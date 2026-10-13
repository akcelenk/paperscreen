"""Command line interface."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import click

from .config import Config, ConfigError, default_config_text, load_config
from .db import Database, utcnow_iso
from .digest import build_digest
from .embeddings import EmbeddingStore
from .rankers import Context


def _db(cfg: Config) -> Database:
    """Open the database and close it when the command finishes (WAL files are cleaned up)."""
    db = Database(cfg.db_path)
    cc = click.get_current_context(silent=True)
    if cc is not None:
        cc.call_on_close(db.close)
    return db


def _open(cfg: Config) -> tuple[Database, Context]:
    db = _db(cfg)
    emb = cfg["embedding"]
    ctx = Context(EmbeddingStore(db), cfg.interests, cfg.seed, emb["model"], emb.get("input_format"))
    return db, ctx


@click.group()
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="YAML config (default: ./config.yaml merged over the packaged defaults).")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """paperscreen: rank new papers by what you have said you like."""
    ctx.obj = {"config_path": config_path}


def _cfg(ctx: click.Context) -> Config:
    if ctx.obj["config_path"] is None and not Path("config.yaml").exists():
        click.echo(f"note: no config.yaml in {Path.cwd()}; using the defaults and the database under this folder. "
                   "Run paperscreen from the project folder (or use the desktop shortcut) to see your own data.", err=True)
    try:
        return load_config(ctx.obj["config_path"])
    except (ConfigError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.option("--write-config", is_flag=True, help="Write the default config to ./config.yaml.")
@click.pass_context
def init(ctx: click.Context, write_config: bool) -> None:
    """Create the database (and optionally a starter config.yaml)."""
    if write_config:
        p = Path("config.yaml")
        if p.exists():
            raise click.ClickException("config.yaml already exists; not overwriting it.")
        p.write_text(default_config_text(), "utf-8")
        click.echo("wrote config.yaml")
    cfg = _cfg(ctx)
    Database(cfg.db_path).close()
    click.echo(f"database ready at {cfg.db_path}")


@main.command()
@click.option("--days", type=int, default=None, help="Look back N days (default: since last run).")
@click.option("--source", "only", multiple=True, type=click.Choice(["arxiv", "pubmed", "openalex"]))
@click.pass_context
def fetch(ctx: click.Context, days: int | None, only: tuple[str, ...]) -> None:
    """Fetch new papers from the enabled sources and store them (deduplicated)."""
    from .sources import build_sources, fetch_all

    cfg = _cfg(ctx)
    db = _db(cfg)
    sources, skipped = build_sources(cfg, list(only) or None)
    for s in skipped:
        click.echo(f"{s.source}: skipped ({s.skipped})")
    for r in fetch_all(db, sources, days=days, max_catchup_days=cfg["sources"].get("max_catchup_days", 60)):
        if r.error:
            click.echo(f"{r.source}: ERROR {r.error}")
        else:
            click.echo(f"{r.source}: {r.fetched} fetched, {r.new} new")
        if r.warning:
            click.echo(f"{r.source}: WARNING {r.warning}")
    click.echo(f"{db.count_papers()} papers in database")


@main.command("import-jsonl")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--limit", type=int, default=None)
@click.pass_context
def import_jsonl(ctx: click.Context, path: str, since: datetime | None, limit: int | None) -> None:
    """Backfill from the Kaggle arXiv metadata snapshot (avoids hammering the API)."""
    from .sources import import_kaggle_jsonl

    cfg = _cfg(ctx)
    db = _db(cfg)
    n = new = 0
    for p in import_kaggle_jsonl(path, since.date() if since else None,
                                 cfg["sources"]["arxiv"]["categories"], limit):
        n += 1
        new += int(db.upsert_paper(p)[1])
    click.echo(f"{n} matched, {new} new; {db.count_papers()} papers in database")


@main.command()
@click.pass_context
def embed(ctx: click.Context) -> None:
    """Pre-compute embeddings for every stored paper (otherwise done lazily)."""
    cfg = _cfg(ctx)
    db, c = _open(cfg)
    papers = db.query_papers()
    c.embed(papers)
    click.echo(f"{len(papers)} papers embedded with {cfg['embedding']['model']}")


@main.command()
@click.option("--date", "day", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.pass_context
def digest(ctx: click.Context, day: datetime | None) -> None:
    """Rank new candidates and write today's digest (also logs what was shown)."""
    cfg = _cfg(ctx)
    db, c = _open(cfg)
    dg = build_digest(db, cfg, c, day.date() if day else None)
    if not dg.entries:
        click.echo("no new candidates; run `paperscreen fetch` first")
        return
    click.echo(f"{len(dg.entries)} papers ({dg.model_id}) -> {dg.path}")


def _parse_tags(raw: str, ids: list[str]) -> list[str]:
    out = []
    for tok in raw.replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(ids):
            out.append(ids[int(tok) - 1])
        elif tok in ids:
            out.append(tok)
        else:
            raise click.BadParameter(f"unknown interest {tok!r}")
    return out


@main.command()
@click.option("--limit", type=int, default=30, show_default=True)
@click.pass_context
def label(ctx: click.Context, limit: int) -> None:
    """Judge shown-but-unlabelled papers: y (interesting), n (not), s (skip), q (quit)."""
    cfg = _cfg(ctx)
    db, c = _open(cfg)
    items = db.unlabeled_impressions()[:limit]
    if not items:
        click.echo("nothing to label; run `paperscreen digest` first")
        return
    ids = cfg.interest_ids
    sims = c.interest_sims(c.embed(items))
    done = 0
    for n, (p, sim) in enumerate(zip(items, sims, strict=True), 1):
        click.echo("\n" + "=" * 78)
        click.echo(f"[{n}/{len(items)}] {p.title}")
        click.echo(f"{p.source} · {p.published}" + (f" · {p.venue}" if p.venue else "") + f"  {p.url}")
        click.echo(textwrap.fill(textwrap.shorten(p.abstract, 700, placeholder=" ..."), 78))
        choice = click.prompt("interesting? [y/n/s/q]", type=click.Choice(["y", "n", "s", "q"]),
                              show_choices=False)
        if choice == "q":
            break
        if choice == "s":
            continue
        tags: list[str] = []
        if choice == "y":
            guess = int(sim.argmax()) + 1
            menu = "  ".join(f"{i}={it.id}" for i, it in enumerate(cfg.interests, 1))
            click.echo(menu)
            raw = click.prompt("interest tag(s), '-' for none", default=str(guess), show_default=True)
            tags = [] if raw.strip() == "-" else _parse_tags(raw, ids)
        db.set_label(p.paper_id, 1 if choice == "y" else 0, tags, utcnow_iso())
        done += 1
    click.echo(f"\nsaved {done} labels")


@main.command("label-id")
@click.argument("paper_id")
@click.option("--yes/--no", "liked", default=True)
@click.option("--tags", default="", help="Comma-separated interest ids.")
@click.pass_context
def label_id(ctx: click.Context, paper_id: str, liked: bool, tags: str) -> None:
    """Label one paper by id (ids appear in the digest)."""
    cfg = _cfg(ctx)
    db = _db(cfg)
    if not db.get_papers([paper_id]):
        raise click.ClickException(f"unknown paper id {paper_id!r}")
    db.set_label(paper_id, int(liked), _parse_tags(tags, cfg.interest_ids) if tags else [])
    click.echo(f"labelled {paper_id} as {'yes' if liked else 'no'}")


@main.command()
@click.option("--no-curve", is_flag=True, help="Skip the label-efficiency curve.")
@click.pass_context
def report(ctx: click.Context, no_curve: bool) -> None:
    """Weekly evaluation: time-split metrics, bootstrap CIs, learning curve -> reports/."""
    from .evaluation import evaluate

    cfg = _cfg(ctx)
    db, c = _open(cfg)
    out = evaluate(db, cfg, c, with_curve=not no_curve)
    click.echo(f"report written to {out['dir']}/report.md")
    if out["main"] is None:
        click.echo("not enough labels for a time-split evaluation yet (see the report for thresholds)")


@main.command()
@click.option("--days", type=int, default=None, help="Look back N days (default: since last run).")
@click.pass_context
def refresh(ctx: click.Context, days: int | None) -> None:
    """Fetch new papers and build today's digest in one step (the daily routine)."""
    from .pipeline import refresh as run_refresh

    cfg = _cfg(ctx)
    db, c = _open(cfg)
    run_refresh(cfg, db, c, days=days, log=click.echo)


@main.command()
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--no-browser", is_flag=True, help="Do not open a browser tab automatically.")
@click.option("--bind", default="127.0.0.1", show_default=True,
              help="Interface to listen on. Only change it inside a container, and publish the port to 127.0.0.1 only.")
@click.pass_context
def app(ctx: click.Context, port: int, no_browser: bool, bind: str) -> None:
    """Open the local web app (review, label, refresh, reports) in your browser."""
    import webbrowser

    from .webui import make_server

    cfg = _cfg(ctx)
    Database(cfg.db_path).close()  # create the database on first run
    try:
        server = make_server(cfg, port, bind)
    except OSError as exc:
        raise click.ClickException(f"cannot listen on port {port}: {exc}. Try --port {port + 1}.") from exc
    url = f"http://127.0.0.1:{port}/"
    click.echo(f"paperscreen is running at {url}  (Ctrl+C to stop; your data stays on this computer)")
    if not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("stopped")
    finally:
        server.server_close()


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Counts: papers per source, labels, impressions by reason."""
    cfg = _cfg(ctx)
    db = _db(cfg)
    click.echo(f"papers: {db.count_papers()}  by source: {db.papers_by_source()}")
    lab = db.labels_frame()
    click.echo(f"labels: {len(lab)} ({int(lab['label'].sum()) if len(lab) else 0} positive)")
    rows = db.conn.execute("SELECT reason, COUNT(*) n FROM impressions GROUP BY reason").fetchall()
    click.echo("impressions: " + (", ".join(f"{r['reason']}={r['n']}" for r in rows) or "none"))
    for s in ("arxiv", "pubmed", "openalex"):
        mark = db.kv_get(f"watermark:{s}")
        if mark:
            click.echo(f"last fetch {s}: {mark}")


@main.command()
@click.option("--workdir", type=click.Path(), default="demo_run", show_default=True)
@click.option("--days", type=int, default=70, show_default=True)
@click.option("--seed", type=int, default=0)
@click.pass_context
def demo(ctx: click.Context, workdir: str, days: int, seed: int) -> None:
    """Run the whole pipeline on synthetic data (no network, no personal data)."""
    from .evaluation import evaluate
    from .simulate import demo_config, populate_demo

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    cfg = demo_config(load_config(ctx.obj["config_path"] or None), seed).with_overrides(
        db_path=str(wd / "demo.sqlite"),
        digest={"out_dir": str(wd / "digests")},
        evaluation={"out_dir": str(wd / "reports")},
        model={"registry_dir": str(wd / "models")},
    )
    db, c = _open(cfg)
    last = populate_demo(db, days=days, seed=seed)
    click.echo(f"simulated {days} days: {db.count_papers()} papers, {len(db.labels_frame())} labels")
    dg = build_digest(db, cfg, c, today=last)
    click.echo(f"digest for {last}: {len(dg.entries)} papers ({dg.model_id}) -> {dg.path}")
    out = evaluate(db, cfg, c)
    click.echo(f"report: {out['dir']}/report.md")
    from .monitor import build_monitor, render_monitor

    (wd / "monitor.md").write_text(render_monitor(build_monitor(db, str(wd / "models"), today=last)), "utf-8")
    click.echo(f"monitoring: {wd / 'monitor.md'}")
    click.echo("These numbers come from invented data and only show that the pipeline runs.")



@main.command()
@click.option("--out", type=click.Path(dir_okay=False), default=None, help="Also write the Markdown to this file.")
@click.pass_context
def monitor(ctx: click.Context, out: str | None) -> None:
    """Live-system health: model picks vs random picks, data quality, freshness, label backlog."""
    from .monitor import build_monitor, render_monitor

    cfg = _cfg(ctx)
    db = _db(cfg)
    m = build_monitor(db, cfg["model"].get("registry_dir"))
    text = render_monitor(m)
    click.echo(text)
    if out:
        Path(out).write_text(text, "utf-8")
    if m.alerts:
        ctx.exit(1)  # non-zero exit for cron / CI style checks


@main.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List saved model versions (newest last) from the model registry."""
    from .registry import list_models

    cfg = _cfg(ctx)
    root = cfg["model"].get("registry_dir")
    cards = list_models(root) if root else []
    if not cards:
        click.echo("no saved models yet (they are saved when a digest is built)")
        return
    for c in cards:
        click.echo(f"{c.version:45s} labels={c.n_labels:5d} pos={c.n_pos:4d} fit={c.fit_seconds:6.2f}s "
                   f"config={c.config} git={c.git or '-'}")

if __name__ == "__main__":  # pragma: no cover
    main()
