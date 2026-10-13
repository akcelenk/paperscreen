"""Daily digest: quota per interest, plus unmarked uniformly random picks.

The random picks are what makes later evaluation honest. Without them every item you ever label was
chosen by a previous model, and a new model can only be measured on that model's taste. Items are
shown without scores or labels, shuffled inside each interest group, so you judge them blind.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .db import Database
from .paper import Paper
from .rankers import Context, TrainData, build_active_ranker


@dataclass
class DigestEntry:
    paper: Paper
    reason: str
    score: float
    rank: int
    interest: str
    similar: list[tuple[Paper, float]] = field(default_factory=list)


@dataclass
class Digest:
    day: str
    model_id: str
    entries: list[DigestEntry]
    n_candidates: int
    markdown: str = ""
    path: str | None = None


def training_data(db: Database, exclude: set[str] | None = None) -> TrainData:
    labels = db.labels_frame()
    by_id = {p.paper_id: p for p in db.get_papers(labels["paper_id"])}
    labeled = set(labels["paper_id"])
    exclude = exclude or set()
    pool = [p for p in db.query_papers() if p.paper_id not in labeled and p.paper_id not in exclude]
    return TrainData(
        papers=[by_id[i] for i in labels["paper_id"]],
        y=labels["label"].to_numpy(dtype=int),
        tags=[tuple(t) for t in labels["tags"]],
        pool=pool,
    )


def _register(cfg: Config, ctx: Context, ranker, ranker_type: str, data: TrainData, fit_seconds: float) -> str:
    """Save the fitted model with a model card; returns the version id stored with each impression."""
    from . import registry

    root = cfg["model"].get("registry_dir")
    snap = registry.label_snapshot(data)
    version = registry.make_version(ranker_type, snap)
    if not root:
        return version
    version = registry.unique_version(root, version)
    card = registry.ModelCard(
        version=version, created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        ranker=ranker_type, params=dict(cfg["model"].get("params") or {}) if ranker_type != "zeroshot" else {},
        embedder=ctx.default_embedder, input_format=ctx.input_format, n_labels=len(data.papers),
        n_pos=data.n_pos, n_neg=data.n_neg, n_pool=len(data.pool), label_snapshot=snap,
        config=cfg.fingerprint(), git=registry.git_sha(), fit_seconds=round(fit_seconds, 3),
        notes="cold start: zero-shot similarity to interest descriptions" if ranker_type == "zeroshot" else "",
    )
    registry.save_model(root, ranker, card)
    registry.prune(root, int(cfg["model"].get("keep_versions", 30)))
    return version


def _access_link(p: Paper, proxy_prefix: str | None) -> str:
    target = f"https://doi.org/{p.doi}" if p.doi else p.url
    return f"{proxy_prefix}{target}" if proxy_prefix and target else ""


def build_digest(db: Database, cfg: Config, ctx: Context, today: date | None = None,
                 write: bool = True) -> Digest:
    today = today or datetime.now(timezone.utc).date()
    dc = cfg["digest"]
    since = (today - timedelta(days=dc["lookback_days"])).isoformat()
    shown, labeled = db.shown_ids(), db.labeled_ids()
    cands = [p for p in db.query_papers("published >= ?", (since,))
             if p.paper_id not in shown and p.paper_id not in labeled]
    if not cands:
        return Digest(today.isoformat(), "none", [], 0, markdown=f"# Digest {today}\n\nNo new candidates.\n")

    data = training_data(db, exclude={p.paper_id for p in cands})
    t0 = time.perf_counter()
    ranker, ranker_type = build_active_ranker(cfg["model"], ctx, data)
    fit_seconds = time.perf_counter() - t0
    model_id = _register(cfg, ctx, ranker, ranker_type, data, fit_seconds) if write else ranker_type
    scores = ranker.score(cands)
    rank_of = {cands[i].paper_id: r + 1 for r, i in enumerate(np.argsort(-scores, kind="stable"))}
    X = ctx.embed(cands)
    assign = ctx.interest_sims(X).argmax(axis=1)  # which interest group each paper is listed under
    reason = "coldstart" if ranker_type == "zeroshot" else "exploit"

    picked: dict[int, str] = {}
    for g in range(len(ctx.interests)):
        idx = [i for i in np.argsort(-scores, kind="stable") if assign[i] == g][: dc["per_interest"]]
        picked.update({int(i): reason for i in idx})
    rng = np.random.default_rng(int(today.strftime("%Y%m%d")) + cfg.seed)
    rest = [i for i in range(len(cands)) if i not in picked]
    for i in rng.choice(rest, size=min(dc["n_random"], len(rest)), replace=False) if rest else []:
        picked[int(i)] = "random"

    # Nearest positives, for a short "why this one" line.
    pos = [p for p, y in zip(data.papers, data.y, strict=True) if y == 1][-2000:]
    sim_idx: dict[int, list[tuple[Paper, float]]] = {}
    if pos:
        P = ctx.embed(pos)
        for i in picked:
            s = P @ X[i]
            top = np.argsort(-s)[:2]
            sim_idx[i] = [(pos[j], float(s[j])) for j in top]

    entries = [
        DigestEntry(cands[i], r, float(scores[i]), rank_of[cands[i].paper_id],
                    ctx.interests[int(assign[i])].id, sim_idx.get(i, []))
        for i, r in picked.items()
    ]
    order = {it.id: k for k, it in enumerate(ctx.interests)}
    shuffle = rng.random(len(entries))  # hide score order and which items are the random picks
    entries = [e for _, e in sorted(zip(zip([order[e.interest] for e in entries], shuffle, strict=True), entries, strict=True),
                                    key=lambda t: t[0])]

    now = f"{today.isoformat()}T{datetime.now(timezone.utc):%H:%M:%S}Z"
    db.add_impressions([{"paper_id": e.paper.paper_id, "shown_at": now, "reason": e.reason,
                         "score": e.score, "rank": e.rank, "model_id": model_id} for e in entries])
    md = render_digest(today.isoformat(), model_id, entries, len(cands), ctx, dc.get("proxy_prefix"))
    dg = Digest(today.isoformat(), model_id, entries, len(cands), markdown=md)
    if write:
        out = Path(dc["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{today.isoformat()}.md"
        n = 2
        while path.exists():  # a second digest on the same day gets a suffix, never overwrites
            path = out / f"{today.isoformat()}-{n}.md"
            n += 1
        path.write_text(md, "utf-8")
        dg.path = str(path)
    return dg


def render_digest(day: str, model_id: str, entries: list[DigestEntry], n_cand: int, ctx: Context,
                  proxy_prefix: str | None) -> str:
    names = {it.id: it.name for it in ctx.interests}
    lines = [
        f"# Paper digest {day}", "",
        f"{len(entries)} papers from {n_cand} new candidates. Ranking: `{model_id}`. "
        "Judge each one on its own: `paperscreen label` (or `label-id`).", "",
    ]
    cur, n = None, 0
    for e in entries:
        if e.interest != cur:
            cur = e.interest
            lines += [f"## {names[cur]}", ""]
        n += 1
        p = e.paper
        meta = " · ".join(x for x in (p.source, p.published, p.venue) if x)
        lines += [f"### {n}. {p.title}", "", f"{meta}  ", f"id: `{p.paper_id}`  "]
        if p.url:
            lines.append(f"[open]({p.url})" + (f" · [via university]({_access_link(p, proxy_prefix)})"
                                                if _access_link(p, proxy_prefix) else ""))
        lines.append("")
        if p.abstract:
            lines += [textwrap.shorten(p.abstract, 600, placeholder=" ..."), ""]
        if e.similar:
            lines += ["Similar to your positives: " + "; ".join(
                f"{textwrap.shorten(q.title, 70, placeholder='...')} ({s:.2f})" for q, s in e.similar), ""]
    return "\n".join(lines)
