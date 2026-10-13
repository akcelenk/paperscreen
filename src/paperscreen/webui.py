"""A small local web app: review the digest, label with one click, run refresh and reports.

Standard library only (no extra dependencies). It binds to 127.0.0.1, checks the Host header (DNS
rebinding) and requires a custom header on every POST (cross-site form posts), because it can
write your labels. Paper text comes from external sources and is only ever inserted with
``textContent`` / escaped HTML.
"""

from __future__ import annotations

import base64
import html
import json
import re
import threading
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .config import Config
from .db import Database, utcnow_iso
from .digest import _access_link, candidates
from .embeddings import EmbeddingStore
from .rankers import Context
from .text import paper_text, text_hash

ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


def open_session(cfg: Config) -> tuple[Database, Context]:
    db = Database(cfg.db_path)
    emb = cfg["embedding"]
    return db, Context(EmbeddingStore(db), cfg.interests, cfg.seed, emb["model"], emb.get("input_format"))


class Job:
    """One background job at a time (fetching and reports are slow; the UI polls this)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.name: str | None = None
        self.running = False
        self.log: list[str] = []
        self.error: str | None = None
        self.done = False

    def start(self, name: str, fn: Callable[[Callable[[str], None]], None]) -> bool:
        with self.lock:
            if self.running:
                return False
            self.name, self.running, self.log, self.error, self.done = name, True, [], None, False

        def run() -> None:
            try:
                fn(self.log.append)
            except Exception as exc:  # shown in the UI
                self.error = f"{type(exc).__name__}: {exc}"
                self.log.append(traceback.format_exc(limit=3))
            finally:
                self.running, self.done = False, True

        threading.Thread(target=run, daemon=True).start()
        return True

    def snapshot(self) -> dict:
        return {"name": self.name, "running": self.running, "done": self.done, "error": self.error,
                "log": self.log[-12:]}


def md_to_html(text: str, base: Path | None = None) -> str:
    """Tiny, safe Markdown subset (headings, tables, quotes, images, code) for the report page."""
    out: list[str] = []
    rows: list[list[str]] = []

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"(?<![\w`])_([^_]+)_(?!\w)", r"<em>\1</em>", s)  # _italic_, not snake_case

    def flush_table() -> None:
        if not rows:
            return
        head, body = rows[0], rows[1:]
        out.append("<div class='tablewrap'><table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head)
                   + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                                                      for r in body) + "</tbody></table></div>")
        rows.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?|---", c) or c == "---" for c in cells):
                rows.append(cells)
            continue
        flush_table()
        m = re.match(r"(#{1,3}) (.*)", line)
        img = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            out.append(f"<h{len(m.group(1)) + 1}>{inline(m.group(2))}</h{len(m.group(1)) + 1}>")
        elif img and base and (base / img.group(2)).is_file() and img.group(2).endswith(".png"):
            data = base64.b64encode((base / img.group(2)).read_bytes()).decode()
            out.append(f"<img alt='{html.escape(img.group(1))}' src='data:image/png;base64,{data}'>")
        elif line.startswith(">"):
            out.append(f"<blockquote>{inline(line.lstrip('> '))}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<p class='li'>• {inline(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    return "\n".join(out)


class Handler(BaseHTTPRequestHandler):
    app: WebApp

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    # ---- plumbing -------------------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode())

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]") in ALLOWED_HOSTS

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._json({"error": "bad host"}, 403)
        path = self.path.split("?")[0]
        try:
            if path == "/":
                page = resources.files("paperscreen").joinpath("web/index.html").read_bytes()
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(self.app.state())
            if path == "/api/items":
                return self._json(self.app.items())
            if path == "/api/report":
                return self._json(self.app.report())
            if path == "/api/monitor":
                return self._json(self.app.monitor())
            if path == "/api/job":
                return self._json(self.app.job.snapshot())
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:
        if not self._host_ok() or self.headers.get("X-Requested-With") != "paperscreen":
            return self._json({"error": "forbidden"}, 403)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.split("?")[0]
            if path == "/api/label":
                return self._json(self.app.label(body))
            if path == "/api/unlabel":
                return self._json(self.app.unlabel(body))
            if path == "/api/run":
                return self._json(self.app.run(body))
            return self._json({"error": "not found"}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


class WebApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.job = Job()

    # ---- API ------------------------------------------------------------------------------
    def state(self) -> dict:
        db = Database(self.cfg.db_path)
        try:
            lab = db.labels_frame()
            m = self.cfg["model"]
            n_pos = int(lab["label"].sum()) if len(lab) else 0
            return {
                "available": len(candidates(db, self.cfg)),
                "cold_start": {"labels": int(len(lab)), "need_labels": m["min_labels"],
                               "pos": n_pos, "need_pos": m["min_positives"],
                               "neg": int(len(lab)) - n_pos, "need_neg": m["min_negatives"]},
                "papers": db.count_papers(), "labels": int(len(lab)),
                "positives": int(lab["label"].sum()) if len(lab) else 0,
                "unlabeled": len(db.unlabeled_impressions()),
                "interests": [{"id": i.id, "name": i.name} for i in self.cfg.interests],
                "watermarks": {s: db.kv_get(f"watermark:{s}") for s in ("arxiv", "pubmed", "openalex")},
                "job": self.job.snapshot(),
            }
        finally:
            db.close()

    def items(self, limit: int = 30) -> list[dict]:
        db, ctx = open_session(self.cfg)
        try:
            items = db.unlabeled_impressions()[:limit]  # no scores or reasons: judging stays blind
            if not items:
                return []
            guess = ctx.interest_sims(ctx.embed(items)).argmax(axis=1)
            prefix = self.cfg["digest"].get("proxy_prefix")
            return [{
                "id": p.paper_id, "title": p.title, "abstract": p.abstract, "source": p.source,
                "published": p.published, "venue": p.venue, "authors": p.authors[:160], "url": p.url,
                "access_url": _access_link(p, prefix), "suggest": self.cfg.interests[int(g)].id,
            } for p, g in zip(items, guess, strict=True)]
        finally:
            db.close()

    def label(self, body: dict) -> dict:
        pid, label = body.get("id"), body.get("label")
        tags = [t for t in body.get("tags", []) if t in self.cfg.interest_ids]
        if label not in (0, 1) or not isinstance(pid, str):
            raise ValueError("need id and label 0/1")
        opened = body.get("opened") if isinstance(body.get("opened"), bool) else None
        db = Database(self.cfg.db_path)
        try:
            found = db.get_papers([pid])
            if not found:
                raise ValueError("unknown paper id")
            db.set_label(pid, label, tags if label == 1 else [], utcnow_iso(), opened=opened,
                         text_hash=text_hash(paper_text(found[0])))
        finally:
            db.close()
        return {"ok": True}

    def unlabel(self, body: dict) -> dict:
        db = Database(self.cfg.db_path)
        try:
            db.delete_label(str(body.get("id", "")))
        finally:
            db.close()
        return {"ok": True}

    def run(self, body: dict) -> dict:
        action = body.get("action")
        if action == "refresh":
            from . import pipeline

            def work(log):
                db, ctx = open_session(self.cfg)
                try:
                    pipeline.refresh(self.cfg, db, ctx, log=log)
                finally:
                    db.close()
        elif action == "more":
            from .digest import build_digest

            def work(log):
                db, ctx = open_session(self.cfg)
                try:
                    log("ranking the next batch from papers already downloaded...")
                    dg = build_digest(db, self.cfg, ctx)
                    log(f"{len(dg.entries)} more papers ({dg.model_id})" if dg.entries
                        else "no unseen papers left in the look-back window: fetch new papers")
                finally:
                    db.close()
        elif action == "report":
            from .evaluation import evaluate

            def work(log):
                db, ctx = open_session(self.cfg)
                try:
                    log("running the evaluation (can take a few minutes)...")
                    out = evaluate(db, self.cfg, ctx)
                    log(f"report written to {out['dir']}")
                finally:
                    db.close()
        else:
            raise ValueError("action must be 'refresh', 'more' or 'report'")
        return {"started": self.job.start(action, work)}

    def monitor(self) -> dict:
        from .monitor import build_monitor, render_monitor

        db = Database(self.cfg.db_path)
        try:
            return {"html": md_to_html(render_monitor(build_monitor(db, self.cfg["model"].get("registry_dir"))))}
        finally:
            db.close()

    def report(self) -> dict:
        root = Path(self.cfg["evaluation"]["out_dir"])
        dirs = sorted(d for d in root.glob("*") if (d / "report.md").is_file()) if root.exists() else []
        if not dirs:
            return {"html": None}
        d = dirs[-1]
        return {"html": md_to_html((d / "report.md").read_text("utf-8"), d), "name": d.name}


def make_server(cfg: Config, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": WebApp(cfg)})
    return ThreadingHTTPServer((host, port), handler)
