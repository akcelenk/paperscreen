"""Evaluation harness: prequential runs, bootstrap statistics, reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..db import Database
from ..rankers import Context, default_specs
from .report import plot_learning_curve, render_report
from .runner import (
    EvalSettings,
    InsufficientData,
    PrequentialResult,
    run_learning_curve,
    run_prequential,
    summarize,
)

__all__ = ["evaluate", "EvalSettings", "InsufficientData", "PrequentialResult",
           "run_prequential", "run_learning_curve", "summarize"]


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def label_snapshot(labels) -> str:
    """Hash of every human label and its time: two reports with the same snapshot saw the same data."""
    rows = sorted(f"{r.paper_id}\t{r.label}\t{' '.join(sorted(r.tags))}\t{r.labeled_at}" for r in labels.itertuples())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:12]


def evaluate(db: Database, cfg: Config, ctx: Context, specs: list[dict] | None = None,
             out_root: str | Path | None = None, with_curve: bool = True) -> dict:
    """Run the full evaluation and write ``report.md`` + CSV/JSON under ``reports/<date>_<hash>/``."""
    ev = cfg["evaluation"]
    settings = EvalSettings.from_config(ev, cfg.seed)
    specs = specs or cfg.get("rankers") or default_specs(
        cfg["model"].get("params"), ev.get("extra_embedders") or [])
    t0 = time.perf_counter()
    labels = db.labels_frame()
    papers = db.query_papers()
    ids = cfg.interest_ids
    primary = ev["primary"]

    main = random = None
    res = None
    try:
        res = run_prequential(labels, papers, specs, ctx, settings)
        main = summarize(res, settings, primary, ids)
        try:
            random = summarize(res.restrict_random(), settings, primary, ids)
        except InsufficientData:
            random = None
    except InsufficientData:
        pass

    curve = None
    if with_curve and len(labels) >= 2:
        sizes = [n for n in ev["learning_curve_sizes"] if n <= int(len(labels) * (1 - ev["test_fraction"]))]
        if sizes:
            curve = run_learning_curve(labels, papers, specs, ctx, settings, sizes, ev["test_fraction"])

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(out_root or ev["out_dir"]) / f"{stamp}_{cfg.fingerprint()}"
    out.mkdir(parents=True, exist_ok=True)
    png = "learning_curve.png" if curve is not None and not curve.empty and plot_learning_curve(
        curve, out / "learning_curve.png") else None
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": cfg.fingerprint(), "git": _git_sha(), "n_labels": int(len(labels)),
        "n_pos": int(labels["label"].sum()) if len(labels) else 0,
        "n_neg": int((labels["label"] == 0).sum()) if len(labels) else 0,
        "n_papers": len(papers), "rankers": [s["name"] for s in specs],
        "label_snapshot": label_snapshot(labels), "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "report.md").write_text(
        render_report(meta, main, random, curve, png, ev.get("min_interest_positives", 20)), "utf-8")
    for tag, summ in (("all", main), ("random", random)):
        if summ:
            summ["map_table"].to_csv(out / f"map_{tag}.csv", index=False)
            summ["window_metrics"].to_csv(out / f"windows_{tag}.csv", index=False)
    if curve is not None:
        curve.to_csv(out / "learning_curve.csv", index=False)
    (out / "meta.json").write_text(json.dumps({**meta, "specs": specs, "primary": primary}, indent=2))
    return {"dir": str(out), "main": main, "random": random, "curve": curve, "meta": meta}
