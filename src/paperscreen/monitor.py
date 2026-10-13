"""Monitoring for the live system: online model quality, data quality, freshness, label backlog.

The random picks in every digest are a built-in control group. Comparing the share of "interesting"
judgements among papers the model chose with the share among the random picks is an online A/B
measurement of the model: it needs no offline replay and it is not biased by earlier models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .db import Database
from .evaluation.report import md_table


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion with a Wilson score interval (sane for small n and for 0 or n successes)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, mid - half), min(1.0, mid + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference of two proportions ``p1 - p2`` with Newcombe's hybrid score interval."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def _first_impressions(db: Database) -> pd.DataFrame:
    """One row per shown paper: its first impression (reason, time, model) and its label, if any."""
    df = pd.read_sql_query(
        "SELECT i.paper_id, i.shown_at, i.reason, i.model_id, l.label, l.labeled_at "
        "FROM impressions i JOIN (SELECT paper_id, MIN(id) AS id FROM impressions GROUP BY paper_id) f "
        "ON f.id = i.id LEFT JOIN labels l ON l.paper_id = i.paper_id AND l.origin = 'human'",
        db.conn,
    )
    df["day"] = pd.to_datetime(df["shown_at"], utc=True, errors="coerce").dt.date
    df["week"] = df["day"].map(lambda d: (d - timedelta(days=d.weekday())).isoformat() if pd.notna(d) else "")
    df["arm"] = df["reason"].map(lambda r: "random" if r == "random" else "model")
    return df


def online_quality(imp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Weekly precision of model picks vs random picks, and the pooled comparison."""
    lab = imp[imp["label"].notna()]
    rows = []
    for (week, arm), g in lab.groupby(["week", "arm"]):
        k, n = int(g["label"].sum()), len(g)
        p, lo, hi = wilson(k, n)
        rows.append({"week": week, "arm": arm, "labelled": n, "interesting": k, "precision": p, "lo": lo, "hi": hi})
    weekly = pd.DataFrame(rows, columns=["week", "arm", "labelled", "interesting", "precision", "lo", "hi"])
    m, r = lab[lab["arm"] == "model"], lab[lab["arm"] == "random"]
    km, nm, kr, nr = int(m["label"].sum()), len(m), int(r["label"].sum()), len(r)
    d, dlo, dhi = newcombe_diff(km, nm, kr, nr)
    pm, pr = wilson(km, nm)[0], wilson(kr, nr)[0]
    pooled = {"model_n": nm, "model_k": km, "model_p": pm, "random_n": nr, "random_k": kr, "random_p": pr,
              "diff": d, "diff_lo": dlo, "diff_hi": dhi,
              "lift": pm / pr if nr and pr > 0 else float("nan")}
    return weekly, pooled


def data_quality(db: Database, today: date, days: int = 30) -> pd.DataFrame:
    """Per source, papers first seen in the last ``days`` days: volume, completeness, freshness."""
    since = (today - timedelta(days=days)).isoformat()
    df = pd.read_sql_query("SELECT source, abstract, authors, published, first_seen FROM papers WHERE first_seen >= ?",
                           db.conn, params=[since])
    rows = []
    multi = int(df["source"].str.contains(",").sum()) if len(df) else 0
    for src in sorted({s for v in df["source"] for s in v.split(",") if s}):
        g = df[df["source"].str.split(",").map(lambda xs, s=src: s in xs)]
        newest = g["published"].max() or ""
        try:
            age = (today - date.fromisoformat(newest)).days
        except ValueError:
            age = None
        rows.append({
            "source": src, "papers": len(g),
            "no_abstract_%": 100 * float((g["abstract"].str.len() < 50).mean()),
            "no_authors_%": 100 * float((g["authors"] == "").mean()),
            "newest_published": newest, "days_since_newest": age,
        })
    out = pd.DataFrame(rows, columns=["source", "papers", "no_abstract_%", "no_authors_%", "newest_published",
                                      "days_since_newest"])
    out.attrs["multi_source"] = multi
    return out


@dataclass
class MonitorReport:
    today: str
    pooled: dict
    weekly: pd.DataFrame
    quality: pd.DataFrame
    backlog: int
    oldest_unlabelled_days: int | None
    labels_last_7d: int
    model: dict | None
    alerts: list[str] = field(default_factory=list)


def build_monitor(db: Database, registry_dir: str | None = None, today: date | None = None,
                  stale_days: int = 4) -> MonitorReport:
    today = today or datetime.now(timezone.utc).date()
    imp = _first_impressions(db)
    weekly, pooled = online_quality(imp)
    quality = data_quality(db, today)
    unl = imp[imp["label"].isna()]
    oldest = int((today - unl["day"].min()).days) if len(unl) and pd.notna(unl["day"].min()) else None
    week_ago = (today - timedelta(days=7)).isoformat()
    n7 = int((imp["labeled_at"].fillna("") >= week_ago).sum())
    model = None
    if registry_dir:
        from .registry import list_models

        cards = list_models(registry_dir)
        if cards:
            c = cards[-1]
            model = {"version": c.version, "ranker": c.ranker, "n_labels": c.n_labels, "n_pos": c.n_pos,
                     "fit_seconds": c.fit_seconds, "versions_kept": len(cards)}

    alerts = []
    for _, r in quality.iterrows():
        if r["days_since_newest"] is not None and r["days_since_newest"] > stale_days:
            alerts.append(f"{r['source']}: newest paper is {r['days_since_newest']} days old (source may be down or lagging)")
        if r["papers"] >= 20 and r["no_abstract_%"] > 20:
            alerts.append(f"{r['source']}: {r['no_abstract_%']:.0f}% of recent papers have no abstract (parser or API change?)")
    if len(unl) > 100 or (oldest is not None and oldest > 14):
        alerts.append(f"label backlog: {len(unl)} shown papers wait for a judgement (oldest {oldest} days)")
    if len(imp) and n7 == 0:
        alerts.append("no judgements in the last 7 days: the model cannot learn and evaluation stalls")
    if pooled["random_n"] >= 30 and pooled["model_n"] >= 30 and pooled["diff_hi"] < 0:
        alerts.append("model picks are judged interesting LESS often than random picks: check the model")
    return MonitorReport(today.isoformat(), pooled, weekly, quality, len(unl), oldest, n7, model, alerts)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def render_monitor(m: MonitorReport) -> str:
    p = m.pooled
    lines = [f"# paperscreen monitoring ({m.today})", ""]
    lines += ["## Alerts", ""] + ([f"- {a}" for a in m.alerts] or ["- none"]) + [""]
    lines += [
        "## Online quality: model picks vs random picks",
        "",
        "Random picks are a control group shown the same way as model picks. If the model helps, its picks "
        "are judged interesting more often.",
        "",
        f"- Model picks: {_pct(p['model_p'])} interesting ({p['model_k']}/{p['model_n']})",
        f"- Random picks: {_pct(p['random_p'])} interesting ({p['random_k']}/{p['random_n']})",
        f"- Difference: {_pct(p['diff'])} (95% CI {_pct(p['diff_lo'])} to {_pct(p['diff_hi'])}); "
        f"lift x{p['lift']:.2f}" if p["lift"] == p["lift"] else
        f"- Difference: {_pct(p['diff'])} (95% CI {_pct(p['diff_lo'])} to {_pct(p['diff_hi'])})",
        "",
        "Model picks include cold-start (zero-shot) weeks. Per week:",
        "",
        md_table(m.weekly) if not m.weekly.empty else "_(no judgements yet)_",
        "",
        "## Data quality (papers first seen in the last 30 days)",
        "",
        md_table(m.quality, nd=1) if not m.quality.empty else "_(no recent papers)_",
        "",
        f"Papers found in more than one source (merged): {m.quality.attrs.get('multi_source', 0)}.",
        "",
        "## Labelling",
        "",
        f"- Waiting for a judgement: {m.backlog}" + (f" (oldest shown {m.oldest_unlabelled_days} days ago)"
                                                    if m.oldest_unlabelled_days is not None else ""),
        f"- Judgements in the last 7 days: {m.labels_last_7d}",
        "",
        "## Current model",
        "",
    ]
    if m.model:
        lines += [f"- Version `{m.model['version']}` ({m.model['ranker']}), trained on {m.model['n_labels']} labels "
                  f"({m.model['n_pos']} positive) in {m.model['fit_seconds']:.2f}s; {m.model['versions_kept']} saved version(s) kept."]
    else:
        lines += ["- No saved model yet (build a digest first)."]
    return "\n".join(lines) + "\n"
