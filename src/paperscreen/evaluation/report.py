"""Markdown/plot rendering for evaluation results."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def _fmt(v, nd: int = 3) -> str:
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 3, index: bool = False) -> str:
    if df.empty:
        return "_(empty)_"
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v, nd) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _section(title: str, summary: dict | None, note: str, min_interest_pos: int) -> str:
    out = [f"## {title}", "", note, ""]
    if summary is None:
        out += ["_Not enough data for this evaluation yet._", ""]
        return "\n".join(out)
    mt = summary["map_table"].copy()
    mt["MAP (95% CI)"] = mt.apply(lambda r: f"{r['map']:.3f} [{r['map_lo']:.3f}, {r['map_hi']:.3f}]", axis=1)
    out += [
        f"Windows used: {summary['n_windows_used']} of {summary['n_windows_total']} "
        f"(a window needs at least one positive and one negative). "
        f"Test labels: {summary['n_test_labels']}, of which {summary['n_test_pos']} positive.",
        "",
        md_table(mt[["ranker", "MAP (95% CI)", "p_at_k", "ndcg"]]),
        "",
        "### Primary comparisons (pre-registered in ANALYSIS_PLAN.md, Holm-corrected)",
        "",
        md_table(summary["primary"]) if not summary["primary"].empty else "_(none computed)_",
        "",
        "`diff` is mean(AP of a) - mean(AP of b) over windows, with a paired cluster-bootstrap CI. "
        "A CI that includes 0 means: not distinguishable with this data.",
        "",
        "### Exploratory comparisons (uncorrected; do not treat as findings)",
        "",
        md_table(summary["exploratory"]) if not summary["exploratory"].empty else "_(none)_",
        "",
        "### Per-interest AP (positives of one interest vs all negatives)",
        "",
        md_table(summary["per_interest"], index=True),
        "",
        "Test positives per interest: " + ", ".join(
            f"{k}={v}" + (" (too few to interpret)" if v < min_interest_pos else "")
            for k, v in summary["per_interest_n"].items()) + ".",
        "",
    ]
    return "\n".join(out)


def plot_learning_curve(curve: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, g in curve.groupby("ranker", sort=False):
        ax.plot(g["n_labels"], g["map"], marker="o", label=name, linewidth=1.5)
        ax.fill_between(g["n_labels"], g["lo"], g["hi"], alpha=0.12)
    ax.set_xlabel("number of training labels (first N in time)")
    ax.set_ylabel("MAP on a fixed later period")
    ax.set_title("Label efficiency")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def render_report(meta: dict, main: dict | None, random: dict | None, curve: pd.DataFrame | None,
                  curve_png: str | None, min_interest_pos: int = 20) -> str:
    head = [
        "# paperscreen evaluation report",
        "",
        f"- Generated: {meta['generated']}   Config: `{meta['config']}`   Git: `{meta.get('git') or 'n/a'}`   "
        f"Label snapshot: `{meta.get('label_snapshot', 'n/a')}`   Runtime: {meta.get('seconds', 'n/a')}s",
        f"- Human labels: {meta['n_labels']} ({meta['n_pos']} positive, {meta['n_neg']} negative); "
        f"papers in database: {meta['n_papers']}",
        f"- Rankers: {', '.join(meta['rankers'])}",
        "",
        "> One user, one period. These numbers describe *this* labelled stream, not a general claim "
        "about the methods. Labels made by an LLM are never used for evaluation.",
        "",
    ]
    body = [
        _section("All labelled items", main,
                 "Higher power but biased: items you saw were chosen by earlier models.", min_interest_pos),
        _section("Random slice only (unbiased)", random,
                 "Only items first shown as uniformly random picks. Unbiased by earlier models but "
                 "has few positives, so expect wide intervals and many empty windows.", min_interest_pos),
    ]
    tail = ["## Label-efficiency curve", ""]
    if curve is None or curve.empty:
        tail += ["_Not enough labels for the configured sizes yet._", ""]
    else:
        if curve_png:
            tail += [f"![learning curve]({curve_png})", ""]
        tail += [md_table(curve), ""]
    return "\n".join(head + body + tail)
