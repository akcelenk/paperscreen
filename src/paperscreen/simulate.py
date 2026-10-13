"""Synthetic data for tests and the offline demo.

The corpus is built from invented pseudo-words grouped into topics, so it says nothing about real
papers. Its only purpose is to exercise the whole pipeline (storage, labelling stream, rankers,
leakage-safe evaluation) without any personal data or network access. Do not read the demo's
numbers as findings about the methods.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .config import Config, Interest
from .db import Database
from .paper import Paper

_SYL = ["ba", "ko", "ri", "tu", "me", "sa", "no", "li", "ve", "da", "pu", "zo", "he", "gi", "fa", "yu"]


def _words(rng: np.random.Generator, n: int, tag: str) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add("".join(rng.choice(_SYL, size=3)) + tag)
    return sorted(out)


def demo_interests(n: int = 5, seed: int = 0) -> tuple[list[Interest], dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    interests, topics = [], {}
    for i in range(n):
        iid = f"topic{i + 1}"
        words = _words(rng, 40, iid[-1])
        topics[iid] = words
        interests.append(Interest(id=iid, name=f"Synthetic topic {i + 1}",
                                  description=" ".join(words[:15]), categories=["cs.LG"]))
    return interests, topics


def demo_config(base: Config, seed: int = 0) -> Config:
    interests, _ = demo_interests(5, seed)
    raw = {
        "interests": [{"id": i.id, "name": i.name, "description": i.description,
                       "categories": i.categories, "anchors": []} for i in interests],
        "embedding": {"model": "hash-256", "input_format": None},
        "evaluation": {"n_boot": 500, "learning_curve_sizes": [50, 100, 200, 400]},
        "model": {"min_labels": 50, "min_positives": 10, "min_negatives": 10},
        "digest": {"per_interest": 2, "n_random": 3},
    }
    return base.with_overrides(**raw)


def populate_demo(db: Database, days: int = 70, per_day: int = 100, start: date = date(2026, 6, 1),
                  seed: int = 0, n_exploit: int = 10, n_random: int = 5, p_interest: float = 0.08,
                  like_interest: float = 0.85, like_other: float = 0.03) -> date:
    """Simulate a labelling stream and return the last simulated day.

    Each day ``per_day`` papers appear. You are shown ``n_exploit`` papers picked by a noisy
    oracle (a stand-in for "the model at the time") plus ``n_random`` uniformly random ones, and you
    judge all of them the same evening.
    """
    rng = np.random.default_rng(seed)
    interests, topics = demo_interests(5, seed)
    ids = [i.id for i in interests]
    other = {f"other{j}": _words(rng, 40, f"o{j}") for j in range(6)}
    common = _words(rng, 80, "c")
    everything = [w for ws in list(topics.values()) + list(other.values()) for w in ws]

    def text(topic_words: list[list[str]], n: int, share: float) -> str:
        toks = []
        for _ in range(n):
            u = rng.random()
            if u < share:
                toks.append(str(rng.choice(topic_words[int(rng.integers(len(topic_words)))])))
            elif u < share + 0.35:
                toks.append(str(rng.choice(common)))
            else:
                toks.append(str(rng.choice(everything)))
        return " ".join(toks)

    for d in range(days):
        day = start + timedelta(days=d)
        day_s = day.isoformat()
        made: list[tuple[str, list[str], float]] = []
        for k in range(per_day):
            if rng.random() < p_interest:
                tp = [str(rng.choice(ids))]
                if rng.random() < 0.1:
                    tp.append(str(rng.choice(ids)))
            else:
                tp = [str(rng.choice(list(other)))]
            tw = [topics[t] if t in topics else other[t] for t in tp]
            title = text(tw, 8, 0.6).capitalize()
            p = Paper(
                title=f"{title} {d}-{k}", abstract=text(tw, 80, 0.45), published=day_s, source="synthetic",
                arxiv_id=f"{day.year % 100:02d}{day.month:02d}.{d * per_day + k + 1:05d}",
                categories="cs.LG", authors="Synthetic",
            )
            pid, _ = db.upsert_paper(p, now=f"{day_s}T06:00:00Z")
            in_topic = [t for t in tp if t in topics]
            made.append((pid, in_topic, float(bool(in_topic)) + rng.normal(0, 0.8)))
        by_oracle = sorted(range(len(made)), key=lambda i: -made[i][2])[:n_exploit]
        rest = [i for i in range(len(made)) if i not in set(by_oracle)]
        rand = list(rng.choice(rest, size=min(n_random, len(rest)), replace=False))
        shown = [(i, "exploit") for i in by_oracle] + [(int(i), "random") for i in rand]
        db.add_impressions([{"paper_id": made[i][0], "shown_at": f"{day_s}T09:00:00Z", "reason": r,
                             "score": made[i][2], "model_id": "oracle"} for i, r in shown])
        for i, _r in shown:
            pid, in_topic, _ = made[i]
            liked = rng.random() < (like_interest if in_topic else like_other)
            tags = in_topic if (liked and in_topic) else ([str(rng.choice(ids))] if liked else [])
            db.set_label(pid, int(liked), tags, labeled_at=f"{day_s}T18:00:00Z")
    return start + timedelta(days=days - 1)
