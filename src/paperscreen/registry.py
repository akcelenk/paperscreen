"""A small file-based model registry: every model that ranked a digest is saved with a model card.

Each version lives in ``<registry_dir>/<version>/`` as ``model.joblib`` (the fitted state, without the
embedding store) and ``card.json`` (what it was trained on and how). Impressions store the version
id, so any digest entry can be traced back to the exact model, training-label snapshot, config and
code that produced it, and the ranking can be reproduced with :func:`load_model`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from . import __version__
from .rankers import RANKER_TYPES, Context, FusionRanker, Ranker, TrainData


@dataclass
class ModelCard:
    version: str
    created: str
    ranker: str
    params: dict
    embedder: str
    input_format: str | None
    n_labels: int
    n_pos: int
    n_neg: int
    n_pool: int
    label_snapshot: str          # hash of (paper_id, label, tags) of the training labels
    config: str                  # config fingerprint
    git: str | None
    fit_seconds: float
    package_version: str = __version__
    python: str = field(default_factory=platform.python_version)
    notes: str = ""


def label_snapshot(data: TrainData) -> str:
    """Order-independent hash of the training labels (what the model learned from)."""
    rows = sorted(f"{p.paper_id}\t{int(y)}\t{' '.join(sorted(t))}" for p, y, t in zip(data.papers, data.y, data.tags, strict=True))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:12]


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def make_version(ranker: str, snapshot: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{ranker}-{now:%Y%m%dT%H%M%S}-{snapshot[:6]}"


def unique_version(root: str | Path, version: str) -> str:
    """Append a counter if a version with this id already exists (two digests in one second)."""
    v, n = version, 2
    while (Path(root) / v).exists():
        v, n = f"{version}-{n}", n + 1
    return v


def save_model(root: str | Path, ranker: Ranker, card: ModelCard) -> Path:
    if isinstance(ranker, FusionRanker):
        raise TypeError("fusion rankers are not saved; save their components")
    out = Path(root) / card.version
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in ranker.__dict__.items() if k != "ctx"}
    joblib.dump({"class": type(ranker).__name__, "state": state}, out / "model.joblib")
    (out / "card.json").write_text(json.dumps(asdict(card), indent=2), "utf-8")
    return out


def load_model(path: str | Path, ctx: Context) -> tuple[Ranker, ModelCard]:
    """Rebuild a saved ranker. ``ctx`` supplies embeddings; use the card's embedder to match."""
    path = Path(path)
    blob = joblib.load(path / "model.joblib")
    classes = {cls.__name__: cls for cls in RANKER_TYPES.values()}
    cls = classes[blob["class"]]
    obj = cls.__new__(cls)
    obj.__dict__.update(blob["state"])
    obj.ctx = ctx
    card = ModelCard(**json.loads((path / "card.json").read_text("utf-8")))
    return obj, card


def list_models(root: str | Path) -> list[ModelCard]:
    root = Path(root)
    if not root.exists():
        return []
    cards = []
    for d in root.iterdir():
        f = d / "card.json"
        if f.is_file():
            try:
                cards.append(ModelCard(**json.loads(f.read_text("utf-8"))))
            except (TypeError, ValueError):
                continue  # a card from an incompatible version: skip, never crash the listing
    return sorted(cards, key=lambda c: (c.created, c.version))


def prune(root: str | Path, keep: int) -> list[str]:
    """Delete all but the newest ``keep`` versions; returns the removed version ids."""
    cards = list_models(root)
    removed = [c.version for c in cards[: max(0, len(cards) - keep)]]
    for v in removed:
        shutil.rmtree(Path(root) / v, ignore_errors=True)
    return removed


def same_scores(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b))
