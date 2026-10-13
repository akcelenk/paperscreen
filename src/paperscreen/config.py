"""Configuration loading (YAML, deep-merged over packaged defaults)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Interest:
    id: str
    name: str
    description: str
    categories: list[str] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """Texts embedded for zero-shot scoring: the description plus any anchor abstracts."""
        return [" ".join(self.description.split())] + [a for a in self.anchors if a]


class ConfigError(ValueError):
    """A config value is missing, misspelt or out of range (all problems are listed at once)."""


KNOWN_EMBEDDERS = ("bge-small", "minilm", "specter2-base", "scincl")
LEARNED_RANKERS = ("emb_logreg", "per_interest_logreg", "tfidf_svm", "prototype")


def validate(raw: dict) -> None:
    """Check the merged config before anything slow runs. Raises ConfigError listing every problem."""
    errs: list[str] = []

    def num(path: str, lo: float | None = None, hi: float | None = None, integer: bool = False) -> None:
        cur: Any = raw
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                errs.append(f"{path}: missing")
                return
            cur = cur[part]
        ok_type = isinstance(cur, int) if integer else isinstance(cur, (int, float))
        if isinstance(cur, bool) or not ok_type:
            errs.append(f"{path}: expected {'an integer' if integer else 'a number'}, got {cur!r}")
        elif (lo is not None and cur < lo) or (hi is not None and cur > hi):
            errs.append(f"{path}: {cur} is outside [{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]")

    if not raw.get("interests"):
        errs.append("interests: at least one interest is needed")
    for i, it in enumerate(raw.get("interests") or []):
        for key in ("id", "name", "description"):
            if not isinstance(it, dict) or not it.get(key):
                errs.append(f"interests[{i}].{key}: missing")
    emb = (raw.get("embedding") or {}).get("model")
    if not (isinstance(emb, str) and (emb in KNOWN_EMBEDDERS or (emb.startswith("hash-") and emb[5:].isdigit()))):
        errs.append(f"embedding.model: {emb!r} is not one of {', '.join(KNOWN_EMBEDDERS)} or hash-<dim>")
    model = raw.get("model") or {}
    if model.get("ranker") not in LEARNED_RANKERS:
        errs.append(f"model.ranker: {model.get('ranker')!r} is not one of {', '.join(LEARNED_RANKERS)}")
    for key, lo, hi in (("C", 1e-9, None), ("S", 1e-9, None), ("V", 0, 1), ("n_random", 0, None)):
        if key in (model.get("params") or {}):
            num(f"model.params.{key}", lo, hi, integer=key == "n_random")
    for path in ("model.min_labels", "model.min_positives", "model.min_negatives", "digest.n_random",
                 "digest.lookback_days", "evaluation.min_train_labels", "evaluation.n_boot"):
        num(path, 0, None, integer=True)
    num("digest.per_interest", 1, None, integer=True)
    num("evaluation.step_days", 1, None, integer=True)
    num("evaluation.test_fraction", 0.01, 0.99)
    for i, pair in enumerate((raw.get("evaluation") or {}).get("primary") or []):
        if not (isinstance(pair, list) and len(pair) == 2):
            errs.append(f"evaluation.primary[{i}]: expected [ranker_a, ranker_b], got {pair!r}")
    prefix = (raw.get("digest") or {}).get("proxy_prefix")
    if prefix and not str(prefix).startswith("https://"):
        errs.append("digest.proxy_prefix: must start with https://")
    if errs:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errs))


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config_text() -> str:
    return resources.files("paperscreen").joinpath("default_config.yaml").read_text("utf-8")


class Config:
    """Thin wrapper over the merged dict. Use ``cfg.raw`` / ``cfg['section']['key']``."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.interests = [Interest(**i) for i in raw.get("interests", [])]
        ids = [i.id for i in self.interests]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate interest ids: {ids}")
        for i in ids:
            if not i or " " in i or "," in i:
                raise ValueError(f"invalid interest id {i!r} (no spaces or commas)")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def interest_ids(self) -> list[str]:
        return [i.id for i in self.interests]

    @property
    def db_path(self) -> str:
        return str(self.raw["db_path"])

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    def secret(self, name: str) -> str | None:
        return os.environ.get(name) or None

    def fingerprint(self) -> str:
        """Short hash of the config, recorded with every run for reproducibility."""
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def with_overrides(self, **sections: Any) -> Config:
        return Config(_deep_merge(self.raw, sections))


def load_config(path: str | Path | None = None) -> Config:
    base = yaml.safe_load(default_config_text())
    p = Path(path) if path else Path("config.yaml")
    if p.exists():
        user = yaml.safe_load(p.read_text("utf-8")) or {}
        base = _deep_merge(base, user)
    elif path:
        raise FileNotFoundError(f"config file not found: {path}")
    validate(base)
    return Config(base)
