"""Rankers: baselines, zero-shot, TF-IDF + SVM, embedding + logistic regression, multi-interest.

Every ranker gets only a ``TrainData`` at ``fit`` time. The evaluation harness builds that object
from labels made *before* the cutoff and from papers published *before* the cutoff; nothing in
this module reads the database, which keeps leakage out by construction.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .config import Interest
from .embeddings import EmbeddingStore, get_embedder
from .paper import Paper
from .text import paper_text


@dataclass
class TrainData:
    """What a ranker may learn from.

    ``papers``/``y``/``tags``: labelled papers (y=1 interesting, 0 not) with their interest tags.
    ``pool``: papers that were available and not yet labelled at fit time (candidates for random negatives).
    """

    papers: list[Paper]
    y: np.ndarray
    tags: list[tuple[str, ...]]
    pool: list[Paper] = field(default_factory=list)

    @property
    def n_pos(self) -> int:
        return int((self.y == 1).sum())

    @property
    def n_neg(self) -> int:
        return int((self.y == 0).sum())


@dataclass
class Context:
    store: EmbeddingStore
    interests: list[Interest]
    seed: int = 0
    default_embedder: str = "bge-small"
    input_format: str | None = None
    _iv: dict = field(default_factory=dict, repr=False)

    def embed(self, papers: Sequence[Paper], name: str | None = None, fmt: str | None = None) -> np.ndarray:
        name = self.default_embedder if name in (None, "default") else name
        return self.store.matrix(list(papers), get_embedder(name), fmt or self.input_format)

    def embed_texts(self, texts: Sequence[str], name: str | None = None) -> np.ndarray:
        name = self.default_embedder if name in (None, "default") else name
        return get_embedder(name).encode(list(texts), kind="query")

    def interest_sims(self, X: np.ndarray, name: str | None = None) -> np.ndarray:
        """Cosine similarity of paper vectors to each interest (max over its description/anchors)."""
        name = self.default_embedder if name in (None, "default") else name
        if name not in self._iv:
            rows, owner = [], []
            for ix, it in enumerate(self.interests):
                m = self.embed_texts(it.texts, name)
                rows.append(m)
                owner += [ix] * len(m)
            self._iv[name] = (np.vstack(rows), np.array(owner))
        M, owner = self._iv[name]
        sims = X @ M.T
        out = np.full((X.shape[0], len(self.interests)), -1.0, dtype=np.float32)
        for ix in range(len(self.interests)):
            out[:, ix] = sims[:, owner == ix].max(axis=1)
        return out


# Pre-registered in ANALYSIS_PLAN.md. Do not copy Scholar Inbox's published values (C=0.1, V=0.8, S=5)
# into these knobs: there V up-weights *explicit* negatives per example (w_N ~ V, w_R ~ 1-V) and S scales
# per-example weights, whereas here V is the share of negative mass on *random* papers and S the total
# negative mass relative to the positives. The exploratory V/S grid in the report tunes them on your data.
DEFAULT_LOGREG = {"C": 1.0, "S": 1.0, "V": 0.5, "n_random": 1000}


def negative_weights(n_pos: int, n_exp: int, n_rand: int, S: float, V: float) -> tuple[float, float]:
    """Per-example weights ``(w_explicit_negative, w_random_negative)``; positives weigh 1.

    The total negative weight is ``S * n_pos``. A share ``V`` of it goes to random unlabeled papers
    (treated as weak negatives), the rest to your explicit "not interested" labels. This
    parametrisation is ours; it is inspired by, not identical to, the weighting scheme in the
    Scholar Inbox paper (arXiv:2504.08385), where V weights explicit negatives instead.
    """
    mass = S * max(n_pos, 1)
    if n_exp > 0 and n_rand > 0:
        return mass * (1 - V) / n_exp, mass * V / n_rand
    if n_exp > 0:
        return mass / n_exp, 0.0
    if n_rand > 0:
        return 0.0, mass / n_rand
    return 0.0, 0.0


def _sample_pool(data: TrainData, n: int, seed: int) -> list[Paper]:
    if n <= 0 or not data.pool:
        return []
    rng = np.random.default_rng(seed + len(data.papers))
    idx = rng.choice(len(data.pool), size=min(n, len(data.pool)), replace=False)
    return [data.pool[i] for i in sorted(idx)]


def _fit_weighted_logreg(X_pos, X_exp, X_rand, C: float, S: float, V: float):
    """Weighted L2 logistic regression; None when a class is missing."""
    w_exp, w_rand = negative_weights(len(X_pos), len(X_exp), len(X_rand), S, V)
    parts, labels, weights = [X_pos], [np.ones(len(X_pos))], [np.ones(len(X_pos))]
    if len(X_exp) and w_exp > 0:
        parts.append(X_exp)
        labels.append(np.zeros(len(X_exp)))
        weights.append(np.full(len(X_exp), w_exp))
    if len(X_rand) and w_rand > 0:
        parts.append(X_rand)
        labels.append(np.zeros(len(X_rand)))
        weights.append(np.full(len(X_rand), w_rand))
    y = np.concatenate(labels)
    if len(X_pos) == 0 or (y == 0).sum() == 0:
        return None
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(np.vstack(parts), y, sample_weight=np.concatenate(weights))
    return clf


class Ranker(ABC):
    needs_fit = True
    # Score at which the model is undecided (used to pick "uncertain" papers); None if it has none.
    boundary: float | None = None

    def __init__(self, ctx: Context, embedder: str | None = None, fmt: str | None = None, **params):
        self.ctx = ctx
        self.embedder = embedder
        self.fmt = fmt
        self.params = params

    @abstractmethod
    def fit(self, data: TrainData) -> Ranker: ...

    @abstractmethod
    def score(self, papers: Sequence[Paper]) -> np.ndarray: ...

    def _X(self, papers: Sequence[Paper]) -> np.ndarray:
        return self.ctx.embed(papers, self.embedder, self.fmt)


# ---- baselines ------------------------------------------------------------------------------
class RandomRanker(Ranker):
    needs_fit = False

    def fit(self, data):
        return self

    def score(self, papers):
        out = []
        for p in papers:
            h = hashlib.md5(f"{self.ctx.seed}:{p.paper_id}".encode()).digest()
            out.append(int.from_bytes(h[:6], "big") / 2**48)
        return np.array(out)


class RecencyRanker(Ranker):
    needs_fit = False

    def fit(self, data):
        return self

    def score(self, papers):
        def day(p: Paper) -> int:
            try:
                return date.fromisoformat(p.published).toordinal()
            except ValueError:
                return 0

        return np.array([day(p) for p in papers], dtype=float)


class CategoryRanker(Ranker):
    """Fraction of a paper's categories that appear in your interests' preferred categories."""

    needs_fit = False

    def fit(self, data):
        return self

    def score(self, papers):
        wanted = {c for it in self.ctx.interests for c in it.categories}
        out = []
        for p in papers:
            cats = p.category_set
            out.append(len(cats & wanted) / len(cats) if cats else 0.0)
        return np.array(out)


class KeywordRanker(Ranker):
    """TF-IDF cosine to the interest descriptions (a no-label lexical baseline)."""

    def fit(self, data):
        texts = [paper_text(p) for p in data.papers + _sample_pool(data, 5000, self.ctx.seed)]
        descs = [t for it in self.ctx.interests for t in it.texts]
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self.vec.fit(texts + descs)
        self.D = self.vec.transform(descs)
        return self

    def score(self, papers):
        X = self.vec.transform([paper_text(p) for p in papers])
        return np.asarray((X @ self.D.T).max(axis=1).todense()).ravel()


class ZeroShotRanker(Ranker):
    """Max cosine between the paper and any interest description. Needs no labels (cold start)."""

    needs_fit = False

    def fit(self, data):
        return self

    def score(self, papers):
        return self.ctx.interest_sims(self._X(papers), self.embedder).max(axis=1)


# ---- learned rankers -------------------------------------------------------------------------
class TfidfSvmRanker(Ranker):
    """Positives vs the rest, TF-IDF + linear SVM, class-balanced (arxiv-sanity-lite's recipe)."""

    boundary = 0.0

    def fit(self, data):
        p = {"C": 0.1, "n_random": 1000, "min_df": 2, **self.params}
        rnd = _sample_pool(data, p["n_random"], self.ctx.seed)
        corpus = [paper_text(x) for x in data.papers + _sample_pool(data, 5000, self.ctx.seed) + rnd]
        # The vectoriser (vocabulary and IDF) is fitted on training-period text only.
        self.vec = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=p["min_df"], max_df=0.5,
            sublinear_tf=True, stop_words="english", dtype=np.float32,
        )
        self.clf = None
        try:
            self.vec.fit(corpus)
        except ValueError:  # tiny or degenerate corpus: relax the frequency filters
            try:
                self.vec.set_params(min_df=1, max_df=1.0)
                self.vec.fit(corpus)
            except ValueError:
                self.vec = None
                return self
        texts = [paper_text(x) for x in data.papers + rnd]
        y = np.concatenate([data.y, np.zeros(len(rnd))])
        if (y == 1).any() and (y == 0).any():
            self.clf = LinearSVC(C=p["C"], class_weight="balanced", random_state=self.ctx.seed).fit(self.vec.transform(texts), y)
        return self

    def score(self, papers):
        if self.clf is None or self.vec is None:
            return np.zeros(len(papers))
        return self.clf.decision_function(self.vec.transform([paper_text(x) for x in papers]))


class EmbeddingLogRegRanker(Ranker):
    """Frozen embedding + class-weighted L2 logistic regression with random weak negatives."""

    boundary = 0.0

    def fit(self, data):
        p = {**DEFAULT_LOGREG, **self.params}
        rnd = _sample_pool(data, p["n_random"], self.ctx.seed)
        X = self._X(data.papers) if data.papers else np.zeros((0, 1))
        X_r = self._X(rnd) if rnd else np.zeros((0, X.shape[1] if X.size else 1))
        pos, neg = data.y == 1, data.y == 0
        self.clf = _fit_weighted_logreg(X[pos], X[neg], X_r, p["C"], p["S"], p["V"])
        return self

    def score(self, papers):
        if self.clf is None:
            return np.zeros(len(papers))
        return self.clf.decision_function(self._X(papers))


class PrototypeRanker(Ranker):
    """Max cosine to per-interest centroids of your positives (parameter-free, interpretable)."""

    def fit(self, data):
        k = int(self.params.get("k", 1))
        X = self._X(data.papers) if data.papers else np.zeros((0, 1))
        groups: dict[str, list[int]] = {}
        for i, (yy, tg) in enumerate(zip(data.y, data.tags, strict=True)):
            if yy != 1:
                continue
            for t in (tg or ("_untagged",)):
                groups.setdefault(t, []).append(i)
        cents = []
        for idx in groups.values():
            V = X[idx]
            if k > 1 and len(idx) >= 3 * k:
                km = KMeans(n_clusters=k, n_init=3, random_state=self.ctx.seed).fit(V)
                C = km.cluster_centers_
            else:
                C = V.mean(axis=0, keepdims=True)
            cents.append(C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12))
        self.C = np.vstack(cents) if cents else None
        return self

    def score(self, papers):
        if self.C is None:
            return np.zeros(len(papers))
        return (self._X(papers) @ self.C.T).max(axis=1)


class PerInterestLogRegRanker(Ranker):
    """One logistic regression per interest (positives with that tag vs the negatives), max-combined.

    Positives of *other* interests are left out of an interest's training set: they are unknown
    for it, not negatives. A global model over all positives is included as a fallback member.
    """

    boundary = 0.5

    def fit(self, data):
        p = {**DEFAULT_LOGREG, "min_pos": 5, "include_global": True, **self.params}
        rnd = _sample_pool(data, p["n_random"], self.ctx.seed)
        X = self._X(data.papers) if data.papers else np.zeros((0, 1))
        X_r = self._X(rnd) if rnd else np.zeros((0, X.shape[1] if X.size else 1))
        neg = data.y == 0
        self.models = {}
        for it in self.ctx.interests:
            idx = [i for i, (yy, tg) in enumerate(zip(data.y, data.tags, strict=True)) if yy == 1 and it.id in tg]
            if len(idx) < p["min_pos"]:
                continue
            clf = _fit_weighted_logreg(X[idx], X[neg], X_r, p["C"], p["S"], p["V"])
            if clf is not None:
                self.models[it.id] = clf
        if p["include_global"] or not self.models:
            g = _fit_weighted_logreg(X[data.y == 1], X[neg], X_r, p["C"], p["S"], p["V"])
            if g is not None:
                self.models["_global"] = g
        return self

    def score(self, papers):
        if not self.models:
            return np.zeros(len(papers))
        X = self._X(papers)
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models.values()])
        return probs.max(axis=0)


class FusionRanker(Ranker):
    """Average of per-component rank-normalised scores (sparse + dense fusion)."""

    needs_fit = False

    def __init__(self, ctx, components: list[Ranker], weights: list[float] | None = None, **kw):
        super().__init__(ctx, **kw)
        self.components = components
        self.weights = weights or [1.0] * len(components)

    def fit(self, data):  # components are fitted by the caller (so each is fitted once)
        return self

    def score(self, papers):
        total = np.zeros(len(papers))
        for w, c in zip(self.weights, self.components, strict=True):
            s = c.score(papers)
            total += w * (rankdata(s) - 1) / max(len(s) - 1, 1)
        return total / sum(self.weights)


RANKER_TYPES: dict[str, type[Ranker]] = {
    "random": RandomRanker, "recency": RecencyRanker, "category": CategoryRanker,
    "keyword": KeywordRanker, "zeroshot": ZeroShotRanker, "tfidf_svm": TfidfSvmRanker,
    "emb_logreg": EmbeddingLogRegRanker, "prototype": PrototypeRanker,
    "per_interest_logreg": PerInterestLogRegRanker,
}


def build_rankers(specs: list[dict], ctx: Context) -> dict[str, Ranker]:
    """Instantiate rankers from specs: ``{name, type, embedder?, input_format?, params?, of?}``.

    A ``fusion`` spec lists other rankers by name in ``of``; they must appear earlier in the list.
    """
    built: dict[str, Ranker] = {}
    for s in specs:
        name, typ = s["name"], s["type"]
        if name in built:
            raise ValueError(f"duplicate ranker name {name!r}")
        if typ == "fusion":
            comps = [built[n] for n in s["of"]]
            built[name] = FusionRanker(ctx, comps, s.get("weights"))
        elif typ in RANKER_TYPES:
            built[name] = RANKER_TYPES[typ](
                ctx, embedder=s.get("embedder"), fmt=s.get("input_format"), **(s.get("params") or {})
            )
        else:
            raise ValueError(f"unknown ranker type {typ!r}")
    return built


def default_specs(model_params: dict | None = None, extra_embedders: Sequence[str] = ()) -> list[dict]:
    """The v1 comparison set. Baselines first, then learned models, then ablation extras."""
    mp = dict(model_params or {})
    specs: list[dict] = [
        {"name": "random", "type": "random"},
        {"name": "recency", "type": "recency"},
        {"name": "category", "type": "category"},
        {"name": "keyword", "type": "keyword"},
        {"name": "zeroshot", "type": "zeroshot"},
        {"name": "tfidf_svm", "type": "tfidf_svm"},
        {"name": "emb_logreg", "type": "emb_logreg", "params": mp},
        {"name": "prototype", "type": "prototype"},
        {"name": "per_interest_logreg", "type": "per_interest_logreg", "params": mp},
        {"name": "fusion_tfidf_emb", "type": "fusion", "of": ["tfidf_svm", "emb_logreg"]},
    ]
    for e in extra_embedders:
        specs.append({"name": f"emb_logreg@{e}", "type": "emb_logreg", "embedder": e, "params": mp})
    return specs


def build_active_ranker(
    cfg_model: dict, ctx: Context, data: TrainData
) -> tuple[Ranker, str]:
    """The ranker used for the digest: zero-shot until there are enough labels, then the learned one."""
    enough = (
        len(data.papers) >= cfg_model["min_labels"]
        and data.n_pos >= cfg_model["min_positives"]
        and data.n_neg >= cfg_model["min_negatives"]
    )
    if not enough:
        return ZeroShotRanker(ctx).fit(data), "zeroshot"
    typ = cfg_model["ranker"]
    r = RANKER_TYPES[typ](ctx, **(cfg_model.get("params") or {}))
    return r.fit(data), typ
