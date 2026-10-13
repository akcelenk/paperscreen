"""Text embedders and a persistent cache keyed by (paper, model, input format, text hash)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .db import Database
from .paper import Paper
from .text import paper_text, text_hash


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class Embedder(ABC):
    key: str
    fmt: str = "title_abstract"  # input format this model was trained for

    @abstractmethod
    def encode(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        """L2-normalised float32 matrix ``(len(texts), dim)``. ``kind`` is 'doc' or 'query'."""


class HashingEmbedder(Embedder):
    """Dependency-free bag-of-words hashing embedder.

    Used for tests, the offline demo and as a cheap lexical baseline. It is not a semantic model.
    """

    def __init__(self, dim: int = 256):
        self.key = f"hash-{dim}"
        self.dim = dim
        self._vec = HashingVectorizer(
            n_features=dim, alternate_sign=True, norm="l2", ngram_range=(1, 2),
            stop_words="english", lowercase=True,
        )

    def encode(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        return _l2(self._vec.transform(list(texts)).toarray())


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, key: str, model_name: str, doc_prefix: str = "", query_prefix: str = "",
                 fmt: str = "title_abstract", revision: str | None = None, batch_size: int = 32):
        self.key = key
        self.model_name = model_name
        self.doc_prefix, self.query_prefix = doc_prefix, query_prefix
        self.fmt = fmt
        self.revision = revision
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the 'embed' extra: pip install 'paperscreen[embed]'") from exc
            self._model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        vecs = self._load().encode(
            [prefix + t for t in texts], batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return _l2(np.asarray(vecs))


class CLSEmbedder(Embedder):
    """BERT-style encoder with [CLS] pooling (SPECTER2 base, SciNCL)."""

    def __init__(self, key: str, model_name: str, fmt: str = "title_sep_abstract",
                 revision: str | None = None, batch_size: int = 16, max_length: int = 512):
        self.key = key
        self.model_name = model_name
        self.fmt = fmt
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("install the 'embed' extra: pip install 'paperscreen[embed]'") from exc
            self._tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self._model = AutoModel.from_pretrained(self.model_name, revision=self.revision).eval()
        return self._tok, self._model

    def encode(self, texts: Sequence[str], kind: str = "doc") -> np.ndarray:
        import torch

        tok, model = self._load()
        out = []
        order = np.argsort([len(t) for t in texts])  # batch similar lengths together
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i : i + self.batch_size]
                enc = tok([texts[j] for j in idx], padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt")
                out.append(model(**enc).last_hidden_state[:, 0, :].cpu().numpy())
        mat = np.vstack(out)
        res = np.empty_like(mat)
        res[order] = mat
        return _l2(res)


@cache
def get_embedder(name: str) -> Embedder:
    """Registry. Model ids and licences: check each model card before relying on them."""
    if name.startswith("hash-"):
        return HashingEmbedder(int(name.split("-", 1)[1]))
    if name == "bge-small":
        return SentenceTransformerEmbedder(name, "BAAI/bge-small-en-v1.5")
    if name == "minilm":
        return SentenceTransformerEmbedder(name, "sentence-transformers/all-MiniLM-L6-v2")
    if name in ("gte-base", "gte-large"):
        # Scholar Inbox found GTE-large best on explicit ratings (title [SEP] abstract input).
        return SentenceTransformerEmbedder(name, f"thenlper/{name}", fmt="title_sep_abstract", batch_size=16)
    if name == "specter2-base":
        return CLSEmbedder(name, "allenai/specter2_base")
    if name == "scincl":
        return CLSEmbedder(name, "malteos/scincl")
    raise ValueError(f"unknown embedder {name!r}")


class EmbeddingStore:
    """Computes missing embeddings in batches and caches them in the database (and in memory)."""

    def __init__(self, db: Database):
        self.db = db
        self._memo: dict[tuple[str, str], tuple[str, np.ndarray]] = {}

    def matrix(self, papers: Sequence[Paper], embedder: Embedder, fmt: str | None = None,
               batch: int = 128) -> np.ndarray:
        fmt = fmt or embedder.fmt
        model_key = f"{embedder.key}|{fmt}"
        texts = [paper_text(p, fmt) for p in papers]
        hashes = [text_hash(t) for t in texts]
        need_db = [p.paper_id for p in papers if (model_key, p.paper_id) not in self._memo]
        if need_db:
            for pid, (h, v) in self.db.get_embeddings(model_key, need_db).items():
                self._memo[(model_key, pid)] = (h, v)
        missing = [i for i, p in enumerate(papers)
                   if self._memo.get((model_key, p.paper_id), ("", None))[0] != hashes[i]]
        for s in range(0, len(missing), batch):
            idx = missing[s : s + batch]
            vecs = embedder.encode([texts[i] for i in idx], kind="doc")
            self.db.put_embeddings(model_key, [(papers[i].paper_id, hashes[i], v) for i, v in zip(idx, vecs, strict=True)])
            for i, v in zip(idx, vecs, strict=True):
                self._memo[(model_key, papers[i].paper_id)] = (hashes[i], v)
        if not papers:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._memo[(model_key, p.paper_id)][1] for p in papers])
