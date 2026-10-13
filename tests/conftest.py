from __future__ import annotations

import pytest

from paperscreen.config import load_config
from paperscreen.db import Database
from paperscreen.embeddings import EmbeddingStore
from paperscreen.rankers import Context
from paperscreen.simulate import demo_config, populate_demo


def labels_of(path):
    """Read the labels table of a database file and close it again."""
    with Database(path) as d:
        return d.labels_frame()


class FakeResponse:
    def __init__(self, text: str = "", status: int = 200, headers: dict | None = None, json_data=None):
        self.text, self.status_code, self.headers, self._json = text, status, headers or {}, json_data

    def json(self):
        return self._json


class FakeSession:
    """Returns queued responses (or calls ``handler(url, params)``) and records requests."""

    def __init__(self, responses=None, handler=None):
        self.headers: dict = {}
        self.responses = list(responses or [])
        self.handler = handler
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        if self.handler:
            return self.handler(url, params)
        return self.responses.pop(0)


@pytest.fixture
def cfg(tmp_path_factory):
    return demo_config(load_config(None), seed=0).with_overrides(
        model={"registry_dir": str(tmp_path_factory.mktemp("models"))})


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


def make_ctx(db, cfg) -> Context:
    return Context(EmbeddingStore(db), cfg.interests, cfg.seed, "hash-256", None)


@pytest.fixture
def ctx(db, cfg):
    return make_ctx(db, cfg)


@pytest.fixture
def demo_db(cfg):
    d = Database(":memory:")
    last = populate_demo(d, days=45, per_day=60, seed=0)
    yield d, last
    d.close()
