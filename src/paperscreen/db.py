"""SQLite storage: papers (deduplicated across sources), labels, impressions, embeddings."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paper import Paper
from .text import arxiv_id_from_doi, normalize_arxiv_id, normalize_doi, title_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL DEFAULT '',
    published   TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    arxiv_id    TEXT, doi TEXT, pmid TEXT, openalex_id TEXT,
    authors     TEXT NOT NULL DEFAULT '',
    categories  TEXT NOT NULL DEFAULT '',
    venue       TEXT NOT NULL DEFAULT '',
    url         TEXT NOT NULL DEFAULT '',
    first_seen  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_keys (
    key      TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
CREATE TABLE IF NOT EXISTS labels (
    paper_id   TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    label      INTEGER NOT NULL CHECK (label IN (0, 1)),
    tags       TEXT NOT NULL DEFAULT '',
    labeled_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    origin     TEXT NOT NULL DEFAULT 'human',
    opened     INTEGER,          -- 1 if a link was opened before judging
    text_hash  TEXT              -- hash of the title+abstract that was judged
);
CREATE TABLE IF NOT EXISTS impressions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    shown_at TEXT NOT NULL,
    reason   TEXT NOT NULL,
    score    REAL,
    rank     INTEGER,
    model_id TEXT,
    propensity   REAL,           -- probability that this arm picked the paper (1.0 = deterministic)
    n_candidates INTEGER,        -- size of the candidate pool when the batch was built
    position     INTEGER         -- position inside the shown batch
);
CREATE INDEX IF NOT EXISTS idx_impr_paper ON impressions(paper_id);
CREATE TABLE IF NOT EXISTS embeddings (
    paper_id  TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL,
    PRIMARY KEY (paper_id, model_key)
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

REASONS = ("exploit", "random", "uncertain", "coldstart")

# Columns added after 0.2.0. Existing databases get them on open; old rows keep NULL.
MIGRATIONS = {
    "labels": {"opened": "INTEGER", "text_hash": "TEXT"},
    "impressions": {"propensity": "REAL", "n_candidates": "INTEGER", "position": "INTEGER"},
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _keys_for(p: Paper) -> list[str]:
    keys = []
    if p.arxiv_id:
        keys.append(f"arxiv:{p.arxiv_id}")
    if p.doi:
        keys.append(f"doi:{p.doi}")
    if p.pmid:
        keys.append(f"pmid:{p.pmid}")
    if p.openalex_id:
        keys.append(f"oa:{p.openalex_id}")
    tk = title_key(p.title)
    if tk:
        keys.append(f"title:{tk}")
    return keys


def _row_to_paper(r: sqlite3.Row) -> Paper:
    return Paper(
        paper_id=r["paper_id"], title=r["title"], abstract=r["abstract"], published=r["published"],
        source=r["source"], arxiv_id=r["arxiv_id"], doi=r["doi"], pmid=r["pmid"],
        openalex_id=r["openalex_id"], authors=r["authors"], categories=r["categories"],
        venue=r["venue"], url=r["url"], first_seen=r["first_seen"],
    )


class Database:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # The web app writes labels while a background refresh writes papers: wait instead of failing.
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")  # readers no longer block the writer
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        with self.conn:
            for table, cols in MIGRATIONS.items():
                have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
                for col, typ in cols.items():
                    if col not in have:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- papers -------------------------------------------------------------------------
    def upsert_paper(self, p: Paper, now: str | None = None) -> tuple[str, bool]:
        """Insert or merge a paper. Returns ``(paper_id, created)``.

        Records are matched on arXiv id, DOI, PMID, OpenAlex id, or an exact long-title match,
        so a preprint on arXiv and its journal version on PubMed/OpenAlex collapse into one row.
        The canonical id is chosen once, at creation: arxiv > doi > pmid > openalex > title.
        """
        p.arxiv_id = normalize_arxiv_id(p.arxiv_id) or arxiv_id_from_doi(p.doi)
        p.doi = normalize_doi(p.doi)
        keys = _keys_for(p)
        if not keys:
            raise ValueError(f"paper has no usable identifier or title: {p.title!r}")
        marks = ",".join("?" * len(keys))
        found = [
            r["paper_id"]
            for r in self.conn.execute(
                f"SELECT DISTINCT paper_id FROM paper_keys WHERE key IN ({marks})", keys
            )
        ]
        now = now or utcnow_iso()
        with self.conn:
            if not found:
                pid = keys[0]
                self.conn.execute(
                    "INSERT INTO papers (paper_id,title,abstract,published,source,arxiv_id,doi,pmid,"
                    "openalex_id,authors,categories,venue,url,first_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, p.title, p.abstract, p.published, p.source, p.arxiv_id, p.doi, p.pmid,
                     p.openalex_id, p.authors, p.categories, p.venue, p.url, now),
                )
                self.conn.executemany(
                    "INSERT OR IGNORE INTO paper_keys (key, paper_id) VALUES (?, ?)",
                    [(k, pid) for k in keys],
                )
                return pid, True
            target = self._pick_target(found)
            for other in found:
                if other != target:
                    self._merge_rows(target, other)
            self._merge_fields(target, p)
            self.conn.executemany(
                "INSERT OR IGNORE INTO paper_keys (key, paper_id) VALUES (?, ?)",
                [(k, target) for k in keys],
            )
            return target, False

    def _pick_target(self, ids: Sequence[str]) -> str:
        marks = ",".join("?" * len(ids))
        row = self.conn.execute(
            f"SELECT paper_id FROM papers WHERE paper_id IN ({marks}) ORDER BY first_seen, paper_id LIMIT 1",
            list(ids),
        ).fetchone()
        return row["paper_id"]

    def _merge_fields(self, target: str, p: Paper) -> None:
        cur = self.conn.execute("SELECT * FROM papers WHERE paper_id=?", (target,)).fetchone()
        upd: dict[str, object] = {}
        for col in ("arxiv_id", "doi", "pmid", "openalex_id"):
            if not cur[col] and getattr(p, col):
                upd[col] = getattr(p, col)
        for col in ("authors", "venue", "url"):
            if not cur[col] and getattr(p, col):
                upd[col] = getattr(p, col)
        if len(p.abstract or "") > len(cur["abstract"] or ""):
            upd["abstract"] = p.abstract
        if p.published and (not cur["published"] or p.published < cur["published"]):
            upd["published"] = p.published
        srcs = sorted(set(filter(None, (cur["source"] + "," + p.source).split(","))))
        upd["source"] = ",".join(srcs)
        cats = sorted(set((cur["categories"] + " " + p.categories).split()))
        upd["categories"] = " ".join(cats)
        sets = ",".join(f"{k}=?" for k in upd)
        self.conn.execute(f"UPDATE papers SET {sets} WHERE paper_id=?", [*upd.values(), target])

    def _merge_rows(self, target: str, other: str) -> None:
        """Fold a duplicate row (discovered when a new record links two existing ones) into target."""
        o = self.conn.execute("SELECT * FROM papers WHERE paper_id=?", (other,)).fetchone()
        if o is None:
            return
        self._merge_fields(target, _row_to_paper(o))
        self.conn.execute("UPDATE OR IGNORE paper_keys SET paper_id=? WHERE paper_id=?", (target, other))
        self.conn.execute("UPDATE impressions SET paper_id=? WHERE paper_id=?", (target, other))
        has_label = self.conn.execute("SELECT 1 FROM labels WHERE paper_id=?", (target,)).fetchone()
        if not has_label:
            self.conn.execute("UPDATE labels SET paper_id=? WHERE paper_id=?", (target, other))
        self.conn.execute("DELETE FROM papers WHERE paper_id=?", (other,))

    def get_papers(self, ids: Iterable[str]) -> list[Paper]:
        ids = list(ids)
        out: dict[str, Paper] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self.conn.execute(f"SELECT * FROM papers WHERE paper_id IN ({marks})", chunk):
                out[r["paper_id"]] = _row_to_paper(r)
        return [out[i] for i in ids if i in out]

    def query_papers(self, where: str = "", params: Sequence = ()) -> list[Paper]:
        sql = "SELECT * FROM papers" + (f" WHERE {where}" if where else "") + " ORDER BY published, paper_id"
        return [_row_to_paper(r) for r in self.conn.execute(sql, params)]

    def count_papers(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def papers_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.conn.execute("SELECT source, COUNT(*) n FROM papers GROUP BY source"):
            for s in r["source"].split(","):
                counts[s] = counts.get(s, 0) + r["n"]
        return counts

    # ---- labels -------------------------------------------------------------------------
    def set_label(
        self, paper_id: str, label: int, tags: Sequence[str] = (), labeled_at: str | None = None,
        origin: str = "human", opened: bool | None = None,
        text_hash: str | None = None,
    ) -> None:
        """Store a judgement. Re-labelling keeps the original ``labeled_at`` (evaluation time)."""
        now = labeled_at or utcnow_iso()
        tag_str = " ".join(dict.fromkeys(tags))
        with self.conn:
            self.conn.execute(
                "INSERT INTO labels (paper_id,label,tags,labeled_at,updated_at,origin,opened,text_hash) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(paper_id) DO UPDATE SET label=excluded.label, tags=excluded.tags, "
                "updated_at=excluded.updated_at, origin=excluded.origin, "
                "opened=excluded.opened, text_hash=excluded.text_hash",
                (paper_id, int(label), tag_str, now, now, origin,
                 None if opened is None else int(bool(opened)), text_hash),
            )

    def labels_frame(self, origin: str | None = "human") -> pd.DataFrame:
        """One row per labelled paper: label, tags (list), labeled_at, first impression reason."""
        sql = (
            "SELECT l.paper_id, l.label, l.tags, l.labeled_at, "
            "(SELECT reason FROM impressions i WHERE i.paper_id=l.paper_id ORDER BY shown_at, id LIMIT 1) "
            "AS first_reason FROM labels l"
        )
        params: list = []
        if origin:
            sql += " WHERE l.origin=?"
            params.append(origin)
        sql += " ORDER BY l.labeled_at, l.paper_id"
        df = pd.read_sql_query(sql, self.conn, params=params)
        df["tags"] = df["tags"].map(lambda s: tuple(s.split()) if s else ())
        return df

    def delete_label(self, paper_id: str) -> None:
        """Undo a judgement: the paper goes back to the unlabelled queue."""
        with self.conn:
            self.conn.execute("DELETE FROM labels WHERE paper_id=?", (paper_id,))

    def labeled_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT paper_id FROM labels")}

    # ---- impressions --------------------------------------------------------------------
    def add_impressions(self, rows: Iterable[dict]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT INTO impressions (paper_id, shown_at, reason, score, rank, model_id, propensity, "
                "n_candidates, position) VALUES (:paper_id, :shown_at, :reason, :score, :rank, :model_id, "
                ":propensity, :n_candidates, :position)",
                [{"score": None, "rank": None, "model_id": None, "propensity": None, "n_candidates": None,
                  "position": None, **r} for r in rows],
            )

    def shown_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT DISTINCT paper_id FROM impressions")}

    def unlabeled_impressions(self) -> list[Paper]:
        rows = self.conn.execute(
            "SELECT p.* FROM papers p JOIN (SELECT paper_id, MIN(shown_at) s, MIN(id) i FROM impressions "
            "GROUP BY paper_id) im ON im.paper_id=p.paper_id LEFT JOIN labels l ON l.paper_id=p.paper_id "
            "WHERE l.paper_id IS NULL ORDER BY im.s, im.i"
        ).fetchall()
        return [_row_to_paper(r) for r in rows]

    # ---- embeddings ---------------------------------------------------------------------
    def get_embeddings(self, model_key: str, ids: Sequence[str]) -> dict[str, tuple[str, np.ndarray]]:
        out: dict[str, tuple[str, np.ndarray]] = {}
        for i in range(0, len(ids), 500):
            chunk = list(ids[i : i + 500])
            marks = ",".join("?" * len(chunk))
            for r in self.conn.execute(
                f"SELECT paper_id, text_hash, vec FROM embeddings WHERE model_key=? AND paper_id IN ({marks})",
                [model_key, *chunk],
            ):
                out[r["paper_id"]] = (r["text_hash"], np.frombuffer(r["vec"], dtype=np.float32))
        return out

    def put_embeddings(self, model_key: str, rows: Iterable[tuple[str, str, np.ndarray]]) -> None:
        data = [
            (pid, model_key, h, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
            for pid, h, v in rows
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (paper_id, model_key, text_hash, dim, vec) VALUES (?,?,?,?,?)",
                data,
            )

    # ---- key/value ----------------------------------------------------------------------
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def kv_set(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
