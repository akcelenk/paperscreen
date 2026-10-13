from paperscreen.db import Database
from paperscreen.paper import Paper

LONG = "A sufficiently long and unique paper title for matching purposes"


def test_same_arxiv_id_different_versions_is_one_row(db):
    a, c1 = db.upsert_paper(Paper(title="Foo bar", abstract="x", published="2026-01-01", source="arxiv", arxiv_id="2401.00001v1"))
    b, c2 = db.upsert_paper(Paper(title="Foo bar", abstract="longer abstract", published="2026-01-02", source="arxiv", arxiv_id="2401.00001v2"))
    assert a == b == "arxiv:2401.00001" and c1 and not c2
    assert db.count_papers() == 1
    p = db.get_papers([a])[0]
    assert p.abstract == "longer abstract" and p.published == "2026-01-01"


def test_preprint_and_journal_version_merge_via_doi_and_title(db):
    pid, _ = db.upsert_paper(Paper(title=LONG, abstract="a", published="2026-01-01", source="arxiv", arxiv_id="2401.00002"))
    pid2, created = db.upsert_paper(Paper(title=LONG + ".", abstract="a much longer abstract", published="2026-03-01",
                                          source="pubmed", pmid="123", doi="10.1/xyz"))
    assert pid2 == pid and not created
    p = db.get_papers([pid])[0]
    assert p.pmid == "123" and p.doi == "10.1/xyz" and p.arxiv_id == "2401.00002"
    assert p.source == "arxiv,pubmed" and p.published == "2026-01-01"
    # now the DOI alone also finds it
    pid3, created3 = db.upsert_paper(Paper(title="Different title altogether here", doi="10.1/xyz", source="openalex", openalex_id="W9"))
    assert pid3 == pid and not created3 and db.count_papers() == 1


def test_linking_record_merges_two_existing_rows_and_keeps_label(db):
    a, _ = db.upsert_paper(Paper(title="Alpha paper about one thing entirely", arxiv_id="2401.00003", published="2026-01-01", source="arxiv"))
    b, _ = db.upsert_paper(Paper(title="Beta paper about another thing entirely", doi="10.2/b", published="2026-01-02", source="openalex"))
    assert a != b and db.count_papers() == 2
    db.set_label(b, 1, ["t1"], "2026-01-05T10:00:00Z")
    t, created = db.upsert_paper(Paper(title="Gamma", arxiv_id="2401.00003", doi="10.2/b", source="openalex"))
    assert not created and db.count_papers() == 1
    lab = db.labels_frame()
    assert len(lab) == 1 and lab.iloc[0]["paper_id"] == t and lab.iloc[0]["label"] == 1


def test_short_titles_do_not_merge(db):
    db.upsert_paper(Paper(title="Attention", arxiv_id="2401.00004", source="arxiv"))
    _, created = db.upsert_paper(Paper(title="Attention", doi="10.3/c", source="openalex"))
    assert created and db.count_papers() == 2


def test_labels_keep_first_time_and_tags(db):
    pid, _ = db.upsert_paper(Paper(title="T", arxiv_id="2401.00005", source="arxiv"))
    db.set_label(pid, 1, ["a", "b"], "2026-02-01T10:00:00Z")
    db.set_label(pid, 0, [], "2026-02-09T10:00:00Z")
    row = db.labels_frame().iloc[0]
    assert row["label"] == 0 and row["labeled_at"] == "2026-02-01T10:00:00Z" and row["tags"] == ()


def test_first_reason_and_unlabeled_impressions(db):
    a, _ = db.upsert_paper(Paper(title="A", arxiv_id="2401.00006", source="arxiv"))
    b, _ = db.upsert_paper(Paper(title="B", arxiv_id="2401.00007", source="arxiv"))
    db.add_impressions([{"paper_id": a, "shown_at": "2026-02-01T09:00:00Z", "reason": "random"},
                        {"paper_id": b, "shown_at": "2026-02-01T09:00:00Z", "reason": "exploit"}])
    db.set_label(a, 1, ["x"])
    assert db.labels_frame().iloc[0]["first_reason"] == "random"
    assert [p.paper_id for p in db.unlabeled_impressions()] == [b]


def test_embedding_roundtrip(db):
    import numpy as np

    a, _ = db.upsert_paper(Paper(title="A", arxiv_id="2401.00008", source="arxiv"))
    db.put_embeddings("m|f", [(a, "h1", np.arange(4, dtype=np.float32))])
    got = db.get_embeddings("m|f", [a])
    assert got[a][0] == "h1" and got[a][1].tolist() == [0, 1, 2, 3]


def test_persistent_file(tmp_path):
    p = tmp_path / "sub" / "x.sqlite"
    d = Database(p)
    d.upsert_paper(Paper(title="T", arxiv_id="2401.00009", source="arxiv"))
    d.close()
    with Database(p) as d2:
        assert d2.count_papers() == 1
