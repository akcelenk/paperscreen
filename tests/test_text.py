import pytest

from paperscreen.paper import Paper
from paperscreen.text import arxiv_id_from_doi, normalize_arxiv_id, normalize_doi, paper_text, title_key, to_iso_date


def test_arxiv_ids():
    assert normalize_arxiv_id("http://arxiv.org/abs/2401.01234v3") == "2401.01234"
    assert normalize_arxiv_id("arXiv:2401.01234") == "2401.01234"
    assert normalize_arxiv_id("hep-th/9901001v2") == "hep-th/9901001"
    assert normalize_arxiv_id("not an id") is None
    assert normalize_arxiv_id(None) is None


def test_doi_and_arxiv_doi():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi: 10.1000/x") == "10.1000/x"
    assert normalize_doi("foo") is None
    assert arxiv_id_from_doi("10.48550/arXiv.2401.01234") == "2401.01234"
    assert arxiv_id_from_doi("10.1000/abc") is None


def test_title_key_needs_length():
    assert title_key("Short title") is None
    assert title_key("A Sufficiently Long Paper Title, Really!") == "asufficientlylongpapertitlereally"


def test_dates():
    assert to_iso_date("2026-10-08T12:00:00Z") == "2026-10-08"
    assert to_iso_date("garbage") == ""


def test_paper_text_formats():
    p = Paper(title="T  x", abstract="A\n b")
    assert paper_text(p, "title") == "T x"
    assert paper_text(p, "title_abstract") == "T x. A b"
    assert paper_text(p, "title_sep_abstract") == "T x [SEP] A b"
    with pytest.raises(ValueError):
        paper_text(p, "nope")
