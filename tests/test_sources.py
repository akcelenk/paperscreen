import json
from datetime import date
from pathlib import Path

import pytest
from conftest import FakeResponse, FakeSession

from paperscreen.http import HttpClient
from paperscreen.sources import fetch_all
from paperscreen.sources.arxiv import ArxivSource, import_kaggle_jsonl, parse_atom
from paperscreen.sources.base import Source, SourceError
from paperscreen.sources.openalex import OpenAlexSource, parse_work, rebuild_abstract
from paperscreen.sources.pubmed import PubMedSource, parse_pubmed_xml

FIX = Path(__file__).parent / "fixtures"
EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _http(session):
    return HttpClient(session=session, sleep=lambda d: None)


def test_parse_arxiv_fixture():
    (p,) = parse_atom((FIX / "arxiv_sample.xml").read_text())
    assert p.arxiv_id == "2610.13935" and p.published == "2026-10-09"
    assert p.categories.split()[0] == "cs.CV" and p.abstract and p.source == "arxiv"
    assert p.url.endswith("2610.13935")


def test_arxiv_source_stops_at_since_and_paginates():
    xml = (FIX / "arxiv_sample.xml").read_text()
    s = FakeSession([FakeResponse(xml), FakeResponse(EMPTY_FEED)])
    got = list(ArxivSource(["cs.LG"], http=_http(s), page_size=1).fetch(date(2026, 10, 1)))
    assert len(got) == 1 and len(s.calls) == 2
    s2 = FakeSession([FakeResponse(xml)])
    assert list(ArxivSource(["cs.LG"], http=_http(s2)).fetch(date(2026, 10, 10))) == []  # all older than since
    q = s.calls[0][1]["search_query"]
    assert q == "cat:cs.LG" and s.calls[0][1]["sortBy"] == "submittedDate"


def test_parse_pubmed_fixture():
    (p,) = parse_pubmed_xml((FIX / "pubmed_sample.xml").read_text())
    assert p.pmid == "42863208" and p.doi == "10.1016/j.neunet.2026.109725"
    assert p.published == "2026-10-08" and "optimization" in p.title and len(p.abstract) > 200
    assert parse_pubmed_xml((FIX / "pubmed_sample.xml").read_text().replace("<Abstract>", "<X>").replace("</Abstract>", "</X>")) == []


def test_pubmed_source_two_step():
    xml = (FIX / "pubmed_sample.xml").read_text()

    def handler(url, params):
        if "esearch" in url:
            return FakeResponse(json_data={"esearchresult": {"idlist": ["42863208"]}})
        assert "efetch" in url and params["id"] == "42863208"
        return FakeResponse(xml)

    s = FakeSession(handler=handler)
    got = list(PubMedSource(["q1", "q2"], http=_http(s)).fetch(date(2026, 10, 1), date(2026, 10, 13)))
    assert [p.pmid for p in got] == ["42863208"]  # second query's duplicate id is not refetched
    assert sum("esearch" in u for u, _ in s.calls) == 2 and sum("efetch" in u for u, _ in s.calls) == 1


def test_rebuild_abstract_orders_by_position():
    assert rebuild_abstract({"b": [1], "a": [0], "c": [2, 3]}) == "a b c c"
    assert rebuild_abstract(None) == ""


def test_parse_openalex_fixture():
    data = json.loads((FIX / "openalex_sample.json").read_text())
    ps = [parse_work(w) for w in data["results"]]
    assert ps[0].doi == "10.1016/j.neunet.2026.109725" and ps[0].pmid == "42863208"
    assert ps[0].openalex_id == "W111" and ps[0].venue == "Neural Networks"
    assert ps[1].arxiv_id == "2410.01234"  # recovered from the arXiv-issued DOI
    assert ps[2].abstract == ""


def test_openalex_needs_key_and_filters_empty_abstracts():
    with pytest.raises(SourceError, match="OPENALEX_API_KEY"):
        OpenAlexSource(["q"], api_key=None)
    data = json.loads((FIX / "openalex_sample.json").read_text())
    s = FakeSession([FakeResponse(json_data=data)])
    got = list(OpenAlexSource(["q"], api_key="k", http=_http(s)).fetch(date(2026, 10, 1), date(2026, 10, 13)))
    assert [p.openalex_id for p in got] == ["W111", "W222"]
    assert s.calls[0][1]["api_key"] == "k" and "has_abstract:true" in s.calls[0][1]["filter"]


def test_cross_source_dedupe(db):
    pm = parse_pubmed_xml((FIX / "pubmed_sample.xml").read_text())[0]
    oa = parse_work(json.loads((FIX / "openalex_sample.json").read_text())["results"][0])
    a, c1 = db.upsert_paper(pm)
    b, c2 = db.upsert_paper(oa)
    assert a == b and c1 and not c2 and db.count_papers() == 1
    p = db.get_papers([a])[0]
    assert p.source == "openalex,pubmed" and p.openalex_id == "W111" and p.pmid == "42863208"


class _Boom(Source):
    name = "boom"

    def fetch(self, since, until=None):
        raise RuntimeError("nope")
        yield  # pragma: no cover


class _One(Source):
    name = "one"

    def fetch(self, since, until=None):
        from paperscreen.paper import Paper

        yield Paper(title="Only paper", arxiv_id="2401.00100", published="", source="one")


def test_fetch_all_isolates_errors_and_sets_watermark(db):
    res = fetch_all(db, [_Boom(), _One()], today=date(2026, 10, 13))
    by = {r.source: r for r in res}
    assert by["boom"].error and by["one"].new == 1
    assert db.kv_get("watermark:one") == "2026-10-13" and db.kv_get("watermark:boom") is None
    assert db.get_papers(["arxiv:2401.00100"])[0].published == "2026-10-13"  # missing date filled


def test_kaggle_import(tmp_path):
    lines = [
        {"id": "2401.00001", "title": "In   scope", "abstract": "abs", "categories": "cs.LG math.OC",
         "update_date": "2024-02-01", "versions": [{"created": "Mon, 1 Jan 2024 10:00:00 GMT"}], "authors": "A"},
        {"id": "2401.00002", "title": "Out of scope", "abstract": "abs", "categories": "physics.optics",
         "update_date": "2024-02-01", "versions": []},
        {"id": "2301.00003", "title": "Too old", "abstract": "abs", "categories": "cs.LG",
         "update_date": "2023-02-01", "versions": [{"created": "Sun, 1 Jan 2023 10:00:00 GMT"}]},
    ]
    f = tmp_path / "k.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines))
    got = list(import_kaggle_jsonl(f, since=date(2024, 1, 1), categories=["cs.LG"]))
    assert [p.arxiv_id for p in got] == ["2401.00001"] and got[0].published == "2024-01-01"
    assert got[0].title == "In scope"
