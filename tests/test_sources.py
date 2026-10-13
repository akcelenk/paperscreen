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


def test_arxiv_queries_each_day_and_paginates():
    xml = (FIX / "arxiv_sample.xml").read_text()

    def handler(url, params):
        # Day 2026-10-09 has one full page (page_size=1) then an empty page; other days are empty.
        q, start = params["search_query"], params["start"]
        return FakeResponse(xml if "20261009" in q and start == 0 else EMPTY_FEED)

    s = FakeSession(handler=handler)
    src = ArxivSource(["cs.LG", "stat.ML"], http=_http(s), page_size=1)
    got = list(src.fetch(date(2026, 10, 8), date(2026, 10, 10)))
    assert [p.arxiv_id for p in got] == ["2610.13935"]
    queries = [c[1]["search_query"] for c in s.calls]
    assert queries[0] == "(cat:cs.LG OR cat:stat.ML) AND submittedDate:[202610100000 TO 202610102359]"
    days = [q.split("[")[1][:8] for q in queries]
    assert days == ["20261010", "20261009", "20261009", "20261008"]  # newest day first, day 9 paged twice
    assert src.truncated_days == []


def test_arxiv_records_days_that_hit_the_cap(db):
    xml = (FIX / "arxiv_sample.xml").read_text()
    s = FakeSession(handler=lambda url, params: FakeResponse(xml))  # every page is full
    src = ArxivSource(["cs.LG"], http=_http(s), page_size=1, max_per_day=3)
    list(src.fetch(date(2026, 10, 9), date(2026, 10, 9)))
    assert src.truncated_days == ["2026-10-09"] and len(s.calls) == 3
    (res,) = fetch_all(db, [src], days=0, today=date(2026, 10, 9))
    assert res.warning and "2026-10-09" in res.warning
    assert db.kv_get("truncated:arxiv") == "2026-10-09"


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


class _DaysSource(Source):
    name = "days"

    def __init__(self):
        self.calls = []

    def fetch(self, since, until=None):
        self.calls.append(since)
        return iter(())


def test_long_break_catches_up_instead_of_skipping_days(db):
    src = _DaysSource()
    db.kv_set("watermark:days", "2026-08-01")
    fetch_all(db, [src], today=date(2026, 9, 10))  # 40 days away: everything since the mark (minus overlap)
    assert src.calls[-1] == date(2026, 7, 29)
    fetch_all(db, [src], today=date(2026, 12, 1), max_catchup_days=60)  # but never more than the cap
    assert src.calls[-1] == date(2026, 10, 2)
