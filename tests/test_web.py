import http.client
import json
import threading
import time

import pytest
from conftest import labels_of

from paperscreen import pipeline
from paperscreen.db import Database
from paperscreen.paper import Paper
from paperscreen.webui import make_server, md_to_html


@pytest.fixture
def server(tmp_path, cfg):
    c = cfg.with_overrides(db_path=str(tmp_path / "w.sqlite"), evaluation={"out_dir": str(tmp_path / "reports")})
    db = Database(c.db_path)
    for i in range(3):
        pid, _ = db.upsert_paper(Paper(title=f"Web paper {i} about {c.interests[0].description[:30]}",
                                       abstract=c.interests[0].description, published="2026-10-12", source="arxiv",
                                       arxiv_id=f"2610.0100{i}", url=f"https://arxiv.org/abs/2610.0100{i}"))
        db.add_impressions([{"paper_id": pid, "shown_at": "2026-10-13T08:00:00Z", "reason": "random",
                             "score": 0.9, "model_id": "m"}])
    db.close()
    srv = make_server(c, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], c
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = dict(headers or {})
    if host:
        h["Host"] = host
    payload = json.dumps(body) if body is not None else None
    if payload is not None:
        h.setdefault("Content-Type", "application/json")
    conn.request(method, path, payload, h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, (json.loads(data) if r.getheader("Content-Type", "").startswith("application/json") else data)


POST = {"X-Requested-With": "paperscreen"}


def test_page_and_blind_items(server):
    port, _ = server
    st, page = call(port, "GET", "/")
    assert st == 200 and b"paperscreen" in page
    st, items = call(port, "GET", "/api/items")
    assert st == 200 and len(items) == 3
    assert not ({"score", "reason", "rank", "model_id"} & set(items[0]))  # judging stays blind
    assert items[0]["suggest"] in {i.id for i in server[1].interests}


def test_label_undo_and_state(server):
    port, cfg = server
    _, items = call(port, "GET", "/api/items")
    pid = items[0]["id"]
    tag = cfg.interests[0].id
    assert call(port, "POST", "/api/label", {"id": pid, "label": 1, "tags": [tag, "bogus"]}, POST)[0] == 200
    st, state = call(port, "GET", "/api/state")
    assert state["labels"] == 1 and state["positives"] == 1 and state["unlabeled"] == 2
    lab = labels_of(cfg.db_path).iloc[0]
    assert lab["tags"] == (tag,)  # unknown tags are dropped
    call(port, "POST", "/api/unlabel", {"id": pid}, POST)
    assert call(port, "GET", "/api/state")[1]["unlabeled"] == 3
    assert call(port, "POST", "/api/label", {"id": pid, "label": 7}, POST)[0] == 400
    assert call(port, "POST", "/api/label", {"id": "nope", "label": 1}, POST)[0] == 400


def test_post_requires_header_and_host_is_checked(server):
    port, _ = server
    _, items = call(port, "GET", "/api/items")
    body = {"id": items[0]["id"], "label": 0}
    assert call(port, "POST", "/api/label", body)[0] == 403  # no custom header: blocked
    assert call(port, "POST", "/api/label", body, POST, host="evil.example.com")[0] == 403
    assert call(port, "GET", "/api/state", host="evil.example.com:8765")[0] == 403
    assert call(port, "GET", "/api/state", host="localhost:8765")[0] == 200


def test_background_job_runs_once_at_a_time(server, monkeypatch):
    port, _ = server
    gate = threading.Event()

    def fake(cfg, db, ctx, days=None, log=print, today=None):
        log("fake fetch")
        gate.wait(5)
        log("fake done")

    monkeypatch.setattr(pipeline, "refresh", fake)
    assert call(port, "POST", "/api/run", {"action": "refresh"}, POST)[1] == {"started": True}
    assert call(port, "POST", "/api/run", {"action": "refresh"}, POST)[1] == {"started": False}
    assert call(port, "GET", "/api/job")[1]["running"] is True
    gate.set()
    for _ in range(50):
        job = call(port, "GET", "/api/job")[1]
        if not job["running"]:
            break
        time.sleep(0.1)
    assert job["done"] and job["error"] is None and "fake done" in job["log"]
    assert call(port, "POST", "/api/run", {"action": "nope"}, POST)[0] == 400


def test_failed_job_reports_error(server, monkeypatch):
    port, _ = server

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(pipeline, "refresh", boom)
    call(port, "POST", "/api/run", {"action": "refresh"}, POST)
    for _ in range(50):
        job = call(port, "GET", "/api/job")[1]
        if not job["running"]:
            break
        time.sleep(0.1)
    assert job["error"] and "network down" in job["error"]


def test_report_endpoint_escapes_markup(server, tmp_path):
    port, cfg = server
    assert call(port, "GET", "/api/report")[1] == {"html": None}
    d = tmp_path / "reports" / "2026-10-13_x"
    d.mkdir(parents=True)
    (d / "report.md").write_text("# Title\n\n| a | b |\n|---|---|\n| <script>x</script> | 1 |\n")
    st, rep = call(port, "GET", "/api/report")
    assert "<script>" not in rep["html"] and "&lt;script&gt;" in rep["html"] and "<table>" in rep["html"]


def test_md_to_html_basics():
    out = md_to_html("## Head\n\n> note\n\nSome `code` and **bold** <b>")
    assert "<h3>Head</h3>" in out and "<blockquote>" in out and "<code>code</code>" in out
    assert "<b>" not in out and "&lt;b&gt;" in out


def test_monitor_endpoint(server):
    port, _ = server
    status, body = call(port, "GET", "/api/monitor")
    assert status == 200 and "Online quality" in body["html"] and "<script" not in body["html"]


def test_md_to_html_italics_but_not_snake_case():
    out = md_to_html("_(empty)_ and no_abstract_% and `a_b_c`")
    assert "<em>(empty)</em>" in out and "no_abstract_%" in out and "<code>a_b_c</code>" in out
