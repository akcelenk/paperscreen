from click.testing import CliRunner
from conftest import labels_of

from paperscreen.cli import main
from paperscreen.db import Database
from paperscreen.paper import Paper


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
        "db_path: db.sqlite\nembedding: {model: hash-256}\ndigest: {out_dir: digests}\n"
        "evaluation: {out_dir: reports}\n"
    )


def test_demo_runs_end_to_end(tmp_path):
    r = CliRunner().invoke(main, ["demo", "--workdir", str(tmp_path / "w"), "--days", "40"])
    assert r.exit_code == 0, r.output
    reports = list((tmp_path / "w" / "reports").glob("*/report.md"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert "Primary comparisons" in text and "Label-efficiency" in text
    assert list((tmp_path / "w" / "digests").glob("*.md"))


def test_label_flow_and_stats(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    db = Database("db.sqlite")
    ids = []
    for i in range(2):
        pid, _ = db.upsert_paper(Paper(title=f"Paper {i} on gradient descent convergence", abstract="adam sgd loss",
                                       published="2026-10-12", source="arxiv", arxiv_id=f"2610.0000{i}"))
        ids.append(pid)
    db.add_impressions([{"paper_id": p, "shown_at": "2026-10-13T08:00:00Z", "reason": "exploit"} for p in ids])
    db.close()
    r = CliRunner().invoke(main, ["label"], input="y\n1\nn\n")
    assert r.exit_code == 0, r.output
    lab = labels_of("db.sqlite").set_index("paper_id")
    assert lab.loc[ids[0], "label"] == 1 and lab.loc[ids[0], "tags"] == ("optimization",)
    assert lab.loc[ids[1], "label"] == 0 and lab.loc[ids[1], "tags"] == ()
    out = CliRunner().invoke(main, ["stats"]).output
    assert "labels: 2 (1 positive)" in out and "exploit=2" in out


def test_label_id_validates(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    db = Database("db.sqlite")
    pid, _ = db.upsert_paper(Paper(title="T", arxiv_id="2610.00009", source="arxiv"))
    db.close()
    ok = CliRunner().invoke(main, ["label-id", pid, "--yes", "--tags", "optimization,mech_interp"])
    assert ok.exit_code == 0 and labels_of("db.sqlite").iloc[0]["tags"] == ("optimization", "mech_interp")
    assert CliRunner().invoke(main, ["label-id", "nope"]).exit_code != 0
    assert CliRunner().invoke(main, ["label-id", pid, "--tags", "bogus"]).exit_code != 0


def test_init_and_fetch_without_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    r = CliRunner().invoke(main, ["init", "--write-config"])
    assert r.exit_code == 0 and (tmp_path / "config.yaml").exists()
    assert CliRunner().invoke(main, ["init", "--write-config"]).exit_code != 0  # never overwrite
    f = CliRunner().invoke(main, ["fetch", "--source", "openalex"])
    assert f.exit_code == 0 and "OPENALEX_API_KEY" in f.output


def test_report_with_no_labels(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    r = CliRunner().invoke(main, ["report"])
    assert r.exit_code == 0 and "not enough labels" in r.output
