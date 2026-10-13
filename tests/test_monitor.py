from datetime import date

import pytest
from click.testing import CliRunner

from paperscreen.cli import main
from paperscreen.monitor import build_monitor, newcombe_diff, render_monitor, wilson


def test_wilson_interval_known_values():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    p, lo, hi = wilson(0, 20)
    assert p == 0 and lo == 0 and 0.1 < hi < 0.2       # never a degenerate [0, 0] interval
    assert all(x != x for x in wilson(0, 0))


def test_newcombe_difference():
    d, lo, hi = newcombe_diff(56, 70, 48, 80)          # Newcombe (1998), example (a)
    assert d == pytest.approx(0.2) and lo == pytest.approx(0.0524, abs=2e-3) and hi == pytest.approx(0.3339, abs=2e-3)


def test_monitor_on_demo_stream(demo_db):
    db, last = demo_db
    m = build_monitor(db, today=last)
    p = m.pooled
    assert p["model_n"] > 0 and p["random_n"] > 0
    assert p["model_p"] > p["random_p"] and p["diff_lo"] > 0     # the demo's oracle picks beat random
    assert set(m.weekly["arm"]) == {"model", "random"}
    assert m.quality["source"].tolist() == ["synthetic"] and m.quality["days_since_newest"].iloc[0] == 0
    assert m.backlog == 0 and not m.alerts
    text = render_monitor(m)
    assert "Online quality" in text and "Alerts" in text


def test_monitor_alerts(demo_db):
    db, last = demo_db
    later = date.fromordinal(last.toordinal() + 30)
    m = build_monitor(db, today=later)
    assert any("synthetic: newest paper" in a for a in m.alerts)
    assert any("no judgements in the last 7 days" in a for a in m.alerts)


def test_monitor_and_models_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("db_path: db.sqlite\nembedding: {model: hash-256}\nmodel: {registry_dir: models}\n")
    r = CliRunner().invoke(main, ["models"])
    assert r.exit_code == 0 and "no saved models" in r.output
    r = CliRunner().invoke(main, ["monitor", "--out", "m.md"])
    assert r.exit_code == 0 and (tmp_path / "m.md").read_text().startswith("# paperscreen monitoring")


def test_monitor_alerts_on_truncated_fetch(demo_db):
    db, last = demo_db
    db.kv_set("truncated:arxiv", "2026-06-20")
    assert any("per-day cap" in a for a in build_monitor(db, today=last).alerts)
    db.kv_set("truncated:arxiv", "")
    assert not any("per-day cap" in a for a in build_monitor(db, today=last).alerts)
