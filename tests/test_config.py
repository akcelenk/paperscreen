import copy

import pytest
from click.testing import CliRunner

from paperscreen.cli import main
from paperscreen.config import ConfigError, load_config, validate


def test_default_config_is_valid():
    validate(load_config(None).raw)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["model"].update(ranker="emb_logrg"), "model.ranker"),
    (lambda r: r["model"]["params"].update(V=1.5), "model.params.V"),
    (lambda r: r["model"]["params"].update(n_random=10.5), "model.params.n_random"),
    (lambda r: r["embedding"].update(model="bge"), "embedding.model"),
    (lambda r: r["digest"].update(per_interest=0), "digest.per_interest"),
    (lambda r: r["evaluation"].update(test_fraction=1.0), "evaluation.test_fraction"),
    (lambda r: r["digest"].update(proxy_prefix="http://proxy?url="), "proxy_prefix"),
    (lambda r: r.update(interests=[]), "interests"),
    (lambda r: r["interests"][0].pop("description"), "interests[0].description"),
    (lambda r: r["evaluation"].update(step_days="7"), "evaluation.step_days"),
])
def test_invalid_values_are_reported(mutate, message):
    raw = copy.deepcopy(load_config(None).raw)
    mutate(raw)
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate(raw)


def test_cli_reports_config_errors_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("model: {ranker: nope}\n")
    r = CliRunner().invoke(main, ["stats"])
    assert r.exit_code == 1 and "model.ranker" in r.output and "Traceback" not in r.output
