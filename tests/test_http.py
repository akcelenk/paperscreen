import pytest
from conftest import FakeResponse, FakeSession

from paperscreen.http import HttpClient, HttpError


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, d):
        self.sleeps.append(d)
        self.t += d


def test_retries_then_succeeds():
    clk = Clock()
    s = FakeSession([FakeResponse(status=503), FakeResponse("ok")])
    c = HttpClient(session=s, sleep=clk.sleep, clock=clk.now)
    assert c.get("https://retry.test/x").text == "ok"
    assert clk.sleeps == [1.0]


def test_gives_up_after_retries():
    clk = Clock()
    s = FakeSession([FakeResponse(status=500)] * 3)
    c = HttpClient(session=s, retries=2, sleep=clk.sleep, clock=clk.now)
    with pytest.raises(HttpError):
        c.get("https://giveup.test/x")
    assert len(s.calls) == 3


def test_client_error_is_not_retried():
    s = FakeSession([FakeResponse("nope", status=404)])
    c = HttpClient(session=s, sleep=lambda d: None)
    with pytest.raises(HttpError):
        c.get("https://e404.test/x")
    assert len(s.calls) == 1


def test_long_retry_after_fails_fast():
    s = FakeSession([FakeResponse(status=429, headers={"Retry-After": "59000"})])
    c = HttpClient(session=s, sleep=lambda d: None)
    with pytest.raises(HttpError, match="Retry-After"):
        c.get("https://quota.test/x")
    assert len(s.calls) == 1


def test_min_interval_is_enforced_per_host():
    clk = Clock()
    s = FakeSession([FakeResponse("a"), FakeResponse("b"), FakeResponse("c")])
    c = HttpClient(min_interval=3.0, session=s, sleep=clk.sleep, clock=clk.now)
    c.get("https://rate.test/1")
    c.get("https://rate.test/2")
    assert clk.sleeps == [3.0]
    c.get("https://other-host.test/3")  # different host: no wait
    assert clk.sleeps == [3.0]
