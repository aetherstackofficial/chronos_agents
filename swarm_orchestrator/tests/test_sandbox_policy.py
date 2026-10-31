"""Sandbox policy: sealed-or-nothing, the container recipe, the runner service.

No Docker is needed: the Docker CLI and the runner are replaced by fakes, and the
tests check what the code asks them to do. The real thing (containers, networks)
is exercised by hand against `docker compose up`.
"""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from runner import sandbox as sandbox_module
from runner.check import static_check
from runner.docker_backend import DockerSandbox, clean_name
from runner.remote import RemoteSandbox, RunnerError
from runner.sandbox import SandboxManager, SandboxUnavailable

GOOD = ("from runner.sdk import Strategy\n"
        "class UserStrategy(Strategy):\n"
        "    def on_tick(self, state):\n"
        "        return [{'action': 'BUY', 'qty': 5}] if state['rsi'] < 30 else []\n")


class FakeSealed:
    """Stands in for DockerSandbox / RemoteSandbox."""

    image = "fake-image"

    def __init__(self, running=True, launch_error=None):
        self.running = running
        self.launch_error = launch_error
        self.calls = []

    def available(self):
        return True

    def dry_run(self, code, ticks):
        self.calls.append(("dry_run", len(ticks)))
        return {"ok": True, "errors": [], "warnings": [], "orders": [{"action": "BUY"}],
                "ticks_tested": len(ticks), "output": ""}

    def launch(self, strategy_id, agent_id, name, code):
        self.calls.append(("launch", strategy_id, agent_id))
        if self.launch_error:
            raise self.launch_error
        return "container-123"

    def is_running(self, strategy_id):
        return self.running

    def stop(self, strategy_id):
        self.calls.append(("stop", strategy_id))
        return True

    def logs(self, strategy_id, tail=200):
        return "strategy output\n"

    def remove(self, strategy_id):
        self.calls.append(("remove", strategy_id))


@pytest.fixture
def manager(tmp_path):
    def make(sealed=None, allow_unsafe=False):
        return SandboxManager(base_dir=str(tmp_path / "strategies"), sealed=sealed,
                              allow_unsafe=allow_unsafe)
    return make


# --- the container recipe ----------------------------------------------------------
def test_launch_args_are_hardened_and_mount_nothing():
    box = DockerSandbox(image="chronos-sandbox:latest", network="chronos_strategies",
                        zmq_host="engine")
    args = box.launch_args("0a1b2c3d", "STRAT_0a1b2c3d", "My Strat!", "print(1)")
    joined = " ".join(args)
    for flag in ("--cap-drop ALL", "--read-only", "--memory 512m", "--memory-swap 512m",
                 "--cpus 1", "--pids-limit 64", "no-new-privileges", "--user 65534:65534",
                 "--network chronos_strategies"):
        assert flag in joined, flag
    assert "-v" not in args and "--volume" not in args and "--mount" not in args   # no host files
    assert "--privileged" not in args and "--network host" not in joined
    assert "ZMQ_HOST=engine" in args
    assert args[-4:] == ["python", "-m", "runner.sandbox_entry", "run"]


def test_launch_args_carry_only_the_strategy_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    monkeypatch.setenv("RUNNER_TOKEN", "another-secret")
    box = DockerSandbox()
    args = box.launch_args("0a1b2c3d", "STRAT_0a1b2c3d", "x", "print('hi')")
    assert "super-secret" not in " ".join(args) and "another-secret" not in " ".join(args)
    env = {a.split("=", 1)[0] for a in args if "=" in a and not a.startswith("--")}
    assert "STRATEGY_B64" in env and "GEMINI_API_KEY" not in env
    encoded = next(a for a in args if a.startswith("STRATEGY_B64=")).split("=", 1)[1]
    assert base64.b64decode(encoded).decode() == "print('hi')"


def test_dry_run_has_no_network_at_all():
    args = DockerSandbox().dryrun_args("chronos_dry_x", GOOD, [{"scenario": "calm"}])
    assert args[args.index("--network") + 1] == "none"
    assert "--rm" in args and "--cap-drop" in args and "--read-only" in args
    assert "runner.sandbox_entry" in args and args[-1] == "dryrun"


def test_clean_name_strips_odd_characters_and_limits_length():
    assert clean_name("a; rm -rf /") == "a rm -rf"[:48].strip() or clean_name("a; rm -rf /")
    assert ";" not in clean_name("a; rm -rf $(x)") and "$" not in clean_name("a; rm -rf $(x)")
    assert len(clean_name("x" * 200)) == 48 and clean_name("!!!") == "strategy"


def test_docker_backend_rejects_bad_ids_and_huge_code():
    box = DockerSandbox()
    with pytest.raises(ValueError):
        box.launch("../../etc", "STRAT_x", "n", GOOD)
    with pytest.raises(ValueError):
        box.launch("0a1b2c3d", "STRAT_0a1b2c3d", "n", "x = 1\n" * 40000)


# --- the policy: sealed or nothing -----------------------------------------------------
def test_bad_code_is_rejected_before_anything_else(manager):
    with pytest.raises(ValueError, match="line"):
        manager().launch("x", "class UserStrategy(object):\n    pass\n")


def test_no_sealed_backend_means_no_run(manager):
    mgr = manager(sealed=None, allow_unsafe=False)
    with pytest.raises(SandboxUnavailable, match="no sealed sandbox"):
        mgr.launch("x", GOOD)
    assert mgr.mode() == "unavailable"
    assert not list(mgr.base_dir.iterdir())               # nothing left behind


def test_check_without_sandbox_still_validates_but_skips_the_dry_run(manager):
    report = manager().check(GOOD)
    assert report["ok"] and report["dry_run"].startswith("skipped")
    assert any("sealed sandbox" in w["message"] for w in report["warnings"])
    broken = manager().check("class UserStrategy(object):\n    pass\n")
    assert not broken["ok"] and broken["errors"]


def test_static_warnings_are_not_repeated_when_the_runner_already_added_them(manager):
    class Echo(FakeSealed):
        def dry_run(self, code, ticks):           # the runner puts the static warnings in its report too
            report = super().dry_run(code, ticks)
            report["warnings"] = static_check(code)["warnings"]
            return report
    code = "import numpy\n" + GOOD
    warnings = manager(sealed=Echo()).check(code)["warnings"]
    assert len([w for w in warnings if "numpy" in w["message"]]) == 1


def test_sealed_launch_checks_then_launches_with_a_derived_agent_id(manager):
    fake = FakeSealed()
    mgr = manager(sealed=fake)
    result = mgr.launch("my strat", GOOD)
    assert result["agent_id"] == f"STRAT_{result['strategy_id']}" and result["backend"] == "docker"
    assert [c[0] for c in fake.calls] == ["dry_run", "launch"]       # check first, then launch
    assert fake.calls[0][1] == 3                                    # three sample ticks
    assert (mgr.base_dir / result["strategy_id"] / "strategy.py").read_text() == GOOD


def test_a_failed_sealed_launch_does_not_fall_back_to_unsafe(manager):
    fake = FakeSealed(launch_error=RuntimeError("docker run failed: no space"))
    mgr = manager(sealed=fake, allow_unsafe=False)
    with pytest.raises(SandboxUnavailable, match="could not start the sealed sandbox"):
        mgr.launch("x", GOOD)
    assert not list(mgr.base_dir.iterdir())                         # strategy dir cleaned up


def test_unsafe_mode_is_opt_in_and_runs_the_subprocess_path(manager, monkeypatch):
    mgr = manager(sealed=None, allow_unsafe=True)
    monkeypatch.setattr(mgr, "_dry_run_local", lambda code: {
        "ok": True, "errors": [], "warnings": [], "orders": [], "ticks_tested": 3, "output": ""})
    started = {}

    def fake_launch(entry, strategy_file, agent_id, name):
        started["agent"] = agent_id
        entry.update(backend="subprocess", pid=1)
    monkeypatch.setattr(mgr, "_launch_subprocess", fake_launch)
    result = mgr.launch("x", GOOD)
    assert result["backend"] == "subprocess" and started["agent"] == result["agent_id"]
    assert mgr.mode() == "unsafe-subprocess"


def test_subprocess_environment_withholds_secrets(manager, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    monkeypatch.setenv("RUNNER_TOKEN", "another-secret")
    monkeypatch.setenv("ZMQ_HOST", "engine")
    seen = {}

    class FakeProc:
        pid = 4242

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        seen["env"] = kw["env"]
        return FakeProc()
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(sandbox_module.threading.Thread, "start", lambda self: None)
    mgr = manager(sealed=None, allow_unsafe=True)
    strategy_file = tmp_path / "strategy.py"
    strategy_file.write_text(GOOD)
    mgr._launch_subprocess({"strategy_id": "ab12cd34"}, strategy_file, "STRAT_ab12cd34", "x")
    assert "GEMINI_API_KEY" not in seen["env"] and "RUNNER_TOKEN" not in seen["env"]
    assert seen["env"]["ZMQ_HOST"] == "engine"


# --- status and logs with a sealed backend ----------------------------------------------
def test_unknown_status_does_not_mark_a_live_strategy_exited(manager):
    fake = FakeSealed(running=None)                 # backend unreachable: is_running -> None
    mgr = manager(sealed=fake)
    sid = mgr.launch("x", GOOD)["strategy_id"]
    assert mgr.list()[0]["status"] == "running"
    assert ("remove", sid) not in fake.calls


def test_stopped_container_is_marked_exited_and_its_logs_are_kept(manager):
    fake = FakeSealed(running=False)
    mgr = manager(sealed=fake)
    sid = mgr.launch("x", GOOD)["strategy_id"]
    assert mgr.list()[0]["status"] == "exited"
    assert (mgr.base_dir / sid / "run.log").read_text() == "strategy output\n"
    assert ("remove", sid) in fake.calls


def test_stop_stops_saves_logs_and_removes(manager):
    fake = FakeSealed()
    mgr = manager(sealed=fake)
    sid = mgr.launch("x", GOOD)["strategy_id"]
    assert mgr.stop(sid) is True
    assert [c[0] for c in fake.calls][-2:] == ["stop", "remove"]
    assert mgr.logs(sid).startswith("strategy output")


# --- the runner service ---------------------------------------------------------------------
@pytest.fixture
def runner_client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from runner import service

    monkeypatch.setenv("RUNNER_TOKEN", "t0ken")
    fake = FakeSealed()
    monkeypatch.setattr(service, "sandbox", fake)
    return TestClient(service.app), fake


HEADERS = {"X-Runner-Token": "t0ken"}


def test_runner_requires_the_token(runner_client):
    client, _ = runner_client
    assert client.get("/health").status_code == 401
    assert client.post("/launch", json={"strategy_id": "0a1b2c3d", "code": GOOD}).status_code == 401
    assert client.get("/health", headers={"X-Runner-Token": "wrong"}).status_code == 401
    assert client.get("/health", headers=HEADERS).status_code == 200


def test_runner_refuses_when_no_token_is_configured(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from runner import service
    monkeypatch.delenv("RUNNER_TOKEN", raising=False)
    assert TestClient(service.app).get("/health", headers={"X-Runner-Token": ""}).status_code == 401


def test_runner_validates_ids_and_code_and_derives_the_agent_id(runner_client):
    client, fake = runner_client
    assert client.post("/launch", headers=HEADERS,
                       json={"strategy_id": "../../x", "code": GOOD}).status_code == 400
    assert client.post("/launch", headers=HEADERS, json={
        "strategy_id": "0a1b2c3d", "code": "class UserStrategy(object):\n    pass\n"}).status_code == 400
    ok = client.post("/launch", headers=HEADERS, json={
        "strategy_id": "0a1b2c3d", "name": "n", "code": GOOD, "agent_id": "STRAT_attacker"})
    assert ok.status_code == 200 and ok.json()["agent_id"] == "STRAT_0a1b2c3d"
    assert fake.calls[-1] == ("launch", "0a1b2c3d", "STRAT_0a1b2c3d")      # caller's agent_id ignored


def test_runner_check_runs_the_static_check_and_limits_ticks(runner_client):
    client, _ = runner_client
    broken = client.post("/check", headers=HEADERS, json={
        "code": "class UserStrategy(object):\n    pass\n", "ticks": [{}]}).json()
    assert broken["ok"] is False and broken["errors"]
    too_many = client.post("/check", headers=HEADERS, json={"code": GOOD, "ticks": [{}] * 9})
    assert too_many.status_code == 400
    ok = client.post("/check", headers=HEADERS, json={"code": GOOD, "ticks": [{"scenario": "calm"}]}).json()
    assert ok["ok"] is True


# --- the HTTP client -------------------------------------------------------------------------
@pytest.fixture
def fake_runner_server():
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("X-Runner-Token") != "t0ken":
                return self._send(401, {"detail": "bad or missing runner token"})
            if self.path == "/health":
                return self._send(200, {"status": "ok", "image_present": True})
            if self.path.startswith("/status/"):
                return self._send(200, {"running": True})
            self._send(404, {"detail": "nope"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.headers.get("X-Runner-Token") != "t0ken":
                return self._send(401, {"detail": "bad or missing runner token"})
            if self.path == "/launch":
                return self._send(200, {"container_id": "cid", "echo": body["strategy_id"]})
            self._send(400, {"detail": "rejected"})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_remote_sandbox_talks_to_the_runner(fake_runner_server):
    remote = RemoteSandbox(fake_runner_server, "t0ken")
    assert remote.available() is True
    assert remote.launch("0a1b2c3d", "STRAT_0a1b2c3d", "n", GOOD) == "cid"
    assert remote.is_running("0a1b2c3d") is True


def test_remote_sandbox_maps_errors(fake_runner_server):
    wrong = RemoteSandbox(fake_runner_server, "wrong")
    assert wrong.available() is False
    with pytest.raises(RunnerError, match="401"):
        wrong.launch("0a1b2c3d", "STRAT_0a1b2c3d", "n", GOOD)
    unreachable = RemoteSandbox("http://127.0.0.1:1", "t0ken")
    assert unreachable.is_running("0a1b2c3d") is None          # unknown, not "stopped"
    assert unreachable.dry_run(GOOD, [])["ok"] is False
