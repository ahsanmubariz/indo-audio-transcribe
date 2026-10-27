"""t14 — safe to leave running, easy to start.

An unauthenticated service must never be reachable from another device, so it
binds loopback. Port 3080 is taken on this machine by another service, so the
default must not be 3080 and must be configurable. And a server that cannot load
its model must say so and stop, rather than start up and fail at the first upload.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTranscriber

from mytranscribe import server
from mytranscribe.app import create_app
from mytranscribe.config import DEFAULT_PORT, AppConfig


# ------------------------------------------------------------------- binding

def test_the_default_host_is_loopback_only() -> None:
    """Not `0.0.0.0`: this app has no authentication."""
    config = AppConfig.from_env({})

    assert config.host == "127.0.0.1"


def test_the_default_port_is_configurable_and_not_3080() -> None:
    assert DEFAULT_PORT != 3080
    assert AppConfig.from_env({}).port == DEFAULT_PORT
    assert AppConfig.from_env({"MYTRANSCRIBE_PORT": "9123"}).port == 9123
    assert AppConfig.from_env({"PORT": "9124"}).port == 9124


def test_the_config_reads_the_environment_for_every_knob() -> None:
    config = AppConfig.from_env(
        {
            "MYTRANSCRIBE_HOST": "127.0.0.1",
            "MYTRANSCRIBE_PORT": "8899",
            "MYTRANSCRIBE_THREADS": "6",
            "MYTRANSCRIBE_RETENTION_SECONDS": "120",
            "MYTRANSCRIBE_WORK_ROOT": "/tmp/somewhere",
        }
    )

    assert config.port == 8899
    assert config.threads == 6
    assert config.retention_seconds == 120.0
    assert config.work_root == Path("/tmp/somewhere")


def test_the_server_binds_loopback_by_default_end_to_end(work_dir: Path) -> None:
    """Start the real server on a real port and check where it is listening."""
    import subprocess
    import sys
    import time

    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mytranscribe",
            "--port",
            str(port),
            "--no-model",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        reachable = False
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    reachable = True
                    break
            except OSError:
                time.sleep(0.1)
        assert reachable, "the server never started listening"

        listening = _listening_addresses(port)
        assert listening, "could not read the listening socket"
        # Every listening address must be loopback.
        for address in listening:
            assert address in {"127.0.0.1", "::1"}, f"bound to {address}, not loopback"
    finally:
        process.terminate()
        process.wait(timeout=20)


# ---------------------------------------------------------------- health flag

def test_health_distinguishes_model_loaded(work_dir: Path) -> None:
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=FakeTranscriber(), model_loaded=True)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "ok"


def test_health_distinguishes_model_missing(work_dir: Path) -> None:
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=None, model_loaded=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["model_loaded"] is False
    assert response.json()["status"] == "degraded"


# --------------------------------------------------------------- startup fail

def test_startup_fails_with_a_clear_message_when_the_model_cannot_load(monkeypatch, capsys) -> None:
    def explode(config):
        raise server.ModelUnavailable(
            "The model PixelML/sea-asr-id could not be loaded, so the app will not start."
        )

    monkeypatch.setattr(server, "load_transcriber", explode)

    exit_code = server.main(["--port", "8898"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "could not be loaded" in captured.err
    assert "will not start" in captured.err
    # A person must be able to read it: no traceback in the startup failure path.
    assert "Traceback" not in captured.err


def test_a_gated_model_failure_explains_how_to_fix_it() -> None:
    error = RuntimeError("401 Client Error: Unauthorized for url (gated model)")

    message = server._model_advice("PixelML/sea-asr-id", error)

    assert "hf auth login" in message
    assert "https://huggingface.co/google/embeddinggemma-2" in message
    assert "terms" in message.lower()
    assert "401" in message


def test_an_unexpected_model_failure_is_still_plain() -> None:
    message = server._model_advice("PixelML/sea-asr-id", RuntimeError("boom"))

    assert "boom" in message
    assert "will not start" in message
    assert "Traceback" not in message


# -------------------------------------------------------------------- logging

def test_startup_logging_states_port_threads_and_model(monkeypatch, capsys) -> None:
    """All three must be visible in the log, so a misconfiguration is obvious."""
    logged: list[str] = []
    monkeypatch.setattr(server, "_log", lambda message: logged.append(message))
    monkeypatch.setattr(server, "load_transcriber", lambda config: FakeTranscriber())

    runner_calls: dict = {}

    def fake_run(application, host, port, log_level):
        runner_calls.update(host=host, port=port)
        raise SystemExit(0)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    with pytest.raises(SystemExit):
        server.main(["--port", "8901", "--threads", "6"])

    joined = "\n".join(logged)
    assert "8901" in joined
    assert "threads=6" in joined
    assert "model=loaded" in joined
    assert runner_calls == {"host": "127.0.0.1", "port": 8901}


def test_the_thread_count_reaches_torch(monkeypatch) -> None:
    """The configured thread count must actually be applied, not just logged."""
    import torch

    seen: dict = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda value: seen.update(threads=value))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)

    class FakeSeaASR:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return FakeTranscriber()

    import sea_asr

    monkeypatch.setattr(sea_asr, "SeaASR", FakeSeaASR)

    server.load_transcriber(AppConfig(threads=6))

    assert seen["threads"] == 6


# ------------------------------------------------------------------- process

def test_a_managed_background_process_survives_its_parent_shell(tmp_path: Path) -> None:
    """The app is started detached, so closing the terminal does not kill it."""
    import subprocess
    import sys
    import time

    port = _free_port()
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "mytranscribe", "--port", str(port), "--no-model"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # A new session detaches it from the shell's process group.
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise AssertionError("the detached server never started")

        # It is in its own session, which is what makes it survive a closed terminal.
        import os

        assert os.getsid(process.pid) == process.pid
    finally:
        process.terminate()
        process.wait(timeout=20)


def test_no_port_collision_with_the_service_on_3080() -> None:
    """Whatever is on 3080, the app must not want it."""
    assert DEFAULT_PORT != 3080
    text = (Path(__file__).resolve().parents[1] / "mytranscribe" / "config.py").read_text()
    assert not re.search(r"=\s*3080\b", text)


# ------------------------------------------------------------------- helpers

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _listening_addresses(port: int) -> set[str]:
    """The addresses a port is bound to, read from /proc.

    The port was free when the test chose it, so any listener on it belongs to
    the server under test.
    """
    found: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 4 or fields[3] != "0A":  # 0A is TCP_LISTEN
                continue
            host_hex, port_hex = fields[1].rsplit(":", 1)
            if int(port_hex, 16) != port:
                continue
            found.add(_hex_to_ip(host_hex))
    return found


def _hex_to_ip(host_hex: str) -> str:
    if len(host_hex) == 8:
        # /proc prints each 32-bit word in host byte order.
        octets = [int(host_hex[i : i + 2], 16) for i in (6, 4, 2, 0)]
        return ".".join(str(octet) for octet in octets)
    if host_hex in {"0" * 31 + "1", "00000000000000000000000001000000"}:
        return "::1"
    return host_hex


# ------------------------------------------- uploads when the model is missing

def test_an_upload_is_refused_when_the_worker_cannot_transcribe(work_dir: Path) -> None:
    """Story 21: 'broken' and 'busy' are different answers, and neither is silence.

    Accepting an upload with no model queues it forever and leaves its audio on
    disk — the Job never moves and nothing ever says why.
    """
    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=None, model_loaded=False)

    with TestClient(app) as client:
        from media import tone_wav

        audio = tone_wav(work_dir / "note.wav", seconds=2.0)
        with audio.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                files={"file": ("note.wav", handle, "audio/wav")},
                data={"format": "txt"},
            )

        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert "model" in detail.lower()
        assert "Traceback" not in detail
        # No Job, and no temp audio left behind.
        assert client.get("/api/jobs").json()["jobs"] == []
        leftovers = [p for p in config.work_root.rglob("*") if p.is_file()]
        assert leftovers == [], f"left behind: {leftovers}"


def test_a_busy_worker_still_accepts_uploads(work_dir: Path) -> None:
    """'Busy' must not be refused: a Queue is the whole point of being busy."""
    from fakes import FakeTranscriber
    from media import tone_wav

    config = AppConfig(work_root=work_dir / "jobs", port=0)
    app = create_app(config, transcriber=FakeTranscriber(per_piece_seconds=0.2), model_loaded=True)

    with TestClient(app) as client:
        audio = tone_wav(work_dir / "note.wav", seconds=40.0)
        for index in range(3):
            with audio.open("rb") as handle:
                response = client.post(
                    "/api/jobs",
                    files={"file": (f"berkas{index}.wav", handle, "audio/wav")},
                    data={"format": "txt"},
                )
            assert response.status_code == 201
        states = [job["state"] for job in client.get("/api/jobs").json()["jobs"]]
        assert states.count("queued") == 2


# ------------------------------------------------- the configured model is used

def test_the_configured_model_appears_in_the_job_and_the_snapshot(work_dir: Path) -> None:
    """A documented knob that nothing reads is a lie in the README."""
    config = AppConfig(work_root=work_dir / "jobs", port=0, model_id="someone/other-model")
    app = create_app(config, transcriber=FakeTranscriber(), model_loaded=True)

    with TestClient(app) as client:
        from media import tone_wav

        audio = tone_wav(work_dir / "note.wav", seconds=20.0)
        with audio.open("rb") as handle:
            response = client.post(
                "/api/jobs",
                files={"file": ("note.wav", handle, "audio/wav")},
                data={"format": "json"},
            )
        job = response.json()
        assert job["model"] == "someone/other-model"
        assert client.get("/health").json()["model"] == "someone/other-model"


def test_the_job_passes_its_own_model_to_the_pipeline(work_dir: Path) -> None:
    """The Worker must not substitute a hardcoded model for the configured one."""
    from mytranscribe import pipeline

    seen: list[str] = []
    original = pipeline.transcribe_audio

    def spy(request, **kwargs):
        seen.append(request.model)
        return original(request, **kwargs)

    config = AppConfig(work_root=work_dir / "jobs", port=0, model_id="someone/other-model")
    app = create_app(config, transcriber=FakeTranscriber(), model_loaded=True)

    import mytranscribe.worker as worker_module

    worker_module.transcribe_audio = spy
    try:
        with TestClient(app) as client:
            from media import tone_wav

            audio = tone_wav(work_dir / "note.wav", seconds=20.0)
            with audio.open("rb") as handle:
                client.post(
                    "/api/jobs",
                    files={"file": ("note.wav", handle, "audio/wav")},
                    data={"format": "txt"},
                )
            import time

            # 20 s of polling, not 8. The Worker runs a real Piece loop against a
            # tone, and on a loaded machine 8 s was not always enough — the test
            # then read an empty `seen` and failed for a reason unrelated to what
            # it asserts.
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                jobs = client.get("/api/jobs").json()["jobs"]
                if jobs and jobs[0]["state"] in {"finished", "failed"}:
                    break
                time.sleep(0.02)
    finally:
        worker_module.transcribe_audio = original

    assert seen == ["someone/other-model"]
