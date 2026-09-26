"""U3: `serve` binds a real socket before the model is ready. A real subprocess and a real
uvicorn server on an ephemeral port; this is what actually exercises uvicorn.Server's
lifespan/signal handling on Windows (see PLAN_0.4.0.md's risk table), which the in-process
TestClient tests in tests/test_serve.py cannot.
"""

import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from tests.conftest import subprocess_env

MODEL = "downshift.demo.clean_mlp:make_model"
BOOT_TIMEOUT = 20


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _status(url: str) -> int | None:
    """The response status, or None if the socket isn't accepting connections yet."""
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return None


def test_ready_is_503_then_200_with_no_restart() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "downshift",
            "serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--warmup",
            "0",
            "--no-access-log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=subprocess_env(),
    )
    # Drained continuously: left unread, a few hundred polls' worth of uvicorn log lines
    # would fill the pipe buffer and block the child on its next write, hanging this test.
    output: list[str] = []
    reader = threading.Thread(target=lambda: output.extend(iter(proc.stdout.readline, "")))
    reader.daemon = True
    reader.start()
    try:
        health_url = f"http://127.0.0.1:{port}/health"
        ready_url = f"http://127.0.0.1:{port}/ready"
        deadline = time.monotonic() + BOOT_TIMEOUT

        while time.monotonic() < deadline and _status(health_url) is None:
            assert proc.poll() is None, f"server exited early:\n{''.join(output)}"
            time.sleep(0.05)
        assert _status(health_url) == 200, "the port never accepted a /health request"

        saw_503 = False
        turned_ready = False
        while time.monotonic() < deadline:
            status = _status(ready_url)
            if status == 503:
                saw_503 = True
            elif status == 200:
                turned_ready = True
                break
            time.sleep(0.05)

        assert turned_ready, "/ready never turned 200"
        assert saw_503, "never observed the not-ready window before /ready turned 200"
        assert _status(ready_url) == 200  # still 200 a moment later: no restart
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
