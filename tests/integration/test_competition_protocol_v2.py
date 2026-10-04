"""A bounded local socket smoke for the documented CLI, using explicit fixtures."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading

import uvicorn

from sat_rs_vlm.competition.protocol_v2.server import create_app
from sat_rs_vlm.competition.protocol_v2.smoke import create_fixture_package


def test_socket_server_and_five_type_smoke_cli(tmp_path, monkeypatch):
    package = create_fixture_package(tmp_path / "package")
    monkeypatch.setenv("COMPETITION_IMAGE_PACKAGE", str(package))
    monkeypatch.setenv("MODEL_API_KEY", "local-socket-test-key")
    started = threading.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                started.set()

    app = create_app("configs/competition/protocol_v2.offline.yaml")
    server = ReadyServer(uvicorn.Config(app, log_level="error", access_log=False))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        thread = threading.Thread(
            target=lambda: asyncio.run(server.serve(sockets=[listener])),
            daemon=True,
        )
        thread.start()
        try:
            assert started.wait(5), "Local smoke server failed to start"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sat_rs_vlm.competition.protocol_v2.smoke",
                    "--endpoint",
                    f"http://127.0.0.1:{port}",
                    "--package",
                    str(package),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                env=os.environ.copy(),
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.splitlines() == [
                "PASS enum",
                "PASS single_choice",
                "PASS integer",
                "PASS short_text",
                "PASS bbox",
            ]
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "Bounded smoke server did not close"
