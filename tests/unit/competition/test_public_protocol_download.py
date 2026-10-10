from __future__ import annotations

import hashlib
import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "public_protocol_preparation", Path("scripts/competition/prepare_public_protocol_v2.py")
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def source():
    payload = b"resume-test-data" * 40000
    requests = []
    settings = {"interrupt": False, "ignore_range": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            requests.append(requested)
            offset = int(requested.removeprefix("bytes=").removesuffix("-")) if requested else 0
            if settings["ignore_range"]:
                offset = 0
            self.send_response(206 if offset else 200)
            self.send_header("Content-Length", str(len(payload) - offset))
            if offset:
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            if settings["interrupt"] and len(requests) == 1:
                self.wfile.write(payload[:300000])
                self.close_connection = True
            else:
                self.wfile.write(payload[offset:])

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/tool", payload, requests, settings
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_interrupted_download_resumes_without_losing_saved_bytes(source, tmp_path):
    url, payload, requests, settings = source
    settings["interrupt"] = True
    output = tmp_path / "tool.exe"
    preparation.download_verified(
        url, output, len(payload), hashlib.sha256(payload).hexdigest(),
        lambda *args, **kwargs: None, retry_delay=0,
    )
    assert output.read_bytes() == payload
    assert requests == [None, "bytes=300000-"]


def test_server_ignoring_range_does_not_append_duplicate_bytes(source, tmp_path):
    url, payload, requests, settings = source
    settings["ignore_range"] = True
    output = tmp_path / "tool.exe"
    output.with_suffix(".download").write_bytes(payload[:12345])
    preparation.download_verified(
        url, output, len(payload), hashlib.sha256(payload).hexdigest(), lambda *args, **kwargs: None,
    )
    assert requests == ["bytes=12345-"]
    assert output.read_bytes() == payload


def test_bad_hash_cannot_be_promoted_to_executable(source, tmp_path):
    url, payload, _, _ = source
    output = tmp_path / "tool.exe"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        preparation.download_verified(url, output, len(payload), "0" * 64, lambda *args, **kwargs: None)
    assert not output.exists()
    assert output.with_suffix(".download").is_file()
