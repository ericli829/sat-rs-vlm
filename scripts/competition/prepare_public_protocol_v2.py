"""Background preparation of a local diagnostic service and temporary HTTPS tunnel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

VERSION = "2026.10.0"
DOWNLOAD = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/cloudflared-windows-amd64.exe"
SIZE = 55365048
SHA256 = "86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_json(url, secret=None):
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    request = urllib.request.Request(url, headers=headers)
    # Never forward credentials to another origin after a redirect.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=5) as response:
        return json.load(response)


def download_verified(url, binary, size, sha256, progress, *, retries=8, retry_delay=2, timeout=60):
    """Resume interrupted responses, but execute only the verified complete file."""
    temporary = binary.with_suffix(".download")
    for attempt in range(1, retries + 1):
        offset = temporary.stat().st_size if temporary.is_file() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError("Partial download exceeds the expected release size")
        headers = {"User-Agent": "sat-rs-vlm-local-protocol"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as source:
                if source.status == 206:
                    content_range = source.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise RuntimeError("Unexpected download Content-Range")
                    mode = "ab"
                elif source.status == 200:
                    # A server that ignores Range returns the entire file.
                    offset = 0
                    mode = "wb"
                else:
                    raise RuntimeError("Unexpected download response status")
                count = offset
                reported = offset
                progress("downloading_cloudflared", download_bytes=count, download_attempt=attempt)
                with temporary.open(mode) as target:
                    # read1 persists available bytes instead of waiting for a full 1 MiB.
                    while block := source.read1(256 * 1024):
                        target.write(block)
                        count += len(block)
                        if count > size:
                            raise RuntimeError("Download exceeds the expected release size")
                        if count - reported >= 1024 * 1024:
                            target.flush()
                            progress("downloading_cloudflared", download_bytes=count, download_attempt=attempt)
                            reported = count
                if count != size:
                    raise EOFError("Download response ended before the complete file")
                break
        except (OSError, EOFError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == retries:
                raise
            saved = temporary.stat().st_size if temporary.is_file() else 0
            progress("download_retry", download_bytes=saved, download_attempt=attempt, download_error=type(exc).__name__)
            time.sleep(min(30, retry_delay * attempt))
    if temporary.stat().st_size != size or hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
        raise RuntimeError("cloudflared release size or SHA256 mismatch")
    temporary.replace(binary)
    progress("download_verified", download_bytes=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--mode", choices=("offline", "qwen2b"), default="qwen2b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reuse-server", action="store_true", help="Keep the existing server; prepare only its HTTPS tunnel")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / "preparation.status.json"
    status = {"job_pid": os.getpid(), "started_at": now(), "mode": args.mode}

    def stage(name, **fields):
        status.update(stage=name, updated_at=now(), **fields)
        write_json(status_path, status)
        print(json.dumps({"stage": name, **fields}, ensure_ascii=True), flush=True)

    try:
        if args.reuse_server:
            stage("checking_existing_local_service")
            existing = json.loads((run / "state.json").read_text(encoding="utf-8-sig"))
            existing_secret = Path(existing["key_file"]).read_text(encoding="utf-8-sig").strip()
            if fetch_json(existing["endpoint"] + "/healthz", existing_secret) != {
                "status": "ready", "protocol_version": "2.0", "dataset_id": "rsu-dev100-v2",
            }:
                raise RuntimeError("Existing local server is not ready with the development package")
        else:
            stage("starting_local_service")
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(repo / "scripts/competition/start_local_protocol_v2.ps1"),
                "-Mode", args.mode, "-Port", str(args.port),
                "-RunDirectory", str(run), "-Python", args.python,
            ]
            # Files, rather than PIPE: the detached server can inherit handles.
            # Waiting for a pipe's EOF would then wait for the server's lifetime.
            with (run / "launcher.stdout.log").open("wb") as stdout, (run / "launcher.stderr.log").open("wb") as stderr:
                launched = subprocess.run(
                    command, stdout=stdout, stderr=stderr, timeout=25,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            if launched.returncode:
                raise RuntimeError("Local service launcher failed; inspect launcher.stderr.log")
        service_state = json.loads((run / "state.json").read_text(encoding="utf-8-sig"))
        stage("downloading_cloudflared", server_pid=service_state["pid"], download_total=SIZE)
        tools_dir = repo / ".tmp/tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        binary = tools_dir / f"cloudflared-{VERSION}-windows-amd64.exe"
        valid = binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest() == SHA256
        if not valid:
            download_verified(DOWNLOAD, binary, SIZE, SHA256, stage)
        write_json(run / "tunnel-tool.json", {"version": VERSION, "url": DOWNLOAD, "sha256": SHA256})
        stage("starting_https_tunnel")
        with (run / "tunnel.stdout.log").open("wb") as stdout, (run / "tunnel.stderr.log").open("wb") as stderr:
            tunnel = subprocess.Popen(
                [str(binary), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", service_state["endpoint"]],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        tunnel_state = {"pid": tunnel.pid, "started_at": now(), "executable": str(binary)}
        write_json(run / "tunnel.state.json", tunnel_state)
        (run / "tunnel.pid").write_text(str(tunnel.pid), encoding="ascii")
        stage("waiting_for_public_health", tunnel_pid=tunnel.pid)
        secret = Path(service_state["key_file"]).read_text(encoding="utf-8-sig").strip()
        deadline = time.monotonic() + 240
        endpoint = None
        last_problem = None
        while time.monotonic() < deadline:
            if tunnel.poll() is not None:
                raise RuntimeError("HTTPS tunnel process exited; inspect tunnel.stderr.log")
            if endpoint is None:
                text = (run / "tunnel.stderr.log").read_text(encoding="utf-8", errors="replace")
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
                if match:
                    endpoint = match.group()
                    (run / "endpoint.txt").write_text(endpoint, encoding="ascii")
                    stage("waiting_for_public_health", public_endpoint=endpoint)
            if endpoint:
                try:
                    health = fetch_json(endpoint + "/healthz", secret)
                    if health != {
                        "status": "ready", "protocol_version": "2.0",
                        "dataset_id": "rsu-dev100-v2",
                    }:
                        raise RuntimeError("Unexpected public health metadata")
                    try:
                        fetch_json(endpoint + "/healthz")
                    except urllib.error.HTTPError as exc:
                        if exc.code != 401:
                            raise
                    else:
                        raise RuntimeError("Public endpoint does not enforce authentication")
                    stage("ready", public_health=health, authentication_check="unauthorized_returns_401")
                    return
                except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
                    last_problem = type(exc).__name__
            # Only this detached preparation job waits; the interactive agent returns immediately.
            time.sleep(2)
        raise RuntimeError(f"Public health was not ready in 240 seconds ({last_problem})")
    except Exception as exc:
        stage("needs_attention", error_type=type(exc).__name__, error=str(exc))
        # Preserve any launched server/tunnel for inspection; do not stop or restart them.
        raise


if __name__ == "__main__":
    main()
