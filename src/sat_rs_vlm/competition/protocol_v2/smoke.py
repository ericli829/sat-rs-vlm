"""Five synthetic contract checks. Does not submit an official evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from .answers import serialize_response
from .image_package import ImagePackage
from .models import AnswerContract, ImageDescriptor, parse_request


def create_fixture_package(root: str | Path) -> Path:
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("Refusing to overwrite an existing image package")
    (root / "assets").mkdir(parents=True, exist_ok=True)
    assets = []
    for index, color in enumerate(("navy", "green")):
        image = Image.new("RGB", (640, 480), color)
        ImageDraw.Draw(image).rectangle((64, 48, 320, 240), fill="white")
        asset_id = f"{index + 1:032x}"
        path = root / "assets" / asset_id
        image.save(path, format="PNG")
        assets.append(
            {
                "asset_id": asset_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mime_type": "image/png",
                "size": path.stat().st_size,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "rsu-sealed-images-v1",
                "dataset_id": "synthetic-protocol-v2",
                "assets": assets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def smoke_requests(package: ImagePackage) -> list[dict]:
    assets = list(package.assets.values())
    descriptors = [
        ImageDescriptor(asset_id=x.asset_id, sha256=x.sha256, mime_type=x.mime_type).model_dump()
        for x in assets[:2]
    ]
    if not descriptors:
        raise ValueError("Smoke needs at least one asset")
    size = package.resolve(ImageDescriptor.model_validate(descriptors[0]))
    specs = [
        (
            "enum",
            "Has the scene changed?",
            {"type": "enum", "values": ["Yes", "No"]},
            {
                "answer_type": "short_text",
                "images": descriptors if len(descriptors) == 2 else descriptors[:1],
            },
        ),
        (
            "single_choice",
            "What is the main land cover?",
            {"type": "single_choice", "values": ["opt-7", "Z", "B"]},
            {
                "answer_type": "single_choice",
                "choices": {"B": "water", "opt-7": "urban", "Z": "forest"},
            },
        ),
        (
            "integer",
            "How many ships are there?",
            {"type": "integer", "minimum": 0},
            {"answer_type": "integer"},
        ),
        (
            "short_text",
            "Describe the main land cover.",
            {"type": "short_text", "min_length": 1},
            {"answer_type": "short_text"},
        ),
        (
            "bbox",
            "Locate the ship.",
            {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            {"answer_type": "bbox", "image_width": size.width, "image_height": size.height},
        ),
    ]
    result = []
    for kind, question, constraint, extra in specs:
        result.append(
            {
                "protocol_version": "2.0",
                "request_id": f"local-smoke:{kind}:1",
                "item_id": f"smoke-{kind}",
                "question": question,
                "images": descriptors[:1],
                "response_constraint": constraint,
                **extra,
            }
        )
    return result


def check_response(payload: dict, response: object) -> None:
    request = parse_request(payload)
    contract = AnswerContract.from_request(request)
    if not isinstance(response, dict) or set(response) != {
        "protocol_version",
        "request_id",
        "item_id",
        "answer",
    }:
        raise ValueError("Response must contain only the protocol echoes and answer")
    value = response["answer"]
    if contract.kind == "integer":
        if not isinstance(value, str) or not value or str(int(value)) != value:
            raise ValueError("Integer answer must be a canonical integer string")
        value = int(value)
    expected = serialize_response(request, contract, value)
    if expected != response:
        raise ValueError("Invalid response echo or answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Local origin or HTTPS deployed origin")
    parser.add_argument("--package")
    parser.add_argument("--api-key", help="Prefer MODEL_API_KEY to keep secrets out of argv")
    parser.add_argument("--create-fixture", metavar="DIRECTORY")
    args = parser.parse_args()
    if args.create_fixture:
        created = create_fixture_package(args.create_fixture).resolve()
        print(f"Created synthetic image package: {created}")
        return
    secret = args.api_key or os.environ.get("MODEL_API_KEY")
    if not args.endpoint or not args.package or not secret:
        parser.error("--endpoint, --package and MODEL_API_KEY (or --api-key) are required")
    origin = args.endpoint.rstrip("/")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            # Official checks require direct responses; never forward Bearer on redirects.
            raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)

    opener = urllib.request.build_opener(NoRedirect())
    try:
        health_request = urllib.request.Request(
            origin + "/healthz", headers={"Authorization": f"Bearer {secret}"}
        )
        with opener.open(health_request, timeout=10) as response:
            health = json.load(response)
        package = ImagePackage(args.package)
        if (
            health.get("status") != "ready"
            or health.get("protocol_version") != "2.0"
            or (health.get("dataset_id") != package.dataset_id)
        ):
            raise ValueError("Health metadata disagrees with the image package")
        for payload in smoke_requests(package):
            req = urllib.request.Request(
                origin + "/v1/predict",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            with opener.open(req, timeout=120) as response:
                if (
                    response.status != 200
                    or response.headers.get_content_type() != "application/json"
                ):
                    raise ValueError("Predict must return HTTP 200 application/json")
                body = json.load(response)
            check_response(payload, body)
            print(f"PASS {payload['response_constraint']['type']}")
    except (urllib.error.URLError, ValueError) as exc:
        # Network exceptions can include URLs; do not print request/credential data.
        parser.exit(1, f"Smoke failed ({type(exc).__name__}).\n")


if __name__ == "__main__":
    main()
