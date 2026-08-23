"""Render blinded before/after/difference contact sheets for visual annotation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    return parser.parse_args()


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("annotation CSV is empty")
    required = {"audit_id", "sample_id", "image_t1_path", "image_t2_path"}
    if not required.issubset(rows[0]):
        raise ValueError("annotation CSV is missing required image/audit fields")
    return rows


def _open_rgb(path: str) -> Image.Image:
    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"image file not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _panel(image: Image.Image, *, width: int, height: int) -> Image.Image:
    panel = image.copy()
    panel.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(panel, ((width - panel.width) // 2, (height - panel.height) // 2))
    return canvas


def main() -> int:
    args = parse_args()
    source = args.annotation_csv.resolve()
    output_dir = args.output_dir.resolve()
    if args.rows_per_sheet < 1:
        raise SystemExit("--rows-per-sheet must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {output_dir}")
    try:
        rows = _read_rows(source)
    except ValueError as exc:
        raise SystemExit(f"contact-sheet rendering failed: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_width, panel_height, label_height = 256, 256, 38
    sheet_width = panel_width * 3
    font = _font(14)
    index: list[dict[str, object]] = []
    for start in range(0, len(rows), args.rows_per_sheet):
        batch = rows[start : start + args.rows_per_sheet]
        sheet = Image.new("RGB", (sheet_width, (panel_height + label_height) * len(batch)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(batch):
            try:
                before = _open_rgb(row["image_t1_path"])
                after = _open_rgb(row["image_t2_path"])
            except ValueError as exc:
                raise SystemExit(f"contact-sheet rendering failed: {exc}") from exc
            difference = ImageChops.difference(before, after)
            top = offset * (panel_height + label_height)
            for column, image in enumerate((before, after, difference)):
                sheet.paste(
                    _panel(image, width=panel_width, height=panel_height),
                    (column * panel_width, top),
                )
            draw.text(
                (4, top + panel_height + 2),
                f"{row['audit_id']}  {row['sample_id']}",
                fill="black",
                font=font,
            )
            draw.text(
                (4, top + panel_height + 19),
                "before              after               abs-difference",
                fill="black",
                font=font,
            )
            index.append(
                {
                    "sheet": f"sheet_{start // args.rows_per_sheet + 1:03d}.png",
                    "row_in_sheet": offset + 1,
                    "audit_id": row["audit_id"],
                    "sample_id": row["sample_id"],
                }
            )
        sheet.save(output_dir / f"sheet_{start // args.rows_per_sheet + 1:03d}.png")
    (output_dir / "contact_sheet_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(index)} blinded rows in {len(set(row['sheet'] for row in index))} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
