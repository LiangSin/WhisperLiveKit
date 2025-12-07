#!/usr/bin/env python3
"""
Decode JSON files that contain escaped Unicode (e.g., '\\uXXXX') and write a
human-readable copy alongside the original.

Usage:
  python scripts/decode_unicode_json.py /path/to/file.json

The decoded file will be saved as `<stem>.decoded<suffix>` in the same
directory (e.g., `foo.json` -> `foo.decoded.json`).
"""

import argparse
import json
from pathlib import Path
from typing import Any


def decode_json_file(src_path: Path, overwrite: bool = False) -> Path:
    if not src_path.is_file():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    content = src_path.read_text(encoding="utf-8")
    try:
        data: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {src_path}: {exc}") from exc

    dst_path = src_path.with_name(f"{src_path.stem}.decoded{src_path.suffix}")
    if dst_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {dst_path} (use --force to overwrite)"
        )

    dst_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode a JSON file with unicode escapes and write a readable copy."
    )
    parser.add_argument("path", type=Path, help="Path to the JSON file to decode")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists",
    )
    args = parser.parse_args()

    try:
        output_path = decode_json_file(args.path, overwrite=args.force)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))
        return

    print(f"Decoded to: {output_path}")


if __name__ == "__main__":
    main()






