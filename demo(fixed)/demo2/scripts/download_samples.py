"""
Utility script to fetch tagged samples from public malware feeds.

The script purposely avoids executing any binary. It only downloads the
encrypted archives (password defaults to 'infected') so that you can unpack
them later inside a detached sandbox.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_samples")

MALWAREBAZAAR_URL = "https://mb-api.abuse.ch/api/v1/"


def save_metadata(entries: List[Dict], output_dir: Path) -> None:
    metadata_path = output_dir / "metadata.json"
    existing: List[Dict] = []
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Existing metadata.json is corrupted. Overwriting.")
    existing.extend(entries)
    metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Saved metadata for %d samples to %s", len(entries), metadata_path)


def fetch_from_malwarebazaar(tag: str, limit: int, api_key: Optional[str], output_dir: Path) -> None:
    logger.info("Querying MalwareBazaar for tag='%s' (limit=%d)", tag, limit)
    resp = requests.post(
        MALWAREBAZAAR_URL,
        data={
            "query": "get_taginfo",
            "tag": tag,
            "limit": limit,
        },
        headers={"API-KEY": api_key} if api_key else None,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("query_status") != "ok":
        raise RuntimeError(f"MalwareBazaar error: {payload.get('query_status')}")

    samples = payload.get("data", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    stored_metadata: List[Dict] = []

    for entry in samples:
        sha256 = entry["sha256_hash"]
        file_name = entry.get("file_name", f"{sha256}.zip")
        archive_path = output_dir / f"{sha256}.zip"
        if archive_path.exists():
            logger.info("Sample %s already exists, skipping download.", sha256)
            stored_metadata.append(entry)
            continue

        logger.info("Downloading sample %s -> %s", sha256, archive_path)
        time.sleep(1)  # be nice to the API
        download_resp = requests.post(
            MALWAREBAZAAR_URL,
            data={"query": "get_file", "sha256_hash": sha256},
            headers={"API-KEY": api_key} if api_key else None,
            timeout=60,
        )
        download_resp.raise_for_status()
        archive_path.write_bytes(download_resp.content)
        stored_metadata.append(entry)

    save_metadata(stored_metadata, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download malware samples from public feeds (password protected archives).",
        epilog="Always handle the downloaded files inside an isolated VM or sandbox.",
    )

    parser.add_argument(
        "--source",
        choices=["malwarebazaar"],
        default="malwarebazaar",
        help="Data feed to query (more sources will be added over time).",
    )
    parser.add_argument("--tag", required=True, help="Tag to filter (e.g., packed, upx, obfuscated).")
    parser.add_argument("--limit", type=int, default=25, help="Number of samples to request.")
    parser.add_argument(
        "--api-key",
        help="Optional API key (MalwareBazaar allows anonymous but rate-limited queries; provide a key for reliability).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/malwarebazaar",
        help="Directory to store downloaded archives and metadata.json",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.source == "malwarebazaar":
        fetch_from_malwarebazaar(args.tag, args.limit, args.api_key, output_dir)
    else:
        raise NotImplementedError(f"Source {args.source} is not supported yet.")

    logger.info("Done.")


if __name__ == "__main__":
    main()

