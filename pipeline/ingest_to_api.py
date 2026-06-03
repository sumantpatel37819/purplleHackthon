"""
Ingest events from JSONL files into the API.
Run this after detect.py to populate the API database.

Usage:
    python pipeline/ingest_to_api.py
    python pipeline/ingest_to_api.py --api-url http://localhost:8080
"""
import json
import sys
import time
import httpx
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EVENTS_DIR = BASE_DIR / "data" / "events"
DEFAULT_API_URL = "http://localhost:8080"
BATCH_SIZE = 200


def ingest_file(api_url: str, jsonl_path: Path) -> dict:
    """Ingest a single JSONL events file into the API."""
    events = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        print(f"  [SKIP] {jsonl_path.name}: no events")
        return {"ingested": 0, "errors": 0}

    print(f"\n  Ingesting {len(events)} events from {jsonl_path.name}...")
    total_ingested = 0
    total_errors = 0

    # Send in batches of BATCH_SIZE
    for i in range(0, len(events), BATCH_SIZE):
        batch = events[i:i + BATCH_SIZE]
        try:
            resp = httpx.post(
                f"{api_url}/events/ingest",
                json={"events": batch},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                total_ingested += data.get("ingested", 0)
                total_errors += data.get("errors", 0)
                print(f"    Batch {i//BATCH_SIZE + 1}: {data.get('ingested', 0)} ingested, "
                      f"{data.get('duplicates', 0)} dupes, {data.get('errors', 0)} errors")
            else:
                print(f"    ERROR {resp.status_code}: {resp.text[:200]}")
                total_errors += len(batch)
        except httpx.ConnectError:
            print(f"    CONNECTION ERROR: Is the API running at {api_url}?")
            sys.exit(1)

    return {"ingested": total_ingested, "errors": total_errors}


def main():
    parser = argparse.ArgumentParser(description="Ingest CCTV events into the Store Intelligence API")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--cam", type=int, help="Only ingest specific camera events (1-5)")
    args = parser.parse_args()

    print(f"Purplle Store Intelligence - Event Ingestion")
    print(f"API URL: {args.api_url}")

    # Check API is reachable
    try:
        resp = httpx.get(f"{args.api_url}/health", timeout=5.0)
        print(f"API Status: {resp.status_code} OK")
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to API at {args.api_url}")
        print("Start the API first: uvicorn app.main:app --reload")
        sys.exit(1)

    # Find JSONL files
    if args.cam:
        files = [EVENTS_DIR / f"cam{args.cam}_events.jsonl"]
    else:
        files = sorted(EVENTS_DIR.glob("cam*_events.jsonl"))

    if not files:
        print(f"No event files found in {EVENTS_DIR}")
        sys.exit(1)

    grand_total = 0
    for f in files:
        if f.exists():
            result = ingest_file(args.api_url, f)
            grand_total += result["ingested"]
        else:
            print(f"  [MISSING] {f}")

    print(f"\n{'='*50}")
    print(f"INGESTION COMPLETE: {grand_total} total events ingested")
    print(f"{'='*50}")

    # Quick verification
    try:
        resp = httpx.get(f"{args.api_url}/stores/STORE_BLR_002/metrics", timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"\nQuick check - /metrics:")
            print(f"  Unique visitors: {data.get('unique_visitors', '?')}")
            print(f"  Conversion rate: {data.get('conversion_rate', '?')}")
    except Exception as e:
        print(f"Could not fetch metrics: {e}")


if __name__ == "__main__":
    main()
