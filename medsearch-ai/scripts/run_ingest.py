#!/usr/bin/env python3
"""
run_ingest.py - One-shot script to ingest medical articles into Endee vector database.
Run this ONCE after starting the Endee server.

Usage:
    cd d:\Endee.io\medsearch-ai
    python scripts/run_ingest.py
"""

import sys
import os
import logging

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app import endee_client
from app.ingest import ingest_all


def main():
    print("\n" + "=" * 60)
    print("  MedSearch AI - Data Ingestion Script")
    print("  Vector Database: Endee (http://localhost:8080)")
    print("=" * 60 + "\n")

    # Check Endee is running
    print(">> Checking Endee server connection...")
    if not endee_client.check_health():
        print("ERROR: Cannot connect to Endee at http://localhost:8080")
        print("\n  Please start Endee first:")
        print("  docker run -p 8080:8080 endeeio/endee-server:latest\n")
        sys.exit(1)
    print("OK: Endee server is running!\n")

    # Run ingestion
    force = "--force" in sys.argv
    if force:
        print("WARNING: Force re-create mode: existing index will be deleted.\n")

    result = ingest_all(force_recreate=force)

    print("\n" + "=" * 60)
    if result["success"]:
        print("DONE: Ingestion Complete!")
        print(f"   Total articles: {result['total_articles']}")
        print(f"   Successfully ingested: {result['ingested']}")
        print(f"   Failed batches: {result['failed_batches']}")
        print("\n>> Now start the API server:")
        print("   cd d:\\Endee.io\\medsearch-ai")
        print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\n>> Then open your browser:")
        print("   http://localhost:8000")
    else:
        print(f"FAILED: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
