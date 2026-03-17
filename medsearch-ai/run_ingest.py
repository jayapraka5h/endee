#!/usr/bin/env python3
"""
Direct ingestion script - bypasses HTTP layer to ingest data immediately
"""
import sys
import os

# Set up path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

from app.ingest import ingest_all

if __name__ == "__main__":
    print("="*60)
    print("MedSearch AI - Direct Data Ingestion")
    print("="*60)
    
    result = ingest_all(force_recreate=False)
    
    print("\n" + "="*60)
    if result.get("success"):
        print(f"SUCCESS: Ingested {result.get('ingested', 0)} articles")
    else:
        print(f"FAILED: {result.get('error', 'Unknown error')}")
    print("="*60)
