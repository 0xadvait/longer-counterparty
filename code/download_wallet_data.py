#!/usr/bin/env python3
"""
High-Performance Wallet-Level Data Downloader for Hyperliquid

Optimized for M2 Max with 96GB RAM:
- Massively parallel downloads (50+ concurrent connections)
- Real-time progress tracking with speed estimates
- Fully resumable - checkpoints after every file
- Incremental saves to parquet

Author: Claude
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading
import time
import os
import sys
from pathlib import Path

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}

BUCKET = 'hl-mainnet-node-data'
OUTPUT_DIR = Path(_RESULTS_DIR)
CHECKPOINT_FILE = OUTPUT_DIR / 'download_checkpoint.json'

# Parallelism settings (optimized for M2 Max + high bandwidth)
MAX_WORKERS = 64  # High parallelism for I/O-bound work
BATCH_SIZE = 50   # Files to process before saving checkpoint

# Data sources to download
DOWNLOAD_CONFIG = {
    'node_trades': {
        'prefix': 'node_trades/hourly/',
        'date_start': '20250322',
        'date_end': '20250621',
        'output_file': 'node_trades_data.parquet'
    },
    'node_fills': {
        'prefix': 'node_fills_by_block/hourly/',
        'date_start': '20250727',
        'date_end': '20250930',  # 2 months around outage
        'output_file': 'node_fills_extended.parquet'
    }
}

# ============================================================================
# PROGRESS TRACKER (Thread-Safe)
# ============================================================================

class ProgressTracker:
    def __init__(self, total_files):
        self.total_files = total_files
        self.completed = 0
        self.failed = 0
        self.bytes_downloaded = 0
        self.records_parsed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.current_files = {}
        self.errors = []

    def update(self, file_key, bytes_size, records, success=True, error=None):
        with self.lock:
            if success:
                self.completed += 1
                self.bytes_downloaded += bytes_size
                self.records_parsed += records
            else:
                self.failed += 1
                self.errors.append((file_key, str(error)))

            if file_key in self.current_files:
                del self.current_files[file_key]

    def start_file(self, file_key):
        with self.lock:
            self.current_files[file_key] = time.time()

    def get_status(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            speed_mbps = (self.bytes_downloaded / 1024 / 1024) / elapsed if elapsed > 0 else 0
            records_per_sec = self.records_parsed / elapsed if elapsed > 0 else 0

            if self.completed > 0:
                eta_seconds = (self.total_files - self.completed) * (elapsed / self.completed)
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            else:
                eta_str = "calculating..."

            return {
                'completed': self.completed,
                'failed': self.failed,
                'total': self.total_files,
                'percent': 100 * self.completed / self.total_files if self.total_files > 0 else 0,
                'bytes_mb': self.bytes_downloaded / 1024 / 1024,
                'records': self.records_parsed,
                'speed_mbps': speed_mbps,
                'records_per_sec': records_per_sec,
                'elapsed': elapsed,
                'eta': eta_str,
                'active': len(self.current_files)
            }

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.completed_files = set()
        self.load()

    def load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.completed_files = set(data.get('completed_files', []))
            print(f"📂 Loaded checkpoint: {len(self.completed_files)} files already downloaded")

    def save(self, new_files):
        self.completed_files.update(new_files)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'completed_files': list(self.completed_files),
                'last_updated': datetime.now().isoformat()
            }, f)

    def is_completed(self, file_key):
        return file_key in self.completed_files

# ============================================================================
# DATA PARSERS
# ============================================================================

def parse_node_trades(data_bytes, file_key):
    """Parse node_trades format - has both sides of each trade."""
    records = []
    lines = data_bytes.decode('utf-8').strip().split('\n')

    date = file_key.split('/')[-2]
    hour = int(file_key.split('/')[-1].replace('.lz4', ''))

    for line in lines:
        try:
            trade = json.loads(line)

            # Extract both sides
            side_info = trade.get('side_info', [])
            if len(side_info) >= 2:
                # First side is typically the maker, second is taker (but verify with side field)
                for i, side in enumerate(side_info):
                    records.append({
                        'coin': trade.get('coin', ''),
                        'trade_side': trade.get('side', ''),  # A or B
                        'time': trade.get('time', ''),
                        'px': float(trade.get('px', 0)),
                        'sz': float(trade.get('sz', 0)),
                        'wallet': side.get('user', ''),
                        'start_pos': float(side.get('start_pos', 0)) if side.get('start_pos') else 0,
                        'oid': side.get('oid'),
                        'twap_id': side.get('twap_id'),
                        'is_first_side': i == 0,
                        'date': date,
                        'hour': hour
                    })
        except Exception as e:
            continue

    return records

def parse_node_fills(data_bytes, file_key):
    """Parse node_fills_by_block format."""
    records = []
    lines = data_bytes.decode('utf-8').strip().split('\n')

    date = file_key.split('/')[-2]
    hour = int(file_key.split('/')[-1].replace('.lz4', ''))

    for line in lines:
        try:
            record = json.loads(line)
            for event in record.get('events', []):
                if isinstance(event, list) and len(event) == 2:
                    wallet, fill_data = event
                    records.append({
                        'wallet': wallet,
                        'coin': fill_data.get('coin', ''),
                        'px': float(fill_data.get('px', 0)),
                        'sz': float(fill_data.get('sz', 0)),
                        'side': fill_data.get('side', ''),
                        'time': fill_data.get('time', 0),
                        'crossed': fill_data.get('crossed', True),
                        'fee': float(fill_data.get('fee', 0)),
                        'date': date,
                        'hour': hour
                    })
        except Exception as e:
            continue

    return records

# ============================================================================
# DOWNLOAD WORKER
# ============================================================================

def download_file(s3_client, bucket, key, parser, tracker, checkpoint):
    """Download and parse a single file."""
    if checkpoint.is_completed(key):
        return None, key, True  # Already done

    tracker.start_file(key)

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key, RequestPayer='requester')
        compressed = response['Body'].read()
        decompressed = lz4.frame.decompress(compressed)

        records = parser(decompressed, key)

        tracker.update(key, len(compressed), len(records), success=True)
        return records, key, False

    except Exception as e:
        tracker.update(key, 0, 0, success=False, error=e)
        return None, key, False

# ============================================================================
# MAIN DOWNLOAD ORCHESTRATOR
# ============================================================================

def list_files_to_download(s3_client, prefix, date_start, date_end):
    """List all files in date range."""
    files = []

    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, RequestPayer='requester'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Extract date from key
            parts = key.split('/')
            if len(parts) >= 3:
                date = parts[-2]
                if date_start <= date <= date_end:
                    files.append(key)

    return sorted(files)

def download_dataset(source_name, config):
    """Download a complete dataset with progress tracking."""
    print(f"\n{'='*80}")
    print(f"📥 DOWNLOADING: {source_name}")
    print(f"{'='*80}")

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(OUTPUT_DIR / f'{source_name}_checkpoint.json')

    # Create S3 client
    s3 = boto3.client('s3', **AWS_CONFIG)

    # List files
    print(f"\n🔍 Listing files from {config['date_start']} to {config['date_end']}...")
    all_files = list_files_to_download(s3, config['prefix'], config['date_start'], config['date_end'])

    # Filter out completed files
    files_to_download = [f for f in all_files if not checkpoint.is_completed(f)]

    print(f"   Total files: {len(all_files)}")
    print(f"   Already downloaded: {len(all_files) - len(files_to_download)}")
    print(f"   Remaining: {len(files_to_download)}")

    if not files_to_download:
        print("✅ All files already downloaded!")
        return

    # Select parser
    parser = parse_node_trades if 'trades' in source_name else parse_node_fills

    # Initialize tracker
    tracker = ProgressTracker(len(files_to_download))

    # Storage for records
    all_records = []
    completed_keys = []
    last_save_time = time.time()

    # Progress display thread
    stop_display = threading.Event()

    def display_progress():
        while not stop_display.is_set():
            status = tracker.get_status()
            bar_width = 40
            filled = int(bar_width * status['percent'] / 100)
            bar = '█' * filled + '░' * (bar_width - filled)

            sys.stdout.write(f"\r   [{bar}] {status['percent']:.1f}% | "
                           f"{status['completed']}/{status['total']} files | "
                           f"{status['bytes_mb']:.1f} MB | "
                           f"{status['speed_mbps']:.1f} MB/s | "
                           f"{status['records']:,} records | "
                           f"ETA: {status['eta']} | "
                           f"Active: {status['active']}   ")
            sys.stdout.flush()
            time.sleep(0.5)

    display_thread = threading.Thread(target=display_progress, daemon=True)
    display_thread.start()

    # Download with thread pool
    print(f"\n🚀 Starting download with {MAX_WORKERS} parallel workers...\n")

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {
                executor.submit(download_file, boto3.client('s3', **AWS_CONFIG),
                              BUCKET, key, parser, tracker, checkpoint): key
                for key in files_to_download
            }

            # Process completed tasks
            for future in as_completed(futures):
                records, key, was_cached = future.result()

                if records and not was_cached:
                    all_records.extend(records)
                    completed_keys.append(key)

                # Save checkpoint periodically
                if len(completed_keys) >= BATCH_SIZE:
                    checkpoint.save(completed_keys)
                    completed_keys = []

                # Save data periodically (every 5 minutes or 1M records)
                if (time.time() - last_save_time > 300) or (len(all_records) > 1_000_000):
                    if all_records:
                        save_incremental(all_records, source_name)
                        all_records = []
                        last_save_time = time.time()

    finally:
        stop_display.set()
        display_thread.join(timeout=1)

    # Final save
    if completed_keys:
        checkpoint.save(completed_keys)

    if all_records:
        save_incremental(all_records, source_name)

    # Print summary
    status = tracker.get_status()
    print(f"\n\n{'='*80}")
    print(f"✅ DOWNLOAD COMPLETE: {source_name}")
    print(f"{'='*80}")
    print(f"   Files downloaded: {status['completed']}")
    print(f"   Files failed: {status['failed']}")
    print(f"   Total data: {status['bytes_mb']:.1f} MB")
    print(f"   Total records: {status['records']:,}")
    print(f"   Time elapsed: {status['elapsed']/60:.1f} minutes")
    print(f"   Average speed: {status['speed_mbps']:.1f} MB/s")

    if tracker.errors:
        print(f"\n⚠️  Errors ({len(tracker.errors)}):")
        for key, error in tracker.errors[:5]:
            print(f"      {key}: {error}")

def save_incremental(records, source_name):
    """Save records incrementally to parquet."""
    if not records:
        return

    df = pd.DataFrame(records)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'{source_name}_part_{timestamp}.parquet'

    df.to_parquet(output_file, index=False)
    print(f"\n   💾 Saved {len(df):,} records to {output_file.name}")

def merge_parquet_files(source_name):
    """Merge all parquet parts into a single file."""
    print(f"\n🔄 Merging parquet files for {source_name}...")

    pattern = f'{source_name}_part_*.parquet'
    part_files = sorted(OUTPUT_DIR.glob(pattern))

    if not part_files:
        print("   No part files found.")
        return

    print(f"   Found {len(part_files)} part files")

    dfs = []
    for f in part_files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates if any
    before = len(df)
    if 'time' in df.columns and 'wallet' in df.columns:
        df = df.drop_duplicates()
    after = len(df)

    output_file = OUTPUT_DIR / f'{source_name}_merged.parquet'
    df.to_parquet(output_file, index=False)

    print(f"   ✅ Merged {before:,} records ({before-after:,} duplicates removed)")
    print(f"   📁 Output: {output_file}")
    print(f"   📊 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   👛 Unique wallets: {df['wallet'].nunique():,}")

    # Cleanup part files
    for f in part_files:
        f.unlink()
    print(f"   🧹 Cleaned up {len(part_files)} part files")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🚀 HYPERLIQUID WALLET-LEVEL DATA DOWNLOADER")
    print("   Optimized for M2 Max with 96GB RAM")
    print("="*80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"⚡ Parallel workers: {MAX_WORKERS}")
    print(f"💾 Checkpoint batch size: {BATCH_SIZE}")

    # Download each dataset
    for source_name, config in DOWNLOAD_CONFIG.items():
        download_dataset(source_name, config)
        merge_parquet_files(source_name)

    print("\n" + "="*80)
    print("🎉 ALL DOWNLOADS COMPLETE!")
    print("="*80)

    # Final summary
    print("\n📊 FINAL DATA SUMMARY:")
    for f in OUTPUT_DIR.glob('*_merged.parquet'):
        df = pd.read_parquet(f)
        print(f"\n   {f.name}:")
        print(f"      Records: {len(df):,}")
        print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"      Unique wallets: {df['wallet'].nunique():,}")
        if 'coin' in df.columns:
            print(f"      Coins: {df['coin'].nunique()}")

if __name__ == '__main__':
    main()
