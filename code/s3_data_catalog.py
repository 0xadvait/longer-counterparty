#!/usr/bin/env python3
"""
S3 Data Catalog for Hyperliquid Research
=========================================

Lists available data in S3 and provides download utilities.
Credentials are loaded from .env file in project root.

Usage:
    python s3_data_catalog.py --list          # List all available data
    python s3_data_catalog.py --download L2   # Download L2 order book data
    python s3_data_catalog.py --info          # Show bucket info

Author: Boyi Shen, London Business School
"""

import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import argparse

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env')

# AWS Configuration from environment
AWS_CONFIG = {
    'region_name': os.getenv('AWS_REGION', 'us-east-1'),
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
}

# S3 Buckets
BUCKETS = {
    'node_data': os.getenv('S3_BUCKET_NODE_DATA', 'hl-mainnet-node-data'),
    'research': os.getenv('S3_BUCKET_RESEARCH', 'hyperliquid-research-bunc')
}

# Data catalog - what's available and where
DATA_CATALOG = {
    'node_trades': {
        'bucket': 'node_data',
        'prefix': 'node_trades/hourly/',
        'description': 'All trades on Hyperliquid with wallet addresses',
        'date_range': '2025-03-22 to present',
        'format': 'LZ4-compressed JSON'
    },
    'node_fills': {
        'bucket': 'node_data',
        'prefix': 'node_fills_by_block/hourly/',
        'description': 'Fill-level data with maker/taker info',
        'date_range': '2025-07-27 to present',
        'format': 'LZ4-compressed JSON'
    },
    'L2_orderbook': {
        'bucket': 'research',
        'prefix': 'l2_snapshots/',
        'description': 'L2 order book snapshots (10-second frequency)',
        'date_range': '2025-07-28 to 2025-07-31',
        'format': 'Parquet'
    }
}

def get_s3_client():
    """Create S3 client with credentials from .env"""
    if not AWS_CONFIG['aws_access_key_id']:
        raise ValueError("AWS credentials not found. Copy .env.example to .env and fill in credentials.")
    return boto3.client('s3', **AWS_CONFIG)

def list_bucket_contents(bucket_name, prefix='', max_keys=100):
    """List contents of an S3 bucket"""
    s3 = get_s3_client()
    bucket = BUCKETS.get(bucket_name, bucket_name)

    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=max_keys
    )

    return response.get('Contents', [])

def get_bucket_size(bucket_name, prefix=''):
    """Get total size of objects in bucket/prefix"""
    s3 = get_s3_client()
    bucket = BUCKETS.get(bucket_name, bucket_name)

    paginator = s3.get_paginator('list_objects_v2')
    total_size = 0
    total_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            total_size += obj['Size']
            total_count += 1

    return total_size, total_count

def format_size(bytes_size):
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024

def show_catalog():
    """Display the data catalog"""
    print("=" * 70)
    print("HYPERLIQUID RESEARCH DATA CATALOG")
    print("=" * 70)

    for name, info in DATA_CATALOG.items():
        print(f"\n{name}")
        print("-" * 40)
        print(f"  Description: {info['description']}")
        print(f"  Bucket:      {BUCKETS[info['bucket']]}")
        print(f"  Prefix:      {info['prefix']}")
        print(f"  Date Range:  {info['date_range']}")
        print(f"  Format:      {info['format']}")

def show_bucket_info():
    """Show info about each bucket"""
    print("=" * 70)
    print("S3 BUCKET INFORMATION")
    print("=" * 70)

    s3 = get_s3_client()

    for name, bucket in BUCKETS.items():
        print(f"\n{name}: {bucket}")
        print("-" * 40)

        try:
            # List top-level prefixes
            response = s3.list_objects_v2(
                Bucket=bucket,
                Delimiter='/',
                MaxKeys=20
            )

            prefixes = response.get('CommonPrefixes', [])
            if prefixes:
                print("  Top-level folders:")
                for p in prefixes[:10]:
                    print(f"    - {p['Prefix']}")
                if len(prefixes) > 10:
                    print(f"    ... and {len(prefixes) - 10} more")

        except Exception as e:
            print(f"  Error accessing bucket: {e}")

def list_data(data_name):
    """List files for a specific data source"""
    if data_name not in DATA_CATALOG:
        print(f"Unknown data source: {data_name}")
        print(f"Available: {', '.join(DATA_CATALOG.keys())}")
        return

    info = DATA_CATALOG[data_name]
    bucket = BUCKETS[info['bucket']]
    prefix = info['prefix']

    print(f"\nListing {data_name} ({bucket}/{prefix})...")

    contents = list_bucket_contents(info['bucket'], prefix, max_keys=20)

    if contents:
        print(f"\nFirst 20 files:")
        for obj in contents:
            size = format_size(obj['Size'])
            print(f"  {obj['Key']} ({size})")

        # Get total size
        print("\nCalculating total size...")
        total_size, total_count = get_bucket_size(info['bucket'], prefix)
        print(f"Total: {total_count:,} files, {format_size(total_size)}")
    else:
        print("No files found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S3 Data Catalog for Hyperliquid Research')
    parser.add_argument('--list', nargs='?', const='all', help='List available data (or specify: node_trades, node_fills, L2_orderbook)')
    parser.add_argument('--info', action='store_true', help='Show bucket information')
    parser.add_argument('--catalog', action='store_true', help='Show data catalog')

    args = parser.parse_args()

    if args.catalog or (not args.list and not args.info):
        show_catalog()

    if args.info:
        show_bucket_info()

    if args.list:
        if args.list == 'all':
            for name in DATA_CATALOG:
                list_data(name)
        else:
            list_data(args.list)
