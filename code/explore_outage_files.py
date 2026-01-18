#!/usr/bin/env python3
"""
Explore the file structure for the outage analysis.
"""

import boto3

AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hyperliquid-research-bunc'

s3 = boto3.client('s3', **AWS_CONFIG)

print("=" * 80)
print("EXPLORING FILE STRUCTURE FOR OUTAGE ANALYSIS")
print("=" * 80)

# Check the actual file names for BTC on outage date
print("\n1. File names for BTC on 2025-07-29:")
prefix = "raw/l2_books/BTC/2025-07-29/"
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
for obj in response.get('Contents', []):
    key = obj['Key']
    size_mb = obj['Size'] / (1024 * 1024)
    print(f"   {key.split('/')[-1]} ({size_mb:.1f} MB)")

# Check for trades data structure
print("\n2. Checking if trades data exists in different format:")
for prefix in ['raw/trades/', 'data/trades/', 'raw/fills/', 'data/fills/']:
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=5)
    if response.get('Contents'):
        print(f"   Found data at: {prefix}")
        for obj in response['Contents'][:3]:
            print(f"      {obj['Key']}")
    elif response.get('CommonPrefixes'):
        print(f"   Found subdirs at: {prefix}")
        for p in response['CommonPrefixes'][:3]:
            print(f"      {p['Prefix']}")

# Check the data/ prefix structure
print("\n3. Checking data/ prefix structure:")
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='data/', Delimiter='/')
for p in response.get('CommonPrefixes', []):
    print(f"   {p['Prefix']}")

# Check if there's processed data
print("\n4. Checking for processed data around outage date:")
for prefix_pattern in ['data/processed/', 'data/hourly/', 'data/minute/']:
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix_pattern, MaxKeys=5)
    if response.get('Contents') or response.get('CommonPrefixes'):
        print(f"   Found: {prefix_pattern}")

# Sample one file to see the data structure
print("\n5. Sampling one L2 book file to understand structure...")
import lz4.frame
import json

# Get first file
prefix = "raw/l2_books/BTC/2025-07-29/"
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
if response.get('Contents'):
    key = response['Contents'][0]['Key']
    print(f"   Downloading: {key}")

    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    compressed = obj['Body'].read()
    decompressed = lz4.frame.decompress(compressed)
    lines = decompressed.decode('utf-8').strip().split('\n')

    print(f"   Total lines: {len(lines)}")

    # Parse first few records
    timestamps = []
    for i, line in enumerate(lines[:1000]):
        try:
            record = json.loads(line)
            # Get timestamp
            ts = record.get('raw', {}).get('data', {}).get('time')
            if ts:
                timestamps.append(ts)
        except:
            pass

    if timestamps:
        from datetime import datetime
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        min_dt = datetime.utcfromtimestamp(min_ts / 1000)
        max_dt = datetime.utcfromtimestamp(max_ts / 1000)
        print(f"   Time range in first 1000 records: {min_dt} to {max_dt}")

        # Check time spacing
        if len(timestamps) > 1:
            diffs = [timestamps[i+1] - timestamps[i] for i in range(min(100, len(timestamps)-1))]
            avg_diff_sec = sum(diffs) / len(diffs) / 1000
            print(f"   Average spacing: {avg_diff_sec:.2f} seconds")

    # Show sample record structure
    if lines:
        sample = json.loads(lines[0])
        print(f"\n   Sample record keys: {list(sample.keys())}")
        if 'raw' in sample and 'data' in sample['raw']:
            print(f"   Data keys: {list(sample['raw']['data'].keys())}")

# Check what hours are covered on outage day
print("\n6. Analyzing hourly coverage on outage day:")
prefix = "raw/l2_books/BTC/2025-07-29/"
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

all_hours = []
for obj in response.get('Contents', []):
    key = obj['Key']
    filename = key.split('/')[-1]
    # Extract hours from filename
    # Typical format might be: BTC_2025-07-29_00-01.lz4 or similar
    print(f"   {filename}")

print("\n" + "=" * 80)
