#!/usr/bin/env python3
"""
Explore S3 data availability around the July 29, 2025 API outage.

Outage window: ~14:10-14:47 UTC on July 29, 2025
"""

import boto3
from datetime import datetime, timedelta

# AWS Configuration
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hyperliquid-research-bunc'

s3 = boto3.client('s3', **AWS_CONFIG)

print("=" * 80)
print("EXPLORING S3 DATA AVAILABILITY FOR JULY 29, 2025 OUTAGE ANALYSIS")
print("=" * 80)

# Check what data types are available
print("\n1. Checking top-level prefixes...")
response = s3.list_objects_v2(Bucket=S3_BUCKET, Delimiter='/')
prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
print(f"   Available prefixes: {prefixes}")

# Check raw data types
print("\n2. Checking raw data types...")
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='raw/', Delimiter='/')
raw_prefixes = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
print(f"   Raw data types: {raw_prefixes}")

# Key dates for the analysis
outage_date = '2025-07-29'
dates_to_check = [
    '2025-07-27',  # 2 days before (placebo)
    '2025-07-28',  # 1 day before (pre-outage exposure calculation)
    '2025-07-29',  # Outage day
    '2025-07-30',  # 1 day after
    '2025-07-31',  # 2 days after
]

# Check which assets have data on the outage date
print(f"\n3. Checking asset coverage on {outage_date}...")

# Check l2_books
print("\n   L2 Order Books:")
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='raw/l2_books/', Delimiter='/')
book_assets = []
for p in response.get('CommonPrefixes', []):
    asset = p['Prefix'].split('/')[-2]
    # Check if this asset has data on outage date
    check_prefix = f"raw/l2_books/{asset}/{outage_date}/"
    check_resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=check_prefix, MaxKeys=1)
    if check_resp.get('Contents'):
        book_assets.append(asset)
print(f"   Assets with L2 books on {outage_date}: {len(book_assets)}")
print(f"   Assets: {book_assets[:20]}..." if len(book_assets) > 20 else f"   Assets: {book_assets}")

# Check trades
print("\n   Trades:")
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='raw/trades/', Delimiter='/')
trade_assets = []
for p in response.get('CommonPrefixes', []):
    asset = p['Prefix'].split('/')[-2]
    check_prefix = f"raw/trades/{asset}/{outage_date}/"
    check_resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=check_prefix, MaxKeys=1)
    if check_resp.get('Contents'):
        trade_assets.append(asset)
print(f"   Assets with trades on {outage_date}: {len(trade_assets)}")
print(f"   Assets: {trade_assets[:20]}..." if len(trade_assets) > 20 else f"   Assets: {trade_assets}")

# Check order events (cancels, updates)
print("\n   Order Events (cancels/updates):")
for prefix_type in ['raw/order_events/', 'raw/orders/', 'raw/cancels/', 'raw/order_updates/']:
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix_type, MaxKeys=5)
    if response.get('Contents') or response.get('CommonPrefixes'):
        print(f"   Found: {prefix_type}")
        # Check structure
        if response.get('CommonPrefixes'):
            print(f"      Subdirs: {[p['Prefix'] for p in response['CommonPrefixes'][:3]]}")

# Check file counts for outage date
print(f"\n4. Checking data volume on outage date ({outage_date})...")

# Pick a major asset to check
test_assets = ['BTC', 'ETH', 'SOL']
for asset in test_assets:
    if asset in book_assets:
        prefix = f"raw/l2_books/{asset}/{outage_date}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        n_files = len(response.get('Contents', []))

        # Check hourly coverage
        if response.get('Contents'):
            hours = set()
            for obj in response['Contents'][:100]:
                # Try to extract hour from filename
                key = obj['Key']
                if '/' in key:
                    filename = key.split('/')[-1]
                    if '_' in filename:
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            try:
                                hour = int(parts[1][:2]) if len(parts[1]) >= 2 else None
                                if hour is not None:
                                    hours.add(hour)
                            except:
                                pass

        print(f"   {asset}: {n_files} files on {outage_date}")

# Check data around the outage window specifically (14:00-15:00 UTC)
print(f"\n5. Checking data granularity around outage window (14:00-15:00 UTC)...")
for asset in ['BTC', 'ETH']:
    if asset not in book_assets:
        continue
    prefix = f"raw/l2_books/{asset}/{outage_date}/"
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

    outage_window_files = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        # Files around 14:xx hour
        if '_14' in key or '_1410' in key or '_1420' in key or '_1430' in key or '_1440' in key or '_1450' in key:
            outage_window_files.append(key.split('/')[-1])

    print(f"   {asset}: {len(outage_window_files)} files in outage window")
    if outage_window_files:
        print(f"      Sample files: {outage_window_files[:5]}")

# Check date range availability
print(f"\n6. Checking date range availability for control periods...")
for date in dates_to_check:
    for asset in ['BTC', 'ETH']:
        if asset not in book_assets:
            continue
        prefix = f"raw/l2_books/{asset}/{date}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=1)
        has_data = '✓' if response.get('Contents') else '✗'
        print(f"   {date} - {asset}: {has_data}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Data appears to be available for the outage analysis:
- L2 order books: {len(book_assets)} assets
- Trades: {len(trade_assets)} assets
- Outage date: {outage_date}
- Outage window: 14:10-14:47 UTC

Next steps:
1. Download L2 books and trades for key assets around the outage window
2. Compute pre-outage exposure metrics (July 28)
3. Run event study comparing outage vs normal periods
""")
