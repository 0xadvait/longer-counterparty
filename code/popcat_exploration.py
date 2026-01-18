#!/usr/bin/env python3
"""
POPCAT Manipulation Episode Exploration
November 12-13, 2025

Objective: Assess whether there are clean pre-trends and sharp breaks
in market quality metrics around the alleged manipulation episode.

IMPORTANT: We are being honest researchers - only proceed with this
analysis if the data shows clear, compelling patterns.
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
warnings.filterwarnings('ignore')

# AWS Configuration
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hyperliquid-research-bunc'

# Study window: Nov 8-16, 2025 (4 days pre, event days 12-13, 3 days post)
START_DATE = '2025-11-08'
END_DATE = '2025-11-16'
EVENT_DATES = ['2025-11-12', '2025-11-13']

# Assets to analyze
TREATED = ['POPCAT']
CONTROLS = ['DOGE', 'WIF', 'kPEPE', 'kSHIB', 'kBONK', 'MEME', 'GOAT', 'PNUT', 'MOODENG']

def download_and_process_file(s3, bucket, key, coin):
    """Download and process a single L2 book file."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        compressed = obj['Body'].read()
        decompressed = lz4.frame.decompress(compressed)
        lines = decompressed.decode('utf-8').strip().split('\n')

        records = []
        for line in lines:
            try:
                data = json.loads(line)
                raw_data = data.get('raw', {}).get('data', {})

                timestamp = raw_data.get('time')
                if not timestamp:
                    continue

                levels = raw_data.get('levels', [[], []])
                bids = levels[0] if len(levels) > 0 else []
                asks = levels[1] if len(levels) > 1 else []

                if not bids or not asks:
                    continue

                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])

                if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
                    continue

                mid = (best_bid + best_ask) / 2
                spread_bps = (best_ask - best_bid) / mid * 10000

                # Calculate depth (top 5 levels)
                bid_depth = sum(float(b['px']) * float(b['sz']) for b in bids[:5])
                ask_depth = sum(float(a['px']) * float(a['sz']) for a in asks[:5])
                total_depth = bid_depth + ask_depth

                # Order imbalance
                bid_sz = sum(float(b['sz']) for b in bids[:5])
                ask_sz = sum(float(a['sz']) for a in asks[:5])
                imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz) if (bid_sz + ask_sz) > 0 else 0

                records.append({
                    'timestamp': timestamp,
                    'coin': coin,
                    'mid': mid,
                    'spread_bps': spread_bps,
                    'depth': total_depth,
                    'imbalance': abs(imbalance),
                    'best_bid': best_bid,
                    'best_ask': best_ask
                })
            except:
                continue

        return records
    except Exception as e:
        print(f"Error processing {key}: {e}")
        return []

def download_coin_data(coin, dates):
    """Download all data for a coin across specified dates."""
    s3 = boto3.client('s3', **AWS_CONFIG)
    all_records = []

    # Build list of all files to download
    files_to_download = []
    for date in dates:
        prefix = f"raw/l2_books/{coin}/{date}/"
        try:
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
            for obj in response.get('Contents', []):
                files_to_download.append((obj['Key'], coin))
        except:
            continue

    print(f"  {coin}: {len(files_to_download)} files to download")

    # Download in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_and_process_file, s3, S3_BUCKET, key, coin): key
            for key, coin in files_to_download
        }
        for future in as_completed(futures):
            records = future.result()
            all_records.extend(records)

    return all_records

def main():
    print("=" * 80)
    print("POPCAT MANIPULATION EPISODE EXPLORATION")
    print(f"Study Window: {START_DATE} to {END_DATE}")
    print(f"Event Dates: {EVENT_DATES}")
    print("=" * 80)

    # Generate date list
    dates = []
    current = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')
    while current <= end:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    print(f"\nDates to analyze: {dates}")

    # Download data for all assets
    all_coins = TREATED + CONTROLS
    all_data = []

    print(f"\nDownloading data for {len(all_coins)} assets...")
    for coin in all_coins:
        print(f"\nProcessing {coin}...")
        records = download_coin_data(coin, dates)
        all_data.extend(records)
        print(f"  {coin}: {len(records):,} records")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("\nERROR: No data downloaded!")
        return

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['datetime'].dt.date.astype(str)
    df['hour'] = df['datetime'].dt.hour

    # Mark event period
    df['is_event'] = df['date'].isin(EVENT_DATES)
    df['is_treated'] = df['coin'].isin(TREATED)

    # Save raw data for later
    cache_file = os.path.join(_DATA_DIR, 'popcat_exploration_data.parquet')
    df.to_parquet(cache_file)
    print(f"\nSaved {len(df):,} records to {cache_file}")

    # ============================================================
    # ANALYSIS 1: Daily Summary Statistics
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: DAILY SUMMARY STATISTICS")
    print("=" * 80)

    daily = df.groupby(['date', 'coin']).agg({
        'spread_bps': ['mean', 'median', 'std'],
        'depth': ['mean', 'median'],
        'imbalance': 'mean',
        'mid': ['first', 'last', 'mean'],
        'timestamp': 'count'
    }).round(4)
    daily.columns = ['_'.join(col) for col in daily.columns]
    daily = daily.reset_index()

    # POPCAT daily stats
    print("\nPOPCAT Daily Statistics:")
    print("-" * 60)
    popcat_daily = daily[daily['coin'] == 'POPCAT'].copy()
    popcat_daily['price_change_pct'] = (popcat_daily['mid_last'] / popcat_daily['mid_first'] - 1) * 100

    for _, row in popcat_daily.iterrows():
        event_marker = " *** EVENT ***" if row['date'] in EVENT_DATES else ""
        print(f"{row['date']}: Spread={row['spread_bps_mean']:.2f}bps, "
              f"Depth=${row['depth_mean']:,.0f}, "
              f"Price={row['mid_mean']:.4f}, "
              f"Obs={row['timestamp_count']:,}{event_marker}")

    # ============================================================
    # ANALYSIS 2: POPCAT vs Controls Comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: POPCAT vs CONTROLS (Pre vs Event Period)")
    print("=" * 80)

    # Pre-event vs event comparison
    pre_event_dates = [d for d in dates if d < '2025-11-12']

    summary_stats = []
    for coin in all_coins:
        coin_data = df[df['coin'] == coin]

        pre = coin_data[coin_data['date'].isin(pre_event_dates)]
        event = coin_data[coin_data['date'].isin(EVENT_DATES)]

        if len(pre) > 0 and len(event) > 0:
            summary_stats.append({
                'coin': coin,
                'pre_spread': pre['spread_bps'].mean(),
                'event_spread': event['spread_bps'].mean(),
                'spread_change': event['spread_bps'].mean() - pre['spread_bps'].mean(),
                'spread_change_pct': (event['spread_bps'].mean() / pre['spread_bps'].mean() - 1) * 100,
                'pre_depth': pre['depth'].mean(),
                'event_depth': event['depth'].mean(),
                'depth_change_pct': (event['depth'].mean() / pre['depth'].mean() - 1) * 100,
                'pre_imbalance': pre['imbalance'].mean(),
                'event_imbalance': event['imbalance'].mean(),
                'pre_obs': len(pre),
                'event_obs': len(event)
            })

    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('spread_change', ascending=False)

    print("\nSpread Changes (Pre-Event → Event):")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        treated_marker = " <-- TREATED" if row['coin'] == 'POPCAT' else ""
        print(f"{row['coin']:10s}: {row['pre_spread']:6.2f} → {row['event_spread']:6.2f} bps "
              f"(Δ = {row['spread_change']:+6.2f}, {row['spread_change_pct']:+6.1f}%){treated_marker}")

    print("\nDepth Changes (Pre-Event → Event):")
    print("-" * 80)
    summary_df_depth = summary_df.sort_values('depth_change_pct')
    for _, row in summary_df_depth.iterrows():
        treated_marker = " <-- TREATED" if row['coin'] == 'POPCAT' else ""
        print(f"{row['coin']:10s}: ${row['pre_depth']:12,.0f} → ${row['event_depth']:12,.0f} "
              f"({row['depth_change_pct']:+6.1f}%){treated_marker}")

    # ============================================================
    # ANALYSIS 3: Hourly Patterns on Event Days
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: HOURLY PATTERNS ON EVENT DAYS (Nov 12-13)")
    print("=" * 80)

    event_data = df[df['date'].isin(EVENT_DATES)]

    # Focus on POPCAT
    popcat_hourly = event_data[event_data['coin'] == 'POPCAT'].groupby(['date', 'hour']).agg({
        'spread_bps': 'mean',
        'depth': 'mean',
        'mid': 'mean',
        'imbalance': 'mean',
        'timestamp': 'count'
    }).round(4)

    print("\nPOPCAT Hourly Metrics (Event Days):")
    print("-" * 70)
    for idx, row in popcat_hourly.iterrows():
        date, hour = idx
        print(f"{date} {hour:02d}:00 - Spread: {row['spread_bps']:6.2f}bps, "
              f"Depth: ${row['depth']:12,.0f}, Price: ${row['mid']:.4f}")

    # ============================================================
    # ANALYSIS 4: Difference-in-Differences Setup Check
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: DIFFERENCE-IN-DIFFERENCES FEASIBILITY")
    print("=" * 80)

    # Calculate DiD estimate
    popcat_pre = df[(df['coin'] == 'POPCAT') & (df['date'].isin(pre_event_dates))]['spread_bps'].mean()
    popcat_event = df[(df['coin'] == 'POPCAT') & (df['date'].isin(EVENT_DATES))]['spread_bps'].mean()

    controls_pre = df[(df['coin'].isin(CONTROLS)) & (df['date'].isin(pre_event_dates))]['spread_bps'].mean()
    controls_event = df[(df['coin'].isin(CONTROLS)) & (df['date'].isin(EVENT_DATES))]['spread_bps'].mean()

    did_spread = (popcat_event - popcat_pre) - (controls_event - controls_pre)

    print(f"\nSpread (bps):")
    print(f"  POPCAT:   Pre = {popcat_pre:.2f}, Event = {popcat_event:.2f}, Δ = {popcat_event - popcat_pre:+.2f}")
    print(f"  Controls: Pre = {controls_pre:.2f}, Event = {controls_event:.2f}, Δ = {controls_event - controls_pre:+.2f}")
    print(f"  DiD Estimate: {did_spread:+.2f} bps")

    # Same for depth
    popcat_pre_depth = df[(df['coin'] == 'POPCAT') & (df['date'].isin(pre_event_dates))]['depth'].mean()
    popcat_event_depth = df[(df['coin'] == 'POPCAT') & (df['date'].isin(EVENT_DATES))]['depth'].mean()

    controls_pre_depth = df[(df['coin'].isin(CONTROLS)) & (df['date'].isin(pre_event_dates))]['depth'].mean()
    controls_event_depth = df[(df['coin'].isin(CONTROLS)) & (df['date'].isin(EVENT_DATES))]['depth'].mean()

    did_depth = (popcat_event_depth - popcat_pre_depth) - (controls_event_depth - controls_pre_depth)

    print(f"\nDepth ($):")
    print(f"  POPCAT:   Pre = ${popcat_pre_depth:,.0f}, Event = ${popcat_event_depth:,.0f}, Δ = ${popcat_event_depth - popcat_pre_depth:+,.0f}")
    print(f"  Controls: Pre = ${controls_pre_depth:,.0f}, Event = ${controls_event_depth:,.0f}, Δ = ${controls_event_depth - controls_pre_depth:+,.0f}")
    print(f"  DiD Estimate: ${did_depth:+,.0f}")

    # ============================================================
    # ANALYSIS 5: Pre-Trend Check (Critical for Honesty!)
    # ============================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: PRE-TREND CHECK (CRITICAL FOR VALIDITY)")
    print("=" * 80)

    # Check if POPCAT and controls had parallel trends pre-event
    print("\nDaily Spread Trends (Pre-Event Period):")
    print("-" * 60)

    for date in pre_event_dates:
        popcat_spread = df[(df['coin'] == 'POPCAT') & (df['date'] == date)]['spread_bps'].mean()
        control_spread = df[(df['coin'].isin(CONTROLS)) & (df['date'] == date)]['spread_bps'].mean()
        diff = popcat_spread - control_spread
        print(f"{date}: POPCAT={popcat_spread:.2f}bps, Controls={control_spread:.2f}bps, Diff={diff:+.2f}bps")

    # Check for trend in the difference
    pre_diffs = []
    for date in pre_event_dates:
        popcat_spread = df[(df['coin'] == 'POPCAT') & (df['date'] == date)]['spread_bps'].mean()
        control_spread = df[(df['coin'].isin(CONTROLS)) & (df['date'] == date)]['spread_bps'].mean()
        pre_diffs.append(popcat_spread - control_spread)

    if len(pre_diffs) >= 2:
        trend = (pre_diffs[-1] - pre_diffs[0]) / (len(pre_diffs) - 1)
        print(f"\nPre-trend in POPCAT-Controls spread difference: {trend:+.2f} bps/day")
        print("(Should be close to 0 for parallel trends assumption)")

    # ============================================================
    # HONEST ASSESSMENT
    # ============================================================
    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT: IS THIS A CLEAN NATURAL EXPERIMENT?")
    print("=" * 80)

    # Criteria for inclusion
    criteria = []

    # 1. Did POPCAT spread increase more than controls?
    popcat_change = popcat_event - popcat_pre
    control_change = controls_event - controls_pre
    spread_effect = popcat_change > control_change * 1.5  # POPCAT should have notably larger increase
    criteria.append(('POPCAT spread ↑ more than controls', spread_effect, f"{popcat_change:.2f} vs {control_change:.2f}"))

    # 2. Is the DiD estimate economically meaningful?
    meaningful_did = abs(did_spread) > 1.0  # At least 1 bps effect
    criteria.append(('DiD spread effect > 1 bps', meaningful_did, f"{did_spread:.2f} bps"))

    # 3. Are pre-trends relatively parallel?
    if len(pre_diffs) >= 2:
        parallel = abs(trend) < 0.5  # Less than 0.5 bps/day trend
        criteria.append(('Pre-trends parallel (|trend| < 0.5)', parallel, f"{trend:.2f} bps/day"))

    # 4. Sufficient observations?
    sufficient_obs = len(df[df['coin'] == 'POPCAT']) > 10000
    criteria.append(('Sufficient observations (>10K)', sufficient_obs, f"{len(df[df['coin'] == 'POPCAT']):,}"))

    print("\nCriteria Assessment:")
    print("-" * 60)
    all_pass = True
    for criterion, passed, detail in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion}: {status} ({detail})")
        if not passed:
            all_pass = False

    print("\n" + "-" * 60)
    if all_pass:
        print("RECOMMENDATION: Data looks clean enough for inclusion in paper.")
        print("Proceed with full DiD analysis and visualization.")
    else:
        print("RECOMMENDATION: Some criteria not met. Proceed with caution.")
        print("Consider whether to include this analysis or note limitations.")

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
