#!/usr/bin/env python3
"""
CROSS-EVENT LEARNING INSTRUMENTAL VARIABLE
==========================================

This script implements a proper cross-event learning IV:
1. Downloads January 2025 wallet fills data (during congestion events)
2. Classifies wallets as "infrastructure-resilient" (stayed active during congestion)
   vs "non-resilient" (dropped out)
3. Uses January 2025 resilience to predict July 2025 behavior
4. Tests whether resilient-wallet concentration predicts spreads during July stress

The key identification: Wallet behavior in January 2025 congestion is PREDETERMINED
relative to July 2025 events - providing a valid instrument.

Author: Boyi Shen, London Business School
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
try:
    from linearmodels.iv import IV2SLS
except ImportError:
    IV2SLS = None
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
from pathlib import Path
import threading
import time
import sys
warnings.filterwarnings('ignore')

# Threading lock for print statements
print_lock = threading.Lock()

# Configuration - using Hyperliquid's public mainnet node data bucket
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hl-mainnet-node-data'  # Hyperliquid's public node data

OUTPUT_DIR = Path(_RESULTS_DIR)
DATA_DIR = OUTPUT_DIR / '_archive' / 'data'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Key dates (using YYYYMMDD format for the bucket)
JAN_2025_DATES = ['20250118', '20250119', '20250120', '20250121', '20250122']
JUL_2025_DATES = ['20250728', '20250729', '20250730']

# Known events
JAN_CONGESTION_1 = {'start': datetime(2025, 1, 20, 17, 7), 'end': datetime(2025, 1, 20, 17, 11)}
JAN_CONGESTION_2 = {'start': datetime(2025, 1, 20, 17, 40), 'end': datetime(2025, 1, 20, 17, 44)}
JUL_OUTAGE = {'start': datetime(2025, 7, 29, 14, 10), 'end': datetime(2025, 7, 29, 14, 47)}

ASSETS = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'XRP', 'AVAX', 'LINK', 'OP', 'SUI']

print("=" * 80)
print("CROSS-EVENT LEARNING INSTRUMENTAL VARIABLE")
print("=" * 80)
print(f"January 2025: {JAN_2025_DATES[0]} to {JAN_2025_DATES[-1]}")
print(f"July 2025: {JUL_2025_DATES[0]} to {JUL_2025_DATES[-1]}")


# =============================================================================
# STEP 1: DOWNLOAD JANUARY 2025 FILLS DATA
# =============================================================================

class DownloadTracker:
    """Thread-safe download progress tracker with live statistics."""

    def __init__(self, total_files):
        self.total_files = total_files
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.total_records = 0
        self.total_bytes = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.asset_stats = {}
        self.date_stats = {}

    def update(self, asset, date_str, success, n_records=0, n_bytes=0):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
                self.total_records += n_records
                self.total_bytes += n_bytes

                # Track per-asset
                if asset not in self.asset_stats:
                    self.asset_stats[asset] = {'files': 0, 'records': 0}
                self.asset_stats[asset]['files'] += 1
                self.asset_stats[asset]['records'] += n_records

                # Track per-date
                if date_str not in self.date_stats:
                    self.date_stats[date_str] = {'files': 0, 'records': 0}
                self.date_stats[date_str]['files'] += 1
                self.date_stats[date_str]['records'] += n_records
            else:
                self.failed += 1

    def get_progress_str(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total_files - self.completed) / rate if rate > 0 else 0

            pct = 100 * self.completed / self.total_files
            mb = self.total_bytes / (1024 * 1024)

            return (f"  [{self.completed:4d}/{self.total_files}] {pct:5.1f}% | "
                    f"✓ {self.successful:4d} ✗ {self.failed:3d} | "
                    f"Records: {self.total_records:,} | "
                    f"{mb:.1f} MB | "
                    f"{rate:.1f} files/sec | "
                    f"ETA: {eta:.0f}s")


def parse_node_fills(data_bytes, date_str, hour):
    """Parse node_fills_by_block format (same as download_wallet_data.py)."""
    records = []
    lines = data_bytes.decode('utf-8').strip().split('\n')

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
                        'date': date_str,
                        'hour': hour
                    })
        except Exception as e:
            continue
    return records


def download_fills_hour(s3, date_str, hour, tracker):
    """Download wallet-level fills for one hour with tracking (all assets)."""
    # Use node_fills_by_block format - includes all assets in one file per hour
    key = f"node_fills_by_block/hourly/{date_str}/{hour}.lz4"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key, RequestPayer='requester')
        compressed = response['Body'].read()
        n_bytes = len(compressed)
        decompressed = lz4.frame.decompress(compressed)

        records = parse_node_fills(decompressed, date_str, hour)

        tracker.update(date_str, str(hour), success=True, n_records=len(records), n_bytes=n_bytes)
        return records
    except Exception as e:
        tracker.update(date_str, str(hour), success=False)
        return []


def massively_parallel_download(dates, max_workers=50, checkpoint_interval=20):
    """
    Massively parallel download with live progress tracking.

    Downloads node_fills_by_block data which contains ALL assets per hourly file.
    Uses 50 concurrent workers by default for good throughput without overwhelming.
    """
    s3 = boto3.client('s3', **AWS_CONFIG)

    # node_fills has one file per date+hour (contains all assets)
    tasks = [(date, hour) for date in dates for hour in range(24)]

    total_files = len(tasks)
    tracker = DownloadTracker(total_files)

    print(f"\n  {'='*70}")
    print(f"  MASSIVELY PARALLEL DOWNLOAD: {total_files} files")
    print(f"  Workers: {max_workers} | Dates: {len(dates)} | Hours per date: 24")
    print(f"  Source: node_fills_by_block (contains ALL assets per file)")
    print(f"  {'='*70}")

    all_records = []
    last_print = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_fills_hour, s3, date, hour, tracker): (date, hour)
            for date, hour in tasks
        }

        for future in as_completed(futures):
            records = future.result()
            all_records.extend(records)

            # Print progress (throttled to avoid spam)
            with tracker.lock:
                completed = tracker.completed

            if completed % 5 == 0 or completed == total_files:
                print(f"\r{tracker.get_progress_str()}", end='', flush=True)

            # Detailed checkpoint
            if completed % checkpoint_interval == 0 and completed > last_print:
                last_print = completed
                print()  # New line
                print(f"\n  --- CHECKPOINT at {completed} files ---")
                with tracker.lock:
                    for date, stats in sorted(tracker.date_stats.items()):
                        print(f"    {date}: {stats['files']:3d} files, {stats['records']:8,} records")
                print()

    # Final summary
    print(f"\n\n  {'='*70}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  {'='*70}")
    print(f"  Total time: {time.time() - tracker.start_time:.1f} seconds")
    print(f"  Total files: {tracker.completed}")
    print(f"  Successful: {tracker.successful}")
    print(f"  Failed: {tracker.failed}")
    print(f"  Total records: {tracker.total_records:,}")
    print(f"  Total data: {tracker.total_bytes / (1024*1024):.1f} MB")

    print(f"\n  BY DATE:")
    for date, stats in sorted(tracker.date_stats.items()):
        print(f"    {date}: {stats['files']:3d} files, {stats['records']:8,} records")

    print(f"  {'='*70}\n")

    return all_records


print("\n[1/6] Loading/downloading wallet fills data...")

# Check for cached January 2025 data
jan_cache_file = DATA_DIR / 'jan2025_wallet_fills.parquet'

if jan_cache_file.exists():
    print("  Loading cached January 2025 wallet data...")
    jan_fills = pd.read_parquet(jan_cache_file)
    print(f"  ✓ Loaded {len(jan_fills):,} January fills")
else:
    print("  Downloading January 2025 wallet data from S3...")
    print("  Using PARALLEL DOWNLOAD (50 workers)")
    print("  NOTE: This downloads from hl-mainnet-node-data (requester pays)")

    all_records = massively_parallel_download(
        dates=JAN_2025_DATES,
        max_workers=50,
        checkpoint_interval=20
    )

    jan_fills = pd.DataFrame(all_records)
    if len(jan_fills) > 0:
        jan_fills['time_dt'] = pd.to_datetime(jan_fills['time'], unit='ms')
        jan_fills.to_parquet(jan_cache_file)
        print(f"  ✓ Downloaded and cached {len(jan_fills):,} January fills")
    else:
        print("  WARNING: No January fills downloaded!")

# Load July 2025 data (already available)
print("  Loading July 2025 wallet data...")
jul_fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
jul_fills['time_dt'] = pd.to_datetime(jul_fills['time'], unit='ms')
print(f"  ✓ Loaded {len(jul_fills):,} July fills")

print(f"\n  January wallets: {jan_fills['wallet'].nunique():,}")
print(f"  July wallets: {jul_fills['wallet'].nunique():,}")


# =============================================================================
# STEP 2: CLASSIFY JANUARY 2025 WALLET RESILIENCE
# =============================================================================

print("\n[2/6] Classifying January 2025 wallet congestion resilience...")

if 'time_dt' not in jan_fills.columns:
    jan_fills['time_dt'] = pd.to_datetime(jan_fills['time'], unit='ms')

# Define congestion windows (January 20, 2025)
jan20 = jan_fills[jan_fills['date'] == '20250120'].copy()
print(f"  January 20 fills: {len(jan20):,}")

# Window definitions
congestion_1_start = JAN_CONGESTION_1['start']
congestion_1_end = JAN_CONGESTION_1['end']
congestion_2_start = JAN_CONGESTION_2['start']
congestion_2_end = JAN_CONGESTION_2['end']

# Pre-congestion: 16:00-17:00 UTC
pre_start = datetime(2025, 1, 20, 16, 0)
pre_end = datetime(2025, 1, 20, 17, 0)

# Post-congestion: 18:00-19:00 UTC
post_start = datetime(2025, 1, 20, 18, 0)
post_end = datetime(2025, 1, 20, 19, 0)

# Tag periods
jan20['period'] = 'other'
jan20.loc[(jan20['time_dt'] >= pre_start) & (jan20['time_dt'] < pre_end), 'period'] = 'pre'
jan20.loc[((jan20['time_dt'] >= congestion_1_start) & (jan20['time_dt'] <= congestion_1_end)) |
          ((jan20['time_dt'] >= congestion_2_start) & (jan20['time_dt'] <= congestion_2_end)), 'period'] = 'during'
jan20.loc[(jan20['time_dt'] >= post_start) & (jan20['time_dt'] <= post_end), 'period'] = 'post'

# Count wallet activity by period
wallet_activity = jan20.groupby(['wallet', 'period']).size().unstack(fill_value=0)
wallet_activity.columns = ['during_count', 'other_count', 'post_count', 'pre_count']
wallet_activity = wallet_activity.reset_index()

# Ensure all columns exist
for col in ['during_count', 'other_count', 'post_count', 'pre_count']:
    if col not in wallet_activity.columns:
        wallet_activity[col] = 0

# Classification: "Resilient" = stayed active DURING congestion relative to pre
wallet_activity['pre_active'] = wallet_activity['pre_count'] > 0
wallet_activity['during_active'] = wallet_activity['during_count'] > 0

# Key metric: Activity during congestion conditional on being active before
active_before = wallet_activity[wallet_activity['pre_active']]
print(f"\n  Wallets active before congestion: {len(active_before):,}")
print(f"  Wallets still active DURING congestion: {active_before['during_active'].sum():,} ({100*active_before['during_active'].mean():.1f}%)")

# Classify wallets
wallet_activity['resilience_score'] = wallet_activity['during_count'] / (wallet_activity['pre_count'] + 1)
resilient_threshold = wallet_activity.loc[wallet_activity['pre_active'], 'resilience_score'].median()

wallet_activity['is_resilient'] = (
    wallet_activity['pre_active'] &
    (wallet_activity['resilience_score'] >= resilient_threshold)
)

print(f"\n  Resilient wallets (above-median activity during congestion): {wallet_activity['is_resilient'].sum():,}")
print(f"  Non-resilient wallets: {(~wallet_activity['is_resilient']).sum():,}")


# =============================================================================
# STEP 3: MATCH TO JULY 2025 WALLETS
# =============================================================================

print("\n[3/6] Matching to July 2025 wallets...")

# Get July wallet activity
jul_wallets = set(jul_fills['wallet'].unique())
jan_wallets = set(wallet_activity['wallet'].unique())
overlap_wallets = jul_wallets & jan_wallets

print(f"  July unique wallets: {len(jul_wallets):,}")
print(f"  January classified wallets: {len(jan_wallets):,}")
print(f"  Overlap (in both periods): {len(overlap_wallets):,}")

# Merge resilience classification to July fills
jul_fills_with_class = jul_fills.merge(
    wallet_activity[['wallet', 'is_resilient', 'resilience_score']],
    on='wallet',
    how='left'
)

# For wallets not in January, mark as "unknown"
jul_fills_with_class['is_resilient'] = jul_fills_with_class['is_resilient'].fillna(False)
jul_fills_with_class['resilience_score'] = jul_fills_with_class['resilience_score'].fillna(0)
jul_fills_with_class['in_jan_sample'] = jul_fills_with_class['wallet'].isin(jan_wallets)

print(f"\n  July fills from wallets in Jan sample: {jul_fills_with_class['in_jan_sample'].sum():,} ({100*jul_fills_with_class['in_jan_sample'].mean():.1f}%)")


# =============================================================================
# STEP 4: BUILD HOURLY PANEL FOR IV
# =============================================================================

print("\n[4/6] Building hourly panel for IV analysis...")

# Tag July outage
jul_fills_with_class['is_outage'] = (
    (jul_fills_with_class['time_dt'] >= JUL_OUTAGE['start']) &
    (jul_fills_with_class['time_dt'] <= JUL_OUTAGE['end'])
)

# Create hour variable
jul_fills_with_class['date_str'] = jul_fills_with_class['date'].astype(str)
jul_fills_with_class['date_str'] = jul_fills_with_class['date_str'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")

# Compute hourly metrics by asset
def compute_hourly_metrics(df, group_cols=['coin', 'date_str', 'hour']):
    """Compute hourly spread and composition metrics."""

    # Spread proxy: use fee structure
    hourly = df.groupby(group_cols).agg({
        'px': ['mean', 'std'],
        'sz': 'sum',
        'wallet': 'nunique',
        'is_resilient': 'mean',  # Share of fills from resilient wallets
        'resilience_score': 'mean',  # Average resilience
        'in_jan_sample': 'mean',  # Share in jan sample
    })
    hourly.columns = ['px_mean', 'px_std', 'volume', 'n_wallets',
                      'resilient_share', 'avg_resilience', 'jan_sample_share']
    hourly = hourly.reset_index()

    # Volatility as spread proxy
    hourly['vol_bps'] = hourly['px_std'] / hourly['px_mean'] * 10000

    return hourly

hourly_panel = compute_hourly_metrics(jul_fills_with_class)
print(f"  ✓ Created {len(hourly_panel):,} hourly observations")

# Merge with outage indicator
hourly_panel['datetime'] = pd.to_datetime(hourly_panel['date_str'] + ' ' + hourly_panel['hour'].astype(str) + ':00:00')
hourly_panel['is_outage'] = (
    (hourly_panel['datetime'] >= JUL_OUTAGE['start']) &
    (hourly_panel['datetime'] <= JUL_OUTAGE['end'])
)

print(f"  Outage hours: {hourly_panel['is_outage'].sum()}")


# =============================================================================
# STEP 5: CROSS-EVENT LEARNING IV ANALYSIS
# =============================================================================

print("\n[5/6] Running cross-event learning IV analysis...")

# Focus on outage day and neighbors
outage_day_data = hourly_panel[hourly_panel['date_str'] == '2025-07-29'].copy()

# Pre-outage: hours 10-13 (before 14:10 outage)
# During-outage: hour 14
# Post-outage: hours 15-18

outage_day_data['period'] = 'pre'
outage_day_data.loc[outage_day_data['hour'] == 14, 'period'] = 'during'
outage_day_data.loc[outage_day_data['hour'] >= 15, 'period'] = 'post'

print(f"\n  Pre-outage observations: {(outage_day_data['period'] == 'pre').sum()}")
print(f"  During-outage observations: {(outage_day_data['period'] == 'during').sum()}")
print(f"  Post-outage observations: {(outage_day_data['period'] == 'post').sum()}")

# Key IV test:
# First stage: Does Jan resilience predict composition during Jul stress?
# Second stage: Does predicted composition affect spreads?

# Compute PRE-outage resilient share by asset
pre_resilience = outage_day_data[outage_day_data['period'] == 'pre'].groupby('coin')['resilient_share'].mean()
pre_resilience.name = 'pre_resilient_share'

# This is our INSTRUMENT: pre-outage share of Jan-2025-resilient wallets
# It is predetermined (from 6 months prior) and predicts who will stay during stress

# Merge instrument to outage period data
outage_analysis = outage_day_data.merge(pre_resilience, on='coin', how='left')

# Outcome: Volatility during outage (proxy for spread)
during_outage = outage_analysis[outage_analysis['period'] == 'during'].copy()
pre_outage = outage_analysis[outage_analysis['period'] == 'pre'].copy()

print(f"\n  During-outage assets: {len(during_outage)}")
print(f"  Mean volatility (bps): {during_outage['vol_bps'].mean():.2f}")

# Simple regression: Does pre-resilient share predict outage volatility?
if len(during_outage) > 5:
    # First stage: Does pre-resilience predict during-resilience?
    X_fs = sm.add_constant(during_outage['pre_resilient_share'])
    y_fs = during_outage['resilient_share']  # During-outage resilient share

    fs_model = OLS(y_fs, X_fs).fit()

    print("\n  FIRST STAGE: Pre-Resilience → During-Resilience")
    print(f"    Coefficient: {fs_model.params.iloc[1]:.4f}")
    print(f"    t-stat: {fs_model.tvalues.iloc[1]:.2f}")
    print(f"    R²: {fs_model.rsquared:.3f}")

    # Reduced form: Does pre-resilience predict volatility?
    y_rf = during_outage['vol_bps']
    rf_model = OLS(y_rf, X_fs).fit()

    print("\n  REDUCED FORM: Pre-Resilience → Volatility")
    print(f"    Coefficient: {rf_model.params.iloc[1]:.4f}")
    print(f"    t-stat: {rf_model.tvalues.iloc[1]:.2f}")

    # OLS: During-resilience on volatility
    X_ols = sm.add_constant(during_outage['resilient_share'])
    ols_model = OLS(y_rf, X_ols).fit()

    print("\n  OLS: During-Resilience → Volatility")
    print(f"    Coefficient: {ols_model.params.iloc[1]:.4f}")
    print(f"    t-stat: {ols_model.tvalues.iloc[1]:.2f}")

    # Wald estimate = Reduced Form / First Stage
    iv_estimate = rf_model.params.iloc[1] / fs_model.params.iloc[1]
    print(f"\n  IV (WALD) ESTIMATE: {iv_estimate:.4f}")


# =============================================================================
# STEP 6: CROSS-ASSET PANEL IV
# =============================================================================

print("\n[6/6] Cross-asset panel IV with asset fixed effects...")

# Use full panel (all hours around outage)
panel_data = outage_day_data.copy()
panel_data['is_stress'] = (panel_data['period'] == 'during').astype(int)

# Create asset dummies
panel_data = pd.get_dummies(panel_data, columns=['coin'], prefix='asset')

# Interaction: pre_resilience × stress
panel_data['resilience_x_stress'] = panel_data['pre_resilient_share'] * panel_data['is_stress']

# Regression with interaction
asset_cols = [c for c in panel_data.columns if c.startswith('asset_')]

# Outcome: volatility
# Key regressor: resilient_share during outage
# Instrument: pre_resilient_share (from Jan 2025)

# Simple panel regression
X_vars = ['is_stress', 'resilient_share', 'resilience_x_stress'] + asset_cols[:-1]
X = sm.add_constant(panel_data[X_vars].values)
y = panel_data['vol_bps'].values

if len(panel_data) > len(X_vars) + 5:
    model = OLS(y, X).fit()

    print("\n  PANEL REGRESSION: Volatility ~ Resilience + Stress + Interaction + Asset FE")
    print(f"    is_stress: {model.params[1]:.2f} (t={model.tvalues[1]:.2f})")
    print(f"    resilient_share: {model.params[2]:.2f} (t={model.tvalues[2]:.2f})")
    print(f"    resilience×stress: {model.params[3]:.2f} (t={model.tvalues[3]:.2f})")
    print(f"    R²: {model.rsquared:.3f}")


# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Summary statistics
results = {
    'sample': {
        'jan_dates': JAN_2025_DATES,
        'jul_dates': JUL_2025_DATES,
        'jan_fills': len(jan_fills),
        'jul_fills': len(jul_fills),
        'overlap_wallets': len(overlap_wallets),
    },
    'wallet_classification': {
        'n_resilient': int(wallet_activity['is_resilient'].sum()),
        'n_non_resilient': int((~wallet_activity['is_resilient']).sum()),
        'resilience_rate': float(wallet_activity['is_resilient'].mean()),
    },
    'iv_analysis': {
        'first_stage': {
            'coef': float(fs_model.params.iloc[1]) if 'fs_model' in dir() else None,
            't_stat': float(fs_model.tvalues.iloc[1]) if 'fs_model' in dir() else None,
            'r2': float(fs_model.rsquared) if 'fs_model' in dir() else None,
        },
        'reduced_form': {
            'coef': float(rf_model.params.iloc[1]) if 'rf_model' in dir() else None,
            't_stat': float(rf_model.tvalues.iloc[1]) if 'rf_model' in dir() else None,
        },
        'iv_wald_estimate': float(iv_estimate) if 'iv_estimate' in dir() else None,
    }
}

with open(OUTPUT_DIR / 'cross_event_iv_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save wallet resilience classification
wallet_activity.to_csv(OUTPUT_DIR / 'wallet_jan_resilience.csv', index=False)

print("✓ Saved: cross_event_iv_results.json")
print("✓ Saved: wallet_jan_resilience.csv")

print("\n" + "=" * 80)
print("CROSS-EVENT LEARNING IV COMPLETE")
print("=" * 80)

print("""
IDENTIFICATION STRATEGY:
========================
1. Wallets that stayed active during Jan 2025 congestion are "infrastructure-resilient"
2. This is PREDETERMINED relative to July 2025 events (6 months prior)
3. Pre-outage concentration of resilient wallets predicts composition during stress
4. This allows us to isolate the CAUSAL effect of composition on spreads

KEY EXCLUSION RESTRICTION:
==========================
Wallet behavior in January 2025 affects July 2025 spreads ONLY through the
composition channel (who stays vs who leaves during stress).

This is cleaner than same-event instruments because January behavior is genuinely
exogenous to July market conditions.
""")
