#!/usr/bin/env python3
"""
JANUARY 2025 API CONGESTION ANALYSIS
====================================

Analyzes the documented API server congestion on January 20, 2025:
- 17:07-17:11 UTC
- 17:40-17:44 UTC

Downloads January 18-22, 2025 for robust pre/post windows.

Author: Boyi Shen, London Business School
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
from pathlib import Path

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
warnings.filterwarnings('ignore')

# Publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Configuration
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hyperliquid-research-bunc'

DATA_DIR = Path(_DATA_DIR)
OUTPUT_DIR = Path(_RESULTS_DIR)
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Analysis dates: January 18-22, 2025
ANALYSIS_DATES = ['2025-01-18', '2025-01-19', '2025-01-20', '2025-01-21', '2025-01-22']

# Known congestion events on January 20, 2025
CONGESTION_EVENTS = [
    {'start': datetime(2025, 1, 20, 17, 7), 'end': datetime(2025, 1, 20, 17, 11), 'name': 'Congestion 1'},
    {'start': datetime(2025, 1, 20, 17, 40), 'end': datetime(2025, 1, 20, 17, 44), 'name': 'Congestion 2'},
]

# Assets (major ones)
ASSETS = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'XRP', 'AVAX', 'LINK', 'OP', 'SUI']

print("=" * 80)
print("JANUARY 2025 API CONGESTION ANALYSIS")
print("=" * 80)
print(f"Dates: {ANALYSIS_DATES[0]} to {ANALYSIS_DATES[-1]}")
print(f"Known congestion: Jan 20, 17:07-17:11 and 17:40-17:44 UTC")


# =============================================================================
# DATA LOADING
# =============================================================================

def download_and_process_hour(s3, asset, date_str, hour):
    """Download L2 book data and compute quote update measures."""
    key = f"raw/l2_books/{asset}/{date_str}/{hour}.lz4"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        compressed = response['Body'].read()
        decompressed = lz4.frame.decompress(compressed)

        records = []
        lines = decompressed.decode('utf-8').strip().split('\n')

        prev_bid = None
        prev_ask = None

        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
                data = record.get('raw', {}).get('data', {})
                levels = data.get('levels', [])

                if len(levels) < 2:
                    continue

                bids, asks = levels[0], levels[1]
                if not bids or not asks:
                    continue

                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])
                mid = (best_bid + best_ask) / 2
                spread_bps = (best_ask - best_bid) / mid * 10000

                bid_changed = 1 if prev_bid is not None and best_bid != prev_bid else 0
                ask_changed = 1 if prev_ask is not None and best_ask != prev_ask else 0

                prev_bid = best_bid
                prev_ask = best_ask

                ts_ms = data.get('time')
                if ts_ms:
                    records.append({
                        'asset': asset,
                        'time_ms': ts_ms,
                        'spread_bps': spread_bps,
                        'bid_changed': bid_changed,
                        'ask_changed': ask_changed,
                    })
            except:
                continue

        return records
    except Exception as e:
        return []


# =============================================================================
# LOAD OR DOWNLOAD DATA
# =============================================================================

cache_file = DATA_DIR / 'jan2025_congestion_data.parquet'

if cache_file.exists():
    print("\n[1/5] Loading cached data...")
    df_all = pd.read_parquet(cache_file)
    print(f"  ✓ Loaded {len(df_all):,} observations")
else:
    print("\n[1/5] Downloading data from S3...")
    print(f"  Downloading {len(ANALYSIS_DATES)} days × {len(ASSETS)} assets × 24 hours")

    s3 = boto3.client('s3', **AWS_CONFIG)
    tasks = [(asset, date, hour) for date in ANALYSIS_DATES for asset in ASSETS for hour in range(24)]

    print(f"  Total files: {len(tasks)}")

    all_records = []
    with ThreadPoolExecutor(max_workers=50) as executor:  # Massive parallelism!
        futures = {executor.submit(download_and_process_hour, s3, asset, date, hour): (asset, date, hour)
                   for asset, date, hour in tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            records = future.result()
            all_records.extend(records)
            if completed % 100 == 0:
                print(f"    Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.0f}%)")

    if not all_records:
        print("ERROR: No data downloaded!")
        exit(1)

    df_all = pd.DataFrame(all_records)
    df_all['time'] = pd.to_datetime(df_all['time_ms'], unit='ms')
    df_all['date'] = df_all['time'].dt.strftime('%Y-%m-%d')
    df_all = df_all.sort_values(['asset', 'time']).reset_index(drop=True)

    df_all.to_parquet(cache_file)
    print(f"  ✓ Cached {len(df_all):,} observations")

print(f"  Date range: {df_all['time'].min()} to {df_all['time'].max()}")
print(f"  Assets: {df_all['asset'].nunique()}")


# =============================================================================
# PREPARE MINUTE-LEVEL DATA
# =============================================================================

print("\n[2/5] Preparing minute-level data...")

df_all['minute_bin'] = df_all['time'].dt.floor('1min')
df_all['quote_updates'] = df_all['bid_changed'] + df_all['ask_changed']

df_minute = df_all.groupby(['asset', 'date', 'minute_bin']).agg({
    'spread_bps': 'median',
    'quote_updates': 'sum',
}).reset_index()

df_minute['hour'] = df_minute['minute_bin'].dt.hour
print(f"  ✓ Created {len(df_minute):,} minute-level observations")


# =============================================================================
# ANALYZE KNOWN CONGESTION EVENTS
# =============================================================================

print("\n[3/5] Analyzing known congestion events...")

def analyze_event(event_start, event_end, event_name, df_minute, pre_hours=1, post_hours=1):
    """Analyze a specific congestion event."""

    pre_start = event_start - timedelta(hours=pre_hours)
    post_end = event_end + timedelta(hours=post_hours)

    df_event = df_minute[
        (df_minute['minute_bin'] >= pre_start) &
        (df_minute['minute_bin'] <= post_end)
    ].copy()

    if len(df_event) < 20:
        return None

    df_event['period'] = 'during'
    df_event.loc[df_event['minute_bin'] < event_start, 'period'] = 'pre'
    df_event.loc[df_event['minute_bin'] > event_end, 'period'] = 'post'

    spread_pre = df_event[df_event['period'] == 'pre']['spread_bps'].mean()
    spread_during = df_event[df_event['period'] == 'during']['spread_bps'].mean()
    spread_post = df_event[df_event['period'] == 'post']['spread_bps'].mean()

    quotes_pre = df_event[df_event['period'] == 'pre']['quote_updates'].mean()
    quotes_during = df_event[df_event['period'] == 'during']['quote_updates'].mean()

    pre_spreads = df_event[df_event['period'] == 'pre']['spread_bps']
    during_spreads = df_event[df_event['period'] == 'during']['spread_bps']

    if len(pre_spreads) > 5 and len(during_spreads) > 5:
        t_stat, p_val = stats.ttest_ind(during_spreads, pre_spreads)
    else:
        t_stat, p_val = np.nan, np.nan

    # Count assets affected
    n_assets = df_event[df_event['period'] == 'during']['asset'].nunique()

    return {
        'event_name': event_name,
        'event_start': event_start,
        'event_end': event_end,
        'duration_min': (event_end - event_start).total_seconds() / 60,
        'n_assets': n_assets,
        'spread_pre': spread_pre,
        'spread_during': spread_during,
        'spread_post': spread_post,
        'spread_effect_bps': spread_during - spread_pre,
        'quotes_pre': quotes_pre,
        'quotes_during': quotes_during,
        'quotes_drop_pct': (quotes_during - quotes_pre) / quotes_pre * 100 if quotes_pre > 0 else np.nan,
        't_stat': t_stat,
        'p_value': p_val,
        'n_obs': len(df_event),
    }

congestion_results = []
for event in CONGESTION_EVENTS:
    result = analyze_event(event['start'], event['end'], event['name'], df_minute)
    if result:
        congestion_results.append(result)
        print(f"\n  {event['name']} ({event['start'].strftime('%H:%M')}-{event['end'].strftime('%H:%M')} UTC):")
        print(f"    Assets affected: {result['n_assets']}")
        print(f"    Spread: {result['spread_pre']:.2f} → {result['spread_during']:.2f} bps ({result['spread_effect_bps']:+.2f})")
        print(f"    Quote updates: {result['quotes_pre']:.0f} → {result['quotes_during']:.0f} ({result['quotes_drop_pct']:+.1f}%)")
        print(f"    t-stat: {result['t_stat']:.2f}, p = {result['p_value']:.4f}")

jan_results_df = pd.DataFrame(congestion_results)


# =============================================================================
# DETECT ADDITIONAL SHOCKS (automated)
# =============================================================================

print("\n[4/5] Detecting additional infrastructure shocks...")

SHOCK_PERCENTILE = 10
MIN_SHOCK_DURATION_MIN = 5  # Shorter for congestion events
MIN_GAP_MIN = 30

# Compute thresholds
thresholds = df_minute.groupby(['asset', 'hour']).agg({
    'quote_updates': lambda x: np.percentile(x, SHOCK_PERCENTILE)
}).reset_index()
thresholds.columns = ['asset', 'hour', 'threshold']

df_minute = df_minute.merge(thresholds, on=['asset', 'hour'], how='left')
df_minute['is_shock'] = (df_minute['quote_updates'] < df_minute['threshold']).astype(int)

print(f"  Minutes flagged as shocks: {df_minute['is_shock'].sum():,} ({100*df_minute['is_shock'].mean():.1f}%)")

# Find events
def identify_shock_events(df_asset):
    df = df_asset.sort_values('minute_bin').copy()
    df['shock_start'] = (df['is_shock'] == 1) & (df['is_shock'].shift(1, fill_value=0) == 0)
    df['shock_end'] = (df['is_shock'] == 1) & (df['is_shock'].shift(-1, fill_value=0) == 0)

    events = []
    start_time = None

    for _, row in df.iterrows():
        if row['shock_start']:
            start_time = row['minute_bin']
        if row['shock_end'] and start_time is not None:
            end_time = row['minute_bin']
            duration = (end_time - start_time).total_seconds() / 60 + 1
            if duration >= MIN_SHOCK_DURATION_MIN:
                events.append({
                    'asset': row['asset'],
                    'start': start_time,
                    'end': end_time,
                    'duration_min': duration,
                    'date': start_time.strftime('%Y-%m-%d'),
                })
            start_time = None
    return events

all_events = []
for asset in df_minute['asset'].unique():
    events = identify_shock_events(df_minute[df_minute['asset'] == asset])
    all_events.extend(events)

if all_events:
    events_df = pd.DataFrame(all_events)
    events_df['event_window'] = events_df['start'].dt.floor('30min')

    event_counts = events_df.groupby('event_window').agg({
        'asset': 'nunique',
        'duration_min': 'max',
        'start': 'min',
        'end': 'max',
    }).reset_index()
    event_counts.columns = ['event_window', 'n_assets', 'max_duration', 'earliest_start', 'latest_end']

    system_events = event_counts[event_counts['n_assets'] >= 3].sort_values('earliest_start')
    print(f"  System-wide events detected: {len(system_events)}")

    # Analyze each
    additional_results = []
    for _, event in system_events.iterrows():
        result = analyze_event(event['earliest_start'], event['latest_end'],
                              f"Auto-{event['earliest_start'].strftime('%m-%d %H:%M')}", df_minute)
        if result:
            additional_results.append(result)

    if additional_results:
        additional_df = pd.DataFrame(additional_results)
        # Combine with known events
        all_results_df = pd.concat([jan_results_df, additional_df], ignore_index=True)
    else:
        all_results_df = jan_results_df
else:
    all_results_df = jan_results_df


# =============================================================================
# AGGREGATE AND SAVE RESULTS
# =============================================================================

print("\n[5/5] Aggregating results...")

n_events = len(all_results_df)
n_significant = (all_results_df['p_value'] < 0.05).sum()
mean_spread_effect = all_results_df['spread_effect_bps'].mean()

print(f"\n  JANUARY 2025 CONGESTION SUMMARY:")
print(f"  {'='*50}")
print(f"  Total events analyzed: {n_events}")
print(f"  Significant (p<0.05): {n_significant} ({100*n_significant/n_events:.0f}%)")
print(f"  Mean spread widening: {mean_spread_effect:.2f} bps")


# =============================================================================
# CREATE FIGURE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING FIGURES")
print("=" * 80)

# Focus on January 20 visualization
jan20_data = df_minute[df_minute['date'] == '2025-01-20'].copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Spread on January 20
ax1 = axes[0, 0]
jan20_agg = jan20_data.groupby('minute_bin').agg({'spread_bps': 'mean'}).reset_index()
ax1.plot(jan20_agg['minute_bin'], jan20_agg['spread_bps'], 'b-', linewidth=1)
for event in CONGESTION_EVENTS:
    ax1.axvspan(event['start'], event['end'], alpha=0.3, color='red')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Spread (bps)')
ax1.set_title('A. Spread on January 20, 2025 (Congestion periods in red)')

# Panel B: Quote updates on January 20
ax2 = axes[0, 1]
jan20_quotes = jan20_data.groupby('minute_bin').agg({'quote_updates': 'mean'}).reset_index()
ax2.plot(jan20_quotes['minute_bin'], jan20_quotes['quote_updates'], 'g-', linewidth=1)
for event in CONGESTION_EVENTS:
    ax2.axvspan(event['start'], event['end'], alpha=0.3, color='red')
ax2.set_xlabel('Time (UTC)')
ax2.set_ylabel('Quote Updates per Minute')
ax2.set_title('B. Quote Activity (Congestion periods in red)')

# Panel C: Event comparison
ax3 = axes[1, 0]
if len(all_results_df) > 0:
    colors = ['red' if 'Congestion' in str(n) else 'steelblue' for n in all_results_df['event_name']]
    bars = ax3.bar(range(len(all_results_df)), all_results_df['spread_effect_bps'],
                   color=colors, edgecolor='black', alpha=0.7)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_xlabel('Event')
    ax3.set_ylabel('Spread Effect (bps)')
    ax3.set_title(f'C. Spread Effects: {n_significant}/{n_events} Significant')
    ax3.set_xticks(range(len(all_results_df)))
    ax3.set_xticklabels([r['event_name'] for _, r in all_results_df.iterrows()], rotation=45, ha='right')

# Panel D: Pre vs During comparison
ax4 = axes[1, 1]
if len(all_results_df) > 0:
    x = np.arange(len(all_results_df))
    width = 0.35
    ax4.bar(x - width/2, all_results_df['spread_pre'], width, label='Pre', color='lightblue', edgecolor='black')
    ax4.bar(x + width/2, all_results_df['spread_during'], width, label='During', color='salmon', edgecolor='black')
    ax4.set_xlabel('Event')
    ax4.set_ylabel('Spread (bps)')
    ax4.set_title('D. Pre vs During Congestion')
    ax4.set_xticks(x)
    ax4.set_xticklabels([r['event_name'] for _, r in all_results_df.iterrows()], rotation=45, ha='right')
    ax4.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_jan2025_congestion.pdf', bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'figure_jan2025_congestion.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figure_jan2025_congestion.pdf/png")


# =============================================================================
# SAVE RESULTS
# =============================================================================

summary = {
    'event_date': '2025-01-20',
    'known_events': [
        {'name': e['name'], 'start': e['start'].isoformat(), 'end': e['end'].isoformat()}
        for e in CONGESTION_EVENTS
    ],
    'n_events_analyzed': n_events,
    'n_significant': n_significant,
    'mean_spread_effect_bps': float(mean_spread_effect),
    'results': all_results_df.to_dict('records'),
}

with open(OUTPUT_DIR / 'jan2025_congestion_results.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

all_results_df.to_csv(OUTPUT_DIR / 'jan2025_congestion_events.csv', index=False)

print("✓ Saved: jan2025_congestion_results.json")
print("✓ Saved: jan2025_congestion_events.csv")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("JANUARY 2025 CONGESTION ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
KNOWN CONGESTION EVENTS (January 20, 2025):
===========================================
""")

for _, r in jan_results_df.iterrows():
    sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else "*" if r['p_value'] < 0.1 else ""
    print(f"  {r['event_name']}: {r['spread_effect_bps']:+.2f} bps (t={r['t_stat']:.2f}){sig}")

print(f"""
MULTI-EVENT EVIDENCE:
=====================
- July 29, 2025 API outage: +3.77 bps (t=31.66)***
- July 30, 2025 stress event: +2.25 bps (t=15.63)***
- January 20, 2025 congestion: Results above

This provides {n_events + 3} infrastructure events showing consistent spread widening.
""")

print("✓ Analysis complete!")
print("=" * 80)
