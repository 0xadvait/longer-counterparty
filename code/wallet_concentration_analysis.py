#!/usr/bin/env python3
"""
Wallet-Level Liquidity Provision Concentration Analysis

Analyzes maker concentration during the July 29, 2025 API outage.
Uses fills data from hl-mainnet-node-data bucket.

Key questions:
1. How concentrated is liquidity provision? (Herfindahl index)
2. Did concentration change during the API outage?
3. Are there "shadow DMMs" - wallets that consistently provide liquidity?
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
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

# AWS Configuration for requester-pays bucket
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
FILLS_BUCKET = 'hl-mainnet-node-data'

# Study dates: July 28-30, 2025 (pre/during/post outage)
STUDY_DATES = ['20250728', '20250729', '20250730']
OUTAGE_DATE = '20250729'
OUTAGE_START_HOUR = 14  # 14:10 UTC
OUTAGE_END_HOUR = 14    # 14:47 UTC (same hour)

# Major assets to focus on
MAJOR_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE', 'ARB', 'OP', 'DOGE', 'LINK', 'AVAX', 'SUI']


def download_fills_hour(s3, date, hour):
    """Download and parse fills for a specific hour."""
    key = f'node_fills_by_block/hourly/{date}/{hour}.lz4'

    try:
        obj = s3.get_object(Bucket=FILLS_BUCKET, Key=key, RequestPayer='requester')
        compressed = obj['Body'].read()
        decompressed = lz4.frame.decompress(compressed)
        lines = decompressed.decode('utf-8').strip().split('\n')

        fills = []
        for line in lines:
            record = json.loads(line)
            block_time = record.get('block_time', '')

            for event in record.get('events', []):
                if isinstance(event, list) and len(event) == 2:
                    wallet, fill_data = event

                    # Only include major assets
                    coin = fill_data.get('coin', '')
                    if coin not in MAJOR_ASSETS:
                        continue

                    fills.append({
                        'wallet': wallet,
                        'coin': coin,
                        'px': float(fill_data.get('px', 0)),
                        'sz': float(fill_data.get('sz', 0)),
                        'side': fill_data.get('side', ''),
                        'time': fill_data.get('time', 0),
                        'crossed': fill_data.get('crossed', True),  # True = taker, False = maker
                        'fee': float(fill_data.get('fee', 0)),
                        'date': date,
                        'hour': hour
                    })

        return fills
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return []


def download_all_fills():
    """Download fills for all study dates."""
    s3 = boto3.client('s3', **AWS_CONFIG)

    # Build list of all hours to download
    hours_to_download = []
    for date in STUDY_DATES:
        for hour in range(24):
            hours_to_download.append((date, hour))

    print(f"Downloading {len(hours_to_download)} hours of fills data...")
    print(f"Dates: {STUDY_DATES}")
    print(f"Assets: {MAJOR_ASSETS}")

    all_fills = []

    # Download in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_fills_hour, s3, date, hour): (date, hour)
            for date, hour in hours_to_download
        }

        completed = 0
        for future in as_completed(futures):
            date, hour = futures[future]
            fills = future.result()
            all_fills.extend(fills)
            completed += 1

            if completed % 12 == 0:
                print(f"  Progress: {completed}/{len(hours_to_download)} hours, {len(all_fills):,} fills")

    print(f"\nTotal fills downloaded: {len(all_fills):,}")
    return pd.DataFrame(all_fills)


def compute_concentration_metrics(df):
    """Compute concentration metrics for makers."""

    # Filter to makers only (crossed = False)
    makers_df = df[df['crossed'] == False].copy()

    print(f"\n{'='*80}")
    print("CONCENTRATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal fills: {len(df):,}")
    print(f"Maker fills: {len(makers_df):,} ({100*len(makers_df)/len(df):.1f}%)")
    print(f"Taker fills: {len(df) - len(makers_df):,} ({100*(len(df)-len(makers_df))/len(df):.1f}%)")

    # Compute hourly maker concentration
    results = []

    for (date, hour), group in makers_df.groupby(['date', 'hour']):
        # Volume by wallet
        wallet_volume = group.groupby('wallet').agg({
            'sz': 'sum',
            'coin': 'count'
        }).rename(columns={'sz': 'volume', 'coin': 'n_fills'})

        total_volume = wallet_volume['volume'].sum()

        if total_volume == 0:
            continue

        # Market shares
        wallet_volume['share'] = wallet_volume['volume'] / total_volume

        # Herfindahl-Hirschman Index (HHI)
        hhi = (wallet_volume['share'] ** 2).sum()

        # Top-N concentration
        sorted_shares = wallet_volume['share'].sort_values(ascending=False)
        top1_share = sorted_shares.iloc[0] if len(sorted_shares) > 0 else 0
        top5_share = sorted_shares.iloc[:5].sum() if len(sorted_shares) >= 5 else sorted_shares.sum()
        top10_share = sorted_shares.iloc[:10].sum() if len(sorted_shares) >= 10 else sorted_shares.sum()

        # Number of unique makers
        n_makers = len(wallet_volume)

        # Is this the outage hour?
        is_outage = (date == OUTAGE_DATE and hour == OUTAGE_START_HOUR)

        results.append({
            'date': date,
            'hour': hour,
            'n_makers': n_makers,
            'total_volume': total_volume,
            'hhi': hhi,
            'top1_share': top1_share,
            'top5_share': top5_share,
            'top10_share': top10_share,
            'is_outage': is_outage,
            'top1_wallet': sorted_shares.index[0] if len(sorted_shares) > 0 else None
        })

    return pd.DataFrame(results)


def analyze_shadow_dmms(df):
    """Identify wallets that consistently provide liquidity."""

    makers_df = df[df['crossed'] == False].copy()

    # Aggregate by wallet
    wallet_stats = makers_df.groupby('wallet').agg({
        'sz': 'sum',
        'coin': 'count',
        'date': 'nunique',
        'hour': 'nunique'
    }).rename(columns={
        'sz': 'total_volume',
        'coin': 'n_fills',
        'date': 'n_days',
        'hour': 'n_hours'
    })

    wallet_stats['fills_per_hour'] = wallet_stats['n_fills'] / wallet_stats['n_hours']
    wallet_stats = wallet_stats.sort_values('total_volume', ascending=False)

    print(f"\n{'='*80}")
    print("SHADOW DMM ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal unique makers: {len(wallet_stats):,}")

    # Top 20 makers
    print(f"\nTop 20 Makers by Volume:")
    print("-" * 100)
    print(f"{'Wallet':<44} {'Volume':>12} {'Fills':>8} {'Days':>6} {'Hours':>6} {'Fills/Hr':>10}")
    print("-" * 100)

    for wallet, row in wallet_stats.head(20).iterrows():
        print(f"{wallet:<44} {row['total_volume']:>12,.0f} {row['n_fills']:>8,} "
              f"{row['n_days']:>6} {row['n_hours']:>6} {row['fills_per_hour']:>10.1f}")

    # Concentration stats
    total_volume = wallet_stats['total_volume'].sum()
    top1_pct = 100 * wallet_stats.iloc[0]['total_volume'] / total_volume
    top5_pct = 100 * wallet_stats.head(5)['total_volume'].sum() / total_volume
    top10_pct = 100 * wallet_stats.head(10)['total_volume'].sum() / total_volume
    top20_pct = 100 * wallet_stats.head(20)['total_volume'].sum() / total_volume

    print(f"\n{'='*60}")
    print("OVERALL CONCENTRATION (3 days)")
    print(f"{'='*60}")
    print(f"Top 1 maker:  {top1_pct:5.1f}% of volume")
    print(f"Top 5 makers: {top5_pct:5.1f}% of volume")
    print(f"Top 10 makers: {top10_pct:5.1f}% of volume")
    print(f"Top 20 makers: {top20_pct:5.1f}% of volume")

    return wallet_stats


def compare_outage_vs_normal(concentration_df):
    """Compare concentration during outage vs normal hours."""

    print(f"\n{'='*80}")
    print("OUTAGE vs NORMAL COMPARISON")
    print(f"{'='*80}")

    outage = concentration_df[concentration_df['is_outage']]
    normal = concentration_df[~concentration_df['is_outage']]

    metrics = ['n_makers', 'hhi', 'top1_share', 'top5_share', 'top10_share']

    print(f"\n{'Metric':<20} {'Normal (mean)':>15} {'Outage':>15} {'Diff':>15} {'% Change':>12}")
    print("-" * 80)

    for metric in metrics:
        normal_mean = normal[metric].mean()
        outage_val = outage[metric].values[0] if len(outage) > 0 else np.nan
        diff = outage_val - normal_mean
        pct_change = 100 * diff / normal_mean if normal_mean != 0 else np.nan

        print(f"{metric:<20} {normal_mean:>15.4f} {outage_val:>15.4f} {diff:>+15.4f} {pct_change:>+11.1f}%")

    return outage, normal


def create_visualizations(concentration_df, wallet_stats, df):
    """Create visualizations of concentration analysis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: HHI over time
    ax1 = axes[0, 0]
    for date in STUDY_DATES:
        date_data = concentration_df[concentration_df['date'] == date]
        label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        color = 'red' if date == OUTAGE_DATE else 'steelblue'
        ax1.plot(date_data['hour'], date_data['hhi'], 'o-', label=label, color=color, alpha=0.7)

    ax1.axvspan(14, 15, alpha=0.3, color='red', label='Outage Hour')
    ax1.set_xlabel('Hour (UTC)')
    ax1.set_ylabel('HHI (Herfindahl Index)')
    ax1.set_title('A. Maker Concentration Over Time')
    ax1.legend()
    ax1.set_ylim(0, None)

    # Panel B: Number of makers over time
    ax2 = axes[0, 1]
    for date in STUDY_DATES:
        date_data = concentration_df[concentration_df['date'] == date]
        label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        color = 'red' if date == OUTAGE_DATE else 'steelblue'
        ax2.plot(date_data['hour'], date_data['n_makers'], 'o-', label=label, color=color, alpha=0.7)

    ax2.axvspan(14, 15, alpha=0.3, color='red', label='Outage Hour')
    ax2.set_xlabel('Hour (UTC)')
    ax2.set_ylabel('Number of Unique Makers')
    ax2.set_title('B. Maker Participation Over Time')
    ax2.legend()

    # Panel C: Top-N concentration
    ax3 = axes[1, 0]
    top_wallets = wallet_stats.head(20)
    total_vol = wallet_stats['total_volume'].sum()
    shares = 100 * top_wallets['total_volume'] / total_vol
    ax3.bar(range(len(shares)), shares, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Maker Rank')
    ax3.set_ylabel('% of Total Volume')
    ax3.set_title('C. Volume Share by Maker Rank')
    ax3.set_xticks(range(0, 20, 2))
    ax3.set_xticklabels(range(1, 21, 2))

    # Panel D: Cumulative concentration
    ax4 = axes[1, 1]
    cumulative = 100 * wallet_stats['total_volume'].cumsum() / total_vol
    ax4.plot(range(1, min(101, len(cumulative)+1)), cumulative.values[:100], 'b-', linewidth=2)
    ax4.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax4.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80%')
    ax4.set_xlabel('Number of Top Makers')
    ax4.set_ylabel('Cumulative % of Volume')
    ax4.set_title('D. Cumulative Concentration Curve')
    ax4.set_xlim(1, 100)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(_FIGURES_DIR, 'figure_maker_concentration.pdf'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(_FIGURES_DIR, 'figure_maker_concentration.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSaved: figures/figure_maker_concentration.pdf/png")


def main():
    print("=" * 80)
    print("WALLET-LEVEL LIQUIDITY PROVISION ANALYSIS")
    print("API Outage Natural Experiment: July 28-30, 2025")
    print("=" * 80)

    # Check for cached data
    cache_file = os.path.join(_DATA_DIR, 'wallet_fills_data.parquet')

    try:
        print(f"\nLoading cached data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        print(f"Loaded {len(df):,} fills from cache")
    except:
        print("\nNo cache found, downloading from S3...")
        df = download_all_fills()
        df.to_parquet(cache_file)
        print(f"Saved to {cache_file}")

    # Compute concentration metrics
    concentration_df = compute_concentration_metrics(df)

    # Analyze shadow DMMs
    wallet_stats = analyze_shadow_dmms(df)

    # Compare outage vs normal
    outage, normal = compare_outage_vs_normal(concentration_df)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(concentration_df, wallet_stats, df)

    # Save results
    concentration_df.to_csv(os.path.join(_RESULTS_DIR, 'maker_concentration_hourly.csv'), index=False)
    wallet_stats.to_csv(os.path.join(_RESULTS_DIR, 'maker_wallet_stats.csv'))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
