#!/usr/bin/env python3
"""
Comprehensive Wallet-Level Analysis for "Resilient but Toxic" Paper

Analyzes 750M+ records across 5 months of Hyperliquid data:
- node_trades: March 22 - June 21, 2025 (pre-outage baseline)
- node_fills: July 27 - September 30, 2025 (outage + post-outage)

Key Analyses:
1. Maker concentration (HHI, top-N shares)
2. Shadow DMM identification
3. Informed vs uninformed trader classification
4. Outage vs normal period comparison
5. Time series visualizations

Author: Claude
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
import gc

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(_DATA_DIR)
FIGURES_DIR = Path(_FIGURES_DIR)
OUTPUT_DIR = Path(_RESULTS_DIR)

FIGURES_DIR.mkdir(exist_ok=True)

# Outage timing
OUTAGE_DATE = '20250729'
OUTAGE_START_HOUR = 14  # 14:10 UTC
OUTAGE_END_HOUR = 14    # 14:47 UTC

# Major assets
MAJOR_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE', 'ARB', 'OP', 'DOGE', 'LINK', 'AVAX', 'SUI']

print("=" * 80)
print("COMPREHENSIVE WALLET-LEVEL ANALYSIS")
print("Resilient but Toxic: Market Quality in On-Chain CLOBs")
print("=" * 80)

# ============================================================================
# PART 1: LOAD NODE_FILLS DATA (for outage analysis)
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: LOADING NODE_FILLS DATA")
print("=" * 80)

def load_node_fills_efficiently():
    """Load node_fills data, filtering to major assets."""
    merged_file = DATA_DIR / 'node_fills_merged.parquet'

    if merged_file.exists():
        print(f"Loading merged file: {merged_file}")
        fills_df = pd.read_parquet(merged_file)
        print(f"  Loaded {len(fills_df):,} total records")

        # Filter to major assets
        fills_df = fills_df[fills_df['coin'].isin(MAJOR_ASSETS)]
        print(f"  Filtered to major assets: {len(fills_df):,} records")
    else:
        # Fall back to part files
        parts = sorted(DATA_DIR.glob('node_fills_part_*.parquet'))
        print(f"Found {len(parts)} node_fills part files")

        dfs = []
        total_records = 0

        for i, p in enumerate(parts):
            df = pd.read_parquet(p)
            df = df[df['coin'].isin(MAJOR_ASSETS)]
            total_records += len(df)
            dfs.append(df)

            if (i + 1) % 50 == 0:
                print(f"  Loaded {i+1}/{len(parts)} files, {total_records:,} records")

            del df
            gc.collect()

        print(f"  Concatenating {len(dfs)} dataframes...")
        fills_df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

    print(f"  Total records (major assets): {len(fills_df):,}")
    print(f"  Date range: {fills_df['date'].min()} to {fills_df['date'].max()}")
    print(f"  Unique wallets: {fills_df['wallet'].nunique():,}")

    return fills_df

fills_df = load_node_fills_efficiently()

# ============================================================================
# PART 2: MAKER CONCENTRATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: MAKER CONCENTRATION ANALYSIS")
print("=" * 80)

def compute_hourly_concentration(df):
    """Compute hourly maker concentration metrics."""
    # Filter to makers only (crossed = False means maker)
    makers = df[df['crossed'] == False].copy()

    print(f"  Total fills: {len(df):,}")
    print(f"  Maker fills: {len(makers):,} ({100*len(makers)/len(df):.1f}%)")

    results = []

    for (date, hour), group in makers.groupby(['date', 'hour']):
        # Volume by wallet
        wallet_vol = group.groupby('wallet')['sz'].sum()
        total_vol = wallet_vol.sum()

        if total_vol == 0:
            continue

        # Market shares
        shares = wallet_vol / total_vol

        # HHI
        hhi = (shares ** 2).sum()

        # Top-N concentration
        sorted_shares = shares.sort_values(ascending=False)
        top1 = sorted_shares.iloc[0] if len(sorted_shares) > 0 else 0
        top5 = sorted_shares.iloc[:5].sum() if len(sorted_shares) >= 5 else sorted_shares.sum()
        top10 = sorted_shares.iloc[:10].sum() if len(sorted_shares) >= 10 else sorted_shares.sum()

        # Number of active makers
        n_makers = len(wallet_vol)

        # Is this the outage hour?
        is_outage = (date == OUTAGE_DATE and hour == OUTAGE_START_HOUR)

        results.append({
            'date': date,
            'hour': hour,
            'datetime': pd.to_datetime(f"{date[:4]}-{date[4:6]}-{date[6:]} {hour:02d}:00:00"),
            'n_makers': n_makers,
            'total_volume': total_vol,
            'hhi': hhi,
            'top1_share': top1,
            'top5_share': top5,
            'top10_share': top10,
            'is_outage': is_outage,
            'top1_wallet': sorted_shares.index[0] if len(sorted_shares) > 0 else None
        })

    return pd.DataFrame(results)

concentration_df = compute_hourly_concentration(fills_df)
print(f"  Computed concentration for {len(concentration_df)} hours")

# Summary stats
print("\n  CONCENTRATION SUMMARY:")
print(f"  Mean HHI: {concentration_df['hhi'].mean():.4f}")
print(f"  Mean Top-1 share: {concentration_df['top1_share'].mean()*100:.1f}%")
print(f"  Mean Top-5 share: {concentration_df['top5_share'].mean()*100:.1f}%")
print(f"  Mean Top-10 share: {concentration_df['top10_share'].mean()*100:.1f}%")
print(f"  Mean # makers: {concentration_df['n_makers'].mean():.0f}")

# ============================================================================
# PART 3: SHADOW DMM IDENTIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SHADOW DMM IDENTIFICATION")
print("=" * 80)

def identify_shadow_dmms(df):
    """Identify wallets that consistently provide liquidity."""
    makers = df[df['crossed'] == False].copy()

    # Aggregate by wallet
    wallet_stats = makers.groupby('wallet').agg({
        'sz': 'sum',
        'coin': 'count',
        'fee': 'sum',
        'date': 'nunique',
        'hour': 'nunique'
    }).rename(columns={
        'sz': 'total_volume',
        'coin': 'n_fills',
        'fee': 'total_fees',
        'date': 'n_days',
        'hour': 'n_hours'
    })

    wallet_stats['volume_per_day'] = wallet_stats['total_volume'] / wallet_stats['n_days']
    wallet_stats['fills_per_hour'] = wallet_stats['n_fills'] / wallet_stats['n_hours']
    wallet_stats = wallet_stats.sort_values('total_volume', ascending=False)

    # Calculate market share
    total_volume = wallet_stats['total_volume'].sum()
    wallet_stats['market_share'] = wallet_stats['total_volume'] / total_volume
    wallet_stats['cumulative_share'] = wallet_stats['market_share'].cumsum()

    return wallet_stats

wallet_stats = identify_shadow_dmms(fills_df)

print(f"\n  Total unique makers: {len(wallet_stats):,}")
print(f"\n  TOP 20 SHADOW DMMs:")
print("-" * 110)
print(f"{'Rank':<6} {'Wallet':<44} {'Volume':>14} {'Share':>8} {'Days':>6} {'Fills/Hr':>10}")
print("-" * 110)

for i, (wallet, row) in enumerate(wallet_stats.head(20).iterrows()):
    print(f"{i+1:<6} {wallet:<44} {row['total_volume']:>14,.0f} {row['market_share']*100:>7.2f}% {row['n_days']:>6} {row['fills_per_hour']:>10.1f}")

# Concentration stats
print(f"\n  OVERALL MAKER CONCENTRATION:")
top1_pct = wallet_stats.iloc[0]['market_share'] * 100
top5_pct = wallet_stats.head(5)['market_share'].sum() * 100
top10_pct = wallet_stats.head(10)['market_share'].sum() * 100
top20_pct = wallet_stats.head(20)['market_share'].sum() * 100

print(f"  Top 1 maker:  {top1_pct:5.1f}% of volume")
print(f"  Top 5 makers: {top5_pct:5.1f}% of volume")
print(f"  Top 10 makers: {top10_pct:5.1f}% of volume")
print(f"  Top 20 makers: {top20_pct:5.1f}% of volume")

# How many makers for 50%, 80%, 90%?
n_50 = (wallet_stats['cumulative_share'] <= 0.50).sum() + 1
n_80 = (wallet_stats['cumulative_share'] <= 0.80).sum() + 1
n_90 = (wallet_stats['cumulative_share'] <= 0.90).sum() + 1
print(f"\n  Makers needed for 50% of volume: {n_50}")
print(f"  Makers needed for 80% of volume: {n_80}")
print(f"  Makers needed for 90% of volume: {n_90}")

# ============================================================================
# PART 4: INFORMED VS UNINFORMED TRADER ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: INFORMED VS UNINFORMED TRADER ANALYSIS")
print("=" * 80)

def classify_traders_by_profitability(df, window_dates):
    """
    Classify takers as informed or uninformed based on realized profitability.

    Informed: trades that are profitable (bought before price up, sold before price down)
    Uninformed: trades that are unprofitable
    """
    # Filter to takers and window
    takers = df[(df['crossed'] == True) & (df['date'].isin(window_dates))].copy()

    if len(takers) == 0:
        return pd.DataFrame()

    # For each coin, compute price change over next hour
    results = []

    for coin in MAJOR_ASSETS:
        coin_df = takers[takers['coin'] == coin].copy()
        if len(coin_df) == 0:
            continue

        # Get hourly prices (use last trade price as proxy)
        hourly_px = coin_df.groupby(['date', 'hour'])['px'].last().reset_index()
        hourly_px['next_px'] = hourly_px['px'].shift(-1)
        hourly_px['price_change'] = (hourly_px['next_px'] - hourly_px['px']) / hourly_px['px']

        # Merge back
        coin_df = coin_df.merge(hourly_px[['date', 'hour', 'price_change']], on=['date', 'hour'], how='left')

        # Classify: Buy before price up = informed, Sell before price down = informed
        coin_df['is_buy'] = coin_df['side'] == 'B'
        coin_df['is_informed'] = (
            ((coin_df['is_buy']) & (coin_df['price_change'] > 0)) |
            ((~coin_df['is_buy']) & (coin_df['price_change'] < 0))
        )

        results.append(coin_df)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# Analyze around the outage: July 28-30
outage_window = ['20250728', '20250729', '20250730']
classified_df = classify_traders_by_profitability(fills_df, outage_window)

if len(classified_df) > 0:
    # Compute I/U ratio by hour
    iu_by_hour = classified_df.groupby(['date', 'hour']).agg({
        'is_informed': ['sum', 'count']
    })
    iu_by_hour.columns = ['informed_count', 'total_count']
    iu_by_hour['uninformed_count'] = iu_by_hour['total_count'] - iu_by_hour['informed_count']
    iu_by_hour['iu_ratio'] = iu_by_hour['informed_count'] / iu_by_hour['uninformed_count'].replace(0, np.nan)
    iu_by_hour = iu_by_hour.reset_index()
    iu_by_hour['is_outage'] = (iu_by_hour['date'] == OUTAGE_DATE) & (iu_by_hour['hour'] == OUTAGE_START_HOUR)

    # Compare outage vs normal
    outage_stats = iu_by_hour[iu_by_hour['is_outage']]
    normal_stats = iu_by_hour[~iu_by_hour['is_outage']]

    print(f"\n  INFORMED/UNINFORMED RATIO ANALYSIS:")
    print(f"  Total classified trades: {len(classified_df):,}")
    print(f"\n  Normal hours (n={len(normal_stats)}):")
    print(f"    Mean I/U ratio: {normal_stats['iu_ratio'].mean():.2f}")
    print(f"    Median I/U ratio: {normal_stats['iu_ratio'].median():.2f}")

    if len(outage_stats) > 0:
        print(f"\n  Outage hour:")
        print(f"    I/U ratio: {outage_stats['iu_ratio'].values[0]:.2f}")
        print(f"    Informed trades: {outage_stats['informed_count'].values[0]:,}")
        print(f"    Uninformed trades: {outage_stats['uninformed_count'].values[0]:,}")

        # Statistical test
        normal_iu = normal_stats['iu_ratio'].dropna()
        outage_iu = outage_stats['iu_ratio'].values[0]
        z_score = (outage_iu - normal_iu.mean()) / normal_iu.std()
        print(f"\n  Z-score (outage vs normal): {z_score:.2f}")

        # Change in informed vs uninformed
        normal_informed_pct = normal_stats['informed_count'].sum() / normal_stats['total_count'].sum() * 100
        outage_informed_pct = outage_stats['informed_count'].values[0] / outage_stats['total_count'].values[0] * 100
        print(f"\n  Informed trader share:")
        print(f"    Normal: {normal_informed_pct:.1f}%")
        print(f"    Outage: {outage_informed_pct:.1f}%")
        print(f"    Change: {outage_informed_pct - normal_informed_pct:+.1f}pp")

# ============================================================================
# PART 5: OUTAGE VS NORMAL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: OUTAGE VS NORMAL PERIOD COMPARISON")
print("=" * 80)

# Get outage hour data
outage_data = concentration_df[concentration_df['is_outage']]
normal_data = concentration_df[~concentration_df['is_outage']]

print(f"\n  CONCENTRATION METRICS:")
print(f"  {'Metric':<20} {'Normal (mean)':>15} {'Outage':>15} {'Change':>15} {'% Change':>12}")
print("-" * 80)

metrics = ['n_makers', 'hhi', 'top1_share', 'top5_share', 'top10_share']
for metric in metrics:
    normal_mean = normal_data[metric].mean()
    if len(outage_data) > 0:
        outage_val = outage_data[metric].values[0]
        diff = outage_val - normal_mean
        pct_change = 100 * diff / normal_mean if normal_mean != 0 else np.nan
        print(f"  {metric:<20} {normal_mean:>15.4f} {outage_val:>15.4f} {diff:>+15.4f} {pct_change:>+11.1f}%")
    else:
        print(f"  {metric:<20} {normal_mean:>15.4f} {'N/A':>15} {'N/A':>15} {'N/A':>12}")

# ============================================================================
# PART 6: GENERATE FIGURES
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: GENERATING FIGURES")
print("=" * 80)

# Figure 1: Maker Concentration Over Time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get July 28-30 data for detailed view
july_data = concentration_df[concentration_df['date'].isin(['20250728', '20250729', '20250730'])]

# Panel A: HHI over time (July 28-30)
ax1 = axes[0, 0]
for date in ['20250728', '20250729', '20250730']:
    date_data = july_data[july_data['date'] == date]
    label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    color = 'red' if date == OUTAGE_DATE else 'steelblue'
    alpha = 1.0 if date == OUTAGE_DATE else 0.6
    ax1.plot(date_data['hour'], date_data['hhi'], 'o-', label=label, color=color, alpha=alpha)

ax1.axvspan(14, 15, alpha=0.3, color='red', label='Outage Hour')
ax1.set_xlabel('Hour (UTC)')
ax1.set_ylabel('HHI (Herfindahl Index)')
ax1.set_title('A. Maker Concentration During Outage Window')
ax1.legend()
ax1.set_xlim(0, 23)

# Panel B: Number of active makers
ax2 = axes[0, 1]
for date in ['20250728', '20250729', '20250730']:
    date_data = july_data[july_data['date'] == date]
    label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    color = 'red' if date == OUTAGE_DATE else 'steelblue'
    alpha = 1.0 if date == OUTAGE_DATE else 0.6
    ax2.plot(date_data['hour'], date_data['n_makers'], 'o-', label=label, color=color, alpha=alpha)

ax2.axvspan(14, 15, alpha=0.3, color='red', label='Outage Hour')
ax2.set_xlabel('Hour (UTC)')
ax2.set_ylabel('Number of Unique Makers')
ax2.set_title('B. Maker Participation During Outage')
ax2.legend()
ax2.set_xlim(0, 23)

# Panel C: Top-N concentration bar chart
ax3 = axes[1, 0]
top_wallets = wallet_stats.head(20)
shares = top_wallets['market_share'] * 100
ax3.bar(range(len(shares)), shares, color='steelblue', alpha=0.7)
ax3.set_xlabel('Maker Rank')
ax3.set_ylabel('% of Total Volume')
ax3.set_title('C. Volume Share by Maker Rank (Full Sample)')
ax3.set_xticks(range(0, 20, 2))
ax3.set_xticklabels(range(1, 21, 2))

# Panel D: Cumulative concentration curve
ax4 = axes[1, 1]
cumulative = wallet_stats['cumulative_share'].values[:100] * 100
ax4.plot(range(1, len(cumulative)+1), cumulative, 'b-', linewidth=2)
ax4.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
ax4.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80%')
ax4.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90%')
ax4.set_xlabel('Number of Top Makers')
ax4.set_ylabel('Cumulative % of Volume')
ax4.set_title('D. Cumulative Concentration Curve')
ax4.set_xlim(1, 100)
ax4.set_ylim(0, 100)
ax4.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_maker_concentration.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'figure_maker_concentration.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure_maker_concentration.pdf/png")

# Figure 2: Informed/Uninformed Analysis
if len(classified_df) > 0 and 'iu_by_hour' in dir():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: I/U ratio over time
    ax1 = axes[0]
    for date in ['20250728', '20250729', '20250730']:
        date_data = iu_by_hour[iu_by_hour['date'] == date]
        if len(date_data) > 0:
            label = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            color = 'red' if date == OUTAGE_DATE else 'steelblue'
            alpha = 1.0 if date == OUTAGE_DATE else 0.6
            ax1.plot(date_data['hour'], date_data['iu_ratio'], 'o-', label=label, color=color, alpha=alpha)

    ax1.axvspan(14, 15, alpha=0.3, color='red', label='Outage Hour')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='I/U = 1')
    ax1.set_xlabel('Hour (UTC)')
    ax1.set_ylabel('Informed/Uninformed Ratio')
    ax1.set_title('A. I/U Ratio During Outage Window')
    ax1.legend()
    ax1.set_xlim(0, 23)

    # Panel B: Distribution comparison
    ax2 = axes[1]
    normal_iu = iu_by_hour[~iu_by_hour['is_outage']]['iu_ratio'].dropna()
    ax2.hist(normal_iu, bins=20, alpha=0.7, color='steelblue', label='Normal Hours', density=True)
    if len(outage_stats) > 0:
        outage_iu_val = outage_stats['iu_ratio'].values[0]
        ax2.axvline(outage_iu_val, color='red', linewidth=2, label=f'Outage Hour ({outage_iu_val:.2f})')
    ax2.axvline(normal_iu.mean(), color='steelblue', linestyle='--', linewidth=2, label=f'Normal Mean ({normal_iu.mean():.2f})')
    ax2.set_xlabel('Informed/Uninformed Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('B. Distribution of I/U Ratio')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_informed_outage.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_informed_outage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figure_informed_outage.pdf/png")

# Figure 3: Time Series of Concentration (Full Sample)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Convert to datetime for plotting
concentration_df['datetime'] = pd.to_datetime(concentration_df['date'], format='%Y%m%d') + pd.to_timedelta(concentration_df['hour'], unit='h')

# Daily averages for cleaner plot
daily_conc = concentration_df.groupby('date').agg({
    'hhi': 'mean',
    'top5_share': 'mean',
    'n_makers': 'mean'
}).reset_index()
daily_conc['datetime'] = pd.to_datetime(daily_conc['date'], format='%Y%m%d')

# Panel A: HHI over full sample
ax1 = axes[0]
ax1.plot(daily_conc['datetime'], daily_conc['hhi'], 'b-', linewidth=1, alpha=0.8)
ax1.axvline(pd.to_datetime('2025-07-29'), color='red', linestyle='--', linewidth=2, label='API Outage')
ax1.set_ylabel('HHI (Daily Average)')
ax1.set_title('A. Maker Concentration Over Time')
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Panel B: Number of makers over time
ax2 = axes[1]
ax2.plot(daily_conc['datetime'], daily_conc['n_makers'], 'g-', linewidth=1, alpha=0.8)
ax2.axvline(pd.to_datetime('2025-07-29'), color='red', linestyle='--', linewidth=2, label='API Outage')
ax2.set_xlabel('Date')
ax2.set_ylabel('Number of Makers (Daily Average)')
ax2.set_title('B. Maker Participation Over Time')
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_concentration_timeseries.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'figure_concentration_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure_concentration_timeseries.pdf/png")

# ============================================================================
# PART 7: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("PART 7: SAVING RESULTS")
print("=" * 80)

# Save concentration data
concentration_df.to_csv(OUTPUT_DIR / 'maker_concentration_hourly.csv', index=False)
print(f"  Saved: maker_concentration_hourly.csv ({len(concentration_df)} rows)")

# Save wallet stats
wallet_stats.to_csv(OUTPUT_DIR / 'maker_wallet_stats.csv')
print(f"  Saved: maker_wallet_stats.csv ({len(wallet_stats)} wallets)")

# Save summary statistics
summary = {
    'analysis_date': datetime.now().isoformat(),
    'data_sources': {
        'node_fills': {
            'records': len(fills_df),
            'date_range': f"{fills_df['date'].min()} to {fills_df['date'].max()}",
            'unique_wallets': fills_df['wallet'].nunique()
        }
    },
    'concentration': {
        'mean_hhi': concentration_df['hhi'].mean(),
        'mean_top1_share': concentration_df['top1_share'].mean(),
        'mean_top5_share': concentration_df['top5_share'].mean(),
        'mean_top10_share': concentration_df['top10_share'].mean(),
        'mean_n_makers': concentration_df['n_makers'].mean()
    },
    'shadow_dmms': {
        'total_makers': len(wallet_stats),
        'top1_share': wallet_stats.iloc[0]['market_share'],
        'top5_share': wallet_stats.head(5)['market_share'].sum(),
        'top10_share': wallet_stats.head(10)['market_share'].sum(),
        'top20_share': wallet_stats.head(20)['market_share'].sum(),
        'makers_for_50pct': int(n_50),
        'makers_for_80pct': int(n_80),
        'makers_for_90pct': int(n_90)
    }
}

import json
with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved: analysis_summary.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
KEY FINDINGS:

1. MAKER CONCENTRATION
   - Mean HHI: {concentration_df['hhi'].mean():.4f}
   - Top 5 makers control: {wallet_stats.head(5)['market_share'].sum()*100:.1f}% of volume
   - Top 10 makers control: {wallet_stats.head(10)['market_share'].sum()*100:.1f}% of volume
   - Only {n_50} makers needed for 50% of volume

2. SHADOW DMMs
   - {len(wallet_stats):,} unique makers identified
   - Top maker: {wallet_stats.iloc[0]['market_share']*100:.1f}% market share
   - Consistent participation across {wallet_stats.iloc[0]['n_days']} days

3. OUTAGE IMPACT
   - During outage, concentration {'increased' if outage_data['hhi'].values[0] > normal_data['hhi'].mean() else 'decreased'}
   - HHI: {normal_data['hhi'].mean():.4f} (normal) vs {outage_data['hhi'].values[0]:.4f} (outage)

FIGURES GENERATED:
   - figure_maker_concentration.pdf
   - figure_informed_outage.pdf
   - figure_concentration_timeseries.pdf

DATA SAVED:
   - maker_concentration_hourly.csv
   - maker_wallet_stats.csv
   - analysis_summary.json
""")
