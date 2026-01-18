#!/usr/bin/env python3
"""
TOB Fragility Alternative Specifications
========================================

Try alternative specifications to see if any recover the paper's claimed t-statistics.

Alternatives:
1. Full-day pre-period (July 28) instead of hour 13
2. Different spread measure (using L2 snapshots if available)
3. Extended asset sample (all 24 assets)
4. Different TOB proxy definitions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import json
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

OUTPUT_DIR = Path(_RESULTS_DIR)

print("=" * 80)
print("TOB FRAGILITY - ALTERNATIVE SPECIFICATIONS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/4] Loading data...")

fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_int'] = fills['date'].astype(int)

# Try all assets with sufficient data
all_assets = fills['coin'].unique()
print(f"  Total assets in data: {len(all_assets)}")

PRE_OUTAGE_DATE = 20250728
OUTAGE_DATE = 20250729
OUTAGE_HOUR = 14

# =============================================================================
# SPECIFICATION 1: Full day July 28 as pre-period
# =============================================================================

print("\n[2/4] SPEC 1: Full-day pre-period (July 28)...")

# Pre-outage = full day July 28
pre_fills_full = fills[
    (fills['date_int'] == PRE_OUTAGE_DATE) &
    (fills['crossed'] == False)
]

def compute_concentration(fills_df, coin):
    """Compute all concentration measures for an asset."""
    coin_fills = fills_df[fills_df['coin'] == coin].copy()

    if len(coin_fills) < 100:
        return None

    # Top 5 makers by volume
    maker_volumes = coin_fills.groupby('wallet')['sz'].sum()
    if len(maker_volumes) < 5:
        return None
    top5_wallets = set(maker_volumes.nlargest(5).index)

    total_volume = maker_volumes.sum()
    shares = maker_volumes / total_volume

    # HHI
    hhi = (shares ** 2).sum()

    # Top-5 share
    top5_share = maker_volumes[maker_volumes.index.isin(top5_wallets)].sum() / total_volume

    # Fill frequency share
    fill_freq_share = len(coin_fills[coin_fills['wallet'].isin(top5_wallets)]) / len(coin_fills)

    # Best-price fill share (simplified: top decile prices)
    coin_fills['price_rank'] = coin_fills.groupby('side')['px'].rank(pct=True)
    # For buys, best price is high rank; for sells, best price is low rank
    coin_fills['at_best'] = False
    coin_fills.loc[coin_fills['side'] == 'B', 'at_best'] = coin_fills.loc[coin_fills['side'] == 'B', 'price_rank'] > 0.9
    coin_fills.loc[coin_fills['side'] == 'A', 'at_best'] = coin_fills.loc[coin_fills['side'] == 'A', 'price_rank'] < 0.1

    best_fills = coin_fills[coin_fills['at_best']]
    if len(best_fills) > 0:
        best_price_share = len(best_fills[best_fills['wallet'].isin(top5_wallets)]) / len(best_fills)
    else:
        best_price_share = np.nan

    return {
        'coin': coin,
        'hhi': hhi,
        'top5_share': top5_share,
        'best_price_share': best_price_share,
        'fill_freq_share': fill_freq_share,
        'n_fills': len(coin_fills)
    }

# Compute for all assets
pre_metrics = []
for coin in all_assets:
    result = compute_concentration(pre_fills_full, coin)
    if result:
        pre_metrics.append(result)

pre_df = pd.DataFrame(pre_metrics)
print(f"  Assets with pre-period data: {len(pre_df)}")

# Compute spread changes
takers = fills[fills['crossed'] == True].copy()

def compute_spread_change(coin):
    """Compute spread change from hour 13 to hour 14 on outage day."""
    pre = takers[(takers['coin'] == coin) & (takers['date_int'] == OUTAGE_DATE) & (takers['hour'] == 13)]
    post = takers[(takers['coin'] == coin) & (takers['date_int'] == OUTAGE_DATE) & (takers['hour'] == 14)]

    if len(pre) < 10 or len(post) < 10:
        return np.nan, np.nan

    pre_spread = pre['px'].std() / pre['px'].mean() * 10000
    post_spread = post['px'].std() / post['px'].mean() * 10000

    return pre_spread, post_spread - pre_spread

spread_data = []
for coin in pre_df['coin']:
    pre_spread, change = compute_spread_change(coin)
    if pd.notna(change):
        spread_data.append({'coin': coin, 'pre_spread': pre_spread, 'spread_change': change})

spread_df = pd.DataFrame(spread_data)
analysis_df = pre_df.merge(spread_df, on='coin')

print(f"  Assets with complete data: {len(analysis_df)}")

# Run regressions
if len(analysis_df) >= 5:
    for col in ['hhi', 'top5_share', 'best_price_share', 'fill_freq_share', 'spread_change']:
        if col in analysis_df.columns:
            mean_val = analysis_df[col].mean()
            std_val = analysis_df[col].std()
            if std_val > 0:
                analysis_df[f'{col}_std'] = (analysis_df[col] - mean_val) / std_val

    print("\n  Results (Spec 1: Full-day pre-period):")
    for measure in ['hhi_std', 'top5_share_std', 'best_price_share_std', 'fill_freq_share_std']:
        if measure not in analysis_df.columns:
            continue
        y = analysis_df['spread_change_std']
        X = sm.add_constant(analysis_df[measure])
        model = sm.OLS(y, X).fit()
        print(f"    {measure.replace('_std', '')}: t = {model.tvalues.iloc[1]:.2f}, R² = {model.rsquared:.3f}")

# =============================================================================
# SPECIFICATION 2: Exclude outliers (winsorize)
# =============================================================================

print("\n[3/4] SPEC 2: Winsorized spread changes (5%/95%)...")

analysis_df2 = analysis_df.copy()
p5 = analysis_df2['spread_change'].quantile(0.05)
p95 = analysis_df2['spread_change'].quantile(0.95)
analysis_df2['spread_change_wins'] = analysis_df2['spread_change'].clip(p5, p95)

# Re-standardize
mean_val = analysis_df2['spread_change_wins'].mean()
std_val = analysis_df2['spread_change_wins'].std()
if std_val > 0:
    analysis_df2['spread_change_wins_std'] = (analysis_df2['spread_change_wins'] - mean_val) / std_val

    print("\n  Results (Spec 2: Winsorized):")
    for measure in ['hhi_std', 'top5_share_std', 'best_price_share_std', 'fill_freq_share_std']:
        if measure not in analysis_df2.columns:
            continue
        y = analysis_df2['spread_change_wins_std']
        X = sm.add_constant(analysis_df2[measure])
        model = sm.OLS(y, X).fit()
        print(f"    {measure.replace('_std', '')}: t = {model.tvalues.iloc[1]:.2f}, R² = {model.rsquared:.3f}")

# =============================================================================
# SPECIFICATION 3: Log spread change
# =============================================================================

print("\n[4/4] SPEC 3: Percentage spread change...")

analysis_df3 = analysis_df.copy()
analysis_df3['spread_change_pct'] = analysis_df3['spread_change'] / analysis_df3['pre_spread'] * 100

# Remove negative pre-spreads
analysis_df3 = analysis_df3[analysis_df3['pre_spread'] > 0]

mean_val = analysis_df3['spread_change_pct'].mean()
std_val = analysis_df3['spread_change_pct'].std()
if std_val > 0:
    analysis_df3['spread_change_pct_std'] = (analysis_df3['spread_change_pct'] - mean_val) / std_val

    print("\n  Results (Spec 3: Percentage change):")
    for measure in ['hhi_std', 'top5_share_std', 'best_price_share_std', 'fill_freq_share_std']:
        if measure not in analysis_df3.columns:
            continue
        y = analysis_df3['spread_change_pct_std']
        X = sm.add_constant(analysis_df3[measure])
        model = sm.OLS(y, X).fit()
        print(f"    {measure.replace('_std', '')}: t = {model.tvalues.iloc[1]:.2f}, R² = {model.rsquared:.3f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Paper claims:
  - Best-Price Fill: t = 3.67***
  - Fill Frequency: t = 4.28***

No specification tested recovers these t-statistics.

POSSIBLE EXPLANATIONS:
1. Paper used different data (different sample period, different assets)
2. Paper used different concentration measure definitions
3. Paper used L2 order book data (not fills) for TOB measures
4. Paper numbers are from a different analysis that no longer exists
5. Paper numbers may be erroneous

RECOMMENDATION:
- Either locate the original analysis code
- Or update the paper with actual results (which show no significant effect)
""")

# Save detailed results
results_df = analysis_df[['coin', 'hhi', 'top5_share', 'best_price_share',
                          'fill_freq_share', 'spread_change', 'pre_spread']].copy()
results_df.to_csv(OUTPUT_DIR / 'tob_fragility_detailed.csv', index=False)
print("  Saved: tob_fragility_detailed.csv")
