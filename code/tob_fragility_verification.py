#!/usr/bin/env python3
"""
TOB Fragility Verification
==========================

Verify the paper's Table 34 claims:
- Fill HHI: t = -0.42
- Top-5 Share: t = -0.21
- Best-Price Fill Share: t = 3.67
- Fill Frequency Share: t = 4.28

The paper claims TOB proxies (Best-Price Fill, Fill Frequency) predict fragility
while fill-based measures (HHI, Top-5 Share) do not.
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
KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE', 'ARB', 'AVAX', 'DOGE', 'LINK', 'OP', 'SUI']

print("=" * 80)
print("TOB FRAGILITY VERIFICATION")
print("=" * 80)
print("\nVerifying paper's Table 34 claims about concentration → fragility")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/5] Loading data...")

fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_int'] = fills['date'].astype(int)
fills = fills[fills['coin'].isin(KEY_ASSETS)]

print(f"  Loaded {len(fills):,} fills")

# =============================================================================
# DEFINE WINDOWS
# =============================================================================

PRE_OUTAGE_DATE = 20250728
OUTAGE_DATE = 20250729
OUTAGE_HOUR = 14

# Get pre-outage maker fills (hour 13 on July 29 - the hour before outage)
pre_fills = fills[
    (fills['date_int'] == OUTAGE_DATE) &
    (fills['hour'] == 13) &
    (fills['crossed'] == False)
]

# Get outage maker fills
outage_fills = fills[
    (fills['date_int'] == OUTAGE_DATE) &
    (fills['hour'] == OUTAGE_HOUR) &
    (fills['crossed'] == False)
]

print(f"  Pre-outage maker fills (hour 13): {len(pre_fills):,}")
print(f"  Outage maker fills (hour 14): {len(outage_fills):,}")

# =============================================================================
# COMPUTE CONCENTRATION MEASURES BY ASSET
# =============================================================================

print("\n[2/5] Computing concentration measures by asset...")

def compute_asset_concentration(fills_df, coin):
    """
    Compute four concentration measures for an asset:
    1. Fill HHI - Herfindahl of volume shares
    2. Top-5 Share - Volume share of top 5 makers
    3. Best-Price Fill Share - Share of best-price fills from top 5 makers
    4. Fill Frequency Share - Share of fill COUNT from top 5 makers
    """
    coin_fills = fills_df[fills_df['coin'] == coin]

    if len(coin_fills) < 50:
        return None

    # Identify top 5 makers by volume
    maker_volumes = coin_fills.groupby('wallet')['sz'].sum()
    top5_wallets = set(maker_volumes.nlargest(5).index)

    total_volume = maker_volumes.sum()
    top5_volume = maker_volumes[maker_volumes.index.isin(top5_wallets)].sum()

    # 1. Fill HHI
    shares = maker_volumes / total_volume
    hhi = (shares ** 2).sum()

    # 2. Top-5 Volume Share
    top5_share = top5_volume / total_volume

    # 3. Best-Price Fill Share of Top 5
    # Compute hourly best prices
    coin_fills = coin_fills.copy()
    coin_fills['minute'] = coin_fills['timestamp'].dt.floor('T')

    # For each minute, find best bid and ask
    minute_best = coin_fills.groupby(['minute', 'side']).agg({
        'px': ['min', 'max']
    }).reset_index()
    minute_best.columns = ['minute', 'side', 'px_min', 'px_max']

    coin_fills = coin_fills.merge(minute_best, on=['minute', 'side'], how='left')

    # A fill is "at best" if within 0.1% of best price
    coin_fills['at_best'] = False
    buy_mask = coin_fills['side'] == 'B'
    sell_mask = coin_fills['side'] == 'A'

    # For buys (maker buying), best bid is px_max
    coin_fills.loc[buy_mask, 'at_best'] = (
        np.abs(coin_fills.loc[buy_mask, 'px'] - coin_fills.loc[buy_mask, 'px_max']) /
        coin_fills.loc[buy_mask, 'px'] < 0.001
    )
    # For sells (maker selling), best ask is px_min
    coin_fills.loc[sell_mask, 'at_best'] = (
        np.abs(coin_fills.loc[sell_mask, 'px'] - coin_fills.loc[sell_mask, 'px_min']) /
        coin_fills.loc[sell_mask, 'px'] < 0.001
    )

    best_fills = coin_fills[coin_fills['at_best']]
    if len(best_fills) > 0:
        top5_best = best_fills[best_fills['wallet'].isin(top5_wallets)]
        best_price_share = len(top5_best) / len(best_fills)
    else:
        best_price_share = np.nan

    # 4. Fill Frequency Share of Top 5
    top5_fills = coin_fills[coin_fills['wallet'].isin(top5_wallets)]
    fill_freq_share = len(top5_fills) / len(coin_fills)

    return {
        'coin': coin,
        'hhi': hhi,
        'top5_share': top5_share,
        'best_price_share': best_price_share,
        'fill_freq_share': fill_freq_share,
        'n_fills': len(coin_fills),
        'n_makers': coin_fills['wallet'].nunique()
    }

# Compute for pre-outage period
pre_metrics = []
for coin in KEY_ASSETS:
    result = compute_asset_concentration(pre_fills, coin)
    if result:
        pre_metrics.append(result)

pre_df = pd.DataFrame(pre_metrics)
print(f"\n  Pre-outage concentration by asset:")
print(pre_df[['coin', 'hhi', 'top5_share', 'best_price_share', 'fill_freq_share']].to_string(index=False))

# =============================================================================
# COMPUTE SPREAD CHANGES
# =============================================================================

print("\n[3/5] Computing spread changes...")

# Get taker fills for spread proxy
takers = fills[fills['crossed'] == True].copy()

def compute_spread_proxy(takers_df, coin, date, hour):
    """Compute spread proxy from price dispersion."""
    subset = takers_df[
        (takers_df['coin'] == coin) &
        (takers_df['date_int'] == date) &
        (takers_df['hour'] == hour)
    ]
    if len(subset) < 10:
        return np.nan
    return subset['px'].std() / subset['px'].mean() * 10000

# Compute spreads
spread_changes = []
for coin in KEY_ASSETS:
    pre_spread = compute_spread_proxy(takers, coin, OUTAGE_DATE, 13)
    outage_spread = compute_spread_proxy(takers, coin, OUTAGE_DATE, 14)

    if pd.notna(pre_spread) and pd.notna(outage_spread):
        spread_changes.append({
            'coin': coin,
            'pre_spread': pre_spread,
            'outage_spread': outage_spread,
            'spread_change': outage_spread - pre_spread,
            'spread_change_pct': (outage_spread - pre_spread) / pre_spread * 100
        })

spread_df = pd.DataFrame(spread_changes)
print(spread_df[['coin', 'pre_spread', 'outage_spread', 'spread_change']].to_string(index=False))

# =============================================================================
# MERGE AND RUN REGRESSIONS
# =============================================================================

print("\n[4/5] Running cross-sectional regressions...")

# Merge concentration and spread data
analysis_df = pre_df.merge(spread_df, on='coin')
print(f"\n  Assets with complete data: {len(analysis_df)}")

if len(analysis_df) < 5:
    print("  ERROR: Insufficient data for regression")
else:
    # Standardize for comparability
    for col in ['hhi', 'top5_share', 'best_price_share', 'fill_freq_share', 'spread_change']:
        mean_val = analysis_df[col].mean()
        std_val = analysis_df[col].std()
        if std_val > 0:
            analysis_df[f'{col}_std'] = (analysis_df[col] - mean_val) / std_val
        else:
            analysis_df[f'{col}_std'] = 0

    print("\n" + "=" * 70)
    print("CROSS-SECTIONAL FRAGILITY TEST: Concentration → Spread Widening")
    print("=" * 70)

    results = {}

    for measure, label in [
        ('hhi_std', 'Fill HHI'),
        ('top5_share_std', 'Top-5 Share'),
        ('best_price_share_std', 'Best-Price Fill'),
        ('fill_freq_share_std', 'Fill Frequency')
    ]:
        y = analysis_df['spread_change_std']
        X = sm.add_constant(analysis_df[measure])
        model = sm.OLS(y, X).fit()

        coef = model.params.iloc[1]
        t_stat = model.tvalues.iloc[1]
        r2 = model.rsquared

        sig = '***' if abs(t_stat) > 2.58 else ('**' if abs(t_stat) > 1.96 else '')

        print(f"\n  {label}:")
        print(f"    Coefficient: {coef:.3f}")
        print(f"    t-statistic: {t_stat:.2f} {sig}")
        print(f"    R²: {r2:.3f}")

        results[label] = {
            'coef': float(coef),
            't_stat': float(t_stat),
            'r2': float(r2)
        }

    # =============================================================================
    # COMPARE TO PAPER'S CLAIMS
    # =============================================================================

    print("\n" + "=" * 70)
    print("COMPARISON TO PAPER'S TABLE 34")
    print("=" * 70)

    paper_claims = {
        'Fill HHI': {'coef': -0.18, 't': -0.42},
        'Top-5 Share': {'coef': -0.09, 't': -0.21},
        'Best-Price Fill': {'coef': 1.42, 't': 3.67},
        'Fill Frequency': {'coef': 1.89, 't': 4.28}
    }

    print(f"\n  {'Measure':<20} {'Paper t':<12} {'Actual t':<12} {'Match?':<10}")
    print("  " + "-" * 54)

    for measure, paper in paper_claims.items():
        actual_t = results[measure]['t_stat']
        match = "YES" if (paper['t'] > 0) == (actual_t > 0) else "NO (sign)"
        if abs(paper['t']) > 2 and abs(actual_t) < 2:
            match = "NO (sig)"
        print(f"  {measure:<20} {paper['t']:<12.2f} {actual_t:<12.2f} {match:<10}")

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    output = {
        'n_assets': len(analysis_df),
        'assets': analysis_df['coin'].tolist(),
        'regression_results': results,
        'paper_claims': paper_claims,
        'data': analysis_df[['coin', 'hhi', 'top5_share', 'best_price_share',
                            'fill_freq_share', 'spread_change']].to_dict('records')
    }

    with open(OUTPUT_DIR / 'tob_fragility_verification.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n  Saved: tob_fragility_verification.json")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
