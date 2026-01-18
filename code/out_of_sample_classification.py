#!/usr/bin/env python3
"""
OUT-OF-SAMPLE INFORMED TRADER CLASSIFICATION
=============================================

Strictly out-of-sample design:
1. CLASSIFICATION PERIOD: July 28, 2025 (day before outage)
2. LABELS FROZEN after classification
3. TEST PERIOD: July 29, 2025 (outage day)

Markout horizons: 1 second, 10 seconds, 1 minute, 5 minutes

Addresses referee concerns:
- Look-ahead bias: Classification uses ONLY pre-outage data
- Multiple horizons: Shows qualitative conclusions are invariant
- Risk adjustment: Reports Sharpe ratios, not just raw profits
- Funding payments: Uses price markouts, not realized PnL with funding

Author: Generated for referee robustness
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
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

# Markout horizons in milliseconds
HORIZONS = {
    '1s': 1000,
    '10s': 10000,
    '1m': 60000,
    '5m': 300000
}

print("="*80)
print("OUT-OF-SAMPLE INFORMED TRADER CLASSIFICATION")
print("="*80)
print("\nMethodology:")
print("  - Classification period: July 28, 2025 (pre-outage)")
print("  - Test period: July 29, 2025 (outage day)")
print("  - Markout horizons: 1s, 10s, 1m, 5m")
print("  - Labels FROZEN before test period")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/5] Loading data...")
fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_int'] = fills['date'].astype(int)

# Filter to key assets
fills = fills[fills['coin'].isin(KEY_ASSETS)]
print(f"  Loaded {len(fills):,} fills across {fills['coin'].nunique()} assets")

# Split into classification and test periods
CLASSIFICATION_DATE = 20250728
TEST_DATE = 20250729
OUTAGE_HOUR = 14

class_fills = fills[fills['date_int'] == CLASSIFICATION_DATE].copy()
test_fills = fills[fills['date_int'] == TEST_DATE].copy()

print(f"  Classification period (July 28): {len(class_fills):,} fills")
print(f"  Test period (July 29): {len(test_fills):,} fills")

# Separate takers (crossed=True) and makers (crossed=False)
class_takers = class_fills[class_fills['crossed'] == True].copy()
test_takers = test_fills[test_fills['crossed'] == True].copy()

print(f"  Classification takers: {len(class_takers):,} ({class_takers['wallet'].nunique():,} wallets)")
print(f"  Test takers: {len(test_takers):,} ({test_takers['wallet'].nunique():,} wallets)")

# =============================================================================
# COMPUTE MARKOUTS AT MULTIPLE HORIZONS
# =============================================================================

print("\n[2/5] Computing markouts at multiple horizons...")

def compute_markouts(df, horizons):
    """
    Compute price markouts at different time horizons.

    Markout = (Future Price - Entry Price) / Entry Price * 10000 (bps)
    Profit = Direction * Markout (positive = correct direction)

    This measures price movement, NOT realized PnL (avoids funding payment confound).
    """
    df = df.sort_values(['coin', 'timestamp']).copy()
    df['time_ms'] = df['time']  # Already in ms
    df['direction'] = np.where(df['side'] == 'B', 1, -1)

    for horizon_name, horizon_ms in horizons.items():
        df[f'markout_{horizon_name}'] = np.nan

        # For each asset, find future price at horizon
        for coin in df['coin'].unique():
            coin_mask = df['coin'] == coin
            coin_df = df[coin_mask].copy()

            if len(coin_df) < 2:
                continue

            # For each trade, find the first trade >= horizon_ms in the future
            times = coin_df['time_ms'].values
            prices = coin_df['px'].values
            idx = coin_df.index.values

            future_prices = np.full(len(coin_df), np.nan)

            for i in range(len(coin_df) - 1):
                target_time = times[i] + horizon_ms
                # Find first trade at or after target_time
                future_idx = np.searchsorted(times[i+1:], target_time)
                if future_idx + i + 1 < len(coin_df):
                    future_prices[i] = prices[future_idx + i + 1]

            # Compute markout in bps
            markouts = (future_prices - prices) / prices * 10000
            df.loc[idx, f'markout_{horizon_name}'] = markouts

        # Profit = direction * markout
        df[f'profit_{horizon_name}'] = df['direction'] * df[f'markout_{horizon_name}']

    return df

# Compute markouts for classification period
print("  Computing classification period markouts...")
class_takers = compute_markouts(class_takers, HORIZONS)

# Compute markouts for test period
print("  Computing test period markouts...")
test_takers = compute_markouts(test_takers, HORIZONS)

# Check markout coverage
for h in HORIZONS.keys():
    coverage = class_takers[f'profit_{h}'].notna().mean() * 100
    print(f"    {h}: {coverage:.1f}% coverage")

# =============================================================================
# CLASSIFICATION: BASED ONLY ON PRE-OUTAGE DATA
# =============================================================================

print("\n[3/5] Classifying wallets (STRICTLY pre-outage)...")

# Aggregate wallet-level statistics from classification period
wallet_stats = {}

for horizon_name in HORIZONS.keys():
    profit_col = f'profit_{horizon_name}'

    stats_df = class_takers.groupby('wallet').agg({
        profit_col: ['mean', 'std', 'count'],
        'sz': ['mean', 'sum']
    }).reset_index()
    stats_df.columns = ['wallet', 'mean_profit', 'profit_std', 'n_trades', 'avg_size', 'total_size']

    # Require minimum trades for reliable classification
    min_trades = 5
    stats_df = stats_df[stats_df['n_trades'] >= min_trades].copy()

    # Compute Sharpe ratio (risk-adjusted)
    stats_df['sharpe'] = stats_df['mean_profit'] / (stats_df['profit_std'] + 1e-6)

    # Classify into quintiles
    stats_df['quintile'] = pd.qcut(
        stats_df['mean_profit'].rank(method='first'), 5,
        labels=['Q1_Uninformed', 'Q2', 'Q3', 'Q4', 'Q5_Informed']
    )

    wallet_stats[horizon_name] = stats_df

    n_q5 = (stats_df['quintile'] == 'Q5_Informed').sum()
    n_q1 = (stats_df['quintile'] == 'Q1_Uninformed').sum()
    print(f"  {horizon_name}: {len(stats_df):,} wallets classified ({n_q5} Q5, {n_q1} Q1)")

# =============================================================================
# TEST: OUT-OF-SAMPLE PERFORMANCE
# =============================================================================

print("\n[4/5] Testing out-of-sample (July 29 - OUTAGE DAY)...")

results = {
    'horizons': {},
    'summary': {}
}

print(f"\n{'Horizon':<8} {'Q5 profit':>12} {'Q1 profit':>12} {'Spread':>10} {'t-stat':>10} {'p-value':>10}")
print("-" * 70)

for horizon_name in HORIZONS.keys():
    profit_col = f'profit_{horizon_name}'

    # Merge classification labels onto test data
    test_with_labels = test_takers.merge(
        wallet_stats[horizon_name][['wallet', 'quintile', 'mean_profit', 'sharpe']].rename(
            columns={'mean_profit': 'class_profit', 'sharpe': 'class_sharpe'}
        ),
        on='wallet',
        how='inner'
    )

    # Compute test period profits by classification
    q5_test = test_with_labels[test_with_labels['quintile'] == 'Q5_Informed'][profit_col].dropna()
    q1_test = test_with_labels[test_with_labels['quintile'] == 'Q1_Uninformed'][profit_col].dropna()

    if len(q5_test) > 10 and len(q1_test) > 10:
        q5_mean = q5_test.mean()
        q1_mean = q1_test.mean()
        spread = q5_mean - q1_mean

        t_stat, p_val = stats.ttest_ind(q5_test, q1_test)

        print(f"{horizon_name:<8} {q5_mean:>12.2f} {q1_mean:>12.2f} {spread:>10.2f} {t_stat:>10.2f} {p_val:>10.4f}")

        results['horizons'][horizon_name] = {
            'q5_mean_profit': float(q5_mean),
            'q1_mean_profit': float(q1_mean),
            'spread_bps': float(spread),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'n_q5_trades': len(q5_test),
            'n_q1_trades': len(q1_test)
        }
    else:
        print(f"{horizon_name:<8} {'Insufficient data'}")

# =============================================================================
# OUTAGE-SPECIFIC ANALYSIS
# =============================================================================

print("\n[5/5] Outage composition analysis...")

# Split test data into outage vs non-outage hours
test_outage = test_takers[test_takers['hour'] == OUTAGE_HOUR].copy()
test_normal = test_takers[test_takers['hour'] != OUTAGE_HOUR].copy()

print(f"  Outage hour fills: {len(test_outage):,}")
print(f"  Normal hours fills: {len(test_normal):,}")

# Use 1m markout for composition analysis
horizon = '1m'
stats_df = wallet_stats[horizon]

def compute_composition(fills_df, stats_df):
    """Compute informed/uninformed composition of trading."""
    merged = fills_df.merge(stats_df[['wallet', 'quintile']], on='wallet', how='left')

    total = len(merged)
    classified = merged['quintile'].notna().sum()

    if total == 0:
        return None

    q5_fills = (merged['quintile'] == 'Q5_Informed').sum()
    q1_fills = (merged['quintile'] == 'Q1_Uninformed').sum()

    return {
        'total_fills': total,
        'classified_fills': classified,
        'pct_classified': 100 * classified / total,
        'pct_q5_informed': 100 * q5_fills / total,
        'pct_q1_uninformed': 100 * q1_fills / total,
        'informed_ratio': q5_fills / (q1_fills + 1) if q1_fills > 0 else np.inf
    }

outage_comp = compute_composition(test_outage, stats_df)
normal_comp = compute_composition(test_normal, stats_df)

if outage_comp and normal_comp:
    print(f"\n  {'Metric':<30} {'Normal Hours':>15} {'Outage Hour':>15} {'Change':>12}")
    print("  " + "-" * 75)
    print(f"  {'Total fills':<30} {normal_comp['total_fills']:>15,} {outage_comp['total_fills']:>15,} {100*(outage_comp['total_fills']/normal_comp['total_fills']*23-1):>+11.1f}%")
    print(f"  {'% Informed (Q5)':<30} {normal_comp['pct_q5_informed']:>15.2f} {outage_comp['pct_q5_informed']:>15.2f} {outage_comp['pct_q5_informed']-normal_comp['pct_q5_informed']:>+11.2f}")
    print(f"  {'% Uninformed (Q1)':<30} {normal_comp['pct_q1_uninformed']:>15.2f} {outage_comp['pct_q1_uninformed']:>15.2f} {outage_comp['pct_q1_uninformed']-normal_comp['pct_q1_uninformed']:>+11.2f}")
    print(f"  {'Informed/Uninformed ratio':<30} {normal_comp['informed_ratio']:>15.2f} {outage_comp['informed_ratio']:>15.2f} {100*(outage_comp['informed_ratio']/normal_comp['informed_ratio']-1):>+11.1f}%")

    results['composition'] = {
        'outage_hour': outage_comp,
        'normal_hours': normal_comp,
        'informed_share_change_pct': outage_comp['pct_q5_informed'] - normal_comp['pct_q5_informed'],
        'uninformed_share_change_pct': outage_comp['pct_q1_uninformed'] - normal_comp['pct_q1_uninformed']
    }

# =============================================================================
# ROBUSTNESS: SHARPE-BASED CLASSIFICATION
# =============================================================================

print("\n" + "="*80)
print("ROBUSTNESS: SHARPE-BASED (RISK-ADJUSTED) CLASSIFICATION")
print("="*80)

# Re-classify using Sharpe ratio instead of raw profit
print("\nClassification by Sharpe ratio (controls for volatility/leverage):")
print(f"\n{'Horizon':<8} {'Q5 Sharpe':>12} {'Q1 Sharpe':>12} {'Test Spread':>12} {'t-stat':>10}")
print("-" * 60)

for horizon_name in HORIZONS.keys():
    profit_col = f'profit_{horizon_name}'
    stats_df = wallet_stats[horizon_name].copy()

    # Re-classify by Sharpe
    stats_df['quintile_sharpe'] = pd.qcut(
        stats_df['sharpe'].rank(method='first'), 5,
        labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']
    )

    # Test out-of-sample
    test_with_labels = test_takers.merge(
        stats_df[['wallet', 'quintile_sharpe', 'sharpe']],
        on='wallet',
        how='inner'
    )

    q5_test = test_with_labels[test_with_labels['quintile_sharpe'] == 'Q5_High'][profit_col].dropna()
    q1_test = test_with_labels[test_with_labels['quintile_sharpe'] == 'Q1_Low'][profit_col].dropna()

    if len(q5_test) > 10 and len(q1_test) > 10:
        q5_sharpe_mean = stats_df[stats_df['quintile_sharpe'] == 'Q5_High']['sharpe'].mean()
        q1_sharpe_mean = stats_df[stats_df['quintile_sharpe'] == 'Q1_Low']['sharpe'].mean()
        spread = q5_test.mean() - q1_test.mean()
        t_stat, _ = stats.ttest_ind(q5_test, q1_test)

        print(f"{horizon_name:<8} {q5_sharpe_mean:>12.3f} {q1_sharpe_mean:>12.3f} {spread:>12.2f} {t_stat:>10.2f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: OUT-OF-SAMPLE CLASSIFICATION VALIDATION")
print("="*80)

print("""
KEY DESIGN FEATURES:
1. Classification uses ONLY July 28 data (pre-outage)
2. Labels are FROZEN before examining July 29 (outage day)
3. Markouts are PRICE-BASED (not realized PnL, avoiding funding confound)
4. Multiple horizons show qualitative robustness

KEY FINDINGS:
""")

all_spreads_positive = True
all_significant = True

for h, res in results.get('horizons', {}).items():
    significant = res['p_value'] < 0.05
    positive = res['spread_bps'] > 0
    status = "✓" if (significant and positive) else "✗"
    print(f"  {h}: Q5-Q1 spread = {res['spread_bps']:+.2f} bps (t={res['t_stat']:.2f}, p={res['p_value']:.4f}) {status}")

    all_spreads_positive &= positive
    all_significant &= significant

if 'composition' in results:
    comp = results['composition']
    print(f"\nCOMPOSITION SHIFT DURING OUTAGE (pre-classified wallets):")
    print(f"  Informed share: {comp['informed_share_change_pct']:+.2f} pp")
    print(f"  Uninformed share: {comp['uninformed_share_change_pct']:+.2f} pp")

print(f"\nOVERALL VERDICT: {'ROBUST' if all_spreads_positive and all_significant else 'NEEDS REVIEW'}")
print("  - Classification predicts out-of-sample" if all_spreads_positive else "  - Classification does NOT predict out-of-sample")
print("  - Results consistent across horizons" if all_significant else "  - Results NOT consistent across horizons")

# Save results
results['summary'] = {
    'all_spreads_positive': all_spreads_positive,
    'all_significant': all_significant,
    'verdict': 'ROBUST' if (all_spreads_positive and all_significant) else 'NEEDS_REVIEW'
}

with open(OUTPUT_DIR / 'oos_classification_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved: oos_classification_results.json")
