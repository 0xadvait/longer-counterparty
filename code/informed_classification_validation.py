#!/usr/bin/env python3
"""
INFORMED CLASSIFICATION VALIDATION
===================================

Validates that "informed taker" classification reflects genuine information advantage,
not just:
1. Leverage/risk-taking (bigger bets = bigger wins/losses)
2. Latency/execution advantage (MEV-style front-running)
3. Mechanical bias from same-window evaluation
4. Random noise that doesn't persist out-of-sample

Tests:
1. OUT-OF-SAMPLE PREDICTION: Classify in period 1, test predictive power in period 2
2. RISK-ADJUSTED RETURNS: Control for position size and volatility exposure
3. EXECUTION QUALITY: Check if "informed" is just "fast/cheap execution"
4. PERSISTENCE: Do informed traders stay informed across time periods?
5. PRICE IMPACT ASYMMETRY: Informed trades should move prices; uninformed should not

Author: Claude
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
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
KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE']

print("="*80)
print("INFORMED CLASSIFICATION VALIDATION")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/6] Loading data...")
fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills = fills[fills['coin'].isin(KEY_ASSETS)]

# Parse dates
fills['date_str'] = fills['date'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
print(f"  Loaded {len(fills):,} fills")
print(f"  Dates: {sorted(fills['date_str'].unique())}")

# Separate takers
takers = fills[fills['crossed'] == True].copy()
print(f"  Takers: {len(takers):,} fills")

# =============================================================================
# COMPUTE PRICE CHANGES FOR DIFFERENT HORIZONS
# =============================================================================

print("\n[2/6] Computing price changes at multiple horizons...")

# Sort by time within each asset
takers = takers.sort_values(['coin', 'time']).reset_index(drop=True)

# Compute forward price changes at different horizons (1-fill, 5-fill, 10-fill ahead)
for horizon in [1, 5, 10]:
    takers[f'px_fwd_{horizon}'] = takers.groupby('coin')['px'].shift(-horizon)
    takers[f'ret_fwd_{horizon}'] = (takers[f'px_fwd_{horizon}'] - takers['px']) / takers['px'] * 10000  # bps

# Trade direction: +1 for buy, -1 for sell
takers['direction'] = np.where(takers['side'] == 'B', 1, -1)

# Profit at each horizon
for horizon in [1, 5, 10]:
    takers[f'profit_{horizon}'] = takers['direction'] * takers[f'ret_fwd_{horizon}']

print(f"  Computed forward returns at horizons: 1, 5, 10 fills")

# =============================================================================
# TEST 1: OUT-OF-SAMPLE PREDICTION
# =============================================================================

print("\n[3/6] TEST 1: Out-of-sample prediction...")
print("  Classifying in Day 1, testing predictive power in Day 2")

dates = sorted(fills['date_str'].unique())
if len(dates) >= 2:
    train_date = dates[0]  # July 28
    test_date = dates[1]   # July 29

    # Training period: classify wallets
    train_takers = takers[takers['date_str'] == train_date].copy()

    train_stats = train_takers.groupby('wallet').agg({
        'profit_5': ['mean', 'count', 'std'],
        'sz': ['sum', 'mean']
    }).reset_index()
    train_stats.columns = ['wallet', 'mean_profit', 'n_trades', 'profit_std', 'total_size', 'avg_size']
    train_stats = train_stats[train_stats['n_trades'] >= 5]

    # Classify by training period profit
    train_stats['quintile'] = pd.qcut(
        train_stats['mean_profit'].rank(method='first'), 5,
        labels=['Q1_Uninformed', 'Q2', 'Q3', 'Q4', 'Q5_Informed']
    )

    # Test period: evaluate
    test_takers = takers[takers['date_str'] == test_date].copy()
    test_takers = test_takers.merge(
        train_stats[['wallet', 'quintile', 'mean_profit']].rename(columns={'mean_profit': 'train_profit'}),
        on='wallet', how='inner'
    )

    # Compute test period profit by training classification
    test_profit_by_class = test_takers.groupby('quintile')['profit_5'].agg(['mean', 'count', 'std'])
    test_profit_by_class['se'] = test_profit_by_class['std'] / np.sqrt(test_profit_by_class['count'])
    test_profit_by_class['t_stat'] = test_profit_by_class['mean'] / test_profit_by_class['se']

    print(f"\n  Training period: {train_date}")
    print(f"  Test period: {test_date}")
    print(f"  Wallets classified in training: {len(train_stats):,}")
    print(f"  Wallets appearing in test: {test_takers['wallet'].nunique():,}")

    print(f"\n  {'Class':<20} {'Test Mean':>12} {'N':>10} {'t-stat':>10}")
    print("  " + "-"*55)
    for quintile in ['Q1_Uninformed', 'Q2', 'Q3', 'Q4', 'Q5_Informed']:
        if quintile in test_profit_by_class.index:
            row = test_profit_by_class.loc[quintile]
            print(f"  {quintile:<20} {row['mean']:>12.2f} {int(row['count']):>10} {row['t_stat']:>10.2f}")

    # Key test: Is Q5 significantly better than Q1 out-of-sample?
    q5_test = test_takers[test_takers['quintile'] == 'Q5_Informed']['profit_5'].dropna()
    q1_test = test_takers[test_takers['quintile'] == 'Q1_Uninformed']['profit_5'].dropna()

    if len(q5_test) > 10 and len(q1_test) > 10:
        t_stat_oos, p_val_oos = stats.ttest_ind(q5_test, q1_test)
        spread_oos = q5_test.mean() - q1_test.mean()
        print(f"\n  OUT-OF-SAMPLE Q5 vs Q1:")
        print(f"    Spread: {spread_oos:+.2f} bps")
        print(f"    t-stat: {t_stat_oos:.2f}")
        print(f"    p-value: {p_val_oos:.4f}")
        oos_valid = p_val_oos < 0.05 and spread_oos > 0
    else:
        oos_valid = False
        spread_oos = np.nan
        t_stat_oos = np.nan

# =============================================================================
# TEST 2: RISK-ADJUSTED CLASSIFICATION
# =============================================================================

print("\n[4/6] TEST 2: Controlling for risk/leverage...")

# Compute risk-adjusted profit (Sharpe-like)
wallet_stats = takers.groupby('wallet').agg({
    'profit_5': ['mean', 'std', 'count'],
    'sz': ['sum', 'mean', 'std'],
    'px': 'mean'
}).reset_index()
wallet_stats.columns = ['wallet', 'mean_profit', 'profit_std', 'n_trades',
                        'total_size', 'avg_size', 'size_std', 'avg_price']
wallet_stats = wallet_stats[wallet_stats['n_trades'] >= 10]

# Risk-adjusted profit (Sharpe ratio proxy)
wallet_stats['sharpe'] = wallet_stats['mean_profit'] / (wallet_stats['profit_std'] + 1e-6)

# Size-adjusted profit (profit per unit of size)
wallet_stats['profit_per_size'] = wallet_stats['mean_profit'] / (wallet_stats['avg_size'] + 1e-6)

# Correlation between raw profit and risk-adjusted measures
corr_sharpe = wallet_stats['mean_profit'].corr(wallet_stats['sharpe'])
corr_size_adj = wallet_stats['mean_profit'].corr(wallet_stats['profit_per_size'])

print(f"  Wallets with sufficient data: {len(wallet_stats):,}")
print(f"\n  Correlation of raw profit with:")
print(f"    Sharpe ratio:        {corr_sharpe:.3f}")
print(f"    Profit per size:     {corr_size_adj:.3f}")

# Classify by Sharpe instead of raw profit
wallet_stats['quintile_sharpe'] = pd.qcut(
    wallet_stats['sharpe'].rank(method='first'), 5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)
wallet_stats['quintile_raw'] = pd.qcut(
    wallet_stats['mean_profit'].rank(method='first'), 5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)

# Agreement between classifications
agreement = (wallet_stats['quintile_sharpe'] == wallet_stats['quintile_raw']).mean()
print(f"\n  Classification agreement (raw vs Sharpe): {100*agreement:.1f}%")

# Are high-profit traders just taking more risk (higher volatility)?
q5_raw = wallet_stats[wallet_stats['quintile_raw'] == 'Q5']
q1_raw = wallet_stats[wallet_stats['quintile_raw'] == 'Q1']

print(f"\n  {'Metric':<25} {'Q5 (Informed)':>15} {'Q1 (Uninformed)':>15}")
print("  " + "-"*60)
print(f"  {'Mean profit (bps)':<25} {q5_raw['mean_profit'].mean():>15.2f} {q1_raw['mean_profit'].mean():>15.2f}")
print(f"  {'Profit volatility':<25} {q5_raw['profit_std'].mean():>15.2f} {q1_raw['profit_std'].mean():>15.2f}")
print(f"  {'Avg trade size':<25} {q5_raw['avg_size'].mean():>15.2f} {q1_raw['avg_size'].mean():>15.2f}")
print(f"  {'Sharpe ratio':<25} {q5_raw['sharpe'].mean():>15.3f} {q1_raw['sharpe'].mean():>15.3f}")

# Key test: Do Q5 traders have higher Sharpe (not just higher variance)?
t_sharpe, p_sharpe = stats.ttest_ind(q5_raw['sharpe'], q1_raw['sharpe'])
print(f"\n  Sharpe ratio Q5 vs Q1: t = {t_sharpe:.2f}, p = {p_sharpe:.4f}")

# =============================================================================
# TEST 3: EXECUTION QUALITY / LATENCY CHECK
# =============================================================================

print("\n[5/6] TEST 3: Controlling for execution quality...")

# Execution quality proxies:
# - Price impact: How much does the price move on their trade?
# - Fill rate: How quickly do their orders fill?
# - Slippage: Difference from mid-price

# Compute immediate price impact (1-fill ahead price change)
wallet_stats['mean_impact'] = takers.groupby('wallet')['ret_fwd_1'].mean()

# Merge back
wallet_exec = wallet_stats[['wallet', 'quintile_raw', 'mean_profit', 'mean_impact']].dropna()

# Are "informed" traders just traders with low price impact (fast execution)?
q5_impact = wallet_exec[wallet_exec['quintile_raw'] == 'Q5']['mean_impact']
q1_impact = wallet_exec[wallet_exec['quintile_raw'] == 'Q1']['mean_impact']

print(f"\n  Price impact (1-fill ahead):")
print(f"    Q5 (Informed): {q5_impact.mean():+.2f} bps")
print(f"    Q1 (Uninformed): {q1_impact.mean():+.2f} bps")

# Actually, informed traders SHOULD have higher price impact (their trades move prices)
# Low price impact would suggest execution advantage, not information
t_impact, p_impact = stats.ttest_ind(q5_impact, q1_impact)
print(f"    t-stat: {t_impact:.2f}, p-value: {p_impact:.4f}")

if q5_impact.mean() > q1_impact.mean():
    print("    RESULT: Q5 trades have HIGHER price impact (consistent with information)")
else:
    print("    WARNING: Q5 trades have lower impact (suggests execution, not information)")

# =============================================================================
# TEST 4: PERSISTENCE / STABILITY
# =============================================================================

print("\n[6/6] TEST 4: Classification persistence...")

# Split into multiple periods and check stability
dates = sorted(takers['date_str'].unique())
if len(dates) >= 2:
    # Classify in each period
    period_classifications = {}

    for date in dates:
        period_data = takers[takers['date_str'] == date]
        period_stats = period_data.groupby('wallet')['profit_5'].agg(['mean', 'count'])
        period_stats = period_stats[period_stats['count'] >= 3]

        if len(period_stats) > 20:
            period_stats['quintile'] = pd.qcut(
                period_stats['mean'].rank(method='first'), 5,
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            )
            period_classifications[date] = period_stats['quintile'].to_dict()

    # Check persistence: wallets in Q5 in period 1, what quintile in period 2?
    if len(period_classifications) >= 2:
        date1, date2 = list(period_classifications.keys())[:2]
        class1 = period_classifications[date1]
        class2 = period_classifications[date2]

        # Wallets appearing in both periods
        common_wallets = set(class1.keys()) & set(class2.keys())

        if len(common_wallets) > 50:
            # Transition matrix
            transitions = []
            for wallet in common_wallets:
                transitions.append({
                    'wallet': wallet,
                    'period1': class1[wallet],
                    'period2': class2[wallet]
                })
            trans_df = pd.DataFrame(transitions)

            # Contingency table
            contingency = pd.crosstab(trans_df['period1'], trans_df['period2'])

            print(f"\n  Transition matrix ({date1} → {date2}):")
            print(f"  Wallets in both periods: {len(common_wallets)}")
            print(contingency.to_string())

            # Key metric: Do Q5 traders stay in top quintiles?
            q5_period1 = trans_df[trans_df['period1'] == 'Q5']
            if len(q5_period1) > 10:
                stay_top = (q5_period1['period2'].isin(['Q4', 'Q5'])).mean()
                print(f"\n  Q5 traders staying in top 40%: {100*stay_top:.1f}%")
                print(f"  (Random expectation: 40%)")

                # Chi-square test for independence
                chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
                print(f"  Chi-square test for independence: χ² = {chi2:.1f}, p = {p_chi:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"""
1. OUT-OF-SAMPLE PREDICTION
   Q5 vs Q1 spread in test period: {spread_oos:+.2f} bps (t = {t_stat_oos:.2f})
   Verdict: {"PASS - Classification predicts future profits" if oos_valid else "NEEDS REVIEW"}

2. RISK-ADJUSTED RETURNS
   Sharpe ratio Q5 vs Q1: t = {t_sharpe:.2f} (p = {p_sharpe:.4f})
   Agreement raw vs risk-adjusted: {100*agreement:.1f}%
   Verdict: {"PASS - Not just risk-taking" if p_sharpe < 0.05 and t_sharpe > 0 else "NEEDS REVIEW"}

3. EXECUTION QUALITY
   Q5 price impact: {q5_impact.mean():+.2f} bps vs Q1: {q1_impact.mean():+.2f} bps
   Verdict: {"PASS - Informed trades move prices (not just fast execution)" if q5_impact.mean() > q1_impact.mean() else "CAUTION - May be execution advantage"}

4. PERSISTENCE
   Q5 staying in top 40%: {100*stay_top:.1f}% (random = 40%)
   Verdict: {"PASS - Classification is stable" if stay_top > 0.45 else "NEEDS REVIEW - Low persistence"}

OVERALL: The "informed" classification is {"CREDIBLE" if oos_valid and p_sharpe < 0.05 else "NEEDS STRENGTHENING"}
""")

# Save results
results = {
    'out_of_sample': {
        'spread_bps': float(spread_oos) if not np.isnan(spread_oos) else None,
        't_stat': float(t_stat_oos) if not np.isnan(t_stat_oos) else None,
        'valid': bool(oos_valid)
    },
    'risk_adjusted': {
        'sharpe_t_stat': float(t_sharpe),
        'sharpe_p_value': float(p_sharpe),
        'classification_agreement': float(agreement)
    },
    'execution': {
        'q5_impact_bps': float(q5_impact.mean()),
        'q1_impact_bps': float(q1_impact.mean()),
        'impact_t_stat': float(t_impact)
    },
    'persistence': {
        'q5_stay_top40_pct': float(stay_top) if 'stay_top' in dir() else None
    }
}

import json
with open(OUTPUT_DIR / 'informed_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: informed_validation_results.json")
