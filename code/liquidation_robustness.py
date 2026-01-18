#!/usr/bin/env python3
"""
Liquidation/Forced-Flow Robustness for Toxicity Classification
================================================================

Perpetual futures have forced trades (liquidations) that could contaminate
the informed/uninformed classification. This script tests robustness by
excluding trades likely to be liquidations.

Liquidation proxies:
1. Extreme volatility hours (top 10% by hourly price range)
2. Extreme trade sizes (top 1% by size within each asset)
3. Both exclusions combined

If the toxicity differential is driven by liquidations, it should weaken
substantially when excluding these trades.

Author: Boyi Shen, London Business School
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'

print("=" * 80)
print("LIQUIDATION/FORCED-FLOW ROBUSTNESS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/6] Loading data...")

fills = pd.read_parquet(DATA_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['hour'] = fills['timestamp'].dt.floor('H')
fills['date_int'] = fills['date'].astype(int)

# Focus on key assets
KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE']
fills = fills[fills['coin'].isin(KEY_ASSETS)]

print(f"  Loaded {len(fills):,} fills across {len(KEY_ASSETS)} key assets")

# =============================================================================
# COMPUTE VOLATILITY MEASURES
# =============================================================================

print("\n[2/6] Computing volatility measures...")

# Compute hourly volatility (price range / mid)
hourly_vol = fills.groupby(['coin', 'hour']).agg({
    'px': ['min', 'max', 'mean']
}).reset_index()
hourly_vol.columns = ['coin', 'hour', 'px_min', 'px_max', 'px_mean']
hourly_vol['hourly_range_pct'] = (hourly_vol['px_max'] - hourly_vol['px_min']) / hourly_vol['px_mean'] * 100

# Flag high-volatility hours (top 10%)
vol_threshold = hourly_vol['hourly_range_pct'].quantile(0.90)
hourly_vol['high_vol'] = hourly_vol['hourly_range_pct'] > vol_threshold

print(f"  Volatility threshold (90th pct): {vol_threshold:.2f}%")
print(f"  High-vol hours: {hourly_vol['high_vol'].sum()} / {len(hourly_vol)}")

# Merge back to fills
fills = fills.merge(hourly_vol[['coin', 'hour', 'high_vol']], on=['coin', 'hour'], how='left')
fills['high_vol'] = fills['high_vol'].fillna(False)

# =============================================================================
# COMPUTE SIZE EXTREMES
# =============================================================================

print("\n[3/6] Identifying extreme trade sizes...")

# Compute size percentiles by asset
size_thresh = fills.groupby('coin')['sz'].quantile(0.99).reset_index()
size_thresh.columns = ['coin', 'size_99pct']

fills = fills.merge(size_thresh, on='coin', how='left')
fills['extreme_size'] = fills['sz'] > fills['size_99pct']

print(f"  Extreme size trades (>99th pct): {fills['extreme_size'].sum():,} / {len(fills):,}")

# =============================================================================
# LINK MAKER-TAKER PAIRS
# =============================================================================

print("\n[4/6] Linking maker-taker pairs...")

takers = fills[fills['crossed'] == True].copy()
makers = fills[fills['crossed'] == False].copy()

pairs = takers.merge(
    makers[['time', 'coin', 'px', 'sz', 'wallet']].rename(columns={'wallet': 'maker_wallet'}),
    on=['time', 'coin', 'px', 'sz'],
    how='inner'
)
pairs = pairs[pairs['wallet'] != pairs['maker_wallet']]

print(f"  Linked {len(pairs):,} maker-taker pairs")

# =============================================================================
# COMPUTE MARKOUTS
# =============================================================================

print("\n[5/6] Computing 1-minute markouts...")

HORIZON_MS = 60000
pairs['direction'] = np.where(pairs['side'] == 'B', 1, -1)

for coin in pairs['coin'].unique():
    mask = pairs['coin'] == coin
    coin_prices = fills[fills['coin'] == coin][['time', 'px']].drop_duplicates().sort_values('time')
    times = coin_prices['time'].values
    prices = coin_prices['px'].values

    pair_times = pairs.loc[mask, 'time'].values
    pair_prices = pairs.loc[mask, 'px'].values

    markouts = []
    for t, p in zip(pair_times, pair_prices):
        future_idx = np.searchsorted(times, t + HORIZON_MS)
        if future_idx < len(times):
            future_price = prices[future_idx]
            markouts.append((future_price - p) / p * 10000)
        else:
            markouts.append(np.nan)

    pairs.loc[mask, 'raw_markout'] = markouts

pairs['taker_profit'] = pairs['direction'] * pairs['raw_markout']
pairs['maker_profit'] = -pairs['taker_profit']
pairs = pairs.dropna(subset=['maker_profit'])
pairs = pairs[np.abs(pairs['maker_profit']) < 1000]

print(f"  Valid pairs: {len(pairs):,}")

# =============================================================================
# CLASSIFICATION (Same as main analysis)
# =============================================================================

TRAIN_DATE = 20250728
TEST_DATES = [20250729, 20250730]

train_pairs = pairs[pairs['date_int'] == TRAIN_DATE]
test_pairs = pairs[pairs['date_int'].isin(TEST_DATES)]

taker_train_stats = train_pairs.groupby('wallet')['taker_profit'].agg(['mean', 'count']).reset_index()
taker_train_stats.columns = ['wallet', 'mean_profit', 'n_trades']

MIN_TRADES = 5
taker_train_stats = taker_train_stats[taker_train_stats['n_trades'] >= MIN_TRADES]

taker_train_stats['quintile'] = pd.qcut(
    taker_train_stats['mean_profit'].rank(method='first'), 5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)

# Merge labels
test_labeled = test_pairs.merge(
    taker_train_stats[['wallet', 'quintile']],
    on='wallet',
    how='inner'
)
test_q1q5 = test_labeled[test_labeled['quintile'].isin(['Q1', 'Q5'])].copy()

# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

print("\n[6/6] Running robustness tests...")

def compute_toxicity_differential(df, label):
    """Compute taker-level toxicity differential."""
    taker_agg = df.groupby(['wallet', 'quintile']).agg({
        'maker_profit': 'mean',
        'time': 'count'
    }).reset_index()
    taker_agg.columns = ['wallet', 'quintile', 'mean_profit', 'n_trades']
    taker_agg = taker_agg.dropna()

    q1_profits = taker_agg[taker_agg['quintile'] == 'Q1']['mean_profit'].values
    q5_profits = taker_agg[taker_agg['quintile'] == 'Q5']['mean_profit'].values

    if len(q1_profits) < 10 or len(q5_profits) < 10:
        return None

    diff = np.mean(q1_profits) - np.mean(q5_profits)
    t_stat, p_value = stats.ttest_ind(q1_profits, q5_profits)

    # Trade-weighted regression
    taker_agg = taker_agg[taker_agg['quintile'].isin(['Q1', 'Q5'])]
    y = taker_agg['mean_profit'].values
    is_uninformed = (taker_agg['quintile'] == 'Q1').astype(float).values
    X = sm.add_constant(is_uninformed)
    weights = taker_agg['n_trades'].values

    try:
        wls = sm.WLS(y, X, weights=weights).fit()
        wls_coef = float(wls.params[1])
        wls_t = float(wls.tvalues[1])
        wls_p = float(wls.pvalues[1])
    except:
        wls_coef = diff
        wls_t = t_stat
        wls_p = p_value

    return {
        'label': label,
        'n_trades': len(df),
        'n_takers_q1': len(q1_profits),
        'n_takers_q5': len(q5_profits),
        'taker_level': {
            'differential_bps': float(diff),
            't_stat': float(t_stat),
            'p_value': float(p_value)
        },
        'trade_weighted': {
            'differential_bps': wls_coef,
            't_stat': wls_t,
            'p_value': wls_p
        }
    }

results = {}

# Baseline
results['baseline'] = compute_toxicity_differential(test_q1q5, 'Baseline (all trades)')
print(f"\n  Baseline:")
print(f"    Trades: {results['baseline']['n_trades']:,}")
print(f"    Taker-level: {results['baseline']['taker_level']['differential_bps']:.2f} bps (t={results['baseline']['taker_level']['t_stat']:.2f})")
print(f"    Trade-weighted: {results['baseline']['trade_weighted']['differential_bps']:.2f} bps (t={results['baseline']['trade_weighted']['t_stat']:.2f})")

# Exclude high-volatility hours
test_no_vol = test_q1q5[~test_q1q5['high_vol']]
results['excl_high_vol'] = compute_toxicity_differential(test_no_vol, 'Exclude high-vol hours')
print(f"\n  Exclude high-volatility hours (top 10%):")
print(f"    Trades: {results['excl_high_vol']['n_trades']:,}")
print(f"    Taker-level: {results['excl_high_vol']['taker_level']['differential_bps']:.2f} bps (t={results['excl_high_vol']['taker_level']['t_stat']:.2f})")
print(f"    Trade-weighted: {results['excl_high_vol']['trade_weighted']['differential_bps']:.2f} bps (t={results['excl_high_vol']['trade_weighted']['t_stat']:.2f})")

# Exclude extreme sizes
test_no_extreme = test_q1q5[~test_q1q5['extreme_size']]
results['excl_extreme_size'] = compute_toxicity_differential(test_no_extreme, 'Exclude extreme sizes (>99th pct)')
print(f"\n  Exclude extreme trade sizes (>99th percentile):")
print(f"    Trades: {results['excl_extreme_size']['n_trades']:,}")
print(f"    Taker-level: {results['excl_extreme_size']['taker_level']['differential_bps']:.2f} bps (t={results['excl_extreme_size']['taker_level']['t_stat']:.2f})")
print(f"    Trade-weighted: {results['excl_extreme_size']['trade_weighted']['differential_bps']:.2f} bps (t={results['excl_extreme_size']['trade_weighted']['t_stat']:.2f})")

# Exclude both
test_conservative = test_q1q5[~test_q1q5['high_vol'] & ~test_q1q5['extreme_size']]
results['excl_both'] = compute_toxicity_differential(test_conservative, 'Exclude both')
print(f"\n  Exclude both high-vol AND extreme sizes:")
print(f"    Trades: {results['excl_both']['n_trades']:,}")
print(f"    Taker-level: {results['excl_both']['taker_level']['differential_bps']:.2f} bps (t={results['excl_both']['taker_level']['t_stat']:.2f})")
print(f"    Trade-weighted: {results['excl_both']['trade_weighted']['differential_bps']:.2f} bps (t={results['excl_both']['trade_weighted']['t_stat']:.2f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output = {
    'methodology': {
        'description': 'Liquidation/forced-flow robustness for toxicity classification',
        'high_vol_threshold': 'Top 10% hourly price range',
        'extreme_size_threshold': 'Top 1% trade size by asset',
        'classification': 'Out-of-sample (train July 28, test July 29-30)',
        'assets': KEY_ASSETS
    },
    'results': results,
    'summary': {
        'baseline_trade_weighted_bps': results['baseline']['trade_weighted']['differential_bps'],
        'baseline_trade_weighted_t': results['baseline']['trade_weighted']['t_stat'],
        'conservative_trade_weighted_bps': results['excl_both']['trade_weighted']['differential_bps'],
        'conservative_trade_weighted_t': results['excl_both']['trade_weighted']['t_stat'],
        'pct_of_baseline': results['excl_both']['trade_weighted']['differential_bps'] / results['baseline']['trade_weighted']['differential_bps'] * 100,
        'conclusion': 'robust' if results['excl_both']['trade_weighted']['t_stat'] > 2 else 'weakened'
    }
}

with open(RESULTS_DIR / 'liquidation_robustness_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved to: {RESULTS_DIR / 'liquidation_robustness_results.json'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: LIQUIDATION/FORCED-FLOW ROBUSTNESS")
print("=" * 80)
print(f"""
Robustness Test: Exclude Likely Liquidations
---------------------------------------------
Perpetual futures have forced trades (liquidations) that could contaminate
the informed/uninformed classification. We test robustness by excluding:
  1. High-volatility hours (top 10% by hourly price range)
  2. Extreme trade sizes (top 1% by asset)
  3. Both exclusions combined

Results (Trade-Weighted):
  Baseline:              {results['baseline']['trade_weighted']['differential_bps']:.2f} bps (t = {results['baseline']['trade_weighted']['t_stat']:.2f})
  Excl. high-vol hours:  {results['excl_high_vol']['trade_weighted']['differential_bps']:.2f} bps (t = {results['excl_high_vol']['trade_weighted']['t_stat']:.2f})
  Excl. extreme sizes:   {results['excl_extreme_size']['trade_weighted']['differential_bps']:.2f} bps (t = {results['excl_extreme_size']['trade_weighted']['t_stat']:.2f})
  Excl. both:            {results['excl_both']['trade_weighted']['differential_bps']:.2f} bps (t = {results['excl_both']['trade_weighted']['t_stat']:.2f})

Conservative estimate is {output['summary']['pct_of_baseline']:.0f}% of baseline.
Conclusion: The toxicity differential is {output['summary']['conclusion'].upper()}.
""")
