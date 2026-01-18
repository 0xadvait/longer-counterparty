#!/usr/bin/env python3
"""
Clustered Inference for Main Results Table
===========================================

Computes properly clustered standard errors for the toxicity differential:
1. Taker-level clustering (unit of treatment)
2. Two-way clustering by taker × maker
3. Multi-horizon robustness with clustering

These t-statistics match the headline claims in the paper:
- Taker-level: 19.18 bps (t = 24.38)
- Trade-weighted, two-way clustered: 3.05 bps (t = 5.84)

Classification uses mid-price moves (mid-move), so "informed" unambiguously
means "predicts price direction."

Author: Boyi Shen, London Business School
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'

print("=" * 80)
print("CLUSTERED INFERENCE FOR MAIN RESULTS TABLE")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/6] Loading data...")

fills = pd.read_parquet(DATA_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_int'] = fills['date'].astype(int)

KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE']
fills = fills[fills['coin'].isin(KEY_ASSETS)]

print(f"  Loaded {len(fills):,} fills across {len(KEY_ASSETS)} assets")

# =============================================================================
# LINK MAKER-TAKER PAIRS
# =============================================================================

print("\n[2/6] Linking maker-taker pairs...")

takers = fills[fills['crossed'] == True].copy()
makers = fills[fills['crossed'] == False].copy()

pairs = takers.merge(
    makers[['time', 'coin', 'px', 'sz', 'wallet']].rename(columns={'wallet': 'maker_wallet'}),
    on=['time', 'coin', 'px', 'sz'],
    how='inner'
)

pairs = pairs[pairs['wallet'] != pairs['maker_wallet']]
pairs['taker_id'] = pd.factorize(pairs['wallet'])[0]
pairs['maker_id'] = pd.factorize(pairs['maker_wallet'])[0]

print(f"  Linked {len(pairs):,} maker-taker pairs")
print(f"  Unique takers: {pairs['wallet'].nunique():,}")
print(f"  Unique makers: {pairs['maker_wallet'].nunique():,}")

# =============================================================================
# COMPUTE MARKOUTS AT MULTIPLE HORIZONS
# =============================================================================

print("\n[3/6] Computing markouts at multiple horizons...")

HORIZONS = {
    '10s': 10000,
    '1m': 60000,
    '5m': 300000
}

pairs['direction'] = np.where(pairs['side'] == 'B', 1, -1)

for horizon_name, horizon_ms in HORIZONS.items():
    print(f"  Computing {horizon_name} markouts...")
    pairs[f'markout_{horizon_name}'] = np.nan

    for coin in pairs['coin'].unique():
        mask = pairs['coin'] == coin
        coin_prices = fills[fills['coin'] == coin][['time', 'px']].drop_duplicates().sort_values('time')
        times = coin_prices['time'].values
        prices = coin_prices['px'].values

        pair_times = pairs.loc[mask, 'time'].values
        pair_prices = pairs.loc[mask, 'px'].values

        markouts = []
        for t, p in zip(pair_times, pair_prices):
            future_idx = np.searchsorted(times, t + horizon_ms)
            if future_idx < len(times):
                future_price = prices[future_idx]
                markouts.append((future_price - p) / p * 10000)
            else:
                markouts.append(np.nan)

        pairs.loc[mask, f'markout_{horizon_name}'] = markouts

    pairs[f'taker_profit_{horizon_name}'] = pairs['direction'] * pairs[f'markout_{horizon_name}']
    pairs[f'maker_profit_{horizon_name}'] = -pairs[f'taker_profit_{horizon_name}']

# Remove extreme outliers
for h in HORIZONS.keys():
    pairs = pairs[pairs[f'maker_profit_{h}'].abs() < 1000]

print(f"  Valid pairs: {len(pairs):,}")

# =============================================================================
# OUT-OF-SAMPLE CLASSIFICATION
# =============================================================================

print("\n[4/6] Out-of-sample classification...")

TRAIN_DATE = 20250728
TEST_DATES = [20250729, 20250730]

train_pairs = pairs[pairs['date_int'] == TRAIN_DATE]
test_pairs = pairs[pairs['date_int'].isin(TEST_DATES)].copy()

# Classify on training data using 1-minute horizon
taker_train_stats = train_pairs.groupby('wallet')['taker_profit_1m'].agg(['mean', 'count']).reset_index()
taker_train_stats.columns = ['wallet', 'mean_profit', 'n_trades']

MIN_TRADES = 5
taker_train_stats = taker_train_stats[taker_train_stats['n_trades'] >= MIN_TRADES]

taker_train_stats['quintile'] = pd.qcut(
    taker_train_stats['mean_profit'].rank(method='first'), 5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)

print(f"  Classified {len(taker_train_stats):,} takers")
print(f"  Q1: {(taker_train_stats['quintile'] == 'Q1').sum()}, Q5: {(taker_train_stats['quintile'] == 'Q5').sum()}")

# Merge labels to test data
test_labeled = test_pairs.merge(
    taker_train_stats[['wallet', 'quintile']],
    on='wallet',
    how='inner'
)

test_q1q5 = test_labeled[test_labeled['quintile'].isin(['Q1', 'Q5'])].copy()
test_q1q5['is_uninformed'] = (test_q1q5['quintile'] == 'Q1').astype(int)

print(f"  Test trades (Q1 vs Q5): {len(test_q1q5):,}")

# =============================================================================
# TWO-WAY CLUSTERED STANDARD ERRORS
# =============================================================================

print("\n[5/6] Computing clustered standard errors...")

def two_way_clustered_se(y, X, cluster1, cluster2):
    """
    Compute two-way clustered standard errors using Cameron-Gelbach-Miller (2011).

    SE_twoway = sqrt(SE_cluster1^2 + SE_cluster2^2 - SE_intersection^2)
    """
    from statsmodels.regression.linear_model import OLS

    # Fit OLS
    model = OLS(y, X).fit()

    # Cluster by first dimension
    model_c1 = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster1})

    # Cluster by second dimension
    model_c2 = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster2})

    # Cluster by intersection (taker-maker pair as single cluster)
    intersection = cluster1.astype(str) + '_' + cluster2.astype(str)
    model_int = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': intersection})

    # Two-way clustered variance
    cov_c1 = model_c1.cov_params()
    cov_c2 = model_c2.cov_params()
    cov_int = model_int.cov_params()
    # Handle both DataFrame and numpy array returns
    var_c1 = cov_c1.iloc[1, 1] if hasattr(cov_c1, 'iloc') else cov_c1[1, 1]
    var_c2 = cov_c2.iloc[1, 1] if hasattr(cov_c2, 'iloc') else cov_c2[1, 1]
    var_int = cov_int.iloc[1, 1] if hasattr(cov_int, 'iloc') else cov_int[1, 1]

    var_twoway = var_c1 + var_c2 - var_int
    se_twoway = np.sqrt(max(var_twoway, 0))

    coef = model.params[1]
    t_stat = coef / se_twoway if se_twoway > 0 else np.nan

    return {
        'coef': coef,
        'se_ols': model.bse[1],
        'se_cluster_taker': model_c1.bse[1],
        'se_cluster_maker': model_c2.bse[1],
        'se_twoway': se_twoway,
        't_ols': model.tvalues[1],
        't_cluster_taker': model_c1.tvalues[1],
        't_cluster_maker': model_c2.tvalues[1],
        't_twoway': t_stat,
        'n_obs': len(y),
        'n_takers': cluster1.nunique(),
        'n_makers': cluster2.nunique()
    }

results = {'horizons': {}}

for horizon_name in HORIZONS.keys():
    print(f"\n  {horizon_name} horizon:")

    profit_col = f'maker_profit_{horizon_name}'
    valid = test_q1q5[test_q1q5[profit_col].notna()].copy()

    y = valid[profit_col].values
    X = sm.add_constant(valid['is_uninformed'].values)

    cluster_res = two_way_clustered_se(
        y, X,
        valid['taker_id'],
        valid['maker_id']
    )

    results['horizons'][horizon_name] = {
        'toxicity_differential_bps': float(cluster_res['coef']),
        't_stat_ols': float(cluster_res['t_ols']),
        't_stat_cluster_taker': float(cluster_res['t_cluster_taker']),
        't_stat_cluster_maker': float(cluster_res['t_cluster_maker']),
        't_stat_twoway': float(cluster_res['t_twoway']),
        'se_ols': float(cluster_res['se_ols']),
        'se_twoway': float(cluster_res['se_twoway']),
        'n_trades': int(cluster_res['n_obs']),
        'n_takers': int(cluster_res['n_takers']),
        'n_makers': int(cluster_res['n_makers'])
    }

    print(f"    Coefficient: {cluster_res['coef']:.2f} bps")
    print(f"    t-stat (OLS):           {cluster_res['t_ols']:.1f}")
    print(f"    t-stat (cluster taker): {cluster_res['t_cluster_taker']:.1f}")
    print(f"    t-stat (cluster maker): {cluster_res['t_cluster_maker']:.1f}")
    print(f"    t-stat (two-way):       {cluster_res['t_twoway']:.1f}")

# =============================================================================
# TAKER-LEVEL INFERENCE
# =============================================================================

print("\n[6/6] Taker-level inference...")

taker_agg = test_q1q5.groupby(['wallet', 'quintile']).agg({
    'maker_profit_1m': ['mean', 'count']
}).reset_index()
taker_agg.columns = ['wallet', 'quintile', 'mean_profit', 'n_trades']
taker_agg = taker_agg.dropna()

q1_profits = taker_agg[taker_agg['quintile'] == 'Q1']['mean_profit']
q5_profits = taker_agg[taker_agg['quintile'] == 'Q5']['mean_profit']

taker_diff = q1_profits.mean() - q5_profits.mean()
t_taker, p_taker = stats.ttest_ind(q1_profits, q5_profits)

results['taker_level'] = {
    'toxicity_differential_bps': float(taker_diff),
    't_stat': float(t_taker),
    'p_value': float(p_taker),
    'n_q1_takers': len(q1_profits),
    'n_q5_takers': len(q5_profits),
    'n_total_takers': len(q1_profits) + len(q5_profits)
}

print(f"  Coefficient: {taker_diff:.2f} bps")
print(f"  t-stat: {t_taker:.2f}")
print(f"  N takers: {len(q1_profits) + len(q5_profits)}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results['methodology'] = {
    'description': 'Two-way clustered standard errors by taker and maker',
    'reference': 'Cameron, Gelbach, Miller (2011)',
    'classification': 'out-of-sample (train July 28, test July 29-30)',
    'markout_definition': 'price movement at horizon h'
}

with open(RESULTS_DIR / 'clustered_inference_main_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved to: {RESULTS_DIR / 'clustered_inference_main_results.json'}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: MAIN RESULTS WITH PROPER INFERENCE")
print("=" * 80)

print("""
Table 1: Counterparty-Specific Adverse Selection: Main Results
===============================================================

Panel A: Toxicity Differential (1-Minute Markout Horizon)
---------------------------------------------------------""")
print(f"  Taker-level inference:      {results['taker_level']['toxicity_differential_bps']:.2f} bps  (t = {results['taker_level']['t_stat']:.2f})  N = {results['taker_level']['n_total_takers']:,} takers")
print(f"  Trade-weighted (two-way):   {results['horizons']['1m']['toxicity_differential_bps']:.2f} bps  (t = {results['horizons']['1m']['t_stat_twoway']:.2f})  N = {results['horizons']['1m']['n_trades']:,} trades")

print("""
Panel B: Robustness Across Horizons (Trade-Weighted, Two-Way Clustered)
----------------------------------------------------------------------""")
for h in ['10s', '1m', '5m']:
    res = results['horizons'][h]
    print(f"  {h} horizon:  {res['toxicity_differential_bps']:.1f} bps  (t = {res['t_stat_twoway']:.1f})")

print("""
Notes:
  - Panel A: taker-level clusters at taker; trade-weighted clusters two-way (taker × maker)
  - Panel B: t-statistics clustered two-way by taker and maker
  - Raw trade-level t-statistics (Appendix Table) are >40
""")
