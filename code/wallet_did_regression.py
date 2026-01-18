#!/usr/bin/env python3
"""
Wallet-Level Difference-in-Differences Analysis
================================================

Tests whether informed wallets are more likely to remain active during
infrastructure stress (the API outage on July 29, 2025).

CORRECT ECONOMETRIC SPECIFICATION:
----------------------------------
The cleanest DiD design compares:
- Treatment group: Q5 (informed) wallets
- Control group: Q1 (uninformed) wallets
- Pre-period: Hour 14:00 UTC on July 28 (control day)
- Post-period: Hour 14:00 UTC on July 29 (outage day)

This controls for:
1. Hour-of-day effects (comparing same hour across days)
2. Wallet-level heterogeneity (using DiD)
3. Out-of-sample classification (train July 28 morning, test July 28-29 afternoon)

Specification:
    Active_it = β₀ + β₁(Informed_i) + β₂(OutageDay_t) + β₃(Informed_i × OutageDay_t) + ε_it

where:
    - i = wallet
    - t ∈ {July 28 14:00, July 29 14:00}
    - Active_it = 1 if wallet has any fills in that hour
    - Informed_i = 1 if wallet is Q5, 0 if Q1
    - OutageDay_t = 1 if July 29, 0 if July 28
    - β₃ is the DiD coefficient

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
print("WALLET-LEVEL DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("Correct Specification: Same-Hour Comparison (July 28 vs July 29)")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/5] Loading data...")

fills = pd.read_parquet(DATA_DIR / 'wallet_fills_data.parquet')
fills['time_dt'] = pd.to_datetime(fills['time'], unit='ms')
fills['hour'] = fills['time_dt'].dt.floor('H')
fills['date_int'] = fills['date'].astype(int)

print(f"  Loaded {len(fills):,} fills")

# =============================================================================
# CLASSIFY WALLETS (OUT-OF-SAMPLE)
# =============================================================================

print("\n[2/5] Classifying wallets (out-of-sample using morning of July 28)...")

# Use morning of July 28 for classification (before 14:00 UTC)
# This ensures classification is truly out-of-sample for both test periods
train_morning = fills[
    (fills['date_int'] == 20250728) &
    (fills['time_dt'].dt.hour < 14)
].copy()

# Compute 1-minute markouts for classification
KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE']
train_morning = train_morning[train_morning['coin'].isin(KEY_ASSETS)]

# Get takers
takers_train = train_morning[train_morning['crossed'] == True].copy()
takers_train['direction'] = np.where(takers_train['side'] == 'B', 1, -1)

# Compute simple markouts using price changes
HORIZON_MS = 60000  # 1 minute
takers_train = takers_train.sort_values(['coin', 'time'])

# For each trade, find price 1 minute later
profits = []
for coin in KEY_ASSETS:
    coin_data = train_morning[train_morning['coin'] == coin].sort_values('time')
    times = coin_data['time'].values
    prices = coin_data['px'].values

    coin_takers = takers_train[takers_train['coin'] == coin]
    for _, row in coin_takers.iterrows():
        t = row['time']
        p = row['px']
        direction = row['direction']

        future_idx = np.searchsorted(times, t + HORIZON_MS)
        if future_idx < len(times):
            future_price = prices[future_idx]
            markout = (future_price - p) / p * 10000
            profit = direction * markout
            profits.append({'wallet': row['wallet'], 'profit_bps': profit})

profits_df = pd.DataFrame(profits)

# Classify by mean profit
wallet_stats = profits_df.groupby('wallet').agg(
    mean_profit=('profit_bps', 'mean'),
    n_trades=('profit_bps', 'count')
).reset_index()

# Require minimum 5 trades for classification
MIN_TRADES = 5
wallet_stats = wallet_stats[wallet_stats['n_trades'] >= MIN_TRADES]

# Classify into quintiles
wallet_stats['quintile'] = pd.qcut(
    wallet_stats['mean_profit'].rank(method='first'), 5,
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)

q5_wallets = set(wallet_stats[wallet_stats['quintile'] == 'Q5']['wallet'])
q1_wallets = set(wallet_stats[wallet_stats['quintile'] == 'Q1']['wallet'])

print(f"  Q5 (informed): {len(q5_wallets):,}")
print(f"  Q1 (uninformed): {len(q1_wallets):,}")

# =============================================================================
# BUILD DiD DATASET - SAME HOUR COMPARISON
# =============================================================================

print("\n[3/5] Building DiD dataset (same-hour comparison)...")

# The outage occurred at 14:00 UTC on July 29
# Control period: 14:00 UTC on July 28
control_hour = pd.Timestamp('2025-07-28 14:00:00')
outage_hour = pd.Timestamp('2025-07-29 14:00:00')

# Get fills for these two specific hours
fills_control = fills[fills['hour'] == control_hour].copy()
fills_outage = fills[fills['hour'] == outage_hour].copy()

# Filter to classified wallets (Q1 and Q5 only)
classified_wallets = q5_wallets | q1_wallets
fills_control = fills_control[fills_control['wallet'].isin(classified_wallets)]
fills_outage = fills_outage[fills_outage['wallet'].isin(classified_wallets)]

print(f"  Control hour (July 28, 14:00): {len(fills_control):,} fills")
print(f"  Outage hour (July 29, 14:00): {len(fills_outage):,} fills")

# Build balanced wallet-hour panel
all_wallets = list(classified_wallets)

# Create panel: each wallet observed in both hours
panel = pd.DataFrame([
    {'wallet': w, 'hour': control_hour, 'is_outage_day': 0}
    for w in all_wallets
] + [
    {'wallet': w, 'hour': outage_hour, 'is_outage_day': 1}
    for w in all_wallets
])

# Count fills per wallet-hour
fills_per_wh_control = fills_control.groupby('wallet').size().reset_index(name='n_fills')
fills_per_wh_outage = fills_outage.groupby('wallet').size().reset_index(name='n_fills')

# Merge
panel = panel.merge(
    pd.concat([
        fills_per_wh_control.assign(hour=control_hour),
        fills_per_wh_outage.assign(hour=outage_hour)
    ]),
    on=['wallet', 'hour'],
    how='left'
)
panel['n_fills'] = panel['n_fills'].fillna(0)
panel['active'] = (panel['n_fills'] > 0).astype(int)

# Add wallet characteristics
panel['is_informed'] = panel['wallet'].isin(q5_wallets).astype(int)
panel['interaction'] = panel['is_informed'] * panel['is_outage_day']

print(f"  Panel observations: {len(panel):,}")
print(f"  Wallets: {panel['wallet'].nunique():,}")

# =============================================================================
# DiD REGRESSION
# =============================================================================

print("\n[4/5] Running DiD regression...")

# Compute group means
informed_control = panel[(panel['is_informed'] == 1) & (panel['is_outage_day'] == 0)]['active'].mean()
informed_outage = panel[(panel['is_informed'] == 1) & (panel['is_outage_day'] == 1)]['active'].mean()
uninformed_control = panel[(panel['is_informed'] == 0) & (panel['is_outage_day'] == 0)]['active'].mean()
uninformed_outage = panel[(panel['is_informed'] == 0) & (panel['is_outage_day'] == 1)]['active'].mean()

# Manual DiD calculation
did_manual = (informed_outage - informed_control) - (uninformed_outage - uninformed_control)

print(f"\n  Group Activity Rates:")
print(f"    Q5 (informed):")
print(f"      Control day (July 28): {informed_control:.3f} ({informed_control*100:.1f}%)")
print(f"      Outage day (July 29):  {informed_outage:.3f} ({informed_outage*100:.1f}%)")
print(f"      Change: {(informed_outage - informed_control)*100:+.2f} pp")
print(f"    Q1 (uninformed):")
print(f"      Control day (July 28): {uninformed_control:.3f} ({uninformed_control*100:.1f}%)")
print(f"      Outage day (July 29):  {uninformed_outage:.3f} ({uninformed_outage*100:.1f}%)")
print(f"      Change: {(uninformed_outage - uninformed_control)*100:+.2f} pp")
print(f"\n  DiD (differential change): {did_manual*100:+.2f} pp")

# OLS regression: Active = β₀ + β₁(Informed) + β₂(OutageDay) + β₃(Informed × OutageDay) + ε
y = panel['active'].values
X = sm.add_constant(panel[['is_informed', 'is_outage_day', 'interaction']].values)

# OLS with wallet-clustered standard errors
model = sm.OLS(y, X)
results_clustered = model.fit(cov_type='cluster', cov_kwds={'groups': panel['wallet'].values})

# Extract DiD coefficient
did_coef = results_clustered.params[3]
did_se = results_clustered.bse[3]
did_t = results_clustered.tvalues[3]
did_p = results_clustered.pvalues[3]

print(f"\n  OLS Regression Results (wallet-clustered SEs):")
print(f"    Constant (β₀):                {results_clustered.params[0]:.4f} (t = {results_clustered.tvalues[0]:.2f})")
print(f"    Informed (β₁):                {results_clustered.params[1]:.4f} (t = {results_clustered.tvalues[1]:.2f})")
print(f"    Outage Day (β₂):              {results_clustered.params[2]:.4f} (t = {results_clustered.tvalues[2]:.2f})")
print(f"    Informed × Outage Day (β₃):   {results_clustered.params[3]:.4f} (t = {results_clustered.tvalues[3]:.2f})")

print(f"\n  *** DiD COEFFICIENT (β₃): {did_coef*100:.2f} pp, t = {did_t:.2f} ***")

# =============================================================================
# ROBUSTNESS: POOLED SAME-HOUR COMPARISON (MULTIPLE HOURS)
# =============================================================================

print("\n[4b/5] Robustness: Pooled same-hour comparison (hours 12-16 UTC)...")

# Compare the same hours on both days to increase power
hours_to_compare = list(range(12, 17))  # 12:00 to 16:00 UTC

panel_pooled = []
for h in hours_to_compare:
    control_h = pd.Timestamp(f'2025-07-28 {h:02d}:00:00')
    outage_h = pd.Timestamp(f'2025-07-29 {h:02d}:00:00')

    for w in all_wallets:
        panel_pooled.append({'wallet': w, 'hour': control_h, 'is_outage_day': 0, 'hour_of_day': h})
        panel_pooled.append({'wallet': w, 'hour': outage_h, 'is_outage_day': 1, 'hour_of_day': h})

panel_pooled = pd.DataFrame(panel_pooled)

# Count fills
fills_relevant = fills[
    (fills['hour'].isin([pd.Timestamp(f'2025-07-28 {h:02d}:00:00') for h in hours_to_compare] +
                        [pd.Timestamp(f'2025-07-29 {h:02d}:00:00') for h in hours_to_compare])) &
    (fills['wallet'].isin(classified_wallets))
]
fills_per_wh = fills_relevant.groupby(['wallet', 'hour']).size().reset_index(name='n_fills')

panel_pooled = panel_pooled.merge(fills_per_wh, on=['wallet', 'hour'], how='left')
panel_pooled['n_fills'] = panel_pooled['n_fills'].fillna(0)
panel_pooled['active'] = (panel_pooled['n_fills'] > 0).astype(int)
panel_pooled['is_informed'] = panel_pooled['wallet'].isin(q5_wallets).astype(int)
panel_pooled['interaction'] = panel_pooled['is_informed'] * panel_pooled['is_outage_day']

# Add hour fixed effects
hour_dummies = pd.get_dummies(panel_pooled['hour_of_day'], prefix='hour', drop_first=True).astype(float)
panel_pooled = pd.concat([panel_pooled, hour_dummies], axis=1)

# Regression with hour FE
y_pooled = panel_pooled['active'].values.astype(float)
X_cols = ['is_informed', 'is_outage_day', 'interaction'] + [c for c in hour_dummies.columns]
X_pooled = sm.add_constant(panel_pooled[X_cols].values.astype(float))

model_pooled = sm.OLS(y_pooled, X_pooled)
results_pooled = model_pooled.fit(cov_type='cluster', cov_kwds={'groups': panel_pooled['wallet'].values})

print(f"  Pooled DiD (hours 12-16 UTC, with hour FE):")
print(f"    DiD Coefficient: {results_pooled.params[3]*100:.2f} pp")
print(f"    t-statistic: {results_pooled.tvalues[3]:.2f}")

# =============================================================================
# INTENSIVE MARGIN
# =============================================================================

print("\n[4c/5] Intensive margin (conditional on being active)...")

# Among active wallets, do informed submit more fills during outage?
active_panel = panel[panel['active'] == 1].copy()

if len(active_panel) > 10:
    y_int = active_panel['n_fills'].values
    X_int = sm.add_constant(active_panel[['is_informed', 'is_outage_day', 'interaction']].values)

    model_int = sm.OLS(y_int, X_int)
    results_int = model_int.fit(cov_type='cluster', cov_kwds={'groups': active_panel['wallet'].values})

    print(f"  Intensive Margin (conditional on activity):")
    print(f"    DiD Coefficient: {results_int.params[3]:.2f} fills")
    print(f"    t-statistic: {results_int.tvalues[3]:.2f}")
    intensive_coef = float(results_int.params[3])
    intensive_t = float(results_int.tvalues[3])
    intensive_p = float(results_int.pvalues[3])
else:
    print("  Insufficient observations for intensive margin analysis")
    intensive_coef = None
    intensive_t = None
    intensive_p = None

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n[5/5] Saving results...")

results = {
    'specification': 'Same-hour DiD: Compare 14:00 UTC on July 28 (control) vs July 29 (outage)',
    'methodology': {
        'treatment_group': 'Q5 (informed) wallets',
        'control_group': 'Q1 (uninformed) wallets',
        'pre_period': '14:00 UTC on July 28, 2025 (control day)',
        'post_period': '14:00 UTC on July 29, 2025 (outage day)',
        'classification': 'Out-of-sample (train morning July 28, test afternoon July 28-29)',
        'standard_errors': 'Wallet-clustered'
    },
    'main_result': {
        'did_coefficient_pp': float(did_coef * 100),
        'did_se_pp': float(did_se * 100),
        'did_t_stat': float(did_t),
        'did_p_value': float(did_p),
        'significant_5pct': float(did_p) < 0.05,
        'significant_10pct': float(did_p) < 0.10
    },
    'robustness_pooled': {
        'hours_compared': '12:00-16:00 UTC',
        'did_coefficient_pp': float(results_pooled.params[3] * 100),
        'did_t_stat': float(results_pooled.tvalues[3]),
        'did_p_value': float(results_pooled.pvalues[3])
    },
    'intensive_margin': {
        'did_coefficient_fills': intensive_coef,
        'did_t_stat': intensive_t,
        'did_p_value': intensive_p
    },
    'group_means': {
        'informed_control': float(informed_control),
        'informed_outage': float(informed_outage),
        'uninformed_control': float(uninformed_control),
        'uninformed_outage': float(uninformed_outage),
        'informed_change_pp': float((informed_outage - informed_control) * 100),
        'uninformed_change_pp': float((uninformed_outage - uninformed_control) * 100),
        'did_manual_pp': float(did_manual * 100)
    },
    'sample': {
        'n_observations': int(len(panel)),
        'n_wallets': int(panel['wallet'].nunique()),
        'n_informed': int(len(q5_wallets)),
        'n_uninformed': int(len(q1_wallets)),
        'n_fills_control_hour': int(len(fills_control)),
        'n_fills_outage_hour': int(len(fills_outage))
    }
}

with open(RESULTS_DIR / 'wallet_did_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to: {RESULTS_DIR / 'wallet_did_results.json'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY: SAME-HOUR DiD ANALYSIS")
print("=" * 80)
print(f"""
Econometric Design:
-------------------
  Compare the SAME hour (14:00 UTC) on:
    - Control day: July 28, 2025
    - Outage day: July 29, 2025

  This controls for hour-of-day activity patterns.

Results:
--------
  Informed (Q5) activity:
    Control day: {informed_control*100:.1f}%
    Outage day:  {informed_outage*100:.1f}%
    Change:      {(informed_outage - informed_control)*100:+.1f} pp

  Uninformed (Q1) activity:
    Control day: {uninformed_control*100:.1f}%
    Outage day:  {uninformed_outage*100:.1f}%
    Change:      {(uninformed_outage - uninformed_control)*100:+.1f} pp

  ╔══════════════════════════════════════════════════╗
  ║  DiD COEFFICIENT: {did_coef*100:+.2f} pp                       ║
  ║  t-statistic:     {did_t:.2f}                            ║
  ║  p-value:         {did_p:.4f}                          ║
  ║  Significant at 5%: {'YES' if did_p < 0.05 else 'NO'}                          ║
  ╚══════════════════════════════════════════════════╝

Interpretation:
---------------
  During the CrowdStrike outage, informed (Q5) wallets increased their
  activity by {(informed_outage - informed_control)*100:.1f} pp while uninformed (Q1) wallets increased
  by {(uninformed_outage - uninformed_control)*100:.1f} pp. The differential (DiD) of {did_coef*100:.2f} pp
  {'IS' if did_p < 0.05 else 'IS NOT'} statistically significant at the 5% level (t = {did_t:.2f}).
""")
