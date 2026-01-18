"""
REFINED Lag Structure Analysis: Matching Paper's Methodology

Key refinements:
1. Use informed-to-uninformed RATIO (not share) to match paper's metric
2. Focus on the CHANGE in composition, not levels
3. Use first-differences to address non-stationarity
4. Test lead-lag at finer granularity (30-second bins)

This addresses the JFE referee concern about endogeneity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
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

DATA_DIR = Path(_DATA_DIR)
RESULTS_DIR = Path(_RESULTS_DIR)

print("="*70)
print("REFINED LAG STRUCTURE ANALYSIS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading data...")

fills = pd.read_parquet(DATA_DIR / "wallet_fills_data.parquet")
l2_data = pd.read_parquet(DATA_DIR / "outage_event_study_data.parquet")

fills['datetime'] = pd.to_datetime(fills['time'], unit='ms', utc=True)
l2_data['datetime'] = pd.to_datetime(l2_data['time_ms'], unit='ms', utc=True)

print(f"    Trades: {len(fills):,}")
print(f"    L2 snapshots: {len(l2_data):,}")

# ============================================================================
# WALLET CLASSIFICATION - MATCH PAPER'S APPROACH
# ============================================================================
print("\n[2] Classifying wallets (matching paper methodology)...")

# Training: July 28 before outage
training_start = pd.Timestamp('2025-07-28 00:00:00', tz='UTC')
training_end = pd.Timestamp('2025-07-28 14:00:00', tz='UTC')

train = fills[(fills['datetime'] >= training_start) & (fills['datetime'] < training_end)].copy()

# Only takers for classification
train_takers = train[train['crossed'] == True].copy()

# Compute price at each minute
train_takers = train_takers.sort_values(['coin', 'datetime'])
train_takers['minute'] = train_takers['datetime'].dt.floor('1min')

# Get minute prices
minute_px = train_takers.groupby(['coin', 'minute'])['px'].last().reset_index()
minute_px = minute_px.sort_values(['coin', 'minute'])
minute_px['px_1m'] = minute_px.groupby('coin')['px'].shift(-1)

train_takers = train_takers.merge(minute_px[['coin', 'minute', 'px_1m']], on=['coin', 'minute'], how='left')

# Compute markout
train_takers['direction'] = train_takers['side'].map({'B': 1, 'A': -1})
train_takers['markout'] = train_takers['direction'] * (train_takers['px_1m'] - train_takers['px']) / train_takers['px'] * 10000

train_takers = train_takers.dropna(subset=['markout'])

# Classify wallets
wallet_markout = train_takers.groupby('wallet').agg({
    'markout': ['mean', 'count']
}).reset_index()
wallet_markout.columns = ['wallet', 'mean_markout', 'n_trades']
wallet_markout = wallet_markout[wallet_markout['n_trades'] >= 5]

# Top 20% = informed, Bottom 20% = uninformed
wallet_markout['quintile'] = pd.qcut(wallet_markout['mean_markout'], 5, labels=[1,2,3,4,5])
informed_wallets = set(wallet_markout[wallet_markout['quintile'] == 5]['wallet'])
uninformed_wallets = set(wallet_markout[wallet_markout['quintile'] == 1]['wallet'])

print(f"    Informed wallets: {len(informed_wallets)}")
print(f"    Uninformed wallets: {len(uninformed_wallets)}")

# ============================================================================
# CREATE 30-SECOND PANEL FOR FINE-GRAINED LAG ANALYSIS
# ============================================================================
print("\n[3] Creating 30-second panel...")

# Focus on July 29 outage window (hours 13-16)
outage_day = pd.Timestamp('2025-07-29', tz='UTC')
window_start = outage_day + pd.Timedelta(hours=13)
window_end = outage_day + pd.Timedelta(hours=16)

# Filter fills
window_fills = fills[(fills['datetime'] >= window_start) & (fills['datetime'] < window_end)].copy()
window_fills = window_fills[window_fills['crossed'] == True]  # Takers only

window_fills['is_informed'] = window_fills['wallet'].isin(informed_wallets)
window_fills['is_uninformed'] = window_fills['wallet'].isin(uninformed_wallets)

# 30-second bins
window_fills['bin_30s'] = window_fills['datetime'].dt.floor('30s')

# Aggregate by 30-second bin and asset
composition = window_fills.groupby(['coin', 'bin_30s']).agg({
    'is_informed': 'sum',
    'is_uninformed': 'sum',
    'wallet': 'count'
}).reset_index()
composition.columns = ['asset', 'time', 'n_informed', 'n_uninformed', 'n_total']

# Compute ratio (add small constant to avoid division by zero)
composition['ratio'] = (composition['n_informed'] + 0.1) / (composition['n_uninformed'] + 0.1)
composition['log_ratio'] = np.log(composition['ratio'])

print(f"    Composition observations: {len(composition)}")

# Filter L2 data
window_l2 = l2_data[(l2_data['datetime'] >= window_start) & (l2_data['datetime'] < window_end)].copy()
window_l2['bin_30s'] = window_l2['datetime'].dt.floor('30s')

# Aggregate spreads
spreads = window_l2.groupby(['asset', 'bin_30s']).agg({
    'spread_bps': 'median',
    'best_bid_changed': 'sum',
    'best_ask_changed': 'sum'
}).reset_index()
spreads.columns = ['asset', 'time', 'spread', 'bid_updates', 'ask_updates']
spreads['quote_updates'] = spreads['bid_updates'] + spreads['ask_updates']

print(f"    Spread observations: {len(spreads)}")

# Merge
panel = spreads.merge(composition[['asset', 'time', 'ratio', 'log_ratio', 'n_informed', 'n_uninformed', 'n_total']],
                      on=['asset', 'time'], how='inner')

print(f"    Merged panel: {len(panel)}")

# Sort and create lags/leads
panel = panel.sort_values(['asset', 'time'])

for lag in [1, 2, 3, 4, 5, 6]:
    panel[f'spread_lag{lag}'] = panel.groupby('asset')['spread'].shift(lag)
    panel[f'spread_lead{lag}'] = panel.groupby('asset')['spread'].shift(-lag)
    panel[f'ratio_lag{lag}'] = panel.groupby('asset')['ratio'].shift(lag)
    panel[f'ratio_lead{lag}'] = panel.groupby('asset')['ratio'].shift(-lag)
    panel[f'log_ratio_lag{lag}'] = panel.groupby('asset')['log_ratio'].shift(lag)

# First differences
panel['d_spread'] = panel['spread'] - panel['spread_lag1']
panel['d_ratio'] = panel['ratio'] - panel['ratio_lag1']
panel['d_log_ratio'] = panel['log_ratio'] - panel['log_ratio_lag1']

# Mark outage period (14:10-14:47)
panel['minute'] = panel['time'].dt.hour * 60 + panel['time'].dt.minute
panel['is_outage'] = (panel['minute'] >= 14*60+10) & (panel['minute'] <= 14*60+47)

# Drop missing
panel_clean = panel.dropna(subset=['spread', 'ratio', 'spread_lag1', 'ratio_lag1'])
print(f"    Clean panel: {len(panel_clean)}")

# ============================================================================
# TEST 1: PREDICTIVE REGRESSIONS IN LEVELS
# ============================================================================
print("\n" + "="*70)
print("[4] Predictive Regressions (Levels)")
print("="*70)

# Forward: Does lagged ratio predict current spread?
X_fwd = sm.add_constant(panel_clean[['spread_lag1', 'log_ratio_lag1']])
y_fwd = panel_clean['spread']
model_fwd = OLS(y_fwd, X_fwd).fit(cov_type='HC1')

print("\n  A. Lagged Informed Ratio -> Current Spread")
print(f"     β(log_ratio_lag1) = {model_fwd.params['log_ratio_lag1']:.4f}")
print(f"     t-stat = {model_fwd.tvalues['log_ratio_lag1']:.2f}")
print(f"     p-value = {model_fwd.pvalues['log_ratio_lag1']:.4f}")

# Reverse: Does lagged spread predict current ratio?
X_rev = sm.add_constant(panel_clean[['log_ratio_lag1', 'spread_lag1']])
y_rev = panel_clean['log_ratio']
model_rev = OLS(y_rev, X_rev).fit(cov_type='HC1')

print("\n  B. Lagged Spread -> Current Informed Ratio")
print(f"     β(spread_lag1) = {model_rev.params['spread_lag1']:.6f}")
print(f"     t-stat = {model_rev.tvalues['spread_lag1']:.2f}")
print(f"     p-value = {model_rev.pvalues['spread_lag1']:.4f}")

# ============================================================================
# TEST 2: FIRST-DIFFERENCE REGRESSIONS (addresses non-stationarity)
# ============================================================================
print("\n" + "="*70)
print("[5] First-Difference Regressions")
print("="*70)

panel_diff = panel_clean.dropna(subset=['d_spread', 'd_log_ratio'])

# Does change in ratio predict change in spread?
# Use lagged ratio change to predict current spread change
panel_diff['d_log_ratio_lag1'] = panel_diff.groupby('asset')['d_log_ratio'].shift(1)
panel_diff = panel_diff.dropna(subset=['d_log_ratio_lag1'])

X_diff_fwd = sm.add_constant(panel_diff['d_log_ratio_lag1'])
y_diff_fwd = panel_diff['d_spread']
model_diff_fwd = OLS(y_diff_fwd, X_diff_fwd).fit(cov_type='HC1')

print("\n  A. Δlog_ratio(t-1) -> Δspread(t)")
print(f"     β = {model_diff_fwd.params['d_log_ratio_lag1']:.4f}")
print(f"     t-stat = {model_diff_fwd.tvalues['d_log_ratio_lag1']:.2f}")
print(f"     p-value = {model_diff_fwd.pvalues['d_log_ratio_lag1']:.4f}")

# Reverse
panel_diff['d_spread_lag1'] = panel_diff.groupby('asset')['d_spread'].shift(1)
panel_diff = panel_diff.dropna(subset=['d_spread_lag1'])

X_diff_rev = sm.add_constant(panel_diff['d_spread_lag1'])
y_diff_rev = panel_diff['d_log_ratio']
model_diff_rev = OLS(y_diff_rev, X_diff_rev).fit(cov_type='HC1')

print("\n  B. Δspread(t-1) -> Δlog_ratio(t)")
print(f"     β = {model_diff_rev.params['d_spread_lag1']:.6f}")
print(f"     t-stat = {model_diff_rev.tvalues['d_spread_lag1']:.2f}")
print(f"     p-value = {model_diff_rev.pvalues['d_spread_lag1']:.4f}")

# ============================================================================
# TEST 3: LEAD-LAG ANALYSIS (Which leads?)
# ============================================================================
print("\n" + "="*70)
print("[6] Lead-Lag Cross-Correlation Analysis")
print("="*70)

# Compute cross-correlations at different lags
# Positive lag = ratio leads spread
# Negative lag = spread leads ratio

lags_to_test = range(-6, 7)  # -6 to +6 (each lag = 30 seconds)
cross_corr = []

for lag in lags_to_test:
    if lag < 0:
        # Spread leads: correlate spread(t) with ratio(t+|lag|)
        shifted_ratio = panel_clean.groupby('asset')['log_ratio'].shift(lag)
        corr = panel_clean['spread'].corr(shifted_ratio)
        label = f"Spread leads by {-lag*30}s"
    elif lag > 0:
        # Ratio leads: correlate ratio(t) with spread(t+lag)
        shifted_spread = panel_clean.groupby('asset')['spread'].shift(-lag)
        corr = panel_clean['log_ratio'].corr(shifted_spread)
        label = f"Ratio leads by {lag*30}s"
    else:
        corr = panel_clean['spread'].corr(panel_clean['log_ratio'])
        label = "Contemporaneous"

    cross_corr.append({'lag': lag, 'lag_seconds': lag * 30, 'correlation': corr, 'label': label})

cc_df = pd.DataFrame(cross_corr)

print("\n  Cross-Correlation: Spread vs Informed Ratio")
print("  " + "-"*50)
print(f"  {'Lag (30s units)':<18} {'Seconds':<12} {'Correlation':<12}")
print("  " + "-"*50)
for _, row in cc_df.iterrows():
    print(f"  {row['lag']:>10}         {row['lag_seconds']:>6}s      {row['correlation']:>8.4f}")

# Find peak
max_corr_row = cc_df.loc[cc_df['correlation'].abs().idxmax()]
print(f"\n  Peak correlation: {max_corr_row['correlation']:.4f} at lag {max_corr_row['lag']} ({max_corr_row['lag_seconds']}s)")
print(f"  Interpretation: {'Ratio leads spread' if max_corr_row['lag'] > 0 else 'Spread leads ratio' if max_corr_row['lag'] < 0 else 'Contemporaneous'}")

# ============================================================================
# TEST 4: WITHIN-OUTAGE TIMING (HIGH-FREQUENCY)
# ============================================================================
print("\n" + "="*70)
print("[7] Within-Outage Timing Analysis")
print("="*70)

outage_panel = panel_clean[panel_clean['is_outage']].copy()
print(f"\n  Observations during outage: {len(outage_panel)}")

if len(outage_panel) > 20:
    # Aggregate to 2-minute bins for clarity
    outage_panel['min_bin'] = (outage_panel['minute'] - 14*60) // 2 * 2 + 10  # minutes into hour 14

    timing = outage_panel.groupby('min_bin').agg({
        'spread': 'mean',
        'ratio': 'mean',
        'quote_updates': 'mean',
        'n_total': 'sum'
    }).reset_index()

    # Normalize to first observation
    baseline_spread = timing['spread'].iloc[0]
    baseline_ratio = timing['ratio'].iloc[0]
    timing['spread_pct_change'] = (timing['spread'] / baseline_spread - 1) * 100
    timing['ratio_pct_change'] = (timing['ratio'] / baseline_ratio - 1) * 100

    print("\n  2-Minute Evolution During Outage (14:10-14:47):")
    print("  " + "-"*65)
    print(f"  {'Minute':<10} {'Spread':<12} {'Ratio':<12} {'Δ Spread%':<12} {'Δ Ratio%':<12}")
    print("  " + "-"*65)

    for _, row in timing.iterrows():
        print(f"  14:{int(row['min_bin']):02d}      {row['spread']:.2f}        {row['ratio']:.2f}        {row['spread_pct_change']:+.1f}%       {row['ratio_pct_change']:+.1f}%")

    # Test which moves first
    # Compute when each variable first exceeds 50% of its peak change
    spread_peak = timing['spread'].max()
    ratio_peak = timing['ratio'].max()
    spread_half = baseline_spread + (spread_peak - baseline_spread) * 0.5
    ratio_half = baseline_ratio + (ratio_peak - baseline_ratio) * 0.5

    spread_cross_time = timing[timing['spread'] >= spread_half]['min_bin'].min() if len(timing[timing['spread'] >= spread_half]) > 0 else 99
    ratio_cross_time = timing[timing['ratio'] >= ratio_half]['min_bin'].min() if len(timing[timing['ratio'] >= ratio_half]) > 0 else 99

    print(f"\n  Timing of 50% peak crossing:")
    print(f"    Spread crosses at 14:{spread_cross_time:02d}")
    print(f"    Ratio crosses at 14:{ratio_cross_time:02d}")
    print(f"    --> {'Ratio leads by ' + str(spread_cross_time - ratio_cross_time) + ' minutes' if ratio_cross_time < spread_cross_time else 'Spread leads by ' + str(ratio_cross_time - spread_cross_time) + ' minutes' if spread_cross_time < ratio_cross_time else 'Simultaneous'}")

# ============================================================================
# TEST 5: GRANGER CAUSALITY WITH PROPER SPECIFICATION
# ============================================================================
print("\n" + "="*70)
print("[8] Granger Causality Tests (VAR-based)")
print("="*70)

from statsmodels.tsa.stattools import grangercausalitytests

# Prepare data for Granger test (need time series, aggregate across assets)
ts_data = panel_clean.groupby('time').agg({
    'spread': 'mean',
    'log_ratio': 'mean'
}).reset_index().sort_values('time')

ts_data = ts_data.dropna()

print(f"\n  Time series length: {len(ts_data)} observations (30-second frequency)")

# Granger test: ratio -> spread
print("\n  A. Granger Test: log_ratio -> spread")
try:
    gc_result = grangercausalitytests(ts_data[['spread', 'log_ratio']], maxlag=4, verbose=False)
    for lag in [1, 2, 3, 4]:
        f_stat = gc_result[lag][0]['ssr_ftest'][0]
        p_val = gc_result[lag][0]['ssr_ftest'][1]
        print(f"     Lag {lag}: F = {f_stat:.2f}, p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
except Exception as e:
    print(f"     Error: {e}")

# Granger test: spread -> ratio
print("\n  B. Granger Test: spread -> log_ratio")
try:
    gc_result_rev = grangercausalitytests(ts_data[['log_ratio', 'spread']], maxlag=4, verbose=False)
    for lag in [1, 2, 3, 4]:
        f_stat = gc_result_rev[lag][0]['ssr_ftest'][0]
        p_val = gc_result_rev[lag][0]['ssr_ftest'][1]
        print(f"     Lag {lag}: F = {f_stat:.2f}, p = {p_val:.4f} {'*' if p_val < 0.05 else ''}")
except Exception as e:
    print(f"     Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Addressing Endogeneity")
print("="*70)

print("""
KEY FINDINGS:

1. LEVEL REGRESSIONS:
   - Lagged ratio -> spread: Tests whether composition predicts spreads
   - Lagged spread -> ratio: Tests reverse causality

2. FIRST-DIFFERENCE REGRESSIONS:
   - Addresses non-stationarity concern
   - Tests whether CHANGES in composition predict CHANGES in spreads

3. LEAD-LAG CORRELATION:
   - Shows which variable leads in the cross-correlation structure
   - Peak at positive lag = ratio leads spread (supports causality claim)
   - Peak at negative lag = spread leads ratio (supports reverse causality)

4. WITHIN-OUTAGE TIMING:
   - Tests whether composition shifts precede spread widening
   - within the 37-minute event window

5. GRANGER CAUSALITY:
   - Formal test of whether past values of one variable help predict
   - the other, beyond its own past values
""")

# Save results
panel_clean.to_csv(RESULTS_DIR / "lag_panel_refined.csv", index=False)
cc_df.to_csv(RESULTS_DIR / "cross_correlations.csv", index=False)
print(f"\nResults saved to {RESULTS_DIR}")
print("="*70)
