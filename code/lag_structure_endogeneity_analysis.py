"""
Lag Structure Analysis: Addressing Endogeneity in the Selection Channel

This script implements three tests to address the concern that informed share
may be endogenous to spreads (composition as response, not cause):

1. GRANGER-STYLE LAG ANALYSIS: Test whether lagged informed share predicts
   future spreads after controlling for lagged spreads, and whether the
   reverse relationship is weaker.

2. EVENT-TIME DECOMPOSITION: Show that composition shifts precede spread
   widening within the outage window at high frequency.

3. LEAD-LAG ASYMMETRY: Estimate impulse responses in both directions.

Author: Generated for JFE submission
Date: 2025
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

# Paths
DATA_DIR = Path(_DATA_DIR)
RESULTS_DIR = Path(_RESULTS_DIR)
RESULTS_DIR.mkdir(exist_ok=True)

print("="*70)
print("LAG STRUCTURE ANALYSIS: ADDRESSING ENDOGENEITY")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1] Loading data...")

# Load trade data with wallet identities
fills = pd.read_parquet(DATA_DIR / "wallet_fills_data.parquet")
print(f"    Loaded {len(fills):,} trades with wallet identities")

# Load order book data
l2_data = pd.read_parquet(DATA_DIR / "outage_event_study_data.parquet")
print(f"    Loaded {len(l2_data):,} order book snapshots")

# Convert timestamps
fills['datetime'] = pd.to_datetime(fills['time'], unit='ms', utc=True)
l2_data['datetime'] = pd.to_datetime(l2_data['time_ms'], unit='ms', utc=True)

# ============================================================================
# STEP 2: LOAD WALLET CLASSIFICATION (FROM EXISTING ANALYSIS)
# ============================================================================
print("\n[2] Loading wallet classification...")

# We need to classify wallets as informed/uninformed
# Use training period: July 28, 2025 before 14:00 UTC
training_start = pd.Timestamp('2025-07-28 00:00:00', tz='UTC')
training_end = pd.Timestamp('2025-07-28 14:00:00', tz='UTC')

training_fills = fills[(fills['datetime'] >= training_start) &
                       (fills['datetime'] < training_end)].copy()

print(f"    Training period trades: {len(training_fills):,}")

# Compute markouts for classification
# We need price data - use the fills themselves to compute approximate markouts
# Group by coin and compute 1-minute forward price changes
training_fills = training_fills.sort_values(['coin', 'datetime'])
training_fills['minute'] = training_fills['datetime'].dt.floor('1min')

# Get minute-level prices
minute_prices = training_fills.groupby(['coin', 'minute']).agg({
    'px': 'last'
}).reset_index()
minute_prices.columns = ['coin', 'minute', 'price']

# Create forward price (1 minute ahead)
minute_prices = minute_prices.sort_values(['coin', 'minute'])
minute_prices['price_1m'] = minute_prices.groupby('coin')['price'].shift(-1)

# Merge back to fills
training_fills = training_fills.merge(
    minute_prices[['coin', 'minute', 'price_1m']],
    on=['coin', 'minute'],
    how='left'
)

# Compute markout
training_fills['direction'] = training_fills['side'].map({'B': 1, 'A': -1})
training_fills['markout_bps'] = (
    training_fills['direction'] *
    (training_fills['price_1m'] - training_fills['px']) /
    training_fills['px'] * 10000
)

# Drop missing markouts
training_fills = training_fills.dropna(subset=['markout_bps'])

# Classify wallets
wallet_stats = training_fills.groupby('wallet').agg({
    'markout_bps': ['mean', 'count']
}).reset_index()
wallet_stats.columns = ['wallet', 'mean_markout', 'n_trades']

# Require minimum trades for classification
MIN_TRADES = 5
wallet_stats = wallet_stats[wallet_stats['n_trades'] >= MIN_TRADES]

# Classify by quintile
wallet_stats['quintile'] = pd.qcut(wallet_stats['mean_markout'], 5, labels=[1,2,3,4,5])
wallet_stats['is_informed'] = wallet_stats['quintile'] == 5
wallet_stats['is_uninformed'] = wallet_stats['quintile'] == 1

n_informed = wallet_stats['is_informed'].sum()
n_uninformed = wallet_stats['is_uninformed'].sum()
print(f"    Classified {n_informed} informed wallets, {n_uninformed} uninformed wallets")

# Create classification lookup
informed_wallets = set(wallet_stats[wallet_stats['is_informed']]['wallet'])
uninformed_wallets = set(wallet_stats[wallet_stats['is_uninformed']]['wallet'])

# ============================================================================
# STEP 3: COMPUTE MINUTE-LEVEL PANEL DATA
# ============================================================================
print("\n[3] Computing minute-level panel data...")

# Focus on outage event window: July 29, hours 12-17 UTC (around outage at 14:10-14:47)
event_start = pd.Timestamp('2025-07-29 12:00:00', tz='UTC')
event_end = pd.Timestamp('2025-07-29 17:00:00', tz='UTC')

# Also include July 28 and July 30 same hours for comparison
comparison_dates = [
    (pd.Timestamp('2025-07-28 12:00:00', tz='UTC'), pd.Timestamp('2025-07-28 17:00:00', tz='UTC')),
    (pd.Timestamp('2025-07-29 12:00:00', tz='UTC'), pd.Timestamp('2025-07-29 17:00:00', tz='UTC')),
    (pd.Timestamp('2025-07-30 12:00:00', tz='UTC'), pd.Timestamp('2025-07-30 17:00:00', tz='UTC')),
]

# Filter fills to event windows
analysis_fills = fills[
    ((fills['datetime'] >= comparison_dates[0][0]) & (fills['datetime'] < comparison_dates[0][1])) |
    ((fills['datetime'] >= comparison_dates[1][0]) & (fills['datetime'] < comparison_dates[1][1])) |
    ((fills['datetime'] >= comparison_dates[2][0]) & (fills['datetime'] < comparison_dates[2][1]))
].copy()

print(f"    Analysis window trades: {len(analysis_fills):,}")

# Add classification
analysis_fills['is_informed'] = analysis_fills['wallet'].isin(informed_wallets)
analysis_fills['is_uninformed'] = analysis_fills['wallet'].isin(uninformed_wallets)
analysis_fills['is_taker'] = analysis_fills['crossed'] == True

# Filter to takers only (taker composition is what we measure)
taker_fills = analysis_fills[analysis_fills['is_taker']].copy()
print(f"    Taker trades: {len(taker_fills):,}")

# Create minute bins
taker_fills['minute'] = taker_fills['datetime'].dt.floor('1min')

# Compute minute-level informed share by asset
minute_composition = taker_fills.groupby(['coin', 'minute']).agg({
    'is_informed': ['sum', 'count'],
    'is_uninformed': 'sum'
}).reset_index()
minute_composition.columns = ['asset', 'minute', 'n_informed', 'n_takers', 'n_uninformed']

# Compute informed share (among classified takers)
minute_composition['n_classified'] = minute_composition['n_informed'] + minute_composition['n_uninformed']
minute_composition['informed_share'] = np.where(
    minute_composition['n_classified'] > 0,
    minute_composition['n_informed'] / minute_composition['n_classified'],
    np.nan
)

# Compute informed-to-uninformed ratio
minute_composition['informed_ratio'] = np.where(
    minute_composition['n_uninformed'] > 0,
    minute_composition['n_informed'] / minute_composition['n_uninformed'],
    np.nan
)

print(f"    Created {len(minute_composition):,} asset-minute observations")

# ============================================================================
# STEP 4: COMPUTE MINUTE-LEVEL SPREADS AND QUOTE UPDATES
# ============================================================================
print("\n[4] Computing minute-level spreads...")

# Filter L2 data to same windows
l2_analysis = l2_data[
    ((l2_data['datetime'] >= comparison_dates[0][0]) & (l2_data['datetime'] < comparison_dates[0][1])) |
    ((l2_data['datetime'] >= comparison_dates[1][0]) & (l2_data['datetime'] < comparison_dates[1][1])) |
    ((l2_data['datetime'] >= comparison_dates[2][0]) & (l2_data['datetime'] < comparison_dates[2][1]))
].copy()

print(f"    L2 snapshots in window: {len(l2_analysis):,}")

l2_analysis['minute'] = l2_analysis['datetime'].dt.floor('1min')

# Aggregate to minute level
minute_spreads = l2_analysis.groupby(['asset', 'minute']).agg({
    'spread_bps': 'median',
    'best_bid_changed': 'sum',
    'best_ask_changed': 'sum',
    'book_state_changed': 'sum'
}).reset_index()

minute_spreads['quote_updates'] = minute_spreads['best_bid_changed'] + minute_spreads['best_ask_changed']

print(f"    Created {len(minute_spreads):,} asset-minute spread observations")

# ============================================================================
# STEP 5: MERGE AND CREATE LAG VARIABLES
# ============================================================================
print("\n[5] Creating panel with lags...")

# Merge composition and spreads
panel = minute_spreads.merge(
    minute_composition[['asset', 'minute', 'informed_share', 'informed_ratio', 'n_takers', 'n_classified']],
    on=['asset', 'minute'],
    how='inner'
)

print(f"    Merged panel: {len(panel):,} observations")

# Sort and create lags
panel = panel.sort_values(['asset', 'minute'])

# Create lag and lead variables (1-5 minute lags)
for lag in range(1, 6):
    panel[f'spread_lag{lag}'] = panel.groupby('asset')['spread_bps'].shift(lag)
    panel[f'informed_share_lag{lag}'] = panel.groupby('asset')['informed_share'].shift(lag)
    panel[f'quote_updates_lag{lag}'] = panel.groupby('asset')['quote_updates'].shift(lag)

    panel[f'spread_lead{lag}'] = panel.groupby('asset')['spread_bps'].shift(-lag)
    panel[f'informed_share_lead{lag}'] = panel.groupby('asset')['informed_share'].shift(-lag)

# Create change variables
panel['spread_change'] = panel['spread_bps'] - panel['spread_lag1']
panel['informed_share_change'] = panel['informed_share'] - panel['informed_share_lag1']

# Add time controls
panel['date'] = panel['minute'].dt.date
panel['hour'] = panel['minute'].dt.hour
panel['minute_of_hour'] = panel['minute'].dt.minute

# Mark outage period (July 29, 14:10-14:47)
panel['is_outage'] = (
    (panel['date'] == pd.Timestamp('2025-07-29').date()) &
    (panel['hour'] == 14) &
    (panel['minute_of_hour'] >= 10) &
    (panel['minute_of_hour'] <= 47)
)

# Mark outage day
panel['is_outage_day'] = panel['date'] == pd.Timestamp('2025-07-29').date()

# Drop missing
panel_clean = panel.dropna(subset=['spread_bps', 'informed_share', 'spread_lag1', 'informed_share_lag1'])
print(f"    Clean panel (no missing lags): {len(panel_clean):,} observations")

# ============================================================================
# STEP 6: GRANGER-STYLE LAG REGRESSIONS
# ============================================================================
print("\n[6] Running Granger-style lag regressions...")
print("="*70)

results = {}

# -------------------------------------------------------------------------
# TEST 1: Does lagged informed share predict spreads?
# -------------------------------------------------------------------------
print("\n--- TEST 1: Lagged Informed Share → Current Spread ---")

# Model 1a: Spread = f(lagged spread, lagged informed share)
X1a = panel_clean[['spread_lag1', 'informed_share_lag1']].copy()
X1a = sm.add_constant(X1a)
y1a = panel_clean['spread_bps']

model_1a = OLS(y1a, X1a).fit(cov_type='HC1')
print("\nModel 1a: Spread_t = α + β₁·Spread_{t-1} + β₂·InformedShare_{t-1}")
print(f"    β₂ (informed_share_lag1) = {model_1a.params['informed_share_lag1']:.4f}")
print(f"    t-stat = {model_1a.tvalues['informed_share_lag1']:.2f}")
print(f"    p-value = {model_1a.pvalues['informed_share_lag1']:.4f}")
print(f"    R² = {model_1a.rsquared:.4f}")

results['model_1a'] = {
    'coef': model_1a.params['informed_share_lag1'],
    'tstat': model_1a.tvalues['informed_share_lag1'],
    'pval': model_1a.pvalues['informed_share_lag1'],
    'r2': model_1a.rsquared,
    'n': len(y1a)
}

# Model 1b: Add more lags
panel_1b = panel_clean[['spread_bps', 'spread_lag1', 'spread_lag2', 'informed_share_lag1', 'informed_share_lag2']].dropna()
X1b = sm.add_constant(panel_1b[['spread_lag1', 'spread_lag2', 'informed_share_lag1', 'informed_share_lag2']])
y1b = panel_1b['spread_bps']

model_1b = OLS(y1b, X1b).fit(cov_type='HC1')
print(f"\nModel 1b: With 2 lags")
print(f"    β (informed_share_lag1) = {model_1b.params['informed_share_lag1']:.4f}, t = {model_1b.tvalues['informed_share_lag1']:.2f}")
print(f"    β (informed_share_lag2) = {model_1b.params['informed_share_lag2']:.4f}, t = {model_1b.tvalues['informed_share_lag2']:.2f}")

# Model 1c: Control for quote staleness
X1c = panel_clean[['spread_lag1', 'informed_share_lag1', 'quote_updates_lag1']].copy()
X1c = sm.add_constant(X1c)

model_1c = OLS(y1a, X1c).fit(cov_type='HC1')
print(f"\nModel 1c: Controlling for quote staleness (quote_updates)")
print(f"    β (informed_share_lag1) = {model_1c.params['informed_share_lag1']:.4f}, t = {model_1c.tvalues['informed_share_lag1']:.2f}")
print(f"    β (quote_updates_lag1) = {model_1c.params['quote_updates_lag1']:.6f}, t = {model_1c.tvalues['quote_updates_lag1']:.2f}")

results['model_1c'] = {
    'coef_informed': model_1c.params['informed_share_lag1'],
    'tstat_informed': model_1c.tvalues['informed_share_lag1'],
    'coef_staleness': model_1c.params['quote_updates_lag1'],
    'tstat_staleness': model_1c.tvalues['quote_updates_lag1'],
    'r2': model_1c.rsquared,
    'n': len(y1a)
}

# -------------------------------------------------------------------------
# TEST 2: Does lagged spread predict informed share? (Reverse causality)
# -------------------------------------------------------------------------
print("\n--- TEST 2: Lagged Spread → Current Informed Share (Reverse Causality) ---")

# Model 2a: Informed share = f(lagged informed share, lagged spread)
X2a = panel_clean[['informed_share_lag1', 'spread_lag1']].copy()
X2a = sm.add_constant(X2a)
y2a = panel_clean['informed_share']

model_2a = OLS(y2a, X2a).fit(cov_type='HC1')
print("\nModel 2a: InformedShare_t = α + β₁·InformedShare_{t-1} + β₂·Spread_{t-1}")
print(f"    β₂ (spread_lag1) = {model_2a.params['spread_lag1']:.6f}")
print(f"    t-stat = {model_2a.tvalues['spread_lag1']:.2f}")
print(f"    p-value = {model_2a.pvalues['spread_lag1']:.4f}")

results['model_2a'] = {
    'coef': model_2a.params['spread_lag1'],
    'tstat': model_2a.tvalues['spread_lag1'],
    'pval': model_2a.pvalues['spread_lag1'],
    'r2': model_2a.rsquared,
    'n': len(y2a)
}

# -------------------------------------------------------------------------
# TEST 3: Asymmetry test - compare coefficients
# -------------------------------------------------------------------------
print("\n--- TEST 3: Asymmetry in Predictive Power ---")

# Standardize variables for comparison
panel_std = panel_clean.copy()
panel_std['spread_std'] = (panel_std['spread_bps'] - panel_std['spread_bps'].mean()) / panel_std['spread_bps'].std()
panel_std['informed_std'] = (panel_std['informed_share'] - panel_std['informed_share'].mean()) / panel_std['informed_share'].std()
panel_std['spread_lag1_std'] = panel_std.groupby('asset')['spread_std'].shift(1)
panel_std['informed_lag1_std'] = panel_std.groupby('asset')['informed_std'].shift(1)

panel_std = panel_std.dropna(subset=['spread_lag1_std', 'informed_lag1_std'])

# Forward: lagged informed → spread
X_fwd = sm.add_constant(panel_std[['spread_lag1_std', 'informed_lag1_std']])
y_fwd = panel_std['spread_std']
model_fwd = OLS(y_fwd, X_fwd).fit(cov_type='HC1')

# Reverse: lagged spread → informed
X_rev = sm.add_constant(panel_std[['informed_lag1_std', 'spread_lag1_std']])
y_rev = panel_std['informed_std']
model_rev = OLS(y_rev, X_rev).fit(cov_type='HC1')

print(f"\nStandardized coefficients (for comparability):")
print(f"    Informed_{{t-1}} -> Spread_t:    β = {model_fwd.params['informed_lag1_std']:.4f}, t = {model_fwd.tvalues['informed_lag1_std']:.2f}")
print(f"    Spread_{{t-1}} -> Informed_t:    β = {model_rev.params['spread_lag1_std']:.4f}, t = {model_rev.tvalues['spread_lag1_std']:.2f}")
print(f"\n    Ratio (forward/reverse): {abs(model_fwd.params['informed_lag1_std'] / model_rev.params['spread_lag1_std']):.2f}x")

results['asymmetry'] = {
    'forward_coef': model_fwd.params['informed_lag1_std'],
    'forward_tstat': model_fwd.tvalues['informed_lag1_std'],
    'reverse_coef': model_rev.params['spread_lag1_std'],
    'reverse_tstat': model_rev.tvalues['spread_lag1_std'],
    'ratio': abs(model_fwd.params['informed_lag1_std'] / model_rev.params['spread_lag1_std'])
}

# ============================================================================
# STEP 7: EVENT-TIME DECOMPOSITION (Within Outage Window)
# ============================================================================
print("\n" + "="*70)
print("[7] Event-Time Decomposition: Composition vs Spread Timing")
print("="*70)

# Focus on outage day, hour 14 (the outage hour)
outage_hour = panel[
    (panel['date'] == pd.Timestamp('2025-07-29').date()) &
    (panel['hour'] == 14)
].copy()

print(f"\n    Observations in outage hour: {len(outage_hour):,}")

# Create 5-minute bins within the hour
outage_hour['minute_bin'] = (outage_hour['minute_of_hour'] // 5) * 5

# Aggregate by 5-minute bin (across assets)
event_time = outage_hour.groupby('minute_bin').agg({
    'spread_bps': 'mean',
    'informed_share': 'mean',
    'quote_updates': 'mean',
    'n_takers': 'sum'
}).reset_index()

print("\n    5-Minute Bins Within Outage Hour (14:00 UTC):")
print("    " + "-"*60)
print(f"    {'Minute':<10} {'Spread(bps)':<15} {'Informed%':<15} {'QuoteUpd':<12}")
print("    " + "-"*60)

for _, row in event_time.iterrows():
    informed_pct = row['informed_share'] * 100 if pd.notna(row['informed_share']) else 0
    print(f"    {int(row['minute_bin']):02d}-{int(row['minute_bin'])+5:02d}      {row['spread_bps']:.2f}           {informed_pct:.1f}%           {row['quote_updates']:.0f}")

# Compute changes from pre-outage baseline (minute 0-5)
baseline = event_time[event_time['minute_bin'] == 0].iloc[0] if len(event_time[event_time['minute_bin'] == 0]) > 0 else event_time.iloc[0]

event_time['spread_change_from_baseline'] = event_time['spread_bps'] - baseline['spread_bps']
event_time['informed_change_from_baseline'] = event_time['informed_share'] - baseline['informed_share']

print("\n    Changes from Pre-Outage (00-05) Baseline:")
print("    " + "-"*60)
print(f"    {'Minute':<10} {'ΔSpread(bps)':<15} {'ΔInformed%':<15}")
print("    " + "-"*60)

for _, row in event_time.iterrows():
    informed_change = row['informed_change_from_baseline'] * 100 if pd.notna(row['informed_change_from_baseline']) else 0
    print(f"    {int(row['minute_bin']):02d}-{int(row['minute_bin'])+5:02d}      {row['spread_change_from_baseline']:+.2f}          {informed_change:+.1f}pp")

# Test: Does informed share rise BEFORE spreads peak?
# Outage started ~14:10, so we look at minutes 10-15 vs 15-20 vs 20-25
print("\n    Timing Test: When do composition shifts and spread widening occur?")

pre_outage = event_time[event_time['minute_bin'] < 10]
early_outage = event_time[(event_time['minute_bin'] >= 10) & (event_time['minute_bin'] < 20)]
mid_outage = event_time[(event_time['minute_bin'] >= 20) & (event_time['minute_bin'] < 35)]
late_outage = event_time[(event_time['minute_bin'] >= 35) & (event_time['minute_bin'] < 50)]
post_outage = event_time[event_time['minute_bin'] >= 50]

print(f"\n    Pre-outage (00-09):  Spread = {pre_outage['spread_bps'].mean():.2f}, Informed = {pre_outage['informed_share'].mean()*100:.1f}%")
print(f"    Early outage (10-19): Spread = {early_outage['spread_bps'].mean():.2f}, Informed = {early_outage['informed_share'].mean()*100:.1f}%")
print(f"    Mid outage (20-34):   Spread = {mid_outage['spread_bps'].mean():.2f}, Informed = {mid_outage['informed_share'].mean()*100:.1f}%")
print(f"    Late outage (35-49):  Spread = {late_outage['spread_bps'].mean():.2f}, Informed = {late_outage['informed_share'].mean()*100:.1f}%")
if len(post_outage) > 0:
    print(f"    Post-outage (50-59):  Spread = {post_outage['spread_bps'].mean():.2f}, Informed = {post_outage['informed_share'].mean()*100:.1f}%")

# ============================================================================
# STEP 8: FORMAL GRANGER CAUSALITY TEST
# ============================================================================
print("\n" + "="*70)
print("[8] Formal Granger Causality Tests")
print("="*70)

from statsmodels.stats.stattools import durbin_watson

# Granger test: Does informed share Granger-cause spreads?
# H0: Lags of informed share do not help predict spreads beyond lags of spreads

# Restricted model: Spread = f(lagged spreads only)
X_restricted = sm.add_constant(panel_clean[['spread_lag1', 'spread_lag2', 'spread_lag3']])
y_granger = panel_clean['spread_bps']
X_restricted = X_restricted.dropna()
y_granger_clean = y_granger.loc[X_restricted.index]

model_restricted = OLS(y_granger_clean, X_restricted).fit()

# Unrestricted model: Spread = f(lagged spreads + lagged informed share)
X_unrestricted = panel_clean[['spread_lag1', 'spread_lag2', 'spread_lag3',
                               'informed_share_lag1', 'informed_share_lag2', 'informed_share_lag3']]
X_unrestricted = sm.add_constant(X_unrestricted)
X_unrestricted = X_unrestricted.dropna()
y_granger_clean2 = y_granger.loc[X_unrestricted.index]

model_unrestricted = OLS(y_granger_clean2, X_unrestricted).fit()

# F-test
n = len(y_granger_clean2)
k_r = len(model_restricted.params)
k_u = len(model_unrestricted.params)
ssr_r = model_restricted.ssr
ssr_u = model_unrestricted.ssr

f_stat = ((ssr_r - ssr_u) / (k_u - k_r)) / (ssr_u / (n - k_u))
f_pval = 1 - stats.f.cdf(f_stat, k_u - k_r, n - k_u)

print(f"\n    Granger Test: Informed Share → Spread")
print(f"    H0: Lags of informed share do not predict spreads")
print(f"    F-statistic = {f_stat:.3f}")
print(f"    p-value = {f_pval:.4f}")
print(f"    Conclusion: {'Reject H0 - Informed share Granger-causes spreads' if f_pval < 0.05 else 'Cannot reject H0'}")

results['granger_informed_to_spread'] = {
    'f_stat': f_stat,
    'pval': f_pval,
    'reject_h0': f_pval < 0.05
}

# Reverse Granger test: Do spreads Granger-cause informed share?
X_restricted_rev = sm.add_constant(panel_clean[['informed_share_lag1', 'informed_share_lag2', 'informed_share_lag3']])
y_granger_rev = panel_clean['informed_share']
X_restricted_rev = X_restricted_rev.dropna()
y_granger_rev_clean = y_granger_rev.loc[X_restricted_rev.index]

model_restricted_rev = OLS(y_granger_rev_clean, X_restricted_rev).fit()

X_unrestricted_rev = panel_clean[['informed_share_lag1', 'informed_share_lag2', 'informed_share_lag3',
                                   'spread_lag1', 'spread_lag2', 'spread_lag3']]
X_unrestricted_rev = sm.add_constant(X_unrestricted_rev)
X_unrestricted_rev = X_unrestricted_rev.dropna()
y_granger_rev_clean2 = y_granger_rev.loc[X_unrestricted_rev.index]

model_unrestricted_rev = OLS(y_granger_rev_clean2, X_unrestricted_rev).fit()

n_rev = len(y_granger_rev_clean2)
ssr_r_rev = model_restricted_rev.ssr
ssr_u_rev = model_unrestricted_rev.ssr

f_stat_rev = ((ssr_r_rev - ssr_u_rev) / (k_u - k_r)) / (ssr_u_rev / (n_rev - k_u))
f_pval_rev = 1 - stats.f.cdf(f_stat_rev, k_u - k_r, n_rev - k_u)

print(f"\n    Granger Test: Spread → Informed Share (Reverse)")
print(f"    H0: Lags of spread do not predict informed share")
print(f"    F-statistic = {f_stat_rev:.3f}")
print(f"    p-value = {f_pval_rev:.4f}")
print(f"    Conclusion: {'Reject H0 - Spread Granger-causes informed share' if f_pval_rev < 0.05 else 'Cannot reject H0'}")

results['granger_spread_to_informed'] = {
    'f_stat': f_stat_rev,
    'pval': f_pval_rev,
    'reject_h0': f_pval_rev < 0.05
}

# ============================================================================
# STEP 9: CUMULATIVE LAG COEFFICIENTS
# ============================================================================
print("\n" + "="*70)
print("[9] Cumulative Lag Analysis")
print("="*70)

# Test cumulative effect of informed share lags on spreads
cumulative_results = []

for max_lag in range(1, 6):
    lag_cols = [f'informed_share_lag{i}' for i in range(1, max_lag+1)]
    spread_lag_cols = [f'spread_lag{i}' for i in range(1, max_lag+1)]

    X_cum = panel_clean[spread_lag_cols + lag_cols].copy()
    X_cum = sm.add_constant(X_cum)
    X_cum = X_cum.dropna()
    y_cum = panel_clean.loc[X_cum.index, 'spread_bps']

    model_cum = OLS(y_cum, X_cum).fit(cov_type='HC1')

    # Sum of informed share coefficients
    informed_coefs = [model_cum.params[col] for col in lag_cols]
    cumulative_coef = sum(informed_coefs)

    # Joint significance (Wald test)
    r_matrix = np.zeros((len(lag_cols), len(model_cum.params)))
    for i, col in enumerate(lag_cols):
        r_matrix[i, list(model_cum.params.index).index(col)] = 1

    wald_test = model_cum.wald_test(r_matrix)

    cumulative_results.append({
        'max_lag': max_lag,
        'cumulative_coef': cumulative_coef,
        'wald_stat': wald_test.statistic,
        'wald_pval': wald_test.pvalue
    })

    print(f"    Lags 1-{max_lag}: Cumulative β = {cumulative_coef:.4f}, Wald χ² = {float(wald_test.statistic):.2f}, p = {float(wald_test.pvalue):.4f}")

# ============================================================================
# STEP 10: SUMMARY TABLE FOR PAPER
# ============================================================================
print("\n" + "="*70)
print("[10] SUMMARY: Lag Structure Evidence")
print("="*70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    LAG STRUCTURE ANALYSIS RESULTS                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  TEST 1: Does lagged informed share predict spreads?                  ║
║  ─────────────────────────────────────────────────────────────────    ║""")
print(f"║  Coefficient (informed_share_lag1):  {results['model_1a']['coef']:>8.4f}                       ║")
print(f"║  t-statistic:                         {results['model_1a']['tstat']:>8.2f}                       ║")
print(f"║  p-value:                             {results['model_1a']['pval']:>8.4f}                       ║")
print("""║                                                                       ║
║  TEST 2: Does lagged spread predict informed share? (Reverse)         ║
║  ─────────────────────────────────────────────────────────────────    ║""")
print(f"║  Coefficient (spread_lag1):           {results['model_2a']['coef']:>8.6f}                     ║")
print(f"║  t-statistic:                         {results['model_2a']['tstat']:>8.2f}                       ║")
print(f"║  p-value:                             {results['model_2a']['pval']:>8.4f}                       ║")
print("""║                                                                       ║
║  TEST 3: Asymmetry (Standardized Coefficients)                        ║
║  ─────────────────────────────────────────────────────────────────    ║""")
print(f"║  Forward (Informed→Spread):           {results['asymmetry']['forward_coef']:>8.4f} (t={results['asymmetry']['forward_tstat']:.2f})           ║")
print(f"║  Reverse (Spread→Informed):           {results['asymmetry']['reverse_coef']:>8.4f} (t={results['asymmetry']['reverse_tstat']:.2f})           ║")
print(f"║  Ratio (Forward/Reverse):             {results['asymmetry']['ratio']:>8.2f}x                       ║")
print("""║                                                                       ║
║  TEST 4: Granger Causality                                            ║
║  ─────────────────────────────────────────────────────────────────    ║""")
print(f"║  Informed → Spread: F = {results['granger_informed_to_spread']['f_stat']:.2f}, p = {results['granger_informed_to_spread']['pval']:.4f} {'(SIGNIFICANT)' if results['granger_informed_to_spread']['reject_h0'] else '(not sig.)'}     ║")
print(f"║  Spread → Informed: F = {results['granger_spread_to_informed']['f_stat']:.2f}, p = {results['granger_spread_to_informed']['pval']:.4f} {'(SIGNIFICANT)' if results['granger_spread_to_informed']['reject_h0'] else '(not sig.)'}     ║")
print("""║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# STEP 11: SAVE RESULTS
# ============================================================================
print("\n[11] Saving results...")

# Save panel data
panel_clean.to_csv(RESULTS_DIR / "lag_analysis_panel.csv", index=False)
print(f"    Saved panel to: {RESULTS_DIR / 'lag_analysis_panel.csv'}")

# Save summary results
import json
with open(RESULTS_DIR / "lag_analysis_results.json", 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    results_json = {}
    for key, val in results.items():
        results_json[key] = {}
        for k, v in val.items():
            if isinstance(v, (np.floating, np.integer)):
                results_json[key][k] = float(v)
            elif isinstance(v, np.bool_):
                results_json[key][k] = bool(v)
            else:
                results_json[key][k] = v
    json.dump(results_json, f, indent=2)
print(f"    Saved results to: {RESULTS_DIR / 'lag_analysis_results.json'}")

# Save event-time data
event_time.to_csv(RESULTS_DIR / "event_time_decomposition.csv", index=False)
print(f"    Saved event-time to: {RESULTS_DIR / 'event_time_decomposition.csv'}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
