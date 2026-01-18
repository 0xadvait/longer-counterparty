#!/usr/bin/env python3
"""
API Outage Natural Experiment Analysis
=======================================

The July 29, 2025 API outage (14:10-14:47 UTC) provides an exogenous shock
to the ability to cancel/update orders. This directly tests the mechanism
underlying market quality: maker's ability to manage quotes.

Identification Strategy:
-----------------------
1. Event Study: Compare outage vs normal periods
2. Triple-Diff: Use pre-outage "exposure" (quote update intensity) as heterogeneity
   Y_{i,t} = α_i + δ_t + β(Outage_t × Exposure_i) + ΓX_{i,t} + ε_{i,t}

Author: Boyi Shen, London Business School
"""

import boto3
import lz4.frame
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===
warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Configuration
AWS_CONFIG = {
    'region_name': 'us-east-1',
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', '')
}
S3_BUCKET = 'hyperliquid-research-bunc'

DATA_DIR = _DATA_DIR
OUTPUT_DIR = _RESULTS_DIR
FIGURES_DIR = OUTPUT_DIR + 'figures/'

# Outage timing
OUTAGE_START = datetime(2025, 7, 29, 14, 10)
OUTAGE_END = datetime(2025, 7, 29, 14, 47)

# Assets to analyze (major ones with good liquidity)
ASSETS = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'XRP', 'AVAX', 'LINK', 'OP', 'SUI',
          'APT', 'MATIC', 'NEAR', 'ATOM', 'FIL', 'LTC', 'UNI', 'AAVE', 'MKR', 'SNX']

# Dates for analysis
OUTAGE_DATE = '2025-07-29'
CONTROL_DATES = ['2025-07-28', '2025-07-30']  # Day before and after for comparison
PLACEBO_DATES = ['2025-07-27', '2025-07-31']  # For placebo tests

print("=" * 80)
print("API OUTAGE NATURAL EXPERIMENT ANALYSIS")
print("=" * 80)
print(f"Outage window: {OUTAGE_START} to {OUTAGE_END} UTC")
print(f"Analyzing {len(ASSETS)} assets")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def download_and_process_hour(s3, asset, date, hour):
    """Download and process one hour of L2 book data."""
    key = f"raw/l2_books/{asset}/{date}/{hour}.lz4"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=key)
        compressed = response['Body'].read()
        decompressed = lz4.frame.decompress(compressed)

        records = []
        lines = decompressed.decode('utf-8').strip().split('\n')

        prev_mid = None
        prev_spread = None

        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
                data = record.get('raw', {}).get('data', {})
                levels = data.get('levels', [])

                if len(levels) < 2:
                    continue

                bids, asks = levels[0], levels[1]
                if not bids or not asks:
                    continue

                best_bid = float(bids[0]['px'])
                best_ask = float(asks[0]['px'])
                mid = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                spread_bps = spread / mid * 10000

                bid_depth = sum(float(b['sz']) * float(b['px']) for b in bids[:5])
                ask_depth = sum(float(a['sz']) * float(a['px']) for a in asks[:5])
                total_depth = bid_depth + ask_depth
                imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

                # Track quote changes (proxy for update activity)
                mid_changed = 1 if prev_mid is not None and mid != prev_mid else 0
                spread_changed = 1 if prev_spread is not None and spread != prev_spread else 0

                prev_mid = mid
                prev_spread = spread

                ts_ms = data.get('time')
                if ts_ms:
                    records.append({
                        'asset': asset,
                        'time_ms': ts_ms,
                        'mid': mid,
                        'spread_bps': spread_bps,
                        'bid_depth': bid_depth,
                        'ask_depth': ask_depth,
                        'total_depth': total_depth,
                        'imbalance': imbalance,
                        'mid_changed': mid_changed,
                        'spread_changed': spread_changed,
                    })
            except:
                continue

        return records
    except Exception as e:
        return []


def load_data_for_date(date, assets=ASSETS, hours=range(10, 24)):
    """Load data for all assets on a given date (parallelized)."""
    s3 = boto3.client('s3', **AWS_CONFIG)

    # Build list of all (asset, hour) combinations
    tasks = [(asset, date, hour) for asset in assets for hour in hours]

    print(f"  Downloading {len(tasks)} hour-files for {date}...")

    all_records = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_and_process_hour, s3, asset, date, hour): (asset, hour)
                   for asset, date, hour in tasks}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            records = future.result()
            all_records.extend(records)
            if completed % 50 == 0:
                print(f"    Progress: {completed}/{len(tasks)}")

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    df['time'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['date'] = date
    df = df.sort_values(['asset', 'time']).reset_index(drop=True)

    return df


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Check for cached data
cache_file = DATA_DIR + 'outage_analysis_data.parquet'

if os.path.exists(cache_file):
    print("\nLoading cached data...")
    df_all = pd.read_parquet(cache_file)
    print(f"  Loaded {len(df_all):,} observations")
else:
    print("\nDownloading data from S3...")

    dfs = []

    # Load outage date
    print(f"\nLoading outage date: {OUTAGE_DATE}")
    df_outage = load_data_for_date(OUTAGE_DATE)
    if df_outage is not None:
        dfs.append(df_outage)
        print(f"  Loaded {len(df_outage):,} observations")

    # Load control dates
    for date in CONTROL_DATES:
        print(f"\nLoading control date: {date}")
        df_ctrl = load_data_for_date(date)
        if df_ctrl is not None:
            dfs.append(df_ctrl)
            print(f"  Loaded {len(df_ctrl):,} observations")

    # Load placebo dates
    for date in PLACEBO_DATES:
        print(f"\nLoading placebo date: {date}")
        df_placebo = load_data_for_date(date)
        if df_placebo is not None:
            dfs.append(df_placebo)
            print(f"  Loaded {len(df_placebo):,} observations")

    if not dfs:
        print("ERROR: No data loaded!")
        exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_parquet(cache_file)
    print(f"\nCached {len(df_all):,} observations to {cache_file}")

# =============================================================================
# CREATE ANALYSIS VARIABLES
# =============================================================================

print("\n" + "=" * 80)
print("CREATING ANALYSIS VARIABLES")
print("=" * 80)

# Add time-based variables
df_all['hour'] = df_all['time'].dt.hour
df_all['minute'] = df_all['time'].dt.minute

# Define outage indicator
df_all['outage'] = (
    (df_all['date'] == OUTAGE_DATE) &
    (df_all['time'] >= OUTAGE_START) &
    (df_all['time'] <= OUTAGE_END)
).astype(int)

# Define outage hour (broader)
df_all['outage_hour'] = (
    (df_all['date'] == OUTAGE_DATE) &
    (df_all['hour'] == 14)
).astype(int)

# Define pre/post outage on outage day
df_all['pre_outage'] = (
    (df_all['date'] == OUTAGE_DATE) &
    (df_all['hour'] < 14)
).astype(int)

df_all['post_outage'] = (
    (df_all['date'] == OUTAGE_DATE) &
    (df_all['hour'] > 14)
).astype(int)

print(f"Observations in outage window: {df_all['outage'].sum():,}")
print(f"Observations in outage hour: {df_all['outage_hour'].sum():,}")

# =============================================================================
# COMPUTE PRE-OUTAGE EXPOSURE METRICS
# =============================================================================

print("\n" + "=" * 80)
print("COMPUTING PRE-OUTAGE EXPOSURE METRICS")
print("=" * 80)

# Use July 28 (day before) to compute exposure
df_preoutage = df_all[df_all['date'] == '2025-07-28'].copy()

if len(df_preoutage) > 0:
    # Compute quote update intensity per asset
    exposure_metrics = df_preoutage.groupby('asset').agg({
        'mid_changed': 'mean',      # Fraction of snapshots with mid change
        'spread_changed': 'mean',   # Fraction of snapshots with spread change
        'spread_bps': 'median',
        'total_depth': 'median',
    }).reset_index()

    exposure_metrics.columns = ['asset', 'mid_update_rate', 'spread_update_rate',
                                 'pre_spread', 'pre_depth']

    # Create composite exposure measure
    exposure_metrics['exposure'] = (
        exposure_metrics['mid_update_rate'] + exposure_metrics['spread_update_rate']
    ) / 2

    # Standardize
    exposure_metrics['exposure_std'] = (
        (exposure_metrics['exposure'] - exposure_metrics['exposure'].mean()) /
        exposure_metrics['exposure'].std()
    )

    print("\nExposure Metrics (pre-outage quote update intensity):")
    print(exposure_metrics.sort_values('exposure', ascending=False).head(10).to_string())

    # Merge back to main data
    df_all = df_all.merge(exposure_metrics[['asset', 'exposure', 'exposure_std']],
                          on='asset', how='left')
else:
    print("WARNING: No pre-outage data found!")
    df_all['exposure'] = 0.5
    df_all['exposure_std'] = 0

# =============================================================================
# AGGREGATE TO MINUTE-LEVEL FOR EVENT STUDY
# =============================================================================

print("\n" + "=" * 80)
print("AGGREGATING TO MINUTE-LEVEL")
print("=" * 80)

df_all['minute_bin'] = df_all['time'].dt.floor('1min')

df_minute = df_all.groupby(['asset', 'date', 'minute_bin']).agg({
    'spread_bps': 'median',
    'total_depth': 'median',
    'imbalance': lambda x: x.abs().median(),
    'mid_changed': 'sum',  # Number of mid changes in minute
    'spread_changed': 'sum',
    'outage': 'max',
    'outage_hour': 'max',
    'exposure': 'first',
    'exposure_std': 'first',
}).reset_index()

df_minute['hour'] = df_minute['minute_bin'].dt.hour
df_minute['minute'] = df_minute['minute_bin'].dt.minute

# Rename count columns
df_minute = df_minute.rename(columns={
    'mid_changed': 'mid_changes',
    'spread_changed': 'spread_changes'
})

print(f"Minute-level observations: {len(df_minute):,}")

# =============================================================================
# EVENT STUDY: COMPARE OUTAGE VS NORMAL HOURS
# =============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY: OUTAGE VS NORMAL HOURS")
print("=" * 80)

# Focus on outage day
df_outage_day = df_minute[df_minute['date'] == OUTAGE_DATE].copy()

# Compare hour 14 (outage) vs hours 13 and 15 (adjacent)
df_outage_day['period'] = 'other'
df_outage_day.loc[df_outage_day['hour'] == 13, 'period'] = 'pre'
df_outage_day.loc[df_outage_day['hour'] == 14, 'period'] = 'outage'
df_outage_day.loc[df_outage_day['hour'] == 15, 'period'] = 'post'

# Summary statistics by period
print("\nSummary by Period (Outage Day):")
summary = df_outage_day.groupby('period').agg({
    'spread_bps': ['mean', 'std'],
    'total_depth': ['mean', 'std'],
    'mid_changes': ['mean', 'std'],
}).round(3)
print(summary)

# Test differences
df_pre = df_outage_day[df_outage_day['period'] == 'pre']
df_out = df_outage_day[df_outage_day['period'] == 'outage']
df_post = df_outage_day[df_outage_day['period'] == 'post']

if len(df_pre) > 0 and len(df_out) > 0:
    from scipy import stats

    print("\n\nT-tests: Outage vs Pre-outage Hour:")
    for var in ['spread_bps', 'total_depth', 'mid_changes']:
        t_stat, p_val = stats.ttest_ind(df_out[var].dropna(), df_pre[var].dropna())
        diff = df_out[var].mean() - df_pre[var].mean()
        print(f"  {var}: diff = {diff:+.3f}, t = {t_stat:.2f}, p = {p_val:.4f}")

# =============================================================================
# TRIPLE-DIFF: OUTAGE × EXPOSURE
# =============================================================================

print("\n" + "=" * 80)
print("TRIPLE-DIFF REGRESSION: OUTAGE × EXPOSURE")
print("=" * 80)

# Prepare regression data
df_reg = df_minute.copy()

# Add fixed effects
df_reg['asset_fe'] = pd.Categorical(df_reg['asset']).codes
df_reg['date_fe'] = pd.Categorical(df_reg['date']).codes
df_reg['hour_fe'] = df_reg['hour']

# Create interaction
df_reg['outage_x_exposure'] = df_reg['outage_hour'] * df_reg['exposure_std']

# Run regressions
outcomes = ['spread_bps', 'total_depth', 'mid_changes']
results = {}

for outcome in outcomes:
    # Clean data
    df_clean = df_reg.dropna(subset=[outcome, 'exposure_std', 'outage_hour'])

    if len(df_clean) < 100:
        continue

    # Model: Y = α_asset + δ_hour + β₁·Outage + β₂·Outage×Exposure + ε
    # Using demeaned approach for fixed effects

    # Add dummies for asset and hour
    asset_dummies = pd.get_dummies(df_clean['asset'], prefix='asset', drop_first=True)
    hour_dummies = pd.get_dummies(df_clean['hour'], prefix='hour', drop_first=True)

    X = pd.concat([
        df_clean[['outage_hour', 'outage_x_exposure']],
        asset_dummies,
        hour_dummies
    ], axis=1)
    X = sm.add_constant(X)

    y = df_clean[outcome]

    model = OLS(y.astype(float), X.astype(float)).fit(cov_type='HC1')
    results[outcome] = model

    print(f"\n{outcome.upper()}:")
    print(f"  Outage Hour (β₁):         {model.params['outage_hour']:+.4f} (t = {model.tvalues['outage_hour']:.2f})")
    print(f"  Outage × Exposure (β₂):   {model.params['outage_x_exposure']:+.4f} (t = {model.tvalues['outage_x_exposure']:.2f})")
    print(f"  R²: {model.rsquared:.3f}, N = {len(df_clean):,}")

# =============================================================================
# PLACEBO TEST: SAME HOUR ON OTHER DAYS
# =============================================================================

print("\n" + "=" * 80)
print("PLACEBO TEST: HOUR 14 ON OTHER DAYS")
print("=" * 80)

placebo_results = []

for date in CONTROL_DATES + PLACEBO_DATES:
    df_placebo = df_minute[df_minute['date'] == date].copy()
    df_placebo['fake_outage'] = (df_placebo['hour'] == 14).astype(int)
    df_placebo['fake_outage_x_exposure'] = df_placebo['fake_outage'] * df_placebo['exposure_std']

    df_clean = df_placebo.dropna(subset=['spread_bps', 'exposure_std'])

    if len(df_clean) < 50:
        continue

    # Simple regression
    asset_dummies = pd.get_dummies(df_clean['asset'], prefix='asset', drop_first=True)
    hour_dummies = pd.get_dummies(df_clean['hour'], prefix='hour', drop_first=True)

    X = pd.concat([
        df_clean[['fake_outage', 'fake_outage_x_exposure']],
        asset_dummies,
        hour_dummies
    ], axis=1)
    X = sm.add_constant(X)

    y = df_clean['spread_bps']

    try:
        model = OLS(y.astype(float), X.astype(float)).fit(cov_type='HC1')
        placebo_results.append({
            'date': date,
            'beta_outage': model.params.get('fake_outage', np.nan),
            't_outage': model.tvalues.get('fake_outage', np.nan),
            'beta_interaction': model.params.get('fake_outage_x_exposure', np.nan),
            't_interaction': model.tvalues.get('fake_outage_x_exposure', np.nan),
        })
        print(f"  {date}: β_outage = {model.params.get('fake_outage', np.nan):+.4f} (t = {model.tvalues.get('fake_outage', np.nan):.2f})")
    except:
        pass

# =============================================================================
# FIGURES
# =============================================================================

print("\n" + "=" * 80)
print("CREATING FIGURES")
print("=" * 80)

# Figure 1: Event Study - Spread and Depth around Outage
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Spread over time on outage day
ax1 = axes[0, 0]
df_plot = df_minute[df_minute['date'] == OUTAGE_DATE].copy()
df_plot_agg = df_plot.groupby('minute_bin').agg({'spread_bps': 'mean'}).reset_index()

ax1.plot(df_plot_agg['minute_bin'], df_plot_agg['spread_bps'], 'b-', linewidth=1)
ax1.axvspan(OUTAGE_START, OUTAGE_END, alpha=0.3, color='red', label='API Outage')
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Spread (bps)')
ax1.set_title('A. Average Spread on July 29, 2025')
ax1.legend()

# Panel B: Quote Updates over time
ax2 = axes[0, 1]
df_plot_agg2 = df_plot.groupby('minute_bin').agg({'mid_changes': 'mean'}).reset_index()

ax2.plot(df_plot_agg2['minute_bin'], df_plot_agg2['mid_changes'], 'g-', linewidth=1)
ax2.axvspan(OUTAGE_START, OUTAGE_END, alpha=0.3, color='red', label='API Outage')
ax2.set_xlabel('Time (UTC)')
ax2.set_ylabel('Mid-Price Changes per Minute')
ax2.set_title('B. Quote Update Intensity')
ax2.legend()

# Panel C: High vs Low Exposure during outage
ax3 = axes[1, 0]

# Split by exposure
if 'exposure_std' in df_plot.columns:
    df_high = df_plot[df_plot['exposure_std'] > 0]
    df_low = df_plot[df_plot['exposure_std'] <= 0]

    if len(df_high) > 0 and len(df_low) > 0:
        df_high_agg = df_high.groupby('minute_bin').agg({'spread_bps': 'mean'}).reset_index()
        df_low_agg = df_low.groupby('minute_bin').agg({'spread_bps': 'mean'}).reset_index()

        ax3.plot(df_high_agg['minute_bin'], df_high_agg['spread_bps'], 'r-', linewidth=1, label='High Exposure')
        ax3.plot(df_low_agg['minute_bin'], df_low_agg['spread_bps'], 'b-', linewidth=1, label='Low Exposure')
        ax3.axvspan(OUTAGE_START, OUTAGE_END, alpha=0.3, color='gray')
        ax3.set_xlabel('Time (UTC)')
        ax3.set_ylabel('Spread (bps)')
        ax3.set_title('C. Spread by Pre-Outage Quote Update Intensity')
        ax3.legend()

# Panel D: Difference-in-differences
ax4 = axes[1, 1]

# Compare outage day vs control days at hour 14
hour14_data = []
for date in [OUTAGE_DATE] + CONTROL_DATES:
    df_h14 = df_minute[(df_minute['date'] == date) & (df_minute['hour'] == 14)]
    if len(df_h14) > 0:
        hour14_data.append({
            'date': date,
            'spread_bps': df_h14['spread_bps'].mean(),
            'is_outage': 1 if date == OUTAGE_DATE else 0
        })

if hour14_data:
    hour14_df = pd.DataFrame(hour14_data)
    colors = ['red' if x == 1 else 'blue' for x in hour14_df['is_outage']]
    ax4.bar(hour14_df['date'], hour14_df['spread_bps'], color=colors)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Average Spread at Hour 14 (bps)')
    ax4.set_title('D. Hour 14 Spread: Outage Day vs Control Days')
    ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(FIGURES_DIR + 'figure_outage_event_study.pdf', bbox_inches='tight')
plt.savefig(FIGURES_DIR + 'figure_outage_event_study.png', dpi=300, bbox_inches='tight')
print("Saved: figure_outage_event_study.pdf")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save regression coefficients
reg_results = []
for outcome, model in results.items():
    reg_results.append({
        'outcome': outcome,
        'beta_outage': model.params.get('outage_hour', np.nan),
        'se_outage': model.bse.get('outage_hour', np.nan),
        't_outage': model.tvalues.get('outage_hour', np.nan),
        'beta_interaction': model.params.get('outage_x_exposure', np.nan),
        'se_interaction': model.bse.get('outage_x_exposure', np.nan),
        't_interaction': model.tvalues.get('outage_x_exposure', np.nan),
        'r2': model.rsquared,
        'n_obs': model.nobs,
    })

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(DATA_DIR + 'outage_regression_results.csv', index=False)
print(f"Saved regression results to: outage_regression_results.csv")

# =============================================================================
# LATEX TABLE
# =============================================================================

print("\n" + "=" * 80)
print("LATEX TABLE")
print("=" * 80)

latex_table = r"""
\begin{table}[H]
\centering
\caption{API Outage Natural Experiment: Triple-Difference Results}
\label{tab:outage}
\small
\begin{tabular}{lccc}
\toprule
& (1) Spread & (2) Depth & (3) Quote Updates \\
& (bps) & (\$) & (per min) \\
\midrule
"""

for outcome in outcomes:
    if outcome in results:
        m = results[outcome]
        b1 = m.params.get('outage_hour', np.nan)
        t1 = m.tvalues.get('outage_hour', np.nan)
        b2 = m.params.get('outage_x_exposure', np.nan)
        t2 = m.tvalues.get('outage_x_exposure', np.nan)

        sig1 = '***' if abs(t1) > 2.58 else '**' if abs(t1) > 1.96 else '*' if abs(t1) > 1.65 else ''
        sig2 = '***' if abs(t2) > 2.58 else '**' if abs(t2) > 1.96 else '*' if abs(t2) > 1.65 else ''

if 'spread_bps' in results and 'total_depth' in results and 'mid_changes' in results:
    m1, m2, m3 = results['spread_bps'], results['total_depth'], results['mid_changes']

    latex_table += f"Outage Hour & {m1.params.get('outage_hour', 0):.3f}{'***' if abs(m1.tvalues.get('outage_hour', 0)) > 2.58 else '**' if abs(m1.tvalues.get('outage_hour', 0)) > 1.96 else ''} & {m2.params.get('outage_hour', 0):.0f} & {m3.params.get('outage_hour', 0):.1f} \\\\\n"
    latex_table += f"& ({m1.tvalues.get('outage_hour', 0):.2f}) & ({m2.tvalues.get('outage_hour', 0):.2f}) & ({m3.tvalues.get('outage_hour', 0):.2f}) \\\\\n"
    latex_table += f"Outage $\\times$ Exposure & {m1.params.get('outage_x_exposure', 0):.3f} & {m2.params.get('outage_x_exposure', 0):.0f} & {m3.params.get('outage_x_exposure', 0):.1f} \\\\\n"
    latex_table += f"& ({m1.tvalues.get('outage_x_exposure', 0):.2f}) & ({m2.tvalues.get('outage_x_exposure', 0):.2f}) & ({m3.tvalues.get('outage_x_exposure', 0):.2f}) \\\\\n"

latex_table += r"""\midrule
Asset FE & Yes & Yes & Yes \\
Hour FE & Yes & Yes & Yes \\
\bottomrule
\multicolumn{4}{l}{\footnotesize $t$-statistics in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$.} \\
\multicolumn{4}{l}{\footnotesize Outage: July 29, 2025, 14:10-14:47 UTC. Exposure: pre-outage quote update intensity.}
\end{tabular}
\end{table}
"""

print(latex_table)

with open(OUTPUT_DIR + 'table_outage.tex', 'w') as f:
    f.write(latex_table)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
API OUTAGE NATURAL EXPERIMENT: SUMMARY
================================================================================

EVENT:
- July 29, 2025, 14:10-14:47 UTC
- API server issues caused order delays/failures
- Exogenous shock to cancel/update ability

DATA:
- {len(df_all):,} raw observations
- {len(df_minute):,} minute-level observations
- {len(ASSETS)} assets analyzed
- Dates: {OUTAGE_DATE} (outage) + {len(CONTROL_DATES)} control + {len(PLACEBO_DATES)} placebo

KEY FINDINGS:
""")

if 'spread_bps' in results:
    m = results['spread_bps']
    print(f"1. Spread during outage: {m.params.get('outage_hour', 0):+.3f} bps (t = {m.tvalues.get('outage_hour', 0):.2f})")
    print(f"   Outage × Exposure:    {m.params.get('outage_x_exposure', 0):+.3f} bps (t = {m.tvalues.get('outage_x_exposure', 0):.2f})")

if 'mid_changes' in results:
    m = results['mid_changes']
    print(f"\n2. Quote updates during outage: {m.params.get('outage_hour', 0):+.1f} per min (t = {m.tvalues.get('outage_hour', 0):.2f})")

print("""
INTERPRETATION:
- The outage reduced makers' ability to update/cancel orders
- This allows testing whether speed/latency causally affects market quality
- High-exposure assets (more reliant on frequent quoting) should suffer more

================================================================================
""")

print("\nAnalysis complete!")
