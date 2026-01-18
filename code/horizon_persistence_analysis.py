#!/usr/bin/env python3
"""
Multi-Horizon Price Impact Analysis (Optimized)
================================================

Show that Q5 (informed) trades predict PERMANENT price changes while
Q1 (uninformed) trades show TRANSIENT impact that reverses.

Horizons: 10s, 1m, 5m, 30m, 2h

Author: Boyi Shen, London Business School
"""

import pandas as pd
import numpy as np
from pathlib import Path
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
print("MULTI-HORIZON PRICE IMPACT ANALYSIS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/4] Loading data...")

fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills['time_dt'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_str'] = fills['date'].astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")

print(f"  Loaded {len(fills):,} fills")

# =============================================================================
# CLASSIFY WALLETS
# =============================================================================

print("\n[2/4] Classifying wallets...")

training = fills[fills['date_str'] == '2025-07-28'].copy()

# Hourly prices for markout
hourly_prices = training.groupby(['coin', 'hour'])['px'].last().reset_index()
hourly_prices = hourly_prices.sort_values(['coin', 'hour'])
hourly_prices['next_px'] = hourly_prices.groupby('coin')['px'].shift(-1)
hourly_prices['price_change_bps'] = (hourly_prices['next_px'] - hourly_prices['px']) / hourly_prices['px'] * 10000

takers_train = training[training['crossed'] == True].copy()
takers_train = takers_train.merge(hourly_prices[['coin', 'hour', 'price_change_bps']], on=['coin', 'hour'], how='left')
takers_train['direction'] = np.where(takers_train['side'] == 'B', 1, -1)
takers_train['profit_bps'] = takers_train['direction'] * takers_train['price_change_bps']

wallet_stats = takers_train.groupby('wallet').agg({'profit_bps': 'mean', 'coin': 'count'}).reset_index()
wallet_stats.columns = ['wallet', 'mean_profit', 'n_trades']
wallet_stats = wallet_stats[wallet_stats['n_trades'] >= 5]
wallet_stats['quintile'] = pd.qcut(wallet_stats['mean_profit'], 5, labels=[1, 2, 3, 4, 5])

q5_wallets = set(wallet_stats[wallet_stats['quintile'] == 5]['wallet'])
q1_wallets = set(wallet_stats[wallet_stats['quintile'] == 1]['wallet'])

print(f"  Q5: {len(q5_wallets):,}, Q1: {len(q1_wallets):,}")

# =============================================================================
# COMPUTE MULTI-HORIZON MARKOUTS (Vectorized)
# =============================================================================

print("\n[3/4] Computing multi-horizon markouts...")

# Test days
test_fills = fills[fills['date_str'].isin(['2025-07-29', '2025-07-30'])].copy()
test_takers = test_fills[test_fills['crossed'] == True].copy()

# Tag quintiles
test_takers['quintile'] = 'Middle'
test_takers.loc[test_takers['wallet'].isin(q5_wallets), 'quintile'] = 'Q5'
test_takers.loc[test_takers['wallet'].isin(q1_wallets), 'quintile'] = 'Q1'
test_takers = test_takers[test_takers['quintile'].isin(['Q5', 'Q1'])].copy()

print(f"  Test trades: {len(test_takers):,}")

# Create price series for each coin (minute-level for efficiency)
price_series = fills.groupby(['coin', fills['time_dt'].dt.floor('10s')])['px'].median().reset_index()
price_series.columns = ['coin', 'time_bin', 'mid_px']

# Horizons
horizons = {'10s': 10, '1m': 60, '5m': 300, '30m': 1800, '2h': 7200}

results = []

for coin in test_takers['coin'].unique():
    coin_takers = test_takers[test_takers['coin'] == coin].copy()
    coin_prices = price_series[price_series['coin'] == coin].sort_values('time_bin')

    if len(coin_takers) == 0 or len(coin_prices) < 10:
        continue

    coin_takers['time_bin'] = coin_takers['time_dt'].dt.floor('10s')
    coin_takers['direction'] = np.where(coin_takers['side'] == 'B', 1, -1)

    # Merge trade price
    coin_takers = coin_takers.merge(coin_prices[['time_bin', 'mid_px']], on='time_bin', how='left')
    coin_takers = coin_takers.rename(columns={'mid_px': 'trade_px'})

    for horizon_name, horizon_sec in horizons.items():
        # Future time bin
        coin_takers['future_bin'] = coin_takers['time_bin'] + pd.Timedelta(seconds=horizon_sec)

        # Merge future price
        merged = coin_takers.merge(
            coin_prices[['time_bin', 'mid_px']].rename(columns={'time_bin': 'future_bin', 'mid_px': 'future_px'}),
            on='future_bin', how='left'
        )

        # Compute markout
        merged['markout_bps'] = merged['direction'] * (merged['future_px'] - merged['trade_px']) / merged['trade_px'] * 10000

        for q in ['Q5', 'Q1']:
            q_data = merged[merged['quintile'] == q]['markout_bps'].dropna()
            if len(q_data) > 10:
                results.append({
                    'coin': coin,
                    'quintile': q,
                    'horizon': horizon_name,
                    'mean_markout': q_data.mean(),
                    'std': q_data.std(),
                    'n': len(q_data)
                })

results_df = pd.DataFrame(results)

# Aggregate across coins
summary = results_df.groupby(['quintile', 'horizon']).agg({
    'mean_markout': lambda x: np.average(x, weights=results_df.loc[x.index, 'n']),
    'n': 'sum'
}).reset_index()

# Compute SE and t-stat
summary['se'] = results_df.groupby(['quintile', 'horizon']).apply(
    lambda x: np.sqrt(np.average(x['std']**2 / x['n'], weights=x['n']))
).reset_index(drop=True)
summary['t_stat'] = summary['mean_markout'] / summary['se']

# Order horizons
horizon_order = ['10s', '1m', '5m', '30m', '2h']
summary['horizon'] = pd.Categorical(summary['horizon'], categories=horizon_order, ordered=True)
summary = summary.sort_values(['quintile', 'horizon'])

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("CUMULATIVE MID-MOVE BY HORIZON")
print("=" * 70)

print("\n  Q5 (Informed):")
q5_results = summary[summary['quintile'] == 'Q5']
for _, row in q5_results.iterrows():
    sig = '***' if abs(row['t_stat']) > 2.58 else '**' if abs(row['t_stat']) > 1.96 else '*' if abs(row['t_stat']) > 1.65 else ''
    print(f"    {row['horizon']:>5}: {row['mean_markout']:>7.2f} bps (t = {row['t_stat']:>5.2f}{sig})")

print("\n  Q1 (Uninformed):")
q1_results = summary[summary['quintile'] == 'Q1']
for _, row in q1_results.iterrows():
    sig = '***' if abs(row['t_stat']) > 2.58 else '**' if abs(row['t_stat']) > 1.96 else '*' if abs(row['t_stat']) > 1.65 else ''
    print(f"    {row['horizon']:>5}: {row['mean_markout']:>7.2f} bps (t = {row['t_stat']:>5.2f}{sig})")

# Persistence ratios
q5_1m = q5_results[q5_results['horizon'] == '1m']['mean_markout'].values[0]
q5_30m = q5_results[q5_results['horizon'] == '30m']['mean_markout'].values[0]
q5_2h = q5_results[q5_results['horizon'] == '2h']['mean_markout'].values[0]
q1_1m = q1_results[q1_results['horizon'] == '1m']['mean_markout'].values[0]
q1_30m = q1_results[q1_results['horizon'] == '30m']['mean_markout'].values[0]
q1_2h = q1_results[q1_results['horizon'] == '2h']['mean_markout'].values[0]

print("\n" + "=" * 70)
print("PERSISTENCE DIAGNOSTIC")
print("=" * 70)
print(f"""
  Q5 (Informed):
    30m/1m ratio: {q5_30m/q5_1m:.2f}x
    2h/1m ratio:  {q5_2h/q5_1m:.2f}x

  Q1 (Uninformed):
    30m/1m ratio: {q1_30m/q1_1m:.2f}x
    2h/1m ratio:  {q1_2h/q1_1m:.2f}x

  INTERPRETATION:
    Q5 > 1: Price impact PERSISTS (genuine information)
    Q1 < 1: Price impact REVERSES (transient/mechanical)
""")

# Save results
output = {
    'q5_markouts': {h: float(q5_results[q5_results['horizon'] == h]['mean_markout'].values[0]) for h in horizon_order},
    'q1_markouts': {h: float(q1_results[q1_results['horizon'] == h]['mean_markout'].values[0]) for h in horizon_order},
    'q5_persistence_30m_1m': float(q5_30m/q5_1m),
    'q5_persistence_2h_1m': float(q5_2h/q5_1m),
    'q1_persistence_30m_1m': float(q1_30m/q1_1m),
    'q1_persistence_2h_1m': float(q1_2h/q1_1m),
}

with open(OUTPUT_DIR / 'horizon_persistence_results.json', 'w') as f:
    json.dump(output, f, indent=2)

summary.to_csv(OUTPUT_DIR / 'horizon_persistence_summary.csv', index=False)

print("\n✓ Saved results")

# LaTeX table
latex = r"""\begin{table}[H]
\centering
\caption{Multi-Horizon Price Impact: Information vs Transient Effects}
\label{tab:horizon_persistence}
\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{5}{c}{\textbf{Horizon}} \\
\cmidrule(lr){2-6}
& 10 sec & 1 min & 5 min & 30 min & 2 hours \\
\midrule
\textbf{Panel A: Markout (bps)} \\
"""

for q, label in [('Q5', 'Q5 (Informed)'), ('Q1', 'Q1 (Uninformed)')]:
    latex += f"{label} "
    qr = summary[summary['quintile'] == q]
    for h in horizon_order:
        val = qr[qr['horizon'] == h]['mean_markout'].values[0]
        t = qr[qr['horizon'] == h]['t_stat'].values[0]
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.65 else ''
        latex += f"& {val:.2f}{sig} "
    latex += "\\\\\n"

latex += "Differential "
for h in horizon_order:
    q5v = q5_results[q5_results['horizon'] == h]['mean_markout'].values[0]
    q1v = q1_results[q1_results['horizon'] == h]['mean_markout'].values[0]
    latex += f"& {q5v - q1v:.2f} "
latex += "\\\\\n"

latex += r"""\midrule
\textbf{Panel B: Persistence (ratio to 1-min)} \\
"""
latex += f"Q5 (Informed) & -- & 1.00 & {q5_results[q5_results['horizon']=='5m']['mean_markout'].values[0]/q5_1m:.2f} & {q5_30m/q5_1m:.2f} & {q5_2h/q5_1m:.2f} \\\\\n"
latex += f"Q1 (Uninformed) & -- & 1.00 & {q1_results[q1_results['horizon']=='5m']['mean_markout'].values[0]/q1_1m:.2f} & {q1_30m/q1_1m:.2f} & {q1_2h/q1_1m:.2f} \\\\\n"

latex += r"""\bottomrule
\multicolumn{6}{p{12cm}}{\footnotesize Panel A: Mean markout (mid-move in trade direction) by horizon. Panel B: Persistence ratio (markout at horizon / 1-min markout). Q5 ratio $>$ 1 indicates permanent price impact (information); Q1 ratio $<$ 1 indicates transient impact (mechanical). *** $p<0.01$.}
\end{tabular}
\end{table}
"""

with open(OUTPUT_DIR / 'table_horizon_persistence.tex', 'w') as f:
    f.write(latex)

print("✓ Saved: table_horizon_persistence.tex")
print("\n" + "=" * 80)
