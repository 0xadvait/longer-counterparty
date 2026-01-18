#!/usr/bin/env python3
"""
COUNTERPARTY-LEVEL ADVERSE SELECTION: ROBUST METHODOLOGY
=========================================================

Addresses referee concerns about "profit" definition:
1. EXPLICIT MARKOUT HORIZON: 1-minute price markouts (not realized PnL)
2. AVOIDS FUNDING PAYMENTS: Uses price movement, not accounting profit
3. OUT-OF-SAMPLE CLASSIFICATION: Classify takers on Day 1, test on Days 2-3
4. NET OF FEES: Reports both gross and net-of-fee results

Markout Definition:
    Taker Markout = Direction × (Price_{t+1m} - Price_t) / Price_t × 10000 (bps)
    Maker Markout = -Taker Markout (zero-sum)

Classification:
    - Training period: July 28, 2025 (Day 1)
    - Test period: July 29-30, 2025 (Days 2-3)
    - Labels frozen after training

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
KEY_ASSETS = ['BTC', 'ETH', 'SOL', 'HYPE']  # Focus on major assets

# Markout horizons
HORIZONS = {
    '1m': 60000,    # 1 minute (primary)
    '10s': 10000,   # 10 seconds (robustness)
    '5m': 300000    # 5 minutes (robustness)
}
PRIMARY_HORIZON = '1m'

print("="*80)
print("COUNTERPARTY-LEVEL ADVERSE SELECTION: ROBUST METHODOLOGY")
print("="*80)
print("\nKey design features:")
print("  1. Explicit markout horizon (1 minute primary, 10s/5m robustness)")
print("  2. Price markouts avoid funding payment confounds")
print("  3. Out-of-sample classification (train Day 1, test Days 2-3)")
print("  4. Reports gross and net-of-fee results")

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("\n[1/6] Loading data...")
fills = pd.read_parquet(OUTPUT_DIR / 'wallet_fills_data.parquet')
fills['timestamp'] = pd.to_datetime(fills['time'], unit='ms')
fills['date_int'] = fills['date'].astype(int)
fills = fills[fills['coin'].isin(KEY_ASSETS)]

print(f"  Loaded {len(fills):,} fills across {fills['coin'].nunique()} assets")
print(f"  Dates: {sorted(fills['date_int'].unique())}")

# Separate takers and makers
takers = fills[fills['crossed'] == True].copy()
makers = fills[fills['crossed'] == False].copy()

print(f"  Takers: {len(takers):,}")
print(f"  Makers: {len(makers):,}")

# =============================================================================
# LINK MAKER-TAKER PAIRS
# =============================================================================

print("\n[2/6] Linking maker-taker pairs...")

# Merge takers and makers on time, coin, price, size
pairs = takers.merge(
    makers[['time', 'coin', 'px', 'sz', 'wallet', 'fee']].rename(columns={'wallet': 'wallet_maker', 'fee': 'fee_maker'}),
    on=['time', 'coin', 'px', 'sz'],
    how='inner'
)

# Ensure different wallets (no self-trades)
pairs = pairs[pairs['wallet'] != pairs['wallet_maker']]

print(f"  Linked {len(pairs):,} maker-taker pairs")
print(f"  Unique takers: {pairs['wallet'].nunique():,}")
print(f"  Unique makers: {pairs['wallet_maker'].nunique():,}")

# =============================================================================
# COMPUTE MARKOUTS AT EXPLICIT HORIZONS
# =============================================================================

print("\n[3/6] Computing markouts at explicit horizons...")

def compute_pair_markouts(pairs_df, all_fills, horizons):
    """
    Compute price markouts for each trade at multiple horizons.

    Markout = Direction × (Future Price - Entry Price) / Entry Price × 10000

    This measures price movement, NOT realized PnL (avoids funding).
    """
    pairs_df = pairs_df.copy()
    pairs_df['direction'] = np.where(pairs_df['side'] == 'B', 1, -1)

    # Create price lookup by coin and time
    for horizon_name, horizon_ms in horizons.items():
        pairs_df[f'markout_{horizon_name}'] = np.nan

        for coin in pairs_df['coin'].unique():
            coin_mask = pairs_df['coin'] == coin
            coin_pairs = pairs_df[coin_mask]

            # Get all prices for this coin (sorted by time)
            coin_prices = all_fills[all_fills['coin'] == coin][['time', 'px']].drop_duplicates()
            coin_prices = coin_prices.sort_values('time')
            times = coin_prices['time'].values
            prices = coin_prices['px'].values

            # For each pair, find future price
            pair_times = coin_pairs['time'].values
            pair_prices = coin_pairs['px'].values
            pair_idx = coin_pairs.index

            markouts = []
            for i, (t, p) in enumerate(zip(pair_times, pair_prices)):
                target_time = t + horizon_ms
                future_idx = np.searchsorted(times, target_time)
                if future_idx < len(times):
                    future_price = prices[future_idx]
                    markout = (future_price - p) / p * 10000
                    markouts.append(markout)
                else:
                    markouts.append(np.nan)

            pairs_df.loc[pair_idx, f'markout_{horizon_name}'] = markouts

        # Taker profit = direction × markout
        pairs_df[f'taker_profit_{horizon_name}'] = pairs_df['direction'] * pairs_df[f'markout_{horizon_name}']
        # Maker profit = -taker profit (zero-sum)
        pairs_df[f'maker_profit_{horizon_name}'] = -pairs_df[f'taker_profit_{horizon_name}']

    return pairs_df

pairs = compute_pair_markouts(pairs, fills, HORIZONS)

for h in HORIZONS.keys():
    coverage = pairs[f'taker_profit_{h}'].notna().mean() * 100
    print(f"  {h}: {coverage:.1f}% coverage")

# =============================================================================
# OUT-OF-SAMPLE CLASSIFICATION
# =============================================================================

print("\n[4/6] Out-of-sample classification (train Day 1, test Days 2-3)...")

TRAIN_DATE = 20250728  # Day 1
TEST_DATES = [20250729, 20250730]  # Days 2-3

train_pairs = pairs[pairs['date_int'] == TRAIN_DATE].copy()
test_pairs = pairs[pairs['date_int'].isin(TEST_DATES)].copy()

print(f"  Training pairs (Day 1): {len(train_pairs):,}")
print(f"  Test pairs (Days 2-3): {len(test_pairs):,}")

# Classify takers using ONLY training data
profit_col = f'taker_profit_{PRIMARY_HORIZON}'

taker_train_stats = train_pairs.groupby('wallet').agg({
    profit_col: ['mean', 'count']
}).reset_index()
taker_train_stats.columns = ['wallet', 'mean_profit', 'n_trades']

# Require minimum trades for reliable classification
MIN_TRADES = 5
taker_train_stats = taker_train_stats[taker_train_stats['n_trades'] >= MIN_TRADES]

# Classify into quintiles
taker_train_stats['quintile'] = pd.qcut(
    taker_train_stats['mean_profit'].rank(method='first'), 5,
    labels=['Q1_Uninformed', 'Q2', 'Q3', 'Q4', 'Q5_Informed']
)

n_classified = len(taker_train_stats)
n_q5 = (taker_train_stats['quintile'] == 'Q5_Informed').sum()
n_q1 = (taker_train_stats['quintile'] == 'Q1_Uninformed').sum()

print(f"  Classified {n_classified:,} takers ({n_q5} Q5 informed, {n_q1} Q1 uninformed)")
print(f"  Labels FROZEN - using only Day 1 data")

# =============================================================================
# TEST: OUT-OF-SAMPLE COUNTERPARTY ANALYSIS
# =============================================================================

print("\n[5/6] Testing out-of-sample (Days 2-3)...")

# Merge classification labels onto test pairs
test_with_labels = test_pairs.merge(
    taker_train_stats[['wallet', 'quintile']],
    on='wallet',
    how='inner'
)

print(f"  Test pairs with labels: {len(test_with_labels):,}")
print(f"  Coverage: {100*len(test_with_labels)/len(test_pairs):.1f}% of test pairs")

# Compute maker profitability by counterparty type
results = {'horizons': {}}

for horizon_name in HORIZONS.keys():
    maker_profit_col = f'maker_profit_{horizon_name}'
    taker_profit_col = f'taker_profit_{horizon_name}'

    # Filter to pairs with valid markouts
    valid_test = test_with_labels[test_with_labels[maker_profit_col].notna()].copy()

    # Maker profit vs informed (Q5) takers
    informed_trades = valid_test[valid_test['quintile'] == 'Q5_Informed']
    maker_vs_informed = informed_trades[maker_profit_col].mean()
    maker_vs_informed_se = informed_trades[maker_profit_col].std() / np.sqrt(len(informed_trades))

    # Maker profit vs uninformed (Q1) takers
    uninformed_trades = valid_test[valid_test['quintile'] == 'Q1_Uninformed']
    maker_vs_uninformed = uninformed_trades[maker_profit_col].mean()
    maker_vs_uninformed_se = uninformed_trades[maker_profit_col].std() / np.sqrt(len(uninformed_trades))

    # Toxicity differential
    toxicity_diff = maker_vs_uninformed - maker_vs_informed

    # T-test for difference
    t_stat, p_val = stats.ttest_ind(
        uninformed_trades[maker_profit_col].dropna(),
        informed_trades[maker_profit_col].dropna()
    )

    # Winner-loser spread (taker perspective)
    q5_taker_profit = informed_trades[taker_profit_col].mean()
    q1_taker_profit = uninformed_trades[taker_profit_col].mean()
    winner_loser = q5_taker_profit - q1_taker_profit

    results['horizons'][horizon_name] = {
        'maker_vs_informed_bps': float(maker_vs_informed),
        'maker_vs_uninformed_bps': float(maker_vs_uninformed),
        'toxicity_differential_bps': float(toxicity_diff),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'winner_loser_spread_bps': float(winner_loser),
        'n_informed_trades': len(informed_trades),
        'n_uninformed_trades': len(uninformed_trades)
    }

# Print results
print(f"\n{'Horizon':<8} {'Maker vs Q5':>14} {'Maker vs Q1':>14} {'Toxicity Diff':>14} {'t-stat':>10}")
print("-" * 65)

for h, res in results['horizons'].items():
    print(f"{h:<8} {res['maker_vs_informed_bps']:>+14.2f} {res['maker_vs_uninformed_bps']:>+14.2f} {res['toxicity_differential_bps']:>+14.2f} {res['t_stat']:>10.2f}")

# =============================================================================
# THE INFORMATION FOOD CHAIN (OUT-OF-SAMPLE)
# =============================================================================

print("\n[6/6] Building information food chain matrix (out-of-sample)...")

# Also classify makers by frequency (for the matrix)
maker_freq = test_pairs.groupby('wallet_maker').size().reset_index(name='n_maker_trades')
maker_freq['maker_quintile'] = pd.qcut(
    maker_freq['n_maker_trades'].rank(method='first'), 5,
    labels=['Q1_Slow', 'Q2', 'Q3', 'Q4', 'Q5_HFT']
)

# Merge maker quintiles
test_full = test_with_labels.merge(
    maker_freq[['wallet_maker', 'maker_quintile']],
    on='wallet_maker',
    how='inner'
)

# Build food chain matrix
profit_col = f'taker_profit_{PRIMARY_HORIZON}'
food_chain = test_full.pivot_table(
    values=profit_col,
    index='quintile',
    columns='maker_quintile',
    aggfunc='mean'
)

print("\nFood Chain Matrix (Taker Profit in bps, Out-of-Sample):")
print(food_chain.round(1).to_string())

# =============================================================================
# FEES ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("FEE ANALYSIS: GROSS VS NET")
print("="*80)

# Compute fee impact
# Fee is already in the data (positive for takers, negative for makers typically)
# Convert to bps relative to trade value

test_with_labels['trade_value'] = test_with_labels['px'] * test_with_labels['sz']
test_with_labels['taker_fee_bps'] = test_with_labels['fee'] / test_with_labels['trade_value'] * 10000

avg_taker_fee = test_with_labels['taker_fee_bps'].mean()
print(f"\nAverage taker fee: {avg_taker_fee:.2f} bps")

# Net-of-fee results
profit_col = f'taker_profit_{PRIMARY_HORIZON}'
test_with_labels['taker_profit_net'] = test_with_labels[profit_col] - test_with_labels['taker_fee_bps']

informed_net = test_with_labels[test_with_labels['quintile'] == 'Q5_Informed']['taker_profit_net'].mean()
uninformed_net = test_with_labels[test_with_labels['quintile'] == 'Q1_Uninformed']['taker_profit_net'].mean()

print(f"\nGross vs Net-of-Fee ({PRIMARY_HORIZON} markout):")
print(f"  Q5 (Informed) taker profit:   Gross {results['horizons'][PRIMARY_HORIZON]['maker_vs_informed_bps']*-1:+.2f} bps, Net {informed_net:+.2f} bps")
print(f"  Q1 (Uninformed) taker profit: Gross {results['horizons'][PRIMARY_HORIZON]['maker_vs_uninformed_bps']*-1:+.2f} bps, Net {uninformed_net:+.2f} bps")
print(f"  Winner-loser spread:          Gross {results['horizons'][PRIMARY_HORIZON]['winner_loser_spread_bps']:.2f} bps, Net {informed_net - uninformed_net:.2f} bps")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: OUT-OF-SAMPLE COUNTERPARTY ANALYSIS")
print("="*80)

primary = results['horizons'][PRIMARY_HORIZON]

print(f"""
METHODOLOGY:
  - Markout horizon: {PRIMARY_HORIZON} (price movement, NOT realized PnL)
  - Classification: Out-of-sample (train July 28, test July 29-30)
  - Labels: FROZEN before test period
  - Avoids: Funding payments, carry costs, mark-to-market noise

KEY RESULTS ({PRIMARY_HORIZON} markout, out-of-sample):
  - Maker profit vs. informed (Q5):   {primary['maker_vs_informed_bps']:+.1f} bps (N = {primary['n_informed_trades']:,})
  - Maker profit vs. uninformed (Q1): {primary['maker_vs_uninformed_bps']:+.1f} bps (N = {primary['n_uninformed_trades']:,})
  - TOXICITY DIFFERENTIAL:            {primary['toxicity_differential_bps']:+.1f} bps (t = {primary['t_stat']:.2f})

ROBUSTNESS ACROSS HORIZONS:
""")

for h, res in results['horizons'].items():
    print(f"  {h}: Toxicity = {res['toxicity_differential_bps']:+.1f} bps (t = {res['t_stat']:.2f})")

# Add summary statistics
results['summary'] = {
    'methodology': {
        'markout_horizon': PRIMARY_HORIZON,
        'classification': 'out-of-sample',
        'train_date': TRAIN_DATE,
        'test_dates': TEST_DATES,
        'avoids_funding': True,
        'min_trades_for_classification': MIN_TRADES
    },
    'primary_results': primary,
    'fee_analysis': {
        'avg_taker_fee_bps': float(avg_taker_fee),
        'informed_net_profit_bps': float(informed_net),
        'uninformed_net_profit_bps': float(uninformed_net)
    }
}

# Save results
with open(OUTPUT_DIR / 'counterparty_analysis_robust.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved: counterparty_analysis_robust.json")

# =============================================================================
# LATEX TABLE FOR PAPER
# =============================================================================

print("\n" + "="*80)
print("LATEX TABLE FOR PAPER")
print("="*80)

print(r"""
\begin{table}[H]
\centering
\caption{Counterparty-Specific Adverse Selection: Out-of-Sample Evidence}
\label{tab:toxicity_robust}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Panel A: Methodology} & & & & \\
\midrule
Markout definition & \multicolumn{4}{l}{Price movement at horizon $h$: $\frac{P_{t+h} - P_t}{P_t} \times 10000$ (bps)} \\
Classification & \multicolumn{4}{l}{Out-of-sample: train July 28, test July 29--30} \\
Avoids & \multicolumn{4}{l}{Funding payments, carry costs, mark-to-market noise} \\
\midrule
\textbf{Panel B: Results by Horizon} & \textbf{10s} & \textbf{1m} & \textbf{5m} & \\
\midrule""")

for h in ['10s', '1m', '5m']:
    res = results['horizons'][h]
    print(f"Maker vs.\ informed (Q5) & {res['maker_vs_informed_bps']:+.1f} & & & \\\\")
    print(f"Maker vs.\ uninformed (Q1) & {res['maker_vs_uninformed_bps']:+.1f} & & & \\\\")
    print(f"\\textbf{{Toxicity differential}} & \\textbf{{{res['toxicity_differential_bps']:+.1f}}} & & ($t = {res['t_stat']:.1f}$) & \\\\")
    break  # Just show 1m for now

print(r"""\midrule
\multicolumn{5}{l}{\footnotesize Classification uses only July 28 data; labels frozen before test period.}\\
\multicolumn{5}{l}{\footnotesize Price markouts avoid funding payment confounds in perpetuals.}
\end{tabular}
\end{table}
""")
