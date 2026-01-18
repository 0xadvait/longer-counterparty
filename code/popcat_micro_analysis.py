#!/usr/bin/env python3
"""
POPCAT Micro-Level Analysis
Focus on the specific hours of the stress event: Nov 12, 14:00-18:00 UTC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === RELATIVE PATH SETUP (Auto-generated for portability) ===
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CODE_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_FIGURES_DIR = os.path.join(_PROJECT_ROOT, 'figures')
# === END RELATIVE PATH SETUP ===

# Load cached data
df = pd.read_parquet(os.path.join(_DATA_DIR, 'popcat_exploration_data.parquet'))

print("=" * 80)
print("POPCAT MICRO-LEVEL STRESS ANALYSIS")
print("=" * 80)

# Focus on Nov 12
nov12 = df[(df['date'] == '2025-11-12')]
popcat_nov12 = nov12[nov12['coin'] == 'POPCAT'].copy()

# Calculate minute-level metrics
popcat_nov12['minute'] = popcat_nov12['datetime'].dt.floor('T')  # Floor to minute
minute_agg = popcat_nov12.groupby('minute').agg({
    'spread_bps': 'mean',
    'depth': 'mean',
    'mid': 'mean',
    'imbalance': 'mean',
    'timestamp': 'count'
}).reset_index()
minute_agg.columns = ['minute', 'spread_bps', 'depth', 'price', 'imbalance', 'obs']

print("\n" + "=" * 80)
print("MINUTE-BY-MINUTE ANALYSIS: Nov 12, 14:00-18:00 UTC (Event Window)")
print("=" * 80)

# Focus on event window
event_window = minute_agg[
    (minute_agg['minute'] >= '2025-11-12 14:00:00') &
    (minute_agg['minute'] <= '2025-11-12 18:00:00')
].copy()

print(f"\nTotal minutes in event window: {len(event_window)}")
print(f"\nKey statistics:")
print(f"  Min spread: {event_window['spread_bps'].min():.2f} bps")
print(f"  Max spread: {event_window['spread_bps'].max():.2f} bps")
print(f"  Min depth: ${event_window['depth'].min():,.0f}")
print(f"  Max depth: ${event_window['depth'].max():,.0f}")
print(f"  Price range: ${event_window['price'].min():.4f} - ${event_window['price'].max():.4f}")

# Identify the crash
price_peak = event_window.loc[event_window['price'].idxmax()]
price_trough = event_window.loc[
    event_window[event_window['minute'] > price_peak['minute']]['price'].idxmin()
]

print(f"\n" + "-" * 60)
print("CRASH DYNAMICS:")
print(f"  Price Peak: ${price_peak['price']:.4f} at {price_peak['minute']}")
print(f"  Price Trough: ${price_trough['price']:.4f} at {price_trough['minute']}")
print(f"  Crash: {(1 - price_trough['price']/price_peak['price'])*100:.1f}%")
print(f"  Duration: {(price_trough['minute'] - price_peak['minute']).total_seconds()/60:.0f} minutes")

# Find max spread moment
max_spread = event_window.loc[event_window['spread_bps'].idxmax()]
print(f"\n  Max Spread: {max_spread['spread_bps']:.2f} bps at {max_spread['minute']}")
print(f"  Depth at max spread: ${max_spread['depth']:,.0f}")

# Show the critical 30 minutes around peak spread
print(f"\n" + "-" * 60)
print("CRITICAL 30-MINUTE WINDOW (around max spread):")
print("-" * 60)

critical = event_window[
    (event_window['minute'] >= max_spread['minute'] - pd.Timedelta(minutes=15)) &
    (event_window['minute'] <= max_spread['minute'] + pd.Timedelta(minutes=15))
]

for _, row in critical.iterrows():
    marker = " <-- MAX SPREAD" if row['minute'] == max_spread['minute'] else ""
    print(f"{row['minute'].strftime('%H:%M')}: Spread={row['spread_bps']:6.2f}bps, "
          f"Depth=${row['depth']:12,.0f}, Price=${row['price']:.4f}{marker}")

# ============================================================
# COMPARE POPCAT vs CONTROLS DURING EVENT WINDOW
# ============================================================
print("\n" + "=" * 80)
print("POPCAT vs CONTROLS: MINUTE-LEVEL COMPARISON (14:00-18:00)")
print("=" * 80)

CONTROLS = ['DOGE', 'WIF', 'kPEPE', 'kSHIB', 'kBONK', 'MEME', 'GOAT', 'PNUT', 'MOODENG']

event_window_all = nov12[
    (nov12['datetime'] >= '2025-11-12 14:00:00') &
    (nov12['datetime'] <= '2025-11-12 18:00:00')
].copy()

# Aggregate by coin and 5-minute intervals
event_window_all['interval'] = event_window_all['datetime'].dt.floor('5T')

interval_agg = event_window_all.groupby(['interval', 'coin']).agg({
    'spread_bps': 'mean',
    'depth': 'mean',
    'mid': 'mean'
}).reset_index()

# Pivot for easier comparison
spread_pivot = interval_agg.pivot(index='interval', columns='coin', values='spread_bps')

print("\nSpread (bps) by 5-minute interval:")
print("-" * 120)
cols = ['POPCAT'] + [c for c in CONTROLS if c in spread_pivot.columns]
print(spread_pivot[cols].to_string())

# Calculate which coins had bigger spread increases during stress
pre_stress = event_window_all[event_window_all['datetime'] < '2025-11-12 15:00:00']
during_stress = event_window_all[
    (event_window_all['datetime'] >= '2025-11-12 15:00:00') &
    (event_window_all['datetime'] < '2025-11-12 17:00:00')
]

print("\n" + "-" * 60)
print("SPREAD CHANGE: Pre-Stress (14:00-15:00) vs During Stress (15:00-17:00)")
print("-" * 60)

spread_changes = []
for coin in ['POPCAT'] + CONTROLS:
    pre = pre_stress[pre_stress['coin'] == coin]['spread_bps'].mean()
    during = during_stress[during_stress['coin'] == coin]['spread_bps'].mean()
    if not np.isnan(pre) and not np.isnan(during):
        spread_changes.append({
            'coin': coin,
            'pre': pre,
            'during': during,
            'change': during - pre,
            'pct_change': (during/pre - 1) * 100
        })

spread_df = pd.DataFrame(spread_changes).sort_values('change', ascending=False)

for _, row in spread_df.iterrows():
    marker = " <-- POPCAT" if row['coin'] == 'POPCAT' else ""
    print(f"{row['coin']:10s}: {row['pre']:.2f} → {row['during']:.2f} bps "
          f"(Δ = {row['change']:+.2f}, {row['pct_change']:+.1f}%){marker}")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATION...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: POPCAT Spread over Nov 12
ax1 = axes[0, 0]
popcat_hourly = popcat_nov12.groupby('hour')['spread_bps'].mean()
ax1.bar(popcat_hourly.index, popcat_hourly.values, color='steelblue', alpha=0.7)
ax1.axvspan(15, 17, alpha=0.3, color='red', label='Stress Period')
ax1.set_xlabel('Hour (UTC)')
ax1.set_ylabel('Spread (bps)')
ax1.set_title('A. POPCAT Spread by Hour (Nov 12, 2025)')
ax1.legend()

# Panel B: POPCAT Price trajectory
ax2 = axes[0, 1]
price_hourly = popcat_nov12.groupby('hour')['mid'].mean()
ax2.plot(price_hourly.index, price_hourly.values, 'o-', color='darkred', linewidth=2)
ax2.axvspan(15, 17, alpha=0.3, color='red', label='Stress Period')
ax2.set_xlabel('Hour (UTC)')
ax2.set_ylabel('Price ($)')
ax2.set_title('B. POPCAT Price Trajectory (Nov 12, 2025)')
ax2.legend()

# Panel C: POPCAT vs Controls spread during stress
ax3 = axes[1, 0]
spread_df_sorted = spread_df.sort_values('change')
colors = ['red' if c == 'POPCAT' else 'steelblue' for c in spread_df_sorted['coin']]
ax3.barh(spread_df_sorted['coin'], spread_df_sorted['change'], color=colors, alpha=0.7)
ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Spread Change (bps)')
ax3.set_title('C. Spread Change During Stress (15:00-17:00 vs 14:00-15:00)')

# Panel D: Minute-level spread during critical period
ax4 = axes[1, 1]
critical_period = minute_agg[
    (minute_agg['minute'] >= '2025-11-12 14:30:00') &
    (minute_agg['minute'] <= '2025-11-12 17:30:00')
]
ax4.plot(critical_period['minute'], critical_period['spread_bps'], 'o-', markersize=2, color='steelblue')
ax4.axvspan(pd.Timestamp('2025-11-12 15:00:00'), pd.Timestamp('2025-11-12 17:00:00'),
            alpha=0.3, color='red', label='Stress Period')
ax4.set_xlabel('Time (UTC)')
ax4.set_ylabel('Spread (bps)')
ax4.set_title('D. POPCAT Minute-Level Spread (Nov 12, 14:30-17:30)')
ax4.tick_params(axis='x', rotation=45)
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(_FIGURES_DIR, 'figure_popcat_stress.pdf'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(_FIGURES_DIR, 'figure_popcat_stress.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print("Saved: figures/figure_popcat_stress.pdf/png")

# ============================================================
# HONEST ASSESSMENT FOR THE PAPER
# ============================================================
print("\n" + "=" * 80)
print("HONEST ASSESSMENT: SHOULD THIS GO IN THE PAPER?")
print("=" * 80)

popcat_change = spread_df[spread_df['coin'] == 'POPCAT']['change'].values[0]
popcat_rank = len(spread_df[spread_df['change'] > popcat_change]) + 1
total_coins = len(spread_df)

print(f"""
FINDINGS SUMMARY:

1. EVENT DOCUMENTATION:
   - A stress event DID occur on Nov 12, 2025, 15:00-17:00 UTC
   - POPCAT price crashed ~30% in ~1 hour
   - Spreads spiked from ~5 bps to ~12.5 bps (2.5x)

2. DiD VALIDITY:
   - PROBLEM: POPCAT was NOT uniquely affected
   - POPCAT spread change: +{popcat_change:.2f} bps (rank {popcat_rank}/{total_coins})
   - Other memecoins (MOODENG, GOAT) had LARGER spread increases
   - This violates the "SUTVA" assumption of DiD

3. WHAT WE CAN SAY:
   - The crypto memecoin sector broadly experienced stress
   - POPCAT's crash was severe but its market quality impact
     was not uniquely worse than other memecoins
   - The $4.9m "bad debt" may be real, but it didn't translate
     to uniquely poor market quality vs. similar assets

4. RECOMMENDATION:
   - NOT suitable for clean DiD causal identification
   - COULD be included as a "stress episode description" if:
     a) We're honest that it's descriptive, not causal
     b) We note that other memecoins were similarly affected
     c) We use it to illustrate general memecoin market fragility

5. ALTERNATIVE FRAMING:
   - "Memecoin Market Stress Episode" (not POPCAT-specific)
   - Document the common vulnerability of memecoin perps
   - Show recovery dynamics across multiple assets
""")

print("=" * 80)
