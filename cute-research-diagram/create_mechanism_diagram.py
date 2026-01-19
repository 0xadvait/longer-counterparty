#!/usr/bin/env python3
"""
Mechanism Diagram for "The Two Margins of Market Quality" Research Paper
Creates a professional, cute diagram with bunny symbols
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon
import numpy as np

# Set up the figure with a clean white background
fig, ax = plt.subplots(1, 1, figsize=(20, 15))
ax.set_xlim(0, 100)
ax.set_ylim(-6, 72)
ax.set_aspect('equal')
ax.axis('off')

# Color palette - professional with soft pastels
colors = {
    'title': '#2C3E50',
    'demand_box': '#E8F4FD',
    'demand_border': '#3498DB',
    'supply_box': '#FDF2E9',
    'supply_border': '#E67E22',
    'shock_box': '#FDEDEC',
    'shock_border': '#E74C3C',
    'outcome_box': '#E8F8F5',
    'outcome_border': '#1ABC9C',
    'arrow': '#7F8C8D',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'bunny_pink': '#FADBD8',
    'bunny_ear': '#F5B7B1',
    'text': '#2C3E50',
    'stat': '#8E44AD',
}

def draw_bunny(ax, x, y, size=1.5, facing='right'):
    """Draw a cute bunny face"""
    # Head
    head = Circle((x, y), size, facecolor=colors['bunny_pink'],
                  edgecolor='#D5A6BD', linewidth=1.5, zorder=10)
    ax.add_patch(head)

    # Ears
    ear_offset = 0.3 if facing == 'right' else -0.3
    ear1 = Ellipse((x - size*0.5, y + size*1.5), size*0.4, size*1.2,
                   facecolor=colors['bunny_ear'], edgecolor='#D5A6BD', linewidth=1, zorder=9)
    ear2 = Ellipse((x + size*0.5, y + size*1.5), size*0.4, size*1.2,
                   facecolor=colors['bunny_ear'], edgecolor='#D5A6BD', linewidth=1, zorder=9)
    ax.add_patch(ear1)
    ax.add_patch(ear2)

    # Inner ears
    ear1_inner = Ellipse((x - size*0.5, y + size*1.5), size*0.2, size*0.8,
                         facecolor='#FADBD8', edgecolor='none', zorder=11)
    ear2_inner = Ellipse((x + size*0.5, y + size*1.5), size*0.2, size*0.8,
                         facecolor='#FADBD8', edgecolor='none', zorder=11)
    ax.add_patch(ear1_inner)
    ax.add_patch(ear2_inner)

    # Eyes
    eye_y = y + size*0.2
    ax.plot(x - size*0.35, eye_y, 'ko', markersize=size*3, zorder=12)
    ax.plot(x + size*0.35, eye_y, 'ko', markersize=size*3, zorder=12)
    # Eye shine
    ax.plot(x - size*0.3, eye_y + size*0.1, 'wo', markersize=size*1.5, zorder=13)
    ax.plot(x + size*0.4, eye_y + size*0.1, 'wo', markersize=size*1.5, zorder=13)

    # Nose
    nose = Polygon([(x, y - size*0.1), (x - size*0.15, y - size*0.3),
                    (x + size*0.15, y - size*0.3)],
                   facecolor='#F1948A', edgecolor='none', zorder=12)
    ax.add_patch(nose)

    # Whiskers
    whisker_y = y - size*0.2
    for i, offset in enumerate([-0.2, 0, 0.2]):
        ax.plot([x - size*0.6, x - size*0.2], [whisker_y + offset*size, whisker_y + offset*size*0.5],
                color='#B5B5B5', linewidth=0.8, zorder=11)
        ax.plot([x + size*0.6, x + size*0.2], [whisker_y + offset*size, whisker_y + offset*size*0.5],
                color='#B5B5B5', linewidth=0.8, zorder=11)

def draw_box(ax, x, y, width, height, facecolor, edgecolor, title, content_lines, title_size=11):
    """Draw a rounded box with title and content"""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.5",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=2.5, zorder=1)
    ax.add_patch(box)

    # Title
    ax.text(x + width/2, y + height - 1.2, title, fontsize=title_size, fontweight='bold',
            ha='center', va='top', color=colors['text'], zorder=5)

    # Content
    start_y = y + height - 3
    for i, line in enumerate(content_lines):
        ax.text(x + width/2, start_y - i*1.5, line, fontsize=9, ha='center', va='top',
                color=colors['text'], zorder=5)

def draw_arrow(ax, start, end, color=None, style='simple', label='', label_pos=0.5, curved=False):
    """Draw an arrow between two points"""
    if color is None:
        color = colors['arrow']

    if curved:
        connectionstyle = "arc3,rad=0.2"
    else:
        connectionstyle = "arc3,rad=0"

    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=2, connectionstyle=connectionstyle, zorder=3)
    ax.add_patch(arrow)

    if label:
        mid_x = start[0] + (end[0] - start[0]) * label_pos
        mid_y = start[1] + (end[1] - start[1]) * label_pos
        if curved:
            mid_y += 1.5
        ax.text(mid_x, mid_y + 0.8, label, fontsize=8, ha='center', va='bottom',
                color=color, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# ============== TITLE ==============
ax.text(50, 68, "The Two Margins of Market Quality", fontsize=20, fontweight='bold',
        ha='center', va='top', color=colors['title'])
ax.text(50, 65.5, "A Mechanism Diagram", fontsize=14, style='italic',
        ha='center', va='top', color='#7F8C8D')

# Draw decorative bunnies near title
draw_bunny(ax, 12, 66, size=1.8)
draw_bunny(ax, 88, 66, size=1.8)

# ============== MAIN SHOCK BOX (CENTER TOP) ==============
shock_box = FancyBboxPatch((35, 52), 30, 10, boxstyle="round,pad=0.02,rounding_size=0.8",
                           facecolor=colors['shock_box'], edgecolor=colors['shock_border'],
                           linewidth=3, zorder=1)
ax.add_patch(shock_box)
ax.text(50, 60, "Infrastructure Stress", fontsize=13, fontweight='bold',
        ha='center', va='top', color=colors['shock_border'])
ax.text(50, 57.5, "(API Outage / Congestion)", fontsize=10, style='italic',
        ha='center', va='top', color=colors['text'])
ax.text(50, 55, "Quote Updates: -40% to -68%", fontsize=9,
        ha='center', va='top', color=colors['stat'])

# ============== KEY INSIGHT BOX ==============
insight_box = FancyBboxPatch((30, 45), 40, 5, boxstyle="round,pad=0.02,rounding_size=0.5",
                             facecolor='#FEF9E7', edgecolor='#F1C40F', linewidth=2, zorder=1)
ax.add_patch(insight_box)
ax.text(50, 48.5, "Key Insight: Infrastructure access correlates with trader type",
        fontsize=10, fontweight='bold', ha='center', va='top', color='#B7950B')
ax.text(50, 46, "on BOTH sides of the market", fontsize=9, style='italic',
        ha='center', va='top', color='#B7950B')

# ============== DEMAND SIDE (LEFT) ==============
demand_box = FancyBboxPatch((5, 18), 28, 24, boxstyle="round,pad=0.02,rounding_size=0.8",
                            facecolor=colors['demand_box'], edgecolor=colors['demand_border'],
                            linewidth=3, zorder=1)
ax.add_patch(demand_box)

# Demand side title and header
ax.text(19, 41, "PARTICIPANT COMPOSITION", fontsize=12, fontweight='bold',
        ha='center', va='top', color=colors['demand_border'])
ax.text(19, 38.5, "(Demand Side: Who Trades)", fontsize=10, style='italic',
        ha='center', va='top', color=colors['demand_border'])

# Informed takers box
informed_box = FancyBboxPatch((7, 31), 11, 6, boxstyle="round,pad=0.01,rounding_size=0.3",
                              facecolor='white', edgecolor=colors['positive'], linewidth=1.5, zorder=2)
ax.add_patch(informed_box)
ax.text(12.5, 36, "Informed", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['positive'])
ax.text(12.5, 34, "Takers (Q5)", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(12.5, 32.2, "+19.2 bps", fontsize=8, ha='center', va='top', color=colors['stat'])

# Uninformed takers box
uninformed_box = FancyBboxPatch((20, 31), 11, 6, boxstyle="round,pad=0.01,rounding_size=0.3",
                                facecolor='white', edgecolor=colors['negative'], linewidth=1.5, zorder=2)
ax.add_patch(uninformed_box)
ax.text(25.5, 36, "Uninformed", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['negative'])
ax.text(25.5, 34, "Takers (Q1)", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(25.5, 32.2, "negative", fontsize=8, ha='center', va='top', color=colors['stat'])

# Mechanism description
ax.text(19, 29.5, "During Stress:", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['text'])
ax.text(19, 27.5, "Informed/Uninformed ratio", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(19, 25.8, "+3.4% to +5.1%", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['positive'])
ax.text(19, 24, "(4/4 events consistent)", fontsize=8, style='italic', ha='center', va='top', color=colors['stat'])

# Toxicity differential result
ax.text(19, 21.5, "Toxicity Differential:", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['text'])
ax.text(19, 19.5, "+3.1 bps (t=5.8)", fontsize=9, ha='center', va='top', color=colors['stat'])

draw_bunny(ax, 8, 19.5, size=1.2)

# ============== SUPPLY SIDE (RIGHT) ==============
supply_box = FancyBboxPatch((67, 18), 28, 24, boxstyle="round,pad=0.02,rounding_size=0.8",
                            facecolor=colors['supply_box'], edgecolor=colors['supply_border'],
                            linewidth=3, zorder=1)
ax.add_patch(supply_box)

# Supply side title and header
ax.text(81, 41, "PRICE-SETTING COMPOSITION", fontsize=12, fontweight='bold',
        ha='center', va='top', color=colors['supply_border'])
ax.text(81, 38.5, "(Supply Side: Who Sets Prices)", fontsize=10, style='italic',
        ha='center', va='top', color=colors['supply_border'])

# High-MPSC makers box
hft_box = FancyBboxPatch((69, 31), 11, 6, boxstyle="round,pad=0.01,rounding_size=0.3",
                         facecolor='white', edgecolor=colors['positive'], linewidth=1.5, zorder=2)
ax.add_patch(hft_box)
ax.text(74.5, 36, "High-MPSC", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['positive'])
ax.text(74.5, 34, "Makers (HFT)", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(74.5, 32.2, "Price setters", fontsize=8, ha='center', va='top', color=colors['stat'])

# Marginal makers box
marginal_box = FancyBboxPatch((82, 31), 11, 6, boxstyle="round,pad=0.01,rounding_size=0.3",
                              facecolor='white', edgecolor='#95A5A6', linewidth=1.5, zorder=2)
ax.add_patch(marginal_box)
ax.text(87.5, 36, "Marginal", fontsize=9, fontweight='bold', ha='center', va='top', color='#7F8C8D')
ax.text(87.5, 34, "Makers", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(87.5, 32.2, "Volume fillers", fontsize=8, ha='center', va='top', color=colors['stat'])

# Mechanism description
ax.text(81, 29.5, "During Stress:", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['text'])
ax.text(81, 27.5, "High-MPSC fill rates: -24%", fontsize=8, ha='center', va='top', color=colors['negative'])
ax.text(81, 25.8, "Other makers: +55%", fontsize=8, ha='center', va='top', color=colors['positive'])
ax.text(81, 24, "(Cannot substitute)", fontsize=8, style='italic', ha='center', va='top', color=colors['stat'])

# Paradox highlight
ax.text(81, 21.5, "THE PARADOX:", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['supply_border'])
ax.text(81, 19.5, "HHI -13%, Participation +84%", fontsize=8, ha='center', va='top', color=colors['stat'])

draw_bunny(ax, 92, 19.5, size=1.2)

# ============== OUTCOME BOX (BOTTOM CENTER) ==============
outcome_box = FancyBboxPatch((32, 2), 36, 12, boxstyle="round,pad=0.02,rounding_size=0.8",
                             facecolor=colors['outcome_box'], edgecolor=colors['outcome_border'],
                             linewidth=3, zorder=1)
ax.add_patch(outcome_box)

ax.text(50, 13, "MARKET QUALITY OUTCOME", fontsize=12, fontweight='bold',
        ha='center', va='top', color=colors['outcome_border'])
ax.text(50, 10.8, "Spread Widening: +1.66 bps (mean)", fontsize=10,
        ha='center', va='top', color=colors['text'])
ax.text(50, 8.8, "Sign test p = 0.0625 (4/4 events positive)", fontsize=9,
        ha='center', va='top', color=colors['stat'])

# Two channels contribution
ax.text(41, 6.5, "Staleness", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['demand_border'])
ax.text(41, 4.8, "Channel", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(41, 3.3, "42%", fontsize=10, fontweight='bold', ha='center', va='top', color=colors['demand_border'])

ax.text(59, 6.5, "Selection", fontsize=9, fontweight='bold', ha='center', va='top', color=colors['supply_border'])
ax.text(59, 4.8, "Channel", fontsize=8, ha='center', va='top', color=colors['text'])
ax.text(59, 3.3, "28%", fontsize=10, fontweight='bold', ha='center', va='top', color=colors['supply_border'])

ax.text(50, 6.5, "+", fontsize=14, fontweight='bold', ha='center', va='top', color=colors['text'])

# Selection predicts beyond staleness
selection_note = FancyBboxPatch((35, -1), 30, 3, boxstyle="round,pad=0.02,rounding_size=0.3",
                                facecolor='#F5EEF8', edgecolor=colors['stat'], linewidth=1.5, zorder=1)
ax.add_patch(selection_note)
ax.text(50, 1.5, "Selection predicts spreads beyond staleness:", fontsize=8, fontweight='bold',
        ha='center', va='top', color=colors['stat'])
ax.text(50, 0, "β = 0.07 (t = 2.51, p < 0.05)", fontsize=9, ha='center', va='top', color=colors['stat'])

# ============== ARROWS ==============
# From shock to key insight
draw_arrow(ax, (50, 52), (50, 50), color=colors['shock_border'])

# From key insight to demand side
draw_arrow(ax, (35, 47.5), (25, 42), color=colors['demand_border'],
           label='Informed invest\nmore in capacity', label_pos=0.4, curved=True)

# From key insight to supply side
draw_arrow(ax, (65, 47.5), (75, 42), color=colors['supply_border'],
           label='HFT depend on\nAPI access', label_pos=0.4, curved=True)

# From demand side to outcome
draw_arrow(ax, (19, 18), (40, 14), color=colors['demand_border'],
           label='+Adverse Selection', label_pos=0.5)

# From supply side to outcome
draw_arrow(ax, (81, 18), (60, 14), color=colors['supply_border'],
           label='−Price-Setting\nCapacity', label_pos=0.5)

# ============== LEGEND / KEY FINDINGS BOX ==============
legend_box = FancyBboxPatch((2, 2), 26, 12, boxstyle="round,pad=0.02,rounding_size=0.5",
                            facecolor='#FDFEFE', edgecolor='#BDC3C7', linewidth=1.5, zorder=1)
ax.add_patch(legend_box)

ax.text(15, 13, "KEY STATISTICS", fontsize=10, fontweight='bold',
        ha='center', va='top', color=colors['title'])

stats = [
    ("Toxicity Differential:", "+3.1 bps (t=5.8)"),
    ("Taker-level:", "+19.2 bps (t=24.4)"),
    ("Spread Effect:", "+2.60 bps (main event)"),
    ("Selection β:", "0.07 (t=2.51)"),
]

for i, (label, value) in enumerate(stats):
    ax.text(4, 11 - i*2.2, label, fontsize=8, ha='left', va='top', color=colors['text'])
    ax.text(26, 11 - i*2.2, value, fontsize=8, ha='right', va='top', color=colors['stat'], fontweight='bold')

# ============== EVENTS BOX (RIGHT) ==============
events_box = FancyBboxPatch((72, 2), 26, 12, boxstyle="round,pad=0.02,rounding_size=0.5",
                            facecolor='#FDFEFE', edgecolor='#BDC3C7', linewidth=1.5, zorder=1)
ax.add_patch(events_box)

ax.text(85, 13, "FOUR EVENTS", fontsize=10, fontweight='bold',
        ha='center', va='top', color=colors['title'])

events = [
    ("Jan 20 Cong. 1:", "+0.70 bps"),
    ("Jan 20 Cong. 2:", "+1.10 bps"),
    ("Jul 29 Outage:", "+2.60 bps"),
    ("Jul 30 Stress:", "+2.25 bps"),
]

for i, (label, value) in enumerate(events):
    ax.text(74, 11 - i*2.2, label, fontsize=8, ha='left', va='top', color=colors['text'])
    ax.text(96, 11 - i*2.2, value, fontsize=8, ha='right', va='top', color=colors['positive'], fontweight='bold')

# ============== FOOTER ==============
ax.text(50, -4, "Shen (2025) | Observable Identity Enables Both Margins to Be Measured",
        fontsize=9, style='italic', ha='center', va='top', color='#95A5A6')

# Small decorative bunnies at bottom corners
draw_bunny(ax, 6, -2.5, size=1.2)
draw_bunny(ax, 94, -2.5, size=1.2)

# Add a subtle grid pattern in background (very light)
for i in range(0, 101, 10):
    ax.axhline(y=i, color='#EAECEE', linewidth=0.3, zorder=0)
    ax.axvline(x=i, color='#EAECEE', linewidth=0.3, zorder=0)

plt.tight_layout()
plt.savefig('/Users/bshen/Documents/GitHub/longer-counterparty/cute-research-diagram/mechanism_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.3)
plt.close()

print("Mechanism diagram saved to: cute-research-diagram/mechanism_diagram.png")
