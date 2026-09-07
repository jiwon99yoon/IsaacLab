#!/usr/bin/env python3
"""
Publication-quality plotting script for RL training curves.
Plots shaped reward comparison between DG5F and Inspire RH56F1.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# Configuration
# ============================================================================
# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DG5F_CSV = os.path.join(SCRIPT_DIR, "shaped_step_reward_dg5f.csv")
INSPIRE_CSV = os.path.join(SCRIPT_DIR, "shaped_step_reward_inspire.csv")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "shaped_reward_comparison.pdf")
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "shaped_reward_comparison.png")

# Plot styling
DG5F_COLOR = '#1f77b4'      # Blue
INSPIRE_COLOR = '#d62728'   # Red
SMOOTHING_WEIGHT = 0.95     # 0.8-0.99 (higher = smoother)
FIGURE_SIZE = (8, 5)        # Width, Height in inches
DPI = 300

# ============================================================================
# Smoothing Function (Exponential Moving Average)
# ============================================================================
def smooth(values, weight=0.9):
    """
    Apply exponential moving average smoothing to reduce noise.

    Args:
        values: Array of values to smooth
        weight: Smoothing factor (0.8-0.99). Higher = smoother

    Returns:
        Smoothed array
    """
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# ============================================================================
# Main Plotting Function
# ============================================================================
def plot_reward_comparison():
    """Generate publication-quality reward comparison plot."""

    print("=" * 70)
    print("Publication-Quality Plotting Script")
    print("=" * 70)

    # 1. Load CSV data
    print(f"\n[1/5] Loading data...")
    print(f"  - DG5F:    {DG5F_CSV}")
    print(f"  - Inspire: {INSPIRE_CSV}")

    try:
        df_dg5f = pd.read_csv(DG5F_CSV)
        df_inspire = pd.read_csv(INSPIRE_CSV)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: CSV file not found!")
        print(f"   {e}")
        print(f"\n   Please ensure CSV files exist in: {SCRIPT_DIR}")
        return

    # Auto-detect column names (TensorBoard uses 'Step', 'Value')
    # Try TensorBoard format first, then custom format
    if 'Step' in df_dg5f.columns and 'Value' in df_dg5f.columns:
        x_col_dg5f = 'Step'
        y_col_dg5f = 'Value'
    elif 'step' in df_dg5f.columns and 'shaped_reward' in df_dg5f.columns:
        x_col_dg5f = 'step'
        y_col_dg5f = 'shaped_reward'
    else:
        print(f"\n❌ ERROR: DG5F CSV has unexpected columns!")
        print(f"   Found: {list(df_dg5f.columns)}")
        print(f"   Expected either: ['Step', 'Value'] or ['step', 'shaped_reward']")
        return

    if 'Step' in df_inspire.columns and 'Value' in df_inspire.columns:
        x_col_inspire = 'Step'
        y_col_inspire = 'Value'
    elif 'step' in df_inspire.columns and 'shaped_reward' in df_inspire.columns:
        x_col_inspire = 'step'
        y_col_inspire = 'shaped_reward'
    else:
        print(f"\n❌ ERROR: Inspire CSV has unexpected columns!")
        print(f"   Found: {list(df_inspire.columns)}")
        print(f"   Expected either: ['Step', 'Value'] or ['step', 'shaped_reward']")
        return

    print(f"  ✓ DG5F data loaded:    {len(df_dg5f)} rows")
    print(f"  ✓ Inspire data loaded: {len(df_inspire)} rows")

    # 2. Extract and smooth data
    print(f"\n[2/5] Processing data (smoothing weight = {SMOOTHING_WEIGHT})...")

    x_dg5f = df_dg5f[x_col_dg5f].values
    y_dg5f_raw = df_dg5f[y_col_dg5f].values
    y_dg5f_smooth = smooth(y_dg5f_raw, weight=SMOOTHING_WEIGHT)

    x_inspire = df_inspire[x_col_inspire].values
    y_inspire_raw = df_inspire[y_col_inspire].values
    y_inspire_smooth = smooth(y_inspire_raw, weight=SMOOTHING_WEIGHT)

    print(f"  ✓ Data smoothed")

    # 3. Configure publication-style plot settings
    print(f"\n[3/5] Configuring plot style...")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["grid.linewidth"] = 0.8

    # 4. Create plot
    print(f"\n[4/5] Generating plot...")

    _, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Plot raw data (faded in background for credibility)
    ax.plot(x_dg5f, y_dg5f_raw, color=DG5F_COLOR, alpha=0.15, linewidth=0.8)
    ax.plot(x_inspire, y_inspire_raw, color=INSPIRE_COLOR, alpha=0.15, linewidth=0.8)

    # Plot smoothed curves (main lines)
    ax.plot(x_dg5f, y_dg5f_smooth,
            label='UR10e + DG5F (Fully-Actuated)',
            color=DG5F_COLOR,
            linewidth=2.5)
    ax.plot(x_inspire, y_inspire_smooth,
            label='UR10e + RH56F1 (Underactuated)',
            color=INSPIRE_COLOR,
            linewidth=2.5)

    # Labels and title
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Shaped Reward', fontsize=14, fontweight='bold')
    ax.set_title('Learning Performance Comparison', fontsize=16, fontweight='bold', pad=15)

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(loc='lower right', frameon=True, fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=False)

    # Tight layout
    plt.tight_layout()

    # 5. Save outputs
    print(f"\n[5/5] Saving outputs...")

    # Save as PDF (vector graphics - recommended for LaTeX)
    plt.savefig(OUTPUT_PDF, format='pdf', dpi=DPI, bbox_inches='tight')
    print(f"  ✓ PDF saved: {OUTPUT_PDF}")

    # Save as PNG (raster graphics - for preview)
    plt.savefig(OUTPUT_PNG, format='png', dpi=DPI, bbox_inches='tight')
    print(f"  ✓ PNG saved: {OUTPUT_PNG}")

    # Display statistics
    print(f"\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"DG5F Final Reward:    {y_dg5f_smooth[-1]:.2f}")
    print(f"Inspire Final Reward: {y_inspire_smooth[-1]:.2f}")
    print(f"Performance Gap:      {y_dg5f_smooth[-1] - y_inspire_smooth[-1]:.2f}")
    print("=" * 70)

    # Show plot
    print(f"\nDisplaying plot... (close window to exit)")
    plt.show()

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    plot_reward_comparison()
