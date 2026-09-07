#!/usr/bin/env python3
"""
Simple publication-quality plotting for TensorBoard CSV files.

Usage:
    python plot_comparison.py \
        --dg5f shaped_step_reward_dg5f.csv \
        --inspire shaped_step_reward_inspire.csv \
        --ylabel "Total Shaped Reward" \
        --output shaped_reward_comparison.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# ============================================================================
# Smoothing Function
# ============================================================================
def smooth(values, weight=0.95):
    """Apply exponential moving average smoothing."""
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
def plot_comparison(dg5f_csv, inspire_csv, ylabel, output_png, smoothing=0.95):
    """
    Generate publication-quality comparison plot.

    Args:
        dg5f_csv: Path to DG5F CSV file
        inspire_csv: Path to Inspire CSV file
        ylabel: Y-axis label
        output_png: Output PNG filename
        smoothing: Smoothing weight (default 0.95)
    """

    print(f"\n{'='*70}")
    print(f"TensorBoard CSV Plotter")
    print(f"{'='*70}")

    # 1. Load data
    print(f"\n[1/3] Loading CSV files...")
    print(f"  DG5F:    {dg5f_csv}")
    print(f"  Inspire: {inspire_csv}")

    try:
        df_dg5f = pd.read_csv(dg5f_csv)
        df_inspire = pd.read_csv(inspire_csv)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return

    # Hardcoded: TensorBoard always uses 'Step' and 'Value'
    if 'Step' not in df_dg5f.columns or 'Value' not in df_dg5f.columns:
        print(f"\n❌ ERROR: DG5F CSV missing 'Step' or 'Value' columns!")
        print(f"   Found: {list(df_dg5f.columns)}")
        return

    if 'Step' not in df_inspire.columns or 'Value' not in df_inspire.columns:
        print(f"\n❌ ERROR: Inspire CSV missing 'Step' or 'Value' columns!")
        print(f"   Found: {list(df_inspire.columns)}")
        return

    print(f"  ✓ DG5F:    {len(df_dg5f)} rows")
    print(f"  ✓ Inspire: {len(df_inspire)} rows")

    # 2. Extract and smooth
    print(f"\n[2/3] Processing (smoothing={smoothing})...")

    x_dg5f = df_dg5f['Step'].values
    y_dg5f_raw = df_dg5f['Value'].values
    y_dg5f_smooth = smooth(y_dg5f_raw, weight=smoothing)

    x_inspire = df_inspire['Step'].values
    y_inspire_raw = df_inspire['Value'].values
    y_inspire_smooth = smooth(y_inspire_raw, weight=smoothing)

    # 3. Plot
    print(f"\n[3/3] Generating plot...")

    # Publication style settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["grid.linewidth"] = 0.8

    _, ax = plt.subplots(figsize=(8, 5))

    # Plot raw data (faded background)
    ax.plot(x_dg5f, y_dg5f_raw, color='#1f77b4', alpha=0.15, linewidth=0.8)
    ax.plot(x_inspire, y_inspire_raw, color='#d62728', alpha=0.15, linewidth=0.8)

    # Plot smoothed curves
    ax.plot(x_dg5f, y_dg5f_smooth,
            label='UR10e + DG5F (Fully-Actuated)',
            color='#1f77b4', linewidth=2.5)
    ax.plot(x_inspire, y_inspire_smooth,
            label='UR10e + RH56F1 (Underactuated)',
            color='#d62728', linewidth=2.5)

    # Labels and formatting
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(loc='best', frameon=True, fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=False)

    plt.tight_layout()

    # Save PNG only
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_png}")

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Final Values:")
    print(f"  DG5F:    {y_dg5f_smooth[-1]:.4f}")
    print(f"  Inspire: {y_inspire_smooth[-1]:.4f}")
    print(f"  Δ:       {y_dg5f_smooth[-1] - y_inspire_smooth[-1]:.4f}")
    print(f"{'='*70}\n")

    plt.show()

# ============================================================================
# Command Line Interface
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard CSV comparison for DG5F vs Inspire',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_comparison.py \\
        --dg5f shaped_reward_dg5f.csv \\
        --inspire shaped_reward_inspire.csv \\
        --ylabel "Total Shaped Reward" \\
        --output shaped_reward.png
        """
    )

    parser.add_argument('--dg5f', required=True,
                        help='Path to DG5F CSV file (TensorBoard export)')
    parser.add_argument('--inspire', required=True,
                        help='Path to Inspire CSV file (TensorBoard export)')
    parser.add_argument('--ylabel', required=True,
                        help='Y-axis label (e.g., "Total Shaped Reward")')
    parser.add_argument('--output', required=True,
                        help='Output PNG filename')
    parser.add_argument('--smoothing', type=float, default=0.95,
                        help='Smoothing weight 0-1 (default: 0.95)')

    args = parser.parse_args()

    plot_comparison(
        dg5f_csv=args.dg5f,
        inspire_csv=args.inspire,
        ylabel=args.ylabel,
        output_png=args.output,
        smoothing=args.smoothing
    )

if __name__ == "__main__":
    main()
