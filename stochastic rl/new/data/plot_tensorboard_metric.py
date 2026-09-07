#!/usr/bin/env python3
"""
Generic publication-quality plotting script for any TensorBoard metric.
Can be used for rewards, entropy, critic loss, ADR curriculum, etc.

Usage:
    python plot_tensorboard_metric.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================================
# Smoothing Function
# ============================================================================
def smooth(values, weight=0.9):
    """Apply exponential moving average smoothing."""
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# ============================================================================
# Plotting Function
# ============================================================================
def plot_comparison(dg5f_csv, inspire_csv, x_col, y_col,
                   x_label, y_label, title,
                   output_name, smoothing=0.95, show_raw=True):
    """
    Generic comparison plotting function.

    Args:
        dg5f_csv: Path to DG5F CSV file
        inspire_csv: Path to Inspire CSV file
        x_col: Column name for x-axis (e.g., 'step', 'epoch')
        y_col: Column name for y-axis (e.g., 'shaped_reward', 'entropy')
        x_label: X-axis label for plot
        y_label: Y-axis label for plot
        title: Plot title
        output_name: Output filename (without extension)
        smoothing: Smoothing weight (0.8-0.99)
        show_raw: Whether to show raw data in background
    """

    print(f"\n{'='*70}")
    print(f"Plotting: {title}")
    print(f"{'='*70}")

    # Load data
    print(f"\n[1/4] Loading data...")
    try:
        df_dg5f = pd.read_csv(dg5f_csv)
        df_inspire = pd.read_csv(inspire_csv)
        print(f"  ✓ DG5F:    {len(df_dg5f)} rows from {os.path.basename(dg5f_csv)}")
        print(f"  ✓ Inspire: {len(df_inspire)} rows from {os.path.basename(inspire_csv)}")
    except FileNotFoundError as e:
        print(f"  ❌ ERROR: {e}")
        return

    # Auto-detect column names (TensorBoard uses 'Step', 'Value')
    # Try to map user-specified column names to actual CSV columns
    def find_column(df, preferred_name):
        """Find column with case-insensitive matching."""
        # First try exact match
        if preferred_name in df.columns:
            return preferred_name
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == preferred_name.lower():
                return col
        # Try common alternatives
        if preferred_name.lower() == 'step':
            return 'Step' if 'Step' in df.columns else None
        if preferred_name.lower() in ['value', 'shaped_reward', 'reward']:
            return 'Value' if 'Value' in df.columns else None
        return None

    x_col_dg5f = find_column(df_dg5f, x_col)
    y_col_dg5f = find_column(df_dg5f, y_col)
    x_col_inspire = find_column(df_inspire, x_col)
    y_col_inspire = find_column(df_inspire, y_col)

    if not x_col_dg5f or not y_col_dg5f:
        print(f"  ❌ ERROR: DG5F missing required columns!")
        print(f"     Looking for: [{x_col}, {y_col}]")
        print(f"     Found:       {list(df_dg5f.columns)}")
        return

    if not x_col_inspire or not y_col_inspire:
        print(f"  ❌ ERROR: Inspire missing required columns!")
        print(f"     Looking for: [{x_col}, {y_col}]")
        print(f"     Found:       {list(df_inspire.columns)}")
        return

    # Extract data
    print(f"\n[2/4] Processing data (smoothing = {smoothing})...")
    x_dg5f = df_dg5f[x_col_dg5f].values
    y_dg5f_raw = df_dg5f[y_col_dg5f].values
    y_dg5f_smooth = smooth(y_dg5f_raw, weight=smoothing)

    x_inspire = df_inspire[x_col_inspire].values
    y_inspire_raw = df_inspire[y_col_inspire].values
    y_inspire_smooth = smooth(y_inspire_raw, weight=smoothing)

    # Configure plot style
    print(f"\n[3/4] Generating plot...")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 12

    _, ax = plt.subplots(figsize=(8, 5))

    # Plot raw data (optional, faded background)
    if show_raw:
        ax.plot(x_dg5f, y_dg5f_raw, color='#1f77b4', alpha=0.15, linewidth=0.8)
        ax.plot(x_inspire, y_inspire_raw, color='#d62728', alpha=0.15, linewidth=0.8)

    # Plot smoothed curves
    ax.plot(x_dg5f, y_dg5f_smooth,
            label='UR10e + DG5F (Fully-Actuated)',
            color='#1f77b4', linewidth=2.5)
    ax.plot(x_inspire, y_inspire_smooth,
            label='UR10e + RH56F1 (Underactuated)',
            color='#d62728', linewidth=2.5)

    # Formatting
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', frameon=True, fontsize=11, framealpha=0.95)

    plt.tight_layout()

    # Save outputs
    print(f"\n[4/4] Saving outputs...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    pdf_path = os.path.join(script_dir, f"{output_name}.pdf")
    png_path = os.path.join(script_dir, f"{output_name}.png")

    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    print(f"  ✓ PDF: {pdf_path}")
    print(f"  ✓ PNG: {png_path}")

    # Statistics
    print(f"\n{'='*70}")
    print(f"Final Values:")
    print(f"  DG5F:    {y_dg5f_smooth[-1]:.4f}")
    print(f"  Inspire: {y_inspire_smooth[-1]:.4f}")
    print(f"  Δ:       {y_dg5f_smooth[-1] - y_inspire_smooth[-1]:.4f}")
    print(f"{'='*70}\n")

    plt.show()

# ============================================================================
# Preset Configurations
# ============================================================================
def main():
    """Main function with preset configurations."""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration menu
    print("\n" + "="*70)
    print("Publication-Quality TensorBoard Plotting")
    print("="*70)
    print("\nAvailable plots:")
    print("  1. Shaped Reward (per step)")
    print("  2. Entropy (exploration diversity)")
    print("  3. Critic Loss (value function error)")
    print("  4. Success Reward")
    print("  5. Position Tracking Reward")
    print("  6. ADR Curriculum Difficulty")
    print("  7. Custom (specify your own)")
    print("="*70)

    choice = input("\nSelect plot [1-7]: ").strip()

    configs = {
        '1': {
            'dg5f_csv': 'shaped_step_reward_dg5f.csv',
            'inspire_csv': 'shaped_step_reward_inspire.csv',
            'x_col': 'step',
            'y_col': 'shaped_reward',
            'x_label': 'Training Steps',
            'y_label': 'Total Shaped Reward',
            'title': 'Learning Performance Comparison',
            'output_name': 'shaped_reward_comparison',
            'smoothing': 0.95
        },
        '2': {
            'dg5f_csv': 'entropy_dg5f.csv',
            'inspire_csv': 'entropy_inspire.csv',
            'x_col': 'step',
            'y_col': 'entropy',
            'x_label': 'Training Steps',
            'y_label': 'Policy Entropy',
            'title': 'Exploration Diversity Comparison',
            'output_name': 'entropy_comparison',
            'smoothing': 0.93
        },
        '3': {
            'dg5f_csv': 'critic_loss_dg5f.csv',
            'inspire_csv': 'critic_loss_inspire.csv',
            'x_col': 'step',
            'y_col': 'critic_loss',
            'x_label': 'Training Steps',
            'y_label': 'Critic Loss (TD Error)',
            'title': 'Value Function Learning Comparison',
            'output_name': 'critic_loss_comparison',
            'smoothing': 0.90
        },
        '4': {
            'dg5f_csv': 'success_reward_dg5f.csv',
            'inspire_csv': 'success_reward_inspire.csv',
            'x_col': 'step',
            'y_col': 'success_reward',
            'x_label': 'Training Steps',
            'y_label': 'Success Reward',
            'title': 'Task Success Comparison',
            'output_name': 'success_reward_comparison',
            'smoothing': 0.95
        },
        '5': {
            'dg5f_csv': 'position_tracking_dg5f.csv',
            'inspire_csv': 'position_tracking_inspire.csv',
            'x_col': 'step',
            'y_col': 'position_tracking',
            'x_label': 'Training Steps',
            'y_label': 'Position Tracking Reward',
            'title': 'Position Tracking Comparison',
            'output_name': 'position_tracking_comparison',
            'smoothing': 0.92
        },
        '6': {
            'dg5f_csv': 'adr_curriculum_dg5f.csv',
            'inspire_csv': 'adr_curriculum_inspire.csv',
            'x_col': 'step',
            'y_col': 'curriculum_difficulty',
            'x_label': 'Training Steps',
            'y_label': 'Curriculum Difficulty (Normalized)',
            'title': 'ADR Curriculum Progression',
            'output_name': 'adr_curriculum_comparison',
            'smoothing': 0.98,
            'show_raw': False  # Curriculum is already smooth
        }
    }

    if choice in configs:
        config = configs[choice]
        # Convert relative paths to absolute
        config['dg5f_csv'] = os.path.join(script_dir, config['dg5f_csv'])
        config['inspire_csv'] = os.path.join(script_dir, config['inspire_csv'])
        plot_comparison(**config)

    elif choice == '7':
        print("\nCustom plot mode:")
        print("Please edit the script to add your custom configuration.")

    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
