#!/usr/bin/env python3
"""
Identify which experiments are Kuka-Allegro vs UR10e/HDR/etc in reorient logs.

This script scans all experiment directories and extracts the environment name
from env.yaml files to identify which robot-hand combination was used.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime

def identify_experiments(log_dir="/home/dyros/IsaacLab/logs/rl_games/reorient"):
    """Scan all experiment directories and identify robot-hand combinations."""

    results = {
        'kuka_allegro': [],
        'ur10e_inspire': [],
        'ur10e_dg5f': [],
        'hdr_dg5f': [],
        'uh035': [],
        'unknown': []
    }

    log_path = Path(log_dir)

    # Get all experiment directories (timestamp format: YYYY-MM-DD_HH-MM-SS)
    exp_dirs = [d for d in log_path.iterdir() if d.is_dir() and not d.name.endswith('.zip')]
    exp_dirs.sort()

    print(f"Scanning {len(exp_dirs)} experiment directories in {log_dir}\n")
    print("=" * 100)

    for exp_dir in exp_dirs:
        env_yaml = exp_dir / "params" / "env.yaml"

        if not env_yaml.exists():
            results['unknown'].append((exp_dir.name, "No env.yaml found"))
            continue

        try:
            # Read env.yaml as text and search for robot USD path
            with open(env_yaml, 'r') as f:
                content = f.read()

            # Extract robot name from USD path
            env_name = "Unknown"

            if 'KukaAllegro' in content or 'kuka_allegro' in content:
                category = 'kuka_allegro'
                env_name = 'Kuka iiwa + Allegro Hand'
            elif 'ur10e' in content.lower() and 'inspire' in content.lower():
                category = 'ur10e_inspire'
                env_name = 'UR10e + Inspire Hand'
            elif 'ur10e' in content.lower() and 'dg5f' in content.lower():
                category = 'ur10e_dg5f'
                env_name = 'UR10e + DG5F Hand'
            elif 'hdr' in content.lower() and 'dg5f' in content.lower():
                category = 'hdr_dg5f'
                env_name = 'HDR + DG5F Hand'
            elif 'uh035' in content.lower():
                category = 'uh035'
                env_name = 'UH035 + Hand'
            else:
                category = 'unknown'

            results[category].append((exp_dir.name, env_name))

        except Exception as e:
            results['unknown'].append((exp_dir.name, f"Error: {str(e)}"))

    # Print results
    print("\n📊 EXPERIMENT CLASSIFICATION RESULTS\n")
    print("=" * 100)

    for category, experiments in results.items():
        if not experiments:
            continue

        print(f"\n{'=' * 100}")
        print(f"🤖 {category.upper().replace('_', ' + ')} ({len(experiments)} experiments)")
        print(f"{'=' * 100}\n")

        for timestamp, env_name in sorted(experiments):
            print(f"  {timestamp:30s}  →  {env_name}")

    print(f"\n{'=' * 100}")
    print("\n📈 SUMMARY")
    print("=" * 100)
    print(f"  Kuka iiwa + Allegro Hand:  {len(results['kuka_allegro']):3d}")
    print(f"  UR10e + Inspire Hand:      {len(results['ur10e_inspire']):3d}")
    print(f"  UR10e + DG5F Hand:         {len(results['ur10e_dg5f']):3d}")
    print(f"  HDR + DG5F Hand:           {len(results['hdr_dg5f']):3d}")
    print(f"  UH035:                     {len(results['uh035']):3d}")
    print(f"  Unknown:                   {len(results['unknown']):3d}")
    print(f"  {'─' * 50}")
    print(f"  TOTAL:                     {sum(len(v) for v in results.values()):3d}")
    print("=" * 100)

    return results

if __name__ == "__main__":
    results = identify_experiments()
