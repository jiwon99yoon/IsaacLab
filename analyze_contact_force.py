#!/usr/bin/env python3
"""
Analyze contact force from Tensorboard logs to check if clipping is appropriate.
"""

import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_tensorboard_logs(log_path, hand_name):
    """Parse Tensorboard logs and analyze contact force distribution."""

    print(f"\n{'='*80}")
    print(f"Analyzing {hand_name}")
    print(f"Log path: {log_path}")
    print(f"{'='*80}\n")

    # Load Tensorboard event file
    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    # List all available tags
    print("Available scalar tags:")
    tags = ea.Tags()['scalars']

    # Filter for contact/force related tags
    contact_tags = [tag for tag in tags if 'contact' in tag.lower() or 'force' in tag.lower()]

    if contact_tags:
        print(f"\n📊 Contact/Force related tags found:")
        for tag in contact_tags:
            print(f"  - {tag}")
    else:
        print("\n⚠️  No contact/force tags found. Checking reward tags instead...")
        contact_tags = [tag for tag in tags if 'finger' in tag.lower()]
        for tag in contact_tags:
            print(f"  - {tag}")

    # Analyze each tag
    results = {}
    for tag in contact_tags:
        try:
            events = ea.Scalars(tag)
            values = np.array([e.value for e in events])

            if len(values) > 0:
                results[tag] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'count': len(values)
                }

                print(f"\n📈 {tag}:")
                print(f"  Mean:   {results[tag]['mean']:8.3f}")
                print(f"  Std:    {results[tag]['std']:8.3f}")
                print(f"  Min:    {results[tag]['min']:8.3f}")
                print(f"  Max:    {results[tag]['max']:8.3f}")
                print(f"  Median: {results[tag]['median']:8.3f}")
                print(f"  95%ile: {results[tag]['p95']:8.3f}")
                print(f"  99%ile: {results[tag]['p99']:8.3f}")
                print(f"  Points: {results[tag]['count']}")

        except Exception as e:
            print(f"  ⚠️  Error reading {tag}: {e}")

    # Check for reward tags
    print("\n\n📊 Key reward metrics:")
    reward_tags = [
        'Train/mean_rewards/episode_reward_mean',
        'Train/mean_rewards/success',
        'Train/mean_rewards/good_finger_contact',
        'Train/mean_rewards/position_tracking',
        'episode_lengths/mean'
    ]

    for tag in reward_tags:
        if tag in tags:
            try:
                events = ea.Scalars(tag)
                values = np.array([e.value for e in events])
                if len(values) > 0:
                    print(f"  {tag.split('/')[-1]:30s}: mean={np.mean(values[-100:]):8.3f}, latest={values[-1]:8.3f}")
            except:
                pass

    return results


def main():
    # Paths to log files
    inspire_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_inspire_right/2025-12-15_15-23-45/summaries"
    dg5f_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right/2025-11-30_05-42-14/summaries"

    # Analyze Inspire
    inspire_results = analyze_tensorboard_logs(inspire_log, "Inspire Hand (6-DOF, clip=±20N)")

    # Analyze DG5F
    dg5f_results = analyze_tensorboard_logs(dg5f_log, "DG5F Hand (20-DOF, clip=±50N)")

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON & RECOMMENDATIONS")
    print("="*80)

    # Check good_finger_contact reward
    inspire_contact_tag = None
    dg5f_contact_tag = None

    for tag in inspire_results.keys():
        if 'good_finger_contact' in tag or 'contact' in tag:
            inspire_contact_tag = tag
            break

    for tag in dg5f_results.keys():
        if 'good_finger_contact' in tag or 'contact' in tag:
            dg5f_contact_tag = tag
            break

    if inspire_contact_tag and dg5f_contact_tag:
        print(f"\n📊 Contact Reward Comparison:")
        print(f"  Inspire ({inspire_contact_tag}): {inspire_results[inspire_contact_tag]['mean']:.3f}")
        print(f"  DG5F    ({dg5f_contact_tag}):    {dg5f_results[dg5f_contact_tag]['mean']:.3f}")

    print("\n🔍 Analysis:")
    print("  1. If Inspire's max contact force is close to 20N → clipping is happening → PROBLEM")
    print("  2. If DG5F uses force range much higher than Inspire → unfair comparison → PROBLEM")
    print("  3. If both stay well below their clip limits → OK to keep different clips")

    print("\n💡 Recommendations:")
    print("  - Check if 'good_finger_contact' reward is similar between hands")
    print("  - Check if episode success rate is comparable")
    print("  - If Inspire is clipping frequently (max ≈ 20N), increase clip to 50N and retrain")
    print("  - If both use similar force ranges, current setup is OK (mention in paper)")


if __name__ == "__main__":
    main()
