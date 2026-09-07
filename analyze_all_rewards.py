#!/usr/bin/env python3
"""
Analyze all rewards to understand learning dynamics.
"""

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def analyze_all_rewards(log_path, hand_name):
    """Parse all reward tags."""

    print(f"\n{'='*80}")
    print(f"{hand_name}")
    print(f"{'='*80}\n")

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    tags = ea.Tags()['scalars']

    # Get episode reward tags
    episode_reward_tags = [tag for tag in tags if 'Episode_Reward' in tag]
    train_reward_tags = [tag for tag in tags if 'mean_rewards' in tag]

    print("📊 Episode Rewards (Last 100 episodes average):")
    for tag in sorted(episode_reward_tags):
        try:
            events = ea.Scalars(tag)
            values = np.array([e.value for e in events])
            if len(values) > 0:
                recent = values[-100:] if len(values) >= 100 else values
                reward_name = tag.split('/')[-1]
                print(f"  {reward_name:30s}: mean={np.mean(recent):8.4f}, std={np.std(recent):8.4f}, min={np.min(recent):8.4f}, max={np.max(recent):8.4f}")
        except Exception as e:
            print(f"  Error reading {tag}: {e}")

    print("\n📈 Training Metrics (Last 100 updates):")
    for tag in sorted(train_reward_tags):
        try:
            events = ea.Scalars(tag)
            values = np.array([e.value for e in events])
            if len(values) > 0:
                recent = values[-100:] if len(values) >= 100 else values
                metric_name = tag.split('/')[-1]
                print(f"  {metric_name:30s}: mean={np.mean(recent):8.4f}, latest={values[-1]:8.4f}")
        except:
            pass

    # Check success rate and episode length
    print("\n🎯 Key Performance Metrics:")
    key_metrics = {
        'Success Rate': 'Episode/Episode_Reward/success',
        'Episode Reward': 'Train/mean_rewards/episode_reward_mean',
        'Episode Length': 'episode_lengths/mean',
        'Position Tracking': 'Episode/Episode_Reward/position_tracking',
        'Fingers to Object': 'Episode/Episode_Reward/fingers_to_object',
    }

    for name, tag in key_metrics.items():
        if tag in tags:
            try:
                events = ea.Scalars(tag)
                values = np.array([e.value for e in events])
                if len(values) > 0:
                    recent = values[-100:] if len(values) >= 100 else values
                    print(f"  {name:20s}: mean={np.mean(recent):8.4f}, latest={values[-1]:8.4f}, trend={'↑' if len(values) > 10 and values[-1] > np.mean(values[:10]) else '↓'}")
            except:
                pass

    # Check for abnormal termination
    if 'Episode/Episode_Reward/early_termination' in tags:
        try:
            events = ea.Scalars('Episode/Episode_Reward/early_termination')
            values = np.array([e.value for e in events])
            recent = values[-100:] if len(values) >= 100 else values
            print(f"\n⚠️  Early Termination: mean={np.mean(recent):8.4f} (negative means frequent abnormal termination)")
        except:
            pass


def main():
    inspire_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_inspire_right/2025-12-15_15-23-45/summaries"
    dg5f_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right/2025-11-30_05-42-14/summaries"

    analyze_all_rewards(inspire_log, "Inspire Hand (Current Training)")
    analyze_all_rewards(dg5f_log, "DG5F Hand (Past Training)")

    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print("\n🔴 CRITICAL ISSUE: good_finger_contact ≈ 0 for BOTH hands!")
    print("\nPossible causes:")
    print("  1. Reward threshold too high (currently threshold=1.0N)")
    print("  2. Contact sensors not detecting contact")
    print("  3. Policy avoiding contact (learned wrong strategy)")
    print("  4. Object falling too fast (gravity curriculum issue)")
    print("\n💡 Next steps:")
    print("  1. Check if success reward is non-zero (object reaching target without contact?)")
    print("  2. Check episode_length (too short = early termination)")
    print("  3. Check position_tracking (high = object moving but no contact?)")
    print("  4. Inspect actual training video to see what's happening")


if __name__ == "__main__":
    main()
