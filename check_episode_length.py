#!/usr/bin/env python3
"""Check episode length to see if episodes are terminating early."""

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def check_episode_length(log_path, hand_name, max_episode_length=240):
    """Check episode length distribution."""

    print(f"\n{'='*80}")
    print(f"{hand_name}")
    print(f"Max episode length: {max_episode_length} steps (4.0s @ 60Hz)")
    print(f"{'='*80}\n")

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()

    tags = ea.Tags()['scalars']

    # Find episode length tag
    length_tags = [tag for tag in tags if 'episode_length' in tag.lower() or 'ep_len' in tag.lower()]

    if not length_tags:
        print("⚠️  No episode length tags found!")
        return

    for tag in length_tags:
        try:
            events = ea.Scalars(tag)
            values = np.array([e.value for e in events])

            if len(values) == 0:
                continue

            recent = values[-1000:] if len(values) >= 1000 else values

            print(f"📊 {tag}:")
            print(f"  Mean:           {np.mean(recent):8.2f} steps ({np.mean(recent)/60:.2f}s)")
            print(f"  Std:            {np.std(recent):8.2f} steps")
            print(f"  Min:            {np.min(recent):8.2f} steps")
            print(f"  Max:            {np.max(recent):8.2f} steps")
            print(f"  Median:         {np.median(recent):8.2f} steps")
            print(f"  % Full Length:  {100 * np.sum(recent >= max_episode_length) / len(recent):.1f}%")
            print(f"  % Very Short:   {100 * np.sum(recent < 50) / len(recent):.1f}% (<50 steps)")

            # Histogram
            print(f"\n  Distribution:")
            bins = [0, 50, 100, 150, 200, max_episode_length, 999]
            labels = ['0-50', '50-100', '100-150', '150-200', '200-240', '240+']
            hist, _ = np.histogram(recent, bins=bins)
            for i, (label, count) in enumerate(zip(labels, hist)):
                pct = 100 * count / len(recent)
                bar = '█' * int(pct / 2)
                print(f"    {label:10s}: {bar} {pct:5.1f}% ({count} episodes)")

        except Exception as e:
            print(f"  Error: {e}")

    # Check abnormal termination rate
    if 'Episode/Episode_Reward/early_termination' in tags:
        try:
            events = ea.Scalars('Episode/Episode_Reward/early_termination')
            values = np.array([e.value for e in events])
            recent = values[-1000:] if len(values) >= 1000 else values

            # early_termination reward is negative when abnormal
            abnormal_rate = 100 * np.sum(recent < -0.001) / len(recent)
            print(f"\n⚠️  Abnormal Termination Rate: {abnormal_rate:.1f}%")

        except:
            pass


def main():
    inspire_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_inspire_right/2025-12-15_15-23-45/summaries"
    dg5f_log = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right/2025-11-30_05-42-14/summaries"

    # Episode length = 4.0s @ dt=1/120 with decimation=2 → 4.0 / (1/60) = 240 steps
    max_ep_len = 240

    check_episode_length(inspire_log, "Inspire Hand (Current Training)", max_ep_len)
    check_episode_length(dg5f_log, "DG5F Hand (Past Training)", max_ep_len)

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("\n🎯 What to look for:")
    print("  - If most episodes reach 240 steps → episodes completing normally")
    print("  - If most episodes < 100 steps → early termination (abnormal_robot_state)")
    print("  - If episodes very short (<50) → physics explosion or immediate failure")
    print("\n💡 Expected behavior:")
    print("  - Early training: episodes should be short (agent doesn't know what to do)")
    print("  - Later training: episodes should get longer as agent learns")
    print("  - Success → episode can end early (object reached target)")


if __name__ == "__main__":
    main()
