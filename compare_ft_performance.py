#!/usr/bin/env python3
"""Compare performance of DG5F experiments with/without FT sensor."""

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Experiments classification (from previous analysis)
WITH_FT = [
    '2025-11-24_11-40-29',
    '2025-11-26_02-41-56',
    '2025-11-27_15-15-51',
    '2025-11-28_09-56-05',
    '2025-11-29_01-42-36',
    '2025-11-30_05-42-14',
    '2025-12-03_16-12-39',
]

WITHOUT_FT = [
    '2025-11-18_20-51-17',
    '2025-11-19_16-03-05',
    '2025-11-20_23-27-37',
    '2025-11-23_23-31-29',
    '2025-11-24_10-23-07',
    '2025-11-29_16-17-49',
    '2025-12-15_15-11-05',
]


def get_metrics(log_dir):
    """Extract performance metrics from tensorboard logs."""
    summaries_dir = f"logs/rl_games/dexsuite_ur10e_dg5f_right/{log_dir}/summaries"

    try:
        ea = event_accumulator.EventAccumulator(summaries_dir)
        ea.Reload()
        tags = ea.Tags()['scalars']

        metrics = {}

        # Key metrics
        metric_map = {
            'success': 'Episode/Episode_Reward/success',
            'contact': 'Episode/Episode_Reward/good_finger_contact',
            'position': 'Episode/Episode_Reward/position_tracking',
            'episode_reward': 'Train/mean_rewards/episode_reward_mean',
            'ep_len': 'episode_lengths/mean',
        }

        for name, tag in metric_map.items():
            if tag in tags:
                try:
                    events = ea.Scalars(tag)
                    values = np.array([e.value for e in events])
                    if len(values) > 0:
                        recent = values[-100:] if len(values) >= 100 else values
                        metrics[name] = {
                            'mean': np.mean(recent),
                            'final': values[-1],
                            'epochs': len(values),
                        }
                except:
                    pass

        # FT specific metrics (only for WITH_FT)
        if 'Episode/Episode_Reward/ati_excessive_force_penalty' in tags:
            try:
                events = ea.Scalars('Episode/Episode_Reward/ati_excessive_force_penalty')
                values = np.array([e.value for e in events])
                if len(values) > 0:
                    recent = values[-100:] if len(values) >= 100 else values
                    metrics['ft_penalty'] = {
                        'mean': np.mean(recent),
                        'final': values[-1],
                    }
            except:
                pass

        return metrics
    except Exception as e:
        return {'error': str(e)}


def main():
    print("="*80)
    print("DG5F PERFORMANCE COMPARISON: WITH vs WITHOUT FT Sensor")
    print("="*80)

    # Collect metrics for each group
    with_ft_metrics = []
    without_ft_metrics = []

    print("\n🔧 WITH FT SENSOR:")
    print("-"*80)
    for exp in WITH_FT:
        metrics = get_metrics(exp)
        if 'error' not in metrics and metrics:
            with_ft_metrics.append(metrics)
            print(f"  {exp}:")
            if 'success' in metrics:
                print(f"    Success:  {metrics['success']['mean']:7.4f} (final: {metrics['success']['final']:7.4f}, epochs: {metrics['success']['epochs']})")
            if 'contact' in metrics:
                print(f"    Contact:  {metrics['contact']['mean']:7.4f}")
            if 'position' in metrics:
                print(f"    Position: {metrics['position']['mean']:7.4f}")
            if 'ft_penalty' in metrics:
                print(f"    FT Penalty: {metrics['ft_penalty']['mean']:7.4f}")
        else:
            print(f"  {exp}: ⚠️  Error or no data")

    print("\n✅ WITHOUT FT SENSOR:")
    print("-"*80)
    for exp in WITHOUT_FT:
        metrics = get_metrics(exp)
        if 'error' not in metrics and metrics:
            without_ft_metrics.append(metrics)
            print(f"  {exp}:")
            if 'success' in metrics:
                print(f"    Success:  {metrics['success']['mean']:7.4f} (final: {metrics['success']['final']:7.4f}, epochs: {metrics['success']['epochs']})")
            if 'contact' in metrics:
                print(f"    Contact:  {metrics['contact']['mean']:7.4f}")
            if 'position' in metrics:
                print(f"    Position: {metrics['position']['mean']:7.4f}")
        else:
            print(f"  {exp}: ⚠️  Error or no data")

    # Aggregate comparison
    print("\n" + "="*80)
    print("📊 AGGREGATE COMPARISON")
    print("="*80)

    metric_names = ['success', 'contact', 'position', 'ep_len']

    for metric in metric_names:
        with_values = [m[metric]['mean'] for m in with_ft_metrics if metric in m]
        without_values = [m[metric]['mean'] for m in without_ft_metrics if metric in m]

        if with_values and without_values:
            print(f"\n{metric.upper()}:")
            print(f"  WITH FT:    mean={np.mean(with_values):8.4f}, std={np.std(with_values):8.4f}, n={len(with_values)}")
            print(f"  WITHOUT FT: mean={np.mean(without_values):8.4f}, std={np.std(without_values):8.4f}, n={len(without_values)}")

            diff = np.mean(without_values) - np.mean(with_values)
            pct_diff = 100 * diff / (abs(np.mean(with_values)) + 1e-9)

            if metric != 'ep_len':
                if abs(diff) > 0.0001:  # Only if meaningful difference
                    better = "WITHOUT FT" if diff > 0 else "WITH FT"
                    print(f"  → {better} is BETTER by {abs(diff):8.4f} ({abs(pct_diff):6.1f}%)")
                else:
                    print(f"  → Similar performance")

    # Detailed analysis
    print("\n" + "="*80)
    print("🎯 DETAILED ANALYSIS")
    print("="*80)

    # Best performing experiments
    if with_ft_metrics:
        best_with_ft = max(with_ft_metrics, key=lambda m: m.get('success', {}).get('mean', -999))
        print(f"\n🏆 Best WITH FT:")
        print(f"   Experiment: {WITH_FT[with_ft_metrics.index(best_with_ft)]}")
        if 'success' in best_with_ft:
            print(f"   Success:    {best_with_ft['success']['mean']:.4f}")
        if 'contact' in best_with_ft:
            print(f"   Contact:    {best_with_ft['contact']['mean']:.4f}")

    if without_ft_metrics:
        best_without_ft = max(without_ft_metrics, key=lambda m: m.get('success', {}).get('mean', -999))
        print(f"\n🏆 Best WITHOUT FT:")
        print(f"   Experiment: {WITHOUT_FT[without_ft_metrics.index(best_without_ft)]}")
        if 'success' in best_without_ft:
            print(f"   Success:    {best_without_ft['success']['mean']:.4f}")
        if 'contact' in best_without_ft:
            print(f"   Contact:    {best_without_ft['contact']['mean']:.4f}")

    # Conclusion
    print("\n" + "="*80)
    print("💡 CONCLUSION")
    print("="*80)

    success_with = [m['success']['mean'] for m in with_ft_metrics if 'success' in m]
    success_without = [m['success']['mean'] for m in without_ft_metrics if 'success' in m]

    if success_with and success_without:
        if np.mean(success_without) > np.mean(success_with) + 0.001:
            print("\n🔴 FT sensor appears to HINDER learning!")
            print("   → Experiments WITHOUT FT sensor perform better on average")
            print("   → Recommendation: Use configuration WITHOUT FT sensor for comparison")
        elif np.mean(success_with) > np.mean(success_without) + 0.001:
            print("\n🟢 FT sensor appears to HELP learning!")
            print("   → Experiments WITH FT sensor perform better on average")
            print("   → Recommendation: Use configuration WITH FT sensor for comparison")
        else:
            print("\n🟡 FT sensor has MINIMAL impact")
            print("   → Both configurations show similar performance")
            print("   → Either configuration can be used for comparison")

    print("\n📋 For논문 작성:")
    print("   → Use the BEST performing configuration (highest success rate)")
    print("   → Document which configuration was used")
    print("   → Mention FT sensor presence/absence in methodology")


if __name__ == "__main__":
    main()
