#!/usr/bin/env python3
"""
Classify DG5F experiments by FT sensor presence and compare performance.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def check_ft_sensor(env_yaml_path):
    """Check if environment has FT sensor observations."""
    try:
        with open(env_yaml_path, 'r') as f:
            env_config = yaml.safe_load(f)

        # Check observations config
        obs_config = env_config.get('observations', {})
        proprio = obs_config.get('proprio', {})

        # Check for FT sensor specific observations
        has_ati_force = 'ati_force' in proprio
        has_ati_force_threshold = 'ati_force_threshold' in proprio
        has_ati_force_torque = 'ati_force_torque' in proprio

        has_ft = has_ati_force or has_ati_force_threshold or has_ati_force_torque

        return has_ft, {
            'ati_force': has_ati_force,
            'ati_force_threshold': has_ati_force_threshold,
            'ati_force_torque': has_ati_force_torque
        }
    except Exception as e:
        return None, {'error': str(e)}


def get_performance_metrics(log_dir):
    """Extract key performance metrics from tensorboard logs."""
    summaries_dir = os.path.join(log_dir, 'summaries')

    if not os.path.exists(summaries_dir):
        return None

    try:
        ea = event_accumulator.EventAccumulator(summaries_dir)
        ea.Reload()

        tags = ea.Tags()['scalars']

        metrics = {}

        # Key metrics to extract
        metric_tags = {
            'success': 'Episode/Episode_Reward/success',
            'good_finger_contact': 'Episode/Episode_Reward/good_finger_contact',
            'position_tracking': 'Episode/Episode_Reward/position_tracking',
            'episode_reward': 'Train/mean_rewards/episode_reward_mean',
            'episode_length': 'episode_lengths/mean',
            'early_termination': 'Episode/Episode_Reward/early_termination',
        }

        # FT sensor specific metrics
        if 'Episode/Episode_Reward/ati_excessive_force_penalty' in tags:
            metric_tags['ati_excessive_force_penalty'] = 'Episode/Episode_Reward/ati_excessive_force_penalty'
        if 'Episode/Episode_Reward/ati_force_magnitude' in tags:
            metric_tags['ati_force_magnitude'] = 'Episode/Episode_Reward/ati_force_magnitude'

        for metric_name, tag in metric_tags.items():
            if tag in tags:
                try:
                    events = ea.Scalars(tag)
                    values = np.array([e.value for e in events])

                    if len(values) > 0:
                        # Get last 100 values for current performance
                        recent = values[-100:] if len(values) >= 100 else values

                        metrics[metric_name] = {
                            'mean': np.mean(recent),
                            'std': np.std(recent),
                            'min': np.min(recent),
                            'max': np.max(recent),
                            'final': values[-1],
                            'total_points': len(values),
                        }
                except:
                    pass

        return metrics

    except Exception as e:
        return {'error': str(e)}


def main():
    base_dir = "/home/dyros/IsaacLab/logs/rl_games/dexsuite_ur10e_dg5f_right"

    # Find all experiment directories
    exp_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir() and d.name.startswith('2025-')])

    print("="*80)
    print("DG5F EXPERIMENT CLASSIFICATION: FT Sensor Presence Analysis")
    print("="*80)

    with_ft = []
    without_ft = []
    unknown = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        env_yaml = exp_dir / 'params' / 'env.yaml'

        if not env_yaml.exists():
            unknown.append({'dir': exp_name, 'reason': 'No env.yaml'})
            continue

        has_ft, ft_details = check_ft_sensor(str(env_yaml))

        if has_ft is None:
            unknown.append({'dir': exp_name, 'reason': ft_details.get('error', 'Unknown')})
            continue

        # Get performance metrics
        metrics = get_performance_metrics(str(exp_dir))

        exp_info = {
            'dir': exp_name,
            'ft_details': ft_details,
            'metrics': metrics
        }

        if has_ft:
            with_ft.append(exp_info)
        else:
            without_ft.append(exp_info)

    # Print results
    print(f"\n📊 CLASSIFICATION RESULTS:")
    print(f"  Total experiments: {len(exp_dirs)}")
    print(f"  With FT sensor:    {len(with_ft)}")
    print(f"  Without FT sensor: {len(without_ft)}")
    print(f"  Unknown:           {len(unknown)}")

    # Print WITH FT sensor experiments
    print(f"\n" + "="*80)
    print(f"🔧 WITH FT SENSOR ({len(with_ft)} experiments)")
    print("="*80)

    for exp in with_ft:
        print(f"\n📁 {exp['dir']}")
        print(f"   FT observations: {', '.join([k for k, v in exp['ft_details'].items() if v])}")

        if exp['metrics'] and 'error' not in exp['metrics']:
            m = exp['metrics']
            print(f"   Performance (last 100 episodes):")

            if 'success' in m:
                print(f"     Success:           {m['success']['mean']:8.4f}")
            if 'good_finger_contact' in m:
                print(f"     Contact:           {m['good_finger_contact']['mean']:8.4f}")
            if 'position_tracking' in m:
                print(f"     Position Tracking: {m['position_tracking']['mean']:8.4f}")
            if 'ati_excessive_force_penalty' in m:
                print(f"     FT Force Penalty:  {m['ati_excessive_force_penalty']['mean']:8.4f}")
            if 'episode_length' in m:
                print(f"     Episode Length:    {m['episode_length']['mean']:8.1f} steps")

            print(f"     Total logged points: {m.get('success', {}).get('total_points', 'N/A')}")
        elif exp['metrics']:
            print(f"   ⚠️  Error loading metrics: {exp['metrics'].get('error', 'Unknown')}")
        else:
            print(f"   ⚠️  No metrics found")

    # Print WITHOUT FT sensor experiments
    print(f"\n" + "="*80)
    print(f"✅ WITHOUT FT SENSOR ({len(without_ft)} experiments)")
    print("="*80)

    for exp in without_ft:
        print(f"\n📁 {exp['dir']}")

        if exp['metrics'] and 'error' not in exp['metrics']:
            m = exp['metrics']
            print(f"   Performance (last 100 episodes):")

            if 'success' in m:
                print(f"     Success:           {m['success']['mean']:8.4f}")
            if 'good_finger_contact' in m:
                print(f"     Contact:           {m['good_finger_contact']['mean']:8.4f}")
            if 'position_tracking' in m:
                print(f"     Position Tracking: {m['position_tracking']['mean']:8.4f}")
            if 'episode_length' in m:
                print(f"     Episode Length:    {m['episode_length']['mean']:8.1f} steps")

            print(f"     Total logged points: {m.get('success', {}).get('total_points', 'N/A')}")
        elif exp['metrics']:
            print(f"   ⚠️  Error loading metrics: {exp['metrics'].get('error', 'Unknown')}")
        else:
            print(f"   ⚠️  No metrics found")

    # Comparison
    if with_ft and without_ft:
        print(f"\n" + "="*80)
        print(f"📈 PERFORMANCE COMPARISON")
        print("="*80)

        # Aggregate metrics
        def aggregate_metric(experiments, metric_name):
            values = []
            for exp in experiments:
                if exp['metrics'] and metric_name in exp['metrics']:
                    values.append(exp['metrics'][metric_name]['mean'])
            return values

        metrics_to_compare = ['success', 'good_finger_contact', 'position_tracking', 'episode_length']

        for metric in metrics_to_compare:
            with_values = aggregate_metric(with_ft, metric)
            without_values = aggregate_metric(without_ft, metric)

            if with_values and without_values:
                print(f"\n{metric.upper()}:")
                print(f"  With FT:    mean={np.mean(with_values):8.4f}, std={np.std(with_values):8.4f}, n={len(with_values)}")
                print(f"  Without FT: mean={np.mean(without_values):8.4f}, std={np.std(without_values):8.4f}, n={len(without_values)}")

                diff = np.mean(without_values) - np.mean(with_values)
                if metric != 'episode_length':
                    if diff > 0:
                        print(f"  → WITHOUT FT is BETTER by {diff:8.4f}")
                    else:
                        print(f"  → WITH FT is BETTER by {-diff:8.4f}")

        print(f"\n🎯 CONCLUSION:")
        print(f"  Check which configuration performs better!")
        print(f"  If 'Without FT' consistently outperforms, FT sensor may be hindering learning.")

    # Print unknown experiments
    if unknown:
        print(f"\n" + "="*80)
        print(f"❓ UNKNOWN ({len(unknown)} experiments)")
        print("="*80)
        for exp in unknown:
            print(f"  {exp['dir']}: {exp['reason']}")


if __name__ == "__main__":
    main()
