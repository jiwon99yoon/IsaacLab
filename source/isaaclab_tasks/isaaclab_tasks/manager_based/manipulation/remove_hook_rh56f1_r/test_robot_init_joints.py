#!/usr/bin/env python3
"""
Test script to verify robot spawns with correct init joint positions.

This script:
1. Spawns the remove_hook environment
2. Checks robot joint positions at Step 0
3. Resets the environment
4. Checks robot joint positions after reset
5. Verifies they match the expected init positions from robot cfg
"""

import torch
import math

# Import the environment
from config.hdr35_20_rh56f1_r.remove_hdr35_20_rh56f1_r_env_cfg import RemoveHookHdr35Env_Cfg

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnv

def main():
    # Expected init joint positions (from robot cfg)
    expected_joint_pos = {
        "j1": math.radians(109),   # 1.902 rad
        "j2": math.radians(74),    # 1.292 rad
        "j3": math.radians(-46),   # -0.803 rad
        "j4": math.radians(-72),   # -1.257 rad
        "j5": math.radians(76),    # 1.326 rad
        "j6": math.radians(54),    # 0.942 rad
    }

    print("=" * 80)
    print("Testing Robot Init Joint Positions")
    print("=" * 80)

    # Create environment
    print("\n[1] Creating environment...")
    env_cfg = RemoveHookHdr35Env_Cfg()
    env_cfg.scene.num_envs = 1  # Single environment for testing
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Get robot
    robot = env.scene["robot"]

    # Check joint positions at spawn
    print("\n[2] Checking joint positions at spawn (Step 0)...")
    joint_pos = robot.data.joint_pos[0].cpu()  # Get first env

    # Get joint names
    joint_names = robot.data.joint_names
    arm_joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]

    print(f"\nRobot joint names: {joint_names}")
    print(f"\nChecking arm joints (j1-j6):")
    print(f"{'Joint':<8} {'Expected (rad)':<15} {'Actual (rad)':<15} {'Expected (deg)':<15} {'Actual (deg)':<15} {'Match?':<10}")
    print("-" * 80)

    all_match = True
    for joint_name in arm_joint_names:
        if joint_name in joint_names:
            joint_idx = joint_names.index(joint_name)
            expected_rad = expected_joint_pos[joint_name]
            actual_rad = joint_pos[joint_idx].item()
            expected_deg = math.degrees(expected_rad)
            actual_deg = math.degrees(actual_rad)

            # Check if they match (within tolerance)
            tolerance = 0.01  # 0.01 rad ≈ 0.57°
            match = abs(expected_rad - actual_rad) < tolerance
            all_match = all_match and match

            match_str = "✓ YES" if match else "✗ NO"
            print(f"{joint_name:<8} {expected_rad:<15.3f} {actual_rad:<15.3f} {expected_deg:<15.1f} {actual_deg:<15.1f} {match_str:<10}")

    print("\n" + "=" * 80)
    if all_match:
        print("✓ SUCCESS: All arm joints match expected init positions!")
    else:
        print("✗ FAILURE: Some arm joints do NOT match expected init positions!")
    print("=" * 80)

    # Test reset behavior
    print("\n[3] Testing reset behavior...")
    print("Performing environment reset...")

    # Reset environment
    env.reset()

    # Check joint positions after reset
    print("\n[4] Checking joint positions after reset...")
    joint_pos_after = robot.data.joint_pos[0].cpu()

    print(f"\nChecking arm joints after reset:")
    print(f"{'Joint':<8} {'Expected (rad)':<15} {'Actual (rad)':<15} {'Expected (deg)':<15} {'Actual (deg)':<15} {'Match?':<10}")
    print("-" * 80)

    all_match_after = True
    for joint_name in arm_joint_names:
        if joint_name in joint_names:
            joint_idx = joint_names.index(joint_name)
            expected_rad = expected_joint_pos[joint_name]
            actual_rad = joint_pos_after[joint_idx].item()
            expected_deg = math.degrees(expected_rad)
            actual_deg = math.degrees(actual_rad)

            # Check if they match (within tolerance)
            match = abs(expected_rad - actual_rad) < tolerance
            all_match_after = all_match_after and match

            match_str = "✓ YES" if match else "✗ NO"
            print(f"{joint_name:<8} {expected_rad:<15.3f} {actual_rad:<15.3f} {expected_deg:<15.1f} {actual_deg:<15.1f} {match_str:<10}")

    print("\n" + "=" * 80)
    if all_match_after:
        print("✓ SUCCESS: Reset correctly restores init positions!")
    else:
        print("✗ FAILURE: Reset does NOT restore init positions!")
    print("=" * 80)

    # Cleanup
    env.close()

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    if all_match and all_match_after:
        print("✓ PASS: Robot spawns and resets with correct init joint positions!")
        print("\nYou can now run RL training with confidence that the robot")
        print("will start each episode in the correct pose for wire grasping.")
        return 0
    else:
        print("✗ FAIL: Robot init joint positions are incorrect!")
        print("\nPlease check:")
        print("1. Robot cfg file has correct init_state.joint_pos")
        print("2. Environment config uses correct robot cfg")
        print("3. reset_robot_joints event is enabled with [0.0, 0.0] ranges")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
