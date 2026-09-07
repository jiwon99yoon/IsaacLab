"""
Simple test to verify cuRobo works with Isaac Lab
WITHOUT pytest (direct python execution)
"""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

print("=" * 80)
print("Isaac Lab + cuRobo 통합 테스트")
print("=" * 80)

# Create simple environment
try:
    print("\n[1/4] Creating Isaac Lab environment...")
    env = gym.make("Isaac-Reach-Franka-v0", num_envs=1, headless=False)
    print("✅ Environment created successfully")

    print("\n[2/4] Getting robot articulation...")
    robot = env.unwrapped.scene["robot"]
    print(f"✅ Robot: {robot}")

    print("\n[3/4] Initializing cuRobo planner...")
    planner_cfg = CuroboPlannerCfg.franka_config()
    planner = CuroboPlanner(env=env.unwrapped, robot=robot, config=planner_cfg)
    print("✅ cuRobo planner initialized successfully!")

    print("\n[4/4] Testing motion planning...")
    # Set a simple target pose
    import isaaclab.utils.math as math_utils
    target_pos = torch.tensor([0.5, 0.0, 0.4], device=env.unwrapped.device)
    target_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.unwrapped.device)

    rot_matrix = math_utils.matrix_from_quat(target_quat.unsqueeze(0))[0]
    target_pose = math_utils.make_pose(target_pos, rot_matrix)

    planner.update_world()
    success = planner.plan_motion(target_pose)

    if success:
        print("✅ Motion planning SUCCESSFUL!")
        print(f"   Planned trajectory has {len(planner.current_plan.position)} waypoints")
    else:
        print("❌ Motion planning FAILED")

    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    if 'env' in locals():
        env.close()
    simulation_app.close()
