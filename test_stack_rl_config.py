"""Test script to verify stack_rl environment config loads correctly"""

from isaaclab_tasks.manager_based.manipulation.stack_rl.config.franka.joint_pos_env_cfg import FrankaCubeStackRLEnvCfg

print("=" * 60)
print("Testing Stack RL Environment Configuration")
print("=" * 60)

# Load config
cfg = FrankaCubeStackRLEnvCfg()
print("✓ Environment config loaded successfully\n")

# Check observations
obs_terms = [k for k in dir(cfg.observations.policy) if not k.startswith("_") and k not in ["enable_corruption", "concatenate_terms"]]
print(f"✓ Observation terms ({len(obs_terms)}):")
for term in obs_terms:
    print(f"  - {term}")

# Check rewards
rew_terms = [k for k in dir(cfg.rewards) if not k.startswith("_") and hasattr(getattr(cfg.rewards, k), "func")]
print(f"\n✓ Reward terms ({len(rew_terms)}):")
for term in rew_terms:
    rew = getattr(cfg.rewards, term)
    print(f"  - {term}: weight={rew.weight}")

# Check key config values
print(f"\n✓ Gripper config:")
print(f"  - gripper_joint_names: {cfg.gripper_joint_names}")
print(f"  - gripper_open_val: {cfg.gripper_open_val}")
print(f"  - gripper_threshold: {cfg.gripper_threshold}")

print("\n" + "=" * 60)
print("All checks passed! Environment config is valid.")
print("=" * 60)
