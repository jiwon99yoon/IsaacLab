# Implementation Details: Controlled Comparison Setup

## Overview

This document provides the exact implementation details for our controlled experimental comparison, ensuring reproducibility and transparency.

---

## 1. Hardware Configurations

### 1.1 UR10e + Inspire Hand (Underactuated, 6-DoF)

**File Locations**:
- Robot config: `source/isaaclab_assets/isaaclab_assets/robots/ur10e_inspire_right.py`
- Environment config: `source/isaaclab_tasks/.../dexsuite_ur10e/config/ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py`
- Gym registration: `Isaac-Dexsuite-Ur10e-Inspire-Lift-v0`

**Hand Specifications**:
```python
# Inspire Hand RH56F1-R (Right)
Total DoF: 6 actuated + mimic joints
Fingers: 5 (thumb, index, middle, ring, little)
- Thumb: 2 actuated joints (abduction + flexion)
- Other fingers: 1 actuated joint each (flexion)
- Mimic joints: Additional passive joints coupled to actuated joints

Control Dimensionality: 12-dim
- 6 arm joints (UR10e)
- 6 hand joints (Inspire)
```

**Arm Initial Configuration** (CRITICAL - matches DG5F):
```python
joint_pos={
    # === Arm (UR10e - 6-DOF) ===
    "shoulder_pan_joint": 0.0,          # 0° - Centered
    "shoulder_lift_joint": -2.0944,     # -120° - Raised
    "elbow_joint": 2.0944,              # 120° - Bent
    "wrist_1_joint": -2.0944,           # -120° - Pitch down
    "wrist_2_joint": 3.14159,           # 180° - ✅ ALIGNED with DG5F
    "wrist_3_joint": -3.14159,          # -180° - ✅ ALIGNED with DG5F

    # === Hand (Inspire Right - 6 actuated DOF) ===
    "right_thumb_1_joint": 1.0,         # Thumb abduction
    "right_thumb_2_joint": 0.0,         # Thumb flexion
    "right_index_1_joint": 0.4,         # Index flexion
    "right_middle_1_joint": 0.3,        # Middle flexion
    "right_ring_1_joint": 0.3,          # Ring flexion
    "right_little_1_joint": 0.3,        # Little flexion
}
```

**Actuator Parameters**:
```python
actuators={
    "ur10e_inspire_actuators": ImplicitActuatorCfg(
        joint_names_expr=[
            r"shoulder_pan_joint", r"shoulder_lift_joint", r"elbow_joint",
            r"wrist_[123]_joint",
            r"right_thumb_(1|2)_joint",
            r"right_(index|middle|ring|little)_1_joint",
        ],
        effort_limit_sim={
            # Arm (UR10e official specs)
            r"shoulder_pan_joint": 330.0,
            r"shoulder_lift_joint": 330.0,
            r"elbow_joint": 150.0,
            r"wrist_1_joint": 54.0,
            r"wrist_2_joint": 54.0,
            r"wrist_3_joint": 54.0,
            # Hand (Inspire - underactuated, low effort)
            r"right_thumb_(1|2)_joint": 1.0,
            r"right_(index|middle|ring|little)_1_joint": 1.0,
        },
        stiffness={
            # Arm
            r"shoulder_pan_joint": 1320.0,
            r"shoulder_lift_joint": 1320.0,
            r"elbow_joint": 600.0,
            r"wrist_1_joint": 200.0,
            r"wrist_2_joint": 200.0,
            r"wrist_3_joint": 200.0,
            # Hand (low stiffness for compliance)
            r"right_thumb_(1|2)_joint": 2.5,
            r"right_(index|middle|ring|little)_1_joint": 2.5,
        },
        damping={
            # Arm
            r"shoulder_pan_joint": 72.66,
            r"shoulder_lift_joint": 72.66,
            r"elbow_joint": 34.64,
            r"wrist_1_joint": 27.0,
            r"wrist_2_joint": 27.0,
            r"wrist_3_joint": 27.0,
            # Hand
            r"right_thumb_(1|2)_joint": 0.25,
            r"right_(index|middle|ring|little)_1_joint": 0.25,
        },
    ),
}
```

**Object Sizes** (0.5x scaling):
```python
class InspireSceneCfg(dexsuite.SceneCfg):
    object: RigidObjectCfg = RigidObjectCfg(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # Cuboids (Kuka-Allegro × 0.5)
                CuboidCfg(size=(0.025, 0.05, 0.05)),   # was (0.05, 0.1, 0.1)
                CuboidCfg(size=(0.025, 0.025, 0.05)),  # was (0.05, 0.05, 0.1)
                # ... (all objects scaled to 0.5x)

                # Spheres (Kuka-Allegro × 0.5)
                SphereCfg(radius=0.025),  # was 0.05
                SphereCfg(radius=0.0125), # was 0.025
                # ... (all objects scaled to 0.5x)
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
    )
```

---

### 1.2 UR10e + DG5F Hand (Fully-Actuated, 20-DoF)

**File Locations**:
- Robot config: `source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f_right.py`
- Environment config: `source/isaaclab_tasks/.../dexsuite_ur10e/config/ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py`
- Gym registration: `Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0`

**Hand Specifications**:
```python
# DG5F Hand (Tesollo) - Right
Total DoF: 20 actuated (fully independent)
Fingers: 5 (each with 4 joints)
- Finger 1 (Thumb): 4 actuated joints (spread + 3x flexion)
- Fingers 2-5: 4 actuated joints each (spread + 3x flexion)

Control Dimensionality: 26-dim
- 6 arm joints (UR10e)
- 20 hand joints (DG5F)
```

**Arm Initial Configuration** (IDENTICAL to Inspire):
```python
joint_pos={
    # === Arm (UR10e - 6-DOF) - IDENTICAL TO INSPIRE ===
    "shoulder_pan_joint": deg2rad(0.0),      # 0° - Centered
    "shoulder_lift_joint": deg2rad(-120.0),  # -120° - Raised
    "elbow_joint": deg2rad(120.0),           # 120° - Bent
    "wrist_1_joint": deg2rad(-120.0),        # -120° - Pitch down
    "wrist_2_joint": deg2rad(180.0),         # 180° - ✅ ALIGNED with Inspire
    "wrist_3_joint": deg2rad(-180.0),        # -180° - ✅ ALIGNED with Inspire

    # === Hand (DG5F Right - 20 actuated DOF) ===
    # Finger 1 (Thumb)
    "rj_dg_1_1": deg2rad(10.0),    # Spread
    "rj_dg_1_2": deg2rad(-60.0),   # Flexion 1
    "rj_dg_1_3": deg2rad(15.0),    # Flexion 2
    "rj_dg_1_4": deg2rad(20.0),    # Flexion 3

    # Fingers 2-4 (Index, Middle, Ring)
    "rj_dg_2_1": deg2rad(-10.0),   # Spread
    "rj_dg_3_1": deg2rad(0.0),     # Spread
    "rj_dg_4_1": deg2rad(5.0),     # Spread
    r"rj_dg_[2-4]_2": deg2rad(30.0),  # Flexion 1
    r"rj_dg_[2-4]_3": deg2rad(20.0),  # Flexion 2
    r"rj_dg_[2-4]_4": deg2rad(30.0),  # Flexion 3

    # Finger 5 (Little)
    "rj_dg_5_1": deg2rad(10.0),    # Spread
    "rj_dg_5_2": deg2rad(20.0),    # Flexion 1
    "rj_dg_5_3": deg2rad(30.0),    # Flexion 2
    "rj_dg_5_4": deg2rad(30.0),    # Flexion 3
}
```

**Actuator Parameters**:
```python
actuators={
    "ur10e_dg5f_actuators": ImplicitActuatorCfg(
        joint_names_expr=[
            r"shoulder_pan_joint", r"shoulder_lift_joint", r"elbow_joint",
            r"wrist_[123]_joint",
            r"rj_dg_[1-5]_[1-4]",  # All 20 hand joints
        ],
        effort_limit_sim={
            # Arm (IDENTICAL to Inspire)
            r"shoulder_pan_joint": 330.0,
            r"shoulder_lift_joint": 330.0,
            r"elbow_joint": 150.0,
            r"wrist_1_joint": 54.0,
            r"wrist_2_joint": 54.0,
            r"wrist_3_joint": 54.0,
            # Hand (DG5F - fully actuated, higher effort)
            r"rj_dg_[1-5]_[1-4]": 5.0,
        },
        stiffness={
            # Arm (IDENTICAL to Inspire)
            r"shoulder_pan_joint": 1320.0,
            r"shoulder_lift_joint": 1320.0,
            r"elbow_joint": 600.0,
            r"wrist_1_joint": 200.0,
            r"wrist_2_joint": 200.0,
            r"wrist_3_joint": 200.0,
            # Hand (higher than Inspire for precise control)
            r"rj_dg_[1-5]_[1-4]": 4.0,
        },
        damping={
            # Arm (IDENTICAL to Inspire)
            r"shoulder_pan_joint": 72.66,
            r"shoulder_lift_joint": 72.66,
            r"elbow_joint": 34.64,
            r"wrist_1_joint": 27.0,
            r"wrist_2_joint": 27.0,
            r"wrist_3_joint": 27.0,
            # Hand
            r"rj_dg_[1-5]_[1-4]": 0.3,
        },
    ),
}
```

**Object Sizes** (1.0x - same as Kuka-Allegro baseline):
```python
class Dg5fSceneCfg(dexsuite.SceneCfg):
    object: RigidObjectCfg = RigidObjectCfg(
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # Cuboids (Kuka-Allegro × 1.0 - baseline)
                CuboidCfg(size=(0.05, 0.1, 0.1)),
                CuboidCfg(size=(0.05, 0.05, 0.1)),
                # ... (all objects at 1.0x baseline size)

                # Spheres (Kuka-Allegro × 1.0 - baseline)
                SphereCfg(radius=0.05),
                SphereCfg(radius=0.025),
                # ... (all objects at 1.0x baseline size)
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
    )
```

---

## 2. Shared Configuration (IDENTICAL for Both)

### 2.1 Task Definition

```python
# File: dexsuite_env_cfg.py (shared base config)

class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Lift task: position-only tracking (no orientation)"""

    def __post_init__(self):
        super().__post_init__()
        # Disable orientation reward
        self.rewards.orientation_tracking = None

        # Enable position-only command
        self.commands.object_pose.position_only = True

        # Success criterion: position only
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = None
            self.curriculum.adr.params["rot_tol"] = None

        # Episode settings
        self.episode_length_s = 4.0  # 480 sim steps @ 120Hz
        self.decimation = 2          # 240 control steps
```

### 2.2 Observation Space

**Structure** (IDENTICAL, dimensions differ due to DoF):
```python
class ObservationsCfg:
    # Policy observations (history=5)
    class PolicyCfg(ObsGroup):
        object_quat_b = ObsTerm(func=mdp.object_quat_b)
        target_object_pose_b = ObsTerm(func=mdp.generated_commands)
        actions = ObsTerm(func=mdp.last_action)

        history_length = 5
        concatenate_terms = True

    # Proprioception observations (history=5)
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            clip=(-2.0, 2.0),
        )
        contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            clip=(-20.0, 20.0),  # Inspire: -20/+20, DG5F: -50/+50
        )

        history_length = 5
        concatenate_terms = True

    # Perception observations (history=5)
    class PerceptionObsCfg(ObsGroup):
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            params={"num_points": 64, "flatten": True},
            clip=(-2.0, 2.0),
        )

        history_length = 5
        concatenate_terms = True
```

**Observation Dimensions**:
```python
# Inspire (6-DoF hand)
Policy obs: 4 + 7 + 12 = 23-dim (per timestep)
Proprio obs: 12 + 12 + (6×13) + 5×3 = 24 + 78 + 15 = 117-dim (per timestep)
Perception obs: 64×3 = 192-dim (per timestep)

# DG5F (20-DoF hand)
Policy obs: 4 + 7 + 26 = 37-dim (per timestep)
Proprio obs: 26 + 26 + (6×13) + 5×3 = 52 + 78 + 15 = 145-dim (per timestep)
Perception obs: 64×3 = 192-dim (per timestep)
```

### 2.3 Reward Function

**Shared Rewards** (IDENTICAL weights and logic):
```python
class RewardsCfg:
    # Action regularization
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    # Task rewards
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.4},
        weight=1.0
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,  # Ignored in Lift task
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    early_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1,
        params={"term_keys": "abnormal_robot"}
    )

    ground_contact_penalty = RewTerm(
        func=mdp.object_ground_contact_penalty,
        weight=-0.3,
        params={"object_cfg": SceneEntityCfg("object")},
    )
```

**Hand-Specific Contact Reward** (SAME weight, different sensor names):
```python
# Inspire Hand
good_finger_contact = RewTerm(
    func=mdp.contacts,
    weight=0.5,  # Kuka-Allegro baseline
    params={
        "threshold": 1.0,
        "contact_sensor_names": [
            "ur10e_RH56F1_R_right_thumb_4_object_s",
            "ur10e_RH56F1_R_right_index_2_object_s",
            "ur10e_RH56F1_R_right_middle_2_object_s",
            "ur10e_RH56F1_R_right_ring_2_object_s",
            "ur10e_RH56F1_R_right_little_2_object_s"
        ]
    },
)

# DG5F Hand
good_finger_contact = RewTerm(
    func=mdp.contacts,
    weight=0.5,  # SAME as Inspire
    params={
        "threshold": 1.0,
        "contact_sensor_names": [
            "ur10e_dg5f_right_new_rl_dg_1_4_object_s",  # Thumb
            "ur10e_dg5f_right_new_rl_dg_2_4_object_s",  # Index
            "ur10e_dg5f_right_new_rl_dg_3_4_object_s",  # Middle
            "ur10e_dg5f_right_new_rl_dg_4_4_object_s",  # Ring
            "ur10e_dg5f_right_new_rl_dg_5_4_object_s"   # Pinky
        ]
    },
)
```

### 2.4 Domain Randomization (IDENTICAL Protocol)

```python
class EventCfg:
    # Pre-startup randomization
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    )

    # Startup randomization
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.9, 1.1],
            "dynamic_friction_range": [0.9, 1.1],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.8, 1.2],
            "damping_distribution_params": [0.8, 1.2],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    # Reset-time randomization
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.2],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]),
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Hand joint randomization (hand-specific, but SAME range)
    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.05, 0.05],  # SAME for both hands
            "velocity_range": [0.0, 0.0],
            # asset_cfg specified in hand-specific mixin
        },
    )

    # Gravity curriculum (ADR)
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )
```

### 2.5 PPO Hyperparameters (IDENTICAL)

**File**: `rl_games_ppo_cfg.yaml` (both Inspire and DG5F use SAME file)

```yaml
params:
  seed: ${...seed}  # Set via command line

  algo:
    name: a2c_continuous

  model:
    name: continuous_a2c_logstd

  network:
    name: actor_critic
    separate: False

    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0.0
        fixed_sigma: True

    mlp:
      units: [512, 256, 128]
      activation: elu
      d2rl: False

      initializer:
        name: default
      regularizer:
        name: None

  load_checkpoint: ${if:${...checkpoint},True,False}
  load_path: ${...checkpoint}

  config:
    name: ${resolve_default:Dexsuite_Ur10e,${....experiment}}
    full_experiment_name: ${.name}
    device: ${....rl_device}
    device_name: ${....rl_device}
    env_name: rlgpu
    multi_gpu: ${....multi_gpu}
    ppo: True
    mixed_precision: False
    normalize_input: True
    normalize_value: True
    num_actors: ${....task.env.num_envs}
    reward_shaper:
      scale_value: 0.01
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    e_clip: 0.2
    entropy_coef: 0.0
    learning_rate: 5.e-4
    lr_schedule: adaptive
    kl_threshold: 0.008
    truncate_grads: True
    grad_norm: 1.0
    horizon_length: 16
    minibatch_size: 16384
    mini_epochs: 8
    critic_coef: 4
    clip_value: True
    seq_len: 4
    bounds_loss_coef: 0.0001

    max_epochs: ${resolve_default:20000,${....max_iterations}}
    save_best_after: 200
    score_to_win: 20000
    save_frequency: 500
    print_stats: True
```

---

## 3. Verification Checklist

### 3.1 Arm Configuration Match
```bash
# Check wrist_2 and wrist_3 alignment
grep "wrist_2_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_inspire_right.py
grep "wrist_3_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_inspire_right.py
grep "wrist_2_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f_right.py
grep "wrist_3_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f_right.py

# Expected output:
# inspire: wrist_2 = 3.14159 (180°), wrist_3 = -3.14159 (-180°)
# dg5f:    wrist_2 = deg2rad(180.0), wrist_3 = deg2rad(-180.0)
```

### 3.2 Reward Weights Match
```bash
# Check contact reward weight
grep "good_finger_contact" -A 5 source/isaaclab_tasks/.../ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py
grep "good_finger_contact" -A 5 source/isaaclab_tasks/.../ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py

# Expected output:
# Both: weight=0.5
```

### 3.3 Object Sizes
```bash
# Check first cuboid size
grep "CuboidCfg(size=" source/isaaclab_tasks/.../ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py | head -1
grep "CuboidCfg(size=" source/isaaclab_tasks/.../ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py | head -1

# Expected output:
# inspire: (0.025, 0.05, 0.05)  # 0.5x
# dg5f:    (0.05, 0.1, 0.1)     # 1.0x
```

---

## 4. Training Commands

### 4.1 Inspire Hand
```bash
# Training (5 seeds)
for seed in 0 1 2 3 4; do
    python scripts/rsl_rl/train.py \
        --task Isaac-Dexsuite-Ur10e-Inspire-Lift-v0 \
        --num_envs 4096 \
        --seed $seed \
        --max_iterations 20000 \
        --experiment Inspire_Seed${seed}
done

# Evaluation
python scripts/rsl_rl/play.py \
    --task Isaac-Dexsuite-Ur10e-Inspire-Lift-Play-v0 \
    --num_envs 64 \
    --checkpoint logs/rsl_rl/Inspire_Seed0/model_XXXXX.pt
```

### 4.2 DG5F Hand
```bash
# Training (5 seeds)
for seed in 0 1 2 3 4; do
    python scripts/rsl_rl/train.py \
        --task Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0 \
        --num_envs 4096 \
        --seed $seed \
        --max_iterations 20000 \
        --experiment DG5F_Seed${seed}
done

# Evaluation
python scripts/rsl_rl/play.py \
    --task Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-Play-v0 \
    --num_envs 64 \
    --checkpoint logs/rsl_rl/DG5F_Seed0/model_XXXXX.pt
```

---

## 5. Data Collection and Analysis

### 5.1 Tensorboard Logs
All metrics automatically logged to:
```
logs/rsl_rl/<experiment_name>/summaries/
```

**Key Metrics**:
- `Train/success_rate`: Success rate over training
- `Train/episode_reward_mean`: Mean episode reward
- `Train/policy_loss`: PPO policy loss
- `Train/value_loss`: PPO value loss
- `Train/learning_rate`: Adaptive learning rate
- `Rewards/<reward_term>`: Individual reward components

### 5.2 Analysis Scripts
```python
# Example: Extract success rate at key checkpoints
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator

def extract_success_rates(log_dir, checkpoints=[1e6, 5e6, 10e6, 20e6]):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    success_rates = {}
    for scalar in ea.Scalars('Train/success_rate'):
        step = scalar.step
        value = scalar.value
        for cp in checkpoints:
            if step >= cp and cp not in success_rates:
                success_rates[cp] = value

    return success_rates
```

---

**Next**: See `04_RESULTS_ANALYSIS.md` for detailed analysis methodology and expected results format.
