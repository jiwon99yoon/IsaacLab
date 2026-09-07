# Stack-RL: PPO Training & MDP Configuration

## Overview

This document provides detailed technical specifications of the **MDP (Markov Decision Process)** formulation, observation/action spaces, reward structure, and control loop for the Stack-RL environment trained with PPO (Proximal Policy Optimization).

---

## 1. MDP Formulation

### State Space (S)

The **true state** of the environment includes:
- Robot joint positions (7 arm + 2 gripper = 9 DoF)
- Robot joint velocities (9 DoF)
- End-effector pose (position + quaternion = 7-dim)
- Cube 1 pose (position + quaternion = 7-dim)
- Cube 2 pose (position + quaternion = 7-dim)
- Cube 3 pose (position + quaternion = 7-dim)
- Cube 1 velocity (linear + angular = 6-dim)
- Cube 2 velocity (linear + angular = 6-dim)
- Cube 3 velocity (linear + angular = 6-dim)

**Total State Dimension**: ~60+ dimensions (fully observable for simulation)

### Observation Space (O)

The **policy observation** is a **56-dimensional vector** constructed as follows:

#### Joint States (11-dim)
- **Joint Positions** (7-dim): Relative to default pose
  - `panda_joint1` to `panda_joint7`
  - Function: `mdp.joint_pos_rel()`
  - Normalized by joint limits
- **Joint Velocities** (4-dim): Relative velocities (subset)
  - Function: `mdp.joint_vel_rel()`
  - Limited to most important joints for speed

#### Object States (21-dim)
- **Cube Positions** (9-dim): World-frame positions
  - Cube 1: [x₁, y₁, z₁]
  - Cube 2: [x₂, y₂, z₂]
  - Cube 3: [x₃, y₃, z₃]
  - Function: `mdp.cube_positions_in_world_frame()`

- **Cube Orientations** (12-dim): World-frame quaternions
  - Cube 1: [qw₁, qx₁, qy₁, qz₁]
  - Cube 2: [qw₂, qx₂, qy₂, qz₂]
  - Cube 3: [qw₃, qx₃, qy₃, qz₃]
  - Function: `mdp.cube_orientations_in_world_frame()`

#### End-Effector States (7-dim)
- **EE Position** (3-dim): [x_ee, y_ee, z_ee]
  - Function: `mdp.ee_frame_pos()`
  - Frame: `panda_hand` + 10.34cm offset (center of gripper)

- **EE Orientation** (4-dim): [qw_ee, qx_ee, qy_ee, qz_ee]
  - Function: `mdp.ee_frame_quat()`

#### Gripper State (2-dim)
- **Gripper Finger Positions** (2-dim): [left_finger, right_finger]
  - Function: `mdp.gripper_pos()`
  - Range: [0.0, 0.04] meters (fully closed to fully open)
  - Joint names: `panda_finger_joint1`, `panda_finger_joint2`

#### Action History (15-dim)
- **Previous Actions** (15-dim): Last action taken
  - Function: `mdp.last_action()`
  - Helps policy maintain temporal consistency

**Total Observation Dimension**: **56** (concatenated into single vector)

#### Observation Normalization
```python
enable_corruption: True  # Add observation noise during training
concatenate_terms: True  # Flatten into single vector
normalize_input: True    # Running mean/std normalization
```
- Running mean and standard deviation tracked during training
- Observation clipping: [-100, +100]

---

### Action Space (A)

#### Joint Position Control (Default)

**Action Dimension**: 8-dim
- **Arm Actions** (7-dim): Target joint positions
  - Joints: `panda_joint1` to `panda_joint7`
  - Type: Absolute target positions (with default offset)
  - Scale: 0.5 (actions are scaled by this factor)
  - Function: `mdp.JointPositionActionCfg`

- **Gripper Action** (1-dim): Binary open/close
  - Type: Binary command {0: close, 1: open}
  - Open position: 0.04m (both fingers)
  - Closed position: 0.0m (both fingers)
  - Function: `mdp.BinaryJointPositionActionCfg`

**Action Processing**:
```python
# Arm action
target_joint_pos = default_pose + (action[0:7] * 0.5)

# Gripper action
if action[7] > 0.5:
    gripper_target = 0.04  # Open
else:
    gripper_target = 0.0   # Close
```

**Action Clipping**: [-100, +100] (before scaling)

#### IK Relative Control (Alternative)

**Action Dimension**: 7-dim
- **Pose Delta** (6-dim): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
  - Type: Relative pose change in end-effector frame
  - Scale: 0.5
  - IK Method: Damped Least Squares (DLS)
  - Controller: `DifferentialIKControllerCfg`
  - Body offset: [0.0, 0.0, 0.107] (gripper center)

- **Gripper Action** (1-dim): Binary open/close

---

### Reward Function (R)

The reward at each timestep is a **weighted sum of 11 terms**:

#### Phase 1: Lift Cube 1 (Base Cube)

**R1: Reaching Cube 1** (weight: 1.0)
```python
r_reach_1 = 1 - tanh(||p_cube1 - p_ee|| / σ)
```
- σ (std) = 0.1
- Smooth reward for approaching cube 1
- Maximum: 1.0 when at cube
- Function: `mdp.object_ee_distance`

**R2: Grasping Cube 1** (weight: 10.0)
```python
r_grasp_1 = 1.0 if (||p_cube1 - p_ee|| < 0.06) AND (gripper_closed)
           = 0.0 otherwise
```
- Binary reward for successful grasp
- Threshold: 6cm distance + gripper closure > 0.005m
- Function: `mdp.object_grasped_reward`

**R3: Lifting Cube 1** (weight: 15.0)
```python
r_lift_1 = 1.0 if (z_cube1 > z_min + 0.08)
          = 0.0 otherwise
```
- Minimal height: 8cm above table
- Encourages lifting the base cube
- Function: `mdp.object_is_lifted`

#### Phase 2: Stack Cube 2 on Cube 1

**R4: Reaching Cube 2** (weight: 1.0)
```python
r_reach_2 = 1 - tanh(||p_cube2 - p_ee|| / 0.1)
```

**R5: Grasping Cube 2** (weight: 10.0)
```python
r_grasp_2 = 1.0 if (||p_cube2 - p_ee|| < 0.06) AND (gripper_closed)
```

**R6: Stacking Cube 2 on Cube 1** (weight: 30.0)
```python
Δp = p_cube2 - p_cube1
xy_dist = ||(Δp_x, Δp_y)||
z_dist = |Δp_z|

r_stack_2on1 = 1.0 if (xy_dist < 0.05) AND
                      (|z_dist - 0.0468| < 0.005) AND
                      (gripper_open)
              = 0.0 otherwise
```
- XY alignment: < 5cm
- Height alignment: 4.68cm ± 0.5cm (cube height)
- Gripper must be open (released)
- Function: `mdp.object_stacked_reward`

#### Phase 3: Stack Cube 3 on Cube 2

**R7: Reaching Cube 3** (weight: 1.0)
```python
r_reach_3 = 1 - tanh(||p_cube3 - p_ee|| / 0.1)
```

**R8: Grasping Cube 3** (weight: 10.0)
```python
r_grasp_3 = 1.0 if (||p_cube3 - p_ee|| < 0.06) AND (gripper_closed)
```

**R9: Stacking Cube 3 on Cube 2** (weight: 50.0)
```python
r_stack_3on2 = 1.0 if (||p_cube3 - p_cube2||_xy < 0.05) AND
                      (|z_cube3 - z_cube2 - 0.0468| < 0.005) AND
                      (gripper_open)
```
- Higher weight (50.0) for final stack

#### Success Bonus

**R10: Task Success** (weight: 100.0)
```python
r_success = 1.0 if (cube2_on_cube1) AND (cube3_on_cube2) AND (gripper_open)
           = 0.0 otherwise
```
- Verifies all 3 cubes properly stacked
- Large bonus for task completion
- Function: `mdp.three_cubes_stacked_success`

#### Regularization Penalties

**R11: Action Rate Penalty** (weight: -1e-4, curriculum: -1e-1)
```python
r_action_rate = -||a_t - a_{t-1}||²
```
- Penalizes large action changes
- Encourages smooth control
- Weight increases via curriculum learning
- Function: `mdp.action_rate_l2`

**R12: Joint Velocity Penalty** (weight: -1e-4, curriculum: -1e-1)
```python
r_joint_vel = -||q̇||²
```
- Penalizes high joint velocities
- Encourages energy-efficient motions
- Function: `mdp.joint_vel_l2`

#### Total Reward
```python
R_total = r_reach_1 + 10*r_grasp_1 + 15*r_lift_1 +
          r_reach_2 + 10*r_grasp_2 + 30*r_stack_2on1 +
          r_reach_3 + 10*r_grasp_3 + 50*r_stack_3on2 +
          100*r_success +
          w_action*r_action_rate + w_vel*r_joint_vel
```
- Reward range: ~[-1, +227] per timestep
- Curriculum: Regularization weights increase over training

---

### Transition Dynamics (T)

#### Physics Simulation
- **Simulator**: NVIDIA PhysX 5 (GPU-accelerated)
- **Integration**: Semi-implicit Euler
- **Timestep**: Δt = 0.01s (100 Hz simulation)
- **Substeps**: 1 (no sub-stepping)

#### Robot Dynamics
- **Robot Model**: Franka Panda (7-DOF arm + 2-DOF gripper)
- **URDF**: `isaaclab_assets.robots.franka.FRANKA_PANDA_CFG`
- **Actuator Model**: Implicit PD controller
  - Position gains (Kp): [Joint-specific]
  - Damping (Kd): [Joint-specific]
  - Max efforts: [Joint-specific]
- **Joint Limits**: Enforced via PhysX articulation
- **Self-Collision**: Enabled

#### Object Dynamics
- **Cubes**: 3x rigid bodies
  - Dimensions: 4.05cm × 4.05cm × 4.68cm (standard blocks)
  - Mass: ~50g (estimated from rigid body properties)
  - Friction: Default PhysX material
  - Solver iterations: 16 (position), 1 (velocity)
  - Max depenetration velocity: 5.0 m/s

#### Contact Dynamics
- **Solver**: TGS (Temporal Gauss-Seidel)
- **Friction Model**: Pyramid approximation
- **Bounce Threshold**: 0.01 m/s
- **Friction Correlation Distance**: 0.00625m
- **GPU Pairs Capacity**: 32K (increased for 3 cubes)

---

### Termination Conditions

Episode terminates if any of the following conditions are met:

**T1: Time Out** (time_out = True)
```python
if t ≥ T_max:
    done = True
```
- Episode length: 15 seconds
- Max steps: 750 (at 50Hz control frequency)
- Function: `mdp.time_out`

**T2: Cube Dropping** (time_out = False)
```python
if z_cube_i < -0.05:  # i ∈ {1, 2, 3}
    done = True
```
- Terminates if any cube falls below table
- Threshold: -5cm (table at z=0)
- Functions: `mdp.root_height_below_minimum`

**T3: Task Success** (time_out = False)
```python
if (cube2_on_cube1) AND (cube3_on_cube2):
    done = True
```
- Early termination on success
- Allows policy to finish quickly
- Function: `mdp.cubes_stacked`

---

## 2. Control Loop & Timing

### Hierarchical Control Architecture

```
┌─────────────────────────────────────────────┐
│  PPO Policy Network (PyTorch)              │ ← 50 Hz
│  Input: observation (56-dim)                │
│  Output: action (8-dim)                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Action Processor                           │ ← 50 Hz
│  - Scale actions                            │
│  - Add default offsets                      │
│  - Clip to limits                           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  PD Controller (Implicit in PhysX)          │ ← 100 Hz
│  τ = Kp(q_target - q) - Kd(q̇)              │  (simulated)
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  PhysX Physics Engine                       │ ← 100 Hz
│  - Forward dynamics                          │
│  - Contact resolution                        │
│  - Integration                               │
└─────────────────────────────────────────────┘
```

### Timing Breakdown

#### Simulation Frequency
- **Physics Timestep**: dt = 0.01s = **100 Hz**
- **PhysX Update Rate**: 100 Hz
- **Forward Dynamics**: Every timestep
- **Contact Solver**: Every timestep

#### Control Frequency
- **Policy Decision Rate**: **50 Hz**
- **Decimation**: 2 (policy acts every 2 simulation steps)
- **Action Hold**: 0.02s (2 physics steps)

#### Rendering Frequency
- **Render Interval**: 2 simulation steps = **50 Hz**
- **Visualization FPS**: ~50 FPS (when not headless)

### Control Flow per Episode Step

```python
for step in range(max_episode_steps):  # 750 steps max
    # Every 2 simulation steps (50 Hz control)
    if step % decimation == 0:
        # 1. Get observation (from previous state)
        obs = get_observation()  # 56-dim

        # 2. Policy forward pass
        with torch.no_grad():
            action, value = policy(obs)  # action: 8-dim

        # 3. Process action
        joint_targets = default_pose + action[:7] * 0.5
        gripper_target = 0.04 if action[7] > 0.5 else 0.0

    # Every simulation step (100 Hz physics)
    # 4. Apply joint targets to robot
    robot.set_joint_position_targets(joint_targets)
    robot.set_gripper_targets(gripper_target)

    # 5. PhysX simulates one step
    #    - Computes τ = Kp(q_target - q) - Kd(q̇)
    #    - Forward dynamics: q̈ = M⁻¹(τ + τ_ext)
    #    - Contact resolution
    #    - Integrate: q, q̇ → q', q̇'
    sim.step(dt=0.01)

    # 6. Update observations for next step
    update_sensors()  # Frame transformer, object states

    # Every 2 simulation steps (50 Hz)
    if step % decimation == 0:
        # 7. Compute reward
        reward = compute_reward()

        # 8. Check termination
        done = check_termination()

        # 9. Store experience in buffer
        buffer.add(obs, action, reward, done, value)
```

### Torque Transmission

**Question**: How are torques transmitted to the robot?

**Answer**: IsaacLab uses **implicit PD control** in PhysX:

1. **Policy outputs**: Target joint positions (not torques)
2. **PhysX PD controller**: Computes torques internally
   ```
   τ_actuated = Kp * (q_target - q_current) - Kd * q̇_current
   ```
3. **Torque limits**: Enforced via `max_effort` in robot config
4. **Applied torques**: τ_total = τ_actuated + τ_gravity + τ_contact
5. **Forward dynamics**: PhysX solves for accelerations
6. **Integration**: Updates positions and velocities

**Key Point**: The policy does **not** directly output joint torques. It outputs target positions, and PhysX handles the low-level torque control.

---

## 3. PPO Algorithm Configuration

### Neural Network Architecture

#### Actor-Critic Network
```
Input (56-dim observation)
    ↓
Dense(256) + ELU
    ↓
Dense(128) + ELU
    ↓
Dense(64) + ELU
    ↓         ↓
Actor       Critic
(8-dim)     (1-dim value)
```

**Architecture Details**:
- **Type**: Shared trunk, separate heads
- **Activation**: ELU (Exponential Linear Unit)
- **Layers**: [256, 128, 64]
- **Separate**: False (shared features)
- **Actor Output**: 8-dim mean (μ)
- **Critic Output**: 1-dim value estimate (V)
- **Policy Distribution**: Diagonal Gaussian
  - Fixed σ (not learned)
  - σ initialized via `const_initializer`

### PPO Hyperparameters

#### Learning Parameters
```yaml
learning_rate: 1e-4          # Adam learning rate
lr_schedule: adaptive        # Decreases based on KL divergence
kl_threshold: 0.01           # Target KL for adaptive LR
gamma: 0.99                  # Discount factor
tau: 0.95                    # GAE lambda
```

#### Experience Collection
```yaml
num_actors: 4096             # Parallel environments
horizon_length: 64           # Steps per rollout
total_steps_per_iter: 262144 # 4096 envs × 64 steps
```

#### Optimization
```yaml
mini_epochs: 8               # PPO update epochs
minibatch_size: 32768        # Batch size per update
num_minibatches: 8           # 262144 / 32768
```

#### Loss Weights
```yaml
critic_coef: 4               # Value loss weight
entropy_coef: 0.001          # Entropy bonus weight
bounds_loss_coef: 0.0001     # Action bounds penalty
```

#### Clipping & Regularization
```yaml
e_clip: 0.2                  # PPO clip range [0.8, 1.2]
clip_value: True             # Clip value loss
grad_norm: 1.0               # Gradient norm clipping
truncate_grads: True         # Enable gradient clipping
```

#### Training Schedule
```yaml
max_epochs: 100000           # Total training epochs
save_frequency: 100          # Save every 100 epochs
save_best_after: 200         # Start tracking best after 200
print_stats: True            # Log statistics
```

### PPO Loss Functions

#### Policy Loss (Clipped Objective)
```python
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε = 0.2
L_policy = -min(ratio * A, clipped_ratio * A)
```

#### Value Loss (Clipped)
```python
V_clipped = V_old + clip(V_new - V_old, -ε, +ε)
L_value = max((V_new - V_target)², (V_clipped - V_target)²)
```

#### Entropy Bonus
```python
L_entropy = -H(π(·|s))  # Negative entropy
```

#### Total Loss
```python
L_total = L_policy + 4*L_value + 0.001*L_entropy + 0.0001*L_bounds
```

### Advantage Estimation (GAE)

```python
δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
```
- λ (tau) = 0.95
- γ (gamma) = 0.99

### Normalization

#### Input Normalization
```python
obs_normalized = (obs - running_mean) / (running_std + ε)
```
- Running statistics updated every step
- Initial std = 1.0
- ε = 1e-8

#### Value Normalization
```python
value_normalized = (value - running_mean_value) / (running_std_value + ε)
```

#### Advantage Normalization
```python
A_normalized = (A - mean(A)) / (std(A) + ε)
```

#### Reward Scaling
```python
reward_scaled = reward * 0.01  # reward_shaper.scale_value
```

---

## 4. Parallelization & Performance

### GPU Parallelization

#### Environment Parallelization
- **Num Environments**: 4096 (default)
- **Parallelization**: GPU tensor operations
- **Memory**: All environments in single GPU memory
- **Observation Tensor**: [4096, 56] on GPU
- **Action Tensor**: [4096, 8] on GPU

#### Physics Parallelization
- **PhysX GPU**: All 4096 scenes simulated in parallel
- **Broadphase**: GPU-accelerated AABB tree
- **Contact Solver**: GPU TGS solver
- **Rigid Body Pipeline**: Fully GPU-based

#### Policy Parallelization
- **Batch Inference**: All 4096 observations → single forward pass
- **Device**: cuda:0
- **Mixed Precision**: False (using FP32)
- **Inference Time**: ~0.1ms for 4096 environments

### Training Loop

```python
for epoch in range(max_epochs):  # 100000 epochs
    # 1. Collect experience (4096 envs × 64 steps)
    for step in range(horizon_length):
        obs = env.get_observations()  # [4096, 56]
        actions, values = policy(obs)  # [4096, 8], [4096, 1]
        obs_next, rewards, dones = env.step(actions)
        buffer.add(obs, actions, rewards, dones, values)

    # 2. Compute advantages
    advantages, returns = compute_gae(buffer, gamma=0.99, tau=0.95)

    # 3. PPO update (8 mini-epochs)
    for mini_epoch in range(8):
        for minibatch in get_minibatches(buffer, batch_size=32768):
            # Forward pass
            actions_new, values_new = policy(minibatch.obs)

            # Compute losses
            loss_policy = ppo_policy_loss(actions_new, minibatch.advantages)
            loss_value = value_loss(values_new, minibatch.returns)
            loss_entropy = entropy_loss(actions_new)

            loss_total = loss_policy + 4*loss_value + 0.001*loss_entropy

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

    # 4. Logging
    if epoch % 100 == 0:
        save_checkpoint(policy, epoch)
```

### Performance Metrics

#### Training Speed
- **FPS (steps/sec)**: ~85,000 - 90,000
- **Steps per epoch**: 262,144 (4096 × 64)
- **Time per epoch**: ~3 seconds
- **Total training time**: ~8-10 hours (100k epochs)

#### Memory Usage
- **GPU Memory**: ~12-16 GB (for 4096 envs)
- **System RAM**: ~32 GB
- **PhysX GPU Memory**: ~4 GB (aggregate pairs, contacts)

#### Bottlenecks
- **Physics Simulation**: 60% of compute time
- **Policy Inference**: 10% of compute time
- **Reward Computation**: 15% of compute time
- **Environment Resets**: 5% of compute time
- **Overhead**: 10% (data transfer, logging)

---

## 5. Curriculum Learning

### Reward Weight Curriculum

Two reward terms have their weights gradually increased:

#### Action Rate Penalty
- **Initial Weight**: -1e-4
- **Final Weight**: -1e-1
- **Schedule**: Linear increase over 10,000 steps
- **Function**: `mdp.modify_reward_weight`

#### Joint Velocity Penalty
- **Initial Weight**: -1e-4
- **Final Weight**: -1e-1
- **Schedule**: Linear increase over 10,000 steps

**Rationale**:
- Early training: Allow exploration, ignore smoothness
- Later training: Enforce smooth, energy-efficient motions

### Potential Future Curricula (Not Implemented)

- **Cube Separation**: Start with wider separation, gradually decrease
- **Episode Length**: Start short (5s), gradually increase to 15s
- **Task Complexity**: Train phase-by-phase (lift → 2-cube → 3-cube)
- **Randomization**: Increase noise in observations over time

---

## 6. Observation & Reward Debugging

### Checking Observations

```python
# Print observation for environment 0
obs = env.get_observations()
print("Joint positions:", obs[0, :7])
print("Cube 1 position:", obs[0, 11:14])
print("EE position:", obs[0, 32:35])
```

### Visualizing Rewards

```python
# Access individual reward terms
rewards_dict = env.reward_manager.compute()
print("Reaching cube 1:", rewards_dict["reaching_cube_1"][0])
print("Grasping cube 1:", rewards_dict["grasping_cube_1"][0])
print("Task success:", rewards_dict["task_success"][0])
```

### Checking Terminations

```python
dones = env.termination_manager.compute()
print("Time out:", dones["time_out"][0])
print("Cube dropping:", dones["cube_1_dropping"][0])
print("Success:", dones["success"][0])
```

---

## 7. Key Differences from Other Environments

### vs. Lift Environment
| Aspect | Lift | Stack-RL |
|--------|------|----------|
| Task | Lift 1 cube | Stack 3 cubes sequentially |
| Episode Length | 5s (250 steps) | 15s (750 steps) |
| Horizon Length | 24 | 64 |
| Max Epochs | 5000 | 100000 |
| Observation Dim | ~45 | 56 |
| Reward Terms | 5 | 11 |
| Success Criteria | Cube lifted | 3 cubes stacked |

### vs. Original Stack (IL)
| Aspect | Stack (IL) | Stack-RL |
|--------|------------|----------|
| Learning Method | Imitation Learning | Reinforcement Learning |
| Rewards | None (uses IL loss) | Dense reward shaping |
| Expert Data | Required (CuRobo) | Not required |
| Training | Supervised | Trial-and-error |
| Episode Length | 30s | 15s (tighter) |

---

## 8. Troubleshooting Observations

### Policy Not Learning?

1. **Check observation ranges**:
   ```python
   print(obs.min(), obs.mean(), obs.max())
   ```
   - Should be normalized (mean ≈ 0, std ≈ 1)

2. **Check reward magnitudes**:
   ```python
   print(env.reward_manager.compute())
   ```
   - Reaching rewards should be ~0.5-1.0 initially
   - Success rewards will be 0.0 until late in training

3. **Check action distribution**:
   ```python
   print(actions.min(), actions.mean(), actions.max())
   ```
   - Should explore initially (high variance)
   - Should converge later (low variance)

### Physics Instability?

1. **Check contact forces**:
   - If cubes are flying away → increase solver iterations
   - If penetration → decrease depenetration velocity

2. **Check joint limits**:
   - If robot flails → ensure joint limits are enforced

3. **Check timestep**:
   - If simulation explodes → decrease dt (e.g., 0.005s)

---

## Summary Table

| Component | Value | Notes |
|-----------|-------|-------|
| **Observation Dim** | 56 | Concatenated vector |
| **Action Dim** | 8 (joint) / 7 (IK) | Joint targets + gripper |
| **Reward Terms** | 11 | 9 task + 2 regularization |
| **Episode Length** | 15s (750 steps) | At 50Hz control |
| **Simulation Freq** | 100 Hz | PhysX timestep |
| **Control Freq** | 50 Hz | Policy decision rate |
| **Decimation** | 2 | Policy acts every 2 sim steps |
| **Num Envs** | 4096 | Parallel GPU environments |
| **Horizon Length** | 64 | PPO rollout steps |
| **Minibatch Size** | 32768 | PPO update batch size |
| **Learning Rate** | 1e-4 | Adaptive based on KL |
| **Max Epochs** | 100000 | Extended for complex task |
| **Training Time** | ~8-10 hours | On RTX 3090/4090 |

---

**Document Status**: ✅ Complete technical specification
**Last Updated**: 2025-12-09
**Related**: See `WORKFLOW.md` for project context and development history

---

**End of PPO Information Document**
