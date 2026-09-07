# Controlled Experimental Design for Systematic Comparison

## Overview

This document describes the **controlled experimental protocol** for comparing underactuated vs fully-actuated hand designs in reinforcement learning-based dexterous manipulation.

---

## 1. Research Question

**How does hand actuation structure (underactuated vs fully-actuated) affect reinforcement learning performance in dexterous manipulation tasks?**

### Hypothesis
- **Underactuated hands** (lower DoF, mechanical coupling) may provide implicit constraints that act as **inductive bias**, leading to:
  - Higher sample efficiency (faster learning)
  - More stable training (lower variance)
  - Reduced exploration difficulty (smaller action space)

- **Fully-actuated hands** (higher DoF, independent control) provide greater expressiveness but may suffer from:
  - Lower sample efficiency (harder exploration)
  - Higher training variance (larger action space)
  - More complex learned policies

---

## 2. Controlled Variables (Fixed Across Conditions)

To isolate the effect of **hand actuation structure**, we fix ALL other variables:

### 2.1 Robot Arm
- **Model**: Universal Robots UR10e (6-DOF)
- **Payload**: 12.5kg
- **Reach**: 1300mm

### 2.2 Arm Initial Configuration (CRITICAL - Must be Identical!)
```python
# UR10e Arm Joint Positions (Both Inspire and DG5F)
"shoulder_pan_joint":   0.0°       # Centered (facing forward)
"shoulder_lift_joint": -120.0°     # Raised for better object approach
"elbow_joint":          120.0°     # Bent for manipulability
"wrist_1_joint":       -120.0°     # Pitch down
"wrist_2_joint":        180.0°     # ✅ ALIGNED (neutral roll)
"wrist_3_joint":       -180.0°     # ✅ ALIGNED (neutral yaw)
```

**Why this matters**: Different arm configurations would confound the comparison by introducing different workspace reachability and joint velocity limits.

### 2.3 Task Definition
- **Task**: Object Lifting (position-only tracking, no orientation)
- **Episode Length**: 4.0 seconds (480 steps @ 120Hz simulation, 2x decimation = 240 control steps)
- **Success Criterion**: Object reaches target position within threshold
- **Objects**: 16 different geometric primitives (cuboids, spheres, capsules, cones)
- **Object Mass**: 0.2 kg (randomized 0.2-2.0x during training via domain randomization)

### 2.4 Observation Space (Identical Structure)
Both configurations receive the same observation structure:
- **Proprio observations** (history=5):
  - Joint positions and velocities
  - Hand fingertip states (position, orientation, velocities)
  - Contact forces at fingertips
- **Task observations** (history=5):
  - Object quaternion (base frame)
  - Target object pose (base frame)
  - Previous actions
- **Perception observations** (history=5):
  - Object point cloud (64 points, flattened)

**Note**: Observation dimension differs due to different number of joints, but the **information structure is identical**.

### 2.5 Reward Function (Identical Weights and Terms)
```python
# Base rewards (both configurations)
action_l2:              -0.005
action_rate_l2:         -0.005
fingers_to_object:       1.0
position_tracking:       2.0
success:                10.0
early_termination:      -1.0
ground_contact_penalty: -0.3

# Hand-specific contact reward (same weight)
good_finger_contact:     0.5  # Kuka-Allegro baseline
```

**Critical**: Contact sensor names differ, but the reward function logic and weights are **identical**.

### 2.6 RL Algorithm and Hyperparameters
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: RL-Games
- **Parallel Environments**: 4096
- **Hyperparameters**: Identical across all conditions
  - Learning rate
  - Batch size
  - Epochs per update
  - Entropy coefficient
  - Value loss coefficient
  - Clipping epsilon
  - GAE lambda
  - Discount factor

### 2.7 Domain Randomization (Identical Protocol)
- **Object scale**: 0.75-1.5x
- **Object mass**: 0.2-2.0x
- **Robot physics material**: friction, restitution randomization
- **Joint parameters**: stiffness, damping, friction randomization
- **Gravity**: Curriculum from 0 to full gravity (ADR)

---

## 3. Experimental Variables (What We Change)

### 3.1 Hand Configuration

| Property | Inspire Hand | DG5F Hand |
|----------|-------------|-----------|
| **DoF** | 6 actuated | 20 actuated |
| **Actuation Type** | Underactuated (mimic joints) | Fully-actuated |
| **Fingers** | 5 (thumb: 2-DoF, others: 1-DoF each) | 5 (each: 4-DoF) |
| **Control Dimensionality** | 12-dim (6 arm + 6 hand) | 26-dim (6 arm + 20 hand) |
| **Mechanical Coupling** | Yes (mimic joints provide coupling) | No (independent control) |
| **Hardware** | RH56F1-R (Inspire Right) | DG5F Right (Tesollo) |

### 3.2 Object Size (Scaled to Match Hand Size)

**Rationale**: Hands have different physical sizes. To ensure fair comparison, objects are scaled proportionally to hand size.

| Configuration | Object Scale | Rationale |
|--------------|-------------|-----------|
| **Kuka-Allegro (Baseline)** | 1.0x | Reference (original DexSuite) |
| **UR10e + DG5F** | 1.0x | DG5F similar size to Allegro |
| **UR10e + Inspire** | 0.5x | Inspire ~50-70% of Allegro size |

**Example Object Sizes**:
```python
# First cuboid dimensions
Kuka-Allegro:  (0.05, 0.1, 0.1) m  # 1.0x
UR10e-DG5F:    (0.05, 0.1, 0.1) m  # 1.0x (same as baseline)
UR10e-Inspire: (0.025, 0.05, 0.05) m  # 0.5x (scaled down)
```

**Why different object sizes is acceptable**:
1. We're comparing **learning dynamics**, not absolute task difficulty
2. Object size is normalized to hand size (relative difficulty is equal)
3. This is standard practice in dexterous manipulation research (e.g., different grippers in RoboCup@Home use appropriately-sized objects)

---

## 4. Metrics and Analysis Protocol

### 4.1 Primary Metrics (Learning Efficiency)

1. **Success Rate vs Training Steps**
   - Success rate at 1M, 5M, 10M, 20M steps
   - Convergence speed (steps to reach 50%, 70%, 80% success)
   - Final performance (success rate at 20M steps)

2. **Sample Efficiency**
   - Steps to first success
   - Steps to reach performance thresholds
   - Area under learning curve (AUC)

3. **Training Stability**
   - Standard deviation across 5 random seeds
   - Maximum/minimum performance variance
   - Coefficient of variation

### 4.2 Secondary Metrics (Exploration and Policy Behavior)

1. **Exploration Metrics**
   - Action entropy over training
   - Action standard deviation over training
   - Action saturation ratio (% of actions near joint limits)

2. **Contact Statistics**
   - Contact duration per episode
   - Number of contact events per episode
   - Ground/table collision frequency
   - Slip detection (if available from sensors)

3. **Policy Characteristics**
   - Value function loss over training
   - Policy loss over training
   - KL divergence between policy updates
   - Gradient norms

### 4.3 Qualitative Analysis

1. **Grasp Strategy Taxonomy**
   - Categorize learned grasps into 3-4 types
   - Analyze which strategies emerge for each hand
   - Compare strategy diversity

2. **Failure Mode Analysis**
   - Categorize failure modes (drop, slip, collision, timeout)
   - Compare failure distributions

---

## 5. Experimental Protocol

### 5.1 Training Setup
```bash
# UR10e + Inspire Hand (6-DoF underactuated)
python scripts/rsl_rl/train.py \
    --task Isaac-Dexsuite-Ur10e-Inspire-Lift-v0 \
    --num_envs 4096 \
    --seed [0,1,2,3,4]  # 5 seeds

# UR10e + DG5F Hand (20-DoF fully-actuated)
python scripts/rsl_rl/train.py \
    --task Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0 \
    --num_envs 4096 \
    --seed [0,1,2,3,4]  # 5 seeds
```

### 5.2 Evaluation Setup
```bash
# Play mode (deterministic policy, visualization)
python scripts/rsl_rl/play.py \
    --task Isaac-Dexsuite-Ur10e-Inspire-Lift-Play-v0 \
    --num_envs 64 \
    --checkpoint <path_to_checkpoint>
```

### 5.3 Data Collection
- **Tensorboard logs**: All metrics logged every 10 iterations
- **Checkpoints**: Save every 500 iterations
- **Video recordings**: Record evaluation episodes every 500 iterations

---

## 6. Expected Outcomes and Interpretation

### 6.1 If Underactuated Hand Learns Faster:
**Interpretation**: Mechanical constraints act as **inductive bias**
- Reduced action space → easier exploration
- Mechanical coupling → implicit coordination between fingers
- Natural compliance → more robust grasps

**Implication**: For sample-efficient RL, prefer underactuated designs when:
- Training data/time is limited
- Task complexity is moderate
- Compliance is beneficial

### 6.2 If Fully-Actuated Hand Learns Faster:
**Interpretation**: Higher expressiveness overcomes exploration difficulty
- Independent control → finer manipulation
- More DoF → richer policy space
- No mechanical constraints → more precise control

**Implication**: For high-performance RL, prefer fully-actuated designs when:
- Sufficient training resources available
- Task requires precise, dexterous manipulation
- Compliance is not critical

### 6.3 If Similar Performance:
**Interpretation**: Actuation structure has less impact than expected
- Task may not require fine manipulation (lifting is relatively simple)
- PPO's exploration may overcome DoF differences
- Other factors (e.g., reward shaping) dominate

**Implication**: Actuation choice driven by other factors:
- Cost (underactuated cheaper)
- Robustness (underactuated more robust)
- Precision requirements (fully-actuated more precise)

---

## 7. Validity Threats and Mitigation

### 7.1 Confounding Variables
- ✅ **Arm configuration**: Fixed and verified identical
- ✅ **Task definition**: Identical task, identical success criteria
- ✅ **Observation structure**: Identical information, different dimensions
- ✅ **Reward function**: Identical logic and weights
- ✅ **RL hyperparameters**: Identical across conditions
- ⚠️ **Object size**: Different but normalized to hand size (acceptable)

### 7.2 Implementation Verification
```python
# Verify arm init pose is identical
assert inspire_config.wrist_2_joint == dg5f_config.wrist_2_joint == 180.0°
assert inspire_config.wrist_3_joint == dg5f_config.wrist_3_joint == -180.0°

# Verify reward weights are identical
assert inspire_config.rewards.position_tracking.weight == 2.0
assert dg5f_config.rewards.position_tracking.weight == 2.0
```

### 7.3 Statistical Significance
- Use 5 random seeds per condition
- Report mean ± standard deviation
- Perform statistical tests (t-test, Mann-Whitney U) when comparing final performance

---

## 8. Contribution Summary

This experimental design enables a **systematic, controlled comparison** of hand actuation structures in RL-based manipulation. Key contributions:

1. **Controlled Protocol**: All variables fixed except actuation structure
2. **Fair Comparison**: Object sizes normalized to hand sizes
3. **Comprehensive Metrics**: Beyond success rate (exploration, stability, strategy)
4. **Reproducible Setup**: Detailed configuration and verification steps

This allows us to make **evidence-based claims** about the role of mechanical design in RL-based manipulation, addressing a gap in the literature where comparisons are often confounded by multiple variables.

---

## References

See `02_RELATED_WORK.md` for detailed literature review and positioning of this work.
