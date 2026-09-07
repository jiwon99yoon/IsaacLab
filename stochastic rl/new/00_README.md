# Systematic Comparison of Underactuated vs Fully-Actuated Hands in RL-Based Dexterous Manipulation

## Overview

This directory contains comprehensive documentation for a controlled experimental study comparing **underactuated (6-DOF Inspire Hand)** vs **fully-actuated (20-DOF DG5F Hand)** designs in reinforcement learning-based dexterous manipulation.

**Key Research Question**: How does hand actuation structure affect RL learning dynamics?

**Main Hypothesis**: Underactuated hands act as **inductive bias** (implicit constraints on policy space), leading to:
- Higher sample efficiency
- More stable training
- More robust grasping behaviors

---

## Document Structure

### 1. [Experimental Design](01_EXPERIMENTAL_DESIGN.md) ⭐ **Start Here**
- Research question and hypothesis
- Controlled variables (what we fix)
- Experimental variables (what we change)
- Metrics and analysis protocol
- Expected outcomes and interpretation

**Why read this first**: Understand the **scientific rigor** behind our comparison.

---

### 2. [Related Work](02_RELATED_WORK.md)
- Dexterous RL (DexPBT, DexArt, etc.)
- Underactuated hand design (SoftHand, etc.)
- Gap in literature and our contribution
- Positioning statement

**Use this for**: Writing the Related Work section, understanding the research landscape.

---

### 3. [Implementation Details](03_IMPLEMENTATION_DETAILS.md) 🔧 **For Reproducibility**
- Exact hardware configurations (UR10e + Inspire/DG5F)
- Arm init pose alignment (CRITICAL - both hands identical)
- Object size scaling (0.5x for Inspire, 1.0x for DG5F)
- Reward function (identical weights and logic)
- PPO hyperparameters (identical)
- Domain randomization (identical protocol)
- Training commands and verification checklist

**Use this for**:
- Reproducing experiments
- Verifying controlled setup
- Writing the Method section

---

### 4. [Contributions and Conclusion](04_CONTRIBUTIONS_AND_CONCLUSION.md) 📝 **For Paper Writing**
- Contributions (English + Korean explanations)
- Conclusion (English + Korean explanations)
- Broader implications
- Future work directions
- Key points to emphasize in each section

**Use this for**:
- Writing Contributions, Conclusion sections
- Understanding the **big picture** of our work
- Preparing presentation slides

---

## Quick Start Guide

### For Writing the Paper:

1. **Introduction**:
   - Problem: Dexterous RL is hard (high-DoF, sparse rewards)
   - Gap: No systematic comparison of actuation structures in RL
   - Our Solution: Controlled experimental protocol
   - Contributions: See `04_CONTRIBUTIONS_AND_CONCLUSION.md` Section 3

2. **Related Work**:
   - Use `02_RELATED_WORK.md` Section 1-3
   - Emphasize the gap (Section 4)
   - Position our work (Section 5)

3. **Method**:
   - Experimental setup: `01_EXPERIMENTAL_DESIGN.md` Section 2-3
   - Implementation details: `03_IMPLEMENTATION_DETAILS.md` Section 1-2
   - Metrics: `01_EXPERIMENTAL_DESIGN.md` Section 4

4. **Results**:
   - [TODO: Add after experiments complete]
   - Follow analysis protocol in `01_EXPERIMENTAL_DESIGN.md` Section 4

5. **Discussion**:
   - Interpretation: `01_EXPERIMENTAL_DESIGN.md` Section 6
   - Inductive bias framing: `04_CONTRIBUTIONS_AND_CONCLUSION.md` Section 4.1
   - Design guidelines: `04_CONTRIBUTIONS_AND_CONCLUSION.md` Section 3.4

6. **Conclusion**:
   - Use `04_CONTRIBUTIONS_AND_CONCLUSION.md` Section 4.1-4.4

---

### For Running Experiments:

1. **Verify Setup**:
```bash
# Check arm configuration alignment
grep "wrist_2_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_inspire_right.py
grep "wrist_2_joint" source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f_right.py
# Both should show 180° (3.14159 or deg2rad(180.0))

# Check reward weights
grep "good_finger_contact" -A 2 source/isaaclab_tasks/.../ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py
grep "good_finger_contact" -A 2 source/isaaclab_tasks/.../ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py
# Both should show weight=0.5

# Check object sizes
grep "CuboidCfg(size=" source/isaaclab_tasks/.../ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py | head -1
# Should show (0.025, 0.05, 0.05) for Inspire

grep "CuboidCfg(size=" source/isaaclab_tasks/.../ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py | head -1
# Should show (0.05, 0.1, 0.1) for DG5F
```

2. **Train Inspire Hand**:
```bash
for seed in 0 1 2 3 4; do
    python scripts/rsl_rl/train.py \
        --task Isaac-Dexsuite-Ur10e-Inspire-Lift-v0 \
        --num_envs 4096 \
        --seed $seed \
        --max_iterations 20000 \
        --experiment Inspire_Seed${seed}
done
```

3. **Train DG5F Hand**:
```bash
for seed in 0 1 2 3 4; do
    python scripts/rsl_rl/train.py \
        --task Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0 \
        --num_envs 4096 \
        --seed $seed \
        --max_iterations 20000 \
        --experiment DG5F_Seed${seed}
done
```

4. **Analyze Results**:
   - See `01_EXPERIMENTAL_DESIGN.md` Section 4 for analysis protocol
   - Use Tensorboard: `tensorboard --logdir logs/rsl_rl`

---

## Key Configuration Summary

| Configuration | Inspire Hand | DG5F Hand | Status |
|--------------|-------------|-----------|--------|
| **Arm** | UR10e (6-DOF) | UR10e (6-DOF) | ✅ IDENTICAL |
| **Arm Init Pose** | wrist_2: 180°, wrist_3: -180° | wrist_2: 180°, wrist_3: -180° | ✅ IDENTICAL |
| **Hand DoF** | 6 actuated | 20 actuated | ❌ DIFFERENT (experimental variable) |
| **Control Dim** | 12-dim (6 arm + 6 hand) | 26-dim (6 arm + 20 hand) | ❌ DIFFERENT (experimental variable) |
| **Object Size** | 0.5x | 1.0x | ❌ DIFFERENT (normalized to hand size) |
| **Reward Weights** | position_tracking: 2.0, contact: 0.5 | position_tracking: 2.0, contact: 0.5 | ✅ IDENTICAL |
| **PPO Hyperparams** | lr: 5e-4, minibatch: 16384, ... | lr: 5e-4, minibatch: 16384, ... | ✅ IDENTICAL |
| **Domain Randomization** | Same protocol | Same protocol | ✅ IDENTICAL |

**Critical Point**: Object sizes differ, but this is **acceptable** because:
1. Objects are normalized to hand sizes (relative difficulty is equal)
2. We're comparing **learning dynamics**, not absolute task difficulty
3. This is standard practice in manipulation research (different grippers use appropriately-sized objects)

---

## File Locations Reference

### Robot Configurations
- **Inspire Hand**: `source/isaaclab_assets/isaaclab_assets/robots/ur10e_inspire_right.py`
- **DG5F Hand**: `source/isaaclab_assets/isaaclab_assets/robots/ur10e_dg5f_right.py`

### Environment Configurations
- **Inspire Env**: `source/isaaclab_tasks/.../dexsuite_ur10e/config/ur10e_inspire/dexsuite_ur10e_inspire_env_cfg.py`
- **DG5F Env**: `source/isaaclab_tasks/.../dexsuite_ur10e/config/ur10e_dg5f_right/dexsuite_ur10e_dg5f_right_env_cfg.py`
- **Base Config**: `source/isaaclab_tasks/.../dexsuite_ur10e/dexsuite_env_cfg.py`

### Gym Registrations
- **Inspire**: `Isaac-Dexsuite-Ur10e-Inspire-Lift-v0` (training)
- **Inspire Play**: `Isaac-Dexsuite-Ur10e-Inspire-Lift-Play-v0` (evaluation)
- **DG5F**: `Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-v0` (training)
- **DG5F Play**: `Isaac-Dexsuite-Ur10e-Dg5fRight-Lift-Play-v0` (evaluation)

---

## Expected Timeline

1. **Week 1-2**: Training (5 seeds × 2 hands × 20k iterations)
2. **Week 3**: Analysis (extract metrics, statistical tests, plots)
3. **Week 4**: Writing (draft paper, prepare figures)
4. **Week 5**: Revision (feedback, polish, final submission)

---

## Contact and Support

For questions about:
- **Experimental setup**: See `01_EXPERIMENTAL_DESIGN.md` or `03_IMPLEMENTATION_DETAILS.md`
- **Theoretical background**: See `02_RELATED_WORK.md` or `04_CONTRIBUTIONS_AND_CONCLUSION.md`
- **Reproducibility**: See verification checklist in `03_IMPLEMENTATION_DETAILS.md` Section 3

---

## Citation (Draft)

```bibtex
@article{yourname2025systematic,
  title={Systematic Comparison of Underactuated vs Fully-Actuated Hands in Reinforcement Learning-Based Dexterous Manipulation},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## License

[Specify license here - typically BSD-3-Clause for Isaac Lab projects]

---

**Last Updated**: 2025-12-15

**Status**: ✅ Configuration Complete, Ready for Training
