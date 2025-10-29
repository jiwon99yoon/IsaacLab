# HDR20-17 + DG5F Hand Dexsuite Environment

**Modified from:** Kuka-Allegro Dexsuite Environment
**Robot:** HDR20-17 (6-DOF industrial arm) + DG5F (20-DOF 5-finger hand)
**Task:** Lift and manipulate objects using multi-finger grasping

---

## Overview

This environment is based on the original Kuka-Allegro Dexsuite task but adapted for the heavier HDR20-17 industrial robot with DG5F hand. The main challenge is that HDR is much heavier than Kuka, requiring different actuator settings and additional rewards to encourage proper grasping behavior.

---

## Key Differences from Kuka-Allegro

### 1. **Robot Configuration** (`hdr_dg5f_new.py`)

#### **Actuator Settings**

| Parameter | Kuka-Allegro | HDR-DG5F | Reason |
|-----------|-------------|----------|---------|
| **Arm Stiffness** | 300 (j1-4), 100/50/25 (j5-7) | 20000 (j1-3), 2000/2000/1000 (j4-6) | HDR arm is ~60kg, needs much higher stiffness |
| **Arm Damping** | 45/20/15/15 | 1500/200/200/100 | Proportional to stiffness increase |
| **Finger Stiffness** | 3.0 | **3.0** (30.0 → 8.0 → 12.0 → 5.0 → 3.0) | **Matched Kuka!** Compliance for natural grasping |
| **Finger Damping** | 0.1 | **0.1** (3.0 → 0.5 → 0.8 → 0.3 → 0.1) | **Matched Kuka!** Fast response for agile grasping |
| **Finger Effort Limit** | 0.5 | **5.0** | DG5F is larger and heavier than Allegro |

**Iteration History:**
- **Original:** stiffness=30.0, damping=3.0 → Too rigid, fingers like "steel rods"
- **1st Try:** stiffness=8.0, damping=0.5 → Improved but exploration found issues
- **2nd Try:** stiffness=12.0, damping=0.8 → Wrong direction! Still too rigid
- **3rd Try:** stiffness=5.0, damping=0.3 → Better but fingers still "tap" instead of "grasp"
- **FINAL:** stiffness=3.0, damping=0.1 → **Matched Kuka exactly!** Natural grasping motion
  - User observation: "손가락을 접기 전에 contact이 일어나서 툭툭 침"
  - Diagnosis: stiffness too high → fingers rigid before curling
  - Solution: Match Kuka values (finger weight similar, compliance is key!)

#### **Initial Joint Positions**

**Fingers 2-5 (Index, Middle, Ring, Pinky):**
- Joint 2: 0.3 → **0.5** (more curved, ready to grasp)
- Joint 3: 0.3 → **0.4** (more curved)
- Joint 4: 0.3 → **0.4** (more curved)

**Thumb (Finger 1):**
- Joint 1: 0.85 → **0.6** (less abduction, closer to object)
- Joint 2: -1.0 → **-1.2** (stronger opposition)
- Joint 4: 0.6 → **0.5** (adjusted)

**Reason:** Initial pose now forms a "pre-grasp" configuration, making it easier for the agent to discover grasping behaviors.

---

### 2. **Contact Sensor Configuration** (`dexsuite_hdr_dg5f_new_env_cfg.py`)

| Aspect | Kuka-Allegro | HDR-DG5F |
|--------|-------------|----------|
| **Number of fingers** | 4 (thumb, index, middle, ring) | 5 (thumb, index, middle, ring, pinky) |
| **Finger tip names** | `*_link_3` | `rl_dg_*_4` |
| **Palm link** | `palm_link` | `rl_dg_palm` |
| **Contact sensor path** | `ee_link/*_link_3` | `dg5f_right_new/rl_dg_*_4` |

---

### 3. **Reward Function Modifications**

#### **A. Modified Rewards**

##### **`success_reward` - Contact Gating Added** (`mdp/rewards.py:87-125`)
- **Kuka-Allegro:** No contact requirement
  ```python
  return (1 - tanh(pos_dist)) * (1 - tanh(rot_dist))
  ```
- **HDR-DG5F:** Gated by contact
  ```python
  base_reward = (1 - tanh(pos_dist)) * (1 - tanh(rot_dist))
  return base_reward * contacts(env, 1.0).float()  # Requires finger contact!
  ```
- **Reason:** Agent was achieving success by pushing object with arm instead of grasping
- **Observation:** Green table (success visualizer) activated without finger contact
- **Fix:** Success now requires both position match AND finger contact (consistent with position/orientation tracking)

##### **`good_finger_contact` - Weight Increased**
- **Kuka-Allegro:** weight = 0.5
- **HDR-DG5F:** weight = **3.0** (6x increase)
- **Reason:** Agent was not learning to make finger contact, prioritizing this reward helps discovery

##### **`contacts()` Function - Condition Relaxed** (`mdp/rewards.py:71-83`)
```python
# ORIGINAL (too strict - middle finger required):
good_contact = (thumb > threshold) & (middle > threshold) &
               ((index > threshold) | (ring > threshold) | (pinky > threshold))

# MODIFIED (Kuka-like - any finger with thumb):
good_contact = (thumb > threshold) &
               ((index > threshold) | (middle > threshold) |
                (ring > threshold) | (pinky > threshold))
```
**Reason:** Requiring middle finger contact was too restrictive with rigid fingers, preventing reward signal.

#### **B. New Rewards (Not in Kuka-Allegro)**

##### **`object_lift` - ~~Lift Height Reward~~ [DISABLED]** (`mdp/rewards.py:145-170`)
- **Status:** ❌ **REMOVED** (found to cause problems)
- **Original Weight:** 5.0
- **Problem:** Agent discovered shortcut - pushing object with arm/wrist instead of grasping with fingers
- **Result:** `good_finger_contact` remained at 0, agent optimized for easy lift reward without learning to grasp
- **Lesson:** High-weight rewards that can be achieved without target behavior create harmful local optima

##### **`ground_contact_penalty` - Ground Touch Penalty** (`mdp/rewards.py:173-200`)
- **Weight:** -1.5 (scaled to match Kuka reward magnitudes)
- **Original:** -3.0 (too high)
- **Function:** Penalizes object touching table surface
- **Reason:** Prevents "bouncing object off table" exploitation strategy
- **Scale Justification:** Kuka rewards are in 0.5-10.0 range, so -1.5 is proportional
- **Implementation:**
  ```python
  height_above_table = object_height - 0.255
  penalty = clamp(1.0 - height_above_table / 0.02, 0, 1)
  ```

##### **`grasp_duration` - Sustained Grasp Reward** (`mdp/rewards.py:203-239`)
- **Weight:** 1.5 (slightly lower than position_tracking=2.0)
- **Original:** 2.0 (adjusted to fit Kuka scale)
- **Function:** Rewards maintaining grasp for 10+ consecutive timesteps
- **Reason:** Encourages sustained grasping vs. momentary tapping
- **Scale Justification:** Between good_finger_contact (3.0) and position_tracking (2.0)
- **Implementation:**
  - Maintains 10-step buffer of contact states
  - Rewards only if ALL recent steps have contact

---

### 4. **Scene Configuration Differences** (`dexsuite_env_cfg.py`)

| Parameter | Kuka-Allegro | HDR-DG5F | Reason |
|-----------|-------------|----------|---------|
| **Object initial position** | (-0.55, 0.1, 0.35) | (0.8, 0.0, 0.27) | Robot faces different direction |
| **Table position** | (-0.55, 0.0, 0.235) | (0.8, 0.0, 0.235) | Aligned with object |
| **Table size** | (0.8, 1.5, 0.04) | (0.8, 1.0, 0.04) | Smaller workspace |
| **Command range (x)** | (-0.7, -0.3) | (0.7, 0.9) | Robot coordinate system |
| **Command range (y)** | (-0.25, 0.25) | (-0.3, 0.3) | Slightly wider |
| **Command range (z)** | (0.55, 0.95) | (0.3, 0.6) | Lower workspace |
| **Out of bound (x)** | (-1.5, 0.5) | (-0.5, 2.0) | Different workspace |
| **Viewer eye** | (-2.25, 0.0, 0.75) | (2.25, 0.0, 0.75) | Flipped viewing angle |

---

### 5. **Domain Randomization Differences**

| Parameter | Kuka-Allegro | HDR-DG5F | Reason |
|-----------|-------------|----------|---------|
| **Robot friction** | [0.5, 1.0] | [0.9, 1.1] | HDR too heavy for wide randomization |
| **Stiffness scale** | [0.5, 2.0] | [0.9, 1.1] | Narrow range to maintain arm stability |
| **Damping scale** | [0.5, 2.0] | [0.9, 1.1] | Narrow range to maintain arm stability |
| **Wrist joint reset** | `iiwa7_joint_7: [-3, 3]` | `j4: [-0.1, 0.1]` | Conservative reset for stability |

**Reason:** HDR's heavy arm cannot support wide parameter variations without collapsing or unstable behavior.

---

## Learning Challenges & Solutions

### **Problem 1: Agent learns to "tap" object instead of grasping**
- **Symptom:** Single finger touches object momentarily, no multi-finger coordination
- **Solution:**
  - Increased `good_finger_contact` weight (0.5 → 3.0)
  - Added `grasp_duration` reward (requires 10-step sustained contact)

### **Problem 2: Agent "bounces" object off table to reach target**
- **Symptom:** Object repeatedly contacts ground, success reward increases without grasping
- **Solution:**
  - Added `ground_contact_penalty` (-1.5 weight, scaled to Kuka magnitudes)
  - ~~Added `object_lift` reward~~ → **REMOVED** (caused worse problems - see Problem 5)

### **Problem 3: Fingers too rigid, act like "steel rods"**
- **Symptom:** Fingers push object away instead of conforming to object shape
- **Solution:**
  - Reduced finger stiffness (30.0 → 8.0 → 12.0 → **5.0**)
  - Reduced finger damping (3.0 → 0.5 → 0.8 → **0.3**)
  - Final values only 1.7x and 3x Kuka (3.0/0.1) for soft compliant grasping

### **Problem 4: `good_finger_contact` reward always zero**
- **Symptom:** Middle finger never touches object due to rigid fingers
- **Solution:**
  - Relaxed contact condition to match Kuka-Allegro pattern
  - Changed from "thumb + middle(required) + other" to "thumb + any finger"

### **Problem 5: `object_lift` reward creates harmful shortcut** ⚠️ **CRITICAL**
- **Symptom:**
  - High average reward but zero `good_finger_contact`
  - Agent pushes object with arm/wrist instead of grasping with fingers
  - `fingers_to_object` stays at 0.1 level
  - No finger contact occurs throughout 7500 epochs
- **Root Cause:**
  - `object_lift` (weight=5.0) can be achieved without finger contact
  - Agent found easier path: push object up with arm motion
  - This became dominant strategy, preventing grasping exploration
- **Solution:**
  - **Removed `object_lift` entirely**
  - Keep reward structure similar to Kuka-Allegro (which works without lift reward)
  - Adjusted remaining reward weights to Kuka scale (0.5-10.0 range)
- **Lesson Learned:**
  - High-weight rewards must **require** the target behavior
  - Adding "helpful" rewards can backfire if they create shortcuts
  - Sometimes simpler is better - Kuka succeeded without lift reward

### **Problem 6: `success` reward achievable without grasping** ⚠️ **CRITICAL**
- **Symptom:**
  - Green table (success visualizer) activates without finger contact
  - Agent achieves success by pushing object to target with arm/wrist
  - Success reward obtained while `good_finger_contact` = 0
- **Root Cause:**
  - Original `success_reward` only checks position/orientation match
  - No requirement for finger contact
  - Inconsistent with `position_tracking` and `orientation_tracking` which ARE gated by contact
- **Solution:**
  - Added contact gating to `success_reward`: `base_reward * contacts(env, 1.0).float()`
  - Now consistent: ALL three tracking rewards require contact
  - Success only counts when object is at target AND fingers are touching
- **Comparison:**
  - **Kuka-Allegro:** No contact gating needed (agent naturally grasps)
  - **HDR-DG5F:** Contact gating necessary to prevent pushing shortcuts
- **Lesson Learned:**
  - Heavier robots find more shortcuts (pushing instead of grasping)
  - Consistency matters: if position/orientation need contact, success should too
  - Visualizers can reveal hidden shortcuts during training

---

## Files Modified

### **Robot Definition:**
- `isaaclab_assets/robots/hdr_dg5f_new.py`
  - Actuator settings (stiffness, damping, effort limits)
  - Initial joint positions

### **Environment Configuration:**
- `dexsuite_hdr/dexsuite_env_cfg.py`
  - Scene setup (object/table positions)
  - Base reward configuration
  - Domain randomization parameters

### **Task-Specific Configuration:**
- `dexsuite_hdr/config/hdr_dg5f_new/dexsuite_hdr_dg5f_new_env_cfg.py`
  - Contact sensor setup
  - Finger-specific rewards
  - Observation configuration

### **Reward Functions:**
- `dexsuite_hdr/mdp/rewards.py`
  - Modified: `contacts()` - relaxed conditions
  - New: `object_lift_height()` - lift reward
  - New: `object_ground_contact_penalty()` - ground penalty
  - New: `grasp_duration()` - sustained grasp reward

---

## Current Status & Next Steps

### **Completed Modifications:**
1. ✅ Finger actuator tuning (stiffness 5.0, damping 0.3)
2. ✅ Initial pose optimization for grasping
3. ✅ Contact condition relaxation (thumb + ring + any - user modified)
4. ❌ ~~Lift reward~~ → **REMOVED** (caused harmful shortcut)
5. ✅ Ground contact penalty (weight -1.5, scaled to Kuka)
6. ✅ Grasp duration reward (weight 1.5, scaled to Kuka)
7. ✅ Increased contact reward weight (0.5 → 3.0)
8. ✅ All reward weights scaled to match Kuka magnitudes (0.5-10.0 range)
9. ✅ **Success reward contact gating** (prevents pushing without grasping)

### **To Monitor (After Removing object_lift):**
- Does `good_finger_contact` increase from 0? (Currently stuck at 0)
- Does `fingers_to_object` increase from 0.1? (Currently stuck at 0.1)
- Does agent discover finger contact without shortcut reward?
- Do multiple fingers make contact simultaneously?
- Does learning curve show Kuka-like "sudden jump" when grasping is discovered?
- Visual check: Do fingers actually touch object?

### **Future Improvements (if needed):**
1. **Postural Synergy Actions:**
   - Reduce 20-DOF finger actions to high-level "grasp closure" parameter
   - Example: action = [arm_actions (6), grasp_amount (0-1)]
   - Would simplify exploration and encourage coordinated finger movement

2. **Curriculum Learning:**
   - Phase 1 (0-2000 epochs): Spawn object very close to hand
   - Phase 2 (2000-5000): Gradually increase object distance
   - Forces agent to experience grasping early in training

3. **Reward Scheduling:**
   - Phase 1: High weight on contact/lift, zero weight on position tracking
   - Phase 2: Gradually shift weight to position tracking and success
   - Ensures grasping is learned before object manipulation

---

## Differences Summary Table

| Category | Aspect | Kuka-Allegro | HDR-DG5F |
|----------|--------|-------------|----------|
| **Robot** | DOF | 7 + 16 = 23 | 6 + 20 = 26 |
| **Robot** | Weight | ~15kg (Kuka arm) | ~60kg (HDR arm) |
| **Actuator** | Finger stiffness | 3.0 | 3.0 (finally matched!) |
| **Actuator** | Finger damping | 0.1 | 0.1 (finally matched!) |
| **Actuator** | Arm stiffness | 300 max | 20000 max |
| **Reward** | good_finger_contact | 0.5 | 3.0 |
| **Reward** | success (contact gating) | ❌ No | ✅ Yes (requires contact) |
| **Reward** | object_lift | ❌ None | ❌ None (tried 5.0, removed - caused shortcut) |
| **Reward** | ground_contact_penalty | ❌ None | ✅ -0.5 (reduced from -1.5) |
| **Reward** | grasp_duration | ❌ None | ✅ 1.5 |
| **Contact** | Required fingers | thumb + any | **any 2 fingers** (maximally relaxed) |
| **Scene** | Object spawn | x=-0.55 | x=0.8 |
| **Randomization** | Stiffness range | [0.5, 2.0] | [0.9, 1.1] |

---

## Training Command

```bash
# Train HDR-DG5F
python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Dexsuite-HDR-DG5F-Lift-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 7500

# Play/Evaluate
python source/standalone/workflows/rl_games/play.py \
    --task Isaac-Dexsuite-HDR-DG5F-Lift-Play-v0 \
    --num_envs 32 \
    --checkpoint <path_to_checkpoint>
```

---

## References

- Original Dexsuite paper: [Link if available]
- Kuka-Allegro implementation: `dexsuite/config/kuka_allegro/`
- HDR20-17 URDF: `/home/dyros/ros2_ws/src/hdr_description/`
- DG5F URDF: `/home/dyros/ros2_ws/src/dg_description/`

---

## Revision History

### **v4 - 2025-10-28 (Current)**
- ✅ Added contact gating to `success_reward` function
- **Problem Found:** Green table (success) activating without finger contact
- **Fix:** Success now requires contact: `base_reward * contacts(env, 1.0).float()`
- **Impact:** Prevents "pushing to target" shortcut, requires actual grasping
- **Status:** Ready for training with comprehensive shortcut prevention

### **v3 - 2025-10-28**
- ❌ Removed `object_lift` reward - caused harmful shortcut behavior
- ✅ Scaled all rewards to match Kuka magnitudes (0.5-10.0 range)
- Ground penalty: -3.0 → -1.5
- Grasp duration: 2.0 → 1.5
- Result: Still found shortcuts through success reward

### **v2 - 2025-10-27**
- Added `object_lift`, `ground_contact_penalty`, `grasp_duration` rewards
- Result: Agent exploited object_lift, ignored finger contact
- Learned: High-weight rewards need careful design

### **v1 - 2025-10-27**
- Initial port from Kuka-Allegro
- Actuator tuning (stiffness 30.0 → 8.0 → 12.0 → 5.0)
- Contact condition relaxation

---

**Last Updated:** 2025-10-28
**Author:** dyros
**Status:** Ready for Training (v4 - success reward contact gating added)
