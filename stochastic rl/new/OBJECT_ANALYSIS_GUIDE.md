# Object-Specific Performance Analysis Guide

## 질문: 물체별 성능 분석이 가능한가?

### 답변: **부분적으로 가능하지만, 코드 수정 없이는 제한적입니다**

---

## 1. 현재 상황 분석

### 1.1 물체 종류 (16가지)

우리 환경은 **16가지 서로 다른 기하학적 primitive 물체**를 사용합니다:

```python
# source/isaaclab_tasks/.../dexsuite_env_cfg.py
object: RigidObjectCfg = RigidObjectCfg(
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            # Cuboids (6가지)
            CuboidCfg(size=(0.05, 0.1, 0.1)),      # 1
            CuboidCfg(size=(0.05, 0.05, 0.1)),     # 2
            CuboidCfg(size=(0.025, 0.1, 0.1)),     # 3
            CuboidCfg(size=(0.025, 0.05, 0.1)),    # 4
            CuboidCfg(size=(0.025, 0.025, 0.1)),   # 5
            CuboidCfg(size=(0.01, 0.1, 0.1)),      # 6

            # Spheres (2가지)
            SphereCfg(radius=0.05),                # 7
            SphereCfg(radius=0.025),               # 8

            # Capsules (6가지)
            CapsuleCfg(radius=0.04, height=0.025), # 9
            CapsuleCfg(radius=0.04, height=0.01),  # 10
            CapsuleCfg(radius=0.04, height=0.1),   # 11
            CapsuleCfg(radius=0.025, height=0.1),  # 12
            CapsuleCfg(radius=0.025, height=0.2),  # 13
            CapsuleCfg(radius=0.01, height=0.2),   # 14

            # Cones (2가지)
            ConeCfg(radius=0.05, height=0.1),      # 15
            ConeCfg(radius=0.025, height=0.1),     # 16
        ],
    ),
)
```

**Inspire Hand는 0.5x 스케일**, **DG5F Hand는 1.0x 스케일** 사용.

---

### 1.2 현재 로깅 상황

**✅ 로그되는 것**:
- Episode reward (전체 합)
- Success rate (전체 평균)
- Episode length (전체 평균)
- 각 reward component (전체 평균)
  - `fingers_to_object`
  - `position_tracking`
  - `good_finger_contact`
  - `success`
  - etc.

**❌ 로그되지 않는 것**:
- **어떤 물체가 spawn되었는지 (object index)**
- **물체별 success rate**
- **물체별 reward**
- **물체별 episode length**

---

### 1.3 물체별 분석이 어려운 이유

**Isaac Lab의 RL-Games 통합은 기본적으로 "전체 평균"만 로깅**합니다:

```python
# tensorboard에 기록되는 값들
self.writer.add_scalar('Train/success_rate', mean_success_rate, self.frame)
self.writer.add_scalar('Train/episode_reward_mean', mean_reward, self.frame)
```

**물체 정보는 environment 내부에만 존재**하고, RL algorithm에 노출되지 않습니다:
- Environment가 spawn한 물체 index는 `env.scene.object` 내부에 저장
- 하지만 이 정보가 Tensorboard 로그에 기록되지 않음

---

## 2. 코드 수정 없이 가능한 분석

### 2.1 간접적 분석: Success Rate 분포

비록 물체별로 직접 분리할 수는 없지만, **전체 success rate의 분포**를 보면 간접적으로 유추 가능:

```python
# Tensorboard에서 확인 가능한 지표
- Train/success_rate_mean: 전체 평균 (예: 75%)
- Train/success_rate_std: 표준편차 (예: 0.15)
```

**해석**:
- **표준편차가 크다** → 물체에 따라 성능 차이 큼
- **표준편차가 작다** → 물체에 관계없이 일정한 성능

### 2.2 Episode Length 분석

```python
# Tensorboard에서 확인 가능
- Train/episode_length_mean: 평균 episode 길이
```

**해석**:
- 평균 episode length가 짧다 → 빨리 성공하거나 빨리 실패
- 평균 episode length가 길다 → 오래 시도하거나 timeout

### 2.3 Reward Component 분석

```python
# 각 reward component의 평균값으로 간접 유추
Rewards/fingers_to_object    # 물체까지 거리
Rewards/good_finger_contact  # 접촉 빈도
Rewards/position_tracking    # 위치 추적 정확도
```

**예시 해석**:
- `good_finger_contact`가 낮다 → 물체가 너무 작거나 접촉하기 어려운 형상
- `fingers_to_object`가 낮다 → 물체를 찾기 어려움 (작은 물체?)

---

## 3. 물체별 분석을 위한 코드 수정 (참고용)

**주의: 현재는 코드 수정 금지이므로, 나중에 필요하면 참고**

### 3.1 Object Index 로깅 추가

```python
# source/isaaclab/envs/manager_based_rl_env.py (수정 예시 - 실제로는 하지 말 것!)

# step() 함수에서 extras에 object index 추가
extras = {
    "log": {
        **log_info,
        "object_index": self.scene.object.data.asset_ids,  # 각 env의 물체 index
    }
}
```

### 3.2 물체별 Success Rate 계산

```python
# scripts/rsl_rl/train.py (수정 예시 - 실제로는 하지 말 것!)

# 물체별로 success rate 집계
object_success_rates = {}
for obj_idx in range(16):  # 16가지 물체
    mask = (infos["object_index"] == obj_idx)
    if mask.sum() > 0:
        success_rate = infos["is_success"][mask].mean()
        object_success_rates[f"Object_{obj_idx}_success"] = success_rate
```

---

## 4. 현재 상황에서 할 수 있는 분석

### 4.1 전체 성능 지표

✅ **가능**:
- Inspire vs DG5F 전체 success rate 비교
- 학습 속도 비교 (sample efficiency)
- 안정성 비교 (variance across seeds)

### 4.2 물체 유형별 간접 유추

⚠️ **제한적으로 가능** (정확하지 않음):

**방법 1: Episode Replay 분석**
- Play 모드로 evaluation 실행
- 각 물체에 대해 100 episode 실행
- 수동으로 success rate 기록

```bash
# 예시: 특정 물체로만 evaluation (가능하면)
python scripts/rsl_rl/play.py \
    --task Isaac-Dexsuite-Ur10e-Inspire-Lift-Play-v0 \
    --num_envs 1 \
    --checkpoint <checkpoint>
# 각 episode마다 어떤 물체인지 육안으로 확인하고 기록
```

**방법 2: Success Threshold 분석**
- 만약 특정 물체가 특별히 어려우면, 전체 success rate에 영향
- 예: 작은 sphere (radius=0.025)는 모든 손이 어려워함
- → Success rate 낮은 구간에서 특정 물체 형상 추론 가능

### 4.3 질적 분석 (Qualitative)

✅ **가능**:
- Play 모드로 영상 녹화
- 각 물체별로 성공/실패 케이스 육안 관찰
- 손별로 "어떤 물체를 잘 잡는가" 정성적 파악

**예시 관찰**:
```
Inspire Hand (6-DoF, underactuated):
  - Sphere: 어려움 (접촉면 작음)
  - Large Cuboid: 쉬움 (넓은 접촉면)
  - Thin Capsule: 중간 (기계적 compliance가 도움)

DG5F Hand (20-DoF, fully-actuated):
  - Sphere: 쉬움 (정밀 제어로 감싸기 가능)
  - Large Cuboid: 쉬움 (충분한 자유도)
  - Thin Capsule: 어려움? (과도한 자유도가 오히려 불안정)
```

---

## 5. 논문에서 물체별 분석을 언급할 수 있는가?

### 5.1 정량적 분석: **❌ 불가능** (코드 수정 없이)

물체별 success rate 수치를 제시할 수 없음:
- ❌ "Inspire Hand achieves 80% success on spheres but 90% on cuboids"
- ❌ "DG5F Hand shows 15% better performance on complex shapes"

### 5.2 질적 관찰: **✅ 가능** (제한적)

Play 모드 영상 기반으로 정성적 언급:
- ✅ "We observe that underactuated hands show more stable grasps on regular geometries (cuboids, large spheres)"
- ✅ "Fully-actuated hands demonstrate better adaptability to thin/elongated objects (capsules)"
- ✅ "Both hands struggle with small spheres (radius < 0.03m), suggesting task difficulty is independent of actuation type for extreme cases"

### 5.3 Future Work로 언급: **✅ 강력 추천**

```markdown
## 5.3 Limitations and Future Work

While our current study provides insights into how hand actuation structure
affects **overall** learning dynamics, we did not analyze performance
differences across **object categories** (spheres, cuboids, capsules, cones).

Future work should investigate:
1. **Object-stratified analysis**: Log object indices during training to
   compute per-object success rates and identify which geometries benefit
   from specific actuation structures.

2. **Shape complexity curriculum**: Design curricula that progressively
   introduce more complex shapes, analyzing how underactuated vs
   fully-actuated hands adapt to increasing geometric diversity.

3. **Sim-to-real transfer**: Evaluate whether object-specific performance
   differences observed in simulation translate to real-world deployments.
```

---

## 6. 결론 및 권장사항

### ✅ 현재 가능한 분석 (코드 수정 없이)

1. **전체 성능 비교** (Inspire vs DG5F):
   - Success rate, sample efficiency, stability
   - ✅ **이것이 우리 논문의 핵심!**

2. **질적 관찰** (Play 모드 영상):
   - 어떤 물체를 잘/못 잡는지 정성적 파악
   - ✅ **Discussion에서 언급 가능**

3. **간접 지표** (Tensorboard 분석):
   - Success rate variance → 물체 간 성능 차이 간접 유추
   - ✅ **보조 지표로 활용**

### ❌ 현재 불가능한 분석

1. **물체별 정량적 성능**:
   - "Cuboid: 85%, Sphere: 65%" 같은 수치
   - ❌ **코드 수정 필요**

2. **물체별 학습 곡선**:
   - "Cuboid는 5M step에 수렴, Sphere는 15M step"
   - ❌ **코드 수정 필요**

### 📝 논문 작성 권장사항

**Main Contribution**:
- Hand actuation structure의 영향 (전체 성능 기준)
- ✅ **이것만으로도 충분히 강력한 contribution!**

**Discussion/Limitation**:
- 물체별 분석은 안 했다고 명시
- Future work로 제시
- 질적 관찰은 언급 가능 (영상 있으면)

**강점**:
- 솔직하게 limitation 인정 → 논문 신뢰도 ↑
- Future work 명확히 제시 → 후속 연구 방향 제시

---

## 7. 요약

| 질문 | 답변 |
|------|------|
| **물체별 정량 분석 가능?** | ❌ 코드 수정 없이는 불가능 |
| **물체별 질적 분석 가능?** | ✅ Play 모드 영상으로 가능 (제한적) |
| **전체 성능 비교 가능?** | ✅ 충분히 가능 (이것이 우리 논문의 핵심!) |
| **논문에 포함해야 하나?** | ⚠️ Limitation/Future Work로 언급 권장 |
| **논문 완성도에 영향?** | ❌ 없음 (오히려 솔직한 limitation이 신뢰도 높임) |

**결론**: 물체별 분석은 흥미로운 아이디어지만, **현재 코드 수정 없이는 불가능**합니다. 하지만 **전체 성능 비교만으로도 충분히 강력한 논문**이 될 수 있습니다. 물체별 분석은 **Future Work**로 남겨두세요!
