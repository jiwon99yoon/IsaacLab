# Stack RL Reward Structure Redesign - Implementation Summary
## 📅 20251210 1601 수정

---

## ✅ 완료된 작업

### 1. 문서 업데이트
- ✅ **WORKFLOW_KR.md**: "20251210 1601 - Reward 구조 완전 재설계" 섹션 추가
- ✅ **PPO_INFO_KR.md**: Observation 및 Reward 변경사항 문서화

### 2. 코드 구현

#### 📝 `mdp/rewards.py` - 14개 함수 추가/수정

**Helper Functions (3개)**
1. `check_cube_2_on_cube_1()` - Masking용: cube_2가 cube_1 위에 쌓였는지 확인
2. `check_object_grasped()` - 물체를 잡았는지 확인 (EE 거리 + gripper 닫힘)
3. `check_gripper_open()` - 그리퍼가 열려있는지 확인

**Observation Function (1개)**
4. `cube_dimensions_in_world_frame()` - 각 큐브의 3D 크기 반환 (9-dim)

**New Reward Functions (8개)**
5. `object_is_lifted_after_grasp()` - 잡은 후 들기 확인
6. `stacking_hybrid_reward()` - Dense (접근) + Sparse (일회성 성공) 하이브리드
7. `reaching_cube_3_masked()` - cube_3 reaching (masked)
8. `grasping_cube_3_masked()` - cube_3 grasping (masked)
9. `lifting_cube_3_masked()` - cube_3 lifting (masked)
10. `stacking_cube_3_on_2_masked()` - cube_3 stacking (masked + hybrid)
11. `object_stays_on_table()` - 물체가 테이블 위에 유지되는지 확인
12. `object_movement_penalty()` - 물체 움직임 페널티

**Stability & Success (2개)**
13. `stack_stability_when_released()` - 그리퍼 열렸을 때 안정성 체크
14. `three_cubes_stacked_success_v2()` - 성공 판정 (cube_1 테이블 체크 추가)

#### 📝 `stack_rl_env_cfg.py` - 2곳 수정

**Observation 추가**
- `ObservationsCfg.PolicyCfg`에 `cube_dimensions` 추가 (9-dim)
- 총 observation 차원: **56-dim → 65-dim**

**Reward 완전 재설계**
- ❌ **삭제**: cube_1 들기 관련 보상 3개 (reaching, grasping, lifting)
- ✅ **Phase 1** (cube_2 쌓기): 4개 보상 (최대 53)
  - reaching_cube_2: 3.0 (1.0에서 증가)
  - grasping_cube_2: 10.0
  - lifting_cube_2: 5.0 (새 함수)
  - stacking_cube_2_on_1: hybrid (5.0 dense + 30.0 sparse)

- ✅ **Phase 2** (cube_3 쌓기, masked): 4개 보상 (최대 73)
  - reaching_cube_3: 3.0 (masked)
  - grasping_cube_3: 10.0 (masked)
  - lifting_cube_3: 5.0 (masked)
  - stacking_cube_3_on_2: hybrid (5.0 + 50.0, masked)

- ✅ **Constraints**: 3개 보상 (최대 3)
  - cube_1_stays_on_table: 2.0
  - cube_1_movement_penalty: -1.0
  - stack_stability: 1.0 (그리퍼 열렸을 때만)

- ✅ **Success**: 1개 보상 (최대 100)
  - task_success: 100.0 (v2 함수 사용)

**새 최대 보상**: 227 → **229**
- 계산: 53 + 73 + 3 + 100 = 229

---

## 🔍 검증 완료

### Syntax Validation
```bash
✓ rewards.py syntax is valid
✓ stack_rl_env_cfg.py syntax is valid
✓ 14 new/modified functions confirmed
```

### Code Structure
- ✅ 모든 함수에 "20251210 1601 수정:" 주석 포함
- ✅ 한글 docstring으로 설명 작성
- ✅ `mdp/__init__.py`에서 자동 export 확인
- ✅ 기존 코드 패턴 준수 (env.cfg 사용)

---

## 🚀 다음 단계: 학습 시작

### 1. 환경 로드 테스트 (Isaac Sim 필요)

```bash
# 환경이 정상적으로 로드되는지 확인
./isaaclab.sh -p scripts/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --headless \
    --num_envs 16 \
    --dry_run  # 환경만 로드하고 종료
```

### 2. 새 학습 시작

```bash
# 처음부터 새로 학습 시작 (이전 체크포인트 사용 안 함)
./isaaclab.sh -p scripts/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 15000
```

### 3. Tensorboard로 모니터링

```bash
tensorboard --logdir logs/rl_games/stack_cube_franka
```

### 4. 확인할 주요 지표

**Reward 분해 (Tensorboard에서 확인)**
- `rewards/reaching_cube_2`: Phase 1 시작 신호
- `rewards/stacking_cube_2_on_1`: Phase 1 성공
- `rewards/reaching_cube_3`: Phase 2 활성화 (masked 해제)
- `rewards/stacking_cube_3_on_2`: Phase 2 성공
- `rewards/cube_1_stays_on_table`: 항상 높아야 함 (~2.0)
- `rewards/task_success`: 최종 성공

**학습 진행 예상**
- **Epoch 0-3000**: cube_2 reaching & grasping 학습
- **Epoch 3000-8000**: cube_2 stacking 성공 증가
- **Epoch 8000-12000**: cube_3 Phase 활성화 (masked 해제)
- **Epoch 12000-15000**: 3-cube stacking 성공률 증가

---

## ⚠️ 주의사항

### 1. 이전 학습 데이터와의 차이
- **이전 최대 보상**: 227
- **새 최대 보상**: 229
- **Observation 차원**: 56 → 65
- ⚠️ **이전 체크포인트 호환 불가**: observation 차원이 달라서 이전 모델 로드 안 됨

### 2. Reward Hacking 방지
- Hybrid reward의 one-time flag가 환경에 저장됨
- Episode reset 시 자동으로 초기화됨
- 한 번 성공하면 그 episode에서는 더 이상 sparse 보상 안 받음

### 3. Masked Rewards
- cube_3 관련 모든 보상은 `check_cube_2_on_cube_1()` 조건 통과해야 활성화
- 조건: xy 거리 < 2cm, z 거리 < 3mm
- Phase 1 완료 전에는 cube_3 reward = 0

### 4. Sim2Real 고려사항
- `stack_stability_when_released()`는 velocity 기반 → 시뮬레이션 전용
- 실제 로봇 배포 시 vision-based stability check로 교체 필요

---

## 📊 기대 효과

### Before (이전 잘못된 학습)
```
Epoch 22000: Reward 219.57/227
❌ cube_1 들어올리기 학습 (불필요한 동작)
❌ cube_2, cube_3 순서 없이 랜덤하게 시도
❌ Reward hacking (cube_2 계속 미세조정)
```

### After (새 설계)
```
Epoch 15000 (예상): Reward ~200+/229
✅ cube_1은 테이블 위 고정 (base 역할)
✅ Phase 1 → Phase 2 순차 실행 (masked)
✅ 한 번 성공하면 다음 phase로 진행 (hybrid)
✅ 정확한 정렬 (xy: 2cm, z: 3mm)
```

---

## 🔧 디버깅 Tips

### 문제: cube_3 reward가 너무 늦게 활성화
**해결**: `check_cube_2_on_cube_1()`의 threshold 완화
```python
# mdp/rewards.py line ~243
def check_cube_2_on_cube_1(..., xy_threshold: float = 0.03, ...):  # 0.02 → 0.03
```

### 문제: Reward hacking이 여전히 발생
**해결**: Hybrid reward의 success_weight 증가
```python
# stack_rl_env_cfg.py
stacking_cube_2_on_1 = RewTerm(..., success_weight=50.0, ...)  # 30 → 50
```

### 문제: 학습이 너무 느림
**해결**: Reaching reward weight 더 증가
```python
# stack_rl_env_cfg.py
reaching_cube_2 = RewTerm(..., weight=5.0, ...)  # 3 → 5
```

---

## 📁 수정된 파일 목록

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_rl/
├── WORKFLOW_KR.md                  # 📝 섹션 추가
├── PPO_INFO_KR.md                  # 📝 업데이트 노트 추가
├── mdp/
│   └── rewards.py                  # ✏️ 14개 함수 추가/수정
└── stack_rl_env_cfg.py             # ✏️ Observation + Reward 재설계
```

---

## ✅ 체크리스트

- [x] 문서 업데이트 (WORKFLOW_KR.md, PPO_INFO_KR.md)
- [x] Helper 함수 3개 구현
- [x] Observation 함수 1개 구현
- [x] Reward 함수 8개 구현
- [x] Success 함수 수정
- [x] Config 파일에 observation 추가
- [x] Config 파일 reward 재설계
- [x] Syntax validation
- [ ] 환경 로드 테스트 (Isaac Sim 필요)
- [ ] 새 학습 시작
- [ ] Tensorboard 모니터링

---

## 📞 문제 발생 시

1. **Syntax 에러**: Python syntax는 검증 완료, import 에러라면 Isaac Sim 환경 확인
2. **Runtime 에러**: `env.cfg` 속성 누락 에러는 robot config (joint_pos_env_cfg.py) 확인
3. **Reward 이상**: Tensorboard에서 각 reward term 별도 확인
4. **학습 안 됨**: Curriculum learning 적용 여부 확인 (CurriculumCfg)

**모든 코드는 timestamp "20251210 1601 수정"으로 검색 가능합니다.**

---

## 📚 참고 문서

- 상세 설계: `WORKFLOW_KR.md` (20251210 1601 섹션)
- PPO 설정: `PPO_INFO_KR.md`
- 기존 워크플로: `WORKFLOW.md`
