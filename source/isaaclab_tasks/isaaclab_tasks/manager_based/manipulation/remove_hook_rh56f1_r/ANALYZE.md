# 기술적 분석 - Remove Hook RH56F1_R 환경

## 문제 분석 (2026-01-12)

### 문제 1: 로봇이 훅을 향해 움직이지 않음

**증상**: 학습 중 로봇이 훅에 접근하지 않고 거의 정지 상태로 유지됨.

**근본 원인**: 보상 `std` 파라미터가 초기 거리에 비해 너무 작음.

보상 함수: `reward = 1 - tanh(distance / std)`

| 보상 항목 | 기존 std | 거리 | tanh(d/std) | 보상 |
|-------------|---------|----------|-------------|--------|
| fingers_to_hook | 0.1 | ~1.5m | tanh(15) ≈ 1.0 | ≈ 0 |
| hook_to_target | 0.02 | ~0.36m | tanh(18) ≈ 1.0 | ≈ 0 |

보상 ≈ 0이면 정책이 학습할 수 있는 gradient가 없음.

**수정**: 초기 거리에서 의미있는 gradient를 제공하도록 `std` 값 증가.

| 보상 항목 | 새 std | 거리 | tanh(d/std) | 보상 |
|-------------|---------|----------|-------------|--------|
| fingers_to_hook | 1.0 | ~1.5m | tanh(1.5) ≈ 0.9 | ≈ 0.1 |
| hook_to_target | 0.2 | ~0.36m | tanh(1.8) ≈ 0.95 | ≈ 0.05 |

**수정된 파일**:
- `config/hdr35_20_rh56f1_r/remove_hdr35_20_rh56f1_r_env_cfg.py`
  - Line 87: `std: 0.1 → 1.0` (fingers_to_hook)
  - Line 109: `std: 0.02 → 0.2` (hook_to_target)

---

### 문제 2: 에피소드가 너무 빨리 종료됨 (abnormal_robot termination)

**증상**: `abnormal_robot_state` 조건으로 인해 에피소드가 매우 일찍 종료됨.

**근본 원인**: HDR35 암이 iiwa(Kuka)보다 100배 높은 stiffness를 가짐.

```
iiwa (Kuka):  Kp = 300
HDR35:       Kp = 30,000  (100배 높음)
```

PD 컨트롤러 토크: `τ = Kp × Δq (position error)`

**핵심 통찰**: 100배 stiffness 차이를 action scale로 100배 보상하면 동일한 토크 범위 확보 가능!

```
iiwa:  τ = 300 × 0.1 = 30 Nm
HDR35: τ = 30,000 × 0.001 = 30 Nm  ← 동일!
```

**수정**:
1. action scale 감소: `0.1 → 0.001` (100배 감소로 토크 일치)
2. 속도 한계 multiplier: `2` 유지 (iiwa와 동일)

**수정된 파일**:
- `config/hdr35_20_rh56f1_r/remove_hdr35_20_rh56f1_r_env_cfg.py`
  - Line 59: `scale=0.1 → scale=0.001`
- `mdp/terminations.py`
  - Line 71: `joint_vel_limits * 2` (기존 유지)

---

## Action Scale 설명

Action scale은 스텝당 최대 조인트 위치 변화량을 결정:

```python
# joint_actions.py에서
processed_action = raw_action × scale

# RelativeJointPositionAction의 경우
target_position = current_position + processed_action
```

`scale=0.001`인 경우:
- Raw action 범위: [-1, 1]
- 스텝당 최대 변화: ±0.001 rad (≈ 0.057°)
- 50 Hz 제어에서: 최대 속도 ≈ 0.05 rad/s

**토크 일치 원리**:
```
토크 = Kp × Δq
iiwa:  300 × 0.1 = 30 Nm
HDR35: 30,000 × 0.001 = 30 Nm  ← 동일!
```

이로써 iiwa와 동일한 동역학 특성을 확보하여 abnormal_robot multiplier도 2로 유지 가능.

---

## Stiffness 비교

### iiwa (Kuka) - `kuka_allegro.py`에서
```python
stiffness={
    "iiwa_joint_1": 300.0,
    "iiwa_joint_2": 300.0,
    "iiwa_joint_3": 300.0,
    "iiwa_joint_4": 300.0,
    # ...
}
```

### HDR35 - `hdr35_20_rh56f1_r_sensor.py`에서
```python
stiffness={
    "j1": 30000.0,
    "j2": 30000.0,
    "j3": 30000.0,
    # ...
}
```

---

### 문제 3: j5/j6 조인트 과도한 회전 (2026-01-12 추가 분석)

**증상**: action scale 0.001, abnormal_robot multiplier 1000 설정에서도 j5/j6 조인트가 심하게 회전함.

**비교 분석: iiwa + Allegro vs HDR35 + RH56F1**

| 파라미터 | iiwa + Allegro | HDR35 + RH56F1 | 비고 |
|----------|----------------|----------------|------|
| 마지막 암 조인트 Kp | j7: 25, j6: 50 | j6: 1,500 | HDR35가 30~60배 높음 |
| 핸드 Kp | 3.0 | 2.5 | 유사 |
| 암 최약 조인트/핸드 비율 | 8.3:1 ~ 16.7:1 | **600:1** | **72배 불균형** |
| Action Scale | 0.1 (전 조인트) | 0.001 (전 조인트) | - |
| 핸드 토크/액션 | 3.0×0.1 = **0.3 Nm** | 2.5×0.001 = **0.0025 Nm** | HDR35 핸드가 1/120 |

**핵심 원인 분석**:

1. **극단적인 Stiffness 불균형**
   - iiwa j7/Allegro = 25/3 = **8.3:1** (균형 잡힘)
   - HDR35 j6/RH56F1 = 1,500/2.5 = **600:1** (72배 더 불균형!)

2. **핸드에 전달되는 토크 절대적 부족**
   - Allegro: 3.0 × 0.1 = 0.3 Nm
   - RH56F1: 2.5 × 0.001 = 0.0025 Nm (**1/120**)
   - 핸드가 움직이지 못하고 그 반작용이 j6에 전달됨

3. **iiwa j7의 "버퍼" 역할**
   - iiwa j7 stiffness 25 → 핸드의 관성/반작용을 흡수
   - HDR35 j6 stiffness 1,500 → 60배 강해서 버퍼 역할 불가

4. **Damping 비율 차이**
   - iiwa j7: damping/stiffness = 15/25 = **60%**
   - HDR35 j6: damping/stiffness = 150/1500 = **10%**
   - iiwa가 6배 더 높은 damping 비율로 진동 억제

**해결책: 조인트별 다른 Action Scale 적용**

stiffness/damping 변경 없이 action scale만 조정:

| 조인트 | Scale | 토크 = Kp × Scale | 비고 |
|--------|-------|-------------------|------|
| j1-5 | 0.01 | 30,000×0.01 = 300 Nm (j1-3), 3,000×0.01 = 30 Nm (j4-5) | 기본 암 |
| j6 | 0.02 | 1,500×0.02 = 30 Nm | 핸드 연결 버퍼 |
| hand | 0.1 | 2.5×0.1 = 0.25 Nm | Allegro 수준 (0.3 Nm) |

**수정된 파일**:
- `config/hdr35_20_rh56f1_r/remove_hdr35_20_rh56f1_r_env_cfg.py`
  ```python
  scale={
      r"j[1-5]": 0.01,                                    # 암 j1-j5
      r"j6": 0.02,                                        # 암 j6 (wrist roll)
      r"right_thumb_(1|2)_joint": 0.1,                    # RH56F1 엄지
      r"right_(index|middle|ring|little)_1_joint": 0.1,  # RH56F1 손가락
  }
  ```

---

## 변경 사항 요약

| 항목 | 변경 전 | 변경 후 | 이유 |
|-----------|--------|-------|--------|
| Action scale (j1-5) | 0.1 → 0.001 | 0.01 | 적절한 토크 범위 |
| Action scale (j6) | 0.1 → 0.001 | 0.02 | 버퍼 역할, j5/j6 회전 방지 |
| Action scale (hand) | 0.1 → 0.001 | 0.1 | Allegro 수준 토크 (0.25 Nm) |
| abnormal_robot multiplier | 2 | 1000 (임시) | 학습 우선, sim-to-real 시 조정 필요 |
| fingers_to_hook std | 0.1 | 1.0 | 초기 거리 ~1.5m에서 gradient 확보 |
| hook_to_target std | 0.02 | 0.2 | 타겟 거리 ~0.36m에서 gradient 확보 |

---

## RH56F1_R 핸드 구조

RH56F1_R (Inspire 오른손)은 6 DoF의 under-actuated 핸드:

- **손가락 끝 Body** (접촉 감지용):
  - `right_thumb_4` - 엄지 끝
  - `right_index_2` - 검지 끝
  - `right_middle_2` - 중지 끝
  - `right_ring_2` - 약지 끝
  - `right_little_2` - 소지 끝

- **Palm Body**: `gripper_base_link`

- **타겟 훅**: `left_ring` (오른손이 왼쪽에서 접근)

---

## 파일 구조

```
remove_hook_rh56f1_r/
├── __init__.py                 # 태스크 등록
├── remove_hook_env_cfg.py      # 기본 환경 설정
├── README.md                   # 개요 문서
├── ANALYZE.md                  # 기술적 분석 (이 파일)
├── config/
│   └── hdr35_20_rh56f1_r/
│       ├── __init__.py
│       ├── remove_hdr35_20_rh56f1_r_env_cfg.py  # 로봇별 설정
│       └── agents/
│           └── rl_games_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    ├── commands.py             # 타겟 위치 명령
    ├── observations.py         # 관측 함수
    ├── rewards.py              # 보상 함수
    └── terminations.py         # 종료 조건
```
