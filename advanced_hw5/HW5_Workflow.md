# HW5 Controller Workflow

## 개요

HW5는 Task Space Control을 구현하며, 각 문제를 **준비 단계(Preparation)**와 **실행 단계(Execution)**로 분리하여 구현되었습니다.

## 키 매핑 및 실행 흐름

### 공통 시작
```bash
python main.py --control_mode torque
```

시뮬레이터 실행 후:
- **키 2**: `joint_ctrl_home` - HW4 초기 위치로 이동 (torque mode 시)

---

## Problem 1-1: 2cm y0 방향 Step Response

### 실행 순서
```
2번 키 → 3번 키 → 4번 키
```

### 단계별 설명
1. **키 2 (joint_ctrl_home)**:
   - 초기 위치로 이동
   - torque mode: `[0, 0, 0, -30°, 0, 90°, 0]`

2. **키 3 (hw5_prepare_problem1)**:
   - qi1 = `[0, 0, 0, -90°, 0, 90°, 45°]`로 이동
   - Joint Space PD + M + G 제어 사용 (HW4_4와 동일)
   - **로깅 없음** (준비 단계)

3. **키 4 (hw5_1_step)**:
   - Task Space Control 시작
   - EE를 y0 방향으로 **2cm step** 이동
   - **로깅 활성화** (x, y, z 위치 기록)
   - 제어기: τ = J^T Λ F*_0 + N^T τ_0 + G

### 로그 파일
- `log.txt`: EE의 x, y, z 위치 기록

---

## Problem 1-2: 10cm y0 방향 Cubic Spline Trajectory

### 실행 순서
```
2번 키 → 3번 키 → 5번 키
```

### 단계별 설명
1. **키 2 (joint_ctrl_home)**:
   - 초기 위치로 이동

2. **키 3 (hw5_prepare_problem1)**:
   - qi1 = `[0, 0, 0, -90°, 0, 90°, 45°]`로 이동
   - Joint Space PD + M + G 제어
   - **로깅 없음**

3. **키 5 (hw5_1_cubic)**:
   - Task Space Control 시작
   - EE를 y0 방향으로 **10cm cubic spline** 이동
   - 이동 시간: 3초
   - **로깅 활성화** (x, y, z 위치 기록)
   - 제어기: τ = J^T Λ F*_0 + N^T τ_0 + G

### 로그 파일
- `log.txt`: EE의 x, y, z 위치 기록

---

## Problem 2: Velocity Saturation Controller

### 실행 순서
```
2번 키 → 6번 키 → 7번 키
```

### 단계별 설명
1. **키 2 (joint_ctrl_home)**:
   - 초기 위치로 이동

2. **키 6 (hw5_prepare_problem2)**:
   - qi2 = `[0, -60°, 0, -90°, 0, 30°, 45°]`로 이동
   - Joint Space PD + M + G 제어
   - **로깅 없음** (준비 단계)

3. **키 7 (hw5_2_velocity_sat)**:
   - Task Space Velocity Saturation Control 시작
   - 목표: xd = `[0.3, -0.012, 0.52]^T` [m]
   - 속도 제한: ẋmax = 0.3 m/s
   - 초기 orientation 유지
   - **로깅 활성화** (x, y, z 위치 기록)
   - 제어기: F* = kv(ẋd - ẋ) with velocity saturation

### 로그 파일
- `log.txt`: EE의 x, y, z 위치 기록

---

## 제어 알고리즘 상세

### 준비 단계 (키 3, 6)
**Joint Space Dynamic Compensation (HW4_4)**
```
τ = M{kp(qi - q) + kv(-q̇)} + G
```
- kp = 400, kv = 40 (ωn = 20 rad/s, ζ = 1)
- 목적: 안정적으로 qi 위치로 이동
- 로깅: 없음

### 실행 단계 (키 4, 5, 7)
**Task Space Dynamic Compensation**

#### 기본 구조
```
τ = J^T Λ F*_0 + N^T τ_0 + G
```

여기서:
- **Λ = (J M^-1 J^T)^-1**: Task space inertia matrix
- **J̄ = M^-1 J^T Λ**: Dynamically consistent inverse
- **N = I - J̄^T J**: Dynamic nullspace projector

#### Task Space Force (F*_0)

**Position Control (F*)**:
- Problem 1-1 (Step): `F* = kp(xd - x) - kv ẋ`
- Problem 1-2 (Cubic): `F* = kp(xd - x) + kv(ẋd - ẋ)`
- Problem 2 (Vel Sat): `F* = kv(ẋd_sat - ẋ)`

**Orientation Control (M*)**:
```
δΦ = 0.5(nd × nc + sd × sc + ad × ac)  # Axis-angle error
M* = -kp δΦ - kv ω
```

**Combined Force**:
```
F*_0 = [F*; M*] ∈ ℝ^6
```

#### Nullspace Control (τ_0)
```
τ_0 = M{kp(qi - q) - kv q̇}
```
- qi: 각 문제의 초기 joint configuration
- Problem 1: qi1 = [0, 0, 0, -π/2, 0, π/2, π/4]
- Problem 2: qi2 = [0, -π/3, 0, -π/2, 0, π/6, π/4]

#### Gains
- kp = 400 (ωn^2)
- kv = 40 (2ζωn)
- ωn = 20 rad/s, ζ = 1 (critically damped)

---

## Velocity Saturation 상세 (Problem 2)

### 알고리즘
```python
# 1. Unconstrained desired velocity
xd_dot_unc = (kp/kv) × (xd - x)  # = 10 × error

# 2. Saturation check
if ||xd_dot_unc|| > xd_max:
    xd_dot = (xd_max / ||xd_dot_unc||) × xd_dot_unc
else:
    xd_dot = xd_dot_unc

# 3. Control force
F* = kv × (xd_dot - ẋ)
```

### 동작 특성
- **큰 오차 (> 3cm)**: 일정한 0.3 m/s 속도로 이동
- **작은 오차 (< 3cm)**: 감속하며 목표 접근
- **Threshold**: ||xd - x|| = xd_max / (kp/kv) = 0.03 m

---

## 로깅 정책

### 로깅 활성화 (키 4, 5, 7)
- Task Space Control이 실행되는 동안만 로깅
- 기록 내용: `play_time x_pos y_pos z_pos`
- 파일: `log.txt`

### 로깅 비활성화 (키 2, 3, 6)
- Joint Space Control 준비 단계에서는 로깅 없음
- 목적: Task Space Control 성능만 순수하게 평가

---

## HW4와의 차이점

### HW4 (Joint Space Control)
- 제어 변수: q ∈ ℝ^7 (joint angles)
- 동역학: M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
- 제어: τ = M{kp(qd-q) + kv(q̇d-q̇)} + G
- 결과: q̈ ≈ kp(qd-q) + kv(q̇d-q̇)

### HW5 (Task Space Control)
- 제어 변수: x ∈ SE(3) (EE pose)
- 동역학: Λ(q)ẍ + μ(q,q̇) + p(q) = F
- 제어: τ = J^T Λ F* + N^T τ_0 + G
- 결과: ẍ ≈ kp(xd-x) + kv(ẋd-ẋ)

### 핵심 차이
1. **제어 공간**: Joint space → Task space
2. **Inertia 보상**: M matrix → Λ matrix
3. **Nullspace**: 없음 → Dynamic nullspace (N)
4. **Orientation**: 직접 제어 안함 → Axis-angle error
5. **경로 계획**: Joint space → Cartesian space (더 직관적)

---

## 실행 예시

### HW5-1-1 (2cm step) 실행
```bash
# Terminal 1
python main.py --control_mode torque

# Viewer에서
[2] → [3] → [4]

# 결과: EE가 y0 방향으로 2cm 상승, log.txt 생성
```

### HW5-1-2 (10cm cubic) 실행
```bash
# Terminal 1
python main.py --control_mode torque

# Viewer에서
[2] → [3] → [5]

# 결과: EE가 y0 방향으로 10cm 부드럽게 상승, log.txt 생성
```

### HW5-2 (velocity saturation) 실행
```bash
# Terminal 1
python main.py --control_mode torque

# Viewer에서
[2] → [6] → [7]

# 결과: EE가 [0.3, -0.012, 0.52]로 속도 제한하며 이동, log.txt 생성
```

---

## 참고사항

1. **Static 변수 초기화**
   - 각 실행 함수는 `is_mode_changed_` 플래그로 static 변수 초기화
   - 모드 전환 시 자동으로 초기 상태 저장

2. **Orientation Error**
   - Axis-angle representation 사용 (quaternion 아님)
   - δΦ = 0.5(nd×nc + sd×sc + ad×ac)
   - 이론: REinteraction-7.pdf, p.19-21

3. **Nullspace Control**
   - Task에 영향 없이 joint configuration 최적화
   - N^T J = 0 (dynamic decoupling)
   - qi 위치 유지 시도

4. **로그 분석**
   - Column 1: time [s]
   - Column 2: x [m]
   - Column 3: y [m]  ← y0 방향 이동 확인
   - Column 4: z [m]

---

## 문제 해결

### 로봇이 움직이지 않을 때
- torque mode로 실행했는지 확인: `--control_mode torque`
- 키 순서 확인 (2→3→4 또는 2→3→5 또는 2→6→7)

### 로그가 생성되지 않을 때
- 키 4, 5, 7에서만 로깅됨 (3, 6은 준비 단계)
- 실행 후 충분한 시간 대기

### 제어가 불안정할 때
- qi 위치에 충분히 도달했는지 확인 (키 3 또는 6 후 2-3초 대기)
- gains 확인: kp=400, kv=40

---

**작성일**: 2025
**과제**: Advanced Robotics HW5 - Task Space Control
**참고**: hw5_discussion.txt
