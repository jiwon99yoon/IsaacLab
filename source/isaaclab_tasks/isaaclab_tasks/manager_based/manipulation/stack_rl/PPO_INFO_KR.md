# Stack-RL: PPO 훈련 및 MDP 설정

## 개요

이 문서는 PPO(Proximal Policy Optimization)로 훈련되는 Stack-RL 환경의 **MDP (Markov Decision Process)** 정식화, 관찰/액션 공간, 보상 구조, 제어 루프에 대한 상세한 기술 사양을 제공합니다.

---

## 1. MDP 정식화

### 상태 공간 (S)

환경의 **실제 상태**는 다음을 포함합니다:
- 로봇 조인트 위치 (7개 암 + 2개 그리퍼 = 9 자유도)
- 로봇 조인트 속도 (9 자유도)
- 엔드 이펙터 자세 (위치 + 쿼터니언 = 7차원)
- Cube 1 자세 (위치 + 쿼터니언 = 7차원)
- Cube 2 자세 (위치 + 쿼터니언 = 7차원)
- Cube 3 자세 (위치 + 쿼터니언 = 7차원)
- Cube 1 속도 (선형 + 각 = 6차원)
- Cube 2 속도 (선형 + 각 = 6차원)
- Cube 3 속도 (선형 + 각 = 6차원)

**전체 상태 차원**: ~60+ 차원 (시뮬레이션에서 완전히 관찰 가능)

### 관찰 공간 (O)

**정책 관찰**은 다음과 같이 구성된 **56차원 벡터**입니다:

#### 조인트 상태 (11차원)
- **조인트 위치** (7차원): 기본 자세에 대한 상대 위치
  - `panda_joint1`부터 `panda_joint7`까지
  - 함수: `mdp.joint_pos_rel()`
  - 조인트 한계로 정규화됨
- **조인트 속도** (4차원): 상대 속도 (부분집합)
  - 함수: `mdp.joint_vel_rel()`
  - 속도를 위해 가장 중요한 조인트로 제한

#### 물체 상태 (21차원)
- **큐브 위치** (9차원): 월드 프레임 위치
  - Cube 1: [x₁, y₁, z₁]
  - Cube 2: [x₂, y₂, z₂]
  - Cube 3: [x₃, y₃, z₃]
  - 함수: `mdp.cube_positions_in_world_frame()`

- **큐브 방향** (12차원): 월드 프레임 쿼터니언
  - Cube 1: [qw₁, qx₁, qy₁, qz₁]
  - Cube 2: [qw₂, qx₂, qy₂, qz₂]
  - Cube 3: [qw₃, qx₃, qy₃, qz₃]
  - 함수: `mdp.cube_orientations_in_world_frame()`

#### 엔드 이펙터 상태 (7차원)
- **EE 위치** (3차원): [x_ee, y_ee, z_ee]
  - 함수: `mdp.ee_frame_pos()`
  - 프레임: `panda_hand` + 10.34cm 오프셋 (그리퍼 중심)

- **EE 방향** (4차원): [qw_ee, qx_ee, qy_ee, qz_ee]
  - 함수: `mdp.ee_frame_quat()`

#### 그리퍼 상태 (2차원)
- **그리퍼 핑거 위치** (2차원): [left_finger, right_finger]
  - 함수: `mdp.gripper_pos()`
  - 범위: [0.0, 0.04] 미터 (완전히 닫힘부터 완전히 열림까지)
  - 조인트 이름: `panda_finger_joint1`, `panda_finger_joint2`

#### 액션 이력 (15차원)
- **이전 액션** (15차원): 마지막으로 취한 액션
  - 함수: `mdp.last_action()`
  - 정책이 시간적 일관성을 유지하는 데 도움

**전체 관찰 차원**: **56** (단일 벡터로 연결됨)

#### 관찰 정규화
```python
enable_corruption: True  # 훈련 중 관찰 노이즈 추가
concatenate_terms: True  # 단일 벡터로 평탄화
normalize_input: True    # 실행 평균/표준편차 정규화
```
- 훈련 중 실행 평균과 표준편차 추적
- 관찰 클리핑: [-100, +100]

---

### 액션 공간 (A)

#### 조인트 위치 제어 (기본)

**액션 차원**: 8차원
- **암 액션** (7차원): 목표 조인트 위치
  - 조인트: `panda_joint1`부터 `panda_joint7`까지
  - 타입: 절대 목표 위치 (기본 오프셋 포함)
  - 스케일: 0.5 (액션이 이 인수로 스케일됨)
  - 함수: `mdp.JointPositionActionCfg`

- **그리퍼 액션** (1차원): 이진 열기/닫기
  - 타입: 이진 명령 {0: 닫기, 1: 열기}
  - 열림 위치: 0.04m (양쪽 핑거)
  - 닫힘 위치: 0.0m (양쪽 핑거)
  - 함수: `mdp.BinaryJointPositionActionCfg`

**액션 처리**:
```python
# 암 액션
target_joint_pos = default_pose + (action[0:7] * 0.5)

# 그리퍼 액션
if action[7] > 0.5:
    gripper_target = 0.04  # 열기
else:
    gripper_target = 0.0   # 닫기
```

**액션 클리핑**: [-100, +100] (스케일링 전)

#### IK 상대 제어 (대안)

**액션 차원**: 7차원
- **자세 변화량** (6차원): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]
  - 타입: 엔드 이펙터 프레임에서의 상대 자세 변화
  - 스케일: 0.5
  - IK 방법: Damped Least Squares (DLS)
  - 제어기: `DifferentialIKControllerCfg`
  - 바디 오프셋: [0.0, 0.0, 0.107] (그리퍼 중심)

- **그리퍼 액션** (1차원): 이진 열기/닫기

---

### 보상 함수 (R)

각 타임스텝의 보상은 **11개 항목의 가중 합**입니다:

#### Phase 1: Cube 1 들어올리기 (베이스 큐브)

**R1: Cube 1에 도달** (가중치: 1.0)
```python
r_reach_1 = 1 - tanh(||p_cube1 - p_ee|| / σ)
```
- σ (std) = 0.1
- cube 1에 접근할 때 부드러운 보상
- 최대값: 큐브에 있을 때 1.0
- 함수: `mdp.object_ee_distance`

**R2: Cube 1 잡기** (가중치: 10.0)
```python
r_grasp_1 = 1.0 if (||p_cube1 - p_ee|| < 0.06) AND (gripper_closed)
           = 0.0 otherwise
```
- 성공적인 잡기에 대한 이진 보상
- 임계값: 6cm 거리 + 그리퍼 닫힘 > 0.005m
- 함수: `mdp.object_grasped_reward`

**R3: Cube 1 들어올리기** (가중치: 15.0)
```python
r_lift_1 = 1.0 if (z_cube1 > z_min + 0.08)
          = 0.0 otherwise
```
- 최소 높이: 테이블 위 8cm
- 베이스 큐브 들어올리기 장려
- 함수: `mdp.object_is_lifted`

#### Phase 2: Cube 2를 Cube 1 위에 쌓기

**R4: Cube 2에 도달** (가중치: 1.0)
```python
r_reach_2 = 1 - tanh(||p_cube2 - p_ee|| / 0.1)
```

**R5: Cube 2 잡기** (가중치: 10.0)
```python
r_grasp_2 = 1.0 if (||p_cube2 - p_ee|| < 0.06) AND (gripper_closed)
```

**R6: Cube 2를 Cube 1 위에 쌓기** (가중치: 30.0)
```python
Δp = p_cube2 - p_cube1
xy_dist = ||(Δp_x, Δp_y)||
z_dist = |Δp_z|

r_stack_2on1 = 1.0 if (xy_dist < 0.05) AND
                      (|z_dist - 0.0468| < 0.005) AND
                      (gripper_open)
              = 0.0 otherwise
```
- XY 정렬: < 5cm
- 높이 정렬: 4.68cm ± 0.5cm (큐브 높이)
- 그리퍼가 열려 있어야 함 (놓음)
- 함수: `mdp.object_stacked_reward`

#### Phase 3: Cube 3을 Cube 2 위에 쌓기

**R7: Cube 3에 도달** (가중치: 1.0)
```python
r_reach_3 = 1 - tanh(||p_cube3 - p_ee|| / 0.1)
```

**R8: Cube 3 잡기** (가중치: 10.0)
```python
r_grasp_3 = 1.0 if (||p_cube3 - p_ee|| < 0.06) AND (gripper_closed)
```

**R9: Cube 3을 Cube 2 위에 쌓기** (가중치: 50.0)
```python
r_stack_3on2 = 1.0 if (||p_cube3 - p_cube2||_xy < 0.05) AND
                      (|z_cube3 - z_cube2 - 0.0468| < 0.005) AND
                      (gripper_open)
```
- 최종 스택을 위한 더 높은 가중치 (50.0)

#### 성공 보너스

**R10: 작업 성공** (가중치: 100.0)
```python
r_success = 1.0 if (cube2_on_cube1) AND (cube3_on_cube2) AND (gripper_open)
           = 0.0 otherwise
```
- 3개 큐브가 모두 제대로 쌓였는지 확인
- 작업 완료에 대한 큰 보너스
- 함수: `mdp.three_cubes_stacked_success`

#### 정규화 페널티

**R11: 액션 변화율 페널티** (가중치: -1e-4, 커리큘럼: -1e-1)
```python
r_action_rate = -||a_t - a_{t-1}||²
```
- 큰 액션 변화에 페널티
- 부드러운 제어 장려
- 커리큘럼 학습을 통해 가중치 증가
- 함수: `mdp.action_rate_l2`

**R12: 조인트 속도 페널티** (가중치: -1e-4, 커리큘럼: -1e-1)
```python
r_joint_vel = -||q̇||²
```
- 높은 조인트 속도에 페널티
- 에너지 효율적인 동작 장려
- 함수: `mdp.joint_vel_l2`

#### 전체 보상
```python
R_total = r_reach_1 + 10*r_grasp_1 + 15*r_lift_1 +
          r_reach_2 + 10*r_grasp_2 + 30*r_stack_2on1 +
          r_reach_3 + 10*r_grasp_3 + 50*r_stack_3on2 +
          100*r_success +
          w_action*r_action_rate + w_vel*r_joint_vel
```
- 보상 범위: 타임스텝당 약 [-1, +227]
- 커리큘럼: 훈련 중 정규화 가중치 증가

---

### 전이 역학 (T)

#### 물리 시뮬레이션
- **시뮬레이터**: NVIDIA PhysX 5 (GPU 가속)
- **적분**: Semi-implicit Euler
- **타임스텝**: Δt = 0.01s (100 Hz 시뮬레이션)
- **서브스텝**: 1 (서브스텝 없음)

#### 로봇 역학
- **로봇 모델**: Franka Panda (7자유도 암 + 2자유도 그리퍼)
- **URDF**: `isaaclab_assets.robots.franka.FRANKA_PANDA_CFG`
- **액추에이터 모델**: 암묵적 PD 제어기
  - 위치 게인 (Kp): [조인트별]
  - 댐핑 (Kd): [조인트별]
  - 최대 토크: [조인트별]
- **조인트 한계**: PhysX articulation을 통해 적용
- **자기 충돌**: 활성화됨

#### 물체 역학
- **큐브**: 3개의 강체
  - 크기: 4.05cm × 4.05cm × 4.68cm (표준 블록)
  - 질량: ~50g (강체 속성에서 추정)
  - 마찰: 기본 PhysX 재질
  - 솔버 반복: 16 (위치), 1 (속도)
  - 최대 침투 제거 속도: 5.0 m/s

#### 접촉 역학
- **솔버**: TGS (Temporal Gauss-Seidel)
- **마찰 모델**: 피라미드 근사
- **바운스 임계값**: 0.01 m/s
- **마찰 상관 거리**: 0.00625m
- **GPU 쌍 용량**: 32K (3개 큐브를 위해 증가)

---

### 종료 조건

다음 조건 중 하나가 충족되면 에피소드가 종료됩니다:

**T1: 시간 초과** (time_out = True)
```python
if t ≥ T_max:
    done = True
```
- 에피소드 길이: 15초
- 최대 스텝: 750 (50Hz 제어 빈도에서)
- 함수: `mdp.time_out`

**T2: 큐브 떨어짐** (time_out = False)
```python
if z_cube_i < -0.05:  # i ∈ {1, 2, 3}
    done = True
```
- 큐브가 테이블 아래로 떨어지면 종료
- 임계값: -5cm (테이블이 z=0에 위치)
- 함수: `mdp.root_height_below_minimum`

**T3: 작업 성공** (time_out = False)
```python
if (cube2_on_cube1) AND (cube3_on_cube2):
    done = True
```
- 성공 시 조기 종료
- 정책이 빠르게 완료 가능
- 함수: `mdp.cubes_stacked`

---

## 2. 제어 루프 및 타이밍

### 계층적 제어 아키텍처

```
┌─────────────────────────────────────────────┐
│  PPO 정책 네트워크 (PyTorch)                │ ← 50 Hz
│  입력: 관찰 (56차원)                         │
│  출력: 액션 (8차원)                          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  액션 프로세서                               │ ← 50 Hz
│  - 액션 스케일링                             │
│  - 기본 오프셋 추가                          │
│  - 한계로 클리핑                             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  PD 제어기 (PhysX에서 암묵적)                │ ← 100 Hz
│  τ = Kp(q_target - q) - Kd(q̇)              │  (시뮬레이션됨)
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  PhysX 물리 엔진                             │ ← 100 Hz
│  - 순방향 역학                               │
│  - 접촉 해결                                 │
│  - 적분                                      │
└─────────────────────────────────────────────┘
```

### 타이밍 분석

#### 시뮬레이션 빈도
- **물리 타임스텝**: dt = 0.01s = **100 Hz**
- **PhysX 업데이트 속도**: 100 Hz
- **순방향 역학**: 매 타임스텝
- **접촉 솔버**: 매 타임스텝

#### 제어 빈도
- **정책 결정 속도**: **50 Hz**
- **데시메이션**: 2 (정책이 2 시뮬레이션 스텝마다 작동)
- **액션 유지**: 0.02s (2 물리 스텝)

#### 렌더링 빈도
- **렌더 간격**: 2 시뮬레이션 스텝 = **50 Hz**
- **시각화 FPS**: ~50 FPS (headless가 아닐 때)

### 에피소드 스텝당 제어 흐름

```python
for step in range(max_episode_steps):  # 최대 750 스텝
    # 2 시뮬레이션 스텝마다 (50 Hz 제어)
    if step % decimation == 0:
        # 1. 관찰 얻기 (이전 상태에서)
        obs = get_observation()  # 56차원

        # 2. 정책 순전파
        with torch.no_grad():
            action, value = policy(obs)  # action: 8차원

        # 3. 액션 처리
        joint_targets = default_pose + action[:7] * 0.5
        gripper_target = 0.04 if action[7] > 0.5 else 0.0

    # 매 시뮬레이션 스텝 (100 Hz 물리)
    # 4. 로봇에 조인트 목표 적용
    robot.set_joint_position_targets(joint_targets)
    robot.set_gripper_targets(gripper_target)

    # 5. PhysX가 한 스텝 시뮬레이션
    #    - τ = Kp(q_target - q) - Kd(q̇) 계산
    #    - 순방향 역학: q̈ = M⁻¹(τ + τ_ext)
    #    - 접촉 해결
    #    - 적분: q, q̇ → q', q̇'
    sim.step(dt=0.01)

    # 6. 다음 스텝을 위한 관찰 업데이트
    update_sensors()  # Frame transformer, 물체 상태

    # 2 시뮬레이션 스텝마다 (50 Hz)
    if step % decimation == 0:
        # 7. 보상 계산
        reward = compute_reward()

        # 8. 종료 확인
        done = check_termination()

        # 9. 버퍼에 경험 저장
        buffer.add(obs, action, reward, done, value)
```

### 토크 전달

**질문**: 토크가 로봇에 어떻게 전달되나요?

**답변**: IsaacLab은 PhysX에서 **암묵적 PD 제어**를 사용합니다:

1. **정책 출력**: 목표 조인트 위치 (토크가 아님)
2. **PhysX PD 제어기**: 내부적으로 토크 계산
   ```
   τ_actuated = Kp * (q_target - q_current) - Kd * q̇_current
   ```
3. **토크 한계**: 로봇 설정의 `max_effort`를 통해 적용
4. **적용된 토크**: τ_total = τ_actuated + τ_gravity + τ_contact
5. **순방향 역학**: PhysX가 가속도 계산
6. **적분**: 위치와 속도 업데이트

**핵심 포인트**: 정책은 조인트 토크를 **직접 출력하지 않습니다**. 목표 위치를 출력하고, PhysX가 저수준 토크 제어를 처리합니다.

---

## 3. PPO 알고리즘 설정

### 신경망 아키텍처

#### Actor-Critic 네트워크
```
입력 (56차원 관찰)
    ↓
Dense(256) + ELU
    ↓
Dense(128) + ELU
    ↓
Dense(64) + ELU
    ↓         ↓
Actor       Critic
(8차원)     (1차원 value)
```

**아키텍처 세부사항**:
- **타입**: 공유 트렁크, 분리된 헤드
- **활성화**: ELU (Exponential Linear Unit)
- **레이어**: [256, 128, 64]
- **분리**: False (특징 공유)
- **Actor 출력**: 8차원 평균 (μ)
- **Critic 출력**: 1차원 가치 추정 (V)
- **정책 분포**: 대각 가우시안
  - 고정된 σ (학습되지 않음)
  - σ는 `const_initializer`를 통해 초기화

### PPO 하이퍼파라미터

#### 학습 파라미터
```yaml
learning_rate: 1e-4          # Adam 학습률
lr_schedule: adaptive        # KL 발산에 따라 감소
kl_threshold: 0.01           # 적응적 LR을 위한 목표 KL
gamma: 0.99                  # 할인 인자
tau: 0.95                    # GAE 람다
```

#### 경험 수집
```yaml
num_actors: 4096             # 병렬 환경
horizon_length: 64           # 롤아웃당 스텝
total_steps_per_iter: 262144 # 4096 envs × 64 steps
```

#### 최적화
```yaml
mini_epochs: 8               # PPO 업데이트 에폭
minibatch_size: 32768        # 업데이트당 배치 크기
num_minibatches: 8           # 262144 / 32768
```

#### 손실 가중치
```yaml
critic_coef: 4               # 가치 손실 가중치
entropy_coef: 0.001          # 엔트로피 보너스 가중치
bounds_loss_coef: 0.0001     # 액션 범위 페널티
```

#### 클리핑 및 정규화
```yaml
e_clip: 0.2                  # PPO 클립 범위 [0.8, 1.2]
clip_value: True             # 가치 손실 클리핑
grad_norm: 1.0               # 그래디언트 노름 클리핑
truncate_grads: True         # 그래디언트 클리핑 활성화
```

#### 훈련 스케줄
```yaml
max_epochs: 100000           # 전체 훈련 에폭
save_frequency: 100          # 100 에폭마다 저장
save_best_after: 200         # 200 이후 최고 추적 시작
print_stats: True            # 통계 로깅
```

### PPO 손실 함수

#### 정책 손실 (클리핑된 목적함수)
```python
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε = 0.2
L_policy = -min(ratio * A, clipped_ratio * A)
```

#### 가치 손실 (클리핑됨)
```python
V_clipped = V_old + clip(V_new - V_old, -ε, +ε)
L_value = max((V_new - V_target)², (V_clipped - V_target)²)
```

#### 엔트로피 보너스
```python
L_entropy = -H(π(·|s))  # 음의 엔트로피
```

#### 전체 손실
```python
L_total = L_policy + 4*L_value + 0.001*L_entropy + 0.0001*L_bounds
```

### 어드밴티지 추정 (GAE)

```python
δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
```
- λ (tau) = 0.95
- γ (gamma) = 0.99

### 정규화

#### 입력 정규화
```python
obs_normalized = (obs - running_mean) / (running_std + ε)
```
- 매 스텝 실행 통계 업데이트
- 초기 std = 1.0
- ε = 1e-8

#### 가치 정규화
```python
value_normalized = (value - running_mean_value) / (running_std_value + ε)
```

#### 어드밴티지 정규화
```python
A_normalized = (A - mean(A)) / (std(A) + ε)
```

#### 보상 스케일링
```python
reward_scaled = reward * 0.01  # reward_shaper.scale_value
```

---

## 4. 병렬화 및 성능

### GPU 병렬화

#### 환경 병렬화
- **환경 수**: 4096 (기본)
- **병렬화**: GPU 텐서 연산
- **메모리**: 단일 GPU 메모리에 모든 환경
- **관찰 텐서**: GPU에 [4096, 56]
- **액션 텐서**: GPU에 [4096, 8]

#### 물리 병렬화
- **PhysX GPU**: 4096개 씬이 병렬로 시뮬레이션됨
- **광역 단계**: GPU 가속 AABB 트리
- **접촉 솔버**: GPU TGS 솔버
- **강체 파이프라인**: 완전한 GPU 기반

#### 정책 병렬화
- **배치 추론**: 4096개 관찰 → 단일 순전파
- **디바이스**: cuda:0
- **혼합 정밀도**: False (FP32 사용)
- **추론 시간**: 4096개 환경에 대해 ~0.1ms

### 훈련 루프

```python
for epoch in range(max_epochs):  # 100000 에폭
    # 1. 경험 수집 (4096 envs × 64 steps)
    for step in range(horizon_length):
        obs = env.get_observations()  # [4096, 56]
        actions, values = policy(obs)  # [4096, 8], [4096, 1]
        obs_next, rewards, dones = env.step(actions)
        buffer.add(obs, actions, rewards, dones, values)

    # 2. 어드밴티지 계산
    advantages, returns = compute_gae(buffer, gamma=0.99, tau=0.95)

    # 3. PPO 업데이트 (8 미니 에폭)
    for mini_epoch in range(8):
        for minibatch in get_minibatches(buffer, batch_size=32768):
            # 순전파
            actions_new, values_new = policy(minibatch.obs)

            # 손실 계산
            loss_policy = ppo_policy_loss(actions_new, minibatch.advantages)
            loss_value = value_loss(values_new, minibatch.returns)
            loss_entropy = entropy_loss(actions_new)

            loss_total = loss_policy + 4*loss_value + 0.001*loss_entropy

            # 역전파
            optimizer.zero_grad()
            loss_total.backward()
            clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

    # 4. 로깅
    if epoch % 100 == 0:
        save_checkpoint(policy, epoch)
```

### 성능 메트릭

#### 훈련 속도
- **FPS (스텝/초)**: ~85,000 - 90,000
- **에폭당 스텝**: 262,144 (4096 × 64)
- **에폭당 시간**: ~3초
- **전체 훈련 시간**: ~8-10시간 (100k 에폭)

#### 메모리 사용량
- **GPU 메모리**: ~12-16 GB (4096개 환경)
- **시스템 RAM**: ~32 GB
- **PhysX GPU 메모리**: ~4 GB (집합 쌍, 접촉)

#### 병목 현상
- **물리 시뮬레이션**: 계산 시간의 60%
- **정책 추론**: 계산 시간의 10%
- **보상 계산**: 계산 시간의 15%
- **환경 리셋**: 계산 시간의 5%
- **오버헤드**: 10% (데이터 전송, 로깅)

---

## 5. 커리큘럼 학습

### 보상 가중치 커리큘럼

두 보상 항목의 가중치가 점진적으로 증가합니다:

#### 액션 변화율 페널티
- **초기 가중치**: -1e-4
- **최종 가중치**: -1e-1
- **스케줄**: 10,000 스텝에 걸쳐 선형 증가
- **함수**: `mdp.modify_reward_weight`

#### 조인트 속도 페널티
- **초기 가중치**: -1e-4
- **최종 가중치**: -1e-1
- **스케줄**: 10,000 스텝에 걸쳐 선형 증가

**근거**:
- 초기 훈련: 탐험 허용, 부드러움 무시
- 후기 훈련: 부드럽고 에너지 효율적인 동작 강제

### 향후 가능한 커리큘럼 (미구현)

- **큐브 간격**: 넓은 간격으로 시작, 점진적으로 감소
- **에피소드 길이**: 짧게 시작 (5초), 점진적으로 15초까지 증가
- **작업 복잡도**: 단계별 훈련 (들기 → 2-큐브 → 3-큐브)
- **무작위화**: 시간에 따라 관찰의 노이즈 증가

---

## 6. 관찰 및 보상 디버깅

### 관찰 확인

```python
# 환경 0의 관찰 출력
obs = env.get_observations()
print("조인트 위치:", obs[0, :7])
print("Cube 1 위치:", obs[0, 11:14])
print("EE 위치:", obs[0, 32:35])
```

### 보상 시각화

```python
# 개별 보상 항목 접근
rewards_dict = env.reward_manager.compute()
print("Cube 1 도달:", rewards_dict["reaching_cube_1"][0])
print("Cube 1 잡기:", rewards_dict["grasping_cube_1"][0])
print("작업 성공:", rewards_dict["task_success"][0])
```

### 종료 확인

```python
dones = env.termination_manager.compute()
print("시간 초과:", dones["time_out"][0])
print("큐브 떨어짐:", dones["cube_1_dropping"][0])
print("성공:", dones["success"][0])
```

---

## 7. 다른 환경과의 주요 차이점

### vs. Lift 환경
| 측면 | Lift | Stack-RL |
|--------|------|----------|
| 작업 | 1개 큐브 들기 | 3개 큐브 순차 쌓기 |
| 에피소드 길이 | 5초 (250 스텝) | 15초 (750 스텝) |
| Horizon Length | 24 | 64 |
| Max Epochs | 5000 | 100000 |
| 관찰 차원 | ~45 | 56 |
| 보상 항목 | 5 | 11 |
| 성공 기준 | 큐브 들어올림 | 3개 큐브 쌓임 |

### vs. 원본 Stack (IL)
| 측면 | Stack (IL) | Stack-RL |
|--------|------------|----------|
| 학습 방법 | Imitation Learning | Reinforcement Learning |
| 보상 | 없음 (IL 손실 사용) | 밀집 보상 설계 |
| 전문가 데이터 | 필요 (CuRobo) | 불필요 |
| 훈련 | 지도 학습 | 시행착오 |
| 에피소드 길이 | 30초 | 15초 (더 타이트) |

---

## 8. 관찰 문제 해결

### 정책이 학습되지 않나요?

1. **관찰 범위 확인**:
   ```python
   print(obs.min(), obs.mean(), obs.max())
   ```
   - 정규화되어야 함 (평균 ≈ 0, std ≈ 1)

2. **보상 크기 확인**:
   ```python
   print(env.reward_manager.compute())
   ```
   - 도달 보상은 초기에 ~0.5-1.0이어야 함
   - 성공 보상은 훈련 후기까지 0.0일 것

3. **액션 분포 확인**:
   ```python
   print(actions.min(), actions.mean(), actions.max())
   ```
   - 초기에는 탐험해야 함 (높은 분산)
   - 나중에 수렴해야 함 (낮은 분산)

### 물리 불안정?

1. **접촉력 확인**:
   - 큐브가 날아가면 → 솔버 반복 증가
   - 침투가 발생하면 → 침투 제거 속도 감소

2. **조인트 한계 확인**:
   - 로봇이 난동을 부리면 → 조인트 한계가 적용되는지 확인

3. **타임스텝 확인**:
   - 시뮬레이션이 폭발하면 → dt 감소 (예: 0.005s)

---

## 요약 표

| 구성요소 | 값 | 비고 |
|-----------|-------|-------|
| **관찰 차원** | 56 | 연결된 벡터 |
| **액션 차원** | 8 (조인트) / 7 (IK) | 조인트 목표 + 그리퍼 |
| **보상 항목** | 11 | 9 작업 + 2 정규화 |
| **에피소드 길이** | 15초 (750 스텝) | 50Hz 제어에서 |
| **시뮬레이션 빈도** | 100 Hz | PhysX 타임스텝 |
| **제어 빈도** | 50 Hz | 정책 결정 속도 |
| **데시메이션** | 2 | 정책이 2 시뮬 스텝마다 작동 |
| **환경 수** | 4096 | 병렬 GPU 환경 |
| **Horizon Length** | 64 | PPO 롤아웃 스텝 |
| **Minibatch Size** | 32768 | PPO 업데이트 배치 크기 |
| **학습률** | 1e-4 | KL 기반 적응적 |
| **Max Epochs** | 100000 | 복잡한 작업을 위해 확장 |
| **훈련 시간** | ~8-10시간 | RTX 3090/4090에서 |

---

## 📅 20251210 1601 업데이트

### 주요 변경사항

#### 1. **Observation 차원 변경**: 56-dim → **65-dim** (+9-dim)
```python
# ✅ 신규 추가
cube_dimensions = ObsTerm(func=mdp.cube_dimensions_in_world_frame)  # 9-dim
# → [cube_1: x,y,z, cube_2: x,y,z, cube_3: x,y,z]
```

**이유**: Policy가 큐브 크기를 알아야 적절한 높이로 들어올려 쌓을 수 있음

#### 2. **Reward 구조 완전 재설계**
- ❌ **삭제**: cube_1 들기 (reaching, grasping, lifting) - 베이스는 테이블 고정!
- ✅ **추가**: Hybrid rewards (Dense + Sparse one-time)
- ✅ **추가**: Masked rewards (cube_2 쌓인 후 cube_3 보상 활성화)
- ✅ **추가**: cube_1 제약 조건 (테이블 위 유지)
- ✅ **추가**: 조건부 안정성 (그리퍼 열렸을 때만)

**새 최대 보상**: 227 → **229**

자세한 내용은 `WORKFLOW_KR.md` 의 "20251210 1601 - Reward 구조 완전 재설계" 섹션 참조

---

**문서 상태**: ✅ 완전한 기술 사양
**마지막 업데이트**: 2025-12-10 (20251210 1601 Reward 재설계)
**관련 문서**: 프로젝트 맥락 및 개발 이력은 `WORKFLOW_KR.md` 참조

---

**PPO 정보 문서 끝**
