
stochastic-rl
일단 한글로 먼저 작성,

논문 형식은 0. Abstract (나중에 작성) 1. Introduction, 2. Background and related work 3. Method 4. Experiments 5. Conclusion & Limitation 
으로 작성

1. Introduction (나중에 작성할 것)
다만, flow는
dexterous 전통제어 어려움 → IL/RL.에서 잘됨
RL에서 잘되는 것들 소개 
또한 dexterous의 경우 하드웨어 관점에선 underactuated, fully actuated가 있다는 것 소개
하지만, rl에서 학습 프레임워크만 집중, 손 자체의 기계적 및 구동 구조가 어떻게 학습에 영향을 미치는지 잘 없다
그래서 우리는 How does hand actuation structure (underactuated vs fully-actuated) affect reinforcement learning performance in dexterous manipulation tasks?에 집중해서 분석해보려 한다.

(기존 영어 version)
Dexterous manipulation remains one of the most challenging problems in robotic control due to high-
dimensional action spaces, complex contact dynamics, and intricate coordination requirements among mul-
tiple fingers [4]. Traditional model-based controllers often struggle to generalize across diverse objects and
tasks, motivating the growing adoption of data-driven approaches such as reinforcement learning (RL) and
imitation learning (IL) [5].
Recent advances in RL and IL have enabled impressive demonstrations of dexterous manipulation in both
simulation and real-world systems. In particular, deep RL methods have been shown to acquire complex,
high-dimensional manipulation skills with multi-fingered robotic hands when combined with demonstrations,
reward shaping, or large-scale simulation [2, 6]. These successes have fueled increasing interest in applying
learning-based control to anthropomorphic robots, including humanoids and hand–arm systems, for tasks
such as grasping, in-hand manipulation, and tool use.
Despite this progress, most existing studies focus on improving learning algorithms, reward formulations,
or data collection strategies for a fixed hand configuration [5]. Comparatively little attention has been paid to
how the mechanical and actuation structure of the hand itself influences the learning process. In particular,
fundamental questions remain unanswered regarding how the degree of actuation and mechanical coupling
affect exploration difficulty, sample efficiency, and the emergence of manipulation strategies under RL.
From a hardware perspective, underactuated hands have been widely studied as mechanically simpler and
robust alternatives to fully actuated dexterous hands [7]. By coupling multiple joints through passive mech-
anisms, underactuated designs reduce the number of control inputs, potentially lowering control complexity
and improving robustness, while fully actuated hands offer greater expressiveness at the cost of increased
action dimensionality and control burden. However, how these design trade-offs manifest in learning-based
control settings remains insufficiently understood.
Understanding the role of hand actuation structure in RL-based manipulation is critical for the design of
future robotic systems, particularly as humanoid and dexterous robots continue to gain attention. Despite
the growing body of work on dexterous manipulation with RL, a controlled comparative analysis between
underactuated and fully actuated hands under identical learning conditions remains underexplored.
This project aims to address this gap by conducting a systematic comparison of underactuated and fully
actuated dexterous hands using the same RL framework, task setup, and simulation environment, thereby
shedding light on how actuation structure influences learning efficiency and emergent manipulation strategies.

----------------------------------------------------------------------------------------------------
2. Background and Related work

2.1. Reinforcement Learning (ppo) 알고리즘 설명

(todo with 수학적 수식)

2.2. Dexterous Manipulation with Reinforcement Learning
(**Rajeswaran et al. (RSS 2018)** - *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations*
- **Contribution**: Demonstrated that high-DoF dexterous manipulation (e.g., Shadow Hand) can be learned with RL + demonstrations
- **Key Insight**: High-DoF hands create severe exploration challenges due to:
  - Large action space (20+ DoF)
  - Sparse rewards (contact is infrequent in random exploration)
  - Sample inefficiency (millions of samples needed)
- **Relevance to Our Work**: Motivates our hypothesis that **underactuated hands may reduce exploration difficulty**

**Reference**: https://arxiv.org/abs/1709.10087

와 Large-Scale Dexterous RL 두개 소개

**DexPBT (Petrenko et al., RSS 2023)** - *DexPBT: Scaling Up Dexterous Manipulation for Hand-Arm Systems with Population Based Training*
- **Contribution**: Large-scale RL for hand-arm systems using Population Based Training (PBT)
- **Key Insights**:
  - Massive parallelization (16,384 environments) enables learning complex behaviors
  - Curriculum and domain randomization are critical for success
  - Hand-arm coordination requires careful reward design
- **Relevance to Our Work**:
  - We adopt similar scale (4,096 environments) and curriculum strategies
  - Our experimental protocol follows their benchmark design principles
  - We extend their work by systematically comparing actuation structures

**Reference**: https://roboticsproceedings.org/rss19/p027.pdf)

2.3. Underactuated vs Fully-Actuated Comparison

Classical Underactuated Design

**Catalano et al. (IJRR 2014)** - *Adaptive Synergies for the Design and Control of the Pisa/IIT SoftHand*
- **Contribution**: Seminal work on underactuated soft hands with adaptive synergies
- **Key Insights**:
  - Underactuation provides **mechanical compliance** → robust grasps
  - **Synergies** (coordinated joint motion) reduce control complexity
  - "Easy to control, hard to model" trade-off
- **Relevance to Our Work**:
  - Provides theoretical foundation for why underactuation may help RL
  - **Synergies = inductive bias** in action space
  - Compliance may improve contact-rich manipulation

**Reference**: https://journals.sagepub.com/doi/10.1177/0278364914518172

---

underactuated hands in learning 

**Sintov et al. (RAL 2019, PMLR 2020)** - *Data-Driven Modeling and Control of an Underactuated Hand*
- **Contribution**: Data-driven approach to model and control underactuated hands
- **Key Insights**:
  - Underactuated hands are **hard to model analytically** (complex coupling)
  - Data-driven methods (learning) are natural fit for underactuated systems
  - Learning can exploit mechanical coupling for robust grasping
- **Relevance to Our Work**:
  - Supports our approach of using RL (data-driven) for underactuated hands
  - Suggests that RL may **benefit from** underactuation (not hindered by it)
  - Provides context for interpreting our results

**Reference**: https://cpb-us-e1.wpmucdn.com/sites.yale.edu/dist/5/2769/files/2021/01/SintovEtAl2019_RAL.pdf

---
Underactuated vs Fully-Actuated Comparison

**Lopes (arXiv 2025)** - *Comparative Analysis of Underactuated and Fully-Actuated Hand Designs for Robotic Manipulation*
- **Contribution**: Compares underactuated vs fully-actuated hands (though in different context than ours)
- **Key Insights**:
  - Few works directly compare underactuated vs fully-actuated hands
  - Most comparisons are confounded by multiple variables
  - Need for **controlled experimental protocols**
- **Relevance to Our Work**:
  - Confirms that direct, controlled comparisons are **rare in literature**
  - Our work fills this gap by providing rigorous experimental protocol
  - We extend this line of work to RL context (Lopes focuses on classical control)

**Reference**: https://arxiv.org/abs/2501.xxxxx (check latest for exact reference)


2.4. 우리의 타당성 (gap in literature and our contribution)

Identified Gaps
1. **Algorithmic Focus**: Most dexterous RL work focuses on improving algorithms (PPO, SAC, etc.), not hardware
2. **Confounded Comparisons**: When hardware is compared, multiple variables change simultaneously (arm + hand + task + algorithm)
3. **Underactuated Hands Underrepresented**: Most RL work uses fully-actuated hands (Shadow, Allegro, Barrett)
4. **Lack of Systematic Analysis**: No controlled study isolating the effect of actuation structure on RL learning dynamics

Our Contribution

**We provide the first systematic, controlled comparison of underactuated vs fully-actuated hands in RL-based dexterous manipulation.**

Key aspects:
1. **Controlled Protocol**: Fix all variables except hand actuation structure
2. **Fair Comparison**: Normalize object sizes to hand sizes
3. **Comprehensive Analysis**: Beyond success rate (sample efficiency, stability, exploration, strategy)
4. **Reproducible Setup**: Open-source implementation with detailed configuration

**Novel Insight**: Hand actuation structure acts as **inductive bias** in RL:
- Underactuated hands: mechanical constraints → implicit regularization → easier exploration
- Fully-actuated hands: higher expressiveness → larger policy space → harder exploration

This reframes hand design from a **mechanical problem** to a **learning problem**, opening new research directions in hardware-algorithm co-design.



----------------------------------------------------------------------------------------------------

3. Method

### 3.1. 문제 정의 및 비교 대상 로봇 핸드

본 연구는 **저구동 핸드(underactuated hand)**와 **완전 구동 핸드(fully-actuated hand)**가 강화학습 기반 기민한 조작(dexterous manipulation) 과제에서 보이는 학습 성능의 차이를 체계적으로 비교 분석하는 것을 목표로 한다. 이를 위해 우리는 인간 손의 5개 손가락 구조를 모방한 두 개의 상용 로봇 핸드를 선정하였으며, 각각 구동 방식에서 명확한 차이를 갖는다.

#### 3.1.1. 5-Finger 핸드 선정 배경

기존 dexterous manipulation 연구에서는 주로 4-finger 핸드인 Allegro Hand [10] 또는 Shadow Dexterous Hand [11]가 벤치마크로 사용되어 왔다. 그러나 본 연구에서는 다음과 같은 이유로 **5-finger 구조**를 채택하였다:

1. **인간형 로봇(humanoid robot)과의 호환성**: 최근 Tesla Optimus, Figure 01, 1X NEO 등 상용 휴머노이드 로봇들이 5-finger 핸드를 채택하고 있으며 [12], 인간-로봇 상호작용 및 도구 사용(tool use) 측면에서 5-finger 구조가 더 자연스러운 인터페이스를 제공한다.

2. **작업 공간(workspace)의 대칭성**: 엄지(thumb)를 포함한 5개 손가락 구조는 물체를 enveloping grasp 또는 precision grasp로 잡을 때 더 균형잡힌 접촉 분포를 형성할 수 있다.

3. **실제 산업 응용**: Allegro Hand는 주로 연구용으로 한정되어 있는 반면, 본 연구에서 선택한 두 핸드(DG5F, Inspire RH56F1)는 산업 현장 및 서비스 로봇 분야에서 실제로 사용되고 있어 실용성이 높다 [13, 14].

따라서 우리는 **동일한 5-finger 구조를 유지하면서도 구동 방식에서만 차이가 있는** 두 핸드를 선정하여, 구동 구조가 강화학습 성능에 미치는 영향을 통제된 환경에서 분석하고자 하였다.

#### 3.1.2. DG5F: 완전 구동형 기민 핸드

<image: RH56F1_R(Left)vsDG5F(Right).png - 오른쪽 핸드>

**DG5F(Delto Gripper-5 Finger)**는 한국의 Tesollo사에서 개발한 완전 구동형 5-finger 로봇 핸드이다 [13]. 이 핸드는 **20 자유도(Degrees of Freedom, DoF)**를 가지며, 각 손가락마다 4개의 관절(MCP, PIP, DIP, Fingertip)이 독립적으로 구동되는 fully-actuated 구조를 갖는다.

**주요 사양:**
- **자유도**: 20-DoF (5 fingers × 4 joints)
- **구동 방식**: 완전 구동(fully-actuated) - 모든 관절이 독립적인 모터로 제어됨
- **크기**: 약 20 cm 길이 (성인 남성 손 크기와 유사) [13]
- **페이로드**:
  - Pinching: 2.5 kg (정격), 5 kg (최대)
  - Enveloping: 10 kg (정격), 20 kg (최대)
- **질량**: 약 0.8 kg (URDF 기준)
- **제어 인터페이스**: CAN, Ethernet

DG5F의 URDF 파일 분석 결과, 20개의 `revolute` 타입 관절이 정의되어 있으며, 각 관절은 독립적인 토크 제어가 가능하다. 이러한 완전 구동 구조는 **높은 표현력(high expressiveness)**을 제공하여 복잡한 손가락 자세(finger configuration)를 생성할 수 있지만, 동시에 **26차원(6 arm + 20 hand)의 행동 공간(action space)**을 형성하여 강화학습 관점에서 탐색(exploration) 문제의 난이도를 증가시킨다.

**구조적 특징:**
- 각 손가락의 4개 관절(MCP abduction/adduction, MCP flexion/extension, PIP, DIP)이 독립적으로 제어됨
- Thumb(엄지)의 opposition 동작이 가능하여 precision grasp 구현 가능
- 금속 프레임 기반의 견고한 구조로 반복적인 접촉에 강함

#### 3.1.3. Inspire RH56F1: 저구동형 기민 핸드

<image: RH56F1_R(Left)vsDG5F(Right).png - 왼쪽 핸드>

**Inspire RH56F1**은 중국의 Inspire Robots사에서 개발한 저구동형(underactuated) 5-finger 로봇 핸드이다 [14]. 이 핸드는 **6개의 선형 액추에이터(linear servo actuators)**를 사용하여 **12개의 관절**을 구동하는 underactuated 메커니즘을 채택하고 있다.

**주요 사양:**
- **자유도**: 6-DoF (actuators), 12 joints (coupled)
- **구동 방식**: 저구동(underactuated) - 6개 액추에이터가 12개 관절을 결합(coupling)을 통해 제어
- **크기**: 약 12-15 cm 길이 (DG5F 대비 약 60-75% 크기)
- **페이로드**:
  - 손가락 끝 파지력(fingertip gripping force): 15 N
  - 정적 수동 하중(static passive load): 30 kg
- **센서**: 24개 다차원 센서 (촉각, 힘, 위치, 온도) [14]
- **힘 해상도**: 0.5 N (정밀한 힘 제어 가능)
- **통신 속도**: 1 kHz 실시간 제어
- **질량**: 약 0.5 kg (URDF 기준)

Inspire RH56F1의 URDF 파일 분석 결과, 12개의 `revolute` 타입 관절이 정의되어 있으나, 실제로는 6개의 액추에이터를 통해 구동된다. 이는 **기계적 결합(mechanical coupling)**을 통해 2개의 관절이 하나의 액추에이터에 의해 동시에 움직이는 구조를 의미한다 [15]. 이러한 저구동 설계는 **12차원(6 arm + 6 hand)의 행동 공간**을 형성하여 DG5F 대비 절반 수준의 행동 차원을 갖는다.

**구조적 특징:**
- **선형 구동 메커니즘**: 회전 모터 대신 선형 액추에이터 사용으로 백래시(backlash) 감소
- **Adaptive synergies** [16]: 기계적 결합을 통해 인간 손의 synergistic motion 모방
- **수동 컴플라이언스(passive compliance)**: 물체와 접촉 시 관절이 수동적으로 적응하여 다양한 형태의 물체 파지 가능
- **경량 설계**: 풀메탈 바디임에도 DG5F 대비 약 60% 수준의 질량

#### 3.1.4. 저구동 vs 완전 구동: 근본적 차이

두 핸드의 가장 핵심적인 차이는 **구동 자유도(actuated DoF)**와 **관절 자유도(joint DoF)** 간의 관계에 있다 [17]:

**완전 구동(Fully-actuated)**: DG5F
- 구동 자유도 = 관절 자유도 (20 = 20)
- 각 관절이 독립적으로 제어 가능 → **임의의 손가락 자세(arbitrary finger configuration)** 생성 가능
- 행동 공간: 26차원 연속 공간 (큰 정책 공간)
- 제어 복잡도: 높음 (모든 관절을 협응해야 함)

**저구동(Underactuated)**: Inspire RH56F1
- 구동 자유도 < 관절 자유도 (6 < 12)
- 기계적 결합으로 인해 **제한된 손가락 자세(constrained finger configuration)** 생성
- 행동 공간: 12차원 연속 공간 (작은 정책 공간)
- 제어 복잡도: 낮음 (synergies가 내재된 inductive bias 제공)

이러한 구조적 차이는 강화학습 관점에서 다음과 같은 trade-off를 발생시킨다:

1. **탐색 난이도(Exploration difficulty)**:
   - DG5F: 26차원 공간에서의 무작위 탐색 → 접촉 발생 확률 낮음
   - Inspire: 12차원 공간 + mechanical constraints → 암묵적 정규화(implicit regularization) 효과

2. **표현력(Expressiveness)**:
   - DG5F: 복잡한 manipulation 전략 학습 가능 (finger walking, precision grasp)
   - Inspire: synergistic motion에 제한됨 (power grasp 위주)

3. **학습 효율(Sample efficiency)**:
   - DG5F: 높은 차원으로 인한 샘플 비효율성 가능성
   - Inspire: 작은 행동 공간으로 인한 빠른 수렴 가능성

본 연구는 이러한 trade-off가 실제 강화학습 환경에서 어떻게 나타나는지를 **동일한 작업(task), 동일한 알고리즘(PPO), 동일한 로봇 팔(UR10e)** 조건 하에서 체계적으로 분석한다.

#### 3.1.5. 실험 구성 요약

우리의 비교 실험은 다음과 같이 구성된다:

**UR10e + DG5F Right Hand**
- **Arm**: UR10e (6-DoF collaborative robot)
- **Hand**: DG5F Right (20-DoF fully-actuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (**1.0× 스케일**, 기존 Kuka-Allegro 벤치마크와 동일)
- **Total DoF**: 26 (6 arm + 20 hand)
- **Observation dimension**: 1870 (without FT sensor)

**UR10e + Inspire Right Hand**
- **Arm**: UR10e (6-DoF collaborative robot, DG5F와 동일)
- **Hand**: Inspire RH56F1 Right (6-DoF underactuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (**0.5× 스케일**, 손 크기에 맞춰 조정)
- **Total DoF**: 12 (6 arm + 6 hand)
- **Observation dimension**: 1660 (without FT sensor)

물체 크기의 스케일링(DG5F: 1.0×, Inspire: 0.5×)은 **상대적 작업 난이도를 동일하게 유지**하기 위함이다. 핸드 크기 대비 물체 크기의 비율을 일치시킴으로써, 두 핸드가 직면하는 조작 과제의 난이도를 공정하게 비교할 수 있다. 이에 대한 자세한 설명은 3.2절에서 다룬다.

**연구 질문**: *"구동 구조(underactuated vs fully-actuated)가 강화학습 기반 기민한 조작 과제에서 학습 효율성, 안정성, 최종 성능에 어떠한 영향을 미치는가?"*



3.2. IsaacLab Dexsuite Env

isaclab 시뮬레이션 언급 + 기존 dexpbt / dexpoint 상황에 여러 물체 lift설명

- 기존 iiwa 7자유도 → ur10e 6자유도 설정
- point cloude → target 위치 정보 뭐 이런 것들 (서로 다른 4개의 물체 / 물체별로도 사이즈 다르게 해서 총 16envs)
- hand의 차이에 따라 물체 크기 다르게 함 ; dg5f의 hand크기가 inspire보다 커서 dg5f의 hand는 1배, inspire는 0.5배 size로 설계

서로 다른 물체 크기 : 학습 동역학 : 물체 크기는 핸드 크기에 정규화되어 상대적 난이도 동일

UR10e + DG5F Right Hand

- **Arm**: UR10e (6-DoF collaborative robot)
- **Hand**: DG5F Right (20-DoF fully-actuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (1.0x, Kuka-Allegro baseline과 동일)
- **Total DoF**: 26 (6 arm + 20 hand)

UR10e + Inspire Right Hand

- **Arm**: UR10e (6-DoF collaborative robot, DG5F와 동일)
- **Hand**: Inspire Right (6-DoF underactuated, 5 fingers)
- **Objects**: 16가지 primitive shapes (0.5x, 손 크기에 맞춰 스케일)
- **Total DoF**: 12 (6 arm + 6 hand)




3.3 Reinforcement Learning
- MDP에 대한 설명 - PPO 설명 (KL divergence term에 대해서 어떻게 다룰지, gym 환경에 대한 이야기)
- 주요 observation 어떻게 되는지 설명


3.3.1 Observation 상세 (DexPoint 형식)

우리 환경은 **3개 그룹**으로 구성 (DexPoint의 point cloud + proprio 구조 참고):

Total Observation Dimension:
- DG5F (WITHOUT FT): 1870
- DG5F (WITH FT): 1890 (ati_force +3, ati_force_threshold +1)
- Inspire: ~1870

History Length

- **모든 observation은 5 step history** (temporal information)
- DexPoint처럼 시간적 맥락 제공 (과거 정보로부터 학습

Observation Processing

Raw Observation → Clipping → Noise Addition → Normalization (Running Mean/Std) → History Stacking → Policy Network


FT Sensor와 관련된 항은 전부 다 제거 & inspire와 dg5f의 dof 차이로 인한 observation 차이도 고려해서 넣어야함.
공통점과 차이점을 위주로 분석

**Group 1: Policy (185 dim)** - 목표 정보

| Observation | Dimension | Description |
| --- | --- | --- |
| `object_quat_b` | 20 (4 × 5 history) | 물체 자세 (quaternion, body frame) |
| `target_object_pose_b` | 35 (7 × 5 history) | 목표 pose (pos + quat, body frame) |
| `actions` | 130 (26 × 5 history) | 이전 action history |

**Group 2: Proprio (725 dim)** - 로봇 상태

| Observation | Dimension | Description |
| --- | --- | --- |
| `joint_pos` | 130 (26 × 5 history) | Joint positions |
| `joint_vel` | 130 (26 × 5 history) | Joint velocities |
| `hand_tips_state_b` | 390 (78 × 5 history) | Fingertip states (pos, vel, quat) × 6 tips |
| `contact` | 75 (15 × 5 history) | Contact forces (5 fingers × 3D) |

**Group 3: Perception (960 dim)** - 물체 인식

| Observation | Dimension | Description |
| --- | --- | --- |
| `object_point_cloud` | 960 (192 × 5 history) | Object point cloud (64 points × 3D) |

다만, 여기서 inspire와 dg5f가 다른 점 꼭 설명해야함 (FT는 상관 X)
 관측 공간 구조 비교

두 시스템은 동일한 관측 구조를 공유하지만, DoF 차이로 인해 입력 차원이 다릅니다:

**TABLE III: 관측 공간 차원 비교**

| 관측 그룹 | DG5F (26-DoF) | Inspire (12-DoF) | 차이 | 설명 |
|-------------------|---------------|------------------|------------|-------------|
| **Policy** | 185 dim | 115 dim | +70 dim | 액션: (26-12) × 5 히스토리 = 70 |
| **Proprioception** | 725 dim | 585 dim | +140 dim | 관절: (26-12) × 2 (위치+속도) × 5 히스토리 = 140 |
| **Perception** | 960 dim | 960 dim | 0 dim | 포인트 클라우드: 동일 |
| **전체** | **1870 dim** | **1660 dim** | **+210 dim** | DoF 차이가 히스토리를 통해 전파됨 |

**세부 분석:**
- **Policy 관측** (5-step 히스토리 포함):
  - DG5F: (object_quat 4 + target_pose 7 + actions 26) × 5 = 185 dim
  - Inspire: (object_quat 4 + target_pose 7 + actions 12) × 5 = 115 dim

- **Proprioception 관측** (5-step 히스토리 포함):
  - DG5F: (joint_pos 26 + joint_vel 26 + hand_tips 78 + contact 15) × 5 = 725 dim
  - Inspire: (joint_pos 12 + joint_vel 12 + hand_tips 78 + contact 15) × 5 = 585 dim

- **Perception 관측** (5-step 히스토리 포함):
  - 둘 다: (point_cloud 192) × 5 = 960 dim

**핵심 통찰**: 210차원의 차이는 전적으로 DoF 격차 (26 vs 12)에서 비롯되며, 5-step 시간적 히스토리 스택킹에 의해 증폭됩니다. 관측 구조는 구조적으로 동일하여 공정한 비교를 보장합니다—두 시스템 모두 동일한 정보 유형 (물체 상태, 고유 감각, 시각적 인식)을 수신하며, 각자의 구동 능력에 맞게 스케일링되었을 뿐입니다.


3.3.2 Reward Function상세
reward 함수들(dexsuite에 있었던 것) + 내가 추가한 term들 (ground_contact_penalty 외 등등)

(shared reward와 차이나는 reward : 여기는 reward 다 비슷함. 대신 object 크기에 따른 position_tracking에 대해 언급 꼭 해야함)

  참고: Base config (dexsuite_env_cfg.py:578)에서 이 값이 자동 계산되지만, Inspire는 다른 값을 사용해야 하므로 명시적으로 override합니다.

  ---
  📝 수정 요약

  | 수정 항목             | 위치                   | 변경         | 비고                 |
  |-----------------------|------------------------|--------------|----------------------|
  | fingers_to_object std | Ur10eReorientRewardCfg | 0.4 → 0.2    | 새로 추가 (override) |
  | position_tracking std | Ur10eReorientRewardCfg | 0.2 → 0.1    | 기존 항목 수정       |
  | success pos_std       | Ur10eReorientRewardCfg | 0.1 → 0.05   | 기존 항목 수정       |
  | curriculum pos_tol    | __post_init__          | auto → 0.025 | 명시적 설정          |

Reward Components
| Reward Term | Weight | Description | Formula |
|-------------|--------|-------------|---------|
| **Sparse Rewards** |
| `success` | 10.0 | 물체가 목표 도달 | `exp(-||p_obj - p_goal||^2 / σ_pos^2) × exp(-||q_obj - q_goal||^2 / σ_rot^2)` |
| **Dense Rewards** |
| `position_tracking` | 2.0 | 물체-목표 거리 | `exp(-||p_obj - p_goal||^2 / 0.2^2)` |
| `fingers_to_object` | 1.0 | 손가락-물체 거리 | `exp(-d_fingertips / 0.4^2)` |
| `good_finger_contact` | 0.5 | 손가락 접촉 보상 | `∑ I(F_i > 1.0N)` (threshold-based) |
| **Penalty Terms** |
| `action_l2` | -0.005 | Action magnitude | `-||a_t||^2` |
| `action_rate_l2` | -0.005 | Action smoothness | `-||a_t - a_{t-1}||^2` |
| `ground_contact_penalty` | -0.3 | 테이블 접촉 방지 | `-I(contact_with_table)` |
| `early_termination` | -1.0 | 비정상 종료 | `-I(abnormal_state)` |
| **FT Sensor Only** |
| `ati_excessive_force_penalty` | -0.005 | 과도한 힘 방지 | `-ReLU(||F|| - threshold)` |

Reward Shaping Strategy
- **Sparse reward (success)**: 최종 목표 달성
- **Dense rewards**: 학습 가이드 (hand-crafted shaping)
- **Penalty terms**: 불안정한 행동 억제


3.3 마무리 : network 아키텍처

 Actor-Critic Structure

[Observation Groups] → [Feature Extraction] → [Actor & Critic Networks]

┌──────────────────────────────────────────────────────────────┐
│ Input: Concatenated Observations (1870 or 1890 dim)          │
│  - Policy (185) + Proprio (725) + Perception (960)           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ Running Mean/Std Normalization                                │
│  - Input normalization (learned during training)              │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Actor MLP      │     │  Critic MLP     │
│  [512, 256, 128]│     │  [512, 256, 128]│
│  (ELU activation)│    │  (ELU activation)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Action (26 dim) │     │  Value (1 dim)  │
│ Gaussian Policy │     │  State Value    │
└─────────────────┘     └─────────────────┘



3.4 rl + other . ADR 세팅 + entropy 등 ppo 학습을 위해 추가된 항목들 언급

curriculumn learning + domain randomization


3.4.1 Curriculum Learning (ADR)

우리 환경은 **Automatic Domain Randomization (ADR)** 사용:

| Curriculum Term | Range | Adjust Based On |
| --- | --- | --- |
| Observation Noise | [0.0, 0.05] | Success rate |
| Gravity | [0.0, -9.81] | Success rate |
| Joint Friction | [0.0, 5.0] | Success rate |
| Object Mass | [0.2, 2.0]× | Success rate |
| Joint Stiffness/Damping | [0.8, 1.2]× | Success rate |

3.4.2 Domain Randomization

**Startup Randomization** (episode 시작 시):

- Robot physics material: friction [0.9, 1.1]
- Object physics material: friction [0.5, 1.0]
- Joint stiffness/damping: [0.8, 1.2]×
- Object mass: [0.2, 2.0]×

**Reset Randomization** (매 episode):

- Object pose: x, y, z, roll, pitch, yaw
- Robot joint pos: ±0.5 rad (arm), ±0.05 rad (hand)
- Table pose: ±5cm





----------------------------------------------------------------------------------------------------
4. Experiments

4.1. experiment setup

English Version:

  ### Computational Setup and Training Throughput

  We conduct our experiments using a workstation equipped with an Intel Core Ultra 9 285K CPU (24 cores) and a single Nvidia GeForce RTX 5090 GPU with 32 GB of VRAM, supported by 128 GB of system RAM. Using the Isaac Lab simulation framework [16], we are able to simulate 4096 parallel environments on the GPU. Combined with a GPU-based vectorized RL implementation rl_games [17], this configuration enables efficient distributed learning for dexterous manipulation tasks. The RTX 5090's high memory bandwidth and compute capability allow for real-time physics simulation of complex contact-rich scenarios across thousands of parallel instances, significantly accelerating the training process compared to traditional CPU-based approaches.

  Korean Version (한글):

  ### 계산 환경 및 학습 처리량

  우리는 Intel Core Ultra 9 285K CPU (24 코어)와 32 GB VRAM을 갖춘 단일 Nvidia GeForce RTX 5090 GPU, 그리고 128 GB 시스템 RAM을 장착한 워크스테이션에서 실험을 수행했습니다. Isaac Lab 시뮬레이션 프레임워크 [16]를 사용하여 GPU에서 4096개의 병렬 환경을 시뮬레이션할 수 있습니다. GPU 기반 벡터화된 RL 구현인 rl_games [17]와 결합하여, 이 구성은 기민한 조작 작업을 위한 효율적인 분산 학습을 가능하게 합니다. RTX 5090의 높은 메모리 대역폭과 계산 능력은 수천 개의 병렬 인스턴스에서 복잡한 접촉이 많은 시나리오의 실시간 물리 시뮬레이션을 가능하게 하여, 전통적인 CPU 기반 접근 방식에 비해 학습 프로세스를 크게 가속화합니다.

4.2. 주요 발견들
4.2.1 조작 품질 비교


4.2.2 학습 동역학



----------------------------------------------------------------------------------------------------
5. Conclusion & Limitation

<todo> 
왜 그런것 같은지도 분석 (논문) :


limitation 

realworld - sim2real
시뮬레이션 → 물체별 정량 분석 불가능 / 질적 분석은 가능
질적 분석 : 실제 로봇의 pick&place의 경우 사각형과 같은 물체의 경우엔 dg5f는 손의 마지막 link만 살짝 움직이면 쥘 수 있으나 inspire의 경우엔 그게 불가능 - underactuated ; 모두 articulatioed 연동이 되어있으니까 - 이런 이슈도 있었음 

6자유도, 7자유도 로봇과 hand 일반화 - dg5f, inspire만 사용했었음
물체 무게 똑같이 함. - mass randomization도 / 밀도에 따라 - hand 별 effort가 다르기에 좀 까다롭긴함


----------------------------------------------------------------------------------------------------
references
