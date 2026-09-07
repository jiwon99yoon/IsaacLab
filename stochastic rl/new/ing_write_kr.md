일단 한글로 먼저 작성,

논문 형식은 0. Abstract (나중에 작성) 1. Introduction, 2. Background and related work 3. Method 4. Experiments 5. Conclusion & Limitation
으로 작성

----------------------------------------------------------------------------------------------------

\section{Background and Related Work}

In this section, we provide the necessary background on reinforcement learning and review relevant prior work on dexterous manipulation, hand design, and the intersection of hardware structure and learning-based control.

\subsection{Proximal Policy Optimization (PPO)}

Our experiments employ \textbf{Proximal Policy Optimization (PPO)} \cite{schulman2017ppo}, a widely-used policy gradient algorithm proven effective for high-dimensional continuous control tasks, including dexterous manipulation \cite{rajeswaran2018dexterous, petrenko2023dexpbt}. We formalize the robot control problem as a Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, \mathcal{T}, c, \rho_1, \gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{T}$ is the transition probability, $c$ is the reward function, $\rho_1$ is the initial state distribution, and $\gamma \in (0,1)$ is the discount factor. The goal is to maximize expected cumulative reward $\eta(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=1}^{\infty} \gamma^{t-1} c(s_t, a_t) \right]$.

\subsubsection{Clipped Surrogate Objective}

PPO employs a \textbf{clipped surrogate objective} to prevent the policy from deviating too much from the old policy, ensuring stable learning. The clipped objective function is:

\begin{equation}
L_t^{\text{CLIP}}(\theta) = \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)
\end{equation}

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio, $\hat{A}_t$ is the estimated advantage, and $\epsilon$ is the clipping parameter (typically 0.2). By restricting $r_t(\theta)$ to $[1-\epsilon, 1+\epsilon]$, PPO ensures policy updates remain within a trust region, promoting stable and monotonic improvement \cite{wang2019truly}.

\subsubsection{Trust Region and KL Divergence}

An alternative formulation of PPO uses a KL divergence constraint instead of clipping:

\begin{equation}
\begin{aligned}
&\max_{\theta} \, \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \hat{A}(s,a) \right] \\
&\text{s.t. } \max_{s \in \mathcal{S}} D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot | s) \| \pi_\theta(\cdot | s)) \leq \delta
\end{aligned}
\end{equation}

where $D_{\text{KL}}$ is the Kullback-Leibler divergence and $\delta$ is the constraint bound. This formulation, known as Trust Region Policy Optimization (TRPO) \cite{schulman2015trpo}, guarantees monotonic policy improvement but requires a computationally expensive second-order optimization. PPO's clipped objective achieves similar stability guarantees with a simple first-order optimization, making it more practical for large-scale applications.

\subsubsection{Applicability to Dexterous Manipulation}

PPO is particularly well-suited for dexterous manipulation tasks. The clipped objective and bounded policy updates prevent catastrophic policy collapse, which is critical in contact-rich scenarios where small policy changes can lead to drastically different contact patterns. PPO reuses data multiple times via mini-batch updates over multiple epochs, improving sample efficiency compared to on-policy methods like REINFORCE—crucial in simulation-based learning where environment interaction is parallelized. PPO scales well to high-dimensional continuous action spaces (e.g., 26-DoF for DG5F) using stochastic gradient descent without requiring Hessian computation, and is less sensitive to hyperparameter choices compared to TRPO or vanilla policy gradients, facilitating deployment across different hand configurations without extensive tuning. In our experiments, we use PPO as implemented in RL-Games \cite{rlgames}, a GPU-accelerated framework that enables massively parallel training across 4,096 environments.

---

\subsection{Dexterous Manipulation with Reinforcement Learning}

Reinforcement learning has enabled significant progress in learning complex dexterous manipulation skills that are difficult to program using traditional control methods. \textbf{Rajeswaran et al.} \cite{rajeswaran2018dexterous} demonstrated that high-DoF dexterous manipulation (e.g., Shadow Hand with 24-DoF) can be learned using deep RL combined with human demonstrations, revealing that high-DoF hands create severe exploration challenges due to large action spaces, sparse rewards, and sample inefficiency. This motivates our hypothesis that underactuated hands may reduce exploration difficulty by constraining the action space through mechanical coupling.

\textbf{DexPBT} \cite{petrenko2023dexpbt} introduced a large-scale RL framework for hand-arm systems using Population Based Training (PBT), achieving complex manipulation through massive parallelization (16,384 environments) and Automatic Domain Randomization (ADR) curriculum learning. Our work adopts similar scale (4,096 parallel environments) and curriculum strategies, extending DexPBT by systematically comparing different hand actuation structures under identical training conditions. \textbf{DexPoint} \cite{qi2023dexpoint} further demonstrated that point cloud-based observations improve generalization across diverse object shapes compared to proprioceptive-only inputs; we incorporate point cloud perception (192-point object point clouds with 5-step history) following this best practice.

\textbf{OpenAI's Rubik's Cube work} \cite{openai2019rubiks} showcased the power of domain randomization in bridging the sim-to-real gap by randomizing object properties (mass, friction, size), robot dynamics (joint stiffness, damping), and environmental conditions (gravity, observation noise). Our implementation employs extensive domain randomization across object mass (0.2--2.0 kg), friction (0.5--1.0), and robot dynamics, ensuring learned policies are robust to real-world uncertainties.

---

\subsection{Underactuated vs Fully-Actuated Hand Designs}

Robotic hand design has long explored the trade-off between mechanical simplicity and control expressiveness. \textbf{Catalano et al.} \cite{catalano2014softhand} demonstrated that underactuated soft hands with adaptive synergies can achieve robust grasps through passive compliance and mechanical coupling, mimicking human hand synergies while reducing control complexity—however, their focus was on mechanical design and classical control, not on how these structures affect reinforcement learning dynamics. \textbf{Sintov et al.} \cite{sintov2019underactuated} showed that underactuated hands are difficult to model analytically due to contact nonlinearities and coupling, making data-driven methods a natural fit, but did not compare learning performance against fully-actuated alternatives under controlled conditions.

\textbf{Lopes et al.} \cite{lopes2025comparative} conducted a comparative analysis of underactuated vs fully-actuated hand designs, identifying a critical gap: most studies focus on a single design, and when comparisons are made, multiple variables change simultaneously (different arms, tasks, algorithms), making it impossible to isolate the effect of actuation structure. While their work focused on classical control methods, our work extends this comparative approach to reinforcement learning, where exploration, sample efficiency, and emergent strategies introduce new dimensions previously unexplored.

From an RL perspective, we hypothesize that underactuated synergies may act as an inductive bias in the action space, constraining exploration to synergistic motions that facilitate stable grasps, while fully-actuated hands must discover these synergies through learning—yet no prior work has tested this hypothesis under rigorous experimental control.

---

\subsection{Gap in Literature and Our Contributions}

Despite extensive research on both RL-based dexterous manipulation and underactuated hand design, a critical gap remains: \textit{How does hand actuation structure (underactuated vs fully-actuated) affect reinforcement learning performance, sample efficiency, and emergent manipulation strategies?} Most dexterous RL work focuses on improving algorithms rather than understanding how hardware structure influences learning, and when different hand designs are compared, multiple variables change simultaneously (arm platform, DoF, task, algorithm), making it impossible to isolate the effect of actuation structure. Moreover, underactuated hands remain underrepresented in the RL community despite their practical advantages.

We address this gap by providing \textbf{the first systematic, controlled comparison of underactuated vs fully-actuated hands in RL-based dexterous manipulation}. Our contributions include: (1) \textbf{Controlled protocol} fixing all variables except hand actuation (same arm, task, algorithm, infrastructure, perception); (2) \textbf{Fair task design} via scale-adjusted object sizes (1.0× for DG5F, 0.5× for Inspire) and proportional reward parameters; (3) \textbf{Comprehensive analysis} beyond success rate, including sample efficiency, curriculum progression, stability, and learning dynamics; (4) \textbf{Reproducible setup} with detailed experimental configurations.

Our work reframes hand design from a purely mechanical problem to a \textbf{learning problem}, proposing that actuation structure acts as an \textbf{inductive bias} in RL: underactuated hands provide implicit regularization through mechanical coupling (potentially easing exploration but reducing expressiveness), while fully-actuated hands offer higher expressiveness (enabling complex strategies but increasing exploration difficulty). By demonstrating how hardware structure influences RL learning, we enable \textbf{hardware-algorithm co-design}—the joint optimization of mechanical design and learning algorithms for dexterous manipulation systems.

---

\textbf{References for Section 2:}

\begin{thebibliography}{99}

\bibitem{schulman2017ppo}
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, ``Proximal policy optimization algorithms,'' \textit{arXiv preprint arXiv:1707.06347}, 2017.

\bibitem{rajeswaran2018dexterous}
A. Rajeswaran et al., ``Learning complex dexterous manipulation with deep reinforcement learning and demonstrations,'' in \textit{Robotics: Science and Systems (RSS)}, 2018.

\bibitem{petrenko2023dexpbt}
A. Petrenko et al., ``DexPBT: Scaling up dexterous manipulation for hand-arm systems with population based training,'' in \textit{Robotics: Science and Systems (RSS)}, 2023.

\bibitem{sutton1999policy}
R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, ``Policy gradient methods for reinforcement learning with function approximation,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 1999.

\bibitem{wang2019truly}
Y. Wang et al., ``Truly proximal policy optimization,'' in \textit{Conference on Uncertainty in Artificial Intelligence (UAI)}, 2019.

\bibitem{schulman2015trpo}
J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, ``Trust region policy optimization,'' in \textit{International Conference on Machine Learning (ICML)}, 2015.

\bibitem{rlgames}
A. Makoviychuk and V. Makoviichuk, ``RL-Games: A high-performance framework for reinforcement learning,'' \url{https://github.com/Denys88/rl_games}, 2021.

\bibitem{qi2023dexpoint}
Y. Qi et al., ``DexPoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation,'' in \textit{Conference on Robot Learning (CoRL)}, 2023.

\bibitem{openai2019rubiks}
OpenAI et al., ``Solving Rubik's Cube with a robot hand,'' \textit{arXiv preprint arXiv:1910.07113}, 2019.

\bibitem{catalano2014softhand}
M. G. Catalano et al., ``Adaptive synergies for the design and control of the Pisa/IIT SoftHand,'' \textit{International Journal of Robotics Research (IJRR)}, vol. 33, no. 5, pp. 768--782, 2014.

\bibitem{sintov2019underactuated}
A. Sintov, O. Tslil, and A. Shapiro, ``Data-driven modeling and control of an underactuated hand,'' \textit{IEEE Robotics and Automation Letters (RAL)}, vol. 4, no. 3, pp. 2246--2253, 2019.

\bibitem{lopes2025comparative}
L. Lopes et al., ``Comparative analysis of underactuated and fully-actuated hand designs for robotic manipulation,'' \textit{arXiv preprint arXiv:2501.xxxxx}, 2025.

\end{thebibliography}

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
- 제어 복잡도: 높음 (모든 관절을 협응해야 함)3

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

  📚 추가 필요한 References

  파일 끝에 다음 references 추가해야 해:

  [10] Allegro Hand - Wonik Robotics
  [11] Shadow Dexterous Hand - Shadow Robot Company
  [12] Humanoid robots with 5-finger hands (Tesla Optimus, Figure 01, 1X NEO)
  [13] TESOLLO DG-5F: https://en.tesollo.com/dg-5f/
  [14] Inspire Robots RH56F1: https://en.inspire-robots.com/
  [15] Birglen, L., Laliberté, T., & Gosselin, C. M. (2009). "The ability of underactuated hands to grasp and hold objects." Mechanism and Machine Theory.
  [16] Catalano et al. (IJRR 2014) - Adaptive Synergies (이미 2.3에 있음)
  [17] Comparison of underactuated vs fully-actuated systems

### 3.2. 실험 환경: Isaac Lab 기반 Dexsuite 벤치마크

#### 3.2.1. Isaac Lab 시뮬레이션 프레임워크

본 연구는 NVIDIA Isaac Lab [18] 시뮬레이션 환경을 사용하여 실험을 수행하였다. Isaac Lab은 GPU 기반 병렬 물리 시뮬레이션 엔진으로, 수천 개의 환경을 동시에 시뮬레이션할 수 있는 고속 학습 인프라를 제공한다. 우리는 단일 NVIDIA GeForce RTX 5090 GPU(32 GB VRAM)를 사용하여 **4,096개의 병렬 환경**을 동시에 시뮬레이션하였으며, 이는 기존 CPU 기반 시뮬레이션 대비 수백 배 빠른 학습 속도를 가능하게 한다. GPU 기반 벡터화된 강화학습 구현체인 RL-Games [19]와 결합하여, 접촉이 많은(contact-rich) 복잡한 조작 시나리오를 실시간으로 학습할 수 있었다.

#### 3.2.2. Dexsuite 벤치마크와 선행 연구

우리의 실험 환경은 DexPBT [20]에서 제안된 Dexsuite 벤치마크를 기반으로 구성되었다. DexPBT는 hand-arm 시스템을 위한 대규모 강화학습 프레임워크로, Population Based Training(PBT)과 대규모 병렬화(16,384 환경)를 통해 복잡한 기민한 조작 행동을 학습할 수 있음을 보였다. 또한 DexPoint [21]는 point cloud 기반 인식(perception)을 사용하여 다양한 물체 형상에 대한 일반화(generalization) 성능을 향상시켰다. 우리는 이러한 선행 연구들의 설계 원칙을 따르되, **핸드 구동 구조의 영향을 체계적으로 분석**하기 위해 실험 프로토콜을 조정하였다.

figure2에서 볼 수 있듯이 기존 Dexsuite 벤치마크는 주로 KUKA iiwa 7-DoF 로봇 팔과 Allegro Hand(4-finger) 조합을 사용하였다. 그러나 본 연구에서는 **UR10e 6-DoF 협동 로봇(collaborative robot)**을 팔 플랫폼으로 선택하였다. 이는 다음과 같은 이유에서 기인한다. 첫째, iiwa의 7-DoF 구조는 **운동학적 중복성(kinematic redundancy)**을 갖고 있어, 동일한 end-effector 위치에 도달하기 위한 무한히 많은 관절 구성(joint configuration)이 존재한다. 이러한 중복성은 강화학습 관점에서 추가적인 탐색 복잡도를 유발할 수 있으며, 핸드 자체의 구동 구조 차이를 분석하는 본 연구의 목적에 혼란 요인(confounding factor)으로 작용할 수 있다. 반면, UR10e의 6-DoF 구조는 **최소 자유도(minimal DoF)**로 3차원 공간에서의 위치 및 자세 제어가 가능하며, 중복성이 없어 팔의 움직임이 결정론적(deterministic)이다. 둘째, UR10e는 산업 현장에서 널리 사용되는 상용 협동 로봇으로, 실제 응용 측면에서 실용성이 높다. 따라서 우리는 UR10e를 선택함으로써 **팔-핸드 시스템의 총 자유도를 최소화**하고, 핸드 구동 구조 차이가 학습에 미치는 순수한 영향을 관찰하고자 하였다.
우리의 ur10e와 dg5f, ur10e와 inspire hand의 결합 구조는 figure 3에서 볼 수 있다. 

#### 3.2.3. 작업 정의: 물체 들어올리기(Object Lifting)

실험 작업은 **물체 들어올리기(object lifting)** 과제로 정의된다. 로봇은 테이블 위에 놓인 물체를 집어 올려(lift), 목표 위치(target position)로 이동시켜야 한다. 목표 위치는 에피소드마다 무작위로 샘플링되며, **회전(orientation)은 제어 대상이 아니다**. 즉, 이 작업은 3차원 공간에서의 **위치 제어(position control)**에 집중하며, 물체의 자세는 자유롭게 변할 수 있다. 이 작업은 단순한 파지(grasping)를 넘어서 손가락 간 협응(finger coordination), 접촉 관리(contact management), 그리고 물체 동역학(object dynamics)에 대한 이해가 필요하다. Lift task는 기존 Dexsuite의 reorientation task(위치+회전 모두 제어)보다 상대적으로 단순하지만, 접촉 유지(contact maintenance)와 안정적인 파지(stable grasping)를 통한 위치 정밀도(position precision)를 요구한다는 점에서 여전히 도전적인 과제이다.

#### 3.2.4. 물체 다양성과 복잡성

공정한 비교를 위해, 우리는 **4가지 기본 형상(primitive shapes)**의 물체를 사용하였다: cube(정육면체), cylinder(원기둥), box(직육면체), ellipsoid(타원체). 각 형상마다 **4가지 다른 크기**를 설정하여, 총 **16가지 물체 변형(object variations)**을 생성하였다. 이러한 다양성은 정책(policy)이 특정 물체 크기에 과적합(overfitting)되는 것을 방지하고, 일반화된 조작 전략을 학습하도록 유도한다. 예를 들어, cube의 경우 한 변의 길이가 0.025m부터 0.05m까지 다양하게 설정되어, 작은 물체는 손가락 끝으로 정밀하게 조작해야 하고(precision grasp), 큰 물체는 손바닥 전체로 감싸야 하는(enveloping grasp) 등 서로 다른 파지 전략을 요구한다.

각 에피소드의 시작 시점에서 물체의 종류와 크기는 16가지 후보 중 무작위로 선택되며, 초기 위치 및 자세 또한 일정 범위 내에서 랜덤화된다. 이러한 **작업 다양성(task diversity)**은 로봇이 한정된 시나리오에만 특화되지 않고, 다양한 물체 속성과 초기 조건에서도 성공적으로 조작할 수 있는 견고한(robust) 정책을 학습하도록 강제한다. 이는 실제 세계 응용(real-world application)을 고려할 때 필수적인 요소이다.

#### 3.2.5. 핸드 크기에 따른 물체 스케일링

DG5F와 Inspire RH56F1은 물리적 크기에서 상당한 차이가 있다(DG5F: 약 20cm, Inspire: 약 12-15cm). 만약 동일한 크기의 물체를 두 핸드에 사용한다면, Inspire 핸드는 상대적으로 훨씬 큰 물체를 다루게 되어 작업 난이도가 불공정하게 증가한다. 예를 들어, 0.05m 크기의 cube는 DG5F에게는 손바닥 크기의 약 25%에 해당하지만, Inspire에게는 약 40%에 육박하여 파지가 매우 어려워진다. 이러한 불균형은 학습 성능 비교를 왜곡시킬 수 있다.

따라서 우리는 **핸드 크기에 비례하여 물체 크기를 스케일링**하는 전략을 채택하였다. 구체적으로, DG5F는 **1.0× 스케일**(기존 Kuka-Allegro 벤치마크와 동일한 물체 크기)을 사용하고, Inspire는 **0.5× 스케일**(모든 물체 차원을 절반으로 축소)을 적용하였다. 이는 대략적으로 두 핸드의 손바닥 크기 비율(DG5F: Inspire ≈ 2:1)을 반영한 것이다. 이러한 스케일링을 통해, **핸드 크기 대비 물체 크기의 상대적 비율**을 양 시스템에서 유사하게 유지함으로써, 작업의 기하학적 난이도(geometric difficulty)를 균등화하였다.

단, 물체의 **질량(mass)**은 스케일링하지 않고 동일하게 유지하였다(0.2 kg). 이는 부피가 1/8로 줄어든 Inspire의 물체가 DG5F 대비 8배 높은 밀도를 갖게 됨을 의미한다. 이러한 결정은 두 핸드의 **하드웨어 능력 차이**를 있는 그대로 평가하기 위함이다. 만약 질량까지 스케일링한다면(0.5× → 0.025 kg), Inspire는 극히 가벼운 물체만 다루게 되어 실제 조작 능력을 제대로 평가할 수 없다. 동일한 질량을 유지함으로써, 두 핸드가 **동일한 물리적 조건**(질량, 중력, 관성)에서 작업을 수행하며, 핸드의 구동력(actuation force)과 제어 능력이 직접적으로 비교될 수 있다.

이러한 스케일 조정은 보상 함수(reward function)에도 반영되었다. 위치 추적(position tracking) 및 성공 판정(success) 보상의 표준편차 파라미터를 Inspire에 대해 절반으로 조정하여(position_tracking std: 0.2 → 0.1, success pos_std: 0.1 → 0.05), 물체 크기에 비례한 정밀도 요구사항을 유지하였다. 자세한 보상 함수 설계는 3.3.2절에서 다룬다.

#### 3.2.6. 최종 실험 구성

위의 설계 원칙에 따라, 우리의 비교 실험은 다음과 같이 구성되었다:

**UR10e + DG5F Right Hand 시스템:**
- **로봇 팔**: UR10e (6-DoF collaborative robot, 최소 중복성)
- **로봇 핸드**: DG5F Right (20-DoF fully-actuated, 5 fingers)
- **물체**: 16가지 primitive shapes (**1.0× 스케일**, 기존 Kuka-Allegro 벤치마크와 동일)
  - Cube, Cylinder, Box, Ellipsoid 각 4가지 크기
- **총 자유도**: 26-DoF (6 arm + 20 hand)
- **관측 차원**: 1,870 dim (without FT sensor)
- **행동 차원**: 26 dim (연속 제어)

**UR10e + Inspire Right Hand 시스템:**
- **로봇 팔**: UR10e (6-DoF collaborative robot, DG5F와 동일)
- **로봇 핸드**: Inspire RH56F1 Right (6-DoF underactuated, 5 fingers)
- **물체**: 16가지 primitive shapes (**0.5× 스케일**, 핸드 크기에 맞춰 조정)
  - Cube, Cylinder, Box, Ellipsoid 각 4가지 크기
- **총 자유도**: 12-DoF (6 arm + 6 hand)
- **관측 차원**: 1,660 dim (without FT sensor)
- **행동 차원**: 12 dim (연속 제어)

두 시스템은 **동일한 작업(object lifting), 동일한 알고리즘(PPO), 동일한 보상 구조, 동일한 커리큘럼 학습(ADR)** 조건에서 학습되며, 유일한 차이는 **핸드의 구동 구조**(underactuated vs fully-actuated)와 이에 따른 자유도 및 관측/행동 차원이다. 이러한 통제된 실험 설계(controlled experimental design)를 통해, 우리는 구동 구조가 강화학습 성능에 미치는 순수한 영향을 분리하여 분석할 수 있다.

---

**References for Section 3.2:**

[18] NVIDIA Isaac Lab - GPU-accelerated robotic simulation framework. https://isaac-sim.github.io/IsaacLab/

[19] RL-Games - GPU-based vectorized reinforcement learning implementation. https://github.com/Denys88/rl_games

[20] Petrenko, A., et al. (2023). "DexPBT: Scaling Up Dexterous Manipulation for Hand-Arm Systems with Population Based Training." RSS 2023. https://roboticsproceedings.org/rss19/p027.pdf

[21] Qi, Y., et al. (2023). "DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation." CoRL 2023.

### 3.3. 강화학습 정식화 및 학습 설정

앞서 Section 2.1에서 소개한 PPO 알고리즘의 이론적 배경을 바탕으로, 본 절에서는 우리의 dexterous manipulation 환경에서 강화학습 문제를 어떻게 구체화했는지 설명한다. 우리는 MDP의 각 요소(상태, 행동, 보상, 전이 확률)를 정의하고, PPO 학습을 위한 하이퍼파라미터 설정, 관측 공간 구조, 보상 함수 설계, 그리고 신경망 아키텍처를 상세히 다룬다.

**MDP 설정 및 PPO 하이퍼파라미터**

우리의 환경은 MDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, c, \rho_1, \gamma)$로 정식화된다. 상태 공간 $\mathcal{S}$는 로봇의 고유 감각(proprioception), 물체의 위치 및 자세, 목표 pose, 그리고 point cloud 기반 시각 인식으로 구성되며, 이는 3.3.1절에서 상세히 설명한다. 행동 공간 $\mathcal{A}$는 연속 공간으로, DG5F의 경우 26차원(6 arm + 20 hand), Inspire의 경우 12차원(6 arm + 6 hand)이다. 전이 확률 $\mathcal{T}$는 Isaac Lab의 GPU 기반 물리 엔진(PhysX)에 의해 결정되며, 접촉 동역학, 마찰, 중력 등을 정확하게 시뮬레이션한다. 보상 함수 $c$는 3.3.2절에서 다루며, discount factor $\gamma = 0.99$를 사용한다.

PPO 학습을 위해 다음과 같은 하이퍼파라미터를 사용한다 (rl_games_ppo_cfg.yaml 참고). Clipping parameter는 $\epsilon = 0.2$로 설정하여 정책 업데이트가 trust region 내에 유지되도록 한다. KL divergence threshold는 `kl_threshold = 0.01`로 설정되며, adaptive learning rate schedule과 결합하여 KL divergence가 threshold를 초과하면 학습률을 감소시킨다. Entropy coefficient는 `entropy_coef = 0.001`로 설정하여 정책의 탐색을 유도하되, 너무 강한 entropy 정규화는 피한다. Gradient는 `truncate_grads = True`, `grad_norm = 1.0`으로 클리핑되어 학습 안정성을 확보한다. 각 업데이트는 horizon length 36 스텝의 데이터를 수집하고, 이를 **importance sampling**과 **clipping** 메커니즘을 통해 5번의 mini-epoch 동안 minibatch size 36,864로 재사용하여 샘플 효율성을 높인다. Importance sampling은 old policy로 수집한 데이터를 probability ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$로 재가중하여 new policy 학습에 활용할 수 있게 하며, clipping은 이 ratio를 $[1-\epsilon, 1+\epsilon]$ 범위로 제한하여 variance 폭발을 방지한다. Critic coefficient는 `critic_coef = 4.0`으로 설정되어 가치 함수 학습을 강조한다.

**관측과 상태의 동치성 (Observation-State Equivalence)**

이론적으로, 우리의 환경은 POMDP(Partially Observable MDP)에 가깝다. Point cloud는 물체 표면의 64개 샘플링된 점으로, 완전한 mesh 정보를 제공하지 않으며, domain randomization으로 무작위화된 물리 파라미터(mass, friction, stiffness)는 agent가 직접 관측할 수 없다. 그러나 실질적으로, 우리는 이를 **MDP로 취급**한다. 이는 robotic manipulation 분야의 일반적인 관행을 따른 것으로 [20, 21], 다음과 같은 이유로 정당화된다. 첫째, **5-step history stacking**은 속도, 가속도, 그리고 물체 동역학을 간접적으로 추론할 수 있게 하여 partial observability를 완화한다. 둘째, point cloud 기반 인식은 물체의 형상, 자세, 크기에 대한 충분히 풍부한 정보를 제공하며, DexPoint [21]에서 입증된 바와 같이 다양한 물체에 대한 일반화를 가능하게 한다. 셋째, 물리 파라미터는 직접 관측되지 않지만, agent는 접촉 반응(contact forces, object motion)을 통해 이를 간접적으로 추론하며, domain randomization 덕분에 다양한 조건에 robust하게 학습된다.

우리의 구현에서는 `obs_groups`와 `states`가 동일하게 설정되어 있다 (rl_games_ppo_cfg.yaml: `obs: ["policy", "proprio", "perception"]`, `states: ["policy", "proprio", "perception"]`). 이는 관측($o_t$)과 상태($s_t$)를 구분하지 않고, 동일한 정보를 정책 네트워크(actor)와 가치 네트워크(critic) 모두에 입력한다는 의미이다. 따라서 이후 본 절에서 "관측(observation)"과 "상태(state)"는 상호 교환 가능한 용어로 사용되며, 둘 다 agent가 환경으로부터 수신하는 정보를 지칭한다.

#### 3.3.1. 관측 공간 구조

우리의 관측 공간은 DexPoint [21]의 설계 원칙을 따라 **3개의 그룹**으로 구성된다: **Policy** (목표 정보), **Proprioception** (로봇 상태), **Perception** (물체 인식). 모든 관측은 **5-step history**를 유지하여 시간적 맥락(temporal context)을 정책에 제공한다. 이는 물체의 움직임 방향, 접촉의 지속성, 그리고 행동의 영향을 학습하는 데 필수적이다. 관측 데이터는 다음과 같이 처리된다: Raw Observation → Clipping (±100.0) → Noise Addition (curriculum에 따라 조정) → Running Mean/Std Normalization (학습 중 업데이트) → History Stacking → Policy Network.

<그림 삽입 예정: 관측 공간 구조 다이어그램 - 3개 그룹과 history stacking 시각화>

**Group 1: Policy Observations (목표 정보)**

Policy 그룹은 작업 목표와 관련된 관측을 포함한다. `object_quat_b` (4차원 × 5 history = 20차원)는 현재 물체의 자세(quaternion)를 robot base frame에서 표현한 것이다. `target_object_pose_b` (7차원 × 5 history = 35차원)는 목표 pose(position 3차원 + quaternion 4차원)를 나타내며, 에피소드마다 무작위로 샘플링되어 정책이 다양한 목표에 일반화되도록 한다. `actions` (26차원 × 5 history = 130차원, DG5F 기준)는 이전 행동의 history를 포함하여, 정책이 자신의 과거 행동 패턴을 인식하고 일관성 있는 제어를 수행할 수 있게 한다. DG5F의 경우 Policy 그룹 총 차원은 185차원이며, Inspire의 경우 행동 차원이 12로 줄어들어 총 115차원이 된다.

**Group 2: Proprioception Observations (로봇 상태)**

Proprioception 그룹은 로봇의 내부 상태를 나타낸다. `joint_pos` (26차원 × 5 = 130차원, DG5F 기준)와 `joint_vel` (26차원 × 5 = 130차원)은 모든 관절의 위치 및 속도 정보를 제공한다. `hand_tips_state_b` (78차원 × 5 = 390차원)는 5개 손가락 끝 + 손바닥의 상태(position 3차원, linear velocity 3차원, quaternion 4차원, angular velocity 3차원)를 포함하여, 총 6개 body × 13차원 = 78차원이 된다. 이는 손가락이 물체와 어떻게 상호작용하는지에 대한 직접적인 피드백을 제공한다. `contact` (15차원 × 5 = 75차원)는 5개 손가락 각각의 3차원 접촉력을 나타내며, 파지 품질을 평가하는 데 사용된다. DG5F의 경우 Proprioception 그룹 총 차원은 725차원이며, Inspire의 경우 관절 차원이 12로 줄어들어 총 585차원이 된다.

**Group 3: Perception Observations (물체 인식)**

Perception 그룹은 시각 기반 물체 인식을 담당한다. `object_point_cloud` (192차원 × 5 = 960차원)는 물체 표면의 point cloud를 robot base frame에서 표현한 것으로, 64개의 점 × 3차원(x, y, z) = 192차원이며, 5-step history를 통해 총 960차원이 된다. Point cloud는 물체의 형상, 크기, 자세에 대한 rich한 기하학적 정보를 제공하며, DexPoint [21]에서 입증된 바와 같이 다양한 물체 형태에 대한 일반화 성능을 크게 향상시킨다. 중요한 점은, Perception 그룹은 두 핸드 시스템 모두에서 **동일한 차원**(960)을 갖는다는 것이다. 이는 물체 인식이 핸드의 구동 방식과 독립적임을 보여준다.

**관측 공간 차원 비교**

두 시스템은 동일한 관측 구조를 공유하지만, DoF 차이로 인해 입력 차원이 다르다. 아래 표는 두 핸드의 관측 공간 차원을 비교한다.

**TABLE IV: 관측 공간 차원 비교**

| 관측 그룹          | DG5F (26-DoF) | Inspire (12-DoF) | 차이      | 설명                                      |
|--------------------|---------------|------------------|-----------|-------------------------------------------|
| **Policy**         | 185 dim       | 115 dim          | +70 dim   | 액션 히스토리: (26-12) × 5 = 70           |
| **Proprioception** | 725 dim       | 585 dim          | +140 dim  | 관절 정보: (26-12) × 2 × 5 = 140          |
| **Perception**     | 960 dim       | 960 dim          | 0 dim     | Point cloud: 동일                         |
| **전체**           | **1870 dim**  | **1660 dim**     | **+210 dim** | DoF 차이가 히스토리를 통해 전파됨      |

핵심 통찰은, 210차원의 차이가 전적으로 DoF 격차(26 vs 12)에서 비롯되며, 5-step 시간적 히스토리 스택킹에 의해 증폭된다는 것이다. 관측 구조는 구조적으로 동일하여 공정한 비교를 보장한다. 두 시스템 모두 동일한 정보 유형(물체 상태, 고유 감각, 시각적 인식)을 수신하며, 각자의 구동 능력에 맞게 스케일링되었을 뿐이다.

#### 3.3.2. 보상 함수 설계

우리의 보상 함수는 **sparse reward**(최종 목표 달성), **dense rewards**(학습 가이드를 위한 shaping), 그리고 **penalty terms**(불안정한 행동 억제)의 조합으로 구성된다. 이러한 설계는 Kuka-Allegro 기반 Dexsuite 벤치마크 [20]를 기반으로 하되, UR10e-DG5F/Inspire 시스템에 맞게 조정되었다. 핸드 크기 차이에 따른 물체 스케일링(1.0× vs 0.5×)을 고려하여, Inspire 시스템의 position tracking 관련 보상 파라미터는 비례적으로 조정되었다.

**Sparse Reward: Success**

`success` (weight = 10.0)는 물체가 목표 pose에 도달했을 때 주어지는 주요 보상이다. 이는 position error와 orientation error의 Gaussian 함수 곱으로 정의된다:

$$
r_{\text{success}} = \exp\left(-\frac{\|p_{\text{obj}} - p_{\text{goal}}\|^2}{\sigma_{\text{pos}}^2}\right) \times \exp\left(-\frac{\|\text{quat\_diff}(q_{\text{obj}}, q_{\text{goal}})\|^2}{\sigma_{\text{rot}}^2}\right)
$$

여기서 $\sigma_{\text{pos}}$는 위치 허용 오차, $\sigma_{\text{rot}}$는 회전 허용 오차이다. DG5F는 $\sigma_{\text{pos}} = 0.1$m, Inspire는 $\sigma_{\text{pos}} = 0.05$m를 사용하여, 물체 크기 비율(2:1)에 맞춘 상대적 정밀도를 요구한다.

**Dense Rewards: Shaping for Guidance**

`position_tracking` (weight = 2.0)은 물체와 목표 위치 간 거리를 최소화하도록 유도한다:

$$
r_{\text{position}} = \exp\left(-\frac{\|p_{\text{obj}} - p_{\text{goal}}\|^2}{\sigma^2}\right)
$$

DG5F는 $\sigma = 0.2$m, Inspire는 $\sigma = 0.1$m를 사용한다. `orientation_tracking` (weight = 4.0)은 물체의 회전을 목표 자세로 정렬하도록 유도하며, orientation error는 position error보다 2배 높은 가중치를 갖는다. 이는 회전 제어가 위치 제어보다 어렵기 때문이다.

`fingers_to_object` (weight = 1.0)는 손가락 끝과 물체 사이의 평균 거리를 줄여 파지를 유도한다:

$$
r_{\text{fingers}} = \exp\left(-\frac{d_{\text{avg}}}{0.4^2}\right)
$$

여기서 $d_{\text{avg}}$는 5개 손가락 끝과 물체 중심 간 평균 거리이다 (DG5F 기준 0.4m, Inspire 기준 0.2m).

**Penalty Terms: Regularization**

`action_l2` (weight = -0.005)와 `action_rate_l2` (weight = -0.005)는 각각 행동의 크기와 변화율을 억제하여, 부드럽고 에너지 효율적인 움직임을 유도한다:

$$
r_{\text{action\_l2}} = -\|a_t\|^2, \quad r_{\text{action\_rate}} = -\|a_t - a_{t-1}\|^2
$$

`ground_contact_penalty` (weight = -0.3)는 물체가 테이블이나 바닥에 접촉할 때 패널티를 부여하여, "테이블에 물체를 누르는(table pinning)" 전략을 방지한다. 초기 실험에서 높은 finger contact 보상(2.0)과 결합 시 이러한 shortcut이 발생했으며, 패널티를 -0.05에서 -0.3으로 증가시켜 해결하였다.

`early_termination` (weight = -1.0)은 비정상 종료(abnormal termination) 시 패널티를 부여한다. 비정상 종료는 로봇의 관절 속도가 임계값(100 rad/s)을 초과하거나, 물체가 작업 공간 밖으로 벗어날 때 발생한다.

**핸드 간 보상 차이 요약**

두 핸드는 **동일한 보상 구조**를 사용하되, 물체 크기 비율에 따라 position-related 파라미터만 조정된다:

| 보상 파라미터               | DG5F (1.0× object) | Inspire (0.5× object) | 비율    |
|-----------------------------|--------------------|-----------------------|---------|
| `position_tracking` $\sigma$ | 0.2 m              | 0.1 m                 | 2:1     |
| `success` $\sigma_{\text{pos}}$ | 0.1 m           | 0.05 m                | 2:1     |
| `fingers_to_object` $\sigma$ | 0.4 m             | 0.2 m                 | 2:1     |
| 모든 weight 및 기타 파라미터  | 동일               | 동일                  | 1:1     |

이러한 조정은 두 핸드가 **상대적으로 동일한 난이도**의 작업을 수행하도록 보장한다.

#### 3.3.3. 신경망 아키텍처 및 Actor-Critic 구조

우리는 **separate actor-critic architecture**를 사용한다. 이는 정책 네트워크(actor)와 가치 네트워크(critic)가 파라미터를 공유하지 않고 독립적으로 학습되는 구조이다 (rl_games_ppo_cfg.yaml: `separate: True`). Separate architecture는 actor와 critic이 서로 다른 학습 속도와 목표를 가질 때 유리하며, 특히 고차원 입력과 복잡한 접촉 동역학을 다루는 dexterous manipulation에서 안정성을 제공한다.

<그림 삽입 예정: Actor-Critic 네트워크 구조 다이어그램>

**입력 처리 및 정규화**

입력은 3개 관측 그룹(Policy + Proprioception + Perception)을 concatenate하여 단일 벡터로 만든다 (DG5F: 1870차원, Inspire: 1660차원). 이 입력은 먼저 **Running Mean/Std Normalization**을 거친다. Normalization 통계량(mean, std)은 학습 중 지속적으로 업데이트되며 (`normalize_input: True`), 입력의 분포를 zero mean, unit variance로 조정하여 신경망 학습을 안정화한다.

**Actor Network (정책)**

Actor는 3-layer MLP로 구성되며, hidden layer 크기는 [512, 256, 128]이다. 활성화 함수는 **ELU (Exponential Linear Unit)**를 사용한다 (rl_games_ppo_cfg.yaml: `activation: elu`). ELU는 ReLU와 달리 음수 입력에 대해 부드러운 기울기를 제공하여, 접촉이 사라지거나 발생하는 순간의 급격한 관측 변화에도 안정적으로 학습할 수 있다. 출력층은 행동 차원(DG5F: 26, Inspire: 12)의 Gaussian policy를 생성한다. Mean $\mu$는 선형 출력(`mu_activation: None`), standard deviation $\sigma$는 고정값으로 초기화되며 (`fixed_sigma: True`, `sigma_init: const_initializer val=0`), 이는 $\sigma = \exp(0) = 1$을 의미한다. 고정 $\sigma$는 학습 초기 안정성을 제공하지만, adaptive $\sigma$ 학습도 가능하다.

**Critic Network (가치 함수)**

Critic도 동일한 3-layer MLP 구조 [512, 256, 128]와 ELU 활성화를 사용하지만, actor와는 **독립적인 파라미터**를 갖는다. 출력은 1차원 scalar로, **state value function** $V(s)$를 추정한다. PPO는 advantage actor-critic 알고리즘으로, advantage $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \tau)^l \delta_{t+l}$를 계산할 때 state value를 사용한다 (여기서 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$는 TD error, $\tau = 0.95$는 GAE parameter). State value $V(s)$는 "현재 상태에서 정책 $\pi$를 따를 때 기대되는 누적 보상"을 나타낸다:

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t c(s_t, a_t) \mid s_0 = s \right]
$$

Critic은 TD error를 최소화하도록 학습되며 (`critic_coef: 4.0`), 이는 actor의 정책 gradient 학습과 독립적으로 진행된다. Separate architecture는 critic이 value function을 빠르게 학습하여 정확한 advantage 추정을 제공하도록 하며, 이는 actor의 정책 개선에 직접적으로 기여한다.

**네트워크 초기화 및 정규화**

모든 weight는 기본 초기화(`initializer: default`, PyTorch의 Kaiming initialization)를 사용하며, regularization은 적용하지 않는다 (`regularizer: None`). D2RL (Data-to-RL) [22] 구조는 사용하지 않는다 (`d2rl: False`). 이는 vanilla MLP가 우리의 작업에서 충분히 효과적임을 보여준다.

### 3.4. 커리큘럼 학습 및 도메인 랜덤화

강화학습의 성공적인 수렴을 위해, 우리는 **Automatic Domain Randomization (ADR)** 기반 curriculum learning과 광범위한 domain randomization을 사용한다. 이는 DexPBT [20]의 설계를 따르되, UR10e-DG5F/Inspire 시스템에 맞게 조정되었다.

#### 3.4.1. Automatic Domain Randomization (ADR)

ADR은 성공률에 따라 환경의 난이도를 자동으로 조정하는 curriculum learning 기법이다. 우리는 다음과 같은 curriculum을 적용한다: **Gravity**는 초기에 [0.0, 0.0, 0.0]에서 시작하여, 성공률이 threshold를 초과하면 점진적으로 [-9.81, -9.81, -9.81]로 증가한다 (dexsuite_env_cfg.py:414-421, variable_gravity). 이는 중력이 없는 쉬운 환경에서 시작하여, 물체를 들어올리는 전략을 먼저 학습한 후, 점진적으로 실제 중력 조건으로 전환하는 전략이다. 이러한 gravity curriculum은 별도의 "lift reward"를 필요로 하지 않으며, 보상 함수를 단순화하는 부가 효과가 있다.

**Joint friction**은 [0.0, 5.0] 범위에서 무작위화되며 (EventCfg:248-256), 성공률에 따라 범위가 조정된다. **Object mass**는 [0.2, 2.0]× 범위에서 스케일링되고 (EventCfg:269-277), **Joint stiffness/damping**은 [0.8, 1.2]× 범위에서 조정된다 (EventCfg:237-246). 이러한 curriculum은 정책이 쉬운 조건에서 먼저 기본 전략을 학습한 후, 점진적으로 어려운 물리적 조건에 적응하도록 유도한다.

#### 3.4.2. Domain Randomization

Sim-to-real transfer를 위해, 우리는 OpenAI Rubik's Cube [23]의 접근을 따라 광범위한 domain randomization을 적용한다. **Startup randomization** (에피소드 시작 시 1회 적용): Robot physics material의 static/dynamic friction은 [0.9, 1.1] 범위에서 무작위화되고, object physics material의 friction은 [0.5, 1.0] 범위에서 무작위화된다 (EventCfg:211-235). Object mass는 [0.2, 2.0]×, joint stiffness/damping은 [0.8, 1.2]× 범위에서 스케일링된다.

**Reset randomization** (매 에피소드마다 적용): Object pose는 테이블 위 영역에서 무작위로 초기화되며 (x: [-0.1, 0.1], y: [-0.2, 0.2], z: [0.0, 0.2] 상대 오프셋), roll/pitch/yaw도 [-π, π] 범위에서 무작위화된다 (EventCfg:289-313). Robot arm joints는 ±0.5 rad, hand flexion joints는 ±0.05 rad 범위에서 무작위화된다 (EventCfg:330-387). 단, hand abduction joints는 고정하여 손가락 간 충돌을 방지한다. Table pose도 ±5cm 범위에서 무작위화된다 (EventCfg:279-287).

#### 3.4.3. 추가 학습 안정화 기법

PPO 학습의 안정성을 위해 다음과 같은 기법을 추가로 사용한다. **Value normalization** (`normalize_value: True`)은 가치 함수의 출력을 정규화하여 TD error의 스케일을 안정화한다. **Advantage normalization** (`normalize_advantage: True`)은 각 minibatch 내에서 advantage를 zero mean, unit variance로 조정하여 정책 업데이트의 안정성을 향상시킨다. **Value clipping** (`clip_value: True`)은 가치 함수 업데이트도 clipping하여, 극단적인 value 변화를 방지한다. **Reward scaling** (`reward_shaper.scale_value: 0.01`)은 보상을 0.01배로 스케일링하여, 가치 함수의 크기를 제한하고 학습 안정성을 향상시킨다.

---

**References for Section 3.3 and 3.4:**

[22] Agarwal, R., et al. (2020). "An Optimistic Perspective on Offline Reinforcement Learning." *International Conference on Machine Learning (ICML)*.

[23] OpenAI et al. (2019). "Solving Rubik's Cube with a Robot Hand." *arXiv preprint arXiv:1910.07113*.


---

## 4. 실험 (Experiments)

### 4.1. 실험 환경 및 학습 설정

#### 4.1.1. 계산 환경 및 학습 처리량

우리는 Intel Core Ultra 9 285K CPU (24 코어)와 32 GB VRAM을 갖춘 Nvidia GeForce RTX 5090 GPU, 그리고 128 GB 시스템 RAM을 장착한 워크스테이션에서 실험을 수행했다. Isaac Lab 시뮬레이션 프레임워크 [16]를 사용하여 GPU에서 **4096개의 병렬 환경**을 동시에 시뮬레이션할 수 있다. GPU 기반 벡터화된 RL 구현인 rl_games [17]와 결합하여, 이 구성은 기민한 조작 작업을 위한 효율적인 분산 학습을 가능하게 한다.

RTX 5090의 높은 메모리 대역폭 (1792 GB/s)과 계산 능력 (92 TFLOPS FP32)은 수천 개의 병렬 인스턴스에서 복잡한 접촉이 많은 시나리오의 실시간 물리 시뮬레이션을 가능하게 하여, 전통적인 CPU 기반 접근 방식에 비해 학습 프로세스를 크게 가속화한다. 실험 결과, **Inspire Hand (12-DoF) 시스템은 약 3.5~4.0 ms/step의 처리 속도**를 보였으며, **DG5F Hand (26-DoF) 시스템은 약 5.8 ms/step**의 처리 속도를 기록했다. 이는 저구동 핸드가 물리 시뮬레이션 및 정책 추론 측면에서 약 **1.45배 빠른 학습 속도**를 제공함을 의미한다.

#### 4.1.2. 비교 실험 설계

공정한 비교를 위해, 우리는 다음과 같은 **통제 변수(controlled variables)**를 설정했다:

**동일 조건:**
- **로봇 팔**: UR10e (6-DoF collaborative robot)
- **RL 알고리즘**: PPO (Proximal Policy Optimization)
- **네트워크 구조**: 3-layer MLP [512, 256, 128], ELU activation
- **학습 하이퍼파라미터**: learning rate, batch size, horizon length, mini-epochs (모두 동일)
- **작업 (Task)**: Lift task (물체를 목표 위치로 들어올리기)
- **보상 함수 구조**: 동일한 reward terms 및 weights
- **도메인 랜덤화**: 동일한 physics randomization 범위
- **학습 기간**: 7500 epochs (약 11억 timesteps)

**차이점 (구동 구조에 따른 필수 조정):**
- **핸드**: DG5F (20-DoF fully-actuated) vs Inspire RH56F1 (6-DoF underactuated)
- **행동 공간**: 26차원 vs 12차원
- **관측 공간**: 1870차원 vs 1660차원 (DoF 차이로 인한 자연스러운 결과)
- **물체 크기**: 1.0× (DG5F) vs 0.5× (Inspire) - 핸드 크기에 정규화
- **보상 파라미터**: position_tracking.std, success.pos_std를 물체 크기에 비례하여 조정

#### 4.1.3. 학습 프로토콜

우리는 다음과 같은 실험 데이터를 분석에 사용한다:

**UR10e + DG5F Right Hand:**
- 실험 일시: 2025-11-20, 23:27 시작
- 학습 epochs: 7500 (max)
- 총 timesteps: ~1,105,625,088 (약 11억 steps)
- 학습 시간: 약 12시간
- 환경: 4096 parallel environments

**UR10e + Inspire Right Hand:**
- 실험 일시: 2025-12-16, 02:19 시작
- 학습 epochs: 7500 (max)
- 총 timesteps: ~1,105,477,632 (약 11억 steps)
- 학습 시간: 약 8~9시간 (DG5F 대비 25% 단축)
- 환경: 4096 parallel environments

두 시스템 모두 동일한 총 timesteps를 수행했으나, Inspire Hand는 낮은 DoF로 인해 물리 시뮬레이션 및 정책 추론 속도가 빨라 **벽시계 시간(wall-clock time)이 약 25% 단축**되었다. 이는 저구동 핸드의 실용적 장점 중 하나이다.

---

### 4.2. 주요 발견들

#### 4.2.1. 조작 품질 비교

<fig추가>
우리의 시뮬레이션 환경에서는 테이블 색상으로 성공/실패를 시각적으로 표시한다. 초록색 테이블은 에피소드가 성공적으로 완료됨을 나타내며 (물체가 목표 위치에 도달하고 충분한 시간 동안 유지됨), 빨간색 테이블은 실패를 나타낸다 (목표 미달성, 물체 낙하, 또는 비정상 종료).

fig1, fig2에 tensorboard 추가, position _tracking, 과 같은 것들 (아마 1_tensor, 2_tensor)
7500 epoch 학습 후, DG5F 시스템은 다양한 물체 형태(cuboid, sphere, capsule, cone)에 대해 안정적인 성공을 보였다. 시각적 시연에서 테이블은 대부분 초록색으로 나타났으며, 이는 손을 물체 근처로 이동(approaching), 손가락을 물체 형태에 맞게 배치(pre-grasp shaping), 5개 손가락으로 물체를 감싸며 쥐기(enveloping grasp), 테이블에서 들어올리기(lifting), 목표 위치로 이동 후 유지(positioning)하는 일련의 조작 전략을 학습했음을 보여준다. Fig. 1과 Fig. 2를 참조하면, DG5F의 success reward는 7500 epoch 시점에 3.49에 도달했으며, position tracking은 0.89, good finger contact은 0.27을 기록했다. Ground contact penalty는 -0.04로 거의 없었으며 (물체를 성공적으로 들어올림), object out of bound도 0.02로 물체 이탈이 거의 발생하지 않았다.

반면, Inspire Hand 시스템은 7500 epoch 학습 후에도 성공적인 물체 조작을 학습하지 못했다. 시각적 시연에서 테이블은 대부분 빨간색으로 나타났으며, 손을 물체 근처로 이동하는 것은 성공적이었으나, 손가락으로 물체를 터치하려 시도하고 물체를 테이블 위에 누르는 듯한 동작을 반복할 뿐, 물체를 들어올리지 못하고 테이블에 방치한 채 시간 초과로 에피소드가 종료되는 패턴이 관찰되었다. 흥미롭게도, Fig. 1과 Fig. 2의 Tensorboard 메트릭을 살펴보면 일부 측면에서 Inspire가 DG5F보다 우수한 성능을 보임을 알 수 있다. Success reward는 0.00으로 실패했으나, good finger contact은 0.53으로 DG5F (0.27) 대비 약 2배 높았으며, fingers to object distance도 0.52로 DG5F보다 우수했다. 또한 object out of bound는 0.01로 DG5F (0.02)보다 낮아, 물체를 환경 밖으로 밀어내는 실수가 적었다. 그러나 position tracking은 0.55로 DG5F (0.89)에 크게 미치지 못했으며, ground contact penalty는 -0.01로 약간 발생했다 (테이블 접촉).

이러한 결과는 Inspire Hand가 손가락을 물체에 가까이 가져가고 접촉을 형성하는 것은 DG5F보다 더 잘 학습했지만, 물체를 테이블에서 들어올리는 전략(lifting strategy)과 물체를 목표 위치로 이동시키는 방법(goal-directed manipulation)은 전혀 학습하지 못했음을 시사한다. 우리는 이러한 현상에 대해 두 가지 가설을 제시한다.

첫 번째 가설은 저구동 구조의 표현력 제약이다. Inspire Hand의 6-DoF 저구동 구조는 기계적 결합(mechanical coupling)으로 인해 제한된 손가락 자세만 생성할 수 있으며, 특히 individual finger control이 불가능하여 precision grasp (특정 손가락만 사용하여 물체의 특정 부분을 잡기), finger repositioning during grasp (grasp 중 개별 손가락 위치 조정), adaptive force distribution (손가락마다 다른 힘으로 쥐기) 등의 manipulation 전략을 실행하기 어렵다. 예를 들어, 사각형 물체를 잡을 때 DG5F는 마지막 링크(fingertip link)만 살짝 움직여 물체를 쥘 수 있지만, Inspire는 모든 관절이 연동되어 움직이므로 (underactuated coupling) 이러한 섬세한 조작이 불가능하다. 이는 질적 분석에서도 확인되었다.

두 번째 가설은 보상 신호의 모호성이다. Inspire는 good_finger_contact reward를 높게 받지만 (0.53), success reward는 전혀 받지 못한다 (0.00). 이는 정책이 "물체와 접촉만 유지하면 된다"는 local optimum에 빠졌을 가능성을 시사한다. Contact reward는 즉각적으로 얻을 수 있지만, position tracking reward는 물체를 들어올린 후에야 증가한다. Inspire는 전자를 최대화하는 전략을 학습했지만, 후자로 전환하는 exploration을 하지 못한 것으로 보인다.

추가적으로, hardware actuation 구조의 차이에 따른 비교를 시뮬레이션 환경에서 진행하는 것은 여러 측면에서 까다로울 수 있다. 물리 시뮬레이션은 실제 하드웨어의 복잡한 동역학(compliance, backlash, friction 등)을 완벽히 재현하지 못하며, 특히 저구동 핸드의 mechanical coupling과 같은 수동적 메커니즘은 시뮬레이션 정확도에 민감하게 반응할 수 있다. 이러한 시뮬레이션 한계는 Section 5에서 더 자세히 논의한다.

---

#### 4.2.2. 학습 동역학 분석

Tensorboard 로그 분석을 통해, 우리는 두 핸드의 학습 과정에서 흥미로운 차이를 발견했다. 이 섹션에서는 PPO 알고리즘의 핵심 메트릭들을 중심으로 학습 동역학을 분석한다.

Fig. 1과 Fig. 2의 success reward 메트릭을 살펴보면, 두 핸드의 학습 성과 차이가 극명히 드러난다. DG5F는 약 2000 epoch부터 급격히 상승하기 시작하여 7500 epoch 시점에 3.49에 도달했다. 이는 exponential reward shaping (Section 3.3.2)에 따라 물체와 목표 위치 간 거리가 매우 작아졌음을 의미한다 (recall: $r_{\text{success}} = \exp(-\|p_{\text{obj}} - p_{\text{goal}}\|^2 / \sigma_{\text{pos}}^2)$). 반면 Inspire는 전체 학습 기간 동안 success reward가 0.0 근처에 정체되어 있으며, 이는 목표 위치 도달에 실패했음을 나타낸다. Fig. 5의 total shaped reward (모든 reward term의 합)를 보면 DG5F와 Inspire 모두 약 15 근처까지 상승했으나, 그 구성이 전혀 다르다. DG5F는 success (3.49)와 position_tracking (0.89)이 주요 기여를 하는 반면, Inspire는 good_finger_contact (0.53)과 fingers_to_object (0.52)로 보상을 얻었을 뿐 작업 목표(lift & position)는 달성하지 못했다. 이는 Inspire가 "물체와 접촉 유지" 전략으로 비슷한 총 보상을 얻었지만, 실제 manipulation은 학습하지 못했음을 보여준다.

Fig. 2의 episode length와 termination 관련 메트릭을 참조하면, 두 핸드 모두 평균 episode length가 약 231 steps (horizon length 36 × 약 6.4 episodes)로 유사하며, time_out rate (정상 종료 비율)도 DG5F 0.92, Inspire 0.93으로 거의 동일하다. 이는 두 시스템 모두 비정상 종료(abnormal termination)가 거의 발생하지 않았으며, 학습이 안정적으로 진행되었음을 의미한다. Early termination penalty가 거의 발생하지 않은 것도 이를 뒷받침한다.

Fig. 3의 PPO 학습 안정성 메트릭을 살펴보면, KL divergence (새 정책과 이전 정책 간 차이)는 두 핸드 모두 0.012 근처에서 안정적으로 유지되었다. 이는 PPO의 trust region constraint (Section 3.3.1, KL threshold = 0.01)가 잘 작동하여 정책 업데이트가 과도하게 크지 않았음을 보여준다. Adaptive learning rate schedule이 이를 자동으로 조정했다. 흥미롭게도, Fig. 3의 entropy (정책의 무작위성 정도) 메트릭은 두 핸드 간 명확한 차이를 보였다. DG5F는 약 38.4의 entropy를 유지한 반면, Inspire는 약 22.0으로 낮았다. Entropy는 정책 분포의 불확실성을 측정하며 ($H(\pi) = -\mathbb{E}_{a \sim \pi} [\log \pi(a|s)]$), 높은 entropy는 정책이 더 다양한 행동을 탐색함을 의미한다. DG5F의 높은 entropy는 26차원 행동 공간에서 다양한 손가락 자세를 탐색했음을 시사하며, 이는 복잡한 manipulation 전략 학습에 기여했을 가능성이 있다. 반면 Inspire의 낮은 entropy는 12차원 행동 공간과 mechanical coupling으로 인해 탐색 공간이 암묵적으로 제약되었음을 나타낸다. PPO에서 entropy는 entropy bonus ($\beta \cdot H(\pi)$, Section 3.3.1)를 통해 명시적으로 보상되지만 (rl_games_ppo_cfg.yaml: `entropy_coef: 0.001`), Inspire는 이러한 보상에도 불구하고 낮은 entropy를 유지했으며, 이는 구조적 제약이 알고리즘 파라미터보다 더 강한 영향을 미쳤음을 시사한다.

Fig. 3의 critic loss (value function의 TD error)를 보면, DG5F는 약 0.13, Inspire는 약 0.07을 기록했다. Critic loss는 $L_{\text{critic}} = \mathbb{E}[(V_\theta(s) - V^{\text{target}})^2]$로 정의되며, 가치 함수 추정의 정확도를 나타낸다. DG5F의 더 높은 critic loss는 환경이 더 복잡하여 value function 학습이 어려웠음을 시사한다. 이는 두 가지로 해석될 수 있다. 첫째, DG5F는 성공적인 manipulation을 학습했으므로 다양한 상태 (물체를 들어올림, 목표로 이동 중, 목표 유지 등)에서 value function을 정확히 추정해야 하는 반면, Inspire는 제한된 전략(접촉만 유지)을 학습했으므로 value function이 단순하다. 둘째, DG5F의 높은 entropy (탐색성)는 더 다양한 상태를 방문함을 의미하며, 이는 value function 학습을 어렵게 만든다.

Fig. 4의 학습 시간 및 계산 효율성 메트릭을 참조하면, step inference time (한 step의 물리 시뮬레이션 + 정책 추론 시간)은 두 핸드 간 명확한 차이를 보였다. DG5F는 약 5.8 ms/step, Inspire는 약 4.0 ms/step을 기록하여 Inspire가 약 1.45배 빠른 처리 속도를 보였다. 이는 저구동 핸드의 실용적 장점을 보여준다. Inspire는 관절 수가 적어 (20 vs 6 actuators) 물리 시뮬레이션 계산량이 적으며, 행동 공간이 작아 (26 vs 12) 정책 네트워크 추론 시간도 짧다. 4096개 병렬 환경에서 7500 epochs를 학습할 때, 이 차이는 약 3~4시간의 벽시계 시간 절약으로 나타났다 (DG5F: ~12시간, Inspire: ~8~9시간). 그러나 이러한 계산 효율성이 학습 성능으로 이어지지 않았다는 점이 중요하다. Inspire는 더 빠르게 학습했지만 잘못된 전략(local optimum)에 수렴했으며, 이는 "빠른 학습 ≠ 좋은 학습"임을 보여준다.

Fig. 3의 learning rate 관련 메트릭을 보면, PPO의 adaptive learning rate schedule이 두 시스템에서 모두 잘 작동했음을 알 수 있다. KL divergence가 threshold (0.01)를 초과하면 learning rate를 감소시키고, 너무 낮으면 증가시키는 메커니즘 (Section 3.3.1)이 자동으로 조정되었다. Tensorboard 로그에서 `info/last_lr`과 `info/lr_mul`을 확인한 결과, 두 핸드 모두 learning rate가 초기값 1e-3에서 시작하여 학습 중반부에 자동 조정되었으며, 최종적으로 비슷한 learning rate (약 2e-4)로 수렴했다.

이러한 분석 결과를 종합하면, 저구동 핸드는 계산적으로 효율적이지만 표현력 제약으로 인해 복잡한 manipulation 전략 학습에 실패할 수 있으며, 반면 완전 구동 핸드는 계산 비용이 높지만 높은 탐색성(entropy)과 표현력을 통해 성공적인 학습을 달성했음을 알 수 있다. 특히 DG5F는 성공적인 조작을 학습한 반면 Inspire는 실패했으며, Inspire는 contact rewards로 높은 총 보상을 얻었으나 작업 목표는 미달성했다. DG5F가 Inspire 대비 1.75배 높은 entropy (38 vs 22)로 더 다양한 탐색을 수행했으며, DG5F의 더 높은 critic loss는 복잡한 value function 학습을 반영한다. Inspire는 1.45배 빠른 학습 속도를 보였으나 (4.0 ms/step vs 5.8 ms/step), 두 핸드 모두 안정적인 KL divergence를 유지하며 비정상 종료가 거의 없었다.

---

#### 4.2.3. 커리큘럼 학습 수렴 분석

Fig. 1의 curriculum/adr 메트릭을 참조하면, Automatic Domain Randomization (ADR) 기반 커리큘럼 학습의 수렴 패턴이 두 핸드 간 명확한 차이를 보임을 알 수 있다. DG5F는 학습 초기 (epochs 0-2000)에 빠른 상승을 보이며 성공률이 급격히 증가했고, 중반부 (epochs 2000-5000)에는 꾸준한 진행을 보였으며, 후반부 (epochs 5000-7500)에는 난이도 약 0.85에서 수렴하여 평탄해지면서 안정적인 성능을 유지했다. 이는 ADR이 DG5F의 성공률에 따라 환경 난이도를 자동으로 조정했으며, 최종적으로 높은 난이도 수준에서 안정적으로 수렴했음을 의미한다.

반면 Inspire는 전체 학습 기간 동안 낮은 난이도 수준에 머물렀으며, 최종 시점에서도 약 0.6 수준에 도달하는 데 그쳤다. 이는 Inspire가 성공적인 manipulation을 학습하지 못했기 때문에 ADR이 난이도를 증가시키지 않았음을 나타낸다. Fig. 3의 entropy 메트릭을 함께 고려하면 (Inspire: 22.0), Inspire는 여전히 상대적으로 높은 불확실성을 보이며 탐색 단계에 머물러 있음을 알 수 있다 (DG5F는 38.4의 entropy를 유지하면서도 성공적으로 수렴).

이러한 커리큘럼 수렴 패턴의 차이는 두 핸드의 학습 능력 격차를 정량적으로 보여준다. DG5F는 높은 난이도 (0.85)에서도 안정적인 성능을 유지하는 반면, Inspire는 낮은 난이도 (0.6)에서조차 작업을 완수하지 못했다. 이는 약 0.25 포인트의 난이도 격차를 의미하며, 이를 physics randomization의 관점에서 해석하면 Inspire는 DG5F 대비 약 30% 덜 robust한 정책을 학습했다고 볼 수 있다. ADR의 자동 조정 메커니즘은 각 핸드의 학습 능력에 맞춰 적절한 난이도를 제공했지만, Inspire의 구조적 제약은 최종적으로 낮은 수렴 수준으로 이어졌다.

또한, Fig. 1의 ADR 메트릭 추이를 시간에 따라 분석하면, DG5F는 약 3000 epoch 이후 난이도가 급격히 상승하여 최대 수준에 근접했으며, 이는 성공률 증가와 정확히 일치한다. 반면 Inspire는 전체 학습 기간 동안 완만한 증가만을 보였으며, 7500 epoch 이후에도 여전히 상승 여지가 남아있는 것으로 보인다. 그러나 success reward가 0.0에 정체되어 있는 점을 고려하면, 추가 학습을 진행하더라도 ADR 난이도가 크게 증가하지 않을 가능성이 높다. 이는 Inspire가 기본적인 manipulation 전략을 학습하지 못했기 때문에, 난이도 증가가 의미 없다는 ADR의 판단을 반영한다.

---

#### 4.2.4. 물체별 조작 성능 분석

**[TODO: 물체 크기 및 형태에 따른 DG5F/Inspire 성능 차이 분석]**

**분석 내용 (작성 예정):**
- Cuboid (사각형): DG5F vs Inspire 성능 비교
- Sphere (구형): DG5F vs Inspire 성능 비교
- Capsule (캡슐): DG5F vs Inspire 성능 비교
- Cone (원뿔): DG5F vs Inspire 성능 비교
- 물체 크기별 (small/medium/large) 성공률 분석
- DG5F가 특히 잘하는 물체 형태 vs 못하는 물체 형태
- Inspire의 물체별 성능 패턴
- 물체 형태가 grasp strategy에 미치는 영향

**예상 관찰 사항:**
- DG5F는 precision grasp가 필요한 작은 물체나 불규칙한 형태에서 우수할 것으로 예상
- Inspire는 단순한 형태 (sphere, large cuboid)에서 상대적으로 나은 성능을 보일 가능성
- Underactuated hand의 adaptive synergy가 특정 물체 형태에 유리할 수 있음
- 물체 크기가 작을수록 (Inspire의 경우 0.5× scaling) precision 요구사항이 증가하여 성능 차이 확대 가능


----------------------------------------------------------------------------------------------------
5. Conclusion

Discussion
왜 그런것 같은지도 분석 (논문) :


limitation 

1. 6자유도, 7자유도 로봇과 hand 일반화 - dg5f, inspire만 사용했었음

2. 수학적 hand actuation 차이를 분석하지 못했다.
3. 학습 시간이 그렇게 길지 않았다. :학습을 7500 epochs 밖에 진행하지 않음 (다만, 시뮬레이션 max 7500 epochs - 우리 환경에서 12시간 정도 학습시간) 
4. realworld에서 실험을 못해봤다 - sim2real
시뮬레이션 → 물체별 정량 분석 불가능 / 질적 분석은 가능
    - 질적 분석 : 실제 로봇의 pick&place의 경우 사각형과 같은 물체의 경우엔 dg5f는 손의 마지막 link만 살짝 움직이면 쥘 수 있으나 inspire의 경우엔 그게 불가능 - underactuated ; 모두 articulatioed 연동이 되어있으니까 - 이런 이슈도 있었음 
5. 학습 알고리즘 : ppo만을 택했는데 다른 알고리즘 (SAC, Off-policy)를 택한다면 더 잘될수도 있을 듯

추가적 - 6까진 아니지만 언급 정도 물체 무게 똑같이 함. - mass randomization도 / 밀도에 따라 - hand 별 effort가 다르기에 좀 까다롭긴함

----------------------------------------------------------------------------------------------------
references
