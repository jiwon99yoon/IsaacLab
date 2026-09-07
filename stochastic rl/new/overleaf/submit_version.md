%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2345678901234567890123456789012345678901234567890123456789012345678901234567890
%        1         2         3         4         5         6         7         8

\documentclass[10pt]{article}       
\usepackage{geometry}               
%\geometry{letterpaper}             

\geometry{letterpaper, margin=1in}

% The following packages can be found on http:\\www.ctan.org
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
%\usepackage{epsfig} % for postscript graphics files
%\usepackage{mathptmx} % assumes new font selection scheme installed
%\usepackage{times} % assumes new font selection scheme installed
\usepackage{amsmath} % assumes amsmath package installed
\usepackage{amssymb}  % assumes amsmath package installed
\usepackage{amsthm}  % assumes amsmath package installed
\usepackage{bm}
\usepackage{lipsum}
%\usepackage[linesnumbered, ruled]{algorithm2e}'
\usepackage{color}
\usepackage{enumitem}
\usepackage{cite}
\usepackage{wrapfig}
\usepackage{float}

\newtheorem{proposition}{Proposition}
\newtheorem{definition}{Definition}
\newtheorem{corollary}{Corollary}
\newtheorem{lemma}{Lemma}
\newtheorem{theorem}{Theorem}
\newtheorem{remark}{Remark}



\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\cart}{\times}

\title{A Comparative Study of Underactuated and Fully Actuated Dexterous Hands in Reinforcement Learning-Based Manipulation}

\author{
Jiwon Yoon\thanks{Department of
Transdisciplinary Studies, Graduate School of Convergence Science and
Technology, Seoul National University({yabc0908@snu.ac.kr}). }
}


\date{}

\begin{document}
\maketitle

\begin{abstract}
Dexterous manipulation with reinforcement learning (RL) has primarily focused on fully-actuated multi-fingered hands, leaving unexplored whether underactuated designs—successful in traditional control—offer advantages in learning-based paradigms. This study presents a systematic comparison between a 20-DoF fully-actuated hand (Tesollo DG5F) and a 6-DoF underactuated hand (Inspire RH56F1) on an object lifting task, isolating actuation structure as the sole experimental variable under identical training conditions (Proximal Policy Optimization, 7500 epochs, 1.1 billion timesteps, curriculum learning with Automatic Domain Randomization).

Results reveal a dramatic performance gap: DG5F successfully learned manipulation (total reward 18--19, success reward $\sim$3.3), while RH56F1 completely failed (total reward $\sim$2.0, zero success) despite superior performance on approach behavior (finger-object distance 0.67 vs 0.52). Analysis identifies a critical \textit{local optimum trap}—RH56F1 maximized dense rewards without discovering grasping strategies, exhibiting 47\% lower policy entropy and zero curriculum progression. This failure stems from three compounding factors: reduced exploration from mechanical coupling, insufficient expressiveness for fine-grained control, and inability to escape reward shaping traps.

These findings suggest underactuated hands face fundamental challenges in RL-based learning for tasks requiring precise manipulation beyond adaptive grasping, highlighting the critical role of actuation structure in determining learning dynamics and emergent behaviors.
\end{abstract}

\section{Introduction}

Dexterous manipulation remains one of the most challenging problems in robotic control due to high-dimensional action spaces, complex contact dynamics, and intricate coordination requirements among multiple fingers \cite{yu2022dexterous}. Recent advances in reinforcement learning (RL) have demonstrated impressive capabilities in learning complex manipulation skills when combined with large-scale GPU-accelerated simulation, curriculum learning, and carefully designed reward structures \cite{rajeswaran2018dexterous, andrychowicz2020learning}. Notable achievements include in-hand object rotation \cite{openai2019rubiks}, robust grasping across diverse object geometries \cite{petrenko2023dexpbt}, and sim-to-real transfer for dexterous tasks \cite{qi2023dexpoint}. These successes have primarily been achieved using fully-actuated multi-fingered hands—systems where each joint is independently controlled, providing maximum expressiveness at the cost of high-dimensional action spaces. Platforms such as the Shadow Hand, Allegro Hand \cite{allegro_hand}, and Tesollo DG5F \cite{tesollo_dg5f} have become standards in RL research due to their ability to execute fine-grained finger motions required for complex manipulation primitives.

Despite this progress, most existing studies focus on algorithmic improvements—such as more sample-efficient RL methods \cite{schulman2017ppo, lopes2025lookahead}, better reward engineering, or advanced sim-to-real transfer strategies \cite{qi2023dexpoint}—while treating the hand's mechanical and actuation structure as a fixed design choice. Comparatively little attention has been paid to how the degree of actuation and mechanical coupling affect the learning process itself. In particular, the fundamental trade-off between fully-actuated hands (high DoF, independent joint control) and underactuated hands (coupled actuation via passive mechanical linkages) has not been systematically investigated in the context of RL-based manipulation.

From a hardware perspective, underactuated hands have been extensively studied as mechanically simpler and more robust alternatives to fully-actuated systems \cite{deimel2013novel, birglen2009underactuated}. By coupling multiple finger joints through passive mechanisms such as tendons, springs, or compliant structures, underactuated designs reduce the number of control inputs while enabling adaptive grasping behaviors that naturally conform to object shapes \cite{catalano2014softhand, dollar2011underactuated}. These properties have proven successful in traditional control settings for industrial grasping and prosthetic applications. However, whether the mechanical simplicity that benefits model-based control translates to advantages in learning-based paradigms remains an open question. Underactuation inherently restricts the hand's expressiveness by constraining finger motions to coordinated synergies rather than independent articulation, potentially affecting exploration efficiency, policy expressiveness, and the ability to discover complex manipulation strategies.

To address this gap, this work conducts a systematic comparison between a fully-actuated 5-finger hand (Tesollo DG5F, 20 DoF) and an underactuated 5-finger hand (Inspire RH56F1, 6 DoF) under rigorously controlled experimental conditions. Both systems are trained using identical RL algorithms (Proximal Policy Optimization \cite{schulman2017ppo}), reward functions, neural network architectures, training durations (7500 epochs), and curriculum learning protocols (Automatic Domain Randomization). The task is object lifting—a canonical manipulation primitive requiring coordinated approach, grasping, lifting, and positioning behaviors. By isolating the actuation structure as the sole experimental variable, this study illuminates how hardware design choices influence learning dynamics, task performance, and emergent manipulation strategies in RL-based dexterous manipulation.

\section{Background and Related Work}

This section provides the necessary background on reinforcement learning and reviews relevant prior work on dexterous manipulation, hand design, and the intersection of hardware structure and learning-based control.

\subsection{Proximal Policy Optimization (PPO)}

The experiments presented in this work employ \textbf{Proximal Policy Optimization (PPO)} \cite{schulman2017ppo}, a widely-used policy gradient algorithm proven effective for high-dimensional continuous control tasks, including dexterous manipulation \cite{rajeswaran2018dexterous, petrenko2023dexpbt}. The robot control problem is formalized as a Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $\mathcal{P}$ is the transition probability, $R$ is the reward function, and $\gamma \in (0,1)$ is the discount factor. The goal is to maximize expected cumulative reward $\eta(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=1}^{\infty} \gamma^{t-1} R(s_t, a_t) \right]$.

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

PPO is particularly well-suited for dexterous manipulation tasks. The clipped objective and bounded policy updates prevent catastrophic policy collapse, which is critical in contact-rich scenarios where small policy changes can lead to drastically different contact patterns. PPO reuses data multiple times via mini-batch updates over multiple epochs, improving sample efficiency compared to on-policy methods like REINFORCE—crucial in simulation-based learning where environment interaction is parallelized. PPO scales well to high-dimensional continuous action spaces (e.g., 26-DoF for DG5F) using stochastic gradient descent without requiring Hessian computation, and is less sensitive to hyperparameter choices compared to TRPO or vanilla policy gradients, facilitating deployment across different hand configurations without extensive tuning. In this work, PPO is used as implemented in RL-Games \cite{rlgames}, a GPU-accelerated framework that enables massively parallel training across 4,096 environments.

\subsection{Dexterous Manipulation with Reinforcement Learning}

Reinforcement learning has enabled significant progress in learning complex dexterous manipulation skills that are difficult to program using traditional control methods. \textbf{Rajeswaran et al.} \cite{rajeswaran2018dexterous} demonstrated that high-DoF dexterous manipulation (e.g., Shadow Hand with 24-DoF) can be learned using deep RL combined with human demonstrations, revealing that high-DoF hands create severe exploration challenges due to large action spaces, sparse rewards, and sample inefficiency. This motivates a key hypothesis that underactuated hands may reduce exploration difficulty by constraining the action space through mechanical coupling.

\textbf{DexPBT} \cite{petrenko2023dexpbt} introduced a large-scale RL framework for hand-arm systems using Population Based Training (PBT), achieving complex manipulation through massive parallelization (16,384 environments) and Automatic Domain Randomization (ADR) curriculum learning. This work adopts similar scale (4,096 parallel environments) and curriculum strategies, extending DexPBT by systematically comparing different hand actuation structures under identical training conditions. \textbf{DexPoint} \cite{qi2023dexpoint} further demonstrated that point cloud-based observations improve generalization across diverse object shapes compared to proprioceptive-only inputs; point cloud perception (192-point object point clouds with 5-step history) is incorporated following this best practice.

\textbf{OpenAI's Rubik's Cube work} \cite{openai2019rubiks} showcased the power of domain randomization in bridging the sim-to-real gap by randomizing object properties (mass, friction, size), robot dynamics (joint stiffness, damping), and environmental conditions (gravity, observation noise). The implementation in this work employs extensive domain randomization across object mass (0.2--2.0 kg), friction (0.5--1.0), and robot dynamics, ensuring learned policies are robust to real-world uncertainties.

\subsection{Underactuated vs Fully-Actuated Hand Designs}

Robotic hand design has long explored the trade-off between mechanical simplicity and control expressiveness. \textbf{Catalano et al.} \cite{catalano2014softhand} demonstrated that underactuated soft hands with adaptive synergies can achieve robust grasps through passive compliance and mechanical coupling, mimicking human hand synergies while reducing control complexity—however, their focus was on mechanical design and classical control, not on how these structures affect reinforcement learning dynamics. \textbf{Sintov et al.} \cite{sintov2019underactuated} showed that underactuated hands are difficult to model analytically due to contact nonlinearities and coupling, making data-driven methods a natural fit, but did not compare learning performance against fully-actuated alternatives under controlled conditions.

\textbf{Lopes et al.} \cite{lopes2025lookahead} conducted a comparative analysis of underactuated vs fully-actuated hand designs, identifying a critical gap: most studies focus on a single design, and when comparisons are made, multiple variables change simultaneously (different arms, tasks, algorithms), making it impossible to isolate the effect of actuation structure. While their work focused on classical control methods, this work extends the comparative approach to reinforcement learning, where exploration, sample efficiency, and emergent strategies introduce new dimensions previously unexplored.

From an RL perspective, it is hypothesized that underactuated synergies may act as an inductive bias in the action space, constraining exploration to synergistic motions that facilitate stable grasps, while fully-actuated hands must discover these synergies through learning—yet no prior work has tested this hypothesis under rigorous experimental control.

\section{Method}

\subsection{Problem Definition and Comparative Robot Hands}

This study aims to systematically compare and analyze the learning performance differences between \textbf{underactuated hands} and \textbf{fully-actuated hands} in reinforcement learning-based dexterous manipulation tasks. To this end, two commercial robot hands that mimic the five-finger structure of the human hand were selected, each with distinct actuation mechanisms.

\subsubsection{5-Finger Hand Selection Rationale}

Existing dexterous manipulation research has predominantly used 4-finger hands such as the Allegro Hand \cite{allegro_hand} as benchmark. However, a \textbf{5-finger structure} was adopted for the following reasons:

First, \textbf{compatibility with humanoid robots}: Recent commercial humanoid robots such as Tesla Optimus, Figure 01, and 1X NEO have adopted 5-finger hands, and the 5-finger structure provides a more natural interface for human-robot interaction and tool use. Second, \textbf{workspace symmetry}: The 5-finger structure including the thumb can form a more balanced contact distribution when grasping objects with enveloping or precision grasps. Third, \textbf{practical industrial applications}: While the Allegro Hand is primarily limited to research applications, the two hands selected in this study (DG5F and Inspire RH56F1) are actually being deployed in industrial sites and service robot fields, offering high practical utility \cite{tesollo_dg5f, inspire_rh56f1}.

Therefore, two hands were selected that \textbf{maintain the same 5-finger structure but differ only in actuation mechanism} to analyze the impact of actuation architecture on reinforcement learning performance in a controlled environment. Figure \ref{fig:hand_comparison} shows the two robotic hands used in this study, visualized in the Isaac Sim environment.

\begin{figure}
    \centering
    \includegraphics[width=0.5\linewidth]{RH56F1_R(Left)vsDG5F(Right).png}
    \caption{Comparison of dexterous hand designs used in this study, visualized in the Isaac Sim \\
    \textbf{Left}: Inspire RH56F1 hand (6-DoF, underactuated).
    \textbf{Right}: Tesollo DG5F hand (20-DoF, fully actuated).}
    \label{fig:hand_comparison}
\end{figure}

\subsubsection{DG5F: Fully-Actuated Dexterous Hand}

As shown in Figure \ref{fig:hand_comparison} (right), the \textbf{DG5F (Delto Gripper-5 Finger)} is a fully-actuated 5-finger robotic hand developed by tesollo in South Korea \cite{tesollo_dg5f}. The hand has \textbf{20 degrees of freedom (20-DoF)} with all joints independently controlled by separate motors. The hand is approximately 20 cm in length with a payload capacity of 2.5--5 kg for pinching and 10--20 kg for enveloping grasps. Analysis of the DG5F URDF file reveals 20 \texttt{revolute} joints, each capable of independent torque control. When combined with the UR10e 6-DoF arm, this creates a \textbf{26-dimensional continuous action space (6 arm + 20 hand)}, which presents significant exploration challenges in reinforcement learning due to the high-dimensional policy space.

\subsubsection{Inspire RH56F1: Underactuated Dexterous Hand}

In contrast, as shown in Figure \ref{fig:hand_comparison} (left), the \textbf{Inspire RH56F1} is an underactuated 5-finger robotic hand developed by Inspire Robots in China \cite{inspire_rh56f1}. The hand uses \textbf{6 linear servo actuators} to drive \textbf{12 joints} through mechanical coupling, where each actuator controls two joints simultaneously. The hand is approximately 12--15 cm in length, about 60--75\% of the DG5F size, with a fingertip gripping force of 15 N. Analysis of the Inspire RH56F1 URDF file shows 12 \texttt{revolute} joints, but only 6 are independently actuated \cite{birglen2009underactuated}. When combined with the UR10e arm, this creates a \textbf{12-dimensional continuous action space (6 arm + 6 hand)}, exactly half the dimensionality of the DG5F system. This underactuated design incorporates adaptive synergies through mechanical coupling, mimicking the synergistic motion patterns of the human hand \cite{catalano2014softhand}, and provides passive compliance that allows the hand to adapt to object shapes during contact.

\subsubsection{Underactuated vs Fully-Actuated: Fundamental Differences}

As can be seen from Figure \ref{fig:hand_comparison}, the fundamental difference between the two hands lies in the relationship between actuated and joint DoF \cite{dollar2011underactuated}: the DG5F (right) has 20 independently actuated joints (actuated DoF = joint DoF), enabling arbitrary finger configurations but resulting in a 26-dimensional action space. The Inspire (left) has only 6 actuators controlling 12 coupled joints (actuated DoF $<$ joint DoF), constraining finger configurations to synergistic patterns but reducing the action space to 12 dimensions.

These architectural differences create three key trade-offs from a reinforcement learning perspective. First, \textbf{exploration difficulty}: the DG5F faces a high-dimensional exploration problem with low contact probability during random exploration, while the Inspire's mechanical constraints provide implicit regularization that may facilitate contact-rich state discovery. Second, \textbf{expressiveness}: the DG5F can learn complex manipulation strategies including finger walking and precision grasps, whereas the Inspire is limited to synergistic motions primarily suited for power grasps. Third, \textbf{sample efficiency}: the DG5F's high dimensionality may require extensive exploration to discover effective policies, while the Inspire's reduced action space may enable faster convergence.

This study systematically analyzes how these trade-offs manifest in practice under \textbf{identical task, algorithm (PPO), and robot arm (UR10e)} conditions.

\subsection{Experimental Environment: Isaac Lab-Based Dexsuite Benchmark}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.5\linewidth]{iiwa_allegro_dexsuite.png}
    \caption{Kuka iiwa7 + allegro hand, visualized in the IsaacLab Dexsuite task. This represents the original Dexsuite benchmark configuration.}
    \label{fig:iiwa_allegro}
\end{figure}

\subsubsection{Isaac Lab Simulation Framework}

The experiments are conducted using the NVIDIA Isaac Lab simulation environment \cite{isaaclab}, a GPU-based parallel physics simulation engine that enables high-speed training infrastructure capable of simulating thousands of environments simultaneously. A single NVIDIA GeForce RTX 5090 GPU (32 GB VRAM) is utilized to simulate \textbf{4,096 parallel environments}, achieving training speeds hundreds of times faster than traditional CPU-based simulation. Combined with RL-Games \cite{rlgames}, a GPU-accelerated vectorized reinforcement learning implementation, complex contact-rich manipulation scenarios can be learned in real-time.

\subsubsection{Dexsuite Benchmark and Prior Work}

The experimental environment is based on the Dexsuite benchmark proposed in DexPBT \cite{petrenko2023dexpbt}, a large-scale reinforcement learning framework for hand-arm systems that demonstrated the ability to learn complex dexterous manipulation behaviors through Population Based Training (PBT) and massive parallelization (16,384 environments). Additionally, DexPoint \cite{qi2023dexpoint} showed that point cloud-based perception improves generalization performance across diverse object shapes compared to proprioceptive-only inputs. This work follows the design principles of these prior works while adjusting the experimental protocol to \textbf{systematically analyze the influence of hand actuation structure}.

As shown in Figure \ref{fig:iiwa_allegro}, the original Dexsuite benchmark primarily used a KUKA iiwa 7-DoF robotic arm paired with the Allegro Hand (4-finger). However, in this study, the \textbf{UR10e 6-DoF collaborative robot} was selected as the arm platform for the following reasons. First, the iiwa's 7-DoF structure possesses \textbf{kinematic redundancy}, meaning infinitely many joint configurations can achieve the same end-effector pose. This redundancy can introduce additional exploration complexity from a reinforcement learning perspective and may act as a confounding factor when analyzing hand actuation structure differences, which is the focus of this study. In contrast, the UR10e's 6-DoF structure provides \textbf{minimal DoF} for position and orientation control in 3D space without redundancy, making arm movements deterministic. Second, the UR10e is a widely deployed commercial collaborative robot in industrial settings, offering high practical utility. By selecting the UR10e, the \textbf{total degrees of freedom of the arm-hand system are minimized}, enabling observation of the pure influence of hand actuation structure on learning. Figure \ref{fig:ur10e_scenes} illustrates the UR10e configurations combined with the DG5F and Inspire hands.

\begin{figure}[t]
    \centering
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{ur10e_dg5f_dexsuite.png}
        \caption{UR10e + Tesollo DG5F (20-DoF) in Isaac Sim}
        \label{fig:ur10e_dg5f_scene}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{ur10e_inspire_dexsuite.png}
        \caption{UR10e + Inspire RH56F1 (6-DoF) in Isaac Sim}
        \label{fig:ur10e_inspire_scene}
    \end{subfigure}

    \caption{Simulation environments used in the controlled comparison.
    (a) UR10e with a fully actuated DG5F hand and (b) UR10e with an underactuated RH56F1 hand.
    Both scenes are rendered in Isaac Sim with identical task definitions and reward structure.}
    \label{fig:ur10e_scenes}
\end{figure}

\subsubsection{Task Definition: Object Lifting}

The experimental task is defined as an \textbf{object lifting} problem where the robot must lift an object from a table and move it to a randomly sampled target position. Unlike the Dexsuite reorientation task, \textbf{orientation is not controlled}—this task focuses solely on \textbf{position control} in 3D space. The task requires finger coordination, contact management, and stable grasping to achieve position precision.

\subsubsection{Object Diversity and Complexity}

The experiment uses \textbf{16 object variations}: four primitive shapes (cube, sphere, capsule, cone) with four different sizes each. This diversity prevents overfitting and encourages generalized manipulation strategies. At each episode start, the object type, size, initial position, and pose are randomly sampled, forcing the robot to learn robust policies across diverse conditions.

\subsubsection{Object Scaling Based on Hand Size}

To ensure fair comparison, \textbf{hand-size-proportional object scaling} is adopted: DG5F uses \textbf{1.0$\times$ scale} (original Kuka-Allegro benchmark sizes), while Inspire uses \textbf{0.5$\times$ scale} (dimensions halved), reflecting their approximate palm size ratio of 2:1. This maintains similar relative object-to-hand size ratios, equalizing geometric difficulty. However, object mass is kept constant at 0.2 kg for both systems to evaluate hardware capability differences under identical physical conditions. Reward function parameters are adjusted proportionally for the Inspire to maintain consistent precision requirements.

\subsection{Reinforcement Learning Formulation and Training Setup}

Building on the PPO algorithm introduced in Section 2.1, this section describes how the reinforcement learning problem is formalized in the dexterous manipulation environment.

\subsubsection{MDP Setup and PPO Hyperparameters}

The environment is formalized as an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma)$. The state space $\mathcal{S}$ comprises robot proprioception, object pose, target pose, and point cloud-based perception (detailed in Section 3.3.2). The action space $\mathcal{A}$ is continuous: 26 dimensions for DG5F (6 arm + 20 hand) and 12 dimensions for Inspire (6 arm + 6 hand). The transition probability $\mathcal{P}$ is determined by Isaac Lab's GPU-based PhysX engine, which accurately simulates contact dynamics, friction, and gravity. The discount factor is $\gamma = 0.99$.

For PPO training, the following key hyperparameters are used: clipping parameter $\epsilon = 0.2$, KL divergence threshold 0.01 with adaptive learning rate, entropy coefficient 0.001, and gradient clipping with norm 1.0. Each update collects data over horizon length 36 steps and reuses it for 5 mini-epochs with minibatch size 36,864, leveraging importance sampling and clipping for sample efficiency. The critic coefficient is set to 4.0 to emphasize value function learning.

\textbf{Observation-State Equivalence:} Although the environment is technically a POMDP (point clouds are partial observations, randomized physics parameters are hidden), it is treated as an MDP following standard practice in robotic manipulation \cite{petrenko2023dexpbt, qi2023dexpoint}. This is justified by: (1) 5-step history stacking enables velocity and dynamics inference, (2) point clouds provide rich shape information sufficient for generalization, and (3) contact feedback allows implicit physics parameter inference. In the implementation, observations and states are identical—both actor and critic receive the same information.

\subsubsection{Observation Space Structure}

Following DexPoint \cite{qi2023dexpoint}, the observation space consists of three groups: \textbf{Policy} (goal information), \textbf{Proprioception} (robot state), and \textbf{Perception} (object recognition), all with 5-step history stacking for temporal context. As illustrated in Figure \ref{fig:observation_pipeline}, observations are processed through clipping (±100.0), noise addition (curriculum-adjusted), running mean/std normalization, history stacking, and finally fed to the policy network.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\linewidth]{dual_track observation pipeline.png}
    \caption{Observation processing pipeline\cite{qi2023dexpoint}. Three observation groups (Policy, Proprioception, Perception) undergo normalization and 5-step history stacking before being concatenated and fed to the actor-critic networks. }
    \label{fig:observation_pipeline}
\end{figure}

\textbf{Group 1: Policy} includes goal-related observations: object quaternion (4$\times$5 = 20 dim), target pose (7$\times$5 = 35 dim), and action history (26$\times$5 = 130 dim for DG5F, 12$\times$5 = 60 dim for Inspire). Total: 185 dim (DG5F) vs 115 dim (Inspire).

\textbf{Group 2: Proprioception} captures robot internal state: joint positions and velocities (26$\times$2$\times$5 = 260 dim for DG5F, 12$\times$2$\times$5 = 120 dim for Inspire), fingertip states (6 bodies $\times$ 13 dim $\times$ 5 = 390 dim), and contact forces (5 fingers $\times$ 3 dim $\times$ 5 = 75 dim). Total: 725 dim (DG5F) vs 585 dim (Inspire).

\textbf{Group 3: Perception} provides visual object recognition: object point cloud (64 points $\times$ 3 dim $\times$ 5 = 960 dim), identical for both hands.

Table \ref{tab:observation_dim} compares the observation dimensions. The 210-dimension difference stems entirely from DoF disparity (26 vs 12), amplified by 5-step history stacking. Both systems receive the same information types, scaled to their actuation capabilities.

\begin{table}[h]
\centering
\caption{Observation Space Dimensionality Comparison}
\label{tab:observation_dim}
\begin{tabular}{lccl}
\hline
\textbf{Group} & \textbf{DG5F} & \textbf{Inspire} & \textbf{Difference} \\
\hline
Policy & 185 dim & 115 dim & +70 dim (action history) \\
Proprioception & 725 dim & 585 dim & +140 dim (joint info) \\
Perception & 960 dim & 960 dim & 0 dim (identical) \\
\hline
\textbf{Total} & \textbf{1870 dim} & \textbf{1660 dim} & \textbf{+210 dim} \\
\hline
\end{tabular}
\end{table}

\subsubsection{Reward Function Design}

The reward function combines sparse rewards (goal achievement), dense rewards (learning guidance), and penalties (unstable behavior suppression), adapted from the Kuka-Allegro Dexsuite benchmark \cite{petrenko2023dexpbt} with position parameters scaled for hand-size differences.

\textbf{Sparse Reward - Success} (weight = 10.0): Awarded when the object reaches the target pose, defined as:
\begin{equation}
r_{\text{success}} = \exp\left(-\frac{\|p_{\text{obj}} - p_{\text{goal}}\|^2}{\sigma_{\text{pos}}^2}\right) \times \exp\left(-\frac{\|\text{quat\_diff}(q_{\text{obj}}, q_{\text{goal}})\|^2}{\sigma_{\text{rot}}^2}\right)
\end{equation}
where $\sigma_{\text{pos}} = 0.1$m (DG5F) and 0.05m (Inspire), reflecting their 2:1 object size ratio.

\textbf{Dense Rewards:} \textit{Position tracking} (weight = 2.0) minimizes object-to-goal distance:
\begin{equation}
r_{\text{position}} = \exp\left(-\frac{\|p_{\text{obj}} - p_{\text{goal}}\|^2}{\sigma^2}\right)
\end{equation}
with $\sigma = 0.2$m (DG5F) and 0.1m (Inspire). \textit{Orientation tracking} (weight = 4.0) aligns object rotation with the target, weighted 2$\times$ higher than position due to increased difficulty. \textit{Fingers-to-object} (weight = 1.0) reduces average fingertip-to-object distance to encourage grasping.

\textbf{Penalties:} \textit{Action L2} (weight = -0.005) and \textit{action rate L2} (weight = -0.005) regularize action magnitude and smoothness. \textit{Ground contact penalty} (weight = -0.3) prevents table-pinning shortcuts.

Both hands use identical reward structure, with only position-related parameters scaled proportionally to maintain relative difficulty.

\subsubsection{Neural Network Architecture}

A \textbf{separate actor-critic architecture} is employed where policy and value networks have independent parameters, providing stability for high-dimensional inputs and complex contact dynamics. Figure \ref{fig:actor_critic} illustrates the network structure.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\linewidth]{actor_critic_network.png}
    \caption{Actor-critic network architecture. Both networks use 3-layer MLPs [512, 256, 128] with ELU activation. The actor outputs a Gaussian policy ($\mu$, fixed $\sigma$), while the critic estimates the state value function $V(s)$.}
    \label{fig:actor_critic}
\end{figure}

Input observations (concatenated 3 groups) undergo running mean/std normalization before being fed to both networks. \textbf{Actor} (policy network) uses a 3-layer MLP with hidden sizes [512, 256, 128] and ELU activation, outputting a Gaussian policy with mean $\mu$ and fixed standard deviation $\sigma = 1.0$. \textbf{Critic} (value network) shares the same MLP structure but with independent parameters, outputting a scalar state value function $V(s)$ that estimates expected cumulative reward:
\begin{equation}
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t) \mid s_0 = s \right]
\end{equation}

The critic is trained by minimizing TD error independently from the actor, enabling accurate advantage estimation ($\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \tau)^l \delta_{t+l}$ with GAE parameter $\tau = 0.95$) that directly improves policy learning.

\subsection{Curriculum Learning and Domain Randomization}

\subsubsection{Automatic Domain Randomization (ADR)}

Following DexPBT \cite{petrenko2023dexpbt}, ADR-based curriculum learning is employed that automatically adjusts environment difficulty based on success rate. \textbf{Gravity curriculum} starts from [0, 0, 0] and gradually increases to [0, 0, -9.81] (z-axis only) as the agent succeeds, enabling grasp strategy learning before introducing gravity. \textbf{Joint friction} randomizes within [0.0, 5.0], \textbf{object mass} scales within [0.2, 2.0]$\times$, and \textbf{joint stiffness/damping} vary within [0.8, 1.2]$\times$, all adjusted by ADR.

\subsubsection{Domain Randomization for Sim-to-Real Transfer}

Extensive domain randomization is applied following OpenAI's Rubik's Cube approach \cite{openai2019rubiks}. \textbf{Startup randomization} (once per episode reset): robot material friction [0.9, 1.1], object material friction [0.5, 1.0], object mass [0.2, 2.0]$\times$, joint stiffness/damping [0.8, 1.2]$\times$. \textbf{Reset randomization} (every episode): object pose randomized within table region (x: [-0.1, 0.1]m, y: [-0.2, 0.2]m, z: [0.0, 0.2]m) with full orientation randomization, robot joint positions randomized within safe ranges, and table pose varied by ±5cm. These ensure learned policies are robust to real-world uncertainties.

\subsubsection{Final Experimental Configuration}

Following the design principles above, the comparative experiment is configured as shown in Table \ref{tab:training_comparison}. The two systems are trained under \textbf{identical task (object lifting), identical algorithm (PPO), identical reward structure, and identical curriculum learning (ADR)} conditions. The only differences are the \textbf{hand actuation structure} (underactuated vs fully-actuated) and the resulting degrees of freedom and observation/action dimensionalities. Through this controlled experimental design, the pure influence of actuation structure on reinforcement learning performance can be isolated and analyzed.

\begin{table}[h]
\centering
\caption{Experimental Configuration and Training Protocol Comparison}
\label{tab:training_comparison}
\begin{tabular}{lcc}
\hline
\textbf{Configuration} & \textbf{UR10e + DG5F} & \textbf{UR10e + Inspire} \\
\hline
\multicolumn{3}{l}{\textit{Identical Conditions (Controlled Variables)}} \\
\hline
Robot Arm & UR10e (6-DoF) & UR10e (6-DoF) \\
RL Algorithm & PPO & PPO \\
Network Architecture & 3-layer MLP [512, 256, 128] & 3-layer MLP [512, 256, 128] \\
Activation Function & ELU & ELU \\
Hyperparameters & Identical (lr, batch, horizon) & Identical (lr, batch, horizon) \\
Task & Object lifting (position-only) & Object lifting (position-only) \\
Reward Structure & Same terms \& weights & Same terms \& weights \\
Domain Randomization & Same physics ranges & Same physics ranges \\
Curriculum Learning & ADR & ADR \\
Training Duration & 7,500 epochs & 7,500 epochs \\
Total Timesteps & $\sim$1.106B steps & $\sim$1.105B steps \\
Parallel Environments & 4,096 & 4,096 \\
\hline
\multicolumn{3}{l}{\textit{Necessary Adjustments (Due to Actuation Structure)}} \\
\hline
Robot Hand & DG5F (20-DoF, fully-act.) & RH56F1 (6-DoF, underact.) \\
Action Space & 26-dim (6 arm + 20 hand) & 12-dim (6 arm + 6 hand) \\
Observation Space & 1,870-dim & 1,660-dim \\
Object Scale & 1.0$\times$ (original) & 0.5$\times$ (hand-proportional) \\
Position Reward Param & $\sigma = 0.2$m & $\sigma = 0.1$m (scaled) \\
Success Position Tol. & $\sigma_{\text{pos}} = 0.1$m & $\sigma_{\text{pos}} = 0.05$m (scaled) \\
\hline
\multicolumn{3}{l}{\textit{Training Protocol and Performance}} \\
\hline
Wall-clock Time & $\sim$12 hours & $\sim$8--9 hours (25\% faster) \\
Step Inference Time & $\sim$5.8 ms/step & $\sim$4.0 ms/step (1.45$\times$ faster) \\
\hline
\end{tabular}
\end{table}

\section{Experiments}

\subsection{Experimental Setup}

The experiments are conducted using a workstation equipped with an Intel Core Ultra 9 285K CPU (24 cores) and a single NVIDIA GeForce RTX 5090 GPU with 32 GB of VRAM, supported by 128 GB of system RAM. Using the Isaac Lab simulation framework \cite{isaaclab}, 4,096 parallel environments can be simulated on the GPU. Combined with a GPU-based vectorized RL implementation RL-Games \cite{rlgames}, this configuration enables efficient distributed learning for dexterous manipulation tasks.

To ensure a fair comparison, the following \textbf{controlled variables} and \textbf{necessary adjustments} are established. Table \ref{tab:training_comparison} summarizes the complete experimental configuration, training protocol, and performance characteristics for both systems.


Both systems were trained for identical durations (7,500 epochs, approximately 1.1 billion timesteps) under the same PPO algorithm, network architecture, reward structure, and curriculum learning protocol. The only differences are the hand actuation structure (fully-actuated vs underactuated) and the resulting degrees of freedom, which naturally lead to different action/observation dimensionalities. Object size and position-related reward parameters were scaled proportionally to maintain equivalent geometric difficulty relative to hand size. Interestingly, the Inspire system demonstrated faster wall-clock training time (8--9 hours vs 12 hours) and step inference speed (4.0 ms/step vs 5.8 ms/step) due to its lower DoF, reducing both physics simulation and policy network computation. However, as demonstrated in Section 4.2, this computational efficiency did not translate to better learning performance.

\subsection{Results and Analysis}

\subsubsection{Learning Performance Comparison}

Figure \ref{fig:rewards} presents the total episode reward trajectories for both systems across the entire training duration. The results reveal a dramatic performance gap between the two hand configurations. Figure \ref{fig:rewardsperstep} shows rewards per training step, while Figure \ref{fig:rewardspertime} presents the same data as a function of wall-clock time—both perspectives yield identical conclusions, ruling out computational efficiency as a confounding factor.

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth, height=4cm]{total_step_reward.png}
        \caption{Total episode reward per training step}
        \label{fig:rewardsperstep}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth, height=4cm]{total_time_reward.png}
        \caption{Total episode reward per wall-clock time}
        \label{fig:rewardspertime}
    \end{subfigure}
    \caption{Learning curves comparing DG5F (blue) and RH56F1 (red). Both per-step and per-time views show consistent patterns: DG5F successfully learns the manipulation task (total reward $\sim$18--19), while RH56F1 fails to progress beyond initial exploration (total reward $\sim$2).}
    \label{fig:rewards}
\end{figure}

The DG5F system (blue curve) demonstrates successful learning with a characteristic three-phase progression. During the initial exploration phase (0--1000 steps, 0--5 hours), the reward gradually increases from near-zero to approximately 5--8 as the agent discovers basic contact and approach strategies. A rapid improvement phase (1000--3000 steps, 5--15 hours) follows, where the reward steeply rises to 15--18, indicating the emergence of successful lifting and positioning behaviors. Finally, the convergence phase (3000--7500 steps, 15--24 hours) shows stable performance at reward 18--19, with minor oscillations reflecting continued exploration under domain randomization.

In contrast, the RH56F1 system (red curve) exhibits learning failure. The reward curve remains nearly flat throughout the entire 7500-epoch training period, plateauing at approximately 2.0 with no discernible improvement trajectory. This indicates that while the underactuated hand learned some basic behaviors (e.g., approaching the object, making finger contact), it failed to discover the critical manipulation strategy required for the lifting task. The absence of any upward trend even after 1.1 billion timesteps suggests a fundamental learning bottleneck rather than mere sample inefficiency.

Critically, both the per-step view (Figure \ref{fig:rewardsperstep}) and per-time view (Figure \ref{fig:rewardspertime}) show identical relative performance, confirming that the RH56F1's 1.45$\times$ faster inference speed (Section 4.1) provides no learning advantage. Despite processing more wall-clock experience in less time, the underactuated hand could not overcome its exploration and expressiveness constraints.

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{dg5f_1.png}
        \caption{DG5F lifting sphere}
        \label{fig:Dg5f lifting sphere}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{dg5f_2.png}
        \caption{DG5F lifting cube}
        \label{fig:Dg5f lifting cube}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{dg5f_3.png}
        \caption{DG5F lifting cone}
        \label{fig:Dg5f lifting cone}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{dg5f_4.png}
        \caption{DG5F lifting capsule}
        \label{fig:Dg5f lifting capsule}
    \end{subfigure}
    \caption{DG5F manipulation behavior after 7500 epochs of training. All subfigures show \textbf{green tables}, indicating consistent success across diverse object geometries. The hand demonstrates proper approach, pre-grasp shaping, enveloping grasp formation, lifting, and goal-directed positioning.}
    \label{fig:dg5f_lifting}
\end{figure}

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{inspire_1.png}
        \caption{RH56F1 lifting sphere}
        \label{fig:RH56F1 lifting sphere}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{inspire_2.png}
        \caption{RH56F1 lifting cube}
        \label{fig:RH56F1 lifting cube}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{inspire_3.png}
        \caption{RH56F1 lifting cone}
        \label{fig:RH56F1 lifting cone}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.45\linewidth}
        \centering
        % height 파라미터 추가 (원하는 cm 단위 입력, 예: 6cm)
        \includegraphics[width=\linewidth, height=4cm]{inspire_4.png}
        \caption{RH56F1 lifting capsule}
        \label{fig:RH56F1 lifting capsule}
    \end{subfigure}
    \caption{RH56F1 manipulation behavior after 7500 epochs of training. Most subfigures show \textbf{red/pink tables}, indicating task failure. Objects remain on the table surface rather than being lifted. The hand makes contact but fails to execute successful grasping and lifting strategies.}
    \label{fig:inspire_lifting}
\end{figure}

\subsubsection{Qualitative Manipulation Comparison}

To validate the quantitative results from Figure \ref{fig:rewards}, the learned manipulation behaviors are examined through visual inspection of the simulation environment. In the Isaac Lab setup, table color serves as a real-time success indicator: \textbf{green tables} signify successful episode completion (object reached target position and maintained for sufficient duration), while \textbf{red/pink tables} indicate failure (target not achieved, object dropped, or abnormal termination).

Figures \ref{fig:dg5f_lifting} and \ref{fig:inspire_lifting} present representative snapshots of the trained policies manipulating four different object geometries: sphere, cube, cone, and capsule. These object types span the diversity of shapes encountered during training, testing the generalization capability of each learned policy.

As shown in Figure \ref{fig:dg5f_lifting}, the DG5F system successfully manipulates all four object types, evidenced by the consistent green table color across all subfigures. Visual inspection reveals a sophisticated manipulation strategy: the hand approaches the object from above, shapes its fingers to match the object geometry (pre-grasp configuration), wraps all five fingers around the object (enveloping grasp), lifts it clear of the table, and positions it toward the goal. Notably, the DG5F can adapt its grasp configuration to object shape—using a power grasp for the sphere (Fig. \ref{fig:dg5f_lifting}a), a precision grasp for the cube (Fig. \ref{fig:dg5f_lifting}b), and adaptive wrapping for irregular shapes like the cone and capsule.

In contrast, Figure \ref{fig:inspire_lifting} reveals that the RH56F1 system fails to achieve successful manipulation, with red/pink tables dominating the visualizations. Across all object types, the objects remain stationary on the table surface rather than being lifted. While the hand does make contact with objects (visible finger-object proximity in all subfigures), it fails to form stable grasps capable of supporting the object against gravity. The learned policy appears to be a contact-seeking behavior where the hand moves toward and touches the object, then remains in place until episode timeout. This behavior is consistent with the flat reward curve in Figure \ref{fig:rewards}—the policy discovered that making contact yields immediate dense rewards (fingers-to-object distance reduction) but never progressed to the more challenging lifting and positioning objectives.

These qualitative observations align perfectly with the quantitative reward trajectories, confirming that the performance gap is not merely numerical but reflects fundamental differences in learned manipulation competence.

\subsubsection{Task Success Metrics}

Figure \ref{fig:task_success} presents the core task performance metrics: success reward and position tracking. These metrics directly measure whether the robot achieved the manipulation objective—lifting the object and positioning it at the target location.

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{success.png}
        \caption{Success reward progression}
        \label{fig:success}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{position_tracking.png}
        \caption{Position tracking reward progression}
        \label{fig:position_tracking}
    \end{subfigure}
    \caption{Task success metrics comparing DG5F and RH56F1. Both metrics show complete task failure for the underactuated hand.}
    \label{fig:task_success}
\end{figure}

The results are unambiguous: DG5F achieves steadily increasing success reward, reaching approximately 3.3 by epoch 7500, while RH56F1 remains at zero throughout training. Similarly, position tracking reward for DG5F converges to approximately 0.9, whereas RH56F1 exhibits negligible values. This indicates that the underactuated hand completely failed to learn the fundamental manipulation skill—lifting and positioning objects—despite 1.1 billion training timesteps.

\subsubsection{Contact Behavior Analysis}

To understand why RH56F1 failed at task completion while presumably making some progress, contact-related metrics are examined in Figure \ref{fig:contact_metrics}.

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{fingers_to_object.png}
        \caption{Fingers-to-object distance reward}
        \label{fig:fingers_to_object}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{good_finger_contact.png}
        \caption{Good finger contact reward}
        \label{fig:good_finger_contact}
    \end{subfigure}
    \caption{Contact-related reward metrics revealing distinct learning strategies between the two systems.}
    \label{fig:contact_metrics}
\end{figure}

Interestingly, the fingers-to-object metric reveals a critical insight: RH56F1 (final value $\sim$0.67) actually outperforms DG5F (final value $\sim$0.52) in bringing fingers close to objects. The underactuated hand rapidly learns this behavior within the first 500 epochs and maintains it consistently. However, good finger contact reward remains near-zero for both systems initially, with DG5F eventually reaching $\sim$0.26 as it learns successful grasping. This disparity exposes the core failure mode: \textit{RH56F1 learned to approach and touch objects but never progressed to forming stable grasps capable of lifting}. The policy converged to a local optimum that maximizes easily-achievable dense rewards (minimizing finger-object distance) without discovering the more complex manipulation primitives (grasping and lifting) required for task success.

\subsubsection{Exploration and Learning Dynamics}

Figure \ref{fig:learning_dynamics} examines how actuation structure influences exploration behavior and value function learning.

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{entropy.png}
        \caption{Policy entropy over training}
        \label{fig:entropy}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{critic_loss.png}
        \caption{Critic loss over training}
        \label{fig:critic_loss}
    \end{subfigure}
    \caption{Exploration and value function learning dynamics reveal fundamental differences in how the two systems learn.}
    \label{fig:learning_dynamics}
\end{figure}

Entropy trajectories show striking differences: DG5F exhibits a U-shaped pattern, starting at $\sim$37, decreasing to $\sim$23 during focused exploitation (epochs 2000--4000), then rising to $\sim$38 as the policy stabilizes with maintained exploration. In contrast, RH56F1 follows a similar initial pattern but plateaus at $\sim$20, approximately 47\% lower than DG5F's final entropy. This entropy gap reflects the implicit action space constraints imposed by mechanical coupling—the underactuated hand has fewer distinct behaviors available for exploration, limiting its ability to discover diverse manipulation strategies despite PPO's entropy bonus.

Critic loss patterns further illuminate this disparity. DG5F's loss initially rises to $\sim$0.42 as the policy encounters increasingly complex states during successful manipulation learning, then gradually decreases to $\sim$0.12 as value estimates stabilize. RH56F1 maintains consistently low critic loss ($\sim$0.06), indicating a simpler value landscape. Paradoxically, DG5F's higher critic loss correlates with successful task performance, suggesting that complex manipulation requires navigating and accurately evaluating diverse high-reward states, whereas RH56F1's simple policy (approach-and-touch without lifting) operates in a restricted state space requiring less sophisticated value estimation.

\subsubsection{Curriculum Progression and Robustness}

Figure \ref{fig:curriculum} presents the ADR curriculum difficulty progression, quantifying how quickly each system masters increasingly challenging randomized environments.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\linewidth]{curriculum_adr.png}
    \caption{ADR curriculum difficulty progression over training, showing the stark robustness gap between fully-actuated and underactuated systems.}
    \label{fig:curriculum}
\end{figure}

DG5F demonstrates rapid curriculum advancement from epoch 2000 onward, reaching normalized difficulty $\sim$0.85 by epoch 5000 and stabilizing thereafter. This indicates the fully-actuated hand learned robust policies capable of handling diverse physics randomization, object masses, friction coefficients, and gravitational loads. In contrast, RH56F1's curriculum remains at zero throughout training—the ADR system never advances difficulty because the agent fails to achieve sufficient success rates even under the easiest conditions. This represents a complete robustness failure: the underactuated hand cannot reliably execute successful manipulation even with minimal environmental variation, let alone under the challenging randomization ranges that DG5F masters.

\section{Conclusion}

\subsection{Discussion}

The experimental results reveal a dramatic performance gap between fully-actuated (DG5F) and underactuated (RH56F1) robotic hands when trained under identical conditions. DG5F successfully learned object manipulation, achieving total episode rewards of 18--19, success rewards of $\sim$3.3, and position tracking rewards of $\sim$0.9. In contrast, RH56F1 failed to progress beyond basic contact behaviors, with total rewards plateauing at $\sim$2.0 and success/position tracking metrics remaining at zero throughout 7500 epochs. This performance gap persisted across all metrics: exploration diversity (entropy: 38 vs 20), value function learning, and curriculum progression (ADR difficulty: 0.85 vs 0.0).

The analysis uncovered a critical paradox that illuminates the failure mode: RH56F1 \textit{outperformed} DG5F on the fingers-to-object distance metric (0.67 vs 0.52), demonstrating faster approach behavior. However, this superior performance on dense intermediate rewards did not translate to task success. The hypothesis is that the underactuated hand fell into a \textbf{local optimum} in the reward landscape—the policy learned to maximize easily-achievable dense rewards (approaching and touching objects) but failed to discover the challenging sparse rewards (forming stable grasps and lifting). This exploration failure is evidenced by 47\% lower policy entropy, indicating that RH56F1 converged to a narrow strategy space early in training and never escaped.

Furthermore, the mechanical constraints of underactuation fundamentally limit the hand's expressiveness: unlike DG5F, which can independently control each finger joint to shape grasps precisely, RH56F1's coupled actuation forces coordinated synergies. While such synergies enable adaptive grasping in some contexts, they appear insufficient for the fine-grained control required to stabilize objects against gravity during lifting. The curriculum progression failure (ADR difficulty remaining at zero) compounds this issue—without advancing to harder randomization ranges, the hand never developed robustness for reliable manipulation. These findings suggest underactuated hands face compounding challenges in RL-based learning: limited exploration due to reduced action space dimensionality, local optima traps in dense reward shaping, and insufficient mechanical expressiveness for complex manipulation primitives.

\subsection{Limitation}

Several limitations warrant consideration. The experimental scope is limited to a single arm-hand configuration for each actuation type (UR10e + DG5F and UR10e + RH56F1), leaving generalization to other robotic platforms untested. Different kinematic structures or hand morphologies may yield different relative performance outcomes. Additionally, the analysis focuses on empirical performance metrics rather than mathematical characterization of actuation differences—a formal analysis of constraint manifolds, null space properties, or grasp stability conditions could offer deeper theoretical insights into why certain manipulation strategies emerge or fail under different actuation paradigms.

The training duration was constrained to 7500 epochs (1.1 billion timesteps), representing the practical upper limit for the computational setup. While this duration proved sufficient for DG5F convergence, it is possible that RH56F1 might eventually discover successful strategies with significantly longer training, though the complete absence of upward trends suggests fundamental bottlenecks rather than sample inefficiency. All experiments were conducted exclusively in simulation using Isaac Lab; real-world deployment would introduce actuator backlash, sensor noise, and contact dynamics modeling errors that may differentially impact actuation types.

This study exclusively employs Proximal Policy Optimization (PPO). Alternative algorithms—such as Soft Actor-Critic (SAC) or other off-policy methods—might exhibit different sample efficiency or exploration characteristics, potentially helping underactuated hands escape local optima more effectively. The domain randomization protocol did not include object mass variation; while randomization was applied to friction coefficients and gravitational loads, incorporating mass variation could provide more comprehensive robustness evaluation, though implementing this fairly across actuation types is nontrivial due to effort/torque scaling differences.


\newpage
\begin{thebibliography}{21}

\bibitem{yu2022dexterous}
C. Yu, P. Wang, X. Ma, and Y. Zhang. Dexterous manipulation for multi-fingered robotic hands: A review. \textit{Frontiers in Neurorobotics}, 16:873802, 2022.

\bibitem{rajeswaran2018dexterous}
A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. In \textit{Robotics: Science and Systems}, 2018.

\bibitem{andrychowicz2020learning}
O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. Learning dexterous in-hand manipulation. \textit{The International Journal of Robotics Research}, 39(1):3--20, 2020.

\bibitem{openai2019rubiks}
OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, et al. Solving Rubik's Cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.

\bibitem{petrenko2023dexpbt}
A. Petrenko, E. Allshire, T. Chen, and G. State. DexPBT: Scaling up dexterous manipulation for hand-arm systems with population based training. In \textit{Robotics: Science and Systems}, 2023.

\bibitem{qi2023dexpoint}
Y. Qi, R. Chen, M. Liu, P. Zeng, G. Wang, Y. Xiang, X. Li, C. Wang, Y. Qin, and H. Su. DexPoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation. In \textit{Conference on Robot Learning}, 2023.

\bibitem{allegro_hand}
Wonik Robotics. Allegro Hand. \url{https://www.wonikrobotics.com/research-robot-hand}.

\bibitem{tesollo_dg5f}
TESOLLO. DG-5F Dexterous Gripper. \url{https://en.tesollo.com/dg-5f/}.

\bibitem{schulman2017ppo}
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

\bibitem{lopes2025lookahead}
A. F. G. Lopes, C. Barata, and P. Moreno. Model-based lookahead reinforcement learning for in-hand manipulation. arXiv preprint arXiv:2510.08884, 2025.

\bibitem{deimel2013novel}
R. Deimel and O. Brock. A novel type of compliant, underactuated robotic hand for dexterous grasping. In \textit{Robotics: Science and Systems}, 2013.

\bibitem{birglen2009underactuated}
L. Birglen, T. Laliberté, and C. M. Gosselin. Underactuated Robotic Hands. \textit{Springer Tracts in Advanced Robotics}, vol. 40. Springer, 2009.

\bibitem{catalano2014softhand}
M. G. Catalano, G. Grioli, A. Farnioli, A. Serio, C. Piazza, and A. Bicchi. Adaptive synergies for the design and control of the Pisa/IIT SoftHand. \textit{The International Journal of Robotics Research}, 33(5):768--782, 2014.

\bibitem{dollar2011underactuated}
A. M. Dollar and R. D. Howe. The highly adaptive SDM hand: Design and performance evaluation. \textit{The International Journal of Robotics Research}, 29(5):585--597, 2010.

\bibitem{wang2019truly}
Y. Wang, H. He, and X. Tan. Truly proximal policy optimization. In \textit{Conference on Uncertainty in Artificial Intelligence}, 2019.

\bibitem{schulman2015trpo}
J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In \textit{International Conference on Machine Learning}, 2015.

\bibitem{rlgames}
A. Makoviychuk and V. Makoviichuk. RL-Games: A high-performance framework for reinforcement learning. \url{https://github.com/Denys88/rl_games}, 2021.

\bibitem{sintov2019underactuated}
A. Sintov, O. Tslil, and A. Shapiro. Data-driven modeling and control of an underactuated hand. \textit{IEEE Robotics and Automation Letters}, 4(3):2246--2253, 2019.

\bibitem{inspire_rh56f1}
Inspire Robots. RH56F1 Robotic Hand. \url{https://en.inspire-robots.com/}.

\bibitem{isaaclab}
M. Mittal, C. Yu, Q. Yu, J. Liu, and N. Rudin. Isaac Lab: A GPU-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

\end{thebibliography}

\end{document}
