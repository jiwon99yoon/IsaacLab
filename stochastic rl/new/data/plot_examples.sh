#!/bin/bash
# Example commands for plotting different metrics

# Figure 6: Shaped Reward
python plot_comparison.py \
    --dg5f shaped_step_reward_dg5f.csv \
    --inspire shaped_step_reward_inspire.csv \
    --ylabel "Total Shaped Reward" \
    --output fig6_shaped_reward.png

# Figure 10a: Entropy
python plot_comparison.py \
    --dg5f entropy_dg5f.csv \
    --inspire entropy_inspire.csv \
    --ylabel "Policy Entropy" \
    --output fig10a_entropy.png

# Figure 10b: Critic Loss
python plot_comparison.py \
    --dg5f critic_loss_dg5f.csv \
    --inspire critic_loss_inspire.csv \
    --ylabel "Critic Loss (TD Error)" \
    --output fig10b_critic_loss.png \
    --smoothing 0.90

# Figure 10c: KL Divergence
python plot_comparison.py \
    --dg5f kl_divergence_dg5f.csv \
    --inspire kl_divergence_inspire.csv \
    --ylabel "KL Divergence" \
    --output fig10c_kl.png

# Figure 10d: Episode Length
python plot_comparison.py \
    --dg5f episode_length_dg5f.csv \
    --inspire episode_length_inspire.csv \
    --ylabel "Episode Length (Steps)" \
    --output fig10d_episode_length.png

# Figure 11: ADR Curriculum
python plot_comparison.py \
    --dg5f adr_curriculum_dg5f.csv \
    --inspire adr_curriculum_inspire.csv \
    --ylabel "Curriculum Difficulty (Normalized)" \
    --output fig11_adr_curriculum.png \
    --smoothing 0.98

# Success Reward
python plot_comparison.py \
    --dg5f success_reward_dg5f.csv \
    --inspire success_reward_inspire.csv \
    --ylabel "Success Reward" \
    --output success_reward.png

# Position Tracking
python plot_comparison.py \
    --dg5f position_tracking_dg5f.csv \
    --inspire position_tracking_inspire.csv \
    --ylabel "Position Tracking Reward" \
    --output position_tracking.png
