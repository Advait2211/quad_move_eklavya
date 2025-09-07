# 8_PPO_bipedal

Uses Proximal Policy Optimization (PPO) to train a policy for the BipedalWalker-v3 environment.
Highlights clipping-based surrogate objective for stable training.

## Run Instructions

```bash
cd 8_PPO_bipedal
python3 biped.py  # Train PPO on BipedalWalker
python3 test_biped.py  # Test and visualize the trained policy
```

![Bipedal Walker](../pre_gifs/bipedal_walker.gif)