# 7_DDQN

Extends DQN into Double DQN (DDQN) to reduce overestimation bias by decoupling action selection and evaluation.
Applied to the LunarLander environment for improved stability.

## Run Instructions

```bash
cd 7_DDQN
python3 train_ddqn.py  # Train DDQN agent
python3 lunar_lander_gpu_ddqn.py  # Visualize using GPU acceleration
```
