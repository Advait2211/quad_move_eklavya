import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm
import wandb
import os
import time

# ========================
# CONFIG
# ========================
ENV_ID = "Ant-v5"
NUM_ENVS = 16  # Reduced from 128 for better CPU utilization
STEPS_PER_ENV = 2048  # Increased from 256 for better sample efficiency
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 10
NUM_MINIBATCHES = 32  # Better minibatch organization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
GRAD_CLIP = 0.5
TOTAL_TIMESTEPS = 10_000_000

# ========================
# WANDB SETUP
# ========================
run_name = f"{ENV_ID}_ppo_go2_{int(time.time())}"
wandb.init(
    project="ppo-go2-optimized",
    name=run_name,
    config={
        "env_id": ENV_ID,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": CLIP_EPS,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "num_minibatches": NUM_MINIBATCHES,
        "device": str(DEVICE),
        "total_timesteps": TOTAL_TIMESTEPS,
    }
)

# ========================
# CHECKPOINT SETUP
# ========================
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, iteration):
    path = os.path.join(CHECKPOINT_DIR, f"ppo_go2_iter_{iteration}.pth")
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    wandb.save(path)
    print(f"[Checkpoint] Saved at iteration {iteration} → {path}")

# ========================
# ENV CREATION
# ========================
def make_env(env_id, idx=0):
    def _init():
        env = gym.make(
            env_id,
            xml_file="./unitree_go2/scene.xml",
            forward_reward_weight=2,
            ctrl_cost_weight=0.1,
            contact_cost_weight=0.01,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.32, 0.55),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.01,
            frame_skip=2,
            max_episode_steps=1000,  # Increased episode length
            render_mode=None,
        )
        # Add standard wrappers for better training
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return _init

# Use SyncVectorEnv instead of AsyncVectorEnv for better CPU efficiency
envs = SyncVectorEnv([make_env(ENV_ID, i) for i in range(NUM_ENVS)])

# ========================
# OPTIMIZED MODEL
# ========================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        
        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ========================
# INITIALIZE AGENT
# ========================
agent = Agent(envs).to(DEVICE)
optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

# ========================
# STORAGE SETUP - ALL ON DEVICE
# ========================
batch_size = int(NUM_ENVS * STEPS_PER_ENV)
minibatch_size = int(batch_size // NUM_MINIBATCHES)
num_iterations = TOTAL_TIMESTEPS // batch_size

obs = torch.zeros((STEPS_PER_ENV, NUM_ENVS) + envs.single_observation_space.shape).to(DEVICE)
actions = torch.zeros((STEPS_PER_ENV, NUM_ENVS) + envs.single_action_space.shape).to(DEVICE)
logprobs = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
rewards = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
dones = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)
values = torch.zeros((STEPS_PER_ENV, NUM_ENVS)).to(DEVICE)

# ========================
# TRAINING LOOP
# ========================
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset()
next_obs = torch.Tensor(next_obs).to(DEVICE)
next_done = torch.zeros(NUM_ENVS).to(DEVICE)

for iteration in tqdm(range(1, num_iterations + 1), desc="PPO Training"):
    # Anneal learning rate
    frac = 1.0 - (iteration - 1.0) / num_iterations
    lrnow = frac * LR
    optimizer.param_groups[0]["lr"] = lrnow
    
    # ========================
    # ROLLOUT PHASE
    # ========================
    for step in range(0, STEPS_PER_ENV):
        global_step += NUM_ENVS
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Execute environment step
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(DEVICE).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(DEVICE), torch.Tensor(next_done).to(DEVICE)

        # Log episode statistics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    wandb.log({
                        "charts/episodic_return": info["episode"]["r"],
                        "charts/episodic_length": info["episode"]["l"],
                        "global_step": global_step
                    })

    # ========================
    # GAE COMPUTATION - VECTORIZED
    # ========================
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(DEVICE)
        lastgaelam = 0
        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + values

    # ========================
    # FLATTEN THE BATCH
    # ========================
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # ========================
    # POLICY UPDATE
    # ========================
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # Calculate approx_kl
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -CLIP_EPS,
                CLIP_EPS,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
            optimizer.step()

    # ========================
    # LOGGING
    # ========================
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    sps = int(global_step / (time.time() - start_time))
    
    wandb.log({
        "iteration": iteration,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "losses/value_loss": v_loss.item(),
        "losses/policy_loss": pg_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/clipfrac": np.mean(clipfracs),
        "losses/explained_variance": explained_var,
        "charts/SPS": sps,
        "global_step": global_step
    })

    print(f"Iteration {iteration}, SPS: {sps}")

    # Save checkpoint
    if iteration % 100 == 0:
        save_checkpoint(agent, optimizer, iteration)

# ========================
# EVALUATION
# ========================
print("Starting evaluation...")
eval_env = make_env(ENV_ID)()
NUM_EVAL_EPISODES = 10

eval_returns = []
for ep in range(NUM_EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done = False
    ep_return = 0.0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
            action = action.cpu().numpy()[0]
        obs, reward, term, trunc, _ = eval_env.step(action)
        ep_return += reward
        done = bool(term or trunc)
    eval_returns.append(ep_return)
    print(f"Eval Episode {ep + 1}: Return = {ep_return:.2f}")

wandb.log({"eval/mean_return": np.mean(eval_returns)})
print(f"Mean Evaluation Return: {np.mean(eval_returns):.2f}")

eval_env.close()
envs.close()

# Save final model
model_path = f"ppo_go2_final_{run_name}.pth"
torch.save(agent.state_dict(), model_path)
wandb.save(model_path)
print(f"Final model saved to {model_path}")
