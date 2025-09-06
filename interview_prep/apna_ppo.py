import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(64, 64),
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        epochs=10,
        minibatches=4,
        entropy_coeff=0.0,
        value_coeff=0.5,
        max_grad_norm=0.5,
        device=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatches = minibatches
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        # policy network
        layers = []
        last_size = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last_size, size), nn.Tanh()]
            last_size = size
        layers += [nn.Linear(last_size, act_dim)]
        self.policy_mean = nn.Sequential(*layers)
        self.policy_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # value network
        v_layers = []
        last_size = obs_dim
        for size in hidden_sizes:
            v_layers += [nn.Linear(last_size, size), nn.Tanh()]
            last_size = size
        v_layers += [nn.Linear(last_size, 1)]
        self.value_net = nn.Sequential(*v_layers)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def forward(self, obs: torch.Tensor):
        mean = self.policy_mean(obs)
        logstd = self.policy_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        return dist

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        dist = self.forward(obs)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_net(obs).squeeze(-1)
        return action, logp, entropy, value

    def compute_gae(self, rewards, values, dones, last_value):
        T, N = rewards.shape
        adv = torch.zeros_like(rewards, device=self.device)
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            nextval = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nextval * nonterminal - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        returns = adv + values
        return adv, returns

    def update(self, batch):
        obs, actions, old_logp, returns, advs = batch
        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            idx = np.arange(batch_size)
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idx[start : start + minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                dist = self.forward(mb_obs)
                mb_logp = dist.log_prob(mb_actions).sum(-1)
                mb_entropy = dist.entropy().sum(-1)
                mb_value = self.value_net(mb_obs).squeeze(-1)

                ratio = (mb_logp - mb_old_logp).exp()
                pg_loss = -torch.min(
                    ratio * mb_advs,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advs,
                ).mean()

                v_unclipped = (mb_value - mb_returns) ** 2
                v_clipped = (mb_value + torch.clamp(mb_value - mb_returns, -self.clip_eps, self.clip_eps)) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = -mb_entropy.mean()
                loss = pg_loss + self.value_coeff * v_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def learn(self, envs, total_steps, steps_per_env):
        num_envs = envs.num_envs
        obs_shape = envs.single_observation_space.shape
        obs_buf = torch.zeros((steps_per_env, num_envs) + obs_shape, device=self.device)
        act_buf = torch.zeros((steps_per_env, num_envs) + (self.act_dim,), device=self.device)
        logp_buf = torch.zeros((steps_per_env, num_envs), device=self.device)
        rew_buf = torch.zeros((steps_per_env, num_envs), device=self.device)
        done_buf = torch.zeros((steps_per_env, num_envs), device=self.device)
        val_buf = torch.zeros((steps_per_env, num_envs), device=self.device)

        obs, _ = envs.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        done = torch.zeros(num_envs, device=self.device)

        total_steps_done = 0
        batch_size = num_envs * steps_per_env

        while total_steps_done < total_steps:
            # rollout
            for t in range(steps_per_env):
                obs_buf[t] = obs
                done_buf[t] = done
                with torch.no_grad():
                    action, logp, _, value = self.get_action_and_value(obs)
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value

                next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
                obs = torch.tensor(next_obs, device=self.device)
                done = torch.tensor(np.logical_or(term, trunc), device=self.device)
                rew_buf[t] = torch.tensor(reward, device=self.device)

            with torch.no_grad():
                _, _, _, last_val = self.get_action_and_value(obs)
            advs, returns = self.compute_gae(rew_buf, val_buf, done_buf, last_val)

            # flatten
            b_obs = obs_buf.reshape(-1, *obs_shape)
            b_act = act_buf.reshape(-1, self.act_dim)
            b_logp = logp_buf.reshape(-1)
            b_rets = returns.reshape(-1)
            b_advs = advs.reshape(-1)

            self.update((b_obs, b_act, b_logp, b_rets, b_advs))

            total_steps_done += batch_size
