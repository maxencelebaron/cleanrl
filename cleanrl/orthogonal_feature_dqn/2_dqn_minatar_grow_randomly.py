# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import collections
import copy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.minatar_wrappers import RenderFix
from cleanrl_utils.plot_plasticity_scatter import (
    scatter_cross_runs,
    scatter_within_runs
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MinAtar/Breakout-v1"
    """the id of the environment. It can be one of the following:
        Asterix-v0, Breakout-v0, Freeway-v0, Seaquest-v0, SpaceInvaders-v0,
        Asterix-v1, Breakout-v1, Freeway-v1, Seaquest-v1, SpaceInvaders-v1.
    """
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10_000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    feature_rank_n_states: int = 2_000
    """number of states to collect for periodic feature rank monitoring"""
    return_window_size: int = 100
    """number of episodes for the rolling mean of episodic return"""
    compute_final_feature_rank: bool = False
    """if toggled, compute feature rank over a policy rollout after training"""
    # final_feature_rank_n_states: int = 10_000
    # """number of states to collect for the final feature rank computation"""
    plasticity_n_steps: int = 2_000
    """number of gradient steps per probe task in plasticity measurement"""
    plasticity_n_tasks: int = 10
    """number of random probe tasks for plasticity measurement"""
    plasticity_n_samples: int = 1_000
    """number of replay buffer samples for plasticity measurement"""
    plasticity_final_n_tasks: int = 50
    """number of random probe tasks for the final plasticity measurement"""
    initial_hidden: int = 8
    """initial encoder hidden size; grows to 128 over n_growth_steps events"""
    n_growth_steps: int = 15
    """number of growth events evenly spaced over training"""
    gradient_steps: int = 4
    """number of gradient steps per training call"""
    eval_frequency: int = 10_000
    """number of environment steps between greedy policy evaluations"""
    eval_episodes: int = 20
    """number of episodes per greedy evaluation"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = RenderFix(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_size: int = 128):
        super().__init__()
        obs_shape = env.single_observation_space.shape  # (H, W, C)
        n_channels = obs_shape[-1]

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy = torch.zeros(1, n_channels, obs_shape[0], obs_shape[1])
        flat_size = int(np.prod(self.conv(dummy).shape[1:]))

        self.flat = nn.Flatten()

        self.encoder = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size, env.single_action_space.n)

    def encode(self, x):
        x = x.permute(0, 3, 1, 2).float()  # (B, H, W, C) -> (B, C, H, W)
        return self.encoder(self.flat(self.conv(x)))

    def forward(self, x):
        return self.q_head(self.encode(x))


def feature_rank(features: np.ndarray, epsilon: float = 0.01):
    """
    Number of singular values of (1/√n)·φ(X) above epsilon where φ(X) is
    the feature matrix.
    """
    phi = features / np.sqrt(features.shape[0])
    singular_values = np.linalg.svd(phi, compute_uv=False)
    rank = int(np.sum(singular_values > epsilon))
    return rank, singular_values


def srank(singular_values: np.ndarray, delta: float = 0.01) -> int:
    """
    Effective rank: smallest k such that the top-k singular values account
    for at least (1 - delta) of the total singular value mass.

    Operates directly on pre-computed singular values (sorted descending),
    so no extra SVD is needed when called after feature_rank().
    """
    total = singular_values.sum()
    if total == 0:
        return 0
    cumulative_ratio = np.cumsum(singular_values) / total
    k = int(np.searchsorted(cumulative_ratio, 1.0 - delta, side="left")) + 1
    return min(k, len(singular_values))


def gradient_metrics(
    network: QNetwork,
    target_network: QNetwork,
    replay_buffer,
    batch_size: int,
    gamma: float,
) -> dict:
    """
    Compute per-layer gradient L1/L2 norms and singular value spectrum of the
    weight gradient matrix for each linear layer.

    Uses a fresh forward+backward on a dedicated replay batch.
    Does not call optimizer.step(), so network weights are unchanged.
    """
    network.zero_grad()

    data = replay_buffer.sample(batch_size)
    with torch.no_grad():
        target_max, _ = target_network(data.next_observations).max(dim=1)
        td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
    old_val = network(data.observations).gather(1, data.actions).squeeze()
    F.mse_loss(td_target, old_val).backward()

    # group flattened gradients and 2D weight grads by top-level module name
    layer_grads: dict[str, list] = {}
    layer_weight_grads: dict[str, torch.Tensor] = {}
    for name, param in network.named_parameters():
        if param.grad is None:
            continue
        layer = name.split(".")[0]
        layer_grads.setdefault(layer, []).append(param.grad.detach().flatten())
        if param.grad.dim() == 2 and name.endswith(".weight"):
            layer_weight_grads[layer] = param.grad.detach()

    result = {}
    for layer, grads in layer_grads.items():
        g = torch.cat(grads)
        result[f"{layer}/grad_l1"] = g.norm(1).item()
        result[f"{layer}/grad_l2"] = g.norm(2).item()

    for layer, g in layer_weight_grads.items():
        result[f"{layer}/grad_svs"] = torch.linalg.svdvals(g).cpu().numpy()

    network.zero_grad()
    return result


def collect_features(
    model: QNetwork,
    env,
    device,
    n_states: int = 10_000
):
    """
    Roll out the current policy and collect encoder activations
    over n_states steps.
    """
    features = []
    obs, _ = env.reset()
    while len(features) < n_states:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode(obs_tensor)
        features.append(feat.squeeze(0).cpu().numpy())
        action = torch.argmax(model(obs_tensor), dim=1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(features[:n_states])


def collect_features_episode(
    model: QNetwork,
    env,
    device,
):
    """
    Roll out the current policy for one complete episode
    and collect encoder activations.
    """
    features = []
    obs, _ = env.reset()
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode(obs_tensor)
        features.append(feat.squeeze(0).cpu().numpy())
        action = torch.argmax(model(obs_tensor), dim=1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.array(features)


def measure_plasticity(
    network: QNetwork,
    replay_buffer,
    optimizer_class=optim.Adam,
    optimizer_params: dict = None,
    n_steps: int = 2_000,
    n_tasks: int = 10,
    n_samples: int = 1_000,
) -> float:
    """
    Plasticity P(θ) = mean(b_k) - mean(l_k) over n_tasks random probe tasks.

    For each task k:
      - sample random init ω₀
      - compute targets y = a + sin(1e5 · f(x; ω₀))  where a = E[f(x; θ_t)]
      - b_k = Var(y)  (target variance, measures task difficulty)
      - fine-tune a probe copy θ_probe of θ_t for n_steps on MSE(f(x; θ_probe), y)
      - l_k = MSE(f(x; θ_probe), y) after fine-tuning  (residual loss)

    Higher P -> more plastic (can reduce loss further relative to task difficulty).
    """
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-4}

    # shared batch across all tasks
    data = replay_buffer.sample(n_samples)
    x = data.observations  # (n_samples, H, W, C)

    # anchor: mean output of current network, no grad  (n_actions,)
    with torch.no_grad():
        a = network(x).mean(dim=0)  # (n_actions,)

    b_list, l_list = [], []
    for _ in range(n_tasks):
        # ω_0 sampled from the same distribution as θ_0 via reset_parameters()
        omega_0 = copy.deepcopy(network)
        for m in omega_0.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        # targets y = a + sin(1e5 · f(x; ω_0))
        with torch.no_grad():
            y = a + torch.sin(1e5 * omega_0(x))  # (n_samples, n_actions)

        b_k = y.var().item()
        b_list.append(b_k)

        # fine-tune a copy of θ_t on this task
        probe = copy.deepcopy(network)
        opt = optimizer_class(probe.parameters(), **optimizer_params)
        for _ in range(n_steps):
            loss = F.mse_loss(probe(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            l_k = F.mse_loss(probe(x), y).item()
        l_list.append(l_k)

    return float(np.mean(b_list) - np.mean(l_list))


def principal_angle_cosines(phi: np.ndarray, phi_new: np.ndarray) -> np.ndarray:
    """
    Cosines of principal angles between column spaces of phi and phi_new.
    Values near 1 = aligned (new neurons replicate existing features).
    Values near 0 = orthogonal (new neurons learn independent features).
    """
    U, _, _ = np.linalg.svd(phi, full_matrices=False)
    V, _, _ = np.linalg.svd(phi_new, full_matrices=False)
    return np.linalg.svd(U.T @ V, compute_uv=False)


def bellman_residual_correlation(phi_new: np.ndarray, R: np.ndarray) -> float:
    """
    Correlation between new features and accumulated Bellman residuals.
    High value = new neurons capture unexplained residual.
    """
    return float(np.linalg.norm(phi_new.T @ R, 'fro'))


def evaluate_policy(net: QNetwork, env, device, n_episodes: int) -> float:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        while True:
            with torch.no_grad():
                action = torch.argmax(
                    net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)),
                    dim=1,
                ).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        returns.append(ep_return)
    return float(np.mean(returns))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def grow_network(
    old_net: QNetwork,
    env,
    device,
    new_hidden: int,
) -> QNetwork:
    """
    Expand QNetwork hidden layer from old_h to new_hidden. Conv stays fixed.

    Old neurons are copied unchanged. New neurons keep their random init from
    QNetwork.__init__ (both encoder rows and q_head columns).
    The optimizer must be rebuilt after calling this function.
    """
    new_net = QNetwork(env, hidden_size=new_hidden).to(device)

    old_h = old_net.q_head.in_features

    with torch.no_grad():
        new_net.conv[0].weight.copy_(old_net.conv[0].weight)
        new_net.conv[0].bias.copy_(old_net.conv[0].bias)

        # encoder[0]: copy old rows, new rows keep random init
        new_net.encoder[0].weight[:old_h, :].copy_(old_net.encoder[0].weight)
        new_net.encoder[0].bias[:old_h].copy_(old_net.encoder[0].bias)

        # q_head: copy old columns, new columns keep random init
        new_net.q_head.weight[:, :old_h].copy_(old_net.q_head.weight)
        new_net.q_head.bias.copy_(old_net.q_head.bias)

    return new_net


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    eval_env = gym.make(args.env_id)

    _FINAL_HIDDEN = 128

    q_network = QNetwork(envs, hidden_size=args.initial_hidden).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, hidden_size=args.initial_hidden).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # growth_schedule: list of (global_step_threshold, new_hidden)
    growth_schedule = [
        (
            args.total_timesteps * k // (args.n_growth_steps + 1),
            args.initial_hidden + (_FINAL_HIDDEN - args.initial_hidden) * k // args.n_growth_steps,
        )
        for k in range(1, args.n_growth_steps + 1)
    ] if args.n_growth_steps > 0 else []
    next_growth_idx = 0

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    return_window = collections.deque(maxlen=args.return_window_size)
    initial_plasticity = None

    feature_split = 0  # hidden size just before last growth; 0 = no growth yet
    monitoring_obs = None    # fixed observation set for fair cross-checkpoint comparison
    monitoring_batch = None  # fixed transition batch (includes next_obs/rewards/actions)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    return_window.append(float(episodic_return))
                    if len(return_window) == args.return_window_size:
                        writer.add_scalar("charts/return_mean", np.mean(return_window), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # GROW: expand network if we've reached the next growth threshold
        if next_growth_idx < len(growth_schedule):
            grow_at, new_nh = growth_schedule[next_growth_idx]
            if global_step >= grow_at:
                feature_split = q_network.q_head.in_features
                q_network = grow_network(q_network, envs, device, new_nh)
                target_network = copy.deepcopy(q_network)
                optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
                writer.add_scalar("growing/hidden_size", new_nh, global_step)
                print(f"global_step={global_step}: grew network to hidden={new_nh}")
                next_growth_idx += 1

        # EVAL: periodic greedy evaluation
        if global_step % args.eval_frequency == 0 and global_step > 0:
            mean_return = evaluate_policy(q_network, eval_env, device, args.eval_episodes)
            writer.add_scalar("charts/mean_episodic_return_per_policy", mean_return, global_step)
            print(f"global_step={global_step}, eval_mean_return={mean_return:.2f}")

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

                for _ in range(args.gradient_steps):
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                if global_step % 20000 == 0:
                    if monitoring_obs is None:
                        monitoring_batch = rb.sample(args.feature_rank_n_states)
                        monitoring_obs = monitoring_batch.observations
                    with torch.no_grad():
                        features = q_network.encode(monitoring_obs).cpu().numpy()
                    rank, svs = feature_rank(features)
                    feature_dim = features.shape[1]
                    writer.add_scalar("features/rank", rank, global_step)
                    writer.add_scalar("features/rank_ratio", rank / feature_dim, global_step)
                    writer.add_histogram("features/singular_values", svs, global_step)

                    sr = srank(svs)
                    writer.add_scalar("features/srank", sr, global_step)
                    writer.add_scalar("features/srank_ratio", sr / feature_dim, global_step)
                    print(f"global_step={global_step}, rank={rank}, rank_ratio={rank / feature_dim:.2f}, srank={sr}, srank_ratio={sr / feature_dim:.2f}")

                    if 0 < feature_split < feature_dim:
                        phi = features[:, :feature_split]
                        phi_new = features[:, feature_split:]

                        rank_old, svs_old = feature_rank(phi)
                        rank_new, svs_new = feature_rank(phi_new)
                        writer.add_scalar("growing/rank_phi_old", rank_old, global_step)
                        writer.add_scalar("growing/rank_phi_new", rank_new, global_step)
                        writer.add_scalar("growing/rank_ratio_phi_old", rank_old / phi.shape[1], global_step)
                        writer.add_scalar("growing/rank_ratio_phi_new", rank_new / phi_new.shape[1], global_step)

                        sr_old = srank(svs_old)
                        sr_new = srank(svs_new)
                        writer.add_scalar("growing/srank_phi_old", sr_old, global_step)
                        writer.add_scalar("growing/srank_phi_new", sr_new, global_step)
                        writer.add_scalar("growing/srank_ratio_phi_old", sr_old / phi.shape[1], global_step)
                        writer.add_scalar("growing/srank_ratio_phi_new", sr_new / phi_new.shape[1], global_step)

                        cosines = principal_angle_cosines(phi, phi_new)
                        writer.add_histogram("growing/principal_angle_cosines", cosines, global_step)
                        writer.add_scalar("growing/angles_min", cosines.min(), global_step)
                        writer.add_scalar("growing/angles_max", cosines.max(), global_step)
                        writer.add_scalar("growing/angles_mean", cosines.mean(), global_step)

                        with torch.no_grad():
                            target_max_brc, _ = target_network(monitoring_batch.next_observations).max(dim=1)
                            td_brc = (monitoring_batch.rewards.flatten() + args.gamma * target_max_brc * (1 - monitoring_batch.dones.flatten())).cpu().numpy()
                            q_brc = q_network(monitoring_batch.observations).gather(1, monitoring_batch.actions).squeeze().cpu().numpy()
                        r_brc = (td_brc - q_brc).reshape(-1, 1)
                        brc = bellman_residual_correlation(phi_new, r_brc)
                        writer.add_scalar("growing/bellman_residual_correlation", brc, global_step)
                        print(f"global_step={global_step}, rank_old={rank_old}, rank_new={rank_new}, brc={brc:.4f}")

                    grad_m = gradient_metrics(
                        q_network,
                        target_network,
                        rb,
                        batch_size=args.plasticity_n_samples,
                        gamma=args.gamma,
                    )
                    for key, val in grad_m.items():
                        if key.endswith("/grad_svs"):
                            writer.add_histogram(f"gradients/{key}", val, global_step)
                        else:
                            writer.add_scalar(f"gradients/{key}", val, global_step)

                    plasticity = measure_plasticity(
                        q_network,
                        rb,
                        optimizer_class=optim.Adam,
                        optimizer_params={"lr": args.learning_rate},
                        n_steps=args.plasticity_n_steps,
                        n_tasks=args.plasticity_n_tasks,
                        n_samples=args.plasticity_n_samples,
                    )
                    writer.add_scalar("charts/plasticity", plasticity, global_step)
                    if initial_plasticity is None:
                        initial_plasticity = plasticity
                    plasticity_loss = initial_plasticity - plasticity
                    writer.add_scalar("charts/plasticity_loss", plasticity_loss, global_step)
                    print(f"global_step={global_step}, plasticity={plasticity:.4f}, plasticity_loss={plasticity_loss:.4f}")

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    if args.compute_final_feature_rank:
        eval_env = gym.make(args.env_id)
        final_features = collect_features_episode(
            q_network,
            eval_env,
            device
        )
        final_rank, final_svs = feature_rank(final_features)
        writer.add_scalar("features/rank", final_rank, args.total_timesteps)
        writer.add_scalar("features/rank_ratio", final_rank / feature_dim, args.total_timesteps)
        final_sr = srank(final_svs)
        writer.add_scalar("features/srank", final_sr, args.total_timesteps)
        writer.add_scalar("features/srank_ratio", final_sr / feature_dim, args.total_timesteps)
        print(f"final_feature_rank={final_rank} (over {len(final_features)} states)")
        eval_env.close()

        final_plasticity = measure_plasticity(
            q_network,
            rb,
            optimizer_class=optim.Adam,
            optimizer_params={"lr": args.learning_rate},
            n_steps=args.plasticity_n_steps,
            n_tasks=args.plasticity_final_n_tasks,
            n_samples=args.plasticity_n_samples,
        )
        writer.add_scalar("charts/plasticity", final_plasticity, args.total_timesteps)
        if initial_plasticity is not None:
            final_plasticity_loss = initial_plasticity - final_plasticity
            writer.add_scalar("charts/plasticity_loss", final_plasticity_loss, args.total_timesteps)
        print(f"final_plasticity={final_plasticity:.4f}")

    eval_env.close()
    envs.close()
    writer.close()

    run_dir = f"runs/{run_name}"
    scatter_within_runs(
        run_dir,
        output_path=f"{run_dir}/plasticity_vs_rank_checkpoints.png"
    )

    if args.compute_final_feature_rank:
        scatter_cross_runs(
            [run_dir],
            output_path=f"{run_dir}/plasticity_vs_rank_final.png"
        )
