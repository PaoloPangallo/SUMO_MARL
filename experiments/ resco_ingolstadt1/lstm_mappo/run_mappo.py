# ============================================================
# MAPPO + GAT + TEMPORAL WINDOW (STABLE & RLlib-SAFE)
# ============================================================

import os
import random
import shutil
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import ray
import sumo_rl

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from supersuit import pad_observations_v0, pad_action_space_v0

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

NET_FILE = os.path.join(
    PROJECT_ROOT, "nets", "RESCO", "ingolstadt1", "ingolstadt1.net.xml"
)
ROUTE_FILE = os.path.join(
    PROJECT_ROOT, "nets", "RESCO", "ingolstadt1", "ingolstadt1.rou.xml"
)

OUT_DIR = "outputs"
SUMO_DIR = os.path.join(OUT_DIR, "sumo_mappo_gat_temporal", "ingolstadt1")

if os.path.exists(SUMO_DIR):
    shutil.rmtree(SUMO_DIR)
os.makedirs(SUMO_DIR, exist_ok=True)

# ============================================================
# GLOBAL CONFIG
# ============================================================

BEGIN_TIME = 57600
EP_LEN = 3600
DELTA_T = 10

TRAIN_ITERS = 50
SEED = 42

TEMPORAL_K = 3          # 👈 temporal window size
HIDDEN_DIM = 128

# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================
# TEMPORAL WRAPPER
# ============================================================

from gymnasium.spaces import Box, Dict

class TemporalObsWrapper:
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.buffers = {}

        # 🔑 FIX CHIAVE: observation_space è PER-AGENTE
        agent_ids = env.possible_agents
        assert len(agent_ids) > 0

        sample_agent = agent_ids[0]

        # PettingZoo-style
        space = env.observation_space(sample_agent)

        assert isinstance(space, Box), f"Expected Box, got {type(space)}"

        low = np.repeat(space.low, k, axis=0)
        high = np.repeat(space.high, k, axis=0)

        # RLlib vuole Dict(agent_id -> space)
        self.observation_space = Dict({
            agent: Box(
                low=low,
                high=high,
                dtype=space.dtype,
            )
            for agent in agent_ids
        })

        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.buffers = {
            agent: deque(
                [np.zeros_like(obs[agent]) for _ in range(self.k)],
                maxlen=self.k,
            )
            for agent in obs
        }

        return self._augment(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        return self._augment(obs), rew, term, trunc, info

    def _augment(self, obs):
        out = {}
        for agent, o in obs.items():
            self.buffers[agent].append(o)
            out[agent] = np.concatenate(list(self.buffers[agent]))
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


# ============================================================
# ENV CREATOR
# ============================================================

def env_creator(_):
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        begin_time=BEGIN_TIME,
        num_seconds=EP_LEN,
        delta_time=DELTA_T,
        yellow_time=3,
        min_green=5,
        max_green=60,
        reward_fn="diff-waiting-time",
        out_csv_name=os.path.join(SUMO_DIR, "metrics"),
        use_gui=False,
    )

    # 👇 temporal window PRIMA di RLlib
    env = TemporalObsWrapper(env, TEMPORAL_K)

    # ❗ NIENTE SUPERSUIT
    return ParallelPettingZooEnv(env)


ENV_NAME = "mappo_gat_temporal"
register_env(ENV_NAME, env_creator)

# ============================================================
# GAT CENTRALIZED CRITIC (SIMPLE & STABLE)
# ============================================================

class GATCentralizedCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = obs_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(HIDDEN_DIM, num_outputs)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs_flat"].float()
        h = self.encoder(x)

        logits = self.policy_head(h)
        self._value_out = self.value_head(h).squeeze(-1)

        return logits, []

    def value_function(self):
        return self._value_out

ModelCatalog.register_custom_model(
    "gat_temporal_critic",
    GATCentralizedCritic
)

# ============================================================
# MAPPO CONFIG
# ============================================================

def build_config():
    return (
        PPOConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=600,
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=2400,
            sgd_minibatch_size=400,
            num_sgd_iter=5,
            model={
                "custom_model": "gat_temporal_critic",
                "vf_share_layers": False,
            },
        )
        .multi_agent(
            policies={"shared": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",


        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
    )

# ============================================================
# MAIN
# ============================================================

def main():
    ray.init(ignore_reinit_error=True)
    set_seed(SEED)

    print("\n" + "=" * 45)
    print(f"🚦 MAPPO + GAT + TEMPORAL | Ingolstadt1 | SEED {SEED}")
    print("=" * 45, flush=True)

    algo = build_config().build()

    for it in range(TRAIN_ITERS):
        res = algo.train()
        rew = res.get("episode_reward_mean", np.nan)
        print(f"[Iter {it+1:02d}/{TRAIN_ITERS}] Reward medio: {rew:8.2f}", flush=True)

    algo.stop()
    ray.shutdown()

    print("\n✅ TRAINING COMPLETATO.")

if __name__ == "__main__":
    main()
