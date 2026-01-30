import os
import random
import numpy as np
import ray
import sumo_rl
import torch
import shutil

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from supersuit import pad_observations_v0, pad_action_space_v0

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

NET_FILE = os.path.join(
    PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.net.xml"
)
ROUTE_FILE = os.path.join(
    PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.rou.xml"
)

OUT_DIR = "outputs"
SUMO_DIR = os.path.join(OUT_DIR, "sumo_mappo_attention", "cologne1")

if os.path.exists(SUMO_DIR):
    shutil.rmtree(SUMO_DIR)
os.makedirs(SUMO_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
BEGIN_TIME = 25200
EP_LEN = 3600
DELTA_T = 10

TRAIN_ITERS = 100
CURRENT_SEED = 42
NUM_AGENTS = 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ENV
# ============================================================
def env_creator(config):
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
        out_csv_name=os.path.join(SUMO_DIR, "mappo_attention"),
        use_gui=False,
    )

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)

    return ParallelPettingZooEnv(env)


ENV_NAME = "mappo_ingolstadt7_attention"
register_env(ENV_NAME, env_creator)

# ============================================================
# MODEL REGISTRATION
# ============================================================
from gat_critic import  GATCentralizedCritic

ModelCatalog.register_custom_model(
    "gat_centralized_critic",
    GATCentralizedCritic,
)

# ============================================================
# MAPPO CONFIG
# ============================================================
def build_config():
    # ricaviamo obs_dim UNA volta
    tmp_env = env_creator({})
    obs_dim = tmp_env.observation_space.shape[0]

    return (
        PPOConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=900,
            batch_mode="truncate_episodes",
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=3600,
            sgd_minibatch_size=500,
            num_sgd_iter=5,
            model={
                "custom_model": "gat_centralized_critic",
                "vf_share_layers": False,
                "custom_model_config": {
                    "num_agents": NUM_AGENTS,  # ✅ nome corretto
                    "embed_dim": 128,
                    "n_heads": 4,
                },
            },
        )
        .multi_agent(
            policies={"shared": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        # 🔒 API legacy (stabile)
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )


# ============================================================
# MAIN
# ============================================================
def main():
    ray.init(ignore_reinit_error=True)

    print("\n" + "=" * 60)
    print(f"🚦 MAPPO + GAT Attention | Ingolstadt7 | SEED {CURRENT_SEED}")
    print("=" * 60, flush=True)

    set_seed(CURRENT_SEED)
    algo = build_config().build()

    for it in range(TRAIN_ITERS):
        res = algo.train()
        reward = res.get("episode_reward_mean", np.nan)

        print(
            f"[Iter {it + 1:02d}/{TRAIN_ITERS}] "
            f"Reward Medio: {reward:8.2f}",
            flush=True,
        )

    algo.stop()
    ray.shutdown()
    print("\n✅ MAPPO + Attention (GAT) completato.")


if __name__ == "__main__":
    main()
