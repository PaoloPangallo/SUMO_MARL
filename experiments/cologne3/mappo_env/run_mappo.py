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
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne3", "cologne3.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne3", "cologne3.rou.xml")

OUT_DIR = "outputs"
SUMO_DIR = os.path.join(OUT_DIR, "sumo_mappo", "cologne3")

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
        out_csv_name=os.path.join(SUMO_DIR, "mappo"),
        use_gui=False,
    )

    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)

    return ParallelPettingZooEnv(env)


ENV_NAME = "mappo_cologne3"
register_env(ENV_NAME, env_creator)

# ============================================================
# REGISTER CENTRALIZED CRITIC
# ============================================================
from centralized_critic import CentralizedCriticModel

ModelCatalog.register_custom_model("centralized_critic", CentralizedCriticModel)

# ============================================================
# MAPPO CONFIG
# ============================================================
def build_config():
    return (
        PPOConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            batch_mode="complete_episodes",
            rollout_fragment_length=900,
            num_rollout_workers=4,
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=3600,
            sgd_minibatch_size=600,
            num_sgd_iter=10,
            model={
                "custom_model": "centralized_critic",
                "vf_share_layers": False,
                "custom_model_config": {
                    # 1 agenti * obs_dim (dopo padding)
                    "critic_obs_space": 3 * env_creator({}).observation_space.shape[0],
                },
            },
        )
        .multi_agent(
            policies={"shared": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
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

    print("\n" + "=" * 40)
    print(f"🚦 MAPPO Cologne - SEED {CURRENT_SEED}")
    print("=" * 40, flush=True)

    set_seed(CURRENT_SEED)
    algo = build_config().build()

    for it in range(TRAIN_ITERS):
        res = algo.train()
        reward = res.get("episode_reward_mean", np.nan)

        print(
            f"[Iter {it + 1:02d}/100] "
            f"Reward Medio: {reward:8.2f}",
            flush=True
        )

    algo.stop()
    ray.shutdown()
    print("\n✅ MAPPO completato.")


if __name__ == "__main__":
    main()
