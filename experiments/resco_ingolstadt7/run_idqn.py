import os
import random
import numpy as np
import ray
import sumo_rl
import torch
import shutil

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.rou.xml")

OUT_DIR = "outputs"
SUMO_DIR = os.path.join(OUT_DIR, "sumo_idqn")

if os.path.exists(SUMO_DIR):
    shutil.rmtree(SUMO_DIR)
os.makedirs(SUMO_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
BEGIN_TIME = 57600
EP_LEN = 3600
DELTA_T = 10
TRAIN_ITERS = 50
CURRENT_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ENV
# ============================================================
from supersuit import pad_observations_v0, pad_action_space_v0

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
        out_csv_name=os.path.join(SUMO_DIR, "idqn_test"),
        use_gui=False,
    )
    env = pad_observations_v0(env)
    env = pad_action_space_v0(env)
    return ParallelPettingZooEnv(env)


ENV_NAME = "idqn_quick_test"
register_env(ENV_NAME, env_creator)


# ============================================================
# DQN CONFIG (IDQN)
# ============================================================
def build_config():
    return (
        DQNConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,
            rollout_fragment_length=900,
        )

        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=3600,
            model={"fcnet_hiddens": [128, 128]},
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 100_000,
            },
            target_network_update_freq=1000,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 20_000,
            }
        )
        .multi_agent(
            policies={"shared": (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared",
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )


# ============================================================
# MAIN
# ============================================================
def main():
    ray.init(ignore_reinit_error=True)

    print("\n" + "=" * 40)
    print(f"🚦 IDQN QUICK TEST - SEED {CURRENT_SEED}")
    print("=" * 40, flush=True)

    set_seed(CURRENT_SEED)
    algo = build_config().build()

    for it in range(TRAIN_ITERS):
        res = algo.train()
        reward = res.get("episode_reward_mean", np.nan)

        print(
            f"[Iter {it + 1:02d}/50] "
            f"Reward Medio: {reward:8.2f}",
            flush=True
        )

    algo.stop()
    ray.shutdown()
    print("\n✅ Test IDQN completato.")


if __name__ == "__main__":
    main()
