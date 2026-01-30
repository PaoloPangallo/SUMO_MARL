import os
import random
import numpy as np
import ray
import sumo_rl
import torch
import shutil
from gymnasium.spaces import Tuple


from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from supersuit import pad_observations_v0, pad_action_space_v0

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.rou.xml")

OUT_DIR = "outputs"
SUMO_DIR = os.path.join(OUT_DIR, "sumo_qmix")

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
# ENV (QMIX requires AGENT GROUPS)
# ============================================================
def env_creator(config):
    base_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        begin_time=BEGIN_TIME,
        num_seconds=EP_LEN,
        delta_time=DELTA_T,
        yellow_time=3,
        min_green=5,
        max_green=60,
        reward_fn="diff-waiting-time",
        out_csv_name=os.path.join(SUMO_DIR, "qmix"),
        use_gui=False,
    )

    base_env = pad_observations_v0(base_env)
    base_env = pad_action_space_v0(base_env)

    # ---- agent ids ----
    agent_ids = list(base_env.possible_agents)

    # ---- build Tuple spaces ----
    obs_spaces = [base_env.observation_space(a) for a in agent_ids]
    act_spaces = [base_env.action_space(a) for a in agent_ids]

    grouped_obs_space = Tuple(obs_spaces)
    grouped_act_space = Tuple(act_spaces)

    # ---- wrap for RLlib ----
    env = ParallelPettingZooEnv(base_env)

    env = env.with_agent_groups(
        groups={"all_tls": agent_ids},
        obs_space=grouped_obs_space,
        act_space=grouped_act_space,
    )

    return env



ENV_NAME = "qmix_ingolstadt1"
register_env(ENV_NAME, env_creator)


# ============================================================
# QMIX CONFIG
# ============================================================
def build_qmix_config():
    return (
        QMixConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .rollouts(
            num_rollout_workers=2,
            rollout_fragment_length=900,
        )
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=3600,
            mixer="qmix",              # per VDN: "vdn"
            double_q=True,
            target_network_update_freq=1000,
        )
        .resources(num_gpus=0)
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .debugging(log_level="ERROR")
    )


# ============================================================
# MAIN
# ============================================================
def main():
    ray.init(ignore_reinit_error=True)

    print("\n" + "=" * 40)
    print(f"🚦 QMIX Ingolstadt1 (GROUPED) - SEED {CURRENT_SEED}")
    print("=" * 40, flush=True)

    set_seed(CURRENT_SEED)

    algo = build_qmix_config().build()

    for it in range(TRAIN_ITERS):
        res = algo.train()
        reward = res.get("episode_reward_mean", np.nan)

        print(
            f"[Iter {it + 1:02d}/{TRAIN_ITERS}] "
            f"Reward Medio: {reward:8.2f}",
            flush=True
        )

    algo.stop()
    ray.shutdown()
    print("\n✅ QMIX Ingolstadt1 completato.")


if __name__ == "__main__":
    main()
