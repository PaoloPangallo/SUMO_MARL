import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
SCENARIO = os.path.basename(os.getcwd())
MIN_EP = 2
SMOOTH_WINDOW = 5

ALGORITHMS = {
    "idqn": "outputs/sumo_idqn",
    "ippo": "outputs/sumo_ippo",
    "qmix" : "outputs/sumo_qmix",
    "mappo": "mappo_env/outputs/sumo_mappo/cologne1",
    "mappo_atn": "mappo_env_att/outputs/sumo_mappo_attention/cologne1",

}

FIXED_TIME_CSV = "outputs_fixed_time/fixed_time_metrics.csv"

OUT_SUMMARY = "comparison_summary.csv"
OUT_CURVES = "training_curves.csv"

# ============================================================
# UTILS
# ============================================================
def extract_ep(fname):
    m = re.search(r"_ep(\d+)", fname)
    return int(m.group(1)) if m else None


def read_sumo_csv(csv):
    try:
        return pd.read_csv(csv, engine="python")
    except Exception:
        return None


# ============================================================
# 1️⃣ FIXED TIME
# ============================================================
summary_rows = []

if os.path.exists(FIXED_TIME_CSV):
    df = pd.read_csv(FIXED_TIME_CSV)
    summary_rows.append({
        "scenario": SCENARIO,
        "controller": "fixed_time",
        "mean_wait": df["mean_wait"].iloc[0],
        "mean_speed": df["mean_speed"].iloc[0],
        "mean_queue": df["mean_queue"].iloc[0],
    })

# ============================================================
# 2️⃣ RL CONTROLLERS — CURVE + BEST
# ============================================================
curve_rows = []

for algo, folder in ALGORITHMS.items():
    if not os.path.exists(folder):
        print(f"⚠️ {algo}: folder non trovata")
        continue

    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)

    best_wait = float("inf")
    best_row = None

    for csv in csvs:
        ep = extract_ep(os.path.basename(csv))
        if ep is None or ep < MIN_EP:
            continue

        df = read_sumo_csv(csv)
        if df is None:
            continue

        if "system_mean_waiting_time" not in df.columns:
            continue

        mean_wait = df["system_mean_waiting_time"].mean()
        mean_speed = df["system_mean_speed"].mean()
        mean_queue = df["system_total_stopped"].mean()

        if not np.isfinite(mean_wait) or mean_wait <= 0:
            continue

        # salva per curva
        curve_rows.append({
            "scenario": SCENARIO,
            "controller": algo,
            "episode": ep,
            "mean_wait": mean_wait,
            "mean_speed": mean_speed,
            "mean_queue": mean_queue,
        })

        # BEST (su mean_wait)
        if mean_wait < best_wait:
            best_wait = mean_wait
            best_row = {
                "scenario": SCENARIO,
                "controller": algo,
                "mean_wait": mean_wait,
                "mean_speed": mean_speed,
                "mean_queue": mean_queue,
            }

    if best_row:
        summary_rows.append(best_row)

# ============================================================
# SAVE CSV
# ============================================================
df_summary = pd.DataFrame(summary_rows)
df_curves = pd.DataFrame(curve_rows)

df_summary.to_csv(OUT_SUMMARY, index=False)
df_curves.to_csv(OUT_CURVES, index=False)

print("\n✅ SUMMARY")
print(df_summary)

# ============================================================
# 3️⃣ PLOT CURVE
# ============================================================
def plot_metric(metric, ylabel):
    plt.figure(figsize=(8, 5))

    for algo in df_curves["controller"].unique():
        sub = df_curves[df_curves["controller"] == algo]
        sub = sub.sort_values("episode")

        y = sub[metric].rolling(
            SMOOTH_WINDOW, min_periods=1
        ).mean()

        plt.plot(sub["episode"], y, label=algo.upper())

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{SCENARIO} – {ylabel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"curve_{metric}.png")
    plt.close()


plot_metric("mean_wait", "Mean Waiting Time (s)")
plot_metric("mean_speed", "Mean Speed (m/s)")
plot_metric("mean_queue", "Mean Queue")

print("\n📈 Curve salvate:")
print(" - curve_mean_wait.png")
print(" - curve_mean_speed.png")
print(" - curve_mean_queue.png")
