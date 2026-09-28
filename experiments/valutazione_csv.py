import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMO_OUT_DIR = PROJECT_ROOT / "experiments" / "resco_ingolstadt21" / "outputs" / "sumo"
SUMO_OUT_DIR = Path(os.getenv("SUMO_OUT_DIR", DEFAULT_SUMO_OUT_DIR))


def plot_sumo_metrics() -> None:
    pattern = str(SUMO_OUT_DIR / "*_ep*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found in {SUMO_OUT_DIR}. Set SUMO_OUT_DIR if needed.")
        return

    data_list = []
    for file_path in files:
        try:
            ep_num = int(file_path.split("_ep")[-1].split(".csv")[0])
            df = pd.read_csv(file_path)
            data_list.append(
                {
                    "episode": ep_num,
                    "waiting_time": df["system_mean_waiting_time"].mean(),
                    "speed": df["system_mean_speed"].mean(),
                    "stopped": df["system_total_stopped"].mean(),
                }
            )
        except Exception as exc:
            print(f"Could not process {file_path}: {exc}")

    if not data_list:
        print("No valid episode data found.")
        return

    full_df = pd.DataFrame(data_list).sort_values("episode")
    summary = full_df.groupby("episode").mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(summary["episode"], summary["waiting_time"], marker="o", linewidth=2, label="Waiting Time")
    ax1.set_ylabel("Avg Waiting Time (s)")
    ax1.set_title("IPPO Performance — Ingolstadt 21")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    ax2.plot(summary["episode"], summary["speed"], marker="s", linewidth=2, label="Mean Speed")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Speed (m/s)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    plot_path = SUMO_OUT_DIR.parent / "performance_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

    last_row = summary.iloc[-1]
    print(f"Final episode: {int(last_row['episode'])}")
    print(f"Mean waiting time: {last_row['waiting_time']:.2f} s")
    print(f"Mean speed: {last_row['speed']:.2f} m/s")


if __name__ == "__main__":
    plot_sumo_metrics()
