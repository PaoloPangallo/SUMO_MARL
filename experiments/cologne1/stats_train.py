import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAZIONE AVANZATA
# ============================================================
SCENARIO = os.path.basename(os.getcwd())
MIN_EP = 2
SMOOTH_WINDOW = 10  # Finestra più ampia per curve più leggibili
LAST_N_EPISODES = 20  # Quanti ultimi episodi usare per il Summary finale (stabilità)

ALGORITHMS = {
    "idqn": "outputs/sumo_idqn",
    "ippo": "outputs/sumo_ippo",
    "qmix": "outputs/sumo_qmix",
    "mappo": "mappo_env/outputs/sumo_mappo/cologne1",
    "mappo_atn": "mappo_env_att/outputs/sumo_mappo_attention/cologne1",
    "mappo_gat": "gat_mappo/outputs/sumo_mappo_attention/cologne1",
}

FIXED_TIME_CSV = "outputs_fixed_time/fixed_time_metrics.csv"
OUT_SUMMARY = "scientific_comparison_summary.csv"
OUT_CURVES = "all_training_data.csv"


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================
def extract_ep(fname):
    m = re.search(r"_ep(\d+)", fname)
    return int(m.group(1)) if m else None


def process_csv(path):
    """Estrae metriche aggregate per l'intero episodio"""
    try:
        df = pd.read_csv(path)
        if df.empty: return None

        # Calcoliamo la media dell'intero episodio per evitare bias da fine-corsa
        return {
            "mean_wait": df["system_mean_waiting_time"].mean(),
            "mean_speed": df["system_mean_speed"].mean(),
            "mean_queue": df["system_total_stopped"].mean(),
            "max_queue": df["system_total_stopped"].max(),
            "efficiency_score": df["system_mean_speed"].mean() / (df["system_mean_waiting_time"].mean() + 1e-5)
        }
    except Exception as e:
        return None


# ============================================================
# CORE PROCESSING
# ============================================================
all_data = []

# 1. Caricamento Baselines (Fixed Time)
if os.path.exists(FIXED_TIME_CSV):
    ft_df = pd.read_csv(FIXED_TIME_CSV)
    baseline_wait = ft_df["mean_wait"].iloc[0]
    print(f"ℹ️ Baseline Fixed Time caricata: {baseline_wait:.2f}s")

# 2. Elaborazione RL
for algo, folder in ALGORITHMS.items():
    if not os.path.exists(folder):
        print(f"⚠️ {algo}: cartella non trovata")
        continue

    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    print(f"Processing {algo}: {len(csvs)} files trovati...")

    for csv_path in csvs:
        ep = extract_ep(os.path.basename(csv_path))
        if ep is None or ep < MIN_EP: continue

        metrics = process_csv(csv_path)
        if metrics:
            all_data.append({
                "scenario": SCENARIO,
                "controller": algo,
                "episode": ep,
                **metrics
            })

df_all = pd.DataFrame(all_data)

# ============================================================
# ANALISI SCIENTIFICA (Summary degli ultimi N episodi)
# ============================================================
summary_list = []

for algo in df_all["controller"].unique():
    algo_data = df_all[df_all["controller"] == algo].sort_values("episode")

    # Prendiamo gli ultimi N episodi per vedere la performance a regime (convergenza)
    last_n = algo_data.tail(LAST_N_EPISODES)

    summary_list.append({
        "Algorithm": algo.upper(),
        "Mean Wait (Last N)": last_n["mean_wait"].mean(),
        "Std Wait": last_n["mean_wait"].std(),
        "Mean Speed (Last N)": last_n["mean_speed"].mean(),
        "Mean Queue (Last N)": last_n["mean_queue"].mean(),
        "Max Convergence Ep": algo_data["episode"].max()
    })

df_summary = pd.DataFrame(summary_list).sort_values("Mean Wait (Last N)")
df_summary.to_csv(OUT_SUMMARY, index=False)
df_all.to_csv(OUT_CURVES, index=False)


# ============================================================
# VISUALIZZAZIONE AVANZATA
# ============================================================
def plot_enhanced_metrics(metric_name, title, ylabel):
    plt.figure(figsize=(12, 6))

    for algo in df_all["controller"].unique():
        subset = df_all[df_all["controller"] == algo].sort_values("episode")

        # Calcolo Media Mobile e Deviazione Standard Mobile
        rolling_mean = subset[metric_name].rolling(window=SMOOTH_WINDOW).mean()
        rolling_std = subset[metric_name].rolling(window=SMOOTH_WINDOW).std()

        line, = plt.plot(subset["episode"], rolling_mean, label=algo.upper(), linewidth=2)
        # Area ombreggiata per la varianza (indica stabilità dell'algoritmo)
        plt.fill_between(subset["episode"],
                         rolling_mean - rolling_std,
                         rolling_mean + rolling_std,
                         alpha=0.15)

    if os.path.exists(FIXED_TIME_CSV) and metric_name == "mean_wait":
        plt.axhline(y=baseline_wait, color='r', linestyle='--', label="FIXED TIME (Baseline)")

    plt.title(f"{SCENARIO.upper()} - {title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"analysis_{metric_name}.png", dpi=300)
    plt.show()


# Generazione Grafici
plot_enhanced_metrics("mean_wait", "Andamento Tempo di Attesa", "Seconds (s)")
plot_enhanced_metrics("mean_speed", "Velocità Media di Sistema", "Speed (m/s)")
plot_enhanced_metrics("mean_queue", "Lunghezza Media Code", "Number of Vehicles")

print("\n🚀 ANALISI COMPLETATA")
print(df_summary)