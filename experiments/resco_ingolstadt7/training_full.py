import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURAZIONE STILI E FAMIGLIE
# ============================================================
SCENARIO = os.path.basename(os.getcwd())
MIN_EP = 2
SMOOTH_WINDOW = 5

# Definizione dei metadati degli algoritmi
ALGO_CONFIG = {
    "fixed_time": {"color": "black", "ls": "--", "label": "Baseline (Fixed)", "family": "Baseline"},
    "idqn": {"color": "#FF9800", "ls": "-", "label": "IDQN", "family": "Independent"},
    "ippo": {"color": "#F44336", "ls": "-", "label": "IPPO", "family": "Independent"},
    "qmix": {"color": "#9C27B0", "ls": "-", "label": "QMIX", "family": "Value-Decomposition"},
    "mappo": {"color": "#2196F3", "ls": "-", "label": "MAPPO (Base)", "family": "Policy-Gradient"},
    "mappo_atn": {"color": "#03A9F4", "ls": "--", "label": "MAPPO + Attn", "family": "Policy-Gradient"},
    "mappo_gat": {"color": "#00BCD4", "ls": ":", "label": "MAPPO + GAT", "family": "Policy-Gradient"},
}

ALGORITHMS = {
    "idqn": "outputs/sumo_idqn",
    "ippo": "outputs/sumo_ippo",
    "qmix": "outputs/sumo_qmix",
    "mappo": "mappo_env/outputs/sumo_mappo/ingolstadt7",
    "mappo_atn": "mappo_env_att/outputs/sumo_mappo_attention/ingolstadt7",
    "mappo_gat": "gat_mappo/outputs/sumo_mappo_attention/ingostaldt7",
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
# 1. DATA COLLECTION
# ============================================================
summary_rows = []
curve_rows = []

# --- Baseline Fixed Time ---
fixed_time_val = None
if os.path.exists(FIXED_TIME_CSV):
    df_ft = pd.read_csv(FIXED_TIME_CSV)
    fixed_time_val = {
        "scenario": SCENARIO,
        "controller": "fixed_time",
        "family": "Baseline",
        "mean_wait": df_ft["mean_wait"].iloc[0],
        "mean_speed": df_ft["mean_speed"].iloc[0],
        "mean_queue": df_ft["mean_queue"].iloc[0],
    }
    summary_rows.append(fixed_time_val)

# --- RL Algorithms ---
for algo, folder in ALGORITHMS.items():
    if not os.path.exists(folder):
        print(f"⚠️ {algo}: folder non trovata")
        continue

    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    best_wait = float("inf")
    best_row = None

    for csv in csvs:
        ep = extract_ep(os.path.basename(csv))
        if ep is None or ep < MIN_EP: continue

        df = read_sumo_csv(csv)
        if df is None or "system_mean_waiting_time" not in df.columns: continue

        m_wait = df["system_mean_waiting_time"].mean()
        m_speed = df["system_mean_speed"].mean()
        m_queue = df["system_total_stopped"].mean()

        if not np.isfinite(m_wait) or m_wait <= 0: continue

        entry = {
            "scenario": SCENARIO, "controller": algo, "episode": ep,
            "family": ALGO_CONFIG[algo]["family"],
            "mean_wait": m_wait, "mean_speed": m_speed, "mean_queue": m_queue,
        }
        curve_rows.append(entry)

        if m_wait < best_wait:
            best_wait = m_wait
            best_row = entry.copy()
            best_row.pop("episode", None)

    if best_row:
        summary_rows.append(best_row)

df_summary = pd.DataFrame(summary_rows)
df_curves = pd.DataFrame(curve_rows)
df_summary.to_csv(OUT_SUMMARY, index=False)
df_curves.to_csv(OUT_CURVES, index=False)


# ============================================================
# 2. PLOTTING FUNCTIONS
# ============================================================

def setup_plot(title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)


def add_baseline(metric):
    if fixed_time_val and metric in fixed_time_val:
        plt.axhline(y=fixed_time_val[metric], color='black', linestyle='--',
                    linewidth=2, label="Fixed Time Baseline", zorder=2)


# --- A. Grafici Globali ---
def plot_global(metric, ylabel):
    setup_plot(f"Global Comparison: {metric.replace('_', ' ').title()}", "Episode", ylabel)
    add_baseline(metric)

    for algo in df_curves["controller"].unique():
        sub = df_curves[df_curves["controller"] == algo].sort_values("episode")
        y_smooth = sub[metric].rolling(SMOOTH_WINDOW, min_periods=1).mean()

        cfg = ALGO_CONFIG.get(algo, {"color": None, "ls": "-", "label": algo})
        plt.plot(sub["episode"], y_smooth, label=cfg["label"],
                 color=cfg["color"], linestyle=cfg["ls"], linewidth=2)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"global_{metric}.png", dpi=300)
    plt.close()


# --- B. Grafici per Famiglia ---
def plot_by_family(metric, ylabel):
    families = [f for f in df_curves["family"].unique() if f != "Baseline"]

    for fam in families:
        setup_plot(f"Family Focus: {fam} ({metric.replace('_', ' ')})", "Episode", ylabel)
        add_baseline(metric)

        sub_fam = df_curves[df_curves["family"] == fam]
        for algo in sub_fam["controller"].unique():
            sub = sub_fam[sub_fam["controller"] == algo].sort_values("episode")
            y_smooth = sub[metric].rolling(SMOOTH_WINDOW, min_periods=1).mean()

            cfg = ALGO_CONFIG[algo]
            plt.plot(sub["episode"], y_smooth, label=cfg["label"],
                     color=cfg["color"], linestyle=cfg["ls"], linewidth=2.5)

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"family_{fam}_{metric}.png", dpi=200)
        plt.close()


# --- C. Bar Chart di Riepilogo (Best Performers) ---
def plot_best_summary():
    plt.figure(figsize=(10, 6))
    # Ordiniamo per mean_wait (più basso è meglio)
    df_plot = df_summary.sort_values("mean_wait")

    colors = [ALGO_CONFIG[c]["color"] for c in df_plot["controller"]]
    sns.barplot(data=df_plot, x="controller", y="mean_wait", palette=colors)

    plt.title("Best Mean Waiting Time per Controller (Lower is Better)", fontsize=14)
    plt.ylabel("Mean Waiting Time (s)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("best_performance_comparison.png")
    plt.close()


# ============================================================
# ESECUZIONE
# ============================================================
metrics = {
    "mean_wait": "Mean Waiting Time (s)",
    "mean_speed": "Mean Speed (m/s)",
    "mean_queue": "Mean Stopped Vehicles"
}

print("🚀 Generazione grafici in corso...")

for m, label in metrics.items():
    plot_global(m, label)  # 1 grafico globale per metrica
    plot_by_family(m, label)  # N grafici per ogni famiglia

plot_best_summary()  # 1 grafico a barre finale

print(f"\n✅ Analisi completata per lo scenario: {SCENARIO}")
print("📁 File generati:")
print(" - global_*.png (Confronto totale)")
print(" - family_*.png (Focus su singole tipologie di algoritmi)")
print(" - best_performance_comparison.png (Ranking finale)")