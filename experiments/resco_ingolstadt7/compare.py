import os
import glob

import numpy as np
import pandas as pd

print("\n🔍 AVVIO COMPARAZIONE CONTROLLERS")

SCENARIO = os.path.basename(os.getcwd())

FIXED_TIME_CSV = "outputs_fixed_time/fixed_time_metrics.csv"
IPPO_DIR = "outputs/sumo_ippo"
IDQN_DIR = "outputs/sumo_idqn"
MAPPO_DIR = "mappo_env/outputs/sumo_mappo/ingolstadt7"
MAPPO_ATN ="mappo_env_att/outputs/sumo_mappo_attention/ingolstadt7"



OUT_CSV = "comparison_results.csv"

rows = []

# ------------------------------------------------------------
# FIXED TIME
# ------------------------------------------------------------
print("\n▶ Fixed-Time")
if os.path.exists(FIXED_TIME_CSV):
    df = pd.read_csv(FIXED_TIME_CSV)
    print("  ✔ trovato:", FIXED_TIME_CSV)

    rows.append({
        "scenario": SCENARIO,
        "controller": "fixed_time",
        "mean_wait": df["mean_wait"].iloc[0],
        "mean_speed": df["mean_speed"].iloc[0],
        "mean_queue": df["mean_queue"].iloc[0],
    })
else:
    print("  ❌ fixed_time_metrics.csv NON trovato")

# ------------------------------------------------------------
# FUNZIONE SUMO
# ------------------------------------------------------------
def load_best_sumo_metrics(folder, label, min_ep=2):
    print(f"\n▶ {label.upper()} (BEST mean_wait, ep ≥ {min_ep})")

    if not os.path.exists(folder):
        print("  ❌ folder non trovata:", folder)
        return None

    csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csvs:
        print("  ❌ nessun CSV trovato in", folder)
        return None

    best_csv = None
    best_wait = float("inf")
    skipped = 0
    parsed = 0

    for csv in csvs:
        fname = os.path.basename(csv)

        # --------------------------------------------------
        # skip ep0 / ep1
        # --------------------------------------------------
        if "_ep" in fname:
            try:
                ep = int(fname.split("_ep")[-1].split(".")[0])
                if ep < min_ep:
                    skipped += 1
                    continue
            except ValueError:
                pass

        # --------------------------------------------------
        # read CSV (robusto)
        # --------------------------------------------------
        try:
            df = pd.read_csv(csv, engine="python")
        except Exception:
            skipped += 1
            continue

        if "system_mean_waiting_time" not in df.columns:
            skipped += 1
            continue

        mean_wait = df["system_mean_waiting_time"].mean()

        # skip CSV vuoti / NaN / zero sospetti
        if not np.isfinite(mean_wait) or mean_wait <= 0.0:
            skipped += 1
            continue

        parsed += 1

        if mean_wait < best_wait:
            best_wait = mean_wait
            best_csv = csv

    if best_csv is None:
        print(f"  ❌ nessun CSV valido trovato (scartati: {skipped})")
        return None

    print(f"  🏆 BEST CSV: {os.path.basename(best_csv)}")
    print(f"      mean_wait = {best_wait:.3f}")
    print(f"      CSV validi analizzati: {parsed}, scartati: {skipped}")

    df = pd.read_csv(best_csv, engine="python")

    return {
        "scenario": SCENARIO,
        "controller": label,
        "mean_wait": df["system_mean_waiting_time"].mean(),
        "mean_speed": df["system_mean_speed"].mean(),
        "mean_queue": df["system_total_stopped"].mean(),
    }





# ------------------------------------------------------------
# IDQN
# ------------------------------------------------------------
m = load_best_sumo_metrics(IDQN_DIR, "idqn")
if m:
    rows.append(m)

# ------------------------------------------------------------
# IPPO
# ------------------------------------------------------------
m = load_best_sumo_metrics(IPPO_DIR, "ippo")
if m:
    rows.append(m)



# ------------------------------------------------------------
# MAPPO
# ------------------------------------------------------------
m = load_best_sumo_metrics(MAPPO_DIR, "mappo")
if m:
    rows.append(m)


m = load_best_sumo_metrics(MAPPO_ATN, "mappo_atn")
if m:
    rows.append(m)



# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------
if not rows:
    print("\n❌ NESSUNA METRICA RACCOLTA → controlla i path")
    exit(0)

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)

print("\n✅ COMPARAZIONE COMPLETATA")
print(df_out)
print(f"\n📄 CSV salvato in: {OUT_CSV}")
