import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurazione path
SUMO_OUT_DIR = r"C:\Users\paolo\SUMORL\experiments\resco_ingolstadt21\outputs\sumo"


def plot_sumo_metrics():
    # 1. Recupera tutti i file CSV che finiscono con _epX.csv
    # Usiamo un pattern che matchi i file generati: ippo_test_connX_epY.csv
    pattern = os.path.join(SUMO_OUT_DIR, "*_ep*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"❌ Nessun file trovato in {SUMO_OUT_DIR}. Controlla il path.")
        return

    # 2. Estraiamo i dati aggregandoli per numero di episodio
    data_list = []
    for f in files:
        try:
            # Estraiamo il numero dell'episodio dal nome del file
            # Esempio: ..._ep10.csv -> 10
            ep_num = int(f.split("_ep")[-1].split(".csv")[0])
            df = pd.read_csv(f)

            # Calcoliamo le medie dell'intero episodio (No Warmup come richiesto)
            data_list.append({
                "episode": ep_num,
                "waiting_time": df["system_mean_waiting_time"].mean(),
                "speed": df["system_mean_speed"].mean(),
                "stopped": df["system_total_stopped"].mean()
            })
        except Exception as e:
            print(f"⚠️ Errore nel processare {f}: {e}")

    # 3. Creiamo un DataFrame e ordiniamo per episodio
    full_df = pd.DataFrame(data_list).sort_values("episode")

    # Raggruppiamo per episodio (nel caso ci siano più worker per lo stesso episodio)
    summary = full_df.groupby("episode").mean().reset_index()

    # 4. Generazione dei Grafici
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Grafico Waiting Time
    ax1.plot(summary["episode"], summary["waiting_time"], color='red', marker='o', linewidth=2, label='Waiting Time')
    ax1.set_ylabel("Avg Waiting Time (s)", fontsize=12, fontweight='bold')
    ax1.set_title("Evoluzione Performance IPPO - Ingolstadt 1", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Grafico Speed
    ax2.plot(summary["episode"], summary["speed"], color='blue', marker='s', linewidth=2, label='Mean Speed')
    ax2.set_xlabel("Episodio", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Avg Speed (m/s)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()

    # Salva il grafico
    plot_path = os.path.join(os.path.dirname(SUMO_OUT_DIR), "performance_plot.png")
    plt.savefig(plot_path)
    print(f"\n✅ Grafico salvato in: {plot_path}")
    plt.show()

    # Mostra i valori finali
    last_row = summary.iloc[-1]
    print("\n--- RISULTATI ULTIMO EPISODIO ---")
    print(f"Episodio: {int(last_row['episode'])}")
    print(f"Waiting Time Medio: {last_row['waiting_time']:.2f} s")
    print(f"Velocità Media: {last_row['speed']:.2f} m/s")


if __name__ == "__main__":
    plot_sumo_metrics()