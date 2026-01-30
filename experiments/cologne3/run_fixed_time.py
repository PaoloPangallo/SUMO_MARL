import os
import csv
import numpy as np
import traci
import sumolib

# ============================================================
# PARAMETRI ESPERIMENTO (ALLINEATI A IPPO)
# ============================================================
BEGIN_TIME = 25200       # usa 0 per sicurezza (traffico presente)
EP_LEN = 3600         # 1 ora
USE_GUI = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne3", "cologne3.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne3", "cologne3.rou.xml")

OUT_DIR = "outputs_fixed_time"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUT_DIR, "fixed_time_metrics.csv")

# ============================================================
# FIXED-TIME CORRETTO (DERIVATO DAL NET)
# ============================================================
FIXED_TIME = {
    "360082": [38, 3, 6, 3, 37, 3],
    "360086": [33, 3, 6, 3, 33, 3, 6, 3],
    "GS_cluster_2415878664_254486231_359566_359576": [33, 3, 6, 3, 33, 3, 6, 3],
}

# ============================================================
# AVVIO SUMO
# ============================================================
sumo_binary = sumolib.checkBinary("sumo-gui" if USE_GUI else "sumo")

traci.start([
    sumo_binary,
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--begin", str(BEGIN_TIME),
    "--end", str(BEGIN_TIME + EP_LEN),
    "--start",
    "--no-step-log",
    "--time-to-teleport", "-1",
])

# ============================================================
# IDENTIFICA SEMAFORI
# ============================================================
tls_ids = traci.trafficlight.getIDList()
print("🚦 TLS trovati:", tls_ids)

# sanity check
for tls in tls_ids:
    assert tls in FIXED_TIME, f"Mancano le fasi per TLS {tls}"

# stato fase per ciascun TLS
tls_phase = {tls: 0 for tls in tls_ids}

# ============================================================
# METRICHE (SUMO-RL / IPPO STYLE)
# ============================================================
wait_list = []
queue_list = []
speed_list = []

sim_time = 0

# ============================================================
# LOOP FIXED-TIME
# ============================================================
while sim_time < EP_LEN:

    # imposta la fase corrente per ciascun TLS
    for tls in tls_ids:
        traci.trafficlight.setPhase(tls, tls_phase[tls])

    # durata sincronizzata (usiamo come clock il TLS semplice)
    ref_tls = "360082"
    phase_duration = FIXED_TIME[ref_tls][tls_phase[ref_tls]]

    for _ in range(phase_duration):
        traci.simulationStep()
        sim_time += 1

        # ===============================
        # METRICHE SISTEMA
        # ===============================
        veh_ids = traci.vehicle.getIDList()

        if veh_ids:
            waits = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            wait_list.append(float(np.mean(waits)))
            speed_list.append(float(np.mean(speeds)))
        else:
            wait_list.append(0.0)
            speed_list.append(0.0)

        # queue = veicoli fermi sugli edge
        stopped = sum(
            traci.edge.getLastStepHaltingNumber(e)
            for e in traci.edge.getIDList()
        )
        queue_list.append(stopped)

        if sim_time >= EP_LEN:
            break

    # advance fase per ogni TLS (modulo corretto)
    for tls in tls_ids:
        tls_phase[tls] = (tls_phase[tls] + 1) % len(FIXED_TIME[tls])

traci.close()

# ============================================================
# RISULTATI FINALI
# ============================================================
mean_wait = float(np.mean(wait_list))
mean_queue = float(np.mean(queue_list))
mean_speed = float(np.mean(speed_list))

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mean_wait", "mean_queue", "mean_speed"])
    writer.writerow([mean_wait, mean_queue, mean_speed])

print("\n✅ FIXED-TIME COMPLETATO (COLOGNE3)")
print(f"Mean wait : {mean_wait:.2f} s")
print(f"Mean queue: {mean_queue:.2f}")
print(f"Mean speed: {mean_speed:.2f} m/s")
print(f"CSV salvato in: {CSV_OUT}")
