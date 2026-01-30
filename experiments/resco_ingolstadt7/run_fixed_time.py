import os
import csv
import numpy as np
import traci
import sumolib

# ============================================================
# PARAMETRI ESPERIMENTO (COERENTI CON IPPO / MAPPO)
# ============================================================
BEGIN_TIME = 57600        # stesso begin di IPPO
EP_LEN = 3600             # 1 ora
USE_GUI = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.rou.xml")

OUT_DIR = "outputs_fixed_time"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUT_DIR, "fixed_time_metrics.csv")

# ============================================================
# FIXED-TIME (DERIVATO 1:1 DA inspect.py)
# ============================================================
FIXED_TIME = {
    "32564122": [42, 3, 42, 3],

    "cluster_1757124350_1757124352": [38, 3, 6, 3, 37, 3],

    "cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190":
        [15, 3, 5, 3, 36, 3],

    "gneJ143": [38, 3, 6, 3, 37, 3],
    "gneJ207": [38, 3, 6, 3, 37, 3],
    "gneJ210": [38, 3, 6, 3, 37, 3],
    "gneJ260": [38, 3, 6, 3, 37, 3],
}

# ============================================================
# AVVIO SUMO
# ============================================================
sumo_binary = sumolib.checkBinary("sumo-gui" if USE_GUI else "sumo")

traci.start([
    sumo_binary,
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--start",
    "--quit-on-end",
    "--no-step-log",
    "--time-to-teleport", "-1",
])

# porta al begin time
while traci.simulation.getTime() < BEGIN_TIME:
    traci.simulationStep()

# ============================================================
# IDENTIFICA TLS
# ============================================================
tls_ids = traci.trafficlight.getIDList()
print(f"🚦 Fixed-Time Ingolstadt7 su TLS: {tls_ids}")

for tls in tls_ids:
    assert tls in FIXED_TIME, f"❌ TLS {tls} non ha fixed-time definito"

# stato per TLS
tls_phase = {tls: 0 for tls in tls_ids}

# ============================================================
# METRICHE
# ============================================================
wait_list = []
queue_list = []
speed_list = []

sim_time = 0

# ============================================================
# LOOP FIXED-TIME MULTI-TLS
# ============================================================
while sim_time < EP_LEN:

    # set phase per ogni TLS
    for tls in tls_ids:
        traci.trafficlight.setPhase(tls, tls_phase[tls])

    # durata di riferimento (allineata)
    ref_tls = tls_ids[0]
    duration = FIXED_TIME[ref_tls][tls_phase[ref_tls]]

    for _ in range(duration):
        traci.simulationStep()
        sim_time += 1

        # -------------------------------
        # METRICHE GLOBALI
        # -------------------------------
        veh_ids = traci.vehicle.getIDList()

        if veh_ids:
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            speed_list.append(float(np.mean(speeds)))
        else:
            speed_list.append(0.0)

        waits = [traci.edge.getWaitingTime(e) for e in traci.edge.getIDList()]
        wait_list.append(float(np.mean(waits)))

        stopped = sum(
            traci.edge.getLastStepHaltingNumber(e)
            for e in traci.edge.getIDList()
        )
        queue_list.append(float(stopped))

        if sim_time >= EP_LEN:
            break

    # avanza fase per ogni TLS
    for tls in tls_ids:
        tls_phase[tls] = (tls_phase[tls] + 1) % len(FIXED_TIME[tls])

traci.close()

# ============================================================
# RISULTATI
# ============================================================
mean_wait = float(np.mean(wait_list))
mean_queue = float(np.mean(queue_list))
mean_speed = float(np.mean(speed_list))

with open(CSV_OUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mean_wait", "mean_queue", "mean_speed"])
    writer.writerow([mean_wait, mean_queue, mean_speed])

print("\n✅ FIXED-TIME COMPLETATO (INGOLSTADT7)")
print(f"Mean wait : {mean_wait:.2f} s")
print(f"Mean queue: {mean_queue:.2f}")
print(f"Mean speed: {mean_speed:.2f} m/s")
print(f"CSV salvato in: {CSV_OUT}")
