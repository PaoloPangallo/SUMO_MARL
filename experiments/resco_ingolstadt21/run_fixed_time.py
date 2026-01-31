import os
import csv
import numpy as np
import traci
import sumolib

# ============================================================
# PARAMETRI ESPERIMENTO
# ============================================================
BEGIN_TIME = 57600
EP_LEN = 3600
USE_GUI = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt21", "ingolstadt21.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt21", "ingolstadt21.rou.xml")

OUT_DIR = "outputs_fixed_time"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUT_DIR, "fixed_time_metrics.csv")

# ============================================================
# AVVIO SUMO (FIXED-TIME NATIVO)
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

tls_ids = traci.trafficlight.getIDList()
print(f"🚦 Fixed-Time Ingolstadt21 (default TLS programs): {tls_ids}")

# ============================================================
# METRICHE
# ============================================================
wait_list, queue_list, speed_list = [], [], []
sim_time = 0

# ============================================================
# LOOP SIMULAZIONE (NESSUN CONTROLLO TLS)
# ============================================================
while sim_time < EP_LEN:
    traci.simulationStep()
    sim_time += 1

    veh_ids = traci.vehicle.getIDList()
    speed_list.append(
        np.mean([traci.vehicle.getSpeed(v) for v in veh_ids]) if veh_ids else 0.0
    )

    wait_list.append(
        np.mean([traci.edge.getWaitingTime(e) for e in traci.edge.getIDList()])
    )

    queue_list.append(
        sum(traci.edge.getLastStepHaltingNumber(e) for e in traci.edge.getIDList())
    )

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

print("\n✅ FIXED-TIME COMPLETATO (INGOLSTADT21 – DEFAULT)")
print(f"Mean wait : {mean_wait:.2f} s")
print(f"Mean queue: {mean_queue:.2f}")
print(f"Mean speed: {mean_speed:.2f} m/s")
print(f"CSV salvato in: {CSV_OUT}")
