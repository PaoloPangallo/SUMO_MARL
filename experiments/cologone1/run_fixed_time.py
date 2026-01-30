import os
import csv
import numpy as np
import traci
import sumolib

# ============================================================
# PARAMETRI ESPERIMENTO
# ============================================================
BEGIN_TIME = 25200
EP_LEN = 3600
USE_GUI = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.rou.xml")

OUT_DIR = "outputs_fixed_time"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_OUT = os.path.join(OUT_DIR, "fixed_time_metrics.csv")

# ============================================================
# FIXED-TIME (DA NET)
# ============================================================
FIXED_TIME = [29, 5, 6, 5, 29, 5, 6, 5]

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
# TLS
# ============================================================
tls_ids = traci.trafficlight.getIDList()
assert len(tls_ids) == 1
TLS_ID = tls_ids[0]

print(f"🚦 Fixed-Time Cologne1 su TLS {TLS_ID}")

# ============================================================
# METRICHE (IPPO STYLE)
# ============================================================
wait_list = []
queue_list = []
speed_list = []

sim_time = 0
current_phase = 0

# ============================================================
# LOOP FIXED-TIME
# ============================================================
while sim_time < EP_LEN:

    traci.trafficlight.setPhase(TLS_ID, current_phase)
    phase_duration = FIXED_TIME[current_phase]

    for _ in range(phase_duration):
        traci.simulationStep()
        sim_time += 1

        veh_ids = traci.vehicle.getIDList()

        if veh_ids:
            waits = [traci.vehicle.getWaitingTime(v) for v in veh_ids]
            speeds = [traci.vehicle.getSpeed(v) for v in veh_ids]
            wait_list.append(float(np.mean(waits)))
            speed_list.append(float(np.mean(speeds)))
        else:
            wait_list.append(0.0)
            speed_list.append(0.0)

        stopped = sum(
            traci.edge.getLastStepHaltingNumber(e)
            for e in traci.edge.getIDList()
        )
        queue_list.append(stopped)

        if sim_time >= EP_LEN:
            break

    current_phase = (current_phase + 1) % len(FIXED_TIME)

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

print("\n✅ FIXED-TIME COMPLETATO (COLOGNE1)")
print(f"Mean wait : {mean_wait:.2f} s")
print(f"Mean queue: {mean_queue:.2f}")
print(f"Mean speed: {mean_speed:.2f} m/s")
print(f"CSV salvato in: {CSV_OUT}")
