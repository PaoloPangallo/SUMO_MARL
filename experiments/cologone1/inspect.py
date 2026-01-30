import os
import traci
import sumolib

USE_GUI = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "cologne1", "cologne1.rou.xml")

sumo_binary = sumolib.checkBinary("sumo-gui" if USE_GUI else "sumo")

traci.start([
    sumo_binary,
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--start",
    "--no-step-log",
])

tls_ids = traci.trafficlight.getIDList()
print("\nTLS trovati:", tls_ids)

for tls in tls_ids:
    print("\n" + "=" * 80)
    print(f"TLS: {tls}")

    # programma attivo
    try:
        cur_prog = traci.trafficlight.getProgram(tls)
        print(f"Programma attivo: {cur_prog}")
    except Exception as e:
        print("Impossibile leggere getProgram:", e)

    # tutte le logiche/programmi disponibili
    logics = traci.trafficlight.getAllProgramLogics(tls)
    print(f"Num logiche disponibili: {len(logics)}")

    for li, logic in enumerate(logics):
        print("-" * 80)
        print(f"[Logic #{li}] programID={logic.programID} type={logic.type} currentPhaseIndex={logic.currentPhaseIndex}")

        phases = logic.phases
        print(f"Num fasi: {len(phases)}")

        for i, ph in enumerate(phases):
            # ph.state: stringa tipo "GGrr..."
            # ph.duration: durata nominale
            # ph.minDur / maxDur possono esserci
            state = getattr(ph, "state", None)
            dur = getattr(ph, "duration", None)
            minDur = getattr(ph, "minDur", None)
            maxDur = getattr(ph, "maxDur", None)

            print(f"  phase {i:02d}: dur={dur}  minDur={minDur}  maxDur={maxDur}  state={state}")

traci.close()
print("\nFatto.")
