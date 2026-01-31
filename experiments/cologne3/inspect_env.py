import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import os
import sys

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SCENARIO = os.path.basename(os.path.dirname(__file__))

NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", SCENARIO, f"{SCENARIO}.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", SCENARIO, f"{SCENARIO}.rou.xml")

BEGIN_TIME = 25200        # coerente con gli esperimenti RL
EP_LEN = 3600             # 1 ora

# ============================================================
# LOAD ROUTES XML
# ============================================================
assert os.path.exists(ROUTE_FILE), f"❌ File routes non trovato: {ROUTE_FILE}"

tree = ET.parse(ROUTE_FILE)
root = tree.getroot()

# ============================================================
# PARSE TRIPS / VEHICLES (ROBUSTO)
# ============================================================
depart_times = []
origins = []
destinations = []

# ---- TRIP-based (Ingolstadt) ----
for trip in root.findall("trip"):
    depart_times.append(float(trip.attrib["depart"]))
    origins.append(trip.attrib.get("from"))
    destinations.append(trip.attrib.get("to"))

# ---- VEHICLE-based (Cologne) ----
for veh in root.findall("vehicle"):
    depart_times.append(float(veh.attrib["depart"]))

    route = veh.find("route")
    if route is not None:
        edges = route.attrib.get("edges", "").split()
        if len(edges) > 0:
            origins.append(edges[0])
            destinations.append(edges[-1])

depart_times = np.array(depart_times)

# ============================================================
# HEADER
# ============================================================
print("\n🚦 ROUTE INSPECTION —", SCENARIO.upper())
print("=" * 60)

print(f"Total trips          : {len(depart_times)}")

if len(depart_times) == 0:
    print("⚠️ Nessun veicolo/trip trovato nel file routes.")
    sys.exit(0)

print(f"Depart time min      : {depart_times.min():.2f}")
print(f"Depart time max      : {depart_times.max():.2f}")
print(f"Simulation window    : [{BEGIN_TIME}, {BEGIN_TIME + EP_LEN}]")

# ============================================================
# FILTER EPISODE WINDOW
# ============================================================
mask = (depart_times >= BEGIN_TIME) & (depart_times <= BEGIN_TIME + EP_LEN)
trips_ep = depart_times[mask]

print(f"Trips in episode     : {len(trips_ep)}")
print(f"Trips / hour         : {len(trips_ep)} veh/h")

# ============================================================
# TEMPORAL DEMAND (5-MIN BINS)
# ============================================================
bins = np.linspace(BEGIN_TIME, BEGIN_TIME + EP_LEN, 13)  # 12 bins = 5 min
hist, _ = np.histogram(trips_ep, bins=bins)

print("\n⏱️ DEMAND OVER TIME (5 min bins)")
for i, h in enumerate(hist):
    t0 = int(bins[i] - BEGIN_TIME)
    t1 = int(bins[i + 1] - BEGIN_TIME)
    print(f"  {t0:4d}–{t1:4d} s : {h:4d} trips")

# ============================================================
# TOP ORIGINS / DESTINATIONS
# ============================================================
orig_counter = Counter(origins)
dest_counter = Counter(destinations)

print("\n🚗 TOP 5 ORIGIN EDGES")
for e, c in orig_counter.most_common(5):
    if e is not None:
        print(f"  {e:40s} : {c}")

print("\n🎯 TOP 5 DESTINATION EDGES")
for e, c in dest_counter.most_common(5):
    if e is not None:
        print(f"  {e:40s} : {c}")

# ============================================================
# PAPER-READY SUMMARY
# ============================================================
print("\n📌 PAPER-READY SUMMARY")
print("-" * 60)
print("• Traffic model   : deterministic trip/vehicle-based (RESCO)")
print(f"• Vehicles        : {len(trips_ep)} in 1h")
print(f"• Avg veh / min   : {len(trips_ep) / 60:.2f}")
print(f"• Peak 5-min load : {hist.max()} vehicles")
print("=" * 60)
