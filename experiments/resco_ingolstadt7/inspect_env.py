import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NET_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "nets", "RESCO", "ingolstadt7", "ingolstadt7.rou.xml")
BEGIN_TIME = 57600                  # stesso begin di RL
EP_LEN = 3600                       # 1h

# ============================================================
# LOAD XML
# ============================================================
assert os.path.exists(ROUTE_FILE), f"File non trovato: {ROUTE_FILE}"

tree = ET.parse(ROUTE_FILE)
root = tree.getroot()

# ============================================================
# PARSE TRIPS
# ============================================================
trips = []
origins = []
destinations = []

for trip in root.findall("trip"):
    depart = float(trip.attrib["depart"])
    frm = trip.attrib.get("from")
    to = trip.attrib.get("to")

    trips.append(depart)
    origins.append(frm)
    destinations.append(to)

trips = np.array(trips)

# ============================================================
# BASIC STATS
# ============================================================
print("\n🚦 ROUTE INSPECTION")
print("=" * 60)

print(f"Total trips          : {len(trips)}")
print(f"Depart time min      : {trips.min():.2f}")
print(f"Depart time max      : {trips.max():.2f}")
print(f"Simulation window    : [{BEGIN_TIME}, {BEGIN_TIME + EP_LEN}]")

# trips in episode window
mask = (trips >= BEGIN_TIME) & (trips <= BEGIN_TIME + EP_LEN)
trips_ep = trips[mask]

print(f"Trips in episode     : {len(trips_ep)}")
print(f"Trips / hour         : {len(trips_ep)} veh/h")

# ============================================================
# TEMPORAL DENSITY
# ============================================================
bins = np.linspace(BEGIN_TIME, BEGIN_TIME + EP_LEN, 13)  # 5-min bins
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
    print(f"  {e:40s} : {c}")

print("\n🎯 TOP 5 DESTINATION EDGES")
for e, c in dest_counter.most_common(5):
    print(f"  {e:40s} : {c}")

# ============================================================
# SUMMARY FOR PAPER
# ============================================================
print("\n📌 PAPER-READY SUMMARY")
print("-" * 60)
print(f"• Traffic model   : deterministic trip-based")
print(f"• Vehicles        : {len(trips_ep)} in 1h")
print(f"• Avg veh / min   : {len(trips_ep) / 60:.2f}")
print(f"• Peak 5-min load : {hist.max()} vehicles")
print("=" * 60)
