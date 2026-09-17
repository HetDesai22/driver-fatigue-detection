"""Smoke-tests the local Tkinter VehicleSimulation (the automatic fallback used when no
CARLA server is reachable) across every driver-state combination it needs to handle:
normal driving, distraction-only, and all three drowsiness strike levels including the
persistent full stop. Confirms update() never raises for any of these states."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from app import VehicleSimulation

root = tk.Tk()
root.geometry("300x200+50+50")
sim = VehicleSimulation(root)

# (strikes, attention_state, direction)
scenarios = [
    (0, "ATTENTIVE", "FORWARD"),
    (0, "DISTRACTED", "LOOKING RIGHT"),
    (0, "DISTRACTED", "LOOKING LEFT"),
    (1, "ATTENTIVE", "FORWARD"),
    (2, "ATTENTIVE", "FORWARD"),
    (3, "DISTRACTED", "LOOKING RIGHT"),  # strikes must still dominate here
    (0, "ATTENTIVE", "FORWARD"),
]

errors = []


def run_step(i):
    if i >= len(scenarios) * 3:
        sim.announce("DRIVER CHANGED — RESUMING NORMAL SPEED")
        sim.update(0, "ATTENTIVE", "FORWARD")
        sim.close()
        root.after(100, root.destroy)
        return
    strikes, state, direction = scenarios[i // 3]
    try:
        sim.update(strikes, state, direction)
    except Exception as e:
        errors.append((strikes, state, direction, str(e)))
    root.after(15, lambda: run_step(i + 1))


root.after(50, lambda: run_step(0))
root.mainloop()

if errors:
    for e in errors:
        print("[FAIL]", e)
    raise AssertionError(f"{len(errors)} scenario(s) raised an exception")

print(f"[PASS] VehicleSimulation.update() ran cleanly across all {len(scenarios)} driver-state "
      f"scenarios plus an announce() call, with no exceptions.")
