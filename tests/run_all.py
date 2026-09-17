"""Runs every test script in this folder and prints a pass/fail summary for each suite.

Usage:
    python tests/run_all.py            # everything, including the slow CARLA connection test
    python tests/run_all.py --fast     # skip the slow CARLA connection test (~12s)

Each test file is a standalone script (not pytest) that prints [PASS]/[FAIL] lines for
individual checks and raises/exits non-zero if any check failed -- this runner just
executes each one as a subprocess and reports whether it exited cleanly.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

FAST_SUITE = [
    "test_head_pose_math.py",
    "test_attention_tracker.py",
    "test_drowsiness_strikes.py",
    "test_carla_vehicle_response.py",
    "test_driver_recognition.py",
    "test_vehicle_simulation.py",
]
SLOW_SUITE = [
    "test_carla_connection.py",  # attempts a real network connection, ~5-12s
]


def run_one(script_name):
    path = HERE / script_name
    print(f"\n{'=' * 70}\n{script_name}\n{'=' * 70}")
    start = time.time()
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    elapsed = time.time() - start
    print(result.stdout.strip())
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr.strip())
    ok = result.returncode == 0
    print(f"\n{'PASS' if ok else 'FAIL'}  ({elapsed:.1f}s)")
    return ok


def main():
    fast_only = "--fast" in sys.argv
    suite = FAST_SUITE + ([] if fast_only else SLOW_SUITE)

    results = {}
    for script in suite:
        results[script] = run_one(script)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    passed = sum(results.values())
    for script, ok in results.items():
        print(f"  {'[PASS]' if ok else '[FAIL]'}  {script}")
    print(f"\n{passed}/{len(results)} test suites passed")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
