# Tests

Standalone scripts (not pytest — plain `assert`/print, run directly with `python`) that verify
the core logic without needing a live webcam or a running CARLA server for most of them.

## Run everything

```
python tests/run_all.py            # all 7 suites, including the slow CARLA connection test (~5-15s)
python tests/run_all.py --fast     # the 6 fast suites only, skips test_carla_connection.py
```

## Suites

| File | What it verifies |
|---|---|
| `test_head_pose_math.py` | solvePnP yaw/pitch/roll math on synthetic landmarks |
| `test_attention_tracker.py` | Distraction state machine: confirmation timing, false-positive protection, calibration, the pitch-wraparound bug fix |
| `test_drowsiness_strikes.py` | Wall-clock drowsiness trigger, FPS-independence, persistent strike escalation |
| `test_carla_vehicle_response.py` | Drowsiness-overrides-distraction priority logic, steering drift-bias math |
| `test_driver_recognition.py` | LBPH driver-change matching and the consecutive-mismatch debounce |
| `test_vehicle_simulation.py` | Local fallback simulation runs cleanly across every driver-state combination |
| `test_carla_connection.py` | **Slow, real network test.** Actually attempts to connect to CARLA — passes either way, whether a server is running (and it drives the vehicle) or not (graceful failure within the timeout) |

Each script imports directly from `app.py` (via a path-relative `sys.path.insert`), so they always
test the actual current application code, not a copy.
