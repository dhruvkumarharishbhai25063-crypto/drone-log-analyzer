# Drone Flight Log Analyzer

Upload an ArduPilot flight log (`.bin` / `.log`) or a CSV and get interactive plots, flight metrics and
rule-based anomaly detection (battery sag, vibration, clipping, altitude loss, current spikes, weak GPS).

## Run locally
```bash
pip install -r requirements.txt
python make_sample.py          # creates a synthetic demo flight
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `log_parser.py` | ArduPilot `.bin` + CSV -> pandas DataFrames |
| `analysis.py` | metrics, anomaly rules, internal-resistance estimate, FFT |
| `plots.py` | Plotly charts (incl. 3D path) |
| `app.py` | Streamlit UI |
| `make_sample.py` | synthetic sample flight with injected faults |

## CSV format
Needs a time column (`t`/`time_s`) plus any of: `Volt, Curr, Alt, Spd, Lat, Lng, Roll, Pitch, Yaw, VibeX, VibeY, VibeZ, Clip, NSats, HDop`.

## Deploy (free)
Push to GitHub -> share.streamlit.io -> select repo -> main file `app.py`.

## Limitations
- Thresholds are defaults; tune them for your aircraft in the sidebar.
- Vibration FFT needs high-rate IMU logging (>= 50 Hz).
- Pack resistance is a rough estimate from voltage vs current.
- PX4 `.ulg` not supported yet (idea: add `pyulog`).
