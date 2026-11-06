"""app.py - Streamlit UI.  Run:  streamlit run app.py"""
import glob
import os
import tempfile

import pandas as pd
import streamlit as st

from analysis import DEFAULTS, analyze, build_report, summary_cards
from log_parser import parse_log
import plots

st.set_page_config(page_title="Drone Flight Log Analyzer", page_icon="🛩️", layout="wide")


@st.cache_data(show_spinner="Parsing log...")
def load(file_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        return parse_log(path)
    finally:
        os.remove(path)


st.title("🛩️ Drone Flight Log Analyzer")
st.caption("Upload an ArduPilot log (.bin/.log) or a CSV and get plots, metrics and anomaly flags.")

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Input")
upload = sb.file_uploader("Flight log", type=["bin", "log", "csv"])
samples = sorted(glob.glob(os.path.join("sample_logs", "*")))
samples = ["sample_flight.csv"] if os.path.exists("sample_flight.csv") else samples
if samples and sb.button("Try a sample flight"):
    st.session_state["sample"] = samples[0]
if upload is not None:
    st.session_state.pop("sample", None)

sb.header("Settings")
cells = sb.number_input("Battery cells (0 = auto)", 0, 14, 0)
min_cell_v = sb.slider("Min cell voltage under load (V)", 3.0, 4.0, DEFAULTS["min_cell_v"], 0.05)
vibe_limit = sb.slider("Vibration limit (m/s²)", 10.0, 60.0, DEFAULTS["vibe_limit"], 1.0)
alt_drop = sb.slider("Altitude-loss rate limit (m/s)", 1.0, 15.0, DEFAULTS["alt_drop_rate"], 0.5)
spike = sb.slider("Current spike (× average)", 1.2, 4.0, DEFAULTS["current_spike_mult"], 0.1)
min_sats = sb.slider("Min GPS satellites", 3, 14, DEFAULTS["min_sats"])
max_hdop = sb.slider("Max HDOP", 0.5, 5.0, DEFAULTS["max_hdop"], 0.1)

# ------------------------------------------------------------------ load
name, blob, suffix = None, None, None
if upload is not None:
    name, blob, suffix = upload.name, upload.getvalue(), os.path.splitext(upload.name)[1]
elif "sample" in st.session_state:
    p = st.session_state["sample"]
    name, suffix = os.path.basename(p), os.path.splitext(p)[1]
    with open(p, "rb") as fh:
        blob = fh.read()

if blob is None:
    st.info("Upload a log in the sidebar, or click **Try a sample flight**.")
    st.stop()

try:
    data = load(blob, suffix)
except Exception as e:  # corrupt / unsupported file
    st.error(f"Could not read this log: {e}")
    st.stop()

cfg = dict(cells=cells, min_cell_v=min_cell_v, vibe_limit=vibe_limit, alt_drop_rate=alt_drop,
           current_spike_mult=spike, min_sats=min_sats, max_hdop=max_hdop)
res = analyze(data, cfg)
summary, anomalies, extras = res["summary"], res["anomalies"], res["extras"]
n_crit = sum(a["severity"] == "critical" for a in anomalies)

st.subheader(f"Flight: {name}")
if n_crit:
    st.error(f"{n_crit} critical issue(s) and {len(anomalies) - n_crit} other finding(s).")
elif anomalies:
    st.warning(f"{len(anomalies)} finding(s), none critical.")
else:
    st.success("No anomalies detected with the current thresholds.")

cards = summary_cards(summary)
for i in range(0, len(cards), 5):
    cols = st.columns(5)
    for col, (label, val) in zip(cols, cards[i:i + 5]):
        col.metric(label, val)


def show(fig, key, msg):
    if fig is None:
        st.info(msg)
    else:
        st.plotly_chart(fig, key=key)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Battery", "Vibration", "Path", "Anomalies", "Report"])

with tab1:
    show(plots.fig_flight(data, anomalies), "flight", "No altitude/speed data in this log.")
    show(plots.fig_attitude(data), "att", "No attitude data in this log.")
with tab2:
    show(plots.fig_battery(data, anomalies, extras.get("cells"), min_cell_v), "bat", "No battery data in this log.")
    if "resistance_mohm" in extras:
        st.caption(f"Estimated pack internal resistance: {extras['resistance_mohm']:.0f} mΩ "
                   "(slope of voltage vs current; rough indicator, compare flights on the same pack).")
with tab3:
    show(plots.fig_vibration(data, anomalies, vibe_limit), "vib", "No VIBE data in this log.")
    show(plots.fig_fft(extras), "fft",
         "No spectrum: needs high-rate IMU data (≥50 Hz) in the log. In ArduPilot raise LOG_BITMASK/IMU logging rate.")
with tab4:
    show(plots.fig_path3d(data), "path", "No GPS position data in this log.")
with tab5:
    if anomalies:
        df = pd.DataFrame(anomalies).rename(columns={"start": "start_s", "end": "end_s"})
        st.dataframe(df)
        st.download_button("Download anomalies (CSV)", df.to_csv(index=False), "anomalies.csv", "text/csv")
    else:
        st.success("Nothing flagged.")
    if "MODE" in data:
        st.markdown("**Flight mode changes**")
        st.dataframe(data["MODE"])
with tab6:
    report = build_report(summary, anomalies, name)
    st.markdown(report)
    st.download_button("Download report (Markdown)", report, "flight_report.md", "text/markdown")
