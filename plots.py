"""plots.py - Plotly figures. Each returns None if the needed data is missing."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis import get_altitude

SHADE = {"critical": "rgba(220,38,38,0.20)", "warning": "rgba(245,158,11,0.20)", "info": "rgba(59,130,246,0.15)"}


def _shade(fig, anomalies, kinds, **kw):
    for a in anomalies:
        if a["type"] in kinds:
            fig.add_vrect(x0=a["start"], x1=max(a["end"], a["start"] + 0.5),
                          fillcolor=SHADE[a["severity"]], line_width=0, layer="below", **kw)


def fig_flight(data, anomalies):
    ta, alt = get_altitude(data)
    gps = data.get("GPS")
    if alt is None:
        return None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ta, y=alt, name="Altitude (m)"), secondary_y=False)
    if gps is not None and "Spd" in gps.columns:
        fig.add_trace(go.Scatter(x=gps["t"], y=gps["Spd"], name="Speed (m/s)"), secondary_y=True)
    _shade(fig, anomalies, {"Sudden altitude loss", "Weak GPS", "Poor GPS accuracy"})
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Altitude (m)", secondary_y=False)
    fig.update_yaxes(title="Speed (m/s)", secondary_y=True)
    fig.update_layout(title="Altitude and speed", hovermode="x unified")
    return fig


def fig_battery(data, anomalies, cells, min_cell_v):
    bat = data.get("BAT")
    if bat is None:
        return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Voltage (V)", "Current (A)", "Power (W)"))
    if "Volt" in bat.columns:
        fig.add_trace(go.Scatter(x=bat["t"], y=bat["Volt"], name="Voltage"), row=1, col=1)
        if cells:
            fig.add_hline(y=cells * min_cell_v, line_dash="dash", line_color="red", row=1, col=1,
                          annotation_text=f"{cells}S limit")
    if "Curr" in bat.columns:
        fig.add_trace(go.Scatter(x=bat["t"], y=bat["Curr"], name="Current"), row=2, col=1)
    if "Volt" in bat.columns and "Curr" in bat.columns:
        fig.add_trace(go.Scatter(x=bat["t"], y=bat["Volt"] * bat["Curr"], name="Power"), row=3, col=1)
    for r in (1, 2, 3):
        _shade(fig, anomalies, {"Battery sag", "Current spike"}, row=r, col=1)
    fig.update_xaxes(title="Time (s)", row=3, col=1)
    fig.update_layout(height=700, showlegend=False, hovermode="x unified")
    return fig


def fig_vibration(data, anomalies, limit):
    v = data.get("VIBE")
    if v is None:
        return None
    fig = go.Figure()
    for c in ("VibeX", "VibeY", "VibeZ"):
        if c in v.columns:
            fig.add_trace(go.Scatter(x=v["t"], y=v[c], name=c))
    fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text="limit")
    _shade(fig, anomalies, {"High vibration", "Accelerometer clipping"})
    fig.update_layout(title="Vibration", xaxis_title="Time (s)", yaxis_title="m/s²", hovermode="x unified")
    return fig


def fig_fft(extras):
    f = extras.get("fft")
    if not f:
        return None
    fig = go.Figure(go.Scatter(x=f["f"], y=f["psd"], name="PSD"))
    fig.update_layout(title=f"Vibration spectrum (peak {f['peak_hz']:.0f} Hz, fs {f['fs']:.0f} Hz)",
                      xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
    return fig


def fig_attitude(data):
    a = data.get("ATT")
    if a is None:
        return None
    fig = go.Figure()
    for c in ("Roll", "Pitch"):
        if c in a.columns:
            fig.add_trace(go.Scatter(x=a["t"], y=a[c], name=c))
    fig.update_layout(title="Attitude (deg)", xaxis_title="Time (s)", hovermode="x unified")
    return fig


def fig_path3d(data):
    g = data.get("GPS")
    if g is None or "Lat" not in g.columns or "Lng" not in g.columns:
        return None
    g = g[(g["Lat"] != 0) & g["Lat"].notna()]
    if len(g) < 3:
        return None
    lat0, lng0 = g["Lat"].iloc[0], g["Lng"].iloc[0]
    x = (g["Lng"] - lng0) * 111320 * np.cos(np.radians(lat0))
    y = (g["Lat"] - lat0) * 111320
    ta, alt = get_altitude(data)
    z = np.interp(g["t"], ta, alt) if alt is not None else np.zeros(len(g))
    color = g["Spd"] if "Spd" in g.columns else z
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="lines+markers",
                                 marker=dict(size=2, color=color, colorscale="Viridis",
                                             colorbar=dict(title="Speed m/s")),
                                 line=dict(width=3, color="gray")))
    fig.update_layout(title="3D flight path", height=600,
                      scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Altitude (m)",
                                 aspectmode="data"))
    return fig
