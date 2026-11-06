"""analysis.py - flight metrics + rule-based anomaly detection."""
import re
import numpy as np
from scipy import signal

DEFAULTS = {
    "cells": 0,               # 0 = auto-detect from starting voltage
    "min_cell_v": 3.5,        # V/cell under load
    "vibe_limit": 30.0,       # m/s^2 (ArduPilot guideline)
    "alt_drop_rate": 5.0,     # m/s downward
    "current_spike_mult": 1.8,
    "min_sats": 6,
    "max_hdop": 2.0,
}

MSG_WORDS = re.compile(r"fail|error|bad |critical|lost|low batt|crash|ekf|unhealthy|timeout", re.I)


# ------------------------------------------------------------------ helpers
def _arr(df, col):
    if df is None or col not in df.columns:
        return None
    return df[col].to_numpy(dtype=float)


def _events(t, mask, min_gap=2.0):
    """Group consecutive True samples into (start, end) events."""
    t = np.asarray(t, float)
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return []
    events, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if t[i] - t[p] > min_gap:
            events.append((t[s], t[p]))
            s = i
        p = i
    events.append((t[s], t[p]))
    return events


def _in(t, arr, ev):
    sel = (t >= ev[0]) & (t <= ev[1])
    return arr[sel]


def _distance_m(lat, lng):
    lat, lng = np.radians(lat), np.radians(lng)
    dlat, dlng = np.diff(lat), np.diff(lng)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlng / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


def get_altitude(data):
    """Relative altitude (m). Prefers barometer, falls back to GPS."""
    for key in ("BARO", "GPS"):
        df = data.get(key)
        if df is None or "Alt" not in df.columns:
            continue
        t, a = df["t"].to_numpy(float), df["Alt"].to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(a)
        t, a = t[ok], a[ok]
        if len(t) > 5:
            return t, a - np.median(a[:20])
    return None, None


def auto_cells(volt):
    v0 = np.nanmedian(volt[: min(20, len(volt))])
    return max(1, int(round(v0 / 3.9)))


# ------------------------------------------------------------------ main
def analyze(data, cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    summary, anomalies, extras = {}, [], {}

    def add(kind, sev, ev, detail):
        anomalies.append({"type": kind, "severity": sev, "start": float(ev[0]),
                          "end": float(ev[1]), "detail": detail})

    # ---- duration
    spans = [df["t"].max() for df in data.values() if "t" in df.columns]
    summary["duration_s"] = float(max(spans)) if spans else 0.0

    # ---- altitude
    ta, alt = get_altitude(data)
    if alt is not None:
        summary["max_alt_m"] = float(np.nanmax(alt))

        tt = np.arange(ta.min(), ta.max(), 1.0)
        if len(tt) > 4:
            aa = np.interp(tt, ta, alt)
            rate = (aa[2:] - aa[:-2]) / 2.0          # m/s over a 2 s window
            for ev in _events(tt[1:-1], rate < -cfg["alt_drop_rate"], 2.0):
                worst = float(rate[(tt[1:-1] >= ev[0]) & (tt[1:-1] <= ev[1])].min())
                add("Sudden altitude loss", "critical", ev,
                    f"Descent up to {abs(worst):.1f} m/s (limit {cfg['alt_drop_rate']} m/s)")

    # ---- GPS
    gps = data.get("GPS")
    if gps is not None:
        tg = gps["t"].to_numpy(float)
        spd = _arr(gps, "Spd")
        if spd is not None:
            summary["max_speed_ms"] = float(np.nanmax(spd))
            summary["avg_speed_ms"] = float(np.nanmean(spd))
        lat, lng = _arr(gps, "Lat"), _arr(gps, "Lng")
        if lat is not None and lng is not None:
            ok = np.isfinite(lat) & np.isfinite(lng) & (lat != 0)
            if ok.sum() > 2:
                summary["distance_m"] = float(_distance_m(lat[ok], lng[ok]).sum())
        sats = _arr(gps, "NSats")
        if sats is not None:
            for ev in _events(tg, sats < cfg["min_sats"], 2.0):
                add("Weak GPS", "warning", ev,
                    f"Satellites down to {int(_in(tg, sats, ev).min())} (min {cfg['min_sats']})")
        hdop = _arr(gps, "HDop")
        if hdop is not None:
            for ev in _events(tg, hdop > cfg["max_hdop"], 2.0):
                add("Poor GPS accuracy", "warning", ev,
                    f"HDOP up to {_in(tg, hdop, ev).max():.1f} (max {cfg['max_hdop']})")

    # ---- battery
    bat = data.get("BAT")
    if bat is not None:
        tb = bat["t"].to_numpy(float)
        volt, curr = _arr(bat, "Volt"), _arr(bat, "Curr")
        if volt is not None:
            cells = cfg["cells"] or auto_cells(volt)
            extras["cells"] = cells
            summary["cells"] = cells
            summary["min_voltage_v"] = float(np.nanmin(volt))
            summary["min_cell_v"] = float(np.nanmin(volt) / cells)
            limit_v = cells * cfg["min_cell_v"]
            for ev in _events(tb, volt < limit_v, 2.0):
                low = _in(tb, volt, ev).min() / cells
                sev = "critical" if low < cfg["min_cell_v"] - 0.2 else "warning"
                add("Battery sag", sev, ev, f"Cell voltage down to {low:.2f} V (limit {cfg['min_cell_v']} V)")
        if curr is not None:
            summary["avg_current_a"] = float(np.nanmean(curr))
            summary["peak_current_a"] = float(np.nanmax(curr))
            active = curr[curr > 1]
            if active.size:
                limit = cfg["current_spike_mult"] * active.mean()
                for ev in _events(tb, curr > limit, 1.0):
                    add("Current spike", "warning", ev,
                        f"Peak {_in(tb, curr, ev).max():.1f} A vs {active.mean():.1f} A average")
            dt = np.diff(tb, prepend=tb[0])
            mah = _arr(bat, "CurrTot")
            summary["mah_used"] = float(np.nanmax(mah)) if mah is not None and np.nanmax(mah) > 0 \
                else float(np.sum(curr * dt) / 3.6)
            if volt is not None:
                summary["energy_wh"] = float(np.sum(volt * curr * dt) / 3600)
                summary["avg_power_w"] = float(np.nanmean(volt * curr))
                # internal resistance: slope of V vs I
                if np.nanmax(curr) - np.nanmin(curr) > 5 and len(curr) > 50:
                    ok = np.isfinite(volt) & np.isfinite(curr)
                    slope, _ = np.polyfit(curr[ok], volt[ok], 1)
                    if slope < 0:
                        extras["resistance_mohm"] = float(-slope * 1000)
                        summary["internal_res_mohm"] = extras["resistance_mohm"]

    # ---- vibration
    vibe = data.get("VIBE")
    if vibe is not None:
        tv = vibe["t"].to_numpy(float)
        axes = [c for c in ("VibeX", "VibeY", "VibeZ") if c in vibe.columns]
        if axes:
            vmax = np.nanmax(np.vstack([vibe[c].to_numpy(float) for c in axes]), axis=0)
            summary["max_vibration"] = float(np.nanmax(vmax))
            for ev in _events(tv, vmax > cfg["vibe_limit"], 2.0):
                add("High vibration", "critical", ev,
                    f"Up to {_in(tv, vmax, ev).max():.1f} m/s² (limit {cfg['vibe_limit']})")
        clip = _arr(vibe, "Clip")
        if clip is not None and len(clip) > 1:
            summary["clip_events"] = int(np.nanmax(clip) - np.nanmin(clip))
            inc = np.diff(clip, prepend=clip[0]) > 0
            for ev in _events(tv, inc, 2.0):
                add("Accelerometer clipping", "critical", ev, "Accel clip counter increased")

    # ---- FFT (needs a high-rate IMU log)
    imu = data.get("IMU")
    if imu is not None and "AccZ" in imu.columns and len(imu) > 256:
        ti, az = imu["t"].to_numpy(float), imu["AccZ"].to_numpy(float)
        fs = 1.0 / np.median(np.diff(ti))
        if fs >= 50:
            f, p = signal.welch(az - np.nanmean(az), fs=fs, nperseg=min(1024, len(az)))
            band = f > 5
            extras["fft"] = {"f": f, "psd": p, "peak_hz": float(f[band][np.argmax(p[band])]), "fs": float(fs)}
            summary["dominant_vibration_hz"] = extras["fft"]["peak_hz"]

    # ---- log messages / errors
    msg = data.get("MSG")
    if msg is not None:
        for _, r in msg.iterrows():
            if MSG_WORDS.search(str(r.get("Message", ""))):
                add("Log message", "info", (r["t"], r["t"]), str(r["Message"]))
    err = data.get("ERR")
    if err is not None:
        for _, r in err.iterrows():
            add("Flight controller error", "warning", (r["t"], r["t"]),
                f"Subsystem {int(r['Subsys'])}, code {int(r['ECode'])}")

    anomalies.sort(key=lambda a: a["start"])
    return {"summary": summary, "anomalies": anomalies, "extras": extras, "cfg": cfg}


# ------------------------------------------------------------------ output
def summary_cards(s):
    """(label, display string) pairs for the UI."""
    def f(key, label, fmt):
        return (label, fmt.format(s[key])) if key in s else None
    mins, secs = divmod(int(s.get("duration_s", 0)), 60)
    cards = [
        ("Duration", f"{mins}m {secs:02d}s"),
        f("max_alt_m", "Max altitude", "{:.1f} m"),
        f("max_speed_ms", "Max speed", "{:.1f} m/s"),
        f("avg_speed_ms", "Avg speed", "{:.1f} m/s"),
        f("distance_m", "Distance", "{:.0f} m"),
        f("mah_used", "Battery used", "{:.0f} mAh"),
        f("min_voltage_v", "Min voltage", "{:.2f} V"),
        f("min_cell_v", "Min cell voltage", "{:.2f} V"),
        f("avg_current_a", "Avg current", "{:.1f} A"),
        f("peak_current_a", "Peak current", "{:.1f} A"),
        f("avg_power_w", "Avg power", "{:.0f} W"),
        f("energy_wh", "Energy used", "{:.1f} Wh"),
        f("internal_res_mohm", "Pack resistance (est.)", "{:.0f} mΩ"),
        f("max_vibration", "Max vibration", "{:.1f} m/s²"),
        f("dominant_vibration_hz", "Dominant vib. freq", "{:.0f} Hz"),
    ]
    return [c for c in cards if c]


def build_report(summary, anomalies, name="flight"):
    lines = [f"# Flight report: {name}", "", "## Summary"]
    lines += [f"- **{k}:** {v}" for k, v in summary_cards(summary)]
    lines += ["", "## Anomalies"]
    if not anomalies:
        lines.append("No anomalies detected with the current thresholds.")
    for a in anomalies:
        lines.append(f"- [{a['severity'].upper()}] {a['type']} at {a['start']:.1f}s"
                     f"{'-' + format(a['end'], '.1f') + 's' if a['end'] > a['start'] else ''}: {a['detail']}")
    crit = sum(a["severity"] == "critical" for a in anomalies)
    lines += ["", f"**Verdict:** {crit} critical and {len(anomalies) - crit} other findings."]
    return "\n".join(lines)
