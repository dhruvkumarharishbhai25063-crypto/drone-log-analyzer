"""log_parser.py - read ArduPilot .bin/.log files or generic CSV into pandas DataFrames.

Every returned DataFrame has a column `t` (seconds from start of log).
Returned dict keys (only present if data exists):
  BAT  : t, Volt, Curr, [CurrTot]
  GPS  : t, [Lat, Lng, Alt, Spd, NSats, HDop]
  BARO : t, Alt
  ATT  : t, Roll, Pitch, Yaw
  VIBE : t, VibeX, VibeY, VibeZ, [Clip]
  IMU  : t, AccX, AccY, AccZ        (used for the FFT, only if logged at high rate)
  MODE : t, Mode
  MSG  : t, Message
  ERR  : t, Subsys, ECode
"""
import os
import numpy as np
import pandas as pd

BIN_TYPES = ["BAT", "CURR", "GPS", "BARO", "ATT", "VIBE", "IMU", "MODE", "MSG", "ERR"]


def parse_log(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return parse_csv(path)
    return parse_bin(path)


# ---------------------------------------------------------------- ArduPilot
def parse_bin(path):
    from pymavlink import mavutil

    mlog = mavutil.mavlink_connection(path)
    rows = {t: [] for t in BIN_TYPES}
    while True:
        m = mlog.recv_match(type=BIN_TYPES)
        if m is None:
            break
        d = m.to_dict()
        d.pop("mavpackettype", None)
        if "TimeUS" not in d and "TimeMS" in d:       # very old logs
            d["TimeUS"] = d["TimeMS"] * 1000
        rows[m.get_type()].append(d)

    frames = {k: pd.DataFrame(v) for k, v in rows.items() if v}
    if not frames:
        raise ValueError("No supported messages found. Is this an ArduPilot DataFlash log?")

    # time zero = first timestamp in the whole log
    t0 = min(df["TimeUS"].min() for df in frames.values() if "TimeUS" in df.columns)
    for df in frames.values():
        if "TimeUS" in df.columns:
            df["t"] = (df["TimeUS"] - t0) / 1e6

    out = {}

    bat = frames.get("BAT")
    if bat is None and "CURR" in frames:              # old firmware
        bat = frames["CURR"]
    if bat is not None:
        if "Inst" in bat.columns:
            bat = bat[bat["Inst"] == 0]
        out["BAT"] = _keep(bat, ["t", "Volt", "Curr", "CurrTot"])

    gps = frames.get("GPS")
    if gps is not None:
        if "I" in gps.columns:
            gps = gps[gps["I"] == 0]
        if "Lat" in gps.columns:
            gps = gps[(gps["Lat"] != 0) & (gps["Lng"] != 0)]
        out["GPS"] = _keep(gps, ["t", "Lat", "Lng", "Alt", "Spd", "NSats", "HDop"])

    if "BARO" in frames:
        b = frames["BARO"]
        if "I" in b.columns:
            b = b[b["I"] == 0]
        out["BARO"] = _keep(b, ["t", "Alt"])

    if "ATT" in frames:
        out["ATT"] = _keep(frames["ATT"], ["t", "Roll", "Pitch", "Yaw"])

    if "VIBE" in frames:
        v = frames["VIBE"].copy()
        clip_cols = [c for c in v.columns if c.lower().startswith("clip")]
        if clip_cols:
            v["Clip"] = v[clip_cols].sum(axis=1)
        out["VIBE"] = _keep(v, ["t", "VibeX", "VibeY", "VibeZ", "Clip"])

    if "IMU" in frames:
        i = frames["IMU"]
        if "I" in i.columns:
            i = i[i["I"] == 0]
        out["IMU"] = _keep(i, ["t", "AccX", "AccY", "AccZ"])

    if "MODE" in frames:
        m = frames["MODE"].copy()
        m["Mode"] = m["ModeNum"] if "ModeNum" in m.columns else m.get("Mode", np.nan)
        out["MODE"] = _keep(m, ["t", "Mode"])

    if "MSG" in frames:
        out["MSG"] = _keep(frames["MSG"], ["t", "Message"])
    if "ERR" in frames:
        out["ERR"] = _keep(frames["ERR"], ["t", "Subsys", "ECode"])

    return {k: v.reset_index(drop=True) for k, v in out.items() if v is not None and len(v)}


def _keep(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy() if "t" in cols else None


# ---------------------------------------------------------------------- CSV
ALIASES = {
    "t": ["t", "time", "time_s", "timestamp", "seconds"],
    "Volt": ["volt", "voltage", "battery_voltage", "vbat"],
    "Curr": ["curr", "current", "battery_current", "ibat"],
    "Alt": ["alt", "altitude", "altitude_m"],
    "Spd": ["spd", "speed", "groundspeed", "speed_ms"],
    "Lat": ["lat", "latitude"],
    "Lng": ["lng", "lon", "long", "longitude"],
    "NSats": ["nsats", "sats", "satellites"],
    "HDop": ["hdop"],
    "Roll": ["roll"],
    "Pitch": ["pitch"],
    "Yaw": ["yaw"],
    "VibeX": ["vibex", "vibe_x", "vibration_x"],
    "VibeY": ["vibey", "vibe_y", "vibration_y"],
    "VibeZ": ["vibez", "vibe_z", "vibration_z"],
    "Clip": ["clip", "clipping"],
}


def parse_csv(path):
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(key):
        for a in ALIASES[key]:
            if a in lower:
                return lower[a]
        return None

    tcol = pick("t")
    if tcol is None:
        raise ValueError("CSV needs a time column (t / time / time_s / timestamp).")
    t = pd.to_numeric(df[tcol], errors="coerce")
    t = t - t.min()

    def frame(keys):
        present = {k: pick(k) for k in keys if pick(k)}
        if not present:
            return None
        f = pd.DataFrame({"t": t})
        for k, c in present.items():
            f[k] = pd.to_numeric(df[c], errors="coerce")
        return f.dropna(subset=["t"]).reset_index(drop=True)

    out = {
        "BAT": frame(["Volt", "Curr"]),
        "GPS": frame(["Lat", "Lng", "Alt", "Spd", "NSats", "HDop"]),
        "ATT": frame(["Roll", "Pitch", "Yaw"]),
        "VIBE": frame(["VibeX", "VibeY", "VibeZ", "Clip"]),
    }
    out = {k: v for k, v in out.items() if v is not None}
    if not out:
        raise ValueError("No recognised columns in CSV.")
    return out


if __name__ == "__main__":
    import sys
    for name, df in parse_log(sys.argv[1]).items():
        print(f"\n== {name} ({len(df)} rows) ==")
        print(df.head())
