"""Generate a synthetic 5-minute flight (CSV) with injected faults so the demo works without a real log."""
import numpy as np
import pandas as pd


def make(path="sample_logs/sample_flight.csv", seed=7):
    rng = np.random.default_rng(seed)
    hz, T = 10, 300
    t = np.arange(0, T, 1 / hz)
    n = len(t)

    alt = np.interp(t, [0, 15, 200, 215, 300], [0, 60, 60, 5, 0])
    alt -= np.interp(t, [120, 122, 126], [0, 18, 0], left=0, right=0)       # sudden drop
    alt += rng.normal(0, 0.15, n)

    th = 2 * np.pi * t / 120
    x, y = 120 * np.cos(th), 120 * np.sin(th)
    spd = np.hypot(np.gradient(x, t), np.gradient(y, t)) + rng.normal(0, 0.1, n)
    lat0, lng0 = 15.40, 73.98
    lat = lat0 + y / 111320
    lng = lng0 + x / (111320 * np.cos(np.radians(lat0)))

    climb = np.gradient(alt, t)
    curr = 11 + 1 * np.clip(climb, 0, None) + rng.normal(0, 0.6, n)
    curr += 30 * ((t >= 90) & (t < 91.5))                                    # current spike
    mah = np.cumsum(curr) / hz / 3.6
    volt = 16.8 - 0.0024 * mah - curr * 0.025 + rng.normal(0, 0.02, n)       # 4S pack

    bad = (t >= 150) & (t < 165)
    vx = np.abs(rng.normal(6, 1.2, n)) + bad * 35
    vy = np.abs(rng.normal(6, 1.2, n)) + bad * 20
    vz = np.abs(rng.normal(9, 1.5, n)) + bad * 25
    clip = np.cumsum(bad & (rng.random(n) < 0.3))

    weak = (t >= 60) & (t < 70)
    df = pd.DataFrame({
        "t": t, "Volt": volt, "Curr": curr, "Alt": alt, "Spd": spd, "Lat": lat, "Lng": lng,
        "Roll": 5 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.5, n),
        "Pitch": 3 * np.sin(2 * np.pi * t / 15) + rng.normal(0, 0.5, n),
        "Yaw": (np.degrees(th) + 90) % 360,
        "VibeX": vx, "VibeY": vy, "VibeZ": vz, "Clip": clip,
        "NSats": np.where(weak, 5, 12), "HDop": np.where(weak, 3.1, 0.9) + rng.normal(0, 0.05, n),
    }).round(3)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    make()
    print("wrote sample_logs/sample_flight.csv")
