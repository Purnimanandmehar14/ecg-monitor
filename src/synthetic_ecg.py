"""
synthetic_ecg.py
-----------------
Generates synthetic ECG records with Normal (N), supraventricular premature
(A -> AAMI 'S') and ventricular premature (V) beats, plus realistic nuisances:
DC offset, baseline wander, 60 Hz mains pickup and broadband noise.

PURPOSE: offline testing and demos only (CI, unit tests, the Streamlit demo
when no real data is at hand). Accuracy numbers measured on this data say
NOTHING about clinical performance -- train on MIT-BIH for that.
"""

import numpy as np

FS = 360


def _g(t, amp, mu, sig):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def _template(t, kind, rng):
    """Beat morphology as a function of time (s) relative to the R peak."""
    j = lambda s: 1.0 + s * rng.randn()          # per-beat jitter factor
    if kind == "V":                               # wide, tall, no P, discordant T
        sig = rng.uniform(0.032, 0.048)
        return (_g(t, 1.5 * j(0.12), 0.0, sig)
                + _g(t, -0.45 * j(0.15), 0.085, sig * 1.1)
                + _g(t, -0.40 * j(0.2), 0.28, 0.07))
    p_amp = 0.15 if kind == "N" else 0.07         # 'A' beats: altered P wave
    p_pos = -0.20 if kind == "N" else -0.15
    return (_g(t, p_amp * j(0.15), p_pos, 0.025)
            + _g(t, -0.12 * j(0.2), -0.04, 0.010)
            + _g(t, 1.0 * j(0.10), 0.0, 0.012)
            + _g(t, -0.20 * j(0.2), 0.035, 0.012)
            + _g(t, 0.30 * j(0.15), 0.25, 0.040))


def generate_record(duration_s=120, fs=FS, seed=0, p_v=0.08, p_s=0.07,
                    noise=0.03, mains_hz=60.0):
    """
    Returns dict with: signal (n,), fs, ann_samples (int array),
    ann_symbols (list of 'N' / 'A' / 'V') -- same layout as data_loader.
    """
    rng = np.random.RandomState(seed)
    n = int(duration_s * fs)
    base_rr = rng.uniform(0.65, 1.0)              # per-record heart rate
    amp_scale = rng.uniform(0.7, 1.4)

    # --- beat sequence with premature beats + compensatory pauses ---------
    times, kinds = [], []
    t = rng.uniform(0.4, 0.8)
    prev_kind = "N"
    while t < duration_s - 0.7:
        r = rng.rand()
        kind = "V" if r < p_v else ("A" if r < p_v + p_s else "N")
        if prev_kind != "N":
            kind = "N"                            # no back-to-back ectopics
        rr = base_rr * (1 + 0.04 * np.sin(2 * np.pi * 0.25 * t)) \
            * (1 + 0.02 * rng.randn())
        if kind == "V":
            rr_before, rr_after = 0.60 * rr, 1.40 * rr
        elif kind == "A":
            rr_before, rr_after = 0.72 * rr, 1.05 * rr
        else:
            rr_before = rr_after = rr
        if prev_kind == "N" and kind != "N":
            t += rr_before - rr                   # shorten the preceding gap
        times.append(t)
        kinds.append(kind)
        t += rr_after if kind != "N" else rr
        prev_kind = kind

    # --- paint beats -------------------------------------------------------
    sig = np.zeros(n)
    half = int(0.5 * fs)
    tt = np.arange(-half, half) / fs
    ann = []
    for tb, k in zip(times, kinds):
        c = int(round(tb * fs))
        if c - half < 0 or c + half > n:
            continue
        sig[c - half:c + half] += amp_scale * _template(tt, k, rng)
        ann.append((c, k))

    # --- nuisances -----------------------------------------------------------
    tvec = np.arange(n) / fs
    sig += 1.0                                                   # DC offset
    sig += 0.25 * np.sin(2 * np.pi * 0.2 * tvec + rng.rand() * 6.28)
    sig += 0.10 * np.sin(2 * np.pi * 0.05 * tvec + rng.rand() * 6.28)
    sig += 0.02 * np.sin(2 * np.pi * mains_hz * tvec)
    sig += noise * rng.randn(n)

    return {
        "signal": sig,
        "fs": fs,
        "ann_samples": np.array([a for a, _ in ann], dtype=int),
        "ann_symbols": [k for _, k in ann],
    }


if __name__ == "__main__":
    rec = generate_record(30, seed=1)
    from collections import Counter
    print("samples:", len(rec["signal"]), "beats:", len(rec["ann_samples"]),
          Counter(rec["ann_symbols"]))
