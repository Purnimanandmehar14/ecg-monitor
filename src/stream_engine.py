"""
stream_engine.py
-----------------
Causal, sample-by-sample ECG analysis engine (the "Real-time ECG Analysis"
stage of the pipeline). This is the REFERENCE implementation; the C port in
embedded/ecg_engine.c mirrors it operation for operation, and tests/test_parity.py
checks that both give the same beats and labels.

Per sample:
  1. Causal band-pass (0.5-40 Hz Butterworth) + mains notch   (SOS biquads)
  2. Running-power normalisation (no global statistics needed)
  3. QRS detection: derivative -> square -> moving-window integral
     -> adaptive threshold -> R-peak = max |y| in the integration window
  4. Beat capture: 0.25 s before / 0.45 s after the R peak -> 6 shape features
  5. Once the beat AFTER next is detected, the RR features are complete
     -> 10-feature vector -> classifier -> event (latency ~ 2 beats, ~1.6 s)

Feature vector (same order/definitions as feature_extraction.py columns 0..9):
  prev_rr, next_rr, local_rr_mean, rr_ratio,
  max_amp, min_amp, mean_amp, std_amp, ptp_amp, skewness
"""

import math
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

from feature_extraction import time_domain_features

# ---- constants shared with the C port (written to ecg_config.h) -----------
FS = 360
MAINS_HZ = 60.0            # use 50.0 for sensors in 50 Hz regions (and retrain)
BEFORE = int(0.25 * FS)    # 90 samples before R
AFTER = int(0.45 * FS)     # 162 samples after R
WIN = BEFORE + AFTER       # 252
MWI_W = int(0.15 * FS)     # 54-sample moving-window integrator
REFRACT = int(0.20 * FS)   # 72 samples (200 ms) refractory period
WARM_N = 2 * FS            # 2 s warm-up before detection starts
NORM_ALPHA = 1.0 / (8 * FS)
THR_FRAC = 0.25
PEAK_ALPHA = 0.25
INIT_FRAC = 0.5            # initial peak level = INIT_FRAC * max MWI during warm-up
DECAY_AFTER = int(1.5 * FS)
DECAY = 0.9995
BUF_LEN = 512
N_FEAT = 10
FEATURE_NAMES = ["prev_rr", "next_rr", "local_rr_mean", "rr_ratio",
                 "max_amp", "min_amp", "mean_amp", "std_amp", "ptp_amp",
                 "skewness"]


def design_filters(fs=FS, mains=MAINS_HZ):
    """Band-pass + notch as one SOS cascade, plus steady-state init values."""
    bp = butter(4, [0.5, 40.0], btype="band", fs=fs, output="sos")
    b, a = iirnotch(mains, 30.0, fs=fs)
    sos = np.vstack([bp, tf2sos(b, a)])
    return sos, sosfilt_zi(sos)


class StreamEngine:
    def __init__(self, model=None, mains=MAINS_HZ):
        self.model = model
        self.sos, self.zi0 = design_filters(FS, mains)
        self.reset()

    def reset(self):
        self.zi = None
        self.n = 0
        self.p = 0.0
        self.ybuf = np.zeros(BUF_LEN)
        self.e_ring = np.zeros(MWI_W)
        self.e_sum = 0.0
        self.y_prev = 0.0
        self.m1 = 0.0
        self.m2 = 0.0
        self.mwi_max = 0.0
        self.peak_level = 0.0
        self.thr = 0.0
        self.last_peak = -10 ** 9
        self.peaks = []          # last <=6 R-peak sample indices
        self.pending = []        # R peaks waiting for their window to complete
        self.records = {}        # R index -> 6 shape features
        self.r_all = []          # every detected R index (for evaluation)

    # ------------------------------------------------------------------ API
    def process(self, x, classify=True):
        """Feed a chunk of raw samples; returns list of beat events."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return []
        if self.zi is None:
            self.zi = self.zi0 * x[0]          # start at steady state (no DC step)
        y_all, self.zi = sosfilt(self.sos, x, zi=self.zi)
        events = []
        for y in y_all.tolist():
            ev = self._step(y)
            if ev is not None:
                events.append(ev)
        if classify and self.model is not None and events:
            self._classify(events)
        return events

    # ------------------------------------------------------------- internals
    def _step(self, y):
        n = self.n
        # 2. running-power normalisation
        if n < WARM_N:
            self.p = (self.p * n + y * y) / (n + 1)
        else:
            self.p += NORM_ALPHA * (y * y - self.p)
        yn = y / math.sqrt(self.p + 1e-12)
        self.ybuf[n % BUF_LEN] = yn

        # 3a. derivative -> square -> moving-window integral
        d = yn - self.y_prev
        self.y_prev = yn
        e = d * d
        slot = n % MWI_W
        self.e_sum += e - self.e_ring[slot]
        self.e_ring[slot] = e
        mwi = self.e_sum / MWI_W

        # 4. finish capturing beats whose window is now complete
        if self.pending:
            for r in [r for r in self.pending if n == r + AFTER - 1]:
                self._capture(r)
                self.pending.remove(r)

        event = None
        if n >= MWI_W and n < WARM_N:
            if mwi > self.mwi_max:
                self.mwi_max = mwi
        if n == WARM_N:
            self.peak_level = INIT_FRAC * self.mwi_max
            self.thr = THR_FRAC * self.peak_level
        if n > WARM_N:
            if n - self.last_peak > DECAY_AFTER:
                self.peak_level *= DECAY
                self.thr = THR_FRAC * self.peak_level
            # 3b. local maximum of the MWI at n-1, above threshold
            if (self.m1 > self.thr and self.m1 >= self.m2 and self.m1 > mwi
                    and (n - 1 - self.last_peak) > REFRACT):
                event = self._on_qrs(n)
        self.m2 = self.m1
        self.m1 = mwi
        self.n = n + 1
        return event

    def _on_qrs(self, n):
        # R peak = max |y| over the integration window ending at n-1
        best, r = -1.0, n - 1
        for i in range(n - 1 - MWI_W, n):
            v = abs(self.ybuf[i % BUF_LEN])
            if v > best:
                best, r = v, i
        self.last_peak = n - 1
        self.peak_level = (1 - PEAK_ALPHA) * self.peak_level + PEAK_ALPHA * self.m1
        self.thr = THR_FRAC * self.peak_level
        self.r_all.append(r)
        self.peaks.append(r)
        if len(self.peaks) > 6:
            self.peaks.pop(0)
        self.pending.append(r)

        # 5. classify the beat TWO back (needs its prev RR and both next RRs)
        if len(self.peaks) >= 4:
            center = self.peaks[-3]
            td = self.records.get(center)
            if td is not None:
                P = self.peaks
                prev_rr = (P[-3] - P[-4]) / FS
                next_rr = (P[-2] - P[-3]) / FS
                diffs = [(P[i + 1] - P[i]) / FS for i in range(len(P) - 1)]
                local = sum(diffs) / len(diffs)
                ratio = prev_rr / next_rr
                feats = [prev_rr, next_rr, local, ratio] + list(td)
                return {"sample": int(center), "time": center / FS,
                        "features": feats, "label": None, "proba": None}
        return None

    def _capture(self, r):
        idx = np.arange(r - BEFORE, r + AFTER) % BUF_LEN
        beat = self.ybuf[idx]
        self.records[r] = time_domain_features(beat[None, :])[0]
        if len(self.records) > 8:
            del self.records[min(self.records)]

    def _classify(self, events):
        X = np.array([e["features"] for e in events])
        proba = self.model.predict_proba(X)
        classes = self.model.classes_
        for e, pr in zip(events, proba):
            k = int(np.argmax(pr))
            e["label"] = str(classes[k])
            e["proba"] = float(pr[k])
