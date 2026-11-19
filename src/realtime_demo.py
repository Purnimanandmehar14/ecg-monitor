"""
realtime_demo.py
-----------------
"Real-time ECG Analysis -> Alert / Display" stage.

Streams a signal through the StreamEngine in small chunks (as a live ADC
feed would arrive), prints an alert for every abnormal beat, and renders a
monitor-style plot.

  python realtime_demo.py                          # synthetic demo signal
  python realtime_demo.py --csv my_ecg.csv         # one column of samples @ 360 Hz
  python realtime_demo.py --realtime               # pace the console output in real time

Not a medical device. For education and prototyping only.
"""

import argparse
import os
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stream_engine as se
from synthetic_ecg import generate_record

CLASS_NAMES = {"N": "Normal", "S": "Supraventricular ectopic",
               "V": "Ventricular ectopic", "F": "Fusion", "Q": "Unknown/paced"}
CLASS_COLORS = {"N": "#2ca02c", "S": "#ff7f0e", "V": "#d62728",
                "F": "#9467bd", "Q": "#7f7f7f"}


def format_alert(ev):
    name = CLASS_NAMES.get(ev["label"], ev["label"])
    return (f"[{ev['time']:7.2f}s] ALERT  {name} beat ({ev['label']}), "
            f"confidence {ev['proba']:.0%}")


def analyze(signal, model, chunk=36, realtime=False, on_alert=print):
    """Stream `signal` through the engine in chunks of `chunk` samples."""
    eng = se.StreamEngine(model=model)
    events = []
    for i in range(0, len(signal), chunk):
        for ev in eng.process(signal[i:i + chunk]):
            events.append(ev)
            if ev["label"] != "N" and on_alert:
                on_alert(format_alert(ev))
        if realtime:
            time.sleep(chunk / se.FS)
    return events


def render_monitor(signal, events, title="ECG monitor", seconds=(0, 20)):
    t = np.arange(len(signal)) / se.FS
    s0, s1 = seconds
    m = (t >= s0) & (t <= s1)
    sig = signal - np.median(signal)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[m], sig[m], color="#222", lw=0.8)
    shown = set()
    for ev in events:
        if not (s0 <= ev["time"] <= s1):
            continue
        c = CLASS_COLORS.get(ev["label"], "k")
        # display only: the causal filter delays detection by ~3 samples, so
        # snap the marker to the top of the QRS (-120 ms .. +60 ms)
        a, b = max(ev["sample"] - 43, 0), min(ev["sample"] + 22, len(sig) - 1)
        i = a + int(np.argmax(sig[a:b + 1]))
        ax.plot(i / se.FS, sig[i], "o", color=c, ms=6,
                label=CLASS_NAMES[ev["label"]] if ev["label"] not in shown else None)
        shown.add(ev["label"])
        if ev["label"] != "N":
            ax.annotate(ev["label"], (i / se.FS, sig[i]), xytext=(0, 10),
                        textcoords="offset points", ha="center", color=c,
                        fontweight="bold")
    n_ab = sum(e["label"] != "N" for e in events)
    ax.set_title(f"{title}  |  {len(events)} beats classified, {n_ab} abnormal")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (a.u.)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))                  # headroom for labels
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV/text file with one ECG sample per line (360 Hz)")
    ap.add_argument("--model", default=os.path.join("..", "models", "ecg_rf_embedded.joblib"))
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--out", default=os.path.join("..", "outputs", "realtime_monitor.png"))
    args = ap.parse_args()

    model = joblib.load(args.model)
    if args.csv:
        signal = np.loadtxt(args.csv, delimiter=",", ndmin=1)
        if signal.ndim > 1:
            signal = signal[:, 0]
        truth = None
    else:
        rec = generate_record(60, seed=777, p_v=0.12, p_s=0.08)
        signal, truth = rec["signal"], rec
    print(f"Streaming {len(signal) / se.FS:.0f} s of ECG at {se.FS} Hz ...\n")
    events = analyze(signal, model, realtime=args.realtime)

    counts = {c: sum(e["label"] == c for e in events) for c in sorted({e["label"] for e in events})}
    print(f"\nSummary: {len(events)} beats classified -> {counts}")
    if truth is not None:      # compare with the generator's ground truth
        sym = {"A": "S", "N": "N", "V": "V"}
        ann = np.array(truth["ann_samples"])
        hit = tot = 0
        for e in events:
            j = int(np.argmin(np.abs(ann - e["sample"])))
            if abs(ann[j] - e["sample"]) <= 43:
                tot += 1
                hit += sym[truth["ann_symbols"][j]] == e["label"]
        print(f"Agreement with synthetic ground truth: {hit}/{tot} beats")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    render_monitor(signal, events, seconds=(0, 25)).savefig(args.out, dpi=130)
    print("Monitor display saved to", args.out)


if __name__ == "__main__":
    main()
