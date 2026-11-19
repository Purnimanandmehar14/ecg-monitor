"""
simulate_video.py
-----------------
Animated live-monitor simulation of the whole pipeline.

The ECG is fed to the StreamEngine in small chunks (as an ADC would deliver
it); every detected beat is classified by the Random Forest and drawn on a
scrolling monitor, with an alert banner for abnormal beats and running counts.

  python simulate_video.py                       # synthetic ECG -> ../outputs/ecg_simulation.mp4
  python simulate_video.py --csv my_ecg.csv      # your own signal (360 Hz, one column)

Needs ffmpeg for MP4 output (falls back to GIF if it is missing).
Not a medical device. For education and prototyping only.
"""

import argparse
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.animation as animation

import stream_engine as se
from synthetic_ecg import generate_record

NAMES = {"N": "Normal", "S": "Supraventricular ectopic", "V": "Ventricular ectopic",
         "F": "Fusion", "Q": "Unknown/paced"}
COLORS = {"N": "#7CFFB2", "S": "#FFC857", "V": "#FF5C7A", "F": "#B388FF", "Q": "#9AA7B5"}
BG, PANEL, FG, SIG = "#04141F", "#0B2A3A", "#EAFBFB", "#4FE3D0"

FPS = 30
CHUNK = se.FS // FPS          # 12 samples per frame
WINDOW_S = 6.0


def run_engine(signal, model):
    """Stream the signal chunk by chunk; return per-frame lists of new events."""
    eng = se.StreamEngine(model=model)
    sig = signal - np.median(signal)
    per_frame = []
    for i in range(0, len(signal), CHUNK):
        out = []
        for ev in eng.process(signal[i:i + CHUNK]):
            # display only: snap the marker to the top of the QRS (-120..+60 ms)
            a, b = max(ev["sample"] - 43, 0), min(ev["sample"] + 22, len(sig) - 1)
            ev["disp"] = a + int(np.argmax(sig[a:b + 1]))
            out.append(ev)
        per_frame.append(out)
    return sig, per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--model", default=os.path.join("..", "models", "ecg_rf_embedded.joblib"))
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--out", default=os.path.join("..", "outputs", "ecg_simulation.mp4"))
    args = ap.parse_args()

    model = joblib.load(args.model)
    if args.csv:
        raw = np.loadtxt(args.csv, delimiter=",", ndmin=1)
        raw = raw[:, 0] if raw.ndim > 1 else raw
        raw = raw[: int(args.seconds * se.FS)]
    else:
        raw = generate_record(args.seconds, seed=777, p_v=0.12, p_s=0.08)["signal"]

    sig, per_frame = run_engine(raw, model)
    n_frames = len(per_frame)
    ylo, yhi = np.percentile(sig, 0.5), np.percentile(sig, 99.5)
    pad = 0.35 * (yhi - ylo)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.05, 0.22, 0.68, 0.56], facecolor=PANEL)
    ax.set_ylim(ylo - pad, yhi + 1.6 * pad)
    ax.set_yticks([])
    ax.grid(color="#16475C", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#16475C")
    ax.tick_params(colors="#9FC3CC")
    ax.set_xlabel("time (s)", color="#9FC3CC")
    line, = ax.plot([], [], color=SIG, lw=1.8)
    marks = {c: ax.plot([], [], "o", color=COLORS[c], ms=8, ls="none")[0] for c in COLORS}

    fig.text(0.05, 0.93, "ECG Arrhythmia Detection", color=FG, fontsize=24, fontweight="bold")
    fig.text(0.05, 0.885, "Live simulation: signal -> filter -> beat detection -> features -> Random Forest -> alert",
             color="#9FC3CC", fontsize=12)
    banner = fig.text(0.05, 0.095, "", fontsize=18, fontweight="bold", va="center",
                      bbox=dict(boxstyle="round,pad=0.5", fc="#0E7C86", ec="none"), color="white")
    fig.text(0.97, 0.02, "Educational prototype, not a medical device", color="#6FA3AE",
             fontsize=10, ha="right")

    # right-hand stats panel
    fig.patches.append(plt.Rectangle((0.76, 0.22), 0.21, 0.56, transform=fig.transFigure,
                                     fc=PANEL, ec="#16475C"))
    fig.text(0.775, 0.745, "HEART RATE", color="#9FC3CC", fontsize=11)
    hr_txt = fig.text(0.775, 0.67, "-- bpm", color=FG, fontsize=30, fontweight="bold")
    fig.text(0.775, 0.60, "BEATS CLASSIFIED", color="#9FC3CC", fontsize=11)
    cnt_txt = {}
    for k, c in enumerate(["N", "S", "V", "F"]):
        y = 0.54 - k * 0.075
        fig.text(0.775, y, c, color=COLORS[c], fontsize=20, fontweight="bold")
        cnt_txt[c] = fig.text(0.96, y, "0", color=FG, fontsize=20, ha="right")
        fig.text(0.81, y + 0.004, NAMES[c].split()[0], color="#9FC3CC", fontsize=11, va="bottom")
    time_txt = fig.text(0.775, 0.245, "t = 0.0 s", color="#9FC3CC", fontsize=12)

    shown, counts, texts = [], {"N": 0, "S": 0, "V": 0, "F": 0}, []
    state = {"alert": None, "alert_until": -1.0, "last_f": -1}

    def update(f):
        # replay events frame by frame (frames are rendered in order)
        new = per_frame[f] if f > state["last_f"] else []
        state["last_f"] = max(state["last_f"], f)
        for ev in new:
            shown.append(ev)
            if ev["label"] in counts:
                counts[ev["label"]] += 1
            if ev["label"] != "N":
                state["alert"], state["alert_until"] = ev, f / FPS + 1.6
        t_end = (f + 1) * CHUNK / se.FS
        t0 = max(0.0, t_end - WINDOW_S)
        i0, i1 = int(t0 * se.FS), (f + 1) * CHUNK
        tt = np.arange(i0, min(i1, len(sig))) / se.FS
        line.set_data(tt, sig[i0:i1])
        ax.set_xlim(t0, t0 + WINDOW_S)
        for t in texts:
            t.remove()
        texts.clear()
        for c, m in marks.items():
            pts = [e["disp"] / se.FS for e in shown if e["label"] == c and t0 <= e["disp"] / se.FS <= t_end]
            m.set_data(pts, [sig[int(p * se.FS)] for p in pts])
        for e in shown:
            x = e["disp"] / se.FS
            if e["label"] != "N" and t0 <= x <= t_end:
                texts.append(ax.annotate(e["label"], (x, sig[e["disp"]]), xytext=(0, 12),
                                         textcoords="offset points", ha="center",
                                         color=COLORS[e["label"]], fontweight="bold", fontsize=16))
        if len(shown) >= 4:
            rr = np.diff([e["sample"] for e in shown[-5:]]) / se.FS
            hr_txt.set_text(f"{60 / rr.mean():.0f} bpm")
        for c, tx in cnt_txt.items():
            tx.set_text(str(counts[c]))
        time_txt.set_text(f"t = {t_end:4.1f} s")
        if state["alert"] and f / FPS < state["alert_until"]:
            e = state["alert"]
            banner.set_text(f"ALERT  {NAMES[e['label']]} beat ({e['label']})   confidence {e['proba']:.0%}")
            banner.get_bbox_patch().set_facecolor(COLORS[e["label"]])
            banner.set_color("#04141F")
        else:
            banner.set_text("Monitoring: rhythm normal")
            banner.get_bbox_patch().set_facecolor("#0E7C86")
            banner.set_color("white")
        return []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    out = args.out
    if animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=1500, codec="libx264",
                                           extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"]))
    else:
        out = os.path.splitext(out)[0] + ".gif"
        anim.save(out, writer=PillowWriter(fps=15))
    print(f"Saved {out}: {n_frames} frames, {n_frames / FPS:.0f} s, "
          f"{len(shown)} beats classified {counts}")


if __name__ == "__main__":
    main()
