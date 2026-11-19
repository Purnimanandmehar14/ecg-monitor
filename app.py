"""
app.py -- Streamlit demo of the ECG arrhythmia monitor.

  pip install -r requirements.txt
  streamlit run app.py

Deploy free on Hugging Face Spaces (SDK: Streamlit) or Streamlit Community Cloud
by pushing this folder to a GitHub repo.

Educational prototype -- NOT a medical device.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import stream_engine as se                                     # noqa: E402
from realtime_demo import analyze, render_monitor, CLASS_NAMES  # noqa: E402
from synthetic_ecg import generate_record                      # noqa: E402

st.set_page_config(page_title="ECG Arrhythmia Monitor", page_icon="🫀", layout="wide")
st.title("🫀 ECG Arrhythmia Monitor")
st.caption("Streaming pipeline: filter → normalise → QRS detect → features → Random Forest → alert. "
           "Educational prototype, not a medical device.")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(ROOT, "models", "ecg_rf_embedded.joblib"))


model = load_model()

with st.sidebar:
    st.header("Input")
    mode = st.radio("Signal source", ["Synthetic demo", "Upload CSV"])
    if mode == "Synthetic demo":
        seed = st.number_input("Random seed", 0, 10_000, 777)
        p_v = st.slider("Ventricular ectopic rate", 0.0, 0.3, 0.10, 0.01)
        p_s = st.slider("Supraventricular ectopic rate", 0.0, 0.3, 0.08, 0.01)
        dur = st.slider("Duration (s)", 20, 120, 60, 5)
        signal = generate_record(dur, seed=int(seed), p_v=p_v, p_s=p_s)["signal"]
    else:
        up = st.file_uploader("CSV/text: one ECG sample per row (first column), 360 Hz", type=["csv", "txt"])
        signal = None
        if up is not None:
            signal = pd.read_csv(up, header=None).iloc[:, 0].astype(float).to_numpy()
    st.info("The bundled model is trained on SYNTHETIC beats so the app runs out of the box. "
            "Retrain on MIT-BIH with `src/train_embedded.py --source mitdb` for real use.")

if signal is None or len(signal) < 5 * se.FS:
    st.warning("Provide at least 5 seconds of ECG sampled at 360 Hz.")
    st.stop()

events = analyze(signal, model, on_alert=None)
labels = [e["label"] for e in events]
n_ab = sum(l != "N" for l in labels)

c1, c2, c3 = st.columns(3)
c1.metric("Beats classified", len(events))
c2.metric("Abnormal beats", n_ab)
c3.metric("Abnormal fraction", f"{n_ab / max(len(events), 1):.0%}")

if n_ab:
    st.error(f"⚠️ {n_ab} abnormal beat(s) detected: " +
             ", ".join(f"{CLASS_NAMES[k]} ×{labels.count(k)}" for k in sorted(set(labels)) if k != "N"))
else:
    st.success("✅ All classified beats look normal.")

start = st.slider("View window start (s)", 0, max(int(len(signal) / se.FS) - 20, 0), 0)
st.pyplot(render_monitor(signal, events, seconds=(start, start + 20)))

st.subheader("Beat log")
st.dataframe(pd.DataFrame([{"time (s)": round(e["time"], 2), "class": CLASS_NAMES[e["label"]],
                            "confidence": round(e["proba"], 2),
                            "RR prev (s)": round(e["features"][0], 3)} for e in events]),
             width="stretch", height=300)
