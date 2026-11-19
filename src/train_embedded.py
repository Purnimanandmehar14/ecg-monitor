"""
train_embedded.py
------------------
Trains the compact model that runs on the microcontroller.

Key idea: the training data is produced by running the SAME streaming engine
that gets deployed (causal filter, running normalisation, detected R peaks),
so there is no train/deploy mismatch from preprocessing. Every detected beat
is labelled from the nearest reference annotation.

Usage (from the src/ folder):
  python train_embedded.py --source synthetic                  # offline demo
  python train_embedded.py --source mitdb --db-dir data/mitdb  # real data

Outputs: models/ecg_rf_embedded.joblib, outputs/metrics_embedded.json,
         outputs/confusion_matrix_embedded.png, embedded/ecg_config.h + ecg_model.h
"""

import argparse
import json
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

import stream_engine as se
from data_loader import AAMI_MAP, load_all_records, download_records, DEFAULT_RECORDS
from synthetic_ecg import generate_record
from export_to_c import export_all

TOL = int(0.12 * se.FS)          # detection<->annotation match tolerance (samples)


def collect_beats(signal, ann_samples, ann_symbols, group, mains=se.MAINS_HZ):
    """Run the streaming engine over one record; label detected beats."""
    keep = [i for i, s in enumerate(ann_symbols) if s in AAMI_MAP]
    a = np.asarray(ann_samples)[keep]
    lab = [AAMI_MAP[ann_symbols[i]] for i in keep]

    eng = se.StreamEngine(mains=mains)
    events = []
    for i in range(0, len(signal), 3600):
        events += eng.process(signal[i:i + 3600], classify=False)

    X, y = [], []
    for ev in events:
        j = int(np.argmin(np.abs(a - ev["sample"])))
        if abs(a[j] - ev["sample"]) <= TOL:
            X.append(ev["features"])
            y.append(lab[j])

    det = np.asarray(eng.r_all)
    post = a > se.WARM_N + 20
    tp_ann = sum(np.min(np.abs(det - x)) <= TOL for x in a[post]) if len(det) else 0
    tp_det = sum(np.min(np.abs(a - d)) <= TOL for d in det) if len(a) else 0
    stats = {"annotated": int(post.sum()), "found": int(tp_ann),
             "detections": int(len(det)), "true_detections": int(tp_det)}
    return np.array(X), np.array(y), np.full(len(y), group), stats


def build_embedded_dataset(source, db_dir, records, n_synth=24, synth_s=120,
                           mains=se.MAINS_HZ):
    Xs, ys, gs = [], [], []
    tot = {"annotated": 0, "found": 0, "detections": 0, "true_detections": 0}
    if source == "synthetic":
        recs = {f"syn{s:02d}": generate_record(synth_s, seed=s) for s in range(n_synth)}
    else:
        download_records(records, db_dir)
        recs = load_all_records(records, db_dir)
    for rid, d in recs.items():
        sig = d["signal"] if source == "synthetic" else d["signal"][:, 0]
        X, y, g, st = collect_beats(sig, d["ann_samples"], d["ann_symbols"], rid, mains)
        if len(y) == 0:
            continue
        Xs.append(X); ys.append(y); gs.append(g)
        for k in tot:
            tot[k] += st[k]
        print(f"  {rid}: {len(y)} labelled beats")
    det = {"sensitivity": round(tot["found"] / max(tot["annotated"], 1), 4),
           "ppv": round(tot["true_detections"] / max(tot["detections"], 1), 4)}
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "mitdb"], default="synthetic")
    ap.add_argument("--db-dir", default="data/mitdb")
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--trees", type=int, default=20)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--mains", type=float, default=se.MAINS_HZ)
    ap.add_argument("--models-dir", default="../models")
    ap.add_argument("--outputs-dir", default="../outputs")
    ap.add_argument("--export-dir", default="../embedded")
    args = ap.parse_args()

    records = args.records or DEFAULT_RECORDS
    print(f"Building dataset from '{args.source}' with the streaming engine ...")
    X, y, groups, det = build_embedded_dataset(args.source, args.db_dir, records,
                                               mains=args.mains)
    print("Detector:", det)
    print("Dataset:", X.shape, {c: int((y == c).sum()) for c in sorted(set(y))})

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(split.split(X, y, groups))
    model = RandomForestClassifier(n_estimators=args.trees, max_depth=args.depth,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])

    acc = accuracy_score(y[te], pred)
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="macro",
                                                  zero_division=0)
    print(f"\nAccuracy {acc:.4f} | macro precision {p:.4f} | recall {r:.4f} | F1 {f1:.4f}")
    print(classification_report(y[te], pred, zero_division=0))

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.models_dir, "ecg_rf_embedded.joblib"))

    labels = list(model.classes_)
    cm = confusion_matrix(y[te], pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Embedded model ({args.source} data)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outputs_dir, "confusion_matrix_embedded.png"), dpi=150)
    plt.close()

    info = export_all(model, args.export_dir, se.FEATURE_NAMES, args.mains)
    metrics = {"source": args.source, "accuracy": acc, "macro_precision": p,
               "macro_recall": r, "macro_f1": f1, "detector": det,
               "classes": [str(c) for c in labels], "model": info,
               "n_train_beats": int(len(tr)), "n_test_beats": int(len(te))}
    with open(os.path.join(args.outputs_dir, "metrics_embedded.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model, metrics, confusion matrix and C headers.")


if __name__ == "__main__":
    main()
