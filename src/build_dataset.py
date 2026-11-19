"""
build_dataset.py
-----------------
Runs the full pipeline for every record:

    load -> preprocess -> segment beats -> extract features -> label

and assembles a single (X, y) dataset ready for ML training.

Labels follow the AAMI 5-class scheme:
    N - Normal
    S - Supraventricular ectopic beat
    V - Ventricular ectopic beat
    F - Fusion beat
    Q - Unknown/paced beat
"""

import numpy as np

from data_loader import load_all_records, DEFAULT_RECORDS, AAMI_MAP
from preprocessing import preprocess_signal, segment_beats
from feature_extraction import extract_features


def build_dataset(record_list=None, db_dir="data/mitdb", channel=0):
    if record_list is None:
        record_list = DEFAULT_RECORDS

    raw_data = load_all_records(record_list, db_dir)

    X_all, y_all, groups_all = [], [], []

    for rec_id, d in raw_data.items():
        signal = d["signal"][:, channel]
        fs = d["fs"]
        ann_samples = d["ann_samples"]
        ann_symbols = d["ann_symbols"]

        # Keep only annotations that map to a valid AAMI class
        keep = [i for i, s in enumerate(ann_symbols) if s in AAMI_MAP]
        r_peaks = ann_samples[keep]
        labels = [AAMI_MAP[ann_symbols[i]] for i in keep]

        clean_signal = preprocess_signal(signal, fs)
        beats, valid_idx = segment_beats(clean_signal, fs, r_peaks)

        if len(beats) == 0:
            continue

        X, feature_names = extract_features(beats, r_peaks, valid_idx, fs)
        y = np.array([labels[i] for i in valid_idx])

        X_all.append(X)
        y_all.append(y)
        groups_all.append(np.full(len(y), rec_id))

        print(f"Record {rec_id}: {len(y)} beats processed.")

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups_all = np.concatenate(groups_all)

    return X_all, y_all, groups_all, feature_names


if __name__ == "__main__":
    X, y, groups, feature_names = build_dataset(["100", "101", "103"])
    print("Feature matrix shape:", X.shape)
    print("Feature names:", feature_names)
    print("Label distribution:", {c: int((y == c).sum()) for c in set(y)})
