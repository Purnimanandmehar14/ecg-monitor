"""
data_loader.py
---------------
Downloads (if needed) and loads records from the MIT-BIH Arrhythmia
Database using the `wfdb` package, which talks directly to PhysioNet.

MIT-BIH records are named by 3-digit patient IDs, e.g. 100, 101, 103...
Each record has:
    - a .dat file (raw signal)
    - a .hea file (header: sampling rate, gain, etc.)
    - a .atr file (beat-by-beat annotations: N, L, R, V, A, F, etc.)
"""

import os
import wfdb

# A commonly used subset of MIT-BIH record numbers.
# Feel free to extend this list to the full 48-record set.
DEFAULT_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111",
    "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203",
    "205", "207", "208", "209", "210", "212", "213", "214",
    "215", "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234"
]

# AAMI-style grouping of MIT-BIH annotation symbols into 5 superclasses.
# N = Normal, S = Supraventricular ectopic, V = Ventricular ectopic,
# F = Fusion, Q = Unknown/paced
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    'P': 'Q', '/': 'Q', 'f': 'Q', 'u': 'Q'
}


def download_records(record_list=None, db_dir="data/mitdb"):
    """
    Downloads the given MIT-BIH records into db_dir if not already present.
    Requires internet access to physionet.org.
    """
    if record_list is None:
        record_list = DEFAULT_RECORDS

    os.makedirs(db_dir, exist_ok=True)
    for rec in record_list:
        rec_path = os.path.join(db_dir, rec + ".dat")
        if not os.path.exists(rec_path):
            print(f"Downloading record {rec} ...")
            wfdb.dl_database("mitdb", dl_dir=db_dir, records=[rec])
        else:
            print(f"Record {rec} already present, skipping.")


def load_record(record_id, db_dir="data/mitdb"):
    """
    Loads a single record's signal + annotations.

    Returns:
        signal (np.ndarray): shape (n_samples, n_channels)
        fs (int): sampling frequency in Hz
        ann_samples (np.ndarray): sample indices of annotated beats
        ann_symbols (list[str]): raw MIT-BIH beat symbols at those indices
    """
    rec_path = os.path.join(db_dir, record_id)
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, "atr")

    signal = record.p_signal          # (n_samples, n_channels)
    fs = record.fs
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    return signal, fs, ann_samples, ann_symbols


def load_all_records(record_list=None, db_dir="data/mitdb"):
    """
    Loads every record in record_list, returning a dict keyed by record id.
    """
    if record_list is None:
        record_list = DEFAULT_RECORDS

    data = {}
    for rec in record_list:
        try:
            signal, fs, ann_samples, ann_symbols = load_record(rec, db_dir)
            data[rec] = {
                "signal": signal,
                "fs": fs,
                "ann_samples": ann_samples,
                "ann_symbols": ann_symbols,
            }
        except FileNotFoundError:
            print(f"Record {rec} not found in {db_dir}, skipping. "
                  f"Run download_records() first.")
    return data


if __name__ == "__main__":
    # Quick smoke test: download a couple of records and print basic info
    test_records = ["100", "101"]
    download_records(test_records)
    data = load_all_records(test_records)
    for rec, d in data.items():
        print(f"Record {rec}: signal shape={d['signal'].shape}, "
              f"fs={d['fs']}, n_beats={len(d['ann_samples'])}")
