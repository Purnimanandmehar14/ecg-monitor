"""
feature_extraction.py
----------------------
Extracts time-domain and frequency-domain features per beat, to feed
into the ML model.

Time-domain:
    - RR interval (previous, next, local average)
    - Beat amplitude stats (max, min, mean, std)
    - QRS width estimate

Frequency-domain:
    - Power spectral density band energies (via Welch's method)
    - Dominant frequency
"""

import numpy as np
from scipy.signal import welch

# numpy >=2.0 renamed trapz -> trapezoid; keep this working on both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def rr_interval_features(r_peak_samples, valid_idx, fs):
    """
    Computes RR-interval-based features for each valid beat:
        - prev_rr:  interval to previous beat (seconds)
        - next_rr:  interval to next beat (seconds)
        - local_rr_mean: mean of the 5 surrounding RR intervals
        - rr_ratio: prev_rr / next_rr (captures irregularity, e.g. PVCs)
    """
    r_peak_samples = np.asarray(r_peak_samples)
    rr_all = np.diff(r_peak_samples) / fs  # seconds, len = n_beats - 1

    features = []
    for idx in valid_idx:
        prev_rr = rr_all[idx - 1] if idx > 0 else np.nan
        next_rr = rr_all[idx] if idx < len(rr_all) else np.nan

        window = rr_all[max(0, idx - 3): idx + 2]
        local_mean = np.nanmean(window) if len(window) > 0 else np.nan

        ratio = prev_rr / next_rr if (next_rr and not np.isnan(next_rr) and next_rr != 0) else np.nan

        features.append([prev_rr, next_rr, local_mean, ratio])

    features = np.array(features, dtype=float)
    # Fill NaNs (first/last beat edge cases) with column means
    col_means = np.nanmean(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_means, inds[1])
    return features  # shape (n_beats, 4)


def time_domain_features(beats):
    """
    Per-beat amplitude/shape statistics.
    beats: shape (n_beats, window_length)
    """
    max_amp = np.max(beats, axis=1)
    min_amp = np.min(beats, axis=1)
    mean_amp = np.mean(beats, axis=1)
    std_amp = np.std(beats, axis=1)
    ptp_amp = max_amp - min_amp  # peak-to-peak, rough QRS amplitude proxy
    skewness = _skew(beats)

    return np.column_stack([max_amp, min_amp, mean_amp, std_amp, ptp_amp, skewness])


def _skew(beats):
    mean = np.mean(beats, axis=1, keepdims=True)
    std = np.std(beats, axis=1, keepdims=True)
    std[std == 0] = 1e-8
    return np.mean(((beats - mean) / std) ** 3, axis=1)


def frequency_domain_features(beats, fs):
    """
    Welch PSD-based features per beat:
        - energy in low band (0.5-10 Hz, dominated by T-wave/baseline)
        - energy in mid band (10-25 Hz, QRS complex)
        - energy in high band (25-45 Hz, noise-ish / fast components)
        - dominant frequency
    """
    feats = []
    for beat in beats:
        freqs, psd = welch(beat, fs=fs, nperseg=min(len(beat), 64))
        low = _trapz(psd[(freqs >= 0.5) & (freqs < 10)])
        mid = _trapz(psd[(freqs >= 10) & (freqs < 25)])
        high = _trapz(psd[(freqs >= 25) & (freqs < 45)])
        dom_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0
        feats.append([low, mid, high, dom_freq])
    return np.array(feats)


def extract_features(beats, r_peak_samples, valid_idx, fs):
    """
    Combines all feature groups into a single feature matrix.

    Returns:
        X (np.ndarray): shape (n_beats, n_features)
        feature_names (list[str])
    """
    rr_feats = rr_interval_features(r_peak_samples, valid_idx, fs)
    td_feats = time_domain_features(beats)
    fd_feats = frequency_domain_features(beats, fs)

    X = np.hstack([rr_feats, td_feats, fd_feats])
    feature_names = [
        "prev_rr", "next_rr", "local_rr_mean", "rr_ratio",
        "max_amp", "min_amp", "mean_amp", "std_amp", "ptp_amp", "skewness",
        "psd_low", "psd_mid", "psd_high", "dominant_freq",
    ]
    return X, feature_names
