"""
preprocessing.py
-----------------
Signal preprocessing stage of the pipeline:

    Noise Removal (baseline wander / motion artifact)
        + Normalization / Filtering

Implements:
    - Bandpass Butterworth filter (removes baseline wander + high-freq noise)
    - Notch filter (removes 50/60 Hz powerline interference)
    - Wavelet-based baseline wander removal (alternative method)
    - Z-score normalization
    - Beat segmentation around R-peaks
"""

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Butterworth bandpass filter.
    0.5 Hz removes baseline wander, 40 Hz removes muscle/motion noise
    while keeping the QRS complex intact.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=60.0, quality=30.0):
    """
    Removes powerline interference (60 Hz in US, 50 Hz in EU/India).
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)


def wavelet_denoise(signal, wavelet="db6", level=8):
    """
    Alternative baseline-wander removal using discrete wavelet transform.
    Zeros out the lowest-frequency approximation coefficients (which
    capture baseline drift) and reconstructs the signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # remove baseline (approx coeffs)
    return pywt.waverec(coeffs, wavelet)[: len(signal)]


def normalize(signal):
    """
    Z-score normalization: zero mean, unit variance.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std


def preprocess_signal(signal, fs, use_wavelet=False, mains_freq=60.0):
    """
    Full preprocessing pipeline for a single-channel ECG signal.
    """
    clean = bandpass_filter(signal, fs)
    clean = notch_filter(clean, fs, freq=mains_freq)
    if use_wavelet:
        clean = wavelet_denoise(clean)
    clean = normalize(clean)
    return clean


def segment_beats(signal, fs, r_peak_samples, window_before=0.25, window_after=0.45):
    """
    Extracts fixed-length beat windows centered on each R-peak.

    window_before/after are in seconds (defaults capture a full QRS-T
    complex: ~0.25s before R-peak, ~0.45s after).

    Returns:
        beats (np.ndarray): shape (n_beats, window_length)
        valid_idx (np.ndarray): indices into r_peak_samples that were kept
                                 (edge beats too close to signal start/end
                                 are dropped)
    """
    before = int(window_before * fs)
    after = int(window_after * fs)
    beats = []
    valid_idx = []

    for i, r in enumerate(r_peak_samples):
        start, end = r - before, r + after
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
        valid_idx.append(i)

    return np.array(beats), np.array(valid_idx)
