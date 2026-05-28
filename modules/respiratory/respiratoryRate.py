import numpy as np
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d


def analyse_respiratory_rate(intensity_log, fps):

    signal = np.array(intensity_log, dtype=float)
    n = len(signal)

    if n < fps * 3:
        return None

    time = np.arange(n) / fps

    # -----------------
    # Step 1: Detrend
    # -----------------
    signal_d = detrend(signal)

    # -----------------
    # Step 2: Smooth
    # -----------------
    smooth_win = max(3, int(fps * 0.5))
    signal_s = uniform_filter1d(signal_d, smooth_win)

    # -----------------
    # Step 3: FFT
    # -----------------
    freqs = np.fft.rfftfreq(n, d=1/fps)
    fft_vals = np.abs(np.fft.rfft(signal_s))

    mask = (freqs >= 0.1) & (freqs <= 0.8)

    fft_bpm = np.nan
    if np.any(mask):
        peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
        fft_bpm = peak_freq * 60

    # -----------------
    # Step 4: Peaks
    # -----------------
    norm = (signal_s - signal_s.min())
    norm = norm / (norm.max() + 1e-6)

    peaks, _ = find_peaks(norm, distance=int(fps))
    peak_bpm = (len(peaks) / (n / fps)) * 60 if len(peaks) > 0 else np.nan

    # -----------------
    # Step 5: Bandpass
    # -----------------
    nyq = fps / 2
    b, a = butter(3, [0.1/nyq, 0.8/nyq], btype="band")
    filtered = filtfilt(b, a, signal_d)

    norm2 = (filtered - filtered.min())
    norm2 = norm2 / (norm2.max() + 1e-6)

    band_peaks, _ = find_peaks(norm2, distance=int(fps))
    band_bpm = (len(band_peaks) / (n / fps)) * 60 if len(band_peaks) > 0 else np.nan

    # -----------------
    # FINAL BPM
    # -----------------
    values = [fft_bpm, peak_bpm, band_bpm]
    values = [v for v in values if not np.isnan(v)]

    final_bpm = float(np.median(values)) if values else None

    return {
        "fft_bpm": float(fft_bpm) if not np.isnan(fft_bpm) else None,
        "peak_bpm": float(peak_bpm) if not np.isnan(peak_bpm) else None,
        "band_bpm": float(band_bpm) if not np.isnan(band_bpm) else None,
        "final_bpm": final_bpm
    }