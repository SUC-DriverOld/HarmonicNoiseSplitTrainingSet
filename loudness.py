"""Integrated LUFS normalization with shared stem gains and 4x peak protection."""

import numpy as np
from numba import njit
import pyloudnorm as pyln
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import firwin


def validate_loudness_options(target_lufs, peak_db):
    if target_lufs is not None and (
        not np.isfinite(target_lufs) or not -60 <= target_lufs <= -5
    ):
        raise ValueError("target_lufs must be finite and in [-60, -5], or None")
    if not np.isfinite(peak_db) or not -12 <= peak_db <= 0:
        raise ValueError("peak_db must be finite and in [-12, 0]")


# The four phases of the centered 81-tap Kaiser FIR used by resample_poly(x, 4, 1).
_FIR = np.pad(4 * firwin(81, 0.25, window=("kaiser", 5.0)), (0, 3))
_PHASES = tuple(np.ascontiguousarray(_FIR[p::4][::-1]) for p in range(4))


@njit(cache=True)
def _filter_phase(padded, taps):
    # Fixed 21-tap inner loop allows SIMD across samples in a single CPU thread.
    out = np.empty(len(padded) - 20)
    for i in range(len(out)):
        value = 0.0
        for j in range(21):
            value += padded[i + j] * taps[j]
        out[i] = value
    return out


def peak_trace(harmonic, noise, mixture=None):
    """4x per-sample peak across stems and mix, including zero-padded edges."""
    harmonic = np.asarray(harmonic, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    mix = harmonic + noise if mixture is None else mixture
    trace = np.maximum(np.maximum(np.abs(harmonic), np.abs(noise)), np.abs(mix))
    hp, npad = np.pad(harmonic, 10), np.pad(noise, 10)
    mp = None if mixture is None else np.pad(mix, 10)
    for taps in _PHASES:
        h = _filter_phase(hp, taps)
        n = _filter_phase(npad, taps)
        # FIR is linear; only the rounded float32 export needs an independent mix pass.
        m = h + n if mp is None else _filter_phase(mp, taps)
        for stream in (h, n, m):
            np.maximum(trace, np.abs(stream, out=stream), out=trace)
    return trace


def normalize_pair(harmonic, noise, sr, target_lufs=-16.5, peak_db=-1.0):
    """Normalize the MIXTURE, applying identical gains to both stems.

    The linked smooth limiter reduces excessive crest factor before remeasuring
    integrated loudness. Every output, including individual stems, is peak guarded.
    Clips shorter than the standard 400 ms gate get peak protection only.
    """
    validate_loudness_options(target_lufs, peak_db)
    harmonic = np.asarray(harmonic, dtype=np.float64).copy()
    noise = np.asarray(noise, dtype=np.float64).copy()
    if harmonic.ndim != 1 or harmonic.shape != noise.shape or not harmonic.size:
        raise ValueError("harmonic and noise must be non-empty, aligned mono arrays")
    if not np.all(np.isfinite(harmonic)) or not np.all(np.isfinite(noise)):
        raise ValueError("audio must be finite")
    if not isinstance(sr, (int, np.integer)) or sr < 8000:
        raise ValueError("sr must be an integer >= 8000")
    meter = pyln.Meter(sr)
    measurable = len(harmonic) >= int(np.ceil(sr * meter.block_size))
    initial_lufs = (
        float(meter.integrated_loudness(harmonic + noise))
        if measurable
        else float("-inf")
    )
    current_lufs = initial_lufs
    ceiling = 10 ** (peak_db / 20)
    # Leave 0.3 dB for reconstruction-filter differences and float32 rounding.
    working_ceiling = ceiling * 10 ** (-0.3 / 20)
    uniform_gain_db = 0.0
    limiter_applied = False
    iterations = 0

    if target_lufs is not None and np.isfinite(current_lufs):
        for iterations in range(1, 17):
            delta = target_lufs - current_lufs
            gain = 10 ** (delta / 20)
            harmonic *= gain
            noise *= gain
            uniform_gain_db += delta
            trace = peak_trace(harmonic, noise)
            if trace.max() > working_ceiling:
                limiter_applied = True
                radius = max(1, int(0.002 * sr))
                # Expand the peaks before averaging the gain twice. This anticipates
                # transients and keeps a continuous common gain below the local limit.
                envelope = maximum_filter1d(trace, size=4 * radius + 1, mode="nearest")
                gain_curve = np.minimum(
                    1.0, working_ceiling / np.maximum(envelope, 1e-12)
                )
                for _ in range(2):
                    gain_curve = uniform_filter1d(
                        gain_curve, size=2 * radius + 1, mode="nearest"
                    )
                harmonic *= gain_curve
                noise *= gain_curve
            current_lufs = float(meter.integrated_loudness(harmonic + noise))
            if abs(current_lufs - target_lufs) <= 0.05:
                break

    # A single check on the actual float32 outputs replaces two float64 peak scans.
    harmonic = harmonic.astype(np.float32)
    noise = noise.astype(np.float32)
    mixture = harmonic + noise

    def output_peak():
        return float(
            peak_trace(
                harmonic.astype(float), noise.astype(float), mixture.astype(float)
            ).max()
        )

    peak = output_peak()
    if peak > working_ceiling:
        gain = working_ceiling / peak
        harmonic *= gain
        noise *= gain
        mixture = harmonic + noise
        peak = output_peak()
        uniform_gain_db += float(20 * np.log10(gain))
        limiter_applied = True
    measured = (
        float(meter.integrated_loudness(mixture)) if measurable else float("-inf")
    )
    if not measurable:
        status = "short_clip_peak_only"
    elif not np.isfinite(measured):
        status = "silence"
    elif target_lufs is None:
        status = "peak_only"
    else:
        status = "ok" if abs(measured - target_lufs) <= 0.15 else "peak_limited"
    metadata = {
        "target_lufs": target_lufs,
        "measured_lufs": measured if np.isfinite(measured) else None,
        "input_lufs": initial_lufs if np.isfinite(initial_lufs) else None,
        "peak_ceiling_db": peak_db,
        "max_stem_or_mix_peak_db_4x": float(20 * np.log10(peak)) if peak > 0 else None,
        "limiter_applied": limiter_applied,
        "uniform_gain_db": uniform_gain_db,
        "iterations": iterations,
        "status": status,
        "meter": "pyloudnorm BS.1770 integrated, 400 ms blocks",
    }
    return harmonic, noise, metadata
