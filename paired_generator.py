"""Shared NumPy/SciPy synthesis for training and WAV export; no torch dependency."""

import numpy as np

from consonants import ConsonantSynthesizer, rms
from random_harmonic import GeneratorHarmonic
from random_noise import GeneratorNoise
from loudness import normalize_pair, validate_loudness_options


class PairedGenerator:
    """Generate aligned periodic/aperiodic stems with a shared articulation plan.

    All friction, release bursts, aspiration and vowel breathiness belong to noise.
    Only periodic excitation (including voiced consonants) belongs to harmonic.
    """

    def __init__(
        self,
        sr=44100,
        hopsize=512,
        random_seed=None,
        harmonic_generator=None,
        noise_generator=None,
        consonants=None,
        snr_db=(-5.0, 20.0),
        background_db=(-42.0, -26.0),
        peak_range=(0.65, 0.95),
        consonant_augmentation=True,
        target_lufs=-16.5,
        peak_db=-1.0,
    ):
        if not isinstance(sr, (int, np.integer)) or sr < 8000:
            raise ValueError("sr must be an integer >= 8000")
        if not isinstance(hopsize, (int, np.integer)) or not 1 <= hopsize <= 2048:
            raise ValueError("hopsize must be an integer in [1, 2048]")
        for name, bounds in (
            ("snr_db", snr_db),
            ("background_db", background_db),
            ("peak_range", peak_range),
        ):
            if (
                len(bounds) != 2
                or not np.all(np.isfinite(bounds))
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must contain two finite, ascending values")
        if not 0 < peak_range[0] <= peak_range[1] < 1:
            raise ValueError("peak_range must lie strictly between 0 and 1")
        # These bounds avoid numerical overflow and unusably small training targets.
        if not -100 <= snr_db[0] <= snr_db[1] <= 100:
            raise ValueError("snr_db must lie in [-100, 100]")
        if not -120 <= background_db[0] <= background_db[1] <= 0:
            raise ValueError("background_db must lie in [-120, 0]")
        validate_loudness_options(target_lufs, peak_db)
        self.target_lufs = target_lufs
        self.peak_db = peak_db
        self.sr = sr
        self.hopsize = hopsize
        self.snr_db = tuple(snr_db)
        self.background_db = tuple(background_db)
        self.peak_range = tuple(peak_range)
        self.consonant_augmentation = consonant_augmentation
        seeds = np.random.SeedSequence(random_seed).spawn(4)
        self.rng = np.random.default_rng(seeds[0])
        self.harmonic_generator = (
            harmonic_generator
            if harmonic_generator is not None
            else GeneratorHarmonic(sr=sr, hopsize=hopsize, random_seed=seeds[1])
        )
        self.noise_generator = (
            noise_generator
            if noise_generator is not None
            else GeneratorNoise(sr=sr, hopsize=hopsize, random_seed=seeds[2])
        )
        self.consonants = ConsonantSynthesizer(sr, seeds[3], consonants)
        for generator in (self.harmonic_generator, self.noise_generator):
            if getattr(generator, "sr", sr) != sr:
                raise ValueError("all generators must use the same sample rate")
        self.last_metadata = None

    @staticmethod
    def _check_audio(audio, n_samples, name):
        audio = np.asarray(audio, dtype=np.float64)
        if audio.shape != (n_samples,) or not np.all(np.isfinite(audio)):
            raise ValueError(
                f"{name} must return exactly {n_samples} finite mono samples"
            )
        return audio

    def generate_pair(self, n_samples):
        """Return (harmonic, noise), each mono float32 with exactly n_samples."""
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        harmonic_source = self.harmonic_generator
        if isinstance(harmonic_source, GeneratorHarmonic):
            harmonic = harmonic_source.generate_harmonic(
                n_samples, apply_mask=not self.consonant_augmentation
            )
        else:
            harmonic = harmonic_source.generate_harmonic(n_samples=n_samples)
        harmonic = self._check_audio(harmonic, n_samples, "harmonic_generator")
        harmonic = 0.15 * harmonic / max(rms(harmonic), 1e-12)

        # Avoid generating a second unrelated consonant schedule inside GeneratorNoise.
        background_method = getattr(self.noise_generator, "generate_background", None)
        if background_method is None:
            background_method = self.noise_generator.generate_noise
        background = self._check_audio(
            background_method(n_samples=n_samples), n_samples, "noise_generator"
        )
        background_level = self.rng.uniform(*self.background_db)
        background = (
            background
            * (0.15 * 10 ** (background_level / 20))
            / max(rms(background), 1e-12)
        )
        events = []
        if self.consonant_augmentation:
            voicing, events = self.consonants.make_plan(n_samples)
            harmonic *= voicing
            noise = background + self.consonants.synthesize(n_samples, events)
        else:
            noise = background

        requested_snr = float(self.rng.uniform(*self.snr_db))
        harmonic_rms, noise_rms = rms(harmonic), rms(noise)
        noise_gain = 1.0
        if harmonic_rms > 1e-12 and noise_rms > 1e-12:
            noise_gain = harmonic_rms / (noise_rms * 10 ** (requested_snr / 20))
            noise *= noise_gain

        # First apply a common peak gain; loudness control below also shares all gains.
        peak = max(
            np.max(np.abs(harmonic)),
            np.max(np.abs(noise)),
            np.max(np.abs(harmonic + noise)),
        )
        target_peak = float(self.rng.uniform(*self.peak_range))
        gain = target_peak / peak if peak > 1e-12 else 1.0
        harmonic = (harmonic * gain).astype(np.float32)
        noise = (noise * gain).astype(np.float32)
        harmonic, noise, loudness = normalize_pair(
            harmonic, noise, self.sr, self.target_lufs, self.peak_db
        )
        actual_snr = None
        if rms(harmonic) > 1e-12 and rms(noise) > 1e-12:
            actual_snr = float(20 * np.log10(rms(harmonic) / rms(noise)))
        self.last_metadata = {
            "sample_rate": self.sr,
            "n_samples": int(n_samples),
            "duration_seconds": n_samples / self.sr,
            "snr_db": actual_snr,
            "requested_snr_db": requested_snr,
            "background_db_before_snr": float(background_level),
            "noise_gain": float(noise_gain),
            "common_gain": float(gain),
            "consonant_augmentation": bool(self.consonant_augmentation),
            "events": [event.to_dict() for event in events],
            "loudness": loudness,
        }
        return harmonic, noise

    def generate(self, n_samples):
        """Return (harmonic, noise, mixture) using the exported float32 stem sum."""
        harmonic, noise = self.generate_pair(n_samples)
        return harmonic, noise, harmonic + noise
