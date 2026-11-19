"""Dataset adapters. The synthesis and offline exporter do not require torch."""

import random

import numpy as np

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    class Dataset:
        """Sequence-compatible fallback when torch is not installed."""


from paired_generator import PairedGenerator
from loudness import normalize_pair, validate_loudness_options


class _GeneratedPairs(Dataset):
    def _configure(
        self,
        harmonic_generator,
        noise_generator,
        sr,
        hop_length,
        cropsize,
        mode,
        random_seed,
        consonant_augmentation,
        target_lufs,
        peak_db,
    ):
        if mode not in ("noise_as_inst", "harmonic_as_inst"):
            raise ValueError(f"Invalid mode: {mode}")
        if not isinstance(cropsize, (int, np.integer)) or cropsize < 2:
            raise ValueError("cropsize must be an integer >= 2")
        if not isinstance(hop_length, (int, np.integer)) or hop_length < 1:
            raise ValueError("hop_length must be a positive integer")
        validate_loudness_options(target_lufs, peak_db)
        self.target_lufs = target_lufs
        self.peak_db = peak_db
        self.harmonic_generator = harmonic_generator
        self.noise_generator = noise_generator
        self.sr = sr
        self.hop_length = hop_length
        self.cropsize = cropsize
        self.mode = mode
        self.n_samples = (cropsize - 1) * hop_length
        self.waveform_sec = self.n_samples / sr
        self.pair_generator = PairedGenerator(
            sr=sr,
            hopsize=hop_length,
            random_seed=random_seed,
            harmonic_generator=harmonic_generator,
            noise_generator=noise_generator,
            consonant_augmentation=consonant_augmentation,
            target_lufs=None,
            peak_db=peak_db,
        )

    def generate_pair(self):
        harmonic, noise = self.pair_generator.generate_pair(self.n_samples)
        return harmonic[None, :], noise[None, :]

    def _generate_item(self):
        harmonic, noise = self.generate_pair()
        target = noise if self.mode == "noise_as_inst" else harmonic
        return harmonic + noise, target

    def _finish_item(self, X, y):
        # Apply the requested loudness AFTER copy/mixup, when the final mixture is known.
        if self.target_lufs is None:
            return X, y
        if self.mode == "noise_as_inst":
            harmonic, noise = X[0] - y[0], y[0]
        else:
            harmonic, noise = y[0], X[0] - y[0]
        harmonic, noise, _ = normalize_pair(
            harmonic, noise, self.sr, self.target_lufs, self.peak_db
        )
        target = noise if self.mode == "noise_as_inst" else harmonic
        return (harmonic + noise)[None, :], target[None, :]


class GeneratedHarmonicNoiseSplitTrainingSet(_GeneratedPairs):
    def __init__(
        self,
        harmonic_generator,
        noise_generator,
        sr,
        hop_length,
        cropsize,
        copy_rate,
        mixup_rate,
        mixup_alpha,
        mode="noise_as_inst",
        random_seed=None,
        consonant_augmentation=True,
        target_lufs=-16.5,
        peak_db=-1.0,
    ):
        if not 0 <= copy_rate <= 1 or not 0 <= mixup_rate <= 1:
            raise ValueError("copy_rate and mixup_rate must be in [0, 1]")
        if not np.isfinite(mixup_alpha) or mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive and finite")
        self._configure(
            harmonic_generator,
            noise_generator,
            sr,
            hop_length,
            cropsize,
            mode,
            random_seed,
            consonant_augmentation,
            target_lufs,
            peak_db,
        )
        self.copy_rate = copy_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        # Preserve DataLoader's global RNG seeding when no explicit seed is supplied.
        self.rng = (
            np.random.default_rng(random_seed) if random_seed is not None else None
        )
        self.py_rng = random.Random(random_seed) if random_seed is not None else random

    def __len__(self):
        return 1000

    def do_aug(self, X, y):
        max_amp = max(float(np.abs(X).max()), float(np.abs(y).max()))
        if max_amp > 0 and self.target_lufs is None:
            max_shift = min(1.0, np.log10(1 / max_amp))
            log10_shift = self.py_rng.uniform(min(-1.0, max_shift), max_shift)
            gain = np.float32(10**log10_shift)
            X, y = X * gain, y * gain
        rng = self.rng if self.rng is not None else np.random
        if rng.random() < self.copy_rate:
            X = y.copy()
        return X, y

    def do_mixup(self, X, y):
        rng = self.rng if self.rng is not None else np.random
        idx = (
            int(rng.integers(len(self)))
            if self.rng is not None
            else int(rng.randint(len(self)))
        )
        X_i, y_i = self.__getitem__(idx, skip_mix=True)
        lam = np.float32(rng.beta(self.mixup_alpha, self.mixup_alpha))
        return lam * X + (1 - lam) * X_i, lam * y + (1 - lam) * y_i

    def __getitem__(self, idx, skip_mix=False):
        X, y = self.do_aug(*self._generate_item())
        rng = self.rng if self.rng is not None else np.random
        if not skip_mix and rng.random() < self.mixup_rate:
            X, y = self.do_mixup(X, y)
        return (X, y) if skip_mix else self._finish_item(X, y)


class GeneratedHarmonicNoiseSplitValidationSet(_GeneratedPairs):
    def __init__(
        self,
        harmonic_generator,
        noise_generator,
        sr,
        cropsize,
        val_size=10,
        mode="noise_as_inst",
        hop_length=None,
        random_seed=None,
        consonant_augmentation=True,
        target_lufs=-16.5,
        peak_db=-1.0,
    ):
        if not isinstance(val_size, (int, np.integer)) or val_size < 1:
            raise ValueError("val_size must be a positive integer")
        # Infer the generator hop for existing callers; cropsize is NOT a hop length.
        if hop_length is None:
            hop_length = getattr(harmonic_generator, "hopsize", 512)
        self._configure(
            harmonic_generator,
            noise_generator,
            sr,
            hop_length,
            cropsize,
            mode,
            random_seed,
            consonant_augmentation,
            target_lufs,
            peak_db,
        )
        self.val_size = val_size
        self.cache = [
            self._finish_item(*self._generate_item()) for _ in range(val_size)
        ]

    def __len__(self):
        return self.val_size

    def __getitem__(self, idx):
        return self.cache[idx]
