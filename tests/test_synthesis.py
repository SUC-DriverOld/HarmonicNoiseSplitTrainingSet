import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from scipy.interpolate import PchipInterpolator

from consonants import CONSONANTS, ConsonantSynthesizer, rms
from dataset import (
    GeneratedHarmonicNoiseSplitTrainingSet,
    GeneratedHarmonicNoiseSplitValidationSet,
)
from generate_wavs import export_pairs
from loudness import normalize_pair, peak_trace
from paired_generator import PairedGenerator
from random_harmonic import GeneratorHarmonic, _sum_partials, upsample
from random_noise import GeneratorNoise


class ToneGenerator:
    sr = 16000
    hopsize = 256

    def generate_harmonic(self, n_samples):
        return np.sin(2 * np.pi * 180 * np.arange(n_samples) / self.sr)


class SilentBackground:
    sr = 16000

    def generate_noise(self, n_samples):
        return np.zeros(n_samples)


def paired(seed=42, **options):
    return PairedGenerator(
        sr=16000,
        hopsize=256,
        random_seed=seed,
        harmonic_generator=ToneGenerator(),
        noise_generator=SilentBackground(),
        **options,
    )


class SynthesisTests(unittest.TestCase):
    def test_harmonic_recurrence_matches_direct_sum_on_glide(self):
        rng = np.random.default_rng(21)
        sr, hop, orders = 44100, 128, 200
        f0_frames = np.array([110.0, 190.0, 1100.0, 120.0, 190.0])
        amplitudes = rng.uniform(0.01, 0.2, (len(f0_frames), orders))
        amplitudes[1] = 0
        offsets = rng.uniform(0, 2 * np.pi, orders)
        interp = PchipInterpolator(np.arange(len(f0_frames)), amplitudes, axis=0)
        # Include high partials crossing Nyquist and the held last frame.
        f0 = upsample(f0_frames, hop)[:-13]
        phase = (np.cumsum(f0 / sr) % 1) * 2 * np.pi
        actual = _sum_partials(phase, f0, interp.c, amplitudes[-1], offsets, hop, sr)
        k = np.arange(1, orders + 1)
        amps = interp(np.minimum(np.arange(len(f0)) / hop, len(f0_frames) - 1))
        taper = np.clip((sr / 2 - f0[:, None] * k) / (0.02 * sr), 0, 1)
        reference = (amps * taper * np.sin(phase[:, None] * k + offsets)).sum(axis=1)
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-12)

    def test_original_generators_exact_lengths_and_short_inputs(self):
        for sr in (8000, 44100, 48000):
            harmonic = GeneratorHarmonic(sr=sr, random_seed=42)
            noise = GeneratorNoise(sr=sr, random_seed=43)
            for length in (1, 17, 511, 512, 513, 1601):
                with self.subTest(sr=sr, length=length):
                    for audio in (
                        harmonic.generate_harmonic(length),
                        noise.generate_noise(length),
                    ):
                        self.assertEqual(audio.shape, (length,))
                        self.assertTrue(np.all(np.isfinite(audio)))

    def test_mask_fixed_intervals_and_interpolation(self):
        harmonic = GeneratorHarmonic(
            mask_min0=0.01, mask_max0=0.01, mask_min1=0.01, mask_max1=0.01
        )
        mask = harmonic.generate_mask(40)
        self.assertTrue(np.any((mask > 0) & (mask < 1)))
        interpolated = upsample(np.array([0.0, 1.0, 0.0, 0.0]), 32)
        self.assertGreaterEqual(interpolated.min(), 0)
        self.assertLessEqual(interpolated.max(), 1)

    def test_unvoiced_friction_and_voiced_friction_labels(self):
        for phone, voiced in (("s", False), ("z", True)):
            gen = paired(consonants=[phone], target_lufs=None)
            harmonic, noise, mixture = gen.generate(16000)
            event = next(
                e for e in gen.last_metadata["events"] if e["kind"] == "frication"
            )
            section = slice(event["start"], event["end"])
            self.assertGreater(rms(noise[section]), 0.001)
            if voiced:
                self.assertGreater(rms(harmonic[section]), 0.001)
            else:
                self.assertTrue(np.all(harmonic[section] == 0))
            np.testing.assert_array_equal(mixture, harmonic + noise)

    def test_stop_closure_burst_and_aspiration(self):
        gen = paired(consonants=["t"], target_lufs=None)
        harmonic, noise = gen.generate_pair(16000)
        events = gen.last_metadata["events"]
        self.assertEqual(
            [e["kind"] for e in events[:3]], ["closure", "burst", "aspiration"]
        )
        closure, burst, aspiration = events[:3]
        self.assertTrue(np.all(harmonic[closure["start"] : aspiration["end"]] == 0))
        self.assertTrue(np.all(noise[closure["start"] : closure["end"]] == 0))
        self.assertGreater(rms(noise[burst["start"] : burst["end"]]), 0.001)
        self.assertGreater(rms(noise[aspiration["start"] : aspiration["end"]]), 0.001)
        self.assertLessEqual(burst["end"] - burst["start"], int(0.012 * 16000))

    def test_all_consonants_finite_and_bounded(self):
        for phone in CONSONANTS:
            with self.subTest(phone=phone):
                gen = paired(consonants=[phone], target_lufs=None)
                h, n, m = gen.generate(8000)
                for audio in (h, n, m):
                    self.assertTrue(np.isfinite(audio).all())
                    self.assertLess(np.abs(audio).max(), 1)
                for event in gen.last_metadata["events"]:
                    self.assertTrue(0 <= event["start"] < event["end"] <= 8000)

    def test_sibilants_have_high_frequency_noise(self):
        energies = {}
        for phone in ("s", "h"):
            synth = ConsonantSynthesizer(44100, 42, [phone])
            _, events = synth.make_plan(44100)
            audio = synth.synthesize(44100, events)
            first = events[0]
            section = audio[first.start : first.end]
            power = np.abs(np.fft.rfft(section)) ** 2
            freq = np.fft.rfftfreq(len(section), 1 / 44100)
            energies[phone] = power[freq > 4000].sum() / power.sum()
        self.assertGreater(energies["s"], 0.6)
        self.assertGreater(energies["s"], 5 * energies["h"])

    def test_seed_reproducibility_and_snr(self):
        a, b, c = (
            paired(seed=10, target_lufs=None, snr_db=(7, 7)),
            paired(seed=10, target_lufs=None, snr_db=(7, 7)),
            paired(seed=11, target_lufs=None),
        )
        first = a.generate(16000)
        second = b.generate(16000)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(first[2], c.generate(16000)[2]))
        self.assertAlmostEqual(
            20 * np.log10(rms(first[0]) / rms(first[1])), 7, places=5
        )

    def test_invalid_parameters(self):
        for options in (
            {"consonants": []},
            {"snr_db": (10, -10)},
            {"target_lufs": float("nan")},
            {"peak_db": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired(**options)
        with self.assertRaises(ValueError):
            paired().generate_pair(0)


class LoudnessTests(unittest.TestCase):
    def test_peak_trace_matches_scipy_at_boundaries_and_cancellation(self):
        rng = np.random.default_rng(3)
        for length in (1, 13, 1503):
            h = rng.normal(0, 0.4, length)
            h[[0, -1]] += 1.1
            n = -h + rng.normal(0, 0.02, length)
            rounded_mix = (h.astype("float32") + n.astype("float32")).astype(float)
            for mix in (None, rounded_mix):
                streams = (h, n, h + n if mix is None else mix)
                expected = np.maximum.reduce(
                    [
                        np.maximum(
                            np.abs(x),
                            np.abs(resample_poly(x, 4, 1)).reshape(-1, 4).max(axis=1),
                        )
                        for x in streams
                    ]
                )
                np.testing.assert_allclose(
                    peak_trace(h, n, mix), expected, rtol=2e-14, atol=2e-14
                )

    def test_requested_lufs_and_peak_for_both_targets(self):
        for target in (-15.0, -18.0):
            gen = paired(consonants=["t", "s"], target_lufs=target)
            h, n, m = gen.generate(32000)
            measured = pyln.Meter(16000).integrated_loudness(m)
            self.assertAlmostEqual(measured, target, delta=0.15)
            for audio in (h, n, m):
                peak = np.max(np.abs(resample_poly(audio.astype(float), 4, 1)))
                self.assertLessEqual(peak, 10 ** (-1 / 20))
            np.testing.assert_array_equal(m, h + n)

    def test_large_transients_keep_alignment_after_limiting(self):
        time = np.arange(48000) / 16000
        harmonic = 0.04 * np.sin(2 * np.pi * 200 * time)
        noise = np.random.default_rng(42).normal(0, 0.004, len(time))
        noise[12000:12016] += 8
        h, n, metadata = normalize_pair(harmonic, noise, 16000, -16.5, -1)
        self.assertTrue(metadata["limiter_applied"])
        self.assertAlmostEqual(
            pyln.Meter(16000).integrated_loudness(h + n), -16.5, delta=0.15
        )
        self.assertLessEqual(metadata["max_stem_or_mix_peak_db_4x"], -1)
        # A common gain preserves the instantaneous source proportion.
        np.testing.assert_allclose(
            h.astype(float) * noise, n.astype(float) * harmonic, rtol=2e-6, atol=1e-9
        )

    def test_short_and_silent_audio(self):
        for length, status in ((10, "short_clip_peak_only"), (16000, "silence")):
            h, n, metadata = normalize_pair(np.zeros(length), np.zeros(length), 16000)
            self.assertEqual(metadata["status"], status)
            self.assertIsNone(metadata["measured_lufs"])
            self.assertTrue(np.isfinite(h + n).all())
            json.dumps(metadata, allow_nan=False)


class DatasetAndExportTests(unittest.TestCase):
    def test_train_validation_lengths_modes_and_mixup(self):
        for mode in ("noise_as_inst", "harmonic_as_inst"):
            train = GeneratedHarmonicNoiseSplitTrainingSet(
                ToneGenerator(),
                SilentBackground(),
                16000,
                256,
                64,
                0,
                1,
                0.4,
                mode=mode,
                random_seed=42,
            )
            X, y = train[0]
            self.assertEqual(X.shape, (1, 63 * 256))
            self.assertEqual(y.dtype, np.float32)
            self.assertTrue(np.isfinite(X).all())
            self.assertAlmostEqual(
                pyln.Meter(16000).integrated_loudness(X[0]), -16.5, delta=0.15
            )
            val = GeneratedHarmonicNoiseSplitValidationSet(
                ToneGenerator(),
                SilentBackground(),
                16000,
                64,
                val_size=1,
                mode=mode,
                random_seed=42,
            )
            self.assertEqual(val[0][0].shape, X.shape)
            np.testing.assert_array_equal(val[0][0], val[0][0])

    def test_copy_augmentation_and_silence_are_safe(self):
        train = GeneratedHarmonicNoiseSplitTrainingSet(
            ToneGenerator(),
            SilentBackground(),
            16000,
            256,
            64,
            1,
            0,
            0.4,
            random_seed=42,
        )
        X, y = train[0]
        np.testing.assert_array_equal(X, y)
        X, y = train.do_aug(
            np.zeros((1, 10), np.float32), np.zeros((1, 10), np.float32)
        )
        self.assertTrue(np.isfinite(X).all())

    def test_wav_round_trip_resume_and_no_overwrite(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(tmp)
            paths = export_pairs(
                root / "batch", count=2, duration=0.6, sr=16000, seed=7, target_lufs=-18
            )
            resumed = export_pairs(
                root / "resume",
                count=1,
                duration=0.6,
                sr=16000,
                seed=7,
                start_index=1,
                target_lufs=-18,
            )
            for name in ("harmonic.wav", "noise.wav", "mixure.wav", "metadata.json"):
                self.assertEqual(
                    (paths[1] / name).read_bytes(), (resumed[0] / name).read_bytes()
                )
            waveforms = [
                sf.read(paths[0] / name, dtype="float32")[0]
                for name in ("harmonic.wav", "noise.wav", "mixure.wav")
            ]
            np.testing.assert_array_equal(waveforms[0] + waveforms[1], waveforms[2])
            self.assertEqual(len(waveforms[0]), 9600)
            self.assertEqual(sf.info(paths[0] / "mixure.wav").subtype, "FLOAT")
            before = (paths[0] / "mixure.wav").read_bytes()
            with self.assertRaises(FileExistsError):
                export_pairs(root / "batch", count=1, duration=0.6, sr=16000)
            self.assertEqual(before, (paths[0] / "mixure.wav").read_bytes())

    def test_cli_runs_with_torch_and_matplotlib_imports_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = """import sys, runpy
class BlockOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'matplotlib'):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptionalImports())
sys.argv = ['generate_wavs.py', '--output-dir', sys.argv[1], '--count', '1', '--duration', '.5', '--sr', '16000', '--lufs', '-18', '--mixture-name', 'mixture.wav']
runpy.run_path('generate_wavs.py', run_name='__main__')
"""
            result = subprocess.run(
                [sys.executable, "-c", code, tmp],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / "000000" / "mixture.wav").is_file())


if __name__ == "__main__":
    unittest.main()
