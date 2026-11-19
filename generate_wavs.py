"""Export sample-aligned harmonic.wav / noise.wav / mixure.wav without torch."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from consonants import CONSONANTS
from paired_generator import PairedGenerator


def export_pairs(
    output_dir,
    count=10,
    duration=5.0,
    sr=44100,
    hopsize=512,
    seed=42,
    start_index=0,
    snr_db=(-5.0, 20.0),
    consonants=None,
    consonant_augmentation=True,
    mixture_name="mixure.wav",
    overwrite=False,
    target_lufs=-16.5,
    peak_db=-1.0,
):
    """Write FLOAT WAVs, returning their directories. Each index has its own seed."""
    if (
        not isinstance(count, int)
        or count < 1
        or not isinstance(start_index, int)
        or start_index < 0
    ):
        raise ValueError(
            "count must be positive and start_index must be nonnegative integers"
        )
    if not isinstance(sr, int) or sr < 8000:
        raise ValueError("sr must be an integer >= 8000")
    if not np.isfinite(duration) or duration <= 0 or int(round(duration * sr)) < 1:
        raise ValueError("duration must be positive and contain at least one sample")
    if target_lufs is not None and int(round(duration * sr)) < int(np.ceil(0.4 * sr)):
        raise ValueError("LUFS-controlled WAV export requires duration >= 0.4 seconds")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if mixture_name not in ("mixure.wav", "mixture.wav"):
        raise ValueError("mixture_name must be mixure.wav or mixture.wav")
    # Validate synthesis options before creating any output files.
    PairedGenerator(
        sr=sr,
        hopsize=hopsize,
        random_seed=seed,
        snr_db=snr_db,
        consonants=consonants,
        consonant_augmentation=consonant_augmentation,
        target_lufs=target_lufs,
        peak_db=peak_db,
    )
    output_dir = Path(output_dir).expanduser().resolve()
    directories = [
        output_dir / f"{index:06d}" for index in range(start_index, start_index + count)
    ]
    for directory in directories:
        if directory.is_symlink():
            raise ValueError(f"refusing to export through a symlink: {directory}")
        if directory.exists() and not overwrite:
            raise FileExistsError(
                f"{directory} already exists; use --start-index or --overwrite"
            )
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(str(directory))
    n_samples = int(round(duration * sr))
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, directory in zip(range(start_index, start_index + count), directories):
        sample_seed = int(
            np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0]
        )
        generator = PairedGenerator(
            sr=sr,
            hopsize=hopsize,
            random_seed=sample_seed,
            snr_db=snr_db,
            consonants=consonants,
            consonant_augmentation=consonant_augmentation,
            target_lufs=target_lufs,
            peak_db=peak_db,
        )
        harmonic, noise, mixture = generator.generate(n_samples)
        directory.mkdir(exist_ok=overwrite)
        # FLOAT avoids separate PCM quantization/clipping corrupting the paired labels.
        for name, audio in (
            ("harmonic.wav", harmonic),
            ("noise.wav", noise),
            (mixture_name, mixture),
        ):
            destination = directory / name
            if destination.is_symlink():
                raise ValueError(f"refusing to overwrite a symlink: {destination}")
            sf.write(destination, audio, sr, subtype="FLOAT", format="WAV")
        metadata = dict(generator.last_metadata)
        metadata.update(
            {
                "format_version": 1,
                "index": index,
                "base_seed": seed,
                "sample_seed": sample_seed,
                "hopsize": hopsize,
                "consonants": list(generator.consonants.consonants),
                "files": {
                    "harmonic": "harmonic.wav",
                    "noise": "noise.wav",
                    "mixture": mixture_name,
                },
                "wav_subtype": "FLOAT",
                "event_units": "samples; end is exclusive; event levels precede noise_gain, common_gain and linked loudness gains",
            }
        )
        destination = directory / "metadata.json"
        if destination.is_symlink():
            raise ValueError(f"refusing to overwrite a symlink: {destination}")
        destination.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        measured = metadata["loudness"]["measured_lufs"]
        level = f"{measured:.2f} LUFS" if measured is not None else "LUFS unavailable"
        status = metadata["loudness"]["status"]
        print(f"[{index - start_index + 1}/{count}] {directory} | {level} | {status}")
    return directories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_audio"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per pair")
    parser.add_argument("--sr", type=int, default=44100, help="sample rate, >= 8000")
    parser.add_argument("--hopsize", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-index", type=int, default=0, help="first output directory index"
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs=2,
        default=(-5.0, 20.0),
        metavar=("MIN", "MAX"),
        help="RMS harmonic/noise ratio, including consonants",
    )
    parser.add_argument(
        "--consonants",
        nargs="+",
        choices=CONSONANTS,
        help="consonant inventory to sample",
    )
    parser.add_argument(
        "--target-lufs",
        "--lufs",
        type=float,
        default=-18.0,
        help="target integrated mixture loudness, default: -18.0 LUFS",
    )
    parser.add_argument(
        "--peak-db",
        type=float,
        default=-1.0,
        help="4x oversampled peak ceiling for all WAVs, default: -1 dB",
    )
    parser.add_argument(
        "--no-consonants",
        action="store_true",
        help="disable consonant augmentation for comparison",
    )
    parser.add_argument(
        "--mixture-name", choices=("mixure.wav", "mixture.wav"), default="mixure.wav"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing files in selected sample directories",
    )
    args = parser.parse_args(argv)
    try:
        export_pairs(
            output_dir=args.output_dir,
            count=args.count,
            duration=args.duration,
            sr=args.sr,
            hopsize=args.hopsize,
            seed=args.seed,
            start_index=args.start_index,
            snr_db=args.snr_db,
            consonants=args.consonants,
            consonant_augmentation=not args.no_consonants,
            mixture_name=args.mixture_name,
            overwrite=args.overwrite,
            target_lufs=args.target_lufs,
            peak_db=args.peak_db,
        )
    except (ValueError, FileExistsError, NotADirectoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
