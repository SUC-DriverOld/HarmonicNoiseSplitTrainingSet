"""Time serial WAV exports in a temporary directory, using one numerical thread."""

import argparse
import contextlib
import cProfile
import io
import os
from pathlib import Path
import pstats
from statistics import median
import tempfile
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile", action="store_true", help="profile a separate batch after timing"
    )
    args = parser.parse_args()
    if min(args.count, args.rounds) < 1:
        parser.error("--count and --rounds must be positive")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"
    start = perf_counter()
    from generate_wavs import export_pairs

    with tempfile.TemporaryDirectory() as tmp:

        def run(name, count):
            with contextlib.redirect_stdout(io.StringIO()):
                export_pairs(
                    Path(tmp) / name,
                    count=count,
                    duration=args.duration,
                    sr=args.sr,
                    seed=args.seed,
                    target_lufs=-16.5,
                )

        run("warmup", 1)
        print(f"Import + first pair: {perf_counter() - start:.3f} s", flush=True)
        times = []
        for i in range(args.rounds):
            start = perf_counter()
            run(str(i), args.count)
            times.append(perf_counter() - start)
            print(f"Round {i + 1}: {times[-1]:.3f} s", flush=True)
        seconds = median(times) / args.count
        print(
            f"Median: {seconds:.3f} s/pair | {args.duration / seconds:.2f} audio seconds/s",
            flush=True,
        )
        if args.profile:
            profiler = cProfile.Profile()
            profiler.runcall(run, "profile", args.count)
            pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
