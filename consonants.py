"""Sample-aligned consonant/voicing plans and aperiodic excitation (no torch).

These are randomized acoustic approximations, not a text-to-speech engine.
Periodic voicing is applied by paired_generator; every excitation here is noise.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq


CONSONANTS = (
    "s",
    "sh",
    "f",
    "h",
    "z",
    "zh",
    "v",
    "p",
    "t",
    "k",
    "b",
    "d",
    "g",
    "ts",
    "ch",
)
VOICED = frozenset(("z", "zh", "v", "b", "d", "g"))
STOPS = frozenset(("p", "t", "k", "b", "d", "g"))
AFFRICATES = frozenset(("ts", "ch"))
PROFILES = {
    "z": "s",
    "zh": "sh",
    "v": "f",
    "b": "p",
    "d": "t",
    "g": "k",
    "ts": "s",
    "ch": "sh",
}


def smooth_envelope(length, attack, release):
    """Raised-cosine edges, including clips shorter than the requested ramps."""
    envelope = np.ones(length, dtype=np.float64)
    attack = min(length, max(1, attack))
    release = min(length, max(1, release))
    envelope[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack)) ** 2
    envelope[-release:] *= np.sin(np.linspace(np.pi / 2, 0, release)) ** 2
    return envelope


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


@dataclass(frozen=True)
class SpeechEvent:
    start: int
    end: int  # exclusive, in samples
    kind: str
    phone: str
    profile: str
    level: float  # excitation RMS before envelope/global gain
    attack: int
    release: int
    voiced: bool = False
    spectral_scale: float = 1.0

    def to_dict(self):
        return asdict(self)


class ConsonantSynthesizer:
    def __init__(self, sr=44100, random_seed=None, consonants=None):
        if not isinstance(sr, (int, np.integer)) or sr < 8000:
            raise ValueError("sr must be an integer >= 8000")
        self.sr = sr
        self.rng = np.random.default_rng(random_seed)
        self.consonants = tuple(CONSONANTS if consonants is None else consonants)
        if not self.consonants or any(
            phone not in CONSONANTS for phone in self.consonants
        ):
            raise ValueError(f"consonants must be a non-empty subset of {CONSONANTS}")

    def make_plan(self, n_samples):
        """Return shared voicing envelope and closure/burst/frication/breath events."""
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        voicing = np.zeros(n_samples, dtype=np.float64)
        events = []
        position = 0
        inventory = []

        def add(
            start,
            duration,
            kind,
            phone,
            profile,
            level,
            voice=0.0,
            attack_ms=5,
            release_ms=10,
            spectral_scale=1.0,
        ):
            length = max(1, int(round(duration * self.sr)))
            end = min(n_samples, start + length)
            if start < n_samples:
                event = SpeechEvent(
                    start,
                    end,
                    kind,
                    phone,
                    profile,
                    float(level),
                    max(1, int(attack_ms * self.sr / 1000)),
                    max(1, int(release_ms * self.sr / 1000)),
                    bool(voice),
                    float(spectral_scale),
                )
                events.append(event)
                if voice:
                    envelope = smooth_envelope(end - start, event.attack, event.release)
                    voicing[start:end] = np.maximum(
                        voicing[start:end], voice * envelope
                    )
            return start + length

        while position < n_samples:
            if not inventory:
                inventory = list(self.rng.permutation(self.consonants))
            phone = str(inventory.pop())
            profile = PROFILES.get(phone, phone)
            is_voiced = phone in VOICED
            scale = self.rng.uniform(0.8, 1.2)
            voice = self.rng.uniform(0.25, 0.65) if is_voiced else 0.0

            if phone in STOPS or phone in AFFRICATES:
                position = add(
                    position,
                    self.rng.uniform(0.02, 0.08),
                    "closure",
                    phone,
                    profile,
                    0,
                    voice * 0.25,
                )
                position = add(
                    position,
                    self.rng.uniform(0.003, 0.012),
                    "burst",
                    phone,
                    profile,
                    self.rng.uniform(0.15, 0.4),
                    attack_ms=0.2,
                    release_ms=2,
                    spectral_scale=scale,
                )
                if phone in STOPS:
                    vot = (
                        self.rng.uniform(0.003, 0.025)
                        if is_voiced
                        else self.rng.uniform(0.02, 0.12)
                    )
                    position = add(
                        position,
                        vot,
                        "aspiration",
                        phone,
                        "h",
                        self.rng.uniform(0.04, 0.13),
                        voice * 0.4,
                        attack_ms=1,
                        release_ms=5,
                        spectral_scale=scale,
                    )

            if phone not in STOPS:
                position = add(
                    position,
                    self.rng.uniform(0.06, 0.24),
                    "aspiration" if phone == "h" else "frication",
                    phone,
                    profile,
                    self.rng.uniform(0.06, 0.18),
                    voice,
                    attack_ms=self.rng.uniform(2, 12),
                    release_ms=self.rng.uniform(5, 18),
                    spectral_scale=scale,
                )

            # Breath noise overlaps the voiced vowel. Stops also have an aspiration tail.
            vowel_start = position
            position = add(
                position,
                self.rng.uniform(0.12, 0.55),
                "breathiness",
                "vowel",
                "h",
                self.rng.uniform(0.006, 0.035),
                self.rng.uniform(0.65, 1.0),
                attack_ms=self.rng.uniform(5, 20),
                release_ms=20,
                spectral_scale=scale,
            )
            if phone in STOPS or phone in AFFRICATES:
                add(
                    vowel_start,
                    self.rng.uniform(0.01, 0.035),
                    "aspiration",
                    phone,
                    "h",
                    self.rng.uniform(0.015, 0.06),
                    attack_ms=1,
                    release_ms=10,
                    spectral_scale=scale,
                )
            if self.rng.random() < 0.35:
                position = add(
                    position,
                    self.rng.uniform(0.025, 0.15),
                    "silence",
                    "silence",
                    "h",
                    0,
                )
        return voicing, events

    def _colored_noise(self, length, profile, spectral_scale):
        """Smooth stochastic spectra, with no common high-frequency roll-off."""
        pad = max(16, int(0.01 * self.sr))
        fft_size = next_fast_len(length + 2 * pad)
        frequencies = rfftfreq(fft_size, 1 / self.sr)
        # Preserve a useful band at low sample rates without exceeding Nyquist.
        scale = min(spectral_scale, self.sr / 22050)
        f = frequencies / scale

        def highpass(cutoff):
            return (f / np.sqrt(f * f + cutoff * cutoff)) ** 4

        def gaussian(center, width):
            return np.exp(-0.5 * ((f - center) / width) ** 2)

        if profile == "s":
            shape = highpass(3200) * (0.15 + gaussian(7000, 2000))
        elif profile == "sh":
            shape = highpass(1300) * (0.08 + gaussian(3300, 1200))
        elif profile == "f":
            shape = highpass(650) / np.sqrt(1 + (f / 9500) ** 4)
        elif profile == "h":
            shape = highpass(100) * (
                0.03
                + gaussian(650, 220)
                + 0.75 * gaussian(1500, 350)
                + 0.4 * gaussian(2700, 500)
            )
        elif profile == "p":
            shape = highpass(120) / np.sqrt(1 + (f / 1600) ** 4)
        elif profile == "t":
            shape = highpass(2500) * (0.25 + gaussian(6500, 2400))
        elif profile == "k":
            shape = highpass(700) * (0.12 + gaussian(2300, 800))
        else:
            raise ValueError(f"unknown noise profile: {profile}")
        shape[0] = 0
        white = self.rng.standard_normal(fft_size)
        colored = irfft(rfft(white) * shape, n=fft_size)[pad : pad + length]
        return colored / max(rms(colored), 1e-12)

    def synthesize(self, n_samples, events):
        """Render only aperiodic components; apply envelopes AFTER spectral filtering."""
        result = np.zeros(n_samples, dtype=np.float64)
        for event in events:
            if not 0 <= event.start < event.end <= n_samples:
                raise ValueError("event bounds must lie within the waveform")
            if event.kind in ("closure", "silence") or event.level == 0:
                continue
            length = event.end - event.start
            excitation = self._colored_noise(
                length, event.profile, event.spectral_scale
            )
            envelope = smooth_envelope(length, event.attack, event.release)
            if event.kind == "burst":
                envelope *= np.exp(-np.linspace(0, self.rng.uniform(2, 5), length))
            else:
                # Irregular turbulence energy within each consonant, rather than stationary hiss.
                count = max(2, int(length / self.sr * 90))
                modulation = np.interp(
                    np.arange(length),
                    np.linspace(0, length - 1, count),
                    self.rng.uniform(0.65, 1.25, count),
                )
                envelope *= modulation
            result[event.start : event.end] += event.level * excitation * envelope
        return result
