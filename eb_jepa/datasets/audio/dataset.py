"""Audio dataset — Speech Commands v2 keyword spotting (audio/speech SSL).

35-keyword spotting on 1-second clips @ 16 kHz. Two input representations selected
by ``mode``:
  - "raw": 1-second waveform        -> [1, 16000]
  - "mel": log-mel spectrogram      -> [1, n_mels, T]  (treated as a 1-channel image)

Two delivery modes selected by ``mode_ssl``:
  - "ssl"        : returns TWO augmented views of the same clip (time-shift, gain,
                   additive noise at random SNR) for a two-view / VICReg objective.
                   (set ``n_views=1`` to get a single augmented view for predictive.)
  - "supervised" : returns (x, label) with a CLEAN, deterministic representation
                   for the frozen-feature probe.

Official splits ship with the dataset (validation_list.txt / testing_list.txt);
everything else is train. Labels = the 35 keyword folders (excl. _background_noise_).

Data loading + augmentation are PROVIDED (plumbing). The modelling choices on top
(encoder, SSL objective, probe) live in ``examples/audio/`` where the ``# TODO``s are.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

SR = 16000
LEN = 16000  # 1 second


def list_labels(root):
    return sorted(
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith("_")
    )


@dataclass
class AudioConfig:
    data_root: str = "/lustre/work/pdl17890/udl806719/datasets/speech_commands_v2"
    mode: str = "raw"  # raw | mel  (input representation)
    split: str = "train"  # train | valid | test
    mode_ssl: str = "ssl"  # ssl (augmented views) | supervised ((x, label))
    n_views: int = 2  # 2 -> two-view VICReg; 1 -> single view (predictive)
    n_classes: int = 35
    sample_rate: int = SR
    # mel front-end
    n_mels: int = 64
    n_fft: int = 400  # 25 ms @ 16 kHz
    hop: int = 160  # 10 ms -> ~101 frames / 1 s
    # waveform augmentation strengths
    shift_frac: float = 0.1
    noise_snr_db: tuple = (5.0, 30.0)
    gain_db: float = 6.0
    specaug: bool = True  # mel-only SpecAugment on the SSL views
    batch_size: int = 128
    num_workers: int = 8


def _read_split_set(root, fname):
    p = os.path.join(root, fname)
    with open(p) as f:
        return set(line.strip() for line in f if line.strip())


class SpeechCommands(torch.utils.data.Dataset):
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        root = cfg.data_root
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{root} not found — download Speech Commands v2 there "
                "(35 keyword folders + validation_list.txt / testing_list.txt)."
            )
        self.labels = list_labels(root)
        self.lab2idx = {l: i for i, l in enumerate(self.labels)}
        val_set = _read_split_set(root, "validation_list.txt")
        test_set = _read_split_set(root, "testing_list.txt")
        all_wavs = []
        for lab in self.labels:
            for p in glob.glob(os.path.join(root, lab, "*.wav")):
                rel = f"{lab}/{os.path.basename(p)}"
                all_wavs.append((p, rel, self.lab2idx[lab]))
        if cfg.split == "valid":
            self.items = [(p, y) for p, rel, y in all_wavs if rel in val_set]
        elif cfg.split == "test":
            self.items = [(p, y) for p, rel, y in all_wavs if rel in test_set]
        else:
            self.items = [
                (p, y)
                for p, rel, y in all_wavs
                if rel not in val_set and rel not in test_set
            ]
        self._mel = None  # lazily built per-worker (torchaudio transform)
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.items)

    # ----- audio helpers -----
    def _load(self, path):
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if len(wav) >= LEN:
            wav = wav[:LEN]
        else:
            wav = np.pad(wav, (0, LEN - len(wav)))
        return wav

    def _augment_wave(self, wav):
        c, rng = self.cfg, self._rng
        # random circular time shift
        s = int(rng.uniform(-c.shift_frac, c.shift_frac) * LEN)
        wav = np.roll(wav, s)
        # random gain
        wav = wav * (10.0 ** (rng.uniform(-c.gain_db, c.gain_db) / 20.0))
        # additive Gaussian noise at random SNR
        snr = rng.uniform(*c.noise_snr_db)
        sig_p = np.mean(wav**2) + 1e-8
        noise_p = sig_p / (10.0 ** (snr / 10.0))
        wav = wav + rng.normal(0, np.sqrt(noise_p), size=wav.shape).astype(np.float32)
        return wav.astype(np.float32)

    def _mel_transform(self):
        if self._mel is None:
            import torchaudio

            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop,
                n_mels=self.cfg.n_mels,
                power=2.0,
            )
        return self._mel

    def _to_repr(self, wav, do_specaug):
        x = torch.from_numpy(wav).float()
        if self.cfg.mode == "raw":
            return x.unsqueeze(0)  # [1, LEN]
        mel = self._mel_transform()(x)  # [n_mels, T]
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        if do_specaug and self.cfg.specaug:
            mel = self._spec_augment(mel)
        return mel.unsqueeze(0)  # [1, n_mels, T]

    def _spec_augment(self, mel, fmax=12, tmax=15):
        nm, nt = mel.shape
        f = int(self._rng.integers(0, fmax + 1))
        f0 = int(self._rng.integers(0, max(1, nm - f)))
        t = int(self._rng.integers(0, tmax + 1))
        t0 = int(self._rng.integers(0, max(1, nt - t)))
        mel = mel.clone()
        mel[f0 : f0 + f, :] = 0.0
        mel[:, t0 : t0 + t] = 0.0
        return mel

    def __getitem__(self, idx):
        # fresh per-item rng so DataLoader workers don't share augmentation state
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        path, y = self.items[idx]
        wav = self._load(path)
        if self.cfg.mode_ssl == "supervised":
            # eval/probe: clean, deterministic representation
            return self._to_repr(wav, do_specaug=False), y
        # SSL: augmented view(s) of the SAME clip
        v1 = self._to_repr(self._augment_wave(wav), do_specaug=True)
        if self.cfg.n_views == 1:
            return v1, y  # predictive JEPA: one view (masking is the SSL signal)
        v2 = self._to_repr(self._augment_wave(wav), do_specaug=True)
        return v1, v2, y  # two-view VICReg


def make_loader(cfg: AudioConfig, shuffle=None):
    ds = SpeechCommands(cfg)
    is_train = cfg.split == "train"
    if shuffle is None:
        shuffle = is_train
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=cfg.mode_ssl == "ssl",
        persistent_workers=cfg.num_workers > 0,
    )
