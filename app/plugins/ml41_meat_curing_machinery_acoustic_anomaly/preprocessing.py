"""Audio -> log-Mel spectrogram preprocessing.

wav_to_logmel is ported verbatim from
inbox/a41/codigo/src/data_processing/preprocess.py (same sr/n_mels/win_length/
hop_length/n_fft/target_frames, same librosa calls, same pad/crop policy) — changing
any of these parameters would produce spectrograms the delivered checkpoints were never
trained on.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import librosa
import numpy as np

from app.domain.services.exceptions import InvalidAudioError
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    AUDIO_SR,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    TARGET_FRAMES,
    WIN_LENGTH,
)


def wav_to_logmel(
    wav_path: str,
    sr: int = AUDIO_SR,
    n_mels: int = N_MELS,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
    n_fft: int = N_FFT,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """Load a WAV file and return its log-Mel spectrogram.

    Returns an array of shape (1, n_mels, target_frames), dtype float32 — axis 0 is the
    channel (always 1, grayscale-like image). Multi-channel input is downmixed to mono
    by librosa.load(..., mono=True), matching the original pipeline exactly.
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=sr // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # (n_mels, T)

    T = log_mel.shape[1]
    if T < target_frames:
        pad = target_frames - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="reflect")
    else:
        log_mel = log_mel[:, :target_frames]

    return log_mel[np.newaxis, :, :]  # (1, n_mels, target_frames)


def audio_base64_to_logmel(audio_base64: str) -> np.ndarray:
    """Decode a base64-encoded WAV and return its raw (unnormalized) log-Mel spectrogram.

    Uses a temp file because librosa's loaders (audioread/soundfile) are most reliably
    fed a real path across formats/platforms — mirrors how the original pipeline always
    reads WAV files from disk.
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise InvalidAudioError(f"Cannot decode audio from base64: {exc}") from exc

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        return wav_to_logmel(tmp_path)
    except Exception as exc:
        raise InvalidAudioError(f"Cannot decode audio as WAV: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def normalize_logmel(spec: np.ndarray, norm_mean: float, norm_std: float) -> np.ndarray:
    """Apply training-time normalization, exactly as predict()/calibrate() do in main.py."""
    return (spec - norm_mean) / (norm_std + 1e-6)
