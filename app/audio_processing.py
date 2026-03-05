"""Advanced audio processing utilities for high-quality transitions.

Provides stem-aware crossfading using Demucs, tempo matching via pyrubberband,
and utility functions for beat detection and crossfading.
"""

import logging
import os

import numpy as np
import torch
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# Configuration via environment variables
USE_STEM_AWARE_CROSSFADE = os.getenv("USE_STEM_AWARE_CROSSFADE", "true").lower() == "true"

# Frequency cutoffs for bass swap technique
BASS_CUTOFF_HZ = 200


def detect_tempo(audio: np.ndarray, sr: int) -> float:
    """Detect tempo (BPM) of audio using librosa.

    Args:
        audio: Audio samples as numpy array
        sr: Sample rate

    Returns:
        Detected tempo in BPM
    """
    try:
        import librosa

        # Ensure mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        # Handle both old and new librosa return types
        if hasattr(tempo, "__iter__"):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        return float(tempo)
    except ImportError:
        logger.warning("librosa not available, returning default tempo")
        return 120.0
    except Exception as e:
        logger.warning(f"Tempo detection failed: {e}, returning default")
        return 120.0


def match_tempo(
    audio: np.ndarray,
    sr: int,
    source_tempo: float,
    target_tempo: float,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Time-stretch audio to match target tempo using pyrubberband.

    Args:
        audio: Audio samples as numpy array (channels, samples) or (samples,)
        sr: Sample rate
        source_tempo: Source tempo in BPM
        target_tempo: Target tempo in BPM
        tolerance: BPM tolerance within which no stretching is applied

    Returns:
        Time-stretched audio
    """
    if abs(source_tempo - target_tempo) < tolerance:
        return audio

    try:
        import pyrubberband as pyrb

        rate = target_tempo / source_tempo

        # pyrubberband expects (samples, channels) or (samples,)
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            # Convert from (channels, samples) to (samples, channels)
            audio = audio.T
            transposed = True
        else:
            transposed = False

        stretched = pyrb.time_stretch(audio, sr, rate)

        if transposed:
            stretched = stretched.T

        logger.debug(f"Time-stretched audio from {source_tempo:.1f} to {target_tempo:.1f} BPM")
        return stretched

    except ImportError:
        logger.warning("pyrubberband not available, skipping tempo matching")
        return audio
    except Exception as e:
        logger.warning(f"Tempo matching failed: {e}, returning original")
        return audio


def equal_power_crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    overlap_samples: int,
) -> np.ndarray:
    """Apply equal-power crossfade between two audio segments.

    Uses sine/cosine curves for constant perceived loudness.

    Args:
        audio1: First audio segment (channels, samples) or (samples,)
        audio2: Second audio segment (channels, samples) or (samples,)
        overlap_samples: Number of samples to crossfade

    Returns:
        Crossfaded audio
    """
    # Handle 1D arrays
    if audio1.ndim == 1:
        audio1 = audio1.reshape(1, -1)
    if audio2.ndim == 1:
        audio2 = audio2.reshape(1, -1)

    # Clamp overlap to available samples
    overlap_samples = min(
        overlap_samples,
        audio1.shape[-1] // 2,
        audio2.shape[-1] // 2,
    )

    # Equal-power fade curves
    t = np.linspace(0, np.pi / 2, overlap_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    # Apply crossfade
    overlap_region = audio1[:, -overlap_samples:] * fade_out + audio2[:, :overlap_samples] * fade_in

    return np.concatenate(
        [audio1[:, :-overlap_samples], overlap_region, audio2[:, overlap_samples:]],
        axis=1,
    )


def _find_nearest_beat(beat_samples: np.ndarray, target: int, search_window: int) -> int:
    """Find nearest beat position to target within search window.

    Args:
        beat_samples: Array of beat positions in samples
        target: Target sample position
        search_window: Maximum distance to search for a beat

    Returns:
        Nearest beat position, or target if no beat found in window
    """
    if len(beat_samples) == 0:
        return target

    distances = np.abs(beat_samples - target)
    valid_mask = distances < search_window

    if valid_mask.any():
        valid_indices = np.where(valid_mask)[0]
        nearest_idx = valid_indices[np.argmin(distances[valid_mask])]
        return int(beat_samples[nearest_idx])

    return target


def bass_swap_transition(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    overlap_samples: int,
    bass_cutoff: int = BASS_CUTOFF_HZ,
) -> np.ndarray:
    """Perform bass swap transition - instant bass swap at beat with gradual high-frequency crossfade.

    This prevents muddy low frequencies during transitions by instantly swapping
    bass at a beat boundary while gradually crossfading higher frequencies.

    Args:
        audio1: First audio segment (channels, samples)
        audio2: Second audio segment (channels, samples)
        sr: Sample rate
        overlap_samples: Number of samples for the transition
        bass_cutoff: Frequency cutoff for bass in Hz

    Returns:
        Transitioned audio
    """
    # Handle 1D arrays
    if audio1.ndim == 1:
        audio1 = audio1.reshape(1, -1)
    if audio2.ndim == 1:
        audio2 = audio2.reshape(1, -1)

    # Design butterworth filters
    nyq = sr / 2
    normalized_cutoff = bass_cutoff / nyq
    # Clamp to valid range
    normalized_cutoff = min(0.99, max(0.01, normalized_cutoff))

    b_low, a_low = butter(4, normalized_cutoff, btype="low")
    b_high, a_high = butter(4, normalized_cutoff, btype="high")

    # Separate bass and rest for overlap regions
    overlap1 = audio1[:, -overlap_samples:]
    overlap2 = audio2[:, :overlap_samples]

    # Apply filters to each channel
    bass1 = np.zeros_like(overlap1)
    bass2 = np.zeros_like(overlap2)
    rest1 = np.zeros_like(overlap1)
    rest2 = np.zeros_like(overlap2)

    for ch in range(overlap1.shape[0]):
        bass1[ch] = filtfilt(b_low, a_low, overlap1[ch])
        bass2[ch] = filtfilt(b_low, a_low, overlap2[ch])
        rest1[ch] = filtfilt(b_high, a_high, overlap1[ch])
        rest2[ch] = filtfilt(b_high, a_high, overlap2[ch])

    # Gradual crossfade for high frequencies
    t = np.linspace(0, 1, overlap_samples)
    rest_blend = rest1 * (1 - t) + rest2 * t

    # Find beat-aligned swap point for bass
    # Try to detect beats in the overlap region of audio1
    target_swap = overlap_samples // 2  # Default to midpoint
    search_window = overlap_samples // 4  # Search within ±25% of midpoint

    try:
        # Detect beats in the overlap region
        overlap_mono = overlap1.mean(axis=0) if overlap1.ndim == 2 else overlap1
        beat_samples = detect_beat_positions(overlap_mono, sr)

        if len(beat_samples) > 0:
            # Find beat nearest to midpoint
            swap_point = _find_nearest_beat(beat_samples, target_swap, search_window)
            logger.debug(f"Bass swap at beat position {swap_point} (target was {target_swap})")
        else:
            swap_point = target_swap
            logger.debug(f"No beats detected, using midpoint {swap_point}")
    except Exception as e:
        logger.debug(f"Beat detection failed: {e}, using midpoint")
        swap_point = target_swap

    # Clamp swap point to valid range (not too close to edges)
    min_swap = int(overlap_samples * 0.2)
    max_swap = int(overlap_samples * 0.8)
    swap_point = max(min_swap, min(swap_point, max_swap))

    bass_blend = np.concatenate([bass1[:, :swap_point], bass2[:, swap_point:]], axis=1)

    # Combine
    blended = rest_blend + bass_blend

    return np.concatenate(
        [audio1[:, :-overlap_samples], blended, audio2[:, overlap_samples:]],
        axis=1,
    )


def separate_stems_for_crossfade(
    audio: torch.Tensor,
    sr: int,
) -> dict[str, np.ndarray]:
    """Separate audio into stems using Demucs for crossfading.

    The 'other' stem is calculated as residual (mixture - drums - bass - vocals)
    to ensure stems sum back exactly to the original audio. This fixes the
    issue where Demucs stems don't perfectly reconstruct the original.

    Args:
        audio: Audio tensor (channels, samples)
        sr: Sample rate

    Returns:
        Dictionary with 'drums', 'bass', 'other', 'vocals' stems as numpy arrays
        at the ORIGINAL sample rate (resampled back if needed)
    """
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from app.musicgen import get_device

    device = get_device()

    # Load model (htdemucs gives us all 4 stems)
    model = get_model("htdemucs")
    if device != "cpu":
        model = model.to(device)

    # Prepare audio
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Ensure float32 for consistency
    audio = audio.float()

    # Keep original for residual calculation (before any resampling)
    original_audio_for_residual = audio.clone()

    # Track if we need to resample back
    needs_resample = sr != model.samplerate

    # Resample to Demucs sample rate if needed
    if needs_resample:
        import torchaudio

        resampler_to_demucs = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=model.samplerate
        )
        audio = resampler_to_demucs(audio)
        # Also resample original for residual calculation
        original_audio_for_residual = resampler_to_demucs(original_audio_for_residual)

    # Add batch dimension and move to device
    audio = audio.unsqueeze(0)
    if device != "cpu":
        audio = audio.to(device)

    # Run separation
    with torch.no_grad():
        sources = apply_model(model, audio, device=device, progress=False)

    # sources shape: (batch, num_sources, channels, samples)
    # source_names = ['drums', 'bass', 'other', 'vocals']

    # Extract individual stems (still on device)
    drums = sources[0, 0]  # drums
    bass = sources[0, 1]   # bass
    # Skip sources[0, 2] (other) - we'll calculate as residual
    vocals = sources[0, 3]  # vocals

    # Calculate 'other' as residual to ensure perfect reconstruction
    # other = original - drums - bass - vocals
    original_on_device = original_audio_for_residual.to(device)
    other = original_on_device - drums - bass - vocals

    logger.debug("Calculated 'other' stem as residual for perfect reconstruction")

    # Resample stems back to original sample rate if needed
    if needs_resample:
        import torchaudio

        resampler_back = torchaudio.transforms.Resample(
            orig_freq=model.samplerate, new_freq=sr
        )

        result = {
            "drums": resampler_back(drums.cpu()).numpy(),
            "bass": resampler_back(bass.cpu()).numpy(),
            "other": resampler_back(other.cpu()).numpy(),
            "vocals": resampler_back(vocals.cpu()).numpy(),
        }

        logger.debug(f"Resampled stems from {model.samplerate}Hz back to {sr}Hz")
        return result

    # No resampling needed - just convert to numpy
    result = {
        "drums": drums.cpu().numpy(),
        "bass": bass.cpu().numpy(),
        "other": other.cpu().numpy(),
        "vocals": vocals.cpu().numpy(),
    }

    return result


def stem_aware_crossfade(
    audio1: torch.Tensor | np.ndarray,
    audio2: torch.Tensor | np.ndarray,
    sr: int,
    overlap_sec: float = 2.0,
) -> np.ndarray:
    """Crossfade two audio segments using stem-aware processing.

    Separates each segment into stems and applies appropriate crossfade
    technique for each:
    - Drums: beat-aligned crossfade
    - Bass: instant swap at midpoint (bass swap technique)
    - Other/Vocals: equal-power crossfade

    Args:
        audio1: First audio segment (channels, samples)
        audio2: Second audio segment (channels, samples)
        sr: Sample rate
        overlap_sec: Overlap duration in seconds

    Returns:
        Crossfaded audio as numpy array
    """
    logger.info("Performing stem-aware crossfade (this may take a moment)")

    # Convert to numpy if needed
    if isinstance(audio1, torch.Tensor):
        audio1 = audio1.cpu().numpy()
    if isinstance(audio2, torch.Tensor):
        audio2 = audio2.cpu().numpy()

    overlap_samples = int(overlap_sec * sr)

    # Separate both segments into stems
    logger.debug("Separating stems for audio 1")
    stems1 = separate_stems_for_crossfade(torch.from_numpy(audio1), sr)
    logger.debug("Separating stems for audio 2")
    stems2 = separate_stems_for_crossfade(torch.from_numpy(audio2), sr)

    # Different crossfade strategy per stem
    result_stems = {}

    # Drums: equal-power crossfade
    result_stems["drums"] = equal_power_crossfade(stems1["drums"], stems2["drums"], overlap_samples)

    # Bass: gradual crossfade (was instant swap, but that caused beat doubling)
    result_stems["bass"] = equal_power_crossfade(stems1["bass"], stems2["bass"], overlap_samples)

    # Other and vocals: equal-power crossfade
    result_stems["other"] = equal_power_crossfade(
        stems1["other"], stems2["other"], overlap_samples
    )
    result_stems["vocals"] = equal_power_crossfade(
        stems1["vocals"], stems2["vocals"], overlap_samples
    )

    # Ensure all stems have the same length (fix for silence bug)
    stem_lengths = [s.shape[-1] for s in result_stems.values()]
    if len(set(stem_lengths)) > 1:
        logger.warning(f"Stem length mismatch: {stem_lengths}, trimming to shortest")
        target_len = min(stem_lengths)
        for name in result_stems:
            result_stems[name] = result_stems[name][:, :target_len]

    # Sum all stems
    result = (
        result_stems["drums"]
        + result_stems["bass"]
        + result_stems["other"]
        + result_stems["vocals"]
    )

    # Normalize to prevent clipping (stems can sum to >1.0)
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        logger.debug(f"Normalizing audio (peak was {max_val:.2f})")
        result = result / max_val

    logger.info("Stem-aware crossfade complete")
    return result


def find_zero_crossing(audio: np.ndarray, target: int, window: int = 256) -> int:
    """Find nearest zero crossing to target position.

    Args:
        audio: Audio samples (channels, samples) or (samples,)
        target: Target sample position
        window: Search window size

    Returns:
        Position of nearest zero crossing
    """
    if audio.ndim == 2:
        audio = audio[0]  # Use first channel

    start = max(0, target - window)
    end = min(len(audio), target + window)

    segment = audio[start:end]
    zero_crossings = np.where(np.diff(np.signbit(segment)))[0]

    if len(zero_crossings) == 0:
        return target

    nearest = zero_crossings[np.argmin(np.abs(zero_crossings - (target - start)))]
    return start + int(nearest)


def detect_beat_positions(audio: np.ndarray, sr: int) -> np.ndarray:
    """Detect beat positions in audio.

    Args:
        audio: Audio samples
        sr: Sample rate

    Returns:
        Array of beat positions in samples
    """
    try:
        import librosa

        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        return librosa.frames_to_samples(beat_frames)
    except ImportError:
        logger.warning("librosa not available for beat detection")
        return np.array([])
    except Exception as e:
        logger.warning(f"Beat detection failed: {e}")
        return np.array([])
