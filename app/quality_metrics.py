"""Audio quality metrics and generate-evaluate-regenerate loop.

Provides transition quality evaluation using spectral analysis and
optional FAD (Fréchet Audio Distance) with CLAP embeddings.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Configuration via environment variables
USE_QUALITY_LOOP = os.getenv("USE_QUALITY_LOOP", "true").lower() == "true"
MAX_REGEN_ATTEMPTS = int(os.getenv("MAX_REGEN_ATTEMPTS", "3"))


@dataclass
class TransitionMetrics:
    """Metrics for evaluating transition quality between segments."""

    mfcc_similarity: float  # Target: > 0.7
    spectral_flux: float  # Target: < 100
    harmonic_continuity: float  # Target: > 0.5
    energy_ratio: float  # Target: < 2.0

    def passes_thresholds(
        self,
        mfcc_threshold: float = 0.7,
        flux_threshold: float = 100.0,
        harmonic_threshold: float = 0.5,
        energy_threshold: float = 2.0,
    ) -> bool:
        """Check if metrics pass quality thresholds.

        Args:
            mfcc_threshold: Minimum MFCC similarity
            flux_threshold: Maximum spectral flux
            harmonic_threshold: Minimum harmonic continuity
            energy_threshold: Maximum energy ratio

        Returns:
            True if all thresholds are met
        """
        return (
            self.mfcc_similarity >= mfcc_threshold
            and self.spectral_flux <= flux_threshold
            and self.harmonic_continuity >= harmonic_threshold
            and self.energy_ratio <= energy_threshold
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "mfcc_similarity": self.mfcc_similarity,
            "spectral_flux": self.spectral_flux,
            "harmonic_continuity": self.harmonic_continuity,
            "energy_ratio": self.energy_ratio,
        }


class QualityEvaluator:
    """Evaluator for audio transition quality."""

    def __init__(self, use_fad: bool = False) -> None:
        """Initialize evaluator.

        Args:
            use_fad: Whether to use FAD (Fréchet Audio Distance) with CLAP.
                     Requires fadtk to be installed.
        """
        self._use_fad = use_fad
        self._fad_model = None

        if use_fad:
            self._init_fad()

    def _init_fad(self) -> None:
        """Initialize FAD model lazily."""
        try:
            from fadtk import FrechetAudioDistance
            from fadtk.model_loader import CLAPLaionModel

            logger.info("Initializing CLAP model for FAD evaluation")
            self._clap_model = CLAPLaionModel("clap-laion-music")
            self._fad = FrechetAudioDistance(self._clap_model)
            logger.info("FAD model initialized")
        except ImportError:
            logger.warning(
                "fadtk not available, FAD evaluation disabled. "
                "Install with: pip install fadtk"
            )
            self._use_fad = False
        except Exception as e:
            logger.warning(f"Failed to initialize FAD: {e}")
            self._use_fad = False

    def evaluate_transition(
        self,
        seg1: np.ndarray | torch.Tensor,
        seg2: np.ndarray | torch.Tensor,
        sr: int,
        boundary_samples: int = 8192,
    ) -> TransitionMetrics:
        """Evaluate transition quality between two audio segments.

        Computes metrics at the boundary between segments:
        - MFCC similarity: Timbral continuity
        - Spectral flux: Discontinuity indicator
        - Harmonic continuity: Pitch/chord consistency
        - Energy ratio: Loudness balance

        Args:
            seg1: First audio segment (channels, samples) or (samples,)
            seg2: Second audio segment (channels, samples) or (samples,)
            sr: Sample rate
            boundary_samples: Number of samples to analyze at boundary

        Returns:
            TransitionMetrics with computed values
        """
        try:
            import librosa
            from scipy.spatial.distance import cosine
        except ImportError:
            logger.warning("librosa/scipy not available, returning default metrics")
            return TransitionMetrics(
                mfcc_similarity=1.0,
                spectral_flux=0.0,
                harmonic_continuity=1.0,
                energy_ratio=1.0,
            )

        # Convert to numpy if needed
        if isinstance(seg1, torch.Tensor):
            seg1 = seg1.cpu().numpy()
        if isinstance(seg2, torch.Tensor):
            seg2 = seg2.cpu().numpy()

        # Ensure 1D (mono) for analysis
        if seg1.ndim == 2:
            seg1 = seg1.mean(axis=0)
        if seg2.ndim == 2:
            seg2 = seg2.mean(axis=0)

        # Extract boundary regions
        end_region = seg1[-boundary_samples:]
        start_region = seg2[:boundary_samples]

        # 1. MFCC Similarity
        try:
            mfcc1 = np.mean(librosa.feature.mfcc(y=end_region, sr=sr, n_mfcc=20), axis=1)
            mfcc2 = np.mean(librosa.feature.mfcc(y=start_region, sr=sr, n_mfcc=20), axis=1)
            mfcc_similarity = float(1 - cosine(mfcc1, mfcc2))
        except Exception:
            mfcc_similarity = 0.5

        # 2. Spectral Flux (discontinuity)
        try:
            spec1 = np.abs(librosa.stft(end_region))
            spec2 = np.abs(librosa.stft(start_region))
            spectral_flux = float(np.sum(np.abs(spec2[:, 0] - spec1[:, -1])))
        except Exception:
            spectral_flux = 50.0

        # 3. Harmonic Continuity (chroma correlation)
        try:
            chroma1 = np.mean(librosa.feature.chroma_cqt(y=end_region, sr=sr)[:, -10:], axis=1)
            chroma2 = np.mean(librosa.feature.chroma_cqt(y=start_region, sr=sr)[:, :10], axis=1)
            harmonic_continuity = float(np.corrcoef(chroma1, chroma2)[0, 1])
            if np.isnan(harmonic_continuity):
                harmonic_continuity = 0.5
        except Exception:
            harmonic_continuity = 0.5

        # 4. Energy Ratio
        try:
            rms1 = np.sqrt(np.mean(end_region**2))
            rms2 = np.sqrt(np.mean(start_region**2))
            energy_ratio = float(max(rms1, rms2) / (min(rms1, rms2) + 1e-8))
        except Exception:
            energy_ratio = 1.0

        return TransitionMetrics(
            mfcc_similarity=mfcc_similarity,
            spectral_flux=spectral_flux,
            harmonic_continuity=harmonic_continuity,
            energy_ratio=energy_ratio,
        )

    def evaluate_overall_quality(
        self,
        audio: np.ndarray | torch.Tensor,
        sr: int,
    ) -> dict[str, float]:
        """Evaluate overall audio quality.

        Args:
            audio: Audio to evaluate
            sr: Sample rate

        Returns:
            Dictionary with quality metrics
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        metrics = {}

        # RMS energy statistics
        window_samples = sr  # 1 second windows
        num_windows = len(audio) // window_samples
        rms_values = []

        for i in range(num_windows):
            start = i * window_samples
            end = start + window_samples
            rms = np.sqrt(np.mean(audio[start:end] ** 2))
            rms_values.append(rms)

        if rms_values:
            metrics["avg_rms"] = float(np.mean(rms_values))
            metrics["min_rms"] = float(np.min(rms_values))
            metrics["max_rms"] = float(np.max(rms_values))
            metrics["rms_std"] = float(np.std(rms_values))

        # Peak detection
        metrics["peak"] = float(np.max(np.abs(audio)))
        metrics["clipping"] = metrics["peak"] > 0.99

        # Silence detection
        silence_threshold = 0.005
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        metrics["silence_ratio"] = float(silent_samples / len(audio))

        return metrics


class QualityOptimizedGenerator:
    """Generator wrapper that retries generation if quality is poor."""

    def __init__(
        self,
        musicgen_wrapper,
        max_attempts: int = MAX_REGEN_ATTEMPTS,
        enabled: bool = USE_QUALITY_LOOP,
    ) -> None:
        """Initialize quality-optimized generator.

        Args:
            musicgen_wrapper: MusicGenWrapper instance
            max_attempts: Maximum generation attempts
            enabled: Whether quality loop is enabled
        """
        self._musicgen = musicgen_wrapper
        self._evaluator = QualityEvaluator()
        self._max_attempts = max_attempts
        self._enabled = enabled

        # Quality thresholds - very relaxed to pass most generations on first try
        # The stem-aware crossfade handles quality; retries add time without benefit
        self._thresholds = {
            "mfcc_threshold": 0.5,      # Timbral similarity (very lenient)
            "flux_threshold": 5000.0,   # Spectral change (allow big changes)
            "harmonic_threshold": -0.5, # Pitch continuity (allow negative correlation)
            "energy_threshold": 10.0,   # Loudness ratio (allow 10x difference)
        }

    async def generate_segment_with_quality(
        self,
        prompt: str,
        duration: int,
        previous_segment: torch.Tensor | None = None,
        temperature: float = 0.85,
        cfg_coef: float = 3.0,
    ) -> tuple[torch.Tensor, TransitionMetrics | None]:
        """Generate a segment with optional quality-based retry.

        Args:
            prompt: Text prompt for generation
            duration: Duration in seconds
            previous_segment: Previous segment for continuation and quality check
            temperature: Sampling temperature
            cfg_coef: Classifier-free guidance coefficient

        Returns:
            Tuple of (generated segment, metrics or None if first segment)
        """
        if not self._enabled or previous_segment is None:
            # Quality loop disabled or first segment - just generate
            segment = await self._generate_segment(
                prompt, duration, previous_segment, temperature, cfg_coef
            )
            return segment, None

        best_segment = None
        best_metrics = None
        best_score = -float("inf")

        for attempt in range(self._max_attempts):
            segment = await self._generate_segment(
                prompt, duration, previous_segment, temperature, cfg_coef
            )

            metrics = self._evaluator.evaluate_transition(
                previous_segment.cpu().numpy(),
                segment.cpu().numpy(),
                self._musicgen.sample_rate,
            )

            # Compute composite score
            score = self._compute_quality_score(metrics)

            if score > best_score:
                best_segment = segment
                best_metrics = metrics
                best_score = score

            if metrics.passes_thresholds(**self._thresholds):
                logger.debug(f"Quality check passed on attempt {attempt + 1}")
                return segment, metrics

            logger.info(
                f"Quality check failed (attempt {attempt + 1}/{self._max_attempts}): "
                f"MFCC={metrics.mfcc_similarity:.2f}, "
                f"flux={metrics.spectral_flux:.1f}, "
                f"harmonic={metrics.harmonic_continuity:.2f}, "
                f"energy={metrics.energy_ratio:.2f}"
            )

        logger.warning(
            f"Quality threshold not met after {self._max_attempts} attempts, "
            f"using best attempt (score={best_score:.2f})"
        )
        return best_segment, best_metrics

    async def _generate_segment(
        self,
        prompt: str,
        duration: int,
        previous_segment: torch.Tensor | None,
        temperature: float,
        cfg_coef: float,
    ) -> torch.Tensor:
        """Generate a single segment."""
        if previous_segment is not None:
            # Use last 5 seconds as conditioning
            context_samples = int(5 * self._musicgen.sample_rate)
            conditioning = previous_segment[:, -context_samples:]
            # Use much lower cfg_coef for continuations to maintain style consistency
            # The audio conditioning should guide the style more than the text prompt
            # 1.2 is very low - the model will mostly follow the audio, with subtle text influence
            continuation_cfg = min(cfg_coef, 1.2)
            return await self._musicgen.generate_continuation_async(
                prompt, duration, conditioning, temperature, continuation_cfg
            )
        else:
            return await self._musicgen.generate_segment_async(
                prompt, duration, temperature, cfg_coef
            )

    def _compute_quality_score(self, metrics: TransitionMetrics) -> float:
        """Compute composite quality score from metrics.

        Args:
            metrics: Transition metrics

        Returns:
            Composite score (higher is better)
        """
        # Normalize each metric to 0-1 range and compute weighted average
        mfcc_score = min(1.0, metrics.mfcc_similarity)
        flux_score = max(0, 1 - metrics.spectral_flux / 200)
        harmonic_score = (metrics.harmonic_continuity + 1) / 2  # -1 to 1 -> 0 to 1
        energy_score = max(0, 1 - (metrics.energy_ratio - 1) / 3)  # 1->1, 4->0

        # Weighted average
        return 0.3 * mfcc_score + 0.2 * flux_score + 0.3 * harmonic_score + 0.2 * energy_score
