"""MusicGen wrapper with lazy loading and sliding window support."""

import logging
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Model configuration - configurable via environment variable
# Options: facebook/musicgen-small (300M), facebook/musicgen-medium (1.5B), facebook/musicgen-large (3.3B)
# Note: Mono models produce better music quality than stereo variants
# Default to musicgen-small for fast generation with good quality
MUSICGEN_MODEL = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
MAX_SEGMENT_DURATION = 30  # MusicGen max duration per segment
OVERLAP_DURATION = 8  # Overlap for smooth transitions (increased from 5 for better context)
STYLE_CONDITION_DURATION = 4.0  # Duration of style conditioning audio (1.5-4.5s recommended)


def get_device() -> str:
    """Get the best available device for inference.

    Note: MPS is available on Apple Silicon but torch.autocast doesn't
    support MPS in PyTorch 2.1.0 which audiocraft requires. So we use
    CPU on macOS for now. This can be changed when audiocraft supports
    newer PyTorch versions with MPS autocast support.

    Returns:
        Device string: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    # MPS is available but audiocraft's autocast doesn't support it in torch 2.1.0
    # Uncomment below when using PyTorch >= 2.4 with MPS autocast support
    # elif torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def clear_device_cache() -> None:
    """Clear GPU/MPS memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class MusicGenWrapper:
    """Wrapper for MusicGen with lazy loading and VRAM management."""

    def __init__(self, output_dir: str = "/data/output") -> None:
        """Initialize wrapper.

        Args:
            output_dir: Directory for output files
        """
        self._model = None
        self._device = get_device()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"MusicGen will use device: {self._device}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        """Get model sample rate (32kHz for MusicGen).

        Returns:
            Sample rate in Hz
        """
        self.load_model()
        assert self._model is not None
        return self._model.sample_rate

    @property
    def model(self):
        """Get loaded model instance for direct access.

        Returns:
            MusicGen model instance
        """
        self.load_model()
        return self._model

    def load_model(self) -> None:
        """Load MusicGen model lazily."""
        if self._model is not None:
            return

        logger.info(f"Loading MusicGen model: {MUSICGEN_MODEL} on {self._device}")
        from audiocraft.models import MusicGen

        self._model = MusicGen.get_pretrained(MUSICGEN_MODEL, device=self._device)
        assert self._model is not None
        self._model.set_generation_params(use_sampling=True, top_k=250)
        logger.info(f"MusicGen model {MUSICGEN_MODEL} loaded successfully")

    def unload_model(self) -> None:
        """Unload model and free VRAM/MPS memory."""
        if self._model is not None:
            del self._model
            self._model = None
            clear_device_cache()
            logger.info("MusicGen model unloaded, memory freed")

    async def generate(
        self,
        prompt: str,
        duration: int,
        mood: str | None = None,
        job_id: str = "",
        progress_callback: Callable | None = None,
    ) -> str:
        """Generate music from text prompt.

        Args:
            prompt: Text description of desired music
            duration: Duration in seconds (10-120)
            mood: Optional mood modifier
            job_id: Job ID for output filename
            progress_callback: Optional callback for progress updates

        Returns:
            Path to generated audio file
        """
        self.load_model()
        assert self._model is not None

        # Combine prompt with mood if provided
        full_prompt = prompt
        if mood:
            full_prompt = f"{mood} {prompt}"

        output_path = self._output_dir / f"{job_id}_generated.wav"

        try:
            if duration <= MAX_SEGMENT_DURATION:
                # Short generation - single pass
                audio = await self._generate_segment(full_prompt, duration)
                if progress_callback:
                    await progress_callback(100.0)
            else:
                # Long generation - sliding window
                audio = await self._generate_long(
                    full_prompt, duration, progress_callback
                )

            # Save audio
            torchaudio.save(
                str(output_path),
                audio.cpu(),
                sample_rate=self._model.sample_rate,
            )
            logger.info(f"Generated audio saved to {output_path}")
            return str(output_path)

        finally:
            # Clean up GPU/MPS memory after generation
            clear_device_cache()

    async def _generate_segment(self, prompt: str, duration: int) -> torch.Tensor:
        """Generate a single audio segment.

        Args:
            prompt: Text prompt
            duration: Duration in seconds

        Returns:
            Audio tensor
        """
        import asyncio

        assert self._model is not None
        self._model.set_generation_params(duration=duration)

        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        model = self._model  # Capture for lambda
        audio = await loop.run_in_executor(
            None,
            lambda: model.generate([prompt]),
        )
        return audio[0]

    async def _generate_long(
        self,
        prompt: str,
        total_duration: int,
        progress_callback: Callable | None = None,
    ) -> torch.Tensor:
        """Generate long audio using sliding window technique.

        Args:
            prompt: Text prompt
            total_duration: Total duration in seconds
            progress_callback: Optional callback for progress updates

        Returns:
            Concatenated audio tensor
        """
        import asyncio

        assert self._model is not None
        model = self._model  # Capture for lambdas
        segments: list[torch.Tensor] = []
        current_pos = 0
        segment_num = 0

        # Calculate number of segments needed
        effective_segment = MAX_SEGMENT_DURATION - OVERLAP_DURATION
        num_segments = (total_duration + effective_segment - 1) // effective_segment

        while current_pos < total_duration:
            segment_duration = min(MAX_SEGMENT_DURATION, total_duration - current_pos)

            logger.info(
                f"Generating segment {segment_num + 1}/{num_segments} "
                f"at position {current_pos}s"
            )

            # Generate segment
            model.set_generation_params(duration=segment_duration)

            loop = asyncio.get_event_loop()
            if segment_num == 0:
                # First segment - no conditioning
                audio = await loop.run_in_executor(
                    None,
                    lambda: model.generate([prompt]),
                )
            else:
                # Subsequent segments - use end of previous as conditioning
                prev_audio = segments[-1]
                # Take last few seconds for conditioning
                condition_samples = int(OVERLAP_DURATION * model.sample_rate)
                condition_audio = prev_audio[:, -condition_samples:]

                audio = await loop.run_in_executor(
                    None,
                    lambda cond=condition_audio, sr=model.sample_rate: model.generate_continuation(
                        cond.unsqueeze(0),
                        sr,
                        [prompt],
                        progress=False,
                    ),
                )

            segments.append(audio[0])
            segment_num += 1

            # Move position (account for overlap)
            if segment_num == 1:
                current_pos += MAX_SEGMENT_DURATION
            else:
                current_pos += MAX_SEGMENT_DURATION - OVERLAP_DURATION

            # Report progress
            if progress_callback:
                progress = min(100.0, (current_pos / total_duration) * 100)
                await progress_callback(progress)

        # Crossfade and concatenate segments
        return self._crossfade_segments(segments)

    def _find_zero_crossing(
        self, audio: torch.Tensor, target: int, window: int = 256
    ) -> int:
        """Find nearest zero crossing to target position to prevent clicks.

        Args:
            audio: Audio tensor [channels, samples]
            target: Target sample position
            window: Search window size in samples

        Returns:
            Sample position of nearest zero crossing
        """
        start = max(0, target - window)
        end = min(audio.shape[-1], target + window)

        # Use first channel for zero-crossing detection
        segment = audio[0, start:end].cpu().numpy()
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]

        if len(zero_crossings) == 0:
            return target

        # Find nearest zero crossing to target
        nearest = zero_crossings[np.argmin(np.abs(zero_crossings - (target - start)))]
        return start + int(nearest)

    def _detect_beat_boundary(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        target_position: int,
        search_window_samples: int = 8000,
    ) -> int:
        """Find nearest beat to target position using librosa.

        Args:
            audio: Audio tensor [channels, samples]
            sample_rate: Sample rate in Hz
            target_position: Target sample position
            search_window_samples: Search window size in samples (~250ms at 32kHz)

        Returns:
            Sample position of nearest beat (or target_position if no beat found)
        """
        try:
            import librosa
        except ImportError:
            logger.debug("librosa not available, using target position")
            return target_position

        # Convert to numpy mono for librosa
        audio_np = audio.cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=0)

        try:
            # Detect beats
            _tempo, beat_frames = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
            beat_samples = librosa.frames_to_samples(beat_frames)

            if len(beat_samples) == 0:
                logger.debug("No beats detected, using target position")
                return target_position

            # Find nearest beat within search window
            distances = np.abs(beat_samples - target_position)
            valid_mask = distances < search_window_samples

            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                nearest_idx = valid_indices[np.argmin(distances[valid_mask])]
                beat_pos = int(beat_samples[nearest_idx])
                logger.debug(f"Found beat at {beat_pos} (target was {target_position})")
                return beat_pos

            logger.debug("No beats in search window, using target position")
            return target_position

        except Exception as e:
            logger.debug(f"Beat detection failed: {e}, using target position")
            return target_position

    def _crossfade_segments(
        self,
        segments: list[torch.Tensor],
        use_beat_aligned: bool = True,
    ) -> torch.Tensor:
        """Crossfade and concatenate audio segments with equal-power crossfade.

        Uses equal-power (sine/cosine) crossfade curves for constant perceived loudness,
        zero-crossing snapping to prevent clicks, and optional beat alignment.

        Args:
            segments: List of audio tensors
            use_beat_aligned: Whether to align crossfades to beat boundaries

        Returns:
            Concatenated audio tensor
        """
        if len(segments) == 1:
            return segments[0]

        assert self._model is not None
        sample_rate = self._model.sample_rate
        default_overlap_samples = int(OVERLAP_DURATION * sample_rate)
        result = segments[0]

        for i, segment in enumerate(segments[1:], 1):
            # Determine crossfade position
            if use_beat_aligned:
                # Find beat-aligned boundary
                target_pos = result.shape[-1] - default_overlap_samples
                aligned_pos = self._detect_beat_boundary(
                    result, sample_rate, target_pos, default_overlap_samples
                )
                overlap_samples = result.shape[-1] - aligned_pos
                # Clamp to reasonable bounds (0.5s to 5s)
                overlap_samples = max(
                    int(0.5 * sample_rate),
                    min(overlap_samples, int(5.0 * sample_rate)),
                )
            else:
                overlap_samples = default_overlap_samples

            # Snap to zero crossings to prevent clicks
            end_pos = self._find_zero_crossing(result, result.shape[-1] - overlap_samples)
            start_pos = self._find_zero_crossing(segment, overlap_samples)

            # Recalculate actual overlap based on zero-crossing positions
            actual_overlap = result.shape[-1] - end_pos
            actual_overlap = min(actual_overlap, start_pos)
            actual_overlap = max(actual_overlap, int(0.1 * sample_rate))  # Min 100ms

            # Equal-power crossfade curves (constant energy)
            t = torch.linspace(0, math.pi / 2, actual_overlap, device=result.device)
            fade_out = torch.cos(t)
            fade_in = torch.sin(t)

            # Apply crossfade to overlap region
            overlap_region = (
                result[:, -actual_overlap:] * fade_out
                + segment[:, :actual_overlap] * fade_in
            )

            # Concatenate: non-overlap from previous + crossfaded overlap + rest of current
            result = torch.cat(
                [result[:, :-actual_overlap], overlap_region, segment[:, actual_overlap:]],
                dim=1,
            )
            logger.debug(f"Crossfaded segment {i} with {actual_overlap} samples overlap")

        return result

    async def generate_segment_async(
        self,
        prompt: str,
        duration: int,
        temperature: float = 0.85,
        cfg_coef: float = 3.0,
    ) -> torch.Tensor:
        """Generate a single audio segment asynchronously.

        Public wrapper for generating audio segments used by soundtrack generator.

        Args:
            prompt: Text prompt for generation
            duration: Duration in seconds
            temperature: Sampling temperature (default 0.85 for coherence)
            cfg_coef: Classifier-free guidance coefficient (default 3.0)

        Returns:
            Generated audio tensor
        """
        import asyncio

        self.load_model()
        assert self._model is not None
        model = self._model  # Capture for lambda
        model.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250,
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: model.generate([prompt], progress=False),
        )
        return audio[0]

    async def generate_continuation_async(
        self,
        prompt: str,
        duration: int,
        conditioning_audio: torch.Tensor,
        temperature: float = 0.85,
        cfg_coef: float = 3.0,
    ) -> torch.Tensor:
        """Generate continuation from audio context asynchronously.

        Public wrapper for continuation generation used by soundtrack generator.

        Args:
            prompt: Text prompt for new segment
            duration: Duration in seconds
            conditioning_audio: Audio tensor to continue from (last N seconds)
            temperature: Sampling temperature (default 0.85 for coherence)
            cfg_coef: Classifier-free guidance coefficient (default 3.0)

        Returns:
            Generated audio tensor
        """
        import asyncio

        self.load_model()
        assert self._model is not None
        model = self._model  # Capture for lambda
        model.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250,
            temperature=temperature,
            cfg_coef=cfg_coef,
        )

        # Ensure proper tensor shape [batch, channels, samples]
        if conditioning_audio.dim() == 2:
            conditioning_audio = conditioning_audio.unsqueeze(0)

        sample_rate = model.sample_rate
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: model.generate_continuation(
                conditioning_audio,
                sample_rate,
                [prompt],
                progress=False,
            ),
        )
        return audio[0]

    async def generate_with_style_async(
        self,
        prompt: str,
        duration: int,
        style_audio: torch.Tensor,
        temperature: float = 0.85,
        cfg_coef: float = 5.0,
        cfg_coef_beta: float = 2.0,
    ) -> torch.Tensor:
        """Generate audio with style conditioning from reference audio.

        Uses MusicGen-Style's double CFG to maintain style from reference audio
        while allowing text prompts to guide mood/energy variations.

        Args:
            prompt: Text prompt for mood/energy guidance
            duration: Duration in seconds
            style_audio: Reference audio tensor for style conditioning (1.5-4.5s recommended)
            temperature: Sampling temperature (default 0.85)
            cfg_coef: Style conditioning strength (default 5.0 - strong style adherence)
            cfg_coef_beta: Text conditioning strength (default 2.0 - subtle text influence)

        Returns:
            Generated audio tensor with consistent style
        """
        import asyncio

        self.load_model()
        assert self._model is not None
        model = self._model  # Capture for lambda

        # Set generation params with double CFG for style model
        model.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250,
            temperature=temperature,
            cfg_coef=cfg_coef,
            cfg_coef_beta=cfg_coef_beta,
        )

        # Ensure proper tensor shape [batch, channels, samples]
        if style_audio.dim() == 2:
            style_audio = style_audio.unsqueeze(0)

        # Trim style audio to recommended duration (1.5-4.5 seconds)
        sample_rate = model.sample_rate
        max_style_samples = int(STYLE_CONDITION_DURATION * sample_rate)
        if style_audio.shape[-1] > max_style_samples:
            # Take from the middle for more representative style
            start = (style_audio.shape[-1] - max_style_samples) // 2
            style_audio = style_audio[:, :, start:start + max_style_samples]

        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: model.generate_with_chroma(
                [prompt],
                style_audio,
                sample_rate,
                progress=False,
            ),
        )
        return audio[0]

    @property
    def supports_style_conditioning(self) -> bool:
        """Check if current model supports style conditioning.

        Returns:
            True if model is musicgen-style, False otherwise
        """
        return "style" in MUSICGEN_MODEL.lower()


# Global instance
musicgen = MusicGenWrapper(output_dir=os.getenv("OUTPUT_DIR", "/data/output"))
