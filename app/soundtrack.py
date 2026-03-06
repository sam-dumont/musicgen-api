"""Multi-mood soundtrack generation with seamless transitions."""

import logging
import math
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from app.audio_processing import USE_STEM_AWARE_CROSSFADE, stem_aware_crossfade

logger = logging.getLogger(__name__)

# Generation parameters
TEMPERATURE = 0.85  # sampling temperature
CFG_COEF = 3.0  # classifier-free guidance coefficient
MAX_SEGMENT_DURATION = 30  # MusicGen max duration per segment
MIN_SEGMENT_DURATION = 5  # MusicGen minimum duration (avoid assertion errors)
OVERLAP_DURATION = 8  # seconds of overlap for continuation (matches musicgen.py)
CROSSFADE_DURATION = 2.0  # crossfade duration in seconds
FADE_BUFFER = 5  # extra seconds at end for video fade out


class SoundtrackGenerator:
    """Generator for multi-mood video soundtracks."""

    def __init__(self, musicgen_wrapper, output_dir: str = "/data/output") -> None:
        """Initialize with MusicGenWrapper instance.

        Args:
            musicgen_wrapper: MusicGenWrapper instance for generation
            output_dir: Directory for output files
        """
        self._musicgen = musicgen_wrapper
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SoundtrackGenerator initialized (multi-mood mode)")

    async def generate_soundtrack(
        self,
        scenes: list[dict[str, Any]],
        base_prompt: str,
        melody_audio_url: str | None = None,
        use_beat_aligned_crossfade: bool = True,
        crossfade_duration: float = 2.0,
        job_id: str = "",
        progress_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
    ) -> str:
        """Generate multi-mood soundtrack from scene timeline.

        Args:
            scenes: List of {mood, duration, prompt} dictionaries
            base_prompt: Base musical description for consistency
            melody_audio_url: Optional URL for melody conditioning (not implemented)
            use_beat_aligned_crossfade: Whether to align crossfades to beats
            crossfade_duration: Crossfade overlap in seconds
            job_id: Job identifier for output filename
            progress_callback: Async callback for progress updates

        Returns:
            Path to generated soundtrack file
        """
        if melody_audio_url:
            logger.warning("Melody conditioning not yet implemented, ignoring melody_audio_url")

        total_scenes = len(scenes)
        scene_audios: list[torch.Tensor] = []
        sample_rate = self._musicgen.sample_rate

        # Calculate total crossfade overhead (each crossfade consumes CROSSFADE_DURATION)
        num_crossfades = max(0, total_scenes - 1)
        total_crossfade_overhead = num_crossfades * CROSSFADE_DURATION

        total_requested = sum(s.get("duration", 30) for s in scenes)
        logger.info(
            f"Generating soundtrack with {total_scenes} scenes, "
            f"{total_requested}s requested + {total_crossfade_overhead}s crossfade overhead"
        )

        for i, scene in enumerate(scenes):
            scene_num = i + 1
            mood = scene.get("mood") or ""
            logger.info(f"Generating scene {scene_num}/{total_scenes}: {mood[:50] if mood else '(no mood)'}")

            # Report progress
            if progress_callback:
                progress = (i / total_scenes) * 90  # Reserve 10% for crossfading
                await progress_callback(
                    progress,
                    current_scene=scene_num,
                    total_scenes=total_scenes,
                    stage="generating",
                )

            # Build prompt for this scene: "mood, base_prompt"
            scene_prompt = self._build_scene_prompt(base_prompt, scene)
            scene_duration = scene.get("duration", 30)

            # Add extra duration to compensate for crossfade loss and fade out
            # Each scene (except last) loses CROSSFADE_DURATION to crossfade
            # Last scene gets FADE_BUFFER extra for video fade out
            # Cap padded duration to MAX_SEGMENT_DURATION to avoid triggering
            # sliding window for tiny overflows (which creates destructive tail segments)
            is_last_scene = (i == total_scenes - 1)
            if is_last_scene:
                padding = min(FADE_BUFFER, MAX_SEGMENT_DURATION - scene_duration)
                padding = max(0, padding)
                scene_duration = int(scene_duration + padding)
            else:
                padding = min(CROSSFADE_DURATION, MAX_SEGMENT_DURATION - scene_duration)
                padding = max(0, padding)
                scene_duration = int(scene_duration + padding)

            # Generate scene audio
            if i == 0:
                # First scene: generate from scratch
                scene_audio = await self._generate_scene_audio(scene_prompt, scene_duration)
                scene_audios.append(scene_audio)
            else:
                # Subsequent scenes: continue from previous scene's audio
                context_samples = int(OVERLAP_DURATION * sample_rate)
                prev_audio = scene_audios[-1]
                conditioning_audio = prev_audio[:, -context_samples:]
                scene_audio = await self._generate_scene_with_continuation(
                    scene_prompt, scene_duration, conditioning_audio
                )
                scene_audios.append(scene_audio)

        # Crossfade all scene audios
        logger.info("Crossfading scenes...")
        if progress_callback:
            await progress_callback(
                90,
                current_scene=total_scenes,
                total_scenes=total_scenes,
                stage="crossfading",
            )

        final_audio = self._crossfade_segments(scene_audios, sample_rate)

        # Save output
        output_path = self._output_dir / f"{job_id}_soundtrack.wav"
        torchaudio.save(str(output_path), final_audio.cpu(), sample_rate)

        if progress_callback:
            await progress_callback(
                100,
                current_scene=total_scenes,
                total_scenes=total_scenes,
                stage="finalizing",
            )

        logger.info(f"Soundtrack saved to {output_path}")
        return str(output_path)

    def _build_scene_prompt(self, base_prompt: str, scene: dict[str, Any]) -> str:
        """Combine base prompt with scene mood/prompt.

        Args:
            base_prompt: Base musical description
            scene: Scene dictionary with mood and optional prompt

        Returns:
            Combined prompt string: "mood, base_prompt", scene's custom prompt,
            or just base_prompt if no mood specified
        """
        mood = scene.get("mood") or ""
        scene_prompt = scene.get("prompt")

        if scene_prompt:
            # Use scene-specific prompt with mood (if any)
            if mood:
                return f"{mood}, {scene_prompt}"
            return scene_prompt
        else:
            # Combine mood with base prompt (or just base prompt if no mood)
            if mood:
                return f"{mood}, {base_prompt}"
            return base_prompt

    async def _generate_scene_audio(
        self,
        prompt: str,
        duration: int,
    ) -> torch.Tensor:
        """Generate audio for a scene, handling long durations with sliding window.

        Uses the same approach as musicgen.py's _generate_long which is proven to work.

        Args:
            prompt: Text prompt for generation
            duration: Duration in seconds

        Returns:
            Generated audio tensor
        """
        # If duration is only slightly over MAX_SEGMENT_DURATION, cap it to avoid
        # creating a useless tiny tail segment in the sliding window. The tiny tail
        # (e.g. 5s with 4s context = 1s new audio) gets destroyed by the 8s crossfade
        # in musicgen._crossfade_segments, causing beat grid destruction and volume drops.
        if duration > MAX_SEGMENT_DURATION and duration <= MAX_SEGMENT_DURATION + OVERLAP_DURATION:
            logger.info(f"  Capping scene duration from {duration}s to {MAX_SEGMENT_DURATION}s (avoids tiny tail segment)")
            duration = MAX_SEGMENT_DURATION

        if duration <= MAX_SEGMENT_DURATION:
            # Short scene: single generation (use direct model call like _generate_long)
            import asyncio
            model = self._musicgen.model
            model.set_generation_params(duration=duration)
            loop = asyncio.get_event_loop()
            audio_result = await loop.run_in_executor(
                None,
                lambda: model.generate([prompt]),
            )
            return audio_result[0]

        # Long scene: use exact same sliding window as musicgen.py _generate_long
        # This approach is proven to work without silence issues
        logger.info(f"  Long scene: {duration}s using sliding window")

        segments: list[torch.Tensor] = []
        current_pos = 0
        segment_num = 0
        sample_rate = self._musicgen.sample_rate

        overlap_duration = OVERLAP_DURATION

        while current_pos < duration:
            seg_duration = min(MAX_SEGMENT_DURATION, duration - current_pos)
            seg_duration = max(MIN_SEGMENT_DURATION, seg_duration)

            if segment_num == 0:
                # First segment - generate fresh (use direct model call like _generate_long)
                logger.info(f"  Generating segment {segment_num + 1} at pos {current_pos}s ({seg_duration}s)...")
                import asyncio
                model = self._musicgen.model
                model.set_generation_params(duration=seg_duration)
                loop = asyncio.get_event_loop()
                audio_result = await loop.run_in_executor(
                    None,
                    lambda m=model: m.generate([prompt]),  # type: ignore[misc]
                )
                audio = audio_result[0]
            else:
                # Continuation - use last N seconds as context
                # Context must be shorter than segment duration (MusicGen requirement)
                context_duration = min(overlap_duration, seg_duration - 1)
                logger.info(f"  Generating segment {segment_num + 1} at pos {current_pos}s ({seg_duration}s, {context_duration}s context)...")
                prev_audio = segments[-1]
                context_samples = int(context_duration * sample_rate)
                conditioning = prev_audio[:, -context_samples:]

                # Use same approach as simple _generate_long (no temperature/cfg_coef)
                import asyncio
                model = self._musicgen.model
                model.set_generation_params(duration=seg_duration)
                loop = asyncio.get_event_loop()
                audio_result = await loop.run_in_executor(
                    None,
                    lambda cond=conditioning, sr=sample_rate, m=model: m.generate_continuation(  # type: ignore[misc]
                        cond.unsqueeze(0),
                        sr,
                        [prompt],
                        progress=False,
                    ),
                )
                audio = audio_result[0]

            segments.append(audio)
            segment_num += 1

            # Move position (same logic as musicgen.py)
            if segment_num == 1:
                current_pos += MAX_SEGMENT_DURATION
            else:
                current_pos += MAX_SEGMENT_DURATION - overlap_duration

        # Use musicgen's crossfade (with beat alignment) instead of our simple one
        return self._musicgen._crossfade_segments(segments)

    async def _generate_scene_with_continuation(
        self,
        prompt: str,
        duration: int,
        conditioning_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a scene continuing from previous audio.

        Uses the same sliding window approach as musicgen.py's _generate_long.

        Args:
            prompt: Text prompt for new scene (includes mood)
            duration: Duration in seconds
            conditioning_audio: Audio tensor to continue from

        Returns:
            Generated audio tensor
        """
        sample_rate = self._musicgen.sample_rate
        overlap_duration = OVERLAP_DURATION

        # If duration is only slightly over MAX_SEGMENT_DURATION, cap it to avoid
        # creating a useless tiny tail segment (same fix as _generate_scene_audio)
        if duration > MAX_SEGMENT_DURATION and duration <= MAX_SEGMENT_DURATION + overlap_duration:
            logger.info(f"  Capping continuation duration from {duration}s to {MAX_SEGMENT_DURATION}s (avoids tiny tail segment)")
            duration = MAX_SEGMENT_DURATION

        if duration <= MAX_SEGMENT_DURATION:
            # Short scene: single continuation (use direct model call like _generate_long)
            # Limit conditioning to be shorter than duration (MusicGen requirement)
            max_context = duration - 1
            if conditioning_audio.shape[-1] > max_context * sample_rate:
                conditioning_audio = conditioning_audio[:, -int(max_context * sample_rate):]
            logger.info(f"  Short scene continuation ({duration}s)...")
            import asyncio
            model = self._musicgen.model
            model.set_generation_params(duration=duration)
            loop = asyncio.get_event_loop()
            # Ensure proper tensor shape [batch, channels, samples]
            cond = conditioning_audio.unsqueeze(0) if conditioning_audio.dim() == 2 else conditioning_audio
            audio_result = await loop.run_in_executor(
                None,
                lambda c=cond, sr=sample_rate, m=model: m.generate_continuation(  # type: ignore[misc]
                    c, sr, [prompt], progress=False
                ),
            )
            return audio_result[0]

        # Long scene: sliding window with continuation (same as musicgen.py)
        logger.info(f"  Long scene with continuation: {duration}s using sliding window")

        segments: list[torch.Tensor] = []
        current_pos = 0
        segment_num = 0

        while current_pos < duration:
            seg_duration = min(MAX_SEGMENT_DURATION, duration - current_pos)
            seg_duration = max(MIN_SEGMENT_DURATION, seg_duration)

            if segment_num == 0:
                # First segment - continue from previous scene's audio (use direct model call like _generate_long)
                # Limit conditioning to be shorter than segment duration (MusicGen requirement)
                max_context = seg_duration - 1
                if conditioning_audio.shape[-1] > max_context * sample_rate:
                    cond = conditioning_audio[:, -int(max_context * sample_rate):]
                else:
                    cond = conditioning_audio
                logger.info(f"  Generating segment {segment_num + 1} at pos {current_pos}s ({seg_duration}s, from prev scene)...")
                import asyncio
                model = self._musicgen.model
                model.set_generation_params(duration=seg_duration)
                loop = asyncio.get_event_loop()
                audio_result = await loop.run_in_executor(
                    None,
                    lambda c=cond, sr=sample_rate, m=model: m.generate_continuation(  # type: ignore[misc]
                        c.unsqueeze(0), sr, [prompt], progress=False
                    ),
                )
                audio = audio_result[0]
            else:
                # Subsequent - continue from within this scene
                # Context must be shorter than segment duration (MusicGen requirement)
                context_duration = min(overlap_duration, seg_duration - 1)
                logger.info(f"  Generating segment {segment_num + 1} at pos {current_pos}s ({seg_duration}s, {context_duration}s context)...")
                prev_audio = segments[-1]
                context_samples = int(context_duration * sample_rate)
                conditioning = prev_audio[:, -context_samples:]

                # Use same approach as simple _generate_long (no temperature/cfg_coef)
                import asyncio
                model = self._musicgen.model
                model.set_generation_params(duration=seg_duration)
                loop = asyncio.get_event_loop()
                audio_result = await loop.run_in_executor(
                    None,
                    lambda cond=conditioning, sr=sample_rate, m=model: m.generate_continuation(  # type: ignore[misc]
                        cond.unsqueeze(0),
                        sr,
                        [prompt],
                        progress=False,
                    ),
                )
                audio = audio_result[0]

            segments.append(audio)
            segment_num += 1

            # Move position (same logic as musicgen.py)
            if segment_num == 1:
                current_pos += MAX_SEGMENT_DURATION
            else:
                current_pos += MAX_SEGMENT_DURATION - overlap_duration

        # Use musicgen's crossfade (with beat alignment)
        return self._musicgen._crossfade_segments(segments)

    def _crossfade_segments(
        self,
        segments: list[torch.Tensor],
        sample_rate: int,
    ) -> torch.Tensor:
        """Crossfade segments with stem-aware or equal-power crossfade.

        Uses stem-aware crossfade (Demucs-based) when enabled for better
        quality transitions.

        Args:
            segments: List of audio tensors
            sample_rate: Sample rate in Hz

        Returns:
            Crossfaded audio tensor
        """
        if len(segments) == 0:
            raise ValueError("No segments to crossfade")

        if len(segments) == 1:
            return segments[0]

        result = segments[0]

        for i, segment in enumerate(segments[1:], 1):
            if USE_STEM_AWARE_CROSSFADE:
                # Use stem-aware crossfade with residual 'other' stem for perfect reconstruction
                logger.info(f"Applying stem-aware crossfade for segment {i}")
                result_np = stem_aware_crossfade(
                    result, segment, sample_rate, CROSSFADE_DURATION
                )
                result = torch.from_numpy(result_np.astype(np.float32))
            else:
                # Simple equal-power crossfade
                overlap_samples = int(CROSSFADE_DURATION * sample_rate)
                result = self._equal_power_crossfade(result, segment, overlap_samples)

        return result

    def _equal_power_crossfade(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        overlap_samples: int,
    ) -> torch.Tensor:
        """Apply equal-power crossfade between two audio segments.

        Args:
            audio1: First audio tensor [channels, samples]
            audio2: Second audio tensor [channels, samples]
            overlap_samples: Number of samples to crossfade

        Returns:
            Crossfaded audio tensor
        """
        overlap_samples = min(
            overlap_samples,
            audio1.shape[-1] // 2,
            audio2.shape[-1] // 2,
        )

        # Equal-power fade curves
        t = torch.linspace(0, math.pi / 2, overlap_samples, device=audio1.device)
        fade_out = torch.cos(t)
        fade_in = torch.sin(t)

        # Apply crossfade
        overlap_region = (
            audio1[:, -overlap_samples:] * fade_out + audio2[:, :overlap_samples] * fade_in
        )

        return torch.cat(
            [audio1[:, :-overlap_samples], overlap_region, audio2[:, overlap_samples:]],
            dim=1,
        )


# Global instance (initialized in main.py)
soundtrack_generator: SoundtrackGenerator | None = None
