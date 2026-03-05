"""Demucs wrapper for audio stem separation with lazy loading."""

import functools
import logging
import os
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchaudio

from app.musicgen import clear_device_cache, get_device

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "htdemucs"  # Hybrid Transformer Demucs
TWO_STEMS = False  # Use 4-stem mode: drums, bass, other, vocals (better for instrumentals)


class DemucsWrapper:
    """Wrapper for Demucs with lazy loading and VRAM management."""

    def __init__(self, output_dir: str = "/data/output") -> None:
        """Initialize wrapper.

        Args:
            output_dir: Directory for output files
        """
        self._model = None
        self._device = get_device()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Demucs will use device: {self._device}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def load_model(self) -> None:
        """Load Demucs model lazily."""
        if self._model is not None:
            return

        logger.info(f"Loading Demucs model: {MODEL_NAME} on {self._device}")
        from demucs.pretrained import get_model

        self._model = get_model(MODEL_NAME)
        assert self._model is not None

        # Move to appropriate device
        if self._device != "cpu":
            self._model = self._model.to(self._device)

        logger.info("Demucs model loaded successfully")

    def unload_model(self) -> None:
        """Unload model and free VRAM/MPS memory."""
        if self._model is not None:
            del self._model
            self._model = None
            clear_device_cache()
            logger.info("Demucs model unloaded, memory freed")

    async def separate(
        self,
        audio_path: str | None = None,
        audio_url: str | None = None,
        job_id: str = "",
        progress_callback: Callable | None = None,
    ) -> list[str]:
        """Separate audio into stems.

        Args:
            audio_path: Path to local audio file
            audio_url: URL of audio file to download
            job_id: Job ID for output filenames
            progress_callback: Optional callback for progress updates

        Returns:
            List of paths to separated stem files
        """
        import asyncio

        self.load_model()
        assert self._model is not None
        model = self._model  # Capture for lambda

        # Get audio file
        if audio_url:
            audio_path = await self._download_audio(audio_url, job_id)
        elif not audio_path:
            raise ValueError("Either audio_path or audio_url must be provided")

        if progress_callback:
            await progress_callback(10.0)

        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.info(f"Loaded audio: {waveform.shape}, {sample_rate}Hz")

            # Resample if necessary
            if sample_rate != model.samplerate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=model.samplerate,
                )
                waveform = resampler(waveform)
                sample_rate = model.samplerate

            # Ensure stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]

            if progress_callback:
                await progress_callback(20.0)

            # Add batch dimension and move to device
            waveform = waveform.unsqueeze(0)
            if self._device != "cpu":
                waveform = waveform.to(self._device)

            # Run separation
            loop = asyncio.get_event_loop()
            sources = await loop.run_in_executor(
                None,
                lambda: self._run_separation(waveform),
            )

            if progress_callback:
                await progress_callback(80.0)

            # Save stems
            output_paths = await self._save_stems(sources, sample_rate, job_id)

            if progress_callback:
                await progress_callback(100.0)

            return output_paths

        finally:
            # Clean up GPU/MPS memory after separation
            clear_device_cache()

    def _run_separation(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run separation on waveform.

        Args:
            waveform: Audio tensor (batch, channels, samples)

        Returns:
            Dictionary of stem name to audio tensor
        """
        from demucs.apply import apply_model

        assert self._model is not None
        with torch.no_grad():
            sources = apply_model(
                self._model,
                waveform,
                device=waveform.device,
                progress=False,
            )

        # sources shape: (batch, num_sources, channels, samples)
        sources = sources[0]  # Remove batch dimension

        # Map source indices to names
        source_names = self._model.sources
        result = {}

        if TWO_STEMS:
            # Combine into vocals and accompaniment
            vocals_idx = source_names.index("vocals") if "vocals" in source_names else None
            if vocals_idx is not None:
                result["vocals"] = sources[vocals_idx]
                # Accompaniment is everything except vocals
                accompaniment = torch.zeros_like(sources[0])
                for i, name in enumerate(source_names):
                    if name != "vocals":
                        accompaniment += sources[i]
                result["accompaniment"] = accompaniment
        else:
            # Return all stems
            for i, name in enumerate(source_names):
                result[name] = sources[i]

        return result

    async def _save_stems(
        self,
        sources: dict[str, torch.Tensor],
        sample_rate: int,
        job_id: str,
    ) -> list[str]:
        """Save separated stems to files.

        Args:
            sources: Dictionary of stem name to audio tensor
            sample_rate: Sample rate
            job_id: Job ID for filenames

        Returns:
            List of output file paths
        """
        import asyncio

        output_paths = []
        loop = asyncio.get_event_loop()

        for name, audio in sources.items():
            output_path = self._output_dir / f"{job_id}_{name}.wav"

            # Move to CPU for saving
            audio_cpu = audio.cpu()

            # Use functools.partial to safely capture loop variables
            save_fn = functools.partial(torchaudio.save, str(output_path), audio_cpu, sample_rate)
            await loop.run_in_executor(None, save_fn)

            output_paths.append(str(output_path))
            logger.info(f"Saved stem {name} to {output_path}")

        return output_paths

    async def _download_audio(self, url: str, job_id: str) -> str:
        """Download audio from URL.

        Args:
            url: URL of audio file
            job_id: Job ID for filename

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If URL scheme is not http/https or hostname resolves to private IP
        """
        import asyncio
        import ipaddress
        import socket
        import urllib.request

        parsed = urlparse(url)

        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Only http/https URLs are allowed, got: {parsed.scheme}")

        # Block requests to private/internal IPs (SSRF protection)
        hostname = parsed.hostname
        if hostname:
            try:
                addr = socket.getaddrinfo(hostname, None)[0][4][0]
                ip = ipaddress.ip_address(addr)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise ValueError("URLs pointing to private/internal addresses are not allowed")
            except socket.gaierror:
                raise ValueError(f"Could not resolve hostname: {hostname}") from None

        ext = Path(parsed.path).suffix or ".wav"
        input_path = self._output_dir / f"{job_id}_input{ext}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: urllib.request.urlretrieve(url, str(input_path)),
        )

        logger.info(f"Downloaded audio from {url} to {input_path}")
        return str(input_path)


# Global instance
demucs = DemucsWrapper(output_dir=os.getenv("OUTPUT_DIR", "/data/output"))
