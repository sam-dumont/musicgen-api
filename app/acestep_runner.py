"""ACE-Step 1.5 wrapper with budget GPU configuration.

Designed for low-VRAM GPUs (8GB) like GTX 1070 and T1000.
Handles Pascal architecture limitations (no bf16, no Flash Attention).

For dual-GPU setups, use separate Docker services (see docker-compose.budget-gpu.yml)
rather than running both models in one process, since ACE-Step only supports
GPU selection via CUDA_VISIBLE_DEVICES which is process-wide.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ACE-Step configuration via environment variables
ACESTEP_DEVICE = os.getenv("ACESTEP_DEVICE", "auto")
ACESTEP_CONFIG_PATH = os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
ACESTEP_LM_MODEL_PATH = os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
ACESTEP_LM_BACKEND = os.getenv("ACESTEP_LM_BACKEND", "pt")
ACESTEP_INIT_LLM = os.getenv("ACESTEP_INIT_LLM", "auto")
# Whether ACE-Step support is enabled at all (disabled in musicgen-only deployments)
ACESTEP_ENABLED = os.getenv("ACESTEP_ENABLED", "true").lower() in ("true", "1", "yes")


def _detect_device() -> str:
    """Detect the best device for ACE-Step inference.

    Returns:
        Device string: 'cuda', 'cpu', etc.
    """
    if ACESTEP_DEVICE != "auto":
        return ACESTEP_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_pascal_gpu() -> bool:
    """Check if the current CUDA GPU is Pascal architecture (compute < 7.0).

    Pascal GPUs (GTX 1070/1080, T1000, P100) lack bf16 and Flash Attention.

    Returns:
        True if Pascal or older GPU detected
    """
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] < 7


def _detect_dtype() -> str:
    """Detect safe dtype for the current GPU.

    Pascal GPUs (compute < 7.0) don't support bf16.
    Turing (7.x) has limited bf16. Ampere+ (8.0+) has full bf16.
    Falls back to float16 on older GPUs.

    Returns:
        Dtype string: 'bfloat16', 'float16', or 'float32'
    """
    if not torch.cuda.is_available():
        return "float32"

    capability = torch.cuda.get_device_capability()
    # bf16 requires compute capability >= 8.0 (Ampere) for full support
    if capability[0] >= 8:
        return "bfloat16"
    logger.info(
        f"GPU compute capability {capability[0]}.{capability[1]} < 8.0, "
        f"using float16 instead of bfloat16 (Pascal/Turing compatibility)"
    )
    return "float16"


class ACEStepWrapper:
    """Wrapper for ACE-Step 1.5 with lazy loading and budget GPU support.

    Uses ACE-Step's built-in GPU tier detection for memory optimization,
    INT8 quantization, and CPU offloading. Adds Pascal-specific workarounds
    for dtype and Flash Attention compatibility.
    """

    def __init__(self, output_dir: str = "/data/output") -> None:
        self._dit_handler = None
        self._llm_handler = None
        self._device = _detect_device()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._enabled = ACESTEP_ENABLED
        if self._enabled:
            logger.info(f"ACE-Step will use device: {self._device}")
        else:
            logger.info("ACE-Step is disabled (ACESTEP_ENABLED=false)")

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    def load_model(self) -> None:
        """Load ACE-Step models lazily.

        Raises:
            RuntimeError: If ACE-Step is disabled via ACESTEP_ENABLED=false
        """
        if self._initialized:
            return

        if not self._enabled:
            raise RuntimeError(
                "ACE-Step is disabled. Set ACESTEP_ENABLED=true or use the "
                "ACE-Step dedicated service in docker-compose.budget-gpu.yml"
            )

        # Set dtype override for Pascal GPUs before importing ACE-Step.
        # ACE_PIPELINE_DTYPE is checked by ACEStepPipeline.__init__()
        dtype = _detect_dtype()
        os.environ["ACE_PIPELINE_DTYPE"] = dtype

        pascal = _is_pascal_gpu()

        logger.info(
            f"Loading ACE-Step: config={ACESTEP_CONFIG_PATH}, "
            f"lm={ACESTEP_LM_MODEL_PATH}, backend={ACESTEP_LM_BACKEND}, "
            f"dtype={dtype}, pascal={pascal}, device={self._device}"
        )

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        # Initialize DiT handler
        self._dit_handler = AceStepHandler()

        # Pascal GPUs: disable Flash Attention (requires SM80+/Ampere)
        init_kwargs = {
            "config_path": ACESTEP_CONFIG_PATH,
            "device": self._device,
        }
        if pascal:
            init_kwargs["use_flash_attention"] = False
        self._dit_handler.initialize_service(**init_kwargs)

        # Initialize LLM handler (if enabled)
        init_llm = ACESTEP_INIT_LLM.lower()
        if init_llm in ("false", "0", "no"):
            self._llm_handler = None
            logger.info("ACE-Step LLM disabled (DiT-only mode)")
        else:
            self._llm_handler = LLMHandler()
            self._llm_handler.initialize(
                lm_model_path=ACESTEP_LM_MODEL_PATH,
                backend=ACESTEP_LM_BACKEND,
                device=self._device,
            )
            logger.info(f"ACE-Step LLM loaded: {ACESTEP_LM_MODEL_PATH} ({ACESTEP_LM_BACKEND})")

        self._initialized = True
        logger.info("ACE-Step models loaded successfully")

    def unload_model(self) -> None:
        """Unload models and free GPU memory."""
        if self._dit_handler is not None:
            del self._dit_handler
            self._dit_handler = None
        if self._llm_handler is not None:
            del self._llm_handler
            self._llm_handler = None
        self._initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ACE-Step models unloaded, memory freed")

    async def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        lyrics: str = "",
        instrumental: bool = True,
        infer_steps: int = 8,
        guidance_scale: float = 7.0,
        batch_size: int = 1,
        seed: int = -1,
        audio_format: str = "wav",
        thinking: bool = True,
        job_id: str = "",
        progress_callback: Callable | None = None,
    ) -> str:
        """Generate music using ACE-Step 1.5.

        Args:
            prompt: Text description of desired music (caption)
            duration: Duration in seconds (10-240)
            lyrics: Lyrics text, or empty for instrumental
            instrumental: Force instrumental generation
            infer_steps: Diffusion steps (8 for turbo, 32-64 for base)
            guidance_scale: Conditioning strength (default 7.0)
            batch_size: Number of parallel generations
            seed: Random seed (-1 for random)
            audio_format: Output format (wav, flac, mp3)
            thinking: Enable LM reasoning/planning
            job_id: Job ID for output filename
            progress_callback: Optional callback for progress updates

        Returns:
            Path to generated audio file
        """
        self.load_model()
        assert self._dit_handler is not None

        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics if lyrics else "[Instrumental]",
            instrumental=instrumental,
            duration=duration,
            inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            thinking=thinking and self._llm_handler is not None,
        )

        config = GenerationConfig(
            batch_size=batch_size,
            audio_format=audio_format,
        )

        save_dir = str(self._output_dir)

        if progress_callback:
            await progress_callback(10.0)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_music(
                self._dit_handler,
                self._llm_handler,
                params,
                config,
                save_dir=save_dir,
            ),
        )

        if progress_callback:
            await progress_callback(90.0)

        # Return the first generated audio path
        if result.success and result.audios:
            output_path = result.audios[0].get("path", "")
            if output_path:
                # Rename to include job_id for consistency
                final_path = self._output_dir / f"{job_id}_acestep.{audio_format}"
                Path(output_path).rename(final_path)
                output_path = str(final_path)
            if progress_callback:
                await progress_callback(100.0)
            logger.info(f"ACE-Step generated audio saved to {output_path}")
            return output_path

        error_msg = getattr(result, "error", "Unknown generation error")
        raise RuntimeError(f"ACE-Step generation failed: {error_msg}")


# Global instance
acestep = ACEStepWrapper(output_dir=os.getenv("OUTPUT_DIR", "/data/output"))
