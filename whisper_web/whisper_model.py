import torch
import asyncio

from pydantic import BaseModel, Field
from typing import Optional, Tuple
from torch import Tensor
from datetime import datetime

from whisper_web.utils import set_device
from whisper_web.events import EventBus, TranscriptionCompleted
from whisper_web.types import Transcription
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging
from dotenv import load_dotenv

load_dotenv()

logging.set_verbosity_error()


class ModelConfig(BaseModel):
    """Configuration for creating and loading the Whisper ASR model.

    This class contains the configuration options required for initializing the Whisper model,
    including the model size, device type, and other parameters related to the inference process.
    """

    model_id: Optional[str] = Field(default=None, description="The model id to be used for loading the model.")
    model_size: str = Field(
        default="large-v3",
        description="The size of the model to be used for inference. choices=['small', 'medium', 'large-v3']",
    )
    device: str = Field(
        default="cuda",
        description="The device to be used for inference. choices=['cpu', 'cuda', 'mps']",
    )
    continuous: bool = Field(default=True, description="Whether to generate audio data continuously or not.")
    use_vad: bool = Field(default=False, description="Whether to use VAD (Voice Activity Detection) or not.")
    samplerate: int = Field(default=16000, description="The sample rate of the generated audio.")


class WhisperModel:
    """Event-driven Whisper ASR model wrapper with optimized inference capabilities.

    This class provides a high-level interface to OpenAI's Whisper models (via Transformers)
    with event-driven architecture, device optimization, and asynchronous processing. It handles
    model loading, configuration, and transcription with automatic device selection and performance
    optimizations.

    **Core Features:**

    - **Event-Driven Architecture**: Publishes transcription results via event bus
    - **Device Optimization**: Automatic device selection with CUDA, MPS, and CPU support
    - **Async Processing**: Non-blocking transcription using thread pools
    - **Model Flexibility**: Supports various Whisper model sizes and custom model IDs
    - **Performance Optimizations**: Includes dtype optimization and CUDA acceleration
    - **Distil-Whisper Integration**: Optimized distilled models for faster inference

    **Supported Models:**

    - **small**: distil-whisper/distil-small.en (English only, fastest)
    - **medium**: distil-whisper/distil-medium.en (English only, balanced)
    - **large-v3**: distil-whisper/distil-large-v3 (Multilingual, most accurate)
    - **Custom**: Any HuggingFace Whisper-compatible model ID

    **Device Support:**

    - **CUDA**: GPU acceleration with float16 precision
    - **MPS**: Apple Silicon GPU acceleration
    - **CPU**: Fallback with float32 precision

    :param model_args: Configuration object specifying model and device settings
    :type model_args: :class:`ModelConfig`
    :param event_bus: Event bus for publishing transcription results
    :type event_bus: :class:`EventBus`

    :ivar device: Torch device used for model inference
    :type device: :class:`torch.device`
    :ivar samplerate: Audio sample rate for processing
    :type samplerate: :class:`int`
    :ivar torch_dtype: Data type used for model computations
    :type torch_dtype: :class:`torch.dtype`
    :ivar speech_model: Loaded Whisper model for conditional generation
    :type speech_model: :class:`Optional[WhisperForConditionalGeneration]`
    :ivar processor: Whisper processor for audio preprocessing
    :type processor: :class:`Optional[WhisperProcessor]`

    .. note::
        The model automatically handles device placement, dtype conversion, and
        publishes results through the event system for loose coupling.
    """

    def __init__(self, model_args: ModelConfig, event_bus: EventBus):
        self.event_bus = event_bus

        self.device = set_device(model_args.device)  # type: ignore
        self.samplerate: int = model_args.samplerate
        self.torch_dtype = torch.float16 if self.device == torch.device("cuda") else torch.float32

        if self.device == torch.device("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore

        self.available_model_sizes = []
        self.model_size = ""
        self.model_id = ""

        self.speech_model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None

        self.load_model(model_args.model_size, model_args.model_id)

    def load_model(self, model_size: str, model_id: Optional[str]) -> None:
        """Load and initialize the Whisper ASR model with optimized configuration.

        This method handles the complete model loading process including model ID resolution,
        cache management, device placement, and performance optimizations. It supports both
        predefined model sizes and custom model IDs from HuggingFace.

        :param model_size: Predefined model size ('small', 'medium', 'large-v3') or custom size
        :type model_size: :class:`str`
        :param model_id: Optional custom HuggingFace model ID. Overrides model_size if provided
        :type model_id: :class:`Optional[str]`

        **Model ID Resolution:**

        - If `model_id` is provided, uses it directly
        - Otherwise, maps `model_size` to appropriate distil-whisper model:
          
          * 'small' → 'distil-whisper/distil-small.en'
          * 'medium' → 'distil-whisper/distil-medium.en'  
          * 'large-v3' → 'distil-whisper/distil-large-v3'
          * 'large' → 'distil-whisper/distil-large-v3' (legacy mapping)

        **Optimizations Applied:**

        - Automatic dtype selection (float16 for CUDA, float32 for CPU/MPS)
        - Low CPU memory usage during loading
        - SafeTensors format for improved security and performance
        - Configurable cache directory via HF_HOME environment variable

        **Cache Management:**

        - Uses HF_HOME environment variable if set
        - Falls back to './.models' directory for local caching
        - Enables offline usage after initial download

        .. note::
            Model loading may take time on first use due to download requirements.
            Subsequent loads use cached models for faster initialization.

        .. warning::
            Ensure sufficient disk space for model caching. Large models can
            require several GB of storage space.
        """
        if model_id is None:
            self.available_model_sizes = ["small", "medium", "large-v3"]

            self.model_size = "large-v3" if model_size == "large" else model_size

            if model_size not in self.available_model_sizes:
                print(f"Model size not supported. Defaulting to {self.model_size}.")

            self.model_id = (
                f"distil-whisper/distil-{self.model_size}.en"
                if self.model_size in self.available_model_sizes[:2]
                else f"distil-whisper/distil-{self.model_size}"
            )
        else:
            self.model_id = model_id

        # Use HF_HOME environment variable if set, otherwise default to relative path
        import os

        cache_dir = os.environ.get("HF_HOME", "./.models")

        self.speech_model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir,
        ).to(self.device)  # type: ignore

        self.processor = WhisperProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)  # type: ignore

    async def _transcribe(self, audio: Tensor) -> str:
        """Perform core speech-to-text transcription on audio tensor.

        This method handles the complete transcription pipeline from audio preprocessing
        to text generation using the loaded Whisper model. It includes device placement,
        async processing, and optimized generation parameters.

        :param audio: Raw audio tensor with shape (samples,) at the configured sample rate
        :type audio: :class:`Tensor`
        :return: Transcribed text with leading/trailing whitespace removed
        :rtype: :class:`str`

        **Processing Pipeline:**

        1. **Audio Preprocessing**: Converts audio tensor to model input format
        2. **Device Placement**: Moves inputs to appropriate device with correct dtype
        3. **Text Generation**: Uses optimized parameters for fast, accurate transcription
        4. **Post-processing**: Decodes tokens to text and cleans output

        **Generation Parameters:**

        - `max_new_tokens=128`: Limits output length for efficiency
        - `num_beams=1`: Greedy decoding for speed
        - `return_timestamps=False`: Text-only output
        - Proper padding and EOS token handling

        **Async Processing:**

        - Model inference runs in thread pool to avoid blocking
        - Token decoding also runs async for consistent performance
        - Maintains responsiveness during heavy computation

        .. note::
            This method assumes valid audio input and properly initialized model/processor.
            Input validation is performed by the calling method.

        .. warning::
            Large audio inputs may consume significant GPU memory. Consider chunking
            very long audio sequences for memory-constrained environments.
        """
        assert isinstance(self.processor, WhisperProcessor), "Processor must be an instance of WhisperProcessor"
        assert isinstance(self.speech_model, WhisperForConditionalGeneration), "Speechmodel must be instance of WhisperForConditionalGeneration"

        inputs = self.processor(
            audio,
            sampling_rate=self.samplerate,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device, dtype=self.torch_dtype)

        generated_ids = await asyncio.to_thread(
            self.speech_model.generate,
            **inputs,
            max_new_tokens=128,
            num_beams=1,
            return_timestamps=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,  # type: ignore
            eos_token_id=self.processor.tokenizer.eos_token_id,  # type: ignore
        )

        transcript = await asyncio.to_thread(
            self.processor.batch_decode, generated_ids, skip_special_tokens=True, decode_with_timestamps=False, return_timestamps=False
        )

        return transcript[-1].strip()

    async def __call__(self, audio_data: Tuple[Tensor, bool]) -> None:
        """Process audio data and publish transcription results via event system.

        This method serves as the main entry point for audio transcription, handling
        input validation, transcription processing, and result publication through
        the event bus. It's designed to be called by the transcription manager.

        :param audio_data: Tuple containing audio tensor and finality flag
        :type audio_data: Tuple[Tensor, bool]

        **Parameters:**

        - `audio_data[0]`: Audio tensor with shape (samples,) at configured sample rate
        - `audio_data[1]`: Boolean flag indicating if this is the final audio chunk

        **Processing Flow:**

        1. **Input Validation**: Ensures audio data is a valid PyTorch tensor
        2. **Transcription**: Calls internal transcription method with audio tensor
        3. **Result Packaging**: Creates Transcription object with text and timestamp
        4. **Event Publication**: Publishes TranscriptionCompleted event with results

        **Event Publication:**

        - Event Type: `TranscriptionCompleted`
        - Payload: Transcription object with text and current date
        - Finality Flag: Passed through from input parameters

        **Error Handling:**

        - Validates tensor input with assertion
        - Async processing handles model inference errors
        - Event publication errors propagate to caller

        .. note::
            This method is typically called automatically by the transcription
            manager's inference loop rather than directly by user code.
        """
        audio, is_final = audio_data
        assert isinstance(audio, Tensor), "Audio data must be a torch.Tensor"

        transcript_text = await self._transcribe(audio)
        transcription = Transcription(transcript_text, datetime.now().date())

        await self.event_bus.publish(TranscriptionCompleted(transcription=transcription, is_final=is_final))
