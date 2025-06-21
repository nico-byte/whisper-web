import torch
import asyncio
import gc

from pydantic import BaseModel, Field
from typing import Optional, Tuple
from torch import Tensor
from datetime import datetime

from whisper_web.utils import set_device, process_transcription_timestamps
from whisper_web.events import EventBus, TranscriptionCompleted
from whisper_web.types import Transcription
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging

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
    batch_size: int = Field(
        default=1,
        description="The batch size to be used for inference. This is the number of audio chunks processed in parallel.",
    )
    batch_timeout_s: float = Field(
        default=0.1,
        description="The timeout in seconds for batch processing. If the batch is not filled within this time, it will be processed anyway.",
    )


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

        self.last_timestamp = 0.0

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

    async def _transcribe(self, audio: list[Tensor]) -> list[str]:
        """Perform batch speech-to-text transcription on audio tensors with timestamps.

        This method handles the complete batch transcription pipeline from audio preprocessing
        to text generation using the loaded Whisper model. It includes tensor-to-numpy conversion,
        device placement, async processing, and optimized generation parameters with timestamp
        extraction for accurate temporal alignment.

        :param audio: List of audio tensors, each with shape (samples,) at the configured sample rate
        :type audio: list[Tensor]
        :return: List of transcribed texts with embedded timestamps in Whisper format
        :rtype: list[str]

        **Processing Pipeline:**

        1. **Tensor Conversion**: Converts PyTorch tensors to numpy arrays for processor compatibility
        2. **Audio Preprocessing**: Batch processes audio with longest padding and attention masks
        3. **Device Placement**: Moves inputs to appropriate device with correct dtype
        4. **Text Generation**: Uses optimized parameters for accurate transcription with timestamps
        5. **Post-processing**: Decodes tokens to text with embedded timestamp markers

        **Generation Parameters:**

        - `max_new_tokens=128`: Limits output length for efficiency
        - `num_beams=4`: Beam search for better quality (increased from 1)
        - `return_timestamps=True`: Enables timestamp generation in output
        - `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`: Temperature fallback for robustness
        - `logprob_threshold=-1.0`: Log probability threshold for quality control
        - `compression_ratio_threshold=1.35`: Compression ratio filtering
        - Proper padding and EOS token handling

        **Timestamp Output Format:**

        - Returns text with embedded timestamps: `<|0.00|> text <|2.34|> more text`
        - Timestamps represent seconds from audio start
        - Used for temporal alignment and continuity across batches

        **Batch Processing:**

        - Processes multiple audio samples simultaneously for efficiency
        - Uses longest padding strategy for variable-length inputs
        - Maintains batch coherence through attention masks

        **Async Processing:**

        - Model inference runs in thread pool to avoid blocking event loop
        - Token decoding also runs async for consistent performance
        - Maintains responsiveness during heavy computation

        **Error Handling:**

        - Comprehensive error handling for preprocessing and generation stages
        - Detailed error logging with audio shape and type information
        - Graceful degradation with empty list return on errors

        .. note::
            This method assumes valid audio input and properly initialized model/processor.
            Input validation is performed by the calling method.

        .. warning::
            Large audio batches may consume significant GPU memory. Monitor memory usage
            with large batch sizes to avoid out-of-memory errors. Batch size affects both
            memory consumption and processing efficiency.
        """
        assert isinstance(self.processor, WhisperProcessor), "Processor must be an instance of WhisperProcessor"
        assert isinstance(self.speech_model, WhisperForConditionalGeneration), "Speechmodel must be instance of WhisperForConditionalGeneration"

        try:
            # TODO: Do this conversion earlier
            np_audio = [wave.numpy() for wave in audio]  # Ensure audio is in numpy format

            inputs = self.processor(
                np_audio,
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                sampling_rate=self.samplerate,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device, dtype=self.torch_dtype)
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            print(f"Audio shapes: {[a.shape if hasattr(a, 'shape') else len(a) for a in audio]}")
            print(f"Audio types: {[type(a) for a in audio]}")
            return []

        try:
            generated_ids = await asyncio.to_thread(
                self.speech_model.generate,
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                condition_on_prev_tokens=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                logprob_threshold=-1.0,
                compression_ratio_threshold=1.35,
                return_timestamps=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,  # type: ignore
                eos_token_id=self.processor.tokenizer.eos_token_id,  # type: ignore
            )
        except Exception as e:
            print(f"Error during model generation: {e}")
            return []

        results = await asyncio.to_thread(
            self.processor.batch_decode, generated_ids, skip_special_tokens=True, decode_with_timestamps=True, return_timestamps=True
        )

        return results

    async def __call__(self, audio_data: Tuple[list[Tensor], list[bool]]) -> None:
        """Process batch of audio data and publish transcription results via event system.

        This method serves as the main entry point for batch audio transcription, handling
        input validation, batch transcription processing, timestamp continuity, and result
        publication through the event bus. It's designed to be called by the transcription
        manager with batched audio data.

        :param audio_data: Tuple containing list of audio tensors and list of finality flags
        :type audio_data: Tuple[list[Tensor], list[bool]]

        **Parameters:**

        - `audio_data[0]`: List of audio tensors, each with shape (samples,) at configured sample rate
        - `audio_data[1]`: List of boolean flags indicating if each audio chunk is final

        **Processing Flow:**

        1. **Input Validation**: Ensures audio data is a valid list of PyTorch tensors
        2. **Batch Transcription**: Calls internal transcription method with audio tensor batch
        3. **Timestamp Processing**: Maintains timestamp continuity across batches using last_timestamp
        4. **Result Packaging**: Creates Transcription objects with processed text and timestamps
        5. **Event Publication**: Publishes TranscriptionCompleted events for all batch results

        **Event Publication:**

        - Event Type: `TranscriptionCompleted`
        - Payload: Transcription object with processed text and current timestamp
        - Finality Flags: Passed through from input parameters for each batch item

        **Timestamp Continuity:**

        - Processes embedded timestamps from Whisper output
        - Maintains continuity across batches by tracking last_timestamp
        - Ensures timestamps don't restart at 0.00 for each new batch

        **Error Handling:**

        - Validates tensor batch input with assertions
        - Async processing handles model inference errors
        - Event publication errors propagate to caller
        - Early return for empty batches

        .. note::
            This method processes multiple audio samples in a single batch for efficiency.
            It's typically called automatically by the transcription manager's inference
            loop rather than directly by user code.

        .. warning::
            The batch size affects GPU memory usage. Monitor memory consumption with
            large batches to avoid out-of-memory errors.
        """
        audio_batch, finals_batch = audio_data

        assert isinstance(audio_batch, list), "Audio data must be a list"
        assert all(isinstance(a, Tensor) for a in audio_batch), "All audio items must be PyTorch tensors"

        if not audio_data:
            return  # Nothing to process

        # Perform batch transcription
        transcriptions = await self._transcribe(audio_batch)
        transcriptions, self.last_timestamp = process_transcription_timestamps(transcriptions, self.last_timestamp)

        print(f"Transcription: {transcriptions}")

        # Create Transcription objects with timestamps
        current_time = datetime.now()
        transcriptions = [Transcription(text.strip(), current_time) for text in transcriptions]

        # Publish all transcription events
        await asyncio.gather(
            *[
                self.event_bus.publish(TranscriptionCompleted(transcription=transcription, is_final=is_final))
                for transcription, is_final in zip(transcriptions, finals_batch)
            ]
        )

    def _cleanup(self):
        """Properly cleanup model resources and free VRAM/memory.

        This method handles the complete cleanup of the Whisper model, including
        moving tensors to CPU, clearing CUDA cache, and performing garbage collection
        to ensure all GPU memory is properly released.

        **Cleanup Process:**

        1. **Model Cleanup**: Moves model to CPU and deletes references
        2. **Processor Cleanup**: Clears processor and tokenizer references
        3. **CUDA Cache**: Empties CUDA cache and synchronizes GPU
        4. **Garbage Collection**: Forces Python garbage collection
        5. **State Reset**: Resets internal state variables

        **Memory Management:**

        - Moves all model tensors from GPU to CPU before deletion
        - Clears CUDA cache to free GPU memory immediately
        - Forces garbage collection to release Python object memory
        - Synchronizes CUDA operations to ensure cleanup completion

        .. note::
            This method should be called before deleting the WhisperModel instance
            to ensure proper cleanup of GPU resources.

        .. warning::
            After calling this method, the model instance becomes unusable.
            Any subsequent transcription calls will fail.
        """
        print("Cleaning up WhisperModel resources...")

        # Clean up speech model
        if hasattr(self, "speech_model") and self.speech_model is not None:
            try:
                # Move model to CPU before deletion
                if hasattr(self.speech_model, "to"):
                    self.speech_model.to("cpu")

                # Delete model reference
                del self.speech_model
                self.speech_model = None
                print("Speech model cleaned up")
            except Exception as e:
                print(f"Error cleaning up speech model: {e}")

        # Clean up processor
        if hasattr(self, "processor") and self.processor is not None:
            try:
                # Clean up tokenizer if it exists
                if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                    del self.processor.tokenizer

                del self.processor
                self.processor = None
                print("Processor cleaned up")
            except Exception as e:
                print(f"Error cleaning up processor: {e}")

        # Reset internal state
        self.last_timestamp = 0.0
        self.model_id = ""
        self.model_size = ""
        self.available_model_sizes = []

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("CUDA cache cleared")
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}")

        print("WhisperModel cleanup completed")

    def cleanup(self):
        """Cleanup resources when the model is deleted."""
        try:
            self._cleanup()
        except Exception as e:
            print(f"Error during WhisperModel destructor cleanup: {e}")
            # Fallback cleanup
            try:
                if hasattr(self, "speech_model") and self.speech_model is not None:
                    if hasattr(self.speech_model, "to"):
                        self.speech_model.to("cpu")
                    del self.speech_model

                if hasattr(self, "processor") and self.processor is not None:
                    del self.processor

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during fallback cleanup: {e}")
