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
    """Event-driven Whisper model with no direct dependencies."""

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
        """
        Loads the Whisper ASR model based on the specified model size or model ID.
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
        """Core transcription logic separated from event handling."""
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
        """Transcribe audio and publish result as event."""
        audio, is_final = audio_data
        assert isinstance(audio, Tensor), "Audio data must be a torch.Tensor"

        transcript_text = await self._transcribe(audio)
        transcription = Transcription(transcript_text, datetime.now().date())

        await self.event_bus.publish(TranscriptionCompleted(transcription=transcription, is_final=is_final))
