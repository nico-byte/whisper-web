from typing import List, Tuple
from asyncio import Queue
import torch
import asyncio

from whisper_web.events import (
    EventBus,
    AudioChunkReceived,
    AudioChunkGenerated,
    AudioChunkNum,
    TranscriptionCompleted,
    TranscriptionUpdated,
)
from whisper_web.types import Transcription, AudioChunk


class TranscriptionManager:
    """Event-driven transcription manager."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # State
        self.transcriptions: List[Transcription] = []
        self.current_transcription: str = ""
        self.audio_queue: Queue[Tuple[torch.Tensor, bool]] = Queue(maxsize=128)
        self.processed_chunks: int = 0
        self.num_chunks: int = 0

        # Subscribe to events
        self.event_bus.subscribe(AudioChunkReceived, self._handle_audio_chunk)  # type: ignore
        self.event_bus.subscribe(TranscriptionCompleted, self._handle_transcription_completed)  # type: ignore

    async def _handle_audio_chunk(self, event: AudioChunkReceived) -> None:
        """Handle incoming audio chunks."""
        if event.chunk.data.numel() > 0:
            await self.audio_queue.put((event.chunk.data, event.is_final))

    async def _handle_transcription_completed(self, event: TranscriptionCompleted) -> None:
        """Handle completed transcriptions."""
        self.current_transcription = event.transcription.text
        self.processed_chunks += 1

        if len(self.transcriptions) < 1 or event.is_final:
            self.transcriptions.append(event.transcription)

        # Publish transcription update event
        await self.event_bus.publish(TranscriptionUpdated(current_text=self.current_transcription, full_text=self.full_transcription))

    async def run_inference(self, model) -> None:
        """Main inference loop - no direct dependency on TranscriptionManager."""
        while True:
            try:
                audio_data: tuple[torch.Tensor, bool] = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            await model(audio_data)

    @property
    def queue_size(self) -> int:
        return self.audio_queue.qsize()

    @property
    def full_transcription(self) -> str:
        full_transcription = " ".join([transcription.text for transcription in self.transcriptions])
        return full_transcription + self.current_transcription

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self.queue_size,
            "processed_chunks": self.processed_chunks,
            "num_transcriptions": len(self.transcriptions),
        }

    def clear_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()


class AudioManager:
    """Event-driven audio chunk manager."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # State
        self.audio_chunk_queue: Queue[Tuple[AudioChunk, bool]] = Queue(maxsize=128)
        self.processed_chunks: int = 0
        self.num_chunks: int = 0

        # Subscribe to events
        self.event_bus.subscribe(AudioChunkGenerated, self._handle_generated_audio_chunk)  # type: ignore
        self.event_bus.subscribe(AudioChunkNum, self._handle_num_chunks_updated)  # type: ignore

    async def _handle_generated_audio_chunk(self, event: AudioChunkGenerated) -> None:
        """Handle incoming audio chunks."""
        try:
            audio_chunk = event.chunk
            is_final = event.is_final
            self.processed_chunks += 1

            if audio_chunk.data.numel() > 0:
                await self.audio_chunk_queue.put((audio_chunk, is_final))
        except Exception as e:
            print(f"Error handling audio chunk: {e}")

    async def _handle_num_chunks_updated(self, event: AudioChunkNum) -> None:
        """Handle completed transcriptions."""
        self.num_chunks = event.num_chunks

    async def get_next_audio_chunk(self) -> Tuple[AudioChunk, bool] | None:
        """Retrieve the next available audio chunk from the queue."""
        try:
            audio_chunk, is_final = await asyncio.wait_for(self.audio_chunk_queue.get(), timeout=1.0)
            return audio_chunk, is_final
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"Error getting audio chunk: {e}")
            return None

    @property
    def queue_size(self) -> int:
        return self.audio_chunk_queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self.queue_size,
            "processed_chunks": self.processed_chunks,
        }

    def clear_audio_queue(self) -> None:
        while not self.audio_chunk_queue.empty():
            self.audio_chunk_queue.get_nowait()
