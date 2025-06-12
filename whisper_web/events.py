from abc import ABC
from typing import Any, Dict, List, Callable
import asyncio

from dataclasses import dataclass

from whisper_web.types import Transcription, AudioChunk


# Event System
class Event(ABC):
    """Base class for all events."""

    pass


@dataclass
class AudioChunkReceived(Event):
    chunk: AudioChunk
    is_final: bool


@dataclass
class AudioChunkGenerated(Event):
    chunk: AudioChunk
    is_final: bool


@dataclass
class AudioChunkNum(Event):
    num_chunks: int


@dataclass
class TranscriptionCompleted(Event):
    transcription: Transcription
    is_final: bool


@dataclass
class TranscriptionUpdated(Event):
    current_text: str
    full_text: str


@dataclass
class DownloadModel(Event):
    model_url: str
    is_finished: bool = False


class EventBus:
    """Simple event bus for decoupled communication."""

    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
