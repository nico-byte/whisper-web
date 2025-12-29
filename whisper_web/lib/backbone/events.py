import asyncio
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from whisper_web.types import AudioChunk, Transcription


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


@dataclass
class DiarizationRequest(Event):
    """Request diarization for provided audio + transcript."""

    audio_waveform: Any  # np.ndarray or torch.Tensor
    start_time: int
    transcript: str
    language: str
    session_id: Optional[str] = None


@dataclass
class DiarizationCompleted(Event):
    """Published after diarization finished."""

    result: Dict[str, Any]
    session_id: Optional[str] = None


class EventBus:
    """Asynchronous event bus implementation for decoupled component communication.

    The EventBus provides a publish-subscribe pattern that enables loose coupling between
    different components of the whisper-web transcription system. Components can subscribe
    to specific event types and publish events without direct knowledge of other components.

    **Key Features:**

    - **Type-Safe Subscriptions**: Events are registered by their concrete type
    - **Async/Sync Handler Support**: Automatically detects and handles both coroutine and regular functions
    - **Multiple Subscribers**: Multiple handlers can subscribe to the same event type
    - **Decoupled Architecture**: Publishers don't need to know about subscribers

    :ivar _subscribers: Internal mapping of event types to their handler lists
    :type _subscribers: :class:`Dict[type, List[Callable]]`
    """

    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable[[Any], None]) -> None:
        """Register a handler function to receive events of a specific type.

        When an event of the specified type is published, the handler will be called
        with the event instance as its argument. Handlers can be either synchronous
        functions or async coroutines.

        :param event_type: The class :class:`type` of events this handler should receive
        :type event_type: :class:`type`
        :param handler: Function or coroutine to call when events are published
        :type handler: :class:`Callable[[Any], None]`
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered subscribers of its type.

        This method delivers the event to all handlers that have subscribed to the
        event's specific type. Both synchronous and asynchronous handlers are supported
        and will be called appropriately.

        :param event: The event instance to publish to subscribers
        :type event: :class:`Event`
        """
        event_type = type(event)
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
