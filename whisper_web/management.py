import asyncio
from asyncio import Queue
from typing import List, Tuple

import torch

from whisper_web.events import (
    AudioChunkGenerated,
    AudioChunkNum,
    AudioChunkReceived,
    EventBus,
    TranscriptionCompleted,
    TranscriptionUpdated,
)
from whisper_web.types import AudioChunk, Transcription


class TranscriptionManager:
    """Manages the complete transcription pipeline from audio input to text output.

    The TranscriptionManager serves as the central coordinator for the real-time transcription
    process, handling audio chunk queuing, transcription state management, and result aggregation.
    It operates through an event-driven architecture, responding to audio events and publishing
    transcription updates.

    **Core Responsibilities:**

    - **Audio Queue Management**: Maintains a thread-safe queue of incoming audio chunks
    - **Transcription State Tracking**: Manages both current and historical transcription results
    - **Event Coordination**: Subscribes to audio events and publishes transcription updates
    - **Inference Loop**: Provides async interface for model inference execution
    - **Result Aggregation**: Combines partial and final transcriptions into complete text

    **Event Subscriptions:**

    - `AudioChunkReceived`: Queues audio data for processing
    - `TranscriptionCompleted`: Updates transcription state and publishes results

    **Event Publications:**

    - `TranscriptionUpdated`: Notifies subscribers of new transcription results

    :param event_bus: Event bus instance for inter-component communication
    :type event_bus: :class:`EventBus`

    :ivar transcriptions: List of completed transcription segments
    :type transcriptions: :class:`List[Transcription]`
    :ivar current_transcription: Current active transcription text
    :type current_transcription: :class:`str`
    :ivar audio_queue: Thread-safe queue for audio processing
    :type audio_queue: :class:`Queue[Tuple[torch.Tensor, bool]]`
    :ivar processed_chunks: Counter for processed audio chunks
    :type processed_chunks: :class:`int`
    :ivar num_chunks: Total expected number of chunks
    :type num_chunks: :class:`int`
    """

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
        """Process incoming audio chunk events and queue valid audio data.

        Validates incoming audio chunks and adds them to the processing queue
        if they contain valid audio data (non-empty tensors).

        :param event: Audio chunk received event containing audio data and finality flag
        :type event: :class:`AudioChunkReceived`
        """
        if event.chunk.data.numel() > 0:
            await self.audio_queue.put((event.chunk.data, event.is_final))

    async def _handle_transcription_completed(self, event: TranscriptionCompleted) -> None:
        """Process completed transcription events and update internal state.

        Updates the current transcription text, increments processed chunk counter,
        and manages the transcription history. Publishes transcription update events
        to notify other components of new results.

        :param event: Transcription completed event with result text and finality flag
        :type event: :class:`TranscriptionCompleted`

        **Behavior:**

        - Updates current transcription text from event
        - Appends to transcription history if final or first result
        - Publishes `TranscriptionUpdated` event with current and full text
        """
        self.current_transcription = event.transcription.text
        self.processed_chunks += 1

        if len(self.transcriptions) < 1 or event.is_final:
            self.transcriptions.append(event.transcription)

        # Publish transcription update event
        await self.event_bus.publish(TranscriptionUpdated(current_text=self.current_transcription, full_text=self.full_transcription))

    async def run_batched_inference(self, model, batch_size: int = 1, batch_timeout_s: float = 0.1) -> None:
        """Execute the main batched inference loop for continuous audio processing.

        Continuously retrieves audio data from the queue and processes it in batches
        for improved efficiency. Collects multiple audio samples up to the specified
        batch size before passing them to the model for transcription. Handles timeouts
        gracefully to prevent blocking and ensures timely processing even with partial batches.

        :param model: Async callable model that processes batched audio tensors
        :type model: Callable[[Tuple[list[torch.Tensor], list[bool]]], Awaitable[None]]
        :param batch_size: Maximum number of audio samples to collect per batch
        :type batch_size: int
        :param batch_timeout_s: Timeout in seconds for collecting additional batch items
        :type batch_timeout_s: float

        **Batching Strategy:**

        - Waits up to 1 second for the first audio sample to avoid busy waiting
        - Collects additional samples with shorter timeout (batch_timeout_s) for responsiveness
        - Processes partial batches when timeout is reached or batch_size is filled
        - Maintains separate lists for audio tensors and finality flags

        **Processing Flow:**

        1. **Initial Wait**: Blocks up to 1s for first audio sample
        2. **Batch Collection**: Gathers additional samples with short timeout
        3. **Batch Processing**: Sends complete batch to model as tuple of lists
        4. **Continuous Loop**: Repeats indefinitely for real-time processing

        **Timeout Behavior:**

        - Long timeout (1s) for first item prevents CPU spinning when queue is empty
        - Short timeout (batch_timeout_s) for additional items ensures low latency
        - Graceful handling of timeouts without error propagation
        - Processes partial batches immediately when collection timeout occurs

        **Batch Format:**

        - Model receives: `(list[torch.Tensor], list[bool])`
        - First list contains audio tensors for batch processing
        - Second list contains corresponding finality flags
        - Maintains order correspondence between tensors and flags

        **Performance Benefits:**

        - Reduced model invocation overhead through batching
        - Improved GPU utilization with parallel processing
        - Lower per-sample processing latency at scale
        - Configurable batch size for memory/latency trade-offs

        .. note::
            This method runs indefinitely and should be executed in a separate task
            or thread. The batch_size parameter affects both memory usage and processing
            efficiency - larger batches improve throughput but increase latency.

        .. warning::
            Large batch sizes may cause GPU memory issues. Monitor memory consumption
            and adjust batch_size accordingly. The method will block if the audio queue
            is consistently empty.
        """
        while True:
            batch_audio = []
            batch_finals = []

            # Get first item (with timeout)
            try:
                audio_data: tuple[torch.Tensor, bool] = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                batch_audio.append(audio_data[0])
                batch_finals.append(audio_data[1])
            except asyncio.TimeoutError:
                continue

            # Collect additional items with short timeout
            while len(batch_audio) < batch_size:
                try:
                    audio_data: tuple[torch.Tensor, bool] = await asyncio.wait_for(self.audio_queue.get(), timeout=batch_timeout_s)
                    batch_audio.append(audio_data[0])
                    batch_finals.append(audio_data[1])
                except asyncio.TimeoutError:
                    # Timeout reached, process current batch
                    break

            # Process the batch
            if batch_audio:
                batch_data = (batch_audio, batch_finals)
                await model(batch_data)

    @property
    def queue_size(self) -> int:
        """Get the current number of audio chunks waiting in the processing queue.

        :return: Number of queued audio chunks
        :rtype: :class:`int`
        """
        return self.audio_queue.qsize()

    @property
    def full_transcription(self) -> str:
        """Get the complete transcription text including all segments.

        Combines all completed transcription segments with the current active
        transcription to provide the full transcribed text.

        :return: Complete transcription text with all segments joined
        :rtype: :class:`str`
        """
        full_transcription = " ".join([transcription.text for transcription in self.transcriptions])
        return full_transcription + self.current_transcription

    @property
    def stats(self) -> dict:
        """Get comprehensive statistics about the transcription process.

        Provides key metrics for monitoring transcription performance and state,
        including queue utilization and processing progress.

        :return: Dictionary containing transcription statistics
        :rtype: :class`dict`

        **Returns:**

        - `queue_size`: Number of audio chunks in processing queue
        - `processed_chunks`: Total number of processed audio chunks
        - `num_transcriptions`: Number of completed transcription segments
        """
        return {
            "queue_size": self.queue_size,
            "processed_chunks": self.processed_chunks,
            "num_transcriptions": len(self.transcriptions),
        }

    def clear_audio_queue(self) -> None:
        """Remove all pending audio chunks from the processing queue.

        Empties the audio queue by discarding all queued audio data. Useful
        for resetting the transcription state or handling session cleanup.

        .. warning::
            This operation discards all pending audio data and cannot be undone.
            Use with caution during active transcription sessions.
        """
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()


class AudioManager:
    """Manages audio chunk lifecycle and distribution in the transcription pipeline.

    The AudioManager handles the flow of raw audio chunks from generation to consumption,
    providing a buffered interface between audio input sources and transcription models.
    It operates through event-driven architecture to ensure loose coupling and scalability.

    **Core Responsibilities:**

    - **Audio Chunk Buffering**: Maintains a thread-safe queue of processed audio chunks
    - **Event-Driven Processing**: Subscribes to audio generation events and manages chunk flow
    - **Chunk Metadata Tracking**: Monitors chunk counts and processing statistics
    - **Async Chunk Distribution**: Provides non-blocking access to queued audio data
    - **Error Handling**: Gracefully handles audio processing errors and timeouts

    **Event Subscriptions:**

    - `AudioChunkGenerated`: Receives and queues newly generated audio chunks
    - `AudioChunkNum`: Updates total expected chunk count for progress tracking

    **Key Features:**

    - **Thread-Safe Operations**: All queue operations are async-safe for concurrent access
    - **Timeout Handling**: Non-blocking audio retrieval with configurable timeouts
    - **Progress Tracking**: Monitors processing progress against expected chunk counts
    - **Error Resilience**: Continues operation despite individual chunk processing errors

    :param event_bus: Event bus instance for inter-component communication
    :type event_bus: :class:`EventBus`

    :ivar audio_chunk_queue: Thread-safe queue for audio chunk storage
    :type audio_chunk_queue: :class:`Queue[Tuple[AudioChunk, bool]]`
    :ivar processed_chunks: Counter for successfully processed audio chunks
    :type processed_chunks: :class:`int`
    :ivar num_chunks: Total expected number of chunks for current session
    :type num_chunks: :class:`int`
    """

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
        """Process generated audio chunk events and queue valid audio data.

        Validates and queues incoming audio chunks, ensuring only chunks with valid
        audio data are added to the processing queue. Includes error handling to
        maintain system stability during audio processing issues.

        :param event: Audio chunk generated event containing audio data and finality flag
        :type event: :class:`AudioChunkGenerated`

        **Behavior:**

        - Validates audio chunk contains non-empty tensor data
        - Increments processed chunk counter for progress tracking
        - Queues valid audio chunks with finality flag for downstream processing
        - Logs errors without interrupting processing flow
        """
        try:
            audio_chunk = event.chunk
            is_final = event.is_final
            self.processed_chunks += 1

            if audio_chunk.data.numel() > 0:
                await self.audio_chunk_queue.put((audio_chunk, is_final))
        except Exception as e:
            print(f"Error handling audio chunk: {e}")

    async def _handle_num_chunks_updated(self, event: AudioChunkNum) -> None:
        """Update the total expected chunk count for progress tracking.

        Receives chunk count updates to maintain accurate progress tracking
        during file processing or streaming sessions.

        :param event: Audio chunk number event containing total expected chunks
        :type event: :class:`AudioChunkNum`
        """
        self.num_chunks = event.num_chunks

    async def get_next_audio_chunk(self) -> Tuple[AudioChunk, bool] | None:
        """Retrieve the next available audio chunk from the processing queue.

        Provides non-blocking access to queued audio chunks with timeout handling.
        Returns None when no audio is available within the timeout period, allowing
        callers to handle empty queue conditions gracefully.

        :return: Tuple of (AudioChunk, is_final) if available, None if timeout/error
        :rtype: :class:`Tuple[AudioChunk, bool]` | :class:`None`

        **Behavior:**

        - Waits up to 1 second for audio chunk availability
        - Returns tuple containing audio chunk and finality flag
        - Returns None on timeout or processing errors
        - Logs errors without raising exceptions
        """
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
        """Get the current number of audio chunks waiting in the processing queue.

        :return: Number of queued audio chunks
        :rtype: :class:`int`
        """
        return self.audio_chunk_queue.qsize()

    @property
    def stats(self) -> dict:
        """Get comprehensive statistics about audio chunk processing.

        Provides key metrics for monitoring audio processing performance,
        including queue utilization and chunk processing progress.

        :return: Dictionary containing audio processing statistics
        :rtype: :class:`dict`

        **Returns:**

        - `queue_size`: Number of audio chunks currently in processing queue
        - `processed_chunks`: Total number of audio chunks processed
        """
        return {
            "queue_size": self.queue_size,
            "processed_chunks": self.processed_chunks,
        }

    def clear_audio_queue(self) -> None:
        """Remove all pending audio chunks from the processing queue.

        Empties the audio chunk queue by discarding all queued audio data.
        Useful for session cleanup or resetting audio processing state.

        .. warning::
            This operation discards all pending audio chunks and cannot be undone.
            Use with caution during active audio processing sessions.
        """
        while not self.audio_chunk_queue.empty():
            self.audio_chunk_queue.get_nowait()
