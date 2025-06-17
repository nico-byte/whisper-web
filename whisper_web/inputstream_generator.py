import numpy as np
import asyncio
import sys
import torch

from pydantic import BaseModel, Field
from typing import AsyncGenerator
import soundfile as sf
import resampy
from datetime import datetime

from whisper_web.events import EventBus, AudioChunkGenerated, AudioChunkNum
from whisper_web.types import AudioChunk

try:
    import sounddevice as sd
except OSError as e:
    print(e)
    print("If `GLIBCXX_x.x.x' not found, try installing it with: conda install -c conda-forge libstdcxx-ng=12")
    sys.exit()


class GeneratorConfig(BaseModel):
    """Configuration model for controlling audio input generation behavior.

    This configuration class is used to define how audio should be captured,
    processed, and segmented before being sent to a speech recognition system.
    """

    samplerate: int = Field(
        default=16000,
        description="The specified samplerate of the audio data.",
    )
    blocksize: int = Field(
        default=6000,
        description="The size of each individual audio chunk.",
    )
    max_length_s: int = Field(
        default=25,
        description="The maximum length of the audio data.",
    )
    adjustment_time: int = Field(
        default=5,
        description="The adjustment_time for setting the silence threshold.",
    )
    min_chunks: int = Field(
        default=3,
        description="The minimum number of chunks to be generated, before feeding it into the asr model.",
    )
    phrase_delta: float = Field(
        default=1.0,
        description="The expected pause between two phrases in seconds.",
    )
    continuous: bool = Field(
        default=True,
        description="Whether to generate audio data conituously or not.",
    )
    from_file: str = Field(
        default="",
        description="The path to the audio file to be used for inference.",
    )


class InputStreamGenerator:
    """Handles real-time or file-based audio input for speech processing and transcription.

    This class manages the lifecycle of audio input—from capturing or loading audio data to detecting
    speech segments and dispatching them for transcription. It supports both live microphone streams
    and pre-recorded audio files, and includes configurable voice activity detection (VAD) heuristics
    and silence detection.

    **Core Features:**

    - **Real-Time Audio Input**: Captures audio using a microphone input stream.
    - **File-Based Input**: Reads and processes audio from a file if specified.
    - **Silence Threshold Calibration**: Dynamically computes the silence threshold based on environmental noise.
    - **Voice Activity Detection (VAD)**: Supports heuristic-based VAD.
    - **Phrase Segmentation**: Aggregates audio buffers into speech phrases based on silence duration and loudness.
    - **Asynchronous Processing**: Fully asynchronous design suitable for non-blocking audio pipelines.

    :param generator_config: Configuration object with audio processing settings
    :type generator_config: :class:`GeneratorConfig`
    :param event_bus: Instance of the EventBus to handle events
    :type event_bus: :class:`EventBus`

    :ivar samplerate: Sample rate for audio processing
    :type samplerate: :class:`int`
    :ivar blocksize: Size of each audio block
    :type blocksize: :class:`int`
    :ivar adjustment_time: Time in seconds for adjusting silence threshold
    :type adjustment_time: :class:`int`
    :ivar min_chunks: Minimum number of chunks to process
    :type min_chunks: :class:`int`
    :ivar continuous: Flag for continuous processing
    :type continuous: :class:`bool`
    :ivar event_bus: Event bus for handling events
    :type event_bus: :class:`EventBus`
    :ivar global_ndarray: Global buffer for audio data
    :type global_ndarray: :class:`np.ndarray`
    :ivar phrase_delta_blocks: Max number of blocks for inbetween phrases
    :type phrase_delta_blocks: :class:`int`
    :ivar silence_threshold: Threshold for silence detection
    :type silence_threshold: :class:`float`
    :ivar max_blocksize: Maximum size of audio block
    :ivar max_blocksize: Maximum size of audio block in samples
    :ivar max_chunks: Maximum number of chunks
    :type max_chunks: :class:`int`
    :ivar from_file: Path to the audio file if specified
    :type from_file: :class:`str`

    .. note::
        Instantiate this class with a `GeneratorConfig` and `EventBus`, then call `process_audio()`
        to start listening or processing input.
    """

    def __init__(self, generator_config: GeneratorConfig, event_bus: EventBus):
        self.samplerate = generator_config.samplerate
        self.blocksize = generator_config.blocksize
        self.adjustment_time = generator_config.adjustment_time
        self.min_chunks = generator_config.min_chunks
        self.continuous = generator_config.continuous

        self.event_bus = event_bus

        self.global_ndarray: np.ndarray = np.array([])

        self.phrase_delta_blocks: int = int((self.samplerate // self.blocksize) * generator_config.phrase_delta)
        self.silence_threshold = -1

        self.max_blocksize = generator_config.max_length_s * self.samplerate
        self.max_chunks = generator_config.max_length_s * self.samplerate / self.blocksize

        self.from_file = generator_config.from_file

    async def process_audio(self) -> None:
        """Entry point for audio processing based on the selected VAD configuration.

        Determines if the input is from a file or a live stream, sets up the silence threshold,
        and processes audio input accordingly.

        .. note::
            If `from_file` is set, it processes the audio from the specified file.
            If `from_file` is not set, it sets the silence threshold and processes audio using heuristics.
        """
        if self.from_file:
            await self.generate_from_file(self.from_file)
        else:
            await self.set_silence_threshold()
            await self.process_with_heuristic()

    async def generate(self) -> AsyncGenerator:
        """Asynchronously generates audio chunks for processing from a live input stream.

        This method acts as a unified audio generator, yielding blocks of audio data for downstream processing.

        **Behavior:**

        - Opens an audio input stream using `sounddevice.InputStream`.
        - Captures audio in blocks of `self.blocksize`, configured for mono 16-bit input.
        - Uses a thread-safe callback to push incoming audio data into an `asyncio.Queue`.
        - Yields `(in_data, status)` tuples from the queue as they become available.

        :return: A :class:`tuple` containing the raw audio block and its status.
        :rtype: :class:`Iterator[Tuple[np.ndarray, CallbackFlags]]`
        """
        q_in = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def callback(in_data, _, __, state):
            loop.call_soon_threadsafe(q_in.put_nowait, (in_data.copy(), state))

        # Default stream args
        stream_args = dict(  # type: ignore
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",  # type: ignore
            blocksize=self.blocksize,
            callback=callback,  # type: ignore
        )

        stream = sd.InputStream(
            **stream_args,
        )
        with stream:
            while True:
                indata, status = await q_in.get()
                yield indata, status

    async def generate_from_file(self, file_path: str) -> None:
        """Processes audio data from a file and simulates streaming for transcription.

        This method reads audio from the given file path, optionally resamples and converts it to mono,
        and then splits the audio into chunks that simulate live microphone input. Each chunk is passed
        to the transcription manager after waiting for the current transcription to complete.

        **Behavior:**

        - Reads audio from the specified file using `soundfile`.
        - Supports multi-channel audio, which is converted to mono by selecting the first channel.
        - If the audio files sample rate differs from the expected rate (`self.samplerate`),
          the data is resampled to match.
        - Audio is divided into blocks of `self.max_blocksize` samples.
        - The final chunk is zero-padded if it is shorter than the expected size.
        - Each chunk is set as the current buffer and dispatched for transcription using `_send_audio()`.
        - Waits for the transcription manager's signal (`transcription_status.wait()`) before continuing.
        - Logs the total time taken to process the file.

        :param file_path: Path to the audio file to be processed
        :type file_path: :class:`str`
        """
        data, samplerate = sf.read(file_path, dtype="float32")

        if np.ndim(data) > 1:
            data = np.mean(data, axis=1)  # type: ignore

        # Resample if needed
        if samplerate != self.samplerate:
            data = resampy.resample(data.astype(np.float32), samplerate, self.samplerate)
            data = (data * 32767).astype(np.float32)

        # Mono conversion (just take first channel)
        if data.ndim > 1:
            data = data[:, 0]

        num_chunks = int(np.ceil(len(data) / self.max_blocksize))
        print(f"Processing file {file_path} with {num_chunks} chunks of size {self.max_blocksize}")
        await self.event_bus.publish(AudioChunkNum(num_chunks=num_chunks))

        # Yield chunks of blocksize
        for i in range(0, len(data), self.max_blocksize):
            chunk = data[i : i + self.max_blocksize]  # type: ignore

            # Pad the last chunk if it's too short
            if len(chunk) < self.max_blocksize:
                chunk = np.pad(chunk, (0, self.max_blocksize - len(chunk)), "constant", constant_values=0)  # type: ignore

            self.global_ndarray = chunk
            await self.send_audio(True)

    async def process_with_heuristic(self) -> None:
        """Continuously processes audio input, detects significant speech segments, and dispatches them for transcription.

        This method operates in an asynchronous loop, consuming real-time audio buffers from `generate()`, aggregating
        meaningful speech segments while filtering out silence or noise based on a calculated silence threshold.

        **Behavior:**

        - Buffers with low average volume (below `self.silence_threshold`) are considered silent.
        - Incoming buffers are accumulated in `self.global_ndarray`.
        - If the accumulated audio exceeds `self.max_chunks`, it is dispatched for transcription.
        - If the size of `self.global_ndarray` is > 0 and the average volume is below the silence threshold,
          empty blocks is incremented by one and if it exceeds `self.phrase_delta_blocks`, the buffer is dispatched.
        - If a buffer does not start or end or is silent the audio is dispatched.
        - In continuous mode (`self.continuous` = True), the method loops indefinitely to process ongoing audio.
        - Otherwise, it exits after the first valid speech phrase is processed.
        """
        empty_blocks = 0
        async for indata, _ in self.generate():
            indata_flattened = abs(indata.flatten())
            silence = np.mean(indata_flattened) <= self.silence_threshold
            ending_silence = np.mean(indata_flattened[-500:-1]) <= self.silence_threshold  # type: ignore
            starting_silence = np.mean(indata_flattened[:500]) <= self.silence_threshold  # type: ignore

            print(f"Silence: {silence}, Starting Silence: {starting_silence}, Ending Silence: {ending_silence}")

            # Process the global ndarray if the max chunks are met
            if self.global_ndarray.size / self.blocksize == self.max_chunks:
                await self.send_audio(True)
                self.global_ndarray = np.array([])  # reset
                empty_blocks = 0

            # continue if start/ending/whole of buffer is not silent
            if not starting_silence or not ending_silence or not silence:
                # concatenate buffers
                if self.global_ndarray.size > 0:
                    self.global_ndarray = np.concatenate((self.global_ndarray, indata), dtype="int16")
                else:
                    self.global_ndarray = indata

            # discard buffers that conain mostly silence
            if self.global_ndarray.size > 0 and silence:
                empty_blocks += 1
                if empty_blocks >= self.phrase_delta_blocks:
                    empty_blocks = 0
                    await self.send_audio(True)
                    self.global_ndarray = np.array([])  # reset
                    if not self.continuous:
                        return
                continue

            empty_blocks = 0

            await self.send_audio() if self.global_ndarray.size / self.blocksize >= self.min_chunks else None

    async def send_audio(self, is_final: bool = False) -> None:
        """Dispatches the collected audio buffer for transcription after normalization.

        This method converts the internal audio buffer (`self.global_ndarray`) from
        16-bit PCM format to a normalized float32 waveform in the range [-1.0, 1.0].
        It then creates an `AudioChunk` instance with the normalized data and publishes
        it as an `AudioChunkGenerated` event to the event bus.

        :param is_final: Indicates if the audio chunk is complete and ready for final processing.
        :type is_final: :class:`bool`
        """
        # Normalize int16 to float32 waveform in range [-1.0, 1.0]
        waveform = torch.from_numpy(self.global_ndarray.flatten().astype("float32") / 32768.0)
        audio_chunk = AudioChunk(data=waveform, timestamp=datetime.now())
        # Publish the audio chunk event
        await self.event_bus.publish(AudioChunkGenerated(audio_chunk, is_final))

    async def set_silence_threshold(self) -> None:
        """Dynamically determines and sets the silence threshold based on initial audio input.

        This method analyzes the average loudness of incoming audio blocks during a short calibration phase
        to determine an appropriate silence threshold. The threshold helps distinguish between background
        noise and meaningful speech during audio processing.

        **Behavior:**

        - Processes audio blocks for a predefined duration (`_adjustment_time` in seconds).
        - For each block, computes the mean absolute loudness and stores it.
        - After enough blocks are collected, calculates the average loudness across all blocks.
        - Sets `self.silence_threshold` to this value, treating it as the baseline for silence.

        .. note::
            This method is skipped if audio is being read from a file (`self.from_file` is set).
            Intended to run once before audio processing begins, helping tailor silence detection to the environment.
        """

        blocks_processed: int = 0
        loudness_values: list = []

        async for indata, _ in self.generate():
            blocks_processed += 1
            indata_flattened: np.ndarray = abs(indata.flatten())

            # Compute loudness over first few seconds to adjust silence threshold
            loudness_values.append(np.mean(indata_flattened))  # type: ignore

            # Stop recording after ADJUSTMENT_TIME seconds
            if blocks_processed >= self.adjustment_time * (self.samplerate / self.blocksize):
                self.silence_threshold = float(np.mean(loudness_values))  # type: ignore
                print(f"Silence threshold set to {self.silence_threshold}")
                break
