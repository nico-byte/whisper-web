from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
import threading
import asyncio
import torchaudio
import io
import uuid
from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field

from whisper_web.whisper_model import ModelConfig, WhisperModel
from whisper_web.management import TranscriptionManager
from whisper_web.events import EventBus, AudioChunkReceived, DownloadModel
from whisper_web.types import AudioChunk
from whisper_web.utils import get_installed_models


# Constants
SESSION_NOT_FOUND_MSG = "Session not found"


# API Schemas
class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session."""

    model_configuration: Optional[ModelConfig] = Field(None, description="Model configuration for the session")
    session_id: Optional[str] = Field(None, description="Optional custom session ID")


class SessionResponse(BaseModel):
    """Response schema for session creation."""

    session_id: str = Field(..., description="Unique session identifier")
    model_configuration: Dict = Field(..., description="Model configuration used for the session")
    message: Optional[str] = Field(None, description="Optional message (e.g., for existing sessions)")


class SessionInfo(BaseModel):
    """Schema for session information in list responses."""

    session_id: str = Field(..., description="Unique session identifier")
    model_configuration: Dict = Field(..., description="Model configuration")
    inference_running: bool = Field(..., description="Whether inference is currently running")
    transcription_count: int = Field(..., description="Number of completed transcriptions")
    current_transcription: str = Field(..., description="Current transcription text")
    audio_queue_size: int = Field(..., description="Number of audio chunks in queue")


class SessionListResponse(BaseModel):
    """Response schema for listing sessions."""

    sessions: List[SessionInfo] = Field(..., description="List of active sessions")
    total_sessions: int = Field(..., description="Total number of active sessions")


class SessionStatusResponse(BaseModel):
    """Response schema for session status."""

    session_id: str = Field(..., description="Session identifier")
    is_downloading: bool = Field(..., description="Whether model is currently downloading")
    inference_running: bool = Field(..., description="Whether inference is currently running")
    audio_queue_size: int = Field(..., description="Number of audio chunks in queue")
    audio_queue_processed: int = Field(..., description="Number of audio chunks processed")
    transcription_count: int = Field(..., description="Number of completed transcriptions")
    model_configuration: Dict = Field(..., description="Model configuration")


class TranscriptionsResponse(BaseModel):
    """Response schema for getting all transcriptions."""

    session_id: str = Field(..., description="Session identifier")
    transcriptions: List[str] = Field(..., description="List of all transcriptions")


class InstalledModelsResponse(BaseModel):
    """Response schema for getting all installed models."""

    installed_models: List[str] = Field(..., description="List of all installed models")


class CurrentTranscriptionResponse(BaseModel):
    """Response schema for current transcription."""

    session_id: str = Field(..., description="Session identifier")
    current_transcription: str = Field(..., description="Current transcription text")


class FinalTranscriptionResponse(BaseModel):
    """Response schema for final transcription."""

    session_id: str = Field(..., description="Session identifier")
    final_transcription: str = Field(..., description="Final transcription text")


class QueueSizeResponse(BaseModel):
    """Response schema for audio queue size."""

    session_id: str = Field(..., description="Session identifier")
    audio_queue_size: int = Field(..., description="Number of audio chunks in queue")


class QueueProcessedResponse(BaseModel):
    """Response schema for processed queue items."""

    session_id: str = Field(..., description="Session identifier")
    audio_queue_processed: int = Field(..., description="Number of audio chunks processed")


class MessageResponse(BaseModel):
    """Generic response schema for operations with messages."""

    message: str = Field(..., description="Operation result message")
    session_id: Optional[str] = Field(None, description="Session identifier")


class SessionOperationResponse(BaseModel):
    """Response schema for session operations like restart."""

    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="Operation result message")
    inference_running: Optional[bool] = Field(None, description="Whether inference is running after operation")


class ClientSession:
    """Represents an isolated client transcription session with dedicated resources.

    Each ClientSession encapsulates a complete transcription pipeline for a single client,
    including its own event bus, transcription manager, and Whisper model instance. This
    design enables concurrent multi-client support with isolated state and configurations.

    **Session Components:**

    - **Event Bus**: Dedicated event system for inter-component communication
    - **Transcription Manager**: Handles audio queuing and transcription state
    - **Whisper Model**: Configured ASR model instance for speech recognition
    - **Inference Task**: Async task managing the transcription processing loop
    - **Download Tracking**: Monitors model download progress

    **Key Features:**

    - **Isolation**: Each session operates independently with separate state
    - **Lifecycle Management**: Handles session startup, operation, and cleanup
    - **Event-Driven Architecture**: Uses publish-subscribe pattern for loose coupling
    - **Async Processing**: Non-blocking inference execution with proper task management
    - **Model Download Handling**: Tracks and responds to model download events

    :param session_id: Unique identifier for this client session
    :type session_id: :class:`str`
    :param model_config: Configuration for the Whisper model used in this session
    :type model_config: :class:`ModelConfig`

    :ivar session_id: Unique session identifier
    :type session_id: :class:`str`
    :ivar model_config: Whisper model configuration
    :type model_config: :class:`ModelConfig`
    :ivar event_bus: Session-specific event bus for component communication
    :type event_bus: :class:`EventBus`
    :ivar manager: Transcription manager handling audio processing
    :type manager: :class:`TranscriptionManager`
    :ivar model: Whisper model instance for speech recognition
    :type model: :class:`WhisperModel`
    :ivar inference_task: Async task running the inference loop
    :type inference_task: :class:`Optional[asyncio.Task]`
    :ivar is_downloading: Flag indicating if model is currently downloading
    :type is_downloading: :class:`bool`
    """

    def __init__(self, session_id: str, model_config: ModelConfig):
        self.session_id = session_id
        self.model_config = model_config
        self.event_bus = EventBus()
        self.manager = TranscriptionManager(self.event_bus)
        self.model = WhisperModel(model_config, self.event_bus)
        self.inference_task: Optional[asyncio.Task] = None
        self.is_downloading: bool = False

        self.event_bus.subscribe(DownloadModel, self.handle_model_download)  # type: ignore

    async def start_inference(self):
        """Start the transcription inference task for this session.

        Creates and starts an async task that runs the main inference loop,
        processing audio chunks through the Whisper model. If an inference
        task is already running, this method has no effect.

        **Behavior:**

        - Creates new inference task if none exists or previous task completed
        - Task runs indefinitely until manually stopped or session cleanup
        - Uses the session's transcription manager and model for processing
        - Handles audio chunks from the session's event bus

        .. note::
            The inference task runs in the background and must be explicitly
            stopped using `stop_inference()` for proper cleanup.
        """
        if self.inference_task is None or self.inference_task.done():
            self.inference_task = asyncio.create_task(
                self.manager.run_batched_inference(self.model, self.model_config.batch_size, self.model_config.batch_timeout_s)
            )

    def _cleanup_model(self):
        """Properly cleanup model and free VRAM."""
        if hasattr(self, "model") and self.model is not None:
            try:
                # Use synchronous cleanup method
                self.model.cleanup()
            except Exception as e:
                print(f"Error during model cleanup: {e}")

            self.model = None

            # Additional cleanup
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"Session {self.session_id} model cleanup completed")

    async def stop_inference(self):
        """Stop the transcription inference task for this session.

        Gracefully cancels the running inference task and waits for proper
        cleanup. Handles cancellation exceptions to ensure clean shutdown
        without propagating cancellation errors.

        **Behavior:**

        - Cancels the inference task if currently running
        - Waits for task cancellation to complete
        - Suppresses CancelledError exceptions from task cleanup
        - Safe to call multiple times or when no task is running

        .. note::
            This method should be called during session cleanup to prevent
            resource leaks and ensure proper task termination.
        """
        if self.inference_task and not self.inference_task.done():
            self.inference_task.cancel()
            try:
                await self.inference_task
            except asyncio.CancelledError:
                pass

        # Clean up model resources
        self._cleanup_model()

    async def handle_model_download(self, event: DownloadModel):
        """Handle model download progress events for this session.

        Updates the session's download status based on model download events,
        allowing the session to track when models are being loaded and when
        they become available for inference.

        :param event: Model download event containing URL and completion status
        :type event: :class:`DownloadModel`

        **Behavior:**

        - Sets `is_downloading` to True when download starts
        - Sets `is_downloading` to False when download completes
        - Logs download status changes for monitoring
        """
        self.is_downloading = not event.is_finished
        print(f"Model download {'started' if not event.is_finished else 'finished'} for session {self.session_id}")

    async def cleanup(self):
        """Cleanup resources when the session is deleted."""
        if hasattr(self, "inference_task") and self.inference_task:
            await self.stop_inference()

        print(f"Session {getattr(self, 'session_id', 'unknown')} resources cleaned up")


class TranscriptionServer:
    """A real-time speech transcription API server built with FastAPI.

    This service provides a full audio transcription pipeline via HTTP and WebSocket endpoints, including:

    - Audio stream ingestion per client.
    - Real-time automatic speech recognition (ASR) using configurable Whisper models.
    - Voice Activity Detection (VAD) to detect speech regions.
    - Status monitoring of transcription, generation, and voice activity per client.
    - Retrieval and posting of transcription data per client.
    - Multi-client support with isolated sessions.

    :param default_model_config: Default configuration for the ASR model (e.g., Whisper), defaults to :class:`ModelConfig()`
    :type default_model_config: :class:`ModelConfig`, optional
    :param host: Hostname for the FastAPI server, defaults to "localhost"
    :type host: :class:`str`, optional
    :param port: Port for the FastAPI server, defaults to 8000
    :type port: :class:`int`, optional

    .. note::
        Each client connection creates its own isolated session with a dedicated TranscriptionManager and WhisperModel.
        This allows multiple clients to use different model configurations and have separate transcription states.

        The server exposes endpoints for creating sessions, managing transcriptions per session,
        and retrieving processing status per client.
    """

    def __init__(
        self,
        default_model_config: Optional[ModelConfig] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self._default_model_config = default_model_config or ModelConfig()
        self.host = host
        self.port = port
        self.app = FastAPI()

        # Dictionary to store client sessions
        self.client_sessions: Dict[str, ClientSession] = {}

        self._setup_ws_route()
        self._setup_api_routes()

    def get_or_create_session(self, session_id: Optional[str] = None, model_config: Optional[ModelConfig] = None) -> ClientSession:
        """Retrieve an existing session or create a new one with specified configuration.

        This method implements the session management logic, ensuring each client gets
        a dedicated session with isolated resources. If a session doesn't exist, it
        creates a new one with the provided or default model configuration.

        :param session_id: Unique identifier for the session. If None, generates UUID
        :type session_id: :class:`Optional[str]`
        :param model_config: Model configuration for new sessions. Uses default if None
        :type model_config: :class:`Optional[ModelConfig]`
        :return: The existing or newly created client session
        :rtype: :class:`ClientSession`

        **Behavior:**

        - Generates UUID if no session_id provided
        - Returns existing session if session_id already exists
        - Creates new session with provided or default model configuration
        - New sessions are immediately ready for inference
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self.client_sessions:
            config = model_config or self._default_model_config
            self.client_sessions[session_id] = ClientSession(session_id, config)

        return self.client_sessions[session_id]

    async def remove_session(self, session_id: str):
        """Remove a client session and perform complete cleanup.

        Safely removes a session by stopping its inference task, cleaning up
        resources, and removing it from the active sessions dictionary. This
        method ensures proper resource cleanup to prevent memory leaks.

        :param session_id: Unique identifier of the session to remove
        :type session_id: :class:`str`

        **Behavior:**

        - Stops the session's inference task gracefully
        - Removes session from active sessions dictionary
        - Performs cleanup of session resources
        - Logs removal for monitoring purposes
        - Safe to call for non-existent sessions (no-op)
        """
        if session_id in self.client_sessions:
            session = self.client_sessions[session_id]
            await session.stop_inference()
            await session.cleanup()
            print(f"Session {session_id} removed and cleaned up")
            del self.client_sessions[session_id]

    async def cleanup_inactive_sessions(self):
        """Remove sessions with failed or completed inference tasks.

        Performs maintenance by identifying and removing sessions whose inference
        tasks have finished or failed. This prevents accumulation of dead sessions
        and ensures system resources are properly reclaimed.

        **Behavior:**

        - Scans all active sessions for completed inference tasks
        - Identifies sessions with failed tasks (exceptions)
        - Marks failed/completed sessions for removal
        - Removes inactive sessions and performs cleanup
        - Logs cleanup activities for monitoring

        **Use Cases:**

        - Periodic maintenance to clean up dead sessions
        - Error recovery after inference failures
        - Resource management in long-running deployments

        .. note::
            This method should be called periodically or after detecting
            inference failures to maintain system health.
        """
        inactive_sessions = []
        for session_id, session in self.client_sessions.items():
            if session.inference_task and session.inference_task.done():
                if session.inference_task.exception():
                    print(f"Error in session {session_id}: {session.inference_task.exception()}")
                    inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            await self.remove_session(session_id)

    def _setup_ws_route(self):
        """Configure WebSocket route for real-time audio streaming and transcription.

        Sets up the primary WebSocket endpoint that handles real-time audio streaming
        from clients. This endpoint manages the complete audio-to-transcription pipeline
        including session management, audio processing, and connection lifecycle.

        **WebSocket Protocol:**

        - **Endpoint**: `/ws/transcribe/{session_id}`
        - **Binary Protocol**: First byte indicates finality flag, remaining bytes contain WAV audio data
        - **Session Management**: Automatically creates or retrieves existing sessions
        - **Real-time Processing**: Streams audio directly to transcription pipeline

        **Connection Lifecycle:**

        1. **Connection**: Accept WebSocket connection for specified session
        2. **Session Setup**: Get or create session with default configuration
        3. **Inference Start**: Begin transcription processing for the session
        4. **Audio Streaming**: Continuously receive and process audio chunks
        5. **Cleanup**: Handle disconnections and errors gracefully

        **Audio Processing:**

        - Receives binary audio data in WAV format
        - Extracts finality flag from protocol header
        - Converts audio to tensor format for processing
        - Publishes audio events to session event bus
        - Maintains real-time processing capabilities

        **Error Handling:**

        - Graceful WebSocket disconnection handling
        - Exception logging without service interruption
        - Automatic session cleanup on connection loss

        .. note::
            This WebSocket endpoint is the primary interface for real-time
            transcription and supports the binary protocol expected by clients.
        """

        @self.app.websocket("/ws/transcribe/{session_id}")
        async def websocket_endpoint_with_session(websocket: WebSocket, session_id: str):
            await websocket.accept()

            # Get or create session
            session = self.get_or_create_session(session_id)
            print(f"WebSocket connection accepted for session: {session.session_id}")

            # Start inference task for this session
            await session.start_inference()

            try:
                while True:
                    message_bytes = await websocket.receive_bytes()

                    # Unpack the binary protocol: first byte is is_final flag
                    is_final = message_bytes[0] == 1  # First byte indicates if final
                    wav_bytes = message_bytes[1:]  # Rest is WAV data

                    audio_buffer = io.BytesIO(wav_bytes)
                    waveform, _ = torchaudio.load(audio_buffer)
                    waveform = waveform.squeeze(0)

                    audio_chunk = AudioChunk(
                        data=waveform,
                        timestamp=datetime.now(),  # Use current time as timestamp
                    )

                    # Append audio tensor to the session's queue
                    await session.event_bus.publish(AudioChunkReceived(chunk=audio_chunk, is_final=is_final))

            except WebSocketDisconnect:
                print(f"WebSocket disconnected for session: {session.session_id}")

            except Exception as e:
                print(f"Error in WebSocket for session {session.session_id}: {e}")

    def _setup_api_routes(self):
        """Configure all HTTP API routes for the FastAPI application.

        This method organizes and sets up the complete REST API interface for the
        transcription server. Routes are grouped by functionality to provide a
        comprehensive API for session management and transcription access.

        **Route Categories:**

        - **Session Management**: Create, delete, list, and manage sessions
        - **Transcription Access**: Retrieve current, final, and historical transcriptions
        - **Queue Monitoring**: Monitor audio processing queues and statistics

        **API Design:**

        - RESTful design with resource-based URLs
        - Consistent response schemas across endpoints
        - Proper HTTP status codes and error handling
        - Session-scoped operations for multi-client support

        .. note::
            Routes are automatically registered with the FastAPI application
            and include OpenAPI documentation for interactive API exploration.
        """
        self._setup_session_management_routes()
        self._setup_transcription_routes()

    def _setup_session_management_routes(self):
        """Configure HTTP routes for session lifecycle management.

        Sets up endpoints that handle the complete session lifecycle including
        creation, deletion, status monitoring, and operational controls. These
        routes provide the foundation for multi-client session management.

        **Endpoints Configured:**

        - `GET /installed_models` - Get list of installed models on the server
        - `POST /sessions` - Create new session with optional configuration
        - `POST /sessions/{session_id}` - Create session with specific ID
        - `DELETE /sessions/{session_id}` - Remove session and cleanup resources
        - `GET /sessions` - List all active sessions with statistics
        - `GET /sessions/{session_id}/status` - Get detailed session status
        - `POST /sessions/{session_id}/clear` - Clear session transcription history
        - `POST /sessions/{session_id}/restart` - Restart session inference

        **Features:**

        - Model configuration per session
        - Session existence validation
        - Graceful resource cleanup
        - Comprehensive status reporting
        - Operational controls for session management
        """

        @self.app.get("/installed_models", summary="Get installed models for the server", response_model=InstalledModelsResponse)
        async def installed_models() -> InstalledModelsResponse:
            installed_models = await asyncio.to_thread(get_installed_models)
            return InstalledModelsResponse(installed_models=installed_models)

        @self.app.post("/sessions", summary="Create a new transcription session", response_model=SessionResponse)
        async def create_session(model_config: Optional[ModelConfig] = None, session_id: Optional[str] = None) -> SessionResponse:
            session_id = session_id or str(uuid.uuid4())
            session = self.get_or_create_session(session_id, model_config)
            return SessionResponse(session_id=session.session_id, model_configuration=session.model_config.model_dump())

        @self.app.post("/sessions/{session_id}", summary="Create a transcription session with specific ID", response_model=SessionResponse)
        async def create_session_with_id(session_id: str, model_config: Optional[ModelConfig] = None) -> SessionResponse:
            if session_id in self.client_sessions:
                existing_session = self.client_sessions[session_id]
                return SessionResponse(
                    session_id=session_id, message="Session already exists", model_configuration=existing_session.model_config.model_dump()
                )

            session = self.get_or_create_session(session_id, model_config)
            return SessionResponse(session_id=session.session_id, model_configuration=session.model_config.model_dump())

        @self.app.delete("/sessions/{session_id}", summary="Remove a transcription session", response_model=MessageResponse)
        async def delete_session(session_id: str) -> MessageResponse:
            if session_id in self.client_sessions:
                print(f"Removing session {session_id}")
                await self.remove_session(session_id)
                return MessageResponse(message=f"Session {session_id} removed successfully", session_id=session_id)
            else:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)

        @self.app.get("/sessions", summary="List all active sessions", response_model=SessionListResponse)
        async def list_sessions() -> SessionListResponse:
            sessions = [
                SessionInfo(
                    session_id=session.session_id,
                    model_configuration=session.model_config.model_dump(),
                    inference_running=session.inference_task is not None and not session.inference_task.done(),
                    transcription_count=len(session.manager.transcriptions),
                    current_transcription=session.manager.current_transcription,
                    audio_queue_size=session.manager.queue_size,
                )
                for session in self.client_sessions.values()
            ]
            return SessionListResponse(sessions=sessions, total_sessions=len(sessions))

        @self.app.get("/sessions/{session_id}/status", summary="Get session status", response_model=SessionStatusResponse)
        async def get_session_status(session_id: str) -> SessionStatusResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return SessionStatusResponse(
                session_id=session_id,
                is_downloading=session.is_downloading,
                inference_running=session.inference_task is not None and not session.inference_task.done(),
                audio_queue_size=session.manager.queue_size,
                audio_queue_processed=session.manager.processed_chunks,
                transcription_count=len(session.manager.transcriptions),
                model_configuration=session.model_config.model_dump(),
            )

        @self.app.post("/sessions/{session_id}/clear", summary="Clear transcriptions for a session", response_model=MessageResponse)
        async def clear_session_transcriptions(session_id: str) -> MessageResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]

            # Clear transcription data
            session.manager.transcriptions.clear()
            session.manager.current_transcription = ""

            return MessageResponse(session_id=session_id, message="Transcriptions cleared successfully")

        @self.app.post("/sessions/{session_id}/restart", summary="Restart inference for a session", response_model=SessionOperationResponse)
        async def restart_session(session_id: str) -> SessionOperationResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)

            session = self.client_sessions[session_id]

            # Stop current inference
            await session.stop_inference()

            # Start new inference
            await session.start_inference()

            return SessionOperationResponse(
                session_id=session_id,
                message="Inference restarted successfully",
                inference_running=session.inference_task is not None and not session.inference_task.done(),
            )

    def _setup_transcription_routes(self):
        """Configure HTTP routes for transcription data access and queue management.

        Sets up endpoints that provide access to transcription results, real-time
        status monitoring, and audio queue management. These routes enable clients
        to retrieve transcription data and monitor processing progress.

        **Transcription Data Endpoints:**

        - `GET /sessions/{session_id}/transcriptions` - Get all completed transcriptions
        - `GET /sessions/{session_id}/transcription/current` - Get current active transcription
        - `GET /sessions/{session_id}/transcription/final` - Get complete final transcription

        **Queue Management Endpoints:**

        - `GET /sessions/{session_id}/queue/size` - Get current audio queue size
        - `GET /sessions/{session_id}/queue/processed` - Get processed chunk count
        - `POST /sessions/{session_id}/queue/clear` - Clear pending audio chunks

        **Features:**

        - Real-time transcription access
        - Processing progress monitoring
        - Queue management and clearing
        - Session-scoped data isolation
        """

        @self.app.get(
            "/sessions/{session_id}/transcriptions", summary="Get all transcriptions for a session", response_model=TranscriptionsResponse
        )
        async def get_transcriptions(session_id: str) -> TranscriptionsResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return TranscriptionsResponse(session_id=session_id, transcriptions=session.manager.transcriptions)

        @self.app.get(
            "/sessions/{session_id}/transcription/current",
            summary="Get the current transcription for a session",
            response_model=CurrentTranscriptionResponse,
        )
        async def get_current(session_id: str) -> CurrentTranscriptionResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return CurrentTranscriptionResponse(session_id=session_id, current_transcription=session.manager.current_transcription)

        @self.app.get(
            "/sessions/{session_id}/transcription/final",
            summary="Get the final transcription for a session",
            response_model=FinalTranscriptionResponse,
        )
        async def get_final(session_id: str) -> FinalTranscriptionResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return FinalTranscriptionResponse(session_id=session_id, final_transcription=session.manager.full_transcription)

        @self.app.get(
            "/sessions/{session_id}/queue/size", summary="Get the current audio queue size for a session", response_model=QueueSizeResponse
        )
        async def get_audio_queue_size(session_id: str) -> QueueSizeResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return QueueSizeResponse(session_id=session_id, audio_queue_size=session.manager.queue_size)

        @self.app.get(
            "/sessions/{session_id}/queue/processed",
            summary="Get num processed audio queue items for a session",
            response_model=QueueProcessedResponse,
        )
        async def get_audio_queue_processed(session_id: str) -> QueueProcessedResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            return QueueProcessedResponse(session_id=session_id, audio_queue_processed=session.manager.processed_chunks)

        @self.app.post("/sessions/{session_id}/queue/clear", summary="Clear audio queue for a session", response_model=MessageResponse)
        async def clear_audio_queue(session_id: str) -> MessageResponse:
            if session_id not in self.client_sessions:
                raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)
            session = self.client_sessions[session_id]
            session.manager.clear_audio_queue()
            return MessageResponse(session_id=session_id, message="Audio queue cleared successfully")

    def run(self):
        """Starts the FastAPI application in a separate thread.

        This method runs the FastAPI server using Uvicorn, which handles the HTTP requests for the transcription service.
        The server is launched in a separate thread, allowing the application to run concurrently with other tasks.
        It uses the `host` and `port` parameters defined in the class to bind the server.

        The server operates as a daemon thread, meaning it will not block the main program from exiting.

        .. note::

            - The application listens for HTTP requests at the specified `host` and `port`.
            - Ensure that the necessary configurations for the host and port are provided when calling this method.
        """
        threading.Thread(
            target=uvicorn.run,
            kwargs={"app": self.app, "host": self.host, "port": self.port, "access_log": False},
            daemon=True,
        ).start()
