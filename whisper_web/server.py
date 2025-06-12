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
    """Represents a client session with its own transcription manager and model."""

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
        """Start the inference task for this client session."""
        if self.inference_task is None or self.inference_task.done():
            self.inference_task = asyncio.create_task(self.manager.run_inference(self.model))

    async def stop_inference(self):
        """Stop the inference task for this client session."""
        if self.inference_task and not self.inference_task.done():
            self.inference_task.cancel()
            try:
                await self.inference_task
            except asyncio.CancelledError:
                pass

    async def handle_model_download(self, event: DownloadModel):
        """Handle model download event."""
        self.is_downloading = not event.is_finished
        print(f"Model download {'started' if not event.is_finished else 'finished'} for session {self.session_id}")


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
    :type host: str, optional
    :param port: Port for the FastAPI server, defaults to 8000
    :type port: int, optional

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
        """Get an existing session or create a new one."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self.client_sessions:
            config = model_config or self._default_model_config
            self.client_sessions[session_id] = ClientSession(session_id, config)

        return self.client_sessions[session_id]

    async def remove_session(self, session_id: str):
        """Remove a client session and stop its inference task."""
        if session_id in self.client_sessions:
            session = self.client_sessions[session_id]
            await session.stop_inference()
            del self.client_sessions[session_id]
            print(f"Session {session_id} removed and cleaned up")

    async def cleanup_inactive_sessions(self):
        """Remove sessions with failed or completed inference tasks."""
        inactive_sessions = []
        for session_id, session in self.client_sessions.items():
            if session.inference_task and session.inference_task.done():
                if session.inference_task.exception():
                    print(f"Session {session_id} has failed inference task, marking for cleanup")
                    inactive_sessions.append(session_id)

        for session_id in inactive_sessions:
            await self.remove_session(session_id)

    def _setup_ws_route(self):
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
                    print(f"Received {len(message_bytes)} bytes from session {session.session_id}")

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
        """
        Sets up the routes for the FastAPI application.

        This method defines several HTTP endpoints that allow clients to interact with their transcription sessions.
        Routes are organized into session management, transcription access, and legacy compatibility endpoints.
        """
        self._setup_session_management_routes()
        self._setup_transcription_routes()

    def _setup_session_management_routes(self):
        """Setup routes for session creation, deletion, and management."""

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
        """Setup routes for accessing transcription data and controlling sessions."""

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

        @self.app.post("/clear", summary="Clear transcriptions (legacy)", response_model=MessageResponse)
        async def clear_legacy() -> MessageResponse:
            session = self.get_or_create_session("default")

            # Clear transcription data
            session.manager.transcriptions.clear()
            session.manager.current_transcription = ""

            return MessageResponse(message="Transcriptions cleared successfully")

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
