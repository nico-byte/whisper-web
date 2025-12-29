import asyncio
import signal
import sys
from whisper_web.lib.backbone.server import TranscriptionServer
from whisper_web.lib.transcription.whisper_model import ModelConfig
from whisper_web.helper import is_running_in_docker

import logging

logger = logging.getLogger(__name__)

# Automatically set HOST based on execution environment
HOST = "0.0.0.0" if is_running_in_docker() else "127.0.0.1"
PORT = 8000


def create_default_model_config() -> ModelConfig:
    """Create a default model configuration."""
    return ModelConfig(
        model_size="small",  # Use small model as default for faster startup
        device="cpu",  # Use CPU for broader compatibility
        continuous=True,
        use_vad=False,
        samplerate=16000,
    )


async def monitor_sessions(server: TranscriptionServer, interval: int = 30):
    """Monitor sessions and clean up inactive ones periodically."""
    while True:
        try:
            await asyncio.sleep(interval)
            await server.cleanup_inactive_sessions()

            active_sessions = len(server.client_sessions)
            if active_sessions > 0:
                logger.info(f"Active sessions: {active_sessions}")
                for session_id, session in server.client_sessions.items():
                    status = (
                        "running"
                        if (
                            session.transcribe_task
                            and not session.transcribe_task.done()
                            or session.diarization_task
                            and not session.diarization_task.done()
                        )
                        else "stopped"
                    )
                    transcribe_queue_size = session.manager.audio_queue.qsize()
                    diarization_queue_size = session.manager.diarization_queue.qsize()
                    transcription_count = len(session.manager.transcriptions)
                    logger.info(
                        f"  {session_id}: {status}, transcribe_queue={transcribe_queue_size} \
                          diarization_queue={diarization_queue_size}, transcriptions={transcription_count}"
                    )

        except Exception as e:
            logger.exception(f"Session monitor error: {e}")


async def main():
    """Main server function."""
    # Create server with default configuration
    default_config = create_default_model_config()
    server = TranscriptionServer(default_model_config=default_config, host=HOST, port=PORT)

    print("Starting Multi-Client Transcription Server")
    print("=" * 40)
    print(f"Environment: {'Docker' if is_running_in_docker() else 'Local/User'}")
    print(f"Host: {server.host}")
    print(f"Port: {server.port}")
    print(f"Default model: {default_config.model_size}")
    print(f"Default device: {default_config.device}")
    print("=" * 40)

    # Start the FastAPI server in a thread
    server.run()

    # Wait a moment for server to start
    await asyncio.sleep(2)

    # Start session monitoring
    monitor_task = asyncio.create_task(monitor_sessions(server))

    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nShutting down server...")

        # Cancel monitoring
        monitor_task.cancel()

        # Cleanup all sessions
        logger.info("Cleaning up sessions...")
        session_ids = list(server.client_sessions.keys())
        for session_id in session_ids:
            try:
                await server.remove_session(session_id)
            except Exception as e:
                logger.exception(f"Error cleaning up session {session_id}: {e}")

        logger.info("Server stopped.")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"\nReceived signal {signum}")
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
