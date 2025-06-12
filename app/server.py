#!/usr/bin/env python3
"""
Example server startup script demonstrating the refactored multi-client, multi-model TranscriptionServer.

This script shows how to:
1. Configure and start the server with default model settings
2. Handle multiple clients with different model configurations
3. Monitor and manage sessions
"""

import asyncio
import signal
import sys
from whisper_web.server import TranscriptionServer
from whisper_web.whisper_model import ModelConfig
from app.helper import is_running_in_docker

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
                print(f"Active sessions: {active_sessions}")
                for session_id, session in server.client_sessions.items():
                    status = "running" if (session.inference_task and not session.inference_task.done()) else "stopped"
                    queue_size = session.manager.audio_queue.qsize()
                    transcription_count = len(session.manager.transcriptions)
                    print(f"  {session_id}: {status}, queue={queue_size}, transcriptions={transcription_count}")

        except Exception as e:
            print(f"Session monitor error: {e}")


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
    print()

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
        print("\nShutting down server...")

        # Cancel monitoring
        monitor_task.cancel()

        # Cleanup all sessions
        print("Cleaning up sessions...")
        session_ids = list(server.client_sessions.keys())
        for session_id in session_ids:
            try:
                await server.remove_session(session_id)
            except Exception as e:
                print(f"Error cleaning up session {session_id}: {e}")

        print("Server stopped.")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\nReceived signal {signum}")
    sys.exit(0)


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)
