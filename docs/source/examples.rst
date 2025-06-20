Examples
=============

Installation/Usage:
*******************
We recommend using ``uv`` as the python package manager.
How to install uv can be found at https://docs.astral.sh/uv/getting-started/installation/.
To install the dependencies, run the following command:
.. code-block::
    $ uv sync

After completing the installation, one can now use the transcriber.

Simply start the server and the cli client:
.. code-block::
    # In two seperate terminal sessions
    $ make local.server
    $ make local.cli

One can also run a frontend app. We provide a simple streamlit app that shows the transcriptions. The CLI Client still has to be running in the background.
We provide a Makefile task to run the server, frontend and cli client all at once:
.. code-block::
    # In another terminal session
    $ make local.run-cli

Low Level Usage:
****************
Running the server in a python script:

.. code-block:: python

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

.. note:: 
    The server will receive the audio data from the client and manages the transcription process.

Running a simple client in a python script:

.. code-block:: python
    
    import asyncio
    import aiohttp
    import websockets
    import soundfile as sf
    import io
    
    from whisper_web.inputstream_generator import GeneratorConfig, InputStreamGenerator
    from whisper_web.management import AudioManager
    from whisper_web.events import EventBus
    from app.helper import get_server_urls
    
    API_BASE_URL, WS_BASE_URL = get_server_urls()
    
    
    async def create_session_with_model(base_url: str, model_size: str = "small", counter: int = 0) -> None|str:
        """Create a new transcription session with a specific model configuration."""
        failed_tries = counter
        try:
            async with aiohttp.ClientSession() as session:
                # Create model config
                model_config = {
                    "model_size": model_size,
                    "device": "cuda",  # Use CPU for this example
                    "continuous": True,
                    "use_vad": False,
                    "samplerate": 16000,
                }
    
                # Create session
                async with session.post(f"{base_url}/sessions", json=model_config) as response:
                    if response.status == 200:
                        data = await response.json()
                        session_id = data["session_id"]
                        print(f"Created session {session_id} with model {model_size}")
                        return session_id
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Failed to create session: {response.status}",
                            headers=response.headers
                        )
        except Exception as e:
            print(f"Error creating session: {e}")
            await asyncio.sleep(1)  # Wait a bit before retrying
            return (
                await create_session_with_model(base_url, model_size, counter + 1)
                if failed_tries < 15
                else None
            )
    
    
    async def list_sessions(base_url: str):
        """List all active sessions."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/sessions") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"\nActive sessions ({data['total_sessions']}):")
                    for session_info in data["sessions"]:
                        print(f"  Session ID: {session_info['session_id']}")
                        print(f"  Model: {session_info['model_configuration']['model_size']}")
                        print(f"  Inference running: {session_info['inference_running']}")
                        print(f"  Transcriptions: {session_info['transcription_count']}")
                        print(f"  Audio queue size: {session_info['audio_queue_size']}")
                        print()
                else:
                    print(f"Failed to list sessions: {response.status}")
    
    
    async def get_session_status(base_url: str, session_id: str):
        """Get detailed status for a specific session."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/sessions/{session_id}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Session {session_id} status:")
                    print(f"  Inference running: {data['inference_running']}")
                    print(f"  Audio queue size: {data['audio_queue_size']}")
                    print(f"  Model: {data['model_configuration']['model_size']}")
                else:
                    print(f"Failed to get session status: {response.status}")
    
    
    async def stream_audio(session_id: str, manager: AudioManager):
        # Connect to WebSocket
        uri = f"{WS_BASE_URL}/ws/transcribe/{session_id}"
    
        # Connect without using context manager to keep connection alive
        ws = await websockets.connect(uri)
    
        try:
            # Monitor and send audio chunks
            while True:
                # Get audio data
                audio_chunk = await manager.get_next_audio_chunk()
                if audio_chunk is None:
                    await asyncio.sleep(0.1)
                    continue  # No audio chunk available, skip iteration
    
                chunk, is_final = audio_chunk
    
                print(f"Processing audio chunk: {chunk.data.shape}, final: {is_final}")
    
                if chunk.data.numel() > 0:
                    # Convert to numpy array and ensure proper format
                    audio_data = chunk.data.detach().cpu().numpy()
    
                    # Convert to WAV bytes
                    with io.BytesIO() as buffer:
                        sf.write(buffer, audio_data, samplerate=16000, format="WAV")
                        wav_bytes = buffer.getvalue()
    
                    # Send to WebSocket with custom binary protocol
                    # First byte indicates if final (1) or not (0)
                    final_flag = b"\x01" if is_final else b"\x00"
                    message = final_flag + wav_bytes
                    await ws.send(message)
    
                await asyncio.sleep(0.1)  # Prevent busy waiting
        except websockets.ConnectionClosedError as e:
            print(f"WebSocket connection closed: {e}")
        except Exception as e:
            print(f"Error during audio streaming: {e}")
    
        await ws.close()
    
    
    async def get_transcriptions(base_url: str, session_id: str):
        """Get transcriptions for a specific session."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/sessions/{session_id}/transcriptions") as response:
                if response.status == 200:
                    data = await response.json()
                    transcriptions = data["transcriptions"]
                    if transcriptions:
                        print(f"Transcriptions for session {session_id}:")
                        for i, transcription in enumerate(transcriptions):
                            print(f"  {i + 1}: {transcription}")
                    else:
                        print(f"No transcriptions yet for session {session_id}")
                else:
                    print(f"Failed to get transcriptions: {response.status}")
    
    
    async def cleanup_session(base_url: str, session_id: str):
        """Clean up a session."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{base_url}/sessions/{session_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Cleaned up session: {data['message']}")
                else:
                    print(f"Failed to cleanup session: {response.status}")
    
    
    async def main():
        print("Multi-Client, Multi-Model Transcription Server Demo")
        print("=" * 50)
    
        # Wait a bit for server to be ready
        await asyncio.sleep(1)
    
        try:
            # Create sessions with different model configurations
            session_id = await create_session_with_model(API_BASE_URL, "small")
            assert session_id is not None, "Failed to create session"
    
            # List all sessions
            await list_sessions(API_BASE_URL)
    
            # Get status for each session
            await get_session_status(API_BASE_URL, session_id)
    
            # Create generator config and manager
            event_bus = EventBus()
            generator_config = GeneratorConfig()
            generator_manager = AudioManager(event_bus)
            generator = InputStreamGenerator(generator_config, event_bus)
            print("InputStreamGenerator created")
            print("AudioManager created")
            print("Starting audio processing...")
    
            audio_task = asyncio.create_task(generator.process_audio())
            print("Audio processing task started")
            stream_task = asyncio.create_task(stream_audio(session_id, generator_manager))
            print("WebSocket streaming task started")
    
            # Run the streaming function
            await asyncio.gather(audio_task, stream_task, return_exceptions=True)
    
        except Exception as e:
            print(f"Demo error: {e}")
    
        finally:
            # Cleanup
            try:
                await cleanup_session(API_BASE_URL, session_id)
            except Exception as e:
                print(f"Failed to cleanup session: {e}")
    
    
    if __name__ == "__main__":
        print("Starting client...")
        print("Make sure the TranscriptionServer is running on localhost:8000")
        print("Press Ctrl+C to stop\n")
    
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nClient stopped by user")
        except Exception as e:
            print(f"CLient failed: {e}")


.. note::
    The client will read the audio data from the input stream and send it to the server for transcription.
    The input stream consists of a custom python module :class:`InputStreamGenerator` that reads audio data from a file or a microphone.