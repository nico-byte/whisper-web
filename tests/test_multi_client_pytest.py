#!/usr/bin/env python3
"""
Pytest version of the multi-client test for the whisper_web transcription server.
Tests multi-client functionality with proper pytest fixtures and async handling.
"""

import asyncio
import httpx
import uvicorn
import threading
import time
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from whisper_web.server import TranscriptionServer
from whisper_web.whisper_model import ModelConfig


@pytest.fixture(scope="session")
def server() -> Generator[TranscriptionServer, None, None]:
    """Set up the transcription server for the entire test session."""
    print("\nSetting up TranscriptionServer...")

    # Use small model by default for faster testing
    default_config = ModelConfig(model_size="small")
    server = TranscriptionServer(default_model_config=default_config, port=8001)

    # Run server in a thread
    thread = threading.Thread(target=uvicorn.run, kwargs={"app": server.app, "host": "127.0.0.1", "port": 8001}, daemon=True)
    thread.start()

    # Give server time to start
    time.sleep(3)
    print("✓ Server started on port 8001")

    yield server

    print("\nTearing down server...")


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an HTTP client with extended timeout for model downloads."""
    # Extended timeout for model downloads
    timeout = httpx.Timeout(120.0, connect=30.0)  # 2 minutes for requests, 30s for connect

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=timeout) as client:
        yield client


@pytest_asyncio.fixture(autouse=True)
async def cleanup_sessions(client: httpx.AsyncClient):
    """Automatically clean up sessions after each test."""
    yield  # Run the test first

    # Clean up any remaining sessions
    try:
        response = await client.get("/sessions")
        if response.status_code == 200:
            sessions = response.json()
            for session in sessions["sessions"]:
                await client.delete(f"/sessions/{session['session_id']}")
                print(f"Cleaned up session: {session['session_id']}")
    except Exception as e:
        print(f"Cleanup error: {e}")


class TestMultiClientAPI:
    """Test class for multi-client API functionality."""

    @pytest.mark.asyncio
    async def test_create_default_session(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test creating a session with default configuration."""
        print("\n1. Creating a new session (may trigger model download)...")
        start_time = time.time()

        response = await client.post("/sessions")
        elapsed = time.time() - start_time
        print(f"   Request completed in {elapsed:.1f} seconds")

        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        print(f"✓ Created session: {session_id}")
        print(f"  Model config: {session_data['model_config']['model_size']}")

        # Verify session exists
        assert session_id is not None
        assert session_data["model_config"]["model_size"] == "small"

    @pytest.mark.asyncio
    async def test_create_custom_session(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test creating a session with custom configuration."""
        print("\n2. Creating session with custom config...")
        custom_config = ModelConfig(model_size="medium")
        start_time = time.time()

        response = await client.post("/sessions", json=custom_config.model_dump())
        elapsed = time.time() - start_time
        print(f"   Request completed in {elapsed:.1f} seconds")

        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        print(f"✓ Created session: {session_id}")
        print(f"  Model config: {session_data['model_config']['model_size']}")

        assert session_data["model_config"]["model_size"] == "medium"

    @pytest.mark.asyncio
    async def test_list_sessions(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test listing all sessions."""
        # Create a couple of sessions first
        session1_response = await client.post("/sessions")
        session2_config = ModelConfig(model_size="medium")
        session2_response = await client.post("/sessions", json=session2_config.model_dump())

        assert session1_response.status_code == 200
        assert session2_response.status_code == 200

        print("\n3. Listing all sessions...")
        response = await client.get("/sessions")
        assert response.status_code == 200

        sessions = response.json()
        print(f"✓ Found {sessions['total_sessions']} sessions")

        assert sessions["total_sessions"] >= 2
        for session in sessions["sessions"]:
            print(f"  - {session['session_id']}: {session['model_config']['model_size']} model")

    @pytest.mark.asyncio
    async def test_session_status(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test getting session status."""
        # Create a session
        response = await client.post("/sessions")
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        print(f"\n4. Getting status for session {session_id}...")
        response = await client.get(f"/sessions/{session_id}/status")
        assert response.status_code == 200

        status = response.json()
        print("✓ Session status retrieved")
        print(f"  Audio queue size: {status['audio_queue_size']}")
        print(f"  Transcription count: {status['transcription_count']}")
        print(f"  Model size: {status['model_config']['model_size']}")

        # Verify status structure
        assert "audio_queue_size" in status
        assert "transcription_count" in status
        assert "model_config" in status
        assert status["audio_queue_size"] >= 0
        assert status["transcription_count"] >= 0

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test concurrent access to different sessions."""
        # Create two sessions
        session1_response = await client.post("/sessions")
        session2_config = ModelConfig(model_size="medium")
        session2_response = await client.post("/sessions", json=session2_config.model_dump())

        assert session1_response.status_code == 200
        assert session2_response.status_code == 200

        session_id_1 = session1_response.json()["session_id"]
        session_id_2 = session2_response.json()["session_id"]

        print("\n5. Testing concurrent access to both sessions...")
        tasks = [
            client.get(f"/sessions/{session_id_1}/transcriptions"),
            client.get(f"/sessions/{session_id_2}/transcriptions"),
            client.get(f"/sessions/{session_id_1}/transcription/current"),
            client.get(f"/sessions/{session_id_2}/transcription/current"),
        ]
        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            assert response.status_code == 200

        print("✓ Concurrent access to multiple sessions successful")

    @pytest.mark.asyncio
    async def test_session_cleanup(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test session deletion and cleanup."""
        # Create sessions
        session1_response = await client.post("/sessions")
        session2_response = await client.post("/sessions")

        assert session1_response.status_code == 200
        assert session2_response.status_code == 200

        session_id_1 = session1_response.json()["session_id"]
        session_id_2 = session2_response.json()["session_id"]

        print("\n6. Cleaning up sessions...")

        # Delete first session
        response = await client.delete(f"/sessions/{session_id_1}")
        assert response.status_code == 200
        print(f"✓ Deleted session {session_id_1}")

        # Delete second session
        response = await client.delete(f"/sessions/{session_id_2}")
        assert response.status_code == 200
        print(f"✓ Deleted session {session_id_2}")

        # Verify cleanup
        print("\n7. Verifying cleanup...")
        response = await client.get("/sessions")
        assert response.status_code == 200

        sessions = response.json()
        print(f"✓ {sessions['total_sessions']} sessions remaining (expected: 0)")
        assert sessions["total_sessions"] == 0

    @pytest.mark.asyncio
    async def test_session_restart_and_clear(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test session restart and clear functionality."""
        # Create a session
        response = await client.post("/sessions")
        assert response.status_code == 200
        session_data = response.json()
        session_id = session_data["session_id"]

        print(f"\n8. Testing session restart for {session_id}...")

        # Test restart
        response = await client.post(f"/sessions/{session_id}/restart")
        assert response.status_code == 200
        print("✓ Session restart successful")

        # Test clear
        response = await client.post(f"/sessions/{session_id}/clear")
        assert response.status_code == 200
        print("✓ Session clear successful")

    @pytest.mark.asyncio
    async def test_invalid_session_operations(self, server: TranscriptionServer, client: httpx.AsyncClient):
        """Test operations on non-existent sessions."""
        non_existent_id = "non-existent-session-id"

        print("\n9. Testing operations on non-existent session...")

        # Try to get status of non-existent session
        response = await client.get(f"/sessions/{non_existent_id}/status")
        assert response.status_code == 404

        # Try to delete non-existent session
        response = await client.delete(f"/sessions/{non_existent_id}")
        assert response.status_code == 404

        # Try to restart non-existent session
        response = await client.post(f"/sessions/{non_existent_id}/restart")
        assert response.status_code == 404

        print("✓ Proper error handling for non-existent sessions")


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    import subprocess

    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "--asyncio-mode=auto"])
    sys.exit(result.returncode)
