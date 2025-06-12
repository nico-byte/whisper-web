#!/usr/bin/env python3
"""
Pytest version of the API-only test for the multi-client whisper_web transcription server.
This test focuses on API functionality without loading models.
"""

import asyncio
import pytest
import pytest_asyncio
import httpx
import uvicorn
import threading
import time
from whisper_web.server import TranscriptionServer
from whisper_web.whisper_model import ModelConfig


@pytest_asyncio.fixture(scope="session")
async def test_server():
    """Fixture to set up and tear down the test server."""
    print("Setting up TranscriptionServer...")
    default_config = ModelConfig(model_size="small", device="cpu", use_vad=False)

    server = TranscriptionServer(
        default_model_config=default_config,
        host="127.0.0.1",
        port=8001,  # Use different port to avoid conflicts
    )

    # Start the server in a background thread
    thread = threading.Thread(target=uvicorn.run, kwargs={"app": server.app, "host": "127.0.0.1", "port": 8001}, daemon=True)
    thread.start()

    # Wait for server to start
    time.sleep(3)
    print("✓ Server started on port 8001")

    yield server

    print("Tearing down server...")


@pytest_asyncio.fixture
async def http_client():
    """Fixture to provide HTTP client for testing."""
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture(autouse=True)
async def cleanup_sessions():
    """Automatically cleanup sessions before and after each test."""
    # Cleanup before test
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=10.0) as client:
        try:
            response = await client.get("/sessions")
            if response.status_code == 200:
                sessions = response.json()
                for session in sessions["sessions"]:
                    await client.delete(f"/sessions/{session['session_id']}")
        except Exception:
            pass  # Ignore cleanup errors

    yield

    # Cleanup after test
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=10.0) as client:
        try:
            response = await client.get("/sessions")
            if response.status_code == 200:
                sessions = response.json()
                for session in sessions["sessions"]:
                    await client.delete(f"/sessions/{session['session_id']}")
        except Exception:
            pass  # Ignore cleanup errors


@pytest.mark.asyncio
async def test_list_empty_sessions(test_server, http_client):
    """Test listing sessions when none exist."""
    response = await http_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert sessions["total_sessions"] == 0


@pytest.mark.asyncio
async def test_create_session_with_custom_id(test_server, http_client):
    """Test creating a session with a custom ID."""
    custom_id = "test-session-123"
    response = await http_client.post(f"/sessions/{custom_id}")
    assert response.status_code == 200
    session_data = response.json()
    assert session_data["session_id"] == custom_id


@pytest.mark.asyncio
async def test_create_duplicate_session(test_server, http_client):
    """Test creating a session that already exists."""
    custom_id = "test-session-duplicate"

    # Create session first time
    response = await http_client.post(f"/sessions/{custom_id}")
    assert response.status_code == 200

    # Create same session again
    response = await http_client.post(f"/sessions/{custom_id}")
    assert response.status_code == 200
    session_data = response.json()
    assert "already exists" in session_data["message"]


@pytest.mark.asyncio
async def test_list_sessions_with_data(test_server, http_client):
    """Test listing sessions when sessions exist."""
    custom_id = "test-session-list"

    # Create a session
    await http_client.post(f"/sessions/{custom_id}")

    # List sessions
    response = await http_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert sessions["total_sessions"] == 1
    assert sessions["sessions"][0]["session_id"] == custom_id


@pytest.mark.asyncio
async def test_get_session_status(test_server, http_client):
    """Test getting status for a session."""
    custom_id = "test-session-status"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Get status
    response = await http_client.get(f"/sessions/{custom_id}/status")
    assert response.status_code == 200
    status = response.json()
    assert status["session_id"] == custom_id
    assert "audio_queue_size" in status
    assert "transcription_count" in status
    assert "model_config" in status


@pytest.mark.asyncio
async def test_get_empty_transcriptions(test_server, http_client):
    """Test getting transcriptions from an empty session."""
    custom_id = "test-session-transcriptions"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Get transcriptions
    response = await http_client.get(f"/sessions/{custom_id}/transcriptions")
    assert response.status_code == 200
    transcriptions = response.json()
    assert len(transcriptions["transcriptions"]) == 0


@pytest.mark.asyncio
async def test_get_current_transcription(test_server, http_client):
    """Test getting current transcription from a session."""
    custom_id = "test-session-current"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Get current transcription
    response = await http_client.get(f"/sessions/{custom_id}/transcription/current")
    assert response.status_code == 200
    current = response.json()
    assert current["current_transcription"] == ""


@pytest.mark.asyncio
async def test_get_final_transcription(test_server, http_client):
    """Test getting final transcription from a session."""
    custom_id = "test-session-final"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Get final transcription
    response = await http_client.get(f"/sessions/{custom_id}/transcription/final")
    assert response.status_code == 200
    final = response.json()
    assert final["final_transcription"] == ""


@pytest.mark.asyncio
async def test_clear_transcriptions(test_server, http_client):
    """Test clearing transcriptions for a session."""
    custom_id = "test-session-clear"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Clear transcriptions
    response = await http_client.post(f"/sessions/{custom_id}/clear")
    assert response.status_code == 200
    clear_result = response.json()
    assert "cleared successfully" in clear_result["message"]


@pytest.mark.asyncio
async def test_session_not_found(test_server, http_client):
    """Test 404 response for non-existent session."""
    response = await http_client.get("/sessions/non-existent/status")
    assert response.status_code == 404
    error = response.json()
    assert "not found" in error["detail"].lower()


@pytest.mark.asyncio
async def test_delete_session(test_server, http_client):
    """Test deleting a session."""
    custom_id = "test-session-delete"

    # Create session
    await http_client.post(f"/sessions/{custom_id}")

    # Delete session
    response = await http_client.delete(f"/sessions/{custom_id}")
    assert response.status_code == 200
    delete_result = response.json()
    assert "removed successfully" in delete_result["message"]

    # Verify session is deleted
    response = await http_client.get(f"/sessions/{custom_id}/status")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_session(test_server, http_client):
    """Test deleting a session that doesn't exist."""
    response = await http_client.delete("/sessions/non-existent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_session_lifecycle(test_server, http_client):
    """Test complete session lifecycle: create, use, delete."""
    custom_id = "test-session-lifecycle"

    # 1. Create session
    response = await http_client.post(f"/sessions/{custom_id}")
    assert response.status_code == 200

    # 2. Verify it exists
    response = await http_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    session_ids = [s["session_id"] for s in sessions["sessions"]]
    assert custom_id in session_ids

    # 3. Use session (get status)
    response = await http_client.get(f"/sessions/{custom_id}/status")
    assert response.status_code == 200

    # 4. Delete session
    response = await http_client.delete(f"/sessions/{custom_id}")
    assert response.status_code == 200

    # 5. Verify it's gone
    response = await http_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    session_ids = [s["session_id"] for s in sessions["sessions"]]
    assert custom_id not in session_ids


@pytest.mark.asyncio
async def test_concurrent_session_access(test_server, http_client):
    """Test concurrent access to multiple sessions."""
    session_id_1 = "test-concurrent-1"
    session_id_2 = "test-concurrent-2"

    # Create two sessions
    await http_client.post(f"/sessions/{session_id_1}")
    await http_client.post(f"/sessions/{session_id_2}")

    # Test concurrent access
    tasks = [
        http_client.get(f"/sessions/{session_id_1}/transcriptions"),
        http_client.get(f"/sessions/{session_id_2}/transcriptions"),
        http_client.get(f"/sessions/{session_id_1}/transcription/current"),
        http_client.get(f"/sessions/{session_id_2}/transcription/current"),
    ]
    responses = await asyncio.gather(*tasks)

    # All requests should succeed
    for response in responses:
        assert response.status_code == 200

    # Clean up
    await http_client.delete(f"/sessions/{session_id_1}")
    await http_client.delete(f"/sessions/{session_id_2}")
