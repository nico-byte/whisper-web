import streamlit as st
import asyncio
import websockets
import soundfile as sf
import io
import tempfile
import os
import requests
import threading
import time
from datetime import datetime
from typing import Optional
from whisper_web.inputstream_generator import GeneratorConfig, InputStreamGenerator
from whisper_web.management import AudioChunkManager
from whisper_web.events import EventBus
from app.helper import get_server_urls

API_BASE_URL, WS_BASE_URL = get_server_urls()

st.set_page_config(page_title="Live Transcription with File Upload", layout="wide", initial_sidebar_state="expanded")

st.title("🎤 Live Transcription with File Upload")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "is_streaming" not in st.session_state:
    st.session_state["is_streaming"] = False
if "refresh_interval" not in st.session_state:
    st.session_state["refresh_interval"] = 1.0  # Default refresh interval
if "needs_refresh" not in st.session_state:
    st.session_state["needs_refresh"] = False
if "transcription_data" not in st.session_state:
    st.session_state["transcription_data"] = {"current": "", "final": "", "queue_size": 0, "error": None, "last_updated": None}
if "websocket_connection" not in st.session_state:
    st.session_state["websocket_connection"] = None
if "streaming_progress" not in st.session_state:
    st.session_state["streaming_progress"] = 0
if "background_result" not in st.session_state:
    st.session_state["background_result"] = None


def create_session(model_args) -> Optional[str]:
    """Create a new transcription session with specified model."""
    try:
        model_config = {
            "device": "cuda",  # Change to "cpu" or "mps" if needed
            "continuous": True,
            "use_vad": False,
            "samplerate": 16000,
        }

        model_config = {**model_config, **model_args}

        response = requests.post(f"{API_BASE_URL}/sessions", json=model_config, timeout=600)
        response.raise_for_status()

        data = response.json()
        session_id = data["session_id"]
        return session_id

    except requests.RequestException as e:
        st.error(f"Failed to create session: {str(e)}")
        return None


async def close_websocket_connection():
    """Close the active WebSocket connection if it exists."""
    if st.session_state.get("websocket_connection"):
        try:
            await st.session_state["websocket_connection"].close()
        except Exception:
            pass  # Ignore errors when closing
        finally:
            st.session_state["websocket_connection"] = None


class BackgroundStreamResult:
    """Thread-safe class to hold background streaming results."""

    def __init__(self):
        self.websocket_connection = None
        self.is_complete = False
        self.progress = 0
        self.error = None
        self.lock = threading.Lock()

    def update_progress(self, progress):
        with self.lock:
            self.progress = progress

    def set_complete(self, websocket_connection=None, error=None):
        with self.lock:
            self.is_complete = True
            self.websocket_connection = websocket_connection
            self.error = error

    def get_status(self):
        with self.lock:
            return {
                "progress": self.progress,
                "is_complete": self.is_complete,
                "websocket_connection": self.websocket_connection,
                "error": self.error,
            }


async def run_async_stream(session_id: str, audio_file_path: str, result_holder: BackgroundStreamResult):
    """Run the async streaming function in a thread."""
    try:
        # Create generator config and manager
        event_bus = EventBus()
        generator_config = GeneratorConfig(from_file=audio_file_path)
        generator_manager = AudioChunkManager(event_bus)
        generator = InputStreamGenerator(generator_config, event_bus)

        audio_task = asyncio.create_task(generator.process_audio())
        stream_task = asyncio.create_task(stream_audio_file_background(session_id, generator_manager, result_holder))

        # Run the streaming function
        ws_connection = await asyncio.gather(audio_task, stream_task, return_exceptions=True)

        # Mark as complete with success
        result_holder.set_complete(websocket_connection=ws_connection)

    except Exception as e:
        # Mark as complete with error
        result_holder.set_complete(error=str(e))
        print(f"Background streaming error: {e}")


def run_async_stream_wrapper(session_id: str, audio_file_path: str, result_holder: BackgroundStreamResult):
    """Wrapper to run async function in a new event loop within the thread."""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async function
        loop.run_until_complete(run_async_stream(session_id, audio_file_path, result_holder))

    except Exception as e:
        result_holder.set_complete(error=str(e))
        print(f"Async stream wrapper error: {e}")
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except Exception:
            pass


async def stream_audio_file_background(session_id: str, manager: AudioChunkManager, result_holder: BackgroundStreamResult):
    """Stream audio file to the transcription server via WebSocket in background."""
    try:
        # Connect to WebSocket
        uri = f"{WS_BASE_URL}/ws/transcribe/{session_id}"

        # Connect without using context manager to keep connection alive
        ws = await websockets.connect(uri)

        total_chunks = manager.num_chunks
        processed_chunks = 0
        failed_chunks = 0

        print(f"Connected to WebSocket for session {session_id}")

        # Monitor and send audio chunks
        while True:
            print(f"Processed chunks: {processed_chunks}, Failed chunks: {failed_chunks}, Total chunks: {total_chunks}")
            if failed_chunks > 10:
                break  # Assume file is transcribed completely if too many failures
            # Get audio data
            audio_chunk = await manager.get_next_audio_chunk()
            if audio_chunk is None:
                await asyncio.sleep(1.0)
                continue  # No audio chunk available, skip iteration

            chunk, is_final = audio_chunk

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
                print(f"Sent audio chunk: {len(wav_bytes)} bytes, final: {is_final}")

                # Update progress
                processed_chunks += 1
                if total_chunks == 0:
                    total_chunks = 100  # Estimate

                progress = int((processed_chunks / total_chunks) * 100)
                result_holder.update_progress(progress)

            await asyncio.sleep(0.1)  # Prevent busy waiting

        # Final progress update
        result_holder.update_progress(100)

        # Return the WebSocket connection to keep it alive
        return ws

    except Exception as e:
        raise RuntimeError(f"Error during background audio streaming: {str(e)}")


def get_session_transcriptions(session_id: str):
    """Fetch current and final transcriptions for a specific session."""
    try:
        # Fetch current transcription
        current_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/transcription/current", timeout=3)
        current_response.raise_for_status()
        current_data = current_response.json()

        # Fetch final transcription
        final_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/transcription/final", timeout=3)
        final_response.raise_for_status()
        final_data = final_response.json()

        # Fetch session status (including queue size)
        status_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/status", timeout=3)
        status_response.raise_for_status()
        status_data = status_response.json()

        return {
            "current": current_data.get("current_transcription", ""),
            "final": final_data.get("final_transcription", ""),
            "is_active": status_data.get("inference_running", False),
            "queue_size": status_data.get("audio_queue_size", 0),
            "processed_chunks": status_data.get("audio_queue_processed", 0),
            "error": None,
            "last_updated": datetime.now(),
        }

    except requests.RequestException as e:
        return {
            "current": "",
            "final": "",
            "queue_size": 0,
            "is_active": False,
            "processed_chunks": 0,
            "error": str(e),
            "last_updated": datetime.now(),
        }


def get_installed_models():
    """Scan the .models folder and return formatted model names."""
    models_path = "./.models"
    available_models = []

    try:
        if os.path.exists(models_path):
            # List all directories in .models folder
            for item in os.listdir(models_path):
                item_path = os.path.join(models_path, item)
                # Check if it's a directory and starts with "models--"
                if os.path.isdir(item_path) and item.startswith("models--"):
                    # Format: models--distil-whisper--distil-large-v3 -> distil-whisper/distil-large-v3
                    parts = item.split("--")
                    if len(parts) >= 3:
                        # Skip the "models" part and join the rest with "/"
                        formatted_name = "/".join(parts[1:])
                        available_models.append(formatted_name)

        # Sort models for consistent ordering
        available_models.sort()

        # Add fallback models if no models found
        if not available_models:
            available_models = ["distil-whisper/distil-small.en", "distil-whisper/distil-medium.en", "distil-whisper/distil-large-v3"]

    except Exception as e:
        st.error(f"Error scanning models folder: {e}")
        # Fallback models
        available_models = ["distil-whisper/distil-small.en", "distil-whisper/distil-medium.en", "distil-whisper/distil-large-v3"]

    return available_models


def display_transcriptions():
    """Display current and final transcriptions with queue status."""
    if st.session_state["session_id"]:
        # Update transcription data
        st.session_state["transcription_data"] = get_session_transcriptions(st.session_state["session_id"])

        data = st.session_state["transcription_data"]

        # Show error if any
        if data.get("error"):
            st.error(f"⚠️ Error: {data['error']}")
            return

        # Display queue status and metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

        with col_metrics1:
            queue_size = data.get("queue_size", 0)
            assert isinstance(queue_size, int), "Queue size must be an integer"
            if queue_size == 0:
                st.session_state["needs_refresh"] = False
            else:
                st.session_state["needs_refresh"] = True
            st.metric(label="📊 Queue Size", value=queue_size, delta=None if queue_size == 0 else f"{queue_size} pending")

        with col_metrics2:
            is_active = data.get("is_active", False)
            status_indicator = "🟢 Active" if is_active else "🔴 Inactive"
            st.metric(label="🔌 Session Status", value=status_indicator)

        with col_metrics3:
            processed_chunks = data.get("processed_chunks", 0)
            assert isinstance(processed_chunks, int), "Processed chunks must be an integer"
            st.metric(label="📦 Processed Chunks", value=processed_chunks)

        # Display transcriptions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔄 Current Transcription")
            current_text = data.get("current", "") or ""
            current_text = current_text.strip() if isinstance(current_text, str) else ""
            if current_text:
                st.info(current_text)

        with col2:
            st.markdown("### ✅ Final Transcription")
            final_text = data.get("final", "") or ""
            final_text = final_text.strip() if isinstance(final_text, str) else ""
            if final_text:
                st.success(final_text)

        # Show timestamp and queue details
        last_updated = data.get("last_updated")
        assert isinstance(last_updated, datetime), "Last updated must be datetime"

        status_info = []
        if last_updated and hasattr(last_updated, "strftime"):
            status_info.append(f"Last updated: {last_updated.strftime('%H:%M:%S')}")
        elif isinstance(last_updated, str):
            status_info.append(f"Last updated: {last_updated}")
        elif last_updated:
            status_info.append(f"Last updated: {str(last_updated)}")

        if status_info:
            st.caption(" | ".join(status_info))


# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Model selection
    installed_models = get_installed_models()

    selected_model = st.selectbox(
        "Select Model",
        installed_models,
        index=0,  # Default to first available model
        help="Select from available models in the .models folder",
    )

    # Option to use custom model ID instead
    use_custom_model = st.checkbox("Use custom model ID", value=False)

    if use_custom_model:
        custom_model_id = st.text_input("Custom Model ID", value="", help="Specify a custom model ID")
        model_args = {
            "model_id": custom_model_id if custom_model_id else None,
        }
    else:
        model_args = {
            "model_id": selected_model,
        }

    # Session management
    st.subheader("Session Management")

    if st.button("🆕 Create New Session", type="primary"):
        with st.spinner("Creating session..."):
            session_id = create_session(model_args)
            if session_id:
                st.session_state["session_id"] = session_id
                st.session_state["is_streaming"] = False
                st.success(f"Session created: {session_id}")
                st.rerun()

    if st.session_state["session_id"]:
        st.success(f"Current session: `{st.session_state['session_id']}`")

        if st.button("🗑️ Delete Session"):
            try:
                # Close WebSocket connection first
                if st.session_state.get("websocket_connection"):
                    asyncio.run(close_websocket_connection())

                # Stop streaming if running
                if st.session_state.get("is_streaming"):
                    st.session_state["is_streaming"] = False

                response = requests.delete(f"{API_BASE_URL}/sessions/{st.session_state['session_id']}")
                if response.status_code == 200:
                    st.session_state["session_id"] = None
                    st.session_state["is_streaming"] = False
                    st.session_state["websocket_connection"] = None
                    st.session_state["streaming_progress"] = 0
                    st.session_state["background_result"] = None
                    st.success("Session deleted successfully")
                    st.rerun()
                else:
                    st.error("Failed to delete session")
            except Exception as e:
                st.error(f"Error deleting session: {str(e)}")

    # Auto-refresh settings
    st.subheader("Display Settings")
    auto_refresh = st.checkbox("Auto refresh transcriptions", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0, 0.1)


# Main content area
if not st.session_state["session_id"]:
    st.info("👆 Please create a session in the sidebar to get started.")
    st.markdown("""
    ### Getting Started
    1. **Create a Session**: Click "Create New Session" in the sidebar
    2. **Upload Audio File**: Use the file uploader below
    3. **Start Transcription**: Click "Start Transcription" to process the audio
    4. **View Results**: Watch the transcriptions appear in real-time
    """)
else:
    # File upload section
    st.header("📁 Audio File Upload")

    uploaded_file = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "flac", "aac", "m4a", "ogg"], help="Upload an audio file to transcribe"
    )

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Audio player
        st.audio(uploaded_file.getvalue(), format=f"audio/{uploaded_file.name.split('.')[-1]}")

        # Transcription controls
        if not st.session_state["is_streaming"]:
            st.info("📤 Ready to start background transcription")
        else:
            if st.session_state.get("streaming_thread") and st.session_state["streaming_thread"].is_alive():
                st.warning("⏳ Transcription running in background...")
            else:
                st.success("✅ Background transcription completed")

        # Progress and transcription display
        if st.session_state["is_streaming"]:
            # Check if background thread is still running
            if st.session_state.get("streaming_thread") and st.session_state["streaming_thread"].is_alive():
                # Get status from background result holder
                result_holder = st.session_state.get("background_result")
                if result_holder:
                    status = result_holder.get_status()
                    progress = status["progress"]

                    progress_bar = st.progress(progress / 100.0)
                    status_text = st.empty()
                    status_text.text(f"Processing audio file in background... {progress}%")

                    # Auto-refresh to update progress
                else:
                    st.error("Background streaming status unavailable")
                    st.session_state["is_streaming"] = False
            else:
                # Background streaming completed - check results
                result_holder = st.session_state.get("background_result")
                if result_holder:
                    status = result_holder.get_status()

                    if status["error"]:
                        st.error(f"❌ Streaming failed: {status['error']}")
                    elif status["progress"] == 100:
                        st.session_state["websocket_connection"] = status["websocket_connection"]
                        st.success("✅ File processed! WebSocket connection active. Check transcriptions below.")
                    else:
                        st.success("✅ File processed! Check transcriptions below.")

                st.session_state["is_streaming"] = False
                st.session_state["streaming_progress"] = 0
                st.session_state["background_result"] = None

        elif uploaded_file is not None and not st.session_state["is_streaming"]:
            # Start background streaming
            if st.session_state["transcription_data"].get("queue_size") == 0:
                if st.session_state.get("session_id") and st.button("🎯 Start Background Transcription", type="primary"):
                    st.session_state["is_streaming"] = True
                    st.session_state["streaming_progress"] = 0

                    # Create result holder for thread-safe communication
                    result_holder = BackgroundStreamResult()
                    st.session_state["background_result"] = result_holder

                    # Start streaming in background thread
                    streaming_thread = threading.Thread(
                        target=run_async_stream_wrapper, args=(st.session_state["session_id"], temp_file_path, result_holder), daemon=True
                    )
                    streaming_thread.start()
                    st.session_state["streaming_thread"] = streaming_thread

                    st.rerun()

        # Clean up temp file if not streaming
        elif "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    # Transcription display section
    st.header("📝 Transcription Results")

    # Always display transcriptions first
    display_transcriptions()

    # Auto-refresh controls (only show if session exists)
    if st.session_state["session_id"]:
        if auto_refresh:
            # Auto-refresh status and controls
            col1, col2 = st.columns([3, 1])

            with col1:
                # Update refresh timestamp and trigger refresh
                st.caption("🔄 Fetching new transcriptions...")
            if st.session_state["needs_refresh"] or st.session_state["is_streaming"]:
                time.sleep(st.session_state["refresh_interval"])
                st.rerun()
        else:
            # Manual refresh mode
            refresh_col1, refresh_col2 = st.columns([4, 1])
            with refresh_col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    <small>Whisper Real-Time Transcription Frontend</small>
</div>
""",
    unsafe_allow_html=True,
)
