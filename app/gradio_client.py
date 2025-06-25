import asyncio
import base64
import json
import logging
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperWebClient:
    def __init__(self, server_url: str = "ws://localhost:8765"):
        self.server_url = server_url
        self.websocket: Optional[Any] = None
        self.session_id = None

    async def connect(self):
        """Connect to the Whisper Web server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Connected to server at {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False

    async def send_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Send audio data to server and get transcription."""
        if not self.websocket:
            return None

        try:
            # Convert audio to base64
            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            message = {"type": "audio", "data": audio_b64, "sample_rate": sample_rate, "session_id": self.session_id}

            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            result = json.loads(response)

            if result.get("type") == "transcription":
                return result.get("text", "")

        except Exception as e:
            logger.error(f"Error sending audio: {e}")

        return None

    async def disconnect(self):
        """Disconnect from server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None


# Global client instance
client = WhisperWebClient()


def preprocess_audio(audio_data: Any) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Preprocess audio data with robust error handling."""
    try:
        if audio_data is None:
            return None, None

        # Handle different input formats
        if isinstance(audio_data, tuple):
            sr, y = audio_data
        elif isinstance(audio_data, dict):
            sr = audio_data.get("sample_rate", 16000)
            y = audio_data.get("array", audio_data.get("data"))
        else:
            logger.error(f"Unknown audio format: {type(audio_data)}")
            return None, None

        if y is None:
            return None, None

        # Convert to numpy array
        y = np.array(y, dtype=np.float32)

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        return y, sr

    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None, None


async def transcribe_async(stream: Optional[np.ndarray], new_chunk: Any) -> Tuple[Optional[np.ndarray], str]:
    """Async transcription function."""
    try:
        processed_audio, sample_rate = preprocess_audio(new_chunk)

        if processed_audio is None:
            return stream, "❌ Error: Invalid audio data"

        # Concatenate with existing stream
        if stream is not None:
            try:
                stream = np.concatenate([stream, processed_audio])
            except ValueError as e:
                logger.error(f"Error concatenating streams: {e}")
                stream = processed_audio
        else:
            stream = processed_audio

        # Send to server for transcription
        if not client.websocket:
            await client.connect()

        if client.websocket:
            transcription = await client.send_audio(stream, sample_rate or 16000)
            if transcription:
                return stream, f"🎯 {transcription}"
            else:
                return stream, "⏳ Processing..."
        else:
            return stream, "❌ Not connected to server"

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return stream, f"❌ Error: {str(e)}"


def transcribe(stream, new_chunk):
    """Sync wrapper for async transcription."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(transcribe_async(stream, new_chunk))


with gr.Blocks(title="Whisper Web - Real-time Transcription") as demo:
    gr.Markdown("# 🎤 Whisper Web - Real-time Speech Transcription")
    gr.Markdown("Connect to your Whisper server and transcribe audio in real-time")

    # Connection status
    connection_status = gr.Textbox(label="Connection Status", value="🔴 Disconnected", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Server Configuration")

            server_url = gr.Textbox(label="Server URL", value="ws://localhost:8765", placeholder="ws://localhost:8765")

            model_dropdown = gr.Dropdown(
                label="Model",
                choices=["whisper-small", "whisper-medium", "whisper-large-v2", "whisper-large-v3"],
                value="whisper-small",
                info="Select the Whisper model to use",
            )

            use_custom_model = gr.Checkbox(label="Use Custom Model", value=False, info="Enable if you want to specify a custom model ID")

            custom_model_id = gr.Textbox(
                label="Custom Model ID", placeholder="openai/whisper-large-v3", visible=False, info="HuggingFace model ID or local path"
            )

            # Show/hide custom model input
            use_custom_model.change(lambda x: gr.update(visible=x), inputs=[use_custom_model], outputs=[custom_model_id])

            with gr.Row():
                connect_btn = gr.Button("🔗 Connect", variant="primary")
                disconnect_btn = gr.Button("❌ Disconnect", variant="secondary")

            # Audio settings
            gr.Markdown("### Audio Settings")
            audio_quality = gr.Radio(
                label="Audio Quality", choices=["16kHz", "44.1kHz", "48kHz"], value="16kHz", info="Higher quality uses more bandwidth"
            )

            enable_vad = gr.Checkbox(label="Voice Activity Detection", value=True, info="Only transcribe when speech is detected")

        with gr.Column(scale=2):
            gr.Markdown("### Real-time Transcription")

            # Audio interface with better configuration
            audio_interface = gr.Interface(
                fn=transcribe,
                inputs=["state", gr.Audio(sources=["microphone"], streaming=True, label="🎙️ Microphone Input", show_label=True)],
                outputs=[
                    "state",
                    gr.Textbox(
                        label="📝 Live Transcription",
                        lines=10,
                        max_lines=20,
                        show_copy_button=True,
                        placeholder="Start speaking to see transcription here...",
                    ),
                ],
                live=True,
                allow_flagging="never",
                show_progress="hidden",
            )

    # Footer with usage tips
    gr.Markdown("""
    ### 💡 Usage Tips
    - Make sure your Whisper server is running before connecting
    - Speak clearly and at a normal pace for best results
    - Use headphones to avoid feedback when using live transcription
    - The transcription updates in real-time as you speak
    """)

    # Connection handlers
    async def handle_connect(url, model, use_custom, custom_id):
        try:
            client.server_url = url
            success = await client.connect()
            if success:
                return "🟢 Connected successfully"
            else:
                return "🔴 Connection failed"
        except Exception as e:
            return f"🔴 Error: {str(e)}"

    async def handle_disconnect():
        try:
            await client.disconnect()
            return "🔴 Disconnected"
        except Exception as e:
            return f"🔴 Error during disconnect: {str(e)}"

    def sync_connect(*args):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(handle_connect(*args))

    def sync_disconnect():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(handle_disconnect())

    connect_btn.click(sync_connect, inputs=[server_url, model_dropdown, use_custom_model, custom_model_id], outputs=[connection_status])

    disconnect_btn.click(sync_disconnect, outputs=[connection_status])


if __name__ == "__main__":
    print("🚀 Starting Whisper Web Client...")
    print("📍 Open http://localhost:7860 in your browser")
    print("🔧 Make sure your Whisper server is running on ws://localhost:8765")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, quiet=False)
