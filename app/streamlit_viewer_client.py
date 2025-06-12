import streamlit as st
import requests
import time
from datetime import datetime
from app.helper import get_server_urls

API_BASE_URL, WS_BASE_URL = get_server_urls()

st.set_page_config(page_title="Live Transcription Viewer", layout="wide")
st.title("📝 Live Transcription Feed")

# Sidebar controls
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 0.5, 10.0, 2.0, 0.1)
show_timestamps = st.sidebar.checkbox("Show timestamps", value=True)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)


def fetch_sessions():
    """Fetch all active sessions from the server"""
    try:
        response = requests.get(f"{API_BASE_URL}/sessions", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("sessions", []), None
    except requests.RequestException as e:
        return [], f"Failed to fetch sessions: {str(e)}"


def fetch_session_transcriptions(session_id):
    """Fetch current and final transcriptions for a specific session"""
    try:
        # Fetch current transcription
        current_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/transcription/current", timeout=3)
        current_response.raise_for_status()
        current_data = current_response.json()

        # Fetch final transcription
        final_response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/transcription/final", timeout=3)
        final_response.raise_for_status()
        final_data = final_response.json()

        return {
            "current": current_data.get("current_transcription", ""),
            "final": final_data.get("final_transcription", ""),
            "error": None,
            "last_updated": datetime.now(),
        }

    except requests.RequestException as e:
        return {"current": "", "final": "", "error": str(e), "last_updated": datetime.now()}


def display_session_box(session, transcription_data):
    """Display a transcription box for a single session"""
    session_id = session["session_id"]

    # Create an expandable container for each session
    with st.expander(f"🎤 Session: {session_id}", expanded=True):
        # Session info
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**Session ID:** `{session_id}`")
        with col2:
            model_config = session.get("model_config", {})
            model_size = model_config.get("model_size", "unknown") if isinstance(model_config, dict) else "unknown"
            st.write(f"**Model:** {model_size}")
        with col3:
            inference_running = session.get("inference_running", False)
            status_emoji = "🟢" if inference_running else "🔴"
            st.write(f"**Status:** {status_emoji} {'Running' if inference_running else 'Stopped'}")

        # Show timestamp if enabled
        if show_timestamps and transcription_data.get("last_updated"):
            st.caption(f"Last updated: {transcription_data['last_updated'].strftime('%H:%M:%S')}")

        # Show error if any
        if transcription_data.get("error"):
            st.error(f"⚠️ Error: {transcription_data['error']}")
            return

        # Current transcription
        current_text = transcription_data.get("current", "").strip()
        if current_text:
            st.markdown("**🔄 Current Transcription:**")
            st.info(current_text)
        else:
            st.markdown("**🔄 Current Transcription:**")
            st.write("_No current transcription_")

        # Final transcription
        final_text = transcription_data.get("final", "").strip()
        if final_text:
            st.markdown("**✅ Final Transcription:**")
            st.success(final_text)
        else:
            st.markdown("**✅ Final Transcription:**")
            st.write("_No final transcription_")

        # Session stats
        st.caption(f"Transcription count: {session.get('transcription_count', 0)} | Audio queue: {session.get('audio_queue_size', 0)}")


def display_content():
    """Display the main content (sessions and transcriptions)"""
    # Fetch sessions
    sessions, error = fetch_sessions()

    # Show error if any
    if error:
        st.error(error)
        return

    if not sessions:
        st.info("🔍 No active sessions found. Start a transcription session to see it here.")
        st.write("**To start a session:**")
        st.code(f"curl -X POST {API_BASE_URL}/sessions", language="bash")
        return

    st.success(f"📡 Found {len(sessions)} active session(s)")

    # Display each session in its own box
    for i, session in enumerate(sessions):
        session_id = session["session_id"]

        # Fetch transcription data for this session
        transcription_data = fetch_session_transcriptions(session_id)

        # Display the session box
        display_session_box(session, transcription_data)

        # Add separator between sessions (except for the last one)
        if i < len(sessions) - 1:
            st.markdown("---")


# Main app execution
if auto_refresh:
    # Auto-refresh mode
    st.info("🔄 Auto-refresh is enabled. The page will update automatically.")

    # Create a placeholder for the content
    content_placeholder = st.empty()

    # Auto-refresh loop
    while True:
        with content_placeholder.container():
            display_content()
        time.sleep(refresh_interval)
else:
    # Manual refresh mode
    if st.button("🔄 Refresh Now", key="manual_refresh_button"):
        st.rerun()

    display_content()
