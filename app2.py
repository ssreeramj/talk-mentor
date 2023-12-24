import logging
import queue

import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from audio_transcription import transcribe_audio
from config import configure_genai_and_streamlit
from llm_feedback import generate_video_feedback
from media_processing import process_media

logger = logging.getLogger(__name__)


def main():
    configure_genai_and_streamlit()
    st.title("Toastmasters Table Topic Master")

    if "raw_video_frames" not in st.session_state:
        st.session_state["raw_video_frames"] = []
        st.session_state["raw_audio"] = None

    # Setup for WebRTC streamer and other Streamlit components
    webrtc_ctx = webrtc_streamer(
        key="audio-video-sendonly",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={  # Add this config
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": True},
        video_receiver_size=16,
        audio_receiver_size=16,
    )

    image_place = st.empty()

    process_media(
        webrtc_ctx=webrtc_ctx,
        image_place=image_place,
    )

    st.write(st.session_state)

    # Handling of the video frames and audio buffer
    if (
        st.session_state["raw_audio"] is not None
        and st.session_state["raw_video_frames"]
    ):
        with st.spinner(text="Generating Feedback"):
            video_feedback = generate_video_feedback()
            st.write(video_feedback)

            st.audio(st.session_state["exported_audio"].read())

            transcription = transcribe_audio(st.session_state["raw_audio"])
            st.write(transcription)


if __name__ == "__main__":
    main()
