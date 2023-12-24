import logging
import os
import queue
import tempfile
from io import BytesIO

import google.generativeai as genai
import PIL
import pydub
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from llm_feedback import generate_text_feedback, generate_video_feedback

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.set_page_config(page_title="TTT", page_icon="microphone")


# Create a function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    client = OpenAI()

    # Create a temporary file with a specific name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name  # Get the temporary file name

        # Export the audio segment to the temporary file
        audio_file.export(temp_filename, format="wav")

        # Do something with the temporary file (e.g., process it further)
        print("Temporary file saved at:", temp_filename)

    with open(temp_filename, "rb") as audio_snippet:
        # Transcribe the temporary audio file
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_snippet,
            response_format="text",
        )

    return transcript


def reset_speech():
    st.session_state["raw_video_frames"] = []
    st.session_state["raw_audio"] = pydub.AudioSegment.empty()
    st.session_state["exported_audio"] = None


if __name__ == "__main__":
    st.title("Toastmasters Table Topic Master")

    logger = logging.getLogger(__name__)

    webrtc_ctx = webrtc_streamer(
        key="audio-video-sendonly",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={  # Add this config
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": True},
        audio_receiver_size=256,
        video_receiver_size=256,
    )

    image_place = st.empty()

    if "raw_video_frames" not in st.session_state:
        st.session_state["raw_video_frames"] = []
        st.session_state["raw_audio"] = pydub.AudioSegment.empty()

    sound_window_len = 500  # 5s
    sound_window_buffer = None

    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            img_rgb = video_frame.to_ndarray(format="rgb24")
            st.session_state["raw_video_frames"].append(img_rgb)
            image_place.image(img_rgb)

        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )

                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk

            st.session_state["raw_audio"] = sound_window_buffer
            st.session_state["exported_audio"] = sound_window_buffer.export()

        else:
            logger.warning("AudioReciver is not set. Abort.")
            break

    st.session_state["num_frames"] = len(st.session_state["raw_video_frames"])
    st.write(st.session_state)

    # Handling of the video frames and audio buffer
    if len(st.session_state["raw_audio"]) > 0 and st.session_state["raw_video_frames"]:
        with st.spinner(text="Generating Feedback"):
            st.audio(st.session_state["exported_audio"].read())

            # transcription = transcribe_audio(st.session_state["raw_audio"])
            with open("./notebooks/sample_text.txt", "r") as f:
                transcription = f.read()
            with st.expander(label="Expand to see your speech"):
                st.write(transcription)

            st.divider()

            video_feedback = generate_video_feedback()
            text_feedback = generate_text_feedback(
                speech_text=transcription, topic="Higher Studies"
            )

    st.button("Reset speech", on_click=reset_speech)
