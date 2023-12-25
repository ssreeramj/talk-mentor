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

from llm_feedback import (generate_text_feedback, generate_video_feedback,
                          get_speech_topic)

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
    st.session_state["frame_idx"] = 0
    st.session_state["theme-input"] = ""
    st.session_state["speech-topic"] = ''


if __name__ == "__main__":
    st.title("TalkMentor: Your Speech Coach")
    st.write("This tool allows users to select speech themes, generating tailored topics. Post-speech, it offers audio/video feedback for improvement. User-friendly interface ensures a seamless experience. Enhance public speaking skills efficiently.")

    logger = logging.getLogger(__name__)

    st.text_input(
        label="Enter a theme",
        placeholder="Science, Cricket, Education",
        key="theme-input",
    )
    if st.session_state["theme-input"]:
        if not st.session_state.get("speech-topic", ''):
            get_speech_topic()
        
        if st.session_state["speech-topic"]:
            st.write(f"Here is your topic: {st.session_state['speech-topic']}")
            st.write(f"Click 'Start' whenever you are ready! All the best!")
            
            webrtc_ctx = webrtc_streamer(
                key="audio-video-sendonly",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration={  # Add this config
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": True},
                audio_receiver_size=512,
                video_receiver_size=512,
            )

            image_place = st.empty()
            # Create the output directory if it doesn't exist
            # os.makedirs("raw_video_frames", exist_ok=True)

            if "raw_video_frames" not in st.session_state:
                st.session_state["raw_video_frames"] = []
                st.session_state["raw_audio"] = pydub.AudioSegment.empty()
                st.session_state["frame_idx"] = 0

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
                    # Save the combined image
                    # output_path = os.path.join("raw_video_frames", f"output_image_{st.session_state['frame_idx'] + 1}.png")
                    # PIL.Image.fromarray(img_rgb).save(output_path)
                    image_place.image(img_rgb)
                    # st.session_state["frame_idx"] += 1

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
            # st.write(st.session_state)

            # Handling of the video frames and audio buffer
            if (
                len(st.session_state["raw_audio"]) > 0
                and st.session_state["raw_video_frames"]
            ):
                with st.spinner(text="Generating Feedback"):
                    st.audio(st.session_state["exported_audio"].read())

                    transcription = transcribe_audio(st.session_state["raw_audio"])
                    # with open("./notebooks/sample_text.txt", "r") as f:
                    #     transcription = f.read()
                    with st.expander(label="Expand to see your speech"):
                        st.write(transcription)

                    st.divider()

                    video_feedback = generate_video_feedback()
                    text_feedback = generate_text_feedback(
                        speech_text=transcription, topic=st.session_state["speech-topic"]
                    )

            st.button("Reset", on_click=reset_speech)
