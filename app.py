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

    audio_snippet = open(temp_filename, "rb")

    # Transcribe the temporary audio file
    transcript = client.audio.translations.create(
        model="whisper-1",
        file=audio_snippet,
        response_format="text",
    )

    return transcript


if __name__ == "__main__":
    st.title("Toastmasters Table Topic Master")

    """A sample to use WebRTC in sendonly mode to transfer frames
    from the browser to the server and to render frames via `st.image`."""

    # from sample_utils.turn import get_ice_servers

    logger = logging.getLogger(__name__)

    webrtc_ctx = webrtc_streamer(
        key="audio-video-sendonly",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={  # Add this config
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": True},
    )

    image_place = st.empty()

    if "raw_video_frames" not in st.session_state:
        st.session_state["raw_video_frames"] = []

    if "raw_audio" not in st.session_state:
        st.session_state["raw_audio"] = None

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
                st.write(
                    audio_frame.to_ndarray().shape,
                    audio_frame.sample_rate,
                    len(audio_frame.layout.channels),
                )

                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk

            st.session_state["raw_audio"] = sound_window_buffer

        else:
            logger.warning("AudioReciver is not set. Abort.")
            break

    st.session_state["len_raw_video_frames"] = len(st.session_state["raw_video_frames"])
    st.write(st.session_state)

    if st.session_state.get("len_raw_video_frames", 0) > 0:
        with st.spinner(text="Generating Feedback"):
            model = genai.GenerativeModel("gemini-pro-vision")

            img_prompt = """
            Context: The assistant receives a tiled series of screenshots from a user's live video feed. These screenshots represent sequential frames from the video, capturing distinct moments. The assistant is to analyze these frames as a continuous video feed.

            1. Address the user directly, and assume that what is shown in the images is what the user is doing.
            2. Use "you" and "your" to refer to the user.
            3. DO NOT mention a series of individual images, a strip, a grid, a pattern or a sequence. Do as if the user and the assistant were both seeing the video.
            4. Keep in mind that the grid of images will show the same object in a sequence of time. E.g. If there are multiple people in different part of the images, it is the same person and NOT multiple people.
            5. Analyze the video of the person and give them feedback on how to improve their body language with respect to public speaking. Only analyze the video, not the audio
            6. Tell them what they did right while speaking, and then give them feedback as to what could be improved. 
            7. Finally conclude your feedback.
            """

            pil_image = PIL.Image.fromarray(st.session_state["raw_video_frames"][0])
            st.image(pil_image)

            response = model.generate_content(contents=[img_prompt, pil_image])

            response.resolve()
            st.write(response.text.strip())

            # Add the function to the streamlit app

            if st.session_state["raw_audio"] is not None:
                # st.write(audio_frame.to_ndarray().shape, audio_frame.sample_rate, len(audio_frame.layout.channels))
                st.session_state["exported_audio"] = st.session_state[
                    "raw_audio"
                ].export()
                st.audio(st.session_state["exported_audio"].read())
                transcription = transcribe_audio(st.session_state["raw_audio"])

                st.write(transcription)
