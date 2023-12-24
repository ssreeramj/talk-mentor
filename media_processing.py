import logging
import queue

import pydub
import streamlit as st

logger = logging.getLogger(__name__)


def process_media(webrtc_ctx, image_place):
    # capture and video and audio of the user
    sound_window_len = 500  # 0.5s
    sound_window_buffer = None

    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Video Queue is empty. Abort.")
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
            logger.warning("AudioReceiver is not set. Abort.")
            break


def process_video(webrtc_ctx, image_place):
    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Video queue is empty. Abort.")
                break

            img_rgb = video_frame.to_ndarray(format="rgb24")
            st.session_state["raw_video_frames"].append(img_rgb)
            image_place.image(img_rgb)

        else:
            logger.warning("VideoReceiver is not set. Abort.")
            break
