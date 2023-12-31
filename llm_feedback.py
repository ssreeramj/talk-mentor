import math
import os
import re
import traceback

import cv2
import google.generativeai as genai
import PIL
import streamlit as st

text_model = genai.GenerativeModel("gemini-pro")
vision_model = genai.GenerativeModel("gemini-pro-vision")


def get_speech_topic():
    try:
        topic_prompt = f"""Context: The assistant receives some themes and it has to return a topic related to one of the themes for the user to speak.
        Eg: if the theme is "Holiday" then the topic could be, "Talk about your last vacation"
        If the theme is "Cricket", then the topic could be, "What's your unforgettable cricket moment".

        Just give the topic, nothing else

        Themes: {st.session_state['theme-input']}
        """
        response = text_model.generate_content(topic_prompt)
        speech_topic = response.text.strip()
    except Exception as e:
        speech_topic = "Tell me about yourself"
        print(f"Error in generating topic: {traceback.format_exc()}")

    st.session_state["speech-topic"] = speech_topic


def process_video():
    # print(len(final_frames))
    # output_directory = "output_images"
    image_size = (100, 100)  # Adjust the size of each frame as needed
    rows, columns = 5, 5
    images_per_output = rows * columns

    # get the video frames to analyze
    # Create a PIL Image object from the NumPy array
    frames = st.session_state["raw_video_frames"]
    final_frames = []
    skip_frames = math.ceil(st.session_state["num_frames"] / (images_per_output * 16))
    # st.write(f"{skip_frames=}")
    for i in range(len(frames[::skip_frames])):
        pil_image = PIL.Image.fromarray(frames[i])
        final_frames.append(pil_image)

    # st.write(f"{len(final_frames)=}")
    total_images = min(math.ceil(len(final_frames) / images_per_output), 16)
    # st.write(f"{total_images=}")

    # Create the output directory if it doesn't exist
    # os.makedirs(output_directory, exist_ok=True)

    new_images = []
    # Iterate through each set of frames for creating images
    frame_number = 0
    for i in range(total_images):
        # Create a new image
        new_image = PIL.Image.new(
            "RGB", (image_size[0] * columns, image_size[1] * rows)
        )

        # Load frames into the image
        for j in range(images_per_output):
            if frame_number == len(final_frames):
                break
            # Open and resize the frame
            # print(frame_number)
            frame = final_frames[frame_number]
            frame = frame.resize(image_size)
            # st.image(frame)

            # Calculate the position to paste the frame
            row_position = (
                (frame_number - (i * images_per_output)) // columns
            ) * image_size[1]
            col_position = (frame_number % columns) * image_size[0]

            # print(f"{row_position=}, {col_position=}")

            # Paste the frame onto the new image
            new_image.paste(frame, (col_position, row_position))
            frame_number += 1

        # Save the combined image
        # output_path = os.path.join(output_directory, f"output_image_{i + 1}.png")
        # new_image.save(output_path)
        # st.image(new_image)
        new_images.append(new_image)

    print("Images created successfully.")
    return new_images


def generate_video_feedback():
    try:
        user_images = process_video()

        img_prompt = """Context: The assistant receives a tiled series of screenshots from a user's live video feed. These screenshots represent sequential frames from the video, capturing distinct moments. The assistant is to analyze these frames as a continuous video feed.

        1. Address the user directly, and assume that what is shown in the images is what the user is doing.
        2. Use "you" and "your" to refer to the user.
        3. DO NOT mention a series of individual images, a strip, a grid, a pattern or a sequence. Do as if the user and the assistant were both seeing the video.
        4. Keep in mind that the grid of images will show the same object in a sequence of time. E.g. If there are multiple people in different part of the images, it is the same person and NOT multiple people.
        5. Analyze the video of the person and give them feedback on how to improve their body language with respect to public speaking. Only analyze the video, not the audio
        6. Tell them what they did right while speaking, and then give them feedback as to what could be improved. 
        7. Finally conclude your feedback.
        """
        response = vision_model.generate_content(contents=[img_prompt, *user_images])
        response.resolve()
        video_feedback = response.text.strip()
        # video_feedback = "Sample Video Feedback"
    except:
        print(f"Excption in generating video feedback: {traceback.format_exc()}")
        video_feedback = "Error in generating video feedback!"

    st.subheader("Video Feedback")
    st.write(video_feedback)
    st.divider()


def get_grammarian_feedback(speech_text, topic):
    grammarian_prompt_template = f"""Imagine you're a dedicated language coach for a public speaking skills improvement club, equipped with the vast knowledge about vocabulary, grammar and languages. Your mission is to empower each member to become a confident and captivating speaker.

    Context: The assistant receives the text of the speech by the user, and the topic on which the user was supposed to speak. Given the text and the topic, here are the things you have to give feedback on:
    Instructions: 1. Address the user directly, and assume that what is shown in the images is what the user is doing. 2. Use "you" and "your" to refer to the user.
    
    Wordsmith Spotlight:
    Did the speaker utilize any unique or under-appreciated words that deserve a shout-out? Share their "Word of the Speech" moment, explaining its meaning and showcasing its potential to enrich future speeches.
    Did you encounter any instances of particularly vivid or impactful phrasing? Highlight these gems and encourage similar strategies.

    Fluency Feedback:
    How smoothly did the speech flow? Did transitions feel natural, or were there any areas that could benefit from tighter phrasing or bridge sentences? Offer specific suggestions for improved coherence and pacing.
    Did you notice any patterns in sentence structure or word choice? Identify areas where speaker shines and suggest techniques to inject further variety and dynamism into their delivery.

    Grammar Guardian:
    Were there any grammatical errors or awkward constructions that could be gently addressed? Craft a private message for speaker providing alternative word choices, clearer sentence structures, and polished punctuation suggestions. Remember, constructive and positive feedback is key!

    Relevance to the Topic:
    Did the user speak on the assigned topic?

    Learning Lighthouse:
    Based on the speech, what specific online resources or language learning activities would benefit speaker most? Recommend tailored materials that address any areas for improvement while also aligning with their individual learning style and interests.

    Overall:
    Provide a personalized and encouraging evaluation that celebrates speaker's strengths while gently guiding them towards even greater linguistic mastery. Remember, your goal is to empower them to find their voice, refine their message, and deliver impactful presentations with increased confidence and clarity.

    Topic given to the user: {topic}

    Here is the speech given by the user
    ```
    {speech_text}
    ```                                                                                                             
    """
    try:
        grammarian_response = text_model.generate_content(grammarian_prompt_template)
        return grammarian_response.text.strip()
    except Exception as e:
        print(f"Exception in grammarian feedback => {traceback.format_exc()}")
        return ""


def count_filler_words(text):
    filler_words = [
        "um",
        "ah",
        "er",
        "so",
        "like",
        "you know",
        "well",
        "i mean",
        "actually",
        "basically",
        "literally",
        "okay",
        "right",
        "sort of",
        "kind of",
        "i guess",
        "i think",
        "i believe",
        "i hope",
        "i feel",
        "you see",
        "you know what",
    ]
    filler_word_counts = {}

    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()

    for word in filler_words:
        filler_word_counts[word] = cleaned_text.split().count(word)

    return dict(sorted(filler_word_counts.items(), key=lambda x: x[1], reverse=True))


def get_ah_counter_feedback(speech_text):
    try:
        filler_word_counts = count_filler_words(text=speech_text)

        ah_counter_prompt_template = f"""Your job is to keep the user aware of how many filler words or empty words they are using in their speech.
        Here is a dictionary of counts of filler words used by the user. Just give the feedback to the user. Highlight the top 5 or 10 filler words used by them. Only highligh the filler words used by the speaker. Don't tell them what they didn't speak.
        ```
        {filler_word_counts}
        ```
        """
        ah_counter_response = text_model.generate_content(ah_counter_prompt_template)
        return ah_counter_response.text.strip()

    except Exception as e:
        print(f"Exception in ah counter feedback => {traceback.format_exc()}")
        return ""


def generate_text_feedback(speech_text, topic):
    """
    function to give feedback on the speech text.
    """
    st.subheader("Speech Feedback")
    # st.write("Sample Speech Feedback")
    ah_counter_feedback = get_ah_counter_feedback(speech_text=speech_text)
    st.write(ah_counter_feedback)
    
    grammarian_feedback = get_grammarian_feedback(
        speech_text=speech_text,
        topic=topic,
    )
    st.write(grammarian_feedback)

