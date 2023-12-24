import tempfile

from openai import OpenAI


def transcribe_audio(audio_file):
    client = OpenAI()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_file.export(temp_file.name, format="wav")
        with open(temp_file.name, "rb") as audio_snippet:
            transcript = client.audio.translations.create(
                model="whisper-1",
                file=audio_snippet,
                response_format="text",
            )
    return transcript
