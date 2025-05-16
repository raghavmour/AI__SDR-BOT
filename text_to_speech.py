from gtts import gTTS
import io
# Load environment variables from .env file


def generate_audio(text):
    tts = gTTS(text)
    audio_bytes_io = io.BytesIO()
    tts.write_to_fp(audio_bytes_io)
    return audio_bytes_io.getvalue()
