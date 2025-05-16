from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load API key from environment
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")  # Make sure the key name matches your .env file

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

def generate_audio(text):
    """
    Converts text to speech using ElevenLabs API and returns audio bytes in MP3 format.
    """
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with your voice ID
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_generator)
    return audio_bytes
