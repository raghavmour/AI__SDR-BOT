
from elevenlabs.client import ElevenLabs
ELEVEN_LABS_API_KEY = "elven_labs_api"  # Replace with your ElevenLabs API key
client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)

def generate_audio(text):
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with your voice id
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_generator)
    return audio_bytes
