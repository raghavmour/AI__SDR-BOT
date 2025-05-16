import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import speech_recognition as sr
import os

    

def record_audio(pause_threshold=0.5, sample_rate=41000):
    """
    Records audio using streamlit audio_recorder and returns the path to a temp WAV file.
    """
    audio_bytes = audio_recorder(pause_threshold=pause_threshold, sample_rate=sample_rate)

    if audio_bytes:
        #st.audio(audio_bytes, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            audio_path = tmpfile.name
        
        #st.success("✅ Audio recorded!")
        return audio_path
    return None

def transcribe_audio(audio_path):
    """
    Transcribes the given WAV audio file using Google Speech Recognition API.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "❌ Could not understand the audio."
    except sr.RequestError as e:
        return f"❌ Google API error: {e}"

# --- Main App Logic ---


