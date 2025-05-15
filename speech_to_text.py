import streamlit as st
import pyaudio
import wave
import numpy as np
import time
from datetime import datetime
import speech_recognition as sr

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence detection
SILENCE_DURATION = 2  # Seconds of silence before stopping

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    st.write("Recording... Speak into the microphone.")
    frames = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)
    
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Convert data to numpy array for amplitude analysis
        audio_data = np.frombuffer(data, dtype=np.int16)
        amplitude = np.abs(audio_data).mean()
        
        # Check for silence
        if amplitude < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        # Stop if silence duration is exceeded
        if silent_chunks > max_silent_chunks:
            break
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"recording_{timestamp}.wav"
    wf = wave.open(audio_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return audio_filename

def transcribe_audio(audio_filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filename) as source:
        audio = recognizer.record(source)
    
    try:
        # Use Google's Speech Recognition API
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def save_text_to_file(text, timestamp):
    text_filename = f"transcription_{timestamp}.txt"
    with open(text_filename, "w") as file:
        file.write(text)
    return text_filename

def main():
    st.title("Voice Recorder with Speech-to-Text")
    st.write("Click the button to start recording. Recording stops after 2 seconds of silence, and the transcribed text is displayed below.")
    
    if st.button("Start Recording"):
        with st.spinner("Recording in progress..."):
            audio_filename = record_audio()
            st.write(f"Recording saved as: {audio_filename}")
            
            # Transcribe the audio
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_filename)
                # Display transcription using st.write
                st.write("### Transcribed Text:")
                st.write(transcription)
                
                # Save transcription to text file
                timestamp = audio_filename.split("_")[1].split(".")[0]
                text_filename = save_text_to_file(transcription, timestamp)
                st.write(f"Transcription saved as: {text_filename}")
                
                # Provide download links
                with open(audio_filename, "rb") as audio_file:
                    st.download_button(
                        label="Download Recording",
                        data=audio_file,
                        file_name=audio_filename,
                        mime="audio/wav"
                    )
                
                with open(text_filename, "rb") as text_file:
                    st.download_button(
                        label="Download Transcription",
                        data=text_file,
                        file_name=text_filename,
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()