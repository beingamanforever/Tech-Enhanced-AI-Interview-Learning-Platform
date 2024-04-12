import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel


DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10


def calculate_speaking_pace(transcription, chunk_length):
    words = transcription.split()
    num_words = len(words)
    speaking_rate = num_words / chunk_length  # Words per second
    return speaking_rate

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False


def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def audio_to_text():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=1)
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"

            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                # print("Customer:{}".format(transcription))
                speaking_pace = calculate_speaking_pace(transcription, DEFAULT_CHUNK_LENGTH)
                print("Speaking pace: {:.2f} words per second".format(speaking_pace))
                output = transcription
                return transcription, speaking_pace

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    audio_to_text()