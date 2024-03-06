import os
import pyaudio
import wave

def record_audio(num_recordings=10, duration=5, channels=1, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()

    # Create folder if it doesn't exist
    folder_name = "data_voice"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i in range(num_recordings):
        filename = os.path.join(folder_name, f"recording_{i+1}.wav")
        
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=chunk_size)

        print(f"Recording {i+1}/{num_recordings}...")

        frames = []

        for _ in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

    audio.terminate()

if __name__ == "__main__":
    record_audio()
