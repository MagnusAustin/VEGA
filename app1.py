import streamlit as st
import numpy as np
from scipy.io.wavfile import read
import tensorflow as tf
from tensorflow import keras

# Function to load the trained model
@st.cache(allow_output_mutation=True)  # Cache the model for faster loading
def load_model():
    model = keras.models.load_model("model.keras")
    return model

# Function to recognize speaker
def recognize_speaker(audio_file):
    try:
        # Load the audio file
        sample_rate, audio_data = read(audio_file)

        # Preprocess audio data (adjust based on your model's requirements)
        audio_data = np.expand_dims(audio_data, axis=0)  # Add additional dimension for model input
        audio_data = preprocess_audio(audio_data)  # Replace with your specific preprocessing logic

        # Perform speaker recognition using the loaded model
        speaker_predictions = model.predict(audio_data)
        speaker_id = np.argmax(speaker_predictions)  # Get speaker ID based on predicted probabilities

        return speaker_id
    except FileNotFoundError:
        st.error("Error: Selected file is not found.")
        return None  # Indicate error for clear feedback

# Function to preprocess audio data (replace with your actual implementation)
def preprocess_audio(audio_data):
    # ... (your specific audio preprocessing logic here)
    # For example, you might resample to 16000 Hz, normalize, etc.
    return audio_data

# Function to render the main app layout
def main():
    st.title("Speaker Recognition App")
    st.subheader("Identify the speaker from an audio file")

    uploaded_file = st.file_uploader("Select an audio file (.wav):", type="audio/wav")

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()

        # Save the uploaded audio file temporarily (optional, adjust path)
        with open("temp_audio.wav", "wb") as buffer:
            buffer.write(bytes_data)

        try:
            # Load the model (cached)
            model = load_model()

            # Perform speaker recognition
            speaker_id = recognize_speaker("temp_audio.wav")

            # Display results
            if speaker_id is not None:  # Handle potential errors from recognize_speaker
                st.success(f"The predicted speaker ID is: {speaker_id}")
            else:
                st.error("An error occurred during processing. Please try again.")

        except Exception as e:  # Catch generic exceptions for better error handling
            st.error(f"An unexpected error occurred: {e}")

        finally:
            # Clean up temporary audio file (optional)
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()
