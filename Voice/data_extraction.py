import librosa
import numpy as np

def extract_features(filename, n_mfcc=13, n_fft=2048, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(filename)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Transpose the feature matrix to have time along the columns
    mfccs = mfccs.T
    
    return mfccs

if __name__ == "__main__":
    filename = r"C:\Users\anike\OneDrive\Desktop\Manas\Python\recorded_audio.wav"  # Change this to your WAV file
    features = extract_features(filename)
    print("MFCCs shape:", features.shape)
