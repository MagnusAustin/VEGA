import os
import numpy as np
import librosa

def extract_mfcc(wav_file, n_mfcc=13, n_fft=2048, hop_length=512):
    # Load the WAV file
    audio, sr = librosa.load(wav_file, sr=None)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Transpose the matrix to have time along the first axis
    mfcc = mfcc.T
    return mfcc

def load_training_data(data_dir):
    X = []
    y = []
    for wav_file in os.listdir(data_dir):
        wav_file_path = os.path.join(data_dir, wav_file)
        mfcc = extract_mfcc(wav_file_path)
        X.append(mfcc)
        y.append(os.path.basename(wav_file))  # Use the file name as the label
    return np.array(X), np.array(y)

# Directory containing your training data
data_dir = r"C:\Users\anike\OneDrive\Desktop\Manas\Python\data_voice"
X_train, y_train = load_training_data(data_dir)

# Now X_train contains the MFCC features and y_train contains the corresponding file names (labels)
# You can use X_train and y_train for training your model
