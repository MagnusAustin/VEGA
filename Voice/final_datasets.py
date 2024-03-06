import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

# Function to extract MFCC features from a WAV file
def extract_mfcc(wav_file, num_mfcc=13):
    # Load audio file
    y, sr = librosa.load(wav_file)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    
    return mfccs

# Function to extract features from a folder of WAV files
def extract_features_from_folder(folder_path):
    # Initialize empty lists to store features and labels
    features = []
    labels = []
    
    # Iterate over files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                # Extract features from each WAV file
                wav_file = os.path.join(root, file)
                mfccs = extract_mfcc(wav_file)
                
                # Append features and label (folder name) to lists
                features.append(mfccs)
                labels.append(os.path.basename(root))
    
    return features, labels

# Function to save features and labels to a file
def save_features_to_file(features, labels, file_path):
    np.savez(file_path, features=features, labels=labels)

if __name__ == "__main__":
    # Path to the folder containing WAV files
    folder_path = r"C:\Users\anike\OneDrive\Desktop\Manas\Python\data_voice"
    
    # Extract features from the folder
    features, labels = extract_features_from_folder(folder_path)
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Save training features and labels to a file
    save_features_to_file(train_features, train_labels, "train_data.npz")
    
    # Save testing features and labels to a file
    save_features_to_file(test_features, test_labels, "test_data.npz")
