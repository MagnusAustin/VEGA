import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from scipy.io.wavfile import read
import tensorflow as tf
from tensorflow import keras

# Function to load the trained model
def load_model():
    model = keras.models.load_model("model.keras")
    return model

# Function to recognize speaker
def recognize_speaker(audio_file):
    # Load the audio file
    sample_rate, audio_data = read(audio_file)
    
    # Preprocess audio data (you may need to adjust this based on your preprocessing)
    # For example, you may need to resample the audio data to 16000 Hz
    
    # Perform speaker recognition using the loaded model
    # (you need to implement this part based on your model)
    # Here's a placeholder implementation:
    speaker_id = np.random.randint(0, 5)  # Placeholder random speaker ID
    
    return speaker_id

# Function to handle file selection
def select_file():
    filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if filepath:
        speaker_id = recognize_speaker(filepath)
        messagebox.showinfo("Speaker Recognition", f"The predicted speaker ID is: {speaker_id}")

# Function to create the GUI
def create_gui():
    # Create the root window
    root = tk.Tk()
    root.title("Speaker Recognition")

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window dimensions and position
    window_width = 400
    window_height = 200
    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")

    # Create GUI elements
    label = ttk.Label(root, text="Select an audio file (.wav) to recognize speaker:")
    label.pack(pady=10)

    button = ttk.Button(root, text="Select File", command=select_file)
    button.pack(pady=5)

    # Start the main event loop
    root.mainloop()

create_gui()
