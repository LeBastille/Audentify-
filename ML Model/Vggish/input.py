import io
import soundfile as sf
import librosa
import sounddevice as sd
import numpy as np
import tensorflow as tf
import joblib
import tkinter as tk
from tkinter import filedialog
import time

max_size = 16000 * 10

# Load the trained model
model = joblib.load('svm_model.pkl') # Replace with your model filename
vggish = tf.saved_model.load("/Users/a_mat/Audentity/vggish")

def predict_genre(val):
    predicted_label = val[0]
    genre_mapping = {0: 'Pop',
                     1: 'Metal',
                     2: 'Disco',
                     3: 'Blues',
                     4: 'Reggae',
                     5: 'Classical',
                     6: 'Rock',
                     7: 'Hip Hop',
                     8: 'Country',
                     9: 'Jazz'
                     }
    predicted_genre = genre_mapping[predicted_label]
    print("The predicted genre is:", predicted_genre)

def file_predict():
    try:

        root = tk.Tk()
        root.withdraw()
        audioFile = filedialog.askopenfilename()
        root.destroy()
        # Loading audio file
        waveform, sr = librosa.load(audioFile, sr=16000)
        print(waveform.dtype)

        segment_length = 1.0
        num_segments = int(waveform.shape[0] / (sr * segment_length))
        features = []

        for i in range(num_segments):
            start = int(i * segment_length * sr)
            end = int(start + sr * segment_length)
            segment = waveform[start:end]
            vgg_features = vggish(segment).numpy()
            features.append(vgg_features)

        # Extracting features using VGGish
        aggregated_features = np.mean(features, axis=0)
        input_features = aggregated_features.flatten()
        prediction = model.predict(input_features.reshape(1, -1))
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")


def realtime_predict(duration=10, sample_rate=16000):
    audio_data = sd.rec(duration * sample_rate, samplerate=sample_rate, channels=1)
    sd.wait()

    # Save as WAV file
    sf.write('input.wav', audio_data, sample_rate)
    print(librosa.get_duration(filename='input.wav'))

    # Load, process segments
    signal, sr = librosa.load('input.wav')
    print(signal.dtype)

    segment_length = 1.0
    num_segments = int(signal.shape[0] / (sr * segment_length))
    features = []

    # process 1-second audio chunks
    for i in range(num_segments):
        start = int(i * segment_length * sr)
        end = int(start + sr * segment_length)
        segment = signal[start:end]
        vgg_features = vggish(segment).numpy()
        features.append(vgg_features)

    aggregated_features = np.mean(features, axis=0)
    input_features = aggregated_features.flatten()
    prediction = model.predict(input_features.reshape(1, -1))
    return prediction



if __name__ == "__main__":
    choice = int(input("Enter your input choice:"))
    if choice == 1:
        genre = file_predict()
        predict_genre(genre)
    elif choice == 2:
        genre2 = realtime_predict()
        predict_genre(genre2)

