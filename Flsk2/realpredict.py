import numpy as np
import tensorflow as tf
import joblib
import tkinter as tk
from tkinter import filedialog
import librosa
import pyaudio
import wave
import numpy as np
from sklearn.impute import SimpleImputer

from flask import Flask, render_template, request, jsonify
import numpy as np
import soundfile as sf

app = Flask(__name__)


@app.route('/predict_realtime')
def predict_realtime():
    prediction = record_audio_and_save_wav(filename="output.wav", duration=10)
    return jsonify({'genre': prediction[0]})  # Convert numpy array to python list

@app.route('/get_prediction')
def get_prediction():
    global prediction
    # Return the prediction if available, otherwise indicate it's still processing
    return jsonify({'genre': prediction[0] if prediction is not None else None})

sample_rate = 16000
model = joblib.load('svm_model.pkl') # Replace with your model filename
vggish = tf.saved_model.load("/Users/a_mat/Audentity/vggish")

def record_audio_and_save_wav(filename="output.wav", duration=10):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    sample_rate = 16000

    p = pyaudio.PyAudio()

    print('Recording...')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sample_rate,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print('Finished recording')

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    waveform, sr = librosa.load("output.wav", sr=16000)

    segment_length = 1.0
    num_segments = int(waveform.shape[0] / (sr * segment_length))
    features = []

    for i in range(num_segments):
        start = int(i * segment_length * sr)
        end = int(start + sr * segment_length)
        segment = waveform[start:end]
        vgg_features = vggish(segment).numpy()
        features.append(vgg_features)

    aggregated_features = np.mean(features, axis=0)
    input_features = aggregated_features.flatten()
    prediction = model.predict(input_features.reshape(1, -1))
    return prediction
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
        waveform, sr = librosa.load(audioFile, sr=16000)

        segment_length = 1.0
        num_segments = int(waveform.shape[0] / (sr * segment_length))
        features = []

        for i in range(num_segments):
            start = int(i * segment_length * sr)
            end = int(start + sr * segment_length)
            segment = waveform[start:end]
            vgg_features = vggish(segment).numpy()
            features.append(vgg_features)

        aggregated_features = np.mean(features, axis=0)
        input_features = aggregated_features.flatten()
        prediction = model.predict(input_features.reshape(1, -1))
        return prediction

    except Exception as e:
        print(f"Error during prediction: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('try.html')  # Serve your HTML page

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    # Directly call your realtime_predict function
    prediction = record_audio_and_save_wav(duration=10, filename="output.wav")
    predicted_label = int(prediction[0])  # Convert to Python int
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
    predicted_genre = genre_mapping.get(predicted_label, "Unknown")  # Handle unknown labels
    return jsonify({'genre': predicted_genre})


if __name__ == "__main__":
    app.run(debug=True)