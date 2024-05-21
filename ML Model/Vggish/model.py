import tensorflow as tf
import pandas as pd
import os, librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from glob import glob
from keras.regularizers import l2

import IPython.display as ipd
import matplotlib.pyplot as plt
import seaborn as sns
from visualkeras import layered_view

import tensorflow_hub as hub
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import layers, Sequential
from keras.callbacks import EarlyStopping

DATASET_PATH = '/Users/a_mat/Audentity/dataset/genres_original'

from warnings import filterwarnings
filterwarnings('ignore')

# Load the model.
vggish = tf.saved_model.load("/Users/a_mat/Audentity/vggish")

def extractFeatures(audioFile):
    try:
        # Loading audio file
        waveform, sr = librosa.load(audioFile)

        # Trimming silence
        waveform, _ = librosa.effects.trim(waveform)

        # Extracting features using VGGish
        return vggish(waveform).numpy()
    except:
        return None

data = []

#extracting features
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    audio_files = glob(os.path.join(folder_path, "*.wav"))

    for file in tqdm(audio_files, desc=f'Processing {folder}'):
        file_path = os.path.join(folder_path, file)

        try:
            # Attempt to extract features and handle potential errors
            features = extractFeatures(file_path)
            if features is not None:
                data.append([features, folder])
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue  # Move on to the next file

data = pd.DataFrame(data, columns=['Features', 'Genre'])
data.head()

x = data['Features'].tolist()
x = pad_sequences(x, dtype='float32', padding='post', truncating='post')
x.shape

encoder = LabelEncoder()
y = encoder.fit_transform(data['Genre'])
y = to_categorical(y)

trainX, testX, trainY, testY = train_test_split(x, y, random_state = 0)

# defining the model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(43, 128, 1)),
    layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Flatten(),

    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 10,
    min_delta = 0.001,
    restore_best_weights = True
)

history = model.fit(
    trainX, trainY,
    validation_data = (testX, testY),
    epochs = 50,
    callbacks = [earlyStopping]
)

model.save('vggish_classifier.keras')

historyDf = pd.DataFrame(history.history)
historyDf.loc[:, ['accuracy', 'val_accuracy']].plot()

score = model.evaluate(testX, testY)[1] * 100
print(f'Validation accuracy of model : {score:.2f}%')