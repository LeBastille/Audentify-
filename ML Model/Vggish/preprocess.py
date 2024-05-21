from tqdm import tqdm
import os
import numpy as np
import librosa
import tensorflow as tf
import json

DATASET_PATH = '/Users/a_mat/Audentity/dataset/genres_original'

vggish = tf.saved_model.load("/Users/a_mat/Audentity/vggish")

data = {
    'mapping': [],
    'labels': [],
    'features': []
}

def extractFeatures(audioFile):
        try:
            # Loading audio file
            waveform, sr = librosa.load(audioFile, sr=16000)

            segment_length = 1.0
            num_segments = int(waveform.shape[0] / (sr * segment_length))
            features = []

            for i in range(num_segments):
                start = int(i*segment_length*sr)
                end = int(start+ sr * segment_length)
                segment = waveform[start:end]
                vgg_features = vggish(segment).numpy()
                features.append(vgg_features)

            # Extracting features using VGGish
            aggregated_features = np.mean(features, axis=0)
            return aggregated_features
        except:

            return None

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

    if dirpath is not DATASET_PATH:
        genre_label = dirpath.split("/")[-1]
        data["mapping"].append(genre_label)

        encoded_label = i-1

        for f in tqdm(filenames, desc=f'Processing {genre_label}'):
            file_path = os.path.join(dirpath, f)

            try:
                # Attempt to extract features and handle potential errors
                features = extractFeatures(file_path)
                if features is not None:
                    data['labels'].append(encoded_label)
                    features = features.flatten()
                    data['features'].append(features.tolist())
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")
                continue  # Move on to the next file

with open('dataval.json', 'w') as fp:
    json.dump(data, fp, indent=4)





