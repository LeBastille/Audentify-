# Audentify-
A web-application designed to perform music genre classification. The application uses VGGish, a deep learning model (CNN) made by Google for the purpose of audio feature extraction combined with an SVM classifier to make the final genre predictions

FEATURES:
- Real Time Genre Prediction: Can capture audio from your device's microphone and use that audio to determine the genre
- File Upload Classification: Upload a song file (The supported format is WAV) and recieve the genre prediction
- User Accounts & History: Each individual user can create their own account
- Recommendations: Song recommendations are provided based on whichever genre has been predicted or even using the search bar by typing in the genre name

Technologies Used:

MACHINE LEARNING:
. VGGish - Google's Pre Trained model for audio feature extraction
. SVM - Support Vector Machine Classifier for genre prediction

BACKEND:
.Flask: Python web framework for handling requests and building the API.
.TensorFlow: For loading and running the VGGish model.
.scikit-learn: For the SVM implementation and evaluation.
.PyAudio: For audio recording.
.Librosa: For audio manipulation and processing.
.soundfile: For saving audio data as WAV files.

FRONTEND:
HTML, CSS, JavaScript: For building the user interface and interacting with the backend.


