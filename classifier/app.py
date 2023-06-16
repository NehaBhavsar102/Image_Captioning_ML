from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
app.debug = True


app.config['UPLOAD_FOLDER'] = 'uploads'
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def classify_genre():
    audio_file = request.files['audio']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)

    # Create the uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    audio_file.save(file_path)
    genre = classify_audio_genre(file_path)
    return render_template('result.html', genre=genre)



   

def classify_audio_genre(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)

    # Perform feature extraction
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # Concatenate features into a single array
    features = np.concatenate((mfccs, chroma, mel, spectral_contrast), axis=0)

    # Transpose the features array
    features = np.transpose(features)

    # Normalize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Reshape the features to match the input shape of the model
   

    # Perform genre classification using the loaded model
    prediction = model.predict(features)
    predicted_genre = np.argmax(prediction)

    # Map the predicted genre index to the corresponding label
    genre_labels = ['classical', 'hip_hop', 'jazz', 'rock']
    predicted_genre_label = genre_labels[predicted_genre]

    return predicted_genre_label


if __name__ == '__main__':
    app.run(debug=True)




