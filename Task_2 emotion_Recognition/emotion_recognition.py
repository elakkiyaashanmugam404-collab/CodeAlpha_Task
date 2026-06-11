import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset Path
dataset_path = "archive"

# Emotion Labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

# MFCC Feature Extraction
def extract_features(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfccs.T, axis=0)

# Read Dataset
for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                emotion_code = file.split("-")[2]
                emotion = emotion_dict[emotion_code]

                file_path = os.path.join(actor_path, file)

                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

# Convert to NumPy Arrays
X = np.array(X)

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Number of Emotions:", len(set(y)))

# Deep Learning Model
model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(40,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(8, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nDeep Learning Model Created Successfully!")

# Train Model
print("\nTraining Started...")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===== EMOTION RECOGNITION RESULTS =====")
print("Model Accuracy:", accuracy)

print("\nEmotion Recognition Project Completed Successfully!")