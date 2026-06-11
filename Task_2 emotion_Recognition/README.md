Emotion Recognition from Speech

Project Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The objective of this project is to recognize human emotions from speech audio using Deep Learning and Speech Signal Processing techniques. The model extracts MFCC (Mel-Frequency Cepstral Coefficients) features from audio files and classifies emotions such as happy, sad, angry, neutral, fearful, disgust, calm, and surprised.

---

Dataset

Dataset Used: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

The dataset contains speech recordings from multiple actors expressing different emotions.

Emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

Technologies Used

- Python
- NumPy
- Librosa
- Scikit-learn
- TensorFlow
- Keras

---

Project Workflow

1. Data Collection

Speech audio files were collected from the RAVDESS dataset.

2. Feature Extraction

MFCC (Mel-Frequency Cepstral Coefficients) features were extracted from audio files using Librosa.

3. Data Preprocessing

- Emotion labels were extracted from file names.
- Labels were encoded using LabelEncoder.
- Dataset was split into training and testing sets.

4. Deep Learning Model

A Neural Network was built using TensorFlow/Keras with:

- Dense Layer (128 neurons)
- Dense Layer (64 neurons)
- Output Layer (8 emotion classes)

5. Model Evaluation

The trained model was evaluated using test data and classification accuracy was calculated.

---

Output

Example Output:

Training Samples: 1152

Testing Samples: 288

Number of Emotions: 8

Deep Learning Model Created Successfully!

Training Started...

Model Accuracy: 0.39

Emotion Recognition Project Completed Successfully!

---

Project Structure

Emotion_Recognition/

│

├── dataset/

│ └── archive (1)/

│

├── emotion_recognition.py

│

└── README.md

---

Future Improvements

- Improve model accuracy using CNN or LSTM.
- Perform audio augmentation.
- Use larger datasets.
- Build a real-time emotion recognition application.

---

Author

ELAKKIYAA S

CodeAlpha Machine Learning Intern