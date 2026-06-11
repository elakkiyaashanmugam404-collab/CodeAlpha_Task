Handwritten Character Recognition

Project Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The objective of this project is to recognize handwritten digits using Image Processing and Deep Learning techniques. A Convolutional Neural Network (CNN) is used to classify handwritten digits from the MNIST dataset.

---

Dataset

Dataset Used: MNIST Dataset

The dataset contains 70,000 handwritten digit images (0–9).

- Training Images: 60,000
- Testing Images: 10,000

---

Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

Project Workflow

1. Dataset Loading

The MNIST dataset is loaded using TensorFlow.

2. Image Preprocessing

- Normalize pixel values.
- Reshape images for CNN input.

3. CNN Model

The model consists of:

- Convolution Layer
- Max Pooling Layer
- Flatten Layer
- Dense Layer
- Output Layer

4. Model Training

The CNN model is trained on handwritten digit images.

5. Model Evaluation

The trained model is evaluated using test data and accuracy is calculated.

---

Output

Example Output:

Dataset Loaded Successfully!

Training Samples: 60000

Testing Samples: 10000

CNN Model Created Successfully!

Model Accuracy: 0.98

Project Completed Successfully!

---

Project Structure

Task_3_Handwritten_Character_Recognition/

├── handwritten_character_recognition.py

└── README.md

---

Future Improvements

- Character recognition using EMNIST dataset.
- Real-time handwritten text recognition.
- Word and sentence recognition using CRNN.

---

Author

ELAKKIYAA .S

CodeAlpha Machine Learning Intern