import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("german.data", sep=" ", header=None)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# Convert categorical values into numerical values
le = LabelEncoder()

for col in data.columns:
    data[col] = le.fit_transform(data[col].astype(str))

# Features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LogisticRegression(max_iter=2000)

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===== CREDIT SCORING MODEL =====")
print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nCredit Scoring Model Completed Successfully!")