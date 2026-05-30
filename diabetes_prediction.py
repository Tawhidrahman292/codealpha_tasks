import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load PIMA Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
           "BMI","DiabetesPedigreeFunction","Age","Outcome"]
data = pd.read_csv(url, names=columns)

print("✔ Dataset Loaded Successfully")

# Split features and labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Diabetes Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('diabetes_confusion_matrix.png')
print("✔ Confusion matrix saved as diabetes_confusion_matrix.png")

with open("results_diabetes.txt", "w") as f:
    f.write("Diabetes Prediction Results\n")
    f.write("===========================\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred):.4f}\n")
    f.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))

print("✔ Results saved to results_diabetes.txt")
