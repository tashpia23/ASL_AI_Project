import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

dataset_path = "dataset"

X = []
y = []

for file in os.listdir(dataset_path):
    if file.endswith(".npy"):
        label = file.split(".")[0]
        data = np.load(os.path.join(dataset_path, file))
        for sample in data:
            X.append(sample)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Classes:", sorted(set(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as asl_model.pkl")