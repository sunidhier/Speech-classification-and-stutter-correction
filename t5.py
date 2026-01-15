import librosa
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Reproducibility
# -------------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_mfcc(file_path, max_len=50):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Pad or truncate
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.flatten()

# -------------------------------
# Load Dataset
# -------------------------------
stuttered_dir = 'dataset/stuttered/'
normal_dir = 'dataset/normal/'

X, y = [], []

for fname in os.listdir(stuttered_dir):
    if fname.endswith('.wav'):
        X.append(extract_mfcc(os.path.join(stuttered_dir, fname)))
        y.append(0)  # 0 = Stuttered

for fname in os.listdir(normal_dir):
    if fname.endswith('.wav'):
        X.append(extract_mfcc(os.path.join(normal_dir, fname)))
        y.append(1)  # 1 = Normal

X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# K-Fold Cross Validation
# -------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies, f1s = [], []

fold = 1
for train_idx, test_idx in kf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    fold += 1

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = to_categorical(y[train_idx]), to_categorical(y[test_idx])

    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        verbose=0
    )

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    accuracies.append(acc)
    f1s.append(f1)

    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

# -------------------------------
# Final Results
# -------------------------------
print("\n=== Cross-Validation Summary ===")
for i, (acc, f1) in enumerate(zip(accuracies, f1s), 1):
    print(f"Fold {i}: Accuracy={acc:.4f}, F1={f1:.4f}")

print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
print(f"Mean F1-score: {np.mean(f1s):.4f}")

# -------------------------------
# Plot Results
# -------------------------------
plt.plot(range(1, len(accuracies)+1), accuracies, marker='o', label="Accuracy")
plt.plot(range(1, len(f1s)+1), f1s, marker='s', label="F1-score")
plt.title("Cross-Validation Results")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.show()
