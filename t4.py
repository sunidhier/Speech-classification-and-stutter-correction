import librosa
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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
        y.append(0)  # 0 for stuttered

for fname in os.listdir(normal_dir):
    if fname.endswith('.wav'):
        X.append(extract_mfcc(os.path.join(normal_dir, fname)))
        y.append(1)  # 1 for normal

X = np.array(X)
y = to_categorical(np.array(y))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Train/Test Splits Evaluation
# -------------------------------
split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # test sizes
results = []

for test_ratio in split_ratios:
    print(f"\n=== Train/Test Split {int((1-test_ratio)*100)}/{int(test_ratio*100)} ===")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    # Build model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    
    # Train
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[es],
        verbose=0
    )
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Accuracy & F1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    results.append((1-test_ratio, test_ratio, acc, f1))
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

# -------------------------------
# Summary of Results
# -------------------------------
print("\n=== Final Results (Train%, Test%, Accuracy, F1-score) ===")
for train_p, test_p, acc, f1 in results:
    print(f"Train {int(train_p*100)}% | Test {int(test_p*100)}% → Accuracy: {acc:.4f}, F1: {f1:.4f}")

# -------------------------------
# Plot Accuracy vs Train Size
# -------------------------------
train_sizes = [int(train_p*100) for train_p, _, _, _ in results]
accuracies = [acc for _, _, acc, _ in results]

plt.plot(train_sizes, accuracies, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs Train Size")
plt.xlabel("Training Set Percentage (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
