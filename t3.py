import librosa
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

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
        y.append(0)  # 0 for stuttered

for fname in os.listdir(normal_dir):
    if fname.endswith('.wav'):
        X.append(extract_mfcc(os.path.join(normal_dir, fname)))
        y.append(1)  # 1 for normal

X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# K-Fold Cross Validation
# -------------------------------
k = 5  # 5-fold cross validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)

all_f1_scores = []

fold = 1
for train_index, test_index in kf.split(X):
    print(f"\n===== Fold {fold} =====")
    fold += 1
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # One-hot encoding for training
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # Build model
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=0, validation_data=(X_test, y_test_cat))
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # F1 score
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    all_f1_scores.append(f1)
    
    # Report for each fold
    print("F1 Score (Macro):", f1)
    print("Classification Report:\n", classification_report(y_test, y_pred_classes, target_names=["Stuttered", "Normal"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

# -------------------------------
# Final Results
# -------------------------------
print("\n===== Final Cross-Validation Results =====")
print("F1 Scores across folds:", all_f1_scores)
print("Mean F1 Score:", np.mean(all_f1_scores))
print("Std Dev of F1:", np.std(all_f1_scores))
