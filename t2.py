import librosa
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Build DNN Model
# -------------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# -------------------------------
# Train Model
# -------------------------------
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[es],
    verbose=1
)

# Save model
model.save("stutter_model.h5")

# -------------------------------
# Evaluation with F1 Score
# -------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute F1 scores
f1_macro = f1_score(y_true, y_pred_classes, average='macro')
f1_weighted = f1_score(y_true, y_pred_classes, average='weighted')

print("\nF1 Score (Macro):", f1_macro)
print("F1 Score (Weighted):", f1_weighted)

# Detailed report
print("\nClassification Report:\n", classification_report(
    y_true, y_pred_classes, target_names=["Stuttered", "Normal"]
))

# Confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred_classes))
