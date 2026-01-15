import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

def extract_mfcc(file_path, max_len=50):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten()

# Load dataset
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build DNN model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))
model.save("stutter_model.h5")
