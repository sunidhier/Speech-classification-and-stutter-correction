# Speech Disfluency Detection & Assistive Audio Correction System

A machine learning-based assistive speech-processing system designed to detect speech disfluencies and generate corrected speech output using acoustic feature extraction, neural networks, and linguistic pattern matching.

## Overview

This project implements a hybrid approach to identifying speech disfluencies such as repetitions, prolongations, and blocks. It combines audio processing, machine learning, acoustic analysis, and text-to-speech synthesis to support more fluent speech output.

## Key Features

- **MFCC Feature Extraction:** Extracts 13-dimensional Mel-Frequency Cepstral Coefficients across 50 time frames, producing 650-dimensional feature vectors.
- **Audio Preprocessing:** Applies Z-score standardization and silence thresholding to support robust feature processing.
- **Deep Learning Classification:** Uses a Multi-Layer Perceptron (MLP) with dropout regularization and early stopping.
- **Hybrid Detection:** Combines neural network predictions, acoustic onset clustering, and regex-based lexical pattern matching.
- **Text Correction:** Applies text-processing logic to identify and correct supported disfluency patterns.
- **Speech Synthesis:** Uses `pyttsx3` to generate spoken output from processed text.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Audio Processing | Librosa, MFCC |
| Text Processing | Regular Expressions (Regex) |
| Text-to-Speech | pyttsx3 |

## Model Performance

The model achieved the following reported results using 5-fold stratified cross-validation:

- **Accuracy:** 98.9%
- **F1-score:** 98.9%

These results reflect the evaluation performed for this project and may vary with different datasets and testing conditions.

## System Workflow

1. Capture or load speech audio.
2. Process the audio and apply speech or silence detection.
3. Extract MFCC features from the audio signal.
4. Standardize features and prepare them for classification.
5. Detect potential disfluencies using the MLP and complementary acoustic and lexical analysis.
6. Apply text-correction logic to supported disfluency patterns.
7. Generate speech output using text-to-speech synthesis.

## Installation and Setup

### Prerequisites

- Python
- pip
- Required audio-processing and machine learning dependencies

### 1. Clone the Repository

```bash
git clone https://github.com/sunidhier/Speech-classification-and-stutter-correction.git
cd Speech-classification-and-stutter-correction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project configuration.

### 4. Run the Project

Run the appropriate Python script from the repository. The entry-point filename and any required model files depend on the project implementation.

**Note:** Audio recording may require microphone permissions, audio-device configuration, and additional system dependencies.

## Copyright Registration

The software work titled **"Hybrid model to detect and correct stuttered speech"** is registered under the Copyright Act, 1957.

- **Certificate Number:** SW-2026023958
- **Application Number:** SW-16774/2026-CO
- **Date of Registration:** 19 October 2026
- **Ownership:** E R Sunidhi is listed as a co-owner in the registration certificate.

## Author

**E R Sunidhi , Monika B M , Manya C**

GitHub: [@sunidhier](https://github.com/sunidhier)

## Acknowledgements

Developed as a collaborative academic project focused on applying machine learning and speech processing to assistive technology.

---

*Exploring machine learning for more accessible and fluent speech communication.*

© 2026 E R Sunidhi and the other applicable rights holders. All rights reserved.
This project is protected by copyright. No permission is granted to reproduce, modify, redistribute, or commercially use the work except as permitted by applicable law or with permission from the relevant rights holders.
