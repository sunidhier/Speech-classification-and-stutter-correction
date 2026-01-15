import matplotlib
matplotlib.use('Agg')   # <<-- use non-GUI backend to avoid Tkinter errors

import os
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.effects

import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import pyttsx3

from tensorflow.keras.models import load_model

# ---------------------------
# CONFIG (tuned for medium voice + noisy classroom)
# ---------------------------
MODEL_PATH = "stutter_model.h5"
SCALER_PATH = "scaler.pkl"
TEMP_WAV = "input.wav"
RECORD_SECONDS = 6
FS = 44100
N_MFCC = 13
MAX_LEN = 50

WAVEFORM_PNG = "waveform.png"
MFCC_PNG = "mfcc.png"

# ---------------------------
# RECORD AUDIO
# ---------------------------
def record_audio(filename=TEMP_WAV, duration=RECORD_SECONDS, fs=FS):
    print(f"[+] Recording {duration}s... (speak clearly when prompted)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print("[+] Recording saved to", filename)
    return filename

# ---------------------------
# SILENCE DETECTION (voice-activity aware)
# ---------------------------
def is_silence(y, sr, min_voiced_sec=0.25):
    """
    Returns True if voiced (non-silent) duration is less than min_voiced_sec.
    Uses librosa.effects.split which is more robust in noisy environments.
    """
    if y is None or len(y) == 0:
        return True
    # y is shape (n,) or (n,1)
    if y.ndim > 1:
        y_mono = np.squeeze(y)
    else:
        y_mono = y
    # return voiced intervals (start, end) in samples
    intervals = librosa.effects.split(y_mono, top_db=30)  # tolerant threshold for noisy rooms
    voiced_duration = sum((end - start) for start, end in intervals) / float(sr)
    return voiced_duration < min_voiced_sec

# ---------------------------
# EXTRACT MFCC + SAVE PLOTS (no GUI)
# ---------------------------
def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN, show_plots=True):
    y, sr = librosa.load(file_path, sr=None)

    # Waveform -> save as PNG
    if show_plots:
        plt.figure(figsize=(10, 2.5))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.tight_layout()
        try:
            plt.savefig(WAVEFORM_PNG)
        except Exception as e:
            print("[!] Failed to save waveform png:", e)
        plt.close()

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad / truncate
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    # MFCC plot -> save as PNG
    if show_plots:
        plt.figure(figsize=(8, 3.5))
        librosa.display.specshow(mfcc, x_axis="time")
        plt.colorbar()
        plt.title("MFCC")
        plt.tight_layout()
        try:
            plt.savefig(MFCC_PNG)
        except Exception as e:
            print("[!] Failed to save mfcc png:", e)
        plt.close()

    return mfcc, mfcc.flatten(), y, sr

# ---------------------------
# SPEECH TO TEXT
# ---------------------------
def speech_to_text(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print("[STT] Recognized:", text)
        return text
    except sr.UnknownValueError:
        print("[STT] Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print("[STT] API error:", e)
        return ""

# ---------------------------
# IMPROVED TEXT RULE (robust but conservative)
# ---------------------------
def detect_stutter_text(text):
    """
    Returns True if the transcribed text shows strong stutter patterns.
    Conservative enough to avoid false positives from noisy STT output,
    but will detect typical demo patterns like "b-b-boy", "I I I", "soooo".
    """
    if not text or not isinstance(text, str):
        return False

    t = text.lower().strip()

    # 1) repeated letters (prolongation) - require 3+ repeats (strong)
    if re.search(r'([a-z])\1{2,}', t):
        return True

    # 2) repeated words (word repetition) - require at least two repeats
    if re.search(r'\b(\w+)(\s+\1\b){1,}', t):
        return True

    # 3) dashed or hyphenated patterns: b-b-boy, ba-ba-banana
    if re.search(r'\b(?:[a-z]{1,4}-){1,}[a-z]{1,}\b', t):
        return True

    # 4) syllable repetition patterns: "ba ba" or "ba ba ba"
    if re.search(r'\b([a-z]{1,3})( \1){1,}\b', t):
        return True

    return False

# ---------------------------
# IMPROVED AUDIO RULE (voice-activity aware, noise-resistant)
# ---------------------------
def detect_stutter_audio(y, sr, window_short=0.18, min_clusters=2):
    """
    Detects repeated rapid onsets inside voiced segments only.
    This ignores background noise by focusing on voiced intervals.
    """
    if y is None or len(y) == 0:
        return False

    # convert to mono
    if y.ndim > 1:
        y_mono = np.squeeze(y)
    else:
        y_mono = y

    # find voiced intervals (ignore background)
    intervals = librosa.effects.split(y_mono, top_db=30)
    if len(intervals) == 0:
        return False

    close = 0
    for (start, end) in intervals:
        segment = y_mono[start:end]
        # detect onsets in this voiced segment
        try:
            onsets = librosa.onset.onset_detect(y=segment, sr=sr)
        except Exception:
            continue
        if len(onsets) < 2:
            continue
        times = librosa.frames_to_time(onsets, sr=sr)
        # count close-onset pairs in this segment
        for i in range(1, len(times)):
            if (times[i] - times[i-1]) <= window_short:
                close += 1
    return close >= min_clusters

# ---------------------------
# TEXT CORRECTION (conservative)
# ---------------------------
def correct_stuttered_text(text):
    if not text:
        return text
    # remove repeated words: "I I I want" -> "I want"
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # shorten repeated letters/prolongation: "soooo" -> "so"
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)
    # remove dashed patterns: "b-b-boy" -> "boy"
    text = re.sub(r'\b(?:\w+-){1,}(\w+)', r'\1', text)
    return text

# ---------------------------
# TTS
# ---------------------------
def speak_text(text, rate=150):
    if not text:
        return
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

# ---------------------------
# LOAD MODEL & SCALER
# ---------------------------
def load_model_and_scaler():
    model = None
    scaler = None

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("[+] Model loaded")
        except Exception as e:
            print("[-] Failed to load model:", e)
            model = None

    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("[+] Scaler loaded")
        except Exception as e:
            print("[-] Failed to load scaler:", e)
            scaler = None

    return model, scaler

# ---------------------------
# MODEL PREDICT (unchanged)
# ---------------------------
def model_predict(model, scaler, mfcc_flat):
    if model is None:
        return None
    x = mfcc_flat.reshape(1, -1)
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception:
            pass
    try:
        pred = model.predict(x)
        return pred[0]
    except Exception:
        return None

# ---------------------------
# FINAL DECISION (CONSERVATIVE for noisy demo)
# ---------------------------
def combine_decision(model_probs, text_rule, audio_rule):
    if text_rule and audio_rule:
        return True
    if model_probs is not None:
        try:
            p_stutter = float(model_probs[0])
            if p_stutter > 0.80:
                return True
        except Exception:
            pass
    # otherwise: normal
    return False

# ---------------------------
# MAIN
# ---------------------------
def main():
    # reference screenshot (for slides / README)
    print("Reference screenshot (for slides): /mnt/data/Screenshot 2025-11-24 224326.png")

    print("=== DEMO-SAFE HYBRID STUTTER DETECTION (Option 2: save plots only) ===")
    choice = input("Record audio? (Y/n): ").strip().lower()
    if choice in ["", "y", "yes"]:
        audio_file = record_audio()
    else:
        audio_file = input("Enter WAV path: ").strip()
        if not os.path.exists(audio_file):
            print("File not found:", audio_file)
            return

    mfcc_mat, mfcc_flat, y, sr = extract_mfcc(audio_file)

    # silence detection
    if is_silence(y, sr):
        print("\n[!] No recorded speech detected.")
        speak_text("No recorded speech detected. Please try again.")
        return

    # speech-to-text (STT)
    text = speech_to_text(audio_file)

    # run rules
    text_rule = detect_stutter_text(text)
    audio_rule = detect_stutter_audio(y, sr)

    print("Text Rule:", text_rule)
    print("Audio Rule:", audio_rule)

    # load model and predict (if available)
    model, scaler = load_model_and_scaler()
    model_probs = model_predict(model, scaler, mfcc_flat)

    final_stutter = combine_decision(model_probs, text_rule, audio_rule)

    print("\n=== FINAL RESULT ===")
    if final_stutter:
        print("Detected: STUTTERED SPEECH")
        speak_text("Stuttered speech detected")
        corrected = correct_stuttered_text(text)
    else:
        print("Detected: NORMAL SPEECH")
        speak_text("Normal speech detected")
        corrected = text

    print("Corrected Text:", corrected)

    # NOTE: plots saved as PNGs in the current working directory:
    print(f"\nSaved plots: {os.path.abspath(WAVEFORM_PNG)}, {os.path.abspath(MFCC_PNG)}")
    speak_text(corrected)

    print("\nDone.")

if __name__ == "__main__":
    main()

