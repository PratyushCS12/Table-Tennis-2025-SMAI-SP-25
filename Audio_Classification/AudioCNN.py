#!/usr/bin/env python3
"""
Hit Segmentation & CNN Classification: Racket, Table, No_Hit
-----------------------------------------------------------------
1. Splits each .wav into 100 ms clips for racket hits, table hits, and true silence.
2. Converts each clip into a fixed-size MFCC spectrogram for CNN input.
3. Trains a small 2D CNN on these spectrograms.
4. Exports:
   • labeled clips in `clips_out/{class}/file_start-end.wav`
   • CSV (`filepath,start_time,end_time,label`)
   • trained model (`--cnn_out`)
   • plots: PCA, confusion matrix, accuracy curve in `--plot_out`

Usage:
  python main.py \
    --audio_dir /path/to/clean_audio \
    --output clusters.csv \
    --clips_out clips \
    --cnn_out cnn_model.h5 \
    --plot_out plots \
    --nohit_per_file 100 \
    --rms_thresh 0.01 \
    --amp_thresh 0.005

Dependencies:
  pip install numpy pandas librosa soundfile tensorflow scikit-learn matplotlib tqdm joblib
"""
import os
import argparse
import numpy as np
np.complex = complex; np.float = float
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Settings
SR = 22050
PRE_SEC, POST_SEC = 0.02, 0.08
CLIP_LEN = PRE_SEC + POST_SEC
N_MFCC = 32
N_FFT = 512
HOP_LEN = 256
EXPECTED_FRAMES = int(np.ceil((CLIP_LEN * SR - N_FFT) / HOP_LEN)) + 1
DEFAULT_NOHIT = 100

# 1) Onset detection
def detect_onsets(y, sr):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    return librosa.frames_to_time(frames, sr=sr)

# 2) Build MFCC spectrogram tensor
def make_mfcc_spec(seg, sr):
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP_LEN)
    if mfcc.shape[1] < EXPECTED_FRAMES:
        pad = EXPECTED_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_FRAMES]
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc[..., np.newaxis]

# 3) Sample true-silence windows
def sample_nohit(y, sr, onsets, nohit_per_file, rms_thresh, amp_thresh):
    duration = len(y) / sr
    half = CLIP_LEN / 2
    centers = np.arange(half, duration - half, CLIP_LEN)
    valid = [c for c in centers if all(abs(c - o) > half for o in onsets)]
    if not valid:
        return []
    rng = np.random.default_rng(0)
    chosen = rng.choice(valid, size=min(nohit_per_file, len(valid)), replace=False)
    clips = []
    for c in chosen:
        start_t = c - half
        end_t = c + half
        st = int(start_t * sr)
        ed = st + int(CLIP_LEN * sr)
        seg = y[st:ed]
        if seg.size == 0:
            continue
        if np.abs(seg).max() >= amp_thresh:
            continue
        if np.sqrt(np.mean(seg**2)) >= rms_thresh:
            continue
        clips.append((start_t, end_t, seg))
    return clips

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir',      required=True)
    parser.add_argument('--output',         default='clusters.csv')
    parser.add_argument('--clips_out',      default='clips')
    parser.add_argument('--cnn_out',        default='cnn_model.h5')
    parser.add_argument('--plot_out',       default='plots')
    parser.add_argument('--nohit_per_file', type=int, default=DEFAULT_NOHIT)
    parser.add_argument('--rms_thresh',     type=float, default=0.01)
    parser.add_argument('--amp_thresh',     type=float, default=0.005)
    args = parser.parse_args()

    os.makedirs(args.clips_out, exist_ok=True)
    os.makedirs(args.plot_out, exist_ok=True)
    for cls in ['racket', 'table', 'no_hit']:
        os.makedirs(os.path.join(args.clips_out, cls), exist_ok=True)

    records = []
    specs = []
    labels = []
    sc_list = []

    for fn in tqdm(sorted(os.listdir(args.audio_dir)), desc='Files'):
        if not fn.lower().endswith('.wav'):
            continue
        path = os.path.join(args.audio_dir, fn)
        y, sr_native = sf.read(path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_native != SR:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=SR)

        onsets = detect_onsets(y, SR)
        hits = []
        for t in onsets:
            start_t = max(t - PRE_SEC, 0)
            end_t = start_t + CLIP_LEN
            st = int(start_t * SR)
            ed = st + int(CLIP_LEN * SR)
            seg = y[st:ed]
            if seg.size == 0:
                continue
            sc = librosa.feature.spectral_centroid(y=seg, sr=SR).mean()
            hits.append((start_t, end_t, seg, sc))
            sc_list.append(sc)

        thr = np.median(sc_list) if sc_list else 0
        for start_t, end_t, seg, sc in hits:
            label = 'racket' if sc >= thr else 'table'
            base = os.path.splitext(fn)[0]
            name = f"{base}_{start_t:.3f}-{end_t:.3f}.wav"
            outp = os.path.join(args.clips_out, label, name)
            max_abs = np.abs(seg).max()
            if max_abs < 1e-8:
                continue
            seg_norm = seg / max_abs
            if not np.all(np.isfinite(seg_norm)):
                continue
            sf.write(outp, (seg_norm * 32767).astype(np.int16), SR)
            spec = make_mfcc_spec(seg_norm, SR)
            specs.append(spec)
            labels.append(label)
            records.append((fn, start_t, end_t, label))

        no_clips = sample_nohit(y, SR, onsets, args.nohit_per_file,
                                 args.rms_thresh, args.amp_thresh)
        for start_t, end_t, seg in no_clips:
            base = os.path.splitext(fn)[0]
            name = f"{base}_{start_t:.3f}-{end_t:.3f}.wav"
            outp = os.path.join(args.clips_out, 'no_hit', name)
            max_abs = np.abs(seg).max()
            if max_abs < 1e-8:
                continue
            seg_norm = seg / max_abs
            if not np.all(np.isfinite(seg_norm)):
                continue
            sf.write(outp, (seg_norm * 32767).astype(np.int16), SR)
            spec = make_mfcc_spec(seg_norm, SR)
            specs.append(spec)
            labels.append('no_hit')
            records.append((fn, start_t, end_t, 'no_hit'))

    pd.DataFrame(records,
        columns=['filepath','start_time','end_time','label']
    ).to_csv(args.output, index=False)
    print(f"Saved CSV to {args.output}")

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    X = np.stack(specs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=0, stratify=y_enc)

    num_classes = len(le.classes_)
    input_shape = X_train.shape[1:]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D((2,2), padding='same'),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D((2,2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    model.save(args.cnn_out)
    print(f"Saved CNN model to {args.cnn_out}")

    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.plot_out,'accuracy_curve.png'), dpi=300)
    plt.show()

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap='Blues'); plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, le.classes_, rotation=45)
    plt.yticks(ticks, le.classes_)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    thresh = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i,j], ha='center', va='center',
                     color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_out,'confusion_matrix.png'), dpi=300)
    plt.show()

    emb = PCA(n_components=2).fit_transform(X_test.reshape(len(X_test), -1))
    plt.figure(figsize=(6,5))
    for idx, cls in enumerate(le.classes_):
        sel = np.where(y_test==idx)
        plt.scatter(emb[sel,0], emb[sel,1], label=cls, alpha=0.6)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('PCA of Test Set Features')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(args.plot_out,'pca_test.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
