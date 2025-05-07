# import os, argparse
# import numpy as np
# # compatibility
# np.complex = complex; np.float = float
# import pandas as pd
# import soundfile as sf
# import librosa
# from tqdm import tqdm
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import joblib

# # Settings
# SR = 22050
# PRE_SEC, POST_SEC = 0.02, 0.08
# DEFAULT_NOHIT = 100

# # 1. Onset detection
# def detect_onsets(y, sr):
#     o_env = librosa.onset.onset_strength(y=y, sr=sr)
#     frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
#     return librosa.frames_to_time(frames, sr=sr)

# # 2. Feature extraction (MFCC, spectral centroid, ZCR)
# def extract_features(y, sr, start_t, end_t):
#     start = int(start_t * sr)
#     end = int(end_t * sr)
#     seg = y[start:end]
#     if seg.size == 0:
#         return None, None, None, None
#     n_fft = min(2048, len(seg))
#     mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=n_fft)
#     feats = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
#     sc = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=n_fft).mean()
#     zcr = librosa.feature.zero_crossing_rate(y=seg, frame_length=n_fft, hop_length=n_fft//2)[0].mean()
#     return feats, sc, zcr, seg

# # 3. Sample no_hit windows (avoid onsets + amplitude criterion)
# def sample_nohit(y, sr, onsets, clip_len, nohit_per_file, rms_thresh, amp_thresh):
#     duration = len(y) / sr
#     half = clip_len / 2
#     centers = np.arange(half, duration-half, clip_len)
#     valid = [c for c in centers if all(abs(c-o) > half for o in onsets)]
#     if not valid:
#         return []
#     rng = np.random.default_rng(0)
#     chosen = rng.choice(valid, size=min(nohit_per_file, len(valid)), replace=False)
#     clips = []
#     for c in chosen:
#         start_t = c - half
#         end_t = c + half
#         start, end = int(start_t * sr), int(end_t * sr)
#         seg = y[start:end]
#         if seg.size == 0: continue
#         if np.max(np.abs(seg)) >= amp_thresh: continue
#         rms = np.sqrt(np.mean(seg**2))
#         if rms >= rms_thresh: continue
#         clips.append((start_t, end_t, seg))
#     return clips

# # Main

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--audio_dir',      required=True)
#     p.add_argument('--output',         default='clusters.csv')
#     p.add_argument('--clips_out',      default='clips')
#     p.add_argument('--svm_out')
#     p.add_argument('--plot_out',       default='plots')
#     p.add_argument('--nohit_per_file', type=int,   default=DEFAULT_NOHIT)
#     p.add_argument('--rms_thresh',     type=float, default=0.01)
#     p.add_argument('--amp_thresh',     type=float, default=0.005)
#     args = p.parse_args()

#     clip_len = PRE_SEC + POST_SEC
#     # Prepare directories
#     os.makedirs(args.clips_out, exist_ok=True)
#     os.makedirs(args.plot_out, exist_ok=True)
#     for cls in ['racket','table','no_hit']:
#         os.makedirs(os.path.join(args.clips_out, cls), exist_ok=True)

#     records, feats_all, labels_all = [], [], []
#     sc_values = []

#     # Process each file
#     for fn in tqdm(sorted(os.listdir(args.audio_dir)), desc='Files'):
#         if not fn.lower().endswith('.wav'): continue
#         path = os.path.join(args.audio_dir, fn)
#         y, sr_nat = sf.read(path, dtype='float32')
#         if y.ndim > 1: y = y.mean(axis=1)
#         if sr_nat != SR: y = librosa.resample(y, orig_sr=sr_nat, target_sr=SR)

#         # 1) Hit detection
#         onsets = detect_onsets(y, SR)
#         temp = []
#         for t in onsets:
#             start_t = max(t - PRE_SEC, 0)
#             end_t = start_t + clip_len
#             feats, sc, zcr, seg = extract_features(y, SR, start_t, end_t)
#             if feats is not None:
#                 temp.append((start_t, end_t, feats, sc, zcr, seg))
#                 sc_values.append(sc)

#         # compute threshold
#         thr = np.median(sc_values) if sc_values else 0
#         # save hits
#         for start_t, end_t, feats, sc, zcr, seg in temp:
#             lbl = 'racket' if sc >= thr else 'table'
#             base = os.path.splitext(fn)[0]
#             start_s = f"{start_t:.3f}"; end_s = f"{end_t:.3f}"
#             outp = os.path.join(args.clips_out, lbl, f"{base}_{start_s}-{end_s}.wav")
#             segn = seg / np.max(np.abs(seg)) if seg.max()!=seg.min() else seg
#             sf.write(outp, (segn*32767).astype(np.int16), SR, subtype='PCM_16')
#             records.append((fn, start_t, end_t, lbl))
#             feats_all.append(np.hstack([feats, sc, zcr])); labels_all.append(lbl)

#         # 2) Sample no_hit
#         no_clips = sample_nohit(y, SR, onsets, clip_len, args.nohit_per_file,
#                                  args.rms_thresh, args.amp_thresh)
#         for start_t, end_t, seg in no_clips:
#             base = os.path.splitext(fn)[0]
#             start_s = f"{start_t:.3f}"; end_s = f"{end_t:.3f}"
#             outp = os.path.join(args.clips_out, 'no_hit', f"{base}_{start_s}-{end_s}.wav")
#             segn = seg / np.max(np.abs(seg)) if seg.max()!=seg.min() else seg
#             sf.write(outp, (segn*32767).astype(np.int16), SR, subtype='PCM_16')
#             feats, sc_n, zcr_n, _ = extract_features(y, SR, start_t, end_t)
#             records.append((fn, start_t, end_t, 'no_hit'))
#             feats_all.append(np.hstack([feats, sc_n, zcr_n])); labels_all.append('no_hit')

#     # Save CSV with window times
#     df = pd.DataFrame(records, columns=['filepath','start_time','end_time','label'])
#     df.to_csv(args.output, index=False)
#     print(f"Saved CSV: {args.output}")

#     # Train SVM
#     le = LabelEncoder(); y_enc = le.fit_transform(labels_all)
#     X_arr = np.vstack(feats_all)
#     svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')
#     scores = cross_val_score(svm, X_arr, y_enc, cv=5)
#     print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
#     svm.fit(X_arr, y_enc)
#     if args.svm_out:
#         joblib.dump((svm, le), args.svm_out)
#         print(f"Saved SVM to {args.svm_out}")

#     # Visualization
#     X2 = PCA(n_components=2).fit_transform(X_arr)
#     fig, ax = plt.subplots(figsize=(8,6))
#     for cls in le.classes_:
#         idx = np.where(np.array(labels_all)==cls)
#         ax.scatter(X2[idx,0], X2[idx,1], label=cls, s=40, alpha=0.7, edgecolors='k')
#     ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('PCA by Class')
#     ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
#     fig.savefig(os.path.join(args.plot_out,'pca.png'), dpi=300); plt.show()

#     preds = svm.predict(X_arr)
#     cm = confusion_matrix(y_enc, preds)
#     fig2, ax2 = plt.subplots(figsize=(5,4))
#     im = ax2.imshow(cm, cmap='Blues'); fig2.colorbar(im, ax=ax2)
#     ticks = np.arange(len(le.classes_))
#     ax2.set_xticks(ticks); ax2.set_xticklabels(le.classes_, rotation=45)
#     ax2.set_yticks(ticks); ax2.set_yticklabels(le.classes_)
#     ax2.set_xlabel('Predicted'); ax2.set_ylabel('True'); ax2.set_title('Confusion Matrix')
#     thresh = cm.max()/2
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax2.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>thresh else 'black')
#     fig2.tight_layout(); fig2.savefig(os.path.join(args.plot_out,'confusion_matrix.png'), dpi=300); plt.show()

# if __name__=='__main__':
#     main()





import os, argparse
import numpy as np
# compatibility
np.complex = complex; np.float = float
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Settings
SR = 22050
PRE_SEC, POST_SEC = 0.02, 0.08
DEFAULT_NOHIT = 100

# 1. Onset detection
def detect_onsets(y, sr):
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    return librosa.frames_to_time(frames, sr=sr)

# 2. Feature extraction (MFCC, spectral centroid, ZCR)
def extract_features(y, sr, start_t, end_t):
    start = int(start_t * sr)
    end = int(end_t * sr)
    seg = y[start:end]
    if seg.size == 0:
        return None, None, None, None
    n_fft = min(2048, len(seg))
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=n_fft)
    feats = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
    sc = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=n_fft).mean()
    zcr = librosa.feature.zero_crossing_rate(y=seg, frame_length=n_fft, hop_length=n_fft//2)[0].mean()
    return feats, sc, zcr, seg

# 3. Sample no_hit windows (avoid onsets + amplitude criterion)
def sample_nohit(y, sr, onsets, clip_len, nohit_per_file, rms_thresh, amp_thresh):
    duration = len(y) / sr
    half = clip_len / 2
    centers = np.arange(half, duration-half, clip_len)
    valid = [c for c in centers if all(abs(c-o) > half for o in onsets)]
    if not valid:
        return []
    rng = np.random.default_rng(0)
    chosen = rng.choice(valid, size=min(nohit_per_file, len(valid)), replace=False)
    clips = []
    for c in chosen:
        start_t = c - half
        end_t = c + half
        start, end = int(start_t * sr), int(end_t * sr)
        seg = y[start:end]
        if seg.size == 0: continue
        if np.max(np.abs(seg)) >= amp_thresh: continue
        rms = np.sqrt(np.mean(seg**2))
        if rms >= rms_thresh: continue
        clips.append((start_t, end_t, seg))
    return clips

# Main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio_dir',      required=True)
    p.add_argument('--output',         default='clusters.csv')
    p.add_argument('--clips_out',      default='clips')
    p.add_argument('--svm_out')
    p.add_argument('--plot_out',       default='plots')
    p.add_argument('--nohit_per_file', type=int,   default=DEFAULT_NOHIT)
    p.add_argument('--rms_thresh',     type=float, default=0.01)
    p.add_argument('--amp_thresh',     type=float, default=0.005)
    args = p.parse_args()

    clip_len = PRE_SEC + POST_SEC
    # Prepare directories
    os.makedirs(args.clips_out, exist_ok=True)
    os.makedirs(args.plot_out, exist_ok=True)
    for cls in ['racket','table','no_hit']:
        os.makedirs(os.path.join(args.clips_out, cls), exist_ok=True)

    records, feats_all, labels_all = [], [], []
    sc_values = []

    # Process each file
    for fn in tqdm(sorted(os.listdir(args.audio_dir)), desc='Files'):
        if not fn.lower().endswith('.wav'): continue
        path = os.path.join(args.audio_dir, fn)
        y, sr_nat = sf.read(path, dtype='float32')
        if y.ndim > 1: y = y.mean(axis=1)
        if sr_nat != SR: y = librosa.resample(y, orig_sr=sr_nat, target_sr=SR)

        # 1) Hit detection
        onsets = detect_onsets(y, SR)
        temp = []
        for t in onsets:
            start_t = max(t - PRE_SEC, 0)
            end_t = start_t + clip_len
            feats, sc, zcr, seg = extract_features(y, SR, start_t, end_t)
            if feats is not None:
                temp.append((start_t, end_t, feats, sc, zcr, seg))
                sc_values.append(sc)

        # compute threshold
        thr = np.median(sc_values) if sc_values else 0
        # save hits
        for start_t, end_t, feats, sc, zcr, seg in temp:
            lbl = 'racket' if sc >= thr else 'table'
            base = os.path.splitext(fn)[0]
            start_s = f"{start_t:.3f}"; end_s = f"{end_t:.3f}"
            outp = os.path.join(args.clips_out, lbl, f"{base}_{start_s}-{end_s}.wav")
            segn = seg / np.max(np.abs(seg)) if seg.max()!=seg.min() else seg
            sf.write(outp, (segn*32767).astype(np.int16), SR, subtype='PCM_16')
            records.append((fn, start_t, end_t, lbl))
            feats_all.append(np.hstack([feats, sc, zcr])); labels_all.append(lbl)

        # 2) Sample no_hit
        no_clips = sample_nohit(y, SR, onsets, clip_len, args.nohit_per_file,
                                 args.rms_thresh, args.amp_thresh)
        for start_t, end_t, seg in no_clips:
            base = os.path.splitext(fn)[0]
            start_s = f"{start_t:.3f}"; end_s = f"{end_t:.3f}"
            outp = os.path.join(args.clips_out, 'no_hit', f"{base}_{start_s}-{end_s}.wav")
            segn = seg / np.max(np.abs(seg)) if seg.max()!=seg.min() else seg
            sf.write(outp, (segn*32767).astype(np.int16), SR, subtype='PCM_16')
            feats, sc_n, zcr_n, _ = extract_features(y, SR, start_t, end_t)
            records.append((fn, start_t, end_t, 'no_hit'))
            feats_all.append(np.hstack([feats, sc_n, zcr_n])); labels_all.append('no_hit')

    # Save CSV with window times
    df = pd.DataFrame(records, columns=['filepath','start_time','end_time','label'])
    df.to_csv(args.output, index=False)
    print(f"Saved CSV: {args.output}")

    # Train SVM
    le = LabelEncoder(); y_enc = le.fit_transform(labels_all)
    X_arr = np.vstack(feats_all)
    svm = SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale')

    # Cross-validation and accuracy plot
    scores = cross_val_score(svm, X_arr, y_enc, cv=5)
    print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Plot fold accuracies
    fig3, ax3 = plt.subplots(figsize=(6,4))
    folds = np.arange(1, len(scores)+1)
    ax3.plot(folds, scores, marker='o', linestyle='-')
    ax3.axhline(scores.mean(), linestyle='--', label=f'Mean = {scores.mean():.3f}')
    ax3.set_xticks(folds)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Cross-validation Accuracy per Fold')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.plot_out, 'accuracy.png'), dpi=300)
    plt.show()

    # Fit final model
    svm.fit(X_arr, y_enc)
    if args.svm_out:
        joblib.dump((svm, le), args.svm_out)
        print(f"Saved SVM to {args.svm_out}")

    # PCA visualization
    X2 = PCA(n_components=2).fit_transform(X_arr)
    fig, ax = plt.subplots(figsize=(8,6))
    for cls in le.classes_:
        idx = np.where(np.array(labels_all)==cls)
        ax.scatter(X2[idx,0], X2[idx,1], label=cls, s=40, alpha=0.7, edgecolors='k')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title('PCA by Class')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
    fig.savefig(os.path.join(args.plot_out,'pca.png'), dpi=300); plt.show()

    # Confusion matrix visualization
    preds = svm.predict(X_arr)
    cm = confusion_matrix(y_enc, preds)
    fig2, ax2 = plt.subplots(figsize=(5,4))
    im = ax2.imshow(cm, cmap='Blues'); fig2.colorbar(im, ax=ax2)
    ticks = np.arange(len(le.classes_))
    ax2.set_xticks(ticks); ax2.set_xticklabels(le.classes_, rotation=45)
    ax2.set_yticks(ticks); ax2.set_yticklabels(le.classes_)
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('True'); ax2.set_title('Confusion Matrix')
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>thresh else 'black')
    fig2.tight_layout(); fig2.savefig(os.path.join(args.plot_out,'confusion_matrix.png'), dpi=300); plt.show()

if __name__=='__main__':
    main()
