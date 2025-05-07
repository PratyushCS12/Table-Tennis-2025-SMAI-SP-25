import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# ── CONFIG ─────────────────────────────────────────────────────────────
# point these to your dirs
VIDEOS_DIR    = Path("/ssd_scratch/karan.p/RightVideo")             # contains 14.mp4, etc.
ANNOT_JSON    = Path("/home/karan.padariya/SMAI-Project/selected_frames_bb.json")   # your JSON from question
OUTPUT_DIR    = Path("/ssd_scratch/karan.p/dataset")            # will be created
CLASS_NAMES   = ["table tennis player"]    # index 0
IMG_FORMAT    = "jpg"
FRAME_DIR     = OUTPUT_DIR / "images" / "train"
LABEL_DIR     = OUTPUT_DIR / "labels" / "train"
VAL_SPLIT     = 0.2                         # if you want a val split

# ── UTILS ──────────────────────────────────────────────────────────────
def yolo_box(box, img_w, img_h):
    # converts [x1,y1,x2,y2] -> [x_center, y_center, w, h] normalized
    x1,y1,x2,y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return xc, yc, w, h

# ── PREPARE OUTPUT DIRS ─────────────────────────────────────────────────
for d in (FRAME_DIR, LABEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── LOAD ANNOTATIONS ───────────────────────────────────────────────────
with open(ANNOT_JSON, "r") as f:
    ann = json.load(f)

# ── EXTRACT FRAMES & WRITE LABELS ───────────────────────────────────────
# for vid_name, vinfo in ann.items():
#     vid_path = VIDEOS_DIR / vid_name
#     cap = cv2.VideoCapture(str(vid_path))
#     if not cap.isOpened():
#         print(f"⚠️  Cannot open {vid_name}, skipping")
#         continue

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     for frame_str, dets in tqdm(vinfo["detections"].items(), desc=vid_name):
#         idx = int(frame_str)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         img_fname  = f"{vid_name[:-4]}_frame{idx:05d}.{IMG_FORMAT}"
#         lbl_fname  = f"{vid_name[:-4]}_frame{idx:05d}.txt"
#         img_path   = FRAME_DIR / img_fname
#         lbl_path   = LABEL_DIR / lbl_fname

#         # save image
#         cv2.imwrite(str(img_path), frame)

#         # write yolo labels
#         with open(lbl_path, "w") as out:
#             for d in dets:
#                 label = d.get("label","").strip()
#                 if not label or label not in CLASS_NAMES:
#                     continue
#                 cls_id = CLASS_NAMES.index(label)
#                 xc,yc,w_n,h_n = yolo_box(d["box"], w, h)
#                 out.write(f"{cls_id} {xc:.6f} {yc:.6f} {w_n:.6f} {h_n:.6f}\n")

#     cap.release()

# ── (OPTIONAL) SPLIT TRAIN/VAL ──────────────────────────────────────────
from sklearn.model_selection import train_test_split
import shutil

# paths
IMG_TRAIN = OUTPUT_DIR / "images" / "train"
LBL_TRAIN = OUTPUT_DIR / "labels" / "train"
IMG_VAL   = OUTPUT_DIR / "images" / "val"
LBL_VAL   = OUTPUT_DIR / "labels" / "val"

# make val dirs
# IMG_VAL.mkdir(parents=True, exist_ok=True)
# LBL_VAL.mkdir(parents=True, exist_ok=True)

# # get all image filenames
# all_imgs = list(IMG_TRAIN.glob(f"*.{IMG_FORMAT}"))
# all_names = [p.stem for p in all_imgs]

# # split
# train_names, val_names = train_test_split(
#     all_names, test_size=VAL_SPLIT, random_state=42
# )

# # move the files
# for name in tqdm(val_names, desc="Moving to VAL"):
#     # image
#     src_img = IMG_TRAIN / f"{name}.{IMG_FORMAT}"
#     dst_img = IMG_VAL   / f"{name}.{IMG_FORMAT}"
#     shutil.move(str(src_img), str(dst_img))

#     # label
#     src_lbl = LBL_TRAIN / f"{name}.txt"
#     dst_lbl = LBL_VAL   / f"{name}.txt"
#     if src_lbl.exists():
#         shutil.move(str(src_lbl), str(dst_lbl))

# ── WRITE data.yaml ────────────────────────────────────────────────────
data_cfg = {
    'train': str(IMG_TRAIN),
    'val':   str(IMG_VAL),
    'nc':    len(CLASS_NAMES),
    'names': CLASS_NAMES
}
with open(OUTPUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_cfg, f)


import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded '{filename}' successfully.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# Example usage
url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'  # Replace with actual URL
filename = 'yolov11n.pt'
download_file(url, filename)


# ── TRAIN YOLOv8n ───────────────────────────────────────────────────────
model = YOLO("yolov11n.pt")     # pretrained nano model
model.train(
    data    = str(OUTPUT_DIR / "data.yaml"),
    epochs  = 10,
    imgsz   = 640,
    batch   = 32,
    name    = "tt_player_tracking"
)
