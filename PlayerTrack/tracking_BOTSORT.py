from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import torch
from boxmot import BotSort

# 1) Video Configuration
IN_VIDEO = "/home/karan.padariya/SMAI-Project/input_video.mp4"
OUT_CSV = "/home/karan.padariya/SMAI-Project/track_id_BotSort.csv"
OUT_VIDEO = "/home/karan.padariya/SMAI-Project/output_BotSort.mp4"


# 2) Initialize YOLO Model with Embeddings
model = YOLO("/home/karan.padariya/runs/detect/tt_player_tracking7/weights/best.pt")

# 3) Configure BoTSORT without ReID
cap = cv2.VideoCapture(IN_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

tracker = BotSort(
    reid_weights = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    half=True,
    with_reid=False,  # Disable external ReID
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=int(fps * 2),
    match_thresh=0.8,
    proximity_thresh=0.5,
    cmc_method="sof",
    frame_rate=fps,
    fuse_first_associate=True,
)

# 4) Tracking Loop with YOLO Embeddings
records = []
frame_idx = 0
box_color = (0, 255, 0)  # Green color for bounding boxes
text_color = (0, 0, 255)  # Red color for text

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.25, verbose=False)
    
    # Extract detection components
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()
    
    # Format detections [x1, y1, x2, y2, conf, cls]
    dets = np.hstack([boxes, confs.reshape(-1, 1), clss.reshape(-1, 1)])

    # Update tracker
    tracks = tracker.update(dets, frame)

    # Draw results on frame
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls = track[:7]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Draw ID and confidence
        label = f"ID: {int(track_id)} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Store tracking data
        records.append({
            "frame": frame_idx,
            "id": int(track_id),
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "area": (x2 - x1) * (y2 - y1),
            "confidence": conf,
            "class": int(cls)
        })

    # Write frame to output video
    out.write(frame)
    frame_idx += 1
    print(f"Processed frame {frame_idx}", end="\r")

cap.release()
out.release()


# 5) Save Results
pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print(f"\n✅ Tracking data saved to {OUT_CSV}")