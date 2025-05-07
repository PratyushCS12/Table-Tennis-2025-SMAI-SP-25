import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Paths
IN_VIDEO  = "/home/karan.padariya/SMAI-Project/input_video.mp4"
OUT_VIDEO = "/home/karan.padariya/SMAI-Project/output_custom_track.mp4"
OUT_CSV   = "/home/karan.padariya/SMAI-Project/custom_track_data.csv"
MODEL_PATH = "/home/karan.padariya/runs/detect/tt_player_tracking7/weights/best.pt"

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(IN_VIDEO)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

# Tracking logic
prev_centers = None
records = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx == 279:
        print("warning!!")
    if frame_idx % 10 == 0:
        print(f"Processing frame {frame_idx}/{total_frames}...")

    # Run YOLO detection
    results = model(frame, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if boxes.shape[0] != 2:
        writer.write(frame)
    else:
        centers = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in boxes]

        if prev_centers is None:
            id_order = [0, 1]
        else:
            # d00 = np.hypot(centers[0][0]-prev_centers[0][0],
            #                centers[0][1]-prev_centers[0][1])
            # d01 = np.hypot(centers[0][0]-prev_centers[1][0],
            #                centers[0][1]-prev_centers[1][1])
            # id_order = [0, 1] if d00 <= d01 else [1, 0]

            cx1 = centers[0][0]
            cx2 = centers[1][0]
            id_order = [0, 1] if cx1 <= cx2 else [1, 0]

        # Draw + record bounding boxes
        for new_idx, id_assigned in enumerate(id_order):
            x1,y1,x2,y2 = boxes[new_idx].astype(int)
            w = x2 - x1
            h = y2 - y1
            a = w * h
            cx, cy = centers[new_idx]
            tid = id_assigned + 1  # Assigned IDs: 1 or 2
            color = (0,255,0) if tid == 1 else (0,0,255)

            # Draw box + label
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save record
            records.append({
                "frame": frame_idx,
                "t_id": tid,
                "x": cx,
                "y": cy,
                "width": w,
                "height": h,
                "area": a
            })

        prev_centers = [centers[id_order[0]], centers[id_order[1]]]
        writer.write(frame)

    frame_idx += 1

# Cleanup
cap.release()
writer.release()
print(f"✅ custom-tracked video saved to {OUT_VIDEO}")

# Save tracking data to CSV
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"✅ tracking data saved to {OUT_CSV}")
