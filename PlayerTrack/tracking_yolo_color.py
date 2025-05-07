import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def get_hsv_distance(hsv1, hsv2):
    """Calculate distance between two HSV colors considering circular hue"""
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2
    
    # Circular hue difference
    h_diff = abs(h1 - h2)
    h_circular_diff = min(h_diff, 180 - h_diff)
    
    # Normalize components to [0, 1]
    h_dist = h_circular_diff / 90.0  # Max circular difference is 90
    s_dist = abs(s1 - s2) / 255.0
    v_dist = abs(v1 - v2) / 255.0
    
    # Weighted distance (adjust weights as needed)
    return h_dist * 0.5 + s_dist * 0.3 + v_dist * 0.2

# Path configurations
IN_VIDEO  = "/home/karan.padariya/SMAI-Project/input_video.mp4"
OUT_VIDEO = "/home/karan.padariya/SMAI-Project/output_color_track.mp4"
OUT_CSV   = "/home/karan.padariya/SMAI-Project/color_track_data.csv"
MODEL_PATH = "/home/karan.padariya/runs/detect/tt_player_tracking7/weights/best.pt"

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Video setup
cap = cv2.VideoCapture(IN_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Tracking variables
ref_colors = None  # Stores reference HSV values for both players
records = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.25, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) != 2:
        writer.write(frame)
        frame_idx += 1
        continue

    if ref_colors is None:
        # First valid frame - establish color references
        sorted_boxes = sorted(boxes, key=lambda b: (b[0] + b[2])/2)  # Sort left to right
        
        # Extract reference colors
        ref_colors = []
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            ref_colors.append((
                np.mean(hsv_roi[:,:,0]),  # H
                np.mean(hsv_roi[:,:,1]),  # S
                np.mean(hsv_roi[:,:,2])   # V
            ))
        id_order = [0, 1]  # Initial left-right order
    else:
        # Subsequent frames - color matching
        current_hsv = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            current_hsv.append((
                np.mean(hsv_roi[:,:,0]),
                np.mean(hsv_roi[:,:,1]),
                np.mean(hsv_roi[:,:,2])
            ))

        # Calculate assignment costs
        cost_assign0 = (get_hsv_distance(current_hsv[0], ref_colors[0]) +
                        get_hsv_distance(current_hsv[1], ref_colors[1]))
        cost_assign1 = (get_hsv_distance(current_hsv[0], ref_colors[1]) +
                        get_hsv_distance(current_hsv[1], ref_colors[0]))

        id_order = [0, 1] if cost_assign0 <= cost_assign1 else [1, 0]

    # Process boxes with determined order
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        tid = id_order[i] + 1  # Convert to 1-based IDs
        color = (0, 255, 0) if tid == 1 else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Record data
        records.append({
            "frame": frame_idx,
            "t_id": tid,
            "x": (x1+x2)/2,
            "y": (y1+y2)/2,
            "width": x2-x1,
            "height": y2-y1,
            "area": (x2-x1)*(y2-y1)
        })

    writer.write(frame)
    frame_idx += 1

# Cleanup
cap.release()
writer.release()
pd.DataFrame(records).to_csv(OUT_CSV, index=False)
print(f"Processing complete! Output saved to {OUT_VIDEO} and {OUT_CSV}")
