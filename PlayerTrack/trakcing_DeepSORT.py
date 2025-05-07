import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# paths
IN_VIDEO  = "/home/karan.padariya/SMAI-Project/input_video.mp4"
OUT_VIDEO = "runs/track/player_deepsort.mp4"
OUT_CSV   = "/home/karan.padariya/SMAI-Project/track_deepsort.csv"

# 1) load YOLO model
model = YOLO("/home/karan.padariya/runs/detect/tt_player_tracking7/weights/best.pt")

# 2) init DeepSORT with more robust parameters
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,  # Slightly increased for tolerance
    nn_budget=100,
    embedder=None,  # Set to None to use default embedder
)

# 3) prepare I/O
cap = cv2.VideoCapture(IN_VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

records = []
frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # 1) YOLO inference
        res    = model(frame)[0]
        boxes  = res.boxes.xyxy.cpu().numpy()    # shape [N,4]
        scores = res.boxes.conf.cpu().numpy()    # shape [N]
        clss   = res.boxes.cls.cpu().numpy().astype(int)  # shape [N]
        
        # 2) build the list of raw detections
        raw_detections = []
        for (x1,y1,x2,y2), score, cls in zip(boxes, scores, clss):
            # Make sure bbox dimensions are valid
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            # Only add detection if box has valid area
            if x2 > x1 and y2 > y1:
                raw_detections.append(((x1, y1, x2, y2), float(score), int(cls), None))
        
        # 3) update DeepSORT with error handling
        try:
            tracks = tracker.update_tracks(raw_detections, frame)
            
            # 4) annotate + log
            for t in tracks:
                if not t.is_confirmed():
                    continue
                    
                tid = t.track_id
                try:
                    x1, y1, x2, y2 = t.to_ltrb()
                    xc, yc = (x1+x2)/2, (y1+y2)/2
                    
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                    cv2.putText(frame, f"ID:{tid}", (int(x1),int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    
                    records.append({
                        "frame": frame_idx,
                        "id": tid,
                        "x": xc,
                        "y": yc
                    })
                except Exception as e:
                    print(f"Error processing track {tid}: {e}")
                    continue
                    
        except ValueError as e:
            print(f"DeepSORT error on frame {frame_idx}: {e}")
            # If tracking fails, we'll still show detections from YOLO
            for i, ((x1, y1, x2, y2), score, cls, _) in enumerate(raw_detections):
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
                cv2.putText(frame, f"Det:{i}", (int(x1),int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        # Add frame number to video
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        out_vid.write(frame)
        frame_idx += 1
        
        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")
            
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Always make sure to release resources
    cap.release()
    out_vid.release()
    
    # 7) save CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUT_CSV, index=False)
        print(f"✅ Video with DeepSORT tracks: {OUT_VIDEO}")
        print(f"✅ Tracking data saved to: {OUT_CSV}")
    else:
        print("⚠️ No tracking records were generated")