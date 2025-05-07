import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import cv2


VIDEO   = "/home/karan.padariya/SMAI-Project/input_video.mp4"
CSV = "/home/karan.padariya/SMAI-Project/color_track_data.csv"
# VIDEO   = "input_video.mp4"
PID     = 1      # player ID to heatmap
BINS    = 64
# VIDEO       = "input_video.mp4"                                      # your downloaded clip

df = pd.read_csv(CSV)
if df.empty:
    raise ValueError("Tracking CSV is empty.")

# ── Distance per ID ─────────────────────────────
MAX_STEP = 50.0  # px; tune this to your frame rate / field scale

distances = {}
for tid, group in df.groupby("t_id"):
    coords = group[["x","y"]].values
    if len(coords) < 2:
        distances[tid] = 0.
        continue

    # compute all per-frame steps
    deltas = coords[1:] - coords[:-1]
    steps  = np.linalg.norm(deltas, axis=1)

    # filter out anomalies
    valid = steps < MAX_STEP
    cleaned_steps = steps[valid]

    # optionally log how many you dropped
    dropped = (~valid).sum()
    if dropped:
        print(f"🔶 Dropped {dropped} outlier steps for ID {tid}")

    distances[tid] = float(cleaned_steps.sum())

print("🏃 Total (filtered) distance per player:")
for tid, d in distances.items():
    print(f" • ID {tid}: {d:.1f}px")

# ── Heatmap for player ID ───────────────────────
cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video to get frame size")
H, W = frame.shape[:2]
cap.release()
for PID in range(1, 3):
    df_pid = df[df["t_id"] == PID]
    if df_pid.empty:
        print(f"No data for player ID {PID}")
    else:
        x, y = df_pid["x"].values, df_pid["y"].values
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=BINS, range=[[0, W], [0, H]])
        plt.imshow(
            heatmap.T,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
        )
        plt.title(f"Heatmap of Player ID {PID}")
        # plt.colorbar(label="Visit Count")
        plt.xlabel("X (px)")
        plt.ylabel("Y (px)")
        plt.savefig(f"/home/karan.padariya/SMAI-Project/heatmap/player_{PID}_heatmap.png", dpi=150)
        plt.show()


# # 2) collect centers per track ID
# positions = {}   # id -> list of (xc, yc)
# for frame in results:          # one Results object per frame
#     for box in frame.boxes:    # each has .xyxy and .id
#         tid = int(box.id)      
#         x1,y1,x2,y2 = box.xyxy.cpu().numpy().tolist()
#         xc = (x1 + x2) / 2
#         yc = (y1 + y2) / 2
#         positions.setdefault(tid, []).append((xc, yc))

# # 3) compute distance per ID
# distances = {}
# for tid, pts in positions.items():
#     pts = np.array(pts)
#     # sum of Euclidean distances between successive centers
#     d = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum() if len(pts)>1 else 0
#     distances[tid] = float(d)
# print("🏃 Distance covered (pixels) per player ID:", distances)

# # 4) heatmap of player #1 (change ID as needed)
# pid = 1
# all_pts = np.array(positions.get(pid, []))
# if all_pts.size:
#     heatmap, xedges, yedges = np.histogram2d(
#         all_pts[:,0], all_pts[:,1], bins=64,
#         range=[[0, frame.orig_shape[1]], [0, frame.orig_shape[0]]]
#     )
#     plt.imshow(heatmap.T, origin='lower',
#                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
#     plt.colorbar(label="Visits")
#     plt.title(f"Heatmap of Player {pid}")
#     plt.xlabel("X (pixels)")
#     plt.ylabel("Y (pixels)")
#     plt.savefig(f"player{pid}_heatmap.png", dpi=150)
#     plt.show()
# else:
#     print(f"No detections for player {pid}, cannot make heatmap.")


# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# import pandas as pd
# import cv2
# import os

# VIDEO   = "/home/karan.padariya/SMAI-Project/input_video.mp4"
# CSV     = "/home/karan.padariya/SMAI-Project/color_track_data.csv"
# FRAME_IMG = "/home/karan.padariya/SMAI-Project/14_frame_0181_bb.jpg"
# OUTPUT_IMG = "/home/karan.padariya/SMAI-Project/heatmap/overlay_player_{pid}.png"
# PID_LIST = [1, 2]
# BINS    = 64

# # Load tracking data
# df = pd.read_csv(CSV)
# if df.empty:
#     raise ValueError("Tracking CSV is empty.")

# # Get frame size from input image
# frame = cv2.imread(FRAME_IMG)
# if frame is None:
#     raise FileNotFoundError(f"Could not read frame image: {FRAME_IMG}")
# H, W = frame.shape[:2]

# # For each player ID
# for PID in PID_LIST:
#     df_pid = df[df["t_id"] == PID]
#     if df_pid.empty:
#         print(f"No data for player ID {PID}")
#         continue

#     # Generate 2D histogram heatmap
#     x, y = df_pid["x"].values, df_pid["y"].values
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=BINS, range=[[0, W], [0, H]])
    
#     # Normalize heatmap and resize to match frame size
#     heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     heatmap_resized = cv2.resize(heatmap.T, (W, H))
#     heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

#     # Overlay heatmap on original frame
#     overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

#     # Save the overlayed image
#     out_path = OUTPUT_IMG.format(pid=PID)
#     cv2.imwrite(out_path, overlay)
#     print(f"✅ Heatmap overlay saved for Player {PID} at: {out_path}")
