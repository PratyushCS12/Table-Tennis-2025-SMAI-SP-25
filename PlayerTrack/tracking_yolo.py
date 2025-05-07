from yt_dlp import YoutubeDL
from ultralytics import YOLO
import pandas as pd

# 1) grab the video
# URL        = "https://youtu.be/jbeXI7zYGlk?si=vgSWkFqDjrjrOFsQ" #"https://youtu.be/gvtJ4B7EqDc?si=gfua4fyf34XXunrr"
IN_VIDEO   = "/home/karan.padariya/SMAI-Project/input_video.mp4"
OUT_CSV = "/home/karan.padariya/SMAI-Project/track_id.csv"
ydl_opts   = {'format':'mp4','outtmpl':IN_VIDEO,'noplaylist':True}
# with YoutubeDL(ydl_opts) as ydl:
    # ydl.download([URL])

# 2) load your trained model
model = YOLO("/home/karan.padariya/runs/detect/tt_player_tracking7/weights/best.pt")

# 3) track players, save in runs/track/player_tracking
#    - tracker="bytetrack.yaml" uses the ByteTrack algorithm
#    - persist=True makes IDs stick if temporarily lost
#    - save=True writes out an mp4 with boxes+IDs
results = model.track(
    source=IN_VIDEO,
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    show=False,
    conf=0.25,            # adjust confidence threshold
    project="runs/track",
    name="player_tracking"
)

print("✅ saved to runs/track/player_tracking")


records = []
for frame_idx, res in enumerate(results):
    boxes = res.boxes
    if boxes.id is None: continue  # no detections
    xyxys = boxes.xyxy.cpu().numpy()
    ids   = boxes.id.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), tid in zip(xyxys, ids):
        width  = x2 - x1
        height = y2 - y1
        area   = width * height
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        records.append({
            "frame": frame_idx,
            "id": tid,
            "x": xc,
            "y": yc,
            "width": width,
            "height": height,
            "area": area
        })


# Save to CSV
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Saved tracking data to {OUT_CSV}")


# import cv2
# # from pytube import YouTube
# from ultralytics import YOLO

# # 1. Download the YouTube video
# # from yt_dlp import YoutubeDL

# # url        = "https://youtu.be/gvtJ4B7EqDc?si=gfua4fyf34XXunrr"
# video_path = "input_video.mp4"

# # ydl_opts = {
# #     'format': 'mp4',
# #     'outtmpl': video_path,
# #     'noplaylist': True
# # }
# # with YoutubeDL(ydl_opts) as ydl:
# #     ydl.download([url])

# # 2. Load your trained model
# model = YOLO("/home/karan.padariya/runs/detect/tt_player_tracking6/weights/best.pt")

# # 3. Open the video, run frame‑wise inference, and write out an annotated video
# cap    = cv2.VideoCapture(video_path)
# fps    = cap.get(cv2.CAP_PROP_FPS)
# w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out    = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (w, h))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # model(frame) returns a list; take the first result
#     res       = model(frame)[0]
#     annotated = res.plot()   # draws boxes+labels onto a copy of `frame`
#     out.write(annotated)

# cap.release()
# out.release()
# print("✅  Inference complete: saved as annotated_output.mp4")
