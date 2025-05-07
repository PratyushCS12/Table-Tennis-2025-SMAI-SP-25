import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torch import nn

# ——— Configuration ———
INPUT_DIR      = "/ssd_scratch/karan.p/RightVideo"
JSON_IN        = "/home/karan.padariya/SMAI-Project/selected_frames.json"
JSON_OUT       = "/home/karan.padariya/SMAI-Project/selected_frames_bb.json"
VIS_DIR        = "/ssd_scratch/karan.p/detected_results/"
PROMPTS        = ["table tennis player"]
MODEL_ID       = "IDEA-Research/grounding-dino-tiny"
BOX_THRESHOLD  = 0.3
TEXT_THRESHOLD = 0.5

def load_model():
    device    = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    if torch.cuda.device_count() > 1:
        print(f"🖥️  {torch.cuda.device_count()} GPUs detected — using DataParallel")
        model = nn.DataParallel(model)
    return model, processor, device

def process_batch(pil_imgs, model, processor, device):
    """
    pil_imgs: list of PIL.Image in RGB
    returns: list of detections lists (one per image)
    """
    bs = len(pil_imgs)
    # repeat the same prompt-list for each image
    text = [PROMPTS] * bs
    # pad to the max size in batch
    inputs = processor(images=pil_imgs, text=text,
                       return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[img.size[::-1] for img in pil_imgs]
        )
    all_detections = []
    for result in results:
        dets = []
        for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
            dets.append({
                "label": label,
                "box":   [round(float(x),2) for x in box],
                "score": round(float(score),3)
            })
        all_detections.append(dets)
    return all_detections

def visualize_frame(frame_bgr, detections, out_path):
    img = frame_bgr.copy()
    for det in detections:
        x1,y1,x2,y2 = map(int, det["box"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = f'{det["label"]}:{det["score"]:.2f}'
        cv2.putText(img, txt, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(out_path, img)

def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    model, processor, device = load_model()
    num_gpus  = torch.cuda.device_count()
    BATCH_SIZE = max(1, num_gpus)      # at least 1

    data = json.load(open(JSON_IN))

    for video_name, info in tqdm(data.items(), desc="Videos"):
        video_path = os.path.join(INPUT_DIR, video_name)
        cap        = cv2.VideoCapture(video_path)
        info["detections"] = {}

        frames = info.get("positive_frames", [])
        # process in chunks of BATCH_SIZE

        for i in tqdm(
                range(0, len(frames), BATCH_SIZE),
                desc=f"    Batches in {video_name}",
                unit="batch",
                leave=False
            ):
            batch_idxs = frames[i : i + BATCH_SIZE]
            pil_imgs   = []
            originals  = []   # store (frame_idx, bgr_frame) for each

            for idx in batch_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    tqdm.write(f"⚠️ Couldn't read frame {idx} of {video_name}")
                    continue
                pil_imgs.append(Image.fromarray(frame[..., ::-1]))
                originals.append((idx, frame))

            if not pil_imgs:
                continue

            # forward pass on batch
            dets_batch = process_batch(pil_imgs, model, processor, device)

            # write out results & visuals
            for (frame_idx, frame), dets in zip(originals, dets_batch):
                info["detections"][str(frame_idx)] = dets
                if dets:
                    out_vis = os.path.join(
                        VIS_DIR,
                        f"{os.path.splitext(video_name)[0]}_f{frame_idx}.jpg"
                    )
                    visualize_frame(frame, dets, out_vis)
        cap.release()

    # save JSON
    with open(JSON_OUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Done. Results in {JSON_OUT} and visuals in {VIS_DIR}")

if __name__ == "__main__":
    main()


# import os
# import json
# import glob
# import torch
# import cv2
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from torch import nn

# # ——— Configuration ———
# INPUT_DIR = "/ssd_scratch/karan.p/RightVideo"
# JSON_IN     = "/home/karan.padariya/SMAI-Project/selected_frames.json"
# JSON_OUT    = "/home/karan.padariya/SMAI-Project/selected_frames_bb.json"
# VIS_DIR     = "/ssd_scratch/karan.p/detected_results/"
# PROMPTS     = ["table tennis player"]
# MODEL_ID    = "IDEA-Research/grounding-dino-tiny"
# BOX_THRESHOLD  = 0.3
# TEXT_THRESHOLD = 0.5

# # ——— Helpers ———
# def load_model():
#     # device    = "cuda" if torch.cuda.is_available() else "cpu"
#     # processor = AutoProcessor.from_pretrained(MODEL_ID)
#     # model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
#     # return model, processor, device

#     # pick cuda (we assume at least one GPU is available)
#     device    = "cuda"
#     processor = AutoProcessor.from_pretrained(MODEL_ID)
#     # load into default GPU
#     model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
#     # if there's more than one GPU, wrap in DataParallel
#     if torch.cuda.device_count() > 1:
#         print(f"🖥️  {torch.cuda.device_count()} GPUs detected — using DataParallel")
#         model = nn.DataParallel(model)
#     return model, processor, device

# def process_image(image_src, model, processor, device):
#     # Accept either a PIL.Image or a filepath
#     if isinstance(image_src, str):
#         img = Image.open(image_src).convert("RGB")
#     else:
#         img = image_src.convert("RGB")

#     inputs = processor(images=img, text=[[p for p in PROMPTS]], return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)

#         result = processor.post_process_grounded_object_detection(
#             outputs,
#             inputs.input_ids,
#             box_threshold=BOX_THRESHOLD,
#             text_threshold=TEXT_THRESHOLD,
#             target_sizes=[img.size[::-1]]
#         )[0]
    
#         detections = []
#         # text_labels is already a list of strings (one per box)
#         for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
#             detections.append({
#                 "label": text_label,
#                 "box":   [round(float(x),2) for x in box],
#                 "score": round(float(score),3)
#             })
#         return detections

# def visualize_frame(frame_bgr, detections, out_path):
#     img = frame_bgr.copy()
#     for det in detections:
#         x1,y1,x2,y2 = map(int, det["box"])
#         cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
#         txt = f'{det["label"]}:{det["score"]:.2f}'
#         cv2.putText(img, txt, (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#     cv2.imwrite(out_path, img)

# # ——— Main ———
# def main():
#     os.makedirs(VIS_DIR, exist_ok=True)
#     model, processor, device = load_model()

#     # load your JSON of positive frames
#     data = json.load(open(JSON_IN))

#     for video_name, info in tqdm(data.items(), desc="Videos"):
#         video_path = os.path.join(INPUT_DIR, video_name)
#         cap = cv2.VideoCapture(video_path)
#         info["detections"] = {}
        
#         for video_name, info in tqdm(data.items(), desc="Videos"):
#             video_path = os.path.join(INPUT_DIR, video_name)
#             cap = cv2.VideoCapture(video_path)
#             info["detections"] = {}

#             frames = info.get("positive_frames", [])
#             # tqdm on frames, leave=False so it clears after each video
#             for frame_idx in tqdm(frames,
#                                 desc=f"    Frames in {video_name}",
#                                 leave=False,
#                                 unit="frame"):
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#                 ret, frame = cap.read()
#                 if not ret:
#                     tqdm.write(f"⚠️ Couldn't read frame {frame_idx} of {video_name}")
#                     continue

#                 pil = Image.fromarray(frame[..., ::-1])
#                 dets = process_image(pil, model, processor, device)
#                 info["detections"][str(frame_idx)] = dets

#                 if dets:
#                     out_vis = os.path.join(
#                         VIS_DIR,
#                         f"{os.path.splitext(video_name)[0]}_f{frame_idx}.jpg"
#                     )
#                     visualize_frame(frame, dets, out_vis)

#         cap.release()

#     # write augmented JSON
#     with open(JSON_OUT, "w") as f:
#         json.dump(data, f, indent=2)

#     print(f"✅ Done. Results in {JSON_OUT} and visualizations in {VIS_DIR}")

# if __name__ == "__main__":
#     main()
