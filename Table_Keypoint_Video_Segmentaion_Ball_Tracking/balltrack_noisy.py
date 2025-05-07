# -*- coding: utf-8 -*-
"""
Script to train a 3D U-Net for segmenting small moving objects (like a ball)
in video sequences, filtering out motion overlapping with detected persons.
Uses a multi-stage caching approach with RESUMABILITY and CHUNKED YOLO:
1. Cache needed frames globally (skips videos if cache dir exists).
2. Run batched YOLO inference on required cached frames (loads existing results,
   runs inference on missing frames in chunks, saves combined results).
3. Generate sequences using cached assets (skips existing .npy sequences).
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from tqdm import tqdm
import gc
from ultralytics import YOLO # Import YOLO
import random
import torch.nn.functional as F
import shutil # For directory removal
import traceback
import time # For timing sections
import math # For chunking calculation

# --- 1. Configuration ---
# Paths
DRIVE_BASE = "/content/drive/MyDrive/SMAI_Project_dataset"
VIDEO_DIR = os.path.join(DRIVE_BASE, "RightVideo")
JSON_PATH = os.path.join(DRIVE_BASE, "selected_frames.json")
OUTPUT_DIR = os.path.join(DRIVE_BASE, "unet_data_v6_resume") # Changed version
CACHE_BASE_DIR = "/content/frame_cache_v6_resume" # Changed version
YOLO_CACHE_FILE = os.path.join(CACHE_BASE_DIR, "global_yolo_results_cache.json")
MODEL_SAVE_PATH = os.path.join(DRIVE_BASE, "3d_unet_ball_segmentation_v6_resume.pth") # Changed version
YOLO_MODEL_NAME = 'yolov8s-world.pt'

# Processing Parameters
ROI_SIZE = 20
MASK_RADIUS_X = 4
BORDER_RADIUS_Y = 8
IOU_THRESHOLD = 0.1
NUM_VIDEOS_TO_PROCESS = 0 # Set to 0 for all videos in selected_frames.json
SEQUENCE_LENGTH = 3
FRAME_DIFF_AREA_MIN = 8
FRAME_DIFF_AREA_MAX = 400
YOLO_CONF_THRESHOLD = 0.4
YOLO_BATCH_SIZE = 16 # Hint for potential internal batching, may not be strictly used by predict()
YOLO_INFERENCE_CHUNK_SIZE = 500 # <<< Process YOLO inference in chunks of this size for memory efficiency

# Training Parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
NUM_WORKERS = 2
EARLY_STOPPING_PATIENCE = 7
BILINEAR_UPSAMPLING = False # False uses ConvTranspose3d, True uses Upsample

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- Create Output and Cache Directories ---
os.makedirs(os.path.join(OUTPUT_DIR, 'sequences'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)
os.makedirs(CACHE_BASE_DIR, exist_ok=True)

# --- 2. Helper Functions ---
# (calculate_iou, apply_otsu_threshold, generate_masks, process_frame_for_bboxes - remain unchanged)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def apply_otsu_threshold(frame):
    if frame is None: return None
    if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else: gray = frame
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def generate_masks(height, width, center, mask_radius, border_radius):
    core_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask = np.zeros((height, width), dtype=np.uint8)
    try: cx, cy = int(center[0]), int(center[1])
    except: return core_mask, border_mask
    mask_radius = max(1, int(mask_radius))
    border_radius = max(mask_radius + 1, int(border_radius))
    cv2.circle(core_mask, (cx, cy), mask_radius, 255, -1)
    cv2.circle(border_mask, (cx, cy), border_radius, 255, -1)
    cv2.circle(border_mask, (cx, cy), mask_radius, 0, -1)
    return core_mask, border_mask

def process_frame_for_bboxes(f0_gray, f1_gray, roi_size=20, min_area=10, max_area=400):
    if f0_gray is None or f1_gray is None: return []
    height, width = f0_gray.shape
    diff_frame = cv2.absdiff(f0_gray, f1_gray)
    thresh_frame = apply_otsu_threshold(diff_frame)
    if thresh_frame is None: return []
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated = cv2.dilate(opened, kernel, iterations=3)
    try: # Add try-except for robustness if connectedComponents fails
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    except Exception as e:
        print(f"Error in connectedComponentsWithStats: {e}")
        return []
    potential_boxes = []
    for i in range(1, num_labels):
        try:
            area = stats[i, cv2.CC_STAT_AREA]
            if not (min_area <= area <= max_area): continue
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            half_size = roi_size // 2
            x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
            x2, y2 = min(width, cx + half_size), min(height, cy + half_size)
            if x1 >= x2 or y1 >= y2: continue
            potential_boxes.append([(cx, cy), (x1, y1, x2, y2)])
        except IndexError:
            print(f"IndexError in process_frame_for_bboxes loop (i={i}, stats shape: {stats.shape}, centroids shape: {centroids.shape}). Skipping component.")
            continue
    return potential_boxes


# --- 3. YOLO-World Loading ---
print("Loading YOLO-World model...")
yolo_model = None
try:
    yolo_model = YOLO(YOLO_MODEL_NAME)
    yolo_model.set_classes(["person"])
    print("YOLO-World model loaded.")
    if device != "cpu":
        try:
             print("Performing dummy YOLO inference..."); _ = yolo_model.predict(np.zeros((64, 64, 3), dtype=np.uint8), device=device, verbose=False); torch.cuda.synchronize(); print("Dummy inference complete.")
        except Exception as e: print(f"Warning: dummy YOLO inference failed: {e}")
    gc.collect(); torch.cuda.empty_cache()
except Exception as e: print(f"Error loading YOLO Model: {e}"); exit()

# --- 4. Data Preparation Script ---
print("\n--- Starting Data Preparation ---")
all_sequences_paths = []
all_masks_paths = []
processed_sequence_count = 0 # Tracks sequences successfully generated and saved across runs
data_prep_start_time = time.time()

try:
    with open(JSON_PATH, 'r') as f: selected_frames_data = json.load(f)
except Exception as e: print(f"Error loading JSON {JSON_PATH}: {e}"); exit()

video_files_in_json = list(selected_frames_data.keys())
if NUM_VIDEOS_TO_PROCESS > 0: video_files_to_process = video_files_in_json[:NUM_VIDEOS_TO_PROCESS]
else: video_files_to_process = video_files_in_json
print(f"Will process {len(video_files_to_process)} videos.")

# --- Stage 1: Global Frame Caching & YOLO Target Collection (with Resume Logic) ---
print("\n--- Stage 1: Caching Frames & Collecting YOLO Targets (Resumable) ---")
frame_caching_start_time = time.time()
all_potential_yolo_paths = [] # Collect ALL potential paths needed for YOLO across all videos
total_frames_actually_cached = 0 # Count only frames written in this run
total_cache_errors = 0

for video_idx, video_name in enumerate(video_files_to_process):
    video_path = os.path.join(VIDEO_DIR, video_name)
    video_name_safe = video_name.replace('.', '_')
    if not os.path.exists(video_path): continue
    if video_name not in selected_frames_data: continue
    positive_indices = sorted(selected_frames_data[video_name].get("positive_frames", []))
    if not positive_indices: continue

    video_cache_dir = os.path.join(CACHE_BASE_DIR, video_name_safe)
    cache_exists = os.path.isdir(video_cache_dir)
    verb = "Scanning" if cache_exists else "Caching"
    # print(f"{verb} Video {video_idx + 1}/{len(video_files_to_process)}: {video_name}") # Less verbose

    # Determine needed indices regardless of cache existence
    cap = cv2.VideoCapture(video_path) # Needed for frame count even if cache exists
    if not cap.isOpened(): print(f"Error opening {video_path}"); continue
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release() # Release early if just getting frame count

    needed_indices_for_video = set(); yolo_needed_indices_for_video = set()
    for frame_idx in positive_indices:
        if 0 < frame_idx < total_video_frames - 1:
            needed_indices_for_video.update([frame_idx - 1, frame_idx, frame_idx + 1])
            yolo_needed_indices_for_video.add(frame_idx)
    if not needed_indices_for_video: continue

    indices_to_scan_or_cache = sorted(list(needed_indices_for_video))

    # Collect potential YOLO paths and cache if necessary
    if not cache_exists:
        os.makedirs(video_cache_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error opening {video_path} for caching"); continue

    cached_count_video = 0; cache_errors_video = 0
    last_read_idx = -1; frame_buffer = {}
    pbar_cache = tqdm(indices_to_scan_or_cache, desc=f"{verb} {video_name_safe[:15]}", leave=False, ncols=100)

    for frame_to_cache_idx in pbar_cache:
        cache_filename = f"frame_{frame_to_cache_idx:06d}.png"
        cache_path = os.path.join(video_cache_dir, cache_filename)

        # --- Collect YOLO Path ---
        if frame_to_cache_idx in yolo_needed_indices_for_video:
            # Add path for YOLO check later, regardless of whether we cache now or it exists
            all_potential_yolo_paths.append(cache_path)

        # --- Caching Logic (only if cache dir didn't exist initially or file is missing) ---
        if not cache_exists or not os.path.exists(cache_path):
            if cache_exists and not os.path.exists(cache_path):
                 pbar_cache.set_description(f"Caching MISSING {video_name_safe[:10]}") # Indicate partial cache
                 if not cap.isOpened(): # Reopen cap if needed for partial cache
                      cap = cv2.VideoCapture(video_path)
                      if not cap.isOpened(): print(f"Error reopening {video_path}"); cache_errors_video += 1; continue

            # Read Frame (only if caching is needed)
            frame = None; ret = False
            if frame_to_cache_idx in frame_buffer: frame = frame_buffer[frame_to_cache_idx]; ret = True
            else:
                seek_needed = (frame_to_cache_idx <= last_read_idx or frame_to_cache_idx >= last_read_idx + 10)
                if seek_needed: frame_buffer.clear(); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_cache_idx); last_read_idx = frame_to_cache_idx - 1
                current_pos = last_read_idx + 1
                while current_pos <= frame_to_cache_idx :
                     ret_read, frame_read = cap.read()
                     if not ret_read: break
                     last_read_idx = current_pos
                     if current_pos >= frame_to_cache_idx - 5:
                         frame_buffer[current_pos] = frame_read
                         if len(frame_buffer) > 10: frame_buffer.pop(min(frame_buffer.keys()), None)
                     if current_pos == frame_to_cache_idx: frame = frame_read; ret = True; break
                     current_pos +=1

            # Write Frame
            if ret and frame is not None:
                try:
                    cv2.imwrite(cache_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    cached_count_video += 1
                except Exception as e: cache_errors_video += 1; print(f"Cache write error: {e}")
            else: cache_errors_video += 1
        # --- End Caching Logic ---

    if cap.isOpened(): cap.release()
    del frame_buffer
    total_frames_actually_cached += cached_count_video
    total_cache_errors += cache_errors_video

all_potential_yolo_paths = sorted(list(set(all_potential_yolo_paths))) # Unique paths
frame_caching_duration = time.time() - frame_caching_start_time
print(f"--- Stage 1 Finished ({frame_caching_duration:.2f}s). Frames written this run: {total_frames_actually_cached}, Total YOLO candidates: {len(all_potential_yolo_paths)}, Errors: {total_cache_errors} ---")
gc.collect()

# --- Stage 2: Batched YOLO Inference (Resumable & Chunked) ---
print("\n--- Stage 2: Running Batched YOLO Inference (Resumable & Chunked) ---")
yolo_start_time = time.time()
loaded_yolo_cache = {}
new_yolo_results = {}

# Load existing results first
if os.path.exists(YOLO_CACHE_FILE):
    print(f"Loading existing YOLO results from {YOLO_CACHE_FILE}...")
    try:
        with open(YOLO_CACHE_FILE, 'r') as f_json:
            loaded_yolo_cache = json.load(f_json)
        print(f"Loaded {len(loaded_yolo_cache)} cached YOLO results.")
    except Exception as e:
        print(f"Error loading YOLO cache file {YOLO_CACHE_FILE}: {e}. Starting fresh cache.")
        loaded_yolo_cache = {}

# Determine which frames still need processing
frames_to_run_yolo = [p for p in all_potential_yolo_paths if p not in loaded_yolo_cache]
frames_to_run_yolo = [p for p in frames_to_run_yolo if os.path.exists(p)] # Ensure frame file actually exists

if frames_to_run_yolo and yolo_model:
    print(f"Found {len(frames_to_run_yolo)} frames needing YOLO inference.")
    num_chunks = math.ceil(len(frames_to_run_yolo) / YOLO_INFERENCE_CHUNK_SIZE)
    yolo_errors = 0

    for i in range(num_chunks):
        chunk_start = i * YOLO_INFERENCE_CHUNK_SIZE
        chunk_end = chunk_start + YOLO_INFERENCE_CHUNK_SIZE
        current_chunk_paths = frames_to_run_yolo[chunk_start:chunk_end]

        if not current_chunk_paths: continue

        print(f"Processing YOLO Chunk {i+1}/{num_chunks} ({len(current_chunk_paths)} frames)...")
        try:
            # Run predict on the chunk
            results_generator = yolo_model.predict(
                source=current_chunk_paths,
                device=device,
                conf=YOLO_CONF_THRESHOLD,
                stream=True,
                verbose=False
            )

            # Process results for the chunk
            pbar_yolo_chunk = tqdm(results_generator, total=len(current_chunk_paths), desc=f"YOLO Chunk {i+1}", ncols=100, leave=False)
            for results in pbar_yolo_chunk:
                current_frame_path = results.path
                person_boxes_list = []
                try:
                    if hasattr(results, 'boxes') and results.boxes is not None:
                        boxes_tensor = results.boxes.xyxy
                        if boxes_tensor.numel() > 0:
                            person_boxes_list = boxes_tensor.cpu().numpy().astype(int).tolist()
                    # Store result for this frame (even if empty)
                    new_yolo_results[current_frame_path] = person_boxes_list
                except Exception as e_inner:
                    new_yolo_results[current_frame_path] = []
                    yolo_errors += 1
                    print(f"\nError processing YOLO result for {current_frame_path}: {e_inner}")

            del results_generator # Explicitly delete generator
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"Chunk {i+1} processed. Current new results: {len(new_yolo_results)}")

        except Exception as e_outer:
            print(f"\nError during YOLO prediction for chunk {i+1}: {e_outer}")
            traceback.print_exc()
            yolo_errors += len(current_chunk_paths) # Count all in chunk as errors if predict fails

    if yolo_errors > 0:
        print(f"Encountered {yolo_errors} errors during YOLO processing.")

    # Merge new results into the loaded cache
    if new_yolo_results:
        print(f"Merging {len(new_yolo_results)} new results into the cache...")
        loaded_yolo_cache.update(new_yolo_results)

        # Save the combined cache
        print(f"Saving combined YOLO results ({len(loaded_yolo_cache)} total) to {YOLO_CACHE_FILE}...")
        try:
            # Backup previous cache before overwriting
            if os.path.exists(YOLO_CACHE_FILE):
                try: shutil.copy2(YOLO_CACHE_FILE, YOLO_CACHE_FILE + '.bak')
                except Exception: print("Warning: Failed to backup YOLO cache.")

            with open(YOLO_CACHE_FILE, 'w') as f_json:
                json.dump(loaded_yolo_cache, f_json) # Save without indent for space
            print("Combined YOLO results cache saved.")
        except Exception as e:
            print(f"Error saving combined YOLO results cache: {e}")

elif not frames_to_run_yolo and all_potential_yolo_paths:
    print("All required YOLO results were already found in the cache.")
else:
    if not all_potential_yolo_paths: print("No frames required YOLO processing.")
    if not yolo_model: print("YOLO model not loaded, skipping inference.")

gc.collect(); torch.cuda.empty_cache()
yolo_duration = time.time() - yolo_start_time
print(f"--- Stage 2 Finished ({yolo_duration:.2f}s) ---")


# --- Stage 3: Sequence Generation using Cached Assets (Resumable) ---
print("\n--- Stage 3: Generating Sequences from Cached Data (Resumable) ---")
sequence_gen_start_time = time.time()
final_sequences_generated_this_run = 0
sequences_skipped = 0

# Determine starting sequence count by checking existing output files
# This is an approximation, assumes files are sequential without gaps if run previously
# A more robust way might be needed if runs create gaps (e.g., saving a manifest)
# Simple approach: find the highest existing sequence number + 1
existing_seq_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, 'sequences')) if f.startswith('seq_') and f.endswith('.npy')]
if existing_seq_files:
    try:
        last_num = max([int(f[4:-4]) for f in existing_seq_files])
        processed_sequence_count = last_num + 1
        print(f"Resuming sequence generation. Found existing sequences up to {last_num}. Starting count from {processed_sequence_count}.")
    except ValueError:
        print("Warning: Could not parse existing sequence numbers. Starting count from 0.")
        processed_sequence_count = 0
else:
    processed_sequence_count = 0 # Start from 0 if no output exists

# --- Reload YOLO cache (might have been updated in Stage 2) ---
if os.path.exists(YOLO_CACHE_FILE):
    try:
        with open(YOLO_CACHE_FILE, 'r') as f_json: loaded_yolo_cache = json.load(f_json)
    except Exception as e: print(f"Error re-loading YOLO cache for Stage 3: {e}"); loaded_yolo_cache = {}
else: loaded_yolo_cache = {}
print(f"Using {len(loaded_yolo_cache)} YOLO cache entries for sequence generation.")


current_video_progress = tqdm(video_files_to_process, desc="Video Sequence Gen", ncols=100)
for video_idx, video_name in enumerate(current_video_progress):
    video_name_safe = video_name.replace('.', '_')
    video_cache_dir = os.path.join(CACHE_BASE_DIR, video_name_safe)

    # **Resume Check 1: Skip video if its cache dir is already cleaned up**
    if not os.path.isdir(video_cache_dir):
        # print(f"Skipping video {video_name}: Cache directory not found (already processed and cleaned).")
        continue # Assume Stage 3 already finished successfully for this video

    if video_name not in selected_frames_data: continue
    positive_indices = sorted(selected_frames_data[video_name].get("positive_frames", []))
    if not positive_indices: continue

    valid_sequences_generated_video = 0 # Count generated sequences for *this video* in *this run*
    sequences_skipped_video = 0 # Count skipped sequences for *this video* in *this run*

    # Estimate starting sequence number for this video based on global count.
    # This isn't strictly needed for the check below but helps understand progress.
    # seq_idx_offset = processed_sequence_count + sequences_skipped # Rough starting point

    current_video_progress.set_description(f"Sequences {video_name_safe[:15]}")
    # pbar_seq = tqdm(positive_indices, desc=f"Frames {video_name_safe[:15]}", leave=False, ncols=100)
    # Iterate through positive frames to check/generate sequences
    for frame_idx in positive_indices:
        # Calculate the *potential* index this sequence would have IF generated
        # This count depends on sequences generated *before* this video + sequences generated *within* this video so far
        potential_output_idx = processed_sequence_count + valid_sequences_generated_video + sequences_skipped_video

        seq_filename_check = f"seq_{potential_output_idx:06d}.npy"
        mask_filename_check = f"mask_{potential_output_idx:06d}.npy"
        seq_path_check = os.path.join(OUTPUT_DIR, 'sequences', seq_filename_check)
        mask_path_check = os.path.join(OUTPUT_DIR, 'masks', mask_filename_check)

        # **Resume Check 2: Skip if output files already exist**
        if os.path.exists(seq_path_check) and os.path.exists(mask_path_check):
            sequences_skipped_video += 1
            continue # Skip this frame_idx, output already exists

        # --- If not skipped, proceed to generate ---
        idx_prev = frame_idx - 1; idx_curr = frame_idx; idx_next = frame_idx + 1
        path_prev = os.path.join(video_cache_dir, f"frame_{idx_prev:06d}.png")
        path_curr = os.path.join(video_cache_dir, f"frame_{idx_curr:06d}.png")
        path_next = os.path.join(video_cache_dir, f"frame_{idx_next:06d}.png")
        if not (os.path.exists(path_prev) and os.path.exists(path_curr) and os.path.exists(path_next)):
             # This shouldn't happen if Stage 1 logic is correct, but good check
             # print(f"Warning: Missing cache frames for seq {frame_idx} in {video_name}, skipping.")
             continue

        try: # Load frames
            frame_prev = cv2.imread(path_prev); frame_curr = cv2.imread(path_curr); frame_next = cv2.imread(path_next)
            if frame_prev is None or frame_curr is None or frame_next is None: raise ValueError("Frame load failed")
        except Exception as e: print(f"Seq frame read error: {e}"); continue

        # Process motion, apply YOLO filter, generate masks (same logic as before)
        f_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY); f_curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
        motion_boxes_info = process_frame_for_bboxes(f_prev_gray, f_curr_gray, roi_size=ROI_SIZE, min_area=FRAME_DIFF_AREA_MIN, max_area=FRAME_DIFF_AREA_MAX)
        if not motion_boxes_info: continue
        person_boxes = loaded_yolo_cache.get(path_curr, [])
        filtered_motion_centers = []
        for motion_center, motion_box in motion_boxes_info:
            is_overlapping = False
            if person_boxes:
                for person_box in person_boxes:
                    if len(person_box) == 4:
                        if calculate_iou(motion_box, person_box) > IOU_THRESHOLD: is_overlapping = True; break
            if not is_overlapping: filtered_motion_centers.append(motion_center)
        if not filtered_motion_centers: continue
        frame_h, frame_w = frame_curr.shape[:2]
        final_core_mask = np.zeros((frame_h, frame_w), dtype=np.uint8); final_border_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for center in filtered_motion_centers:
            core_mask, border_mask = generate_masks(frame_h, frame_w, center, MASK_RADIUS_X, BORDER_RADIUS_Y)
            final_core_mask = cv2.bitwise_or(final_core_mask, core_mask); final_border_mask = cv2.bitwise_or(final_border_mask, border_mask)
        if np.sum(final_core_mask) == 0 and np.sum(final_border_mask) == 0: continue

        # Resize
        try:
            frame_prev_resized = cv2.resize(frame_prev, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA); frame_curr_resized = cv2.resize(frame_curr, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA); frame_next_resized = cv2.resize(frame_next, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            final_core_mask_resized = cv2.resize(final_core_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST); final_border_mask_resized = cv2.resize(final_border_mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        except cv2.error as e: print(f"Resize error: {e}"); continue

        # Stack
        sequence = np.stack([frame_prev_resized, frame_curr_resized, frame_next_resized], axis=0); sequence = np.transpose(sequence, (3, 0, 1, 2))
        masks = np.stack([final_core_mask_resized, final_border_mask_resized], axis=0)

        # Save using the calculated potential_output_idx
        seq_path_save = seq_path_check # Use the paths already constructed
        mask_path_save = mask_path_check

        try:
            np.save(seq_path_save, sequence.astype(np.uint8))
            np.save(mask_path_save, (masks > 0).astype(np.uint8))
            # Only increment count and add paths if save is successful
            valid_sequences_generated_video += 1
            final_sequences_generated_this_run += 1
            all_sequences_paths.append(seq_path_save) # Collect paths for dataset
            all_masks_paths.append(mask_path_save)
        except Exception as e:
            print(f"Numpy save error for seq {potential_output_idx}: {e}")
            if os.path.exists(seq_path_save): os.remove(seq_path_save) # Clean up partial save
            if os.path.exists(mask_path_save): os.remove(mask_path_save)
            continue # Don't increment counts or add paths

    # After processing all frames for a video
    # Update the global *starting* count for the *next* video
    processed_sequence_count += (valid_sequences_generated_video + sequences_skipped_video)
    sequences_skipped += sequences_skipped_video

    # Clean up this video's cache directory now that sequence generation is done/skipped
    # print(f"Cleaning up frame cache: {video_cache_dir}") # Less verbose
    try: shutil.rmtree(video_cache_dir)
    except OSError as e: print(f"Error removing cache dir {video_cache_dir}: {e}")
    del f_prev_gray, f_curr_gray; gc.collect()
    # --- End of video loop in Stage 3 ---

sequence_gen_duration = time.time() - sequence_gen_start_time
print(f"--- Stage 3 Finished ({sequence_gen_duration:.2f}s). Generated {final_sequences_generated_this_run} new sequences this run. Skipped {sequences_skipped} existing sequences. ---")
total_data_prep_duration = time.time() - data_prep_start_time
print(f"Total Data Preparation Time: {total_data_prep_duration:.2f}s")


# --- 5. PyTorch Dataset and DataLoaders ---
# (Dataset and DataLoader code remains the same)
class BallSegmentDataset(Dataset):
    def __init__(self, sequence_paths, mask_paths, transform=None):
        self.sequence_paths = sequence_paths; self.mask_paths = mask_paths
        # We collect paths *only* for sequences generated *in this run*.
        # If resuming training, this dataset might only contain newly generated data.
        # If the goal is to train on *all* data (old + new), we need to scan the OUTPUT_DIR instead.
        # For now, assume we train on what was just generated/verified in this run.
        if not self.sequence_paths:
             print("Warning: No sequence paths collected in Stage 3. Dataset might be empty.")
        assert len(self.sequence_paths) == len(self.mask_paths), "Seq/Mask path mismatch"
        self.transform = transform
    def __len__(self): return len(self.sequence_paths)
    def __getitem__(self, idx):
        seq_path = self.sequence_paths[idx]; mask_path = self.mask_paths[idx]
        try:
            sequence_uint8 = np.load(seq_path); masks_uint8 = np.load(mask_path)
            sequence_float = sequence_uint8.astype(np.float32) / 255.0
            masks_float = masks_uint8.astype(np.float32)
            sequence_tensor = torch.from_numpy(sequence_float); masks_tensor = torch.from_numpy(masks_float)
            if self.transform: pass
            return sequence_tensor, masks_tensor
        except FileNotFoundError: raise FileNotFoundError(f"Missing files idx {idx}: {seq_path} or {mask_path}")
        except Exception as e: raise RuntimeError(f"Failed loading idx {idx} ({seq_path}): {e}")

# --- Create Dataset & Dataloaders ---
# Option 1: Use only paths collected in this run (current implementation)
print(f"Creating dataset from {len(all_sequences_paths)} sequences generated/verified in this run.")
# Option 2: Scan OUTPUT_DIR for *all* existing sequences (if resuming training on full dataset is desired)
# if RESUME_TRAINING_ON_ALL_DATA: # Add a flag for this if needed
#    print("Scanning OUTPUT_DIR for all existing sequences...")
#    all_sequences_paths = sorted([os.path.join(OUTPUT_DIR, 'sequences', f) for f in os.listdir(os.path.join(OUTPUT_DIR, 'sequences')) if f.endswith('.npy')])
#    all_masks_paths = sorted([os.path.join(OUTPUT_DIR, 'masks', f.replace('seq_', 'mask_')) for f in os.listdir(os.path.join(OUTPUT_DIR, 'sequences')) if f.endswith('.npy')])
#    print(f"Found {len(all_sequences_paths)} total sequences in OUTPUT_DIR.")

if not all_sequences_paths or not all_masks_paths: print("Error: No sequence paths for dataset creation."); exit()
if len(all_sequences_paths) != len(all_masks_paths): print("Error: Seq/Mask path count mismatch for dataset."); exit()

try: dataset = BallSegmentDataset(all_sequences_paths, all_masks_paths)
except Exception as e: print(f"Error creating Dataset: {e}"); exit()
total_size = len(dataset); print(f"Final Dataset size for training/validation: {total_size}")
if total_size == 0: print("Error: Dataset empty. No sequences available for training."); exit()

val_size = int(total_size * VAL_SPLIT)
if val_size == 0 and total_size > 0 and VAL_SPLIT > 0: val_size = 1
if val_size >= total_size and total_size > 0: val_size = total_size - 1
train_size = total_size - val_size; print(f"Split: Train={train_size}, Val={val_size}")
if train_size <= 0: print("Error: Train size <= 0."); exit()

train_dataset, val_dataset = None, None
try:
    if val_size > 0 and train_size > 0: train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    elif train_size > 0 : train_dataset = dataset # Handle case with only train data
    else: raise ValueError("Cannot split dataset with non-positive sizes")
except Exception as e: print(f"Error during random_split: {e}"); exit()

persistent_workers_flag = (NUM_WORKERS > 0)
train_loader, val_loader = None, None
try:
    if train_dataset: train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=persistent_workers_flag)
    if val_dataset: val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False, persistent_workers=persistent_workers_flag)
except Exception as e:
    print(f"DataLoader error (retrying with NUM_WORKERS=0): {e}")
    NUM_WORKERS = 0; persistent_workers_flag = False
    if train_dataset: train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    if val_dataset: val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
if not train_loader: print("Error: Train loader creation failed."); exit()


# --- 6. 3D U-Net Model Definition ---
# (DoubleConv3D, Down3D, CORRECTED Up3D, OutConv3D, UNet3D - remain the same as previous version)
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x): return self.double_conv(x)

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)), DoubleConv3D(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv. Handles bilinear or ConvTranspose."""
    def __init__(self, in_channels_total, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.out_channels = out_channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels_total, out_channels, mid_channels=in_channels_total // 2)
        else:
            ch_skip = out_channels; ch_below = in_channels_total - ch_skip
            if ch_below <= 0: raise ValueError(f"Up3D CT init error: ch_below ({ch_below}) <= 0. In={in_channels_total}, Out={out_channels}")
            up_out_channels = ch_skip
            self.up = nn.ConvTranspose3d(ch_below, up_out_channels, kernel_size=(1,2,2), stride=(1,2,2))
            conv_in_channels = ch_skip + ch_skip
            self.conv = DoubleConv3D(conv_in_channels, out_channels)
    def forward(self, x1, x2):
        x1_up = self.up(x1)
        diffZ = x2.size(2) - x1_up.size(2); diffY = x2.size(3) - x1_up.size(3); diffX = x2.size(4) - x1_up.size(4)
        if diffZ > 0 or diffY > 0 or diffX > 0: x2 = x2[:, :, diffZ//2 : x2.size(2) - (diffZ - diffZ//2), diffY//2 : x2.size(3) - (diffY - diffY//2), diffX//2 : x2.size(4) - (diffX - diffX//2)]
        if diffZ < 0 or diffY < 0 or diffX < 0: x1_up = F.pad(x1_up, [abs(diffX)//2, abs(diffX) - abs(diffX)//2, abs(diffY)//2, abs(diffY) - abs(diffY)//2, abs(diffZ)//2, abs(diffZ) - abs(diffZ)//2])
        x = torch.cat([x2, x1_up], dim=1); return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels): super(OutConv3D, self).__init__(); self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3D, self).__init__(); self.n_channels, self.n_classes, self.bilinear = n_channels, n_classes, bilinear
        ch1, ch2, ch3, ch4, ch5 = 32, 64, 128, 256, 512
        self.inc = DoubleConv3D(n_channels, ch1); self.down1 = Down3D(ch1, ch2); self.down2 = Down3D(ch2, ch3); self.down3 = Down3D(ch3, ch4); self.down4 = Down3D(ch4, ch5)
        self.up1 = Up3D(ch5 + ch4, ch4, bilinear); self.up2 = Up3D(ch4 + ch3, ch3, bilinear); self.up3 = Up3D(ch3 + ch2, ch2, bilinear); self.up4 = Up3D(ch2 + ch1, ch1, bilinear)
        self.outc = OutConv3D(ch1, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits_3d = self.outc(x)
        if logits_3d.size(2) == SEQUENCE_LENGTH: logits_2d = logits_3d[:, :, SEQUENCE_LENGTH // 2, :, :]
        else: logits_2d = logits_3d[:, :, logits_3d.size(2) // 2, :, :] # Fallback
        return logits_2d

model = UNet3D(n_channels=3, n_classes=2, bilinear=BILINEAR_UPSAMPLING).to(device)
print(f"UNet initialized with bilinear={BILINEAR_UPSAMPLING}")

# --- 7. Training Loop ---
# (Loss function, optimizer, scheduler, and training loop logic remain the same)
def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
    target_flat = target.reshape(target.size(0), target.size(1), -1)
    intersection = (pred_flat * target_flat).sum(dim=2)
    dice_score = (2. * intersection + smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + smooth)
    return 1. - dice_score.mean()

def combined_loss(pred, target):
    target = target.to(pred.dtype)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    dice = dice_loss(pred, target)
    return bce + dice

criterion = combined_loss
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, verbose=True)

print("\n--- Starting Training ---")
training_start_time = time.time()
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    epoch_train_loss = 0.0
    pbar_epoch = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False, ncols=100)
    for batch_idx, batch in enumerate(pbar_epoch):
        if batch is None: print("W: None train batch"); continue
        try: sequences, masks = batch; sequences=sequences.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
        except Exception as e: print(f"E: Train batch move: {e}"); continue
        optimizer.zero_grad(set_to_none=True)
        try:
            outputs = model(sequences)
            if outputs.shape[-2:] != masks.shape[-2:] or outputs.shape[1] != masks.shape[1]: print(f"E: Train shape mismatch: Out={outputs.shape}, Mask={masks.shape}"); continue
            loss = criterion(outputs, masks)
            if torch.isnan(loss) or torch.isinf(loss): print(f"W: Train NaN/Inf loss: {loss.item()}"); continue
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if (batch_idx + 1) % 20 == 0: pbar_epoch.set_postfix({'AvgLoss': f'{epoch_train_loss / (batch_idx + 1):.4f}'})
        except Exception as e: print(f"\nE: Train Step: {e}"); traceback.print_exc(); continue
    avg_train_loss = epoch_train_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0.0
    pbar_epoch.close()

    model.eval()
    epoch_val_loss = 0.0
    avg_val_loss = float('inf')
    if val_loader:
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False, ncols=100)
        with torch.no_grad():
             for batch_idx, batch in enumerate(pbar_val):
                if batch is None: print("W: None val batch"); continue
                try: sequences, masks = batch; sequences=sequences.to(device,non_blocking=True); masks=masks.to(device,non_blocking=True)
                except Exception as e: print(f"E: Val batch move: {e}"); continue
                try:
                    outputs = model(sequences)
                    if outputs.shape[-2:] != masks.shape[-2:] or outputs.shape[1] != masks.shape[1]: print(f"E: Val shape mismatch: Out={outputs.shape}, Mask={masks.shape}"); continue
                    loss = criterion(outputs, masks)
                    if not (torch.isnan(loss) or torch.isinf(loss)): epoch_val_loss += loss.item()
                    else: print(f"W: Val NaN/Inf loss: {loss.item()}")
                    if (batch_idx + 1) % 10 == 0: pbar_val.set_postfix({'AvgLoss': f'{epoch_val_loss / (batch_idx + 1):.4f}'})
                except Exception as e: print(f"\nE: Val Step: {e}"); traceback.print_exc(); continue
        avg_val_loss = epoch_val_loss / len(val_loader) if val_loader and len(val_loader) > 0 else float('inf')
        pbar_val.close()

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} ({epoch_duration:.2f}s) -> Train Loss: {avg_train_loss:.4f}", end="")
    if val_loader: print(f", Val Loss: {avg_val_loss:.4f}")
    else: print(" (No Validation)")

    if val_loader and avg_val_loss != float('inf'): # Check avg_val_loss is valid
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try: torch.save(model.state_dict(), MODEL_SAVE_PATH); print(f"*** Best model saved (Val Loss: {best_val_loss:.4f}) ***")
            except Exception as e: print(f"E: Model save error: {e}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs w/o improvement: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE: print("Early stopping!"); break
    elif not val_loader: # No validation loader
         if (epoch + 1) == NUM_EPOCHS: # Save last if no validation
              try: torch.save(model.state_dict(), MODEL_SAVE_PATH); print(f"Model saved (final epoch)")
              except Exception as e: print(f"E: Error saving final model: {e}")

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

training_duration = time.time() - training_start_time
print(f"\n--- Training Finished ({training_duration:.2f}s) ---")
if val_loader and best_val_loss != float('inf'): print(f"Best Val Loss: {best_val_loss:.4f}. Model saved to {MODEL_SAVE_PATH}")
else: print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

# Optional: Final cleanup
# try:
#     if os.path.exists(CACHE_BASE_DIR): print(f"Cleaning up base cache: {CACHE_BASE_DIR}"); shutil.rmtree(CACHE_BASE_DIR)
# except Exception as e: print(f"Error cleaning base cache: {e}")

