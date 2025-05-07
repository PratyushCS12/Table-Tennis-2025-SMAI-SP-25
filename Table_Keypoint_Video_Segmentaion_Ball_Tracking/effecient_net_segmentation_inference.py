import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import time
import gc
import traceback
from tqdm import tqdm
import json
import math

# --- Configuration ---

# **** IMPORTANT: Verify this path points to the BEST saved checkpoint file ****
CHECKPOINT_PATH = "/content/drive/MyDrive/SMAI_Project_Checkpoints/efficientnetB0_chunked_frames_best.pth"

# **** IMPORTANT: Verify this path points to your ORIGINAL video directory ****
VIDEO_INPUT_DIR = "/content/drive/MyDrive/SMAI_Project_dataset/RightVideo"

# **** IMPORTANT: Set the path for the output JSON file ****
OUTPUT_JSON_PATH = "/content/drive/MyDrive/SMAI_Project_Labeled_Videos/video_frame_labels.json"

# Model Input Size (should match training)
IMG_SIZE = 224

# Inference Batch Size (Process multiple frames at once for speed)
# Adjust based on GPU memory. Higher is faster if memory allows.
INFERENCE_BATCH_SIZE = 64 # Can often be higher when not storing/writing frames

# --- Chunking Configuration ---
# How much video data (in seconds) to load into RAM at once for processing
# Can potentially be larger now since we are not holding original frames for writing
CHUNK_DURATION_SECONDS = 300 # Load 5-minute chunks

# --- Optional: Skip processing if output JSON already exists? ---
# Set to False to overwrite or append (current code overwrites)
SKIP_IF_EXISTS = False # Set to True to skip if JSON exists

# --- Mount Google Drive ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    # Create parent directory for JSON if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    print(f"Output JSON will be saved to: {OUTPUT_JSON_PATH}")
except ImportError:
    print("Not running in Google Colab or Drive mounting failed.")
    print("Ensure all paths (checkpoint, input, output) are accessible locally.")
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True) # Try creating locally
except Exception as e:
    print(f"Error mounting Google Drive or creating output directory: {e}")
    exit()

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Load Model Function (Same as before) ---
def load_model_from_checkpoint(checkpoint_path, device):
    print(f"Loading model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at '{checkpoint_path}'")
        return None, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(num_ftrs, 1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        class_to_idx = checkpoint.get('class_to_idx')
        idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else {0: 'negative', 1: 'positive'}
        model.eval()
        model = model.to(device)
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')} (Val Acc: {checkpoint.get('accuracy', 'N/A'):.4f})")
        print(f"Using class mapping: {idx_to_class}")
        return model, idx_to_class
    except Exception as e:
        print(f"\nERROR loading checkpoint: {e}"); traceback.print_exc(); return None, None


# --- Define Inference Transformations (Same as before) ---
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
print("Inference transformations defined.")


# --- Main Processing Function (Chunked Read + Batched Inference -> JSON Output) ---
def classify_video_frames_to_json(input_video_path, model, idx_to_class_map, transform, device, inference_batch_size, chunk_duration_seconds):
    """
    Reads video in chunks, classifies frames in batches, and returns a list
    of {'frame_index': index, 'label': 'positive'/'negative'} dictionaries.
    """
    video_name = os.path.basename(input_video_path)
    print(f"\nProcessing: {video_name}")
    start_time = time.time()
    cap = None
    frame_labels = [] # List to store results for this video
    total_frames_read_overall = 0
    video_props = {}

    try:
        # --- Open Input Video & Get Properties ---
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"  ERROR: Could not open input video."); return None, {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_props = {
            'fps': fps, 'width': frame_width, 'height': frame_height,
            'total_frames': total_frames_in_video
        }
        print(f"  Input properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, ~{total_frames_in_video} frames")

        if fps <= 0 or frame_width <= 0 or frame_height <= 0 or total_frames_in_video <= 0:
             print(f"  ERROR: Invalid video properties retrieved. Skipping."); return None, video_props

        chunk_size_frames = math.ceil(chunk_duration_seconds * fps)
        if chunk_size_frames <= 0: chunk_size_frames = 5000 # Fallback
        print(f"  Processing in chunks of up to {chunk_size_frames} frames ({chunk_duration_seconds}s)")

        # --- Loop Through Chunks ---
        current_chunk_start_frame = 0
        pbar = tqdm(total=total_frames_in_video, desc="  Frames processed", unit="frame")

        while current_chunk_start_frame < total_frames_in_video:
            chunk_end_frame = min(current_chunk_start_frame + chunk_size_frames, total_frames_in_video)
            current_chunk_frames_bgr = [] # Holds frames just for preprocessing
            chunk_load_success = True
            effective_chunk_end_frame = chunk_end_frame

            try:
                # --- Load current chunk into RAM ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_chunk_start_frame)
                frames_read_in_chunk = 0

                for i in range(chunk_size_frames):
                    actual_frame_index = current_chunk_start_frame + i
                    if actual_frame_index >= effective_chunk_end_frame: break
                    ret, frame = cap.read()
                    if not ret:
                        effective_chunk_end_frame = actual_frame_index
                        chunk_load_success = False; break
                    current_chunk_frames_bgr.append(frame) # Store temporarily
                    frames_read_in_chunk += 1
                    total_frames_read_overall += 1

                if len(current_chunk_frames_bgr) == 0 and effective_chunk_end_frame > current_chunk_start_frame:
                     chunk_load_success = False

                # --- Process Frames within the Loaded Chunk in Inference Batches ---
                if chunk_load_success or len(current_chunk_frames_bgr) > 0:
                    inference_batch_tensors = []
                    # Keep track of original frame indices for this batch
                    batch_frame_indices = []

                    for frame_index_in_chunk, original_frame in enumerate(current_chunk_frames_bgr):
                        absolute_frame_idx = current_chunk_start_frame + frame_index_in_chunk
                        batch_frame_indices.append(absolute_frame_idx) # Store original index

                        # Preprocess frame for model
                        try:
                            frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                            img_pil = Image.fromarray(frame_rgb)
                            img_tensor = transform(img_pil)
                            inference_batch_tensors.append(img_tensor)
                        except Exception as preprocess_e:
                             print(f"\n      Error preprocessing frame {absolute_frame_idx}: {preprocess_e}")
                             # Remove corresponding index if preprocess failed
                             if batch_frame_indices and batch_frame_indices[-1] == absolute_frame_idx:
                                 batch_frame_indices.pop()
                             continue # Skip adding tensor to batch

                        # If inference batch is full, process it
                        if len(inference_batch_tensors) == inference_batch_size:
                            try:
                                input_tensor = torch.stack(inference_batch_tensors).to(device)
                                with torch.no_grad():
                                    outputs = model(input_tensor)
                                    probs = torch.sigmoid(outputs)
                                    preds_indices = (probs > 0.5).int().cpu()

                                # Append results to the video's list
                                for i in range(len(batch_frame_indices)):
                                    pred_idx = preds_indices[i].item()
                                    pred_name = idx_to_class_map.get(pred_idx, "unknown")
                                    frame_labels.append({
                                        'frame_index': batch_frame_indices[i],
                                        'label': pred_name
                                    })
                                pbar.update(len(batch_frame_indices)) # Update progress bar

                            except Exception as batch_inf_e:
                                 print(f"\n      Error during batch inference: {batch_inf_e}")

                            # Clear inference batches
                            inference_batch_tensors.clear()
                            batch_frame_indices.clear()

                    # --- Process Remaining Partial Inference Batch (after chunk loop) ---
                    if inference_batch_tensors:
                        try:
                            input_tensor = torch.stack(inference_batch_tensors).to(device)
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probs = torch.sigmoid(outputs)
                                preds_indices = (probs > 0.5).int().cpu()

                            for i in range(len(batch_frame_indices)):
                                pred_idx = preds_indices[i].item()
                                pred_name = idx_to_class_map.get(pred_idx, "unknown")
                                frame_labels.append({
                                    'frame_index': batch_frame_indices[i],
                                    'label': pred_name
                                })
                            pbar.update(len(batch_frame_indices))

                            inference_batch_tensors.clear()
                            batch_frame_indices.clear()
                        except Exception as final_batch_inf_e:
                             print(f"\n      Error processing final inference batch: {final_batch_inf_e}")

            except MemoryError:
                print(f"\n\n     CRITICAL: MemoryError loading chunk [{current_chunk_start_frame}-{chunk_end_frame-1}] for video {video_name}!")
                print(f"     Try reducing CHUNK_DURATION_SECONDS ({chunk_duration_seconds}s).")
                # Stop processing this video on MemoryError
                print(f"     Stopping processing for this video.")
                pbar.close() # Close progress bar early
                return None, video_props # Return None for labels to indicate failure

            finally:
                # --- Free chunk memory ---
                del current_chunk_frames_bgr
                gc.collect()

            # --- Move to the next chunk ---
            next_chunk_start = effective_chunk_end_frame

            # Stalling Check
            if next_chunk_start <= current_chunk_start_frame and current_chunk_start_frame < total_frames_in_video:
                # print(f"\n     INFO: Chunk end ({next_chunk_start}) didn't advance. Likely read errors. Ending.")
                break # Exit chunk loop safely

            current_chunk_start_frame = next_chunk_start

        pbar.close() # Close progress bar after all chunks are processed

    except Exception as e:
        print(f"\n  An unexpected error occurred during processing: {e}")
        traceback.print_exc()
        return None, video_props # Indicate failure
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

    end_time = time.time()
    duration = end_time - start_time
    print(f"  Finished processing.")
    print(f"  Total frames read: {total_frames_read_overall}")
    print(f"  Labels generated for: {len(frame_labels)} frames")
    print(f"  Duration: {duration:.2f} seconds")
    if len(frame_labels) != total_frames_read_overall and total_frames_read_overall > 0:
         print(f"  WARNING: Number of labels ({len(frame_labels)}) differs from frames read ({total_frames_read_overall}). Check logs.")

    return frame_labels, video_props # Return the list of label dictionaries


# --- Main Execution Logic ---
if __name__ == "__main__":
    print("\n--- Starting Video Frame Labeling to JSON Script ---")

    # 1. Load the trained model
    model, idx_to_class = load_model_from_checkpoint(CHECKPOINT_PATH, device)
    if model is None: exit()

    # 2. Check if output JSON should be skipped
    if SKIP_IF_EXISTS and os.path.exists(OUTPUT_JSON_PATH):
        print(f"\nOutput JSON file '{OUTPUT_JSON_PATH}' already exists and SKIP_IF_EXISTS is True.")
        print("Skipping processing.")
        print("\n--- Script Finished ---")
        exit()

    # 3. Get list of videos to process
    print(f"\nSearching for videos in: {VIDEO_INPUT_DIR}")
    try:
        input_video_files = sorted([ f for f in os.listdir(VIDEO_INPUT_DIR)
                                     if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                                     and os.path.isfile(os.path.join(VIDEO_INPUT_DIR, f)) ])
        if not input_video_files: print(f"ERROR: No video files found in '{VIDEO_INPUT_DIR}'."); exit()
        print(f"Found {len(input_video_files)} video files to process.")
    except FileNotFoundError: print(f"ERROR: Input video directory not found: '{VIDEO_INPUT_DIR}'"); exit()
    except Exception as e: print(f"ERROR listing input video files: {e}"); exit()

    # 4. Process each video and collect results
    all_video_results = {} # Dictionary to store results: {video_name: {'properties': {...}, 'labels': [...]}}
    successful_videos = 0
    failed_videos = 0

    for video_filename in input_video_files:
        input_path = os.path.join(VIDEO_INPUT_DIR, video_filename)

        # Classify frames for the video
        labels_list, props = classify_video_frames_to_json(
                                 input_path, model, idx_to_class,
                                 inference_transform, device,
                                 INFERENCE_BATCH_SIZE, CHUNK_DURATION_SECONDS
                             )

        if labels_list is not None: # Check if processing was successful
            all_video_results[video_filename] = {
                'properties': props,
                'labels': labels_list
            }
            successful_videos += 1
        else:
            # Store minimal info for failed videos if needed, or just count failure
            all_video_results[video_filename] = {
                'properties': props, # Store properties even if labeling failed
                'labels': None,      # Indicate failure
                'error': 'Processing failed, check logs.'
            }
            failed_videos += 1

        # Optional: Aggressive cleanup after each video
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 5. Save all results to a single JSON file
    print(f"\nSaving results for {len(all_video_results)} videos to: {OUTPUT_JSON_PATH}")
    try:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_video_results, f, indent=4) # Use indent for readability
        print("Successfully saved results to JSON.")
    except Exception as e:
        print(f"\nERROR saving results to JSON file '{OUTPUT_JSON_PATH}': {e}")
        traceback.print_exc()

    # 6. Final Summary
    print("\n" + "="*50); print("      JSON Labeling Summary"); print("="*50)
    print(f"Successfully processed: {successful_videos} video(s)")
    print(f"Failed processing:      {failed_videos} video(s)")
    print(f"Results saved in:       {OUTPUT_JSON_PATH}")

    # 7. Cleanup
    print("\n--- Script Finished ---")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
