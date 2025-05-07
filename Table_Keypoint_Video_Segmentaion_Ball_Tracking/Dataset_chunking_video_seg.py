import cv2
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import gc # Garbage Collector
import time
import math
import traceback

# --- Configuration ---
# Input Data
JSON_PATH = "/content/drive/MyDrive/SMAI_Project_dataset/selected_frames.json"
VIDEO_BASE_DIR = "/content/drive/MyDrive/SMAI_Project_dataset/RightVideo"

# Frame Extraction Output Location
EXTRACTION_BASE_DIR = "/content/extracted_frames_chunked" # Store extracted frames here

# Parameters
VALIDATION_SPLIT = 0.2 # Fraction for validation set
RANDOM_SEED = 42       # For reproducible train/val split
IMG_SIZE = 224         # Target image size (width, height) for resizing
JPEG_QUALITY = 95      # Quality for saving JPG images (0-100)

# --- Chunking Configuration ---
CHUNK_DURATION_SECONDS = 300 # Process in 5-minute (300 seconds) chunks

# --- Optional: Skip extraction if output dir exists? ---
SKIP_IF_EXISTS = True # Set to False to always re-extract

# --- Mount Google Drive (if using Colab) ---
# This is needed to access your JSON and Videos if they are on Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab or Drive mounting failed. Ensure paths are accessible.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    # Depending on where your data is, you might not need Drive

# --- Start Extraction Process ---
print("="*50)
print("      Frame Extraction Script (Chunking Strategy)")
print("="*50)

# Check if extraction should be skipped
if SKIP_IF_EXISTS and \
   os.path.exists(os.path.join(EXTRACTION_BASE_DIR, 'train')) and \
   os.path.exists(os.path.join(EXTRACTION_BASE_DIR, 'val')):
    print(f"Output directory '{EXTRACTION_BASE_DIR}' already exists and SKIP_IF_EXISTS is True.")
    print("Skipping frame extraction.")
    print("="*50)
    exit() # Exit the script cleanly if skipping

# --- Load Frame Data from JSON ---
print(f"\n1. Loading frame data from: {JSON_PATH}")
try:
    with open(JSON_PATH, 'r') as f:
        selected_frames_data = json.load(f)
    print(f"   Loaded data for {len(selected_frames_data)} videos.")
except FileNotFoundError:
    print(f"   ERROR: JSON file not found at {JSON_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"   ERROR: Could not decode JSON from {JSON_PATH}")
    exit()
except Exception as e:
    print(f"   ERROR: An unexpected error occurred while loading JSON: {e}")
    exit()

# --- Prepare Master Frame List & Check Video Files ---
print("\n2. Preparing master list of frames to extract and checking video files...")
all_frames_to_extract = []
videos_found_count = 0
videos_missing_count = 0
unique_videos = set()

for video_name, data in tqdm(selected_frames_data.items(), desc="Checking videos"):
    video_path = os.path.join(VIDEO_BASE_DIR, video_name)
    if os.path.exists(video_path):
        videos_found_count += 1
        unique_videos.add(video_path)
        # Add positive frames (label 1)
        for frame_idx in data.get("positive_frames", []):
            all_frames_to_extract.append({'video_path': video_path,
                                          'video_name': video_name,
                                          'frame_idx': frame_idx,
                                          'label': 1})
        # Add negative frames (label 0)
        for frame_idx in data.get("negative_frames", []):
            all_frames_to_extract.append({'video_path': video_path,
                                          'video_name': video_name,
                                          'frame_idx': frame_idx,
                                          'label': 0})
    else:
        videos_missing_count += 1
        # print(f"   Warning: Video file not found, skipping: {video_path}") # Optional verbose warning

print(f"   Checked {len(selected_frames_data)} videos listed in JSON.")
print(f"   Found {videos_found_count} video files.")
if videos_missing_count > 0:
    print(f"   WARNING: Skipped {videos_missing_count} videos (files not found).")
if not all_frames_to_extract:
    print("\n   ERROR: No valid frames found to extract (check JSON content and video file availability).")
    exit()
print(f"   Total frames to extract: {len(all_frames_to_extract)}")

# --- Split Data into Train/Validation Sets ---
print("\n3. Splitting frame list into training and validation sets...")
try:
    train_frames_meta, val_frames_meta = train_test_split(
        all_frames_to_extract,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        # Stratify based on labels to maintain class balance in splits
        stratify=[item['label'] for item in all_frames_to_extract]
    )
    print(f"   Training frames: {len(train_frames_meta)}")
    print(f"   Validation frames: {len(val_frames_meta)}")
except ValueError as e:
    print(f"   WARNING during train/val split: {e}")
    print("   This might happen if a class has too few samples for stratification.")
    # Decide how to proceed: exit, or split without stratification?
    # For simplicity, let's try without stratification if the first attempt failed.
    print("   Attempting split without stratification...")
    try:
       train_frames_meta, val_frames_meta = train_test_split(
           all_frames_to_extract,
           test_size=VALIDATION_SPLIT,
           random_state=RANDOM_SEED
       )
       print(f"   Training frames (non-stratified): {len(train_frames_meta)}")
       print(f"   Validation frames (non-stratified): {len(val_frames_meta)}")
    except Exception as e_nostrat:
        print(f"   ERROR: Failed to split even without stratification: {e_nostrat}")
        exit()


# --- Extraction Process (Chunking Strategy) ---
print(f"\n4. Starting frame extraction to: {EXTRACTION_BASE_DIR}")
print(f"   (Using chunking strategy: {CHUNK_DURATION_SECONDS}s chunks)")

os.makedirs(EXTRACTION_BASE_DIR, exist_ok=True)
extraction_errors = 0 # Count for read/write/resize errors
memory_errors = 0     # Count for MemoryError during chunk load
start_extract_time = time.time()

# Loop through train and validation splits
for split_name, frame_meta_list in [('train', train_frames_meta), ('val', val_frames_meta)]:
    print(f"\n   Processing split: {split_name.upper()}")
    split_dir = os.path.join(EXTRACTION_BASE_DIR, split_name)
    positive_dir = os.path.join(split_dir, 'positive')
    negative_dir = os.path.join(split_dir, 'negative')
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    # Group frames by video path for efficient processing
    frames_by_video_path = {}
    for item in frame_meta_list:
        video_path = item['video_path']
        if video_path not in frames_by_video_path:
            frames_by_video_path[video_path] = []
        # Store only necessary info for the video loop
        frames_by_video_path[video_path].append({'video_name': item['video_name'],
                                                 'frame_idx': item['frame_idx'],
                                                 'label': item['label']})

    # Process video by video within the current split
    for video_path, frames_info in tqdm(frames_by_video_path.items(), desc=f"  Videos ({split_name})"):
        if not frames_info: continue # Skip if no frames needed for this video

        # Sort required frames by index - CRUCIAL for chunking
        frames_info.sort(key=lambda x: x['frame_idx'])

        cap = None
        required_frame_pointer = 0 # Index into sorted frames_info list for this video
        total_frames_saved_for_video = 0
        video_processed_successfully = True # Flag for final message

        try:
            # --- Open Video and Get Properties ---
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"\n     WARNING: Could not open video {video_path}. Skipping {len(frames_info)} frames.")
                extraction_errors += len(frames_info) # Count all frames for this failed video as errors
                video_processed_successfully = False
                continue # Skip to next video

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_name_short = os.path.basename(video_path)

            # Basic validation of video properties
            if fps <= 0 or total_frames_in_video <= 0:
                print(f"\n     WARNING: Invalid metadata (FPS:{fps:.2f} Total Frames:{total_frames_in_video}) for {video_name_short}. Skipping.")
                extraction_errors += len(frames_info) # Count all frames for this failed video as errors
                video_processed_successfully = False
                continue # Skip to next video

            # --- Calculate Chunk Size ---
            chunk_size_frames = math.ceil(CHUNK_DURATION_SECONDS * fps)
            if chunk_size_frames <= 0: chunk_size_frames = 5000 # Reasonable fallback if FPS is weird
            # print(f"\n     Processing {video_name_short} (FPS:{fps:.2f}, Frames:{total_frames_in_video}, Chunk:{chunk_size_frames})") # Optional verbose log

            # --- Loop Through Chunks ---
            current_chunk_start_frame = 0
            while current_chunk_start_frame < total_frames_in_video and required_frame_pointer < len(frames_info):
                chunk_end_frame = min(current_chunk_start_frame + chunk_size_frames, total_frames_in_video)
                # print(f"       Chunk: Frames {current_chunk_start_frame} - {chunk_end_frame-1}") # Optional verbose log

                current_chunk_frames = [] # List to hold frames for this chunk
                chunk_load_success = True
                try:
                    # --- Load current chunk into RAM ---
                    # print(f"         Loading chunk...", end="") # Optional verbose log
                    load_chunk_start_time = time.time()
                    # Seek once to the start of the chunk
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_chunk_start_frame)
                    frames_read_in_chunk = 0
                    effective_chunk_end_frame = chunk_end_frame # Store original end frame for logic below

                    for i in range(chunk_size_frames):
                        actual_frame_index = current_chunk_start_frame + i
                        if actual_frame_index >= effective_chunk_end_frame: break # Reached end boundary

                        ret, frame = cap.read()
                        if not ret:
                            # Failed to read a frame within the expected chunk range
                            # print(f"\n         Warning: Failed to read frame {actual_frame_index} in {video_name_short}.")
                            effective_chunk_end_frame = actual_frame_index # Adjust effective end frame based on where read failed
                            chunk_load_success = False # Mark chunk as potentially incomplete
                            break # Stop loading this chunk
                        current_chunk_frames.append(frame)
                        frames_read_in_chunk += 1
                    # load_chunk_end_time = time.time()
                    # print(f" Done ({len(current_chunk_frames)} frames in {load_chunk_end_time - load_chunk_start_time:.2f}s).") # Optional verbose log

                    if len(current_chunk_frames) == 0 and effective_chunk_end_frame > current_chunk_start_frame:
                         # print(f"         Warning: Failed to load any frames for chunk {current_chunk_start_frame}-{effective_chunk_end_frame-1}")
                         chunk_load_success = False # Confirm chunk load failure

                    # --- Process required frames that fall within this loaded chunk ---
                    if chunk_load_success or len(current_chunk_frames) > 0: # Proceed if we loaded at least some frames
                        frames_processed_in_chunk = 0
                        while required_frame_pointer < len(frames_info):
                            req_frame_info = frames_info[required_frame_pointer]
                            required_idx = req_frame_info['frame_idx']

                            # Check if the required frame is within the *intended* chunk range
                            # Use effective_chunk_end_frame because we might not have read the whole chunk
                            if current_chunk_start_frame <= required_idx < effective_chunk_end_frame:
                                index_in_chunk = required_idx - current_chunk_start_frame

                                # Check if frame was actually loaded into RAM for this chunk
                                if 0 <= index_in_chunk < len(current_chunk_frames):
                                    frame_to_process = current_chunk_frames[index_in_chunk]

                                    # --- Resize ---
                                    try:
                                      frame_resized = cv2.resize(frame_to_process, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                                    except Exception as resize_e:
                                       # Log resize error specifically
                                       print(f"\n         ERROR resizing frame {required_idx} from {video_name_short}: {resize_e}")
                                       extraction_errors += 1
                                       required_frame_pointer += 1 # Move pointer even on error
                                       continue # Skip saving this frame

                                    # --- Determine Save Path ---
                                    label = req_frame_info['label']
                                    video_name = req_frame_info['video_name'] # Original video name for file
                                    output_dir = positive_dir if label == 1 else negative_dir
                                    base_video_name = os.path.splitext(video_name)[0].replace('.', '_').replace(' ', '_') # Sanitize
                                    output_filename = f"{base_video_name}_frame_{required_idx}.jpg"
                                    output_path = os.path.join(output_dir, output_filename)

                                    # --- Save Image ---
                                    try:
                                        cv2.imwrite(output_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                                        total_frames_saved_for_video += 1
                                        frames_processed_in_chunk += 1
                                    except Exception as write_e:
                                        # Log write error specifically
                                        print(f"\n         ERROR writing {output_path}: {write_e}")
                                        extraction_errors += 1

                                else:
                                    # Frame index was in range, but frame wasn't loaded (read error earlier in chunk)
                                    # print(f"\n         Warning: Required frame {required_idx} (idx_in_chunk {index_in_chunk}) was in range but not loaded for {video_name_short}. Total loaded in chunk: {len(current_chunk_frames)}")
                                    extraction_errors += 1 # Count as error

                                required_frame_pointer += 1 # Move pointer only if processed or encountered specific error above

                            elif required_idx >= effective_chunk_end_frame:
                                # This required frame is beyond the frames we successfully loaded/intended for this chunk
                                break # Break inner loop to load next chunk
                            else:
                                # This required frame index is *before* current chunk start.
                                # Should not happen with sorted list and correct outer loop logic. Log warning.
                                # print(f"\n         Warning: Logic error? Required frame {required_idx} is before chunk start {current_chunk_start_frame}.")
                                required_frame_pointer += 1 # Skip this frame and move pointer

                        # if frames_processed_in_chunk > 0: # Optional verbose log
                        #     print(f"       Processed {frames_processed_in_chunk} required frames from chunk.")

                except MemoryError:
                    print(f"\n\n     CRITICAL: MemoryError loading chunk [{current_chunk_start_frame}-{chunk_end_frame-1}] for video {video_name_short}!")
                    print(f"     Try reducing CHUNK_DURATION_SECONDS ({CHUNK_DURATION_SECONDS}s) or ensure sufficient RAM.")
                    memory_errors += 1
                    remaining_frames = len(frames_info) - required_frame_pointer
                    if remaining_frames > 0:
                        print(f"     Skipping remaining {remaining_frames} required frames for this video.")
                        extraction_errors += remaining_frames # Count skipped frames as errors
                    video_processed_successfully = False
                    break # Stop processing chunks for this video

                finally:
                    # --- CRITICAL: Free chunk memory ---
                    # print(f"       Clearing memory for chunk...") # Optional verbose log
                    del current_chunk_frames # Delete the list holding frames
                    gc.collect() # Explicitly run garbage collection
                    # print("       Memory cleared.") # Optional verbose log

                # --- Move to the next chunk ---
                # Use the potentially adjusted effective_chunk_end_frame to ensure progress even if reads failed
                next_chunk_start = effective_chunk_end_frame

                # **UPDATED Stalling Check**
                # Check if the *next* start frame failed to advance past the *current* one,
                # but we are not at the end of the video yet.
                if next_chunk_start <= current_chunk_start_frame and current_chunk_start_frame < total_frames_in_video:
                    # Optional: Print an informational message instead of an error
                    # print(f"\n     INFO: Chunk end ({next_chunk_start}) didn't advance past start ({current_chunk_start_frame}) for {video_name_short}. Likely due to read errors. Ending processing for this video.")
                    video_processed_successfully = False # Mark as having issues
                    # Errors should have been counted during failed read attempts
                    break # Exit chunk loop for this video safely

                # Update the start frame for the next iteration
                current_chunk_start_frame = next_chunk_start


        except Exception as e:
            # Catch-all for unexpected errors during the processing of a single video
            print(f"\n     UNEXPECTED ERROR during processing for video {video_name_short}: {e}")
            traceback.print_exc()
            # Count remaining frames as errors if processing fails mid-way
            remaining_frames = len(frames_info) - required_frame_pointer
            if remaining_frames > 0:
                 print(f"     Assuming remaining {remaining_frames} required frames for this video could not be processed.")
                 extraction_errors += remaining_frames
            video_processed_successfully = False

        finally:
            # --- Release Video Capture Object ---
            if cap is not None and cap.isOpened():
                cap.release()
            # Optional: Log success/failure per video
            # if video_processed_successfully:
            #     print(f"     Finished {video_name_short}. Total frames saved: {total_frames_saved_for_video}")
            # else:
            #     print(f"     Finished {video_name_short} with errors.")

# --- Final Summary ---
extract_time_elapsed = time.time() - start_extract_time
print("\n" + "="*50)
print("      Frame Extraction Summary")
print("="*50)
print(f"Extraction finished in {extract_time_elapsed // 60:.0f} minutes {extract_time_elapsed % 60:.1f} seconds.")
print(f"Frames saved to: {EXTRACTION_BASE_DIR}")
if memory_errors > 0:
    print(f"WARNING: Encountered MemoryError for {memory_errors} chunk(s). Consider reducing CHUNK_DURATION_SECONDS.")
print(f"Total extraction errors (open/read/resize/write/skip): {extraction_errors}")

# Optional: Verify final counts
print("\nVerifying extracted file counts...")
try:
  train_pos_dir = os.path.join(EXTRACTION_BASE_DIR, 'train', 'positive')
  train_neg_dir = os.path.join(EXTRACTION_BASE_DIR, 'train', 'negative')
  val_pos_dir = os.path.join(EXTRACTION_BASE_DIR, 'val', 'positive')
  val_neg_dir = os.path.join(EXTRACTION_BASE_DIR, 'val', 'negative')

  train_pos = len(os.listdir(train_pos_dir)) if os.path.exists(train_pos_dir) else 0
  train_neg = len(os.listdir(train_neg_dir)) if os.path.exists(train_neg_dir) else 0
  val_pos = len(os.listdir(val_pos_dir)) if os.path.exists(val_pos_dir) else 0
  val_neg = len(os.listdir(val_neg_dir)) if os.path.exists(val_neg_dir) else 0

  total_saved = train_pos + train_neg + val_pos + val_neg
  print(f" Found: Train(Pos:{train_pos}, Neg:{train_neg}), Val(Pos:{val_pos}, Neg:{val_neg})")
  print(f" Total files saved: {total_saved}")
  # Calculate expected based on original list minus total errors counted
  expected_total = len(all_frames_to_extract) - extraction_errors
  print(f" Expected (approx based on errors): {expected_total}")
  if total_saved < expected_total:
       print(f" INFO: Saved count ({total_saved}) is less than expected ({expected_total}). Check logs for specific errors.")
  elif total_saved > expected_total:
       print(f" WARNING: Saved count ({total_saved}) is MORE than expected ({expected_total}). This might indicate an issue in error counting.")

except Exception as count_e:
  print(f" Could not verify final counts: {count_e}")

print("\n--- Extraction Script Finished ---")
