import os
import json
import cv2
import numpy as np
from scipy.spatial import distance as dist # For ordering points
from tqdm import tqdm
import datetime
import shutil
import warnings
import random

# --- Configuration ---
INPUT_CLUSTERED_JSON = "/content/drive/MyDrive/SMAI_Data/clustered_segmentation_quad_improved.json"
VIDEOS_DIR = "/content/RightVideo"
# Temporary storage on the local Colab machine for faster I/O
TEMP_DIR = "/content/temp_table_data" # Base temporary directory
TEMP_FRAMES_DIR = os.path.join(TEMP_DIR, "frames")
TEMP_MASKS_DIR = os.path.join(TEMP_DIR, "masks")
# Final output directory on Drive for processed data index
OUTPUT_DATA_DIR = "/content/drive/MyDrive/SMAI_Data/keypoint_training_data_improved"
OUTPUT_METADATA_JSON = os.path.join(OUTPUT_DATA_DIR, "training_metadata.json")

HIGH_QUALITY_CLUSTER = 0  # Cluster ID for high-quality tables (after remapping)

# Keypoint Mask Configuration
KEYPOINT_RADIUS = 5  # Radius of the central keypoint mask (in pixels)
BORDER_THICKNESS = 3 # Thickness of the border ring around the keypoint
MASK_VALUE = 255     # Pixel value for the keypoint mask center
BORDER_VALUE = 128   # Pixel value for the border ring

# --- Helper Functions ---

def load_data(json_path):
    """Loads data from a JSON file."""
    print(f"Loading data from: {json_path}")
    if not os.path.exists(json_path):
        print(f"Error: Input file not found at {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} entries.")
        return data
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_path}: {e}")
        return None

def save_metadata(metadata, json_path):
    """Saves metadata to a JSON file."""
    print(f"Saving metadata for {len(metadata)} samples to: {json_path}")
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Metadata saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving metadata to {json_path}: {e}")

def order_points(pts):
    """Orders 4 points: top-left, top-right, bottom-right, bottom-left."""
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # Grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :] # Flipped order to get br, tr

    # Return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int")


def find_table_corners_from_polygon(polygon_points):
    """
    Finds the four most likely corner points of a table from a polygon.
    Uses the minimum area rectangle approach for robustness.

    Args:
        polygon_points (list): List of [x, y] coordinates for the polygon.

    Returns:
        numpy.ndarray: Array of 4 corner points [x, y] in a consistent order
                       (TL, TR, BR, BL), or None if insufficient points.
    """
    if not polygon_points or len(polygon_points) < 4:
        return None

    try:
        points_array = np.array(polygon_points, dtype=np.float32)
        contour = points_array.reshape((-1, 1, 2)).astype(np.int32)

        # Use Minimum Area Rectangle to find the four bounding corners
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect) # Get the 4 corners: TL, TR, BR, BL (order may vary)
        box = np.intp(box) # Convert to integers

        # Ensure the points are ordered consistently (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        if len(box) == 4:
            ordered_box = order_points(box)
            return ordered_box
        else:
            # Fallback or error if boxPoints doesn't return 4 points (shouldn't happen with minAreaRect)
            print(f"Warning: minAreaRect did not yield 4 box points for a polygon.")
            return None

    except Exception as e:
        print(f"Error finding table corners: {e}")
        return None

def generate_keypoint_mask(image_shape, center_point, radius, border_thickness):
    """
    Generates a mask image with a central keypoint circle and a surrounding border ring.

    Args:
        image_shape (tuple): Shape of the original image (h, w) or (h, w, c).
        center_point (tuple): (x, y) coordinates of the keypoint center.
        radius (int): Radius of the central keypoint mask.
        border_thickness (int): Thickness of the border ring.

    Returns:
        numpy.ndarray: Grayscale mask image (dtype=uint8) or None on error.
    """
    try:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = int(round(center_point[0])), int(round(center_point[1]))

        # Ensure coordinates are within bounds
        if not (0 <= center_x < w and 0 <= center_y < h):
             print(f"Warning: Center point ({center_x}, {center_y}) outside image bounds ({w}x{h}). Skipping mask.")
             return None # Skip if center is outside

        # Draw the border ring
        # Use clipping aware drawing if possible, otherwise simple check
        outer_radius = radius + border_thickness
        cv2.circle(mask, (center_x, center_y), outer_radius, BORDER_VALUE, thickness=border_thickness)

        # Draw the central keypoint mask (filled circle)
        cv2.circle(mask, (center_x, center_y), radius, MASK_VALUE, thickness=-1)

        return mask
    except Exception as e:
        print(f"Error generating keypoint mask at {center_point}: {e}")
        return None


# --- Main Execution ---
def main():
    """Main function to extract frames, find corners, and generate keypoint masks."""
    print("--- Starting Keypoint Data Preparation ---")
    start_time = datetime.datetime.now()

    # 1. Load Clustered Data
    clustered_data = load_data(INPUT_CLUSTERED_JSON)
    if not clustered_data:
        print("Exiting due to issues loading clustered data.")
        return

    # 2. Setup Temporary Directories
    print(f"Setting up temporary directories in: {TEMP_DIR}")
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR) # Clean up previous runs
    os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
    os.makedirs(TEMP_MASKS_DIR, exist_ok=True)

    # 3. Process High-Quality Table Frames
    print(f"Processing frames with cluster label {HIGH_QUALITY_CLUSTER}...")
    processed_samples_metadata = [] # To store info about generated frames/masks
    processed_count = 0
    skipped_count = 0

    # Filter items first to potentially parallelize video reading later if needed
    items_to_process = []
    for idx, item in enumerate(clustered_data):
        cluster_label = item.get("segmentation", {}).get("quad_cluster_label")
        if cluster_label == HIGH_QUALITY_CLUSTER:
            items_to_process.append((idx, item)) # Store original index and item

    print(f"Found {len(items_to_process)} potential high-quality table entries.")

    # Process items video by video to minimize opening/closing files
    items_to_process.sort(key=lambda x: (x[1].get("video_name", ""), x[1].get("frame_idx", 0)))

    current_video_name = None
    cap = None

    for original_idx, item in tqdm(items_to_process, desc="Processing Frames"):
        video_name = item.get("video_name")
        frame_idx = item.get("frame_idx")
        polygon = item.get("segmentation", {}).get("mask_polygon")

        if not all([video_name, frame_idx is not None, polygon]):
            # print(f"Warning: Skipping item {original_idx} due to missing data (video/frame/polygon).")
            skipped_count += 1
            continue

        # --- Video Handling ---
        try:
            if video_name != current_video_name:
                if cap is not None:
                    cap.release()
                video_path = os.path.join(VIDEOS_DIR, video_name)
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found '{video_path}'. Skipping entries for this video.")
                    # Skip all remaining items for this video if not found
                    while items_to_process and items_to_process[0][1].get("video_name") == video_name:
                        items_to_process.pop(0)
                    current_video_name = None
                    cap = None
                    skipped_count +=1 # Count this one as skipped
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video '{video_path}'. Skipping.")
                    cap = None
                    current_video_name = None
                    skipped_count += 1
                    continue
                current_video_name = video_name
                # print(f"Opened video: {video_name}") # Can be verbose

            # Seek and read frame
            if cap is None: # Should not happen if logic is correct, but safety check
                 skipped_count += 1
                 continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Failed to read frame {frame_idx} from video '{video_name}'.")
                skipped_count += 1
                continue

        except Exception as e:
            print(f"Error processing video/frame ({video_name}, {frame_idx}): {e}")
            skipped_count += 1
            if cap is not None:
                 cap.release()
                 cap = None
                 current_video_name = None
            continue

        # --- Corner Finding ---
        corners = find_table_corners_from_polygon(polygon)

        if corners is None or len(corners) != 4:
            # print(f"Warning: Could not find 4 corners for item {original_idx}. Skipping.")
            skipped_count += 1
            continue

        # --- Save Frame and Generate/Save Masks ---
        frame_filename = f"frame_{video_name}_idx{frame_idx:06d}.png"
        frame_filepath = os.path.join(TEMP_FRAMES_DIR, frame_filename)

        try:
            # Save the original frame
            cv2.imwrite(frame_filepath, frame)

            # Generate and save masks for each corner
            corner_mask_paths = []
            valid_corners_found = True
            for i, corner in enumerate(corners):
                mask = generate_keypoint_mask(frame.shape, corner, KEYPOINT_RADIUS, BORDER_THICKNESS)
                if mask is not None:
                    mask_filename = f"mask_{video_name}_idx{frame_idx:06d}_corner{i}.png"
                    mask_filepath = os.path.join(TEMP_MASKS_DIR, mask_filename)
                    cv2.imwrite(mask_filepath, mask)
                    corner_mask_paths.append(os.path.join("masks", mask_filename)) # Relative path for metadata
                else:
                     print(f"Warning: Failed to generate mask for corner {i} in frame {frame_idx} of {video_name}.")
                     valid_corners_found = False
                     break # If one mask fails, skip this frame sample

            if valid_corners_found and len(corner_mask_paths) == 4 :
                 # Add metadata for this sample
                 processed_samples_metadata.append({
                     "original_index": original_idx,
                     "video_name": video_name,
                     "frame_index": frame_idx,
                     "input_frame_path": os.path.join("frames", frame_filename), # Relative path
                     "corner_mask_paths": corner_mask_paths, # List of 4 relative mask paths
                     "corners_coordinates_xy": corners.tolist() # Save corner coordinates (TL, TR, BR, BL)
                 })
                 processed_count += 1
            else:
                 # Clean up the frame if masks weren't fully generated
                 if os.path.exists(frame_filepath):
                     os.remove(frame_filepath)
                 # Clean up any masks that were generated before failure
                 for mask_rel_path in corner_mask_paths:
                     full_mask_path = os.path.join(TEMP_MASKS_DIR, os.path.basename(mask_rel_path))
                     if os.path.exists(full_mask_path):
                          os.remove(full_mask_path)
                 skipped_count += 1


        except Exception as e:
            print(f"Error saving frame/masks for item {original_idx}: {e}")
            skipped_count += 1
             # Clean up potentially partially saved files
            if os.path.exists(frame_filepath):
                 os.remove(frame_filepath)
            for i in range(4):
                 mask_filename_check = f"mask_{video_name}_idx{frame_idx:06d}_corner{i}.png"
                 mask_filepath_check = os.path.join(TEMP_MASKS_DIR, mask_filename_check)
                 if os.path.exists(mask_filepath_check):
                     os.remove(mask_filepath_check)


    # Release the last video capture object
    if cap is not None:
        cap.release()
        # print("Released final video capture.")

    print(f"\nProcessed {processed_count} frames successfully.")
    print(f"Skipped {skipped_count} frames due to errors or missing data.")

    # 4. Save Metadata
    if processed_samples_metadata:
        save_metadata(processed_samples_metadata, OUTPUT_METADATA_JSON)
    else:
        print("No samples were processed successfully. Metadata file not saved.")

    # 5. Final Instructions
    end_time = datetime.datetime.now()
    print("\n--- Keypoint Data Preparation Complete ---")
    print(f"Duration: {end_time - start_time}")
    print(f"Input frames saved to: {TEMP_FRAMES_DIR}")
    print(f"Keypoint masks saved to: {TEMP_MASKS_DIR}")
    print(f"Metadata index saved to: {OUTPUT_METADATA_JSON}")
    print(f"\nNext steps:")
    print(f"1. Review some images in '{TEMP_FRAMES_DIR}' and '{TEMP_MASKS_DIR}' to verify correctness.")
    print(f"2. Consider copying the contents of '{TEMP_DIR}' to a persistent location (like '{OUTPUT_DATA_DIR}') if needed, or use directly from /content for training.")
    print(f"3. Use '{OUTPUT_METADATA_JSON}' to build a Dataset loader (e.g., PyTorch/TensorFlow) for training your keypoint detection model.")
    print(f"   - The loader should read frame paths and corresponding mask paths.")
    print(f"   - Implement batching (e.g., batch_size=512) and shuffling, ensuring diversity from different videos within batches.")

if __name__ == "__main__":
    # Suppress specific OpenCV warnings if they become too noisy
    warnings.filterwarnings("ignore", category=UserWarning, module='cv2')
    main()
