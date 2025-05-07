import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split # Keep for consistency if needed later
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import segmentation_models_pytorch as smp # Using a library for U-Net
import random

# --- Configuration (Should match training script) ---
METADATA_JSON_PATH = "/content/drive/MyDrive/SMAI_Data/keypoint_training_data_improved/training_metadata.json"
BASE_DATA_DIR = "/content/temp_table_data"
MODEL_SAVE_DIR = "/content/drive/MyDrive/SMAI_Models/TableCornerUNet_Improved"
CHECKPOINT_NAME = "unet_table_corners_best.pth"

# Model & Training Parameters
MODEL_ARCHITECTURE = "unet"
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet" # Or None if trained from scratch
NUM_CLASSES = 4 # TL, TR, BR, BL
ACTIVATION = None # Matched training (used None for BCEWithLogitsLoss)

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4 # Smaller batch size for inference/visualization
VALIDATION_SPLIT = 0.15 # Needed to recreate the same validation set split
NUM_WORKERS = 2
MASK_THRESHOLD = 100 # Same as training

# Inference Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES_TO_SHOW = 5 # How many random validation samples to visualize
PREDICTION_THRESHOLD = 0.5 # Threshold for sigmoid output to create binary mask
CORNER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # BGR for OpenCV: Red, Green, Blue, Yellow
CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# --- Dataset Class (Copied from training script for consistency) ---
class TableCornerDataset(Dataset):
    def __init__(self, metadata, base_dir, image_transform=None, mask_transform=None, mask_threshold=100):
        self.metadata = metadata
        self.base_dir = base_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_threshold = mask_threshold
        # Store original image size if needed for scaling back later, but not strictly necessary for this visualization
        self._get_original_size_example()

    def _get_original_size_example(self):
        # Helper to potentially get original size, not strictly needed here
        # as we visualize on the resized image
        if self.metadata:
            item = self.metadata[0]
            frame_rel_path = item["input_frame_path"]
            frame_path = os.path.join(self.base_dir, frame_rel_path)
            try:
                img = cv2.imread(frame_path)
                if img is not None:
                    self.original_height, self.original_width = img.shape[:2]
                else:
                    self.original_height, self.original_width = None, None
            except:
                self.original_height, self.original_width = None, None
        else:
            self.original_height, self.original_width = None, None


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        frame_rel_path = item["input_frame_path"]
        frame_path = os.path.join(self.base_dir, frame_rel_path)
        try:
            image = cv2.imread(frame_path)
            if image is None: raise IOError(f"Could not read image: {frame_path}")
            original_image_for_viz = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0])) # Resize for viz
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {frame_path}: {e}")
            dummy_image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)
            dummy_mask = torch.zeros((NUM_CLASSES, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)
            dummy_orig_image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
            return dummy_image, dummy_mask, dummy_orig_image # Return dummy original too

        masks = []
        mask_shape_ref = None
        for mask_rel_path in item["corner_mask_paths"]:
            mask_path = os.path.join(self.base_dir, mask_rel_path)
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None: raise IOError(f"Could not read mask: {mask_path}")

                if mask_shape_ref is None: mask_shape_ref = mask.shape
                elif mask.shape != mask_shape_ref:
                    mask = cv2.resize(mask, (mask_shape_ref[1], mask_shape_ref[0]), interpolation=cv2.INTER_NEAREST)

                mask = (mask >= self.mask_threshold).astype(np.float32)
                masks.append(mask)
            except Exception as e:
                 print(f"Error loading mask {mask_path}: {e}")
                 masks.append(np.zeros(IMAGE_SIZE, dtype=np.float32)) # Use target size

        if len(masks) == NUM_CLASSES:
            target_mask = np.stack(masks, axis=-1)
        else:
             # Pad if necessary (shouldn't happen with good data)
             while len(masks) < NUM_CLASSES:
                 masks.append(np.zeros(IMAGE_SIZE, dtype=np.float32))
             target_mask = np.stack(masks, axis=-1)

        # Apply transformations
        processed_image = image # Keep original RGB numpy for ToPILImage
        if self.image_transform:
             processed_image = self.image_transform(processed_image) # Apply normalization etc.

        if self.mask_transform:
             target_mask = self.mask_transform(target_mask)
        else:
             target_mask = torch.from_numpy(target_mask).permute(2, 0, 1).float()
             target_mask = transforms.functional.resize(target_mask, IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)


        # Return processed image, target mask, and the resized *original BGR* image for visualization
        return processed_image, target_mask, original_image_for_viz

# --- Transformations (Only need validation transforms) ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

val_image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize,
])

# Mask transform (only resize needed if applied in Dataset __getitem__)
mask_transforms = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float()), # to C, H, W tensor
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
])


# --- Model Definition (Copied from training script) ---
def build_model(architecture, encoder, weights, in_channels=3, out_classes=4, activation=None):
    model = smp.create_model(
        arch=architecture,
        encoder_name=encoder,
        encoder_weights=weights, # Use None if you want to load weights strictly and encoder was trained from scratch
        in_channels=in_channels,
        classes=out_classes,
        activation=activation
    )
    return model

# --- Helper Function for Visualization ---
def overlay_masks(image, masks, colors, alpha=0.5):
    """Overlays masks on the image with specified colors and transparency."""
    overlay = image.copy()
    for i, mask in enumerate(masks):
        if mask.max() > 0: # Only draw if mask is not empty
             colored_mask = np.zeros_like(image, dtype=np.uint8)
             colored_mask[mask == 1] = colors[i % len(colors)]
             overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    return overlay

def find_mask_centroid(mask):
    """Finds the centroid of the largest contour in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None # No contour found

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)

    # Calculate centroid
    if M["m00"] == 0:
        return None # Avoid division by zero

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


# --- Main Inference and Post-processing ---
if __name__ == "__main__":
    print("--- Starting Table Corner Inference and Post-processing ---")

    # 1. Load Metadata and setup validation set
    print(f"Loading metadata from: {METADATA_JSON_PATH}")
    if not os.path.exists(METADATA_JSON_PATH):
        print(f"Error: Metadata file not found at {METADATA_JSON_PATH}. Exiting.")
        exit()
    try:
        with open(METADATA_JSON_PATH, 'r') as f:
            all_metadata = json.load(f)
        print(f"Loaded metadata for {len(all_metadata)} samples.")
    except Exception as e:
        print(f"Error reading metadata file: {e}. Exiting.")
        exit()

    if not all_metadata:
        print("Metadata is empty. Exiting.")
        exit()

    # Ensure we use the *exact same* validation split as during training
    _, val_meta = train_test_split(all_metadata, test_size=VALIDATION_SPLIT, random_state=42) # Use same random_state
    print(f"Using {len(val_meta)} validation samples.")

    # Create validation dataset - modified to return original image for visualization
    val_dataset = TableCornerDataset(val_meta, BASE_DATA_DIR,
                                     image_transform=val_image_transforms, # Apply normalization etc.
                                     mask_transform=mask_transforms, # Apply resize
                                     mask_threshold=MASK_THRESHOLD)


    if len(val_dataset) == 0:
         print("Error: Validation dataset is empty. Check paths and split.")
         exit()

    # 2. Load Model
    model_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_NAME)
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Exiting.")
        exit()

    model = build_model(MODEL_ARCHITECTURE, ENCODER_NAME, weights=None, out_classes=NUM_CLASSES, activation=ACTIVATION) # weights=None, we load our trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in this script matches the saved checkpoint.")
        exit()

    # 3. Perform Inference and Visualization on Random Samples
    if len(val_dataset) < NUM_SAMPLES_TO_SHOW:
        print(f"Warning: Requested {NUM_SAMPLES_TO_SHOW} samples, but validation set only has {len(val_dataset)}. Showing all.")
        indices_to_show = list(range(len(val_dataset)))
    else:
        indices_to_show = random.sample(range(len(val_dataset)), NUM_SAMPLES_TO_SHOW)

    for i in indices_to_show:
        print(f"\nProcessing sample {i+1}/{len(indices_to_show)}...")

        # Get data (processed image, ground truth mask, original BGR image for viz)
        processed_img_tensor, target_mask_tensor, original_img_bgr = val_dataset[i]

        # Prepare input tensor for model
        input_tensor = processed_img_tensor.unsqueeze(0).to(DEVICE) # Add batch dim and move to device

        with torch.no_grad():
            logits = model(input_tensor) # Shape: [1, NUM_CLASSES, H, W]
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits) # Shape: [1, NUM_CLASSES, H, W]

        # Process output: move to CPU, remove batch dim, convert to numpy
        pred_masks_prob = probs.squeeze(0).cpu().numpy() # Shape: [NUM_CLASSES, H, W]

        # Create binary masks based on threshold
        pred_masks_binary = (pred_masks_prob > PREDICTION_THRESHOLD).astype(np.uint8) # Shape: [NUM_CLASSES, H, W]

        # --- Visualization 1: Raw Mask Overlay ---
        print("Visualizing raw predicted masks...")
        viz_img_raw_overlay = overlay_masks(original_img_bgr.copy(), pred_masks_binary, CORNER_COLORS)

        # --- Post-processing: Find Keypoints (Centroids) ---
        print("Post-processing: Finding keypoint centroids...")
        keypoints = {}
        viz_img_keypoints = original_img_bgr.copy()

        for class_idx in range(NUM_CLASSES):
            mask_single_channel = pred_masks_binary[class_idx] # Shape: [H, W]
            centroid = find_mask_centroid(mask_single_channel)

            if centroid is not None:
                corner_name = CORNER_NAMES[class_idx]
                keypoints[corner_name] = centroid
                print(f"  Found {corner_name}: {centroid}")
                # Draw the keypoint on the visualization image
                cv2.circle(viz_img_keypoints, centroid, radius=5, color=CORNER_COLORS[class_idx], thickness=-1) # Filled circle
                cv2.putText(viz_img_keypoints, corner_name, (centroid[0] + 10, centroid[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLORS[class_idx], 2)
            else:
                print(f"  {CORNER_NAMES[class_idx]} not detected (no contour or empty mask).")


        # --- Display Results ---
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image (Sample {i})")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(viz_img_raw_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Raw Predicted Mask Overlay")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(viz_img_keypoints, cv2.COLOR_BGR2RGB))
        plt.title("Post-Processed Keypoints (Centroids)")
        plt.axis('off')

        plt.tight_layout()
        plt.show() # Display the plots for the current sample

    print("\n--- Inference and Visualization Complete ---")
