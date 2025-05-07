import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import segmentation_models_pytorch as smp # Using a library for U-Net

# --- Configuration ---
METADATA_JSON_PATH = "/content/drive/MyDrive/SMAI_Data/keypoint_training_data_improved/training_metadata.json"
# Base directory where 'frames' and 'masks' folders reside
BASE_DATA_DIR = "/content/temp_table_data"
MODEL_SAVE_DIR = "/content/drive/MyDrive/SMAI_Models/TableCornerUNet_Improved"
CHECKPOINT_NAME = "unet_table_corners_best.pth"

# Model & Training Parameters
MODEL_ARCHITECTURE = "unet" # Or 'unetplusplus', etc. from segmentation-models-pytorch
ENCODER_NAME = "resnet34"  # Backbone encoder
ENCODER_WEIGHTS = "imagenet" # Use pre-trained weights
NUM_CLASSES = 4  # Four corners (TL, TR, BR, BL) - outputting 4 masks
ACTIVATION = None # Use None for BCEWithLogitsLoss, 'sigmoid' for BCELoss/DiceLoss

IMAGE_SIZE = (256, 256)  # Resize images/masks to this size
BATCH_SIZE = 32         # START SMALLER! 512 is likely too large for most GPUs with 256x256 images. Adjust based on VRAM.
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50          # Adjust as needed
VALIDATION_SPLIT = 0.15  # 15% for validation
NUM_WORKERS = 2         # Number of parallel workers for data loading

# Mask processing: Treat both keypoint (255) and border (128) as positive signal
MASK_THRESHOLD = 100 # Pixels >= this value in loaded mask become 1.0

# --- Dataset Class ---

class TableCornerDataset(Dataset):
    def __init__(self, metadata, base_dir, image_transform=None, mask_transform=None, mask_threshold=100):
        """
        Args:
            metadata (list): List of dictionaries from the metadata JSON.
            base_dir (str): Path to the directory containing 'frames' and 'masks'.
            image_transform (callable, optional): Transformations for input images.
            mask_transform (callable, optional): Transformations for target masks.
            mask_threshold (int): Value above which mask pixels are considered positive (1).
        """
        self.metadata = metadata
        self.base_dir = base_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # Load Frame
        frame_rel_path = item["input_frame_path"]
        frame_path = os.path.join(self.base_dir, frame_rel_path)
        try:
            # Read image using OpenCV (BGR)
            image = cv2.imread(frame_path)
            if image is None:
                raise IOError(f"Could not read image: {frame_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
        except Exception as e:
            print(f"Error loading image {frame_path}: {e}")
            # Return a dummy sample or handle appropriately
            dummy_image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)
            dummy_mask = torch.zeros((NUM_CLASSES, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=torch.float32)
            return dummy_image, dummy_mask


        # Load and Stack Masks
        masks = []
        mask_shape = None
        for mask_rel_path in item["corner_mask_paths"]:
            mask_path = os.path.join(self.base_dir, mask_rel_path)
            try:
                # Read mask as grayscale
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                     raise IOError(f"Could not read mask: {mask_path}")
                if mask_shape is None:
                    mask_shape = mask.shape
                elif mask.shape != mask_shape:
                     # Should not happen if generated correctly, but safety check
                     print(f"Warning: Mask shape mismatch for item {idx}. Resizing {mask_path} from {mask.shape} to {mask_shape}")
                     mask = cv2.resize(mask, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)

                # --- Crucial Mask Processing ---
                # Thresholding: Convert 0, 128, 255 -> 0.0 or 1.0
                mask = (mask >= self.mask_threshold).astype(np.float32)
                masks.append(mask)
            except Exception as e:
                 print(f"Error loading mask {mask_path}: {e}")
                 # Append a dummy mask or handle appropriately
                 masks.append(np.zeros(IMAGE_SIZE, dtype=np.float32)) # Use target size

        # Stack masks along a new dimension (channel dimension)
        # Target shape: [H, W, Num_Classes]
        if len(masks) == NUM_CLASSES:
            target_mask = np.stack(masks, axis=-1)
        else:
            # Handle cases where fewer than 4 masks were loaded (should ideally not happen with good data prep)
             print(f"Warning: Expected {NUM_CLASSES} masks, got {len(masks)} for item {idx}. Padding with zeros.")
             while len(masks) < NUM_CLASSES:
                 masks.append(np.zeros(IMAGE_SIZE, dtype=np.float32)) # Use target size
             target_mask = np.stack(masks, axis=-1)


        # Apply Transformations (must handle image and mask consistently, especially spatial ones)
        # Note: Most segmentation libraries' transforms handle paired image/mask augmentation
        if self.image_transform:
            # Augmentations often require image and mask to be passed together
            # For simple resizing/normalization, apply separately
            # For complex augmentation (like albumentations), adapt this part
             image = self.image_transform(image)

        if self.mask_transform:
             # Ensure mask transformations don't use interpolation that breaks binary values
             # Usually involves converting to tensor first, then applying transforms
             target_mask = self.mask_transform(target_mask)
        else:
             # Basic conversion if no specific mask transform
             # Permute mask from [H, W, C] to [C, H, W] for PyTorch
             target_mask = torch.from_numpy(target_mask).permute(2, 0, 1).float()


        return image, target_mask

# --- Transformations ---
# Define transforms for training (with augmentation) and validation (no augmentation)
# Normalization parameters typical for ImageNet pre-trained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_image_transforms = transforms.Compose([
    transforms.ToPILImage(), # Convert numpy array (H, W, C) to PIL Image
    transforms.Resize(IMAGE_SIZE),
    # Add Augmentations here (optional but recommended)
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.ToTensor(), # Convert PIL Image (H, W, C) to Tensor (C, H, W) and scale to [0, 1]
    normalize,
])

val_image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize,
])

# Simple mask transform (resize only, ensure no bad interpolation)
# For more complex transforms (augmentation), use libraries like Albumentations
# that handle image+mask consistency.
mask_transforms = transforms.Compose([
    # Input is numpy H, W, C
    transforms.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float()), # to C, H, W tensor
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for masks
])


# --- Model Definition ---
def build_model(architecture, encoder, weights, in_channels=3, out_classes=4, activation=None):
    model = smp.create_model(
        arch=architecture,
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=out_classes,
        activation=activation # Important: Set to None for BCEWithLogitsLoss
    )
    return model

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets) # Ensure targets are float [0, 1]
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                val_pbar.set_postfix({'val_loss': loss.item()})

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saved model to {model_save_path}")

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    return history


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Table Corner Model Training ---")
    start_time = datetime.datetime.now()

    # 1. Load Metadata
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
        print("Metadata is empty. No data to train on. Exiting.")
        exit()

    # 2. Split Data
    train_meta, val_meta = train_test_split(all_metadata, test_size=VALIDATION_SPLIT, random_state=42)
    print(f"Split data: {len(train_meta)} training samples, {len(val_meta)} validation samples.")

    # 3. Create Datasets and DataLoaders
    train_dataset = TableCornerDataset(train_meta, BASE_DATA_DIR,
                                       image_transform=train_image_transforms,
                                       mask_transform=mask_transforms,
                                       mask_threshold=MASK_THRESHOLD)
    val_dataset = TableCornerDataset(val_meta, BASE_DATA_DIR,
                                     image_transform=val_image_transforms, # No augmentation for validation
                                     mask_transform=mask_transforms,
                                     mask_threshold=MASK_THRESHOLD)

    # Check if datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         print("Error: One or both datasets are empty after initialization. Check paths and data.")
         exit()

    # WARNING: Check BATCH_SIZE based on GPU memory! Start low (e.g., 8, 16, 32).
    print(f"Using Batch Size: {BATCH_SIZE}. Adjust if you encounter CUDA out-of-memory errors.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 4. Setup Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(MODEL_ARCHITECTURE, ENCODER_NAME, ENCODER_WEIGHTS, out_classes=NUM_CLASSES, activation=ACTIVATION)
    model.to(device)

    # Loss Function: Binary Cross Entropy with Logits is suitable for multi-label mask outputs
    # It expects raw logits from the model (activation=None) and targets between 0 and 1.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Train Model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, CHECKPOINT_NAME)
    print(f"Model checkpoints will be saved to: {model_save_path}")

    history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, model_save_path)

    # 6. Plot Training History (Optional)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_save_path = os.path.join(MODEL_SAVE_DIR, "training_loss_plot.png")
    plt.savefig(plot_save_path)
    print(f"Saved training history plot to: {plot_save_path}")
    # plt.show()


    end_time = datetime.datetime.now()
    print("\n--- Training Complete ---")
    print(f"Duration: {end_time - start_time}")
    print(f"Best model saved to: {model_save_path}")
