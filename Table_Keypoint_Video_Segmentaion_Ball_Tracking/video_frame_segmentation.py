#############################################################################
# Part 0: Setup, Imports, Configuration
#############################################################################
print("--- Part 0: Setup ---")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms # Use datasets.ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import gc
import traceback

# --- Configuration ---
# **** IMPORTANT: Verify this path matches the output of your extraction script ****
EXTRACTION_BASE_DIR = "/content/extracted_frames_chunked"

# **** IMPORTANT: Verify this path exists on your Google Drive ****
CHECKPOINT_DIR = "/content/drive/MyDrive/SMAI_Project_Checkpoints"
# Define the name for the best model checkpoint file
BEST_MODEL_FILENAME = "efficientnetB0_chunked_frames_best.pth"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, BEST_MODEL_FILENAME)

# Hyperparameters
IMG_SIZE = 224  # EfficientNet-B0 input size
BATCH_SIZE = 256 # Use large batch size (ensure GPU RAM is sufficient)
NUM_EPOCHS = 15 # Adjust as needed (start with 15-25)
LEARNING_RATE = 0.001
# Note: Validation split is implicitly handled by the folder structure

# --- Mount Google Drive ---
# Needed for saving checkpoints to Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    # Ensure the checkpoint directory exists on Drive
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoint directory set to: {CHECKPOINT_DIR}")
except ImportError:
    print("Not running in Google Colab or Drive mounting failed.")
    print("Checkpoints will be saved locally if CHECKPOINT_DIR is a local path.")
    # If running locally, ensure CHECKPOINT_DIR exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
except Exception as e:
    print(f"Error mounting Google Drive or creating checkpoint directory: {e}")
    print("Please ensure the CHECKPOINT_DIR path is correct and accessible.")
    # Decide if you want to exit or continue saving locally
    # exit() # Uncomment to exit if Drive mount fails

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#############################################################################
# Part 1: Data Loading and Preparation
#############################################################################
print("\n--- Part 1: Data Loading & Preparation ---")

# --- Define Transformations ---
# Use ImageNet stats as EfficientNet was pretrained on it
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    # Training transforms: Resize, Augmentation (Flip), Convert to Tensor, Normalize
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Simple augmentation
        transforms.ToTensor(),
        normalize
    ]),
    # Validation transforms: Resize, Convert to Tensor, Normalize (NO augmentation)
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ]),
}

# --- Create Datasets and DataLoaders using ImageFolder ---
print(f"Loading datasets from: {EXTRACTION_BASE_DIR}")
try:
    # Create datasets using ImageFolder, pointing to the base extraction directory
    image_datasets = {x: datasets.ImageFolder(os.path.join(EXTRACTION_BASE_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # Increase num_workers if possible (reading images is faster than video)
    # Start with 4 if using GPU, adjust based on system performance/errors
    num_workers = 4 if torch.cuda.is_available() else 0
    print(f"Using num_workers = {num_workers} for DataLoaders.")

    # Create DataLoaders for batching and parallel loading
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=BATCH_SIZE,
                                 shuffle=(x == 'train'), # Shuffle only training data
                                 num_workers=num_workers,
                                 pin_memory=torch.cuda.is_available(), # Speeds up CPU->GPU transfer
                                 persistent_workers=True if num_workers > 0 else False) # Keep workers alive
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes # Get class names ['negative', 'positive'] or similar

    # Validate dataset loading
    if not dataset_sizes['train'] or not dataset_sizes['val']:
        print("\nERROR: Training or validation dataset is empty!")
        print(f"Please check the contents of '{EXTRACTION_BASE_DIR}/train' and '{EXTRACTION_BASE_DIR}/val'.")
        print("Ensure the frame extraction script ran successfully and populated these folders.")
        exit()

    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")
    print(f"Classes found: {class_names} -> Indices: {image_datasets['train'].class_to_idx}")

    # Determine the index for the positive class (important for metrics)
    positive_class_index = image_datasets['train'].class_to_idx.get('positive', -1)
    if positive_class_index == -1:
        print("\nWARNING: Class 'positive' not found in training data folders.")
        print("         Please ensure your positive frames are in a subfolder named 'positive'.")
        print("         Metrics like Precision/Recall might be incorrect.")
        # Assuming the second class found is positive if 'positive' is missing
        if len(class_names) > 1:
             positive_class_index = 1
             print(f"         Assuming positive class is '{class_names[1]}' with index {positive_class_index}.")
        else:
             print("         Cannot determine positive class index.")
             # Decide how to proceed - maybe exit? For now, continue with potential metric issues.
    else:
         print(f"Positive class ('positive') has index: {positive_class_index}")


except FileNotFoundError:
    print(f"\nERROR: Extracted data directory not found at '{EXTRACTION_BASE_DIR}'")
    print("Please ensure the frame extraction script (extract_frames_chunked.py or similar) ran successfully")
    print(f"and created the '{EXTRACTION_BASE_DIR}/train' and '{EXTRACTION_BASE_DIR}/val' subdirectories.")
    exit()
except Exception as e:
    print(f"\nERROR: An unexpected error occurred while creating datasets/dataloaders: {e}")
    traceback.print_exc()
    exit()


#############################################################################
# Part 2: Model Definition
#############################################################################
print("\n--- Part 2: Model Definition ---")

print("Loading pre-trained EfficientNet-B0 model...")
# Load EfficientNet-B0 with pre-trained weights from ImageNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# --- Freeze Feature Extractor Layers ---
# We only want to train the classifier initially
print("Freezing feature extractor layers (model.features)...")
for param in model.features.parameters():
    param.requires_grad = False

# --- Modify the Classifier ---
# Get the number of input features for the existing classifier
num_ftrs = model.classifier[1].in_features
print(f"Number of features in original classifier: {num_ftrs}")

# Replace the classifier with a new sequence: Dropout -> Linear layer
# Output is 1 neuron for binary classification (using BCEWithLogitsLoss)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), # Use dropout probability from EfficientNet paper
    nn.Linear(num_ftrs, 1)
)
print("Replaced model classifier for binary classification.")

# --- Initialize the New Classifier Layer ---
# Apply He (Kaiming) Normal initialization to the weights of the new linear layer
print("Applying He Normal initialization to the new classifier weights...")
for m in model.classifier.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # Initialize bias to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --- Move Model to Device ---
model = model.to(device)
print(f"Model loaded, modified, initialized, and moved to {device}.")
# Optional: Print model summary
# print(model)


#############################################################################
# Part 3: Training Setup (Loss, Optimizer, Scheduler)
#############################################################################
print("\n--- Part 3: Training Setup ---")

# --- Loss Function ---
# BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy - numerically stable
# Expects raw logits from the model (output of the linear layer)
criterion = nn.BCEWithLogitsLoss()
print(f"Loss function: {criterion}")

# --- Optimizer ---
# AdamW is often a good choice for fine-tuning
# IMPORTANT: We pass *only* the parameters of the new classifier to the optimizer
# This ensures that the frozen feature extractor layers are not updated
optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
print(f"Optimizer: AdamW (Optimizing only classifier parameters) with LR={LEARNING_RATE}")

# --- Learning Rate Scheduler (Optional but Recommended) ---
# Reduces the learning rate by a factor of 'gamma' every 'step_size' epochs
# Helps in fine-tuning as training progresses
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print(f"LR Scheduler: StepLR (step_size=7, gamma=0.1)")


#############################################################################
# Part 4: Training Loop
#############################################################################
print("\n--- Part 4: Training Loop ---")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    start_train_time = time.time() # Track total training time

    # Variables to track the best model based on validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0 # For early stopping

    # Dictionary to store metrics history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'val_precision': [], 'val_recall': [], 'val_f1': []}

    print("\nStarting Training...")
    print("-" * 20)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_samples_processed = 0 # Track samples accurately per epoch

            # For validation phase, collect all predictions and labels for sklearn metrics
            all_preds_val = []
            all_labels_val = []

            # Iterate over data using tqdm for a progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}")
            for inputs, labels_idx in progress_bar: # labels are indices (0 or 1) from ImageFolder

                # Move inputs and labels to the configured device (GPU/CPU)
                inputs = inputs.to(device)
                # Convert label indices to float, add dimension for loss function, move to device
                labels = labels_idx.float().unsqueeze(1).to(device) # Shape: [batch_size, 1], Type: float

                # Zero the parameter gradients before each batch
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train phase for gradient calculation
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs (raw logits)
                    outputs = model(inputs)
                    # Calculate the loss
                    loss = criterion(outputs, labels)

                    # Apply sigmoid to logits to get probabilities (0-1 range)
                    # Then threshold at 0.5 to get binary predictions (0 or 1)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() # Calculate gradients
                        optimizer.step()  # Update weights

                # --- Statistics Calculation ---
                batch_loss = loss.item() * inputs.size(0) # Loss for the current batch
                batch_corrects = torch.sum(preds == labels.data) # Correct predictions in batch

                running_loss += batch_loss
                running_corrects += batch_corrects
                num_samples_processed += inputs.size(0)

                # Store predictions and true labels for validation phase
                if phase == 'val':
                    all_preds_val.extend(preds.cpu().numpy()) # Store predictions as numpy array
                    all_labels_val.extend(labels_idx.cpu().numpy()) # Store original labels (indices) as numpy array

                # Update progress bar postfix with batch loss and accuracy
                progress_bar.set_postfix(loss=(batch_loss / inputs.size(0)),
                                         acc=(batch_corrects.double() / inputs.size(0)).item())

            # --- Epoch Metrics Calculation ---
            if num_samples_processed == 0:
                print(f"  WARNING: No samples processed in {phase} phase for epoch {epoch}. Skipping metrics.")
                # Assign default values or handle as appropriate
                epoch_loss, epoch_acc = 0.0, 0.0
                if phase == 'val': epoch_prec, epoch_recall, epoch_f1 = 0.0, 0.0, 0.0
            else:
                epoch_loss = running_loss / num_samples_processed
                epoch_acc = (running_corrects.double() / num_samples_processed).item() # Final epoch accuracy

                # --- Calculate Metrics and Print Results ---
                if phase == 'train':
                    # Store training metrics
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc)
                    print(f'  {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    # Step the learning rate scheduler after training phase
                    scheduler.step()
                    print(f"  LR Scheduler stepped. Current LR: {scheduler.get_last_lr()[0]:.6f}")

                else: # Validation phase
                    # Convert collected lists to numpy arrays for sklearn
                    np_preds = np.array(all_preds_val).flatten().astype(int)
                    np_labels = np.array(all_labels_val).flatten().astype(int)

                    # Calculate Precision, Recall, F1-Score using sklearn
                    if len(np_preds) > 0:
                         # average='binary' assumes positive_label=1 by default.
                         # Explicitly set pos_label based on ImageFolder's assignment.
                         precision, recall, f1, _ = precision_recall_fscore_support(
                            np_labels, np_preds, average='binary', zero_division=0,
                            pos_label=positive_class_index # Use determined positive class index
                        )
                         epoch_prec = precision
                         epoch_recall = recall
                         epoch_f1 = f1
                         # Note: epoch_acc is already calculated above
                    else: # Handle case where validation set might be empty or skipped
                        epoch_prec, epoch_recall, epoch_f1 = 0.0, 0.0, 0.0

                    # Store validation metrics
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc)
                    history['val_precision'].append(epoch_prec)
                    history['val_recall'].append(epoch_recall)
                    history['val_f1'].append(epoch_f1)

                    print(f'  {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    print(f'           Prec: {epoch_prec:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1:.4f}')

                    # --- Checkpointing ---
                    # Save the model if validation accuracy has improved
                    if epoch_acc > best_acc:
                        print(f'  Validation accuracy improved ({best_acc:.4f} --> {epoch_acc:.4f}). Saving model...')
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_model_wts = copy.deepcopy(model.state_dict()) # Get current best weights
                        epochs_no_improve = 0 # Reset patience counter

                        # Save checkpoint to Google Drive (or local path)
                        try:
                            # Save necessary information
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': best_model_wts,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': epoch_loss, # Validation loss at best epoch
                                'accuracy': best_acc, # Best validation accuracy
                                'class_to_idx': image_datasets['train'].class_to_idx, # Save class mapping
                                'history': history # Optionally save history in checkpoint
                            }, BEST_MODEL_PATH)
                            print(f"  Checkpoint successfully saved to {BEST_MODEL_PATH}")
                        except Exception as e:
                            print(f"\n  ERROR saving checkpoint to {BEST_MODEL_PATH}: {e}")
                            traceback.print_exc()
                    else:
                         epochs_no_improve += 1
                         print(f'  Validation accuracy did not improve ({best_acc:.4f}). Patience: {epochs_no_improve}/{patience}')


        print() # Print a newline after each epoch

        # --- Early Stopping ---
        if epochs_no_improve >= patience:
             print(f"Early stopping triggered after {patience} epochs with no improvement.")
             break # Exit training loop

    # --- End of Training ---
    train_time_elapsed = time.time() - start_train_time
    print(f'\nTraining complete in {train_time_elapsed // 60:.0f}m {train_time_elapsed % 60:.0f}s')
    print(f'Best validation Accuracy: {best_acc:4f} at epoch {best_epoch}')

    # Load best model weights back into the model
    print("Loading best model weights...")
    model.load_state_dict(best_model_wts)

    return model, history

# --- Execute Training ---
# Set patience for early stopping (e.g., 5 epochs)
EARLY_STOPPING_PATIENCE = 5
model, history = train_model(model, criterion, optimizer, exp_lr_scheduler,
                             num_epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE)


#############################################################################
# Part 5: Plotting History
#############################################################################
print("\n--- Part 5: Plotting History ---")

def plot_history(history):
    # Check if history object is valid and contains expected keys
    if not history or not history.get('train_loss'):
      print("No history data found to plot.")
      return

    epochs = range(len(history['train_loss']))
    plt.style.use('ggplot') # Use a nice style for plots
    plt.figure(figsize=(14, 10)) # Larger figure size

    # Plot Training & Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training & Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Validation Precision & Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history.get('val_precision', [0]*len(epochs)), 'go-', label='Validation Precision') # Use .get for safety
    plt.plot(epochs, history.get('val_recall', [0]*len(epochs)), 'yo-', label='Validation Recall')
    plt.title('Validation Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Plot Validation F1-Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history.get('val_f1', [0]*len(epochs)), 'mo-', label='Validation F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.suptitle("Training Metrics History", fontsize=16, y=1.02) # Add overall title
    plt.show()

print("Plotting training history...")
plot_history(history)


#############################################################################
# Part 6: Visualize Sample Results
#############################################################################
print("\n--- Part 6: Visualize Sample Results ---")

def visualize_results(model, num_images=10):
    print(f"Visualizing {num_images} sample predictions from validation set...")
    was_training = model.training # Store original mode
    model.eval() # Set model to evaluation mode
    images_so_far = 0

    # Calculate rows needed for the subplot grid
    rows = (num_images + 1) // 2
    fig = plt.figure(figsize=(16, rows * 4 )) # Adjust figure size based on rows

    # Create a dataloader iterator for the validation set
    # Use a separate dataloader instance for visualization if needed, or reuse existing
    val_loader_iter = iter(dataloaders['val'])

    # Define the inverse normalization transform for displaying images correctly
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    # Get class names from the dataset used by the loader
    viz_class_names = image_datasets['val'].classes

    with torch.no_grad(): # Turn off gradients for visualization
        batch_count = 0
        # Limit batches to prevent infinite loop if loader is strange
        max_batches_to_try = (dataset_sizes['val'] // BATCH_SIZE) + 2

        while images_so_far < num_images and batch_count < max_batches_to_try:
            try:
                 # Get a batch of inputs and labels (indices)
                 inputs, labels_idx = next(val_loader_iter)
                 batch_count += 1
            except StopIteration:
                print("  Reached end of validation loader during visualization.")
                break # Stop if loader is exhausted
            except Exception as viz_e:
                print(f"  Error getting batch during visualization: {viz_e}")
                break # Stop on other errors

            inputs = inputs.to(device) # Move inputs to GPU/CPU
            # Note: Labels are NOT moved to device here, used as indices later

            outputs = model(inputs) # Get model logits
            probs = torch.sigmoid(outputs) # Convert logits to probabilities
            preds_indices = (probs > 0.5).int().cpu().flatten() # Get binary predictions (0 or 1) and move to CPU

            # Display images in the current batch
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images: break # Stop if enough images shown

                images_so_far += 1
                ax = plt.subplot(rows , 2, images_so_far) # Create subplot
                ax.axis('off') # Turn off axis lines and labels

                # Get true and predicted class names using the indices
                true_label_idx = labels_idx[j].item()
                pred_label_idx = preds_indices[j].item()
                true_label_name = viz_class_names[true_label_idx]
                pred_label_name = viz_class_names[pred_label_idx]
                pred_prob = probs[j].item() # Probability of the positive class

                # Set title with prediction, probability, and true label
                # Color title green if correct, red if incorrect
                ax.set_title(f'Pred: {pred_label_name} ({pred_prob:.2f})\nTrue: {true_label_name}',
                             color=("green" if pred_label_idx == true_label_idx else "red"),
                             fontsize=10) # Adjust font size if needed

                # Prepare image for display: move to CPU, unnormalize, permute channels
                img_display = inputs.cpu().data[j]
                img_display = inv_normalize(img_display) # Reverse normalization
                img_display = img_display.permute(1, 2, 0) # Change from (C, H, W) to (H, W, C) for matplotlib
                img_display = torch.clip(img_display, 0, 1) # Clip values to [0, 1] range after un-normalize

                # Display the image
                plt.imshow(img_display.numpy())


        # Restore the model's original training mode
        model.train(mode=was_training)
        # Adjust layout and show the plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.95) # Add space for suptitle if needed
        plt.suptitle("Sample Model Predictions on Validation Set", fontsize=14)
        plt.show()

# --- Load the Best Model from Checkpoint and Visualize ---
print("\nLoading best model weights from checkpoint for visualization...")
# Create a new instance of the model structure
vis_model = models.efficientnet_b0(weights=None) # Load structure without weights
num_ftrs = vis_model.classifier[1].in_features
vis_model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(num_ftrs, 1))

# Check if the best model checkpoint file exists
if os.path.exists(BEST_MODEL_PATH):
    try:
        # Load the saved state dict
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=device) # Load to current device
        vis_model.load_state_dict(checkpoint['model_state_dict'])
        vis_model = vis_model.to(device) # Ensure model is on the correct device
        print(f"Successfully loaded best model weights from {BEST_MODEL_PATH}")
        print(f"(Achieved validation accuracy: {checkpoint.get('accuracy', 'N/A'):.4f} at epoch {checkpoint.get('epoch', 'N/A')})")

        # Run visualization
        visualize_results(vis_model, num_images=10) # Visualize 10 images

    except Exception as e:
        print(f"\nERROR loading checkpoint or visualizing results: {e}")
        traceback.print_exc()
        print("Visualization skipped due to error.")
else:
    print(f"\nWARNING: Best model checkpoint not found at {BEST_MODEL_PATH}.")
    print("Cannot visualize results using the best model. Ensure training saved a checkpoint.")
    print("Consider running visualization with the model currently in memory (last epoch state):")
    # visualize_results(model, num_images=10) # Uncomment to visualize last epoch state

#############################################################################
# Part 7: Cleanup
#############################################################################
print("\n--- Part 7: Cleanup ---")

# Garbage collection to free up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache() # Clear GPU cache if using CUDA
print("Garbage collection and CUDA cache cleared (if applicable).")

print("\n--- Training Script Finished ---")
