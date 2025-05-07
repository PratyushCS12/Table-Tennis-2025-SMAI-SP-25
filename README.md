# Table Tennis Analysis Project

This repository contains a collection of Python scripts for comprehensive analysis of table tennis videos. It includes modules for player detection and tracking, audio event classification, and advanced table/keypoint/ball tracking and segmentation.

## Overview

The project is organized into three main modules:
1.  **`Audio_Classification/`**: Contains scripts for classifying audio events like ball hits (racket vs. table) using CNN or SVM models.
2.  **`PlayerTrack/`**: Focuses on detecting table tennis players (using Grounding DINO for annotation and training a custom YOLO model), tracking players with various algorithms (ByteTrack, DeepSORT, BoTSORT, custom methods), and generating player position heatmaps.
3.  **`Table_Keypoint_Video_Segmentaion_Ball_Tracking/`**: Includes scripts for dataset preparation (frame extraction, labeling), training a 3D U-Net for ball segmentation, and potentially table keypoint detection, video segmentation, and trajectory rendering.

## Prerequisites

* Python 3.x
* GPU recommended for model training and inference.
* Access to Google Drive if using scripts configured for it (primarily in `Table_Keypoint_Video_Segmentaion_Ball_Tracking/`).
* Required Python packages:
    ```bash
    pip install opencv-python torch torchvision torchaudio ultralytics transformers Pillow scikit-learn pandas numpy matplotlib tqdm pyyaml soundfile librosa tensorflow joblib requests deep-sort-realtime boxmot
    ```
    (Note: It's advisable to create a virtual environment.)

## Directory Structure

/home/pratyush-harsh/Documents/smai_project/Code_Repository_SMAI/
├── Audio_Classification/
│   ├── AudioCNN.py
│   └── AudioClassification.py
├── PlayerTrack/
│   ├── groundingdino.py
│   ├── heatmap.py
│   ├── tracking_BOTSORT.py
│   ├── tracking_yolo.py             # Uses ByteTrack via Ultralytics
│   ├── tracking_yolo_color.py
│   ├── tracking_yolo_custom.py
│   ├── train_yolo.py
│   └── tracking_DeepSORT.py         # Corrected filename from trakcing_DeepSORT.py
├── README.md
├── Table_Keypoint_Video_Segmentaion_Ball_Tracking/
│   ├── Dataset_chunking_video_seg.py
│   ├── balltrack_noisy.py
│   ├── dataset_prepration_video_labeling_smai_project.py
│   ├── effecient_net_segmentation_inference.py
│   ├── keypoint_data_prep.py
│   ├── keypoint_inference.py
│   ├── keypoint_training.py
│   ├── polygon_smoothing.py
│   ├── trajectory_render.py
│   ├── unet_3d_unet.py
│   └── video_frame_segmentation.py

*(Storage for datasets, models, and Google Drive mount points should be configured as per user's setup, e.g., `Google_Drive_SMAI_Project_dataset/` or a `local_data/` directory).*

## Configuration

Before running any script, ensure you update the path configurations defined at the beginning of each file to match your environment and data locations.
* **Google Drive Paths:** For scripts in `Table_Keypoint_Video_Segmentaion_Ball_Tracking/` like `Dataset_chunking_video_seg.py` and `balltrack_noisy.py`, ensure paths point to your Google Drive locations if used.
* **Local Paths:** For scripts in `PlayerTrack/` and `Audio_Classification/`, update paths to your local storage.
* **Model Paths:** Ensure paths to pre-trained or custom-trained models are correctly set in the respective scripts.

## Key Trained Model Paths

This section outlines where key trained models are typically saved or should be configured:

* **Custom YOLO Player Detector:**
    * Trained by: `PlayerTrack/train_yolo.py`
    * Typical Output Path: `./runs/detect/tt_player_trackingX/weights/best.pt` (where X is an experiment number). This path is then used as input for the various tracking scripts in `PlayerTrack/`.
* **3D U-Net Ball Segmentation Model:**
    * Trained by: `Table_Keypoint_Video_Segmentaion_Ball_Tracking/balltrack_noisy.py`
    * Configured Output Path: `MODEL_SAVE_PATH` within the script, e.g., `/content/drive/MyDrive/SMAI_Project_dataset/3d_unet_ball_segmentation_v6_resume.pth` (if using Google Drive).
* **Audio Hit Classification Models:**
    * CNN Model: Trained by `Audio_Classification/AudioCNN.py`. Path specified by `--cnn_out` argument (e.g., `audio_cnn_model.h5`).
    * SVM Model: Trained by `Audio_Classification/AudioClassification.py`. Path specified by `--svm_out` argument (e.g., `audio_svm_model.joblib`).
* **Keypoint Detector Model:**
    * Trained by: `Table_Keypoint_Video_Segmentaion_Ball_Tracking/keypoint_training.py` (Assumed).
    * Output Path: **Please check `keypoint_training.py` for its model saving path configuration.** This path will be needed for `keypoint_inference.py`.
* **Frame Classification Model:**
    * Training Script: **To be specified.** (It's unclear from the provided file list which script trains a generic "frame classification model" and where its weights are saved. If this refers to pixel-level segmentation, `balltrack_noisy.py` or scripts in `Table_Keypoint_Video_Segmentaion_Ball_Tracking/` like `video_frame_segmentation.py` might be relevant.)
    * Output Path: **To be specified by the user/developer based on the chosen training script.**

## Execution Pipeline

### Module 1: Audio Classification (`Audio_Classification/`)

* **`AudioCNN.py` / `AudioClassification.py`**:
    * Purpose: Classifies audio events (racket hit, table hit, no hit).
    * Inputs: `--audio_dir` (directory with `.wav` files).
    * Outputs: CSV of hits, audio clips, trained model, plots.
    * Run (e.g., `AudioCNN.py`):
        ```bash
        python Audio_Classification/AudioCNN.py --audio_dir path/to/your/audio_files --output hit_events.csv --clips_out ./audio_clips --cnn_out audio_cnn_model.h5 --plot_out ./audio_plots
        ```

### Module 2: Player Detection & Tracking (`PlayerTrack/`)

**Step 1: Player Bounding Box Annotation (`groundingdino.py`)**
* Uses Grounding DINO for zero-shot detection of "table tennis player".
* Inputs: `INPUT_DIR` (videos), `JSON_IN` (`selected_frames.json` - this file needs to be prepared, listing videos and frames of interest).
* Outputs: `JSON_OUT` (e.g., `selected_frames_bb.json` with detections), `VIS_DIR` (visualizations).
* Run: `python PlayerTrack/groundingdino.py`

**Step 2: Train Custom Player Detector (`train_yolo.py`)**
* Trains a YOLO model using annotations from `selected_frames_bb.json`.
* Inputs: `VIDEOS_DIR`, `ANNOT_JSON` (`selected_frames_bb.json`).
    * **Note:** Requires frames corresponding to `ANNOT_JSON` to be extracted and placed in the YOLO dataset structure (`OUTPUT_DIR / "images" / "train"`). The script `extract_positive_frames.py` (not listed in the new tree) was previously used for this. Ensure frames are prepared.
* Outputs: YOLO dataset structure, trained YOLO model (e.g., in `./runs/detect/tt_player_trackingX/weights/best.pt`).
* Run: `python PlayerTrack/train_yolo.py`

**Step 3: Player Tracking (Choose one or more methods)**
* These scripts use the YOLO model trained in Step 2.
    * `tracking_yolo_color.py`: Custom tracking with HSV color matching.
    * `tracking_yolo.py`: Ultralytics' built-in tracking with ByteTrack.
    * `tracking_DeepSORT.py`: DeepSORT algorithm.
    * `tracking_BOTSORT.py`: BoTSORT algorithm.
    * `tracking_yolo_custom.py`: Simpler custom tracking logic.
* Inputs: `IN_VIDEO`, Path to trained YOLO model.
* Outputs: Annotated video and a CSV file with trajectory data.
* Run (example for DeepSORT): `python PlayerTrack/tracking_DeepSORT.py`

**Step 4: Player Position Analysis - Heatmaps (`heatmap.py`)**
* Generates heatmaps from the CSV output of a tracking script.
* Inputs: `VIDEO` (for dimensions), `CSV` (tracking data from Step 3).
* Outputs: Saved heatmap images.
* Run: `python PlayerTrack/heatmap.py`

### Module 3: Advanced Segmentation & Tracking (`Table_Keypoint_Video_Segmentaion_Ball_Tracking/`)

This module contains scripts for more complex scene understanding.

**Step 1: Dataset Preparation (Frame Extraction - `Dataset_chunking_video_seg.py`)**
* Extracts frames based on `selected_frames.json`, splits into train/val. Often uses Google Drive for I/O.
* Run: `python Table_Keypoint_Video_Segmentaion_Ball_Tracking/Dataset_chunking_video_seg.py`

**Step 2: Ball Segmentation - 3D U-Net Training (`balltrack_noisy.py`)**
* Trains a 3D U-Net for ball segmentation. Often uses Google Drive for I/O and model saving.
* Run: `python Table_Keypoint_Video_Segmentaion_Ball_Tracking/balltrack_noisy.py`

**Other Scripts in `Table_Keypoint_Video_Segmentaion_Ball_Tracking/`:**
These scripts suggest functionalities for table/court segmentation, keypoint detection, and trajectory visualization. Their specific usage and interplay require direct examination:
* `dataset_prepration_video_labeling_smai_project.py`: Video labeling and dataset preparation.
* `effecient_net_segmentation_inference.py`: Inference with a segmentation model.
* `keypoint_data_prep.py`: Data preparation for keypoint detection.
* `keypoint_training.py`: **Trains the keypoint detection model.** Check this script for model output paths.
* `keypoint_inference.py`: Inference using the trained keypoint model.
* `polygon_smoothing.py`: Utility for smoothing polygonal outputs.
* `trajectory_render.py`: Visualizing trajectories.
* `unet_3d_unet.py`: Contains 3D U-Net model definitions.
* `video_frame_segmentation.py`: Generic frame segmentation tasks. Could be related to the "frame classification model" or table segmentation.

## General Notes
* Many scripts involve model loading or data processing that can be time-consuming.
* Ensure correct path configurations and that the output of one script is correctly fed as input to the next.
* Adjust parameters within scripts (batch sizes, epochs, etc.) based on your hardware and dataset.
* The file `selected_frames.json` (used by `PlayerTrack/groundingdino.py` and scripts in `Table_Keypoint_Video_Segmentaion_Ball_Tracking/`) is a crucial input that defines the frames/videos to be processed. It needs to be created or provided.



