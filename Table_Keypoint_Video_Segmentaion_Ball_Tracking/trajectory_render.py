# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
import collections # For deque
from torchvision import transforms
import segmentation_models_pytorch as smp
from shapely.geometry import Point, Polygon, LineString # Added LineString for clipping

# --- Configuration: 3D U-Net (Ball/Object Segmentation) ---
CONFIG_3D_UNET = {
    "DRIVE_BASE": "/content/drive/MyDrive/SMAI_Project_dataset",
    "VIDEO_DIR_NAME": "RightVideo",
    "JSON_PATH_NAME": "selected_frames.json",
    "MODEL_SAVE_PATH_NAME": "3d_unet_ball_segmentation_v6_resume.pth",
    "OUTPUT_VIDEO_NAME": "combined_extrapolated_trajectory.mp4", # New output name
    "SEQUENCE_LENGTH": 3,
    "BILINEAR_UPSAMPLING": False,
    "IMG_HEIGHT": 128,
    "IMG_WIDTH": 128,
    "CONFIDENCE_THRESHOLD": 0.3,
    "OVERLAY_ALPHA_SEG": 0.3,
    "BALL_TRAJECTORY_INSIDE_COLOR": [255, 100, 0],  # Bright Orange for inside
    "BALL_TRAJECTORY_EXTRAPOLATED_COLOR": [255, 180, 100], # Lighter Orange for extrapolated
    "OTHER_MOTION_COLOR": [200, 200, 200],
    "MIN_MOTION_AREA_FOR_CENTROID": 10,
    "TRAJECTORY_THICKNESS": 2,
    "EXTRAPOLATION_STEPS": 5, # How many steps to extrapolate
    "MIN_POINTS_FOR_VELOCITY": 3, # Min points in path to calculate velocity
    "MIN_INSIDE_POINTS_FOR_BALL": 3, # Min consecutive points inside to consider it a ball path
}

# --- Configuration: 2D Keypoint U-Net (Table Corners) ---
CONFIG_KEYPOINT_UNET = {
    "MODEL_SAVE_DIR": "/content/drive/MyDrive/SMAI_Models/TableCornerUNet_Improved",
    "CHECKPOINT_NAME": "unet_table_corners_best.pth",
    "MODEL_ARCHITECTURE": "unet",
    "ENCODER_NAME": "resnet34",
    "NUM_CLASSES": 4,
    "ACTIVATION": None,
    "IMAGE_SIZE_H": 256,
    "IMAGE_SIZE_W": 256,
    "PREDICTION_THRESHOLD": 0.5,
    "CORNER_COLORS": [(0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)],
    "CORNER_NAMES": ["TL", "TR", "BR", "BL"],
    "TABLE_POLYGON_COLOR": [0, 200, 0],
    "TABLE_POLYGON_THICKNESS": 2,
    "KEYPOINT_RADIUS": 5,
    "KEYPOINT_THICKNESS": -1,
}

# --- General Inference Parameters ---
INFERENCE_DURATION_SECONDS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Model Definitions (Identical to previous script) ---
# ... (UNet3D, DoubleConv3D, Down3D, Up3D, OutConv3D, build_keypoint_model)
# === 1.1 3D U-Net Model (Ball/Object Segmentation) ===
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
    def __init__(self, n_channels, n_classes, bilinear=False, sequence_length=3):
        super(UNet3D, self).__init__(); self.n_channels, self.n_classes, self.bilinear = n_channels, n_classes, bilinear
        self.sequence_length = sequence_length
        ch1, ch2, ch3, ch4, ch5 = 32, 64, 128, 256, 512
        self.inc = DoubleConv3D(n_channels, ch1); self.down1 = Down3D(ch1, ch2); self.down2 = Down3D(ch2, ch3); self.down3 = Down3D(ch3, ch4); self.down4 = Down3D(ch4, ch5)
        self.up1 = Up3D(ch5 + ch4, ch4, bilinear); self.up2 = Up3D(ch4 + ch3, ch3, bilinear); self.up3 = Up3D(ch3 + ch2, ch2, bilinear); self.up4 = Up3D(ch2 + ch1, ch1, bilinear)
        self.outc = OutConv3D(ch1, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits_3d = self.outc(x)
        if logits_3d.size(2) == self.sequence_length:
            logits_2d = logits_3d[:, :, self.sequence_length // 2, :, :]
        else:
            # print(f"Warning: Logits Z-dim {logits_3d.size(2)} != SEQ_LEN {self.sequence_length}. Using middle slice.") # Less verbose
            logits_2d = logits_3d[:, :, logits_3d.size(2) // 2, :, :]
        return logits_2d

# === 1.2 2D Keypoint U-Net Model (Table Corners) ===
def build_keypoint_model(architecture, encoder, weights, in_channels=3, out_classes=4, activation=None):
    model = smp.create_model(
        arch=architecture,
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=out_classes,
        activation=activation
    )
    return model

# --- 2. Preprocessing and Helper Functions ---

# === 2.1 For 3D U-Net ===
def preprocess_frame_3dunet(frame, target_height, target_width):
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    return normalized_frame

def get_mask_centroids(mask, min_area):
    centroids = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        M = cv2.moments(contour)
        area = M["m00"]
        if area > min_area:
            if area == 0: continue
            cX = int(M["m10"] / area)
            cY = int(M["m01"] / area)
            centroids.append(((cX, cY), area))
    return centroids

# === 2.2 For 2D Keypoint U-Net ===
keypoint_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
keypoint_image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CONFIG_KEYPOINT_UNET["IMAGE_SIZE_H"], CONFIG_KEYPOINT_UNET["IMAGE_SIZE_W"])),
    transforms.ToTensor(),
    keypoint_normalize,
])

def find_single_mask_centroid(mask_channel):
    contours, _ = cv2.findContours(mask_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0: return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

# === 2.3 Trajectory Helper ===
def get_velocity(path, num_points_for_velocity):
    if len(path) < num_points_for_velocity:
        return None
    # Average velocity over the last few points
    vx_sum, vy_sum = 0, 0
    count = 0
    for i in range(len(path) - num_points_for_velocity + 1, len(path)):
        vx_sum += path[i][0] - path[i-1][0]
        vy_sum += path[i][1] - path[i-1][1]
        count +=1
    if count == 0: return None
    return np.array([vx_sum / count, vy_sum / count])

def clip_line_to_polygon(line_points, polygon_shapely):
    """Clips a line (represented by a list of points) to a polygon.
       Returns the segment of the line that is inside the polygon.
    """
    if not polygon_shapely or not polygon_shapely.is_valid or len(line_points) < 2:
        return []
    line = LineString(line_points)
    intersection = line.intersection(polygon_shapely)

    if intersection.is_empty:
        return []
    if isinstance(intersection, LineString):
        return list(intersection.coords)
    elif isinstance(intersection, Point): # Should not happen often for a line segment
        return [tuple(intersection.coords[0])]
    elif hasattr(intersection, 'geoms'): # MultiLineString or GeometryCollection
        # Take the longest linestring segment from the intersection
        longest_segment = []
        max_len = 0
        for geom in intersection.geoms:
            if isinstance(geom, LineString):
                if geom.length > max_len:
                    max_len = geom.length
                    longest_segment = list(geom.coords)
        return longest_segment
    return []


# --- 3. Load Models (Identical to previous script) ---
print("--- Loading Models ---")
# ... (Model loading code for both models - same as your last script) ...
# === 3.1 Load 3D U-Net Model ===
path_3d_unet_model = os.path.join(CONFIG_3D_UNET["DRIVE_BASE"], CONFIG_3D_UNET["MODEL_SAVE_PATH_NAME"])
model_3dunet = UNet3D(n_channels=3, n_classes=2,
                      bilinear=CONFIG_3D_UNET["BILINEAR_UPSAMPLING"],
                      sequence_length=CONFIG_3D_UNET["SEQUENCE_LENGTH"])
if os.path.exists(path_3d_unet_model):
    model_3dunet.load_state_dict(torch.load(path_3d_unet_model, map_location=DEVICE))
    model_3dunet.to(DEVICE)
    model_3dunet.eval()
    print(f"3D U-Net model loaded from {path_3d_unet_model}")
else:
    print(f"Error: 3D U-Net model not found at {path_3d_unet_model}. Exiting.")
    exit()

# === 3.2 Load 2D Keypoint U-Net Model ===
path_keypoint_model = os.path.join(CONFIG_KEYPOINT_UNET["MODEL_SAVE_DIR"], CONFIG_KEYPOINT_UNET["CHECKPOINT_NAME"])
model_keypoint = build_keypoint_model(
    CONFIG_KEYPOINT_UNET["MODEL_ARCHITECTURE"],
    CONFIG_KEYPOINT_UNET["ENCODER_NAME"],
    weights=None,
    out_classes=CONFIG_KEYPOINT_UNET["NUM_CLASSES"],
    activation=CONFIG_KEYPOINT_UNET["ACTIVATION"]
)
if os.path.exists(path_keypoint_model):
    model_keypoint.load_state_dict(torch.load(path_keypoint_model, map_location=DEVICE))
    model_keypoint.to(DEVICE)
    model_keypoint.eval()
    print(f"Keypoint U-Net model loaded from {path_keypoint_model}")
else:
    print(f"Error: Keypoint U-Net model not found at {path_keypoint_model}. Exiting.")
    exit()

# --- 4. Inference and Trajectory Logic ---
print("\n--- Starting Combined Inference with Trajectory Logic ---")

video_dir_3dunet = os.path.join(CONFIG_3D_UNET["DRIVE_BASE"], CONFIG_3D_UNET["VIDEO_DIR_NAME"])
json_path_3dunet = os.path.join(CONFIG_3D_UNET["DRIVE_BASE"], CONFIG_3D_UNET["JSON_PATH_NAME"])
output_video_path = os.path.join(CONFIG_3D_UNET["DRIVE_BASE"], CONFIG_3D_UNET["OUTPUT_VIDEO_NAME"])

try:
    with open(json_path_3dunet, 'r') as f: selected_frames_data = json.load(f)
    if not selected_frames_data: raise ValueError("JSON file is empty.")
    first_video_name = list(selected_frames_data.keys())[0]
    video_path = os.path.join(video_dir_3dunet, first_video_name)
except Exception as e:
    print(f"Error accessing video from 3D U-Net JSON {json_path_3dunet}: {e}. Exiting.")
    exit()

if not os.path.exists(video_path):
    print(f"Video file not found: {video_path}. Exiting.")
    exit()

print(f"Processing video: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): print(f"Error opening video file: {video_path}. Exiting."); exit()

fps = cap.get(cv2.CAP_PROP_FPS)
original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames_to_process = int(fps * INFERENCE_DURATION_SECONDS)
print(f"Video FPS: {fps}, Dimensions: {original_w}x{original_h}")
print(f"Processing up to {num_frames_to_process} frames (approx. {INFERENCE_DURATION_SECONDS} seconds).")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (original_w, original_h))
if not out_video.isOpened(): print(f"Error: Could not open video writer. Exiting."); cap.release(); exit()

frame_buffer_3dunet = collections.deque(maxlen=CONFIG_3D_UNET["SEQUENCE_LENGTH"])
original_frame_buffer = collections.deque(maxlen=CONFIG_3D_UNET["SEQUENCE_LENGTH"])
processed_output_frame_count = 0

# Track format: {id: {"path": [(x,y),...], "is_ball_candidate": False, "last_velocity": None, "extrapolated_path": [], "last_seen_frame": frame_num, "consecutive_inside_count": 0}}
active_tracks = {}
next_track_id = 0

table_polygon_shapely = None # Store the shapely polygon globally for the current video segment

with torch.no_grad():
    for frame_num in tqdm(range(num_frames_to_process), desc="Inferencing"):
        ret, current_original_frame = cap.read()
        if not ret: break

        original_frame_buffer.append(current_original_frame.copy())
        processed_frame_for_3dunet = preprocess_frame_3dunet(
            current_original_frame, CONFIG_3D_UNET["IMG_HEIGHT"], CONFIG_3D_UNET["IMG_WIDTH"]
        )
        frame_buffer_3dunet.append(processed_frame_for_3dunet)

        final_overlayed_frame = current_original_frame.copy()
        
        if len(frame_buffer_3dunet) == CONFIG_3D_UNET["SEQUENCE_LENGTH"]:
            frame_to_process_models_on = original_frame_buffer[CONFIG_3D_UNET["SEQUENCE_LENGTH"] // 2].copy()
            final_overlayed_frame = frame_to_process_models_on.copy()

            # --- 2D Keypoint U-Net (Update table_polygon_shapely) ---
            keypoint_input_frame_rgb = cv2.cvtColor(frame_to_process_models_on, cv2.COLOR_BGR2RGB)
            keypoint_input_tensor = keypoint_image_transforms(keypoint_input_frame_rgb).unsqueeze(0).to(DEVICE)
            logits_keypoint = model_keypoint(keypoint_input_tensor)
            probs_keypoint = torch.sigmoid(logits_keypoint)
            pred_masks_binary_keypoint = (probs_keypoint > CONFIG_KEYPOINT_UNET["PREDICTION_THRESHOLD"]).squeeze(0).cpu().numpy().astype(np.uint8)

            kp_coords_scaled_dict = {}
            model_input_h_kp = CONFIG_KEYPOINT_UNET["IMAGE_SIZE_H"]
            model_input_w_kp = CONFIG_KEYPOINT_UNET["IMAGE_SIZE_W"]
            temp_table_polygon_coords_list = [None] * 4

            for class_idx in range(CONFIG_KEYPOINT_UNET["NUM_CLASSES"]):
                mask_single_channel = pred_masks_binary_keypoint[class_idx]
                centroid_kp_coords = find_single_mask_centroid(mask_single_channel)
                if centroid_kp_coords is not None:
                    scaled_cX = int(centroid_kp_coords[0] * (original_w / model_input_w_kp))
                    scaled_cY = int(centroid_kp_coords[1] * (original_h / model_input_h_kp))
                    corner_name = CONFIG_KEYPOINT_UNET["CORNER_NAMES"][class_idx]
                    kp_coords_scaled_dict[corner_name] = (scaled_cX, scaled_cY)
                    if corner_name == "TL": temp_table_polygon_coords_list[0] = (scaled_cX, scaled_cY)
                    elif corner_name == "TR": temp_table_polygon_coords_list[1] = (scaled_cX, scaled_cY)
                    elif corner_name == "BR": temp_table_polygon_coords_list[2] = (scaled_cX, scaled_cY)
                    elif corner_name == "BL": temp_table_polygon_coords_list[3] = (scaled_cX, scaled_cY)

            if all(p is not None for p in temp_table_polygon_coords_list):
                try:
                    # Ensure correct winding order for shapely if needed (usually clockwise or counter-clockwise)
                    # For a typical screen coordinate system, TL, TR, BR, BL should be fine.
                    table_polygon_shapely = Polygon(temp_table_polygon_coords_list)
                    if not table_polygon_shapely.is_valid: # Check if polygon is valid
                        print(f"Warning: Shapely polygon from {temp_table_polygon_coords_list} is not valid.")
                        table_polygon_shapely = None
                except Exception as e:
                    print(f"Warning: Could not form shapely polygon: {e}")
                    table_polygon_shapely = None
            else:
                table_polygon_shapely = None # Invalidate if not all corners found

            # Draw Keypoints and Polygon
            if table_polygon_shapely:
                cv2.polylines(final_overlayed_frame, [np.array(list(table_polygon_shapely.exterior.coords), dtype=np.int32)],
                              isClosed=True, color=CONFIG_KEYPOINT_UNET["TABLE_POLYGON_COLOR"],
                              thickness=CONFIG_KEYPOINT_UNET["TABLE_POLYGON_THICKNESS"])
            for name, (kx, ky) in kp_coords_scaled_dict.items():
                idx = CONFIG_KEYPOINT_UNET["CORNER_NAMES"].index(name)
                cv2.circle(final_overlayed_frame, (kx, ky), CONFIG_KEYPOINT_UNET["KEYPOINT_RADIUS"],
                           CONFIG_KEYPOINT_UNET["CORNER_COLORS"][idx], CONFIG_KEYPOINT_UNET["KEYPOINT_THICKNESS"])

            # --- 3D U-Net Inference for Motion ---
            sequence_np = np.stack(list(frame_buffer_3dunet), axis=0)
            sequence_np = np.transpose(sequence_np, (3, 0, 1, 2))
            sequence_tensor = torch.from_numpy(sequence_np).unsqueeze(0).to(DEVICE, non_blocking=True)
            output_logits_3dunet = model_3dunet(sequence_tensor)
            output_probs_3dunet = torch.sigmoid(output_logits_3dunet)
            output_masks_binary_3dunet = (output_probs_3dunet > CONFIG_3D_UNET["CONFIDENCE_THRESHOLD"]).cpu().numpy().astype(np.uint8)
            
            seg_core_mask_pred = output_masks_binary_3dunet[0, 0, :, :] # Assuming core mask is class 0
            seg_core_mask_resized = cv2.resize(seg_core_mask_pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            current_motion_centroids_data = get_mask_centroids(seg_core_mask_resized, CONFIG_3D_UNET["MIN_MOTION_AREA_FOR_CENTROID"])

            # --- Track Motion and Extrapolate ---
            current_frame_matched_track_ids = set()

            for centroid_xy, area in current_motion_centroids_data:
                matched_id = None
                min_dist = 50 # Max distance to associate with an existing track

                for track_id, track_data in active_tracks.items():
                    if track_data["path"]:
                        dist = np.linalg.norm(np.array(track_data["path"][-1]) - np.array(centroid_xy))
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = track_id
                
                if matched_id is not None:
                    active_tracks[matched_id]["path"].append(centroid_xy)
                    active_tracks[matched_id]["last_seen_frame"] = frame_num
                    active_tracks[matched_id]["extrapolated_path"] = [] # Reset extrapolation on new detection
                    current_frame_matched_track_ids.add(matched_id)
                else: # New track
                    active_tracks[next_track_id] = {
                        "path": [centroid_xy], "is_ball_candidate": False, "last_velocity": None,
                        "extrapolated_path": [], "last_seen_frame": frame_num, "consecutive_inside_count": 0
                    }
                    current_frame_matched_track_ids.add(next_track_id)
                    next_track_id += 1

            # Update track states (is_ball_candidate, velocity, extrapolation)
            ids_to_remove = []
            for track_id, track_data in active_tracks.items():
                if track_id not in current_frame_matched_track_ids: # Track not seen this frame
                    if frame_num - track_data["last_seen_frame"] > 3: # Timeout
                        ids_to_remove.append(track_id)
                        continue
                    # Try to extrapolate if it's a ball candidate and has velocity
                    if track_data["is_ball_candidate"] and track_data["last_velocity"] is not None:
                        if not track_data["extrapolated_path"]: # Start extrapolation
                            track_data["extrapolated_path"].append(track_data["path"][-1]) # Start from last known point
                        
                        if len(track_data["extrapolated_path"]) < CONFIG_3D_UNET["EXTRAPOLATION_STEPS"] +1 :
                            next_extrap_point = tuple(np.array(track_data["extrapolated_path"][-1]) + track_data["last_velocity"])
                            track_data["extrapolated_path"].append(next_extrap_point)
                    else: # Not a ball or no velocity, mark for removal if unseen too long
                         if frame_num - track_data["last_seen_frame"] > 5 : ids_to_remove.append(track_id)

                else: # Track was seen this frame
                    # Update velocity
                    track_data["last_velocity"] = get_velocity(track_data["path"], CONFIG_3D_UNET["MIN_POINTS_FOR_VELOCITY"])
                    
                    # Check if inside polygon and update ball candidacy
                    if table_polygon_shapely and track_data["path"]:
                        last_pt_in_path = Point(track_data["path"][-1])
                        if table_polygon_shapely.contains(last_pt_in_path):
                            track_data["consecutive_inside_count"] += 1
                        else: # Reset if it goes outside
                            if track_data["is_ball_candidate"] and track_data["consecutive_inside_count"] >= CONFIG_3D_UNET["MIN_INSIDE_POINTS_FOR_BALL"]:
                                # If it was a ball and just left, keep velocity for extrapolation
                                pass
                            else: # Not a confirmed ball or was outside already
                                track_data["consecutive_inside_count"] = 0
                                track_data["is_ball_candidate"] = False # Reset candidacy if it's outside for too long
                        
                        if track_data["consecutive_inside_count"] >= CONFIG_3D_UNET["MIN_INSIDE_POINTS_FOR_BALL"]:
                            track_data["is_ball_candidate"] = True
                    else: # No table polygon, reset inside count
                        track_data["consecutive_inside_count"] = 0
                        track_data["is_ball_candidate"] = False


            for r_id in ids_to_remove:
                if r_id in active_tracks: del active_tracks[r_id]

            # --- Drawing ---
            for track_id, track_data in active_tracks.items():
                path_to_draw = track_data["path"]
                
                if track_data["is_ball_candidate"]:
                    # Draw actual path inside polygon
                    inside_segment = []
                    if table_polygon_shapely and len(path_to_draw) > 1:
                        inside_segment = clip_line_to_polygon(path_to_draw, table_polygon_shapely)
                    
                    if len(inside_segment) > 1:
                        cv2.polylines(final_overlayed_frame, [np.array(inside_segment, dtype=np.int32).reshape((-1,1,2))],
                                      isClosed=False, color=CONFIG_3D_UNET["BALL_TRAJECTORY_INSIDE_COLOR"],
                                      thickness=CONFIG_3D_UNET["TRAJECTORY_THICKNESS"], lineType=cv2.LINE_AA)
                    elif len(inside_segment) == 1 and path_to_draw: # Single point inside
                         if table_polygon_shapely and Point(path_to_draw[-1]).within(table_polygon_shapely):
                            cv2.circle(final_overlayed_frame, tuple(map(int, path_to_draw[-1])), CONFIG_3D_UNET["TRAJECTORY_THICKNESS"], CONFIG_3D_UNET["BALL_TRAJECTORY_INSIDE_COLOR"], -1)


                    # Draw extrapolated path if it exists and originated from an inside segment
                    # We check if the *start* of extrapolation was inside or very near the boundary
                    if track_data["extrapolated_path"] and len(track_data["extrapolated_path"]) > 1 :
                        # Check if the actual path ended inside or was leaving
                        ended_inside_or_leaving = False
                        if table_polygon_shapely and path_to_draw:
                            if Point(path_to_draw[-1]).within(table_polygon_shapely):
                                ended_inside_or_leaving = True
                            elif len(path_to_draw) >1 and Point(path_to_draw[-2]).within(table_polygon_shapely) and not Point(path_to_draw[-1]).within(table_polygon_shapely):
                                ended_inside_or_leaving = True # It was leaving

                        if ended_inside_or_leaving:
                            cv2.polylines(final_overlayed_frame, [np.array(track_data["extrapolated_path"], dtype=np.int32).reshape((-1,1,2))],
                                          isClosed=False, color=CONFIG_3D_UNET["BALL_TRAJECTORY_EXTRAPOLATED_COLOR"],
                                          thickness=CONFIG_3D_UNET["TRAJECTORY_THICKNESS"]-1, lineType=cv2.LINE_AA) # Slightly thinner

                elif path_to_draw: # Not a ball candidate, draw as "other motion" if outside
                    # Draw last point as a small circle
                    if table_polygon_shapely and not Point(path_to_draw[-1]).within(table_polygon_shapely):
                         cv2.circle(final_overlayed_frame, tuple(map(int, path_to_draw[-1])), 2, CONFIG_3D_UNET["OTHER_MOTION_COLOR"], -1)
                    elif not table_polygon_shapely: # No table, draw all "other"
                         cv2.circle(final_overlayed_frame, tuple(map(int, path_to_draw[-1])), 2, CONFIG_3D_UNET["OTHER_MOTION_COLOR"], -1)


            out_video.write(final_overlayed_frame)
            processed_output_frame_count += 1

# Release resources
cap.release()
out_video.release()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print(f"\n--- Combined Inference Finished ---")
print(f"Processed and wrote {processed_output_frame_count} frames to video.")
print(f"Output video saved to: {output_video_path}")
