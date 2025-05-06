# src/data/preprocess.py
import os
import cv2
import json
import numpy as np
import mediapipe as mp
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from ..utils.analyzer import ASDBehaviorAnalyzer
from ..utils.constants import (
    RAW_DATA_DIR, JSON_DIR, PROCESSED_DATA_DIR, MEDIAPIPE_TO_BODY25
)

def process_video(video_path, output_dir, class_label="unknown"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_output = os.path.join(output_dir, "json", f"{video_name}_json")
    os.makedirs(json_output, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    all_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        frame_keypoints = []
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            keypoints = np.zeros((25, 3))
            for mp_idx, body25_idx in MEDIAPIPE_TO_BODY25.items():
                landmark = results.pose_landmarks.landmark[mp_idx]
                if landmark.visibility > 0.2:
                    keypoints[body25_idx] = [landmark.x * w, landmark.y * h, landmark.visibility]
                else:
                    keypoints[body25_idx] = [0, 0, 0]
            
            if keypoints[9][2] > 0.2 and keypoints[12][2] > 0.2:
                keypoints[8] = [(keypoints[9][0] + keypoints[12][0]) / 2,
                               (keypoints[9][1] + keypoints[12][1]) / 2,
                               min(keypoints[9][2], keypoints[12][2])]
            
            frame_keypoints.append(keypoints)
        
        all_keypoints.append(frame_keypoints)
        json_path = os.path.join(json_output, f"frame_{frame_count:06d}.json")
        with open(json_path, 'w') as f:
            json.dump([kp.tolist() for kp in frame_keypoints], f)
        
        frame_count += 1

    cap.release()
    pose.close()

    selected_keypoints = []
    for frame_keypoints in all_keypoints:
        if frame_keypoints:
            selected_keypoints.append(frame_keypoints[0])
        else:
            selected_keypoints.append(np.zeros((25, 3)))

    return {
        "video_id": video_name,
        "class_label": class_label,
        "keypoints": selected_keypoints,
        "fps": fps,
        "frame_count": frame_count
    }

def process_video_directory(json_dir, class_label):
    video_id = os.path.basename(json_dir).replace("_json", "")
    keypoints = []
    for frame_file in sorted(os.listdir(json_dir)):
        with open(os.path.join(json_dir, frame_file), 'r') as f:
            frame_keypoints = json.load(f)
            if frame_keypoints:
                keypoints.append(np.array(frame_keypoints[0]))
    
    if not keypoints:
        print(f"[WARNING] No keypoints found for {video_id}")
        return None
    
    try:
        analyzer = ASDBehaviorAnalyzer(keypoints, fps=30)
        features = analyzer.extract_features()
        features["video_id"] = video_id
        features["label"] = class_label
        return features
    except Exception as e:
        print(f"[ERROR] Failed to process {video_id}: {str(e)}")
        return None

def generate_behavior_csv():
    all_features = []
    exclude_videos = [
        "-hSduu8zDzI - Trim",
        "Pexels_5621396-uhd_2160_3840_24fps"
    ]
    for class_label in ["asd", "non_asd"]:
        json_root = os.path.join(JSON_DIR, class_label)
        video_dirs = [d for d in os.listdir(json_root) if os.path.isdir(os.path.join(json_root, d))]
        for video_dir in tqdm(video_dirs, desc=f"Analyzing {class_label}"):
            video_id = video_dir.replace("_json", "")
            if video_id in exclude_videos:
                print(f"[INFO] Skipping video {video_id} due to incorrect person detection")
                continue
            json_dir = os.path.join(json_root, video_dir)
            features = process_video_directory(json_dir, class_label)
            if features:
                all_features.append(features)
    
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(os.path.join(PROCESSED_DATA_DIR, "1_MediaPipe_ASD_Behavior_Features.csv"), index=False)
        print("Features saved to", os.path.join(PROCESSED_DATA_DIR, "1_MediaPipe_ASD_Behavior_Features.csv"))
    else:
        print("No valid features extracted!")