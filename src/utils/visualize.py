# src/utils/visualize.py
import os
import cv2
import json
import numpy as np
import mediapipe as mp
from ..utils.constants import VISUALIZATION_DIR, MEDIAPIPE_TO_BODY25

def visualize_keypoints_with_indices(args):
    video_path, output_dir, class_label = args
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(VISUALIZATION_DIR, f"{video_name}_keypoints.mp4")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    json_dir = os.path.join(output_dir, "json", f"{video_name}_json")
    child_keypoints = []
    for frame_file in sorted(os.listdir(json_dir)):
        with open(os.path.join(json_dir, frame_file), 'r') as f:
            frame_keypoints = json.load(f)
            child_keypoints.append(np.array(frame_keypoints[0]) if frame_keypoints else np.zeros((25, 3)))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = np.zeros((25, 3))
            h, w = frame.shape[:2]
            for mp_idx, body25_idx in MEDIAPIPE_TO_BODY25.items():
                landmark = results.pose_landmarks.landmark[mp_idx]
                if landmark.visibility > 0.2:
                    keypoints[body25_idx] = [landmark.x * w, landmark.y * h, landmark.visibility]
            
            valid_kps = keypoints[keypoints[:, 2] > 0.2][:, :2]
            if len(valid_kps) > 1:
                x_min, x_max = np.min(valid_kps[:, 0]), np.max(valid_kps[:, 0])
                y_min, y_max = np.min(valid_kps[:, 1]), np.max(valid_kps[:, 1])
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                cv2.putText(frame, "Child", (int(x_min), int(y_min)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            for kp in keypoints:
                if kp[2] > 0.2:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    pose.close()