# src/data/webcam.py
import os
import cv2
import json
import numpy as np
import mediapipe as mp
import time
from ..utils.constants import JSON_DIR, MEDIAPIPE_TO_BODY25

def process_webcam(output_dir, duration=30):
    video_name = f"webcam_{int(time.time())}"
    json_output = os.path.join(output_dir, "json", f"{video_name}_json")
    os.makedirs(json_output, exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return None

    fps = 30
    frame_count = 0
    all_keypoints = []
    start_time = time.time()

    print("Recording webcam input... Press 'q' to stop early.")
    while cap.isOpened() and (time.time() - start_time) < duration:
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

        if results.pose_landmarks:
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
        
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    selected_keypoints = []
    for frame_keypoints in all_keypoints:
        if frame_keypoints:
            selected_keypoints.append(frame_keypoints[0])
        else:
            selected_keypoints.append(np.zeros((25, 3)))

    return {
        "video_id": video_name,
        "class_label": "unknown",
        "keypoints": selected_keypoints,
        "fps": fps,
        "frame_count": frame_count
    }