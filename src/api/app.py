import os
import cv2
import numpy as np
import json
import pickle
import pandas as pd
import logging
import base64
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mediapipe as mp
import google.generativeai as genai
from src.data.preprocess import process_video
from src.data.webcam import process_webcam
from src.utils.analyzer import ASDBehaviorAnalyzer
from src.models.predict import predict_asd, load_model_and_scaler
from src.utils.visualize import visualize_keypoints_with_indices
from src.utils.constants import OUTPUT_DIR, JSON_DIR, VISUALIZATION_DIR, MEDIAPIPE_TO_BODY25

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(OUTPUT_DIR, "uploaded_videos")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Load model and scaler
try:
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        raise RuntimeError("Failed to load model or scaler")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler: {str(e)}")
    raise

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-actual-api-key")  # Replace with your key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
    logger.info("Gemini API configured successfully with model gemini-1.5-flash")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

@app.route('/predict/webcam', methods=['POST'])
def predict_webcam():
    try:
        logger.debug("Received /predict/webcam request")
        data = request.json
        if not data or 'frames' not in data:
            logger.error("No frames provided in request")
            return jsonify({"error": "No frames provided"}), 400

        frames = data.get('frames', [])
        if not frames:
            logger.error("Empty frame list")
            return jsonify({"error": "Empty frame list"}), 400

        video_id = f"webcam_{int(time.time())}"
        json_output = os.path.join(JSON_DIR, f"{video_id}_json")
        os.makedirs(json_output, exist_ok=True)
        logger.debug(f"Processing webcam video ID: {video_id}")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        )

        all_keypoints = []
        frame_count = 0
        fps = 30

        for frame_data in frames:
            try:
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning(f"Failed to decode frame {frame_count}")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                logger.debug(f"Processed frame {frame_count} with MediaPipe")

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
                logger.debug(f"Saved keypoints for frame {frame_count}")

                frame_count += 1
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                continue

        pose.close()
        logger.debug(f"Processed {frame_count} frames")

        selected_keypoints = []
        for frame_keypoints in all_keypoints:
            if frame_keypoints:
                selected_keypoints.append(frame_keypoints[0])
            else:
                selected_keypoints.append(np.zeros((25, 3)))

        if not selected_keypoints:
            logger.error("No valid keypoints extracted")
            return jsonify({"error": "No valid keypoints extracted"}), 400

        analyzer = ASDBehaviorAnalyzer(selected_keypoints, fps)
        features = analyzer.extract_features()
        logger.debug(f"Extracted features: {features}")
        pred_label, asd_prob = predict_asd(features, model, scaler)
        logger.debug(f"Prediction: {pred_label}, Probability: {asd_prob}")

        return jsonify({
            "video_id": video_id,
            "label": pred_label,
            "probability": float(asd_prob),
            "features": features
        })
    except Exception as e:
        logger.error(f"Error in /predict/webcam: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    try:
        logger.debug("Received /predict/upload request")
        if 'video' not in request.files:
            logger.error("No video file provided")
            return jsonify({"error": "No video file provided"}), 400

        file = request.files['video']
        if not file.filename.endswith(('.mp4', '.avi')):
            logger.error("Invalid file format")
            return jsonify({"error": "Invalid file format. Use .mp4 or .avi"}), 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        logger.debug(f"Saved video to {video_path}")

        result = process_video(video_path, OUTPUT_DIR, class_label="unknown")
        if result is None:
            logger.error("Failed to process video")
            return jsonify({"error": "Failed to process video"}), 500

        visualize_keypoints_with_indices((video_path, OUTPUT_DIR, "unknown"))
        analyzer = ASDBehaviorAnalyzer(result["keypoints"], result["fps"])
        features = analyzer.extract_features()
        pred_label, asd_prob = predict_asd(features, model, scaler)
        logger.debug(f"Prediction: {pred_label}, Probability: {asd_prob}")

        pred_df = pd.DataFrame([{
            "video_id": result["video_id"],
            "predicted_label": pred_label,
            "asd_probability": asd_prob
        }])
        pred_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
        pred_df.to_csv(pred_csv, mode='a', header=not os.path.exists(pred_csv), index=False)
        logger.debug(f"Saved prediction to {pred_csv}")

        return jsonify({
            "video_id": result["video_id"],
            "label": pred_label,
            "probability": float(asd_prob),
            "features": features,
            "visualization": f"/visualize/{result['video_id']}_keypoints.mp4"
        })
    except Exception as e:
        logger.error(f"Error in /predict/upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_features():
    try:
        logger.debug("Received /analyze request")
        data = request.json
        features = data.get('features', {})
        if not features:
            logger.error("No features provided")
            return jsonify({"error": "No features provided"}), 400

        analysis = {
            "arm_flapping": {
                "detected": bool(features.get("arm_flapping")),
                "left_flap_count": features.get("left_flap_count", 0),
                "right_flap_count": features.get("right_flap_count", 0),
                "mean_flap_rate": features.get("mean_flap_rate", 0),
                "description": "Repetitive arm movements indicative of ASD."
            },
            "head_banging": {
                "detected": bool(features.get("head_banging")),
                "count": features.get("head_bang_count", 0),
                "rate": features.get("head_bang_rate", 0),
                "description": "Repeated head movements, potentially self-injurious."
            },
            "body_rocking": {
                "detected": bool(features.get("body_rocking")),
                "frequency": features.get("rocking_frequency", 0),
                "description": "Rhythmic body movements."
            },
            "spinning": {
                "detected": bool(features.get("spinning")),
                "ratio": features.get("spin_ratio", 0),
                "velocity": features.get("spin_velocity", 0),
                "description": "Rotational movements of the body."
            }
        }
        logger.debug(f"Analysis result: {analysis}")
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        logger.debug("Received /chat request")
        data = request.json
        user_message = data.get('message', '')
        if not user_message:
            logger.error("No message provided")
            return jsonify({"error": "No message provided"}), 400

        response = gemini_model.generate_content(
            f"You are a helpful assistant for an ASD prediction system. Answer questions about ASD behaviors, model predictions, or system usage. User: {user_message}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )
        logger.debug(f"Gemini response: {response.text}")
        return jsonify({"response": response.text})
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualize/<filename>')
def serve_visualization(filename):
    try:
        logger.debug(f"Received /visualize request for {filename}")
        file_path = os.path.join(VISUALIZATION_DIR, filename)
        if not os.path.exists(file_path):
            logger.error(f"Visualization file not found: {file_path}")
            return jsonify({"error": "Visualization not found"}), 404
        logger.debug(f"Serving visualization: {file_path}")
        return send_file(file_path, mimetype='video/mp4')
    except Exception as e:
        logger.error(f"Error in /visualize: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)