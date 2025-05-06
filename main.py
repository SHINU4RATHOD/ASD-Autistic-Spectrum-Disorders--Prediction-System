# main.py
import pandas as pd
import numpy as np
import os
import argparse
from src.data.preprocess import process_video, generate_behavior_csv
from src.data.webcam import process_webcam
from src.models.train import train_models
from src.models.predict import predict_asd, load_model_and_scaler
from src.utils.analyzer import ASDBehaviorAnalyzer
from src.utils.visualize import visualize_keypoints_with_indices
from src.utils.constants import OUTPUT_DIR, JSON_DIR, VISUALIZATION_DIR

def main(mode):
    print("=== ASD Prediction System ===")
    
    if mode in ["preprocess", "all"]:
        print("\n=== Preprocessing ===")
        for class_label in ["asd", "non_asd"]:
            video_dir = os.path.join("data/raw", class_label)
            output_dir = os.path.join(OUTPUT_DIR, class_label)
            os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

            video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi"))]
            args = [(video_file, output_dir, class_label) for video_file in video_files]

            from multiprocessing import Pool
            with Pool() as pool:
                results = list(tqdm(pool.imap(process_video, args), total=len(args), desc=f"Processing {class_label} videos"))
            
            with Pool() as pool:
                pool.map(visualize_keypoints_with_indices, args)
            
            for result in results:
                selected_keypoints = []
                for frame_keypoints in result["keypoints"]:
                    if frame_keypoints:
                        selected_keypoints.append(frame_keypoints[0])
                    else:
                        selected_keypoints.append(np.zeros((25, 3)))
                result["keypoints"] = selected_keypoints
                json_output = os.path.join(output_dir, "json", f"{result['video_id']}_json")
                for frame_idx, keypoints in enumerate(result["keypoints"]):
                    json_path = os.path.join(json_output, f"frame_{frame_idx:06d}.json")
                    with open(json_path, 'w') as f:
                        json.dump([keypoints.tolist()], f)
        
        generate_behavior_csv()

    if mode in ["train", "all"]:
        print("\n=== Training ===")
        model, scaler = train_models()
        if model is None or scaler is None:
            print("Training failed!")
            return

    if mode in ["test", "all"]:
        print("\n================================ Testing Phase ================================")
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            return

        while True:
            choice = input("Choose input type for testing (webcam/video/quit): ").lower()
            if choice in ["webcam", "video", "quit"]:
                break
            print("Invalid choice. Please enter 'webcam', 'video', or 'quit'.")

        if choice == "quit":
            print("Exiting...")
            return

        predictions = []
        test_output_dir = os.path.join(OUTPUT_DIR, "test")
        os.makedirs(os.path.join(test_output_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(test_output_dir, "visualizations"), exist_ok=True)

        if choice == "video":
            video_path = input("Enter video file path (e.g., data/test/video.mp4): ")
            if not os.path.exists(video_path) or not video_path.endswith((".mp4", ".avi")):
                print("[ERROR] Invalid video path or format. Must be .mp4 or .avi")
                return

            print(f"Processing test video: {video_path}")
            result = process_video(video_path, test_output_dir)
            visualize_keypoints_with_indices((video_path, test_output_dir, "unknown"))
            
            analyzer = ASDBehaviorAnalyzer(result["keypoints"], result["fps"])
            features = analyzer.extract_features()
            if not features:
                print("[ERROR] Failed to extract features")
                return

            pred_label, asd_prob = predict_asd(features, model, scaler)
            predictions.append({
                "video_id": result["video_id"],
                "predicted_label": pred_label,
                "asd_probability": asd_prob
            })
            
            print(f"\nPrediction for {result['video_id']}:")
            print(f"Label: {pred_label}, ASD Probability: {asd_prob:.4f}")

        elif choice == "webcam":
            print("Starting webcam... Recording for 30 seconds or press 'q' to stop.")
            result = process_webcam(test_output_dir)
            if not result:
                return

            analyzer = ASDBehaviorAnalyzer(result["keypoints"], result["fps"])
            features = analyzer.extract_features()
            if not features:
                print("[ERROR] Failed to extract features")
                return

            pred_label, asd_prob = predict_asd(features, model, scaler)
            predictions.append({
                "video_id": result["video_id"],
                "predicted_label": pred_label,
                "asd_probability": asd_prob
            })
            
            print(f"\nPrediction for webcam input {result['video_id']}:")
            print(f"Label: {pred_label}, ASD Probability: {asd_prob:.4f}")

        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
            print("\nPredictions saved to", os.path.join(OUTPUT_DIR, "predictions.csv"))
            print(pred_df)

        print("\nTesting complete! Check visualizations in", os.path.join(test_output_dir, "visualizations"))
        print("==============================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASD Prediction Pipeline")
    parser.add_argument("--mode", choices=["preprocess", "train", "test", "all"], default="all",
                        help="Pipeline mode: preprocess, train, test, or all")
    args = parser.parse_args()
    main(args.mode)



