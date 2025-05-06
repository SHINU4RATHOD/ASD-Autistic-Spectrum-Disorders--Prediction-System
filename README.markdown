# ASD Prediction System
![WhatsApp Image 2025-05-04 at 10 59 08_3f936ac8](https://github.com/user-attachments/assets/2d526b0f-9f83-41b2-bd8c-f36c516b1b03)


A machine learning system to predict Autism Spectrum Disorder (ASD) behaviors in children using video and webcam input. The system leverages MediaPipe for pose estimation to extract behavioral features (e.g., arm flapping, head banging) and a custom ML model for binary classification (ASD vs. non-ASD). It includes a React frontend, Flask backend, and Dockerized deployment for scalability.

## Project Overview

The ASD Prediction System aims to assist in the early diagnosis of Autism Spectrum Disorder by analyzing video footage of children. It extracts pose landmarks using MediaPipe, derives behavioral features, and uses a machine learning model to predict ASD likelihood. The system supports both pre-recorded video uploads and real-time webcam analysis, with additional features like a chatbot (powered by Gemini API) and an AI analyzer for detailed behavioral insights.

- **Model Accuracy**: 72.8% (custom ML model, TensorFlow Lite).
- **Deployment**: Dockerized Flask backend and React frontend.
- **Target Audience**: Researchers, clinicians, and developers working on ASD diagnosis tools.

## Prerequisites

Before cloning and setting up the project, ensure you have the following installed:

- **Python 3.10**: For running the backend and ML pipeline.
- **Conda**: For environment management (e.g., Anaconda or Miniconda).
- **Node.js and npm**: For the React frontend (Node.js v16+ recommended).
- **Docker and Docker Compose**: For containerized deployment.
- **Git**: For cloning the repository.
- **Git Bash (Windows Users)**: For running shell commands.
- **Stable Internet Connection**: For downloading dependencies and accessing external data.

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/SHINU4RATHOD/ASD-Autistic-Spectrum-Disorders--DetectNet.git
cd ASD-Autistic-Spectrum-Disorders--DetectNet
```

### 2. Set Up the Python Environment
Create and activate a Conda environment, then install dependencies:
```bash
conda create -n asd_clean python=3.10
conda activate asd_clean
pip install -r ASD_Prediction/requirements.txt
```

### 3. Set Up the React Frontend
Navigate to the frontend directory, install dependencies, and build the app:
```bash
cd ASD_Prediction/frontend
npm install
npm run build
```

### 4. Download Raw Data
The raw video data is not included in the repository due to GitHub's file size limits. Download the data from the following external links and place it in the appropriate directories:

- **ASD Videos**: [Google Drive Link](https://drive.google.com/placeholder-asd-videos) → Place in `ASD_Prediction/data/raw/asd/`.
- **Non-ASD Videos**: [Google Drive Link](https://drive.google.com/placeholder-non-asd-videos) → Place in `ASD_Prediction/data/raw/non_asd/`.
- **Test Videos**: [Google Drive Link](https://drive.google.com/placeholder-test-videos) → Place in `ASD_Prediction/data/test/`.

**Note**: Create the `data/raw/asd/`, `data/raw/non_asd/`, and `data/test/` directories if they don’t exist.

### 5. (Optional) Set Up Docker
If you prefer to run the application in containers, ensure Docker and Docker Compose are installed, then build and run:
```bash
cd ASD_Prediction
docker-compose up --build
```
- The Flask backend will run on `http://localhost:5000`.
- The React frontend will be accessible at `http://localhost:3000`.

## Usage

The project supports multiple workflows for preprocessing, training, testing, and deployment.

### 1. Run the ML Pipeline
Navigate to the project root and use the `main.py` script to execute different modes:
```bash
cd ASD_Prediction
python src/main.py --mode [preprocess|train|test|all]
```

- **`preprocess`**: Processes videos using MediaPipe, extracts 35 features (e.g., `arm_flap`, `wrist_nose_distance`), and saves them to `data/processed/`.
- **`train`**: Trains the custom ML model (72.8% accuracy) and saves it to `models/` as a TensorFlow Lite file.
- **`test`**: Tests the model with video or webcam input, saving results to `outputs/`.
- **`all`**: Runs all steps sequentially.

### 2. Run the Web Application
To use the full application (including the React frontend and Flask backend):
1. Start the Flask backend:
   ```bash
   cd ASD_Prediction
   python src/api/app.py
   ```
   - The backend will run on `http://localhost:5000`.

2. In a separate terminal, start the React frontend:
   ```bash
   cd ASD_Prediction/frontend
   npm start
   ```
   - The frontend will run on `http://localhost:3000`.

3. Access the application in your browser at `http://localhost:3000`. You can:
   - Upload videos via `/predict/upload`.
   - Use webcam for real-time analysis via `/predict/webcam`.
   - Chat with the Gemini-powered chatbot via `/chat`.
   - Analyze behaviors (e.g., flapping frequency) via `/analyze`.

### 3. Run with Docker
If you set up Docker, the application will already be running after `docker-compose up`. Access the frontend at `http://localhost:3000`.

## Folder Structure

Here’s an overview of the project’s directory structure:

- `ASD_Prediction/`
  - `src/`: Python modules for preprocessing, training, prediction, and Flask API.
    - `api/app.py`: Flask backend for API endpoints.
    - `main.py`: Main script for ML pipeline.
  - `notebooks/`: Jupyter Notebooks for experimentation (e.g., `EDA1_train_compare_models.ipynb`).
  - `data/`: Directory for raw and processed data (not tracked in Git).
    - `raw/asd/`: Raw ASD videos (download from external link).
    - `raw/non_asd/`: Raw non-ASD videos (download from external link).
    - `test/`: Test videos (download from external link).
    - `processed/`: Processed features (generated during preprocessing).
  - `models/`: Trained models and scalers (e.g., TensorFlow Lite model).
  - `outputs/`: Results from testing.
    - `predictions.csv`: Prediction results.
    - `visualizations/`: Visual outputs (e.g., pose estimation overlays).
    - `json/`: Extracted keypoints in JSON format.
  - `tests/`: Unit tests for Python modules.
  - `frontend/`: React frontend application.
    - `src/`: React components (`Upload.js`, `Webcam.js`, `Chatbot.js`, `Analyzer.js`).
    - `build/`: Built frontend assets (not tracked in Git).
  - `requirements.txt`: Python dependencies.
  - `Dockerfile` & `docker-compose.yaml`: Docker configuration files.

## Testing

The testing phase allows you to evaluate the model on new inputs. Run the test mode:
```bash
python src/main.py --mode test
```

You’ll be prompted to choose an option:
- **`video`**: Provide a video file path (e.g., `data/test/test_asd_1.mp4`).
- **`webcam`**: Records for 30 seconds or until you press `q`.
- **`quit`**: Exits the testing phase.

**Outputs**:
- Predictions are saved to `outputs/predictions.csv`.
- Visualizations (e.g., pose estimation overlays) are saved to `outputs/visualizations/`.

## API Endpoints

The Flask backend provides the following endpoints (accessible at `http://localhost:5000`):
- `/predict/upload`: Upload a video file for ASD prediction (returns probability).
- `/predict/webcam`: Stream webcam data for real-time prediction (~30 FPS).
- `/chat`: Interact with the Gemini-powered chatbot for user queries.
- `/analyze`: Analyze behavioral patterns (e.g., flapping frequency).

## Contributing

We welcome contributions to improve the ASD Prediction System! To contribute:

1. **Fork the Repository**:
   ```bash
   git fork https://github.com/SHINU4RATHOD/ASD-Autistic-Spectrum-Disorders--DetectNet.git
   ```

2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes and Commit**:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

4. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Go to the GitHub repository page and create a pull request from your branch.

**Guidelines**:
- Follow the existing code style (e.g., PEP 8 for Python, ESLint for React).
- Write unit tests for new features in the `tests/` directory.
- Avoid committing large files (>100 MB). Use external storage and link in the README.
- Document your changes in the README or technical documentation.

## Troubleshooting

Here are solutions to common issues you might encounter:

- **Large File Errors on Git Push**:
  - **Problem**: GitHub rejects pushes with files larger than 100 MB.
  - **Solution**: Ensure large files are excluded in `.gitignore`. If already committed, remove them from history using `git filter-repo` (see commit history for examples).
  - **External Storage**: Raw videos are stored on Google Drive (links above).

- **Docker Timeout (Gunicorn 120s)**:
  - **Problem**: Video processing exceeds Gunicorn’s 120-second timeout.
  - **Solution**: Use Celery for asynchronous processing (planned improvement). For now, reduce video size or process locally using `main.py`.

- **Missing Keypoints in Low-Quality Videos**:
  - **Problem**: MediaPipe fails to detect keypoints in low-quality frames.
  - **Solution**: Use higher-quality videos or implement frame interpolation (future work).

- **Webcam Frame Issues**:
  - **Problem**: Webcam predictions fail due to empty frames.
  - **Solution**: Ensure proper lighting and camera permissions. Restart the webcam stream if needed.

- **Dependencies Not Found**:
  - **Problem**: Missing Python or Node.js packages.
  - **Solution**: Re-run `pip install -r requirements.txt` (Python) or `npm install` (frontend). Ensure the correct environment is activated (`conda activate asd_clean`).

## Future Improvements

- **Model Accuracy**: Improve from 72.8% to 85% by implementing multi-frame processing for dynamic behavior detection.
- **Video Processing**: Optimize by downscaling frames and using a lighter MediaPipe model.
- **Asynchronous Processing**: Integrate Celery to handle long-running video processing tasks.
- **Dataset Expansion**: Collect more labeled videos to improve model generalization.
- **Real-Time Feedback**: Add behavior alerts (e.g., "High flapping frequency detected").

## References

- MediaPipe Pose Estimation Guide (Google).
- “Video-Based ASD Detection Study” (2023).
- TensorFlow Lite Documentation.
- Flask and React Documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Developed by Shinu Rathod**  
For questions or support, contact [shinu.rathod@example.com](mailto:shinu.rathod@example.com).
