# ASD Prediction System

A machine learning system to predict Autism Spectrum Disorder (ASD) behaviors from video and webcam input using MediaPipe for pose estimation.

## Setup

1. **Create Environment**:
   ```bash
   conda create -n asd_clean python=3.10
   conda activate asd_clean
   pip install -r requirements.txt
   ```

2. **Organize Data**:
   - Place raw videos in `data/raw/asd/` and `data/raw/non_asd/`.
   - Place test videos in `data/test/`.

## Usage

Run the pipeline with:
```bash
python main.py --mode [preprocess|train|test|all]
```

- **preprocess**: Process videos, extract features, save to `data/processed/`.
- **train**: Train models, save to `models/`.
- **test**: Test with video or webcam input, save to `outputs/`.
- **all**: Run all steps.

## Folder Structure

- `src/`: Python modules for preprocessing, training, and prediction.
- `notebooks/`: Jupyter Notebooks for experimentation.
- `data/`: Raw and processed data.
- `models/`: Trained models and scalers.
- `outputs/`: Visualizations, JSON keypoints, and predictions.
- `tests/`: Unit tests.

## Testing

The testing phase prompts for:
- **video**: Enter a video file path (e.g., `data/test/test_asd_1.mp4`).
- **webcam**: Records for 30 seconds or until 'q' is pressed.
- **quit**: Exits.

Results are saved to `outputs/predictions.csv` and visualizations to `outputs/visualizations/`.