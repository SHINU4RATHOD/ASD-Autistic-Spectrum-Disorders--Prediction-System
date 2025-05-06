FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Copy backend files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application code and resources
COPY src/ src/
COPY models/ models/
COPY outputs/ outputs/
COPY config.yaml .

# Copy pre-downloaded MediaPipe model to the correct location
COPY models/mediapipe/pose_landmark_heavy.tflite /usr/local/lib/python3.10/site-packages/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite

# Expose port for Flask app
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "src.api.app:app"]