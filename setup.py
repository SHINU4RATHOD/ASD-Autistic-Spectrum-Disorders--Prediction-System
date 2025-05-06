from setuptools import setup, find_packages

setup(
    name="ASD_Prediction",
    version="0.1.0",
    author="SHINU4RATHOD",
    author_email="shinukrathod0143@gmail.com",
    description="A machine learning system for predicting Autism Spectrum Disorder behaviors from video and webcam input using MediaPipe",
    packages=find_packages(),
    install_requires=[
        "mediapipe==0.10.21",
        "numpy==1.26.4",
        "opencv-contrib-python==4.10.0.84",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "scipy>=1.9.0",
        "tqdm>=4.65.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)