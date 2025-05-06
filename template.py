import os
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_name = "ASD_Prediction"

list_of_files = [
    # Source code
    f"src/__init__.py",
    f"src/data/__init__.py",
    f"src/data/preprocess.py",
    f"src/data/webcam.py",
    f"src/models/__init__.py",
    f"src/models/train.py",
    f"src/models/predict.py",
    f"src/utils/__init__.py",
    f"src/utils/analyzer.py",
    f"src/utils/visualize.py",
    f"src/utils/constants.py",
    # Notebooks
    f"notebooks/train_compare_models.ipynb",
    f"notebooks/explore_features.ipynb",
    # Tests
    f"tests/__init__.py",
    f"tests/test_preprocess.py",
    f"tests/test_predict.py",
    # Data directories (empty, for videos/CSVs)
    f"data/raw/asd/.gitkeep",
    f"data/raw/non_asd/.gitkeep",
    f"data/processed/.gitkeep",
    f"data/test/.gitkeep",
    # Model and output directories
    f"models/.gitkeep",
    f"outputs/visualizations/.gitkeep",
    f"outputs/json/.gitkeep",
    # Root files
    "requirements.txt",
    "README.md",
    "config.yaml",
    "setup.py",
    "main.py"
]

# Template for Jupyter Notebook files
notebook_template = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ASD Prediction Notebook\n",
                "This notebook is part of the ASD_Prediction project."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "asd_clean",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.16"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    # Create directory if it doesn't exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Created directory: {filedir}")
    
    # Skip if file exists and is non-empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logger.info(f"File already present and non-empty: {filepath}")
        continue
    
    # Handle Jupyter Notebook files
    if filename.endswith('.ipynb'):
        with open(filepath, "w") as f:
            json.dump(notebook_template, f, indent=4)
        logger.info(f"Created Jupyter Notebook: {filepath}")
    else:
        # Create empty file or .gitkeep
        with open(filepath, "w") as f:
            pass
        logger.info(f"Created file: {filepath}")