# gender-classification

Gender Classification Project - Machine Learning course (COSC 6342)

University of Houston - Spring 2025

### Team Members

- Minh Nguyen
- Mahtab Jeyhani

## Project Overview

Given a set of labeled blogs written by males and females, predict
the gender of the author of a new blog.

## Dataset

- Sample blog author dataset used in [Mukherjee and Liu, EMNLP 2010] available
  from: http://www.cs.uic.edu/~liub/FBS/blog-gender-dataset.rar
  or you can find in `data/raw/blog-gender-dataset.zip`
- The extracted file is a xlsx file, we converted it to csv format and save it as
  `gender-classification.csv` in `data/raw/`

## Requirements

- Python 3.12+
- Jupyter Notebook

## Installation

1. Clone the project
2. Create a virtual environment
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main training pipeline script (If no trained model is available in `models/`):

***NOTE: This step may take a long time to run, depending on the size of the dataset and the hardware used.***
  ```bash
  python pipeline.py
  ```
  This will execute the entire training and evaluation pipeline, including data preprocessing contrastive learning, supervised fine-tuning, and evaluation.

  - Or you can run each step separately by running the corresponding scripts in the `src/` directory.
  - For example, to run the data preprocessing step:
    ```bash
    python src/data_preprocessing.py
    ```
  - To run the contrastive learning step:
    ```bash
    python src/contrastive_learning.py
    ```
  - To run the supervised fine-tuning step (after contrastive learning, or if a contrastive pre-trained model is available):
    ```bash
    python src/supervised_fine_tune.py
    ```

2. Perform model evaluation on train/val/test datasets:
    ```bash
    python evaluate.py
    ```


## Project Structure

- `data/`: Contains the dataset and any processed data.
  - `raw/`: Original dataset files.
  - `processed/`: Processed dataset files.
- `models/`: Contains the trained models.
- `src/`: Source code for data processing, model training, and evaluation.
  - `config.py`: Configuration file for setting parameters and paths.
  - `data_preprocessing.py`: Code for loading and processing the dataset.
  - `data_augmentation.py`: Code for augmenting the dataset.
  - `dataset.py`: Code for creating custom dataset classes for contrastive learning and supervised fine-tuning.
  - `model.py`: Code for defining and training the machine learning model.
  - `contrastive_learning.py`: odeC for implementing contrastive learning.
  - `supervised_fine_tune.py`: Code for fine-tuning the model with supervised learning.
  - `evaluation.py`: Code for evaluating the model's performance with various metrics.
  - `utils.py`: Utility functions
- `training_pipeline.py`: Main script for running the entire training pipeline.
- `evaluate.py`: Script for evaluating the model on train/val/test datasets.
- `pipeline.ipynb`: Jupyter notebook version of the pipeline script for visualization.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, providing an overview of the project.
