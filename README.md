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

## Project Structure

- `data/`: Contains the dataset and any processed data.
  - `raw/`: Original dataset files.
  - `processed/`: Processed dataset files.
- `models/`: Contains the trained models.
- `src/`: Source code for data processing, model training, and evaluation.
  - `config.py`: Configuration file for setting parameters and paths.
  - `data_preprocessing.py`: Code for loading and processing the dataset.
  - `data_augmentation.py`: Code for augmenting the dataset.
  - `model.py`: Code for defining and training the machine learning model.
  - `contrastive_learning.py`: Code for implementing contrastive learning.
  - `train.py`: Code for training the model.
  - `evaluate.py`: Code for evaluating the model.
  - `utils.py`: Utility functions for data processing and model evaluation.
- `pipelines.ipynb`: Jupyter notebook for running the entire pipeline from data loading to model evaluation.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, providing an overview of the project.
