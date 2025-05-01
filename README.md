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
- Blog Authorship Corpus from [Kaggle](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus) is used for supervised contrastive pre-training.

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
1. Download the pre-trained model from our [Hugging Face Model Hub](https://huggingface.co/ndminhvn/BertContrastiveModel/tree/main):
- `bert_supervised_contrastive_pretrained_final_pca.pth` (Pre-trained contrastive model)
- `best_bert_supervised_final_pca.pth` (Fine-tuned supervised model)
- Place the downloaded files in the `models/` directory.

2. Run the main training pipeline script (Optional - if no trained model is available in `models/`, or if you want to retrain the model):

**_NOTE: This step may take a long time to run, depending on the size of the dataset and the hardware used._**
- Run code in `pipeline_final_pca.ipynb` to execute the entire training pipeline.

<!-- ```bash
python pipeline.py
``` -->

This will execute the entire training and evaluation pipeline, including data preprocessing, supervised contrastive learning, supervised fine-tuning, and evaluation.

<!-- - Or you can run each step separately by running the corresponding scripts in the `src/` directory.
- For example, to run the data preprocessing step (optional - as the processed data is already available in `data/processed/`):
  ```bash
  python src/data_preprocessing.py
  ```
- To run the contrastive learning step (run if no contrastive pre-trained model is available in `models/`):

  **_NOTE: Run only if you want to retrain the contrastive model._**
  ```bash
  python src/contrastive_learning.py
  ```

- To run the supervised fine-tuning step (after contrastive learning, or if a contrastive pre-trained model is available):

  **_NOTE: Run only if you want to retrain the supervised model._**
  ```bash
  python src/supervised_fine_tune.py
  ```

3. Perform model evaluation on train/val/test datasets (Assuming a trained model is available in `models/`):
```bash
python evaluate.py
``` -->

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
  - `contrastive_learning.py`: Code for implementing contrastive learning.
  - `supervised_fine_tune.py`: Code for fine-tuning the model with supervised learning.
  - `evaluation.py`: Code for evaluating the model's performance with various metrics.
  - `utils.py`: Utility functions
- `pipeline_final_pca.ipynb`: Jupyter notebook version of the pipeline script for visualization.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, providing an overview of the project.
