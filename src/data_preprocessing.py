import html
import os
import re
import string

import nltk
import pandas as pd

from src.config import NLTK_DATA_DIR

# Download the stopwords and punkt tokenizer from nltk
# and save them in the venv/nltk_data directory
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("stopwords", download_dir=NLTK_DATA_DIR)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a csv file.
    Expected data format:
    - Column 1: text
    - Column 2: label (gender)

    Args:
        data_path (str): Path to the csv file

    Returns:
        pd.DataFrame: Dataframe containing the data
    """
    try:
        # data = pd.read_csv(data_path, encoding="ISO-8859-1")
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully from: {data_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at {data_path}")
    except pd.errors.ParserError:
        raise pd.errors.ParserError("Failed to parse the csv file")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def clean_text(text: str) -> str:
    """Clean the text by
     - Converting to lowercase
     - Removing special characters (including HTML characters)
     - Remove URLs
     - Removing extra whitespace
     - Removing stopwords (english)
     - Removing punctuation marks
     - Tokenizing the text

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()

    # Convert HTML characters to corresponding unicode characters
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stopwords])

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Disable the tokenizer for now (we use the tokenizer in the model)
    # Tokenize the text
    # text = " ".join(nltk.word_tokenize(text))

    return text


def normalize_labels(
    data: pd.DataFrame, label_column: str, convert_to_int: bool = True
) -> pd.DataFrame:
    """Normalize the labels (genders) in the dataframe
    - Remove spaces
    - Convert labels to uppercase  (if not already)
    - Convert labels to binary (0, 1) if they are not already


    Args:
        data (pd.DataFrame): Dataframe containing the data
        label_column (str): Column name containing the labels
        convert_to_int (bool): Whether to convert labels to int (0, 1) - default is True

    Returns:
        pd.DataFrame: Dataframe with normalized labels
    """
    # Ensure label column is string type
    data[label_column] = data[label_column].astype(str).fillna("")

    # Remove spaces from labels
    data[label_column] = data[label_column].str.strip()

    # Convert labels to uppercase
    data[label_column] = data[label_column].str.upper()

    # Convert labels to binary (0, 1) if they are not already
    if convert_to_int:
        data[label_column] = data[label_column].map({"M": 0, "F": 1})
        data[label_column] = data[label_column].astype(int)

    return data


def preprocess_data(
    data: pd.DataFrame, text_column: str, label_column: str
) -> pd.DataFrame:
    """Preprocess the data and return a new dataframe (without modifying the original dataframe)
    - Clean the text
    - Normalize the labels
    - Drop rows with missing values

    Args:
        data (pd.DataFrame): Dataframe containing the data
        text_column (str): Column name containing the text
        label_column (str): Column name containing the labels

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Create a copy of the data
    new_data = data.copy()

    # Drop rows with NaN values
    new_data = new_data.dropna()

    # Ensure text column is string type and fill missing values with empty string
    new_data[text_column] = new_data[text_column].astype(str).fillna("")

    # Clean the text and normalize labels
    new_data[text_column] = new_data[text_column].apply(clean_text)
    new_data = normalize_labels(new_data, label_column)

    print("Data preprocessing completed successfully!")
    return new_data


def save_processed_data(data: pd.DataFrame, output_dir: str, filename: str):
    """Save the processed data to a csv file

    Args:
        data (pd.DataFrame): Dataframe containing the data
        output_dir (str): Output directory to save the file
        filename (str): Name of the file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        data.to_csv(output_path, index=False)
        print(f"Processed data saved at: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving processed data: {str(e)}")


# Example usage, otherwise used in the pipeline
if __name__ == "__main__":
    raw_data_path = "data/raw/gender-classification.csv"
    processed_data_path = "data/processed"
    processed_data_filename = "processed_data.csv"
    text_column = "text"
    label_column = "gender"
    try:
        raw_data = load_data(raw_data_path)
        preprocessed_data = preprocess_data(raw_data, text_column, label_column)
        save_processed_data(
            preprocessed_data, processed_data_path, processed_data_filename
        )
    except Exception as e:
        print(f"Error: {str(e)}")
