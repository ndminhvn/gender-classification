import os
import random
import re

import nltk
import pandas as pd
from nltk.corpus import wordnet

from config import NLTK_DATA_DIR

# Download the WordNet and Open Multilingual WordNet (OMW) corpora from nltk
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)
nltk.download("omw-1.4", download_dir=NLTK_DATA_DIR)


def get_synonyms(word: str) -> set:
    """Get synonyms of a word using WordNet.

    Args:
        word (str): The word to get synonyms for.

    Returns:
        set: A set of synonyms for the word.
    """
    synonyms = set()
    # Get the synsets for the word
    for syn in wordnet.synsets(word):
        # Get the lemmas of the synset (n, v, adj, adv)
        for lemma in syn.lemmas():
            # Remove the underscore and convert to lowercase to match the input word
            synonym = lemma.name().replace("_", " ").lower()

            # Add the synonym to the set
            if synonym != word:
                synonyms.add(synonym)
    return synonyms


def synonym_replacement(text: str, n: int = 1) -> str:
    """
    Replace randomly n words in the sentence with one of their synonyms.

    Args:
        text (str): The input text.
        n (int): Number of words to replace with synonyms.

    Returns:
        str: The text after synonym replacement.
    """
    words = text.split()
    new_words = words.copy()

    # Filter words that have synonyms available
    candidate_words = [word for word in words if get_synonyms(word)]
    if not candidate_words:
        return text

    # Shuffle the candidate words
    random.shuffle(candidate_words)

    num_replaced = 1
    for word in candidate_words:
        # Get synonyms for the current word
        synonyms = get_synonyms(word)

        # If synonyms are available, replace the word with a random synonym
        if synonyms:
            synonym = random.choice(list(synonyms))
            # Replace the word in the text with the synonym (in the new_words list)
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1

        # Stop if n words have been replaced
        if num_replaced >= n:
            break

    return " ".join(new_words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete words from the sentence (text) with a probability p.

    Args:
        text (str): The input text.
        p (float): Probability of deleting each word.

    Returns:
        str: The text after random deletion.
    """
    words = text.split()

    # If there's less than 5 words, keep it unchanged
    if len(words) <= 5:
        return text

    # Delete words with probability p while ensuring at least 5 words remain
    new_words = [word for word in words if random.random() > p]

    if len(new_words) < 5:
        # Add remaining words from the original text
        new_words += random.sample(words, 5 - len(new_words))

    return " ".join(new_words)


def augment_text(text: str, del_prob: float = 0.1, replace_count: int = 1) -> list:
    """
    Generate augmented versions of the input text using multiple augmentation strategies.

    Args:
        text (str): The original text.
        del_prob (float): Probability used for random deletion.
        replace_count (int): Number of words to replace in synonym replacement.

    Returns:
        list: A list of augmented texts including the original.
    """
    # Maintain the original text
    augmented_texts = [text]

    # Augment the text using random deletion
    augmented_texts.append(random_deletion(text, p=del_prob))

    # Augment the text using synonym replacement
    augmented_texts.append(synonym_replacement(text, n=replace_count))

    return augmented_texts


def add_augmented_data_to_dataframe(
    data: pd.DataFrame, text_column: str, label_column: str
) -> pd.DataFrame:
    """Add augmented data to the dataframe by applying data augmentation techniques.

    Args:
        data (pd.DataFrame): Dataframe containing the data.
        text_column (str): Column name containing the text.
        label_column (str): Column name containing the labels.

    Returns:
        pd.DataFrame: Dataframe with augmented data.
    """
    # Create a copy of the data
    augmented_data = data.copy()

    # Apply data augmentation to each row in the dataframe
    for index, row in data.iterrows():
        # Augment the text in the current row
        augmented_texts = augment_text(row[text_column], del_prob=0.3, replace_count=2)

        # Add the augmented texts to the augmented_data dataframe
        # Each augmented text will have the same label as the original text
        for text in augmented_texts:
            augmented_data.loc[len(augmented_data)] = [text, row[label_column]]

    return augmented_data


def save_augmented_data(data: pd.DataFrame, output_dir: str, filename: str):
    """Save the augmented data to a csv file

    Args:
        data (pd.DataFrame): Dataframe containing the augmented data.
        output_dir (str): Directory where the csv file will be saved.
        filename (str): Name of the file to save the augmented data.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        data.to_csv(output_path, index=False)
        print(f"Augmented data saved at: {output_path}")
    except Exception as e:
        print(f"Error saving augmented data: {e}")


# Example usage, otherwise used in the pipeline
if __name__ == "__main__":
    sample_text = "This is an example blog post written by a user."
    print("Original:", sample_text)
    print("After Random Deletion:", random_deletion(sample_text, p=0.3))
    print("After Synonym Replacement:", synonym_replacement(sample_text, n=2))
    print(
        "All Augmentations:",
        augment_text(sample_text, del_prob=0.3, replace_count=2),
    )
