from collections import Counter


def build_vocab(data_df, text_column="text", min_freq=1):
    """Build a vocabulary dictionary from the tokenized text in the DataFrame

    Args:
        data_df (pd.DataFrame): DataFrame containing a text column
        text_column (str): Name of the column with tokenized text
        min_freq (int): Minimum frequency for a token to be included

    Returns:
        dict: Mapping from token to unique index
    """
    tokens = []
    for text in data_df[text_column]:
        # Assuming text is already tokenized (space-separated tokens)
        tokens.extend(text.split())
    token_counts = Counter(tokens)

    # Reserve index 0 for padding and 1 for unknown tokens.
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in token_counts.items():
        if count >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def tokenizer_to_ids(text, vocab, max_length):
    """Converts tokenized text (space separated) to a list of token IDs using a given vocabulary

    Args:
        text (str): Tokenized text
        vocab (dict): A mapping from token to index
        max_length (int): Maximum sequence length

    Returns:
        List[int]: List of token IDs (padded to max_length).
    """
    tokens = text.split()[:max_length]
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(token_ids) < max_length:
        token_ids += [vocab["<PAD>"]] * (max_length - len(token_ids))

    return token_ids
