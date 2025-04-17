import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def extract_embeddings(model, dataset, device, batch_size=32):
    model.eval()
    embeddings = []
    labels_list = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            # Assuming for the supervised dataset, batch is: (encoded, labels)
            encoded, labels = batch
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            token_type_ids = None
            if "token_type_ids" in encoded:
                token_type_ids = encoded["token_type_ids"].to(device)

            # Extract features using the encoder (or using forward_projection)
            # If using forward_classifier, you may lose the fine-grained features.
            feats = model.encode(input_ids, attention_mask, token_type_ids)

            embeddings.append(feats.cpu())
            labels_list.append(labels.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels


def plot_embeddings_tsne(embeddings, labels, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels.numpy(),
        cmap="viridis",
        alpha=0.7,
    )
    plt.title("t-SNE Visualization of Learned Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label="Label")
    plt.show()
