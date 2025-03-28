import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from data_augmentation import augment_text
from data_preprocessing import clean_text  # reuse cleaning if needed
from model import ContrastiveModel
from utils import build_vocab, tokenizer_to_ids


# --- Custom Dataset for Contrastive Learning ---
class ContrastiveBlogDataset(Dataset):
    """
    Contrastive Dataset for generating augmented pairs from the same blog post
    Each sample returns two augmented views
    """

    def __init__(self, data_df, vocab, max_length=256):
        self.data_df = data_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row["text"]
        # Generate two augmented views from the same text
        augmented_views = augment_text(text, del_prob=0.1, replace_count=1)
        # Ensure we have at least two views; if not, duplicate the original.
        if len(augmented_views) < 2:
            view1, view2 = text, text
        else:
            view1, view2 = augmented_views[0], augmented_views[1]

        # Convert the augmented texts into token IDs using your vocabulary mapping.
        view1_ids = tokenizer_to_ids(view1, self.vocab, self.max_length)
        view2_ids = tokenizer_to_ids(view2, self.vocab, self.max_length)

        return {
            "view1": torch.tensor(view1_ids, dtype=torch.long),
            "view2": torch.tensor(view2_ids, dtype=torch.long),
        }


# --- Contrastive Loss: InfoNCE ---
def info_nce_loss(features, temperature=0.5):
    """
    Computes the InfoNCE loss.

    Args:
        features (Tensor): Shape (2B, proj_dim), concatenated representations for two augmented views.
        temperature (float): Temperature scaling factor.

    Returns:
        loss (Tensor): Scalar loss.
    """
    batch_size = features.shape[0] // 2

    # Normalize features
    features = nn.functional.normalize(features, dim=1)

    # Compute similarity matrix (2B x 2B)
    similarity_matrix = torch.matmul(features, features.T)

    # Mask self-similarities
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    similarity_matrix.masked_fill_(mask, -9e15)

    # Scale by temperature
    similarity_matrix /= temperature

    # For each example, the positive is the other augmented view.
    # Correct labels: for first half, positive index is i + batch_size; for second half, i - batch_size
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device),
        ],
        dim=0,
    )
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

    return loss


# --- Training Loop for Contrastive Pre-training ---
def contrastive_train_loop(model, dataloader, optimizer, device, temperature=0.5):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)

        # Concatenate along batch dimension: shape becomes (2B, sequence_length)
        inputs = torch.cat([view1, view2], dim=0)

        optimizer.zero_grad()
        # Obtain projections using the contrastive branch
        proj_features = model.forward_contrastive(inputs)  # shape: (2B, proj_dim)
        loss = info_nce_loss(proj_features, temperature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


# def main():
#     # --- Load Processed Data ---
#     processed_path = os.path.join("data", "processed", "processed_data.csv")
#     data_df = pd.read_csv(processed_path)

#     # For contrastive pre-training we only need text, labels are not used.
#     # Optionally, drop rows with missing text.
#     data_df = data_df.dropna(subset=["text"])

#     # --- Create Contrastive Dataset ---
#     max_length = 256
#     dataset = ContrastiveBlogDataset(
#         data_df, tokenizer=simple_tokenizer, max_length=max_length, augmentation=True
#     )

#     # Split into training and test sets (for contrastive pre-training, test split may be used later for fine-tuning)
#     total_len = len(dataset)
#     test_size = int(0.2 * total_len)
#     train_size = total_len - test_size
#     train_dataset, _ = random_split(dataset, [train_size, test_size])

#     # --- DataLoader ---
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # --- Model Setup ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vocab_size = 10000
#     embedding_dim = 128
#     hidden_dim = 256
#     proj_dim = 64
#     num_classes = 2  # Not used during contrastive pre-training.

#     model = ContrastiveModel(
#         vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes
#     )
#     model.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     # --- Contrastive Pre-training ---
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         loss = contrastive_train_loop(
#             model, train_loader, optimizer, device, temperature=0.5
#         )
#         print(f"Epoch {epoch+1}/{num_epochs}: Contrastive Loss = {loss:.4f}")

#     # --- Save the Pre-trained Model ---
#     os.makedirs("models", exist_ok=True)
#     torch.save(
#         model.state_dict(), os.path.join("models", "contrastive_pretrained_model.pth")
#     )
#     print("Contrastive pre-training complete and model saved.")


def main():
    # Load the processed data (assumed to be pre-cleaned & tokenized text)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = pd.read_csv(processed_path)

    # Build vocabulary from the processed text column
    vocab = build_vocab(data_df, text_column="text", min_freq=1)
    print("Vocabulary size:", len(vocab))

    # Create the contrastive dataset using augmented pairs
    max_length = 256
    dataset = ContrastiveBlogDataset(data_df, vocab, max_length=max_length)

    # Optionally, split dataset (here we use most of it for training)
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model Setup: Initialize ContrastiveModel with a BiLSTM encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    proj_dim = 64
    num_classes = 2  # Although classification head is not used in pre-training
    model = ContrastiveModel(
        vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes, bidirectional=True
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = contrastive_train_loop(
            model, train_loader, optimizer, device, temperature=0.5
        )
        print(f"Epoch {epoch+1}/{num_epochs} - Contrastive Loss: {loss:.4f}")

    # Save the pre-trained model for subsequent supervised fine-tuning
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "contrastive_pretrained.pth"))
    print("Contrastive pre-training complete and model saved.")


if __name__ == "__main__":
    main()
