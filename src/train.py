import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset, random_split

from data_augmentation import augment_text
from model import ContrastiveModel
from utils import build_vocab, tokenizer_to_ids


# ---------------------------
# Custom Dataset for Supervised Training
# ---------------------------
class BlogDataset(Dataset):
    def __init__(self, data_df, vocab, max_length=256, augmentation=False):
        """
        Args:
            data_df (DataFrame): Preprocessed data with 'text' and 'gender' columns
            vocab (dict): Vocabulary mapping from token to index
            max_length (int): Maximum sequence length
            augmentation (bool): Whether to apply dynamic augmentation
        """
        self.data_df = data_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row["text"]
        label = row["gender"]
        # Optionally apply dynamic augmentation
        if self.augmentation:
            # Generate augmented versions; choose one randomly
            aug_texts = augment_text(text, del_prob=0.1, replace_count=1)
            text = random.choice(aug_texts)
        # Convert tokens to ids using the vocabulary
        token_ids = tokenizer_to_ids(text, self.vocab, self.max_length)
        sample = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "gender": torch.tensor(label, dtype=torch.long),
        }
        return sample


# # --- Custom Dataset ---
# class BlogDataset(Dataset):
#     def __init__(self, data_df, tokenizer, max_length=256):
#         """
#         Args:
#             data_df (DataFrame): Preprocessed data with 'text' and 'label' columns
#             tokenizer (callable): Function to tokenize text
#             max_length (int): Maximum sequence length
#         """
#         self.data_df = data_df
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data_df)

#     def __getitem__(self, idx):
#         row = self.data_df.iloc[idx]
#         text = row["text"]
#         label = row["gender"]
#         # Tokenize the text into a sequence of token IDs
#         token_ids = self.tokenizer(text, self.max_length)
#         sample = {
#             "input_ids": torch.tensor(token_ids, dtype=torch.long),
#             "gender": torch.tensor(label, dtype=torch.long),
#         }
#         return sample


# --- Simple Tokenizer ---
# def simple_tokenizer(text, max_length):
#     """
#     A very simple tokenizer that splits on whitespace and converts tokens to IDs
#     In a real scenario, use a proper tokenizer and build a vocabulary
#     """
#     tokens = text.split()[:max_length]
#     # For demonstration: assign each token an ID using a hash modulo vocab_size
#     vocab_size = 10000
#     token_ids = [hash(token) % vocab_size for token in tokens]
#     # Pad the sequence if it's shorter than max_length
#     if len(token_ids) < max_length:
#         token_ids += [0] * (max_length - len(token_ids))
#     return token_ids


# # --- Training Loop ---
# def train_loop(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for batch in dataloader:
#         inputs = batch["input_ids"].to(device)
#         labels = batch["gender"].to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)  # Classification forward pass
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * inputs.size(0)
#     return total_loss / len(dataloader.dataset)


# ---------------------------
# Training Loop
# ---------------------------
def train(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["gender"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # Forward pass for classification
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc, val_f1, val_cm, val_report = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        print("Confusion Matrix:")
        print(val_cm)
        print("Classification Report:")
        print(val_report)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join("models", "best_supervised_model.pth")
            )
            print("Best model saved.")


# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["gender"].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["M", "F"])
    return acc, f1, cm, report


# # --- Evaluation Loop ---
# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch["input_ids"].to(device)
#             labels = batch["gender"].to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item() * inputs.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#     avg_loss = total_loss / len(dataloader.dataset)
#     accuracy = correct / len(dataloader.dataset)
#     return avg_loss, accuracy


# def main():
#     # --- Load Processed Data ---
#     processed_path = os.path.join("data", "processed", "processed_data.csv")
#     data_df = pd.read_csv(processed_path)

#     # Map labels from characters to numerical values (e.g., 'M': 0, 'F': 1)
#     label_map = {"M": 0, "F": 1}
#     data_df["gender"] = data_df["gender"].map(label_map)

#     # --- Create Dataset ---
#     max_length = 256
#     dataset = BlogDataset(data_df, tokenizer=simple_tokenizer, max_length=max_length)

#     # --- Split Data ---
#     total_len = len(dataset)
#     test_size = int(0.2 * total_len)
#     train_val_size = total_len - test_size
#     val_size = int(0.1 * train_val_size)
#     train_size = train_val_size - val_size

#     train_dataset, val_dataset, test_dataset = random_split(
#         dataset, [train_size, val_size, test_size]
#     )

#     # --- Create DataLoaders ---
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     # --- Model Setup ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     vocab_size = 10000  # Should match the tokenizer's vocabulary size
#     embedding_dim = 128
#     hidden_dim = 256
#     proj_dim = 64
#     num_classes = 2  # For gender: M and F

#     model = ContrastiveModel(
#         vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes
#     )
#     model.to(device)

#     # --- Optimizer and Loss ---
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()

#     # --- Training ---
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         train_loss = train_loop(model, train_loader, optimizer, criterion, device)
#         val_loss, val_acc = evaluate(model, val_loader, criterion, device)
#         print(
#             f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, "
#             f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
#         )

#     # --- Optionally, Save the Model for Later Evaluation ---
#     torch.save(model.state_dict(), os.path.join("models", "contrastive_model.pth"))
#     print("Training complete.")


def main():
    # Load the processed data (assumed to be pre-tokenized and cleaned)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = pd.read_csv(processed_path)

    # Map labels from characters to numerical values: for example, 'M': 0, 'F': 1
    label_map = {"M": 0, "F": 1}
    data_df["gender"] = data_df["gender"].map(label_map)

    # Build vocabulary from the training data
    # Use entire dataset for vocabulary building
    vocab = build_vocab(data_df, text_column="text", min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")

    # Create the dataset. Set augmentation=True if you want dynamic augmentation.
    max_length = 256
    dataset = BlogDataset(data_df, vocab, max_length=max_length, augmentation=False)

    # Split data: 20% test; from remaining, 10% for validation.
    total_len = len(dataset)
    test_size = int(0.2 * total_len)
    train_val_size = total_len - test_size
    val_size = int(0.1 * train_val_size)
    train_size = train_val_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256
    proj_dim = 64
    num_classes = 2  # For gender: M and F

    # Instantiate model with a BiLSTM (bidirectional=True)
    model = ContrastiveModel(
        vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes, bidirectional=True
    )
    model.to(device)

    # Train and evaluate the model
    num_epochs = 10
    train(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4)

    # Evaluate on the test set using the best saved model
    model.load_state_dict(
        torch.load(os.path.join("models", "best_supervised_model.pth"))
    )
    test_acc, test_f1, test_cm, test_report = evaluate(model, test_loader, device)
    print("Test Set Performance:")
    print(f"Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    print("Classification Report:")
    print(test_report)


if __name__ == "__main__":
    main()
