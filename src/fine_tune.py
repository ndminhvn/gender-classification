import os
import random
from collections import Counter

import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset, random_split

from model import ContrastiveModel
from utils import build_vocab, tokenizer_to_ids


# ---------------------------
# Custom Dataset for Supervised Fine-Tuning
# ---------------------------
class BlogDataset(Dataset):
    def __init__(self, data_df, vocab, max_length=256, augmentation=False):
        """
        Args:
            data_df (DataFrame): Processed data with columns "text" and "label".
            vocab (dict): Mapping of token to index.
            max_length (int): Maximum token sequence length.
            augmentation (bool): If True, you may apply dynamic augmentation.
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
        # Optionally, apply augmentation here if needed (for fine-tuning, we keep it off)
        # For example:
        # if self.augmentation:
        #     text = random.choice(augment_text(text, deletion_prob=0.1, replace_count=1))
        token_ids = tokenizer_to_ids(text, self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "gender": torch.tensor(label, dtype=torch.long),
        }


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
            outputs = model(inputs)  # using the classification forward pass
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["M", "F"])
    return acc, f1, cm, report


# ---------------------------
# Training Loop for Supervised Fine-Tuning
# ---------------------------
def train_supervised(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
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
            outputs = model(inputs)  # classification forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc, val_f1, val_cm, val_report = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        print("Confusion Matrix:\n", val_cm)
        print("Classification Report:\n", val_report)

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(
                model.state_dict(), os.path.join("models", "best_supervised_model.pth")
            )
            print("Best supervised model saved.")
    return best_val_acc


# ---------------------------
# Main Function: Supervised Fine-Tuning Pipeline
# ---------------------------
def main():
    # Load the processed dataset (processed CSV should have cleaned, tokenized text)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = pd.read_csv(processed_path)

    # Map labels from characters to numerical values (e.g., 'M': 0, 'F': 1)
    label_map = {"M": 0, "F": 1}
    data_df["gender"] = data_df["gender"].map(label_map)

    # Build vocabulary from the processed text
    vocab = build_vocab(data_df, text_column="text", min_freq=1)
    print("Vocabulary size:", len(vocab))

    # Create the supervised dataset
    max_length = 256
    dataset = BlogDataset(data_df, vocab, max_length=max_length, augmentation=False)

    # Split data: 20% as test; from remaining, 10% as validation
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
    num_classes = 2  # for gender: M and F

    # Instantiate the model
    # Loading pre-trained weights from contrastive pre-training
    model = ContrastiveModel(
        vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes, bidirectional=True
    )
    model.to(device)

    # Load pre-trained weights from contrastive learning
    pretrained_path = os.path.join("models", "contrastive_pretrained.pth")
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        print("Loaded pre-trained contrastive model weights.")
    else:
        print("Pre-trained model not found. Training from scratch.")

    # Fine-tune the model using labeled data
    print("Starting supervised fine-tuning...")
    train_supervised(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3)

    # Evaluate on the test set using the best saved model
    model.load_state_dict(
        torch.load(os.path.join("models", "best_supervised_model.pth"))
    )
    test_acc, test_f1, test_cm, test_report = evaluate(model, test_loader, device)
    print("Test Set Performance:")
    print(f"Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:\n", test_cm)
    print("Classification Report:\n", test_report)


if __name__ == "__main__":
    main()
