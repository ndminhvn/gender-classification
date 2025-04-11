import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from src.data_preprocessing import load_data
from src.dataset import SupervisedTextDataset
from src.evaluation import evaluate_classifier
from src.model import BertContrastiveModel


def fine_tune_supervised(
    model, train_loader, val_loader, optimizer, device, num_epochs=20, patience=3
):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    # Track training/validation loss and accuracy
    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            encoded, labels = batch
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            token_type_ids = encoded["token_type_ids"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model.forward_classifier(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader.dataset)

        # Evaluate on validation set
        val_eval = evaluate_classifier(model, val_loader, device, full_report=False)
        val_acc = val_eval["accuracy"]
        val_loss = val_eval["loss"]

        # Store training and validation metrics
        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()

            file_name = "best_bert_supervised.pth"
            os.makedirs("models", exist_ok=True)
            file_path = os.path.join("models", file_name)

            torch.save(best_model_state, file_path)

            print(f"Best validation accuracy updated: {best_val_acc:.4f}")
            print(f"Best supervised model updated and saved at {file_path}")

        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print("Supervised fine-tuning complete.")
    return history, best_val_acc


def supervised_fine_tune_pipeline():
    # Load the processed dataset (processed CSV should have cleaned text)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = load_data(processed_path)

    texts = data_df["text"].tolist()
    labels = data_df["gender"].tolist()

    print("Data loaded and preprocessed.")
    print(f"Number of samples: {len(texts)}")
    print(f"Number of labels: {len(labels)}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets and dataloaders for supervised learning
    supervised_dataset = SupervisedTextDataset(
        texts, labels, tokenizer, max_length=128, augment=False
    )

    # Split data: 20% as test; from remaining, 20% as validation
    total_len = len(supervised_dataset)
    test_len = int(0.2 * total_len)
    remaining = total_len - test_len
    val_len = int(0.2 * remaining)
    train_len = remaining - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        supervised_dataset, [train_len, val_len, test_len]
    )

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data loaded and split into train, validation, and test sets.")

    # Initialize our model: BERT encoder with projection and classifier heads.
    model = BertContrastiveModel(proj_dim=64, num_labels=2, dropout_prob=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model initialized and moved to device.")

    # Load pre-trained contrastive model weights
    pre_trained_path = os.path.join("models", "bert_contrastive_pretrained.pth")

    # Check if the pre-trained model weights exist
    if not os.path.exists(pre_trained_path):
        raise FileNotFoundError(
            f"Pre-trained model weights not found at {pre_trained_path}. Please run contrastive pre-training first."
        )

    model.load_state_dict(torch.load(pre_trained_path))

    print("Loaded pre-trained contrastive model weights.")

    # SUPERVISED FINE-TUNING PHASE
    optimizer_ft = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)

    _, best_val_acc = fine_tune_supervised(
        model, train_loader, val_loader, optimizer_ft, device, num_epochs=20, patience=3
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Supervised fine-tuning complete. Model saved.")


if __name__ == "__main__":
    supervised_fine_tune_pipeline()
