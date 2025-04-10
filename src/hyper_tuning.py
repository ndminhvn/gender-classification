import os
import random

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from bert import (BertContrastiveModel, SupervisedTextDataset,
                  evaluate_classifier)
from data_preprocessing import load_data


##########################################
# Training Pipelines
##########################################
def fine_tune_supervised_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    lr,
    weight_decay,
    num_epochs=20,
    patience=3,
):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate_classifier(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()
            torch.save(
                best_model_state,
                os.path.join("models", "best_bert_supervised_hyper.pth"),
            )
            print("Best supervised model updated.")

        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print("Supervised fine-tuning complete.")
    return best_val_acc


######################################
# Hyperparameter Optimization Objective Function
######################################
def objective(trial):
    # Hyperparameters to tune:
    dropout_prob = trial.suggest_categorical("dropout_prob", [0.1, 0.3, 0.5])
    # lr = trial.suggest_categorical("lr", [2e-5, 1e-6, 1e-4])
    # weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-3])
    lr = 1e-3
    weight_decay = 1e-4
    num_epochs = trial.suggest_categorical("num_epochs", [10, 15, 20])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Load dataset
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = load_data(processed_path)

    texts = data_df["text"].tolist()
    labels = data_df["gender"].tolist()

    print("Data loaded and preprocessed.")
    print(f"Number of samples: {len(texts)}")
    print(f"Number of labels: {len(labels)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create supervised dataset
    dataset = SupervisedTextDataset(
        texts, labels, tokenizer, max_length=128, augment=False
    )

    # Split into train (64%), validation (16%), test (20%)
    total_len = len(dataset)
    test_len = int(0.2 * total_len)
    remaining = total_len - test_len
    val_len = int(0.2 * remaining)
    train_len = remaining - val_len
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with hyperparameters from trial.
    model = BertContrastiveModel(proj_dim=64, num_labels=2, dropout_prob=dropout_prob)
    model.to(device)

    # Optionally, load pre-trained weights:
    model.load_state_dict(
        torch.load(os.path.join("models", "bert_contrastive_pretrained_e5.pth"))
    )

    print("Loaded pre-trained contrastive model weights.")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fine-tune
    print("=== Starting supervised fine-tuning... ===")
    print(f"Dropout: {dropout_prob}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"Num Epochs: {num_epochs}, Batch Size: {batch_size}")

    patience = 3  # number of epochs to wait before early stopping.
    val_acc = fine_tune_supervised_with_early_stopping(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
    )

    # Return the negative of val_acc to minimize the objective
    # or return 1 - val_acc
    # return 1.0 - val_acc
    return -1.0 * val_acc


##########################################
# Main Routine
##########################################


def main():
    # Create output folder for models
    os.makedirs("models", exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
