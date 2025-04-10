import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel, BertTokenizer

from data_augmentation import random_deletion
from data_preprocessing import load_data

##########################################
# Dataset classes
##########################################


class ContrastiveTextDataset(Dataset):
    """
    Dataset for contrastive pre-training.
    Each sample is a single post (text) - no label is needed.
    Two augmented versions are generated on the fly.
    """

    def __init__(self, texts, tokenizer, max_length=128, augment=True):
        self.texts = texts  # list of posts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # post = clean_text(self.texts[idx])
        # text is already cleaned in the preprocessing step (and saved as processed_data.csv)
        post = self.texts[idx]

        if self.augment:
            view1 = random_deletion(post, p=0.3)
            # view1 = augment_text(view1, del_prob=0.1, replace_count=1)[0]
            view2 = random_deletion(post, p=0.3)
        else:
            view1 = post
            view2 = post

        # Tokenize the two views using BERT tokenizer
        encoded_view1 = self.tokenizer.encode_plus(
            view1,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded_view2 = self.tokenizer.encode_plus(
            view2,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Squeeze the extra dimension (batch dimension from tokenizer)
        for key in encoded_view1:
            encoded_view1[key] = encoded_view1[key].squeeze(0)
            encoded_view2[key] = encoded_view2[key].squeeze(0)
        return (encoded_view1, encoded_view2)


class SupervisedTextDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    Each sample is a (post, label) pair.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128, augment=False):
        self.texts = texts
        self.labels = labels  # list/array of labels (e.g., 0 for M, 1 for F)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # post = clean_text(self.texts[idx])
        # text is already cleaned in the preprocessing step (and saved as processed_data.csv)
        post = self.texts[idx]

        # For supervised fine-tuning, we don't augment the text
        if self.augment:
            post = random_deletion(post, p=0.1)

        encoded = self.tokenizer.encode_plus(
            post,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        for key in encoded:
            encoded[key] = encoded[key].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded, label


##########################################
# Model Definition
##########################################


class BertContrastiveModel(nn.Module):
    def __init__(self, proj_dim=64, num_labels=2, dropout_prob=0.3):
        super(BertContrastiveModel, self).__init__()
        # BERT encoder (base model without classification head)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

        # Classifier head for gender classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_labels),
        )

    def encode(self, input_ids, attention_mask, token_type_ids):
        # Get BERT representations (pooled output typically corresponds to [CLS])
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        return pooled_output

    # Forward pass for contrastive learning (projection head)
    # For classifier, we may not need to use the projection head
    def forward_contrastive(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.encode(input_ids, attention_mask, token_type_ids)
        projection = self.projection_head(pooled_output)
        return projection

    def forward_classifier(self, input_ids, attention_mask, token_type_ids):
        pooled_output = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)
        return logits


##########################################
# Contrastive Loss (InfoNCE)
##########################################


def info_nce_loss(features, temperature=0.5):
    """
    Computes InfoNCE loss.
    Args:
        features: Tensor of shape (2B, proj_dim), where B is batch size.
    """
    batch_size = features.size(0) // 2
    features = nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.t())

    # Mask self-similarities
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    similarity_matrix.masked_fill_(mask, -9e15)

    similarity_matrix /= temperature

    # Create labels such that for the i-th example in the first half,
    # the positive example is at index i + batch_size and vice versa.
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device),
        ]
    )
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
    return loss


##########################################
# Training Pipelines
##########################################


def contrastive_pretrain(
    model, dataloader, optimizer, device, num_epochs=5, temperature=0.5
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            (encoded1, encoded2) = (
                batch  # each is a dict with input_ids, attention_mask, token_type_ids
            )

            # Concatenate the two views along the batch dimension
            input_ids = torch.cat(
                [encoded1["input_ids"], encoded2["input_ids"]], dim=0
            ).to(device)

            attention_mask = torch.cat(
                [encoded1["attention_mask"], encoded2["attention_mask"]], dim=0
            ).to(device)

            token_type_ids = torch.cat(
                [encoded1["token_type_ids"], encoded2["token_type_ids"]], dim=0
            ).to(device)

            optimizer.zero_grad()

            projections = model.forward_contrastive(
                input_ids, attention_mask, token_type_ids
            )

            loss = info_nce_loss(projections, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(
            f"Contrastive Pre-training Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}"
        )
    # Optionally save the pre-trained model
    torch.save(
        model.state_dict(), os.path.join("models", f"bert_contrastive_pretrained.pth")
    )
    print("Contrastive pre-training complete.")


def fine_tune_supervised_with_early_stopping(
    model, train_loader, val_loader, optimizer, device, num_epochs=20, patience=3
):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

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
                best_model_state, os.path.join("models", "best_bert_supervised.pth")
            )
            print("Best supervised model updated.")

        # Early stopping check
        if epoch - best_epoch >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print("Supervised fine-tuning complete.")
    return best_val_acc


def evaluate_classifier(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            encoded, labels = batch
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            token_type_ids = encoded["token_type_ids"].to(device)
            logits = model.forward_classifier(input_ids, attention_mask, token_type_ids)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    from sklearn.metrics import accuracy_score

    return accuracy_score(all_labels, all_preds)


##########################################
# Main Routine
##########################################


def main():
    # Create output folder for models
    os.makedirs("models", exist_ok=True)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the processed dataset (processed CSV should have cleaned text)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = load_data(processed_path)

    texts = data_df["text"].tolist()
    labels = data_df["gender"].tolist()

    print("Data loaded and preprocessed.")
    print(f"Number of samples: {len(texts)}")
    print(f"Number of labels: {len(labels)}")

    # Split the data appropriately for contrastive pre-training and supervised fine-tuning.
    # For contrastive pre-training we only need texts (we generate two augmented views).
    contrastive_dataset = ContrastiveTextDataset(
        texts, tokenizer, max_length=256, augment=True
    )
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=8, shuffle=True)

    # For supervised fine-tuning:
    supervised_dataset = SupervisedTextDataset(
        texts, labels, tokenizer, max_length=256, augment=False
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

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data loaded and split into train, validation, and test sets.")

    # Initialize our model: BERT encoder with projection and classifier heads.
    model = BertContrastiveModel(proj_dim=64, num_labels=2, dropout_prob=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model initialized and moved to device.")

    # CONTRASTIVE PRE-TRAINING PHASE
    optimizer_ctr = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer_ctr = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    num_epochs_ctr = 5  # adjust as needed
    contrastive_pretrain(
        model,
        contrastive_loader,
        optimizer_ctr,
        device,
        num_epochs=num_epochs_ctr,
        temperature=0.5,
    )

    print("Contrastive pre-training complete. Model saved.")

    # Optionally, load pre-trained weights:
    model.load_state_dict(
        torch.load(os.path.join("models", "bert_contrastive_pretrained.pth"))
        # torch.load(os.path.join("models", "bert_contrastive_pretrained_e5_m256.pth"))
    )

    print("Loaded pre-trained contrastive model weights.")

    # SUPERVISED FINE-TUNING PHASE
    # optimizer_ft = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = fine_tune_supervised_with_early_stopping(
        model, train_loader, val_loader, optimizer_ft, device, num_epochs=15, patience=3
    )
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join("models", "best_bert_supervised.pth"))
    )
    print("Loaded best supervised model weights.")
    test_acc = evaluate_classifier(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")

    # val_acc = evaluate_classifier(model, val_loader, device)
    # print(f"Final validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
