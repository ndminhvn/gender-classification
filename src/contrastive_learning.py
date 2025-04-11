import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data_preprocessing import load_data
from src.dataset import ContrastiveTextDataset
from src.model import BertContrastiveModel


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
    batch_size = features.size(0) // 2
    features = nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.t())

    # Mask self-similarities
    mask = torch.eye(2 * batch_size, device=features.device).bool()
    similarity_matrix.masked_fill_(mask, -9e15)

    # Scale by temperature
    similarity_matrix /= temperature

    # For each sample, the positive is the other augmented view
    # Correct labels: for first half, positive index is i + batch_size; for second half, i - batch_size
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device),
        ],
    )
    loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

    return loss


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

    # Save the pre-trained model after training
    # The weights can be loaded into the model for fine-tuning later
    file_name = "bert_contrastive_pretrained.pth"
    os.makedirs("models", exist_ok=True)
    file_path = os.path.join("models", file_name)

    torch.save(model.state_dict(), file_path)

    print(f"Model saved at {file_path}")
    print("Contrastive pre-training complete.")


def contrastive_pretrain_pipeline():
    # Load the processed dataset (processed CSV should have cleaned text)
    processed_path = os.path.join("data", "processed", "processed_data.csv")
    data_df = load_data(processed_path)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Texts will be used for contrastive learning
    texts = data_df["text"].tolist()

    print("Data loaded and preprocessed.")
    print(f"Number of samples: {len(texts)}")

    contrastive_dataset = ContrastiveTextDataset(
        texts, tokenizer, max_length=128, augment=True
    )
    contrastive_loader = DataLoader(contrastive_dataset, batch_size=8, shuffle=True)

    print("Contrastive dataset created (with data augmentation).")

    # Initialize our model: BERT encoder with projection and classifier heads.
    model = BertContrastiveModel(proj_dim=64, num_labels=2, dropout_prob=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model initialized and moved to device.")

    # CONTRASTIVE PRE-TRAINING PHASE
    optimizer_ctr = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    contrastive_pretrain(
        model,
        contrastive_loader,
        optimizer_ctr,
        device,
        num_epochs=5,
        temperature=0.5,
    )

    print("Contrastive pre-training complete. Model saved.")


if __name__ == "__main__":
    contrastive_pretrain_pipeline()
