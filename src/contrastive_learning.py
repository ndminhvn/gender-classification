import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data_preprocessing import load_data
from src.dataset import ContrastiveTextDataset
from src.model import BertContrastiveModel

# from sklearn.decomposition import PCA
import joblib


def supervised_contrastive_loss(features, labels, temperature=0.2):
    """Computes the supervised contrastive loss.
    This loss encourages the model to learn representations that are
    invariant to different augmentations of the same input.

    It does so by maximizing the cosine similarity between augmented
    representations of the same input while minimizing the similarity
    between different inputs.

    For each sample, the positive is the other samples with the same label.

    The loss is computed as the negative log probability of the positive
    pairs divided by the sum of all pairs.

    The loss is averaged over the batch size.

    The temperature parameter controls the scale of the similarity
    scores. A smaller temperature leads to sharper distributions,
    while a larger temperature leads to smoother distributions.

    Args:
        features (Tensor): Shape (batch_size, proj_dim), projected representations.
        labels (Tensor): Shape (batch_size,), class labels for each sample.
        temperature (float, optional): Temperature scaling factor. Default is 0.2.

    Returns:
        _type_: _description_
    """
    device = features.device
    batch_size = features.size(0)

    # Normalize features
    features = nn.functional.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.t())  # (batch_size, batch_size)

    # Expand labels to create a mask
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(device)  # (batch_size, batch_size)

    # Mask out self-similarity
    mask_self = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    similarity_matrix = similarity_matrix.masked_fill(mask_self, -1e9)

    # Create labels matrix for matching samples
    # labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Compute positive similarity scores
    positive_similarity = similarity_matrix[mask.bool()]

    # Compute negative similarity scores
    negative_similarity = similarity_matrix[~mask.bool()]
    
    # Compute loss
    numerator = torch.exp(positive_similarity / temperature)
    denominator = numerator.sum() + torch.exp(negative_similarity / temperature).sum()
    loss = -torch.log(numerator.sum() / denominator).mean()
    
    # Compute positive and negative similarity scores
    # positive_similarity = similarity_matrix[mask.bool()].reshape(batch_size, -1)
    # negative_similarity = similarity_matrix[~mask.bool()].reshape(batch_size, -1)

    # # Compute loss
    # numerator = torch.exp(positive_similarity / temperature).sum(dim=1)
    # denominator = numerator + torch.exp(negative_similarity / temperature).sum(dim=1)
    # loss = -torch.log(numerator / denominator).mean()

    return loss


# Supervised Contrastive Loss
# def supervised_contrastive_loss(features, labels, temperature=0.5):
#     """
#     Computes the supervised contrastive loss.

#     Args:
#         features (Tensor): Shape (batch_size, proj_dim), projected representations.
#         labels (Tensor): Shape (batch_size,), class labels for each sample.
#         temperature (float): Temperature scaling factor.

#     Returns:
#         loss (Tensor): The scalar supervised contrastive loss.
#     """
#     device = features.device
#     batch_size = features.size(0)

#     # Normalize features
#     features = nn.functional.normalize(features, dim=1)

#     # Compute similarity matrix
#     similarity_matrix = torch.matmul(features, features.t())  # (batch_size, batch_size)

#     # Expand labels to create a mask
#     labels = labels.view(-1, 1)
#     mask = torch.eq(labels, labels.t()).float().to(device)  # (batch_size, batch_size)

#     # Remove self-similarity: set diagonal to 0 - subtracting an identity matrix
#     self_mask = torch.eye(batch_size, device=device)
#     mask = mask - self_mask

#     # Compute logits
#     logits = similarity_matrix / temperature

#     # Subtract the maximum per row for numerical stability
#     logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#     logits = logits - logits_max.detach()

#     # Compute log_prob
#     exp_logits = torch.exp(logits) * (1 - self_mask)
#     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

#     # Sum log-probabilities of positive pairs and normalize by number of positives
#     mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

#     # Loss is the negative mean of these
#     loss = -mean_log_prob_pos.mean()
#     return loss


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


def supervised_contrastive_pretrain(
    model,
    dataloader,
    optimizer,
    device,
    temperature=0.2,
    num_epochs=5,
    file_name="bert_supervised_contrastive_pretrained.pth",
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            encoded1, encoded2, labels = (
                batch  # each is a dict with input_ids, attention_mask, token_type_ids
            )

            # Concatenate the two views (batch becomes 2N, labels are duplicated)
            input_ids = torch.cat(
                [encoded1["input_ids"], encoded2["input_ids"]], dim=0
            ).to(device)

            attention_mask = torch.cat(
                [encoded1["attention_mask"], encoded2["attention_mask"]], dim=0
            ).to(device)

            token_type_ids = torch.cat(
                [encoded1["token_type_ids"], encoded2["token_type_ids"]], dim=0
            ).to(device)

            labels = labels.to(device)

            # Duplicate the labels for both views
            labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            
            # # 1) get BERT pooled outputs
            # with torch.no_grad():  # no grad for BERT during PCA transform
            #     pooled = model.encode(input_ids, attention_mask, token_type_ids)  # (2N, 768)

            # # 2) apply PCA (on CPU)
            # # pca = joblib.load("models/bert_pca_256.joblib")
            # pca = joblib.load("models/bert_pca_64.joblib")
            # pooled_np = pooled.cpu().numpy()
            # reduced = pca.transform(pooled_np)                     # (2N, K)
            # reduced_t = torch.from_numpy(reduced).to(device)       # back to GPU

            # # 3) run through projection head only (bypass forward_contrastive)
            # features = model.projection_head(reduced_t)            # (2N, proj_dim)
            
            
            features = model.forward_contrastive(
                input_ids, attention_mask, token_type_ids
            )

            # Compute supervised contrastive loss
            loss = supervised_contrastive_loss(features, labels, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(
            f"Supervised Contrastive Pre-training Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}"
        )

    # Save the pre-trained model after training
    os.makedirs("models", exist_ok=True)
    file_path = os.path.join("models", file_name)

    torch.save(model.state_dict(), file_path)

    print(f"Model saved at {file_path}")
    print("Supervised Contrastive pre-training complete.")


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
