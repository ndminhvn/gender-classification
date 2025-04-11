import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_classifier(model, dataloader, device, full_report=True):
    """Evaluates the classifier model on the given dataloader.
    This function computes the average loss, accuracy, F1 score,
    confusion matrix, and classification report for the model's predictions.

    Args:
        model (BertContrastiveModel): The classifier model to evaluate.
        dataloader (DataLoader): The dataloader containing the evaluation data.
        device (torch.device): The device (CPU or GPU).
        full_report (bool): If True, includes the full classification report.
            If False, only includes accuracy and loss.
            Default is True.

    Returns:
        dict: A dictionary containing evaluation metrics including
            loss, accuracy, f1 score, confusion matrix, and classification report.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            encoded, labels = batch
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            token_type_ids = encoded["token_type_ids"].to(device)
            labels = labels.to(device)

            # Get logits from the classifier head
            logits = model.forward_classifier(input_ids, attention_mask, token_type_ids)

            # Calculate batch loss
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            # Compute predictions
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute average loss
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    # If full_report is False, return only loss and accuracy
    # This is useful for quick evaluations in epochs (in fine-tuning)
    if not full_report:
        return {
            "loss": avg_loss,
            "accuracy": acc,
        }

    # If full_report is True, compute additional metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=["M", "F"]  # 0 is M, 1 is F
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
