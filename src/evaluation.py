import torch


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
