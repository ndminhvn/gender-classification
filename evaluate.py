import os

import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from src.data_preprocessing import load_data
from src.dataset import SupervisedTextDataset
from src.evaluation import evaluate_classifier
from src.model import BertContrastiveModel


def print_eval_results(results):
    """Prints the evaluation results.

    Args:
        results (dict): A dictionary containing evaluation metrics including
            loss, accuracy, f1 score, confusion matrix, and classification report.
    """
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    print("Classification Report:")
    print(results["report"])
    print("=" * 30)


def run_evaluation():
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

    # Initialize our model: BERT encoder with projection and classifier heads
    model = BertContrastiveModel(proj_dim=64, num_labels=2, dropout_prob=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model initialized and moved to device.")

    # Load pre-trained contrastive model weights - optional
    # pre_trained_path = os.path.join("models", "bert_contrastive_pretrained.pth")

    # # Check if the pre-trained model weights exist
    # if not os.path.exists(pre_trained_path):
    #     raise FileNotFoundError(
    #         f"Pre-trained model weights not found at {pre_trained_path}. Please run contrastive pre-training first."
    #     )

    # model.load_state_dict(torch.load(pre_trained_path))

    # print("Loaded pre-trained contrastive model weights.")

    # Load the supervised model weights
    supervised_model_path = os.path.join("models", "best_bert_supervised.pth")
    # Check if the supervised model weights exist
    if not os.path.exists(supervised_model_path):
        raise FileNotFoundError(
            f"Supervised model weights not found at {supervised_model_path}. Please run supervised fine-tuning first."
        )

    model.load_state_dict(torch.load(supervised_model_path))

    print("Loaded best supervised model weights.")

    # Evaluate the model on the all 3 sets
    print("Evaluating on train set:")
    train_eval = evaluate_classifier(model, train_loader, device)
    print_eval_results(train_eval)

    print("Evaluating on validation set:")
    val_eval = evaluate_classifier(model, val_loader, device)
    print_eval_results(val_eval)

    print("Evaluating on test set:")
    test_eval = evaluate_classifier(model, test_loader, device)
    print_eval_results(test_eval)

    print("Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
