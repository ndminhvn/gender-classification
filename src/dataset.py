import torch
from torch.utils.data import Dataset

from src.data_augmentation import random_deletion

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
        # text is already cleaned in the preprocessing step (and saved as processed_data.csv)
        post = self.texts[idx]

        if self.augment:
            # view1 = augment_text(view1, del_prob=0.3, replace_count=2)[0]
            view1 = random_deletion(post, p=0.3)
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
        self.labels = labels  # list of labels (0 for M, 1 for F)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # text is already cleaned in the preprocessing step (and saved as processed_data.csv)
        post = self.texts[idx]

        # For supervised fine-tuning, we don't need to augment the text
        # but we can still apply augmentation if `self.augment` is True
        if self.augment:
            post = random_deletion(post, p=0.3)

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
