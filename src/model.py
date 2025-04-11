import torch.nn as nn
from transformers import BertModel

##########################################
# Model Definition
##########################################


class BertContrastiveModel(nn.Module):
    """
    BERT-based model for contrastive learning and classification.

    This model uses a BERT encoder for feature extraction and includes
    a projection head for contrastive learning. Additionally, it has a
    classifier head for down-stream classification tasks.

    The model is initialized with a projection dimension, number of labels,
    and dropout probability for the classifier head.

    The `encode` method is used to obtain the BERT representations
    (pooled output - corresponding to the [CLS] token) for the input
    sequences. The input sequences are passed through the BERT model
    to obtain the pooled output, which is then passed through the
    projection head for contrastive learning.

    The contrastive learning part of the model is implemented in the
    `forward_contrastive` method, while the classification part is
    implemented in the `forward_classifier` method.
    """

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
        """Encode input sequences using BERT.

        This method takes the input sequences and their corresponding
        attention masks and token type IDs, and passes them through the
        BERT model to obtain the pooled output (typically corresponding
        to the [CLS] token). The pooled output is then returned.

        The input sequences are expected to be tokenized and padded
        appropriately before being passed to this method.

        The pooled output is a tensor of shape (batch_size, hidden_size),
        where batch_size is the number of input sequences and hidden_size
        is the dimensionality of the BERT hidden states.

        Args:
            input_ids (_type_): tokenized input sequences
            attention_mask (_type_): indicates which tokens are padding tokens
            token_type_ids (_type_): used to distinguish between different segments
                                    in the input sequences

        Returns:
            tensor: pooled output from BERT
        """
        # Get BERT representations (pooled output typically corresponds to [CLS])
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        return pooled_output

    # For classifier, we may not need to use the projection head
    def forward_contrastive(self, input_ids, attention_mask, token_type_ids):
        """Forward pass for contrastive learning using the projection head.

        The projection head is a simple feed-forward neural network that
        takes the BERT pooled output and projects it into
        a lower-dimensional space for contrastive learning.

        Args:
            input_ids (_type_): tokenized input sequences
            attention_mask (_type_): used to indicate which tokens are padding tokens
            token_type_ids (_type_): distinguish between different segments in the input sequences

        Returns:
            tensor: projected output from the projection head
        """
        pooled_output = self.encode(input_ids, attention_mask, token_type_ids)
        projection = self.projection_head(pooled_output)
        return projection

    # We can use the pooled output directly for classification
    def forward_classifier(self, input_ids, attention_mask, token_type_ids):
        """Forward pass for classification using the classifier head.

        The classifier head is a feed-forward neural network that takes the
        BERT pooled output and produces logits for the classification task.
        Args:
            input_ids (_type_): tokenized input sequences
            attention_mask (_type_): used to indicate which tokens are padding tokens
            token_type_ids (_type_): distinguish between different segments in the input sequences
        Returns:
            tensor: logits for the classification task
        """
        pooled_output = self.encode(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)
        return logits
