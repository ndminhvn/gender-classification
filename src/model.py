import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """
    A simple text encoder using an Embedding layer followed by an BiLSTM
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length) with token indices
        Returns a representation for each input in the batch
        """
        # (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        # LSTM output shape: (batch_size, sequence_length, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(embedded)
        # Use mean pooling over the time dimension as the representation.
        representation = torch.mean(lstm_out, dim=1)
        return representation


class ProjectionHead(nn.Module):
    """
    Projection head for mapping encoder outputs to a latent space for contrastive learning
    """

    def __init__(self, input_dim: int, proj_dim: int):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, proj_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ClassifierHead(nn.Module):
    """
    A simple classifier head for predicting class labels
    """

    def __init__(self, input_dim: int, num_classes: int):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits


class ContrastiveModel(nn.Module):
    """
    Complete model combining a text encoder, projection head (for contrastive learning),
    and classifier head (for supervised classification)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        proj_dim: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super(ContrastiveModel, self).__init__()
        self.encoder = TextEncoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        # Adjust the encoder output dimension based on whether it is bidirectional
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection_head = ProjectionHead(encoder_output_dim, proj_dim)
        self.classifier_head = ClassifierHead(encoder_output_dim, num_classes)

    def forward(self, x):
        """
        Forward pass for classification
        Returns class logits
        """
        representation = self.encoder(x)
        logits = self.classifier_head(representation)
        return logits

    def forward_contrastive(self, x):
        """
        Forward pass for contrastive learning
        Returns the projected latent vectors
        """
        representation = self.encoder(x)
        proj = self.projection_head(representation)
        return proj


# Example usage:
# if __name__ == "__main__":
#     batch_size = 4
#     seq_length = 10
#     vocab_size = 10000
#     embedding_dim = 128
#     hidden_dim = 256
#     proj_dim = 64
#     num_classes = 2

#     # Create a dummy batch of tokenized sequences
#     dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

#     # Instantiate the model
#     model = ContrastiveModel(
#         vocab_size, embedding_dim, hidden_dim, proj_dim, num_classes
#     )

#     # Classification forward pass
#     logits = model(dummy_input)
#     print("Classification logits shape:", logits.shape)

#     # Contrastive forward pass
#     proj_vectors = model.forward_contrastive(dummy_input)
#     print("Projection vectors shape:", proj_vectors.shape)
