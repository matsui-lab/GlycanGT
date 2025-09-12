"""
Modified from https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/modules/feedforward.py
"""
# feedforward.py
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A simple feed-forward network.

    This module consists of two linear layers with a specified activation
    function and dropout applied in between and after the layers.
    """
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_fn: str,
        activation_dropout: float,
        dropout: float,
    ):
        """
        :param embedding_dim:
            The input and output dimensionality of the network.
        :param ffn_embedding_dim:
            The inner dimensionality of the network.
        :param activation_fn:
            The name of the activation function to use.
        :param activation_dropout:
            The dropout probability to apply after the activation function.
        :param dropout:
            The dropout probability to apply to the output of the network.
        """
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        
        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "silu":
            self.activation_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.activation_dropout_module = nn.Dropout(p=activation_dropout)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.dropout_module = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
            The input tensor.

        :returns:
            The output tensor.
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x
