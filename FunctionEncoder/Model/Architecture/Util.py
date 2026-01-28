import torch
import torch.nn as nn
from typing import Optional, Union


class VariateEmbedding(nn.Module):
    """
    Turns one univariate series of length T into a D-dim token.
    Operates on the last dim, so we expect x shape (B, T, N).
    """
    def __init__(self, input_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_len, d_model)   # works on length-T axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x  : (B, T, N) --> (B, N, T) so each variate is contiguous
        x = x.permute(0, 2, 1)
        # apply shared MLP → (B, N, D)
        return self.proj(x)


# --------------------------------------------------------------
# 2) Classic Transformer feed-forward block (shared for all N)
# --------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# 3) One iTransformer block = attention across variates + FFN
# ------------------------------------------------------------------
class ITBlock(nn.Module):
    """
    Multivariate attention across the N variate tokens, followed by a
    token-wise feed-forward network.  Residual + LayerNorm as usual.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention across the N variate tokens
        attn_out, _ = self.attn(x, x, x)            # (B, N, D)
        x = self.norm1(x + attn_out)                # residual 1
        # Shared FFN (broadcasted to every token)
        ffn_out = self.ffn(x)                       # (B, N, D)
        x = self.norm2(x + ffn_out)                 # residual 2
        return x


class ITransformerEncoder(nn.Module):
    """
    ITransformer-based encoder for timeseries history data.
    """
    def __init__(self, history_length: int = 4, hidden_size: int = 128, num_heads: int = 4, num_layers: int = 2):
        super(ITransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = VariateEmbedding(history_length, hidden_size)
        self.blocks = nn.ModuleList(
            [
                ITBlock(
                    d_model=hidden_size,
                    n_heads=num_heads,
                    ffn_hidden=hidden_size,
                )
                for _ in range(num_layers) # 2 layers
            ]
        )
        self.project = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.project(x).squeeze(-1)
        
        return x


class AttentionHistoryEncoder(nn.Module):
    """
    Attention-based encoder for timeseries history data.
    Uses transformer encoder layers with positional encodings.
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_heads: int = 4, history_length: int = 5):
        super(AttentionHistoryEncoder, self).__init__()
        # Project input features to hidden_size
        self.proj = nn.Linear(input_size, hidden_size)
        # Learnable positional encodings
        self.pos_encodings = nn.Parameter(torch.zeros(1, history_length, hidden_size))
        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for attention-based history encoding.
        
        Args:
            x: Input tensor of shape (batch_size, history_length, input_size)
            mask: Optional mask tensor for padding positions
            
        Returns:
            Encoded representation of shape (batch_size, hidden_size)
        """
        # x shape: (batch_size, history_length, input_size)
        batch_size = x.shape[0]
        x_proj = self.proj(x)  # (batch_size, history_length, hidden_size)
        x_pos = x_proj + self.pos_encodings  # Add positional info
        output = self.encoder_layer(x_pos, src_key_padding_mask=mask)  # (batch_size, history_length, hidden_size)
        
        # Take the last time step as the encoded representation
        # For masked sequences, we should take the last valid position, not just the last position
        if mask is not None:
            # Create a tensor to store the encoded representations
            encoded_repr = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
            for i in range(batch_size):
                if torch.all(mask[i]):
                    # If all positions are masked (shouldn't happen), use the first position
                    encoded_repr[i] = output[i, 0]
                else:
                    # Find the last valid position (where mask is False)
                    valid_indices = torch.where(~mask[i])[0]
                    if len(valid_indices) > 0:
                        last_valid_idx = valid_indices[-1]
                        encoded_repr[i] = output[i, last_valid_idx]
                    else:
                        # Fallback to the last position (shouldn't happen)
                        encoded_repr[i] = output[i, -1]
        else:
            # If no mask, just take the last position
            encoded_repr = output[:, -1, :]  # (batch_size, hidden_size)
            
        return encoded_repr


class LSTMHistoryEncoder(nn.Module):
    """
    LSTM-based encoder for timeseries history data.
    """
    def __init__(self, input_size: int = 4, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super(LSTMHistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for LSTM-based history encoding.
        
        Args:
            x: Input tensor of shape (batch_size, history_length, input_size)
            mask: Optional mask tensor for padding positions (not used in LSTM)
            
        Returns:
            Encoded representation of shape (batch_size, hidden_size)
        """
        # x shape: (batch_size, history_length, input_size)
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward pass through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Return the last hidden state from the last layer
        return hn[-1]  # Shape: (batch_size, hidden_size)


def get_encoder(encoder_type: str, 
                input_size: int = 4, 
                hidden_size: int = 128, 
                **kwargs) -> Union[AttentionHistoryEncoder, LSTMHistoryEncoder, ITransformerEncoder]:
    """
    Factory function to create different types of encoders for timeseries data.
    
    Args:
        encoder_type: Type of encoder ('attention', 'attn', 'lstm')
        input_size: Size of input features
        hidden_size: Size of hidden representation
        **kwargs: Additional arguments specific to each encoder type
        
    Returns:
        Encoder instance (AttentionHistoryEncoder or LSTMHistoryEncoder)
        
    Raises:
        ValueError: If encoder_type is not supported
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "attention":
        # Extract attention-specific parameters
        num_heads = kwargs.get('num_heads', 4)
        history_length = kwargs.get('history_length', 5)
        
        return AttentionHistoryEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            history_length=history_length
        )
    
    elif encoder_type == 'lstm':
        # Extract LSTM-specific parameters
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.0)
        
        return LSTMHistoryEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    elif encoder_type == 'itransformer':
        num_layers = kwargs.get('num_layers', 2)
        num_heads = kwargs.get('num_heads', 4)
        history_length = kwargs.get('history_length', 5)

        return ITransformerEncoder(
            history_length=history_length,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}. "
                        f"Supported types are: 'attention', 'lstm'")
