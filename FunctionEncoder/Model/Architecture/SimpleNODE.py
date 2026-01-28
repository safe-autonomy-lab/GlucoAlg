import torch
import torch.nn as nn
from FunctionEncoder.Model.Architecture.Util import get_encoder

class SimpleNeuralODEBaseline(nn.Module):
    """
    Minimal neural ODE-style baseline that rolls forward a latent state for BG.
    State is initialized from the flattened history window and rolled out with Euler steps.
    """
    def __init__(
        self,
        history_length: int,
        feature_dim: int,
        pred_len: int,
        hidden: int = 64,
        encoder_type: str = None,
        encoder_kwargs: dict = None,
    ):
        super().__init__()
        self.history_length = history_length
        self.feature_dim = feature_dim
        self.pred_len = pred_len
        self.state_dim = 1  # focus on BG channel
        self.history_encoder = None
        self.enc_proj = None
        if encoder_type is not None:
            kwargs = encoder_kwargs or {}
            hidden_size = kwargs.pop("hidden_size", 64)
            self.history_encoder = get_encoder(
                encoder_type=encoder_type,
                input_size=feature_dim,
                hidden_size=hidden_size,
                **kwargs,
            )
            enc_out_dim = feature_dim if encoder_type == "itransformer" else hidden_size
            self.enc_proj = nn.Linear(enc_out_dim, self.state_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(history_length * feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.state_dim),
            )
        self.derivative = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.state_dim),
        )

    def forward(self, x: torch.Tensor, pred_len: int = None):
        # x: (N, B, H, F)
        prediction_len = pred_len if pred_len is not None else self.pred_len
        N, B, H, Fdim = x.shape
        if self.history_encoder is not None:
            reshaped = x.reshape(N * B, H, Fdim)
            enc = self.history_encoder(reshaped)  # (N*B, hidden)
            state = self.enc_proj(enc)            # (N*B, 1)
        else:
            flat = x.reshape(N * B, H * Fdim)
            state = self.encoder(flat)  # (N*B, 1)
        preds = []
        for _ in range(prediction_len):
            deriv = self.derivative(state)
            state = state + deriv  # Euler step dt=1
            preds.append(state)
        preds = torch.stack(preds, dim=1).reshape(N, B, prediction_len, self.state_dim)
        return preds.squeeze(-1)  # (N, B, P)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)