import torch
import torch.nn as nn
from tqdm import trange
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from typing import Optional, Dict, Any


# --------------------------------------------------------------------
# 1) Embed each variate's raw time-series (length T) into a D-dim token
# --------------------------------------------------------------------
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


# ----------------------------------------------------
# 4) The tiny but complete iTransformer skeleton
# ----------------------------------------------------
class ITransformer(nn.Module):
    """
    Args
    ----
    input_len : T  -> look-back window
    pred_len  : S  -> horizon to predict
    n_variates: N  -> number of input channels / sensors
    """
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_variates: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_hidden: int = 128,
        predict_only_bg: bool = True,
    ):
        super().__init__()
        self.name = "itransformer"
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_variates = n_variates
        self.predict_only_bg = predict_only_bg
        self.output_variates = 1 if predict_only_bg else n_variates
        self.embed = VariateEmbedding(input_len, d_model)

        self.blocks = nn.ModuleList(
            [
                ITBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_hidden=ffn_hidden,
                )
                for _ in range(num_layers)
            ]
        )

        # projects the final token back to an S-long series
        self.project = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input  : x (B, T, N)
        Output : y_hat (B, S, 1) if predict_only_bg else (B, S, N)
        """
        reshape = False
        if len(x.shape) == 4:
            # Reshape from 4D to 3D for history encoder
            # (n_functions, n_examples, history_length, feature_dim) -> (batch_size, history_length, feature_dim)
            original_shape = x.shape
            batch_size = x.shape[0] * x.shape[1]  # n_functions * n_examples
            history_length = x.shape[2]
            feature_dim = x.shape[3]
            x = x.reshape(batch_size, history_length, feature_dim)
            reshape = True
        h = self.embed(x)                       # (B, N, D)

        for block in self.blocks:               # L ×
            h = block(h)                        # (B, N, D)

        y_hat = self.project(h)                 # (B, N, S)
        if self.predict_only_bg:
            # keep only BG channel (index 0)
            y_hat = y_hat[:, :1, :]             # (B, 1, S)
            y_hat = y_hat.permute(0, 2, 1)      # (B, S, 1)
        else:
            y_hat = y_hat.permute(0, 2, 1)      # (B, S, N)

        if reshape:
            y_hat = y_hat.view(original_shape[0], original_shape[1], self.pred_len, self.output_variates)
        return y_hat
    
    @property
    def history_encoder(self):
        return None
    
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, learn_basis_functions=True, *args, **kwargs):
        # Estimate parameters for ITransformer
        # This is a rough estimate or we can instantiate a dummy model
        # Args passed from FunctionEncoder:
        # input_size: tuple (history_len, n_variates)
        # output_size: tuple (pred_len, out_dim)
        # kwargs: d_model, n_heads, etc.
        
        # Extract params
        input_len = input_size[0]
        n_variates = input_size[1]
        pred_len = output_size[0]
        
        d_model = kwargs.get('d_model', 64)
        n_heads = kwargs.get('n_heads', 4)
        num_layers = kwargs.get('num_layers', 2)
        ffn_hidden = kwargs.get('ffn_hidden', 128)
        
        # Embed: Linear(input_len, d_model) -> input_len * d_model + d_model
        embed_params = input_len * d_model + d_model
        
        # Blocks:
        # MultiheadAttention: 4 * d_model * d_model + 4 * d_model (in_proj) + d_model * d_model + d_model (out_proj)
        # Norms: 2 * d_model * 2 (gamma + beta)
        # FFN: d_model * ffn_hidden + ffn_hidden + ffn_hidden * d_model + d_model
        
        attn_params = 4 * d_model**2 + 4 * d_model  # in_proj_weight + in_proj_bias
        attn_params += d_model**2 + d_model         # out_proj
        
        norm_params = 2 * d_model * 2
        
        ffn_params = d_model * ffn_hidden + ffn_hidden
        ffn_params += ffn_hidden * d_model + d_model
        
        block_params = attn_params + norm_params + ffn_params
        
        # Project: Linear(d_model, pred_len) -> d_model * pred_len + pred_len
        project_params = d_model * pred_len + pred_len
        
        total = embed_params + num_layers * block_params + project_params
        return total

    def parameter_count(self) -> int:
        """Number of trainable parameters for this ITransformer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self, function_encoder: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Return parameter counts for fair comparison.

        Args:
            function_encoder: optionally pass the FunctionEncoder instance to sum parameters.
        """
        report = {"itransformer_params": self.parameter_count()}
        if function_encoder is not None:
            fe_params = sum(p.numel() for p in function_encoder.parameters() if p.requires_grad)
            report["function_encoder_params"] = fe_params
            report["total_params"] = fe_params + report["itransformer_params"]
        return report
    
    def set_optimizer(self, optim):
        self.optim = optim
    
    def train_model(self, dataset, epochs=100, callback=None):
        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        losses = []
        bar = trange(epochs)
        for epoch in bar:
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()            
            y_hats = self(query_xs)
            prediction_loss = torch.nn.MSELoss()(y_hats, query_ys)

            loss = prediction_loss
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())



# ----------------------------------------------------
# 5) Quick smoke-test
# ----------------------------------------------------
if __name__ == "__main__":
    B, T, S, N = 32, 24, 12, 10        # example dims
    model = ITransformer(
        input_len=T,
        pred_len=S,
        n_variates=N,
        d_model=64,
        n_heads=4,
        num_layers=3,
        ffn_hidden=128,
    )

    dummy = torch.randn(B, T, N)
    out = model(dummy)                 # (B, S, N)
    print(out.shape)                   # torch.Size([32, 12, 10])
