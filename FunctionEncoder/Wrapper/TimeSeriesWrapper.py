import torch
import logging
import gc
from tqdm import trange
from typing import Dict, Any, Optional, Tuple, Union
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
import numpy as np
# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TimeSeriesWrapper(torch.nn.Module):
    """
    Wrapper around FunctionEncoder to convert delta predictions to absolute position predictions.
    
    The FunctionEncoder predicts position deltas (changes). This wrapper accumulates these 
    deltas to predict absolute positions over a time horizon.
    
    It also handles normalization/denormalization if mean and std are provided.
    """
    
    def __init__(self, 
                 function_encoder: FunctionEncoder, 
                 feature_dim: int = 2, 
                 history_length: int = 10, 
                 normalization_mean: Optional[Union[torch.Tensor, float]] = None, 
                 normalization_std: Optional[Union[torch.Tensor, float]] = None):
        super().__init__()
        self.function_encoder = function_encoder
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.prediction_horizon = function_encoder.model.prediction_length
        
        # Register normalization parameters as buffers
        self._register_normalization_param('normalization_mean', normalization_mean)
        self._register_normalization_param('normalization_std', normalization_std)

        self.name = 'function_encoder'

    def _register_normalization_param(self, name: str, value: Any):
        """Helper to register normalization parameters as buffers."""
        if value is not None:
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32).to(self.function_encoder.device)
            self.register_buffer(name, value)
        else:
            self.register_buffer(name, None)

    def compute_representation(self, example_xs: torch.Tensor, example_ys: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        """Computes the representation using the underlying FunctionEncoder."""
        prediction_horizon = self.prediction_horizon
        return self.function_encoder.compute_representation(example_xs, example_ys, prediction_horizon=prediction_horizon, **kwargs)

    def predict(self, 
                x: torch.Tensor, 
                coeffs: torch.Tensor, 
                prediction_horizon: int, 
                denormalize: bool = False, 
                index: Optional[int] = None,
                train: bool = False,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict absolute positions for `prediction_horizon` steps.
        
        Args:
            x: Input history. Shape (N_funcs, N_queries, history_length, feature_dim)
            coeffs: Representation coefficients.
            prediction_horizon: Number of steps to predict.
            denormalize: If True, applies (x * std + mean) to the output.
                         Assumes the internal model works in normalized space.
            
        Returns:
            absolute_predictions: (N_funcs, N_queries, prediction_horizon, output_dim)
            delta_predictions: (N_funcs, N_queries, prediction_horizon, output_dim)
        """
        assert prediction_horizon >= 1, f"prediction_horizon must be >= 1, got {prediction_horizon}"    
        
        # Prediction loop in latent/normalized space
        current_input = x.clone()        
        # Predict delta: (N, Q, target_dim)
        delta = self.function_encoder.predict(current_input, coeffs, prediction_horizon=prediction_horizon, **kwargs)
        if train:
            return None, delta
        # Get last known position: (N, Q, target_dim)
        # Assuming target features are the first ones
        # We only interested in the blood glucose features
        last_pos = current_input[:, :, -1, :1]

        # Absolute position (still normalized)
        # pred_pos = last_pos + delta.cumsum(dim=-1)
        pred_pos = last_pos + delta
            
        if denormalize and self.normalization_mean is not None and self.normalization_std is not None:
            # Apply denormalization: y_real = y_norm * std + mean
            # Ensure shapes broadcast correctly
            # normalization buffers are usually (N_funcs, 1, 1) or (1, 1, 1) depending on data
            if index is not None:
                pred_pos = pred_pos * self.normalization_std[index: index + 1] + self.normalization_mean[index: index + 1]
            else:
                pred_pos = pred_pos * self.normalization_std + self.normalization_mean
        
        return pred_pos, delta

    def train_model(self,
                    dataset, 
                    epochs: int,
                    batch_size: int = 32,
                    progress_bar: bool = True,
                    callback = None,
                    save_best_path: Optional[str] = None,
                    **kwargs: Any):
        """ 
        Trains the function encoder on the dataset.
        """
        dataset.check_dataset()
        
        # Ensure optimizer exists
        if not hasattr(self.function_encoder, 'opt'):
             self.function_encoder.opt = torch.optim.Adam(self.function_encoder.parameters(), lr=1e-3)

        losses = []
        iterator = trange(epochs, desc="Epochs") if progress_bar else range(epochs)
        prediction_horizon = self.prediction_horizon
        denormalize = True if self.normalization_mean is None else False
        best_eval_error = float('inf')
        for epoch in iterator:
            if hasattr(dataset, 'reset_batch_state'):
                dataset.reset_batch_state()
            
            original_n_funcs = dataset.n_functions
            total_seqs = dataset.train_X.shape[1]
            n_batches = max(1, (total_seqs + batch_size - 1) // batch_size)
            
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                # Adjust dataset to serve a batch of sequences
                dataset.n_sequences = min(batch_size, total_seqs - batch_idx * batch_size)
                if dataset.n_sequences <= 0: break
            
                # Sample batch
                # query_ys: (N, Q, P, D)
                ex_xs, ex_ys, q_xs, q_ys, _ = dataset.sample()
                
                # Compute representation
                # Handle shape mismatch if dataset returns (N, Ex, P, D) but encoder wants (N, Ex, D)
                rep_ys = ex_ys
                if rep_ys.ndim == 4:
                    rep_ys = rep_ys.squeeze(-1)

                
                representation, gram = self.function_encoder.compute_representation(
                    ex_xs, rep_ys, method=self.function_encoder.method, prediction_horizon=prediction_horizon, **kwargs
                )

                # Predict 1-step ahead
                # y_hats: (N, Q, 1, D)
                # during training, we target delta instead of absolu pos
                _, y_hats = self.predict(q_xs, representation, prediction_horizon=prediction_horizon, denormalize=denormalize, train=True)

                # print("--------------------------------")
                # print('representation', representation)
                # print('y_hats', y_hats[:, 0, :])
                # print('q_ys', q_ys[:, 0, :, 0])
                # print("shape of y_hats", y_hats.shape)
                # print("shape of q_ys", q_ys.shape)
                # print("--------------------------------")
                # exit()

                y_hats = y_hats.squeeze(-1)
                q_ys = q_ys.squeeze(-1)

                # Prepare target
                prediction_loss = self.function_encoder._distance(y_hats, q_ys, squared=True).mean()

                loss = prediction_loss
                if self.function_encoder.method == "least_squares":
                    norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()
                    loss = loss + self.function_encoder.regularization_parameter * norm_loss
                
                epoch_loss += loss.item()
                
                # Optimization step
                loss.backward()
                if (batch_idx + 1) % self.function_encoder.gradient_accumulation == 0 or batch_idx == n_batches - 1:
                    torch.nn.utils.clip_grad_norm_(self.function_encoder.parameters(), 1.0)
                    self.function_encoder.opt.step()
                    self.function_encoder.opt.zero_grad()
                
                # Cleanup
                del loss, representation, y_hats, gram
                # break

            if callback:
                eval_error = callback.on_step({
                    'self': self.function_encoder,
                })
                if eval_error < best_eval_error:
                    best_eval_error = eval_error
                    if save_best_path:
                        torch.save(self.function_encoder.state_dict(), save_best_path)
                    
                dataset.n_functions = original_n_funcs
                avg_loss = epoch_loss / n_batches
                losses.append(avg_loss)
                
                if progress_bar and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({'loss': f'{avg_loss:.6f}'})

        if callback: callback.on_training_end(locals())
        return losses
