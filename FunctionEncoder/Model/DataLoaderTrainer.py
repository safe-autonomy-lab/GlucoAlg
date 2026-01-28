from typing import Union, Tuple, Dict, Any
import logging
import torch
import gc
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DataLoaderTrainer:
    """
    Efficient trainer for FunctionEncoder using PyTorch DataLoader.
    This class handles training with automatic memory management and batching.
    """
    
    def __init__(self, model, optimizer=None):
        """
        Args:
            model: FunctionEncoder model to train
            optimizer: PyTorch optimizer (will use model's optimizer if None)
        """
        self.model = model
        self.optimizer = optimizer or model.opt
        
    def train_with_dataloader(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        epochs: int = 100,
        progress_bar: bool = True,
        callback=None,
        **kwargs
    ):
        """
        Train the model using DataLoader for memory efficiency.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: DataLoader for evaluation data (optional)
            epochs: Number of training epochs
            progress_bar: Whether to show progress bar
            callback: Callback for logging/monitoring
            
        Returns:
            List of losses per epoch
        """
        device = next(self.model.parameters()).device
        model = self.model
        
        # Let callbacks know training is starting
        if callback is not None:
            callback.on_training_start(locals())

        losses = []
        epoch_bar = trange(epochs, desc="Epochs") if progress_bar else range(epochs)
        
        for epoch in epoch_bar:
            epoch_loss = self._train_epoch(train_dataloader, progress_bar, callback, epoch)
            losses.append(epoch_loss)
            
            # Evaluation phase
            if eval_dataloader is not None:
                eval_loss = self._eval_epoch(eval_dataloader, callback, epoch)
                
            # Update progress bar
            if progress_bar and hasattr(epoch_bar, 'set_postfix'):
                postfix = {'train_loss': f'{epoch_loss:.6f}'}
                if eval_dataloader is not None:
                    postfix['eval_loss'] = f'{eval_loss:.6f}'
                epoch_bar.set_postfix(postfix)
            
        # Let callbacks know training is complete
        if callback is not None:
            callback.on_training_end(locals())
            
        return losses
    
    def _train_epoch(self, dataloader: DataLoader, progress_bar: bool, callback, epoch: int) -> float:
        """Train for one epoch."""
        model = self.model
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False) if progress_bar else dataloader
        
        for batch_idx, (example_xs, example_ys, query_xs, query_ys, info) in enumerate(batch_bar):
            # Ensure tensors are on correct device
            device = next(model.parameters()).device
            example_xs = example_xs.to(device)
            example_ys = example_ys.to(device)
            query_xs = query_xs.to(device) 
            query_ys = query_ys.to(device)
            
            # Forward pass
            batch_loss = self._compute_batch_loss(example_xs, example_ys, query_xs, query_ys)
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping and optimization step
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Accumulate loss
            epoch_loss += batch_loss.item()
            num_batches += 1
            
            # Memory cleanup
            del example_xs, example_ys, query_xs, query_ys, batch_loss
            
            # Periodic memory cleanup
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Callback for batch
            if callback is not None:
                batch_locals = {
                    'batch_idx': batch_idx,
                    'epoch': epoch,
                    'batch_loss': epoch_loss / (batch_idx + 1),  # running average
                    'info': info
                }
                callback.on_step(batch_locals)
            
            # Update batch progress bar
            if progress_bar and hasattr(batch_bar, 'set_postfix'):
                batch_bar.set_postfix({'loss': f'{batch_loss.item():.6f}'})
        
        return epoch_loss / max(num_batches, 1)
    
    def _eval_epoch(self, dataloader: DataLoader, callback, epoch: int) -> float:
        """Evaluate for one epoch."""
        model = self.model
        model.eval()
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for example_xs, example_ys, query_xs, query_ys, info in dataloader:
                # Ensure tensors are on correct device
                device = next(model.parameters()).device
                example_xs = example_xs.to(device)
                example_ys = example_ys.to(device)
                query_xs = query_xs.to(device)
                query_ys = query_ys.to(device)
                
                # Forward pass only
                batch_loss = self._compute_batch_loss(example_xs, example_ys, query_xs, query_ys)
                eval_loss += batch_loss.item()
                num_batches += 1
                
                # Memory cleanup
                del example_xs, example_ys, query_xs, query_ys, batch_loss
        
        avg_eval_loss = eval_loss / max(num_batches, 1)
        
        # Callback for evaluation
        if callback is not None:
            eval_locals = {
                'epoch': epoch,
                'eval_loss': avg_eval_loss,
                'phase': 'eval'
            }
            callback.on_step(eval_locals)
            
        return avg_eval_loss
    
    def _compute_batch_loss(self, example_xs, example_ys, query_xs, query_ys):
        """Compute loss for a batch."""
        model = self.model
        device = next(model.parameters()).device
        
        # Train average function if it exists
        expected_yhats = None
        average_function_loss = 0.0
        if model.average_function is not None:
            expected_yhats = model.average_function.forward(query_xs).to(device)
            average_function_loss = model._distance(expected_yhats, query_ys, squared=True).mean()
            expected_yhats = expected_yhats.detach()
        
        # Compute representation and predictions
        representation, gram = model.compute_representation(example_xs, example_ys, method=model.method)
        y_hats = model.predict(query_xs, representation, precomputed_average_ys=expected_yhats)
        prediction_loss = model._distance(y_hats, query_ys, squared=True).mean()
        
        # Regularization for least squares
        norm_loss = 0.0
        if model.method == "least_squares":
            norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()
        
        # Combine losses
        total_loss = prediction_loss
        if model.method == "least_squares":
            total_loss = total_loss + model.regularization_parameter * norm_loss
        if model.average_function is not None:
            total_loss = total_loss + average_function_loss
            
        return total_loss
