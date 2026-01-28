from typing import Tuple, Dict, Any
import torch
import numpy as np
from einops import rearrange
from .BaseDataset import BaseDataset


class SequentialTransitionDataset(BaseDataset):
    """
    A dataset class for generating sequential transition data using PyTorch.
    Splits long trajectories into sliding windows for time-series forecasting.
    """
    def __init__(
        self,
        train_transitions,
        eval_transitions,
        history_length: int = 12,
        prediction_length: int = 1,
        batch_size: int = 32,
        step_size: int = 1,
        dtype: torch.dtype = torch.float32,
        use_normalization: bool = True,
        device: str = "cpu",
    ): 
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.step_size = step_size
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self._use_normalization = use_normalization
        
        # Process transitions into sliding windows
        # Returns X, Y, Mean, Std
        train_X_raw, train_Y_raw, train_mean, train_std = self._process_transitions(train_transitions)
        eval_X_raw, eval_Y_raw, _, _ = self._process_transitions(eval_transitions)
        
        # Setup normalization
        # We primarily use training statistics for the model's persistent state
        self.series_mean = train_mean
        self.series_std = train_std
        self._setup_normalization(train_X_raw, train_Y_raw)
        
        # Normalize data
        # We use the specific set's statistics to normalize it (per-patient normalization)
        # This assumes the model should see normalized inputs ~ N(0,1) regardless of the patient
        self.train_X, self.train_Y = self._normalize_data(train_X_raw, train_Y_raw, train_mean, train_std)
        self.eval_X, self.eval_Y = self._normalize_data(eval_X_raw, eval_Y_raw, train_mean, train_std)
        
        # Dataset properties
        num_functions = self.train_X.shape[0]
        self.n_variates = self.train_X.shape[-1]
        self.target_dim = self.train_Y.shape[-1]
        input_size = (self.history_length, self.n_variates)
        output_size = (self.prediction_length, self.target_dim)
        
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            total_n_functions=num_functions,
            total_n_samples_per_function=self.batch_size,
            data_type="deterministic",
            n_functions=num_functions,
            n_examples=self.batch_size // 2,
            n_queries=self.batch_size,
            dtype=dtype,
        )
        
        self._batch_start_idx = 0

    @property
    def use_normalization(self):
        return self._use_normalization
    
    def _setup_normalization(self, X_raw, Y_raw):
        """Calculate and store normalization statistics."""
        if self.use_normalization:
            # Statistics are already computed in _process_transitions and stored in self.series_mean/std
            
            self.series_mean_blood_glucose = self.series_mean[:, :, :1]
            self.series_std_blood_glucose = self.series_std[:, :, :1]
            
            self.torch_series_mean = torch.tensor(self.series_mean, dtype=self.dtype, device=self.device)
            self.torch_series_std = torch.tensor(self.series_std, dtype=self.dtype, device=self.device)
            self.torch_series_mean_blood_glucose = torch.tensor(self.series_mean_blood_glucose, dtype=self.dtype, device=self.device)
            self.torch_series_std_blood_glucose = torch.tensor(self.series_std_blood_glucose, dtype=self.dtype, device=self.device)
        else:
            # Defaults for no normalization
            self.series_mean = 0.
            self.series_std = 1.
            self.series_mean_blood_glucose = 0.
            self.series_std_blood_glucose = 1.
            
            self.torch_series_mean = torch.tensor(self.series_mean, dtype=self.dtype, device=self.device)
            self.torch_series_std = torch.tensor(self.series_std, dtype=self.dtype, device=self.device)
            self.torch_series_mean_blood_glucose = torch.tensor(self.series_mean_blood_glucose, dtype=self.dtype, device=self.device)
            self.torch_series_std_blood_glucose = torch.tensor(self.series_std_blood_glucose, dtype=self.dtype, device=self.device)

    def _normalize_data(self, X_raw, Y_raw, mean, std):
        """Normalize input and target data."""
        num_sequences = X_raw.shape[1]
        
        if self.use_normalization:
            # Flatten for batch processing
            X_flat = rearrange(X_raw, 's w t f -> s (w t) f')
            Y_flat = rearrange(Y_raw, 's w t f -> s (w t) f')
            
            # Mean and Std for Blood Glucose (Index 0)
            # mean shape: (S, 1, F), std shape: (S, 1, F)
            mean_bg = mean[:, :, :1]
            std_bg = std[:, :, :1]
            
            # Prevent division by zero
            std = np.where(std == 0.0, 1.0, std)
            std_bg = np.where(std_bg == 0.0, 1.0, std_bg)

            # Normalize
            X_flat_norm = (X_flat - mean) / std
            
            # CORRECTED: Do NOT subtract mean from Delta (Y)
            # Y represents a change, so it is zero-centered relative to the position.
            # Normalization should only scale it.
            Y_flat_norm = Y_flat / std_bg
            
            # Reshape back
            X_norm = rearrange(X_flat_norm, 's (w t) f -> s w t f', w=num_sequences)
            Y_norm = rearrange(Y_flat_norm, 's (w t) f -> s w t f', w=num_sequences)
            
            return X_norm, Y_norm
        else:
            return X_raw, Y_raw

    def denormalize(self, X_norm):
        return X_norm * self.series_std[:, np.newaxis, :] + self.series_mean[:, np.newaxis, :]

    def denormalize_y(self, y_norm):
        """Denormalize Y values (predictions or targets)."""
        if isinstance(y_norm, torch.Tensor):
            if len(y_norm.shape) == 4:
                y_norm = y_norm.squeeze(-1)
            return y_norm * self.torch_series_std_blood_glucose
        else:
            return y_norm * self.series_std_blood_glucose

    def _process_transitions(self, transitions):
        X_list = []
        Y_list = []
        
        # Calculate minimum length to ensure all patients have same number of samples
        min_T_len = float('inf')
        for key in transitions['Y']:
            T_len = transitions['Y'][key].shape[0]
            min_T_len = min(min_T_len, T_len)
        
        # Adjust for prediction length
        min_T_len = min_T_len - self.prediction_length + 1

        # Statistics collection
        bg_mean_X = []
        bg_std_X = []
    
        for key in transitions['X']:
            traj_X = transitions['X'][key]
            traj_Y = transitions['Y'][key]
            
            # Collect stats for normalization
            bg_mean_X.append(traj_X.mean(axis=0)[0])
            bg_std_X.append(traj_X.std(axis=0)[0])
            
            T_len = traj_X.shape[0]

            one_example_X = []
            one_example_Y = []
            
            # Create sliding windows
            for k in range(0, T_len - self.prediction_length + 1, self.step_size):
                # Input window
                start_idx = k - self.history_length + 1
                end_idx = k + 1
                
                if start_idx < 0:
                    pad_len = abs(start_idx)
                    pad = np.tile(traj_X[0], (pad_len, 1))
                    available_data = traj_X[0 : end_idx]
                    x_window = np.concatenate([pad, available_data], axis=0)
                else:
                    x_window = traj_X[start_idx : end_idx]
    
                # Output window
                y_abs_window = []
                for j in range(self.prediction_length):
                    idx = k + j
                    val = traj_Y[idx, 0]
                    y_abs_window.append(val)
                    
                y_window = np.array(y_abs_window).reshape(-1, 1) # (P, 1)
                
                one_example_X.append(x_window)
                one_example_Y.append(y_window)
            
            # Truncate to min length for uniform batches
            X_list.append(np.array(one_example_X)[None, :min_T_len, ...])
            Y_list.append(np.array(one_example_Y)[None, :min_T_len, ...])

        # Compute Statistics
        n_patients = len(X_list)
        feature_dim = X_list[0].shape[-1]
        
        current_series_mean = np.zeros((n_patients, 1, feature_dim), dtype=np.float32)
        current_series_std = np.ones((n_patients, 1, feature_dim), dtype=np.float32)
        
        current_series_mean[:, :, 0] = np.array(bg_mean_X)[:, np.newaxis]
        current_series_std[:, :, 0] = np.array(bg_std_X)[:, np.newaxis]

        X_list = np.concatenate(X_list, axis=0)
        Y_list = np.concatenate(Y_list, axis=0)
        
        return X_list, Y_list, current_series_mean, current_series_std

    def sample(self, phase: str = "train", use_sequential_batching: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if phase == "train":
            source_X, source_Y = self.train_X, self.train_Y
        elif phase == "eval":
            source_X, source_Y = self.eval_X, self.eval_Y
        else:
            raise ValueError(f"Invalid phase: {phase}")

        # First axis is the number of different patients
        num_samples = source_X.shape[1]
        batch_size = min(num_samples, self.batch_size)
        
        if use_sequential_batching:
            start = self._batch_start_idx
            end = min(start + batch_size, num_samples)
            indices = np.arange(start, end)
            
            # Wrap around
            if len(indices) < batch_size:
                remaining = batch_size - len(indices)
                indices = np.concatenate([indices, np.arange(0, remaining)])
                self._batch_start_idx = remaining
            else:
                self._batch_start_idx = end % num_samples
        else:
            indices = np.random.choice(num_samples, batch_size, replace=False)  
        
        # Extract batch
        # Shape: (N_Patients, Batch, History, Features)
        batch_X = source_X[:, indices]
        batch_Y = source_Y[:, indices]

        query_xs = torch.tensor(batch_X, dtype=self.dtype, device=self.device)
        query_ys = torch.tensor(batch_Y, dtype=self.dtype, device=self.device).cumsum(dim=-2)
        source_X_sample = source_X.shape[1]

        n_examples = np.array(range(min(self.n_examples, source_X_sample)))
        example_batch_X = source_X[:, n_examples]
        example_batch_Y = source_Y[:, n_examples]

        example_xs = torch.tensor(example_batch_X, dtype=self.dtype, device=self.device)
        example_ys = torch.tensor(example_batch_Y, dtype=self.dtype, device=self.device).cumsum(dim=-2)
        return example_xs, example_ys, query_xs, query_ys, {}

    def reset_batch_state(self):
        """Reset batch iteration state for new epoch."""
        self._batch_start_idx = self.n_examples


class TransitionDataset(BaseDataset):
    """
    A dataset class for generating transition data using PyTorch.
    """
    def __init__(
        self,
        train_transitions,
        eval_transitions,
        n_functions:int=1,
        n_examples:int=20,
        n_queries:int=20,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ): 
        train_transitions_X = train_transitions['X']
        train_transitions_Y = train_transitions['Y']
        eval_transitions_X = eval_transitions['X']
        eval_transitions_Y = eval_transitions['Y']

        self.train_hidden_parameters = list(train_transitions_X.keys())
        self.eval_hidden_parameters = list(eval_transitions_X.keys())
        self.n_queries, self.input_size = np.shape(train_transitions_X[self.train_hidden_parameters[0]])
        _, self.output_size = np.shape(train_transitions_Y[self.train_hidden_parameters[0]])
        
        # Keep data as numpy arrays initially - only convert to tensors when needed
        self.train_X = np.stack([train_transitions_X[key] for key in self.train_hidden_parameters], axis=0)
        self.train_Y = np.stack([train_transitions_Y[key] for key in self.train_hidden_parameters], axis=0)
        self.eval_X = np.stack([eval_transitions_X[key] for key in self.eval_hidden_parameters], axis=0)
        self.eval_Y = np.stack([eval_transitions_Y[key] for key in self.eval_hidden_parameters], axis=0)
        self.dt = 0.02
        
        super().__init__(
            input_size=(self.input_size, ),
            output_size=(self.output_size, ),
            total_n_functions=float("inf"),
            total_n_samples_per_function=float("inf"),
            data_type="deterministic",
            n_functions=n_functions,
            n_examples=n_examples,
            n_queries=self.n_queries,
            dtype=dtype,
        )
        self.n_queries = self.n_queries - self.n_examples
        self.device = device
        
        # Add state for batch iteration
        self._current_epoch = 0
        self._batch_start_idx = 0


    def sample(self, phase: str = "train", use_sequential_batching: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert phase in ["train", "eval"], f"Invalid phase: {phase}. Please specify 'train' or 'eval'."

        if phase == "train":
            num_fns = self.train_X.shape[0]
            n_functions = min(num_fns, self.n_functions)
            source_X, source_Y = self.train_X, self.train_Y
            hidden_params = self.train_hidden_parameters
        else:
            num_fns = self.eval_X.shape[0]
            n_functions = min(num_fns, self.n_functions)
            source_X, source_Y = self.eval_X, self.eval_Y
            hidden_params = self.eval_hidden_parameters
            
        if use_sequential_batching:
            start_idx = self._batch_start_idx
            end_idx = min(start_idx + n_functions, num_fns)
            
            if end_idx - start_idx < n_functions:
                indices_1 = list(range(start_idx, num_fns))
                indices_2 = list(range(0, n_functions - len(indices_1)))
                fn_indices = np.array(indices_1 + indices_2)
                if phase == "train": self._batch_start_idx = len(indices_2)
            else:
                fn_indices = np.arange(start_idx, end_idx)
                if phase == "train": self._batch_start_idx = end_idx % num_fns
        else:
            fn_indices = np.random.choice(num_fns, n_functions, replace=False)
        
        example_xs = source_X[fn_indices, :self.n_examples, ...]
        example_ys = source_Y[fn_indices, :self.n_examples, ...]
        xs = source_X[fn_indices, self.n_examples:, ...]
        ys = source_Y[fn_indices, self.n_examples:, ...]
        info = {f'{phase}_hidden_parameters': [hidden_params[i] for i in fn_indices]}

        return (torch.tensor(example_xs, dtype=self.dtype, device=self.device),
                torch.tensor(example_ys, dtype=self.dtype, device=self.device),
                torch.tensor(xs, dtype=self.dtype, device=self.device),
                torch.tensor(ys, dtype=self.dtype, device=self.device),
                info)
    
    def reset_batch_state(self):
        """Reset batch iteration state for new epoch."""
        self._batch_start_idx = 0