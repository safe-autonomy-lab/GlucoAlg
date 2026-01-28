from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from stable_baselines3.common.logger import configure


class LoggerCallback(BaseCallback):

    def __init__(self,
                 model:FunctionEncoder,
                 testing_dataset:BaseDataset,
                 logdir: Union[str, None] = None,
                 prefix="test",
                 ):        
        super(LoggerCallback, self).__init__()
        self.logger = configure(logdir)
        self.total_epochs = 0
        self.model = model
        self.testing_dataset = testing_dataset

    def on_step(self, locals:dict):
        """
        Evaluate once per epoch over the entire train/eval splits.
        Assumes caller invokes on_step at epoch boundaries.
        """
        with torch.no_grad():
            model = self.model
            param = next(model.parameters()) if any(p.requires_grad for p in model.parameters()) else None
            device = param.device if param is not None else torch.device("cpu")
            dtype = param.dtype if param is not None else torch.float32
            prediction_horizon = getattr(model, "model", model).prediction_length if hasattr(getattr(model, "model", model), "prediction_length") else None

            def _evaluate_split(phase: str):
                source_X = self.testing_dataset.train_X if phase == "train" else self.testing_dataset.eval_X
                source_Y = self.testing_dataset.train_Y if phase == "train" else self.testing_dataset.eval_Y
                if source_X.shape[0] == 0 or source_X.shape[1] == 0:
                    return 0.0

                X_flat_before_reshape = torch.as_tensor(source_X, device=device, dtype=dtype)
                Y_flat_before_reshape = torch.as_tensor(source_Y, device=device, dtype=dtype)

                # merge patient and window dims: (P, B, H, F) -> (P*B, H, F)
                X_flat = X_flat_before_reshape.reshape(-1, source_X.shape[-2], source_X.shape[-1])
                Y_flat = Y_flat_before_reshape.reshape(-1, source_Y.shape[-2], source_Y.shape[-1])
                # Limit to available patients/windows to avoid empty batches.
                n_patients = source_X.shape[0]
                n_windows = source_X.shape[1]
                subset_size = min(5, n_patients)
                n_examples = min(20, max(1, n_windows - 1))
                sum_abs_error = 0
                sum_mse_error = 0
                sum_l1_error = 0
                sum_last_l1_error = 0
                sum_last_mse_error = 0
                sum_accum_l1_error = 0
                sum_accum_mse_error = 0
                num_subsets = 0
                for i in range(0, n_patients, subset_size):
                    if hasattr(model, "compute_representation"):
                        example_xs = X_flat_before_reshape[i:i+subset_size, :n_examples]
                        example_ys = Y_flat_before_reshape[i:i+subset_size, :n_examples]
                        query_xs = X_flat_before_reshape[i:i+subset_size, n_examples:]
                        query_ys = Y_flat_before_reshape[i:i+subset_size, n_examples:]
                        if query_xs.shape[1] == 0:
                            continue
                        representation, _ = model.compute_representation(
                            example_xs, example_ys.squeeze(-1), method=getattr(model, "method", "least_squares"), prediction_horizon=prediction_horizon
                        )                        
                        y_hats_flat = model.predict(query_xs, representation, prediction_horizon=prediction_horizon)
                        y_hats = y_hats_flat.reshape(query_ys.shape)
                        diff = query_ys - y_hats
                        diff = diff.squeeze(-1)
                        
                    else:
                        preds = model(X_flat.unsqueeze(0))
                        if preds.ndim == 4:
                            preds = preds[..., 0]
                        target = Y_flat.unsqueeze(0)
                        if target.ndim == 4:
                            target = target.squeeze(-1)
                        diff = target - preds

                    if hasattr(self.testing_dataset, "torch_series_std_blood_glucose"):
                        std_bg = self.testing_dataset.torch_series_std_blood_glucose.to(device=device, dtype=dtype)
                        diff_abs = diff * std_bg[i:i+subset_size]
                    else:
                        diff_abs = diff

                    abs_errors = torch.abs(diff_abs)
                    mse_abs = torch.mean(diff_abs**2)
                    l1_abs = torch.mean(abs_errors)
                    last_l1_abs = abs_errors[..., -1].mean()
                    last_mse_abs = (diff_abs[..., -1] ** 2).mean()
                    accum_l1_abs = abs_errors.sum(dim=-1).mean()
                    accum_mse_abs = (diff_abs**2).sum(dim=-1).mean()

                    sum_abs_error += abs_errors.sum().item()
                    sum_mse_error += mse_abs.item()
                    sum_l1_error += l1_abs.item()
                    sum_last_l1_error += last_l1_abs.item()
                    sum_last_mse_error += last_mse_abs.item()
                    sum_accum_l1_error += accum_l1_abs.item()
                    sum_accum_mse_error += accum_mse_abs.item()
                    num_subsets += 1
                
                if num_subsets == 0:
                    return 0.0
                mean_abs_error = sum_abs_error / num_subsets
                mean_mse_error = sum_mse_error / num_subsets
                mean_l1_error = sum_l1_error / num_subsets
                mean_last_l1_error = sum_last_l1_error / num_subsets
                mean_last_mse_error = sum_last_mse_error / num_subsets
                mean_accum_l1_error = sum_accum_l1_error / num_subsets
                mean_accum_mse_error = sum_accum_mse_error / num_subsets

                self.logger.record(f"{phase}_abs_error", mean_abs_error)
                self.logger.record(f"{phase}_mse_abs", mean_mse_error)
                self.logger.record(f"{phase}_l1_abs", mean_l1_error)
                self.logger.record(f"{phase}_last_l1_abs", mean_last_l1_error)
                self.logger.record(f"{phase}_last_mse_abs", mean_last_mse_error)
                self.logger.record(f"{phase}_accum_l1_abs", mean_accum_l1_error)
                self.logger.record(f"{phase}_accum_mse_abs", mean_accum_mse_error)
                return mean_accum_mse_error

            eval_mse_abs = 0.0
            for phase in ["train", "eval"]:
                acc_mse_abs = _evaluate_split(phase)
                if phase == "eval":
                    eval_mse_abs = acc_mse_abs

            self.total_epochs += 1
            self.logger.dump(self.total_epochs)

        return eval_mse_abs
