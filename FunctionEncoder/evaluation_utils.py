import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_function_encoder(
    model,
    dataset,
    transitions,
    args,
    save_folder,
    device,
    example_nbr: int = 20,
    per_patient_samples: int = None,
):
    """
    Evaluate a BA-NODE checkpoint by computing and saving trajectory representations.
    Mirrors the protocol used in `run_function_encoder` when `--load_model` is set.
    """
    history_length = args.history_length
    pred_len = args.prediction_length
    X_dict = transitions.get("X", {})
    Y_dict = transitions.get("Y", {})

    if not X_dict:
        print("No trajectories provided for BA-NODE evaluation.")
        return

    patient_keys = sorted(list(X_dict.keys()))
    key_to_idx = {k: i for i, k in enumerate(patient_keys)}

    print(f"Evaluating BA-NODE on {len(patient_keys)} trajectories (prediction horizon {pred_len})...")
    representation_dict = {}
    rng = np.random.default_rng(0)

    for key in patient_keys:
        traj_X = np.array(X_dict[key])
        traj_Y = np.array(Y_dict.get(key, []))
        if len(traj_X) <= history_length:
            continue

        patient_idx = key_to_idx[key]
        windows = []
        windows_y = []

        for t in range(history_length, len(traj_X) - pred_len + 1):
            window = traj_X[t - history_length : t]

            # Build target window (delta BG over prediction horizon)
            y_abs_window = []
            for j in range(pred_len):
                idx = t - 1 + j
                y_abs_window.append(traj_Y[idx, 0])
            y_window = np.array(y_abs_window).reshape(-1, 1)

            if dataset.use_normalization:
                window = (window - dataset.series_mean[patient_idx, :, :]) / dataset.series_std[patient_idx, :, :]
                y_window = y_window / dataset.series_std_blood_glucose[patient_idx]

            windows.append(window)
            windows_y.append(y_window)

        if not windows:
            continue

        batch_X = torch.from_numpy(np.array(windows)).float().to(device).unsqueeze(0)
        batch_Y = torch.from_numpy(np.array(windows_y)).float().to(device).unsqueeze(0)

        with torch.no_grad():
            if per_patient_samples is not None and per_patient_samples > 1:
                total_windows = batch_X.shape[1]
                sample_count = min(per_patient_samples, total_windows)
                window_count = min(example_nbr, total_windows)
                if sample_count <= 0:
                    continue
                indices = np.linspace(0, total_windows - 1, sample_count, dtype=int)
                reps = []
                for _ in indices:
                    sample_idx = rng.choice(
                        total_windows,
                        size=window_count,
                        replace=total_windows < window_count,
                    )
                    example_X = batch_X[:, sample_idx]
                    example_Y = batch_Y[:, sample_idx].cumsum(dim=-2).squeeze(-1)
                    representation, _ = model.compute_representation(example_X, example_Y)
                    reps.append(representation.cpu().numpy().reshape(-1))
                if reps:
                    representation_dict[key] = np.stack(reps, axis=0)
            else:
                example_X = batch_X[:, :example_nbr]
                example_Y = batch_Y[:, :example_nbr].cumsum(dim=-2).squeeze(-1)
                representation, _ = model.compute_representation(example_X, example_Y)
                representation_dict[key] = representation.cpu().numpy()

    if not representation_dict:
        print("No valid windows found for BA-NODE evaluation.")
        return
    
    rep_path = os.path.join(save_folder, "representation_dict.npz")
    with open(rep_path, "wb") as f:
        np.savez(f, **representation_dict)
    print(f"Saved BA-NODE representations to {rep_path}")
    # Print evaluation summary similar to multistep results logging
    print("\nBA-NODE Evaluation Summary:")
    print(f"  Patients evaluated: {len(representation_dict)}")
    for k, rep in representation_dict.items():
        print(f"  {k}: representation shape {rep.shape}")
