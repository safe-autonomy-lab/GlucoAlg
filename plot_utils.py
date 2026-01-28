import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import os
import logging
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FONT_SIZES = {
    "title": 12,
    "label": 10,
    "tick": 8,
    "legend": 8,
}

REGION_COLORS = {
    "hypo": "#9ecae1",
    "safe": "#c7e9c0",
    "hyper": "#fdd0a2",
}

PREDICTION_COLORS = {
    "pred": "#4C72B0",
    "true": "#000000",
}


def create_diabetes_animation(history: dict, episode: int, patient_type: str, algorithm: str, patient_name: str, 
                            save_seed: int, tir_pct: float, reward: float, cost: float, shield_type: str = 'none'):
    """Create an animated visualization of the diabetes episode."""
    try:
        # Convert history to DataFrame with time index
        df = pd.DataFrame(history)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Create output directory
        shield_str = "with_shield" if shield_type != 'none' else "no_shield"
        render_dir = f"./diabetes_evaluation/{patient_type}/{algorithm}/{patient_name}/seed{save_seed}/{shield_str}/animations"
        os.makedirs(render_dir, exist_ok=True)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{patient_type.upper()} Patient - {patient_name} - {algorithm} - Episode {episode}\n'
                    f'TIR: {tir_pct:.1f}% | Reward: {reward:.2f} | Cost: {cost:.2f}', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Glucose levels with zones
        axes[0, 0].plot(df.index, df['CGM'], 'b-', linewidth=2, label='CGM')
        axes[0, 0].axhspan(70, 180, alpha=0.3, color='green', label='Target Zone')
        axes[0, 0].axhspan(0, 70, alpha=0.3, color='red', label='Hypoglycemic')
        axes[0, 0].axhspan(180, 400, alpha=0.3, color='orange', label='Hyperglycemic')
        axes[0, 0].set_ylabel('Glucose (mg/dL)')
        axes[0, 0].set_title('Blood Glucose Levels')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 300])
        
        # Plot 2: Insulin delivery
        axes[0, 1].plot(df.index, df['insulin'], 'purple', linewidth=2, label='Insulin')
        axes[0, 1].set_ylabel('Insulin (U/min)')
        axes[0, 1].set_title('Insulin Delivery')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: IOB and COB
        axes[1, 0].plot(df.index, df['IOB'], 'orange', linewidth=2, label='IOB')
        axes[1, 0].plot(df.index, df['COB'], 'brown', linewidth=2, label='COB')
        axes[1, 0].set_ylabel('Units/Grams')
        axes[1, 0].set_title('Insulin/Carb On Board')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Risk indices
        axes[1, 1].plot(df.index, df['LBGI'], 'red', linewidth=2, label='Hypo Risk')
        axes[1, 1].plot(df.index, df['HBGI'], 'orange', linewidth=2, label='Hyper Risk')
        axes[1, 1].plot(df.index, df['Risk'], 'black', linewidth=2, label='Total Risk')
        axes[1, 1].set_ylabel('Risk Index')
        axes[1, 1].set_title('Glycemic Risk')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Rewards and costs
        axes[2, 0].plot(df.index, np.cumsum(df['reward']), 'green', linewidth=2, label='Cumulative Reward')
        axes[2, 0].plot(df.index, np.cumsum(df['cost']), 'red', linewidth=2, label='Cumulative Cost')
        axes[2, 0].set_ylabel('Cumulative Value')
        axes[2, 0].set_title('Rewards and Costs')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Actions
        axes[2, 1].plot(df.index, [a[0] for a in df['action']], 'blue', linewidth=2, label='Action 0')
        if len(df['action'].iloc[0]) > 1:
            axes[2, 1].plot(df.index, [a[1] for a in df['action']], 'red', linewidth=2, label='Action 1')
        if len(df['action'].iloc[0]) > 2:
            axes[2, 1].plot(df.index, [a[2] for a in df['action']], 'green', linewidth=2, label='Action 2')
        axes[2, 1].set_ylabel('Action Values')
        axes[2, 1].set_title('RL Agent Actions')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save as high-quality PNG
        png_filename = f"{render_dir}/episode_{episode}_TIR_{tir_pct:.1f}pct.png"
        plt.savefig(png_filename, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"  ✅ Saved episode plot: {png_filename}")
        
        # Save data as CSV for further analysis
        csv_filename = f"{render_dir}/episode_{episode}_data.csv"
        df.to_csv(csv_filename)
        logger.info(f"  ✅ Saved episode data: {csv_filename}")
        
        plt.close(fig)
        
        # Create a simple GIF showing glucose trajectory
        create_glucose_gif(df, episode, render_dir, tir_pct)
        
    except Exception as e:
        logger.error(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()

def create_glucose_gif(df: pd.DataFrame, episode: int, render_dir: str, tir_pct: float):
    """Create a simple animated GIF of glucose levels over time."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plot
        ax.set_xlim(df.index.min(), df.index.max())
        ax.set_ylim(0, 300)
        ax.axhspan(70, 180, alpha=0.3, color='green', label='Target Zone')
        ax.axhspan(0, 70, alpha=0.3, color='red', label='Hypoglycemic')
        ax.axhspan(180, 300, alpha=0.3, color='orange', label='Hyperglycemic')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title(f'Glucose Trajectory - Episode {episode} (TIR: {tir_pct:.1f}%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        line, = ax.plot([], [], 'b-', linewidth=3, label='CGM')
        point, = ax.plot([], [], 'ro', markersize=8)
        
        def animate(frame):
            # Show glucose up to current frame
            current_data = df.iloc[:frame+1]
            line.set_data(current_data.index, current_data['CGM'])
            
            # Show current point
            if frame < len(df):
                point.set_data([df.index[frame]], [df['CGM'].iloc[frame]])
            
            return line, point
        
        # Create animation (sample every 5th frame to reduce file size)
        frames = range(0, len(df), max(1, len(df) // 50))  # Max 50 frames
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, 
                                     blit=True, repeat=True)
        
        # Save as GIF
        gif_filename = f"{render_dir}/episode_{episode}_glucose_trajectory.gif"
        anim.save(gif_filename, writer='pillow', fps=5)
        logger.info(f"  ✅ Saved glucose GIF: {gif_filename}")
        
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating glucose GIF: {e}")
        import traceback
        traceback.print_exc()

def _add_background(ax: plt.Axes) -> None:
    ax.axhspan(0, 70, color=REGION_COLORS["hypo"], alpha=0.3, linewidth=0)
    ax.axhspan(70, 180, color=REGION_COLORS["safe"], alpha=0.3, linewidth=0)
    ax.axhspan(180, 400, color=REGION_COLORS["hyper"], alpha=0.3, linewidth=0)

def _parse_transition_key(key: str):
    parts = key.split("_")
    if len(parts) < 3:
        return None, None, None
    patient_id = parts[-1]
    cohort = parts[-2]
    diabetes_type = "_".join(parts[:-2])
    return diabetes_type, cohort, patient_id


def _plot_predictions(save_path, query_times, true_y, pred_y, title):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_outputs = true_y.shape[-1]
    fig, axes = plt.subplots(num_outputs, 1, figsize=(10, max(3, 2 * num_outputs)), sharex=True)
    if num_outputs == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(query_times, true_y[:, idx], label="True", color="tab:blue")
        ax.plot(query_times, pred_y[:, idx], label="Predicted", color="tab:orange", linestyle="--")
        ax.set_ylabel(f"Output {idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def evaluate_dataset(model, dataset, phase, prediction_horizon, save_dir):
    """Evaluate single or multi-step predictions and log sample plots."""
    assert phase in ("train", "eval")
    
    source_X = dataset.train_X if phase == "train" else dataset.eval_X
    source_Y = dataset.train_Y if phase == "train" else dataset.eval_Y

    device = dataset.device
    dtype = dataset.dtype
    
    save_dir = os.path.join(save_dir, phase)
    os.makedirs(save_dir, exist_ok=True)
    
    n_samples = source_X.shape[1]
    n_functions = source_X.shape[0]
    
    num_visualize = 3
    indices = np.random.choice(n_samples, num_visualize, replace=False) if n_samples > num_visualize else np.arange(n_samples)

    with torch.no_grad():
        # Select batch
        batch_X = torch.tensor(source_X[:, indices], dtype=dtype, device=device)
        # batch_Y: (N_funcs, Batch, PredictionLength, TargetDim)
        batch_Y = torch.tensor(source_Y[:, indices], dtype=dtype, device=device)
        
        # Prepare context
        bg_seq = batch_X[..., 0]
        diffs = bg_seq[..., 1:] - bg_seq[..., :-1]
        mean_deltas = diffs.mean(dim=-1, keepdim=True)
        # example_Y should match prediction horizon of the model input
        example_Y = mean_deltas.repeat(1, 1, prediction_horizon).cumsum(dim=-1)
        
        representation, _ = model.compute_representation(batch_X, example_Y)

        # Predict
        # prediction_horizon=1 means one block inference
        abs_preds, _ = model.predict(batch_X, representation, prediction_horizon=prediction_horizon, denormalize=True)
        # Check shapes
        if abs_preds.shape[2] != prediction_horizon:
             print(f"Warning: Model prediction horizon {abs_preds.shape[2]} does not match requested {prediction_horizon}")
        
        # Use actual prediction length for time axis
        actual_horizon = abs_preds.shape[2]
        time_axis = np.arange(1, actual_horizon + 1) * (5.0 / 60.0)

        # Denormalize Ground Truth for comparison
        mean = dataset.torch_series_mean_blood_glucose.to(device)
        std = dataset.torch_series_std_blood_glucose.to(device)
        
        # Ensure correct broadcasting
        # std: (N, 1, 1) or (N, 1, 1, 1)
        if std.ndim == 3: std = std.unsqueeze(1)
        if mean.ndim == 3: mean = mean.unsqueeze(1)

        # Reconstruct true absolute trajectory from deltas
        # batch_Y is normalized Delta.
        true_deltas = batch_Y * std
        
        # Get start point from history (Last observed BG)
        # batch_X shape: (N_funcs, Batch, History, Features)
        # Extract last normalized observation and expand dims to match (N, B, P, 1)
        last_bg_norm = batch_X[..., -1, :1].unsqueeze(-2) # (N, B, 1, 1)
        
        # Denormalize start
        start_bg = last_bg_norm * std + mean
        
        # True Absolute Trajectory
        true_abs = start_bg + true_deltas.cumsum(dim=-2)
        
        # Plotting
        for j, idx in enumerate(indices): 
            for f_idx in range(n_functions): 
                pred_traj = abs_preds[f_idx, j].cpu().numpy() # (P, D)
                true_traj = true_abs[f_idx, j].cpu().numpy()  # (P, D)

                # Ensure 2D (Time, Channels)
                if pred_traj.ndim == 1:
                    pred_traj = pred_traj[:, None]
                if true_traj.ndim == 1:
                    true_traj = true_traj[:, None]

                # Ensure time_axis matches
                if len(time_axis) != len(true_traj):
                     print(f"Shape mismatch: time_axis {len(time_axis)}, true_traj {len(true_traj)}")
                     # Adjust time axis if needed
                     time_axis_adj = np.arange(1, len(true_traj) + 1) * (5.0 / 60.0)
                else:
                     time_axis_adj = time_axis

                plot_name = f"p{f_idx}_s{idx}"
                plot_path = os.path.join(save_dir, f"{phase}_traj_{plot_name}_abs.png")
                
                _plot_predictions(
                    plot_path,
                    time_axis_adj,
                    true_traj,
                    pred_traj,
                    f"{phase.capitalize()} P{f_idx} Sample {idx} ({actual_horizon}-step)",
                )
    
    print(f"[{phase}] Generated visualization plots in {save_dir}")
    return {}

def plot_prediction_horizon(model, dataset, phase, prediction_horizon, save_dir, model_type):
    """Plot fixed-horizon prediction matching for FE/ITF/NODE models."""
    assert phase in ("train", "eval")
    assert model_type in ("fe", "ba_node", "fe_node", "itf", "node")

    device = dataset.device
    dtype = dataset.dtype

    save_dir = os.path.join(save_dir, phase)
    os.makedirs(save_dir, exist_ok=True)

    num_visualize = 3

    with torch.no_grad():
        mean = dataset.torch_series_mean_blood_glucose.to(device)
        std = dataset.torch_series_std_blood_glucose.to(device)
        if std.ndim == 3:
            std = std.unsqueeze(1)
        if mean.ndim == 3:
            mean = mean.unsqueeze(1)

        if model_type in ("fe", "ba_node", "fe_node"):
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample(phase=phase)
            n_samples = query_xs.shape[1]
            n_functions = query_xs.shape[0]
            if n_samples <= num_visualize:
                indices = np.arange(n_samples)
            else:
                indices = np.random.choice(n_samples, num_visualize, replace=False)

            example_ys_rep = example_ys
            if example_ys_rep.ndim == 4:
                example_ys_rep = example_ys_rep.squeeze(-1)
            representation, _ = model.compute_representation(
                example_xs, example_ys_rep
            )
            abs_preds, _ = model.predict(
                query_xs,
                representation,
                prediction_horizon=prediction_horizon,
                denormalize=True,
            )
            pred_abs = abs_preds

            last_bg_norm = query_xs[..., -1, :1].unsqueeze(-2)
            start_bg = last_bg_norm * std + mean
            true_abs = start_bg + (query_ys * std)
        else:
            source_X = dataset.train_X if phase == "train" else dataset.eval_X
            source_Y = dataset.train_Y if phase == "train" else dataset.eval_Y
            n_samples = source_X.shape[1]
            n_functions = source_X.shape[0]
            if n_samples <= num_visualize:
                indices = np.arange(n_samples)
            else:
                indices = np.random.choice(n_samples, num_visualize, replace=False)

            batch_X = torch.tensor(source_X[:, indices], dtype=dtype, device=device)
            batch_Y = torch.tensor(source_Y[:, indices], dtype=dtype, device=device)

            last_bg_norm = batch_X[..., -1, :1].unsqueeze(-2)
            start_bg = last_bg_norm * std + mean

            true_deltas = batch_Y * std
            true_abs = start_bg + true_deltas.cumsum(dim=-2)

            if model_type == "itf":
                preds = model(batch_X)
            else:
                preds = model(batch_X, pred_len=prediction_horizon)
            if preds.ndim == 3:
                preds = preds.unsqueeze(-1)
            if preds.shape[-1] > 1:
                preds = preds[..., :1]
            if model_type == "node":
                pred_abs = preds * std + mean
            else:
                pred_abs = start_bg + preds * std

        actual_horizon = pred_abs.shape[2]
        time_axis = np.arange(1, actual_horizon + 1) * (5.0 / 60.0)

        for j, idx in enumerate(indices):
            for f_idx in range(n_functions):
                pred_traj = pred_abs[f_idx, j].cpu().numpy()
                true_traj = true_abs[f_idx, j].cpu().numpy()

                if pred_traj.ndim == 1:
                    pred_traj = pred_traj[:, None]
                if true_traj.ndim == 1:
                    true_traj = true_traj[:, None]

                if len(time_axis) != len(true_traj):
                    time_axis_adj = np.arange(1, len(true_traj) + 1) * (5.0 / 60.0)
                else:
                    time_axis_adj = time_axis

                plot_name = f"p{f_idx}_s{idx}"
                plot_path = os.path.join(save_dir, f"{phase}_traj_{plot_name}_abs.png")
                _plot_predictions(
                    plot_path,
                    time_axis_adj,
                    true_traj,
                    pred_traj,
                    f"{phase.capitalize()} P{f_idx} Sample {idx} ({actual_horizon}-step)",
                )

    print(f"[{phase}] Generated prediction plots in {save_dir}")
    return {}

def plot_prediction_grid(
    model,
    dataset,
    keys_in_order,
    prediction_horizon,
    save_path,
    model_type,
    phase="eval",
    cohorts=None,
    diabetes_types=None,
    n_query_samples=10,
    n_patients_per_group=10,
    rng_seed=0,
    transitions=None,
):
    """Plot a 3x3 grid of mean predictions with std error bands per cohort/type."""
    assert phase in ("train", "eval")
    assert model_type in ("fe", "ba_node", "fe_node", "itf", "node")

    cohorts = cohorts or ["child", "adolescent", "adult"]
    diabetes_types = diabetes_types or ["t1d", "t2d", "t2d_no_pump"]

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "grid"])
    except Exception:
        pass

    device = dataset.device
    dtype = dataset.dtype

    mean = dataset.torch_series_mean_blood_glucose.to(device)
    std = dataset.torch_series_std_blood_glucose.to(device)
    if std.ndim == 3:
        std = std.unsqueeze(1)
    if mean.ndim == 3:
        mean = mean.unsqueeze(1)

    with torch.no_grad():
        rng = np.random.default_rng(rng_seed)
        history_length = dataset.history_length
        pred_len = prediction_horizon

        if transitions is not None:
            X_dict = transitions["X"]
            Y_dict = transitions["Y"]
            n_functions = len(keys_in_order)
            example_count = min(getattr(dataset, "n_examples", 50), 50)

            ex_xs_list = []
            ex_ys_list = []
            q_xs_list = []
            q_ys_list = []

            for idx, key in enumerate(keys_in_order):
                traj_X = np.asarray(X_dict[key])
                traj_Y = np.asarray(Y_dict[key])
                total_windows = traj_X.shape[0] - pred_len + 1
                if total_windows <= 0:
                    continue

                ex_count = min(example_count, total_windows)
                q_count = min(n_query_samples, total_windows)
                replace = total_windows < q_count
                q_idx = rng.choice(total_windows, size=q_count, replace=replace)

                def _build_window(k):
                    start_idx = k - history_length + 1
                    end_idx = k + 1
                    if start_idx < 0:
                        pad_len = abs(start_idx)
                        pad = np.tile(traj_X[0], (pad_len, 1))
                        x_window = np.concatenate([pad, traj_X[:end_idx]], axis=0)
                    else:
                        x_window = traj_X[start_idx:end_idx]

                    y_window = []
                    for j in range(pred_len):
                        idx_j = k + j
                        y_window.append(traj_Y[idx_j, 0])
                    y_window = np.array(y_window).reshape(-1, 1)
                    return x_window, y_window

                ex_x = []
                ex_y = []
                for k in range(ex_count):
                    x_w, y_w = _build_window(k)
                    ex_x.append(x_w)
                    ex_y.append(y_w)

                q_x = []
                q_y = []
                for k in q_idx:
                    x_w, y_w = _build_window(k)
                    q_x.append(x_w)
                    q_y.append(y_w)

                ex_x = np.stack(ex_x, axis=0)
                ex_y = np.stack(ex_y, axis=0)
                q_x = np.stack(q_x, axis=0)
                q_y = np.stack(q_y, axis=0)

                if dataset.use_normalization:
                    mean_np = dataset.series_mean[idx]
                    std_np = dataset.series_std[idx]
                    std_bg = dataset.series_std_blood_glucose[idx]
                    ex_x = (ex_x - mean_np) / std_np
                    q_x = (q_x - mean_np) / std_np
                    ex_y = ex_y / std_bg
                    q_y = q_y / std_bg

                ex_xs_list.append(ex_x)
                ex_ys_list.append(ex_y)
                q_xs_list.append(q_x)
                q_ys_list.append(q_y)

            example_xs = torch.tensor(np.stack(ex_xs_list, axis=0), dtype=dtype, device=device)
            example_ys = torch.tensor(np.stack(ex_ys_list, axis=0), dtype=dtype, device=device)
            query_xs = torch.tensor(np.stack(q_xs_list, axis=0), dtype=dtype, device=device)
            query_y = torch.tensor(np.stack(q_ys_list, axis=0), dtype=dtype, device=device)
        else:
            source_X = dataset.train_X if phase == "train" else dataset.eval_X
            source_Y = dataset.train_Y if phase == "train" else dataset.eval_Y
            n_samples = source_X.shape[1]
            n_functions = source_X.shape[0]
            example_count = min(getattr(dataset, "n_examples", 50), max(n_samples - 1, 1))

            example_xs = torch.tensor(source_X[:, :example_count], dtype=dtype, device=device)
            example_ys = torch.tensor(source_Y[:, :example_count], dtype=dtype, device=device)
            query_xs_list = []
            query_y_list = []
            for idx in range(n_functions):
                replace = n_samples < n_query_samples
                q_idx = rng.choice(n_samples, size=n_query_samples, replace=replace)
                query_xs_list.append(source_X[idx, q_idx])
                query_y_list.append(source_Y[idx, q_idx])
            query_xs = torch.tensor(np.stack(query_xs_list, axis=0), dtype=dtype, device=device)
            query_y = torch.tensor(np.stack(query_y_list, axis=0), dtype=dtype, device=device)
            print('shape of query_xs: ', query_xs.shape)
            print('shape of query_y: ', query_y.shape)
            
        
        last_bg_norm = query_xs[..., -1, :1].unsqueeze(-2)
        start_bg = last_bg_norm * std + mean
        true_abs = start_bg + (query_y * std).cumsum(dim=-2)

        if model_type in ("fe", "ba_node", "fe_node"):
            example_ys_rep = example_ys.cumsum(dim=-2)
            print('shape of example_ys_rep: ', example_ys_rep.shape)
            if example_ys_rep.ndim == 4:
                example_ys_rep = example_ys_rep.squeeze(-1)
            representation, _ = model.compute_representation(
                example_xs, example_ys_rep
            )
            pred_abs, _ = model.predict(
                query_xs,
                representation,
                prediction_horizon=prediction_horizon,
                denormalize=True,
            )

        else:
            if model_type == "itf":
                preds = model(query_xs)
            else:
                preds = model(query_xs, pred_len=prediction_horizon)
            if preds.ndim == 3:
                preds = preds.unsqueeze(-1)
            if preds.shape[-1] > 1:
                preds = preds[..., :1]
            if model_type == "node":
                pred_abs = preds * std + mean
            else:
                pred_abs = start_bg + preds * std

        if pred_abs.ndim == 3:
            pred_abs = pred_abs.unsqueeze(-1)
        if true_abs.ndim == 3:
            true_abs = true_abs.unsqueeze(-1)

    actual_horizon = pred_abs.shape[2]
    time_axis = np.arange(1, actual_horizon + 1) * 5.0

    group_indices = {(c, d): [] for c in cohorts for d in diabetes_types}
    for idx, key in enumerate(keys_in_order):
        diabetes_type, cohort, _ = _parse_transition_key(key)
        if diabetes_type in diabetes_types and cohort in cohorts:
            group_indices[(cohort, diabetes_type)].append(idx)

    # print("group_indices: ", group_indices)
    # exit()

    fig, axes = plt.subplots(
        nrows=len(cohorts),
        ncols=len(diabetes_types),
        figsize=(12, 9),
        sharey=False,
    )

    for row_idx, cohort in enumerate(cohorts):
        for col_idx, diabetes_type in enumerate(diabetes_types):
            ax = axes[row_idx, col_idx]
            indices = group_indices.get((cohort, diabetes_type), [])
            if not indices:
                ax.set_title(f"{diabetes_type} {cohort}", fontsize=FONT_SIZES["title"])
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=9,
                    transform=ax.transAxes,
                )
                ax.tick_params(labelsize=FONT_SIZES["tick"])
                continue

            if len(indices) > n_patients_per_group:
                indices = rng.choice(indices, size=n_patients_per_group, replace=False).tolist()
            print("indices: ", indices)
            pred_group = pred_abs[indices, :, :, 0].cpu().numpy()
            true_group = true_abs[indices, :, :, 0].cpu().numpy()
            print("shape of pred_group: ", pred_group.shape)
            print("shape of true_group: ", true_group.shape)
            # exit()
            pred_group = pred_group.reshape(-1, actual_horizon)
            true_group = true_group.reshape(-1, actual_horizon)
            err_group = pred_group - true_group

            mean_pred = pred_group.mean(axis=0)
            mean_true = true_group.mean(axis=0)
            sample_count = err_group.shape[0] * err_group.shape[1]
            std = err_group.std(axis=0, ddof=1) if sample_count > 1 else np.zeros_like(mean_pred)
            std_err = std / np.sqrt(max(sample_count, 1))

            ax.plot(
                time_axis,
                mean_true,
                label="True",
                color=PREDICTION_COLORS["true"],
                linewidth=1.4,
            )
            ax.plot(
                time_axis,
                mean_pred,
                label="Predicted",
                color=PREDICTION_COLORS["pred"],
                linewidth=1.6,
            )
            ax.fill_between(
                time_axis,
                mean_pred - std_err,
                mean_pred + std_err,
                color=PREDICTION_COLORS["pred"],
                alpha=0.2,
                linewidth=0,
            )

            ax.set_title(f"{diabetes_type} {cohort}", fontsize=FONT_SIZES["title"])
            y_min = mean_true.min() - 5.0
            y_max = mean_true.max() + 5.0
            ax.set_ylim(y_min, y_max)
            ax.tick_params(labelsize=FONT_SIZES["tick"])

            if row_idx == len(cohorts) - 1:
                ax.set_xlabel("Time (minutes)", fontsize=FONT_SIZES["label"])
            if col_idx == 0:
                ax.set_ylabel("Blood glucose (mg/dL)", fontsize=FONT_SIZES["label"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=FONT_SIZES["legend"])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[{phase}] Saved prediction grid to {save_path}")

def save_prediction_group_csvs(
    model,
    dataset,
    keys_in_order,
    prediction_horizon,
    output_dir,
    model_type,
    phase="eval",
    cohorts=None,
    diabetes_types=None,
    n_query_samples=10,
    n_patients_per_group=10,
    rng_seed=0,
    transitions=None,
):
    """Save per-group mean predictions and standard error as CSV files."""
    assert phase in ("train", "eval")
    assert model_type in ("fe", "ba_node", "fe_node", "itf", "node")

    cohorts = cohorts or ["child", "adolescent", "adult"]
    diabetes_types = diabetes_types or ["t1d", "t2d", "t2d_no_pump"]

    device = dataset.device
    dtype = dataset.dtype

    mean = dataset.torch_series_mean_blood_glucose.to(device)
    std = dataset.torch_series_std_blood_glucose.to(device)
    if std.ndim == 3:
        std = std.unsqueeze(1)
    if mean.ndim == 3:
        mean = mean.unsqueeze(1)

    with torch.no_grad():
        rng = np.random.default_rng(rng_seed)
        history_length = dataset.history_length
        pred_len = prediction_horizon

        if transitions is not None:
            X_dict = transitions["X"]
            Y_dict = transitions["Y"]
            n_functions = len(keys_in_order)
            example_count = min(getattr(dataset, "n_examples", 50), 50)

            ex_xs_list = []
            ex_ys_list = []
            q_xs_list = []
            q_ys_list = []

            for idx, key in enumerate(keys_in_order):
                traj_X = np.asarray(X_dict[key])
                traj_Y = np.asarray(Y_dict[key])
                total_windows = traj_X.shape[0] - pred_len + 1
                if total_windows <= 0:
                    continue

                ex_count = min(example_count, total_windows)
                q_count = min(n_query_samples, total_windows)
                replace = total_windows < q_count
                q_idx = rng.choice(total_windows, size=q_count, replace=replace)

                def _build_window(k):
                    start_idx = k - history_length + 1
                    end_idx = k + 1
                    if start_idx < 0:
                        pad_len = abs(start_idx)
                        pad = np.tile(traj_X[0], (pad_len, 1))
                        x_window = np.concatenate([pad, traj_X[:end_idx]], axis=0)
                    else:
                        x_window = traj_X[start_idx:end_idx]

                    y_window = []
                    for j in range(pred_len):
                        idx_j = k + j
                        y_window.append(traj_Y[idx_j, 0])
                    y_window = np.array(y_window).reshape(-1, 1)
                    return x_window, y_window

                ex_x = []
                ex_y = []
                for k in range(ex_count):
                    x_w, y_w = _build_window(k)
                    ex_x.append(x_w)
                    ex_y.append(y_w)

                q_x = []
                q_y = []
                for k in q_idx:
                    x_w, y_w = _build_window(k)
                    q_x.append(x_w)
                    q_y.append(y_w)

                ex_x = np.stack(ex_x, axis=0)
                ex_y = np.stack(ex_y, axis=0)
                q_x = np.stack(q_x, axis=0)
                q_y = np.stack(q_y, axis=0)

                if dataset.use_normalization:
                    mean_np = dataset.series_mean[idx]
                    std_np = dataset.series_std[idx]
                    std_bg = dataset.series_std_blood_glucose[idx]
                    ex_x = (ex_x - mean_np) / std_np
                    q_x = (q_x - mean_np) / std_np
                    ex_y = ex_y / std_bg
                    q_y = q_y / std_bg

                ex_xs_list.append(ex_x)
                ex_ys_list.append(ex_y)
                q_xs_list.append(q_x)
                q_ys_list.append(q_y)

            example_xs = torch.tensor(np.stack(ex_xs_list, axis=0), dtype=dtype, device=device)
            example_ys = torch.tensor(np.stack(ex_ys_list, axis=0), dtype=dtype, device=device)
            query_xs = torch.tensor(np.stack(q_xs_list, axis=0), dtype=dtype, device=device)
            query_y = torch.tensor(np.stack(q_ys_list, axis=0), dtype=dtype, device=device)
        else:
            source_X = dataset.train_X if phase == "train" else dataset.eval_X
            source_Y = dataset.train_Y if phase == "train" else dataset.eval_Y
            n_samples = source_X.shape[1]
            n_functions = source_X.shape[0]
            example_count = min(getattr(dataset, "n_examples", 50), max(n_samples - 1, 1))

            example_xs = torch.tensor(source_X[:, :example_count], dtype=dtype, device=device)
            example_ys = torch.tensor(source_Y[:, :example_count], dtype=dtype, device=device)
            query_xs_list = []
            query_y_list = []
            for idx in range(n_functions):
                replace = n_samples < n_query_samples
                q_idx = rng.choice(n_samples, size=n_query_samples, replace=replace)
                query_xs_list.append(source_X[idx, q_idx])
                query_y_list.append(source_Y[idx, q_idx])
            query_xs = torch.tensor(np.stack(query_xs_list, axis=0), dtype=dtype, device=device)
            query_y = torch.tensor(np.stack(query_y_list, axis=0), dtype=dtype, device=device)

        last_bg_norm = query_xs[..., -1, :1].unsqueeze(-2)
        start_bg = last_bg_norm * std + mean
        true_abs = start_bg + (query_y * std).cumsum(dim=-2)

        if model_type in ("fe", "ba_node", "fe_node"):
            example_ys_rep = example_ys.cumsum(dim=-2)
            if example_ys_rep.ndim == 4:
                example_ys_rep = example_ys_rep.squeeze(-1)
            representation, _ = model.compute_representation(
                example_xs, example_ys_rep
            )
            pred_abs, _ = model.predict(
                query_xs,
                representation,
                prediction_horizon=prediction_horizon,
                denormalize=True,
            )
        else:
            if model_type == "itf":
                preds = model(query_xs)
            else:
                preds = model(query_xs, pred_len=prediction_horizon)
            if preds.ndim == 3:
                preds = preds.unsqueeze(-1)
            if preds.shape[-1] > 1:
                preds = preds[..., :1]
            pred_abs = start_bg + preds * std

        if pred_abs.ndim == 3:
            pred_abs = pred_abs.unsqueeze(-1)
        if true_abs.ndim == 3:
            true_abs = true_abs.unsqueeze(-1)

    actual_horizon = pred_abs.shape[2]
    time_axis = np.arange(1, actual_horizon + 1) * 5.0

    group_indices = {(c, d): [] for c in cohorts for d in diabetes_types}
    for idx, key in enumerate(keys_in_order):
        diabetes_type, cohort, _ = _parse_transition_key(key)
        if diabetes_type in diabetes_types and cohort in cohorts:
            group_indices[(cohort, diabetes_type)].append(idx)

    os.makedirs(output_dir, exist_ok=True)
    for cohort in cohorts:
        for diabetes_type in diabetes_types:
            indices = group_indices.get((cohort, diabetes_type), [])
            if not indices:
                continue
            if len(indices) > n_patients_per_group:
                indices = rng.choice(indices, size=n_patients_per_group, replace=False).tolist()

            pred_group = pred_abs[indices, :, :, 0].cpu().numpy()
            true_group = true_abs[indices, :, :, 0].cpu().numpy()
            pred_group = pred_group.reshape(-1, actual_horizon)
            true_group = true_group.reshape(-1, actual_horizon)
            err_group = pred_group - true_group

            mean_pred = pred_group.mean(axis=0)
            mean_true = true_group.mean(axis=0)
            sample_count = err_group.shape[0]
            std = err_group.std(axis=0, ddof=1) if sample_count > 1 else np.zeros_like(mean_pred)
            stderr = std / np.sqrt(max(sample_count, 1))

            df = pd.DataFrame(
                {
                    "time_minutes": time_axis,
                    "mean_true": mean_true,
                    "mean_pred": mean_pred,
                    "stderr": stderr,
                    "n_samples": sample_count,
                    "cohort": cohort,
                    "diabetes_type": diabetes_type,
                }
            )
            out_path = os.path.join(output_dir, f"p{prediction_horizon}_{diabetes_type}_{cohort}.csv")
            df.to_csv(out_path, index=False)
    print(f"[{phase}] Saved prediction CSVs in {output_dir}")

def visualize_full_trajectories(model, transitions, history_length, prediction_horizon, dataset, save_dir, phase='eval', example_nbr: int = 100):
    """
    Visualizes full trajectories by sliding the window over the continuous data.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.function_encoder.parameters()).device
    pred_len = prediction_horizon if prediction_horizon else 1
    Y_dict = transitions['Y']
    X_dict = transitions['X']
    
    print(f"Visualizing {len(X_dict)} trajectories for {phase} with prediction horizon {pred_len}...")
    
    # Use sorted keys to align with dataset normalization params if they are per-patient
    sorted_keys = sorted(list(X_dict.keys()))

    for idx, key in enumerate(X_dict):
        traj_X = np.array(X_dict[key])[: 500]
        traj_Y = np.array(Y_dict[key])[: 500]
        if len(traj_X) <= history_length: continue
            
        windows = []
        valid_indices = []
        windows_y = []
        for t in range(history_length, len(traj_X) - pred_len + 1):
            window = traj_X[t - history_length : t]
            # Find correct patient index for normalization
            patient_idx = sorted_keys.index(key)
            
            # Create target window (Absolute BG)
            y_abs_window = []
            for j in range(pred_len):
                idx = t - 1 + j
                # val = traj_X[idx, 0] + traj_Y[idx, 0]
                val = traj_Y[idx, 0]
                y_abs_window.append(val)
                
            y_window = np.array(y_abs_window).reshape(-1, 1)

            if dataset.use_normalization:
                window = (window - dataset.series_mean[patient_idx, :, :]) / dataset.series_std[patient_idx, :, :]
                y_window = y_window / dataset.series_std_blood_glucose[patient_idx]
                
            windows.append(window)
            windows_y.append(y_window)
            valid_indices.append(t)
            
        if not windows: continue        
        
        # Create batch of all windows: (Batch, H, D) -> (1, Batch, H, D) for broadcast compatibility
        batch_X = torch.from_numpy(np.array(windows)).float().to(device).unsqueeze(0)
        batch_Y = torch.from_numpy(np.array(windows_y)).float().to(device).unsqueeze(0)
        # Compute example X and example Y
        with torch.no_grad():
            example_X = batch_X[:, :example_nbr]
            example_Y = batch_Y[:, :example_nbr].cumsum(dim=-2).squeeze(-1)
            representation, _ = model.compute_representation(example_X, example_Y)
            
        start_idx = example_nbr
        n_samples = batch_X.shape[1]
        chunk_size = 128
        pred_abs_list = []
        
        with torch.no_grad():
            for i in range(start_idx, n_samples, chunk_size):
                bx = batch_X[:, i: i + chunk_size]
                # Context heuristic
                # bg_seq = bx[..., 0]
                # mean_deltas = (bg_seq[..., 1:] - bg_seq[..., :-1]).mean(dim=-1, keepdim=True)
                # example_Y = mean_deltas.repeat(1, 1, prediction_horizon).cumsum(dim=-1) 

                # representation, _ = model.compute_representation(bx, example_Y)
                
                # Predict (denormalize=False to manual handle)
                abs_preds, _ = model.predict(bx, representation, prediction_horizon=prediction_horizon, denormalize=True, index=patient_idx)
                abs_preds = abs_preds[0]
                
                pred_abs_list.append(abs_preds.cpu().numpy())

        # (N_samples, Prediction Horizon, D)
        pred_abs = np.concatenate(pred_abs_list, axis=0) 
        
        # (TotalSamples, P)
        pred_1step = pred_abs[:, 0] 
        pred_Pstep = pred_abs[:, -1]
        
        true_bg_seq = traj_X[:, 0]
        dt_minutes = 5.0
        full_time_axis = np.arange(len(true_bg_seq)) * dt_minutes / 60.0
        time_1step = np.array(valid_indices[start_idx: ]) * dt_minutes / 60.0
        time_Pstep = (np.array(valid_indices[start_idx: ]) + pred_len - 1) * dt_minutes / 60.0
        
        plt.figure(figsize=(12, 6))
        plt.plot(full_time_axis, true_bg_seq, label='True BG', color='black', linewidth=1.5, alpha=0.8)
        plt.plot(time_1step, pred_1step, label='1-Step Ahead', color='darkorange', linestyle='--', linewidth=1.0, alpha=0.8)
        if prediction_horizon > 1:
            plt.plot(time_Pstep, pred_Pstep, label=f'{prediction_horizon}-Step Ahead', color='forestgreen', linestyle='--', linewidth=1.0, alpha=0.8)
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Blood Glucose (mg/dL)')
        plt.title(f'Trajectory {key} ({phase}) - Horizon {prediction_horizon}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'full_traj_{phase}_{key.replace("/", "_")}.png'), dpi=150)
        plt.close()
