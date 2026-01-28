import os
import torch
from tqdm import trange
from shield.util import save_config, load_model
from FunctionEncoder.Callbacks.LoggerCallback import LoggerCallback
from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from FunctionEncoder.Wrapper.TimeSeriesWrapper import TimeSeriesWrapper
from FunctionEncoder.Model.Architecture.ITransformer import ITransformer
from FunctionEncoder.Model.Architecture.SimpleNODE import SimpleNeuralODEBaseline



def run_ba_node(args, dataset, device, save_folder):
    config_path = f"{save_folder}/config.yaml"
    save_path = f"{save_folder}/model.pth"
    save_best_path = f"{save_folder}/best_model.pth"

    if not args.load_model:
        config = {
            "input_size": (args.history_length, dataset.n_variates),
            "output_size": (args.prediction_length * dataset.train_Y.shape[-1],), 
            "data_type": "deterministic",
            "n_basis": args.n_basis, 
            "model_type": "BA_NODE",
            "method": "least_squares",
            "use_residuals_method": False, 
            "model_kwargs": {
                "d_model": 64,
                "n_heads": 4,
                "hidden_size": 128,
                "num_layers": 2,
                "ffn_hidden": 128,
                "encoder_type": "itransformer",
                # "encoder_type": "attention",
                "encoder_kwargs": {
                    "history_length": args.history_length,
                    "num_heads": 4,
                    "num_layers": 2
                },
                "ode_state_size": 64,
                "prediction_length": args.prediction_length,
                "activation": "silu",
            },  
            "device": device,
            "optimizer_kwargs": {"lr": 1e-4}
        }

        if os.path.exists(save_path):
            print(f"Overwriting existing checkpoint and config in {save_folder}")

        save_config(config, config_path)

        function_encoder = FunctionEncoder(**config).to(device)
        
        model = TimeSeriesWrapper(
            function_encoder,
            feature_dim=dataset.n_variates,
            history_length=args.history_length,
            normalization_mean=dataset.series_mean_blood_glucose,
            normalization_std=dataset.series_std_blood_glucose,
        )
        fe_params = sum(p.numel() for p in model.function_encoder.parameters() if p.requires_grad)
        print(f"FunctionEncoder trainable params: {fe_params}")

        print("Starting training with BA-NODE pipeline...")
        model_label = "ba_node" if args.model == "ba_node" else args.model
        fe_callback = LoggerCallback(
            model.function_encoder,
            dataset,
            logdir=os.path.join(save_folder, "logs"),
        )
        model.train_model(
            dataset,
            epochs=int(args.epochs),
            batch_size=args.batch_size,
            callback=fe_callback,
            save_best_path=save_best_path,
        )

        # Save final model
        model.function_encoder.save(save_path)
        # Save periodic checkpoints every 100 epochs if the logger recorded them
        if hasattr(fe_callback, "total_epochs"):
            for e in range(100, int(args.epochs) + 1, 100):
                ckpt_path = os.path.join(save_folder, f"model_epoch{e}.pth")
                model.function_encoder.save(ckpt_path)
        model.function_encoder.eval()
    
    else:
        model = load_function_encoder(save_folder, dataset)
        print(f"Loaded BA-NODE from {save_path}")
        return model
        

def load_function_encoder(save_folder, dataset):
    function_encoder = load_model(save_folder)
    loaded_hist_len, loaded_feature_dim = function_encoder.input_size

    # Ensure dataset matches the loaded model
    assert loaded_feature_dim == dataset.n_variates, (
        f"Dataset features ({dataset.n_variates}) do not match checkpoint ({loaded_feature_dim})"
    )
    
    model = TimeSeriesWrapper(
        function_encoder, 
        feature_dim=loaded_feature_dim, 
        history_length=loaded_hist_len,
        normalization_mean=dataset.series_mean_blood_glucose,
        normalization_std=dataset.series_std_blood_glucose
    )
    model.function_encoder.eval()
    return model
    
    
def run_function_encoder(args, dataset, device, save_folder):
    config_path = f"{save_folder}/config.yaml"
    save_path = f"{save_folder}/model.pth"
    save_best_path = f"{save_folder}/best_model.pth"

    if not args.load_model:
        config = {
            "input_size": (args.history_length, dataset.n_variates),
            "output_size": (args.prediction_length * dataset.train_Y.shape[-1],),
            "data_type": "deterministic",
            "n_basis": 5,
            "model_type": "NeuralODE",
            "method": "least_squares",
            "use_residuals_method": False,
            "model_kwargs": {
                "hidden_size": 128,
                "n_layers": 4,
                "ode_state_size": 64,
                "prediction_length": args.prediction_length,
                "activation": "silu",
                "encoder_type": "itransformer",
                "encoder_kwargs": {
                    "history_length": args.history_length,
                    "num_heads": 4,
                    "num_layers": 2,
                },
            },
            "device": device,
            "optimizer_kwargs": {"lr": 1e-4},
        }

        if os.path.exists(save_path):
            print(f"Overwriting existing checkpoint and config in {save_folder}")

        save_config(config, config_path)

        function_encoder = FunctionEncoder(**config).to(device)
        model = TimeSeriesWrapper(
            function_encoder,
            feature_dim=dataset.n_variates,
            history_length=args.history_length,
            normalization_mean=dataset.series_mean_blood_glucose,
            normalization_std=dataset.series_std_blood_glucose,
        )
        fe_params = sum(p.numel() for p in model.function_encoder.parameters() if p.requires_grad)
        print(f"FunctionEncoder trainable params (no ensemble): {fe_params}")

        print("Starting training with FE-NODE (no ensemble)...")
        fe_callback = LoggerCallback(
            model.function_encoder,
            dataset,
            logdir=os.path.join(save_folder, "logs"),
        )
        model.train_model(
            dataset,
            epochs=int(args.epochs),
            batch_size=args.batch_size,
            callback=fe_callback,
            save_best_path=save_best_path,
        )

        model.function_encoder.save(save_path)
        model.function_encoder.eval()
        return model

    model = load_function_encoder(save_folder, dataset)
    print(f"Loaded FE-NODE (no ensemble) from {save_path}")
    return model



def run_itransformer(args, dataset, device, save_folder):
    save_path = f"{save_folder}/model.pth"
    save_best_path = f"{save_folder}/best_model.pth"
    itf = ITransformer(
        input_len=args.history_length,
        pred_len=args.prediction_length,
        n_variates=dataset.n_variates,
        d_model=args.itf_d_model,
        n_heads=args.itf_n_heads,
        num_layers=args.itf_num_layers,
        ffn_hidden=args.itf_ffn_hidden,
    ).to(device)
    
    if args.load_model:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Missing checkpoint at {save_path}")
        itf.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded ITransformer baseline from {save_path}")
        return itf
    else:
        callback = LoggerCallback(
            itf,
            dataset,
            logdir=os.path.join(save_folder, "logs"),
        )
        optimizer = torch.optim.Adam(itf.parameters(), lr=args.itf_lr)
        mse = torch.nn.MSELoss()

        print("Starting training with ITransformer baseline...")
        total_epochs = int(args.epochs)
        iterator = trange(total_epochs, desc="Epochs")
        for epoch_idx in iterator:
            if hasattr(dataset, "reset_batch_state"):
                dataset.reset_batch_state()
            total_seqs = dataset.train_X.shape[1]
            n_batches = max(1, (total_seqs + args.batch_size - 1) // args.batch_size)
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                dataset.n_sequences = min(args.batch_size, total_seqs - batch_idx * args.batch_size)
                if dataset.n_sequences <= 0:
                    break
                _, _, q_xs, q_ys, _ = dataset.sample()
                preds = itf(q_xs)
                target = q_ys.squeeze(-1)  # (N, B, P)
                
                pred_bg = preds[..., 0]    # use BG channel
                
                loss = mse(pred_bg, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                
            if callback:
                eval_error = callback.on_step({})
                if eval_error is not None and eval_error < getattr(run_itransformer, "_best_eval", float("inf")):
                    run_itransformer._best_eval = eval_error
                    torch.save(itf.state_dict(), save_best_path)
                    print(f"Saved best ITransformer to {save_best_path} (eval={eval_error:.6f})")
            iterator.set_postfix({'loss': f'{epoch_loss / n_batches:.6f}'})

            # Periodic checkpoint
            if (epoch_idx + 1) % 100 == 0:
                ckpt_path = os.path.join(save_folder, f"model_epoch{epoch_idx + 1}.pth")
                torch.save(itf.state_dict(), ckpt_path)

        torch.save(itf.state_dict(), save_path)
        print(f"Saved ITransformer baseline to {save_path}")
    # Parameter report for fairness
    itf_params = itf.parameter_count()
    print(f"ITransformer params: {itf_params}")

def run_neural_ode(args, dataset, device, save_folder):
    save_path = f"{save_folder}/model.pth"
    save_best_path = f"{save_folder}/best_model.pth"
    node = SimpleNeuralODEBaseline(
        history_length=args.history_length,
        feature_dim=dataset.n_variates,
        pred_len=args.prediction_length,
        hidden=128,
        encoder_type=None if args.node_encoder == "none" else args.node_encoder,
        encoder_kwargs={
            "history_length": args.history_length,
            "num_heads": args.node_encoder_heads,
            "num_layers": args.node_encoder_layers,
            "hidden_size": args.node_encoder_hidden,
        } if args.node_encoder != "none" else {},
    ).to(device)
    
    if args.load_model:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Missing checkpoint at {save_path}")
        node.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded Neural ODE baseline from {save_path}")
        return node
    else:
        callback = LoggerCallback(
            node,
            dataset,
            logdir=os.path.join(save_folder, "logs"),
        )
        optimizer = torch.optim.Adam(node.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()
        print("Starting training with Neural ODE baseline...")
        total_epochs = int(args.epochs)
        iterator = trange(total_epochs, desc="Epochs")
        for epoch_idx in iterator:
            if hasattr(dataset, "reset_batch_state"):
                dataset.reset_batch_state()
            total_seqs = dataset.train_X.shape[1]
            n_batches = max(1, (total_seqs + args.batch_size - 1) // args.batch_size)
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                dataset.n_sequences = min(args.batch_size, total_seqs - batch_idx * args.batch_size)
                if dataset.n_sequences <= 0:
                    break
                _, _, q_xs, q_ys, _ = dataset.sample()
                preds = node(q_xs)          # (N, B, P)
                target = q_ys.squeeze(-1)    # (N, B, P)
                loss = mse(preds, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            if callback:
                eval_error = callback.on_step({})
                if eval_error is not None and eval_error < getattr(run_neural_ode, "_best_eval", float("inf")):
                    run_neural_ode._best_eval = eval_error
                    torch.save(node.state_dict(), save_best_path)
                    print(f"Saved best Neural ODE to {save_best_path} (eval={eval_error:.6f})")
            iterator.set_postfix({'loss': f'{epoch_loss / n_batches:.6f}'})
            # Periodic checkpoint
            if (epoch_idx + 1) % 100 == 0:
                ckpt_path = os.path.join(save_folder, f"model_epoch{epoch_idx + 1}.pth")
                torch.save(node.state_dict(), ckpt_path)
        torch.save(node.state_dict(), save_path)
        print(f"Saved Neural ODE baseline to {save_path}")
    print(f"Neural ODE params: {node.parameter_count()}")
