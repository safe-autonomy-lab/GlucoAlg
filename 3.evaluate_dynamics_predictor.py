import argparse
import os
import torch

from FunctionEncoder.Dataset.TransitionDataset import SequentialTransitionDataset
from FunctionEncoder.Model.Architecture.ITransformer import ITransformer
from FunctionEncoder.Model.Architecture.SimpleNODE import SimpleNeuralODEBaseline
from dynamics_utils import set_seed, load_aggregated_data
from dynamics_trainers import load_function_encoder
from FunctionEncoder.evaluation_utils import evaluate_function_encoder
from shield.util import load_config


def _get_save_folder(args):
    if args.model == "ba_node":
        return (
            f"saved_files/dynamics_predictor/ba_node_b{args.n_basis}_p"
            f"{args.prediction_length}_h{args.history_length}/{args.seed}"
        )
    return (
        f"saved_files/dynamics_predictor/{args.model}_p{args.prediction_length}_h"
        f"{args.history_length}/{args.seed}"
    )


def _sync_fe_args_from_config(args, save_folder):
    config_path = os.path.join(save_folder, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config at {config_path}")
    config = load_config(config_path)
    args.history_length = config["input_size"][0]
    args.prediction_length = config["model_kwargs"].get(
        "prediction_length", config["output_size"][0]
    )
    args.n_basis = config.get("n_basis", args.n_basis)
    return args


def _build_dataset(args, device, train_transitions, eval_transitions):
    if len(train_transitions["X"]) == 0:
        raise RuntimeError("No training data found for evaluation.")
    return SequentialTransitionDataset(
        train_transitions,
        eval_transitions,
        history_length=args.history_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        dtype=torch.float32,
        use_normalization=True,
        device=device,
    )


def _load_model(args, dataset, device, save_folder):
    if not args.load_model:
        raise RuntimeError("Evaluation requires --load_model to load a checkpoint.")
    save_path = os.path.join(save_folder, "best_model.pth")
    if args.model == "ba_node":
        model = load_function_encoder(save_folder, dataset)
        model = model.to(device)
        model.eval()
        return model
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Missing checkpoint at {save_path}")
    if args.model == "itf":
        model = ITransformer(
            input_len=args.history_length,
            pred_len=args.prediction_length,
            n_variates=dataset.n_variates,
            d_model=args.itf_d_model,
            n_heads=args.itf_n_heads,
            num_layers=args.itf_num_layers,
            ffn_hidden=args.itf_ffn_hidden,
        ).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        return model
    model = SimpleNeuralODEBaseline(
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
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--history_length", type=int, default=24)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--n_basis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_seed", type=int, default=None)
    parser.add_argument("--model", type=str, choices=["ba_node", "itf", "node"], default="ba_node")
    parser.add_argument("--itf_d_model", type=int, default=128)
    parser.add_argument("--itf_n_heads", type=int, default=4)
    parser.add_argument("--itf_num_layers", type=int, default=4)
    parser.add_argument("--itf_ffn_hidden", type=int, default=256)
    parser.add_argument("--node_encoder", type=str, choices=["none", "itransformer", "attention", "lstm"], default="itransformer")
    parser.add_argument("--node_encoder_hidden", type=int, default=128)
    parser.add_argument("--node_encoder_heads", type=int, default=4)
    parser.add_argument("--node_encoder_layers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    save_folder = _get_save_folder(args)
    if args.model == "ba_node" and args.load_model:
        args = _sync_fe_args_from_config(args, save_folder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_transitions, eval_transitions = load_aggregated_data()
    dataset = _build_dataset(args, device, train_transitions, eval_transitions)
    model = _load_model(args, dataset, device, save_folder)
    if args.model == "ba_node":
        args.prediction_length = model.prediction_horizon
        evaluate_function_encoder(
            model,
            dataset,
            train_transitions,
            args,
            save_folder,
            device,
            example_nbr=100,
            per_patient_samples=20,
        )

if __name__ == "__main__":
    main()
