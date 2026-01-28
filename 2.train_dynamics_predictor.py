import argparse
import os
import torch
from FunctionEncoder.Dataset.TransitionDataset import SequentialTransitionDataset
from shield.util import load_config
from dynamics_utils import set_seed, load_aggregated_data
from dynamics_trainers import run_ba_node, run_itransformer, run_neural_ode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--history_length", type=int, default=24)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--n_basis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, choices=["ba_node", "itf", "node"], default="ba_node", help="Model to train: ba_node (BA-NODE), itf (ITransformer baseline), node (Neural ODE baseline).")
    parser.add_argument("--itf_d_model", type=int, default=128)
    parser.add_argument("--itf_n_heads", type=int, default=4)
    parser.add_argument("--itf_num_layers", type=int, default=4)
    parser.add_argument("--itf_ffn_hidden", type=int, default=256)
    parser.add_argument("--itf_lr", type=float, default=1e-3)
    parser.add_argument("--node_encoder", type=str, choices=["none", "itransformer", "attention", "lstm"], default="itransformer")
    parser.add_argument("--node_encoder_hidden", type=int, default=128)
    parser.add_argument("--node_encoder_heads", type=int, default=4)
    parser.add_argument("--node_encoder_layers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model == "ba_node":
        save_folder = f"saved_files/dynamics_predictor/ba_node_b{args.n_basis}_p{args.prediction_length}_h{args.history_length}/{args.seed}"
    else:
        save_folder = f"saved_files/dynamics_predictor/{args.model}_p{args.prediction_length}_h{args.history_length}/{args.seed}"
    config_path = f"{save_folder}/config.yaml"
    save_path = f"{save_folder}/model.pth"
    os.makedirs(save_folder, exist_ok=True)

    # Sync args with saved config when loading (FunctionEncoder only)
    if args.load_model and args.model in ("ba_node"):
        if not os.path.exists(config_path) or not os.path.exists(save_path):
            raise FileNotFoundError(f"Missing checkpoint or config in {save_folder}")
        loaded_config = load_config(config_path)
        args.history_length = loaded_config["input_size"][0]
        args.prediction_length = loaded_config["model_kwargs"].get(
            "prediction_length", loaded_config["output_size"][0]
        )
        args.n_basis = loaded_config.get("n_basis", args.n_basis)

    # This is order for normalization, eaach patient has a different mean blood glucose and std
    # We normalize using those values
    # The order is 
    # t1d: adult, child, adolescent: indexes from 0 to 29
    # t2d: adult, child, adolescent: indexes from 30 to 59
    # t2d_no_pump: adult, child, adolescent: indexes from 60 to 89
    train_transitions, eval_transitions = load_aggregated_data()

    if len(train_transitions['X']) == 0:
        print("No training data found. Exiting.")
        exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SequentialTransitionDataset(
        train_transitions,
        eval_transitions,
        history_length=args.history_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        dtype=torch.float32,
        use_normalization=True,
        device=device,
    )

    if args.model == "ba_node":
        model = run_ba_node(args, dataset, device, save_folder)
    elif args.model == "itf":
        model = run_itransformer(args, dataset, device, save_folder)
    else:
        model = run_neural_ode(args, dataset, device, save_folder)
