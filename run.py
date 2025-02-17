import argparse
import torch

from quixer.setup_training import get_train_evaluate


##################################################
# Default hyperparameters for each of the models #
##################################################

quixer_hparams = {
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Quixer",
    "print_iter": 50,
}


lstm_hparams = {
    "layers": 2,
    "window": 32,
    "residuals": False,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.30,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "print_iter": 50,
}


fnet_hparams = {
    "layers": 2,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "FNet",
    "print_iter": 50,
}


transformer_hparams = {
    "layers": 1,
    "heads": 1,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Transformer",
    "print_iter": 50,
}

##################################################


# Embedding dimensions
classical_embedding_dimensions = [96, 128]
quantum_embedding_dimensions = [512]

# Dictionary defining available models along with associated hyperparameters
model_map = {
    "Quixer": (quixer_hparams, quantum_embedding_dimensions),
    "Transformer": (transformer_hparams, classical_embedding_dimensions),
    "LSTM": (lstm_hparams, classical_embedding_dimensions),
    "FNet": (fnet_hparams, classical_embedding_dimensions),
}
available_models = list(model_map.keys())

# Parse command line arguments
args = argparse.ArgumentParser(
    prog="Quixer", description="Runs the Quixer model and/or classical baselines"
)
args.add_argument(
    "-m",
    "--model",
    default="Quixer",
    choices=available_models,
    nargs="*",
    help="Model(s) to run.",
)
args.add_argument("-d", "--device", default="cpu", help="Device to run training on.")
parsed = args.parse_args()

device_name = parsed.device
models_to_run = parsed.model if type(parsed.model) is list else [parsed.model]

# Make algorithms deterministic for reproducibility
torch.backends.cudnn.deterministic = True


device = torch.device(device_name)
print(f"Running on device: {device}")

train_evaluate = get_train_evaluate(device)


for model_name in models_to_run:
    hyperparameters, embedding_dimensions = model_map[model_name]
    for embedding_dimension in embedding_dimensions:
        for seed in torch.randint(high=1000000, size=(10,)).tolist():
            hyperparameters["model"] = model_name
            hyperparameters["dimension"] = embedding_dimension
            hyperparameters["seed"] = seed

            train_evaluate(hyperparameters)
