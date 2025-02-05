import random
import os
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable

import numpy as np

import torch
from torch.types import Device
from torch.nn.modules.loss import _Loss
import torchtext

from quixer.quixer_model import Quixer
from quixer.baseline_models import Transformer, LSTM, FNet

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def batchify_s2s(
    data: torch.Tensor,
    tokens_per_batch: int,
    window_size: int,
    pad_token: int,
    device: Device,
) -> torch.Tensor:
    nr_of_batches = (data.size(0) - 1) // tokens_per_batch
    batched_data = (
        data[: nr_of_batches * tokens_per_batch].view(tokens_per_batch, nr_of_batches).T
    )

    # Take last `window_size` elements for all but the last batch
    window_data = torch.cat(
        (
            torch.full((window_size, 1), pad_token, device=device),
            batched_data[-window_size:, :-1],
        ),
        dim=1,
    )

    return torch.cat((window_data, batched_data))


def init_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device, batch_size: int, window_size: int
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    # Download / load dataset

    raw_dset = load_dataset("ptb_text_only")

    train_iter = raw_dset["train"].data[0]
    train_iter = [s.as_py() for s in train_iter]

    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), specials=["<pad>", "<unk>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    PAD_TOK = vocab["<pad>"]

    def data_process(raw_text_iter) -> torch.Tensor:
        """Converts raw text into a flat Tensor."""
        data = [
            torch.tensor(vocab(tokenizer(item)) + [vocab["eos"]], dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 1, data))).to(device)

    # Convert from arrow array to native list
    train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
    val_sents = [s.as_py() for s in raw_dset["validation"].data[0]]
    test_sents = [s.as_py() for s in raw_dset["test"].data[0]]

    # Flatten datasets into one long tokenised string each
    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)

    # Prepare (x, y) pairs for batches
    train_iter = batchify_s2s(
        train_flat, batch_size * window_size, window_size, PAD_TOK, device
    )
    val_iter = batchify_s2s(
        val_flat, batch_size * window_size, window_size, PAD_TOK, device
    )
    test_iter = batchify_s2s(
        test_flat, batch_size * window_size, window_size, PAD_TOK, device
    )

    return vocab, (train_iter, val_iter, test_iter), PAD_TOK


def get_batch_s2s(source: torch.Tensor, i: int, window_size: int, *args):
    return source[i : i + window_size].T, source[i + window_size]


def create_model(
    hyperparams: dict[str, Any], device: Device, vocab_size: int
) -> torch.nn.Module:
    model_str = hyperparams["model"]
    model: torch.nn.Module
    if model_str == "Quixer":
        model = Quixer(
            n_qubits=hyperparams["qubits"],
            n_words=hyperparams["window"],
            degree=hyperparams["layers"],
            n_ansatz_layers=hyperparams["ansatz_layers"],
            vocab_size=vocab_size,
            embedding_dim=hyperparams["dimension"],
            dropout=hyperparams["dropout"],
            batch_size=hyperparams["batch_size"],
            device=device,
        )
    elif model_str == "FNet":
        model = FNet(
            vocab_size=vocab_size,
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            dropout=hyperparams["dropout"],
        )
    elif model_str == "Transformer":
        model = Transformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocab_size,
            dropout=hyperparams["dropout"],
        )
    elif model_str == "LSTM":
        model = LSTM(
            emb_dim=hyperparams["dimension"],
            hid_dim=hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            vocab_size=vocab_size,
            dropout=hyperparams["dropout"],
        )
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: _Loss,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    window_size: int,
    device: Device,
    batch_size: int,
):
    model.train()

    epoch_loss = 0

    n_batches = iterator.shape[0] - window_size

    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size, device, batch_size)
        optimizer.zero_grad()

        yhat, norm_avg = model(x)

        loss = loss_function(yhat, y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    loss_function: _Loss,
    window_size: int,
    device: Device,
    batch_size: int,
) -> float:
    model.eval()

    epoch_loss = 0

    n_batches = iterator.shape[0] - window_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(iterator, batch_idx, window_size, device, batch_size)

            yhat, _ = model(x)

            loss = loss_function(yhat, y)

            epoch_loss += loss.item()

    return epoch_loss / n_batches


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    device: Device,
    train_iter: torch.Tensor,
    val_iter: torch.Tensor,
    test_iter: torch.Tensor
) -> float:

    checkpoint_fpath = f"./trained_models/q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )

    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    loss_function = torch.nn.CrossEntropyLoss()

    def _evaluate(iter: torch.Tensor):
        return evaluate(
            model,
            iter,
            loss_function,
            hyperparams["window"],
            device,
            hyperparams["batch_size"],
        )

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            loss_function,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["window"],
            device,
            hyperparams["batch_size"],
        )

        valid_loss = _evaluate(val_iter)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {math.exp(train_loss)}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")

    model.load_state_dict(torch.load(checkpoint_fpath))

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss)}")

    return test_loss


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and then compute an evaluation metric.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        init_weights(model)

        model = model.to(device)

        valid_loss = train_cycle(
            model,
            parameterization,
            device,
            train_iter,
            val_iter,
            test_iter
        )

        return valid_loss

    return train_evaluate
