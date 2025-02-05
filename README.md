# Quixer

This is the code repository for the paper [Quixer: A Quantum Transformer Model](https://arxiv.org/abs/2406.04305). It contains the implementation of the classical simulation of Quixer, along with [Transformer](https://arxiv.org/abs/1706.03762), [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) and [FNet](https://arxiv.org/abs/2105.03824) baselines.

## Installation

> [!IMPORTANT]
> It is recommended that you first install the version of `torch` 2.2.2 suitable to your platform by following the instructions [here](https://pytorch.org/get-started/previous-versions/#v222), or the command below may not work.
> When installing using the above link, add `torchtext==0.17.1` to the installation command. For example
> ``` conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 torchtext==0.17.1 cpuonly -c pytorch ```

```
pip install -e .
```

## Running the models

On CPU:
```
python3 run.py -d cpu -m Quixer Transformer LSTM FNet
```

On Nvidia GPU:
```
python3 run.py -d cuda -m Quixer Transformer LSTM FNet
```

You can exclude any of the models in the commands above and it will not be run.