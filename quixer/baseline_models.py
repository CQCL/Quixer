import math
import torch


class PositionalEncoding(torch.nn.Module):
    # Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hid_dim: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float,
    ):

        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        encoder_layers = torch.nn.TransformerEncoderLayer(
            emb_dim, n_heads, hid_dim, dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device
        )

        x = self.transformer_encoder(x, mask=causal_mask)

        # Only return output of the final word
        return self.project(x[:, -1, :]), None


class FNet(torch.nn.Module):

    def __init__(
        self, emb_dim: int, hid_dim: int, n_layers: int, vocab_size: int, dropout: float
    ):

        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, dropout)

        self.dropout = torch.nn.Dropout(dropout)

        self.ffs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    self.dropout,
                    torch.nn.Linear(emb_dim, hid_dim),
                    torch.nn.ReLU(),
                    self.dropout,
                    torch.nn.Linear(hid_dim, emb_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.ln1s = torch.nn.ModuleList(
            [torch.nn.LayerNorm([emb_dim]) for _ in range(n_layers)]
        )
        self.ln2s = torch.nn.ModuleList(
            [torch.nn.LayerNorm([emb_dim]) for _ in range(n_layers)]
        )

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.pos_enc(self.embedding(x))

        for ff, ln1, ln2 in zip(self.ffs, self.ln1s, self.ln2s):
            x = ln1(torch.fft.fft2(x).real + x)
            x = ln2(ff(x) + x)

        # Only return output of the final word
        return self.project(x[:, -1, :]), None


class LSTM(torch.nn.Module):
    def __init__(
        self, emb_dim: int, hid_dim: int, n_layers: int, vocab_size: int, dropout: float
    ):

        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)

        self.rnn = torch.nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.project = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        rnn_op, _ = self.rnn(x)

        # Only return output of the final word
        return self.project(rnn_op[:, -1, :]), None
