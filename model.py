import itertools
from math import log2

import torch
import torchquantum as tq


def sim14_encoder(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i + 1) % n_wires]}
                   for i in range(n_wires - 1, -1, -1)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i - 1) % n_wires]}
                    for i in [n_wires - 1] + list(range(n_wires - 1))])
    return enc


def evaluate_polynomial_state(base_states, unitary_params, enc, qdev, n_qbs, lcu_coeffs, poly_coeffs):
    acc = poly_coeffs[0] * base_states
    working_register = base_states

    for c in poly_coeffs[1:]:
        working_register = apply_unitaries(working_register, unitary_params, enc, qdev, n_qbs, lcu_coeffs)
        acc = acc + c * working_register

    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


def apply_unitaries(base_states, unitary_params, enc, qdev, n_qbs, coeffs):

    repeated_base = base_states.repeat(1, unitary_params.shape[1]).view(-1, 2 ** n_qbs)
    qdev.set_states(repeated_base)

    enc(qdev, unitary_params.view(-1, unitary_params.shape[-1]))

    states = qdev.get_states_1d().view(*unitary_params.shape[:2], 2 ** n_qbs)

    lcs = torch.einsum('bwi,bw->bi', states, coeffs)

    return lcs


class Quixer(torch.nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_words: int,
                 degree: int,
                 n_ansatz_layers: int,
                 vocab_size: int,
                 embedding_dim: int,
                 dropout: float,
                 device):
        """
        n_qubits: int
            Number of qubits per word.
        n_words: int
            Context length.
        degree: int
            Degree of polynomial.
        n_ansatz_layers: int
            Number of layers of circ 14.
        vocab_size: int
            Number of words in vocab. Used for embedding.
        embedding_dim: int
            Size of embedding vector for each word, before angles.
        dropout: float
            Dropout rate.
        device:
            Torch device.
        """

        super().__init__()

        self.n_words = n_words
        self.n_qubits = n_qubits

        assert degree > 0
        self.degree = degree
        self.device = device

        assert n_words != 0
        self.n_ctrl_qubits = int(log2(n_words))

        # Sim14 spec
        self.n_rots = 4 * n_qubits * n_ansatz_layers

        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(vocab_size,
                                            self.embedding_dim)

        torch.nn.init.xavier_uniform_(self.embedding.weight)

        self.emb2rot = torch.nn.Linear(in_features=self.embedding_dim,
                                       out_features=self.n_rots)

        self.dropout = torch.nn.Dropout(dropout)
        self.rot_sigm = torch.nn.Sigmoid()

        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

        # Preparation of word unitaries
        self.word_qencoder = tq.GeneralEncoder(sim14_encoder(n_qubits, n_ansatz_layers))
        self.word_qencoder.n_wires = self.n_qubits

        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = torch.nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = torch.nn.Parameter(torch.rand(self.n_words, dtype=torch.complex64))

        self.qff = tq.GeneralEncoder(sim14_encoder(n_qubits))
        self.qff_params = torch.nn.Parameter(torch.rand(self.n_rots))

        self.measure_all_xyz = tq.MeasureMultipleTimes(
            [{'wires': range(n_qubits), 'observables': ['x'] * n_qubits},
             {'wires': range(n_qubits), 'observables': ['y'] * n_qubits},
             {'wires': range(n_qubits), 'observables': ['z'] * n_qubits}])

        self.n_measures = 3 * n_qubits

        self.output_ff = torch.nn.Sequential(
            torch.nn.Linear(self.n_measures, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, vocab_size),
        )

    def forward(self, x):

        # [bsz, n_words]
        bsz = x.shape[0]

        mix_coeffs = self.mix_coeffs.repeat(bsz, 1)
        mix_coeffs = torch.nn.functional.normalize(mix_coeffs, p=1)
        # [bsz, n_words]

        x = self.embedding(x)
        # [bsz, n_words, embedding_dim]

        word_params = self.emb2rot(self.dropout(x))
        # [bsz, n_words, n_rots]

        base_states = torch.zeros(bsz, 2 ** self.n_qubits, dtype=torch.complex64, device=self.device)
        base_states[:, 0] = 1.0
        mixed_word = evaluate_polynomial_state(base_states,
                                               word_params,
                                               self.word_qencoder,
                                               self.q_device,
                                               self.n_qubits,
                                               mix_coeffs,
                                               self.poly_coeffs)

        # [bsz, 2 ** n_qbs]

        final_probs = torch.linalg.vector_norm(mixed_word, dim=-1)

        self.q_device.set_states(torch.nn.functional.normalize(mixed_word, dim=-1))
        self.qff(self.q_device, self.qff_params.repeat(1, bsz))

        exps = self.measure_all_xyz(self.q_device)
        exps = exps.reshape(3, bsz, self.n_qubits).moveaxis(0,1).reshape(bsz, -1)

        op = self.output_ff(exps)
        return op, torch.mean(final_probs)
