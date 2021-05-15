import matplotlib.pyplot as plt
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, n_features, max_sentence_length=100):
        super(PositionalEncoding, self).__init__()
        self._n_features = n_features
        self._max_sentence_length = max_sentence_length

        pe = self.generate_positional_embedding(n_features, max_sentence_length)
        self.register_buffer("_pe", pe)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, x):
        B, W, N = x.shape
        if W > self._max_sentence_length:
            pe = self.generate_positional_embedding(self._n_features, W)
            self.register_buffer("_pe", pe.to(x.device))

        return x + self._pe[:, :W]

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @staticmethod
    def generate_positional_embedding(n_embedding, max_sentence_length):
        p = torch.arange(max_sentence_length).unsqueeze(1)
        sin_angles = p / (10000 ** ((2 * torch.arange(0, n_embedding // 2)) / n_embedding))
        cos_angles = p / (10000 ** ((2 * torch.arange(0, n_embedding // 2) + 1) / n_embedding))
        sin = torch.sin(sin_angles)
        cos = torch.cos(cos_angles)
        return torch.stack([sin, cos], 2).view(max_sentence_length, n_embedding).unsqueeze(0)

    @staticmethod
    def plot(positional_embedding):
        B, W, N = positional_embedding.shape
        pe = positional_embedding[0].view(W, N // 2, 2)
        sin, cos = pe[:, :, 0], pe[:, :, 1]
        sin = torch.stack([sin, torch.full_like(sin, fill_value=sin.min())], 2).view(W, N)
        cos = torch.stack([torch.full_like(cos, fill_value=cos.min()), cos], 2).view(W, N)
        sin_cos = positional_embedding[0]

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.set_title("sin")
        ax2.set_title("cos")
        ax3.set_title("Concat(sin cos)")
        axes = [ax1, ax2, ax3]
        ax1.imshow(sin, cmap='viridis')
        ax2.imshow(cos, cmap='viridis')
        im = ax3.imshow(sin_cos, cmap='viridis')

        for ax in axes:
            ax.set_xlabel('Embedding Vector Position')
            ax.set_ylabel('Word Position')
            ax.ylabel = 'Word Position'
            ax.set_aspect("equal")

        plt.colorbar(im)
        plt.show()
