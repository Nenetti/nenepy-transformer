from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, n_embeddings, dropout_rate=0.1):
        """

        Args:
            n_embeddings (int):
            dropout_rate (float):

        """
        super(ResidualBlock, self).__init__()
        self._layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(n_embeddings),
        )

    def forward(self, x, skip_x):
        x = self._layers(x)
        return x + skip_x
