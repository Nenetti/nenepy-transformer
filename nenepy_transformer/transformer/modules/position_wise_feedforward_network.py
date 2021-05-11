from torch import nn


class PositionWiseFeedForwardNetwork(nn.Module):

    def __init__(self, n_embeddings):
        """

        Args:
            n_embeddings (int):

        """
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self._layers = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_embeddings * 4, n_embeddings),
        )

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, x):
        return self._layers(x)
