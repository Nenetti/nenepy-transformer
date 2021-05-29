from torch import nn


class PositionWiseFeedForwardNetwork(nn.Module):

    def __init__(self, n_embeddings):
        """

        Args:
            n_embeddings (int):

        """
        super(PositionWiseFeedForwardNetwork, self).__init__()
        self._layers = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_embeddings * 4, n_embeddings, bias=False),
        )

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, x):
        return self._layers(x)
