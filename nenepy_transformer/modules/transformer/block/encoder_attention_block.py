from torch import nn

from nenepy_transformer.modules.transformer import SelfAttention, PositionWiseFeedForwardNetwork, ResidualBlock, SelfMultiHeadAttention


class EncoderAttentionBlock(nn.Module):

    def __init__(self, n_embeddings, n_head, dropout_rate=0.1):
        """

        Args:
            n_embeddings (int):
            n_head (int):
            dropout_rate (float):

        """
        super(EncoderAttentionBlock, self).__init__()

        if n_head > 1:
            self._attention = SelfMultiHeadAttention(n_embeddings, n_head, dropout_rate)
        else:
            self._attention = SelfAttention(n_embeddings, dropout_rate)

        self._ffn = PositionWiseFeedForwardNetwork(n_embeddings)
        self._attention_dropout_norm = ResidualBlock(n_embeddings, dropout_rate)
        self._ffn_dropout_norm = ResidualBlock(n_embeddings, dropout_rate)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, input_tensor, attention_masks=None):
        attention_output, attention = self._attention(input_tensor, attention_masks)
        attention_output = self._attention_dropout_norm(attention_output, input_tensor)

        ffn_output = self._ffn(attention_output)
        ffn_output = self._ffn_dropout_norm(ffn_output, attention_output)

        return ffn_output, attention
