from torch import nn

from nenepy_transformer.attention.modules.self import MultiHeadSelfAttention, SingleHeadSelfAttention
from nenepy_transformer.attention.modules.source_target import SourceTargetMultiHeadAttention, SourceTargetSingleHeadAttention
from nenepy_transformer.transformer.modules import PositionWiseFeedForwardNetwork, ResidualBlock


class DecoderAttentionBlock(nn.Module):

    def __init__(self, n_embeddings, n_head, dropout_rate=0.1, depth=-1):
        """

        Args:
            n_embeddings (int):
            n_head (int):
            dropout_rate (float):

        """
        super(DecoderAttentionBlock, self).__init__()
        if n_head > 1:
            self._self_attention = MultiHeadSelfAttention(n_embeddings, n_head, dropout_rate, depth)
            self._attention = SourceTargetMultiHeadAttention(n_embeddings, n_head, dropout_rate, depth)
        else:
            self._self_attention = SingleHeadSelfAttention(n_embeddings, dropout_rate)
            self._attention = SourceTargetSingleHeadAttention(n_embeddings, dropout_rate)

        self._ffn = PositionWiseFeedForwardNetwork(n_embeddings)
        self._self_attention_dropout_norm = ResidualBlock(n_embeddings, dropout_rate)
        self._attention_dropout_norm = ResidualBlock(n_embeddings, dropout_rate)
        self._ffn_dropout_norm = ResidualBlock(n_embeddings, dropout_rate)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, input_tensor, memory_tensor, self_attention_masks=None, source_target_attention_mask=None):
        self_attention_output, _ = self._self_attention(input_tensor, self_attention_masks)
        self_attention_output = self._self_attention_dropout_norm(self_attention_output, input_tensor)

        source_target_attention_output, attention = self._attention(self_attention_output, memory_tensor, source_target_attention_mask)
        source_target_attention_output = self._attention_dropout_norm(source_target_attention_output, input_tensor)

        ffn_output = self._ffn(source_target_attention_output)
        ffn_output = self._ffn_dropout_norm(ffn_output, source_target_attention_output)

        return ffn_output, attention
