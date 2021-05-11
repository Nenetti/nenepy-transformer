from nenepy_transformer.attention.blocks import DecoderAttentionBlock
from nenepy_transformer.transformer.modules import WordEmbedding, PositionalEncoding
from torch import nn


class TransformerDecoder(nn.Module):

    def __init__(self, n_attention_block, n_head, embedding_dim, dropout_rate=0.1, padding_idx=None):
        super(TransformerDecoder, self).__init__()
        self._self_attentions = nn.ModuleList(
            [DecoderAttentionBlock(embedding_dim, n_head, dropout_rate) for _ in range(n_attention_block)]
        )

        self._n_attention_block = n_attention_block

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, x, memories, attention_masks=None):
        output = x

        outputs = [None] * self._n_attention_block
        attentions = [None] * self._n_attention_block
        for i, block in enumerate(self._self_attentions):
            output, attention = block(output, memories[i], attention_masks)
            outputs[i] = output
            attentions[i] = attention

        return outputs, attentions
