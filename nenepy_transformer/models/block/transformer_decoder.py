from nenepy_transformer.attention.blocks import DecoderAttentionBlock
from torch import nn


class TransformerDecoder(nn.Module):

    def __init__(self, n_attention_block, n_head, embedding_dim, dropout_rate=0.1, padding_idx=None):
        super(TransformerDecoder, self).__init__()
        self._self_attentions = nn.ModuleList(
            [DecoderAttentionBlock(embedding_dim, n_head, dropout_rate, i) for i in range(n_attention_block)]
        )

        self._n_attention_block = n_attention_block

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, inputs, memories, self_attention_masks=None, source_target_attention_masks=None):
        output = inputs

        if source_target_attention_masks is None:
            source_target_attention_masks = [None] * self._n_attention_block
        else:
            source_target_attention_masks = source_target_attention_masks.detach()

        outputs = []
        attentions = []
        for i, block in enumerate(self._self_attentions):
            output, attention = block(output, memories[i], self_attention_masks, source_target_attention_masks[i])
            outputs.append(output)
            attentions.append(attention)

        return outputs, attentions
