from torch import nn

from nenepy_transformer.modules.transformer import WordEmbedding, PositionalEncoding, DecoderAttentionBlock


class TransformerDecoder(nn.Module):

    def __init__(self, n_attention_block, n_head, n_words, n_embedding_dim, max_sentence_length, dropout_rate=0.1, padding_idx=None):
        super(TransformerDecoder, self).__init__()
        self._word_embedding = WordEmbedding(n_words, n_embedding_dim, padding_idx)
        self._positional_encoding = PositionalEncoding(n_embedding_dim, max_sentence_length)
        self._self_attentions = nn.ModuleList([DecoderAttentionBlock(n_embedding_dim, n_head, dropout_rate) for _ in range(n_attention_block)])

        self._n_attention_block = n_attention_block

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, x):
        embeddings, attention_masks = self._word_embedding(x)
        embeddings = self._positional_encoding(embeddings)
        output = embeddings

        outputs = [None] * self._n_attention_block
        attentions = [None] * self._n_attention_block
        for i, block in enumerate(self._self_attentions):
            # 仮設定
            output, attention = block(output, output, attention_masks)
            outputs[i] = output
            attentions[i] = attention

        return outputs, attentions
