import torch
from nenepy_transformer.attention.utils import AttentionMask
from torch import nn


class WordEmbedding(nn.Module):

    def __init__(self, n_words, embedding_dim, padding_idx=None):
        super(WordEmbedding, self).__init__()
        self._embedding = nn.Embedding(num_embeddings=n_words, embedding_dim=embedding_dim, padding_idx=padding_idx)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward(self, batch_words):
        """

        Args:
            batch_words (list[torch.Tensor]):

        Returns:

        """
        if isinstance(batch_words, (tuple, list)):
            max_length = max([len(words) for words in batch_words])
            batch_words = torch.stack([self.shape_arrange(words, max_length) for words in batch_words], dim=0)

        embeddings = self._embedding(batch_words)
        attention_masks = AttentionMask().generate_disable_next_step_and_padding(batch_words)
        return embeddings, attention_masks

    @staticmethod
    def shape_arrange(words, length):
        t = torch.zeros(length, dtype=words.dtype, device=words.device)
        t[:len(words)] = words
        return t
