import torch

# https://www.acceluniverse.com/blog/developers/2019/08/attention.html
# https://qiita.com/halhorn/items/c91497522be27bde17ce
from nenepy_transformer.attention.modules.base import SingleHeadAttention


class MultiHeadAttention(SingleHeadAttention):

    def __init__(self, n_embeddings, n_head, dropout_rate=0.1):
        """
        
        Args:
            n_embeddings (int):
            n_head (int):
            dropout_rate (float):

        """
        super().__init__(n_embeddings, dropout_rate)

        self._n_heads = n_head
        self._head_in_features = n_embeddings // n_head
        self._head_out_features = n_embeddings // n_head

    def forward(self, input_tensor, memory_tensor, attention_mask=None):
        """

        Args:
            input_tensor (torch.Tensor):
            memory_tensor (torch.Tensor)
            attention_mask (None or torch.Tensor)

        Returns:
            (torch.Tensor, torch.Tensor):

        """
        query, key, value = self.forward_query_key_value(input_tensor, memory_tensor)

        query = self._to_multi_head_patch(query)
        key = self._to_multi_head_patch(key)
        value = self._to_multi_head_patch(value)

        attention_weight = torch.matmul(query, key.transpose(-2, -1))
        if attention_mask is not None:
            attention_weight += attention_mask.unsqueeze(dim=1)

        attention_weight = torch.softmax(attention_weight * self._scale, dim=-1)
        attention_weight = self._dropout_layer(attention_weight)

        output = torch.matmul(attention_weight, value)
        output = self._concat_head(output)
        output = self._output_layer(output)

        return output, attention_weight

    def _to_multi_head_patch(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        B, W, _ = x.shape
        x = x.view(B, W, self._n_heads, self._head_out_features)
        return x.permute(0, 2, 1, 3)

    def _concat_head(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        x = x.permute(0, 2, 1, 3)
        B, W, _, _ = x.shape
        return x.contiguous().view(B, W, self._n_embeddings)
