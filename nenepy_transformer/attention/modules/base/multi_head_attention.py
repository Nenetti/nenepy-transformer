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

        memory_tensor = self._fit_shape(memory_tensor, input_tensor)

        query, key, value = self.forward_query_key_value(input_tensor, memory_tensor)
        query = self._to_multi_head_patch(query)
        key = self._to_multi_head_patch(key)
        value = self._to_multi_head_patch(value)

        attention_weight = torch.matmul(query, key.transpose(-2, -1))
        if attention_mask is not None:
            # print("A", attention_weight.shape, self._fit_attention_mask(attention_mask, attention_weight).shape)
            attention_weight += self._fit_shape(attention_mask, attention_weight)
            # attention_weight += attention_mask.unsqueeze(dim=1)
        attention_weight = torch.softmax(attention_weight * self._scale, dim=-1)
        attention_weight = self._dropout_layer(attention_weight)

        output = torch.matmul(attention_weight, value)
        output = self._concat_head(output)
        output = self._output_layer(output)

        return output, attention_weight

    @staticmethod
    def _fit_shape(source, target):
        if source.ndim == target.ndim:
            return source

        n_dim = target.ndim
        shape = [1] * n_dim
        shape[0] = target.shape[0]
        shape[-2] = target.shape[-2]
        shape[-1] = target.shape[-1]
        return source.view(shape)

    def _to_multi_head_patch(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        shape = list(x.shape)[:-1] + [self._n_heads, self._head_out_features]
        x = x.view(shape)
        return x.transpose(-2, -3)

    def _concat_head(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        x = x.transpose(-2, -3)
        shape = list(x.shape)[:-2] + [self._n_embeddings]
        return x.contiguous().view(shape)
