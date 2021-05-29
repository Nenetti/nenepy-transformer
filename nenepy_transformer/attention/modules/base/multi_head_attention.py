import torch

# https://www.acceluniverse.com/blog/developers/2019/08/attention.html
# https://qiita.com/halhorn/items/c91497522be27bde17ce
from nenepy_transformer.attention.modules.base import SingleHeadAttention


class MultiHeadAttention(SingleHeadAttention):

    def __init__(self, n_embeddings, n_head, dropout_rate=0.1, depth=-1):
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
        self._depth = depth

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

        query, key, value = self._to_query_key_value(input_tensor, memory_tensor)
        query, key, value = self._to_multi_head_patches(query, key, value)

        attention_weight = torch.matmul(query, key.transpose(-2, -1))
        if attention_mask is not None:
            attention_weight += self._fit_shape(attention_mask, attention_weight)

        attention_weight = torch.softmax(attention_weight * self._scale, dim=-1)
        attention_weight = self._dropout_layer(attention_weight)
        output = torch.matmul(attention_weight, value)
        output = self._concat_head(output)
        output = self._output_layer(output)
        return output, attention_weight

    # def forward_self_regression(self, input_tensor, memory_tensor, attention_mask=None, past_qkv=None):
    #     memory_tensor = self._fit_shape(memory_tensor, input_tensor)
    #
    #     query, key, value = self._to_query_key_value(input_tensor, memory_tensor)
    #     query, key, value = self._to_multi_head_patches(query, key, value)
    #
    #     if past_qkv is not None:
    #         past_q, past_k, past_v = past_qkv
    #         query = torch.cat([past_q, query], dim=-2)
    #         key = torch.cat([past_k, key], dim=-2)
    #         value = torch.cat([past_v, value], dim=-2)
    #
    #     attention_weight = torch.matmul(query, key.transpose(-2, -1))
    #     if attention_mask is not None:
    #         attention_weight += self._fit_shape(attention_mask, attention_weight)
    #         # print(self.training)
    #         # attention_weight += attention_mask.unsqueeze(dim=1)
    #     attention_weight = torch.softmax(attention_weight * self._scale, dim=-1)
    #     attention_weight = self._dropout_layer(attention_weight)
    #     output = torch.matmul(attention_weight, value)
    #     output = self._concat_head(output)
    #     # if not self.training:
    #     #     print(output.shape, output)
    #     output = self._output_layer(output)
    #     if past_qkv is not None:
    #         output = output[:, -1:]
    #     return output, attention_weight, (query, key, value)

    @staticmethod
    def _fit_shape(source, target):
        if source.ndim == target.ndim:
            return source

        n_dim = target.ndim
        # shapes1 = [1] * (n_dim - 2) + [target.shape[-2], target.shape[-1]]
        # shapes2 = list(target.shape[:-2]) + [-1, -1]
        shape = [1] * n_dim
        shape[0] = target.shape[0]
        shape[-2] = target.shape[-2]
        shape[-1] = target.shape[-1]
        return source.view(shape)
        # print(shapes1, shapes2, source.shape)
        # return source.view(shapes1).expand(shapes2)

    def _to_multi_head_patches(self, query, key, value):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        q = self._to_multi_head_patch(query)
        k = self._to_multi_head_patch(key)
        v = self._to_multi_head_patch(value)
        return q, k, v

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
