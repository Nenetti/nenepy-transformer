import math

import torch
from torch import nn


class SingleHeadAttention(nn.Module):

    def __init__(self, n_embeddings, dropout_rate=0.1):
        """
        
        Args:
            n_embeddings (int):
            dropout_rate (float):

        """
        super().__init__()
        self._query_layer = nn.Linear(n_embeddings, n_embeddings, bias=False)
        self._key_value_layer = nn.Linear(n_embeddings, n_embeddings * 2, bias=False)
        self._output_layer = nn.Linear(n_embeddings, n_embeddings, bias=False)
        self._dropout_layer = nn.Dropout(p=dropout_rate)
        # self._norm_layer = nn.LayerNorm(out_features)

        self._n_embeddings = n_embeddings
        self._scale = math.sqrt(n_embeddings)
        # self._min_inf = torch.finfo(torch.float16).min
        # self._min_inf = -1e9

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
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

        attention_weight = torch.matmul(query, key.transpose(-2, -1))
        if attention_mask is not None:
            attention_weight += attention_mask

        attention_weight = torch.softmax(attention_weight * self._scale, dim=-1)
        attention_weight = self._dropout_layer(attention_weight)

        output = torch.matmul(attention_weight, value)
        output = self._output_layer(output)

        return output, attention_weight

    def forward_query_key_value(self, input_tensor, memory_tensor):
        """
        Args:
            input_tensor (torch.Tensor):
            memory_tensor (torch.Tensor):

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):

        """
        query = self._query_layer(input_tensor)
        kv = self._key_value_layer(memory_tensor)
        shape = list(kv.shape)
        shape[-1] = self._n_embeddings
        key, value = kv.view(2, *shape)

        return query, key, value
