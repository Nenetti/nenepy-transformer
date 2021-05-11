import numpy as np
import torch
from nenepy_summary import TorchSummary

from nenepy_transformer.models.block import TransformerDecoder

# model = Attention(in_features=100, out_features=768)

# model = MultiHeadAttention(in_features=100, out_features=768, n_head=4)
# model = TransformerEncoder(n_attention_block=2, n_head=3, n_words=10000, n_embedding_dim=384, max_sentence_length=1000, dropout_rate=0.1, padding_idx=None)

model = TransformerDecoder(n_attention_block=2, n_head=1, n_words=10000, embedding_dim=384, max_sentence_length=1000, dropout_rate=0.1, padding_idx=None)
# print(model)
# sys.exit()

summary = TorchSummary(model, batch_size=5)
words = torch.zeros([1, 6], dtype=torch.int32).cuda()
# model(words)
words = [[torch.from_numpy(np.array([1, 2, 3, 4, 0])), torch.from_numpy(np.array([3, 4, 5]))]]
summary.forward_tensor(words)
# x = torch.rand([5, 10, 100]).cuda()
# attention_mask = torch.zeros([5, 10, 10]).cuda()
#
# summary.forward_tensor([x, attention_mask])
# model(x, attention_mask)
# with Timer() as t:
#     for i in range(1000):
#         model(x, x, attention_mask)
