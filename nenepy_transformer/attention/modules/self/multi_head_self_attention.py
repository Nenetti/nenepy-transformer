from nenepy_transformer.attention.modules.base import MultiHeadAttention


class MultiHeadSelfAttention(MultiHeadAttention):

    def forward(self, input_tensor, attention_mask=None):
        return super(MultiHeadSelfAttention, self).forward(input_tensor, input_tensor, attention_mask)
