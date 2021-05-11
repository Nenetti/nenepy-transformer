from nenepy_transformer.attention.modules.base import SingleHeadAttention


class SingleHeadSelfAttention(SingleHeadAttention):

    def forward(self, input_tensor, attention_mask=None):
        return super(SingleHeadSelfAttention, self).forward(input_tensor, input_tensor, attention_mask)
