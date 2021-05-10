from nenepy_transformer.modules.attentions.base import SingleHeadAttention


class SelfAttention(SingleHeadAttention):

    def forward(self, input_tensor, attention_mask=None):
        return super(SelfAttention, self).forward(input_tensor, input_tensor, attention_mask)
