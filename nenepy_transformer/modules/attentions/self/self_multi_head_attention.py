from nenepy_transformer.modules.attentions.base import MultiHeadAttention


class SelfMultiHeadAttention(MultiHeadAttention):

    def forward(self, input_tensor, attention_mask=None):
        return super(SelfMultiHeadAttention, self).forward(input_tensor, input_tensor, attention_mask)
