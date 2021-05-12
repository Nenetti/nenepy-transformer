import torch


class AttentionMask:

    def __init__(self, min_inf=-1e9):
        self._min_inf = min_inf

    def generate_disable_padding(self, batch_words):
        attention_masks = self._generate_disable_padding(batch_words).type(torch.float32)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    def generate_disable_next_step(self, batch_words):
        attention_masks = self._generate_disable_next_step(batch_words)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    def generate_disable_next_step_and_padding(self, batch_words, padding_id=0):
        attention_masks_a = self._generate_disable_padding(batch_words, padding_id)
        attention_masks_b = self._generate_disable_next_step(batch_words)
        attention_masks = (attention_masks_a + attention_masks_b.type(torch.bool)).type(torch.float32)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    @staticmethod
    def _generate_disable_padding(batch_words, padding_id=0):
        if padding_id == 0:
            attention_masks_1d = torch.logical_not(batch_words.type(torch.bool))
        else:
            attention_masks_1d = (batch_words != padding_id)

        W = batch_words.shape[-1]
        shapes_a = [-1] * (batch_words.ndim + 1)
        shapes_b = [-1] * (batch_words.ndim + 1)
        shapes_a[-2] = W
        shapes_b[-1] = W
        attention_masks_1d_a = attention_masks_1d.unsqueeze(-2).expand(shapes_a)
        attention_masks_1d_b = attention_masks_1d.unsqueeze(-1).expand(shapes_b)
        attention_masks = (attention_masks_1d_a + attention_masks_1d_b)
        return attention_masks

    @staticmethod
    def _generate_disable_next_step(batch_words):
        B, W = batch_words.shape[0], batch_words.shape[-1]
        attention_masks = torch.triu(torch.ones(W, W), diagonal=1)

        shapes = [-1] * (batch_words.ndim + 1)
        shapes[0] = B
        attention_masks = attention_masks.unsqueeze(0).expand(shapes)
        return attention_masks.to(batch_words.device)
