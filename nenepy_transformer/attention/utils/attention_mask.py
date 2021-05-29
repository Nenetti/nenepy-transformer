import torch


class AttentionMask:

    def __init__(self, min_inf=-1e9, padding_id=0, device="cpu"):
        self._min_inf = min_inf
        self._device = device
        self._padding_id = padding_id

    def generate_disable_next_step(self, batch_size, size):
        attention_masks = self._generate_disable_next_step(batch_size, size, torch.float32, self._device)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    def generate_disable_padding(self, batch_words):
        attention_masks = self._generate_disable_padding(batch_words).type(torch.float32)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    def generate_disable_next_step_and_padding(self, batch_words):
        padding_masks = self._generate_disable_padding(batch_words, self._padding_id)
        next_step_masks = self._generate_disable_next_step(batch_words.shape[0], batch_words.shape[1], torch.bool, self._device)
        attention_masks = (padding_masks + next_step_masks).type(torch.float32)
        attention_masks = (attention_masks * self._min_inf)
        return attention_masks

    @staticmethod
    def _generate_disable_padding(batch_words, padding_id=0):
        paddings = (batch_words == padding_id)
        shape_a = paddings.unsqueeze(-2)
        shape_b = paddings.unsqueeze(-1)
        return shape_a + shape_b

    @staticmethod
    def _generate_disable_next_step(batch_size, size, dtype, device):
        """

        Args:
            size (int):
            device (str):
            dtype (dtype):

        Returns:
            torch.Tensor

        """
        x = torch.triu(torch.ones((size, size), dtype=dtype, device=device), diagonal=1)
        return x.unsqueeze(0).expand(batch_size, -1, -1)
