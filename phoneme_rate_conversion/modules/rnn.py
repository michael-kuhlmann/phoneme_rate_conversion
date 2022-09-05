import torch
from einops import rearrange

import padertorch as pt


class RCNN1d(pt.Module):

    def __init__(
        self, rnn=None, cnn=None, linear=None, reverse=False,
    ):
        super().__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.linear = linear

        self.reverse = reverse

    def conv(self, x, sequence_lengths):
        if self.cnn is None:
            return x, sequence_lengths
        h = rearrange(x, 'b t f -> b f t')
        h, seq_len_h = self.cnn(h, sequence_lengths)
        h = rearrange(h, 'b f t -> b t f')
        return h, seq_len_h

    def forward(
        self, x, sequence_lengths=None, segment_starts=None, segment_stops=None
    ):
        assert x.ndim == 3, x.shape
        h = x
        if self.reverse:
            h = torch.flip(x, (1,))
        if self.rnn is not None:
            h, _ = self.rnn(h)
        if self.linear is not None:
            h = self.linear(h)
        h, seq_len_h = self.conv(h, sequence_lengths)
        return h, seq_len_h