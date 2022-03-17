import torch.nn.functional as F
import torch.nn as nn

import math

from model_component import downsample2, upsample2

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

class Demucs(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.chin = 1
        self.chout = 1
        self.stride = 4
        self.hidden = args.hidden
        self.depth = args.depth
        self.kernel_size = args.kernel_size
        self.causal = True
        self.floor = 1e-3
        self.resample = 4
        self.growth = 2
        self.max_hidden = 10000
        self.normalize = True
        self.sample_rate = 16000
        self.glu = nn.GLU(1)
        ch_scale = 2
        causal = True
        rescale = 0.1

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(self.chin, self.hidden, self.kernel_size, self.stride),
                nn.ReLU(),
                nn.Conv1d(self.hidden, self.hidden * ch_scale, 1), self.glu,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(self.hidden, ch_scale * self.hidden, 1), self.glu,
                nn.ConvTranspose1d(self.hidden, self.chout, self.kernel_size, self.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            self.chout = self.hidden
            self.chin = self.hidden
            self.hidden = min(int(self.growth * self.hidden), self.max_hidden)

        self.lstm = BLSTM(self.chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x
