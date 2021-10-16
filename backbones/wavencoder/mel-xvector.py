import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from backbones.wavencoder.stft import STFT
from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

class TacotronSTFT(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(sampling_rate, filter_length, 
                                   n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
    
    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output
    
class MelFilterbanks(nn.Module):
    """Computes mel-filterbanks."""
    def __init__(self, 
                 n_filters: int = 40,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 window_len: float = 25.,
                 window_stride: float = 10.,
                 min_freq: float = 60.0,
                 max_freq: float = 7800.0):
        """Constructor of a MelFilterbanks frontend.
            Args:
              n_filters: the number of mel_filters.
              sample_rate: sampling rate of input waveforms, in samples.
              n_fft: number of frequency bins of the spectrogram.
              window_len: size of the window, in seconds.
              window_stride: stride of the window, in seconds.
              compression_fn: a callable, the compression function to use.
              min_freq: minimum frequency spanned by mel-filters (in Hz).
              max_freq: maximum frequency spanned by mel-filters (in Hz).
              **kwargs: other arguments passed to the base class, e.g. name.
            """
        super().__init__()
        self._n_filters = n_filters
        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._window_len = int(sample_rate * window_len // 1000 + 1)
        self._window_stride = int(sample_rate * window_stride // 1000)
        self._min_freq = min_freq
        self._max_freq = max_freq if max_freq else sample_rate / 2.
        
        self.stft = TacotronSTFT(filter_length=self._window_len, hop_length=self._window_stride, 
                                 win_length=self._window_len, n_mel_channels=self._n_filters, 
                                 sampling_rate=self._sample_rate, mel_fmin=self._min_freq,
                                 mel_fmax=self._max_freq)
    def forward(self, inputs):
        """Computes mel-filterbanks of a batch of waveforms.
            Args:
              inputs: input audio of shape (batch_size, num_samples).
            Returns:
              Mel-filterbanks of shape (batch_size, time_frames, freq_bins).
        """
        mel_filterbanks = self.stft.mel_spectrogram(inputs)
        return mel_filterbanks

class architecture(nn.Module):
    def __init__(self, embed_dim=512):
        super(architecture, self).__init__()
        self.melfbank = MelFilterbanks(n_filters=30)

    def forward(self, x):
        out = self.melfbank(x)
        return out
    