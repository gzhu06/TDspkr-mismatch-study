# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division

import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.fft as fft
from backbones.wavencoder.frontend_utils import norm_block, Fp32GroupNorm, DSTCNBlock, ConvDS

EPSILON = torch.finfo(torch.float32).eps
def window(window_type, N):
    def hanning(n):
        return 0.5*(1 - np.cos(2 * np.pi * (n - 1) / (N - 1)))

    def hamming(n):
        return 0.54 - 0.46 * np.cos(2 * np.pi * (n - 1) / (N - 1))

    if window_type == 'hanning':
        return np.asarray([hanning(n) for n in range(N)])
    else:
        return np.asarray([hamming(n) for n in range(N)])

class Gabor(object):
    def __init__(self,
                 nfilters=40,
                 min_freq=16,
                 max_freq=8000,
                 fs=16000,
                 wlen=25,
                 wstride=10,
                 nfft=512,
                 normalize_energy=False):
            if not nfilters > 0:
                raise(Exception,
                'Number of filters must be positive, not {0:%d}'.format(nfilters))
            if max_freq > fs // 2:
                raise(Exception,
                'Upper frequency %f exceeds Nyquist %f' % (max_freq, fs // 2))
            self.nfilters = nfilters
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.fs = fs
            self.wlen = wlen
            self.wstride = wstride
            self.nfft = nfft
            self.normalize_energy = normalize_energy
            self._build_mels()
            self._build_gabors()

    def _hz2mel(self, f):
        # Converts a frequency in hertz to mel
        return 2595 * np.log10(1+f/700)

    def _mel2hz(self, m):
        # Converts a frequency in mel to hertz
        return 700 * (np.power(10, m/2595) - 1)

    def _gabor_wavelet(self, eta, sigma):
        T = self.wlen * self.fs / 1000
        # Returns a gabor wavelet on a window of size T

        def gabor_function(t):
            return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(1j * eta * t) * np.exp(-t**2/(2 * sigma**2))
        return np.asarray([gabor_function(t) for t in np.arange(-T/2,T/2 + 1)])

    def _gabor_params_from_mel(self, mel_filter):
        # Parameters in radians
        coeff = np.sqrt(2*np.log(2))*self.nfft
        mel_filter = np.sqrt(mel_filter)
        center_frequency = np.argmax(mel_filter)
        peak = mel_filter[center_frequency]
        half_magnitude = peak/2.0
        spread = np.where(mel_filter >= half_magnitude)[0]
        width = max(spread[-1] - spread[0],1)
        return center_frequency*2*np.pi/self.nfft, coeff/(np.pi*width)

    def _melfilter_energy(self, mel_filter):
        # Computes the energy of a mel-filter (area under the magnitude spectrum)
        height = max(mel_filter)
        hz_spread = (len(np.where(mel_filter > 0)[0])+2)*2*np.pi/self.nfft
        return 0.5 * height * hz_spread

    def _build_mels(self):
        # build mel filter matrix
        self.melfilters = [np.zeros(self.nfft//2 + 1) for i in range(self.nfilters)]
        dfreq = self.fs / self.nfft

        melmax = self._hz2mel(self.max_freq)
        melmin = self._hz2mel(self.min_freq)
        dmelbw = (melmax - melmin) / (self.nfilters + 1)
        # filter edges in hz
        filt_edge = self._mel2hz(melmin + dmelbw *
                                 np.arange(self.nfilters + 2, dtype='d'))
        self.filt_edge = filt_edge
        for filter_idx in range(0, self.nfilters):
            # Filter triangles in dft points
            leftfr = min(round(filt_edge[filter_idx] / dfreq), self.nfft//2)
            centerfr = min(round(filt_edge[filter_idx + 1] / dfreq), self.nfft//2)
            rightfr = min(round(filt_edge[filter_idx + 2] / dfreq), self.nfft//2)
            height = 1
            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.melfilters[filter_idx][int(freq)] = (freq - leftfr) * leftslope
                freq += 1
            if freq == centerfr:
                self.melfilters[filter_idx][int(freq)] = height
                freq += 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.melfilters[filter_idx][int(freq)] = (freq - rightfr) * rightslope
                freq += 1
            if self.normalize_energy:
                energy = self._melfilter_energy(self.melfilters[filter_idx])
                self.melfilters[filter_idx] /= energy

    def _build_gabors(self):
        self.gaborfilters = []
        self.sigmas = []
        self.center_frequencies = []
        for mel_filter in self.melfilters:
            center_frequency, sigma = self._gabor_params_from_mel(mel_filter)
            self.sigmas.append(sigma)
            self.center_frequencies.append(center_frequency)
            gabor_filter = self._gabor_wavelet(center_frequency, sigma)
            # Renormalize the gabor wavelets
            gabor_filter = gabor_filter * np.sqrt(self._melfilter_energy(mel_filter)*2*np.sqrt(np.pi)*sigma)
            self.gaborfilters.append(gabor_filter)
            
class AnalyticFreeFB(nn.Module):
    """Free analytic (fully learned with analycity constraints) filterbank.
    For more details, see [1].
    Args:
        n_filters (int): Number of filters. Half of `n_filters` will
            have parameters, the other half will be the hilbert transforms.
            `n_filters` should be even.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.
    Attributes:
        n_feats_out (int): Number of output filters.
    References
        [1] : "Filterbank design for end-to-end speech separation". ICASSP 2020.
        Manuel Pariente, Samuele Cornell, Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, 
                 wlen=25, wstride=10, fs=16000, 
                 train_stage=False, threshold=1, 
                 log_sigma2=-10.0):
        super().__init__()
        
        self._filters = n_filters
        self._kernel_size = kernel_size
        self.wlen = wlen
        self.wstride = wstride
        self.fs = fs
        self.train_stage = train_stage

        # kernel for real part
        self._kernel = nn.Parameter(torch.ones(self._filters, 1, self._kernel_size), 
                                    requires_grad=True)
        
        # vd parameters
        self.threshold = threshold
        # Initialize the variance term with same dimensionality as weights
        self.log_sigma2 = nn.Parameter(log_sigma2*torch.ones(self._filters, 1, self._kernel_size),
                                       requires_grad=True)
        # Initialize the field for the dropout rate
        self.log_alpha = None
        
        # Create frequency axis
        f = torch.cat(
            [
             torch.true_divide(torch.arange(0, (self._kernel_size-1)//2+1), self._kernel_size),
             torch.true_divide(torch.arange(-(self._kernel_size//2), 0), self._kernel_size),
            ]
        )
        
        # Create step function
        u = torch.heaviside(f, torch.tensor([0.5])).unsqueeze_(0)
        self.register_buffer("u_t", u.unsqueeze_(0))
        self.kernel_initialize()
    
    def kernel_initialize(self, min_freq=60.0,
                          max_freq=7800.0, nfft=512,
                          normalize_energy=False):
        # Initialize complex convolution
        self.complex_init = Gabor(self._filters,
                                  min_freq, max_freq,
                                  self.fs, self.wlen,
                                  self.wstride, nfft,
                                  normalize_energy)
        for idx, gabor in enumerate(self.complex_init.gaborfilters):
            self._kernel.data[idx][0].copy_(torch.from_numpy(np.real(gabor)))

    def forward(self, x):
        
        # compute imag filter
        xf = fft.fft(self._kernel, n=self._kernel_size, dim=-1)
        ht = fft.ifft(xf * 2 * self.u_t, dim=-1)
        real_filters = torch.real(ht)
        imag_filters = torch.imag(ht)
        imag_feats = F.conv1d(x, weight=imag_filters)
        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(EPSILON + torch.abs(self._kernel))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        if self.train_stage:
            # compute real vd conv
            gamma = F.conv1d(x, weight=real_filters)
            # Compute the variance of the weights
            sigma2 = torch.exp(self.log_sigma2)
            delta = F.conv1d(x ** 2, weight=sigma2)
            sqrt_delta = torch.sqrt(delta + EPSILON)
            noise = torch.cuda.FloatTensor(*gamma.size()).normal_()
            real_feats = gamma + sqrt_delta * noise
        else:
            mask = self.log_alpha > self.threshold
            # Feed the input features through the convolutional layer with masked weights
            real_feats = F.conv1d(x, weight=self._kernel.masked_fill(mask, 0))

        # Concatenate the features along a new dimension
        stacked_feats = torch.stack([real_feats, imag_feats], dim=-1)
        # Switch filter and frame dimension
        stacked_feats = stacked_feats.transpose(1, 2)
        # Collapse the last dimension to zip the features
        # and make the real/imag responses adjacent
        stacked_feats = stacked_feats.reshape(tuple(list(stacked_feats.shape[:-2]) + [-1]))
        stacked_feats = stacked_feats.transpose(1, 2)
        return stacked_feats
    
    def kld(self):
        """
        Compute the approximate KL-divergence of the current weights.

        Returns
        ----------
        kld : float
          KL-divergence averaged across all weights of the layer
        """

        # Terms for approximate KL-divergence
        k = [0.63576, 1.87320, 1.48695]

        # Compute the approximate negative KL-divergence
        nkl = k[0] * torch.sigmoid(k[1] + k[2] * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha)) - k[0]

        # Flip sign and average the KL-divergence across the weights
        kld = -torch.sum(nkl) / (1 * self._filters * self._kernel_size)
        return kld
            
class TDFbanks(nn.Module):
    def __init__(self,
                 nfilters,
                 samplerate=16000,
                 wlen=25,
                 compression='log',
                 preemp=True,
                 mvn=True):
        super(TDFbanks, self).__init__()
        window_size = samplerate * wlen // 1000 + 1
        window_stride = 5
        padding_size = (window_size - 1) // 2
        self.nfilters = nfilters
        self.fs = samplerate
        self.wlen = wlen
        self.compression = compression
        self.mvn = mvn
        self.preemp = None
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, 1, padding=1, groups=1, bias=False)
        self.modulus = nn.LPPool1d(2, 2, stride=2)
        self.complex_conv = AnalyticFreeFB(nfilters, window_size)
        self.lowpass = nn.Conv1d(nfilters, nfilters, window_size, window_stride,
                                 padding=0, groups=nfilters, bias=False)
        if preemp:
            self.preemp.weight.requires_grad = False
        self.lowpass.weight.requires_grad = False
        if mvn:
            self.instancenorm = norm_block('in', dim=nfilters, affine=True)
    
    def initialize(self, window_type='hamming', alpha=0.97):
        # Initialize preemphasis
        if self.preemp:
            self.preemp.weight.data[0][0][0] = -alpha
            self.preemp.weight.data[0][0][1] = 1
        
        # Initialize lowpass
        self.lowpass_init = window(window_type, (self.fs * self.wlen)//1000 + 1)
        for idx in range(self.nfilters):
            self.lowpass.weight.data[idx][0].copy_(
                torch.from_numpy(self.lowpass_init))

    def forward(self, x):

        # Preemphasis
        if self.preemp:
            x = self.preemp(x)
        # Complex convolution
        x = self.complex_conv(x)
        # Squared modulus operator
        x = x.transpose(1, 2)
#         x = F.avg_pool1d(x.pow(2), 2, 2, 0, False).mul(2)
        x = self.modulus(x)
        x = x.transpose(1, 2)
        x = self.lowpass(x)
            
        x = x.abs()
        x = x + 1
        if self.compression == 'log':
            x = x.log()
        # The dimension of x is 1, n_channels, seq_length
        if self.mvn:
            x = self.instancenorm(x)
        return x

class architecture(nn.Module):
    
    def __init__(self, nfilters=30, wlen=25, fs = 16000,
                 ds_layers=[(128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2)],
                 embed_dim=512):
        
        super(architecture, self).__init__()

        self.nfilters = nfilters
        self.fs = fs
        self.wlen = wlen
        self.tdfbanks = TDFbanks(nfilters=self.nfilters, 
                                 samplerate=self.fs,
                                 wlen=self.wlen)
        
        # Initialization parameters
        init_params = dict(alpha=0.97)              
        self.tdfbanks.initialize(**init_params)
        
        # downsampling stacks
        self.convds = ConvDS(ds_layers=ds_layers, in_dim=nfilters)
        
    def forward(self, x):

        # neil's implementation
        outputs = self.tdfbanks(x)
        outputs = self.convds(outputs)
        return outputs
    
if __name__ == '__main__':
    from pytorch_model_summary import summary
    model = architecture()
    print(summary(model, torch.rand((4, 1, 64000)), max_depth=None, show_input=False))