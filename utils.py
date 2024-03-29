import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_audio(spec, sr, hop_length, is_stft=False):
    """
    Generate audio from a given (1) mag_stft or (2) separated real/complex matrices
    specified by is_stft.

    Parameters
    ----------
    spec : ndarray
        Takes either (1)mag_stft or (2) separated real/complex matrices
    sr : int
        Sampling rate for generating audio
    hop_length : int
        Hop length used for istft
    is_stft : bool
        Specify the input spec type.

    Returns
    -------
    audio : ndarray
        Generated and normalized audio.

    """

    stft = spec[0, ...] + spec[1, ...] * 1j if not is_stft else spec

    # the dc component is removed during data-processing stage,
    # concatenate zero-filled dc component back for istft
    dc = np.zeros((1, stft.shape[1]), dtype=np.complex64)
    stft = np.concatenate((dc, stft), axis=0)
    audio = librosa.istft(stft, hop_length=hop_length)
    librosa.util.valid_audio(audio, mono=False)
    audio = librosa.util.normalize(audio, norm=np.inf, axis=None)

    return audio

def generate_spec_img(spec, is_stft=False, is_amp=False):
    """
    Generate spectrogram (image) from a given (1) mag_stft or (2) separated real/complex matrices
    specified by is_stft (or is_amp).

    Parameters
    ----------
    spec : ndarray
        Takes either (1)mag_stft or (2) separated real/complex matrices
    is_stft : bool
        Specify the input spec type.
    is_amp : bool
        Specify whether the input spec is already a amplitude spectrogram.

    Returns
    -------
    img : ndarray
        Image of the given spectrogram.

    """


    if not is_amp:
        stft = spec[0, ...] + spec[1, ...] * 1j if not is_stft else spec
        D = librosa.amplitude_to_db(stft, ref=np.max)
    else:
        D = spec

    fig = plt.figure(figsize=(3, 2))
    librosa.display.specshow(D, y_axis="linear")
    #plt.colorbar(format="%+2.0f dB")
    plt.colorbar()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()

    return img

def griffin_lim(spec, n_fft, hop_length, n_iter):
    """
    Generate phase components for the given spec, and output the reconstructed
    audio clips.

    Parameters
    ----------
    spec : ndarray
        stft spectrogram.
    n_fft : int
        n_fft for stft.
    hop_length : int
        Hop length for stft.
    n_iter : int
        The number of griffin lim iterations for phase reconstruction.

    Returns
    -------
    recon_aud : ndarray
        Reconstructed audio.
    new_spec : ndarray
        Reconstructed spectrogram.
    loss : float
        RMSE loss of griffin lim.
    """


    n = n_iter

    audio = librosa.istft(spec, hop_length=hop_length)
    # start with random vector (extracting phase from random vector)
    recon_aud = np.random.randn(audio.shape[0])

    while n > 0:
        n -= 1
        recon_spec = librosa.stft(recon_aud, n_fft=n_fft, hop_length=hop_length)
        recon_spec = np.delete(recon_spec, (0), axis=0)
        recon_phase = np.angle(recon_spec) # extract phase

        new_spec = spec * np.exp(1.0j * recon_phase)
        prev_aud = recon_aud # for loss computation

        recon_aud = librosa.istft(new_spec, hop_length=hop_length)
        loss = np.sqrt(np.sum((recon_aud - prev_aud)**2 / recon_aud.size))

    # normalize the generated audio
    librosa.util.valid_audio(recon_aud, mono=False)
    recon_aud = librosa.util.normalize(recon_aud, norm=np.inf, axis=None)

    return recon_aud, new_spec, loss

def generate_waveplot(audio, sr):
    fig = plt.figure(figsize=(3, 2))
    librosa.display.waveplot(audio, sr=sr)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()
    return img

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, input):
        t = torch.transpose(input, self.dim0, self.dim1)
        return t.contiguous()

class EnergyLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(EnergyLoss, self).__init__()
        self.tensor = tensor
        self.loss = nn.MSELoss()

    def __call__(self, a, b):
        amp_a = self._calc_amp(a)
        amp_b = self._calc_amp(b)
        return self.loss(amp_a, amp_b)

    def _calc_amp(self, a):
        return torch.sqrt(a[:, 0, ...]**2 + a[:, 1, ...]**2 + 1e-10)


class GANLoss(nn.Module):
    def __init__(self, real_label=1., fake_label=0., tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.tensor = tensor
        self.real_label = real_label
        self.fake_label = fake_label
        self.real_var = None
        self.fake_var = None
        self.loss = nn.MSELoss()

    def __call__(self, input, is_real):
        target = self.get_target(input, is_real)
        return self.loss(input, target)

    def get_target(self, input, is_real):
        target = None
        if is_real:
            if (self.real_var is None) or \
                   (self.real_var.numel() != input.numel()):
                real_tensor = self.tensor(input.size()).fill_(self.real_label)
                self.real_var = Variable(real_tensor, requires_grad=False)
            target = self.real_var
        else:
            if (self.fake_var is None) or \
                   (self.fake_var.numel() != input.numel()):
                fake_tensor = self.tensor(input.size()).fill_(self.fake_label)
                self.fake_var = Variable(fake_tensor, requires_grad=False)
            target = self.fake_var
        return target


class Pool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.n = 0
        self.samples = []

    def draw(self, samples):

        if self.pool_size == 0:
            return Variable(samples)

        drawn = []
        for s in samples:
            s = torch.unsqueeze(s, 0)
            if self.n < self.pool_size:
                self.n += 1
                self.samples.append(s)
                drawn.append(s)
            else:
                p = np.random.uniform()
                if p > 0.5:
                    ind = np.random.randint(0, self.pool_size-1)
                    tmp = self.samples[ind].clone()
                    self.samples[ind] = s
                    drawn.append(tmp)
                else:
                    drawn.append(s)
        drawn = Variable(torch.cat(drawn, 0))
        return drawn

    def get_samples(self, n_sample):
        samples = []
        if self.n < 0:
            raise "Empty pool!"
        if self.n == 1:
            samples.append(self.samples[0])
        else:
            for i in range(n_sample):
                ind = np.random.randint(0, self.n - 1)
                samples.append(self.samples[ind])
        samples = torch.cat(samples, 0)

        """
        if isinstance(samples, torch.cuda.FloatTensor):
            return samples.cpu()
        """
        return samples

