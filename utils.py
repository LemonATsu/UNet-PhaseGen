import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import librosa

def generate_audio(matrix, sr, hop_length):
    stft = matrix[0, ...] + matrix[1, ...] * 1j
    # concatenate zero-filled dc component
    dc = np.zeros((1, stft.shape[1]), dtype=np.complex64)
    stft = np.concatenate((dc, stft), axis=0)
    audio = librosa.istft(stft, hop_length=hop_length)
    librosa.util.valid_audio(audio, mono=False)
    audio = librosa.util.normalize(audio, norm=np.inf, axis=None)
    return audio

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

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
        for i in range(n_sample):
            ind = np.random.randint(0, len(self.samples)-1)
            samples.append(self.samples[ind])
        samples = torch.cat(samples, 0)

        if isinstance(samples, torch.cuda.FloatTensor):
            return samples.cpu().numpy()
        return samples.numpy()
