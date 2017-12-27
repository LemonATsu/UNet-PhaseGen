import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset, ConcatDataset

def get_fft_npy_loader(paths, labels=None, batch_size=1):

    if not isinstance(paths, list):
        paths = [paths]
    if labels is None:
        labels = [0]

    datasets = []
    for p, l in zip(paths, labels):
        if os.path.exists(p):
            print("{} exists, start loading ...".format(p))
            dataset = torch.from_numpy(get_real_and_imag(np.load(p, mmap_mode="r")))
            targets = torch.ones(dataset.size(0), 1) * l
            datasets.append(TensorDataset(dataset, targets))
    datasets = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=True)
    return loader

def get_real_and_imag(data):
    if data.dtype != np.complex64:
        return data
    real = np.real(data)
    imag = np.imag(data)
    return np.concatenate([real[:, np.newaxis, ...], imag[:, np.newaxis, ...]], axis=1)


if __name__ == "__main__":
    import time
    from cycleGAN import NLayerDiscriminator

    model = NLayerDiscriminator(2, ndf=32, n_layers=4, gpu_ids=[3])
    loader = get_fft_npy_loader(
            ["unvoiced/Jazz.npy", "unvoiced/Classical.npy"],
            [0, 1], batch_size=32)

    lr = 1e-3
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCELoss()
    m = torch.nn.Sigmoid()

    j = 0

    while True:
        for i, d in enumerate(loader):
            start = time.time()
            optim.zero_grad()
            pred = model.forward(Variable(d[0]))
            loss = lossf(m(pred), Variable(d[1]))
            loss.backward()
            optim.step()
            predicted = m(pred).data
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            correct = (predicted == Variable(d[1]).data).sum()
            print("Forward done, {} elasped, accuracy {} ({}:{})".format(time.time()-start, correct / 32.0, (predicted==1).sum(), (predicted==0).sum()))
        print("Epoch {} finished.".format(j))
        j += 1
