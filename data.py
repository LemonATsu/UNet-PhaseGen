import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset, ConcatDataset

def get_fft_npy_loader(paths, labels=None, batch_size=1, norm=True):

    if not isinstance(paths, list):
        paths = [paths]
    if labels is None:
        labels = [0]

    #dataset = None
    dataset = []
    for p, l in zip(paths, labels):
        if os.path.exists(p):
            print("{} exists, start loading ...".format(p))
            d = np.load(p, mmap_mode="r")
            d = torch.from_numpy(d)
            targets = torch.ones(d.size(0), 1) * l
            dataset.append(TensorDataset(d, targets))
            
    dataset = ConcatDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def _norm(data):
    """
    if (data.max() - data.min()) == 0:
        return data
    return 2 * (data - data.min()) / (data.max() - data.min()) - 1
    """
    return (data - data.mean()) / data.std()

def normalize(data):
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            data[i, j] = _norm(data[i, j])
    return data

def get_real_and_imag(data, norm):
    if data.dtype != np.complex64:
        return data
    real = np.real(data)
    imag = np.imag(data)
    if norm:
        print("Get norm ...")
        real = _norm(real)
        imag = _norm(imag)
        print("Done.")
    return np.concatenate([real, imag], axis=1)
    #return np.concatenate([real[:, np.newaxis, 0, ...], imag[:, np.newaxis, 0, ...]], axis=1)


if __name__ == "__main__":
    import time
    from logger import Logger
    from utils import generate_spec_img, generate_audio
    from cycleGAN import NLayerDiscriminator, AEModel, AxisCompressGenerator, XModel
    from collections import OrderedDict

    # model = NLayerDiscriminator(2, ndf=32, n_layers=4, gpu_ids=[3])
    # model = AxisCompressGenerator(2, 2, 8, gpu_ids=[0])
    # model_r = AEModel(1024, 1, gpu_ids=[0], batch_size=32)
    # model_i = AEModel(128, 1, gpu_ids=[0], batch_size=32, trp=True)
    # model = XModel(1024, 1, batch_size=32)
    model = AEModel(2048, 1, gpu_ids=[0], batch_size=32)
    loader = get_fft_npy_loader(
            ["dataset/Jazz_audio_train.npy"],
            #["dataset/Jazz_audio_train.npy", "dataset/Pop_audio_train.npy"],
            [0, 1], batch_size=32)
    val_loader = get_fft_npy_loader(
            ["dataset/Jazz_audio_val.npy"],
            #["dataset/Jazz_audio_val.npy", "dataset/Pop_audio_val.npy"],
            [0, 1], batch_size=32)

    lr = 0.0001
    """
    optim_r = torch.optim.Adam(model_r.parameters(), lr=lr)
    optim_i = torch.optim.Adam(model_i.parameters(), lr=lr)
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lossf_l1 = torch.nn.L1Loss()
    lossf_mse = torch.nn.MSELoss()
    logger = Logger("test/")
    fake_phase = np.ones((1, 1024, 128), dtype=np.float32) * 0.5

    j = 0
    cnt = 0
    while True:
        for i, d in enumerate(loader):
            if d[0].size(0) < 32: 
                continue
            cnt +=1 
            start = time.time()
            optim.zero_grad()
            pred = model.forward(Variable(d[0]))
            length = torch.sqrt(pred[:, :1024, ...]**2 + pred[:, 1024:, ...]**2)
            length_r = torch.sqrt(Variable(d[0][:, 0, ...]**2)+Variable(d[0][:, 1,...]**2))
            loss_r = lossf_l1(pred[:, :1024, ...], Variable(d[0][:, 0, ...])) 
            loss_i = lossf_l1(pred[:, 1024:, ...], Variable(d[0][:, 1, ...])) 
            loss_l = lossf_mse(length, length_r)
            #loss = lossf(pred, Variable(d[0]))
            loss = loss_r + loss_i
            loss.backward()
            optim.step()

            """
            optim_r.zero_grad()
            optim_i.zero_grad()
            pred_r = model_r.forward(Variable(d[0])[:, 0, ...])
            pred_i = model_i.forward(Variable(d[0])[:, 1, ...])
            loss_r = lossf(pred_r, Variable(d[0])[:, 0, ...])
            loss_i = lossf(pred_i, Variable(d[0])[:, 1, ...])
            loss_r.backward()
            loss_i.backward()
            optim_r.step()
            optim_i.step()
            """
            print("Forward done, {} elasped, loss: {})"
                .format(time.time()-start, loss.data[0]))
            if cnt % 5 == 0:
                _orig = d[0].cpu().numpy()[0]
                #_gen = pred[0].data.cpu().numpy()
                _gen = pred[0].data.cpu().numpy()
                #_gen = np.concatenate([_orig[np.newaxis,1, ...], fake_phase])
                #_gen = np.concatenate([ _orig[np.newaxis, 1, ...], _orig[np.newaxis, 0, ...]])
                _gen = np.concatenate([_gen[np.newaxis, :1024, ...], _gen[np.newaxis, 1024:, ...]]) 
                orig = generate_spec_img(_orig)
                gen = generate_spec_img(_gen)
                report_i = OrderedDict([("Origin_{}".format(cnt), orig), ("Gen_{}".format(cnt), gen)])
                orig = generate_audio(_orig, sr=8000, hop_length=512)
                gen = generate_audio(_gen, sr=8000, hop_length=512)
                report_a = OrderedDict([("wav_Origin_{}".format(cnt), orig), ("wav_Gen_{}".format(cnt), gen)])
                logger.log(cnt, report_i, log_type="image")
                logger.log(cnt, report_a, log_type="audio", sr=8000)
                logger.write()
                logger.flush()
            if cnt % 40 == 0:
                for j, d in enumerate(val_loader):
                    _orig = d[0].cpu().numpy()[0]
                    #_gen = model.forward(Variable(d[0]))[0, ...].data.cpu().numpy()
                    _gen = model.forward(Variable(d[0])).data.cpu().numpy()
                    orig = generate_spec_img(_orig)
                    gen = generate_spec_img(_gen)
                    report_i = OrderedDict([("val_Origin_{}".format(cnt), orig), ("val_Gen_{}".format(cnt), gen)])
                    orig = generate_audio(_orig, sr=8000, hop_length=512)
                    gen = generate_audio(_gen, sr=8000, hop_length=512)
                    report_a = OrderedDict([("ival_wav_Origin_{}".format(cnt), orig), ("ival_wav_Gen_{}".format(cnt), gen)])
                    logger.log(cnt, report_i, log_type="image")
                    logger.log(cnt, report_a, log_type="audio", sr=8000)
                    break
                logger.write()
                logger.flush()
                    

            
        print("Epoch {} finished.".format(j))
        j += 1
