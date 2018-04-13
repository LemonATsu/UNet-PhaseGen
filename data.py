import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset, ConcatDataset

def get_fft_npy_loader(paths, labels=None, batch_size=1, norm=True, precon=False):

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
            if precon:
                d = get_spec_and_angle(d)
            d = torch.from_numpy(d)
            targets = torch.ones(d.size(0), 1) * l
            dataset.append(TensorDataset(d, targets))

    dataset = ConcatDataset(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def _norm(data):
    return (data - data.mean()) / data.std()

def normalize(data):
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            data[i, j] = _norm(data[i, j])
    return data

def get_spec_and_angle(data, use_exp=True):
    fft = data[:, 0, ...] + data[:, 1, ...] * 1j
    spec = np.abs(fft)

    if use_exp:
        spec = np.log1p(spec)
    angle = np.angle(fft)

    return np.concatenate([spec[:, np.newaxis, ...], angle[:, np.newaxis, ...]], axis=1)

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
    from utils import generate_spec_img, generate_audio, griffin_lim
    from cycleGAN import AEModel, UNetModel
    from collections import OrderedDict

    gpu_id = 2
    torch.cuda.set_device(gpu_id)
    batch_size = 16
    model = UNetModel(1024, 1024*2, gpu_ids=[gpu_id]).cuda(gpu_id)
    sr = 16000
    # model_r = AEModel(1024, 1, gpu_ids=[2], batch_size=batch_size).cuda(2)
    loader = get_fft_npy_loader(
            ["dataset/Pop_audio_train.npy"],
            #["dataset/Pop_audio_val.npy"],
            #["dataset/dummy.npy"],
            [0, 1], batch_size=batch_size, precon=True)
    val_loader = get_fft_npy_loader(
            ["dataset/Pop_audio_val.npy"],
            [0, 1], batch_size=3, precon=True)

    lr = 0.001
    """
    optim_r = torch.optim.Adam(model_r.parameters(), lr=lr)
    optim_i = torch.optim.Adam(model_i.parameters(), lr=lr)
    """
    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    lossf = torch.nn.MSELoss()
    logger = Logger("unet_llr/")

    j = 0
    cnt = 0
    ang_losses, mag_losses = [], []
    while True:
        start = time.time()
        for i, d in enumerate(loader):
            if d[0].size(0) < batch_size:
                continue
            cnt +=1
            optim.zero_grad()
            pred = model.forward(Variable(d[0][:, 0, ...]).cuda(gpu_id))
            pred_p, pred_m = pred[:, :1024], pred[:, 1024:]
            cos_loss = lossf(torch.cos(pred_p), Variable(d[0][:, 1, ...].cos()).cuda(gpu_id))
            sin_loss = lossf(torch.sin(pred_p), Variable(d[0][:, 1, ...].sin()).cuda(gpu_id))
            mag_loss = lossf(pred_m, Variable(d[0][:, 0, ...]).cuda(gpu_id))
            ang_loss = cos_loss + sin_loss

            loss = ang_loss + mag_loss * 0.2
            loss.backward()
            optim.step()

            """
            optim_r.zero_grad()
            pred_r= model_r.forward(Variable(d[0][:, 0, ...]).cuda(2))
            loss_r = lossf(pred_r, Variable(d[0][:, 0, ...]).cuda(2))
            loss_r.backward()
            optim_r.step()
            """

            ang_losses.append(ang_loss.data[0])
            mag_losses.append(mag_loss.data[0])
            #losses_r.append(loss_r.data[0])


            if cnt % 2000 == 0:
                val_d = val_loader.__iter__().__next__()[0]
                mses = []
                nop_mses = []
                lim_mses = []
                for c, vd in enumerate(val_d):
                    vd = vd.unsqueeze(0)
                    pred = model.forward(Variable(vd[:, 0], volatile=True).cuda(gpu_id))
                    _orig = vd.cpu().numpy()[0]
                    _gen = pred.data.cpu().numpy()[0, :1024, ...]
                    #_rgen = pred_r[0].data.cpu().numpy()

                    orig = (np.exp(_orig[0]) - 1) * np.exp(_orig[1] * 1.j)
                    #gen = (np.exp(_rgen) - 1) * np.exp(_gen * 1.j)
                    hybrid = (np.exp(_orig[0]) - 1) * np.exp(_gen * 1.j)
                    no_phase = np.exp(_orig[0]) - 1
                    #bridhy = (np.exp(_rgen) - 1) * np.exp(_orig[1] * 1.j)

                    orig_ = generate_spec_img(orig, is_stft=True)
                    #gen_ = generate_spec_img(gen, is_stft=True)
                    hyb_ = generate_spec_img(hybrid, is_stft=True)
                    nop_ = generate_spec_img(no_phase, is_stft=True)
                    #byh_ = generate_spec_img(bridhy, is_stft=True)

                    report_i = OrderedDict([
                        ("Origin_{}_{}".format(cnt, c), orig_),
                        #("Gen_{}".format(cnt), gen_),
                        ("Hybrid_{}_{}".format(cnt, c), hyb_),
                        #("Bridhy_{}".format(cnt), byh_),
                        ("NP_{}_{}".format(cnt, c), nop_),
                    ])

                    orig = generate_audio(orig, sr=sr, hop_length=512, is_stft=True)
                    #gen = generate_audio(gen, sr=8000, hop_length=512, is_stft=True)
                    hyb = generate_audio(hybrid, sr=sr, hop_length=512, is_stft=True)
                    #bhy = generate_audio(bridhy, sr=8000, hop_length=512, is_stft=True)
                    nop = generate_audio(no_phase, sr=sr, hop_length=512, is_stft=True)
                    lim, _, _ = griffin_lim(no_phase, n_fft=2048, hop_length=512, n_iter=250)

                    mse = np.sqrt((orig - hyb)**2)
                    nmse = np.sqrt((orig - nop)**2)
                    lmse = np.sqrt((orig - lim)**2)
                    mses.extend(mse)
                    nop_mses.extend(nmse)
                    lim_mses.extend(lmse)

                    report_a = OrderedDict([
                        ("wav_Origin_{}_{}".format(cnt, c), orig),
                        #("wav_Gen_{}".format(cnt), gen),
                        ("wav_Hyb_{}_{}".format(cnt, c), hyb),
                        #("wav_Bhy_{}".format(cnt), bhy),
                        ("wav_Nop_{}_{}".format(cnt, c), nop),
                        ("wav_GLim_{}_{}".format(cnt, c), lim),
                    ])

                    logger.log(cnt, report_i, log_type="image")
                    logger.log(cnt, report_a, log_type="audio", sr=sr)
                    logger.write()
                    logger.flush()
                logger.log(cnt, OrderedDict([("MSE", np.mean(mses)), ("NOPMSE", np.mean(nop_mses)), ("LMSE", np.mean(lim_mses))]))
                logger.write()
                logger.flush()
                if cnt % 4000 == 0:
                    model.save("unet_llr/ckpt_{}".format(cnt))
        j += 1

        print("Epoch {} done, {} elasped, mag loss: {}, ang loss: {}".format(j, time.time()-start, np.mean(mag_losses), np.mean(ang_losses)))
        logger.log(j, OrderedDict([("Ang Loss", np.mean(ang_losses)), ("Mag Loss", np.mean(mag_losses))]))
        logger.write()
        logger.flush()
        #print("Epoch {} done, {} elasped, loss: {}, loss_r: {}".format(j, time.time()-start, np.mean(losses), np.mean(losses_r)))

