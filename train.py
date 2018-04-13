import time
import torch
import numpy as np
from logger import Logger
from utils import generate_spec_img, generate_audio, griffin_lim
from model import UNetModel
from collections import OrderedDict
from data import get_fft_npy_loader
from torch.autograd import Variable

log_dir = "unet_llr/"
gpu_id = 2
torch.cuda.set_device(gpu_id)
batch_size = 16
model = UNetModel(1024, 1024*2, gpu_ids=[gpu_id]).cuda(gpu_id)
sr = 16000

loader = get_fft_npy_loader(
        ["dataset/Pop_audio_train.npy"],
        [0, 1], batch_size=batch_size, precon=True)

val_loader = get_fft_npy_loader(
        ["dataset/Pop_audio_val.npy"],
        [0, 1], batch_size=3, precon=True)

lr = 0.001
optim = torch.optim.Adam(model.parameters(), lr=lr)
lossf = torch.nn.MSELoss()

logger = Logger(log_dir)

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
        pred = model.forward(Variable(d[0][:, 0]).cuda(gpu_id))

        # predict both magnitude and phase angle
        pred_p, pred_m = pred[:, :1024], pred[:, 1024:]

        # compute angle loss
        # here, we use cos/sin loss
        cos_loss = lossf(torch.cos(pred_p), Variable(d[0][:, 1, ...].cos()).cuda(gpu_id))
        sin_loss = lossf(torch.sin(pred_p), Variable(d[0][:, 1, ...].sin()).cuda(gpu_id))
        ang_loss = cos_loss + sin_loss


        # predict magnitude
        # we found that predicting magnitude and angle at the same time can
        # slight increase the training speed.
        mag_loss = lossf(pred_m, Variable(d[0][:, 0, ...]).cuda(gpu_id))

        # scale the magnitude loss.
        loss = ang_loss + mag_loss * 0.2
        loss.backward()
        optim.step()

        ang_losses.append(ang_loss.data[0])
        mag_losses.append(mag_loss.data[0])


        # dump log
        if cnt % 2000 == 0:
            val_d = val_loader.__iter__().__next__()[0]
            mses = []
            nop_mses = []
            lim_mses = []
            for c, vd in enumerate(val_d):
                vd = vd.unsqueeze(0)
                pred = model.forward(Variable(vd[:, 0], volatile=True).cuda(gpu_id))
                _orig = vd.cpu().numpy()[0]
                _gen = pred.data.cpu().numpy()[0, :1024]

                # the pre-processed data is log-magnitude spectrogram
                # convert it back to spectrogram here.
                orig = (np.exp(_orig[0]) - 1) * np.exp(_orig[1] * 1.j)
                hybrid = (np.exp(_orig[0]) - 1) * np.exp(_gen * 1.j)
                no_phase = np.exp(_orig[0]) - 1

                # generate and dump spectrogram to tensorboard
                orig_ = generate_spec_img(orig, is_stft=True) # ground truth
                hyb_ = generate_spec_img(hybrid, is_stft=True) # ground truth with fake phase
                nop_ = generate_spec_img(no_phase, is_stft=True) # magnitude only

                report_i = OrderedDict([
                    ("Origin_{}_{}".format(cnt, c), orig_),
                    ("Hybrid_{}_{}".format(cnt, c), hyb_),
                    ("NP_{}_{}".format(cnt, c), nop_),
                    ])

                # generate audio and put them to tensorboard
                orig = generate_audio(orig, sr=sr, hop_length=512, is_stft=True)
                hyb = generate_audio(hybrid, sr=sr, hop_length=512, is_stft=True)
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
                    ("wav_Hyb_{}_{}".format(cnt, c), hyb),
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
                model.save(log_dir + "/ckpt_{}".format(cnt))
    j += 1

    print("Epoch {} done, {} elasped, mag loss: {}, ang loss: {}".format(j, time.time()-start, np.mean(mag_losses), np.mean(ang_losses)))
    logger.log(j, OrderedDict([("Ang Loss", np.mean(ang_losses)), ("Mag Loss", np.mean(mag_losses))]))
    logger.write()
    logger.flush()


