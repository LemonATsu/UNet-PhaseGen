import torch
from torch.autograd import Variable
import os, time
from logger import Logger
import numpy as np
from cycleGAN import AEModel
from data import get_fft_npy_loader
from utils import generate_spec_img
from collections import OrderedDict

lr = 1e-3
torch.cuda.set_device(0)
loader = get_fft_npy_loader(
            ["dataset/Jazz_audio_train.npy"],
            [0, 1], batch_size=32, precon=True
        )
model = AEModel(1024, 1, gpu_ids=[0], batch_size=32).cuda()
optim = torch.optim.Adam(model.parameters(), lr=lr)
lossf = torch.nn.L1Loss()
logger = Logger("test/")

cnt = 0
while True:
    for i, d in enumerate(loader):
        if d[0].size(0) < 32:
            continue
        cnt += 1
        start = time.time()

        optim.zero_grad()
        pred = model.forward(Variable(d[0][:, 0, ...]).cuda())

        loss = lossf(pred, Variable(d[0][:, 0, ...]).cuda())
        loss.backward()
        optim.step()

        print("Forward done, {} elasped, loss: {}".format(time.time()-start, loss.data[0]))

        if cnt % 200 == 0:
            orig = d[0][0, 0, ...].cpu().numpy()
            pred = pred[0].data.cpu().numpy()

            spec_orig = generate_spec_img(orig, is_amp=True)
            spec_pred = generate_spec_img(pred, is_amp=True)

            report_i = OrderedDict([
                    ("Origin_{}".format(cnt), spec_orig),
                    ("Pred_{}".format(cnt), spec_pred)
                ])
            logger.log(cnt, report_i, log_type="image")
            logger.write()
            logger.flush()


