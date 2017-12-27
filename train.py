import torch
import argparse, math
import data, utils
from logger import Logger
from cycleGAN import CycleGAN
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("--dataset", default="unvoiced/Jazz.npy,unvoiced/Classical.npy", type=str, help="comma-separated dataset paths")
#parser.add_argument("--dataset", default="unvoiced/dummyA.npy,unvoiced/dummyB.npy", type=str, help="comma-separated dataset paths")
parser.add_argument("--output-dir", default="output/", type=str, help="dir for outputting log/model")
parser.add_argument("--lr", default=0.0002, type=float, help="init learning rate")
parser.add_argument("--lamb-A", default=10., type=float, help="weight for A->B->A'")
parser.add_argument("--lamb-B", default=10., type=float, help="weight for B->A->B'")
parser.add_argument("--lamb-I", default=0.1, type=float, help="weight for identity loss")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--pool-size", default=50, type=int, help="data pool size for discriminator")
parser.add_argument("--gpu-ids", default=[3], type=list, help="comma-separated gpu ids")
parser.add_argument("--n-fft", default=2048, type=int, help="n fft")
parser.add_argument("--hop-length", default=512, type=int, help="n fft")
parser.add_argument("--n-sec", default=8.16, type=float, help="n sec for each chunk")
parser.add_argument("--sr", default=16000, type=int, help="n sec for each chunk")
parser.add_argument("--n-epoch", default=10, type=int, help="total training epoch")
parser.add_argument("--log-interval", default=10, type=int, help="steps between each log")

args = parser.parse_args()
logger = Logger("logdir")

if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

shape = (args.n_fft // 2, int(math.ceil(args.n_sec * args.sr / args.hop_length)+1))
model = CycleGAN(args.lr, args.lamb_A, args.lamb_B, args.lamb_I, args.output_dir, shape, args.batch_size, args.pool_size, args.gpu_ids, True)

datasets = args.dataset.split(",")
datasetA = data.get_fft_npy_loader(datasets[0], batch_size=1)
datasetB = data.get_fft_npy_loader(datasets[1], batch_size=1)

cnt = 0
for i in range(args.n_epoch):
    for j, (a, b) in enumerate(zip(datasetA, datasetB)):
        model.set_input({"A": a[0], "B": b[0]})
        model.optimize()
        errors = model.report_errors()
        cnt += 1
        if cnt % args.log_interval == 0:
            audio = OrderedDict()
            logger.log(cnt, errors, log_type="scalar", text=True)
            fake_As, fake_Bs = model.get_samples(5)
            for k, (a, b) in enumerate(zip(fake_As, fake_Bs)):
                a = utils.generate_audio(a, sr=args.sr, hop_length=args.hop_length)
                b = utils.generate_audio(b, sr=args.sr, hop_length=args.hop_length)
                audio["A_{}_{}".format(cnt, k)] = a
                audio["B_{}_{}".format(cnt, k)] = b
            logger.log(cnt, audio, log_type="audio", sr=args.sr)
            logger.write()
            logger.flush()

