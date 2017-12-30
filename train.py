import torch
import argparse, math
import data, utils, time, os
from utils import generate_audio, Pool
from logger import Logger
from cycleGAN import CycleGAN
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Training parameters")
#parser.add_argument("--dataset", default="unvoiced/Jazz_val.npy,unvoiced/Pop_val.npy", type=str, help="comma-separated dataset paths")
parser.add_argument("--dataset", default="unvoiced/Jazz_train.npy,unvoiced/Pop_train.npy", type=str, help="comma-separated dataset paths")
#parser.add_argument("--dataset", default="unvoiced/dummyA.npy,unvoiced/dummyB.npy", type=str, help="comma-separated dataset paths")
parser.add_argument("--output-dir", default="model_weight", type=str, help="dir for outputting log/model")
parser.add_argument("--lr", default=0.0002, type=float, help="init learning rate")
parser.add_argument("--lamb-A", default=10., type=float, help="weight for A->B->A'")
parser.add_argument("--lamb-B", default=10., type=float, help="weight for B->A->B'")
parser.add_argument("--lamb-I", default=0.01, type=float, help="weight for identity loss")
parser.add_argument("--lamb-E", default=0.0, type=float, help="weight for identity loss")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--n-D", default=5, type=int, help="update freq of D")
parser.add_argument("--pool-size", default=50, type=int, help="data pool size for discriminator")
parser.add_argument("--gpu-ids", default="3", type=str, help="comma-separated gpu ids")
parser.add_argument("--n-fft", default=2048, type=int, help="n fft")
parser.add_argument("--hop-length", default=512, type=int, help="n fft")
parser.add_argument("--n-sec", default=8.16, type=float, help="n sec for each chunk")
parser.add_argument("--sr", default=16000, type=int, help="n sec for each chunk")
parser.add_argument("--n-epoch", default=1000, type=int, help="total training epoch")
parser.add_argument("--log-interval", default=1, type=int, help="steps between each log")

args = parser.parse_args()
logger = Logger("logdir")

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

args.gpu_ids = args.gpu_ids.split(",")
args.gpu_ids = [int(g) for g in args.gpu_ids]
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

shape = (args.n_fft // 2, int(math.ceil(args.n_sec * args.sr / args.hop_length)+1))
model = CycleGAN(args.lr, args.lamb_A, args.lamb_B, args.lamb_I, args.lamb_E,
            args.output_dir, shape, args.batch_size, args.pool_size, args.gpu_ids, True)

datasets = args.dataset.split(",")
train_A = data.get_fft_npy_loader(datasets[0], batch_size=1)
train_B = data.get_fft_npy_loader(datasets[1], batch_size=1)
val_A = data.get_fft_npy_loader("unvoiced/Jazz_val.npy", batch_size=1)
val_B = data.get_fft_npy_loader("unvoiced/Pop_val.npy", batch_size=1)

cnt = 0

def accumulate_report(old_report, new_report=None):
    result = OrderedDict()
    if new_report is None:
        for ka, va in old_report.items():
            result[ka] = va / args.log_interval
    else:
        for (ka, va), (kb, vb) in zip(old_report.items(), new_report.items()):
            result[ka] = va + vb / args.log_interval
    return result

errors = None

for i in range(args.n_epoch):
    print("Epoch {} started.".format(i))
    model.sched_step()
    for j, (a, b) in enumerate(zip(train_A, train_B)):
        start = time.time()
        model.set_input({"A": a[0], "B": b[0]})
        model.optimize(n_D=args.n_D)
        cnt += 1

        if errors is None:
            errors = accumulate_report(model.report_errors())
        else:
            errors = accumulate_report(errors, model.report_errors())

        if cnt % args.log_interval == 0:
            logger.log(cnt, errors, log_type="scalar", text=True)

            if cnt % (args.log_interval * 20) == 0:
                audio = OrderedDict()
                real_As = model.rplA.get_samples(2)
                real_Bs = model.rplB.get_samples(2)
                for k, (a, b, va, vb) in enumerate(zip(real_As, real_Bs, val_A, val_B)):
                    model.set_input({"A": a, "B": b})
                    fake_A, fake_B, recn_A, recn_B = model.generate()
                    audio["A_train_{}_{}".format(cnt, k)] = generate_audio(a.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                    audio["B_train_{}_{}".format(cnt, k)] = generate_audio(b.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                    audio["FA_train_{}_{}".format(cnt, k)] = generate_audio(fake_A.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                    audio["FB_train_{}_{}".format(cnt, k)] = generate_audio(fake_B.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                    audio["RA_train_{}_{}".format(cnt, k)] = generate_audio(recn_A.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                    audio["RB_train_{}_{}".format(cnt, k)] = generate_audio(recn_B.cpu().numpy(), sr=args.sr, hop_length=args.hop_length)
                logger.log(cnt, audio, log_type="audio", sr=args.sr)
                model.save(j)

            logger.write()
            logger.flush()
            errors = None
            print("{} sec elapsed.".format(time.time()-start))

