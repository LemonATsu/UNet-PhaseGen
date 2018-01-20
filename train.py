import torch
import argparse, math
import data, utils, time, os
from utils import generate_audio, generate_spec_img, generate_waveplot, Pool
from logger import Logger
from cycleGAN import CycleGAN
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("--dataset", default="dataset/Jazz_audio_train.npy,dataset/Pop_audio_train.npy", type=str, help="comma-separated dataset paths")
#parser.add_argument("--dataset", default="dataset/Jazz_audio_val.npy,dataset/Pop_audio_val.npy", type=str, help="comma-separated dataset paths")
parser.add_argument("--output-dir", default="model_weight", type=str, help="dir for outputting log/model")
parser.add_argument("--lr", default=0.00008, type=float, help="init learning rate")
parser.add_argument("--lamb-A", default=1., type=float, help="weight for A->B->A'")
parser.add_argument("--lamb-B", default=1., type=float, help="weight for B->A->B'")
parser.add_argument("--lamb-I", default=0.05, type=float, help="weight for identity loss")
parser.add_argument("--lamb-E", default=0.01, type=float, help="weight for identity loss")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--n-D", default=1, type=int, help="update freq of D")
parser.add_argument("--n-G", default=1, type=int, help="update freq of G")
parser.add_argument("--pool-size", default=96, type=int, help="data pool size for discriminator")
parser.add_argument("--gpu-ids", default="0", type=str, help="comma-separated gpu ids")
parser.add_argument("--n-fft", default=2048, type=int, help="n fft")
parser.add_argument("--hop-length", default=512, type=int, help="n fft")
parser.add_argument("--n-sec", default=8.16, type=float, help="n sec for each chunk")
parser.add_argument("--sr", default=8000, type=int, help="n sec for each chunk")
parser.add_argument("--n-epoch", default=1000, type=int, help="total training epoch")
parser.add_argument("--log-interval", default=100, type=int, help="steps between each log")

args = parser.parse_args()
logger = Logger(args.output_dir)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

args.gpu_ids = args.gpu_ids.split(",")
args.gpu_ids = [int(g) for g in args.gpu_ids]
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

fc = [0, 1]
mc = [0, 1]
shape = (args.n_fft // 2, int(math.ceil(args.n_sec * args.sr / args.hop_length)))
model = CycleGAN(args.lr, args.lamb_A, args.lamb_B, args.lamb_I, args.lamb_E,
            args.output_dir, shape, args.batch_size, args.pool_size, args.gpu_ids, True, fc=fc, mc=mc, mtype="BE")

datasets = args.dataset.split(",")
train_A = data.get_fft_npy_loader(datasets[0], batch_size=args.batch_size)
train_B = data.get_fft_npy_loader(datasets[1], batch_size=args.batch_size)
val_A = data.get_fft_npy_loader("dataset/Jazz_audio_val.npy", batch_size=1)
val_B = data.get_fft_npy_loader("dataset/Pop_audio_val.npy", batch_size=1)

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
    start = time.time()
    for j, (a, b) in enumerate(zip(train_A, train_B)):
        if a[0].size() != b[0].size():
            continue
        model.set_input({"A": a[0], "B": b[0]})
        model.optimize(n_G=args.n_G, n_D=args.n_D, mtype="BE")
        cnt += 1

        if errors is None:
            errors = accumulate_report(model.report_errors())
        else:
            errors = accumulate_report(errors, model.report_errors())

        if cnt % args.log_interval == 0:
            logger.log(cnt, errors, log_type="scalar", text=True)

            if cnt % (args.log_interval * 20) == 0:
                audio = OrderedDict()
                image = OrderedDict()
                real_As = model.rplA.get_samples(1)
                real_Bs = model.rplB.get_samples(1)
                for k, (a, b, va, vb) in enumerate(zip(real_As, real_Bs, val_A, val_B)):
                    model.set_input({"A": a, "B": b})
                    fake_A, fake_B, recn_A, recn_B = model.generate()
                    # put it back to cpu
                    a = a.cpu().numpy()
                    b = b.cpu().numpy()
                    fake_A = fake_A.cpu().numpy()
                    fake_B = fake_B.cpu().numpy()
                    recn_A = recn_A.cpu().numpy()
                    recn_B = recn_B.cpu().numpy()

                    # generate audio
                    A_train = generate_audio(a, sr=args.sr, hop_length=args.hop_length)
                    B_train = generate_audio(b, sr=args.sr, hop_length=args.hop_length)
                    FA_train = generate_audio(fake_A, sr=args.sr, hop_length=args.hop_length)
                    FB_train = generate_audio(fake_B, sr=args.sr, hop_length=args.hop_length)
                    RA_train = generate_audio(recn_A, sr=args.sr, hop_length=args.hop_length)
                    RB_train = generate_audio(recn_B, sr=args.sr, hop_length=args.hop_length)

                    # put to dict
                    audio["A_train_{}_{}".format(cnt, k)] = A_train
                    audio["B_train_{}_{}".format(cnt, k)] = B_train
                    audio["FA_train_{}_{}".format(cnt, k)] = FA_train
                    audio["FB_train_{}_{}".format(cnt, k)] = FB_train
                    audio["RA_train_{}_{}".format(cnt, k)] = RA_train
                    audio["RB_train_{}_{}".format(cnt, k)] = RB_train

                    image["spec_A_train_{}_{}".format(cnt, k)] = generate_spec_img(a)
                    image["A_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(A_train, sr=args.sr)
                    image["spec_B_train_{}_{}".format(cnt, k)] = generate_spec_img(b)
                    image["B_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(B_train, sr=args.sr)

                    image["spec_FA_train_{}_{}".format(cnt, k)] = generate_spec_img(fake_A)
                    image["FA_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(FA_train, sr=args.sr)
                    image["spec_FB_train_{}_{}".format(cnt, k)] = generate_spec_img(fake_B)
                    image["FB_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(FB_train, sr=args.sr)

                    image["spec_RA_train_{}_{}".format(cnt, k)] = generate_spec_img(recn_A)
                    image["RA_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(RA_train, sr=args.sr)
                    image["spec_RB_train_{}_{}".format(cnt, k)] = generate_spec_img(recn_B)
                    image["RB_train_{}_{}_wav".format(cnt, k)] = generate_waveplot(RB_train, sr=args.sr)

                logger.log(cnt, audio, log_type="audio", sr=args.sr)
                logger.log(cnt, image, log_type="image", sr=args.sr)
                model.save(cnt)

            logger.write()
            logger.flush()
            errors = None
            print("%.4f sec elapsed." % (time.time()-start))
            start = time.time()

