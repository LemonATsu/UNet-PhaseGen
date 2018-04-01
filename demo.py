import torch, argparse, time
import numpy as np
from data import get_fft_npy_loader
from utils import generate_audio, griffin_lim
from cycleGAN import UNetModel
from librosa.output import write_wav
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="Arguments for generating demo clips.")
parser.add_argument("--genre", required=True)
parser.add_argument("--n_songs", default=5, type=int)
parser.add_argument("--n_fft", default=2048)
parser.add_argument("--sr", default=16000)
parser.add_argument("--hop", default=512)
parser.add_argument("--gpu", default=3)
parser.add_argument("--weight", required=True)

args = parser.parse_args()
loader = get_fft_npy_loader(
                ["dataset/" + args.genre + "_audio_val.npy"],
                [0, 1], batch_size=args.n_songs, precon=True,
        )

torch.cuda.set_device(args.gpu)
model = UNetModel(1024, 1024*2, gpu_ids=[args.gpu]).cuda(args.gpu)
model.load(args.weight)

data = loader.__iter__().__next__()[0]

# NN
runtimes = []

for c, d in enumerate(data):
    d = d.unsqueeze(0)
    start = time.time()
    pred = model.forward(Variable(d[:, 0], volatile=True).cuda(args.gpu))
    mag = d.cpu().numpy()[0]
    pred = pred.data.cpu().numpy()[0, :1024, ...]
    stft = (np.exp(mag[0]) - 1) * np.exp(pred * 1.j)
    audio = generate_audio(stft, sr=args.sr, hop_length=args.hop, is_stft=True)
    end = time.time() - start
    runtimes.append(end)
    write_wav("demo/unet_{}_{}.wav".format(args.genre, c), audio, sr=args.sr)

print("UNet - avg {} sec per clip.".format(np.mean(runtimes)))

# Griffin-Lim
runtimes = []

for c, d in enumerate(data):
    d = d.unsqueeze(0)
    start = time.time()
    mag = d.cpu().numpy()[0]
    mag = np.exp(mag[0]) - 1
    lim, _, _ = griffin_lim(mag, n_fft=args.n_fft, hop_length=args.hop, n_iter=250)
    end = time.time() - start
    runtimes.append(end)
    write_wav("demo/gl_{}_{}.wav".format(args.genre, c), lim, sr=args.sr)

print("GL - avg {} sec per clip".format(np.mean(runtimes)))

