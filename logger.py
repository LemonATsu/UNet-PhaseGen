import os
from tensorboardX import SummaryWriter

LOG_TYPE = ["scalar", "audio", "image"]

class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter()

    def log(self, n_iter, report, log_type="scalar", sr=None, text=False):
        if log_type not in LOG_TYPE:
            raise("Wrong data type for logger.")

        if log_type == "scalar":
            if text:
                self._print_scalars(n_iter, report)
            for k, v in report.items():
                self.writer.add_scalar("scalar/{}".format(k), v, n_iter)
        elif log_type == "audio":
            if sr is None:
                raise("Sample rate is required for saving audio data.")
            for k, v in report.items():
                self.writer.add_audio(k, v, n_iter, sample_rate=sr)

    def _print_scalars(self, n_iter, report):
        print("---------------------------")
        print("n_iter : {}".format(n_iter))
        for k, v in report.items():
            print("{} : {:.4f}".format(k, v))
        print("---------------------------")

    def write(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer.export_scalars_to_json(os.path.join(self.log_dir, "log.json"))

    def flush(self):
        self.writer.file_writer.flush()

    def close(self):
        self.writer.close()

if __name__ == "__main__":
    import numpy as np
    from utils import generate_audio
    from collections import OrderedDict
    logger = Logger("test/")
    x = np.load("unvoiced/Jazz.npy", mmap_mode="r")
    r, i = np.real(x[20]), np.imag(x[20])
    c = np.concatenate([r[np.newaxis, ...], i[np.newaxis, ...]], axis=0)
    audio = generate_audio(c, sr=16000, hop_length=512)
    print("generated")
    report = OrderedDict([("a20.wav", audio)])
    print(report)
    logger.log(1, report, log_type="audio", sr=16000)
    logger.write()
    logger.flush()
    print("log done")

