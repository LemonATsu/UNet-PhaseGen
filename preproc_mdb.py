import medleydb as mdb
import numpy as np
import librosa
import argparse
import time
import os.path
from medleydb.mix import mix_no_vocals, VOCALS, mix_multitrack

def _get_path(track, filename):
    stems = track.stems
    stem_indices = []
    has_vocal = False
    for i in stems.keys():
        not_vocal = all([inst not in VOCALS for inst in stems[i].instrument])
        if not_vocal:
            stem_indices.append(i)
        else:
            has_vocal = True

    if has_vocal:
        if not os.path.exists(filename):
            print("Unvoice version for {} does not exist, generate one.".format(filename))
            mix_multitrack(track, filename, stem_indices=stem_indices)
        return filename
    else:
        return track.mix_path

def get_paths(tracks, genre, unvoice):
    paths = []
    for t in tracks:
        if t.genre == genre:
            if unvoice:
                filename = t.mix_path.split("/")[-1].split(".wav")[0] + "_no_vocal.wav"
                prefix = "/".join(t.mix_path.split("/")[:-1])
                paths.append(_get_path(t, prefix+filename))
            else:
                paths.append(t.mix_path)
    return paths

def chunk_audio(audio, t_slice, n_fft, hop_length, n_random):
    a_len = len(audio)
    bnd = a_len - t_slice // 2
    results = []

    for i in range(0, a_len, t_slice):
        chunk = audio[i:i+t_slice]
        if len(chunk) < t_slice:
            chunk = np.pad(chunk, (0, t_slice -len(chunk)), "constant")
        # remove dc components, for that it's not important for our task at all
        stft = np.delete(librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length), (0), axis=0)
        results.append(stft)

        # along with one extra random chunk
        for j in range(n_random):
            b = np.random.randint(0, bnd)
            chunk = audio[b:b+t_slice]
            if len(chunk) < t_slice:
                chunk = np.pad(chunk, (0, t_slice -len(chunk)), "constant")
            stft = np.delete(librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length), (0), axis=0)
            results.append(stft)
    return results

def compute_feature(n_fft, hop_length, genres, chunk, rsr, unvoice, n_random, n_val):
    genres = genres.split(",")
    t_slice = int(chunk * rsr)

    for genre in genres:
        print("Searching for {}".format(genre))
        dataset = mdb.load_all_multitracks()
        paths = get_paths(dataset, genre, unvoice)

        if len(paths) == 0:
            print("{} genre not found.".format(genre))
            continue

        print("Extracting {}".format(genre))
        cks = []
        start = time.time()
        for path in paths:
            mix = librosa.load(path, sr=44100)[0]
            mix = librosa.resample(mix, orig_sr=44100, target_sr=rsr)
            cks.extend(chunk_audio(mix, t_slice, n_fft, hop_length, n_random))
            print("{} extracted.".format(path.split("/")[-1]))
        print("{} chunks are generated for {} in {} secs.".format(len(cks), genre, time.time()-start))
        cks = np.array(cks)
        val_idx = np.random.choice(cks.shape[0], n_val, replace=False)
        cks_v = cks[val_idx]
        cks_t = cks[cks not in val_idx]
        np.save("./output/{}_val.npy".format(genre), cks_v)
        np.save("./output/{}_train.npy".format(genre), cks_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select the genre(s) for extracting feature")
    parser.add_argument("--n_fft", default=2048, type=int, help="n_fft")
    parser.add_argument("--hop_length", default=512, type=int, help="hop_length")
    parser.add_argument("--chunk", default=8.16, type=float, help="size of data (in second)")
    parser.add_argument("--n_random", default=30, type=int, help="number of randomly generated clip for each chunk")
    parser.add_argument("--unvoice", default=True, action="store_false", help="use unvoice version of tracks")
    parser.add_argument("--rsr", default=16000, type=int, help="sample rate after being resampled")
    parser.add_argument("--n_val", default=1000, type=int, help="sample rate after being resampled")
    parser.add_argument("--genres", required=True, type=str, help="genres: comma separate")
    args = vars(parser.parse_args())
    compute_feature(**args)

