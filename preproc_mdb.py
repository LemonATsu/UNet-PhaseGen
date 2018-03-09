import medleydb as mdb
import numpy as np
import librosa
import argparse
import time
import os.path
from medleydb.mix import mix_no_vocals, VOCALS, mix_multitrack

def test_and_gen_mix(track, fn, stem_indices):
    if not os.path.exists(fn):
        print("{} does not exist, generate one.".format(fn))
        mix_multitrack(track, fn, stem_indices=stem_indices)
    return fn

def get_metadata(track, unvoice, melody, bass):
    stems = track.stems
    non_voice_stem_indices = []
    melody_stem_indices = []
    bass_stem_indices = []

    unvoice_fn = track.mix_path if unvoice is False else track.mix_path.split(".wav")[0] + "_no_vocal.wav"
    melody_fn = track.mix_path.split(".wav")[0] + "_melody.wav"
    bass_fn = track.mix_path.split(".wav")[0] + "_bass.wav"

    has_vocal = False
    metadata = {
        "mix_path": track.mix_path,
        "melody_path": None,
        "unvoice_path": None,
        "bass_path": None,
        "instruments": track.stem_instruments if unvoice is False \
              else [i for i in track.stem_instruments if i is not "vocal"],
        "genre": track.genre
    }

    # search and append instrumental stems
    for i in stems.keys():
        not_vocal = all([inst not in VOCALS for inst in stems[i].instrument])

        # For generating unvoiced mix
        if not_vocal:
            non_voice_stem_indices.append(i)
        else:
            has_vocal = True

        # For generating melody only mix
        if stems[i].component == "melody" and melody:
            melody_stem_indices.append(i)

        if stems[i].component == "bass" and bass:
            bass_stem_indices.append(i)

    if melody:
        metadata["melody_path"] = test_and_gen_mix(track, melody_fn, melody_stem_indices)

    if bass:
        metadata["bass_path"] = test_and_gen_mix(track, bass_fn, bass_stem_indices)

    if has_vocal and unvoice:
        metadata["unvoice_path"] = test_and_gen_mix(track, unvoice_fn, non_voice_stem_indices)
    else:
        metadata["unvoice_path"] = track.mix_path

    return metadata

def chunk_audio(audio, t_slice, n_fft, hop_length, n_random):
    # expect audio to be an np array
    a_len = np.min([len(a) for a in audio])
    audio = np.array([a[:a_len] for a in audio])
    bnd = a_len - t_slice // 1.3
    results = []

    for i in range(0, a_len, t_slice):
        stfts = _chunk_and_stft(audio, i, t_slice, n_fft, hop_length)
        results.append(stfts)

        for j in range(n_random):
            b = np.random.randint(0, bnd)
            stfts = _chunk_and_stft(audio, b, t_slice, n_fft, hop_length)
            results.append(stfts)

    return results

def _chunk_and_stft(audio, start, t_slice, n_fft, hop_length):
    stfts = []
    chunk = audio[:, start:start+t_slice]
    if len(chunk[0]) < t_slice:
        chunk = np.pad(chunk, [(0, 0), (0, t_slice - len(chunk[0]))], "constant")
        # print("padded {}".format(chunk.shape))

    for c in chunk:
        # remove dc components, for that it's not important for our task at all
        stft = np.delete(librosa.stft(c, n_fft=n_fft, hop_length=hop_length), (0), axis=0)
        r, i = np.real(stft), np.imag(stft)
        stft = np.concatenate([r[np.newaxis, ...], i[np.newaxis, ...]], axis=0)
        stfts.append(stft)
    return stfts

def get_gid(genre, genres):
    for i, x in enumerate(genres):
        if x == genre:
            return i
    raise "Undefined genre!"

def get_mix_chunks(fn, t_slice, n_fft, hop_length, n_random, rsr, osr=44100):
    # we use tuple here to avoid confusion
    if not isinstance(fn, tuple):
        fn = (fn,)
    mix = []

    for f in fn:
        m = librosa.load(f, sr=osr)[0]
        if rsr != osr:
            m = librosa.resample(m, orig_sr=osr, target_sr=rsr)
        mix.append(m)
    return chunk_audio(np.asarray(mix), t_slice, n_fft, hop_length, n_random)

def get_instruments_attrs(instruments, instruments_dict, n):
    onehot = [0] * len(instruments_dict)
    for i in instruments:
        onehot[instruments_dict[i]] = 1
    return [onehot] * n


def compute_feature(n_fft, hop_length, genres, chunk, rsr, unvoice, melody, bass,
                        n_random, n_val, get_attr):
    # init
    genres = genres.split(",")
    t_slice = int(chunk * rsr)
    tracks = mdb.load_all_multitracks()
    instruments_dict = {}
    train_set = {}
    val_set = {}
    tracks_meta = [[] for i in range(len(genres))]
    idx_cnt = 0

    print("Reading the metadata and build dictionary for instruments ...")
    for t in tracks:
        if t.genre in genres:
            # get the mix audio as well as build a table for instrumental attributes
            metadata = get_metadata(t, unvoice, melody, bass)
            if get_attr:
                for instrument in metadata["instruments"]:
                    if instrument not in instruments_dict.keys():
                        instruments_dict[instrument] = idx_cnt
                        idx_cnt += 1
            tracks_meta[get_gid(t.genre, genres)].append(metadata)

    print("Done! Start generating features ...")
    target_path = "mix_path" if not unvoice else "unvoice_path"
    # generate data for each genre
    for gid, g in enumerate(genres):
        start = time.time()
        audio_chunks = []
        instruments_attrs = []
        print("Generating features for {} ...".format(g))
        file_path = "{}.hdf5".format(g)

        for md in tracks_meta[gid]:
            print("Converting {} ...".format(md[target_path]))
            audio = (md[target_path],)
            audio += (md["melody_path"],) if melody else () # append the melody part if needed.
            audio += (md["bass_path"],) if bass else ()
            # get the atual mix, and turn them into stft.
            chunks = get_mix_chunks(audio, t_slice, n_fft, hop_length, n_random[gid], rsr)
            audio_chunks.extend(chunks)
            print("{} clips for {}.".format(len(chunks), md[target_path]))
            if get_attr:
                attrs = get_instruments_attrs(md["instruments"], instruments_dict, len(chunks))
                instruments_attrs.extend(attrs)

        # get random indices
        print("We have {} clips for {}.".format(len(audio_chunks), g))
        idx = np.linspace(0, len(audio_chunks)-1, len(audio_chunks), dtype=int)
        np.random.shuffle(idx)

        audio_chunks = np.asarray(audio_chunks, dtype=np.float32)
        print("shape: {}, type: {}".format(audio_chunks.shape, audio_chunks.dtype))
        if audio_chunks.shape[1] == 1:
            audio_chunks = np.squeeze(audio_chunks, axis=1)
        print(audio_chunks.shape)
        audio_chunks = (audio_chunks - audio_chunks.mean()) / audio_chunks.std()
        train_set["audio"] = audio_chunks[idx][n_val:, ...]
        val_set["audio"] = audio_chunks[idx][:n_val, ...]

        print(train_set["audio"].shape)
        print(val_set["audio"].shape)

        if get_attr:
            train_set["attrs"] = np.asarray(instruments_attrs)[idx][n_val:, ...]
            val_set["attrs"] = np.asarray(instruments_attrs)[idx][:n_val, ...]

        print("Saving ...")
        for k in train_set.keys():
            np.save("./output/{}_{}_val.npy".format(g, k), val_set[k])
            np.save("./output/{}_{}_train.npy".format(g, k), train_set[k])

        print("Generation for {} is complete, {} sec elapsed.".format(g, time.time()-start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select the genre(s) for extracting feature")
    parser.add_argument("--n_fft", default=2048, type=int, help="n_fft")
    parser.add_argument("--hop_length", default=512, type=int, help="hop_length")
<<<<<<< HEAD
    parser.add_argument("--chunk", default=4.064, type=float, help="size of data (in second)")
    parser.add_argument("--n_random", default=[30], nargs="+", type=int, help="number of randomly generated clip for each chunk")
    parser.add_argument("--unvoice", default=False, action="store_true", help="use unvoice version of tracks")
    parser.add_argument("--melody", default=False, action="store_true", help="use melody tracks")
    parser.add_argument("--bass", default=False, action="store_true", help="use bass track")
    parser.add_argument("--rsr", default=16000, type=int, help="sample rate after being resampled")
    parser.add_argument("--n_val", default=1000, type=int, help="sample rate after being resampled")
    parser.add_argument("--genres", required=True, type=str, help="genres: comma separate")
    parser.add_argument("--get_attr", action="store_true", help="extract the instruments as one-hot vector")
    args = vars(parser.parse_args())
    compute_feature(**args)

