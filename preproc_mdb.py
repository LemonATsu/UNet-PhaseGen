import medleydb as mdb
import numpy as np
import librosa
import argparse
import time
import os.path
from medleydb.mix import mix_no_vocals, VOCALS, mix_multitrack

def get_metadata(track, unvoice):
    stems = track.stems
    stem_indices = []
    filename = track.mix_path if unvoice is False else track.mix_path.split(".wav")[0] + "_no_vocal.wav"
    has_vocal = False
    metadata = {
        "filename": None,
        "instruments": track.stem_instruments if unvoice is False \
              else [i for i in track.stem_instruments if i is not "vocal"],
        "genre": track.genre
    }

    if unvoice:
        # search and append instrumental stems
        for i in stems.keys():
            not_vocal = all([inst not in VOCALS for inst in stems[i].instrument])
            if not_vocal:
                stem_indices.append(i)
            else:
                has_vocal = True

        if has_vocal:
            if not os.path.exists(filename):
                print("Unvoice version for {} does not exist, generate one.".format(filename))
                mix_multitrack(track, filename, stem_indices=stem_indices) # create mix track
            metadata["filename"] = filename

    if metadata["filename"] is None:
        metadata["filename"] = track.mix_path

    return metadata



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

def get_gid(genre, genres):
    for i, x in enumerate(genres):
        if x == genre:
            return i
    raise "Undefined genre!"

def get_mix_chunks(filename, t_slice, n_fft, hop_length, n_random, rsr):
    mix = librosa.load(filename, sr=44100)[0]
    if rsr != 44100:
        mix = librosa.resample(mix, orig_sr=44100, target_sr=rsr)
    return chunk_audio(mix, t_slice, n_fft, hop_length, n_random)

def get_instruments_attrs(instruments, instruments_dict, n):
    onehot = [0] * len(instruments_dict)
    for i in instruments:
        onehot[instruments_dict[i]] = 1
    return [onehot] * n


def compute_feature(n_fft, hop_length, genres, chunk, rsr, unvoice, n_random, n_val, instruments_attr):
    genres = genres.split(",")
    t_slice = int(chunk * rsr)
    tracks = mdb.load_all_multitracks()
    instruments_dict = {}
    tracks_meta = [[] for i in range(len(genres))]
    #audio_chunks = [[] for i in range(len(genres))] # init list for each genre
    #instruments_attrs = [[] for i in range(len(genres))] # init list for each genre
    idx_cnt = 0

    print("Reading the metadata and build dictionary for instruments ...")
    for t in tracks:
        if t.genre in genres:    
            # get the mix audio as well as build a table for instrumental attributes
            metadata = get_metadata(t, unvoice)
            for instrument in metadata["instruments"]:
                if instrument not in instruments_dict.keys():
                    instruments_dict[instrument] = idx_cnt
                    idx_cnt += 1
            tracks_meta[get_gid(t.genre, genres)].append(metadata)
    print(instruments_dict)

    print("Done! Start generating features ...")

    for gid, g in enumerate(genres):
        start = time.time()
        audio_chunks = []
        instruments_attrs = []
        train_set = {}
        val_set = {}
        print("Generating features for {} ...".format(g))
        for md in tracks_meta[gid]:
            chunks = get_mix_chunks(md["filename"], t_slice, n_fft, hop_length, n_random, rsr)
            attrs = get_instruments_attrs(md["instruments"], instruments_dict, len(chunks))
            audio_chunks.extend(chunks)
            instruments_attrs.extend(attrs)

        print("We have {} clips for {}.".format(len(audio_chunks), g))
        idx = np.linspace(0, len(audio_chunks)-1, len(audio_chunks), dtype=int)
        np.random.shuffle(idx)

        train_set =  {
            "audio": np.asarray(audio_chunks)[idx][n_val:, ...], 
            "attrs": np.asarray(instruments_attrs)[idx][n_val:, ...]
        }
        val_set =  {
            "audio": np.asarray(audio_chunks)[idx][:n_val, ...], 
            "attrs": np.asarray(instruments_attrs)[idx][:n_val, ...]
        }
        print("Saving ...")
        for k in train_set.keys():
            np.save("./output/{}_{}_val.npy".format(g, k), val_set[k])
            np.save("./output/{}_{}_train.npy".format(g, k), train_set[k])

        print("Generation for {} is complete, {} sec elapsed.".format(g, time.time()-start))


    """
    for md in tracks_meta:
        gid = get_gid(md["genre"], genres)
        chunks = get_mix_chunks(md["filename"], t_slice, n_fft, hop_length, n_random, rsr)
        attrs = get_instruments_attrs(md["instruments"], instruments_dict, len(chunks))
        audio_chunks[gid].extend(chunks)
        instruments_attrs[gid].extend(attrs)
    print("Feature generation done! {} sec elapsed ...", time.time()-start)

    for i, (genre, chunks, attrs) in enumerate(zip(genres, np.asarray(audio_chunks), np.asarray(instruments_attrs))):
        # for each genre
        print("We have {} clips for {}.".format(len(chunks), genre))
        idx = np.linspace(0, len(chunks)-1, len(chunks), dtype=int)
        np.random.shuffle(idx)
        train_set =  {"audio": np.asarray(chunks)[idx][n_val:, ...], "attrs": np.asarray(attrs)[idx][n_val:, ...]}
        val_set =  {"audio": np.asarray(chunks)[idx][:n_val, ...], "attrs": np.asarray(attrs)[idx][:n_val, ...]}
        print("Saving ...")
        np.save("./output/{}_val.npy".format(genre), val_set)
        np.save("./output/{}_train.npy".format(genre), train_set)

        print("Erasing array to save some memory space.")
        audio_chunks[i] = []
        instruments_attrs[i] = []
    """
        
        

"""
def get_metadata(tracks, genre, unvoice, instruments_attr):
    tracks_metadata = []
    for t in tracks:
        if t.genre == genre:
            tracks_metadata.append(_get_metadata(t, unvoice))
    return tracks_metadata
def compute_feature(n_fft, hop_length, genres, chunk, rsr, unvoice, n_random, n_val, instruments_attr):
    genres = genres.split(",")
    t_slice = int(chunk * rsr)

    for genre in genres:
        print("Searching for {}".format(genre))
        dataset = mdb.load_all_multitracks()
        tracks_metadata = get_metadata(dataset, genre, unvoice)

        if len(paths) == 0:
            print("{} genre not found.".format(genre))
            continue

        print("Extracting {}".format(genre))
        cks = []
        start = time.time()
        for md in tracks_metadata:
            # get mixed data
            mix = librosa.load(md["filename"], sr=44100)[0]
            mix = librosa.resample(mix, orig_sr=44100, target_sr=rsr)
            cks.extend(chunk_audio(mix, t_slice, n_fft, hop_length, n_random))
            print("{} extracted.".format(path.split("/")[-1]))
        print("{} chunks are generated for {} in {} secs.".format(len(cks), genre, time.time()-start))
        cks = np.array(cks)
        np.random.shuffle(cks)
        np.save("./output/{}_val.npy".format(genre), cks[:n_val, ...])
        np.save("./output/{}_train.npy".format(genre), cks[n_val:, ...])
"""

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
    parser.add_argument("--instruments-attr", action="store_true", help="extract the instruments as one-hot vector")
    args = vars(parser.parse_args())
    compute_feature(**args)

