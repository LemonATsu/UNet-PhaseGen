import medleydb as mdb

genres = ["Jazz", "Pop"]
types = [{}, {}]

for j, genre in enumerate(genres):
    tracks = mdb.load_all_multitracks()
    for t in tracks:
        if t.genre == genre:
            for i in t.stem_instruments:
                types[j][i] = 1
    print("for {}, we have {} different instruments:".format(genre, len(types[j])))
    for t in types[j]:
        print(t)

                
