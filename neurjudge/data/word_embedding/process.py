import json
import numpy as np
from tqdm import tqdm

word2id = json.load(open("../laic/word2id.json"))
print("word count:", len(word2id))

word2vec = {}
with open("sgns.wiki.word") as f:
    for line in tqdm(f.readlines()):
        if len(line) < 100:
            continue
        line = line.strip().replace("\n", "")
        w = line.split(" ")[0]
        emb = line.split(" ")[1:]
        emb = [float(i) for i in emb]
        word2vec[w] = emb

embeddings = []
cnt = 0
for w, i in tqdm(word2id.items()):
    emb = [0] * 300
    if w in word2vec.keys():
        emb = word2vec[w]
        # print(w)
        cnt += 1
    embeddings.append(emb)
embeddings = np.array(embeddings)
np.savetxt("laic_embeddings.txt", embeddings)
print("word count:", len(word2id))
print(cnt)
