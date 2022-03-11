import numpy as np
import json

embeddings = np.loadtxt("data/word_embedding/embeddings.txt")
word2id = json.load(open('data/word2id.json'))
print(embeddings.shape)
print(len(word2id))