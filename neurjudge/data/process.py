
from calendar import c
import json
import jieba
from tqdm import tqdm


def getcharge2id():
    charges = set()
    with open("cail/final_all_data/exercise_contest/data_valid.json") as f:
        for line in f.readlines():
            line = json.loads(line)
            charge = line['meta']['accusation']
            for c in charge:
                c = c.replace("[","").replace("]","")
                charges.add(c)
    charge2id = {}
    id2charge = {}
    idx = 0
    for c in charges:
        charge2id[c] = idx
        id2charge[str(idx)]=c
        idx += 1
    with open("charge2id.json", "w") as f:
        json.dump(charge2id, f, ensure_ascii=False)
    with open("id2charge.json", "w") as f:
        json.dump(id2charge, f, ensure_ascii=False)


def getword2id():
    word2id = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
    id2word = {0: "[PAD]", 1: "[SOS]", 2: "[EOS]", 3: "[UNK]"}
    file_path = "cail/final_all_data/exercise_contest/data_valid.json"
    corpus = []
    with open(file_path) as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            corpus.append(json_obj["fact"])

    for line in tqdm(corpus):
        line = jieba.lcut(line)
        for w in line:
            if w not in word2id.keys():
                idx = len(word2id)
                word2id[w] = idx
                id2word[idx] = w

    with open("word2id.json", "w") as f:
        json.dump(word2id, f, ensure_ascii=False)
    with open("id2word.json", "w") as f:
        json.dump(id2word, f, ensure_ascii=False)


getcharge2id()
