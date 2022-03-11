
from calendar import c
import json
import jieba
from tqdm import tqdm


def getcharge2id(type="charge"):
    charges = set()
    with open("cail/processed/train.json") as f:
        for line in f.readlines():
            line = json.loads(line)

            if type=="charge":
                charge = line['meta']['accusation'][0].replace("[","").replace("]","")
                charges.add(charge)
            elif type=="article":
                article = line['meta']['relevant_articles'][0]
                article = str(article)
                charges.add(article)

    charge2id = {}
    id2charge = {}
    idx = 0
    for c in charges:
        charge2id[c] = idx
        id2charge[str(idx)] = c
        idx += 1
    with open("{}2id.json".format(type), "w") as f:
        json.dump(charge2id, f, ensure_ascii=False)
    with open("id2{}.json".format(type), "w") as f:
        json.dump(id2charge, f, ensure_ascii=False)


class Vocab:
    def __init__(self):
        self.word2id = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.id2word = {0: "[PAD]", 1: "[SOS]", 2: "[EOS]", 3: "[UNK]"}
        self.word2count = {"[PAD]": 0, "[SOS]": 0, "[EOS]": 0, "[UNK]": 0}

    def addword(self, w):
        if w not in self.word2id.keys():
            idx = len(self.word2id)
            self.word2id[w] = idx
            self.id2word[idx] = w

            self.word2count[w] = 1
        else :
            self.word2count[w] += 1

def getword2id():
    file_path = "laic/train.json"
    corpus = []
    with open(file_path) as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            corpus.append(json_obj["fact"])

    vocab=Vocab()
    for line in tqdm(corpus):
        line = jieba.lcut(line)
        for w in line:
            vocab.addword(w)

    vocab2=Vocab()
    for w in vocab.word2count.keys():
        if vocab.word2count[w]<=10:
            continue
        vocab2.addword(w)
    
    print("word count:",len(vocab2.word2id))

    with open("laic/word2id.json", "w") as f:
        json.dump(vocab2.word2id, f, ensure_ascii=False)
    with open("laic/id2word.json", "w") as f:
        json.dump(vocab2.id2word, f, ensure_ascii=False)


def gettime2id():
    time2id = json.load(open("time2id.json"))
    cur_id = 0
    new_time2id = {}
    for time in range(400):
        if str(time) in time2id.keys():
            cur_id = time2id[str(time)]
        new_time2id[str(time)] = cur_id
    with open("tim2id_new.json", "w") as f:
        json.dump(new_time2id, f)
    # print(new_time2id)


# getcharge2id("charge")
# getcharge2id("article")

getword2id()