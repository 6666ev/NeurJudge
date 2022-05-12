import json
import jieba
from tqdm import tqdm
import pandas as pd


charge_tong = {}
with open("charge_tong.json") as f:
    charge_tong = json.load(f)

charges = charge_tong.keys()

charge2idx = {}
idx2charge = {}
for i in range(len(charges)):
    charge2idx[charges[i]] = i
    idx2charge[str(i)] = charges[i]

with open("charge2id.json","w") as f:
    json.dump(charge2idx, ensure_ascii=False)
with open("id2charge.json","w") as f:
    json.dump(idx2charge, ensure_ascii=False)

art_tong = {}
with open("art_tong.json") as f:
    art_tong = json.load(f)

articles = art_tong.keys()

article2idx = {}
idx2article = {}
for i in range(len(articles)):
    article2idx[articles[i]] = i
    idx2article[str(i)] = articles[i]

with open("article2id.json","w") as f:
    json.dump(article2idx, ensure_ascii=False)
with open("id2article.json","w") as f:
    json.dump(idx2article, ensure_ascii=False)