import json
import pandas as pd
from tqdm import tqdm
import jieba


def chinese2arabic(cn: str) -> int:
    CN_NUM = {
        '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
        '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2, '两': 2,
    }

    CN_UNIT = {
        '十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '万': 10000, '萬': 10000, '亿': 100000000, '億': 100000000, '兆': 1000000000000,
    }
    unit = 0   # current
    ldig = []  # digest
    for cndig in reversed(cn):
        if cndig in CN_UNIT:
            unit = CN_UNIT.get(cndig)
            if unit == 10000 or unit == 100000000:
                ldig.append(unit)
                unit = 1
        else:
            dig = CN_NUM.get(cndig)
            if unit:
                dig *= unit
                unit = 0
            ldig.append(dig)
    if unit == 10:
        ldig.append(10)
    val, tmp = 0, 0
    for x in reversed(ldig):
        if x == 10000 or x == 100000000:
            val += tmp * x
            tmp = 0
        else:
            tmp += x
    val += tmp
    return val


def laic2cail(df, data_name):
    data = []
    for i in tqdm(range(len(df))):
        json_obj = {}

        charge = df["charge"][i]
        if charge == "非法生产、买卖、运输制毒物品、走私制毒物品罪":
            charge = "非法买卖制毒物品"
        charge = charge.replace("[", "").replace("]", "").replace("罪", "")

        article = df["article"][i]
        article = str(chinese2arabic(article))

        json_obj["fact"] = " ".join(jieba.lcut(df["justice"][i]))
        json_obj["meta"] = {
            'accusation': [charge],
            'relevant_articles': [article],
            'term_of_imprisonment': {
                'imprisonment': int(df["judge"][i]),
                'death_penalty': False,
                'life_imprisonment': False,
            }
        }
        data.append(json.dumps(json_obj))

    with open("{}.json".format(data_name), "w") as f:
        for line in data:
            f.write(line+"\n")


def trans_cail(data_name):
    charge_tong = json.load(open("../charge_tong.json"))
    article_tong = json.load(open("../art_tong.json"))

    data = []
    with open("{}.json".format(data_name)) as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            
            data.append(line)
    print(len(data))

    new_data = []
    for line in data:
        charge = line['meta']['accusation'][0]
        article = line['meta']['relevant_articles'][0]
        if charge not in charge_tong.keys():
            continue
        if article not in article_tong.keys():
            continue
        new_data.append(line)
    print(len(new_data))

    with open("{}.json".format(data_name), "w") as f:
        for line in new_data:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line+"\n")


df_train = pd.read_csv("train.csv")
df_valid = pd.read_csv("valid.csv")
df_test = pd.read_csv("test.csv")

# laic2cail(df_train,"train")
# laic2cail(df_train, "train")
# trans_cail("train")

laic2cail(df_valid, "valid")
trans_cail("valid")

# laic2cail(df_test, "test")
# trans_cail("test")