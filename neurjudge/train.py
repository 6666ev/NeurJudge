import json
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from utils import Data_Process
import torch
import torch.nn as nn
from model import NeurJudge
import logging
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os
from test import do_test
# from word2vec import toembedding

# trainset_path = "data/cail/processed/train.json"
trainset_path = "data/laic/train.json"
model_save_dir = "logs/laic/"
embeddings = np.loadtxt("data/word_embedding/laic_embeddings.txt")
batch_size = 128

random.seed(42)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#embedding = toembedding()
# embedding = 1
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeurJudge(embeddings)
model = model.to(device)

data_all = []

f = open(trainset_path,'r')

for index,line in enumerate(f):
    data_all.append(line)

random.shuffle(data_all)

print(len(data_all))
dataloader = DataLoader(data_all, batch_size = batch_size, shuffle=True, num_workers=0, drop_last=False)
criterion = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

process = Data_Process()
legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()

legals,legals_len,arts,arts_sent_lent = legals.to(device),legals_len.to(device),arts.to(device),arts_sent_lent.to(device)
num_epoch = 16
global_step = 0

def save_embeddings(model):
    emb = model.embs.weight.data
    emb = emb.cpu().numpy()
    np.savetxt("logs/laic/embeddings/emb.txt",emb)

for epoch in range(num_epoch):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    logger.info("Training Epoch: {}/{}".format(epoch+1, int(num_epoch)))

    # save_embeddings(model)
    # PATH = model_save_dir+'_neural_judge_'+str(epoch)
    # torch.save(model.state_dict(), PATH)
    # do_test('_neural_judge_'+str(epoch))

    tq = tqdm(dataloader)
    for step,batch in enumerate(tq):
        global_step += 1
        nb_tr_steps += 1
        model.train()
        optimizer.zero_grad()

        charge_label,article_label,time_label,documents,sent_lent = process.process_data(batch)
       
        documents = documents.to(device)
        charge_label = charge_label.to(device)
        article_label = article_label.to(device)
        time_label = time_label.to(device)
       
        sent_lent = sent_lent.to(device)
        charge_out,article_out,time_out = model(legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art,documents,sent_lent,process,device)
        
        loss_charge = nn.CrossEntropyLoss()(charge_out,charge_label)
        loss_art = nn.CrossEntropyLoss()(article_out,article_label)
        loss_time = nn.L1Loss()(time_out.squeeze(),time_label.squeeze())

        loss = ( loss_charge + loss_art + loss_time )/3
        tr_loss+=loss_time.item()
        loss.backward()
        optimizer.step()

        tq.set_postfix(loss_term=loss_time.item())
        # if global_step%1000 == 0:
        #     logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))
    save_embeddings(model)
    PATH = model_save_dir+'_neural_judge_'+str(epoch)
    torch.save(model.state_dict(), PATH)

    do_test('_neural_judge_'+str(epoch))


