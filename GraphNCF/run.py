import torch
from torch import nn as nn
from toyDataset.loaddata import load100KRatings
from wsdream.loaddata import loadWsdreamRt, loadWsdreamUserLocation, loadWsdreamItemLocationOrgin, loadServiceCountryAs, PerformanceComput
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from numpy import diag
from GraphNCF.GCFmodel import GCF
from torch.utils.data import DataLoader
from GraphNCF.dataPreprosessing import Wsdream
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
import os
from wsdream.DataSet import DataSet

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

dateset = DataSet('rt', 0.30)
data_train = dateset.train
data_test = dateset.test
data_shape = dateset.shape
userNum = data_shape[0]
itemNum = data_shape[1]


# rt, userNum, itemNum= loadWsdreamRt()
#
# rt['userId'] = rt['userId'].astype(int) - 1
# rt['itemId'] = rt['itemId'].astype(int) - 1

dist_u = loadWsdreamUserLocation()
dist_u['user1Id'] = dist_u['user1Id'].astype(int) - 1
dist_u['user2Id'] = dist_u['user2Id'].astype(int) - 1

dist_i = loadWsdreamItemLocationOrgin()
dist_i['item1Id'] = dist_i['item1Id'].astype(int)
dist_i['item2Id'] = dist_i['item2Id'].astype(int)
train_service_num = len(data_train['[Service ID]'].unique())
ucountryidx = {}
for i in range(len(data_train['[User Country]'])):
    ucountryidx[data_train['[User ID]'][i]] = data_train['[User Country]'][i]
    if len(ucountryidx) == userNum:
        break
uasidx = {}
for i in range(len(data_train['[User AS]'])):
    uasidx[data_train['[User ID]'][i]] = data_train['[User AS]'][i]
    if len(uasidx) == userNum:
        break
serviceToCA = loadServiceCountryAs()
ucountryidx = [ucountryidx[i] for i in range(userNum)]
uasidx = [uasidx[i] for i in range(userNum)]
icountryidx = serviceToCA['countryId'].values.tolist()
iasidx = serviceToCA['asId'].values.tolist()
del_item = []
for i in range(len(icountryidx)):
    if i not in range(itemNum):
        del_item.append(i)
for i in range(len(del_item)):
    icountryidx.pop()
    iasidx.pop()

uca = {'userCountry':ucountryidx,'userAs':uasidx}
ica = {'itemCountry':icountryidx,'itemAs':iasidx}
uca = pd.DataFrame(uca)
ica = pd.DataFrame(ica)
para = {
    'epoch':50,
    'lr':0.001,
    'batch_size':1024,
    'train':0.8,
    'embedding_size':80,
    'layers':[240,80,80,80],
    'location_layers':[80,80,80,80],
    'useLocation':True,
    'togetherEmbd':True
}
d_train = Wsdream(data_train)
d_train, _ = random_split(d_train,[len(d_train),0])
dl = DataLoader(d_train,batch_size=para['batch_size'],shuffle=True,pin_memory=True)


model = GCF(userNum, itemNum, data_train,dist_u,dist_i, uca,ica,para['embedding_size'], layers=[240,80,80,80],
            LocationLayers=[80,80,80,80],useLocation=para['useLocation'], togetherEmbedding=para['togetherEmbd']).cuda()
# model = SVD(userNum,itemNum,50).cuda()
# model = NCF(userNum,itemNum,64,layers=[128,64,32,16,8]).cuda()
optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
lossfn = MSELoss()
prediction_all = torch.empty(para['batch_size']).cuda()
label_all = torch.empty(para['batch_size']).cuda()
user_index = torch.empty(para['batch_size']).cuda()
for i in range(para['epoch']):

    for id,batch in enumerate(dl):
        print('epoch:',i,' batch:',id)
        optim.zero_grad()
        prediction = model(batch[0].cuda(), batch[1].cuda())
        loss = lossfn(batch[2].float().cuda(),prediction)
        loss.backward()
        optim.step()
        print('MSE:', loss)
        print('RMSE:', torch.sqrt(loss))

d_test = Wsdream(data_test)
test1Len = int(len(d_test)*0.5)
# d_test1, d_test2 = random_split(d_test,[test1Len,len(d_test)-test1Len])
# testdl1 = DataLoader(d_test1,batch_size=len(d_test1),)
# testdl2 = DataLoader(d_test2,batch_size=len(d_test2),)
d_test,_ = random_split(d_test,[len(d_test),0])
testdl = DataLoader(d_test,batch_size=int(len(d_test)*0.3))
prediction_all = torch.empty(int(len(d_test)*0.3)).cuda()
label_all = torch.empty(int(len(d_test)*0.3)).cuda()
user_index = torch.empty(int(len(d_test)*0.3)).cuda()
for id,batch in enumerate(testdl):
    print(' test_batch:',id)
    prediction = model(batch[0].cuda(),batch[1].cuda())
    loss = lossfn(batch[2].float().cuda(),prediction)
    print('test_loss:', loss) # MSEloss
    print('test_RMSELoss:', torch.sqrt(loss))
    prediction_all = torch.cat([prediction_all, prediction], dim=0)
    label_all = torch.cat([label_all, batch[2].float().cuda()], dim=0)
    user_index = torch.cat([user_index.long(), batch[0].cuda()], dim=0)

select_index = torch.LongTensor([i for i in range(int(len(d_test)*0.3), len(prediction_all))])
user_index = torch.index_select(user_index, 0, select_index.cuda())
label_all = torch.index_select(label_all, 0, select_index.cuda())
prediction_all = torch.index_select(prediction_all, 0, select_index.cuda())
top_label, top_prediction = PerformanceComput(user_index, label_all, prediction_all)
top_label = torch.FloatTensor(top_label)
top_prediction = torch.FloatTensor(top_prediction)
loss = lossfn(top_label.cuda(), top_prediction.cuda())
print('topK_loss:', loss)
print('topK_RMSELoss:', torch.sqrt(loss))


