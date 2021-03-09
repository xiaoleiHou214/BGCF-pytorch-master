import pandas as pd
from os import path
from GraphNCF.dataPreprosessing import Wsdream
from wsdream.DataSet import DataSet
import numpy as np

path = path.dirname(__file__) + r'/data/'
# fileName = 'itemLocation.csv'
# filePath = path + fileName
# itemLocation = pd.read_csv(filePath)
# print(itemLocation[0:5])
# print(itemLocation.columns)
# itemLocation['item1Id'] = itemLocation['item1Id'].astype(int) - 1
# itemLocation['item2Id'] = itemLocation['item2Id'].astype(int) - 1
# print(itemLocation[0:5])
#
# dataset = DataSet('rt', 0.30)
# data_train = dataset.train
# data_test = dataset.test
# data_shape = dataset.shape
# userNum = data_shape[0]
# itemNum = data_shape[1]
# print(len(dataset.data['[Service ID]'].unique()))
# serviceId = dataset.data['[Service ID]'].unique()
# dele = []
# for i in range(len(itemLocation)):
#     if (itemLocation['item1Id'][i] not in serviceId) | (itemLocation['item2Id'][i] not in serviceId):
#         dele.append(i)
#
# print(len(dele))
# print(len(itemLocation))
# itemLocation.drop(dele,inplace=True)
# print(len(itemLocation))
# print(dataset.data['[Service ID]'].max())
# print(itemLocation['item1Id'].max())
# fileName = 'itemLocation_Orgin.csv'
# file_path_w = path + fileName
# itemLocation.to_csv(file_path_w,index=0)
fileName = 'wslist.txt'
file_path = path + fileName
resopnse = []
with open(file_path, encoding='gbk') as f:
    i = 0
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split('\t')
            resp = [i for i in l[:]]
            resopnse.append(resp)
resopnse = np.array(resopnse)
resopnse = np.delete(resopnse, [0, 1], axis=0)

serviceId = []
countryId = []
asId = []
for i in range(len(resopnse)):
    serviceId.append(int(resopnse[i][0]))
    countryId.append(str(resopnse[i][4]))
    asId.append(str(resopnse[i][6]))

serviceToCA = {'id':serviceId,'countryId':countryId,'asId':asId}
serviceToCA = pd.DataFrame(serviceToCA)
countrylist = serviceToCA['countryId'].unique()
aslist = serviceToCA['asId'].unique()
counDict = {}
for i in range(len(countrylist)):
    counDict[countrylist[i]] = i
asDict = {}
for i in range(len(aslist)):
    asDict[aslist[i]] = i

for i in range(len(serviceToCA)):
    serviceToCA['countryId'][i] = counDict[serviceToCA['countryId'][i]]
    serviceToCA['asId'][i] = asDict[serviceToCA['asId'][i]]
fileName = 'serviceCA.csv'
file_path_w = path + fileName
serviceToCA.to_csv(file_path_w,index=0)