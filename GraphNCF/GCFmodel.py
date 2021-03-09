import torch
import torch.nn as nn
from torch.nn import Module
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import numpy as np


# several models for recommendations

# RMSE
# SVD dim = 50 50 epoch RMSE = 0.931
# GNCF dim = 64 layer = [64,64,64] nn = [128,64,32,] 50 epoch RMSE = 0.916/RMSE =0.914
# NCF dim = 64 50 nn = [128,54,32] epoch 50 RMSE = 0.928

class GNNLayer(Module):

    def __init__(self,inF,outF):

        super(GNNLayer,self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF,out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF,out_features=outF)
        self.leaky1 = torch.nn.LeakyReLU()
        self.leaky2 = torch.nn.LeakyReLU()

    def forward(self, laplacianMat,selfLoop,features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # laplacianMat L = D^-1(A)D^-1 # 拉普拉斯矩阵
        L1 = laplacianMat + selfLoop
        L2 = laplacianMat.cuda()
        L1 = L1.cuda()
        inter_feature = torch.mul(features,features)

        inter_part1 = self.leaky1(self.linear(torch.sparse.mm(L1,features)))
        inter_part2 = self.leaky2(self.interActTransform(torch.sparse.mm(L2,inter_feature)))

        return inter_part1+inter_part2

class GCF(Module):

    def __init__(self,userNum,itemNum,rt,dist_u,dist_i,uca,ica,embedSize=100,layers=[100,80,50],LocationLayers=[80,80,80,80],useLocation=True,togetherEmbedding=True,useCuda=True):

        super(GCF,self).__init__()
        self.useCuda = useCuda
        self.userNum = userNum
        self.itemNum = itemNum
        self.uEmbd = nn.Embedding(userNum,embedSize)
        self.iEmbd = nn.Embedding(itemNum,embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.LaplacianMat = self.buildLaplacianMat(rt) # sparse format
        self.leakyRelu = nn.LeakyReLU()
        self.rt = rt
        self.uca = uca
        self.ica = ica
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)
        self.useLocation = useLocation
        self.ulEmbd = nn.Embedding(userNum, embedSize)
        self.ilEmbd = nn.Embedding(itemNum, embedSize)
        self.GNNlayersForUL = torch.nn.ModuleList()
        self.GNNlayersForIL = torch.nn.ModuleList()
        self.LaplacianMatForUL = self.buildLaplacianMatForUserLocation(dist_u)
        self.LaplacianMatForIL = self.buildLaplacianMatForItemLocation(dist_i)
        self.selfLoopForUL = self.getSparseEye(self.userNum + self.userNum)
        self.selfLoopForIL = self.getSparseEye(self.itemNum + self.itemNum)
        self.togetherEmbedding = togetherEmbedding
        self.user_countryNum = rt['[User Country]'].max()+1
        self.user_acNum = rt['[User AS]'].max()+1
        self.item_countryNum = ica['itemCountry'].max()+1
        self.item_acNum = ica['itemAs'].max()+1
        self.user_country_embedding = nn.Embedding(self.user_countryNum, embedSize)
        self.user_ac_embedding = nn.Embedding(self.user_acNum, embedSize)
        self.item_country_embedding = nn.Embedding(self.item_countryNum, embedSize)
        self.item_ac_embedding = nn.Embedding(self.item_acNum, embedSize)
        linear_input = sum(layers)*2
        if self.useLocation:
            linear_input = linear_input + sum(LocationLayers)*2
        self.transForm1 = nn.Linear(in_features=linear_input,out_features=256)
        self.transForm2 = nn.Linear(in_features=256,out_features=128)
        self.transForm3 = nn.Linear(in_features=128,out_features=64)
        self.transForm4 = nn.Linear(in_features=64, out_features=32)
        self.transForm5 = nn.Linear(in_features=32, out_features=1)
        for From,To in zip(layers[:-1],layers[1:]):
            self.GNNlayers.append(GNNLayer(From,To))
        for From, To in zip(LocationLayers[:-1], LocationLayers[1:]):
            self.GNNlayersForUL.append(GNNLayer(From,To))
            self.GNNlayersForIL.append(GNNLayer(From,To))

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def buildLaplacianMat(self,rt):

        rt_item = rt['[Service ID]'] + self.userNum
        uiMat = coo_matrix((rt['[RT]'], (rt['[User ID]'], rt['[Service ID]'])))

        uiMat_upperPart = coo_matrix((rt['[RT]'], (rt['[User ID]'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def buildLaplacianMatForUserLocation(self, dist_u):
        dist_user = dist_u['user2Id'] + self.userNum
        uiMat = coo_matrix((dist_u['dist'], (dist_u['user1Id'], dist_u['user2Id'])), shape=(self.userNum, self.userNum))


        uiMat_upperPart = coo_matrix((dist_u['dist'], (dist_u['user1Id'], dist_user)), shape=(self.userNum, self.userNum
                                                                                              + self.userNum))
        uiMat = uiMat.transpose()
        uiMat.resize((self.userNum, self.userNum + self.userNum))

        A = sparse.vstack([uiMat_upperPart, uiMat])
        selfLoop = sparse.eye(self.userNum + self.userNum)
        sumArr = (A > 0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag, -0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        print(row,col)
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size([self.userNum+self.userNum, self.userNum+self.userNum]))
        return SparseL

    def buildLaplacianMatForItemLocation(self,dist_i):

        dist_item = dist_i['item2Id'] + self.itemNum
        uiMat = coo_matrix((dist_i['dist'], (dist_i['item1Id'], dist_i['item2Id'])), shape=(self.itemNum, self.itemNum))

        uiMat_upperPart = coo_matrix((dist_i['dist'], (dist_i['item1Id'], dist_item)), shape=(self.itemNum, self.itemNum
                                                                                              + self.itemNum))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.itemNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.itemNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size([self.itemNum+self.itemNum, self.itemNum+self.itemNum]))
        return SparseL

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])

        ucountryidx = torch.LongTensor(self.uca['userCountry'].values.tolist())
        uasidx = torch.LongTensor(self.uca['userAs'].values.tolist())
        icountryidx = torch.LongTensor(self.ica['itemCountry'].values.tolist())
        iasidx = torch.LongTensor(self.ica['itemAs'].values.tolist())
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()
            ucountryidx = ucountryidx.cuda()
            uasidx = uasidx.cuda()
            icountryidx = icountryidx.cuda()
            iasidx = iasidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)

        ucountryEmbd = self.user_country_embedding(ucountryidx)
        uasEmbd = self.user_ac_embedding(uasidx)
        icountryEmbd = self.item_country_embedding(icountryidx)
        iasEmbd = self.item_ac_embedding(iasidx)

        iLocationFeatures = torch.cat([icountryEmbd, iasEmbd], dim=1)
        uLocationFeatures = torch.cat([ucountryEmbd,uasEmbd],dim=1)
        uiLocationFeatures = torch.cat([uLocationFeatures,iLocationFeatures],dim=0)

        if self.togetherEmbedding:
            final_features = torch.cat([features,uiLocationFeatures],dim=1)

        else:
            final_features = features


        userLocationEmbd = self.ulEmbd(uidx)
        ul_features = torch.cat([userLocationEmbd, userLocationEmbd], dim=0)

        itemLocationEmbd = self.ilEmbd(iidx)
        il_features = torch.cat([itemLocationEmbd, itemLocationEmbd], dim=0)

        return final_features, ul_features, il_features


    def forward(self,userIdx,itemIdx):

        itemLoctionIdx = itemIdx
        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
        # gcf data propagation
        # user and item
        features, ul_features, il_features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            #features = nn.ReLU()(features)
            features = torch.nn.functional.normalize(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        #if self.useLocation:
        # user location
        ul_finalEmbd = ul_features.clone()
        for gnn in self.GNNlayersForUL:
            ul_features = gnn(self.LaplacianMatForUL,self.selfLoopForUL,ul_features)
            #ul_features = nn.ReLU()(ul_features)
            ul_features = torch.nn.functional.normalize(ul_features)
            ul_finalEmbd = torch.cat([ul_finalEmbd,ul_features.clone()],dim=1)

        # item location
        il_finalEmbd = il_features.clone()
        for gnn in self.GNNlayersForIL:
            il_features = gnn(self.LaplacianMatForIL,self.selfLoopForIL,il_features)
            #il_features = nn.ReLU()(il_features)
            il_features = torch.nn.functional.normalize(il_features)
            il_finalEmbd = torch.cat([il_finalEmbd,il_features.clone()],dim=1)


        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        # if self.useLocation:
        userLocationEmbd = ul_finalEmbd[userIdx]
        itemLocationEmbd = il_finalEmbd[itemLoctionIdx]
        if self.useLocation:
            embd = torch.cat([userEmbd,itemEmbd,userLocationEmbd,itemLocationEmbd],dim=1)
        else:
            embd = torch.cat([userEmbd, itemEmbd], dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        # embd = self.transForm1(embd)
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        embd = self.transForm4(embd)
        embd = nn.ReLU()(self.transForm5(embd))
        prediction = embd.flatten()

        return prediction