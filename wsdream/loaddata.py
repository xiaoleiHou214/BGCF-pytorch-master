import pandas as pd
from os import path as pathos
import numpy as np
from math import *
import sys
from time import time

path = pathos.dirname(__file__) + r'/data/'
path2 = pathos.dirname(__file__) + r'/Dataset#1/'
def loadWsdreamRt():
    fileName = 'rtMatrix.txt'
    file_path = path + fileName
    resopnse = []
    with open(file_path) as f:
        i = 0
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split('\t')
                resp = [float(i) for i in l[:-1]]
                resopnse.append(resp)
    resopnse = np.array(resopnse)

    user_num = resopnse.shape[0]
    item_num = resopnse.shape[1]

    data = []
    for i in range(user_num):
        for j in range(item_num):
            if resopnse[i][j] != -1:
                data.append(i+1)
                data.append(j+1)
                data.append(resopnse[i][j])
    data = np.array(data)
    data = data.reshape((-1, 3))
    data = pd.DataFrame(data,columns=['userId', 'itemId', 'rating'])

    return data, user_num, item_num

def loadWsdreamUserLocation():
    fileName = 'userlist.txt'
    file_path = path + fileName
    resopnse = []
    with open(file_path) as f:
        i = 0
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split('\t')
                resp = [i for i in l[:]]
                resopnse.append(resp)
    resopnse = np.array(resopnse)
    resopnse = np.delete(resopnse, [0, 1], axis=0)

    location_u = np.empty([len(resopnse), 3])
    for i in range(len(resopnse)):
        location_u[i][0] = int(resopnse[i][0])
        location_u[i][1] = float(resopnse[i][5])
        location_u[i][2] = float(resopnse[i][6])

    dist_u = np.zeros([len(resopnse), len(resopnse)], float)
    for i in range(len(resopnse)):
        for j in range(len(resopnse)):
            if i == j:
                dist_u[i][j] = 0
            else:
                try:
                    dist_u[i][j] = GreatCircleDistance(location_u[i][1], location_u[i][2],
                                                       location_u[j][1], location_u[j][2])
                    dist_u[i][j] = weight_function(dist_u[i][j])
                except Exception:
                    print("dist(i, j):", (i, j))
                    print("userA_location:", location_u[i])
                    print("userB_location:", location_u[j])

    data = []
    for i in range(len(resopnse)):
        for j in range(len(resopnse)):
            if float(format(dist_u[i][j], '.3f')) > 0:
                data.append(i + 1)
                data.append(j + 1)
                data.append(float(format(dist_u[i][j], '.3f')))
    data = np.array(data)
    data = data.reshape((-1, 3))
    data = pd.DataFrame(data, columns=['user1Id', 'user2Id', 'dist'])
    sys.stdout.write('\rLoading UserLocation data completes')
    return data


def loadWsdreamItemLocation():
    fileName = 'wslist.txt'
    file_path = path + fileName
    resopnse = []
    with open(file_path,encoding='gbk') as f:
        i = 0
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split('\t')
                resp = [i for i in l[:]]
                resopnse.append(resp)
    resopnse = np.array(resopnse)
    resopnse = np.delete(resopnse, [0, 1], axis=0)

    location_s= np.empty([len(resopnse), 3])
    count_s = 0
    for i in range(len(resopnse)):
        try:
            location_s[i][0] = int(resopnse[i][0])
            location_s[i][1] = float(resopnse[i][7])
            location_s[i][2] = float(resopnse[i][8])
        except Exception:
            count_s = count_s + 1
            location_s[i][1] = nan
            location_s[i][2] = nan
    print('service location is nan counts:',count_s)

    dist_s = np.zeros([len(resopnse), len(resopnse)], float)
    for i in range(len(resopnse)):
        for j in range(len(resopnse)):
            if isnan(location_s[i][1]) | isnan(location_s[j][1]):
                dist_s[i][j] = 0
            elif i == j:
                dist_s[i][j] = 0
            else:
                try:
                    dist_s[i][j] = GreatCircleDistance(location_s[i][1], location_s[i][2],
                                                       location_s[j][1], location_s[j][2])
                    dist_s[i][j] = weight_function(dist_s[i][j])
                except Exception:
                    print("dist(i, j):", (i, j))
                    print("itemA_location:", location_s[i])
                    print("itemB_location:", location_s[j])

    data = []
    for i in range(len(resopnse)):
        for j in range(len(resopnse)):
            if float(format(dist_s[i][j], '.3f')) > 0:
                data.append(i + 1)
                data.append(j + 1)
                data.append(float(format(dist_s[i][j], '.3f')))
    data = np.array(data)
    data = data.reshape((-1, 3))
    data = pd.DataFrame(data, columns=['item1Id', 'item2Id', 'dist'])
    fileName = 'itemLocation.csv'
    file_path_w = path + fileName
    data.to_csv(file_path_w,index=0)
    sys.stdout.write('\rLoading ItemLocation data completes')
    return data
def loadWsdreamItemLocationOrgin():
    fileName = 'itemLocation_Orgin.csv'
    file_path = path + fileName
    itemLocation = pd.read_csv(file_path)
    return itemLocation

def GreatCircleDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    ra = 6378.140  # 赤道半径 (km)
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    a = sin(rad_lat_A) * sin(rad_lat_B)
    b = cos(rad_lat_A) * cos(rad_lat_B)
    c = cos(rad_lng_A - rad_lng_B)
    d = a + b * c
    if d > 1:
        d = 1
    elif d < -1:
        d = -1
    xx = acos(d)
    distance = ra*xx
    return distance

def weight_function(dist):
    a = 2/(np.e**dist+1)
    return a

def loadServiceCountryAs():
    fileNameW = 'serviceCA.csv'
    file_path = path + fileNameW
    if (pathos.exists(file_path)):
        serviceToCA = pd.read_csv(file_path)
        return serviceToCA
    else:
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

        serviceToCA = {'id': serviceId, 'countryId': countryId, 'asId': asId}
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
        serviceToCA.to_csv(file_path_w, index=0)
        return serviceToCA

def PerformanceComput(UserId,Label,Prediction,topK=10):
    result = {}
    result_sort = {}
    for i in range(len(UserId)):
        if UserId[i] in result.keys():
            result[UserId[i]].append((i, Prediction[i]))
        else:
            result[UserId[i]] = [(i, Prediction[i])]
    for i in result.keys():
        result[i].sort(key=lambda x:x[1])
    label = []
    prediction = []
    index = []
    for i in result.keys():
        count = 0
        for j in result[i]:
            if count > 1:
                break
            label.append(Label[j[0]])
            prediction.append(j[1])
            index.append(j[0])
            count += 1
    return label, prediction