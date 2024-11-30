import numpy as np
from collections import Counter

def zscore(data,data2):
    data[data == 'NA'] = 0
    data = data.astype(float)
    data2[data2 == 'NA'] = 0
    data2 = data2.astype(float)
    m = np.mean(data)
    s = np.std(data,ddof=1)
    return (data-m)/s,(data2-m)/s

def trainingSplit(data):
    np.random.shuffle(data)
    trainingIndex = int(len(data)/3)
    trainingData = data[:-trainingIndex,:]
    validationData = data[(len(data) - trainingIndex):,:]
    return trainingData,validationData

def getFieldIndex(header,fld):
      return np.where(header == (fld))[0][0]

#loading in data and splitting it
csv = np.loadtxt('./data/games.csv',dtype=str,delimiter=",")
np.random.seed(0)
headers = csv[0,:]
data = csv[1:,:]
trainingData, validationData = trainingSplit(data)

#zscore required fields
zscoreFields = ["home_rest","away_rest","pastYear_home_Offense_yards_gained","past4_home_Offense_yards_gained","pastYear_home_Offense_touchdown","past4_home_Offense_touchdown","pastYear_home_Offense_sack","past4_home_Offense_sack","pastYear_home_Defense_yards_gained","past4_home_Defense_yards_gained","pastYear_home_Defense_touchdown","past4_home_Defense_touchdown","pastYear_home_Defense_sack","past4_home_Defense_sack","pastYear_home_top","past4_home_top","pastYear_away_Offense_yards_gained","past4_away_Offense_yards_gained","pastYear_away_Offense_touchdown","past4_away_Offense_touchdown","pastYear_away_Offense_sack","past4_away_Offense_sack","pastYear_away_Defense_yards_gained","past4_away_Defense_yards_gained","pastYear_away_Defense_touchdown","past4_away_Defense_touchdown","pastYear_away_Defense_sack","past4_away_Defense_sack","pastYear_away_top","past4_away_top"]
for fld in zscoreFields:
    i = getFieldIndex(headers,fld)
    trainingData[:,i],validationData[:,i] = zscore(trainingData[:,i],validationData[:,i])

#run KNN based on only scored fields
K = 3
goalIndex = getFieldIndex(headers,"home_win")
right = 0
right100 = 0
total100 = 0
for i in range(0,validationData.shape[0]):
    kClosest = []
    kClosestClass = []
    for p in range(0,trainingData.shape[0]):
        distance = 0
        for fld in zscoreFields:
            fldI = getFieldIndex(headers,fld)
            distance += (validationData[i,fldI].astype(float) - trainingData[p,fldI].astype(float))**2
        distance = distance ** 0.5
        for k in range(0,K):
            if len(kClosest) == k or kClosest[k] > distance:
                kClosest.insert(k,distance)
                kClosestClass.insert(k,trainingData[p,goalIndex])
                break
        if len(kClosest) > K:
            kClosest.pop()
            kClosestClass.pop()
    counter = Counter(kClosestClass)
    predict = counter.most_common(1)[0][0]
    confidence = counter[predict] / 5
    actual = validationData[i,goalIndex]
    if confidence == 1:
        total100 +=1
        if predict == actual:
            right100 += 1
    if predict == actual:
         right += 1
print(f"Acuracy:{right/validationData.shape[0]}")
print(f"Acuracy100:{right100/total100}")