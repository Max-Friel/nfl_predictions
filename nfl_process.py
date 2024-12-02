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

def onehot(data,name):
    values = list(set(data))
    arrRet = np.zeros(shape=(len(data),len(values)))
    for x in range(0,len(data)):
        for y in range(0,len(values)):
            arrRet[x,y] = values[y] == data[x]
    values = np.char.add(name + "_", values)
    return values,arrRet

def trainingSplit(data):
    np.random.shuffle(data)
    trainingIndex = int(len(data)/3)
    trainingData = data[:-trainingIndex,:]
    validationData = data[(len(data) - trainingIndex):,:]
    return trainingData,validationData

def rmse(y,yHat):
	return np.sqrt(np.mean((yHat-y)**2))

def smape(y,yHat):
	return np.mean(np.abs(y-yHat)/(np.abs(y) + np.abs(yHat)))

def getFieldIndex(header,fld):
      return np.where(header == ('"' + fld + '"'))[0][0]

def linreg(training,validation,headers,flds,c):
    cols = []
    for fld in flds:
        cols.append(getFieldIndex(headers,fld))
    cI = getFieldIndex(headers,c)
    X = np.column_stack((np.ones((training.shape[0],1)),training[:,cols])).astype(float)
    Y = training[:,cI].astype(float)
    w = np.linalg.pinv(X.T@X)@X.T@Y
    train = X@w
    print("RMSE Training:" + str(rmse(Y,train)))
    print("SMAPE Training:" + str(smape(Y,train)))
    X = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y = validation[:,cI].astype(float)
    yHat = X@w
    print("RMSE Validation:" + str(rmse(Y,yHat)))
    print("SMAPE Validation:" + str(smape(Y,yHat)))
    return yHat

def knn(trainingData,validationData,flds,c,K):
    goalIndex = getFieldIndex(headers,c)
    right = 0
    results = []
    for i in range(0,validationData.shape[0]):
        kClosest = []
        kClosestClass = []
        for p in range(0,trainingData.shape[0]):
            distance = 0
            for fld in flds:
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
        results.append(predict)
        confidence = counter[predict] / 5
        actual = validationData[i,goalIndex]
        if predict == actual:
            right += 1
    print(f"Acuracy:{right/validationData.shape[0]}")
    return results

#loading in data and splitting it
csv = np.loadtxt('./data/games.csv',dtype=str,delimiter=",")
np.random.seed(0)
headers = csv[0,:]
home_score_col = getFieldIndex(headers,"home_score")
away_score_col = getFieldIndex(headers,"away_score")
data = csv[1:,:]
data = data[data[:,home_score_col] != "NA"]
data = data[data[:,away_score_col] != "NA"]

oneHotFields = ["weekday","gametime","roof","surface","stadium_id"]
oneHotFieldsAfter = {}
for fld in oneHotFields:
    col = getFieldIndex(headers,fld)
    newHeaders,newData = onehot(data[:,col],fld)
    headers = np.delete(headers,col)
    headers = np.append(headers,newHeaders)
    data = np.delete(data,col,axis=1)
    data = np.hstack((data,newData))
    oneHotFieldsAfter[fld] = newHeaders

trainingData, validationData = trainingSplit(data)

#zscore required fields
zscoreFields = ["home_rest","away_rest","pastYear_home_Offense_yards_gained","past4_home_Offense_yards_gained","pastYear_home_Offense_touchdown","past4_home_Offense_touchdown","pastYear_home_Offense_sack","past4_home_Offense_sack","pastYear_home_Offense_penalty_yards","past4_home_Offense_penalty_yards","pastYear_home_Offense_fumble_lost","past4_home_Offense_fumble_lost","pastYear_home_Offense_interception","past4_home_Offense_interception","pastYear_home_Defense_yards_gained","past4_home_Defense_yards_gained","pastYear_home_Defense_touchdown","past4_home_Defense_touchdown","pastYear_home_Defense_sack","past4_home_Defense_sack","pastYear_home_Defense_penalty_yards","past4_home_Defense_penalty_yards","pastYear_home_Defense_fumble_lost","past4_home_Defense_fumble_lost","pastYear_home_Defense_interception","past4_home_Defense_interception","pastYear_home_top","past4_home_top","pastYear_away_Offense_yards_gained","past4_away_Offense_yards_gained","pastYear_away_Offense_touchdown","past4_away_Offense_touchdown","pastYear_away_Offense_sack","past4_away_Offense_sack","pastYear_away_Offense_penalty_yards","past4_away_Offense_penalty_yards","pastYear_away_Offense_fumble_lost","past4_away_Offense_fumble_lost","pastYear_away_Offense_interception","past4_away_Offense_interception","pastYear_away_Defense_yards_gained","past4_away_Defense_yards_gained","pastYear_away_Defense_touchdown","past4_away_Defense_touchdown","pastYear_away_Defense_sack","past4_away_Defense_sack","pastYear_away_Defense_penalty_yards","past4_away_Defense_penalty_yards","pastYear_away_Defense_fumble_lost","past4_away_Defense_fumble_lost","pastYear_away_Defense_interception","past4_away_Defense_interception","pastYear_away_top","past4_away_top"]
for fld in zscoreFields:
    i = getFieldIndex(headers,fld)
    trainingData[:,i],validationData[:,i] = zscore(trainingData[:,i],validationData[:,i])

#run KNN based on only zscored fields
knn(trainingData,validationData,zscoreFields,"home_win_spread",10)

#run LinReg
validationData = np.column_stack((validationData,linreg(trainingData,validationData,headers,zscoreFields,"home_score")))
validationData = np.column_stack((validationData,linreg(trainingData,validationData,headers,zscoreFields,"away_score")))
hwI = getFieldIndex(headers,"home_win_spread")
spreadI = getFieldIndex(headers,"spread_line")
right = 0;
for i in range(0,validationData.shape[0]):
    home = validationData[i,-2].astype(float)
    away = validationData[i,-1].astype(float)
    home_win = validationData[i,hwI] == "TRUE"
    home_win_pred = home > (away + validationData[i,spreadI].astype(float)) 
    if home_win == home_win_pred:
         right += 1
print(f"right:{right} size:{validationData.shape[0]} %:{right/validationData.shape[0]}")