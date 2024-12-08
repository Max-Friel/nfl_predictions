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

def diag(X):
    xDiag = np.zeros((X.shape[0],X.shape[0]),int)
    np.fill_diagonal(xDiag,X)
    return xDiag

def linregstats(Y,yHat):
    print("RMSE Training:" + str(rmse(Y,yHat)))
    print("SMAPE Training:" + str(smape(Y,yHat)))

def svmstats(yHat,Y):
    correct = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    p = 0
    n = 0
    yAbs = np.abs(yHat)
    res = np.hstack((np.where(yHat > 0, 1,-1),np.atleast_2d(Y).T,(np.where(yAbs > .5,1,0))))
    #res = res[res[:,2] == 1]
    tp = np.sum((res[:,0] == 1) & (res[:,1] == 1))
    fp = np.sum((res[:,0] == 1) & (res[:,1] == -1))
    tn = np.sum((res[:,0] == -1) & (res[:,1] == 1))
    fn = np.sum((res[:,0] == -1) & (res[:,1] == 1))
    correct = np.sum(res[:,0] == res[:,1])
    n = np.sum(res[:,1] == -1)
    p = np.sum(res[:,1] == 1)
    precision = tp/(tp+fp)
    recall = tp / (tp + fn)
    fMeasure = (2*precision*recall)/(precision+recall)
    print(f"Class priors 1:{p/res.shape[0]} 0:{n/res.shape[0]}")
    print(f"Accuracy: {correct/res.shape[0]}")
    print(f"Precison: {precision}")
    print(f"Recall: {recall}")
    print(f"FMeasure: {fMeasure}")

def alpha(X,Y,k):
    yDiag = diag(Y)
    ones = np.ones((X.shape[0],1))
    return np.linalg.pinv(yDiag@k(X,X)@yDiag)@ones

def k(a,b):
    return a@b.T

def svm(training,validation,headers,flds,c):
    cols = []
    for fld in flds:
        cols.append(getFieldIndex(headers,fld))
    cI = getFieldIndex(headers,c)
    X = np.column_stack((np.ones((training.shape[0],1)),training[:,cols])).astype(float)
    Y = np.where(training[:,cI] == 'TRUE',1,-1)
    a = alpha(X,Y,k)
    w = X.T@diag(Y)@a
    yHat = X@w
    print("Training")
    svmstats(yHat,Y)
    print()
    X = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y = np.where(validation[:,cI] == 'TRUE',1,-1)
    yHat = X@w
    print("Validation")
    svmstats(yHat,Y)
    return np.where(yHat > 0,"TRUE","FALSE")



def kernel(a,b):
    return ((a@b.T) + 1)**2

def svmk(training,validation,headers,flds,c):
    cols = []
    for fld in flds:
        cols.append(getFieldIndex(headers,fld))
    cI = getFieldIndex(headers,c)
    X = np.column_stack((np.ones((training.shape[0],1)),training[:,cols])).astype(float)
    Y = np.where(training[:,cI] == 'TRUE',1,-1)
    a = alpha(X,Y,kernel)
    w = X.T@diag(Y)@a
    yHat = kernel(X,X)@diag(Y)@a
    print("Training")
    svmstats(yHat,Y)
    print()
    X2 = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y2 = np.where(validation[:,cI] == 'TRUE',1,-1)
    yHat = kernel(X2,X)@diag(Y)@a
    print("Validation")
    svmstats(yHat,Y2)
    return np.where(yHat > 0,"TRUE","FALSE")

def logreg(training,validation,headers,flds,c):
    #needs code filled in
    return np.full((validationData.shape[0],1),"TRUE")

def linreg(training,validation,headers,flds,c):
    cols = []
    for fld in flds:
        cols.append(getFieldIndex(headers,fld))
    cI = getFieldIndex(headers,c)
    X = np.column_stack((np.ones((training.shape[0],1)),training[:,cols])).astype(float)
    Y = training[:,cI].astype(float)
    w = np.linalg.pinv(X.T@X)@X.T@Y
    train = X@w
    print("Training LinReg:")
    linregstats(Y,train)
    X = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y = validation[:,cI].astype(float)
    yHat = X@w
    print("Validation LinReg:")
    linregstats(Y,yHat)
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
#knn(trainingData,validationData,zscoreFields,"home_win_spread",10)

#run LinReg
validationData = np.column_stack((validationData,linreg(trainingData,validationData,headers,zscoreFields,"home_score")))
validationData = np.column_stack((validationData,linreg(trainingData,validationData,headers,zscoreFields,"away_score")))
hwI = getFieldIndex(headers,"home_win_spread")
spreadI = getFieldIndex(headers,"spread_line")

validationData = np.column_stack((validationData,validationData[:,-2].astype(float)-(validationData[:,-1].astype(float) + validationData[:,spreadI].astype(float))))
validationData = np.column_stack((validationData,np.where(validationData[:,-1].astype(float) > 0,"TRUE","FALSE")))
print()
#validationData = validationData[np.abs(validationData[:,-2].astype(float)) > 3]
right = np.sum(validationData[:,-1] == validationData[:,hwI])
print(f"right:{right} size:{validationData.shape[0]} %:{right/validationData.shape[0]}")
for i in range(1,np.max(validationData[:,-2].astype(float)).astype(int)):
    print(f"i:{i}")
    d = validationData[np.abs(validationData[:,-2].astype(float)) > i]
    right = np.sum(d[:,-1] == d[:,hwI])
    print(f"right:{right} size:{d.shape[0]} %:{right/d.shape[0]}")

print()
#run SVM
validationData = np.column_stack((validationData,svm(trainingData,validationData,headers,zscoreFields,"home_win_spread")))

print()
#run LogReg
validationData = np.column_stack((validationData,logreg(trainingData,validationData,headers,zscoreFields,"home_win_spread")))

#creates a result 2d array where 
#column 1 is the linear regression result
#column 2 is the SVM result
#column 3 is the logisitical regression result with 
#column 4 being the actual result 
results = np.column_stack((validationData[:,-3:],validationData[:,hwI]))