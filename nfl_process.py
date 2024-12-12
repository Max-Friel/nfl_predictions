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

def trainingSplitSFolds(data,n):
    np.random.shuffle(data)
    sectionSize = int(len(data)/n)
    sections = []
    for i in range(0,n):
        begin = i*sectionSize
        end = i*sectionSize + sectionSize
        if end > len(data):
            end = len(data)
        sections.append(data[begin:end,:])
    ret = []
    for i in range(0,n):
        valid = sections[i]
        train = sections[:i] + sections[i:]
        t = train[0]
        for i in range(1,len(train)):
            t = np.vstack((t,train[i]))
        ret.append((t,valid))
    return ret

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
    print("SVM Training")
    svmstats(yHat,Y)
    print()
    X = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y = np.where(validation[:,cI] == 'TRUE',1,-1)
    yHat = X@w
    print("SVM Validation")
    svmstats(yHat,Y)
    return np.where(yHat > 0,"TRUE","FALSE")



def kernel(a,b):
    return ((a@b.T) + 1)**47

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
    print("SVMK Training")
    svmstats(yHat,Y)
    print()
    X2 = np.column_stack((np.ones((validation.shape[0],1)),validation[:,cols])).astype(float)
    Y2 = np.where(validation[:,cI] == 'TRUE',1,-1)
    yHat = kernel(X2,X)@diag(Y)@a
    print("SVMK Validation")
    svmstats(yHat,Y2)
    return np.where(yHat > 0,"TRUE","FALSE")

def feature_engineering_manual(data, headers, fields):
    indices = [getFieldIndex(headers, field) for field in fields]
    selected_features = data[:, indices].astype(float)
    interaction_features = []
    for i in range(selected_features.shape[1]):
        for j in range(i + 1, selected_features.shape[1]):
            interaction_features.append((selected_features[:, i] * selected_features[:, j]).reshape(-1, 1))

    interaction_features = np.hstack(interaction_features) if interaction_features else selected_features
    scaled_features = []
    for column in interaction_features.T:
        mean = np.mean(column)
        std = np.std(column, ddof=1)
        scaled_column = (column - mean) / std
        scaled_features.append(scaled_column.reshape(-1, 1))

    return np.hstack(scaled_features)

def logreg(training, validation, headers, fields, target, learning_rate=0.01, epochs=1000, lambda_=0.1):
    cols = [getFieldIndex(headers, field) for field in fields]
    target_col = getFieldIndex(headers, target)
    X_train = np.column_stack((np.ones((training.shape[0], 1)), training[:, cols].astype(float)))
    Y_train = np.where(training[:, target_col] == "TRUE", 1, 0)
    w = np.zeros(X_train.shape[1])
    for epoch in range(epochs):
        linear_combination = X_train @ w
        predictions = 1 / (1 + np.exp(-linear_combination))
        error = predictions - Y_train
        gradient = (X_train.T @ error / Y_train.size) + lambda_ * w
        w -= learning_rate * gradient

        if epoch % 100 == 0:
            loss = -np.mean(Y_train * np.log(predictions) + (1 - Y_train) * np.log(1 - predictions)) + (lambda_ * np.sum(w**2)) / 2
            #print(f"Epoch {epoch}: Loss = {loss}")
    X_val = np.column_stack((np.ones((validation.shape[0], 1)), validation[:, cols].astype(float)))
    Y_val = np.where(validation[:, target_col] == "TRUE", 1, 0)
    validation_scores = 1 / (1 + np.exp(-(X_val @ w)))
    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0.5
    for threshold in thresholds:
        predictions = (validation_scores >= threshold).astype(int)
        accuracy = np.mean(predictions == Y_val)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    #print(f"Optimal Threshold: {best_threshold}")
    predictions = (validation_scores >= best_threshold).astype(int)
    accuracy = np.mean(predictions == Y_val)
    print(f"Logistic Regression Validation Accuracy: {accuracy * 100:.2f}%")

    return predictions
    #return np.full((validationData.shape[0],1),"TRUE")

def linreg2(training,validation):
    cI = 0
    X = np.column_stack((np.ones((training.shape[0],1)),training[:,1:])).astype(float)
    Y = training[:,cI].astype(float)
    w = np.linalg.pinv(X.T@X)@X.T@Y
    train = X@w
    print("Training LinReg:")
    linregstats(Y,train)
    X = np.column_stack((np.ones((validation.shape[0],1)),validation[:,1:])).astype(float)
    Y = validation[:,cI].astype(float)
    yHat = X@w
    print("Validation LinReg:")
    linregstats(Y,yHat)
    return yHat

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

#for trainingData,validationData in trainingSplitSFolds(data,5):
#zscore required fields
zscoreFields = ["home_rest","away_rest","pastYear_home_Offense_yards_gained","past4_home_Offense_yards_gained","pastYear_home_Offense_touchdown","past4_home_Offense_touchdown","pastYear_home_Offense_sack","past4_home_Offense_sack","pastYear_home_Offense_penalty_yards","past4_home_Offense_penalty_yards","pastYear_home_Offense_fumble_lost","past4_home_Offense_fumble_lost","pastYear_home_Offense_interception","past4_home_Offense_interception","pastYear_home_Defense_yards_gained","past4_home_Defense_yards_gained","pastYear_home_Defense_touchdown","past4_home_Defense_touchdown","pastYear_home_Defense_sack","past4_home_Defense_sack","pastYear_home_Defense_penalty_yards","past4_home_Defense_penalty_yards","pastYear_home_Defense_fumble_lost","past4_home_Defense_fumble_lost","pastYear_home_Defense_interception","past4_home_Defense_interception","pastYear_home_top","past4_home_top","pastYear_away_Offense_yards_gained","past4_away_Offense_yards_gained","pastYear_away_Offense_touchdown","past4_away_Offense_touchdown","pastYear_away_Offense_sack","past4_away_Offense_sack","pastYear_away_Offense_penalty_yards","past4_away_Offense_penalty_yards","pastYear_away_Offense_fumble_lost","past4_away_Offense_fumble_lost","pastYear_away_Offense_interception","past4_away_Offense_interception","pastYear_away_Defense_yards_gained","past4_away_Defense_yards_gained","pastYear_away_Defense_touchdown","past4_away_Defense_touchdown","pastYear_away_Defense_sack","past4_away_Defense_sack","pastYear_away_Defense_penalty_yards","past4_away_Defense_penalty_yards","pastYear_away_Defense_fumble_lost","past4_away_Defense_fumble_lost","pastYear_away_Defense_interception","past4_away_Defense_interception","pastYear_away_top","past4_away_top"]
for fld in zscoreFields:
    i = getFieldIndex(headers,fld)
    trainingData[:,i],validationData[:,i] = zscore(trainingData[:,i],validationData[:,i])

#run KNN based on only zscored fields
#knn(trainingData,validationData,zscoreFields,"home_win_spread",10)

#run LinReg
homeFields = ["home_score","home_rest","pastYear_home_Offense_yards_gained","past4_home_Offense_yards_gained","pastYear_home_Offense_touchdown","past4_home_Offense_touchdown","pastYear_home_Offense_sack","past4_home_Offense_sack","pastYear_home_Offense_penalty_yards","past4_home_Offense_penalty_yards","pastYear_home_Offense_fumble_lost","past4_home_Offense_fumble_lost","pastYear_home_Offense_interception","past4_home_Offense_interception","pastYear_home_Defense_yards_gained","past4_home_Defense_yards_gained","pastYear_home_Defense_touchdown","past4_home_Defense_touchdown","pastYear_home_Defense_sack","past4_home_Defense_sack","pastYear_home_Defense_penalty_yards","past4_home_Defense_penalty_yards","pastYear_home_Defense_fumble_lost","past4_home_Defense_fumble_lost","pastYear_home_Defense_interception","past4_home_Defense_interception","pastYear_home_top","past4_home_top"]
awayFields = ["away_score","away_rest","pastYear_away_Offense_yards_gained","past4_away_Offense_yards_gained","pastYear_away_Offense_touchdown","past4_away_Offense_touchdown","pastYear_away_Offense_sack","past4_away_Offense_sack","pastYear_away_Offense_penalty_yards","past4_away_Offense_penalty_yards","pastYear_away_Offense_fumble_lost","past4_away_Offense_fumble_lost","pastYear_away_Offense_interception","past4_away_Offense_interception","pastYear_away_Defense_yards_gained","past4_away_Defense_yards_gained","pastYear_away_Defense_touchdown","past4_away_Defense_touchdown","pastYear_away_Defense_sack","past4_away_Defense_sack","pastYear_away_Defense_penalty_yards","past4_away_Defense_penalty_yards","pastYear_away_Defense_fumble_lost","past4_away_Defense_fumble_lost","pastYear_away_Defense_interception","past4_away_Defense_interception","pastYear_away_top","past4_away_top"]
hcols = []
for fld in homeFields:
    hcols.append(getFieldIndex(headers,fld))
acols = []
for fld in awayFields:
    acols.append(getFieldIndex(headers,fld))
td = np.vstack((trainingData[:,hcols],trainingData[:,acols]))
vd = np.vstack((validationData[:,hcols],validationData[:,acols]))
results = linreg2(td,vd)
halfRes = int(results.shape[0]/2) 
validationData = np.column_stack((validationData,results[:halfRes]))
validationData = np.column_stack((validationData,results[halfRes:]))
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

#run SVM
validationData = np.column_stack((validationData,svmk(trainingData,validationData,headers,zscoreFields,"home_win_spread")))

print()
#run LogReg
validationData = np.column_stack((validationData,logreg(trainingData,validationData,headers,zscoreFields,"home_win_spread")))

#creates a result 2d array where 
#column 1 is the linear regression result
#column 2 is the SVM result
#column 3 is the logisitical regression result with 
#column 4 being the actual result 
results = np.column_stack((validationData[:,-3:],validationData[:,hwI]))


### Ensemble Bagging Method

# Convert 'TRUE'/'FALSE' to 1/0
bg_results = np.where(results == 'TRUE', 1, 0).astype(int)

X = bg_results[:, :3]  
y_class = bg_results[:, 3]
y_reg = bg_results[:, 3].astype(float)

split_index = int(0.7 * len(X))  
X_train, X_test = X[:split_index], X[split_index:]
y_class_train, y_class_test = y_class[:split_index], y_class[split_index:]
y_regr_train, y_regr_test = y_reg[:split_index], y_reg[split_index:]

# Bagging for Classification
num_models = 10
classification_predictions = []

for _ in range(num_models):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample, y_sample = X_train[indices], y_class_train[indices]
    
    majority_class = Counter(y_sample).most_common(1)[0][0]
    model_prediction = [majority_class] * len(X_test)
    classification_predictions.append(model_prediction)

final_classification_prediction = np.round(np.mean(classification_predictions, axis=0)).astype(int)
classification_accuracy = np.mean(final_classification_prediction == y_class_test)

regression_predictions = []

for _ in range(num_models):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample, y_sample = X_train[indices], y_regr_train[indices]
    
    model_prediction = [np.mean(y_sample)] * len(X_test)
    regression_predictions.append(model_prediction)

final_regression_prediction = np.mean(regression_predictions, axis=0)
regression_mse = np.mean((final_regression_prediction - y_regr_test) ** 2)

# Output results
print("BAGGING - Classification Accuracy:", classification_accuracy)
print("BAGGING - Regression Mean Squared Error (MSE):", regression_mse)


###Ensemble Random Forests Method
# Convert 'TRUE'/'FALSE' to 1/0
rf_results = np.where(results == 'TRUE', 1, 0).astype(int)

X = rf_results[:, :3]  
y_class = rf_results[:, 3]  
y_reg = rf_results[:, 3].astype(float) 

split_index = int(0.7 * len(X)) 
X_train, X_test = X[:split_index], X[split_index:]
y_class_train, y_class_test = y_class[:split_index], y_class[split_index:]
y_regr_train, y_regr_test = y_reg[:split_index], y_reg[split_index:]

# Random Forests for Classification
num_trees = 10
max_features = 2 
classification_predictions = []

for _ in range(num_trees):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample, y_sample = X_train[indices], y_class_train[indices]
    
    selected_features = np.random.choice(X_sample.shape[1], max_features, replace=False)
    X_sample_reduced = X_sample[:, selected_features]
    X_test_reduced = X_test[:, selected_features]
    
    majority_class = Counter(y_sample).most_common(1)[0][0]
    model_prediction = [majority_class] * len(X_test_reduced)
    classification_predictions.append(model_prediction)

final_classification_prediction = np.round(np.mean(classification_predictions, axis=0)).astype(int)
classification_accuracy = np.mean(final_classification_prediction == y_class_test)

# Random Forests for Regression
regression_predictions = []

for _ in range(num_trees):
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample, y_sample = X_train[indices], y_regr_train[indices]
    
    selected_features = np.random.choice(X_sample.shape[1], max_features, replace=False)
    X_sample_reduced = X_sample[:, selected_features]
    X_test_reduced = X_test[:, selected_features]
    
    model_prediction = [np.mean(y_sample)] * len(X_test_reduced)
    regression_predictions.append(model_prediction)

final_regression_prediction = np.mean(regression_predictions, axis=0)
regression_mse = np.mean((final_regression_prediction - y_regr_test) ** 2)

# Output results
print("RANDOM FOREST - Classification Accuracy:", classification_accuracy)
print("RANDOM FOREST - Regression Mean Squared Error (MSE):", regression_mse)