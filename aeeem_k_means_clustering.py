import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

## Importing the dataset

files = ["PDE", "EQ", "JDT", "LC", "ML"]



dict1 = {"bal": 0, "auc": 0, "recall": 0,"accu":0,"bal1": 0, "auc1": 0, "recall1": 0, "accu1":0}
for index in range(len(files)):
  path = "/content/" + files[index] + ".csv"
  dataset = pd.read_csv(path)
  X = dataset.iloc[:,:-1]
  y = dataset.iloc[:,-1]
  y = np.array(y)
  scaler = MinMaxScaler()
  X=scaler.fit_transform(X)
  print(files[index])
  for ind in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,shuffle= True)
    n = X_train.shape[0] # number of instances in D
    d = np.sum(y_train == 1) #  no of defective instances in D
    p = d/n 
    m = 40
    k = int(n//m)

    kmeans = KMeans(n_clusters = k, init = 'k-means++')
    y_kmeans = kmeans.fit_predict(X_train)
    col = list(dataset.columns)
    col.append('Cluster')
    #
    train_df = pd.DataFrame(X_train,columns= col[:-2])
    train_df
    train_df['class'] = y_train
    train_df['Cluster'] = y_kmeans
    # compute r for all clusters
    final = pd.DataFrame(columns = col)
    for i in range(k):
      temp_df = train_df.loc[train_df['Cluster']==i]
      temp_df = temp_df.dropna()
      D = len(temp_df[temp_df['class']==1])
      nd = len(temp_df[temp_df['class']==-1])
      r = D/len(temp_df)
      if r > p:
        chosendf = temp_df[temp_df['class']==1]
      else:
        chosendf = temp_df[temp_df['class']==-1]
      
      final = pd.concat([final,chosendf],axis=0)

    final = final.drop(columns= ['Cluster'], axis = 1)


    x_train_new= np.array(final.iloc[:,:-1])
    y_train_new= final.iloc[:,-1]
    y_train_new = np.array(y_train_new).astype('int')
    #s=LogisticRegression(C=1.0)
    #s=KNeighborsClassifier()
    #s = SVC()
    #s=GaussianNB()
    s=RandomForestClassifier(n_estimators=100)
    s.fit(X_train, y_train)
    pred = s.predict(X_test)

    n0 = 0
    n1 = 0

    for i in range(pred.shape[0]):
      if ((y_test[i] == 1) and (pred[i] == -1)):
        n0 = n0 + 1

    for i in range(y_test.shape[0]):
      if y_test[i] == -1:
        n1 = n1 + 1

    accu = metrics.accuracy_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred)
    mc = metrics.matthews_corrcoef(y_test, pred)

    recall = metrics.recall_score(y_test, pred)
    pf = n0 / n1

    bal = 1 - (np.sqrt(pf * pf + (1 - recall) * (1 - recall))) / np.sqrt(2)

    dict1["bal"] += bal
    dict1["auc"] += auc
    dict1["recall"] += recall
    dict1["accu"]+=accu
    s.fit(x_train_new, y_train_new)
    pred1 = s.predict(X_test)

    accu1 = metrics.accuracy_score(y_test, pred1)
    auc1 = metrics.roc_auc_score(y_test, pred1)
    mc1 = metrics.matthews_corrcoef(y_test, pred1)
    recall1 = metrics.recall_score(y_test, pred1)
    n0 = 0
    n1 = 0
    for i in range(pred1.shape[0]):
      if y_test[i] == 1 and pred1[i] == -1:
        n0 = n0 + 1

    for i in range(y_test.shape[0]):
      if y_test[i] == -1:
        n1 = n1 + 1

    if n1 != 0:
      pf = n0 / n1
      bal = 1 - (np.sqrt(pf * pf + (1 - recall) * (1 - recall)))/ np.sqrt(2)

    else:
      bal = 0

    dict1["bal1"] += bal
    dict1["auc1"] += auc1
    dict1["recall1"] += recall1
    dict1["accu1"]+=accu1

  for k in dict1.keys():
    dict1[k] = dict1[k]/20

  keys = list(dict1.keys())
  values = list(dict1.values())
  for t in range(len(dict1.keys())):
    if t==4:
      print()
    
    print(keys[t], end= " ")
    print(values[t], end= " ")
  print()