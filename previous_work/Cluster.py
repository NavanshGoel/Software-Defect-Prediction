import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    df= pd.read_csv('D:\\myoverlap\\Data1\\NASA\\PC3.csv')
    x_train = df.iloc[:, :df.shape[1]-1].values
    y_train = df.iloc[:, df.shape[1]-1].values
    #print(df.shape)

    # df_test = pd.read_csv('D:\\myoverlap\\Data1\\NASA\\PC4.csv')
    # x_test = df_test.iloc[:, :df_test.shape[1] - 1].values
    # y_test = df_test.iloc[:, df_test.shape[1] - 1].values

   # x_train = np.log1p(x_train)
    #x_test=np.log1p(x_test)

    #x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.15)

    # scaler = MinMaxScaler()
    # x=scaler.fit_transform(x)

    p = np.sum(y_train == 1) / np.sum(y_train ==-1)
    #p=1/2
    k=int(df.shape[0]/40)
    print(k,p)
    random_state = 200

    kmean= KMeans(n_clusters=k, random_state=random_state)
    y_pred=kmean.fit_predict(x_train)

    df_x=pd.DataFrame(x_train)
    df_y=pd.DataFrame(y_train)
    df_y_pred=pd.DataFrame(y_pred)
    df_new=pd.concat((df_x,df_y,df_y_pred),axis=1)
    df_new.columns = ['LOC_BLANK','BRANCH_COUNT','CALL_PAIRS','LOC_CODE_AND_COMMENT','LOC_COMMENTS','CONDITION_COUNT',
                        'CYCLOMATIC_COMPLEXITY','CYCLOMATIC_DENSITY','DECISION_COUNT','DECISION_DENSITY','DESIGN_COMPLEXITY',
                        'DESIGN_DENSITY','EDGE_COUNT','ESSENTIAL_COMPLEXITY','ESSENTIAL_DENSITY','LOC_EXECUTABLE',
                        'PARAMETER_COUNT','HALSTEAD_CONTENT','HALSTEAD_DIFFICULTY','HALSTEAD_EFFORT','HALSTEAD_ERROR_EST',
                        'HALSTEAD_LENGTH','HALSTEAD_LEVEL','HALSTEAD_PROG_TIME','HALSTEAD_VOLUME','MAINTENANCE_SEVERITY',
                        'MODIFIED_CONDITION_COUNT','MULTIPLE_CONDITION_COUNT','NODE_COUNT','NORMALIZED_CYLOMATIC_COMPLEXITY',
                        'NUM_OPERANDS','NUM_OPERATORS','NUM_UNIQUE_OPERANDS','NUM_UNIQUE_OPERATORS','NUMBER_OF_LINES',
                        'PERCENT_COMMENTS','LOC_TOTAL','Defective','label']

    # df_new.columns= ['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn', 'numberOfNonTrivialBugsFoundUntil',
    #                                   'WCHU_numberOfPublicAttributes','WCHU_numberOfAttributes','CvsWEntropy',  'LDHH_numberOfPublicMethods',
    #                                   'WCHU_fanIn','LDHH_numberOfPrivateAttributes','CvsEntropy','LDHH_numberOfPublicAttributes',
    #                                   'WCHU_numberOfPrivateMethods','WCHU_numberOfMethods','ck_oo_numberOfPublicAttributes',
    #                                   'ck_oo_noc','numberOfCriticalBugsFoundUntil','ck_oo_wmc','LDHH_numberOfPrivateMethods',
    #                                   'WCHU_numberOfPrivateAttributes','CvsLogEntropy','WCHU_noc','LDHH_numberOfAttributesInherited',
    #                                   'WCHU_wmc','ck_oo_fanOut','ck_oo_numberOfLinesOfCode','ck_oo_numberOfAttributesInherited',
    #                                   'ck_oo_numberOfMethods','ck_oo_dit','ck_oo_fanIn','LDHH_noc','WCHU_dit','ck_oo_lcom',
    #                                   'WCHU_numberOfAttributesInherited','ck_oo_rfc','LDHH_wmc','LDHH_numberOfAttributes',
    #                                   'LDHH_numberOfLinesOfCode','WCHU_fanOut','WCHU_lcom','ck_oo_cbo','WCHU_rfc','ck_oo_numberOfAttributes',
    #                                   'numberOfHighPriorityBugsFoundUntil','ck_oo_numberOfPrivateAttributes','numberOfMajorBugsFoundUntil',
    #                                   'WCHU_numberOfPublicMethods','LDHH_dit','WCHU_cbo','CvsLinEntropy','WCHU_numberOfMethodsInherited',
    #                                   'numberOfBugsFoundUntil','LDHH_fanOut','LDHH_numberOfMethodsInherited','LDHH_rfc',
    #                                   'ck_oo_numberOfMethodsInherited','ck_oo_numberOfPublicMethods','LDHH_cbo','WCHU_numberOfLinesOfCode',
    #                                   'CvsExpEntropy','LDHH_numberOfMethods','class','label']

    # df_new.columns=['total_loc','blank_loc','comment_loc','code_and_comment_loc','executable_loc','unique_operands',
    #                   'unique_operators','total_operands','total_operators','halstead_vocabulary','halstead_length',
    #                   'halstead_volume','halstead_level','halstead_difficulty','halstead_effort','halstead_error',
    #                   'halstead_time','branch_count','decision_count','call_pairs','condition_count','multiple_condition_count',
    #                   'cyclomatic_complexity','cyclomatic_density','decision_density','design_complexity','design_density',
    #                   'normalized_cyclomatic_complexity','formal_parameters','defects','label']

    # df_new.columns = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential', 'AvgLine',
    #               'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'CountLine', 'CountLineBlank', 'CountLineCode',
    #               'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountSemicolon', 'CountStmt',
    #               'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
    #               'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
    #               'isDefective','label']

    # df_new.columns = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa',
    #                       'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc', 'bug','label']

    #df_new.columns= ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15', 'bug','label']


    df_train=pd.DataFrame([])

    for i in range(k):
        arr=df_new[df_new['label']==i]
        n0=arr[arr['Defective']==-1].shape[0]
        n1=arr[arr['Defective']==1].shape[0]
        if n0==0 or n1/n0>=p :
            data=arr[arr['Defective']==1]
            df_train= pd.concat([df_train,data], axis=0)
        else:
            data = arr[arr['Defective'] == -1]
            df_train = pd.concat([df_train, data], axis=0)


        #print(n0,n1)

    df_train.to_csv('D:\\myoverlap\\Data1\\NASA\\PC3_40.csv', index=False)


    # x_train_new= df_train.iloc[:, :df.shape[1]-1].values
    # y_train_new=df_train.iloc[:, df.shape[1]-1].values
    #
    # #s=SVC()
    # s=GaussianNB()

    # s.fit(x_train,y_train)
    #
    # pred=s.predict(x_test)
    #
    #
    #
    # accu = metrics.accuracy_score(y_test, pred)
    # auc = metrics.roc_auc_score(y_test, pred)
    # mc = metrics.matthews_corrcoef(y_test, pred)
    # print(metrics.classification_report(y_test, pred))
    # print(metrics.confusion_matrix(y_test, pred))
    # print(auc, mc)
    # recall=metrics.recall_score(y_test, pred)
    # pf = n0 / n1
    #
    # bal = 1 - np.sqrt(pf * pf + (1 - recall) * (1 - recall)) / np.sqrt(2)
    # print(bal)
    # s.fit(x_train_new, y_train_new)
    #
    # pred1 = s.predict(x_test)
    #
    # n0 = 0
    # n1 = 0
    # # for i in range(pred1.shape[0]):
    # #     if y_test[i] == 0 and pred1[i] == 1:
    # #         n0 = n0 + 1
    #
    # for i in range(y_test.shape[0]):
    #     if y_test[i] == -1:
    #         n1 = n1 + 1
    #
    # accu1 = metrics.accuracy_score(y_test, pred1)
    # auc1 = metrics.roc_auc_score(y_test, pred1)
    # mc1 = metrics.matthews_corrcoef(y_test, pred1)
    # print(metrics.classification_report(y_test, pred1))
    # print(metrics.confusion_matrix(y_test, pred1))
    # print(auc1, mc1)
    # recall1=metrics.recall_score(y_test, pred1)
    # n2=0
    # for i in range(pred1.shape[0]):
    #     if y_test[i]==-1 and pred1[i]==1:
    #           n2=n2+1
    #
    # pf=n2/n1
    #
    # bal1=1-np.sqrt(pf*pf+(1-recall1)*(1-recall1))/np.sqrt(2)
    # print(bal1)
    #

  #   center=kmean.cluster_centers_
  #
  #   print(center)
  #   plt.figure(figsize=(2, 12))
  #   plt.subplot(221)  # 在2图里添加子图2
  #   plt.scatter(x[:, 0], x[:, 1],x[:, 2], c=y)
  #
  #   plt.subplot(222)  # 在2图里添加子图2
  #   plt.scatter(x[:, 0], x[:, 1],x[:, 2], c=y_pred)
  # #  plt.title("Anisotropicly Distributed Blobs")
  #
  #   plt.show()
  #


