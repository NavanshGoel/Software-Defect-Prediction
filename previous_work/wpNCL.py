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
from collections import Counter
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def __populate(nnarray,n, x, y):
    test_data1 = pd.DataFrame([])
    #print(nnarray)
    for i in range(n):
        if y[nnarray[i]]==-1:
            new_data1 = pd.DataFrame(x[nnarray[i] - 1:nnarray[i], :])

            new_data1['Defective'] = y[nnarray[i]]
            # new_data1=pd.concat([X_data1,y_data1],axis=1)

            # print(new_data1.shape)
            # new_data1['bug']=y
            test_data1 = pd.concat([test_data1, new_data1], axis=0)

    #print(test_data1.shape)
    return test_data1

if __name__ == '__main__':
    df= pd.read_csv('D:\\myoverlap\\data1\\NASA\\PC4.csv')
    x= df.iloc[:, :df.shape[1]-1].values
    y= df.iloc[:, df.shape[1]-1].values
    #print(df.shape)

    # df_test = pd.read_csv('E:\\myoverlap\\Data\\new\\MORPH\\skarbonka.csv')
    # x_test = df_test.iloc[:, :df_test.shape[1] - 1].values
    # y_test = df_test.iloc[:, df_test.shape[1] - 1].values
    # #
    # x_train = np.log1p(x_train)
    # x_test=np.log1p(x_test)

    x=np.log1p(x)
    for i in range (20):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.15)

        df_x=pd.DataFrame(x_train)
        df_y=pd.DataFrame(y_train)

        dff=pd.concat([df_x,df_y],axis=1)
        dff.columns =['LOC_BLANK','BRANCH_COUNT','CALL_PAIRS','LOC_CODE_AND_COMMENT','LOC_COMMENTS','CONDITION_COUNT',
                            'CYCLOMATIC_COMPLEXITY','CYCLOMATIC_DENSITY','DECISION_COUNT','DECISION_DENSITY','DESIGN_COMPLEXITY',
                            'DESIGN_DENSITY','EDGE_COUNT','ESSENTIAL_COMPLEXITY','ESSENTIAL_DENSITY','LOC_EXECUTABLE',
                            'PARAMETER_COUNT','HALSTEAD_CONTENT','HALSTEAD_DIFFICULTY','HALSTEAD_EFFORT','HALSTEAD_ERROR_EST',
                            'HALSTEAD_LENGTH','HALSTEAD_LEVEL','HALSTEAD_PROG_TIME','HALSTEAD_VOLUME','MAINTENANCE_SEVERITY',
                            'MODIFIED_CONDITION_COUNT','MULTIPLE_CONDITION_COUNT','NODE_COUNT','NORMALIZED_CYLOMATIC_COMPLEXITY',
                            'NUM_OPERANDS','NUM_OPERATORS','NUM_UNIQUE_OPERANDS','NUM_UNIQUE_OPERATORS','NUMBER_OF_LINES',
                            'PERCENT_COMMENTS','LOC_TOTAL','Defective']

        arr1 = dff[dff['Defective'] == 1]
        arr0 = dff[dff['Defective'] == -1]

        X_source = dff.iloc[:, :dff.shape[1] - 1].values
        y_source = dff.iloc[:, dff.shape[1] - 1].values

        X_source1 = arr1.iloc[:, :arr1.shape[1] - 1].values
        y_source1 = arr1.iloc[:, arr1.shape[1] - 1].values

        # print(y_source)
        X_source0 = arr0.iloc[:, :arr0.shape[1] - 1].values
        y_source0 = arr0.iloc[:, arr0.shape[1] - 1].values
        # df_test = pd.read_csv('E:\\myoverlap\\Data\\NASA\\PC4.csv')
        # x_test=df_test.iloc[:, :df_test.shape[1]-1].values
        # y_test=df_test.iloc[:, df_test.shape[1]-1].values
       # print(X_source1.shape, X_source0.shape)

        # print(y_test)

        # x_test=np.log1p(x_test)

        nbrs1 = NearestNeighbors(n_neighbors=3, algorithm="auto")
        nbrs1.fit(X_source)
        nnarray1 = nbrs1.kneighbors(X_source1)[1]
        test_data1 = pd.DataFrame([])

        for i in range(nnarray1.shape[0]):
            new_data1 = __populate(nnarray1[i], 3, X_source, y_source)
            test_data1 = pd.concat([test_data1, new_data1], axis=0)
        # print(test_data1.shape)
        # test_data1.columns = ['total_loc','blank_loc','comment_loc','code_and_comment_loc','executable_loc','unique_operands',
        #                   'unique_operators','total_operands','total_operators','halstead_vocabulary','halstead_length',
        #                   'halstead_volume','halstead_level','halstead_difficulty','halstead_effort','halstead_error',
        #                   'halstead_time','branch_count','decision_count','call_pairs','condition_count','multiple_condition_count',
        #                   'cyclomatic_complexity','cyclomatic_density','decision_density','design_complexity','design_density',
        #                   'normalized_cyclomatic_complexity','formal_parameters','defects']
        test_data1.columns = ['LOC_BLANK','BRANCH_COUNT','CALL_PAIRS','LOC_CODE_AND_COMMENT','LOC_COMMENTS','CONDITION_COUNT',
                            'CYCLOMATIC_COMPLEXITY','CYCLOMATIC_DENSITY','DECISION_COUNT','DECISION_DENSITY','DESIGN_COMPLEXITY',
                            'DESIGN_DENSITY','EDGE_COUNT','ESSENTIAL_COMPLEXITY','ESSENTIAL_DENSITY','LOC_EXECUTABLE',
                            'PARAMETER_COUNT','HALSTEAD_CONTENT','HALSTEAD_DIFFICULTY','HALSTEAD_EFFORT','HALSTEAD_ERROR_EST',
                            'HALSTEAD_LENGTH','HALSTEAD_LEVEL','HALSTEAD_PROG_TIME','HALSTEAD_VOLUME','MAINTENANCE_SEVERITY',
                            'MODIFIED_CONDITION_COUNT','MULTIPLE_CONDITION_COUNT','NODE_COUNT','NORMALIZED_CYLOMATIC_COMPLEXITY',
                            'NUM_OPERANDS','NUM_OPERATORS','NUM_UNIQUE_OPERANDS','NUM_UNIQUE_OPERATORS','NUMBER_OF_LINES',
                            'PERCENT_COMMENTS','LOC_TOTAL','Defective']
        # test_data1.columns = ['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn', 'numberOfNonTrivialBugsFoundUntil',
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
        #                                   'CvsExpEntropy','LDHH_numberOfMethods','class']
        # test_data1.columns=['total_loc','blank_loc','comment_loc','code_and_comment_loc','executable_loc','unique_operands',
        #                   'unique_operators','total_operands','total_operators','halstead_vocabulary','halstead_length',
        #                   'halstead_volume','halstead_level','halstead_difficulty','halstead_effort','halstead_error',
        #                   'halstead_time','branch_count','decision_count','call_pairs','condition_count','multiple_condition_count',
        #                   'cyclomatic_complexity','cyclomatic_density','decision_density','design_complexity','design_density',
        #                   'normalized_cyclomatic_complexity','formal_parameters','defects']
        # test_data1.columns = ['AvgCyclomatic', 'AvgCyclomaticModified', 'AvgCyclomaticStrict', 'AvgEssential', 'AvgLine',
        #               'AvgLineBlank', 'AvgLineCode', 'AvgLineComment', 'CountLine', 'CountLineBlank', 'CountLineCode',
        #               'CountLineCodeDecl', 'CountLineCodeExe', 'CountLineComment', 'CountSemicolon', 'CountStmt',
        #               'CountStmtDecl', 'CountStmtExe', 'MaxCyclomatic', 'MaxCyclomaticModified', 'MaxCyclomaticStrict',
        #               'RatioCommentToCode', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
        #               'isDefective']

        # test_data1.columns = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam',
        #                       'moa',
        #                       'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
        #print(dff.shape, test_data1.shape)
        arr0 = pd.concat([arr0, test_data1], axis=0)
        df1 = arr0.drop_duplicates(keep=False)
        # print(df.shape)
        # print(df1.shape)


        df_train = pd.concat([df1, arr1], axis=0)

        x_train_new = df_train.iloc[:, :df.shape[1] - 1].values
        y_train_new = df_train.iloc[:, df.shape[1] - 1].values

        s = LogisticRegression(C=1.0)
        #s = KNeighborsClassifier()
        #s = SVC()
        #s=GaussianNB()
        #s=RandomForestClassifier(n_estimators=100)

        s.fit(x_train, y_train)

        pred = s.predict(x_test)

        n0 = 0
        n1 = 0
        for i in range(pred.shape[0]):
            if y_test[i] == -1 and pred[i] == 1:
                n0 = n0 + 1

        for i in range(y_test.shape[0]):
            if y_test[i] == -1:
                n1 = n1 + 1

        accu = metrics.accuracy_score(y_test, pred)
        auc = metrics.roc_auc_score(y_test, pred)
        mc = metrics.matthews_corrcoef(y_test, pred)
        print(metrics.classification_report(y_test, pred))
        print(metrics.confusion_matrix(y_test, pred))
        print(auc, mc)
        recall = metrics.recall_score(y_test, pred)
        pf = n0 / n1

        bal = 1 - np.sqrt(pf * pf + (1 - recall) * (1 - recall)) / np.sqrt(2)
        print(bal)

        # s.fit(x_train_new, y_train_new)
        #
        # pred1 = s.predict(x_test)
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
        #     if y_test[i]==0 and pred1[i]==1:
        #           n2=n2+1
        #
        # pf=n2/n1
        #
        # bal1=1-np.sqrt(pf*pf+(1-recall1)*(1-recall1))/np.sqrt(2)
        # print(bal1)


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


