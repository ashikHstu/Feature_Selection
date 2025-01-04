def all_result(x_train,x_test,y_train,y_test):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import seaborn as sb
    # Import essential libraries
    import os
    import joblib
    import numpy as np
    import pandas as pd
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from sklearn.linear_model import LogisticRegression
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    import os
    from numpy.random import rand
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import random
    from random import randrange
    import time
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LinearRegression
    import xgboost as xg
    x_train_smt = x_train
    x_test_smt = x_test
    y_train_smt = y_train
    y_test_smt  = y_test

    algo_names=[]
    all_accuracy=[]
    all_sensitivity=[]
    all_specificity=[]
    all_f1Score=[]


    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    modelSVC = SVC(probability=True)
    modelSVC.fit(x_train_smt, y_train_smt)
    #print(modelSVC.score(x_test_smt, y_test_smt))

    y_pred = modelSVC.predict(x_test_smt)
    res=y_pred
    y_final = y_test
    y_pred_svc = y_pred
    ac = accuracy_score(y_test_smt, y_pred)
    # print(ac)
    # Performance Measure of SVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # y_pred = modelSVC.predict(x_test_smt)
    # print(confusion_matrix(y_test_smt, y_pred))
    # print(classification_report(y_test_smt, y_pred))

    from sklearn.metrics import cohen_kappa_score
    cmSVC = confusion_matrix(y_test_smt, modelSVC.predict(x_test_smt))
    TP = cmSVC[1,1]
    TN = cmSVC[0,0]
    FP = cmSVC[0,1]
    FN = cmSVC[1,0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    # Precision or positive predictive value
    PPV = TP/float(TP+FP)
    # Negative predictive value
    NPV = TN/float(TN+FN)
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    # False negative rate
    FNR = FN/float(TP+FN)
    # False discovery rate
    FDR = FP/float(TP+FP)
    totalSVC=sum(sum(cmSVC))
    Accuracy = (TN+TP)/totalSVC
    # MCC
    val = (TP * TN) - (FP * FN)
    MCC_SVC = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    algo_names.append("SVM")
    all_accuracy.append(ac)
    all_sensitivity.append(TPR)
    all_specificity.append(TNR)
    f1Score=2*(TPR*PPV)/(TPR+PPV)
    all_f1Score.append(f1Score)

    ####################### Logistic Regression ##############################################3
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    modelLR = LogisticRegression()
    modelLR.fit(x_train_smt, y_train_smt)
    # print(modelLR.score(x_test_smt, y_test_smt))


    # Predicting the Test set results
    y_pred = modelLR.predict(x_test_smt)
    res=res+y_pred
    ac = accuracy_score(y_test_smt, y_pred)
    # print(ac)


    # Performance Measure of Logistic regression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # y_pred = modelLR.predict(x_test_smt)
    # print(confusion_matrix(y_test_smt, y_pred))
    # print(classification_report(y_test_smt, y_pred))
    # from sklearn.metrics import cohen_kappa_score
    cmLR = confusion_matrix(y_test_smt, modelLR.predict(x_test_smt))
    TP = cmLR[1,1]
    TN = cmLR[0,0]
    FP = cmLR[0,1]
    FN = cmLR[1,0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    # Precision or positive predictive value
    PPV = TP/float(TP+FP)
    # Negative predictive value
    NPV = TN/float(TN+FN)
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    # False negative rate
    FNR = FN/float(TP+FN)
    # False discovery rate
    FDR = FP/float(TP+FP)
    # Accuracy
    totalLR=sum(sum(cmLR))
    Accuracy = (TN+TP)/totalLR
    # MCC
    val = (TP * TN) - (FP * FN)
    MCC_LR = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # Cohen Kappa
    # Y_pred = modelLR.predict(x_test_smt)
    # cohen_score = cohen_kappa_score(y_test_smt, Y_pred)
    algo_names.append("LogisticRegression")
    all_accuracy.append(ac)
    all_sensitivity.append(TPR)
    all_specificity.append(TNR)
    f1Score=2*(TPR*PPV)/(TPR+PPV)
    all_f1Score.append(f1Score)

    ############################## Decission Tree ################################3
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    modelDTC = tree.DecisionTreeClassifier()
    modelDTC.fit(x_train_smt, y_train_smt)
    # print(modelDTC.score(x_test_smt, y_test_smt))
    # Predicting the Test set results
    y_pred = modelDTC.predict(x_test_smt)
    res=res+y_pred
    ac = accuracy_score(y_test_smt, y_pred)
    # print(ac)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # y_pred = modelDTC.predict(x_test_smt)
    # print(confusion_matrix(y_test_smt, y_pred))
    # print(classification_report(y_test_smt, y_pred))
    # from sklearn.metrics import cohen_kappa_score
    cmDTC = confusion_matrix(y_test_smt, modelDTC.predict(x_test_smt))
    TP = cmDTC[1,1]
    TN = cmDTC[0,0]
    FP = cmDTC[0,1]
    FN = cmDTC[1,0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    # Precision or positive predictive value
    PPV = TP/float(TP+FP)
    # Negative predictive value
    NPV = TN/float(TN+FN)
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    # False negative rate
    FNR = FN/float(TP+FN)
    # False discovery rate
    FDR = FP/float(TP+FP)
    # Accuracy
    totalDTC=sum(sum(cmDTC))
    Accuracy = (TN+TP)/totalDTC
    # MCC
    val = (TP * TN) - (FP * FN)
    MCC_DTC = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # Cohen Kappa
    # Y_pred = modelDTC.predict(x_test_smt)
    # cohen_score = cohen_kappa_score(y_test_smt, Y_pred)
    algo_names.append("DecissionTree")
    all_accuracy.append(ac)
    all_sensitivity.append(TPR)
    all_specificity.append(TNR)
    f1Score=2*(TPR*PPV)/(TPR+PPV)
    all_f1Score.append(f1Score)

    ############################# Random Forest #########################################################
    from sklearn import ensemble
    from sklearn.metrics import accuracy_score
    modelRFC = ensemble.RandomForestClassifier()
    modelRFC.fit(x_train_smt, y_train_smt)
    # print(modelRFC.score(x_test_smt, y_test_smt))
    # Predicting the Test set results
    y_pred = modelRFC.predict(x_test_smt)
    res=res+y_pred
    ac = accuracy_score(y_test_smt, y_pred)
    # y_pred_rf = y_pred
    # print(ac)
    # Performance Measure of RFC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    # y_pred = modelRFC.predict(x_test_smt)
    # cr=classification_report(y_test_smt,y_pred)
    # from sklearn.metrics import cohen_kappa_score
    cmRFC = confusion_matrix(y_test_smt, modelRFC.predict(x_test_smt))
    TP = cmRFC[1,1]
    TN = cmRFC[0,0]
    FP = cmRFC[0,1]
    FN = cmRFC[1,0]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    # Precision or positive predictive value
    PPV = TP/float(TP+FP)
    # Negative predictive value
    NPV = TN/float(TN+FN)
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    # False negative rate
    FNR = FN/float(TP+FN)
    # False discovery rate
    FDR = FP/float(TP+FP)
    # Accuracy
    totalRFC=sum(sum(cmRFC))
    Accuracy = (TN+TP)/totalRFC
    # MCC
    val = (TP * TN) - (FP * FN)
    MCC_RFC = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # # Cohen Kappa
    # Y_pred = modelRFC.predict(x_test_smt)
    # cohen_score = cohen_kappa_score(y_test_smt, Y_pred)
    algo_names.append("RandomForest")
    all_accuracy.append(ac)
    all_sensitivity.append(TPR)
    all_specificity.append(TNR)
    f1Score=2*(TPR*PPV)/(TPR+PPV)
    all_f1Score.append(f1Score)

    ############################ KNN Algorithm ############################################################

    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.metrics import accuracy_score

    # modelKNN = KNeighborsClassifier()
    # modelKNN.fit(x_train_smt, y_train_smt)
    # # print(modelKNN.score(x_test_smt, y_test_smt))

    # # Predicting the Test set results
    # y_pred = modelKNN.predict(x_test_smt)
    # res=res+y_pred
    # ac = accuracy_score(y_test_smt, y_pred)
    # # print(ac)

    # # Performance Measure of KNN
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import classification_report

    # # y_pred = modelKNN.predict(x_test_smt)
    # # print(confusion_matrix(y_test_smt, y_pred))
    # # print(classification_report(y_test_smt, y_pred))
    # # from sklearn.metrics import cohen_kappa_score
    # cmKNN = confusion_matrix(y_test_smt, modelKNN.predict(x_test_smt))
    # TP = cmKNN[1,1]
    # TN = cmKNN[0,0]
    # FP = cmKNN[0,1]
    # FN = cmKNN[1,0]

    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/float(TP+FN)
    # # Specificity or true negative rate
    # TNR = TN/float(TN+FP)
    # # Precision or positive predictive value
    # PPV = TP/float(TP+FP)
    # # Negative predictive value
    # NPV = TN/float(TN+FN)
    # # Fall out or false positive rate
    # FPR = FP/float(FP+TN)
    # # False negative rate
    # FNR = FN/float(TP+FN)
    # # False discovery rate
    # FDR = FP/float(TP+FP)
    # # Accuracy
    # totalKNN = sum(sum(cmKNN))
    # Accuracy = (TN+TP)/totalKNN
    # # MCC
    # val = (TP * TN) - (FP * FN)
    # MCC_KNN = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # # Cohen Kappa
    # # Y_pred = modelKNN.predict(x_test_smt)
    # # cohen_score = cohen_kappa_score(y_test_smt, Y_pred)
    # algo_names.append("KNN")
    # all_accuracy.append(ac)
    # all_sensitivity.append(TPR)
    # all_specificity.append(TNR)
    # f1Score=2*(TPR*PPV)/(TPR+PPV)
    # all_f1Score.append(f1Score)
    for i in range(len(res)):
        if res[i]>2:
            res[i]=1
        else:
            res[i]=0
    y_pred=res
    ac = accuracy_score(y_test_smt, y_pred)
    # print(ac)
    # Performance Measure of KNN
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    cmKNN = confusion_matrix(y_test_smt, y_pred)
    TP = cmKNN[1,1]
    TN = cmKNN[0,0]
    FP = cmKNN[0,1]
    FN = cmKNN[1,0]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/float(TP+FN)
    # Specificity or true negative rate
    TNR = TN/float(TN+FP)
    # Precision or positive predictive value
    PPV = TP/float(TP+FP)
    # Negative predictive value
    NPV = TN/float(TN+FN)
    # Fall out or false positive rate
    FPR = FP/float(FP+TN)
    # False negative rate
    FNR = FN/float(TP+FN)
    # False discovery rate
    FDR = FP/float(TP+FP)
    # Accuracy
    totalKNN = sum(sum(cmKNN))
    Accuracy = (TN+TP)/totalKNN
    # MCC
    val = (TP * TN) - (FP * FN)
    MCC_KNN = val / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    algo_names.append("Voting")
    all_accuracy.append(ac)
    all_sensitivity.append(TPR)
    all_specificity.append(TNR)
    f1Score=2*(TPR*PPV)/(TPR+PPV)
    all_f1Score.append(f1Score)
    Performance_Matrix = {
    "Algo Names": algo_names,
    "Accuracy": all_accuracy,
    "Sensitivity": all_sensitivity,
    "Specificity": all_specificity,
    "F1-Score": all_f1Score
    }
    Performance_Matrix = pd.DataFrame(Performance_Matrix)

    return Performance_Matrix

def trainTestSplit(Data,class_name,test_size,random_state):
    X = Data.drop(columns=['class'])  # Features
    # X=X[column_names]
    y = Data['class'] 
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.20, random_state=42)

def quick_result(Data,class_name,test_size,random_state):
    X_train, X_test, y_train, y_test = trainTestSplit(Data,class_name='class',test_size=0.2,random_state=42)
    return all_result(X_train,X_test,y_train,y_test)


def svc_result(Data,class_name,test_size,random_state):
    X_train, X_test, y_train, y_test = trainTestSplit(Data,class_name='class',test_size=0.2,random_state=42)
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    modelSVC = SVC(probability=True)
    modelSVC.fit(X_train, y_train)
    y_pred = modelSVC.predict(X_test)
    y_final = y_test
    y_pred_svc = y_pred
    ac = accuracy_score(y_test, y_pred)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    cm=confusion_matrix(y_test, y_pred)
    cr=classification_report(y_test, y_pred)
    return ac,cm,cr

def rf_result(Data,class_name,test_size,random_state,n_estimators = 100):
    X_train, X_test, y_train, y_test = trainTestSplit(Data,class_name='class',test_size=0.2,random_state=42)
    from sklearn import ensemble
    from sklearn.metrics import accuracy_score
    modelRFC = ensemble.RandomForestClassifier(n_estimators = 100)
    modelRFC.fit(X_train, y_train)
    y_pred = modelRFC.predict(X_test)
    y_final = y_test
    y_pred_svc = y_pred
    ac = accuracy_score(y_test, y_pred)
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    cm=confusion_matrix(y_test, y_pred)
    cr=classification_report(y_test, y_pred)
    return ac,cm,cr

import time

def animate_running_method(message):
    indicator = "|/-\\"
    while True:
        for char in indicator:
            print(f"\r{message}... {char}", end="")
            time.sleep(0.1)
           