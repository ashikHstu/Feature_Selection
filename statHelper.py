# find and remove correlated features
from helper import *
class myResult:
    def __init__(self, result, features, cromosome,data):
        self.result = result
        self.features = features
        self.cromosome = cromosome
        self.data=data

    def __repr__(self):
        return f"MyDataContainer(result={self.result}, features={self.features}, cromosome={self.cromosome},data={self.data})"

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    
    corr_matrix = dataset.corr()
    
    for i in range(len(corr_matrix.columns)):
    
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import DropCorrelatedFeatures, SmartCorrelatedSelection
from sklearn.model_selection import train_test_split, cross_validate
def SmartCorrelatinSelection1(data,class_name='class'):
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=[class_name], axis=1),
    data[class_name],
    test_size=0.3,
    random_state=0)
    
    rf = RandomForestClassifier(
    n_estimators=10,
    random_state=20,
    n_jobs=4,
    )

    # correlation selector
    sel = SmartCorrelatedSelection(
        variables=None, # if none, selector examines all numerical variables
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=rf,
        scoring="roc_auc",
        cv=3,
    )

    # this may take a while, because we are training
    # a random forest per correlation group

    sel.fit(X_train, y_train)
    
    # groups of correlated features

    sel.correlated_feature_sets_

        # lets examine the performace of a random forest based on
    # each feature from the second group, to understand
    # what the transformer is doing

    # select second group of correlated features
    group = sel.correlated_feature_sets_[1]

    # build random forest with cross validation for
    # each feature

    for f in group:

        model = cross_validate(
            rf,
            X_train[f].to_frame(),
            y_train,
            cv=3,
            return_estimator=False,
            scoring='roc_auc',
        )

        print(f, model["test_score"].mean())
        
    return sel.features_to_drop_
def SmartCorrelatinSelection2(data,class_name='class'):
    # correlation selector
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(labels=['class'], axis=1),
        data['class'],
        test_size=0.3,
        random_state=0)
    sel = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="variance",
        estimator=None,
        scoring="roc_auc",
        cv=3,
    )

    sel.fit(X_train, y_train)
    
    # let's examine the variance of the features from the second group of
    # correlated ones

    group = list(sel.correlated_feature_sets_[1])

    X_train[group].std()
    return sel.features_to_drop_

def CorrelationReduction(data,class_name="class",threshold=0.9,type=0):
    X=data.drop(labels=[class_name], axis=1)
    Y=data[class_name]
    if type==0:
        corr_features = correlation(X,threshold)
    elif type==1:
        corr_features=SmartCorrelatinSelection1(data.copy(),class_name='class')
    elif type==2:
        corr_features=SmartCorrelatinSelection2(data.copy(),class_name='class')
    else:
        print("Unexpected type!")
        corr_features=['Fp1-LE_alpha']

    data.drop(labels=corr_features, axis=1, inplace=True)
    return data
def getCromosome(data1,data2):
    data1=data1.drop(labels=["class"], axis=1)
    data2=data2.drop(labels=["class"], axis=1)
    column_names1 = data1.columns.tolist()  
    column_names2 = data2.columns.tolist()
    cromo_size=len(column_names1)
    cromo = [0] * cromo_size
    cur=0
    for feature in column_names1:
        if feature in column_names2:
            cromo[cur]=1
        cur=cur+1
    return cromo

def demoResult(data,data2):
    features = data2.columns.tolist()
    features.remove("class")
    res=quick_result(data2.copy(),class_name="class",test_size=0.20,random_state=42)
    cromo=getCromosome(data.copy(),data2.copy())
    makeResult=myResult(res,features,cromo,data2)
    return makeResult

def correlationResult(data,class_name='class',Type=0):
    data2=CorrelationReduction(data.copy(),class_name=class_name,threshold=0.85,type=Type)
    res=demoResult(data.copy(),data2.copy())
    return res
    
    