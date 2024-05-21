import pickle
import pandas as pd
import numpy as np
import Get_data
import functools
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
recall_partial = functools.partial(recall_score, average = 'weighted')
f1_partial = functools.partial(recall_score, average = 'weighted')
def parse_results(array_proba,rookies,verbose = False):
    y_pred = np.zeros((array_proba.shape[0], 1))
    if not rookies:
        first_5_idx = array_proba[:, 1].argsort()[-5:][::-1]
        array_proba[first_5_idx, :] = -1
        y_pred[first_5_idx, :] = 1
        second_5_idx = (array_proba[:, 2]).argsort()[-5:][::-1]
        array_proba[second_5_idx, :] = -1
        y_pred[second_5_idx, :] = 2
        third_5_idx = (array_proba[:, 3]).argsort()[-5:][::-1]
        y_pred[third_5_idx, :] = 3
        if verbose:
            print(X_test.iloc[first_5_idx])
            print(X_test.iloc[second_5_idx])
            print(X_test.iloc[third_5_idx])
    else:
        first_rookie_5_idx = array_proba[:, 1].argsort()[-5:][::-1]
        array_proba[first_rookie_5_idx, :] = -1
        y_pred[first_rookie_5_idx, :] = 4
        second_rookie_5_idx = (array_proba[:, 2]).argsort()[-5:][::-1]
        y_pred[second_rookie_5_idx, :] = 5
        if verbose:
            print(X_rookies_test.iloc[first_rookie_5_idx])
            print(X_rookies_test.iloc[second_rookie_5_idx])

    return y_pred

def custom_scorrer(y_true,y_pred):

        score = 0
        bonus_idx = [0,0,0,0,0]
        bonus_pts = [0,0,5,10,20,40]
        if y_pred.shape[1] > 3:
            rookies = False
        else:
            rookies = True
        y_pred = parse_results(y_pred,rookies)
        for a,b in zip(np.nditer(y_true),np.nditer(y_pred)):
            if a == b and a != 0 and b!= 0:
                score += 10
                bonus_idx[int(a)-1] += 1
            if (abs(a-b) == 1 and b !=  0 and a!= 0):
                score += 8
            if (abs(a-b) == 2 and b != 0 and a != 0):
                score += 6
        for x in bonus_idx:
            score += bonus_pts[x]
        return score


if __name__ == "__main__":

    seasons = []
    rookie_data_names = []
    data_names = []
    for i in range (25):
        seasons.append(str(2022 - i) + '-' + str(((23 - i) % 100)).zfill(2))
        data_names.append(seasons[i] + "_" + "Regular Season")
        rookie_data_names.append(seasons[i] + "_" + "Regular Season" + "_Rookies")

    Season_data = Get_data.Read_Data_from_Pickles(data_names,"./Data/")
    Season_data_rookies = Get_data.Read_Data_from_Pickles(rookie_data_names, "./Data/")
    target = Get_data.Load_data_form_CSV(["target"], "./Target/")
    Season_data_annotated = []
    Season_data_rookies_annotated = []

    for Data,Data_Rookies,season in zip(Season_data,Season_data_rookies,seasons):
        Season_data_annotated.append(Get_data.Combine_Data_and_target(Data,target[0],season,rookies_only=False))
        Season_data_rookies_annotated.append(Get_data.Combine_Data_and_target(Data_Rookies, target[0], season, rookies_only=True))

    playtime_th = -400

    X_train = Get_data.drop_low_playtime(pd.concat(Season_data_annotated[3:29],ignore_index=True),playtime_th)
    X_test = Get_data.drop_low_playtime(Season_data_annotated[0],playtime_th)
    X_rookies_train = Get_data.drop_low_playtime(pd.concat(Season_data_rookies_annotated[3:29],ignore_index=True),playtime_th)
    X_rookies_test = Get_data.drop_low_playtime(Season_data_rookies_annotated[0],playtime_th)

    y_train = X_train['Target']
    y_test = X_test['Target']
    y_rookies_train = X_rookies_train['Target']
    y_rookies_test = X_rookies_test['Target']

    gnb = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
    gnb_rookies = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
    clf_SVC = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True,kernel='rbf',class_weight='balanced'))
    clf_SVC_Rookies = make_pipeline(StandardScaler(), SVC(gamma='scale',probability=True,kernel='rbf'))
    rf = make_pipeline(StandardScaler(),RandomForestClassifier(random_state=42,n_estimators=5000,max_depth=5,max_features=10,criterion='entropy',oob_score = f1_partial))
    rf_rookies = make_pipeline(StandardScaler(),RandomForestClassifier(random_state=42, n_estimators=5000, max_depth=5,max_features=10,criterion='entropy',oob_score = f1_partial))
    rf.fit(X_train.loc[:,~X_train.columns.isin(['PLAYER','TEAM','Target'])], y_train)
    rf_rookies.fit(X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_rookies_train)
    gnb.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_train)
    gnb_rookies.fit(
        X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM', 'Target'])],
        y_rookies_train)
    clf_SVC.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_train)
    clf_SVC_Rookies.fit(X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM', 'Target'])],y_rookies_train)

    result = rf.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_rookies = rf_rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_gnb = gnb.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_rookies_gnb = gnb_rookies.predict_proba(
        X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_SVC = clf_SVC.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_rookies_SVC = clf_SVC_Rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])

    scorer = make_scorer(custom_scorrer,greater_is_better=True,response_method='predict_proba')

    y_pred = parse_results(result,False,True)
    y_pred_rookies = parse_results(result_rookies, True,True)
    a = scorer(rf,X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])],y_test)
    b = scorer(rf_rookies, X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_rookies_test)
    print("Wynik: ", a + b, "/450")
    print("################################WIELKIE SVC##########################################")
    y_pred = parse_results(result_SVC, False,True)
    y_pred_rookies = parse_results(result_rookies_SVC, True,True)
    a = scorer(clf_SVC, X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_test)
    b = scorer(clf_SVC_Rookies,X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_rookies_test)
    print("Wynik: ",a+b,"/450")
    print("################################NAIVE BAYES##########################################")
    y_pred = parse_results(result_gnb, False, True)
    y_pred_rookies = parse_results(result_rookies_gnb, True, True)
    a = scorer(gnb, X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_test)
    b = scorer(gnb_rookies,
               X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])],
               y_rookies_test)
    print("Wynik: ", a + b, "/450")
    #print(X_train.head(40))

