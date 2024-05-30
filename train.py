import pickle
import pandas as pd
import numpy as np
import Get_data
import functools
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

"""Przygotowanie częciowe funkcji wykorzystanych w trakcie oceniania modeli"""
recall_partial = functools.partial(recall_score, average = 'weighted')
f1_partial = functools.partial(recall_score, average = 'weighted')

"""Funkcja przetwarzająca wynik w postaci prawdopodobieństwa na klasy <- wybór 5 o najwyższym prawdopodobieństwie dla danej"
 "klasy"""
def parse_results(array_proba,rookies):
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

    else:
        first_rookie_5_idx = array_proba[:, 1].argsort()[-5:][::-1]
        array_proba[first_rookie_5_idx, :] = -1
        y_pred[first_rookie_5_idx, :] = 4
        second_rookie_5_idx = (array_proba[:, 2]).argsort()[-5:][::-1]
        y_pred[second_rookie_5_idx, :] = 5

    return y_pred
"""Wyświetlenie wyników predykcji"""
def print_results(y_pred,X_test,rookies = False):
    if not rookies:
        print(X_test.iloc[np.where(y_pred == 1)[0]])
        print(X_test.iloc[np.where(y_pred == 2)[0]])
        print(X_test.iloc[np.where(y_pred == 3)[0]])
        return [X_test.iloc[np.where(y_pred == 1)[0]],X_test.iloc[np.where(y_pred == 2)[0]],X_test.iloc[np.where(y_pred == 3)[0]]]
    else:
        print(X_test.iloc[np.where(y_pred == 4)[0]])
        print(X_test.iloc[np.where(y_pred == 5)[0]])
        return [X_test.iloc[np.where(y_pred == 4)[0]], X_test.iloc[np.where(y_pred == 5)[0]]]
"""Funkcja obliczająca uzyskany wynik według kryteriów zawartych w opisie projektu"""
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

        """Odczytanie zapisanych danych"""
        seasons = []
        rookie_data_names = []
        data_names = []
        for i in range (30):
            seasons.append(str(2023 - i) + '-' + str(((24 - i) % 100)).zfill(2))
            data_names.append(seasons[i] + "_" + "Regular Season")
            rookie_data_names.append(seasons[i] + "_" + "Regular Season" + "_Rookies")

        Season_data = Get_data.Read_Data_from_Pickles(data_names,"./Data/")
        Season_data_rookies = Get_data.Read_Data_from_Pickles(rookie_data_names, "./Data/")
        target = Get_data.Load_data_form_CSV(["target"], "./Target/")
        Season_data_annotated = []
        Season_data_rookies_annotated = []
        """Połączenie danych z Targetem"""
        for Data,Data_Rookies,season in zip(Season_data,Season_data_rookies,seasons):
            Season_data_annotated.append(Get_data.Combine_Data_and_target(Data,target[0],season,rookies_only=False))
            Season_data_rookies_annotated.append(Get_data.Combine_Data_and_target(Data_Rookies, target[0], season, rookies_only=True))

        """Usunięcie graczy o zbyt niskim czasie gry - https://www.statmuse.com/nba/ask/fewest-games-played-by-all-nba-player """
        """ Dodatkiwy margines błędu dla rookich - z powodu niższej liczby danych usuwanie mniejszej ilości graczy 
        skutkowało polepszeniem wyniku"""
        playtime_th = 1200
        playtime_th_rookies = 800

        """ Utworzenie zbiorów testowych oraz treningowych - ostatecznie dane brane są z ostatnich 8 lat"""
        X_train = Get_data.drop_low_playtime(pd.concat(Season_data_annotated[1:8],ignore_index=True),playtime_th)
        X_test = Get_data.drop_low_playtime(Season_data_annotated[0],playtime_th)
        X_rookies_train = Get_data.drop_low_playtime(pd.concat(Season_data_rookies_annotated[1:8],ignore_index=True),playtime_th_rookies)
        X_rookies_test = Get_data.drop_low_playtime(Season_data_rookies_annotated[0],playtime_th_rookies)
        y_train = X_train['Target']
        y_test = X_test['Target']
        y_rookies_train = X_rookies_train['Target']
        y_rookies_test = X_rookies_test['Target']

        """ Siatka parametrów gridsearch dla klasyfikatora RF - który wstępnie dawał najlepsze wyniki i dla niego przeprowadzono
        przeszukiwanie parametrów"""
        param_grid = {'randomforestclassifier__n_estimators': [50,100,500,2000],
                      'randomforestclassifier__max_depth':[2,5,10,15],
                      'randomforestclassifier__max_features':[2,5,10,20,'sqrt','log2'],
                      'randomforestclassifier__ccp_alpha':[0.0,0.005],
                      'randomforestclassifier__criterion':['gini', 'entropy', 'log_loss'],
                      'randomforestclassifier__class_weight':[None,'balanced']}

        """Testowane Modele to RF, SVC oraz K-Neigbours, parametry dla RF wybrane przez gridsearch, dla pozostałych wstępnie dobrane
        ręcznie, dla każdego modelu przeprowadzenie jest skalowanie danych poprzez standardScaler"""
        gnb = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
        gnb_rookies = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
        clf_SVC = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True,kernel='rbf',class_weight='balanced'))
        clf_SVC_Rookies = make_pipeline(StandardScaler(), SVC(gamma='scale',probability=True,kernel='rbf'))
        rf = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=50,max_depth=15,max_features=20,criterion='log_loss',class_weight='balanced',random_state=42))
        rf_rookies = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=50,max_depth=15,max_features=20,criterion='log_loss',class_weight='balanced',random_state=42))
        #50 15 20 log_loss balanced
        """Przy dobieraniu parametrów dla RF wykorzystano F1 score zamiast domyślnego accuracy"""

        scorr = make_scorer(f1_score, response_method='predict', average='micro')
        """
        Z powodu czasu obliczeń, parametry dobrane dla 1 modelu wykorzystane zostaną dla obydwóch modelów
        """
        #search = GridSearchCV(rf,param_grid,n_jobs=-1,verbose=5,refit=True,scoring=scorr)
        #search.fit(X_train.loc[:,~X_train.columns.isin(['PLAYER','TEAM','Target'])], y_train)
        #print(search.best_params_)
        #filehandler_param = open("Best_params.pkl", 'w')
        #pickle.dump(search.best_params_, filehandler_param)

        """Trenowanie modeli, modele przyjmują całą tabelkę lederboards z wyjątkiem imienia i drużyny gracza
        https://www.nba.com/stats/leaders?SeasonType=Regular+Season&PerMode=Totals"""

        gnb.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM','Target'])], y_train)
        gnb_rookies.fit(
            X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM', 'Target'])],
            y_rookies_train)
        clf_SVC.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM', 'Target'])], y_train)
        clf_SVC_Rookies.fit(X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM', 'Target'])],y_rookies_train)
        rf.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM',  'Target'])], y_train)
        rf_rookies.fit(X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM',  'Target'])],
                            y_rookies_train)

        """Save the fitted model"""

        filehandler_S = open("./models/Best_S.pkl", 'wb')
        filehandler_R = open("./models/Best_R.pkl", 'wb')

        pickle.dump(rf, filehandler_S)
        pickle.dump(rf_rookies, filehandler_R)

        filehandler_S.close()
        filehandler_R.close()

        """predykcja"""
        result = rf.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies = rf_rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_gnb = gnb.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies_gnb = gnb_rookies.predict_proba(
            X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_SVC = clf_SVC.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies_SVC = clf_SVC_Rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])

        """Ocena oraz wyświetlenie otrzymanych wyników zgodnie z funkcją opisaną w opisie projektu"""
        scorer = make_scorer(custom_scorrer,greater_is_better=True,response_method='predict_proba')

        print("################################RANDOM FOREST##########################################")
        y_pred = parse_results(result,False)
        y_pred_rookies = parse_results(result_rookies, True)
        a = scorer(rf, X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target', 'Target_1'])], y_test)
        b = scorer(rf_rookies ,
                   X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target', 'Target_1'])],
                   y_rookies_test)
        print_results(y_pred,X_test)
        print_results(y_pred_rookies,X_rookies_test, True)
        print("Wynik: ", a + b, "/450")
        print("################################SVC##########################################")
        y_pred = parse_results(result_SVC, False)
        y_pred_rookies = parse_results(result_rookies_SVC, True)
        a = scorer(clf_SVC, X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target','Target_1'])], y_test)
        b = scorer(clf_SVC_Rookies,X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target','Target_1'])], y_rookies_test)
        print_results(y_pred, X_test)
        print_results(y_pred_rookies, X_rookies_test,
        True)
        print("Wynik: ",a+b,"/450")
        print("################################K-neigbours##########################################")
        y_pred = parse_results(result_gnb, False)
        y_pred_rookies = parse_results(result_rookies_gnb, True)
        a = scorer(gnb, X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target','Target_1'])], y_test)
        b = scorer(gnb_rookies,
                   X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target','Target_1'])],
                   y_rookies_test)
        print_results(y_pred, X_test)
        print_results(y_pred_rookies, X_rookies_test, True)
        print("Wynik: ", a + b, "/450")
        #print(X_train.head(40))