# Opis Projektu WZUM - All NBA Team Predictor
Plik ten zawiera opis działania programu który jest częścią zaliczenia zajęć laboratoryjnych z przedmiotu WZUM na Politechnice Poznańskiej. Zadaniem programu jest wybranie zawodników do nagrody [All NBA Team](https://www.nba.com/news/history-all-nba-teams) na podstawie statystyk zgromadzonych na oficjalniej stronie [NBA.com](https://www.nba.com/stats/)

![alt](/images/Screenshot_63.png)

# Opis działania programu
## Pozyskiwanie danych
Dane pozyskane zostały przy pomocy [nba_api](https://github.com/swar/nba_api) z wykorzystaniem modułu leaderboards fragment kodu pozyskującego dane widoczny jest na poniższym listingu
```python
from nba_api.stats.endpoints import leagueleaders
import pickle
import pandas as pd
"""Pobranie danych z modłu drabinek dla dowolnego typu sezonu z podziałem na tabelkę Seniorów oraz Rookich"""
def Get_n_Year_data(N_of_years,Stat_type,Player_type,User_name = ''):
    leagueleader = []
    names = []
    for i in range(N_of_years):
        season = str(2023-i) + '-' +str(((24-i)%100)).zfill(2)
        names.append(season +"_"+Stat_type + User_name)
        print(season)
        data = leagueleaders.LeagueLeaders(league_id = '00',per_mode48='Totals',season=season,
                                                        season_type_all_star=Stat_type
                                               ,scope=Player_type).get_data_frames()[0]
        leagueleader.append(data)
    return leagueleader,  names
```
Dane te następnie przechowane zostały lokalnie w postaci plików .plk Kod służący do manipulacji, zapisania oraz pozyskania danych znajduje się w pliku **GetData.py**

Dane zostały dodatkowo opatrzone etykietami na podstawie pliku **target.csv** utworzonego ręcznie na podstawie wyników nagrody All NBA Team z ostatnich 30 lat.
## Trenowanie modelu
Skrypt treningowy zawarty jest w pliku **train.py**. W tym miejscu opisany zostanie sposób postępowania który doprowadził do uzyskania finalnego modelu.

### Testowane modele

Posługując się informacjami z dokumentacji biblioteki [scikit-learn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) do wstępnych testów wybrano trzy klasyfikatory
+ SVC
+ KNeigbours Classifier
+ Random Forest Classifier
![Schemat_wyboru](images/ml_map.svg)
Modele były wstępnie testowane dla parametrów dobranych ręcznie oraz dla danych z ostatnich 25lat. Z czego 3 lata były wykorzystywane w tym etapie prac jako zbiór testowy. Podział danych dokonany został w następujący sposób:
```python

 """ Utworzenie zbiorów testowych oraz treningowych - ostatecznie dane brane są z ostatnich 8 lat"""
        X_train = Get_data.drop_low_playtime(pd.concat(Season_data_annotated[1:8],ignore_index=True),playtime_th)
        X_test = Get_data.drop_low_playtime(Season_data_annotated[0],playtime_th)
        X_rookies_train = Get_data.drop_low_playtime(pd.concat(Season_data_rookies_annotated[1:8],ignore_index=True),playtime_th_rookies)
        X_rookies_test = Get_data.drop_low_playtime(Season_data_rookies_annotated[0],playtime_th_rookies)
        y_train = X_train['Target']
        y_test = X_test['Target']
        y_rookies_train = X_rookies_train['Target']
        y_rookies_test = X_rookies_test['Target']

```
W tym etapie zdecydowano się na odrzucenie graczy z zbyt niskim czasem gry w sezonie na podstawie informacji zawartych na [stronie](https://www.statmuse.com/nba/ask/least-amount-of-games-played-by-a-player-to-make-all-nba-team-since-2000). Gdzie dla nowych graczy próg ustawiono dodatkowo niżej aby nie ograniczyć zbytnio zbioru treningowego.

Jak widać na powyższym kodzie istnieją dwie bazy treningowe. Wszystkich graczy oraz tylko graczy którzy grają swój pierwszy sezon. Z tego też powodu zdecydowano się na utworzenie dwóch osobnych klasyfikatorów których zadaniem było odpowiednio dobranie 3 najlepszych All NBA teams oraz 2 najlepszych All NBA rookie team. Ostatecznie 1 klasyfikator był wstanie uzyskać porównywalne wyniki do 2 uczących się na osobnych zbiorach dedykowanych do danego przedziału graczy jednakże zachowano rozwiązanie wykorzystujące 2 osobne klasyfikatory.

Wszystkie testowane klasyfikatory posiadały w swoim pipelinie dodatkowy scaller danych wejściowych:

```python
gnb = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
gnb_rookies = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=20,weights='uniform',algorithm='brute'))
clf_SVC = make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True,kernel='rbf',class_weight='balanced'))
clf_SVC_Rookies = make_pipeline(StandardScaler(), SVC(gamma='scale',probability=True,kernel='rbf'))
rf = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=50,max_depth=15,max_features=20,criterion='log_loss',class_weight='balanced',random_state=42))
rf_rookies = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=50,max_depth=15,max_features=20,criterion='log_loss',class_weight='balanced',random_state=42))
```

Testowano dodatkowo rozwiązanie wykorzystujące funkcję ```SelectKBest()``` jako drugą część potoku, jednakże wpływała ona negatywnie na otrzymywane wyniki. Ostatecznie wszystkie modele otrzymywały pełną tabelę danych z wyłączeniem imienia i nazwiska gracza oraz nazwy jego drużyny.

```python
rf.fit(X_train.loc[:, ~X_train.columns.isin(['PLAYER', 'TEAM',  'Target'])], y_train)
rf_rookies.fit(X_rookies_train.loc[:, ~X_rookies_train.columns.isin(['PLAYER', 'TEAM',  'Target'])],y_rookies_train)
```
Predykcja modeli odbywała się w trybie generacji prawdopodobieństwa:
```python
 """predykcja"""
        result = rf.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies = rf_rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_gnb = gnb.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies_gnb = gnb_rookies.predict_proba(
        X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_SVC = clf_SVC.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
        result_rookies_SVC = clf_SVC_Rookies.predict_proba(X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
```
Działanie to miało na celu umożliwić dodatkowe przetworzenie uzyskanego wyniku a nie otrzymanie gotowych klas. W ten sposób wymuszono klasyfikację tylko 5 zawodników do klas odpowiadającym poszczególnym najlepszym piątkom.
Dokonano tego przy pomocy funkcji ```parse_results()```
```python

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

```
Która przypisywała odpowiednie klasy a następnie zwracała ostateczną predykcję.
### Ocena modeli

W celu zmierzenia jakości działania modeli utworzono własną funkcję która przypisywała modelom punkty od 0/450 zgodnie z opisem projektu zaliczeniowego.

```python
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
```
### Dobór parametrów oraz rozmiaru zbioru uczącego - Wybór Random Forest

Po wstępnej próbie testów najlepsze wyniki otrzymywano dla klasyfikatora Random Forest. Dlatego zdecydowano się na dobranie dla niego odpowiednich parametrów. W tym celu wykorzystano funkcję ```GridSearchCV()``` z poniższą siatką parametrów

```python
 """ Siatka parametrów gridsearch dla klasyfikatora RF - który wstępnie dawał najlepsze wyniki i dla niego przeprowadzono
        przeszukiwanie parametrów"""
        param_grid = {'randomforestclassifier__n_estimators': [50,100,500,2000],
                      'randomforestclassifier__max_depth':[2,5,10,15],
                      'randomforestclassifier__max_features':[2,5,10,20,'sqrt','log2'],
                      'randomforestclassifier__ccp_alpha':[0.0,0.005],
                      'randomforestclassifier__criterion':['gini', 'entropy', 'log_loss'],
                      'randomforestclassifier__class_weight':[None,'balanced']}
```
W wyniku długiego czasu trwania przesukiwania zdecydowano się dokonać go tylko dla klasyfikatora dla wszystkich graczy a znalezione parametry zastosować do obu końcowych klasyfikatorów. Do ocenienia jakości wyszukanych parametrów zastosowano f1_score zdefiniowany w następujący sposób:

```python
scorr = make_scorer(f1_score, response_method='predict', average='micro')
search = GridSearchCV(rf,param_grid,n_jobs=-1,verbose=5,refit=True,scoring=scorr)
```

Następnie przyjrzano się rozmiarowi zbioru treningowego. Zaobserwowano już podczas wstępnych testów pogorszenie się predykcji przy wykorzystywaniu danych z dawnych lat. Ostatecznie przyjęto za zbiór treningowy dla finalnych klasyfikatorów okres 8 sezonów porzedzający sezon 2023-24.

Wyuczone modele zostały następnie zapisane do plików .plk

```
"""Save the fitted model"""

        filehandler_S = open("./models/Best_S.pkl", 'wb')
        filehandler_R = open("./models/Best_R.pkl", 'wb')

        pickle.dump(rf, filehandler_S)
        pickle.dump(rf_rookies, filehandler_R)

        filehandler_S.close()
        filehandler_R.close()
```
## Predykcja sezonu 2023-24

Plik dokonujący predykcji na bierzący sezon NBA nazywa się **predict.py**. Przyjmuje on 1 parametr wejściowy który służy jako lokalizacja dla wynikowego pliku .json zawierającego wynik predykcji. Dane wejściowe znajdują się w folderze Data a modele w folderze models. Skrypt wykorzystuje wcześniej wuczone modele z pliku **Train.py**