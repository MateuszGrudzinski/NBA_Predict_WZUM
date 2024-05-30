import Get_data
import pandas as pd
from pickle import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train import parse_results
from train import print_results
import argparse
import numpy as np
from pathlib import Path
import json
parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str)
args = parser.parse_args()

results_file = Path(args.results_file)

if __name__ == "__main__":
    """Pobranie danych oraz odrzucenie graczy o niskim czasie gry"""
    season = "2023-24"
    data_name = season + "_" + "Regular Season"
    rookie_data_name = season + "_" + "Regular Season" + "_Rookies"

    Season_data = Get_data.Read_Data_from_Pickles([data_name],"./Data/")
    Season_data_rookies = Get_data.Read_Data_from_Pickles([rookie_data_name], "./Data/")
    target = Get_data.Load_data_form_CSV(["target"], "./Target/")

    playtime_th = 1200
    playtime_th_rookies = 200

    X_test = Get_data.drop_low_playtime(Season_data[0],playtime_th)
    X_rookies_test = Get_data.drop_low_playtime(Season_data_rookies[0],playtime_th_rookies)
    """Załadowanie modelu (pipeline)"""
    with open("./models/Best_S.pkl", "rb") as f:
        clf = load(f)
    with open("./models/Best_R.pkl", "rb") as f:
        clf_r = load(f)

    "predykcj"
    result = clf.predict_proba(X_test.loc[:, ~X_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])
    result_rookies = clf_r.predict_proba(
        X_rookies_test.loc[:, ~X_rookies_test.columns.isin(['PLAYER', 'TEAM', 'Target'])])

    y_pred = parse_results(result, False)
    y_pred_rookies = parse_results(result_rookies, True)

    first,second,third = print_results(y_pred, X_test)
    first_R,second_R = print_results(y_pred_rookies, X_rookies_test, True)

    first.reset_index()
    """Zapisanie i wyświetlenie wyniku predykcji"""
    prediction_dict = {}
    predictions = []
    names = ["first all-nba team","second all-nba team","third all-nba team","first rookie all-nba team","second rookie all-nba team"]
    idx = 0
    for frame in [first,second,third,first_R,second_R]:
        predictions = []
        for index, row in frame.iterrows():
            predictions.append(row['PLAYER'])
        prediction_dict[names[idx]] = predictions
        idx+=1
    print(prediction_dict)
    with results_file.open('w') as output_file:
        json.dump(prediction_dict, output_file, indent=4)
