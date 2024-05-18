from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leagueleaders
import pickle
import pandas as pd
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

def Save_Data_to_Pickle(leagueleader,names,dir_path):
    i = 0
    for df in leagueleader:
        df.to_pickle(dir_path + names[i] + ".pkl")
        i += 1

def Read_Data_from_Pickles(names,dir_path):
    leagueladers = []
    for name in names:
            df = pd.read_pickle(dir_path + name + ".pkl")
            leagueladers.append(df)
    return leagueladers

def Load_data_form_CSV(names,dir_path):
    data = []
    for name in names:
            df = pd.read_csv(dir_path + name + ".csv")
            data.append(df)
    return data

def Combine_Data_and_target(Data,Target,season):
    result = Data.copy()
    result["Target"] = 0
    target_class = 1
    for i in range (25):
        result.loc[result["PLAYER"] == Target[season].iloc[i],'Target'] = target_class
        if i % 5 == 4:
            target_class += 1
    return result

if __name__ == "__main__":
    Data, Names = Get_n_Year_data(30,"Regular Season",'RS')
    Save_Data_to_Pickle(Data,Names,"./Data/")
    #Data, Names = Get_n_Year_data(30,"Regular Season",'Rookies','_Rookies')
    #Save_Data_to_Pickle(Data,Names,"./Data/")
    Data_loaded = Read_Data_from_Pickles(Names,"./Data/")
    target = Load_data_form_CSV(["target"],"./Target/")
    combined = Combine_Data_and_target(Data[0],target[0],"2022-23")
    combined_tr = combined[['PLAYER','Target']].copy()
    print(combined[['PLAYER','Target']])


