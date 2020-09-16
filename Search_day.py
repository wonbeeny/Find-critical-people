from datetime import timedelta
import numpy as np, pandas as pd
from tqdm import tqdm, tqdm_notebook

def Search_day_make_csv(data_name, day, save_file = None ,file_name = None):
    df = pd.read_csv(f'{data_name}')
    
    df['start_time'] = pd.to_datetime(df['start_time'], format = '%Y-%m-%d %H:%M')
    df['end_time'] = pd.to_datetime(df['end_time'], format = '%Y-%m-%d %H:%M')
    check_end_time = pd.to_datetime(f'{day}') + timedelta(days=1)
    check_start_time = check_end_time - timedelta(days = 15)
    print("체크해야 하는 시작 시점 :", check_start_time, " 발생 일자 + 1일 :", check_end_time)

    fr_danger_to_occur_route_detail = pd.DataFrame([], columns = ['new_p_id', 'start_time', 'end_time', 'stay_time', 'fr_danger_to_occur_route'])


    for i in tqdm_notebook(range(len(df['new_p_id'].unique()))):
        temp = df.loc[(df['new_p_id'] == df['new_p_id'].unique()[i])].reset_index(drop=True)
        temp = temp.loc[(temp['start_time'] > check_start_time) |
                              (temp['end_time'] < check_end_time)].reset_index(drop=True).reset_index(drop=True)

        new_p_id = temp['new_p_id'][0]
        start_time = temp['start_time'][0]
        if check_start_time > start_time:
            start_time = check_start_time
        end_time = temp['end_time'][len(temp)-1]
        stay_time = start_time - end_time
        fr_danger_to_occur_route = '->'.join(temp['new_bts_id'])

        temp1 = pd.DataFrame( [[new_p_id, start_time, end_time, stay_time, fr_danger_to_occur_route]],
                            columns=['new_p_id', 'start_time', 'end_time', 'stay_time', 'fr_danger_to_occur_route'] )
        fr_danger_to_occur_route_detail = pd.concat([fr_danger_to_occur_route_detail, temp1], axis=0).reset_index(drop=True)

    if save_file == True:
        fr_danger_to_occur_route_detail.to_csv('danger_route_{}.csv'.format(f'{file_name}'), index = False)
    
    return fr_danger_to_occur_route_detail