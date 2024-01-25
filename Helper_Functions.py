import math
import json
import re
import ast

import numpy as np
import pandas as pd

import soccerdata as sd

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from mplsoccer import Pitch, VerticalPitch



def get_data():
    '''
    in: -
    out: dataframe with match events
    '''
    df = pd.read_csv('match_events.csv')
    return df

def transform_sort_data(data):
    '''
    in: match events dataframe
    out: transformed match events df
    '''
    data.sort_values(by=['index'],inplace=True)
    data.set_index('index',inplace=True)
    data.drop(data[data['type'].isin(['Starting XI','50/50'])].index,inplace=True)
    data.reset_index(drop=True,inplace=True)
    home_team, away_team = 'Newcastle United', 'Southampton'
    data['home_team'] = home_team
    data['away_team'] = away_team
    data['home_score'] = 0
    data['away_score'] = 0
    # st.write(data['timestamp'])
    # st.write(data['timestamp'].astype('timedelta64[s]'))
    data['timedelta'] = data['timestamp'].astype('timedelta64[s]')
    data[data['period'] == 2]['timedelta'] = data[data['period'] == 2]['timedelta'] + pd.Timedelta(seconds=45)
    data['score_changed_duration'] = np.nan
    data.loc[0,'score_changed_duration'] = pd.to_timedelta(0)
    # st.write(data['timedelta'])

    for i in data[data['shot_outcome'] == 'Goal'].index:
        if data.loc[i]['team'] == home_team:
            data.loc[i:,'home_score'] = data.loc[i,'home_score'] + 1
        else:
            data.loc[i:,'away_score'] = data.loc[i,'away_score'] + 1
        # st.write(data.loc[i,'timedelta'])
        # st.write(data.loc[data.loc[:i,'score_changed_duration'].last_valid_index(),'score_changed_duration'])
        # st.write(type(data.loc[i,'timedelta']))
        # st.write(type(data.loc[data.loc[:i,'score_changed_duration'].last_valid_index(),'score_changed_duration']))
        data.loc[i,'score_changed_duration'] = pd.to_timedelta(data.loc[i,'timedelta'] - data.loc[data.loc[:i,'score_changed_duration'].last_valid_index(),'timedelta']).seconds
        # st.write(data.loc[i,'score_changed_duration']/60)
    # df.loc[len(df)-1,'score_changed_duration'] = pd.to_timedelta(len(df)-1 - df.loc[df['score_changed_duration'].last_valid_index(),'timedelta']).seconds
    data['score_margin'] = np.clip(data['home_score'] - data['away_score'],-2,2)
    data['home_team_status'] = np.clip(data['home_score'] - data['away_score'],-1,1)
    return data,home_team,away_team

    
def map_event_columns(df):
    '''
    
    '''
    events_columns_mapper = {
        'Pass':[i for i in df.columns if i[:5] == 'pass_'],
        'Shot':[i for i in df.columns if i[:5] == 'shot_'],
        'Ball Receipt*': [i for i in df.columns if i[:13] == 'ball_receipt_'],
        'Goal Keeper': [i for i in df.columns if i[:11] == 'goalkeeper_'],
        'Carry': [i for i in df.columns if i[:6] == 'carry_'],
        'Pressure': [i for i in df.columns if i[:9] == 'pressure_'],
        'Ball Recovery': [i for i in df.columns if i[:14] == 'ball_recovery_'],
        'Duel': [i for i in df.columns if i[:5] == 'duel_'],
        'Clearance': [i for i in df.columns if i[:10] == 'clearance_'],
        'Block': [i for i in df.columns if i[:6] == 'block_'],
        'Foul Committed': [i for i in df.columns if i[:5] == 'foul_'],
        'Dribble': [i for i in df.columns if i[:8] == 'dribble_']
    }
    return events_columns_mapper