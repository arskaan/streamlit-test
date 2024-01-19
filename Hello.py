# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

import math
import json
import re
import ast

import numpy as np
import pandas as pd

import soccerdata as sd

import matplotlib.pyplot as plt
# from mpltools import style

import plotly.express as px
import plotly.graph_objects as go

from mplsoccer import Pitch, VerticalPitch

# style.use('ggplot')


def run():
    st.set_page_config(
        page_title="My first football app",
        page_icon="⚽",
    )
    st.subheader('Match Report')

    # ws = sd.WhoScored(leagues="ENG-Premier League", seasons=2021)
    # epl_schedule = ws.read_schedule()
    
    df = pd.read_csv('match_events.csv')
    # st.dataframe(df.head())
    df.sort_values(by=['period','timestamp'],inplace=True)
    df.drop(df[df['type'].isin(['Starting XI','50/50'])].index,inplace=True)
    df.set_index('index',inplace=True)
    home_team, away_team = 'Newcastle United', 'Southampton'
    df['home_team'] = home_team
    df['away_team'] = away_team
    df['home_score'] = 0
    df['away_score'] = 0
    for i in df[df['shot_outcome'] == 'Goal'].index:
        if df.loc[i]['team'] == home_team:
            df.loc[i:,'home_score'] = df.loc[i,'home_score'] + 1
        else:
            df.loc[i:,'away_score'] = df.loc[i,'away_score'] + 1
    df['score_margin'] = np.clip(df['home_score'] - df['away_score'],-2,2)
    df['home_team_status'] = np.clip(df['home_score'] - df['away_score'],-1,1)

    with st.sidebar:
        
        events_filter = st.multiselect(options=df['type'].unique(),label='Event Type',default='Shot')

        col1, col2 = st.columns([4,3])

        with col1:
            st.subheader("Event Types")
            st.dataframe(df['type'].value_counts(sort=True,ascending=False),height=600)
            

        with col2:
            st.subheader("Table Attributes")
            st.dataframe(df.columns,hide_index=True,height=600)
    
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

    # general_info_items = ['timestamp','period','minute','second','team','player','play_pattern','type','possession','possession_team',,'home_team','away_team','home_score','away_score']
    general_info_items = ['timestamp','period','team','player','play_pattern','type','possession','possession_team','score_margin']

    current_event = 'Shot'
    if len(events_filter) == 1:
        current_event = events_filter[0]
        if events_filter[0] in events_columns_mapper.keys():
            general_info_items.extend(events_columns_mapper[events_filter[0]])

    # Create a pitch
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='white', positional=True)
    # pitch_type is one of the following: ‘statsbomb’, ‘opta’, ‘tracab’, ‘wyscout’, ‘uefa’, ‘metricasports’, ‘custom’, ‘skillcorner’, ‘secondspectrum’ and ‘impect’

    fig, axes = plt.subplots(2,df['home_team_status'].nunique(), figsize=(12, 10))
    condition_event = df['type']== current_event
    event_end_location = f'{current_event.lower()}_end_location'
    event_outcome = f'{current_event.lower()}_outcome'
    label_colors = {}

    for i in range(2):
        for x,y in enumerate(df['home_team_status'].unique()):
            condition_match_status = df['home_team_status'] == y
            pitch.draw(ax=axes[i,x])
            if i == 0:
                condition_team = df['team'] == home_team
                axes[i,0].set_ylabel(home_team)
                # axes[i,0].set_title(home_team)
            else:
                condition_team = df['team'] == away_team
                axes[1,0].set_ylabel(away_team)
                # axes[i,0].set_title(away_team)
            if y == 0:
                axes[0,x].set_title('Draw')
            elif y == -1:
                axes[0,x].set_title(f'{away_team} lead')
            elif y == 1:
                axes[0,x].set_title(f'{home_team} lead')

            colors = ['red','green','blue','yellow','orange','purple','black','gray']
            if event_outcome in df.columns:
                for l,k in enumerate(df[condition_event][event_outcome].unique()):
                    condition_outcome = df[event_outcome] == k if pd.notna(k) else df[event_outcome].isna()
                    final_condition = condition_event&condition_team&condition_outcome&condition_match_status
                    start_x = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[0])
                    end_x = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[0]) if event_end_location in df.columns else start_x
                    start_y = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[1])
                    end_y = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[1]) if event_end_location in df.columns else start_y
                    
                    if min(len(start_x),len(start_y)) >0:
                        pitch.arrows(
                            start_x,
                            start_y,
                            end_x,
                            end_y,
                            # lw=df[condition_event&condition_team]['shot_statsbomb_xg'],
                            color=colors[l],
                            ax=axes[i,x],
                            label=k
                        )
                        label_colors[k] = colors[l]
                    
            else:
                final_condition = condition_event&condition_team&condition_match_status
                start_x = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[0])
                end_x = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[0]) if event_end_location in df.columns else start_x
                start_y = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[1])
                end_y = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[1]) if event_end_location in df.columns else start_y
                
                if min(len(start_x),len(start_y)) >0:
                    pitch.arrows(
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        # lw=df[condition_event&condition_team]['shot_statsbomb_xg'],
                        color='red',
                        ax=axes[i,x]
                    )
                    label_colors[current_event] = 'red'

    # axes[0,0].set_ylabel('Newcastle United', rotation=0, labelpad=100)
    legend_labels = [plt.Line2D([], [], color=v, label=k) for k,v in label_colors.items()]
    fig.legend(handles=legend_labels,loc='lower center',ncol=len(legend_labels))

    st.pyplot(fig)

    # st.container(height=20,border=False)
    
    st.data_editor(df[df['type'].isin(list(events_filter))][general_info_items].set_index('timestamp').head(400),use_container_width=True,height=800)
    # st.table(df.iloc[0:10])

    st.json({'foo':'bar','fu':'ba'})
    st.metric(label="Temp", value="273 K", delta="1.2 K")

    
    
    st.text('Fixed width text')
    st.code('for i in range(8): foo()')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    st.markdown(f"Streamlit version: {st.__version__}")
    st.markdown('_Example Data: Statsbomb EPL Match Events. Magpies vs Saints 2015/16_') # see #*

if __name__ == "__main__":
    run()
