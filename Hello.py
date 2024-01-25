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
from Helper_Functions import get_data, transform_sort_data, map_event_columns

def run():
    st.set_page_config(
        page_title="My first football app",
        page_icon="⚽",
    )
    st.subheader('Match Report')

    # ws = sd.WhoScored(leagues="ENG-Premier League", seasons=2021)
    # epl_schedule = ws.read_schedule()
    
    df = get_data()
    df, home_team, away_team = transform_sort_data(df)

    with st.sidebar:
        
        players_filter = st.multiselect(options=df['player'].dropna().unique(),label='Players',default=np.array(df['player'].dropna().value_counts(sort=True,ascending=False).index[:2]))
        events_filter = st.multiselect(options=df['type'].value_counts(sort=True,ascending=False).index,label='Event Type',default=df['type'].value_counts(sort=True,ascending=False).index[0])

        col1, col2 = st.columns([4,3])

        with col1:
            st.subheader("Event Types")
            st.dataframe(df['type'].value_counts(sort=True,ascending=False),height=600)
            

        with col2:
            st.subheader("Table Attributes")
            st.dataframe(df.columns,hide_index=True,height=600)
    
    events_columns_mapper = map_event_columns(df)

    # general_info_items = ['timestamp','period','minute','second','team','player','play_pattern','type','possession','possession_team','score_margin','score_changed_duration','home_team','away_team','home_score','away_score']
    general_info_items = ['timestamp','period','team','player','play_pattern','type','possession','possession_team','score_margin','score_changed_duration']

    for i in events_filter:
        if i in events_columns_mapper.keys():
            general_info_items.extend(events_columns_mapper[i])

    # Create a pitch
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='white', positional=True)
    # pitch_type is one of the following: ‘statsbomb’, ‘opta’, ‘tracab’, ‘wyscout’, ‘uefa’, ‘metricasports’, ‘custom’, ‘skillcorner’, ‘secondspectrum’ and ‘impect’

    fig, axes = plt.subplots(2,df['home_team_status'].nunique(), figsize=(12, 10))

    colors = ['red','green','blue','orange','purple','black','gray']
    for i in range(20):
        colors.extend(colors)
    print(len(colors))
    label_colors = {}

    # For every team
    for team_index in range(2):
        # print(f"inside team, team index = {team_index}")
        # for every match status change
        for match_status_index, match_status in enumerate(df['home_team_status'].unique()):
            # print(f"inside match status, match_status_index = {match_status_index}")
            condition_match_status = df['home_team_status'] == match_status
            pitch.draw(ax=axes[team_index,match_status_index])
            if team_index == 0:
                condition_team = df['team'] == home_team
                axes[team_index,0].set_ylabel(home_team)
            else:
                condition_team = df['team'] == away_team
                axes[1,0].set_ylabel(away_team)
            if match_status == 0:
                axes[0,match_status_index].set_title('Draw')
            elif match_status == -1:
                axes[0,match_status_index].set_title(f'{away_team} lead')
            elif match_status == 1:
                axes[0,match_status_index].set_title(f'{home_team} lead')
            # for every player
            for player_index,player in enumerate(players_filter):
                # print(f"inside player loop, player = {player}, player_index={player_index}")
                condition_player = df['player']== player
                # for every event
                for event_index,event in enumerate(events_filter):
                    # print(f"inside event loop, event = {event}, event_index={event_index}")
                    condition_event = df['type'] == event
                    event_end_location = f'{event.lower()}_end_location'
                    event_outcome = f'{event.lower()}_outcome' if f'{event.lower()}_outcome' in df.columns else 'type'
                    outcome_loop = df[condition_event][event_outcome].unique()
                    # for every event outcome
                    for outcome_index,outcome in enumerate(outcome_loop):
                        # print(f"inside outcome loop, outcome = {outcome}, outcome_index={outcome_index}")
                        condition_outcome = df[event_outcome] == outcome if pd.notna(outcome) else df[event_outcome].isna()
                        # condition_outcome = df[event_outcome] == outcome
                        final_condition = condition_event&condition_team&condition_outcome&condition_match_status&condition_player
                        start_x = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[0])
                        end_x = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[0]) if event_end_location in df.columns else start_x
                        start_y = df[final_condition]['location'].apply(lambda x: ast.literal_eval(x)[1])
                        end_y = df[final_condition][event_end_location].apply(lambda x: ast.literal_eval(x)[1]) if event_end_location in df.columns else start_y
                        # print(f'event_outcome count: {len(start_x)}')
                        label_name = f"{event}: {outcome}"
                        # color_index = outcome_index + event_index*10
                        alpha = 1-outcome_index/len(outcome_loop)
                        color = colors[event_index]

                        
                        if min(len(start_x),len(start_y)) >0:
                            pitch.arrows(
                                start_x,
                                start_y,
                                end_x,
                                end_y,
                                color=color,
                                ax=axes[team_index,match_status_index],
                                label=label_name,
                                alpha=alpha
                            )
                            label_colors[label_name] = color,alpha

    legend_labels = [plt.Line2D([], [], color=v[0], label=k,alpha=v[1]) for k,v in label_colors.items()]
    if len(legend_labels) > 0:
        fig.legend(handles=legend_labels,loc='lower center',ncol=min(len(legend_labels),5))

    st.pyplot(fig)

    # st.container(height=20,border=False)

    new_df = pd.DataFrame(columns=[
        'Yellow Cards',
        'Red Cards',

        'Goals',
        'Assists',
        'xG',
        'Shots',
        'Crosses',
        # Shots inside the box
        # Big Chances
        # Big Chance Conversion Rate
        'xG - Goals',
        # goals + assists
        'Carries',
        'Passes',
        # Pass %
        # Shot %
        'Dribbles',
        # Dribble %
        # Crosses



        # Goals Conceded
        # xG Against
        # Shots Conceded
        # Shots Conceded inside the box
        # Big Chances Allowed
        # Clean Sheets
        # Saves
        # Interceptions
        # Duels
        # Duel %
        # Aerial Duels
        # Aerial Duel %
        # Blocks
        # Recoveries
    ])
    for i in players_filter:
        condition_player = df['player'] == i
        events = {
        'Yellow Cards':len(df[condition_player&(df['foul_committed_card']== 'Yellow Card')]),
        'Red Cards': len(df[condition_player&(df['foul_committed_card']== 'Red Card')]),

        'Goals': len(df[condition_player&(df['shot_outcome']== 'Goal')]),
        'Assists': 0,
        'xG': df[condition_player]['shot_statsbomb_xg'].sum(),
        'Shots': len(df[condition_player&(df['type']== 'Shot')]),
        'Crosses': len(df[condition_player&(df['pass_cross']== True)]),
        # Shots inside the box
        # Big Chances
        # Big Chance Conversion Rate
        'xG - Goals': df[condition_player]['shot_statsbomb_xg'].sum() - len(df[condition_player&(df['shot_outcome']== 'Goal')]),
        # goals + assists
        'Carries': len(df[condition_player&(df['type']== 'Carry')]),
        'Passes': len(df[condition_player&(df['type']== 'Pass')]),
        # Pass %
        # Shot %
        'Dribbles':  len(df[condition_player&(df['type']== 'Dribble')]),
        # Dribble %
        # Crosses



        # Goals Conceded
        # xG Against
        # Shots Conceded
        # Shots Conceded inside the box
        # Big Chances Allowed
        # Clean Sheets
        # Saves
        # Interceptions
        # Duels
        # Duel %
        # Aerial Duels
        # Aerial Duel %
        # Blocks
        # Recoveries
        }
        new_df.loc[i] = events

    st.data_editor(new_df)
    condition_player = df['player'].isin(players_filter)
    st.data_editor(df[df['type'].isin(events_filter)&condition_player][general_info_items].set_index('timestamp').head(400),use_container_width=True,height=800)

    st.json({'foo':'bar','fu':'ba'})
    st.metric(label="Temp", value="273 K", delta="1.2 K")

    
    
    st.text('Fixed width text')
    st.code('for i in range(8): foo()')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    st.markdown(f"Streamlit version: {st.__version__}")
    st.markdown('_Example Data: Statsbomb EPL Match Events. Magpies vs Saints 2015/16_') # see #*

if __name__ == "__main__":
    run()
