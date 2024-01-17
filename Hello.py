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
from streamlit.logger import get_logger

import math
import json
import re
import ast

import numpy as np
import pandas as pd

import soccerdata as sd

import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib.legend_handler import HandlerTuple
# import matplotlib.patheffects as path_effects
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import plotly.express as px
import plotly.graph_objects as go

from mplsoccer import Pitch, VerticalPitch
# from mplsoccer import Pitch, FontManager, Sbopen, VerticalPitch

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="My first football app",
        page_icon="⚽",
    )
    st.title('Football Analytics Dashboard')
    st.header(f"Streamlit version: {st.__version__}")
    # st.subheader('Example Data: Statsbomb EPL Match Events')
    st.markdown('_Example Data: Statsbomb EPL Match Events. Magpies vs Saints 2015/16_') # see #*
    # st.caption('Example Data: Statsbomb EPL Match Events')

    # ws = sd.WhoScored(leagues="ENG-Premier League", seasons=2021)
    # epl_schedule = ws.read_schedule()
    
    df = pd.read_csv('match_events.csv')
    # st.dataframe(df.head())
    df.sort_values(by=['period','timestamp'],inplace=True)
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
    # df.style.hide(axis="index")


    with st.sidebar:
        col1, col2 = st.columns([4,3])

        with col1:
            st.subheader("Event Types")
            st.dataframe(df['type'].value_counts(sort=True,ascending=False))
            

        with col2:
            st.subheader("Table Attributes")
            st.dataframe(df.columns,hide_index=True)

    # Create a pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass')

    # Plot passes entering final third and penalty area for home team
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pitch.draw(ax=axes[0])
    pitch.arrows(
        df[df['type']== 'Shot']['location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['type']== 'Shot']['location'].apply(lambda x: ast.literal_eval(x)[1]),
        df[df['type']== 'Shot']['shot_end_location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['type']== 'Shot']['shot_end_location'].apply(lambda x: ast.literal_eval(x)[1]),
        color='red',
        ax=axes[0]
    )
    pitch.arrows(
        df[df['shot_outcome']== 'Goal']['location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['shot_outcome']== 'Goal']['location'].apply(lambda x: ast.literal_eval(x)[1]),
        df[df['shot_outcome']== 'Goal']['shot_end_location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['shot_outcome']== 'Goal']['shot_end_location'].apply(lambda x: ast.literal_eval(x)[1]),
        color='green',
        ax=axes[0]
    )

    pitch.draw(ax=axes[1])
    pitch.arrows(
        df[df['type']== 'Pass']['location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['type']== 'Pass']['location'].apply(lambda x: ast.literal_eval(x)[1]),
        df[df['type']== 'Pass']['pass_end_location'].apply(lambda x: ast.literal_eval(x)[0]),
        df[df['type']== 'Pass']['pass_end_location'].apply(lambda x: ast.literal_eval(x)[1]),
        color='red',
        ax=axes[1]
    )
    # pitch.arrows(
    #     df[df['pass_outcome'].isnull()]['location'].apply(lambda x: ast.literal_eval(x)[0]),
    #     df[df['pass_outcome'].isnull()]['location'].apply(lambda x: ast.literal_eval(x)[1]),
    #     df[df['pass_outcome'].isnull()]['pass_end_location'].apply(lambda x: ast.literal_eval(x)[0]),
    #     df[df['pass_outcome'].isnull()]['pass_end_location'].apply(lambda x: ast.literal_eval(x)[1]),
    #     color='green',
    #     ax=axes[1]
    # )
    st.pyplot(fig)

    # pitch.arrows(passes_home_penalty_area.start_x, passes_home_penalty_area.start_y,
    #             passes_home_penalty_area.end_x, passes_home_penalty_area.end_y,
    #             color='red', ax=axes[1])
    
    # ‘statsbomb’, ‘opta’, ‘tracab’, ‘wyscout’, ‘uefa’, ‘metricasports’, ‘custom’, ‘skillcorner’, ‘secondspectrum’ and ‘impect’


    # v_pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='grass')
    # fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # st.pyplot(v_pitch.draw())

    events_filter = st.multiselect(options=df['type'].unique(),label='Event Type',default=df['type'].unique())
    # st.write(pd.DataFrame([pd.Series(df.columns).iloc[i:i+len(df.columns)//3] for i in range(0,len(df.columns),len(df.columns)//3)]))
    
    
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
    if len(events_filter) == 1:
        if events_filter[0] in events_columns_mapper.keys():
            general_info_items.extend(events_columns_mapper[events_filter[0]])

    st.container(height=40,border=False)
    
    st.data_editor(df[df['type'].isin(list(events_filter))][general_info_items].head(400),use_container_width=True,height=800,hide_index=True)
    # st.table(df.iloc[0:10])

    st.json({'foo':'bar','fu':'ba'})
    st.metric(label="Temp", value="273 K", delta="1.2 K")

    
    
    st.text('Fixed width text')
    st.code('for i in range(8): foo()')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

if __name__ == "__main__":
    run()
