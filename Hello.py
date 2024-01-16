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

import numpy as np
import pandas as pd

import soccerdata as sd

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import plotly.express as px
import plotly.graph_objects as go

from mplsoccer import Pitch
from mplsoccer import Pitch, FontManager, Sbopen, VerticalPitch
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="My first football app",
        page_icon="⚽",
    )
    st.title('Football Analytics Dashboard')

    # ws = sd.WhoScored(leagues="ENG-Premier League", seasons=2021)
    # epl_schedule = ws.read_schedule()
    
    # st.dataframe(epl_schedule.head())
    # st.table(epl_schedule.iloc[0:10])
    st.json({'foo':'bar','fu':'ba'})
    st.metric(label="Temp", value="273 K", delta="1.2 K")

    st.header('My header')
    st.subheader('My sub')
    st.text('Fixed width text')
    st.code('for i in range(8): foo()')
    st.markdown('_Markdown_') # see #*
    st.caption('Balloons. Hundreds of them...')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    st.sidebar.success("Select a demo above.")

if __name__ == "__main__":
    run()
