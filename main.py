import streamlit as st
import pandas as pd
import numpy as np

from custom_utils import load_data

st.set_page_config(
    page_title="QC Leptospirosis and Disease Forecasting",
    page_icon="🔍",
    layout="wide",                  # "centered" | "wide"
    initial_sidebar_state="expanded",  # "auto" | "expanded" | "collapsed"
)

# session state defaults
defaults = {
    "user": None,
    "theme": "light",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# navigation
cluster_info  = st.Page("pages/cluster_info.py",     title="Barangay Cluster Guide",     icon="ℹ️")
map     = st.Page("pages/map.py",     title="Data Map",     icon="🗺️")
dashboard     = st.Page("pages/dashboard.py",     title="Dashboard",     icon="📊")

pg = st.navigation(
    # {
    #     "Home":     [home],
    #     "Map":     [map],
    #     "Dashboard":     [dashboard],
    #     "About":     [about],
    # },
    [dashboard, map, cluster_info],
    position="sidebar",   # "sidebar" | "hidden"
    expanded=True,
)

# shared sidebar widgets (appear on every page)
with st.sidebar:
    # Select disease
    disease = st.radio("Disease", ["Dengue", "Leptospirosis"])
    st.session_state["disease"] = disease

    # Select horizon
    horizons = [1, 2, 4, 8] if disease == "Leptospirosis" else [1, 2, 4, 8, 12]
    horizon = st.radio(
        "Forecast horizon (weeks ahead):",
        horizons,
        index=2,
        horizontal=False,
        key="dashboard_horizon"
    )
    st.session_state["horizon"] = horizon

# load data
(
    lepto, dengue, geo_brgy, geo_cluster,
    lepto_merged, dengue_merged,
    dict_lepto, dict_dengue
) = load_data()

# run the selected page
pg.run()