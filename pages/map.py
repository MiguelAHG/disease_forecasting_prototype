import streamlit as st
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from shapely.ops import unary_union
import plotly.graph_objects as go
import time

from custom_utils import load_data, build_choropleth, format_label_int, format_label_decimal

st.title("Data Map")

# Set up data

(
    lepto, dengue, geo_brgy, geo_cluster,
    lepto_merged, dengue_merged,
    dict_lepto, dict_dengue
) = load_data()

disease = st.session_state["disease"]
horizon = st.session_state["horizon"]

st.markdown(f"Showing results for __{disease}__.")

st.divider()

gdf = geo_cluster
merged = lepto_merged if disease == "Leptospirosis" else dengue_merged
merged = merged.loc[merged["horizon"] == horizon]
dict_features = dict_lepto if disease == "Leptospirosis" else dict_dengue

# Column selector

first_cols = ["prediction", "lower", "upper", "true_count"]
exclude_cols = ["area_km2", "cluster", "interval_start", "geometry", "lon", "lat",
                "cluster_index", "horizon", "interval_start_at_inference_time"]
available_cols = first_cols + [c for c in merged.columns if c not in (first_cols + exclude_cols)]
assert all([c in dict_features for c in available_cols])

# selected_col = st.selectbox(
#     "Color map by:",
#     available_cols,
#     format_func=lambda col: dict_features[col]["display_title"]
# )

# format_label = format_label_int if dict_features[selected_col]["data_type"] == "int" else format_label_decimal

# Interval slider

all_intervals = sorted(merged["interval_start"].unique())

default_interval = "2025-07-04"
if default_interval not in all_intervals:
    default_interval = all_intervals[-1]

# Map

@st.fragment
def render_map(disease, horizon,
            #    selected_col,
               merged, gdf,
            #    format_label,
               all_intervals, default_interval):

    if "map_interval_idx" not in st.session_state:
        st.session_state["map_interval_idx"] = all_intervals.index(default_interval)
    if "map_playing" not in st.session_state:
        st.session_state["map_playing"] = False

    selected_interval = st.select_slider(
        "Select week:",
        options=all_intervals,
        value=all_intervals[st.session_state["map_interval_idx"]],
    )
    # sync slider position back to index
    st.session_state["map_interval_idx"] = all_intervals.index(selected_interval)

    col1, col2, col3, col4 = st.columns([1, 1, 2, 6])
    with col1:
        if st.button("▶ Play"):
            st.session_state["map_playing"] = True
    with col2:
        if st.button("⏸ Pause"):
            st.session_state["map_playing"] = False
    with col3:
        delay = st.number_input("Seconds per frame", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with col4:
        selected_col = st.selectbox(
            "Color map by:",
            available_cols,
            format_func=lambda col: dict_features[col]["display_title"]
        )

        format_label = format_label_int if dict_features[selected_col]["data_type"] == "int" else format_label_decimal


    with st.spinner("Loading map..."):
        fig = build_choropleth(
            disease, horizon, selected_col, selected_interval,
            _merged=merged, _gdf=gdf, _format_label=format_label
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state["map_playing"]:
        current_idx = st.session_state["map_interval_idx"]
        if current_idx < len(all_intervals) - 1:
            time.sleep(delay)
            st.session_state["map_interval_idx"] = current_idx + 1
            st.rerun(scope="fragment")
        else:
            st.session_state["map_playing"] = False

render_map(disease, horizon,
        #    selected_col,
           merged, gdf,
        #    format_label,
           all_intervals, default_interval)

