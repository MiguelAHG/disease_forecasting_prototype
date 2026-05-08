import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.ops import unary_union
import plotly.express as px
import plotly.graph_objects as go

def format_label_int(x):
    try:
        return f"{int(x)}"
    except (ValueError, TypeError):
        return str(x)

def format_label_decimal(x):
    try:
        return f"{float(x):.2f}"
    except (ValueError, TypeError):
        return str(x)

@st.cache_data(ttl=None, persist="disk")
def load_data():
    lepto_h2 = pd.read_csv("data/lepto_rf_upsampled_aci_intervals_h2.csv").drop(columns="forecast_start")

    dengue_h1 = pd.read_csv("data/dengue_svm_nonupsampled_aci_intervals_h1.csv").drop(columns="forecast_start")

    dengue_h2 = pd.read_csv("data/dengue_xgb_nonupsampled_aci_intervals_h2.csv").drop(columns="forecast_start")

    dengue_lstm = pd.read_csv("data/aci_lstm_dengue_upsampled.csv")
    dengue_lstm = dengue_lstm.loc[dengue_lstm["horizon"].isin([4,8,12])]

    lepto_mhgcn = pd.read_csv("data/lepto_mhgcn_unreduced_nonupsampled_h1_and_h4.csv")
    lepto_mhgcn = lepto_mhgcn.loc[lepto_mhgcn["horizon"].isin([1,4])]

    lepto_h8 = pd.read_csv("data/lepto_mhgcn_unreduced_upsampled_h8.csv")
    lepto_h8 = lepto_h8.loc[lepto_h8["horizon"] == 8]

    lepto_intervals = pd.concat(
        objs = [lepto_h2, lepto_mhgcn, lepto_h8]
    ).sort_values(["horizon", "cluster_index", "interval_start_when_prediction_relevant"])

    dengue_intervals = pd.concat(
        objs = [dengue_h1, dengue_h2, dengue_lstm]
    ).sort_values(["horizon", "cluster_index", "interval_start_when_prediction_relevant"])

    # Get features and metadata
    lepto_features = pd.read_csv("data/fixed_lepto_all_merged_weekly_with_lagged_targets.csv")
    dengue_features = pd.read_csv("data/fixed_dengue_all_merged_weekly_with_lagged_targets.csv")

    with open("data/reduced_lepto_features_DETAILED.json", "r") as f:
        dict_lepto = json.load(f)

    with open("data/reduced_dengue_features_DETAILED.json", "r") as f:
        dict_dengue = json.load(f)

    # Filter to just the features used in models
    lepto_features = lepto_features[["interval_start", "cluster"] + list(dict_lepto.keys())]
    dengue_features = dengue_features[["interval_start", "cluster"] + list(dict_dengue.keys())]

    # Merge predictions and features
    lepto = pd.merge(
        lepto_intervals,
        lepto_features,
        how = "inner",
        left_on = ["interval_start_when_prediction_relevant", "cluster"],
        right_on = ["interval_start", "cluster"]
    ).drop(columns="interval_start_when_prediction_relevant")

    dengue = pd.merge(
        dengue_intervals,
        dengue_features,
        how = "inner",
        left_on = ["interval_start_when_prediction_relevant", "cluster"],
        right_on = ["interval_start", "cluster"]
    ).drop(columns="interval_start_when_prediction_relevant")

    # Update metadata on features
    extra_cols = {
        "true_count": {
                "display_title": "Actual number of cases",
                "data_type": "int",
                "temporal_type": "varying",
                "variable_type": "Epidemiological"
            },
        "prediction": {
                "display_title": "Predicted number of cases",
                "data_type": "dec",
                "temporal_type": "varying",
                "variable_type": "Prediction"
            },
        "lower": {
                "display_title": "Lower bound of predicted number of cases",
                "data_type": "dec",
                "temporal_type": "varying",
                "variable_type": "Prediction"
            },
        "upper": {
                "display_title": "Upper bound of predicted number of cases",
                "data_type": "dec",
                "temporal_type": "varying",
                "variable_type": "Prediction"
            },
    }
    dict_lepto.update(extra_cols)
    dict_dengue.update(extra_cols)

    # Set up geometries
    geo_brgy = (
        gpd.read_file("data/barangay_and_cluster_geo_with_districts.gpkg")
        .drop(columns="weight")
        .set_geometry("geometry")
        .to_crs(epsg=4326)
    )

    geo_brgy["lon"] = geo_brgy.geometry.centroid.x
    geo_brgy["lat"] = geo_brgy.geometry.centroid.y

    geo_cluster = (
        geo_brgy.groupby("cluster")
        .agg(
            geometry=("geometry", unary_union),
            area_km2=("area_km2", "sum"),
        )
        .reset_index(drop=False)
    ).set_geometry("geometry").set_crs(geo_brgy.crs)

    geo_cluster["lon"] = geo_cluster.geometry.centroid.x
    geo_cluster["lat"] = geo_cluster.geometry.centroid.y

    lepto_merged = geo_cluster.merge(lepto, on="cluster").set_crs(geo_brgy.crs)
    dengue_merged = geo_cluster.merge(dengue, on="cluster").set_crs(geo_brgy.crs)

    # precompute RMSEs
    # compute_cluster_rmse("Dengue", horizon=4, rainy_only=True)
    for disease in ["Leptospirosis", "Dengue"]:
        for horizon in [1,2,4,8,12]:
            if disease == "Leptospirosis" and horizon==12:
                continue

            compute_cluster_rmse(disease, horizon, rainy_only=True, _lepto_merged=lepto_merged, _dengue_merged=dengue_merged)

    return (
        lepto, dengue, geo_brgy, geo_cluster,
        lepto_merged, dengue_merged,
        dict_lepto, dict_dengue
    )

@st.cache_data(ttl=None, persist="disk", max_entries=365)
def build_choropleth(disease, horizon, selected_col, selected_interval, _merged, _gdf, _format_label):
    """Build a single-interval choropleth figure. The Streamlit slider in map.py
    controls which interval is passed in, so this is cached per (disease, horizon,
    selected_col, selected_interval) combination."""

    interval_df = _merged[_merged["interval_start"] == selected_interval].copy()

    vmin = _merged[selected_col].min()
    vmax = _merged[selected_col].max()

    choropleth = go.Choroplethmapbox(
        geojson=_gdf.set_index("cluster").__geo_interface__,
        locations=interval_df["cluster"],
        z=interval_df[selected_col],
        zmin=vmin,
        zmax=vmax,
        colorscale="Viridis",
        marker_opacity=0.6,
        marker_line_width=2,
        marker_line_color="white",
        showscale=True,
        customdata=interval_df[["true_count", "prediction", "lower", "upper"]].values,
        hovertemplate=(
            "<b>Cluster %{location}</b><br>"
            "Actual no. cases: %{customdata[0]:.0f}<br>"
            "Predicted no. cases: %{customdata[1]:.2f}<br>"
            "Lower bound: %{customdata[2]:.2f}<br>"
            "Upper bound: %{customdata[3]:.2f}"
            "<extra></extra>"
        ),
    )

    label = go.Scattermapbox(
        lat=interval_df["lat"],
        lon=interval_df["lon"],
        mode="markers+text",
        marker=dict(size=65, color="white", opacity=0.7),
        text=interval_df.apply(
            lambda row: f"Cluster {row['cluster']}<br>{_format_label(row[selected_col])}",
            axis=1
        ),
        textfont=dict(size=13, color="black"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[choropleth, label])

    fig.update_layout(
        height=500,
        mapbox=dict(
            style="open-street-map",
            center={"lat": 14.6760, "lon": 121.0437},
            zoom=11.3,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig

@st.cache_data(ttl=None, persist="disk")
def compute_cluster_rmse(disease, horizon, rainy_only, _lepto_merged, _dengue_merged):
    """Returns df_filtered with rmse column, and overall_rmse float."""

    df = _lepto_merged if disease == "Leptospirosis" else _dengue_merged
    df = df[df["horizon"] == horizon].copy()
    df["current_week"]   = pd.to_datetime(df["interval_start_at_inference_time"])
    df["predicted_week"] = pd.to_datetime(df["interval_start"])
    df["month"] = df["predicted_week"].dt.month

    if rainy_only:
        df = df[(df["month"] >= 6) & (df["month"] <= 11)]

    df["squared_error"] = (df["true_count"] - df["prediction"]) ** 2

    cluster_rmse = (
        df.groupby("cluster")["squared_error"]
        .mean()
        .apply(np.sqrt)
        .reset_index()
        .rename(columns={"squared_error": "rmse"})
    )

    df = df.merge(cluster_rmse, on="cluster")
    overall_rmse = np.sqrt(df["squared_error"].mean())

    # convert datetime cols to string for serialization
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    return df, overall_rmse

@st.cache_data(ttl=None, persist="disk")
def build_rmse_choropleth(disease, horizon, rainy_only, _geo_cluster):
    df_filtered, overall_rmse = compute_cluster_rmse(disease, horizon, rainy_only)

    cluster_rmse = df_filtered[["cluster", "rmse"]].drop_duplicates(subset=["cluster"])
    gdf = _geo_cluster.merge(cluster_rmse, on="cluster").to_crs("EPSG:4326")

    choropleth = go.Choroplethmapbox(
        geojson=gdf.set_index("cluster").__geo_interface__,
        locations=gdf["cluster"],
        z=gdf["rmse"],
        colorscale="Reds",
        zmin=gdf["rmse"].min(),
        zmax=gdf["rmse"].max(),
        marker_opacity=0.6,
        marker_line_width=2,
        marker_line_color="white",
        showscale=True,
        hovertemplate="<b>Cluster %{location}</b><br>RMSE: %{z:.4f}<extra></extra>",
    )

    label = go.Scattermapbox(
        lat=gdf.geometry.centroid.y,
        lon=gdf.geometry.centroid.x,
        mode="markers+text",
        marker=dict(size=60, color="white", opacity=0.7),
        text=gdf.apply(lambda row: f"Cluster {row['cluster']}<br>{row['rmse']:.4f}", axis=1),
        textfont=dict(size=13, color="black"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[choropleth, label])
    fig.update_layout(
        height=600,
        mapbox=dict(
            style="open-street-map",
            center={"lat": 14.6760, "lon": 121.0437},
            zoom=11.3,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig, overall_rmse