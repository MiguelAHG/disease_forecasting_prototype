import streamlit as st
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from shapely.ops import unary_union
import plotly.express as px
import plotly.graph_objects as go

from custom_utils import load_data, build_choropleth, format_label_int, format_label_decimal, compute_cluster_rmse, build_rmse_choropleth

st.title("Barangay, Cluster, and District Map")

# Set up data
(
    lepto, dengue, geo_brgy, geo_cluster,
    lepto_merged, dengue_merged,
    dict_lepto, dict_dengue
) = load_data()

disease = st.session_state["disease"]
horizon = st.session_state["horizon"]

# Data
df_raw = lepto_merged if disease == "Leptospirosis" else dengue_merged
st.markdown(f"Showing results for __{disease}__.")

df = df_raw[df_raw["horizon"] == horizon].copy()
df["current_week"]   = pd.to_datetime(df["interval_start_at_inference_time"])
df["predicted_week"] = pd.to_datetime(df["interval_start"])
df["year"]  = df["current_week"].dt.year
df["month"] = df["current_week"].dt.month
df["prediction_year"]  = df["predicted_week"].dt.year
df["prediction_month"] = df["predicted_week"].dt.month

# Map of barangays, their clusters, and their districts

@st.fragment
def render_brgy_map(geo_brgy):

    fig = go.Figure()

    # Make colorscale
    import plotly.colors as pc
    clusters = sorted(geo_brgy["cluster"].unique())
    n = len(clusters)
    # pick n colors evenly spaced from a qualitative palette
    palette = pc.qualitative.Safe
    colors = [palette[i % len(palette)] for i in range(n)]
    # map cluster letter to an integer for z
    cluster_to_int = {c: i for i, c in enumerate(clusters)}
    geo_brgy["cluster_int"] = geo_brgy["cluster"].map(cluster_to_int)
    # build a discrete colorscale from the palette
    n_colors = len(clusters)
    colorscale = []
    for i, color in enumerate(colors):
        colorscale.append([i / n_colors, color])
        colorscale.append([(i + 1) / n_colors, color])

    # layer 1: barangays colored by cluster
    clusters = sorted(geo_brgy["cluster"].unique())
    for cluster in clusters:
        subset = geo_brgy[geo_brgy["cluster"] == cluster]
        fig.add_trace(go.Choroplethmapbox(
            geojson=subset.set_index("barangay").__geo_interface__,
            locations=subset["barangay"],
            z=[1] * len(subset),          # dummy z, just for coloring
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # transparent fill
            showscale=False,
            marker_line_width=0.5,
            marker_line_color="gray",
            name=f"Cluster {cluster}",
            legendgroup=f"Cluster {cluster}",
            hovertemplate="<b>%{location}</b><br>Cluster " + str(cluster) + "<extra></extra>",
            # showlegend=True,
        ))

    # color each cluster
    fig.add_trace(go.Choroplethmapbox(
        geojson=geo_brgy.set_index("barangay").__geo_interface__,
        locations=geo_brgy["barangay"],
        z=geo_brgy["cluster"].apply(ord) - ord("A"),
        colorscale=colorscale,
        showscale=False,
        marker_line_width=0.5,
        marker_line_color="white",
        marker_opacity=0.7,
        name="Barangays",
        customdata=geo_brgy[["cluster", "district"]].values,
        hovertemplate="<b>%{location}</b><br>Cluster %{customdata[0]}<br>District %{customdata[1]}<extra></extra>",
        # showlegend=True,
    ))

    # after the main choropleth trace, add dummy legend entries per cluster
    for i, cluster in enumerate(clusters):
        fig.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode="markers",
            marker=dict(size=12, color=colors[i]),
            name=f"Cluster {cluster}",
            legendgroup="Clusters",
            legendgrouptitle_text="Clusters" if i == 0 else "",
            # showlegend=True,
            hoverinfo="skip",
        ))

    # layer 2: district boundaries (toggleable via legend)
    # Dissolve geo_brgy by district.
    geo_districts = (
        geo_brgy.groupby("district")
        .agg(geometry=("geometry", unary_union))
        .reset_index(drop=False)
    ).set_geometry("geometry").set_crs(geo_brgy.crs)

    district_colors = {
        1: "red",
        2: "blue",
        3: "green",
        4: "orange",
        5: "purple",
        6: "brown",
    }

    for _, row in geo_districts.iterrows():
        color = district_colors[row["district"]]
        
        # boundary outline
        boundary = row["geometry"].boundary
        if boundary.geom_type == "LineString":
            coords = list(boundary.coords)
            lons, lats = zip(*[(c[0], c[1]) for c in coords])
        else:
            lons, lats = [], []
            for line in boundary.geoms:
                for c in line.coords:
                    lons.append(c[0])
                    lats.append(c[1])
                lons.append(None)
                lats.append(None)

        fig.add_trace(go.Scattermapbox(
            lon=list(lons), lat=list(lats),
            mode="lines",
            line=dict(color=color, width=7),
            name=f"District {row['district']}",
            legendgroup="Districts",
            legendgrouptitle_text="Districts" if row["district"] == geo_districts["district"].iloc[0] else "",
            visible="legendonly",
            hoverinfo="skip",
            showlegend=True,
        ))

        # label at centroid
        centroid = row["geometry"].centroid
        fig.add_trace(go.Scattermapbox(
            lon=[centroid.x], lat=[centroid.y],
            mode="markers+text",
            marker=dict(size=30, color="white", opacity=0.8),
            text=[f"D{row['district']}"],
            textfont=dict(size=14, color=color),
            name=f"District {row['district']} label",
            legendgroup="Districts",
            visible="legendonly",
            hoverinfo="skip",
            showlegend=False,
        ))
    # end loop

    fig.update_layout(
        hoverlabel=dict(font_size=16),
        showlegend=True,
        mapbox=dict(
            style="open-street-map",
            center={"lat": 14.6760, "lon": 121.0437},
            zoom=11,
        ),
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            title="Layers",
            bgcolor="rgba(255,255,255,0.8)",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown(
"""##### Hover the cursor over a barangay to know its Cluster and District.
    
##### In the legend on the right, click 'Districts' to show QC legislative districts on the map."""
)

render_brgy_map(geo_brgy)


# Cluster-specific information

with st.expander("More information on clusters", expanded=True):
    selected_cluster = st.radio(
        "Select a cluster to see its barangays.",
        geo_cluster["cluster"].tolist(),
        horizontal=True,
        format_func = lambda cluster: f"Cluster {cluster}"
    )
    st.header(f"Cluster {selected_cluster}")

    filtered = geo_brgy.loc[geo_brgy["cluster"]==selected_cluster]
    brgy_list_markdown = "\n".join(f"- {b}" for b in filtered["barangay"].tolist())
    n_brgys = filtered.shape[0]

    st.markdown(f"This cluster includes the following {n_brgys} barangay{'s' if n_brgys >= 2 else ''}:")

    # with st.expander("View list of barangays"):
    st.markdown(brgy_list_markdown)

# RMSE of clusters

with st.container(border=True):
    st.header("Model performance on different clusters")

    to_filter = st.radio(
        "Select the type of metric.",
        ["Filtered (Rainy season)", "Unfiltered (Year-round)"]
    )

    rainy_only = (to_filter == "Filtered (Rainy season)")
    # df_filtered, overall_rmse = compute_cluster_rmse(disease, horizon, rainy_only)

    # with st.container(border=True):
    #     st.metric(label="Overall Root Mean Squared Error (RMSE) for 2023-2025", value=f"{overall_rmse:.4f} cases")

    # # 2. Convert Timestamp/Datetime columns to string
    # for col in df_filtered.columns:
    #     if pd.api.types.is_datetime64_any_dtype(df_filtered[col]):
    #         df_filtered[col] = df_filtered[col].astype(str)

    # # 3. Ensure GeoDataFrame is in a format Plotly can process
    # if df_filtered.crs is None or df_filtered.crs != "EPSG:4326":
    #     df_filtered = df_filtered.to_crs("EPSG:4326")

    fig, overall_rmse = build_rmse_choropleth(disease, horizon, rainy_only, _geo_cluster=geo_cluster)
    st.plotly_chart(fig)
    
    # geojson_data = json.loads(df_filtered.to_json())

    # # Create the choropleth map colored by the 'rmse' column
    # fig = px.choropleth_mapbox(
    #     df_filtered,
    #     geojson=geojson_data,
    #     locations=df_filtered.index,
    #     color="rmse",  # <--- Colored by cluster RMSE
    #     color_continuous_scale="Reds", # Reds or Viridis works well for errors
    #     mapbox_style="carto-positron",
    #     zoom=11.3,
    #     center={"lat": df_filtered.geometry.centroid.y.mean(), "lon": df_filtered.geometry.centroid.x.mean()},
    #     opacity=0.5,
    #     title="Cluster Performance (RMSE) Map",
    #     labels={"rmse": "RMSE"},
    #     range_color=[df_filtered['rmse'].min(), df_filtered['rmse'].max()]
    # )

    # # Add labels at the centroid of each cluster with a white circle background
    # cluster_unique = df_filtered[['cluster', 'lat', 'lon', 'rmse']].drop_duplicates(subset=['cluster'])
    
    # fig.add_trace(go.Scattermapbox(
    #     lat=cluster_unique["lat"],
    #     lon=cluster_unique["lon"],
    #     mode="markers+text",
    #     marker=dict(
    #         size=60,
    #         color="white",
    #         opacity=0.7,
    #     ),
    #     text=cluster_unique.apply(
    #         lambda row: f"Cluster {row['cluster']}<br>{row['rmse']:.4f}",
    #         axis=1
    #     ),
    #     textfont=dict(size=13, color="black"),
    #     hoverinfo="skip",
    #     showlegend=False,
    # ))

    # fig.update_layout(height=700, margin={"r":0,"t":0,"l":0,"b":0})
    # st.plotly_chart(fig)