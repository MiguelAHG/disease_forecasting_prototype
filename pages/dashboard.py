import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from custom_utils import load_data

st.title("Dashboard")

# Set up data

(
    lepto, dengue, geo_brgy, geo_cluster,
    lepto_merged, dengue_merged,
    dict_lepto, dict_dengue
) = load_data()

disease = st.session_state["disease"]
horizon = st.session_state["horizon"]

df_raw = lepto if disease == "Leptospirosis" else dengue
st.markdown(f"Showing results for __{disease}__.")

df = df_raw[df_raw["horizon"] == horizon].copy()
df["current_week"]   = pd.to_datetime(df["interval_start_at_inference_time"])
df["predicted_week"] = pd.to_datetime(df["interval_start"])
df["year"]  = df["current_week"].dt.year
df["month"] = df["current_week"].dt.month
df["prediction_year"]  = df["predicted_week"].dt.year
df["prediction_month"] = df["predicted_week"].dt.month

# Citywide weekly aggregates keyed by current week
weekly = (
    df.groupby("current_week")
    .agg(
        predicted_week=("predicted_week", "first"),
        true_count=("true_count", "sum"),
        prediction=("prediction", "sum"),
        lower=("lower", "sum"),
        upper=("upper", "sum"),
        year=("year", "first"),
        month=("month", "first"),
        prediction_year=("prediction_year", "first"),
        prediction_month=("prediction_month", "first"),
    )
    .reset_index()
    .sort_values("current_week")
)

# Global x bounds across all horizons
df_all = df_raw.copy()
df_all["current_week"] = pd.to_datetime(df_all["interval_start_at_inference_time"])
x_min = df_all["current_week"].min().strftime("%Y-%m-%d")
x_max = (df_all["current_week"].max() + pd.Timedelta(days=70)).strftime("%Y-%m-%d")
# Note x_max has an extra 4-week buffer so that there is space for text

# # TEST ONLY. For inspecting the index of each week.
# m = df["predicted_week"].drop_duplicates().reset_index(drop=True)
# m

# Week selector

all_current_weeks = sorted(weekly["current_week"].unique())

default_week = pd.Timestamp("2025-07-04")
if default_week not in all_current_weeks:
    default_week = all_current_weeks[-1]

selected_current_week = st.select_slider(
    "Select current week (week when forecast is made):",
    options=all_current_weeks,
    value=default_week,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
)
selected_current_week = pd.Timestamp(selected_current_week)

# Derive predicted week
selected_row = weekly[weekly["current_week"] == selected_current_week]
if not selected_row.empty:
    selected_predicted_week = pd.Timestamp(selected_row["predicted_week"].values[0])
else:
    selected_predicted_week = selected_current_week + pd.Timedelta(weeks=horizon)

# cluster level
selected_cluster_rows = df[df["current_week"] == selected_current_week]

st.markdown(
    f"Current week: **{selected_current_week.strftime('%b %d, %Y')}**.\n\n"
    f"Predicting **{horizon} week{'s' if horizon >=2 else ''} ahead**, to "
    f"the week of **{selected_predicted_week.strftime('%b %d, %Y')}**."
)

st.divider()

# KPI cards

st.subheader("Key Indicators")

monthly = weekly.groupby(["year", "month"])["true_count"].sum()

def get_monthly_total(year, month):
    return monthly.get((year, month), None)

sel_year  = selected_current_week.year
sel_month = selected_current_week.month

cur_monthly  = get_monthly_total(sel_year, sel_month)
prev_monthly = get_monthly_total(sel_year - 1, sel_month)

if cur_monthly is not None and prev_monthly is not None and prev_monthly > 0:
    pct_change = (cur_monthly - prev_monthly) / prev_monthly * 100
    kpi1_value = f"{int(cur_monthly):,}"
    kpi1_delta = f"{pct_change:+.1f}% vs {sel_year - 1}"
else:
    kpi1_value = f"{int(cur_monthly):,}" if cur_monthly is not None else "N/A"
    kpi1_delta = "No prior year data"

weekly_avg = weekly["true_count"].mean()

year_data = weekly[weekly["year"] == sel_year]
if not year_data.empty:
    peak_idx   = year_data["true_count"].idxmax()
    peak_row   = year_data.loc[peak_idx]
    peak_value = f"{int(peak_row['true_count']):,} cases"
    peak_date  = peak_row["current_week"].strftime("Week of %b %d, %Y")
else:
    peak_value = "N/A"
    peak_date  = ""

if not selected_row.empty:
    forecast_val = selected_row["prediction"].values[0]
    lower_val    = selected_row["lower"].values[0]
    upper_val    = selected_row["upper"].values[0]
    kpi4_value   = f"{forecast_val:.1f} cases"
    kpi4_delta   = f"Interval: [{lower_val:.1f}, {upper_val:.1f}]"
else:
    kpi4_value = "N/A"
    kpi4_delta = ""

col11, col12, col13 = st.columns(3)
col21, col22, col23 = st.columns(3)

with col11:
    st.metric(
        label=f"Cases this month ({pd.Timestamp(sel_year, sel_month, 1).strftime('%b %Y')})",
        value=kpi1_value,
        delta=kpi1_delta,
        delta_color="inverse",
    )
with col12:
    st.metric(
        label="All-time citywide weekly average",
        value=f"{weekly_avg:.1f} cases",
    )
with col13:
    st.metric(
        label=f"Peak week in {sel_year}",
        value=peak_value,
        delta=peak_date,
        delta_color="off",
        delta_arrow="off",
    )
with col22:
    st.metric(
        label=f"FORECAST for week of {selected_predicted_week.strftime('%b %d, %Y')}",
        value=kpi4_value,
        delta=kpi4_delta,
        delta_color="off",
        delta_arrow="off",
    )

st.divider()

with st.expander("Cluster-level Forecasts for Number of Cases", expanded=True):
    row1 = st.columns(5)
    row2 = st.columns(5)
    row3 = st.columns(5)
    cells = row1 + row2 + row3

    for i, row in selected_cluster_rows.reset_index().iterrows():

        cluster_name = row["cluster"]

        if not row.empty:
            forecast_val = row["prediction"]
            lower_val    = row["lower"]
            upper_val    = row["upper"]
            kpi_value   = f"{forecast_val:.1f}"
            kpi_delta   = f"Interval: [{lower_val:.1f}, {upper_val:.1f}]"
        else:
            kpi_value = "N/A"
            kpi_delta = ""

        with cells[i]:
            st.metric(
                label=f"Cluster {cluster_name}",
                value=kpi_value,
                delta=kpi_delta,
                delta_color="off",
                delta_arrow="off",
            )

st.divider()

# Forecast bar chart by horizon

with st.expander("Average Forecasted Weekly Cases by Horizon", expanded=False):

    st.subheader("Average Forecasted Weekly Cases by Horizon")
    st.markdown(
        "- Each bar shows the mean citywide weekly predicted case count across all weeks "
        "in the last three years, for each forecast horizon.\n"
        "- Error bars show the mean half-width "
        "of the prediction interval."
    )

    all_horizons = [1, 2, 4, 8] if disease == "Leptospirosis" else [1, 2, 4, 8, 12]
    bar_rows = []

    for h in all_horizons:
        h_df = df_raw[df_raw["horizon"] == h].copy()
        h_df["interval_start"] = pd.to_datetime(h_df["interval_start"])
        h_weekly = (
            h_df.groupby("interval_start")[["prediction", "lower", "upper"]]
            .sum()
            .reset_index()
        )
        mean_pred      = h_weekly["prediction"].mean()
        mean_halfwidth = ((h_weekly["upper"] - h_weekly["lower"]) / 2).mean()
        bar_rows.append({"horizon": h, "mean_prediction": mean_pred, "error": mean_halfwidth})

    bar_df = pd.DataFrame(bar_rows)
    bar_df["error_minus"] = (bar_df["mean_prediction"] - bar_df["error"]).clip(lower=0)
    bar_df["error_minus"] = bar_df["mean_prediction"] - bar_df["error_minus"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[f"{h} wk" for h in bar_df["horizon"]],
        y=bar_df["mean_prediction"],
        error_y=dict(
            type="data",
            array=bar_df["error"].tolist(),
            arrayminus=bar_df["error_minus"].tolist(),
            visible=True,
            color="gray"
        ),
        marker_color="#1f77b4",
        hovertemplate="<b>%{x} ahead</b><br>Mean forecast: %{y:.1f}<br><extra></extra>",
    ))
    fig_bar.update_layout(
        xaxis_title="Forecast Horizon",
        yaxis_title="Mean Weekly Cases (citywide)",
        height=350,
        margin={"t": 20, "b": 40},
        plot_bgcolor="white",
        yaxis=dict(gridcolor="lightgrey"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Helper: build a time series figure given a weekly-aggregated dataframe

def build_timeseries(weekly_df, title, show_forecast, show_ci, show_rain,
                     hide_future_actual,
                     selected_current_week, selected_predicted_week,
                     horizon, x_min, x_max, y_min_range):

    fig = go.Figure()

    if show_rain:
        years_in_data = weekly_df["prediction_year"].unique()
        first = True
        for yr in sorted(years_in_data):
            fig.add_vrect(
                x0=f"{yr}-06-01", x1=f"{yr}-11-30",
                fillcolor="lightblue", opacity=0.15,
                line_width=0,
                annotation_text="Rainy Season" if first else "",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="steelblue",
            )
            first = False


    actual_data = weekly_df.copy()
    forecast_data = weekly_df.copy()
    if hide_future_actual:
        actual_data = actual_data[actual_data["predicted_week"] <= selected_current_week]
        forecast_data = forecast_data[forecast_data["predicted_week"] <= selected_predicted_week]

    if show_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_data["predicted_week"], forecast_data["predicted_week"][::-1]]),
            y=pd.concat([forecast_data["upper"], forecast_data["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255, 160, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prediction interval",
            hoverinfo="skip",
        ))

    if show_forecast:
        fig.add_trace(go.Scatter(
            x=forecast_data["predicted_week"],
            y=forecast_data["prediction"],
            mode="lines",
            name="Forecast",
            line=dict(color="orange", width=2, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: %{y:.1f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=actual_data["predicted_week"],
        y=actual_data["true_count"],
        mode="lines",
        name="Actual cases",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.0f}<extra></extra>",
    ))

    cur_str  = selected_current_week.strftime("%Y-%m-%d")
    pred_str = selected_predicted_week.strftime("%Y-%m-%d")

    fig.add_shape(
        type="line", x0=cur_str, x1=cur_str,
        y0=0, y1=1, yref="paper",
        line=dict(color="red", dash="dot", width=2),
    )
    fig.add_annotation(
        x=cur_str, y=0.98, yref="paper",
        text=f"Current week<br>{selected_current_week.strftime('%b %d')}",
        showarrow=False,
        font=dict(color="red", size=14),
        bgcolor="rgba(255,255,255,0.7)",
        xanchor="right",
    )

    if selected_predicted_week != selected_current_week:
        fig.add_shape(
            type="line", x0=pred_str, x1=pred_str,
            y0=0, y1=1, yref="paper",
            line=dict(color="green", dash="dot", width=2),
        )
        fig.add_annotation(
            x=pred_str, y=0.98, yref="paper",
            text=f"Predicted week<br>{selected_predicted_week.strftime('%b %d')}",
            showarrow=False,
            font=dict(color="green", size=14),
            bgcolor="rgba(255,255,255,0.7)",
            xanchor="left",
        )
        fig.add_annotation(
            x=pred_str, y=0.82, yref="paper",
            ax=-105, ay=0,
            text=f"{horizon}wk horizon",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gray",
            font=dict(color="gray", size=14),
            xanchor="left",
        )

    y_axis_max = max(
        weekly_df["true_count"].max(),
        weekly_df["upper"].max() if show_ci else 0,
        y_min_range,
    )

    fig.update_layout(
        xaxis_title="Week when forecast is relevant",
        yaxis_title="Case count",
        height=480,
        margin={"t": 20, "b": 40},
        plot_bgcolor="white",
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(gridcolor="lightgrey", range=[0, y_axis_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )

    return fig


# Layer toggle checkboxes (shared by both time series)

y_min_range = 1100 if disease == "Dengue" else 350

c1, c2, c3, c4 = st.columns(4)
with c1:
    show_forecast = st.checkbox("Show model forecast", value=True)
with c2:
    show_ci = st.checkbox("Show prediction interval", value=True)
with c3:
    show_rain = st.checkbox("Highlight rainy season (Jun–Nov)", value=True)
with c4:
    hide_future_actual = st.checkbox("Hide data after current week", value=True)

# Citywide time series

st.subheader("Citywide Weekly Case Count")

fig_city = build_timeseries(
    weekly_df=weekly,
    title="Citywide",
    show_forecast=show_forecast,
    show_ci=show_ci,
    show_rain=show_rain,
    hide_future_actual=hide_future_actual,
    selected_current_week=selected_current_week,
    selected_predicted_week=selected_predicted_week,
    horizon=horizon,
    x_min=x_min,
    x_max=x_max,
    y_min_range=y_min_range,
)
st.plotly_chart(fig_city, use_container_width=True)

st.divider()

# Cluster-level time series

st.subheader("Barangay Cluster Weekly Case Count")

clusters = sorted(df["cluster"].unique())
selected_cluster = st.radio(
    "Select cluster:",
    clusters,
    horizontal=True,
    format_func=lambda c: f"Cluster {c}",
    key="dashboard_cluster_radio",
)

cluster_df = df[df["cluster"] == selected_cluster].copy()
cluster_weekly = (
    cluster_df.groupby("current_week")
    .agg(
        predicted_week=("predicted_week", "first"),
        true_count=("true_count", "sum"),
        prediction=("prediction", "sum"),
        lower=("lower", "sum"),
        upper=("upper", "sum"),
        year=("year", "first"),
        month=("month", "first"),
        prediction_year=("prediction_year", "first"),
        prediction_month=("prediction_month", "first"),
    )
    .reset_index()
    .sort_values("current_week")
)

fig_cluster = build_timeseries(
    weekly_df=cluster_weekly,
    title=f"Cluster {selected_cluster}",
    show_forecast=show_forecast,
    show_ci=show_ci,
    show_rain=show_rain,
    hide_future_actual=hide_future_actual,
    selected_current_week=selected_current_week,
    selected_predicted_week=selected_predicted_week,
    horizon=horizon,
    x_min=x_min,
    x_max=x_max,
    y_min_range=0,  # no minimum y for cluster-level
)
st.plotly_chart(fig_cluster, use_container_width=True)