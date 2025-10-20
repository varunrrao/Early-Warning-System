# ============================================================
#                      IMPORT SECTION
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# ============================================================
#                      CONFIGURATION
# ============================================================
APP_TITLE = "Early Warning System for Control Valve Failure Prediction"
DATE_COL = "Date"
COL_FINAL = ".FINAL_VALUE"
COL_POSITION = ".FINAL_POSITION_VALUE"
SUPPORTED_EXTENSIONS = ("csv", "xlsx", "xlsb")
EXPORT_PATH = "C:/Control Valve Diagnostics/Diagnostic_Results.xlsx"

# ML configuration parameters
CLUSTER_COUNT = 2
ANOMALY_CONTAMINATION = 0.1
ARIMA_ORDER = (1, 1, 1)
SCORE_WEIGHT_THRESHOLD = 0.7
SCORE_WEIGHT_VARIATION = 0.3
SCORE_MAX = 100
SEED = 42

# ============================================================
#                      DATA INGESTION
# ============================================================
@st.cache_data
def read_file(uploaded_file: BytesIO) -> pd.DataFrame:
    """Reads input file into pandas DataFrame. Supports CSV, XLSX, XLSB."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xlsb"):
            import pyxlsb
            df = pd.read_excel(uploaded_file, engine="pyxlsb")
        else:
            st.error("❌ Unsupported file format.")
            return None
        return df
    except Exception as e:
        st.error(f"File read error: {e}")
        return None

# ============================================================
#                   MODELING AND COMPUTATION
# ============================================================
def compute_failure_score(df_tag: pd.DataFrame):
    """
    Computes valve failure prediction score using:
    - Error and AbsError
    - 95th percentile threshold-based severity classification
    - ARIMA trend modeling
    - KMeans clustering
    - Isolation Forest anomaly detection
    """

    df_tag = df_tag.copy()
    df_tag[COL_FINAL] = pd.to_numeric(df_tag[COL_FINAL], errors="coerce")
    df_tag[COL_POSITION] = pd.to_numeric(df_tag[COL_POSITION], errors="coerce")
    df_tag.dropna(subset=[COL_FINAL, COL_POSITION], inplace=True)

    if df_tag.empty:
        return 0, 0, df_tag

    # --- Error and Absolute Error ---
    df_tag["Error"] = df_tag[COL_FINAL] - df_tag[COL_POSITION]
    df_tag["AbsError"] = df_tag["Error"].abs()

    # --- Statistical Thresholds ---
    threshold_95 = df_tag["AbsError"].quantile(0.95)

    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=SEED, n_init=10)
    df_tag["Cluster"] = kmeans.fit_predict(df_tag[["AbsError"]])

    # --- ARIMA Modeling ---
    try:
        model = ARIMA(df_tag["Error"], order=ARIMA_ORDER)
        results = model.fit()
        df_tag["ARIMA_Predicted"] = results.fittedvalues
    except Exception:
        df_tag["ARIMA_Predicted"] = df_tag["Error"]

    # --- Isolation Forest Anomaly Detection ---
    iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=SEED)
    df_tag["Anomaly"] = iso.fit_predict(df_tag[["AbsError"]])

    # --- Calculate Anomaly-Based Ratios ---
    anomaly_counts = df_tag["Anomaly"].value_counts().to_dict()
    normal_count = anomaly_counts.get(1, 0)
    abnormal_count = anomaly_counts.get(-1, 0)
    total_count = normal_count + abnormal_count
    abnormal_ratio = abnormal_count / total_count if total_count > 0 else 0

    # --- Threshold-based Risk Classification ---
    if threshold_95 <= 5:
        base_score = np.interp(threshold_95, [0, 5], [0, 20])  # Low failure risk (0–20)
    elif 5 < threshold_95 <= 20:
        base_score = np.interp(threshold_95, [5, 20], [20, 40])  # Moderate failure (20–40)
    else:
        base_score = np.interp(threshold_95, [20, threshold_95 + 1], [40, 100])  # High failure (40–100)

    # --- Adjust by anomaly severity ---
    score = base_score * (1 + abnormal_ratio)
    score = np.clip(score, 0, 100)  # keep within 0–100

    # --- Assign final results ---
    df_tag["FailureScore"] = round(score, 2)

    return threshold_95, round(score, 2), df_tag
# ============================================================
#                      VALVE PROCESSING
# ============================================================
def process_valve_data(df: pd.DataFrame):
    all_data = {}
    result_rows = []

    tags = [col.replace(COL_FINAL, "") for col in df.columns if COL_FINAL in col]
    progress_bar = st.progress(0)
    total = len(tags)

    for i, tag in enumerate(tags):
        final_col, pos_col = tag + COL_FINAL, tag + COL_POSITION
        if final_col not in df.columns or pos_col not in df.columns:
            continue

        df_tag = df[[DATE_COL, final_col, pos_col]].copy()
        df_tag.columns = [DATE_COL, COL_FINAL, COL_POSITION]
        df_tag[DATE_COL] = pd.to_datetime(df_tag[DATE_COL], errors="coerce")
        df_tag.dropna(subset=[DATE_COL], inplace=True)

        threshold_95, score, df_proc = compute_failure_score(df_tag)

        result_rows.append(
            {
                "Valve Tag": tag,
                "95th '%' Error Threshold": round(threshold_95, 3),
                "Failure Prediction Score (%)": score,
            }
        )
        all_data[tag] = df_proc
        progress_bar.progress((i + 1) / total)

    progress_bar.empty()
    result_df = pd.DataFrame(result_rows)
    result_df.sort_values(by="Failure Prediction Score (%)", ascending=False, inplace=True)
    return result_df, all_data

# ============================================================
#                      VISUALIZATION
# ============================================================
def plot_trend(df_tag, tag):
    """Plots Error trend and highlights the 95th percentile threshold as a dotted line."""
    threshold_95 = df_tag["AbsError"].quantile(0.95)

    fig = go.Figure()

    # Error trend line
    fig.add_trace(go.Scatter(
        x=df_tag[DATE_COL],
        y=df_tag["Error"],
        mode="lines",
        name="Error",
        line=dict(color="steelblue", width=2)
    ))

    # 95th percentile threshold line
    fig.add_hline(
        y=threshold_95,
        line=dict(color="red", width=2, dash="dot"),
        annotation_text=f"95th % Threshold = {threshold_95:.2f}",
        annotation_position="top left"
    )

    fig.add_hline(
        y=-threshold_95,
        line=dict(color="red", width=2, dash="dot"),
        annotation_text=f"-95th % Threshold = {-threshold_95:.2f}",
        annotation_position="bottom left"
    )

    fig.update_layout(
        title=f"Error Trend with 95th Percentile Threshold - {tag}",
        xaxis_title="Date",
        yaxis_title="Error",
        template="plotly_white",
        
    )
    st.plotly_chart(fig, use_container_width=True)
def plot_value_trend(df_tag, tag):
    """Plots FINAL_VALUE and FINAL_POSITION_VALUE trends against Date."""

    fig = go.Figure()

    # FINAL_VALUE trend line
    fig.add_trace(go.Scatter(
        x=df_tag[DATE_COL],
        y=df_tag[COL_FINAL],
        mode="lines",
        name="FINAL_VALUE",
        line=dict(color="green", width=2)
    ))

    # FINAL_POSITION_VALUE trend line
    fig.add_trace(go.Scatter(
        x=df_tag[DATE_COL],
        y=df_tag[COL_POSITION],
        mode="lines",
        name="FINAL_POSITION_VALUE",
        line=dict(color="orange", width=2)
    ))

    fig.update_layout(
        title=f"FINAL_VALUE vs FINAL_POSITION_VALUE Trend - {tag}",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
def plot_gauge(score, threshold, tag): 
    fig = go.Figure(go.Indicator( mode="gauge+number+delta", value=score, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': f"Failure Prediction - {tag}<br>95th Percentile: {threshold:.2f}"}, gauge={ 'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"}, 'steps': [ {'range': [0, 50], 'color': "#098a20"}, {'range': [50, 75], 'color': "#ffc966"}, {'range': [75, 100], 'color': '#990000'} ], 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score} })) 
    st.plotly_chart(fig, use_container_width=True)
def plot_heatmap(df_tag, tag):
    df_tag["DateOnly"] = df_tag[DATE_COL].dt.date
    df_tag["Hour"] = df_tag[DATE_COL].dt.hour
    heatmap_data = df_tag.pivot_table(values="AbsError", index="Hour", columns="DateOnly", aggfunc="mean")
    heatmap_fig = px.imshow(
        heatmap_data,
        color_continuous_scale="RdBu_r",
        labels=dict(x="Date", y="Hour", color="Absolute Error"),
        title=f"Absolute Error Heatmap - {tag}",
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
def plot_abnormalities_per_date(df_tag, tag):
    """
    Plots the number of abnormalities detected per date for a specific valve tag.
    Abnormalities are determined by rows where 'Anomaly Type' is not 'No Anomaly Detected'.

    Parameters:
        df_tag (pd.DataFrame): Data for a single valve tag (must include 'Anomaly Type' and 'Date' columns).
        tag (str): Valve tag name for display in the plot title.
    """

    st.subheader(f"📊 Number of Abnormalities Per Date — {tag}")
       # === Constants ===
    IF_ANOMALY_FLAG = -1        # Isolation Forest anomaly indicator
    KMEANS_ANOMALY_CLUSTER = 1 
    # === Ensure 'Date' column exists and is datetime ===
    if "Date" not in df_tag.columns:
        st.warning("⚠️ No 'Date' column found in the data.")
        return

    df_tag = df_tag.copy()
    df_tag["Date"] = pd.to_datetime(df_tag["Date"], errors='coerce')
      # === Determine anomaly type ===
    anomaly_types = []
    for if_flag, km_flag in zip(df_tag.get("Anomaly", [0]), df_tag.get("Cluster", [0])):
        if if_flag == IF_ANOMALY_FLAG and km_flag == KMEANS_ANOMALY_CLUSTER:
            anomaly_types.append("Isolation-Forest+K-Means")
        elif if_flag == IF_ANOMALY_FLAG:
            anomaly_types.append("Isolation-Forest")
        elif km_flag == KMEANS_ANOMALY_CLUSTER:
            anomaly_types.append("K-Means")
        else:
            anomaly_types.append("No Anomaly Detected")

    df_tag["Anomaly Type"] = anomaly_types
    # === Filter rows with abnormalities ===
    filtered_abnormalities = df_tag.loc[df_tag["Anomaly Type"] != "No Anomaly Detected", ["Date", "Anomaly Type"]]
    if filtered_abnormalities.empty:
        st.info("✅ No abnormalities detected for this tag.")
        return

    # === Count abnormalities per date ===
    filtered_abnormalities = filtered_abnormalities.set_index("Date")
    abnormality_counts = filtered_abnormalities.resample("D").size()

    if abnormality_counts.sum() > 0:
        # === Plot anomalies counts ===
        fig, ax = plt.subplots(figsize=(8, 5))
        abnormality_counts.index = abnormality_counts.index.strftime("%Y-%m-%d")
        abnormality_counts.plot(kind="bar", color="purple", alpha=0.7, ax=ax)
        ax.set_xlabel("Date", fontsize=6)
        ax.set_ylabel("Number of Anomalies", fontsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.xticks(rotation=45, ha="right")
        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=max(1, len(abnormality_counts) // 20))
        )

        st.pyplot(fig)
    else:
        st.info("✅ No abnormalities recorded for this tag.")
def show_valve_table(df_tag, tag):
    """
    Displays raw data table (Date, FINAL_VALUE, FINAL_POSITION_VALUE, Error, AbsError, ARIMA Predicted Error)
    and visually highlights anomalies detected by Isolation Forest or K-Means.
    Also includes an 'Anomaly Type' column for clarity.

    Parameters:
        df_tag (pd.DataFrame): Data for a single valve tag with Error, AbsError, ARIMA_Predicted,
                               Anomaly, and Cluster columns.
        tag (str): Valve tag name.
    """

    # === Constants ===
    IF_ANOMALY_FLAG = -1        # Isolation Forest anomaly indicator
    KMEANS_ANOMALY_CLUSTER = 1  # KMeans cluster number indicating anomaly

    # === Ensure ARIMA column exists ===
    if "ARIMA_Predicted" not in df_tag.columns:
        df_tag["ARIMA_Predicted"] = np.nan  # placeholder to avoid KeyError

    # === Prepare display dataframe ===
    display_df = df_tag[
        [DATE_COL, COL_FINAL, COL_POSITION, "Error", "AbsError", "ARIMA_Predicted"]
    ].copy()
    display_df = display_df.sort_values(by=DATE_COL)

    # === Determine anomaly type ===
    anomaly_types = []
    for if_flag, km_flag in zip(df_tag.get("Anomaly", [0]), df_tag.get("Cluster", [0])):
        if if_flag == IF_ANOMALY_FLAG and km_flag == KMEANS_ANOMALY_CLUSTER:
            anomaly_types.append("Isolation-Forest+K-Means")
        elif if_flag == IF_ANOMALY_FLAG:
            anomaly_types.append("Isolation-Forest")
        elif km_flag == KMEANS_ANOMALY_CLUSTER:
            anomaly_types.append("K-Means")
        else:
            anomaly_types.append("No Anomaly Detected")

    display_df["Anomaly Type"] = anomaly_types

    # === Streamlit Section Header ===
    st.subheader(f"📋 Raw ISAE Control Valve Data with  Anomaly Detection — {tag}")

    # === Define row highlighting logic ===
    def highlight_anomalies(row):
        if row["Anomaly Type"] == "Isolation-Forest+K-Means":
            return ["background-color: #cc0000; color: white;"] * len(row)  # dark red
        elif row["Anomaly Type"] == "Isolation-Forest":
            return ["background-color: #ff9999;"] * len(row)  # light red
        elif row["Anomaly Type"] == "K-Means":
            return ["background-color: #fff5ba;"] * len(row)  # pale yellow
        else:
            return [""] * len(row)

    # === Display the dataframe with formatting ===
    st.dataframe(
        display_df.style
        .apply(highlight_anomalies, axis=1)
        .format({
            COL_FINAL: "{:.3f}",
            COL_POSITION: "{:.3f}",
            "Error": "{:.3f}",
            "AbsError": "{:.3f}",
            "ARIMA_Predicted": "{:.3f}",
        }),
        width='stretch',
    )
    display_df["Anomaly Type"] = anomaly_types

def export_results(all_data):
    with pd.ExcelWriter(EXPORT_PATH, engine="openpyxl") as writer:
        for tag, df_tag in all_data.items():
            safe_tag = re.sub(r"[\\/*?:\[\]]", "_", tag)[:31]
            df_tag.reset_index(drop=True).to_excel(writer, index=False, sheet_name=safe_tag)
    st.success(f"✅ Results exported successfully: {EXPORT_PATH}")
# ============================================================
#                      DRIVER CODE
# ============================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.radio("Select Page", ["Bad Ranking + Detailed Result", "Absolute Error Heatmap of all Tags"])

    st.title(APP_TITLE)
    st.caption("Hybrid AI/ML Model: ARIMA + K-Means Clustering + Isolation Forest")
    uploaded_file = st.file_uploader(
        "📂 Upload the Raw ISAE Control Valve Data file (.csv, .xlsx, .xlsb)", type=list(SUPPORTED_EXTENSIONS)
    )

    if not uploaded_file:
        st.info("Please upload a valve data file to continue.")
        return

    df = read_file(uploaded_file)
    if df is None or df.empty:
        st.error("Invalid or empty dataset.")
        return

    # Process data
    result_df, all_data = process_valve_data(df)

    if page == "Bad Ranking + Detailed Result":
        st.header("📊 Bad Ranking Table")
        st.dataframe(
            result_df.style.apply(
                lambda row: [
            "font-weight: bold; color: black;" if i == 0 else (
                "font-weight: bold; color: black; background-color: #99e699;" if row["Failure Prediction Score (%)"] < 25 else
                "font-weight: bold; color: black; background-color: #ffeb99;" if row["Failure Prediction Score (%)"] < 75 else
                "font-weight: bold; color: black; background-color: #ff6666;"
            )
            for i in range(len(row))
        ],
        axis=1), width='stretch')
        selected_tag = st.selectbox("Select Valve Tag for Detailed Analysis", result_df["Valve Tag"])
        if selected_tag in all_data:
            df_tag = all_data[selected_tag]
            threshold_95 = df_tag["Error"].quantile(0.95)
            score = df_tag["FailureScore"].iloc[0]
            plot_gauge(score, threshold_95, selected_tag)
            #plot_trend(df_tag, selected_tag)
            plot_value_trend(df_tag, selected_tag)
            plot_heatmap(df_tag, selected_tag)
            show_valve_table(df_tag, selected_tag)
            plot_abnormalities_per_date(df_tag, selected_tag)
        if st.button("💾 Export Results to Excel"):
            export_results(all_data)

    elif page == "Absolute Error Heatmap of all Tags":
        st.header("Individual Heatmaps for All Valve Tags")
        for tag, df_tag in all_data.items():
            plot_heatmap(df_tag, tag)

# ============================================================
#                       ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()



