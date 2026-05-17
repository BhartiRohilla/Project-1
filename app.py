import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="HHS Care Forecast Studio",
    layout="wide"
)

# Custom CSS for right-aligned date/time display
st.markdown("""
    <style>
    .datetime-container {
        position: fixed;
        top: 55px;
        right: 20px;
        z-index: 999;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Arial', sans-serif;
        text-align: right;
        color: white;
    }
    .date-text {
        font-size: 12px;
        opacity: 0.9;
    }
    .time-text {
        font-size: 18px;
        font-weight: bold;
    }
    @media (max-width: 768px) {
        .datetime-container {
            position: relative;
            text-align: center;
            margin-bottom: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header section
col_title, col_spacer, col_datetime = st.columns([2, 1, 1])

with col_title:
    st.title("🏥 HHS Future Care Load Dashboard")

with col_datetime:
    # Get current date and time
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y")
    current_time = now.strftime("%I:%M:%S %p")
    
    # Display date and time with custom styling
    st.markdown(f"""
    <div class="datetime-container">
        <div class="date-text">📅 {current_date}</div>
        <div class="time-text">⏰ {current_time}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# 2. FILE PATHS
# =========================================================
DATA_PATH = 'Data/processed_hhs_data.csv'
MODEL_PATH = 'models/random_forest_model.pkl'

# =========================================================
# 3. LOAD DATA + MODEL WITH ERROR HANDLING
# =========================================================
@st.cache_resource
def load_assets():
    try:
        df = pd.read_csv(DATA_PATH)
        
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Create proper timeline (2023–2025)
        df['Date'] = pd.date_range(
            start='2023-01-01',
            periods=len(df),
            freq='D'
        )
        
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Load model with error handling
        try:
            model = joblib.load(MODEL_PATH)
        except:
            st.warning("⚠️ Model file not found. Using baseline model only.")
            model = None
        
        return df, model
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Create sample data if real data not found
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            'Children in HHS Care': np.random.randint(1000, 5000, size=365)
        }, index=dates)
        return df, None

df, model = load_assets()

# =========================================================
# 4. TARGET COLUMN
# =========================================================
target_col = "Children in HHS Care"

# =========================================================
# 5. SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("⚙ User Controls")

horizon = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)

model_choice = st.sidebar.radio(
    "Model Selection",
    ["Random Forest", "Baseline (MA7)"]
)

# Disable Random Forest if model not available
if model is None and model_choice == "Random Forest":
    st.sidebar.warning("⚠️ Random Forest model not available. Using Baseline (MA7).")
    model_choice = "Baseline (MA7)"

scenario = st.sidebar.selectbox(
    "Scenario Simulation",
    ["Normal Flow", "High Intake Surge (+20%)", "Optimized Discharges (+15%)"]
)

# ✅ DATE RANGE FILTER FOR GRAPH
st.sidebar.divider()
st.sidebar.subheader("📅 Graph Date Range Filter")

# Get date range from data
min_date = df.index.min().date()
max_date = df.index.max().date()

# Quick select options for graph
graph_date_range = st.sidebar.selectbox(
    "Select Graph Date Range",
    ["Last 30 Days", "Last 90 Days", "Last 180 Days", "Last Year", "All Time", "Custom Range"]
)

# Set date range based on selection
today = datetime.now().date()
if graph_date_range == "Last 30 Days":
    graph_start_date = today - timedelta(days=30)
    graph_end_date = max_date
elif graph_date_range == "Last 90 Days":
    graph_start_date = today - timedelta(days=90)
    graph_end_date = max_date
elif graph_date_range == "Last 180 Days":
    graph_start_date = today - timedelta(days=180)
    graph_end_date = max_date
elif graph_date_range == "Last Year":
    graph_start_date = today - timedelta(days=365)
    graph_end_date = max_date
elif graph_date_range == "All Time":
    graph_start_date = min_date
    graph_end_date = max_date
else:  # Custom Range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        graph_start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=30),
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        graph_end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

# Validate date range
if graph_start_date > graph_end_date:
    st.sidebar.error("❌ Start date must be before end date")
    graph_start_date, graph_end_date = graph_end_date, graph_start_date

# ✅ FUTURE DATE INPUT
st.sidebar.divider()
st.sidebar.subheader("📅 Future Date Prediction")

selected_date = st.sidebar.date_input(
    "Select Future Date",
    value=pd.to_datetime("2026-01-15")
)

# =========================================================
# 6. FORECAST GENERATION
# =========================================================
last_date = df.index.max()

# Create forecast dates
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=horizon,
    freq='D'
)

forecast_values = []
current_series = df[target_col].tolist()

# Add progress bar for long operations
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

for i in range(horizon):
    status_text.text(f"Generating forecast... Day {i+1}/{horizon}")
    progress_bar.progress((i+1)/horizon)
    
    if model_choice == "Baseline (MA7)":
        next_pred = np.mean(current_series[-7:])
    else:
        try:
            lag_1 = current_series[-1]
            lag_7 = current_series[-7]
            
            day_of_week = forecast_dates[i].dayofweek
            month = forecast_dates[i].month
            
            features = pd.DataFrame(
                [[lag_1, lag_7, day_of_week, month]],
                columns=[
                    'Children in HHS Care_lag_1',
                    'Children in HHS Care_lag_7',
                    'day_of_week',
                    'month'
                ]
            )
            
            if model is not None:
                next_pred = model.predict(features)[0]
            else:
                next_pred = np.mean(current_series[-7:])
        except Exception as e:
            st.warning(f"Prediction error: {str(e)}. Using baseline.")
            next_pred = np.mean(current_series[-7:])
    
    # Scenario adjustment
    if scenario == "High Intake Surge (+20%)":
        next_pred *= 1.20
    elif scenario == "Optimized Discharges (+15%)":
        next_pred *= 0.85
    
    forecast_values.append(next_pred)
    current_series.append(next_pred)

# Clear progress indicators
progress_bar.empty()
status_text.empty()

forecast_df = pd.DataFrame(
    {target_col: forecast_values},
    index=forecast_dates
)

# =========================================================
# 7. FUTURE DATE PREDICTION ENGINE
# =========================================================
selected_date = pd.to_datetime(selected_date)
days_ahead = (selected_date - last_date).days

predicted_value_for_date = None

if days_ahead > 0 and days_ahead <= 365:
    temp_series = df[target_col].tolist()
    
    with st.spinner(f"Calculating prediction for {selected_date.date()}..."):
        for i in range(days_ahead):
            if model_choice == "Baseline (MA7)":
                next_pred = np.mean(temp_series[-7:])
            else:
                try:
                    lag_1 = temp_series[-1]
                    lag_7 = temp_series[-7]
                    
                    future_day = last_date + pd.Timedelta(days=i + 1)
                    day_of_week = future_day.dayofweek
                    month = future_day.month
                    
                    features = pd.DataFrame(
                        [[lag_1, lag_7, day_of_week, month]],
                        columns=[
                            'Children in HHS Care_lag_1',
                            'Children in HHS Care_lag_7',
                            'day_of_week',
                            'month'
                        ]
                    )
                    
                    if model is not None:
                        next_pred = model.predict(features)[0]
                    else:
                        next_pred = np.mean(temp_series[-7:])
                except:
                    next_pred = np.mean(temp_series[-7:])
            
            # Scenario adjustment
            if scenario == "High Intake Surge (+20%)":
                next_pred *= 1.20
            elif scenario == "Optimized Discharges (+15%)":
                next_pred *= 0.85
            
            temp_series.append(next_pred)
        
        predicted_value_for_date = int(temp_series[-1])
elif days_ahead > 365:
    st.sidebar.warning("⚠️ Date too far in the future. Please select a date within 1 year.")

# =========================================================
# 8. KPI DASHBOARD
# =========================================================
st.subheader("📋 Operational Capacity Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    current_value = int(df[target_col].iloc[-1])
    st.metric("Current Load", f"{current_value:,}")

with col2:
    end_forecast = int(forecast_values[-1]) if forecast_values else 0
    delta = end_forecast - current_value
    st.metric("End Forecast", f"{end_forecast:,}", delta=f"{delta:+,}")

with col3:
    discharge_demand = int(forecast_values[0] * 0.062) if forecast_values else 0
    st.metric("Discharge Demand (Next 24h)", f"{discharge_demand:,}")

# =========================================================
# FUTURE DATE RESULT
# =========================================================
if predicted_value_for_date is not None:
    st.subheader("📌 Future Date Forecast")
    st.info(
        f"Estimated patients on **{selected_date.date()}** : "
        f"**{predicted_value_for_date:,}**"
    )
elif days_ahead > 0:
    st.warning(f"⚠️ Unable to calculate prediction for {selected_date.date()}")

# =========================================================
# 9. VISUALIZATION - FIXED VERSION
# =========================================================
st.subheader("📈 Interactive Timeline Tracking")

# Convert graph dates to pandas Timestamp for filtering
graph_start_timestamp = pd.Timestamp(graph_start_date)
graph_end_timestamp = pd.Timestamp(graph_end_date)

# Filter historical data based on selected date range
historical_filtered = df[(df.index >= graph_start_timestamp) & (df.index <= graph_end_timestamp)]

# Filter forecast data correctly
forecast_filtered = forecast_df[(forecast_df.index >= graph_start_timestamp) & (forecast_df.index <= graph_end_timestamp)]

# Check if we have data to display
has_historical = len(historical_filtered) > 0
has_forecast = len(forecast_filtered) > 0

if has_historical or has_forecast:
    # Create a combined dataframe for plotting
    plot_data = []
    
    if has_historical:
        hist_df = historical_filtered[[target_col]].copy()
        hist_df['Type'] = 'Historical'
        plot_data.append(hist_df)
    
    if has_forecast:
        forecast_plot_df = forecast_filtered[[target_col]].copy()
        forecast_plot_df['Type'] = 'Forecast'
        plot_data.append(forecast_plot_df)
    
    if plot_data:
        combined_plot_df = pd.concat(plot_data)
        combined_plot_df = combined_plot_df.reset_index()
        combined_plot_df.columns = ['Date', target_col, 'Type']
        
        # Create the plot using plotly express
        fig = px.line(
            combined_plot_df,
            x='Date',
            y=target_col,
            color='Type',
            title=f"Time-Series Progression ({graph_start_date} to {graph_end_date})",
            color_discrete_map={'Historical': '#636EFA', 'Forecast': '#EF553B'}
        )
        
        # Update forecast line to be dashed
        for trace in fig.data:
            if trace.name == 'Forecast':
                trace.line.dash = 'dash'
        
        # Add a marker for the current date
        current_date = last_date
        if graph_start_timestamp <= current_date <= graph_end_timestamp:
            current_value = df.loc[current_date, target_col] if current_date in df.index else None
            if current_value is not None:
                fig.add_scatter(
                    x=[current_date],
                    y=[current_value],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='circle', line=dict(color='darkred', width=2)),
                    name='Present',
                    showlegend=True
                )
        
        # Update layout
        fig.update_layout(
            template="plotly_white",
            hovermode='x unified',
            legend_title_text='Data Type',
            xaxis_title='Date',
            yaxis_title=target_col,
            height=500
        )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(count=90, label="3m", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display date range info
        total_records = len(historical_filtered) + len(forecast_filtered)
        st.caption(f"📊 Showing data from {graph_start_date} to {graph_end_date} | Total records: {total_records}")
    else:
        st.warning(f"No data available for the selected date range")
else:
    st.warning(f"No data available for the selected date range: {graph_start_date} to {graph_end_date}")

# =========================================================
# 10. ADDITIONAL DATA TABLE
# =========================================================
with st.expander("📊 View Historical & Forecast Data"):
    if has_historical or has_forecast:
        display_data = []
        
        if has_historical:
            hist_display = historical_filtered[[target_col]].copy()
            hist_display['Data Type'] = 'Historical'
            display_data.append(hist_display)
        
        if has_forecast:
            forecast_display = forecast_filtered[[target_col]].copy()
            forecast_display['Data Type'] = 'Forecast'
            display_data.append(forecast_display)
        
        if display_data:
            display_df = pd.concat(display_data)
            display_df = display_df.reset_index()
            display_df.columns = ['Date', target_col, 'Data Type']
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"hhs_data_{graph_start_date}_to_{graph_end_date}.csv",
                mime="text/csv"
            )
    else:
        st.info("No data to display for the selected date range")

# =========================================================
# 11. EARLY WARNING SYSTEM
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("⚠️ Early Warning System")

if forecast_values:
    current_load = df[target_col].iloc[-1]
    forecast_peak = max(forecast_values)
    capacity_threshold = current_load * 1.2
    
    if forecast_peak > capacity_threshold:
        st.sidebar.error(f"🚨 CRITICAL: Forecast exceeds capacity by {((forecast_peak/capacity_threshold)-1)*100:.1f}%")
        st.sidebar.markdown(f"**Peak Forecast:** {int(forecast_peak):,} children")
        st.sidebar.markdown(f"**Safe Capacity:** {int(capacity_threshold):,} children")
    elif forecast_peak > current_load * 1.1:
        st.sidebar.warning(f"⚠️ WARNING: Capacity stress expected within {horizon} days")
        st.sidebar.markdown(f"**Expected Peak:** {int(forecast_peak):,} children")
    else:
        st.sidebar.success("✅ Operations within normal range")

# =========================================================
# 12. MODEL INFO
# =========================================================
with st.expander("ℹ️ About the Forecasting Models"):
    st.markdown("""
    ### 🤖 How the Forecast Works
    
    **Random Forest Model Features:**
    - 📊 Lag values (1 and 7 days)
    - 📅 Day of week patterns
    - 🌙 Monthly seasonality
    
    **Why Random Forest?**
    - Handles non-linear relationships
    - Robust to outliers
    - Provides feature importance
    
    **Limitations:**
    - Cannot predict sudden policy changes
    - Requires historical patterns to continue
    - Uncertainty increases with horizon length
    
    **Baseline Model (MA7):**
    - Simple 7-day moving average
    - Used for comparison and fallback
    """)

# =========================================================
# 13. SUMMARY STATISTICS - FIXED
# =========================================================
if has_historical and len(historical_filtered) > 0:
    with st.expander("📈 Summary Statistics for Selected Range"):
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate statistics safely
        avg_value = historical_filtered[target_col].mean()
        max_value = historical_filtered[target_col].max()
        min_value = historical_filtered[target_col].min()
        std_value = historical_filtered[target_col].std()
        
        with col1:
            st.metric(
                "Average Daily Load",
                f"{int(avg_value):,}" if not pd.isna(avg_value) else "N/A"
            )
        with col2:
            st.metric(
                "Maximum Load",
                f"{int(max_value):,}" if not pd.isna(max_value) else "N/A"
            )
        with col3:
            st.metric(
                "Minimum Load",
                f"{int(min_value):,}" if not pd.isna(min_value) else "N/A"
            )
        with col4:
            st.metric(
                "Std Deviation",
                f"{int(std_value):,}" if not pd.isna(std_value) else "N/A"
            )

# =========================================================
# 14. FOOTER
# =========================================================
st.divider()
st.caption("🏥 AI-Powered HHS Forecasting Dashboard | Data-driven decisions for child welfare")