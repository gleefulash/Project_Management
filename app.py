import streamlit as st
import pandas as pd
import plotly.express as px
import os
import hashlib
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import base64
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Web Analytics", layout="wide")

# Convert image to base64
def get_base64_image():
    try:
        with open("homepage.jpg", "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.warning("homepage.jpg not found. Using default background.")
        return ""

# Embed base64 image in CSS
background_image = get_base64_image()
background_css = f"""
    <style>
    /* General styles */
    .block-container {{ padding: 0.5rem; max-width: 1200px; margin: auto; }}
    .plotly-chart {{
        width: 100% !important;
        border-radius: 10px;
    }}
    .css-1d391kg {{ width: 200px; padding: 0.5rem; border-radius: 6px; }}
    .stButton>button {{
        background: linear-gradient(45deg, #ff4500, #ff8c00);
        color: white; font-family: 'Poppins', sans-serif;
        font-size: 0.9rem; padding: 0.4rem 0.8rem; border: none; border-radius: 20px;
        width: 100%; cursor: pointer; transition: transform 0.2s;
    }}
    .stButton>button:hover {{ transform: scale(1.05); background: linear-gradient(45deg, #e63900, #e07b00); }}
    .st-subheader {{
        font-family: 'Poppins', sans-serif; color: #ffffff; font-size: 1rem; margin-bottom: 0.3rem;
    }}

    /* Title */
    .dashboard-title-container {{
        background: linear-gradient(135deg, #ff4500, #ff8c00);
        padding: 0.5rem; text-align: center; border-radius: 6px; margin-bottom: 0.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }}
    .dashboard-title {{
        font-family: 'Poppins', sans-serif; font-size: min(2rem, 6vw); color: #ffffff; margin: 0;
    }}

    /* Metric cards */
    .metric-card {{
        border-radius: 6px; padding: 0.5rem; margin: 0.1rem; text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); transition: transform 0.2s;
        width: 100px; height: 100px; display: flex; flex-direction: column; justify-content: center;
    }}
    .metric-card:hover {{ transform: translateY(-3px); }}
    .metric-label {{
        font-family: 'Poppins', sans-serif; font-size: 0.6rem; color: #e0e0e0; opacity: 0.9; margin-bottom: 0.2rem;
    }}
    .metric-value {{
        font-family: 'Poppins', sans-serif; font-size: 1rem; color: #ffffff; font-weight: bold;
    }}

    /* Sales Overview */
    .sales-overview {{
        font-family: 'Poppins', sans-serif; font-size: 1.5rem; color: #ffffff;
        text-align: center; margin: 0.5rem 0; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }}

    /* Homepage */
    .homepage {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 100vh;
        background-image: url('data:image/png;base64,{background_image}');
        background-size: cover; background-position: center; background-repeat: no-repeat;
    }}
    .homepage-title {{
        font-family: 'Poppins', sans-serif; font-size: min(2rem, 7vw); color: #ffffff;
        margin-bottom: 1rem; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }}
    .dashboard-button {{
        background: linear-gradient(45deg, #ff4500, #ff8c00);
        color: white; font-family: 'Poppins', sans-serif;
        font-size: 0.9rem; padding: 0.5rem 1rem; border: none; border-radius: 20px;
        cursor: pointer; text-decoration: none; transition: transform 0.2s;
    }}
    .dashboard-button:hover {{ transform: scale(1.05); background: linear-gradient(45deg, #e63900, #e07b00); }}

    /* Responsive adjustments */
    @media (max-width: 600px) {{
        .block-container {{ padding: 0.3rem; }}
        .css-1d391kg {{ width: 100%; }}
        .stButton>button, .dashboard-button {{ font-size: 0.8rem; padding: 0.3rem 0.6rem; }}
        .metric-card {{ margin: 0.05rem; width: 80px; height: 80px; }}
        .metric-label {{ font-size: 0.5rem; }}
        .metric-value {{ font-size: 0.8rem; }}
        .st-subheader {{ font-size: 0.9rem; }}
        .sales-overview {{ font-size: 1.3rem; }}
    }}
    </style>
"""

st.markdown(background_css, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Customer Segmentation'
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'last_data_gen' not in st.session_state:
    st.session_state.last_data_gen = time.time()
if 'last_metrics_update' not in st.session_state:
    st.session_state.last_metrics_update = time.time()

# Fallback dataset generator
def generate_fallback_dataset(n_rows=100, append=False):
    st.write(f"Debug: {'Appending' if append else 'Generating'} {'1-3 rows' if append else f'{n_rows} rows'}")
    np.random.seed(int(time.time()))  # Dynamic seed for varied data
    if append:
        n_rows = np.random.randint(1, 4)  # 1-3 new rows
        current_time = datetime.now()
        start_date = current_time - timedelta(minutes=n_rows)
        timestamps = [start_date + timedelta(minutes=i) for i in range(n_rows)]
    else:
        start_date = datetime(2025, 1, 1)
        timestamps = [start_date + timedelta(minutes=i) for i in range(n_rows)]

    data = {
        'timestamp': timestamps,
        'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Canada'], n_rows),
        'service_type': np.random.choice(['Consulting', 'Support', 'Development'], n_rows),
        'demo_request': np.random.choice([0, 1], n_rows),
        'request_category': np.random.choice(['Email', 'Social Media', 'Search', 'Referral'], n_rows),
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], n_rows),
        'transaction_id': [f'txn_{int(time.time())}_{i:03d}' for i in range(n_rows)],
        'revenue': np.random.uniform(100, 1000, n_rows),
        'product_id': [f'prod_{i%10:03d}' for i in range(n_rows)],
        'session_duration': np.random.uniform(60, 600, n_rows),
        'customer_behavior': np.random.choice(['Browsing', 'Engaged', 'Converted', 'Abandoned'], n_rows),
        'subscription_status': np.random.choice(['Active', 'Inactive', 'Trial'], n_rows),
        'types_of_jobs_requested': np.random.choice(['Analysis,Reporting', 'Development,Testing', 'Support'], n_rows)
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if append and not st.session_state.data.empty:
        st.session_state.data = pd.concat([st.session_state.data, df], ignore_index=True)
    else:
        st.session_state.data = df

    try:
        st.session_state.data.to_csv('sales_log_dataset.csv', index=False)
        st.write("Debug: Dataset saved to sales_log_dataset.csv")
    except Exception as e:
        st.warning(f"Debug: Failed to save dataset: {e}")

    return st.session_state.data

# File watcher for auto-refresh
class CSVHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
    def on_modified(self, event):
        if event.src_path.endswith('sales_log_dataset.csv'):
            st.write(f"Debug: CSV modified at {event.src_path}")
            self.callback()

# Load data from file or uploaded CSV
def load_data(file_hash, uploaded_file=None):
    required_fields = ['timestamp', 'country', 'service_type', 'demo_request', 'request_category', 'browser', 'transaction_id', 'revenue', 'product_id', 'session_duration', 'customer_behavior', 'subscription_status', 'types_of_jobs_requested']
    csv_path = 'sales_log_dataset.csv'

    if uploaded_file is not None:
        st.write(f"Debug: Loading uploaded CSV")
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Debug: Uploaded CSV loaded with {len(df)} rows, columns: {list(df.columns)}")
            if df.empty:
                return pd.DataFrame(), "Uploaded CSV is empty"
            available_fields = [f for f in required_fields if f in df.columns]
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                return pd.DataFrame(), f"Missing required fields in uploaded CSV: {missing_fields}"
            df = df[available_fields]
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isna().any():
                return pd.DataFrame(), "Invalid timestamp format in uploaded CSV. Expected: %Y-%m-%d %H:%M:%S"
            st.write(f"Debug: Uploaded CSV validated successfully")
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"Error loading uploaded CSV: {str(e)}"

    st.write(f"Debug: Checking CSV at {os.path.abspath(csv_path)}")
    if not os.path.exists(csv_path):
        return pd.DataFrame(), f"CSV file not found at {os.path.abspath(csv_path)}"
    try:
        df = pd.read_csv(csv_path)
        st.write(f"Debug: CSV loaded with {len(df)} rows, columns: {list(df.columns)}")
        if df.empty:
            return pd.DataFrame(), "CSV file is empty"
        available_fields = [f for f in required_fields if f in df.columns]
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            return pd.DataFrame(), f"Missing fields in CSV: {missing_fields}"
        df = df[available_fields]
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            return pd.DataFrame(), "Invalid timestamp format in CSV. Expected: %Y-%m-%d %H:%M:%S"
        st.write(f"Debug: CSV validated successfully")
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error loading CSV: {str(e)}"

# Update data
def update_data(uploaded_file=None):
    csv_path = 'sales_log_dataset.csv'
    file_hash = None
    if uploaded_file is None and os.path.exists(csv_path):
        with open(csv_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
    st.write(f"Debug: Updating data, file_hash={file_hash}, previous_hash={st.session_state.file_hash}")
    if file_hash != st.session_state.file_hash or uploaded_file is not None:
        st.session_state.file_hash = file_hash
        df, error = load_data(file_hash, uploaded_file)
        if error:
            st.error(error)
            st.session_state.data = pd.DataFrame()
        else:
            st.session_state.data = df
            st.write(f"Debug: Data updated with {len(df)} rows")
    else:
        st.write("Debug: No data update needed (hash unchanged)")

# Setup file watcher
if 'observer' not in st.session_state:
    st.session_state.observer = None
if not st.session_state.observer and os.path.exists('sales_log_dataset.csv'):
    observer = Observer()
    observer.schedule(CSVHandler(lambda: update_data()), path='sales_log_dataset.csv', recursive=False)
    observer.start()
    st.session_state.observer = observer
    st.write("Debug: File watcher started")

# Function to display metrics
def display_metrics(filtered_df):
    if filtered_df.empty:
        st.warning("Debug: No data available for metrics")
        return

    # Check for metrics update every 5 seconds
    if time.time() - st.session_state.last_metrics_update >= 5:
        st.write("Debug: Triggering metrics refresh with new data")
        # Append new rows
        generate_fallback_dataset(append=True)
        st.session_state.last_metrics_update = time.time()
        filtered_df = st.session_state.data.copy()
        if 'start_date' in st.session_state and 'end_date' in st.session_state:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= st.session_state.start_date) &
                (filtered_df['timestamp'].dt.date <= st.session_state.end_date)
            ]
        if 'location' in st.session_state and st.session_state.location != 'All Locations':
            filtered_df = filtered_df[filtered_df['country'] == st.session_state.location]
        st.rerun()

    st.markdown('<div class="sales-overview">Sales Overview</div>', unsafe_allow_html=True)

    # Metrics with uniform spacing
    cols = st.columns(4, gap="small")
    with cols[0]:
        total_revenue = filtered_df['revenue'].sum()
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL REVENUE</div>
                <div class="metric-value">${total_revenue:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        profit = total_revenue * 0.7
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PROFIT</div>
                <div class="metric-value">${profit:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        avg_target = filtered_df['revenue'].mean() if not filtered_df.empty else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AVG TARGET</div>
                <div class="metric-value">${avg_target:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        avg_session = filtered_df['session_duration'].mean() / 60 if not filtered_df.empty else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AVG SESSION (min)</div>
                <div class="metric-value">{avg_session:.1f}</div>
            </div>
        """, unsafe_allow_html=True)

    cols2 = st.columns(2, gap="small")
    with cols2[0]:
        conversion_rate = (filtered_df[filtered_df['revenue'] > 0]['product_id'].nunique() / filtered_df['product_id'].nunique() * 100) if filtered_df['product_id'].nunique() > 0 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CONVERSION RATE</div>
                <div class="metric-value">{conversion_rate:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with cols2[1]:
        top_service = filtered_df['service_type'].mode()[0] if not filtered_df['service_type'].empty else "N/A"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOP SERVICES</div>
                <div class="metric-value">{top_service}</div>
            </div>
        """, unsafe_allow_html=True)

# Homepage
def show_homepage():
    st.markdown("""
        <div class="homepage">
            <h1 class="homepage-title">Web Analytics</h1>
            <a href="?page=dashboard" class="dashboard-button">Go to Dashboard</a>
        </div>
    """, unsafe_allow_html=True)
    if st.query_params.get('page') == 'dashboard':
        st.session_state.page = 'dashboard'
        st.rerun()

# Dashboard
def show_dashboard():
    # Title
    st.markdown("""
        <div class="dashboard-title-container">
            <h1 class="dashboard-title">Web Analytics Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button("CUSTOMER SEGMENTATION", key="nav_segmentation"):
            st.session_state.page = 'charts'
            st.session_state.active_tab = 'CUSTOMER SEGMENTATION'
            st.rerun()
    with cols[1]:
        if st.button("CONSUMER ACTIONS", key="nav_actions"):
            st.session_state.page = 'charts'
            st.session_state.active_tab = 'CONSUMER ACTIONS'
            st.rerun()
    with cols[2]:
        if st.button("PRODUCT PERFORMANCE", key="nav_performance"):
            st.session_state.page = 'charts'
            st.session_state.active_tab = 'PRODUCT PERFORMANCE'
            st.rerun()
    with cols[3]:
        if st.button("MARKETING & JOBS", key="nav_marketing"):
            st.session_state.page = 'charts'
            st.session_state.active_tab = 'MARKETING & JOBS'
            st.rerun()

    # Attempt to load or generate data
    if st.session_state.data.empty:
        csv_path = 'sales_log_dataset.csv'
        st.write(f"Debug: Initial data load attempt, CSV exists: {os.path.exists(csv_path)}")
        if not os.path.exists(csv_path):
            st.write("Debug: CSV missing, generating fallback dataset")
            st.session_state.data = generate_fallback_dataset()
        else:
            update_data()

    df = st.session_state.data
    if not df.empty:
        filtered_df = df.copy()

        # Sidebar filters and actions
        with st.sidebar:
            st.subheader("AI INSIGHTS")
            if st.button("GENERATE INSIGHTS", key="generate_insights"):
                high_traffic = filtered_df['country'].value_counts().index[0]
                top_channel = filtered_df['request_category'].value_counts().index[0]
                avg_revenue = filtered_df['revenue'].mean()
                behavior_gender_map = {
                    'Browsing': 'Male', 'Engaged': 'Female', 'Converted': 'Male',
                    'Abandoned': 'Female'
                }
                filtered_df['inferred_gender'] = filtered_df['customer_behavior'].map(behavior_gender_map).fillna('Unknown')
                gender_revenue = filtered_df.groupby('inferred_gender')['revenue'].sum()
                top_gender = gender_revenue.idxmax() if not gender_revenue.empty else "N/A"
                sub_status_counts = filtered_df['subscription_status'].value_counts()
                top_sub_status = sub_status_counts.idxmax() if not sub_status_counts.empty else "N/A"

                # Unusual customer behavior
                behavior_counts = filtered_df['customer_behavior'].value_counts(normalize=True)
                abandon_rate = behavior_counts.get('Abandoned', 0) * 100
                session_duration_mean = filtered_df['session_duration'].mean() / 60
                session_duration_std = filtered_df['session_duration'].std() / 60
                high_session_threshold = session_duration_mean + 2 * session_duration_std
                unusual_sessions = len(filtered_df[filtered_df['session_duration'] / 60 > high_session_threshold])

                st.write(f"**High-Traffic Region**: {high_traffic}")
                st.write(f"**Top Converting Channel**: {top_channel}")
                st.write(f"**Average Revenue per Transaction**: ${avg_revenue:,.2f}")
                st.write(f"**Top Gender by Revenue**: {top_gender} (${gender_revenue.get(top_gender, 0):,.2f})")
                st.write(f"**Top Subscription Status**: {top_sub_status} ({sub_status_counts.get(top_sub_status, 0)} users)")
                st.write("**Unusual Customer Behavior**:")
                if abandon_rate > 30:
                    st.write(f"- High abandonment rate: {abandon_rate:.1f}% of sessions are abandoned (above typical 30% threshold).")
                else:
                    st.write(f"- Abandonment rate: {abandon_rate:.1f}% (within typical range).")
                st.write(f"- Unusual session durations: {unusual_sessions} sessions exceed {high_session_threshold:.1f} minutes (2 std above mean).")

            with st.expander("FILTERS", expanded=True):
                min_date = df['timestamp'].min().date() if not df.empty else pd.to_datetime('2025-01-01').date()
                max_date = df['timestamp'].max().date() if not df.empty else pd.to_datetime('2025-12-31').date()
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                locations = ['All Locations'] + (sorted(df['country'].unique().tolist()) if not df.empty else [])
                location = st.selectbox("Location", locations)
                if st.button("APPLY FILTERS", key="apply_filters"):
                    filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
                    if location != 'All Locations':
                        filtered_df = filtered_df[filtered_df['country'] == location]
                    st.session_state.start_date = start_date
                    st.session_state.end_date = end_date
                    st.session_state.location = location

                st.subheader("DATA ACTIONS")
                col1, col2 = st.columns([1, 1])
                with col1:
                    full_csv_path = 'sales_log_dataset.csv'
                    if os.path.exists(full_csv_path):
                        with open(full_csv_path, 'rb') as f:
                            st.download_button(
                                label="Download Full CSV",
                                data=f,
                                file_name="sales_data_full.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                    else:
                        st.warning("Full CSV not available for download.")
                with col2:
                    if st.button("Generate Dataset", key="generate_dataset"):
                        st.session_state.data = generate_fallback_dataset()
                        update_data()

                uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], key="upload_csv")
                if uploaded_file is not None:
                    update_data(uploaded_file=uploaded_file)

                if st.button("Back to Home", key="back_to_home"):
                    st.session_state.page = 'home'
                    st.rerun()

        # Display metrics
        display_metrics(filtered_df)

        # Data Sample
        st.write("**Data Loaded Successfully**")
        st.subheader("Data Sample")
        st.write(df.tail(5))  # Show latest rows to reflect updates
    else:
        st.error("No data loaded. Generating fallback dataset.")
        st.session_state.data = generate_fallback_dataset()
        st.rerun()

# Charts Page
def show_charts():
    st.markdown("""
        <div class="dashboard-title-container">
            <h1 class="dashboard-title">ANALYTICS CHARTS</h1>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Back to Dashboard", key="back_to_dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()

    df = st.session_state.data
    if df.empty:
        st.error("No data loaded. Generating fallback dataset.")
        st.session_state.data = generate_fallback_dataset()
        st.rerun()
        return

    filtered_df = df.copy()
    if 'start_date' in st.session_state and 'end_date' in st.session_state:
        filtered_df = df[(df['timestamp'].dt.date >= st.session_state.start_date) & (df['timestamp'].dt.date <= st.session_state.end_date)]
    if 'location' in st.session_state and st.session_state.location != 'All Locations':
        filtered_df = filtered_df[filtered_df['country'] == st.session_state.location]

    # Add inferred gender
    behavior_gender_map = {
        'Browsing': 'Male', 'Engaged': 'Female', 'Converted': 'Male',
        'Abandoned': 'Female'
    }
    filtered_df['inferred_gender'] = filtered_df['customer_behavior'].map(behavior_gender_map).fillna('Unknown')

    # Display metrics
    display_metrics(filtered_df)

    tab1, tab2, tab3, tab4 = st.tabs(["CUSTOMER SEGMENTATION", "CONSUMER ACTIONS", "PRODUCT PERFORMANCE", "MARKETING & JOBS"])

    with tab1:
        # Row 1: Two charts
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            country_counts = filtered_df['country'].value_counts().head(10).reset_index()
            country_counts.columns = ['Country', 'Visits']
            st.write(f"Top Countries: {', '.join(country_counts['Country'].tolist())}")
            fig = px.area(country_counts, x='Country', y='Visits', title="")
            fig.update_layout(
                xaxis_title="Country", yaxis_title="Visits", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            gender_counts = filtered_df['inferred_gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            st.write(f"Genders: {', '.join(gender_counts['Gender'].tolist())}")
            fig = px.violin(gender_counts, y='Count', x='Gender', title="",
                            color='Gender', color_discrete_sequence=px.colors.sequential.Hot)
            fig.update_layout(
                height=300, width=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12), margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        # Row 2: Two charts
        col3, col4 = st.columns([1, 1], gap="medium")
        with col3:
            fig = px.box(filtered_df, y='session_duration', title="")
            fig.update_layout(
                yaxis_title="Session Duration (seconds)", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col4:
            trends_df = filtered_df.groupby(filtered_df['timestamp'].dt.date)['product_id'].nunique().reset_index()
            trends_df.columns = ['Date', 'Unique Visitors']
            st.write(f"Unique Visitors Data: {trends_df.shape[0]} rows")
            fig = px.scatter(trends_df, x='Date', y='Unique Visitors', title="")
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Visitors", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

    with tab2:
        # Row 1: Two charts
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            channel_counts = filtered_df['request_category'].value_counts().head(10).reset_index()
            channel_counts.columns = ['Channel', 'Count']
            st.write(f"Activity Channels: {', '.join(channel_counts['Channel'].tolist())}")
            fig = px.bar(channel_counts, x='Channel', y='Count', title="",
                         color="Channel", color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                xaxis_title="Channel", yaxis_title="Count", height=300, width=400, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            behavior_counts = filtered_df['customer_behavior'].value_counts().head(10).reset_index()
            behavior_counts.columns = ['Behavior', 'Count']
            st.write(f"Behaviors: {', '.join(behavior_counts['Behavior'].tolist())}")
            fig = px.pie(behavior_counts, names='Behavior', values='Count', title="")
            fig.update_layout(
                height=300, width=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12), margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_traces(textfont=dict(size=12))
            st.plotly_chart(fig, use_container_width=False)

        # Row 2: Two charts
        col3, col4 = st.columns([1, 1], gap="medium")
        with col3:
            demo_df = filtered_df.groupby(filtered_df['timestamp'].dt.date)['demo_request'].sum().reset_index()
            demo_df.columns = ['Date', 'Demo Requests']
            st.write(f"Demo Requests Data: {demo_df.shape[0]} rows")
            fig = px.area(demo_df, x='Date', y='Demo Requests', title="")
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Demo Requests", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col4:
            browser_counts = filtered_df['browser'].value_counts().head(10).reset_index()
            browser_counts.columns = ['Browser', 'Count']
            st.write(f"Browsers: {', '.join(browser_counts['Browser'].tolist())}")
            fig = px.scatter(browser_counts, x='Browser', y='Count', size='Count', title="")
            fig.update_layout(
                xaxis_title="Browser", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

    with tab3:
        # Row 1: Two charts
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            revenue_df = filtered_df.groupby(filtered_df['timestamp'].dt.date)['revenue'].sum().reset_index()
            revenue_df.columns = ['Date', 'Revenue']
            st.write(f"Revenue Data: {revenue_df.shape[0]} rows")
            fig = px.line(revenue_df, x='Date', y='Revenue', title="")
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Revenue ($)", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            revenue_country = filtered_df.groupby('country')['revenue'].sum().sort_values(ascending=False).head(10).reset_index()
            revenue_country.columns = ['Country', 'Revenue']
            st.write(f"Revenue Countries: {', '.join(revenue_country['Country'].tolist())}")
            fig = px.bar(revenue_country, x='Country', y='Revenue', title="",
                         color="Country", color_discrete_sequence=px.colors.sequential.Hot)
            fig.update_layout(
                xaxis_title="Country", yaxis_title="Revenue ($)", height=300, width=400, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        # Row 2: Two charts
        col3, col4 = st.columns([1, 1], gap="medium")
        with col3:
            service_counts = filtered_df['service_type'].value_counts().head(10).reset_index()
            service_counts.columns = ['Service Type', 'Count']
            st.write(f"Service Types: {', '.join(service_counts['Service Type'].tolist())}")
            fig = px.violin(service_counts, x='Service Type', y='Count', title="",
                            color='Service Type', color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                xaxis_title="Service Type", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col4:
            subscription_counts = filtered_df['subscription_status'].value_counts().head(10).reset_index()
            subscription_counts.columns = ['Subscription Status', 'Count']
            st.write(f"Subscription Statuses: {', '.join(subscription_counts['Subscription Status'].tolist())}")
            fig = px.box(subscription_counts, x='Subscription Status', y='Count', title="",
                         color='Subscription Status', color_discrete_sequence=px.colors.sequential.Hot)
            fig.update_layout(
                xaxis_title="Subscription Status", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

    with tab4:
        # Row 1: Three charts
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        with col1:
            channel_revenue = filtered_df.groupby('request_category')['revenue'].sum().reset_index()
            channel_revenue = channel_revenue.sort_values('revenue', ascending=False).head(10)
            st.write(f"Top Channels: {', '.join(channel_revenue['request_category'].tolist())}")
            fig = px.bar(channel_revenue, x='request_category', y='revenue', title="",
                         color='request_category', color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                xaxis_title="Channel", yaxis_title="Revenue ($)", height=300, width=400, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col2:
            job_types = filtered_df['types_of_jobs_requested'].dropna().str.split(',', expand=True).stack().str.strip()
            job_counts = job_types.value_counts().head(10).reset_index()
            job_counts.columns = ['Job Type', 'Count']
            job_types_list = job_counts['Job Type'].tolist() if not job_counts.empty else ['None']
            st.write(f"Job Types: {', '.join(job_types_list)}")
            fig = px.scatter(job_counts, x='Job Type', y='Count', size='Count', title="")
            fig.update_layout(
                xaxis_title="Job Type", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col3:
            demo_revenue = filtered_df.groupby('demo_request')['revenue'].sum().reset_index()
            demo_revenue['demo_request'] = demo_revenue['demo_request'].map({0: 'No Demo', 1: 'Demo Requested'})
            st.write(f"Demo Requests: {demo_revenue.to_dict('records')}")
            fig = px.bar(demo_revenue, x='demo_request', y='revenue', title="",
                         color='demo_request', color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                xaxis_title="Demo Request", yaxis_title="Revenue ($)", height=300, width=400, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        # Row 2: Three charts
        col4, col5, col6 = st.columns([1, 1, 1], gap="medium")
        with col4:
            channel_time = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'request_category']).size().reset_index(name='Count')
            channel_time = channel_time[channel_time['request_category'].isin(channel_time['request_category'].value_counts().head(5).index)]
            st.write(f"Top Channels: {', '.join(channel_time['request_category'].unique().tolist())}")
            fig = px.line(channel_time, x='timestamp', y='Count', color='request_category', title="")
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col5:
            service_country = filtered_df.groupby(['country', 'service_type']).size().reset_index(name='Count')
            service_country = service_country[service_country['country'].isin(filtered_df['country'].value_counts().head(5).index)]
            st.write(f"Top Countries: {', '.join(service_country['country'].unique().tolist())}")
            fig = px.bar(service_country, x='country', y='Count', color='service_type', title="")
            fig.update_layout(
                xaxis_title="Country", yaxis_title="Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col6:
            product_sales = filtered_df.groupby('product_id').size().reset_index(name='Count')
            product_sales = product_sales.sort_values('Count', ascending=False).head(10)
            st.write(f"Top Products: {', '.join(product_sales['product_id'].astype(str).tolist())}")
            fig = px.scatter(product_sales, x='product_id', y='Count', size='Count', title="")
            fig.update_layout(
                xaxis_title="Product ID", yaxis_title="Sales Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        # Row 3: Two charts
        col7, col8 = st.columns([1, 1], gap="medium")
        with col7:
            trans_sub = filtered_df.groupby('subscription_status')['transaction_id'].count().reset_index()
            trans_sub.columns = ['Subscription Status', 'Count']
            st.write(f"Statuses: {', '.join(trans_sub['Subscription Status'].tolist())}")
            fig = px.bar(trans_sub, x='Subscription Status', y='Count', title="",
                         color='Subscription Status', color_discrete_sequence=px.colors.sequential.YlOrRd)
            fig.update_layout(
                xaxis_title="Subscription Status", yaxis_title="Transaction Count", height=300, width=400,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)

        with col8:
            product_counts = filtered_df['product_id'].value_counts().head(10).reset_index()
            product_counts.columns = ['Product ID', 'Count']
            st.write(f"Top Products: {', '.join(product_counts['Product ID'].astype(str).tolist())}")
            fig = px.bar(product_counts, x='Product ID', y='Count', title="",
                         color='Product ID', color_discrete_sequence=px.colors.sequential.Hot)
            fig.update_layout(
                xaxis_title="Product ID", yaxis_title="Count", height=300, width=400, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(size=12),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(tickfont=dict(size=12), title_font=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=12), title_font=dict(size=14))
            st.plotly_chart(fig, use_container_width=False)


# Render page
if st.session_state.page == 'home':
    show_homepage()
elif st.session_state.page == 'charts':
    show_charts()
else:
    show_dashboard()