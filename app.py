import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Trading Comparator Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Initialize session state for synthetic data
if 'synthetic_demo' not in st.session_state:
    st.session_state.synthetic_demo = None
if 'synthetic_live' not in st.session_state:
    st.session_state.synthetic_live = None


# Theme toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


# Apply theme
if st.session_state.dark_mode:
    theme_colors = {
        'bg_color': '#1a1a1a',
        'text_color': '#ffffff',
        'metric_bg': '#2d2d2d',
        'border_color': '#4a4a4a',
        'accent': '#bb86fc'
    }
else:
    theme_colors = {
        'bg_color': '#ffffff',
        'text_color': '#1e3a8a',
        'metric_bg': '#f0f2f6',
        'border_color': '#667eea',
        'accent': '#667eea'
    }

# Custom CSS with theme support
st.markdown(f"""
    <style>
    .main {{
        padding: 0rem 1rem;
        background-color: {theme_colors['bg_color']};
    }}
    .stMetric {{
        background-color: {theme_colors['metric_bg']};
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .metric-container {{
        background: linear-gradient(135deg, {theme_colors['accent']} 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }}
    h1 {{
        color: {theme_colors['text_color']};
        font-weight: 700;
        margin-bottom: 30px;
    }}
    h2 {{
        color: {theme_colors['text_color']};
        border-bottom: 2px solid {theme_colors['border_color']};
        padding-bottom: 10px;
        margin-top: 30px;
    }}
    h3 {{
        color: {theme_colors['text_color']};
        margin-top: 20px;
    }}
    .alert-box {{
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .alert-high {{
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
    }}
    .alert-medium {{
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }}
    .alert-low {{
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
    }}
    .match-success {{
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
    }}
    </style>
    """, unsafe_allow_html=True)

# Main title with theme toggle
col_title, col_theme = st.columns([10, 1])
with col_title:
    st.markdown("# 📊 Trading Comparator Pro")
    st.markdown("### Advanced analysis with anomaly detection and order matching")
with col_theme:
    st.button("🌓", on_click=toggle_theme, help="Toggle Dark/Light Mode")

st.markdown("---")


# Function to generate synthetic data with noise
def generate_noisy_data(original_df, noise_level=0.05, order_variation=0.2, modify_order_count=True):
    """
    Generate a synthetic dataset based on original with added noise

    Parameters:
    - noise_level: Amount of price noise (0.05 = 5%)
    - order_variation: Percentage of orders to add/remove if modify_order_count is True
    - modify_order_count: If True, add/remove orders. If False, keep same number of orders
    """
    noisy_df = original_df.copy()

    # Add very small price noise - only to SOME prices, not all
    if 'Price' in noisy_df.columns:
        # Only modify 30% of prices to keep most data identical
        n_prices_to_modify = int(len(noisy_df) * 0.3)
        if n_prices_to_modify > 0:
            indices_to_modify = np.random.choice(noisy_df.index, size=n_prices_to_modify, replace=False)
            for idx in indices_to_modify:
                # Very small variation using uniform distribution for more control
                # noise_level is the maximum variation, most will be smaller
                variation = np.random.uniform(-noise_level, noise_level)
                noisy_df.loc[idx, 'Price'] = noisy_df.loc[idx, 'Price'] * (1 + variation)

    # Keep quantities almost completely identical
    if 'Quantity' in noisy_df.columns:
        # Only modify 5% of quantities
        n_qty_to_modify = int(len(noisy_df) * 0.05)
        if n_qty_to_modify > 0:
            indices_to_modify = np.random.choice(noisy_df.index, size=n_qty_to_modify, replace=False)
            for idx in indices_to_modify:
                # Add or subtract 1-2 units only, no percentage changes
                qty_change = np.random.choice([-2, -1, 1, 2])
                new_qty = noisy_df.loc[idx, 'Quantity'] + qty_change
                noisy_df.loc[idx, 'Quantity'] = max(1, new_qty)

    # Recalculate Value to be consistent
    if 'Value' in noisy_df.columns and 'Price' in noisy_df.columns and 'Quantity' in noisy_df.columns:
        noisy_df['Value'] = noisy_df['Price'] * noisy_df['Quantity']

    # Add small time variations (only to some orders)
    if 'Time' in noisy_df.columns:
        # Only shift 20% of times
        n_times_to_shift = int(len(noisy_df) * 0.2)
        if n_times_to_shift > 0:
            indices_to_shift = np.random.choice(noisy_df.index, size=n_times_to_shift, replace=False)
            for idx in indices_to_shift:
                # Small shift: -5 to +5 minutes
                time_shift = np.random.randint(-5, 6)
                noisy_df.loc[idx, 'Time'] = noisy_df.loc[idx, 'Time'] + pd.Timedelta(minutes=time_shift)

    # Change very few order statuses
    if 'Status' in noisy_df.columns:
        n_status_changes = max(1, int(len(noisy_df) * 0.02))  # Only 2% of statuses
        if n_status_changes > 0:
            indices_to_change = np.random.choice(noisy_df.index, size=n_status_changes, replace=False)
            statuses = ['Filled', 'Partial', 'Cancelled']
            for idx in indices_to_change:
                current_status = noisy_df.loc[idx, 'Status']
                new_statuses = [s for s in statuses if s != current_status]
                if new_statuses:
                    noisy_df.loc[idx, 'Status'] = np.random.choice(new_statuses)

    # Only modify order count if requested
    if modify_order_count and order_variation > 0:
        # Remove some orders
        n_to_remove = int(len(noisy_df) * order_variation * 0.5)
        if n_to_remove > 0 and len(noisy_df) > n_to_remove + 10:
            indices_to_remove = np.random.choice(noisy_df.index, size=n_to_remove, replace=False)
            noisy_df = noisy_df.drop(indices_to_remove)

        # Add some new orders - but keep them very similar to existing ones
        n_to_add = int(len(original_df) * order_variation * 0.5)
        if n_to_add > 0:
            # Limit the number of orders to add to avoid volume explosion
            n_to_add = min(n_to_add, max(1, len(noisy_df) // 50))  # Max 2% more orders

            # Select random orders to duplicate
            indices_to_dup = np.random.choice(noisy_df.index, size=n_to_add, replace=True)
            new_orders = noisy_df.loc[indices_to_dup].copy()

            if len(new_orders) > 0:
                # Very minimal modifications to duplicated orders
                for idx in new_orders.index:
                    # Tiny price variation: max ±1% of the noise level
                    tiny_variation = np.random.uniform(-noise_level * 0.3, noise_level * 0.3)
                    new_orders.loc[idx, 'Price'] = new_orders.loc[idx, 'Price'] * (1 + tiny_variation)

                    # Keep quantity mostly the same (90% unchanged)
                    if np.random.random() < 0.1:
                        new_orders.loc[idx, 'Quantity'] = max(1, new_orders.loc[idx, 'Quantity'] + np.random.choice(
                            [-1, 0, 1]))

                    # Shift time
                    new_orders.loc[idx, 'Time'] = new_orders.loc[idx, 'Time'] + pd.Timedelta(
                        minutes=np.random.randint(10, 60))

                # Recalculate value
                new_orders['Value'] = new_orders['Price'] * new_orders['Quantity']

                # Combine with original
                noisy_df = pd.concat([noisy_df, new_orders], ignore_index=True)

    # Sort by time and reset index
    noisy_df = noisy_df.sort_values('Time').reset_index(drop=True)

    # Recalculate derived columns
    if 'Time' in noisy_df.columns:
        noisy_df['Hour'] = noisy_df['Time'].dt.hour
        noisy_df['DayOfWeek'] = noisy_df['Time'].dt.day_name()
        noisy_df['Month'] = noisy_df['Time'].dt.month_name()
        noisy_df['Date'] = noisy_df['Time'].dt.date

    return noisy_df


# Function to load and prepare data
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        df['Time'] = pd.to_datetime(df['Time'])
        df['Hour'] = df['Time'].dt.hour
        df['DayOfWeek'] = df['Time'].dt.day_name()
        df['Month'] = df['Time'].dt.month_name()
        df['Date'] = df['Time'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Order Matching Engine
def match_orders(demo_df, live_df, time_window_minutes=5, price_threshold=0.02):
    """Match orders between demo and live accounts"""
    matches = []
    unmatched_demo = []
    unmatched_live = []

    demo_used = set()
    live_used = set()

    for idx_d, demo_order in demo_df.iterrows():
        matched = False
        for idx_l, live_order in live_df.iterrows():
            if idx_l in live_used:
                continue

            # Check if orders match (same symbol, similar time, similar price)
            time_diff = abs((demo_order['Time'] - live_order['Time']).total_seconds() / 60)
            price_diff = abs(demo_order['Price'] - live_order['Price']) / demo_order['Price']

            if (demo_order['Symbol'] == live_order['Symbol'] and
                    demo_order['Type'] == live_order['Type'] and
                    time_diff <= time_window_minutes and
                    price_diff <= price_threshold):
                matches.append({
                    'Symbol': demo_order['Symbol'],
                    'Type': demo_order['Type'],
                    'Demo_Time': demo_order['Time'],
                    'Live_Time': live_order['Time'],
                    'Demo_Price': demo_order['Price'],
                    'Live_Price': live_order['Price'],
                    'Demo_Quantity': demo_order['Quantity'],
                    'Live_Quantity': live_order['Quantity'],
                    'Time_Diff_Min': time_diff,
                    'Price_Diff_%': price_diff * 100,
                    'Slippage': live_order['Price'] - demo_order['Price']
                })
                demo_used.add(idx_d)
                live_used.add(idx_l)
                matched = True
                break

        if not matched:
            unmatched_demo.append(demo_order)

    for idx_l, live_order in live_df.iterrows():
        if idx_l not in live_used:
            unmatched_live.append(live_order)

    return pd.DataFrame(matches), pd.DataFrame(unmatched_demo), pd.DataFrame(unmatched_live)


# Anomaly Detection Functions
def detect_anomalies(demo_df, live_df):
    """Detect various anomalies between demo and live trading"""
    anomalies = []

    # 1. Price difference anomalies
    common_symbols = set(demo_df['Symbol']) & set(live_df['Symbol'])
    for symbol in common_symbols:
        demo_prices = demo_df[demo_df['Symbol'] == symbol]['Price']
        live_prices = live_df[live_df['Symbol'] == symbol]['Price']

        if len(demo_prices) > 0 and len(live_prices) > 0:
            demo_avg = demo_prices.mean()
            live_avg = live_prices.mean()
            price_diff = abs(demo_avg - live_avg) / demo_avg * 100

            if price_diff > 5:  # More than 5% difference
                anomalies.append({
                    'Type': 'Price Divergence',
                    'Severity': 'High' if price_diff > 10 else 'Medium',
                    'Symbol': symbol,
                    'Details': f"Avg price difference: {price_diff:.2f}%",
                    'Demo_Avg': demo_avg,
                    'Live_Avg': live_avg
                })

    # 2. Volume anomalies
    demo_volume = demo_df.groupby('Symbol')['Value'].sum()
    live_volume = live_df.groupby('Symbol')['Value'].sum()

    for symbol in common_symbols:
        if symbol in demo_volume.index and symbol in live_volume.index:
            vol_diff = abs(demo_volume[symbol] - live_volume[symbol]) / demo_volume[symbol] * 100
            if vol_diff > 50:  # More than 50% volume difference
                anomalies.append({
                    'Type': 'Volume Anomaly',
                    'Severity': 'High' if vol_diff > 100 else 'Medium',
                    'Symbol': symbol,
                    'Details': f"Volume difference: {vol_diff:.2f}%",
                    'Demo_Volume': demo_volume[symbol],
                    'Live_Volume': live_volume[symbol]
                })

    # 3. Trading pattern anomalies
    demo_hourly = demo_df.groupby('Hour').size()
    live_hourly = live_df.groupby('Hour').size()

    for hour in range(24):
        demo_count = demo_hourly.get(hour, 0)
        live_count = live_hourly.get(hour, 0)
        if demo_count > 0:
            pattern_diff = abs(demo_count - live_count) / demo_count * 100
            if pattern_diff > 75:
                anomalies.append({
                    'Type': 'Pattern Anomaly',
                    'Severity': 'Low',
                    'Symbol': f"Hour {hour}:00",
                    'Details': f"Order count difference: {pattern_diff:.2f}%",
                    'Demo_Count': demo_count,
                    'Live_Count': live_count
                })

    # 4. Execution anomalies
    demo_fill_rate = (demo_df['Status'] == 'Filled').mean() * 100
    live_fill_rate = (live_df['Status'] == 'Filled').mean() * 100
    fill_diff = abs(demo_fill_rate - live_fill_rate)

    if fill_diff > 10:
        anomalies.append({
            'Type': 'Execution Anomaly',
            'Severity': 'High' if fill_diff > 20 else 'Medium',
            'Symbol': 'Overall',
            'Details': f"Fill rate difference: {fill_diff:.2f}%",
            'Demo_Rate': demo_fill_rate,
            'Live_Rate': live_fill_rate
        })

    return pd.DataFrame(anomalies)


# Sidebar with enhanced filters
with st.sidebar:
    st.markdown("## 📁 File Upload")
    st.markdown("---")

    demo_file = st.file_uploader(
        "📈 **DEMO Account File**",
        type=['csv'],
        help="Select the CSV file with demo account orders"
    )

    live_file = st.file_uploader(
        "💼 **LIVE Account File**",
        type=['csv'],
        help="Select the CSV file with live account orders"
    )

    # Test data generation option
    st.markdown("---")
    st.markdown("### 🧪 Test Data Generator")

    # Check if only one file is uploaded
    if (demo_file and not live_file) or (live_file and not demo_file):
        st.info("📌 Only one file uploaded. You can generate synthetic data for comparison.")

        if st.checkbox("Generate synthetic comparison data"):
            noise_level = st.slider(
                "Price noise level (%)",
                min_value=0.5,
                max_value=5.0,  # Changed to float
                value=2.0,
                step=0.5,
                help="Amount of random price variation to add (realistic: 1-3%)"
            ) / 100

            modify_order_count = st.checkbox(
                "Modify number of orders",
                value=False,
                help="If checked, will add/remove orders. If unchecked, keeps same order structure"
            )

            order_variation = 0
            if modify_order_count:
                order_variation = st.slider(
                    "Order variation (%)",
                    min_value=2.0,  # Changed to float
                    max_value=20.0,  # Changed to float
                    value=5.0,  # Changed to float
                    step=1.0,  # Changed to float
                    help="Percentage of orders to add/remove (realistic: 5-10%)"
                ) / 100

            if noise_level > 0.03:
                st.warning("⚠️ High noise levels (>3%) may create unrealistic differences")

            if st.button("🎲 Generate Test Data"):
                if demo_file and not live_file:
                    # Load demo file and generate synthetic live data
                    source_df = load_data(demo_file)
                    if source_df is not None:
                        st.session_state.synthetic_live = generate_noisy_data(source_df, noise_level, order_variation,
                                                                              modify_order_count)
                        original_volume = source_df['Value'].sum()
                        synthetic_volume = st.session_state.synthetic_live['Value'].sum()
                        volume_diff = ((synthetic_volume / original_volume - 1) * 100)
                        st.success(f"✅ Synthetic LIVE data generated!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Volume Change", f"{volume_diff:+.1f}%")
                        with col2:
                            st.metric("Orders", f"{len(source_df)} → {len(st.session_state.synthetic_live)}")
                elif live_file and not demo_file:
                    # Load live file and generate synthetic demo data
                    source_df = load_data(live_file)
                    if source_df is not None:
                        st.session_state.synthetic_demo = generate_noisy_data(source_df, noise_level, order_variation,
                                                                              modify_order_count)
                        original_volume = source_df['Value'].sum()
                        synthetic_volume = st.session_state.synthetic_demo['Value'].sum()
                        volume_diff = ((synthetic_volume / original_volume - 1) * 100)
                        st.success(f"✅ Synthetic DEMO data generated!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Volume Change", f"{volume_diff:+.1f}%")
                        with col2:
                            st.metric("Orders", f"{len(source_df)} → {len(st.session_state.synthetic_demo)}")
    elif not demo_file and not live_file:
        st.info("📁 Please upload at least one CSV file to begin")
    else:
        # Both files uploaded, clear synthetic data
        st.session_state.synthetic_demo = None
        st.session_state.synthetic_live = None

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Options")

    # Advanced Filters
    st.markdown("#### 🔍 Advanced Filters")

    # Date range filter
    use_date_filter = st.checkbox("Enable date range filter", value=False)
    if use_date_filter:
        date_range = st.date_input(
            "Select date range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )

    # Symbol filter
    use_symbol_filter = st.checkbox("Filter by symbols", value=False)

    # Amount filter
    use_amount_filter = st.checkbox("Filter by amount", value=False)
    if use_amount_filter:
        min_amount = st.number_input("Minimum amount ($)", value=0.0, min_value=0.0)
        max_amount = st.number_input("Maximum amount ($)", value=1000000.0, min_value=0.0)

    st.markdown("---")
    st.markdown("#### 🎯 Matching Parameters")

    time_window = st.slider(
        "Time window for matching (minutes)",
        min_value=1.0,  # Changed to float
        max_value=30.0,  # Changed to float
        value=5.0,  # Changed to float
        step=1.0,  # Added step as float
        help="Maximum time difference to consider orders as matching"
    )

    price_threshold = st.slider(
        "Price threshold for matching (%)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Maximum price difference to consider orders as matching"
    ) / 100

    st.markdown("---")
    st.markdown("#### 📊 Display Options")

    show_raw_data = st.checkbox("Show raw data", value=False)
    show_anomalies = st.checkbox("Show anomaly detection", value=True)
    show_matching = st.checkbox("Show order matching", value=True)
    show_time_analysis = st.checkbox("Time analysis", value=True)

# Main application body
# Determine which data to use
demo_df = None
live_df = None
using_synthetic = False

# Load or use demo data
if demo_file:
    demo_df = load_data(demo_file)
elif st.session_state.synthetic_demo is not None:
    demo_df = st.session_state.synthetic_demo
    using_synthetic = True

# Load or use live data
if live_file:
    live_df = load_data(live_file)
elif st.session_state.synthetic_live is not None:
    live_df = st.session_state.synthetic_live
    using_synthetic = True

# Process data if both are available
if demo_df is not None and live_df is not None:

    # Show notification if using synthetic data
    if using_synthetic:
        st.info("📊 Using synthetic data for comparison. Upload both files to use real data.")

    # Apply filters
    if use_date_filter and len(date_range) == 2:
        demo_df = demo_df[(demo_df['Date'] >= date_range[0]) & (demo_df['Date'] <= date_range[1])]
        live_df = live_df[(live_df['Date'] >= date_range[0]) & (live_df['Date'] <= date_range[1])]

    if use_amount_filter:
        demo_df = demo_df[(demo_df['Value'] >= min_amount) & (demo_df['Value'] <= max_amount)]
        live_df = live_df[(live_df['Value'] >= min_amount) & (live_df['Value'] <= max_amount)]

    # Symbol filter
    if use_symbol_filter:
        all_symbols = sorted(list(set(demo_df['Symbol'].unique()) | set(live_df['Symbol'].unique())))
        selected_symbols = st.sidebar.multiselect("Select symbols", all_symbols)
        if selected_symbols:
            demo_df = demo_df[demo_df['Symbol'].isin(selected_symbols)]
            live_df = live_df[live_df['Symbol'].isin(selected_symbols)]

    # Tabs for different analyses
    tabs = st.tabs([
        "📊 Overview",
        "🔍 Anomaly Detection",
        "🔗 Order Matching",
        "📈 Performance",
        "⏰ Time Analysis",
        "🎯 Symbol Details"
    ])

    with tabs[0]:  # Overview Tab
        st.markdown("## 📊 Account Overview")

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📝 Demo Orders",
                len(demo_df),
                delta=None
            )

        with col2:
            st.metric(
                "💼 Live Orders",
                len(live_df),
                delta=f"{len(live_df) - len(demo_df):+d}"
            )

        with col3:
            demo_value = demo_df['Value'].sum()
            st.metric(
                "💰 Demo Volume",
                f"${demo_value:,.2f}",
                delta=None
            )

        with col4:
            live_value = live_df['Value'].sum()
            st.metric(
                "💎 Live Volume",
                f"${live_value:,.2f}",
                delta=f"{((live_value / demo_value - 1) * 100):.1f}%" if demo_value > 0 else None
            )

        # Quick anomaly summary
        if show_anomalies:
            anomalies_df = detect_anomalies(demo_df, live_df)
            if not anomalies_df.empty:
                st.markdown("### ⚠️ Anomaly Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_anomalies = len(anomalies_df[anomalies_df['Severity'] == 'High'])
                    st.metric("🔴 High Severity", high_anomalies)
                with col2:
                    medium_anomalies = len(anomalies_df[anomalies_df['Severity'] == 'Medium'])
                    st.metric("🟡 Medium Severity", medium_anomalies)
                with col3:
                    low_anomalies = len(anomalies_df[anomalies_df['Severity'] == 'Low'])
                    st.metric("🔵 Low Severity", low_anomalies)

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Order Type Distribution - DEMO")
            demo_type_counts = demo_df['Type'].value_counts()
            fig_demo_type = px.pie(
                values=demo_type_counts.values,
                names=demo_type_counts.index,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_demo_type.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_demo_type, key="demo_type_pie")

        with col2:
            st.markdown("### 📊 Order Type Distribution - LIVE")
            live_type_counts = live_df['Type'].value_counts()
            fig_live_type = px.pie(
                values=live_type_counts.values,
                names=live_type_counts.index,
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig_live_type.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_live_type, key="live_type_pie")

    with tabs[1]:  # Anomaly Detection Tab
        st.markdown("## 🔍 Anomaly Detection & Alerts")

        anomalies_df = detect_anomalies(demo_df, live_df)

        if anomalies_df.empty:
            st.success("✅ No significant anomalies detected!")
        else:
            # Anomaly statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Anomalies", len(anomalies_df))
            with col2:
                unique_symbols = anomalies_df['Symbol'].nunique()
                st.metric("Affected Symbols", unique_symbols)
            with col3:
                anomaly_types = anomalies_df['Type'].nunique()
                st.metric("Anomaly Types", anomaly_types)
            with col4:
                high_severity = len(anomalies_df[anomalies_df['Severity'] == 'High'])
                st.metric("High Severity", high_severity)

            st.markdown("---")

            # Display anomalies by severity
            st.markdown("### 🚨 Anomaly Details")

            # High severity anomalies
            high_anomalies = anomalies_df[anomalies_df['Severity'] == 'High']
            if not high_anomalies.empty:
                st.markdown("#### 🔴 High Severity Anomalies")
                for _, anomaly in high_anomalies.iterrows():
                    st.markdown(f"""
                    <div class="alert-box alert-high">
                        <strong>{anomaly['Type']}</strong> - {anomaly['Symbol']}<br>
                        {anomaly['Details']}
                    </div>
                    """, unsafe_allow_html=True)

            # Medium severity anomalies
            medium_anomalies = anomalies_df[anomalies_df['Severity'] == 'Medium']
            if not medium_anomalies.empty:
                st.markdown("#### 🟡 Medium Severity Anomalies")
                for _, anomaly in medium_anomalies.iterrows():
                    st.markdown(f"""
                    <div class="alert-box alert-medium">
                        <strong>{anomaly['Type']}</strong> - {anomaly['Symbol']}<br>
                        {anomaly['Details']}
                    </div>
                    """, unsafe_allow_html=True)

            # Low severity anomalies
            low_anomalies = anomalies_df[anomalies_df['Severity'] == 'Low']
            if not low_anomalies.empty:
                with st.expander("🔵 Low Severity Anomalies"):
                    for _, anomaly in low_anomalies.iterrows():
                        st.markdown(f"""
                        <div class="alert-box alert-low">
                            <strong>{anomaly['Type']}</strong> - {anomaly['Symbol']}<br>
                            {anomaly['Details']}
                        </div>
                        """, unsafe_allow_html=True)

            # Anomaly visualization
            st.markdown("### 📊 Anomaly Distribution")

            col1, col2 = st.columns(2)
            with col1:
                fig_severity = px.pie(
                    values=anomalies_df['Severity'].value_counts().values,
                    names=anomalies_df['Severity'].value_counts().index,
                    title="Anomalies by Severity",
                    color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#3b82f6'}
                )
                st.plotly_chart(fig_severity, key="anomaly_severity_pie")

            with col2:
                fig_type = px.bar(
                    x=anomalies_df['Type'].value_counts().index,
                    y=anomalies_df['Type'].value_counts().values,
                    title="Anomalies by Type",
                    color=anomalies_df['Type'].value_counts().values,
                    color_continuous_scale='Reds'
                )
                fig_type.update_layout(showlegend=False)
                st.plotly_chart(fig_type, key="anomaly_type_bar")

            # Detailed anomaly table
            st.markdown("### 📋 Anomaly Details Table")
            st.dataframe(
                anomalies_df[['Type', 'Severity', 'Symbol', 'Details']],
                height=400
            )

    with tabs[2]:  # Order Matching Tab
        st.markdown("## 🔗 Order Matching & Reconciliation")

        # Perform order matching
        matched_orders, unmatched_demo, unmatched_live = match_orders(
            demo_df, live_df, time_window, price_threshold
        )

        # Matching statistics
        total_demo = len(demo_df)
        total_live = len(live_df)
        matched_count = len(matched_orders)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            match_rate_demo = (matched_count / total_demo * 100) if total_demo > 0 else 0
            st.metric("Demo Match Rate", f"{match_rate_demo:.1f}%")
        with col2:
            match_rate_live = (matched_count / total_live * 100) if total_live > 0 else 0
            st.metric("Live Match Rate", f"{match_rate_live:.1f}%")
        with col3:
            st.metric("Matched Orders", matched_count)
        with col4:
            st.metric("Unmatched Total", len(unmatched_demo) + len(unmatched_live))

        st.markdown("---")

        # Matching visualization
        st.markdown("### 📊 Matching Overview")

        matching_data = pd.DataFrame({
            'Category': ['Matched', 'Unmatched Demo', 'Unmatched Live'],
            'Count': [matched_count, len(unmatched_demo), len(unmatched_live)]
        })

        fig_matching = px.bar(
            matching_data,
            x='Category',
            y='Count',
            color='Category',
            color_discrete_map={
                'Matched': '#10b981',
                'Unmatched Demo': '#3b82f6',
                'Unmatched Live': '#ef4444'
            },
            title="Order Matching Results"
        )
        st.plotly_chart(fig_matching, key="matching_overview_bar")

        # Matched orders analysis
        if not matched_orders.empty:
            st.markdown("### ✅ Matched Orders Analysis")

            col1, col2 = st.columns(2)
            with col1:
                avg_slippage = matched_orders['Slippage'].mean()
                st.metric("Average Slippage", f"${avg_slippage:.4f}")

                # Slippage distribution
                fig_slippage = px.histogram(
                    matched_orders,
                    x='Slippage',
                    nbins=30,
                    title="Slippage Distribution"
                )
                st.plotly_chart(fig_slippage, key="slippage_hist")

            with col2:
                avg_time_diff = matched_orders['Time_Diff_Min'].mean()
                st.metric("Avg Time Difference", f"{avg_time_diff:.2f} min")

                # Time difference distribution
                fig_time_diff = px.histogram(
                    matched_orders,
                    x='Time_Diff_Min',
                    nbins=20,
                    title="Time Difference Distribution"
                )
                st.plotly_chart(fig_time_diff, key="time_diff_hist")

            # Best and worst matches
            st.markdown("### 🎯 Match Quality")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🏆 Best Matches (Lowest Slippage)")
                best_matches = matched_orders.nsmallest(5, 'Price_Diff_%')[
                    ['Symbol', 'Type', 'Price_Diff_%', 'Time_Diff_Min']
                ]
                st.dataframe(best_matches)

            with col2:
                st.markdown("#### ⚠️ Worst Matches (Highest Slippage)")
                worst_matches = matched_orders.nlargest(5, 'Price_Diff_%')[
                    ['Symbol', 'Type', 'Price_Diff_%', 'Time_Diff_Min']
                ]
                st.dataframe(worst_matches)

            # Detailed matched orders table
            with st.expander("📋 View All Matched Orders"):
                st.dataframe(
                    matched_orders[[
                        'Symbol', 'Type', 'Demo_Time', 'Live_Time',
                        'Demo_Price', 'Live_Price', 'Slippage', 'Price_Diff_%'
                    ]].style.format({
                        'Demo_Price': '${:.2f}',
                        'Live_Price': '${:.2f}',
                        'Slippage': '${:.4f}',
                        'Price_Diff_%': '{:.2f}%'
                    }),
                    height=400
                )

        # Unmatched orders analysis
        st.markdown("### ❌ Unmatched Orders")

        tab_unmatch1, tab_unmatch2 = st.tabs(["Demo Only", "Live Only"])

        with tab_unmatch1:
            if not unmatched_demo.empty:
                st.warning(f"Found {len(unmatched_demo)} unmatched demo orders")

                # Group by symbol
                unmatched_demo_summary = unmatched_demo.groupby('Symbol').agg({
                    'Value': ['count', 'sum'],
                    'Price': 'mean'
                }).round(2)
                unmatched_demo_summary.columns = ['Count', 'Total Value', 'Avg Price']
                st.dataframe(unmatched_demo_summary)
            else:
                st.success("All demo orders matched!")

        with tab_unmatch2:
            if not unmatched_live.empty:
                st.warning(f"Found {len(unmatched_live)} unmatched live orders")

                # Group by symbol
                unmatched_live_summary = unmatched_live.groupby('Symbol').agg({
                    'Value': ['count', 'sum'],
                    'Price': 'mean'
                }).round(2)
                unmatched_live_summary.columns = ['Count', 'Total Value', 'Avg Price']
                st.dataframe(unmatched_live_summary)
            else:
                st.success("All live orders matched!")

    with tabs[3]:  # Performance Tab
        st.markdown("## 📈 Performance Analysis")

        # Calculate performance metrics
        demo_filled = demo_df[demo_df['Status'] == 'Filled'] if 'Filled' in demo_df['Status'].values else pd.DataFrame()
        live_filled = live_df[live_df['Status'] == 'Filled'] if 'Filled' in live_df['Status'].values else pd.DataFrame()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Fill Rate")
            demo_fill_rate = (len(demo_filled) / len(demo_df) * 100) if len(demo_df) > 0 else 0
            live_fill_rate = (len(live_filled) / len(live_df) * 100) if len(live_df) > 0 else 0

            fig_fill = go.Figure(data=[
                go.Bar(name='Demo', x=['Fill Rate'], y=[demo_fill_rate], marker_color='royalblue'),
                go.Bar(name='Live', x=['Fill Rate'], y=[live_fill_rate], marker_color='crimson')
            ])
            fig_fill.update_layout(yaxis_title="Percentage (%)", showlegend=True, height=300)
            st.plotly_chart(fig_fill, key="fill_rate_bar")

            st.metric("Fill Rate Difference", f"{live_fill_rate - demo_fill_rate:.2f}%")

        with col2:
            st.markdown("### 💰 Average Volume per Order")
            demo_avg_volume = demo_df['Value'].mean()
            live_avg_volume = live_df['Value'].mean()

            fig_avg = go.Figure(data=[
                go.Bar(name='Demo', x=['Average Volume'], y=[demo_avg_volume], marker_color='lightseagreen'),
                go.Bar(name='Live', x=['Average Volume'], y=[live_avg_volume], marker_color='darkorange')
            ])
            fig_avg.update_layout(yaxis_title="Volume ($)", showlegend=True, height=300)
            st.plotly_chart(fig_avg, key="avg_volume_bar")

            diff_percent = ((live_avg_volume / demo_avg_volume - 1) * 100) if demo_avg_volume > 0 else 0
            st.metric("Average Volume Difference", f"{diff_percent:.2f}%")

        # Time evolution
        st.markdown("### 📈 Cumulative Volume Evolution")

        demo_daily = demo_df.groupby(demo_df['Time'].dt.date)['Value'].sum().cumsum()
        live_daily = live_df.groupby(live_df['Time'].dt.date)['Value'].sum().cumsum()

        fig_evolution = go.Figure()
        fig_evolution.add_trace(go.Scatter(
            x=demo_daily.index,
            y=demo_daily.values,
            mode='lines',
            name='Demo',
            line=dict(color='blue', width=2)
        ))
        fig_evolution.add_trace(go.Scatter(
            x=live_daily.index,
            y=live_daily.values,
            mode='lines',
            name='Live',
            line=dict(color='red', width=2)
        ))
        fig_evolution.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative Volume ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_evolution, key="cumulative_evolution")

    with tabs[4]:  # Time Analysis Tab
        if show_time_analysis:
            st.markdown("## ⏰ Time Analysis")

            # Hourly analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🕐 Activity by Hour - DEMO")
                demo_hourly = demo_df.groupby('Hour').size()
                fig_hour_demo = px.bar(
                    x=demo_hourly.index,
                    y=demo_hourly.values,
                    color=demo_hourly.values,
                    color_continuous_scale='Blues'
                )
                fig_hour_demo.update_layout(
                    xaxis_title="Hour",
                    yaxis_title="Number of Orders",
                    showlegend=False
                )
                st.plotly_chart(fig_hour_demo, key="demo_hourly_bar")

            with col2:
                st.markdown("### 🕐 Activity by Hour - LIVE")
                live_hourly = live_df.groupby('Hour').size()
                fig_hour_live = px.bar(
                    x=live_hourly.index,
                    y=live_hourly.values,
                    color=live_hourly.values,
                    color_continuous_scale='Reds'
                )
                fig_hour_live.update_layout(
                    xaxis_title="Hour",
                    yaxis_title="Number of Orders",
                    showlegend=False
                )
                st.plotly_chart(fig_hour_live, key="live_hourly_bar")

            # Heatmap by day of week
            st.markdown("### 📅 Heatmap - Activity by Day")

            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            demo_pivot = demo_df.pivot_table(
                values='Value',
                index='DayOfWeek',
                columns='Hour',
                aggfunc='count',
                fill_value=0
            ).reindex(days_order, fill_value=0)

            fig_heatmap = px.imshow(
                demo_pivot,
                labels=dict(x="Hour", y="Day", color="Number of Orders"),
                color_continuous_scale='Viridis',
                aspect='auto'
            )
            fig_heatmap.update_layout(title="Order Distribution (Demo)")
            st.plotly_chart(fig_heatmap, key="activity_heatmap")

    with tabs[5]:  # Symbol Details Tab
        st.markdown("## 🎯 Detailed Analysis by Symbol")

        # Symbol selection
        all_symbols = sorted(list(set(demo_df['Symbol'].unique()) | set(live_df['Symbol'].unique())))
        selected_symbol = st.selectbox("Select a symbol:", all_symbols)

        if selected_symbol:
            col1, col2 = st.columns(2)

            demo_symbol_data = demo_df[demo_df['Symbol'] == selected_symbol]
            live_symbol_data = live_df[live_df['Symbol'] == selected_symbol]

            with col1:
                st.markdown(f"### 📊 {selected_symbol} - DEMO")
                if not demo_symbol_data.empty:
                    st.metric("Number of Orders", len(demo_symbol_data))
                    st.metric("Total Volume", f"${demo_symbol_data['Value'].sum():,.2f}")
                    st.metric("Average Price", f"${demo_symbol_data['Price'].mean():,.2f}")
                    st.metric("Average Quantity", f"{demo_symbol_data['Quantity'].mean():,.2f}")

                    # Price distribution
                    fig_price_demo = px.histogram(
                        demo_symbol_data,
                        x='Price',
                        nbins=20,
                        title="Price Distribution"
                    )
                    st.plotly_chart(fig_price_demo, key=f"demo_price_dist_{selected_symbol}")
                else:
                    st.warning("No data for this symbol in demo")

            with col2:
                st.markdown(f"### 💼 {selected_symbol} - LIVE")
                if not live_symbol_data.empty:
                    st.metric("Number of Orders", len(live_symbol_data))
                    st.metric("Total Volume", f"${live_symbol_data['Value'].sum():,.2f}")
                    st.metric("Average Price", f"${live_symbol_data['Price'].mean():,.2f}")
                    st.metric("Average Quantity", f"{live_symbol_data['Quantity'].mean():,.2f}")

                    # Price distribution
                    fig_price_live = px.histogram(
                        live_symbol_data,
                        x='Price',
                        nbins=20,
                        title="Price Distribution"
                    )
                    st.plotly_chart(fig_price_live, key=f"live_price_dist_{selected_symbol}")
                else:
                    st.warning("No data for this symbol in live")

            # Price comparison chart over time
            if not demo_symbol_data.empty or not live_symbol_data.empty:
                st.markdown(f"### 📈 Price Evolution - {selected_symbol}")

                fig_price_evolution = go.Figure()

                if not demo_symbol_data.empty:
                    fig_price_evolution.add_trace(go.Scatter(
                        x=demo_symbol_data['Time'],
                        y=demo_symbol_data['Price'],
                        mode='markers+lines',
                        name='Demo',
                        marker=dict(size=8, color='blue', opacity=0.6)
                    ))

                if not live_symbol_data.empty:
                    fig_price_evolution.add_trace(go.Scatter(
                        x=live_symbol_data['Time'],
                        y=live_symbol_data['Price'],
                        mode='markers+lines',
                        name='Live',
                        marker=dict(size=8, color='red', opacity=0.6)
                    ))

                fig_price_evolution.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_price_evolution, key=f"price_evolution_{selected_symbol}")

    # Display raw data if requested
    if show_raw_data:
        st.markdown("---")
        st.markdown("## 📋 Raw Data")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### DEMO Data")
            st.dataframe(demo_df, height=400)

        with col2:
            st.markdown("### LIVE Data")
            st.dataframe(live_df, height=400)

else:
    # Welcome message if no files are loaded
    st.markdown("""
    <div style='text-align: center; padding: 50px; background-color: #f8fafc; border-radius: 15px;'>
        <h2 style='color: #667eea;'>👋 Welcome to Trading Comparator Pro</h2>
        <p style='font-size: 18px; color: #64748b;'>
            Advanced trading analysis with anomaly detection and order matching
        </p>
        <p style='font-size: 16px; color: #94a3b8;'>
            📁 Expected format: CSV with columns Time, Symbol, Price, Quantity, Type, Status, Value, Tag
        </p>
        <div style='margin-top: 20px; padding: 15px; background-color: #e0f2fe; border-radius: 8px;'>
            <p style='color: #0369a1; font-weight: bold;'>
                💡 Tip: Upload one file to generate synthetic test data for comparison!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Example of expected data structure
    with st.expander("📋 View data structure example"):
        example_data = pd.DataFrame({
            'Time': ['2024-01-15 10:30:00', '2024-01-15 11:45:00', '2024-01-15 14:20:00'],
            'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
            'Price': [150.25, 2800.50, 380.75],
            'Quantity': [100, 50, 75],
            'Type': ['Market', 'Limit', 'Market'],
            'Status': ['Filled', 'Filled', 'Partial'],
            'Value': [15025.00, 140025.00, 28556.25],
            'Tag': ['Strategy1', 'Strategy2', 'Strategy1']
        })
        st.dataframe(example_data)