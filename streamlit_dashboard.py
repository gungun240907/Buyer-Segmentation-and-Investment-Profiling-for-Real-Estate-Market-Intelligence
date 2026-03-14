"""
Streamlit Dashboard: Real Estate Buyer Segmentation
Parcl Co. Limited - Market Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Parcl Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
        padding: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stDataFrame {
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CACHING
# ============================================================
@st.cache_data
def load_data():
    """Load the segmented buyer data"""
    try:
        df = pd.read_csv('segmented_buyers.csv')
        return df
    except FileNotFoundError:
        st.error("Please run buyer_segmentation.py first to generate the data!")
        return None

# Load data
df = load_data()

if df is None:
    st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("🏠 Parcl Analytics")
st.sidebar.image("https://via.placeholder.com/150x50?text=Parcl+Co", width=150)

st.sidebar.subheader("📊 Filter Options")

# Country Filter
country_list = sorted(df['country'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "🌍 Select Country",
    options=country_list,
    default=country_list
)

# Region Filter
region_list = sorted(df['region'].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "🌎 Select Region",
    options=region_list,
    default=region_list
)

# Client Type Filter
client_type_list = sorted(df['client_type'].unique().tolist())
selected_client_types = st.sidebar.multiselect(
    "👤 Select Client Type",
    options=client_type_list,
    default=client_type_list
)

# Cluster/Segment Filter
segment_list = sorted(df['segment_name'].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "🎯 Select Segment",
    options=segment_list,
    default=segment_list
)

# Apply Filters
df_filtered = df[
    (df['country'].isin(selected_countries)) &
    (df['region'].isin(selected_regions)) &
    (df['client_type'].isin(selected_client_types)) &
    (df['segment_name'].isin(selected_segments))
]

# ============================================================
# MAIN DASHBOARD
# ============================================================
st.markdown('<div class="main-header">🏠 Real Estate Buyer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Parcl Co. Limited - Market Intelligence Platform")
st.markdown("---")

# ============================================================
# KPI METRICS
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Buyers", f"{len(df_filtered):,}", delta=f"{len(df_filtered)-len(df):+d}")

with col2:
    st.metric("Avg Satisfaction", f"{df_filtered['satisfaction_score'].mean():.2f}/5.0")

with col3:
    st.metric("Avg Property Value", f"${df_filtered['property_value'].mean():,.0f}")

with col4:
    st.metric("Avg Annual Income", f"${df_filtered['annual_income'].mean():,.0f}")

with col5:
    st.metric("Avg Investment Horizon", f"{df_filtered['investment_horizon_years'].mean():.1f} yrs")

st.markdown("---")

# ============================================================
# ROW 1: VISUALIZATIONS
# ============================================================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Segment Distribution")
    
    segment_counts = df_filtered['segment_name'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    fig_pie = px.pie(
        segment_counts, 
        values='Count', 
        names='Segment',
        title='Buyer Segments',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("🎯 Cluster Statistics")
    
    cluster_stats = df_filtered.groupby('segment_name').agg({
        'client_id': 'count',
        'age': 'mean',
        'annual_income': 'mean',
        'property_value': 'mean',
        'satisfaction_score': 'mean'
    }).round(2)
    
    cluster_stats.columns = ['Count', 'Avg Age', 'Avg Income', 'Avg Property', 'Avg Satisfaction']
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'), use_container_width=True)

# ============================================================
# ROW 2: GEOGRAPHIC & CLIENT ANALYSIS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Buyers by Country")
    
    country_counts = df_filtered['country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    fig_country = px.bar(
        country_counts,
        x='Country',
        y='Count',
        title='Distribution by Country',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig_country.update_layout(xaxis_title='Country', yaxis_title='Number of Buyers')
    st.plotly_chart(fig_country, use_container_width=True)

with col2:
    st.subheader("👥 Buyers by Client Type")
    
    client_counts = df_filtered['client_type'].value_counts().reset_index()
    client_counts.columns = ['Client Type', 'Count']
    
    fig_client = px.bar(
        client_counts,
        x='Client Type',
        y='Count',
        title='Distribution by Client Type',
        color='Count',
        color_continuous_scale='Plasma'
    )
    fig_client.update_layout(xaxis_title='Client Type', yaxis_title='Number of Buyers')
    st.plotly_chart(fig_client, use_container_width=True)

# ============================================================
# ROW 3: ACQUISITION PURPOSE & INVESTMENT ANALYSIS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Acquisition Purpose Distribution")
    
    purpose_counts = df_filtered['acquisition_purpose'].value_counts().reset_index()
    purpose_counts.columns = ['Purpose', 'Count']
    
    fig_purpose = px.bar(
        purpose_counts,
        x='Purpose',
        y='Count',
        title='Acquisition Purpose Analysis',
        color='Count',
        color_continuous_scale='Tealgrn'
    )
    fig_purpose.update_layout(xaxis_title='Purpose', yaxis_title='Count')
    st.plotly_chart(fig_purpose, use_container_width=True)

with col2:
    st.subheader("💰 Investment Horizon Analysis")
    
    fig_horizon = px.box(
        df_filtered,
        x='segment_name',
        y='investment_horizon_years',
        title='Investment Horizon by Segment',
        color='segment_name'
    )
    fig_horizon.update_layout(xaxis_title='Segment', yaxis_title='Years')
    fig_horizon.update_xaxes(tickangle=45)
    st.plotly_chart(fig_horizon, use_container_width=True)

# ============================================================
# ROW 4: CORRELATION & SCATTER ANALYSIS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("💵 Income vs Property Value")
    
    fig_scatter = px.scatter(
        df_filtered,
        x='annual_income',
        y='property_value',
        color='segment_name',
        size='satisfaction_score',
        hover_data=['client_id', 'country'],
        title='Income vs Property Value by Segment',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_scatter.update_layout(xaxis_title='Annual Income ($)', yaxis_title='Property Value ($)')
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("⭐ Satisfaction by Segment")
    
    satisfaction_by_segment = df_filtered.groupby('segment_name')['satisfaction_score'].mean().reset_index()
    satisfaction_by_segment.columns = ['Segment', 'Avg Satisfaction']
    satisfaction_by_segment = satisfaction_by_segment.sort_values('Avg Satisfaction', ascending=True)
    
    fig_satisfaction = px.bar(
        satisfaction_by_segment,
        x='Avg Satisfaction',
        y='Segment',
        orientation='h',
        title='Average Satisfaction Score by Segment',
        color='Avg Satisfaction',
        color_continuous_scale='RdYlGn'
    )
    fig_satisfaction.update_layout(xaxis_title='Satisfaction Score (1-5)', yaxis_title='Segment')
    st.plotly_chart(fig_satisfaction, use_container_width=True)

# ============================================================
# ROW 5: DETAILED DATA TABLE
# ============================================================
st.markdown("---")
st.subheader("📋 Detailed Buyer Data")

# Add search functionality
search_term = st.text_input("🔍 Search by Client ID", "")

if search_term:
    df_display = df_filtered[df_filtered['client_id'].str.contains(search_term, case=False)]
else:
    df_display = df_filtered

# Format currency columns for display
df_display_formatted = df_display.copy()
df_display_formatted['annual_income'] = df_display_formatted['annual_income'].apply(lambda x: f"${x:,.0f}")
df_display_formatted['property_value'] = df_display_formatted['property_value'].apply(lambda x: f"${x:,.0f}")
df_display_formatted['budget_range'] = df_display_formatted['budget_range'].apply(lambda x: f"${x:,.0f}")
df_display_formatted['satisfaction_score'] = df_display_formatted['satisfaction_score'].round(2)

# Display with pagination
st.dataframe(
    df_display_formatted,
    use_container_width=True,
    height=400
)

st.markdown(f"Showing {len(df_display)} of {len(df_filtered)} filtered records")

# ============================================================
# EXPORT OPTIONS
# ============================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name='filtered_buyers.csv',
        mime='text/csv'
    )

with col2:
    # Summary statistics export
    summary_stats = df_filtered.groupby('segment_name').agg({
        'client_id': 'count',
        'age': ['mean', 'std'],
        'annual_income': ['mean', 'std'],
        'property_value': ['mean', 'std'],
        'satisfaction_score': 'mean',
        'investment_horizon_years': 'mean'
    }).round(2)
    summary_csv = summary_stats.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Segment Summary",
        data=summary_csv,
        file_name='segment_summary.csv',
        mime='text/csv'
    )

with col3:
    st.info("💡 Tip: Use sidebar filters to narrow down your analysis!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Parcl Co. Limited - Real Estate Market Intelligence Platform</p>
        <p>Powered by Machine Learning & AI</p>
    </div>
    """, 
    unsafe_allow_html=True
)
