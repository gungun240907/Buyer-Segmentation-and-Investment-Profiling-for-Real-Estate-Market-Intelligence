"""
Real Estate Buyer Segmentation Application
Parcl Co. Limited - Market Intelligence Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Parcl Buyer Segmentation", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: bold; color: #1f77b4; text-align: center; padding: 20px; }
    .sub-header { font-size: 24px; font-weight: bold; color: #2ca02c; padding: 10px; }
    .success-box { padding: 15px; background-color: #d4edda; border-radius: 10px; border: 1px solid #c3e6cb; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_and_process_data():
    np.random.seed(42)
    n_samples = 1000
    
    client_types = ['Individual', 'Corporate', 'Institutional', 'Foreign_Investor', 'Real_Estate_Fund']
    acquisition_purposes = ['Primary_Residence', 'Investment_Rental', 'Flip_Resale', 'Portfolio_Diversification', 'Commercial_Use']
    countries = ['USA', 'UK', 'UAE', 'Singapore', 'Germany', 'Canada', 'Australia']
    regions = ['North_America', 'Europe', 'Middle_East', 'Asia_Pacific']
    property_types = ['Residential', 'Commercial', 'Mixed_Use', 'Industrial']
    
    data = {
        'client_id': [f'CL{i:05d}' for i in range(1, n_samples + 1)],
        'client_type': np.random.choice(client_types, n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        'acquisition_purpose': np.random.choice(acquisition_purposes, n_samples),
        'country': np.random.choice(countries, n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07]),
        'region': np.random.choice(regions, n_samples),
        'property_type': np.random.choice(property_types, n_samples),
        'age': np.random.randint(22, 75, n_samples),
        'annual_income': np.random.randint(30000, 500000, n_samples),
        'budget_range': np.random.randint(100000, 5000000, n_samples),
        'satisfaction_score': np.random.uniform(1.0, 5.0, n_samples),
        'property_value': np.random.randint(150000, 8000000, n_samples),
        'investment_horizon_years': np.random.randint(1, 20, n_samples),
        'num_prior_investments': np.random.randint(0, 25, n_samples),
        'risk_tolerance': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    missing_indices_age = np.random.choice(df.index, 30, replace=False)
    missing_indices_satisfaction = np.random.choice(df.index, 25, replace=False)
    df.loc[missing_indices_age, 'age'] = np.nan
    df.loc[missing_indices_satisfaction, 'satisfaction_score'] = np.nan
    
    df_clean = df.copy()
    
    categorical_cols = ['client_type', 'acquisition_purpose', 'country', 'region', 'property_type', 'risk_tolerance']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    numerical_cols = ['age', 'satisfaction_score', 'annual_income', 'budget_range', 
                      'property_value', 'investment_horizon_years', 'num_prior_investments']
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    categorical_features = ['client_type', 'acquisition_purpose', 'country']
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=False)
    numerical_features = ['age', 'satisfaction_score', 'annual_income', 'budget_range',
                         'property_value', 'investment_horizon_years', 'num_prior_investments']
    
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    feature_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    X = df_scaled[feature_columns].values
    
    inertias = []
    silhouettes = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
    
    optimal_k = list(K_range)[np.argmax(silhouettes)]
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    kmeans_labels = kmeans_final.fit_predict(X)
    
    hier_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hier_labels = hier_clustering.fit_predict(X)
    
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    hier_silhouette = silhouette_score(X, hier_labels)
    
    df_clean['cluster_kmeans'] = kmeans_labels
    df_clean['cluster_hierarchical'] = hier_labels
    
    def assign_cluster_names(df):
        segment_names = []
        for idx, row in df.iterrows():
            income = row['annual_income']
            horizon = row['investment_horizon_years']
            client = row['client_type']
            
            if income > 350000:
                segment_names.append('Global High-Net-Worth Investors')
            elif horizon > 12:
                segment_names.append('Long-Term Portfolio Builders')
            elif client in ['institutional', 'real_estate_fund']:
                segment_names.append('Institutional Investors')
            elif income > 250000:
                segment_names.append('Affluent Property Seekers')
            elif client == 'corporate':
                segment_names.append('Corporate Buyers')
            else:
                segment_names.append('Standard Retail Clients')
        return segment_names
    
    df_clean['segment_name'] = assign_cluster_names(df_clean)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    return {
        'df': df_clean,
        'X_pca': X_pca,
        'kmeans_labels': kmeans_labels,
        'hier_labels': hier_labels,
        'optimal_k': optimal_k,
        'kmeans_silhouette': kmeans_silhouette,
        'hier_silhouette': hier_silhouette,
        'K_range': list(K_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'pca': pca
    }


with st.spinner('Generating and processing buyer data...'):
    results = generate_and_process_data()
    df = results['df']

# Sidebar navigation
st.sidebar.title("🏠 Parcl Analytics")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("📑 Select Page", ["Home", "Analysis", "Dashboard", "Data"])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Parcl Buyer Segmentation**
- Machine Learning powered buyer profiling
- K-Means & Hierarchical Clustering
- Real-time analytics dashboard
""")

# ===================== HOME PAGE =====================
if page == "Home":
    st.markdown('<div class="main-header">🏠 Real Estate Buyer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("### Parcl Co. Limited - Market Intelligence Platform")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Buyers", f"{len(df):,}")
    with col2:
        st.metric("Optimal Clusters", f"{results['optimal_k']}")
    with col3:
        st.metric("K-Means Score", f"{results['kmeans_silhouette']:.4f}")
    with col4:
        st.metric("Data Sources", "7 Countries")
    
    st.markdown("---")
    
    st.subheader("📊 Segment Overview")
    
    segment_counts = df['segment_name'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    segment_counts['Percentage'] = (segment_counts['Count'] / segment_counts['Count'].sum() * 100).round(1)
    
    fig_pie = px.pie(segment_counts, values='Count', names='Segment', title='Buyer Segments Distribution',
                      hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Analysis Complete!</h4>
        <p>The buyer segmentation analysis has been successfully completed with {results['optimal_k']} optimal clusters.</p>
        <p>Use the sidebar to navigate between pages.</p>
    </div>
    """, unsafe_allow_html=True)


# ===================== ANALYSIS PAGE =====================
elif page == "Analysis":
    st.markdown('<div class="main-header">📊 Cluster Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("🔧 Cluster Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_inertia = px.line(x=results['K_range'], y=results['inertias'],
                              labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}, title='Elbow Method')
        fig_inertia.update_traces(mode='lines+markers', line_color='#1f77b4')
        st.plotly_chart(fig_inertia, use_container_width=True)
    
    with col2:
        fig_silhouette = px.line(x=results['K_range'], y=results['silhouettes'],
                                 labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'}, title='Silhouette Score')
        fig_silhouette.update_traces(mode='lines+markers', line_color='#2ca02c')
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    st.info(f"Optimal number of clusters: **{results['optimal_k']}** (based on highest silhouette score)")
    
    st.markdown("---")
    
    st.subheader("⚖️ Clustering Method Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("K-Means Silhouette Score", f"{results['kmeans_silhouette']:.4f}")
    with col2:
        st.metric("Hierarchical Silhouette Score", f"{results['hier_silhouette']:.4f}")
    
    st.markdown("---")
    
    st.subheader("🎯 PCA Cluster Visualization")
    
    df_viz = df.copy()
    df_viz['PC1'] = results['X_pca'][:, 0]
    df_viz['PC2'] = results['X_pca'][:, 1]
    df_viz['Cluster'] = results['kmeans_labels']
    
    fig_pca = px.scatter(df_viz, x='PC1', y='PC2', color='Cluster',
                          title=f'K-Means Clustering (PCA)', color_continuous_scale='Viridis')
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("👥 Cluster Profiles")
    
    cluster_profiles = df.groupby('cluster_kmeans').agg({
        'client_id': 'count', 'age': 'mean', 'annual_income': 'mean', 'property_value': 'mean',
        'satisfaction_score': 'mean', 'investment_horizon_years': 'mean', 'num_prior_investments': 'mean'
    }).round(2)
    
    cluster_profiles.columns = ['Count', 'Avg Age', 'Avg Income', 'Avg Property', 'Avg Satisfaction', 'Avg Horizon', 'Avg Prior']
    st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'), use_container_width=True)


# ===================== DASHBOARD PAGE =====================
elif page == "Dashboard":
    st.markdown('<div class="main-header">📈 Interactive Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        country_list = sorted(df['country'].unique().tolist())
        selected_countries = st.multiselect("🌍 Select Country", options=country_list, default=country_list)
    
    with col_filter2:
        segment_list = sorted(df['segment_name'].unique().tolist())
        selected_segments = st.multiselect("🎯 Select Segment", options=segment_list, default=segment_list)
    
    with col_filter3:
        client_list = sorted(df['client_type'].unique().tolist())
        selected_clients = st.multiselect("👤 Select Client Type", options=client_list, default=client_list)
    
    df_filtered = df[
        (df['country'].isin(selected_countries)) &
        (df['segment_name'].isin(selected_segments)) &
        (df['client_type'].isin(selected_clients))
    ]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Buyers", f"{len(df_filtered):,}")
    with col2:
        st.metric("Avg Satisfaction", f"{df_filtered['satisfaction_score'].mean():.2f}/5.0")
    with col3:
        st.metric("Avg Property Value", f"${df_filtered['property_value'].mean():,.0f}")
    with col4:
        st.metric("Avg Annual Income", f"${df_filtered['annual_income'].mean():,.0f}")
    with col5:
        st.metric("Avg Horizon", f"{df_filtered['investment_horizon_years'].mean():.1f} yrs")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country_counts = df_filtered['country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        fig_country = px.bar(country_counts, x='Country', y='Count', title='Buyers by Country', color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        client_counts = df_filtered['client_type'].value_counts().reset_index()
        client_counts.columns = ['Client Type', 'Count']
        fig_client = px.bar(client_counts, x='Client Type', y='Count', title='Buyers by Client Type', color='Count', color_continuous_scale='Plasma')
        st.plotly_chart(fig_client, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        purpose_counts = df_filtered['acquisition_purpose'].value_counts().reset_index()
        purpose_counts.columns = ['Purpose', 'Count']
        fig_purpose = px.bar(purpose_counts, x='Purpose', y='Count', title='Acquisition Purpose', color='Count', color_continuous_scale='Tealgrn')
        st.plotly_chart(fig_purpose, use_container_width=True)
    
    with col2:
        fig_horizon = px.box(df_filtered, x='segment_name', y='investment_horizon_years', title='Investment Horizon by Segment', color='segment_name')
        fig_horizon.update_xaxes(tickangle=45)
        st.plotly_chart(fig_horizon, use_container_width=True)
    
    st.markdown("---")
    
    fig_scatter = px.scatter(df_filtered, x='annual_income', y='property_value', color='segment_name',
                            size='satisfaction_score', hover_data=['client_id', 'country'],
                            title='Income vs Property Value by Segment', color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    satisfaction_by_segment = df_filtered.groupby('segment_name')['satisfaction_score'].mean().reset_index()
    satisfaction_by_segment.columns = ['Segment', 'Avg Satisfaction']
    satisfaction_by_segment = satisfaction_by_segment.sort_values('Avg Satisfaction', ascending=True)
    
    fig_satisfaction = px.bar(satisfaction_by_segment, x='Avg Satisfaction', y='Segment', orientation='h',
                              title='Average Satisfaction Score by Segment', color='Avg Satisfaction', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_satisfaction, use_container_width=True)


# ===================== DATA PAGE =====================
elif page == "Data":
    st.markdown('<div class="main-header">📋 Buyer Data</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    search_term = st.text_input("🔍 Search by Client ID", "")
    
    if search_term:
        df_display = df[df['client_id'].str.contains(search_term, case=False)]
    else:
        df_display = df
    
    df_formatted = df_display.copy()
    df_formatted['annual_income'] = df_formatted['annual_income'].apply(lambda x: f"${x:,.0f}")
    df_formatted['property_value'] = df_formatted['property_value'].apply(lambda x: f"${x:,.0f}")
    df_formatted['budget_range'] = df_formatted['budget_range'].apply(lambda x: f"${x:,.0f}")
    df_formatted['satisfaction_score'] = df_formatted['satisfaction_score'].round(2)
    
    st.dataframe(df_formatted, use_container_width=True, height=500)
    
    st.markdown(f"Showing {len(df_display)} of {len(df)} records")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Filtered Data (CSV)", data=csv, file_name='filtered_buyers.csv', mime='text/csv')
    
    with col2:
        summary_stats = df.groupby('segment_name').agg({
            'client_id': 'count', 'age': 'mean', 'annual_income': 'mean', 'property_value': 'mean',
            'satisfaction_score': 'mean', 'investment_horizon_years': 'mean'
        }).round(2)
        summary_csv = summary_stats.to_csv().encode('utf-8')
        st.download_button(label="📥 Download Segment Summary", data=summary_csv, file_name='segment_summary.csv', mime='text/csv')

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;"><p>Parcl Co. Limited - Real Estate Market Intelligence Platform</p><p>Powered by Machine Learning & AI</p></div>', unsafe_allow_html=True)
