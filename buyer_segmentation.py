"""
Machine Learning based Buyer Segmentation and Investment Profiling 
for Real Estate Market Intelligence

Author: Senior Data Scientist
Organization: Parcl Co. Limited
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: DATA GENERATION (Synthetic Data)
# ============================================================
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

# Introduce missing values
missing_indices_age = np.random.choice(df.index, 30, replace=False)
missing_indices_satisfaction = np.random.choice(df.index, 25, replace=False)
df.loc[missing_indices_age, 'age'] = np.nan
df.loc[missing_indices_satisfaction, 'satisfaction_score'] = np.nan

# Introduce inconsistent labels
df.loc[10, 'client_type'] = 'individual'
df.loc[20, 'client_type'] = 'CORPORATE'
df.loc[30, 'country'] = 'usa'
df.loc[40, 'country'] = 'U.S.A.'

df.to_csv('real_estate_buyer_data.csv', index=False)
print(f"Generated {n_samples} buyer records with synthetic data")
print(df.head())

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
def clean_data(df):
    df_clean = df.copy()
    
    # Normalize categorical labels (lowercase, strip whitespace)
    categorical_cols = ['client_type', 'acquisition_purpose', 'country', 'region', 'property_type', 'risk_tolerance']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
    
    # Handle missing values
    # Numerical: fill with median
    numerical_cols = ['age', 'satisfaction_score', 'annual_income', 'budget_range', 
                      'property_value', 'investment_horizon_years', 'num_prior_investments']
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Categorical: fill with mode
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    return df_clean

df_clean = clean_data(df)
print(f"\nData cleaned: {df_clean.isnull().sum().sum()} missing values remaining")

# ============================================================
# STEP 3: FEATURE ENGINEERING (One-Hot Encoding)
# ============================================================
def engineer_features(df):
    df_eng = df.copy()
    
    categorical_features = ['client_type', 'acquisition_purpose', 'country']
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_eng, columns=categorical_features, drop_first=False)
    
    # Keep numerical features as-is (will be scaled later)
    numerical_features = ['age', 'satisfaction_score', 'annual_income', 'budget_range',
                         'property_value', 'investment_horizon_years', 'num_prior_investments']
    
    return df_encoded, numerical_features

df_encoded, numerical_features = engineer_features(df_clean)

# Store original columns for dashboard
original_columns = df_clean.columns.tolist()

print(f"Features after encoding: {df_encoded.shape[1]}")

# ============================================================
# STEP 4: SCALING (StandardScaler)
# ============================================================
def scale_features(df, numerical_cols):
    df_scaled = df.copy()
    
    scaler = StandardScaler()
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_scaled, scaler

df_scaled, scaler = scale_features(df_encoded, numerical_features)

# Prepare features for clustering - only use numeric columns
feature_columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
X = df_scaled[feature_columns].values

print(f"Clustering features shape: {X.shape}")

# ============================================================
# STEP 5: K-MEANS CLUSTERING with Elbow Method & Silhouette
# ============================================================
def find_optimal_clusters(X, max_k=10):
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
    
    return K_range, inertias, silhouettes

K_range, inertias, silhouettes = find_optimal_clusters(X)

# Plot Elbow Method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14)
plt.grid(True, alpha=0.3)

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score for Optimal k', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_optimization.png', dpi=150, bbox_inches='tight')
plt.close()

# Find optimal k based on silhouette score
optimal_k = list(K_range)[np.argmax(silhouettes)]
print(f"\nOptimal number of clusters (based on Silhouette Score): {optimal_k}")
print(f"Silhouette Scores: {dict(zip(K_range, [round(s, 3) for s in silhouettes]))}")

# Final K-Means with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
kmeans_labels = kmeans_final.fit_predict(X)

# ============================================================
# STEP 6: HIERARCHICAL CLUSTERING
# ============================================================
def perform_hierarchical_clustering(X, n_clusters):
    hier_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    hier_labels = hier_clustering.fit_predict(X)
    return hier_labels

hier_labels = perform_hierarchical_clustering(X, optimal_k)

# Compare silhouette scores
kmeans_silhouette = silhouette_score(X, kmeans_labels)
hier_silhouette = silhouette_score(X, hier_labels)

print(f"\nK-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Hierarchical Silhouette Score: {hier_silhouette:.4f}")

# ============================================================
# STEP 7: CLUSTER PROFILING
# ============================================================
df_clean['cluster_kmeans'] = kmeans_labels
df_clean['cluster_hierarchical'] = hier_labels

def profile_clusters(df, cluster_col):
    profiles = {}
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'avg_age': cluster_data['age'].mean(),
            'avg_income': cluster_data['annual_income'].mean(),
            'avg_satisfaction': cluster_data['satisfaction_score'].mean(),
            'avg_property_value': cluster_data['property_value'].mean(),
            'avg_investment_horizon': cluster_data['investment_horizon_years'].mean(),
            'client_type_distribution': cluster_data['client_type'].value_counts().head(3).to_dict(),
            'acquisition_purpose_distribution': cluster_data['acquisition_purpose'].value_counts().head(3).to_dict(),
            'country_distribution': cluster_data['country'].value_counts().head(3).to_dict(),
            'risk_tolerance_distribution': cluster_data['risk_tolerance'].value_counts().to_dict()
        }
        profiles[f'Cluster_{cluster}'] = profile
    
    return profiles

cluster_profiles = profile_clusters(df_clean, 'cluster_kmeans')

# Assign meaningful names to clusters
def assign_cluster_names(profiles):
    cluster_names = {}
    for cluster_id, profile in profiles.items():
        avg_income = profile['avg_income']
        avg_horizon = profile['avg_investment_horizon']
        client_types = profile['client_type_distribution']
        investment_purpose = profile['acquisition_purpose_distribution']
        
        # Heuristic naming based on characteristics
        if avg_income > 300000 and 'investment_rental' in str(investment_purpose).lower():
            cluster_names[cluster_id] = 'Global High-Net-Worth Investors'
        elif 'institutional' in str(client_types).lower():
            cluster_names[cluster_id] = 'Institutional Investors'
        elif 'first_time' in str(investment_purpose).lower() or 'primary_residence' in str(investment_purpose).lower():
            cluster_names[cluster_id] = 'First-Time Home Buyers'
        elif avg_horizon > 10:
            cluster_names[cluster_id] = 'Long-Term Portfolio Builders'
        elif avg_income > 200000:
            cluster_names[cluster_id] = 'Affluent Property Seekers'
        elif 'corporate' in str(client_types).lower():
            cluster_names[cluster_id] = 'Corporate Buyers'
        else:
            cluster_names[cluster_id] = 'Standard Retail Clients'
    
    return cluster_names

cluster_names = assign_cluster_names(cluster_profiles)
df_clean['segment_name'] = df_clean['cluster_kmeans'].map(cluster_names)

# Print cluster profiles
print("\n" + "="*80)
print("CLUSTER PROFILES")
print("="*80)
for cluster_id, profile in cluster_profiles.items():
    print(f"\n{cluster_id}: {cluster_names.get(cluster_id, 'Unknown')}")
    print(f"  Size: {profile['size']} ({profile['percentage']:.1f}%)")
    print(f"  Avg Age: {profile['avg_age']:.1f}")
    print(f"  Avg Income: ${profile['avg_income']:,.0f}")
    print(f"  Avg Satisfaction: {profile['avg_satisfaction']:.2f}")
    print(f"  Avg Property Value: ${profile['avg_property_value']:,.0f}")
    print(f"  Avg Investment Horizon: {profile['avg_investment_horizon']:.1f} years")
    print(f"  Top Client Types: {profile['client_type_distribution']}")
    print(f"  Top Acquisition Purposes: {profile['acquisition_purpose_distribution']}")
    print(f"  Top Countries: {profile['country_distribution']}")

# ============================================================
# STEP 8: VISUALIZATION
# ============================================================
# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('K-Means Clustering Visualization', fontsize=14)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='plasma', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
plt.title('Hierarchical Clustering Visualization', fontsize=14)

plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Dendrogram for hierarchical clustering
plt.figure(figsize=(14, 6))
linkage_matrix = linkage(X[:100], method='ward')  # Sample for visualization
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
plt.xlabel('Sample Index / (Cluster Size)', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# STEP 9: EXPORT RESULTS
# ============================================================
df_clean.to_csv('segmented_buyers.csv', index=False)
print("\n" + "="*80)
print("RESULTS EXPORTED")
print("="*80)
print("1. cluster_optimization.png - Elbow Method & Silhouette Scores")
print("2. cluster_visualization.png - PCA-based Cluster Visualization")
print("3. dendrogram.png - Hierarchical Clustering Dendrogram")
print("4. segmented_buyers.csv - Segmented Buyer Data")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Total Buyers: {len(df_clean)}")
print(f"Optimal Clusters: {optimal_k}")
print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Hierarchical Silhouette Score: {hier_silhouette:.4f}")
print("\nSegment Distribution:")
print(df_clean['segment_name'].value_counts())
