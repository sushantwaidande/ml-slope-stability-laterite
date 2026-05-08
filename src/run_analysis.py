"""
=============================================================================
AUGMENTED ML PIPELINE FOR LATERITE SLOPE STABILITY PREDICTION
=============================================================================
Title: Comparative Evaluation of Machine Learning Models for Predicting
       Monsoon-Induced Slope Failures in Lateritic Terrain

Method: Monte Carlo Simulation for Data Augmentation
        - Generates synthetic samples from multivariate normal distributions
        - Preserves inter-parameter correlations observed in field data
        - Applies physical constraints (grain size sums to ~100%, etc.)
        - Separates generation by site/parent rock type to maintain geological realism

Original Data: 10 samples (6 Ratnagiri/Basalt + 4 Karwar/Granite)
Augmented Data: 200 samples (120 Basalt + 80 Granite), preserving 60:40 ratio

Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                      LeaveOneOut, KFold, RepeatedKFold,
                                      GridSearchCV)
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# Create output directories
for d in ['data', 'figures', 'results', 'logs']:
    os.makedirs(f'/home/ubuntu/ml_pipeline/{d}', exist_ok=True)

print("=" * 80)
print("LATERITE SLOPE STABILITY - AUGMENTED ML ANALYSIS")
print("Monte Carlo Simulation + Machine Learning Pipeline")
print("=" * 80)

# ============================================================================
# SECTION 1: ORIGINAL DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: ORIGINAL FIELD DATA")
print("=" * 80)

# Site 1 - Ratnagiri (Basalt parent rock)
site1_data = pd.DataFrame({
    'Sample': ['S1-1', 'S1-2', 'S1-3', 'S1-4', 'S1-5', 'S1-6'],
    'Site': [1, 1, 1, 1, 1, 1],
    'Parent_Rock': ['Basalt', 'Basalt', 'Basalt', 'Basalt', 'Basalt', 'Basalt'],
    'Gravel_pct': [69.0, 41.0, 10.0, 72.0, 83.0, 65.0],
    'Sand_pct': [12.0, 17.0, 34.0, 12.0, 4.0, 18.0],
    'Silt_pct': [18.0, 41.0, 54.0, 16.0, 13.0, 17.0],
    'Clay_pct': [3.0, 2.0, 3.0, 0.0, 1.0, 15.0],
    'Liquid_Limit': [57.0, 57.0, 53.0, 50.0, 56.0, 47.0],
    'Plastic_Limit': [33.0, 38.0, 35.0, 35.0, 36.0, 31.0],
    'Sp_Gravity': [2.69, 2.73, 2.70, 2.75, 2.73, 2.68],
    'Moisture_pct': [10.0, 12.0, 11.0, 14.0, 15.0, 16.0],
    'Cohesion_kPa': [30.0, 40.0, 25.0, 20.0, 45.0, 39.0],
    'Friction_Angle_deg': [26.0, 36.0, 32.0, 33.0, 25.0, 24.0]
})

# Site 2 - Karwar (Granite parent rock)
site2_data = pd.DataFrame({
    'Sample': ['S2-1', 'S2-2', 'S2-3', 'S2-4'],
    'Site': [2, 2, 2, 2],
    'Parent_Rock': ['Granite', 'Granite', 'Granite', 'Granite'],
    'Gravel_pct': [0.0, 30.0, 40.0, 80.0],
    'Sand_pct': [10.0, 32.0, 20.0, 5.0],
    'Silt_pct': [71.0, 0.0, 10.0, 15.0],
    'Clay_pct': [19.0, 37.0, 30.0, 1.0],
    'Liquid_Limit': [80.0, 48.0, 30.0, 55.0],
    'Plastic_Limit': [52.0, 36.0, 40.0, 38.0],
    'Sp_Gravity': [2.78, 2.86, 2.74, 2.69],
    'Moisture_pct': [15.0, 14.0, 20.0, 10.0],
    'Cohesion_kPa': [26.0, 33.0, 32.0, 26.0],
    'Friction_Angle_deg': [32.0, 27.0, 26.0, 31.0]
})

original_data = pd.concat([site1_data, site2_data], ignore_index=True)
print(f"Original samples: {len(original_data)} ({len(site1_data)} Basalt + {len(site2_data)} Granite)")

# ============================================================================
# SECTION 2: MONTE CARLO DATA AUGMENTATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: MONTE CARLO DATA AUGMENTATION")
print("=" * 80)

def generate_monte_carlo_samples(site_data, n_samples, site_num, parent_rock, seed_offset=0):
    """
    Generate synthetic geotechnical samples using multivariate normal distribution.
    
    Method:
    1. Compute mean vector and covariance matrix from original samples
    2. Generate samples from multivariate normal distribution
    3. Apply physical constraints:
       - Grain size fractions (Gravel + Sand + Silt + Clay) normalized to ~100%
       - All percentages clipped to [0, 100]
       - Specific gravity in realistic range [2.60, 2.95]
       - Moisture content > 0
       - Cohesion > 0
       - Friction angle in [15, 45] degrees
       - Liquid Limit > Plastic Limit (Plasticity Index > 0)
    4. Add controlled noise to prevent exact replication
    """
    np.random.seed(42 + seed_offset)
    
    numeric_cols = ['Gravel_pct', 'Sand_pct', 'Silt_pct', 'Clay_pct',
                    'Liquid_Limit', 'Plastic_Limit', 'Sp_Gravity',
                    'Moisture_pct', 'Cohesion_kPa', 'Friction_Angle_deg']
    
    data_matrix = site_data[numeric_cols].values
    
    # Compute statistics
    mean_vec = data_matrix.mean(axis=0)
    cov_matrix = np.cov(data_matrix, rowvar=False)
    
    # Regularize covariance matrix (add small diagonal for numerical stability)
    cov_matrix += np.eye(len(numeric_cols)) * 0.01
    
    # Generate from multivariate normal
    synthetic = np.random.multivariate_normal(mean_vec, cov_matrix, size=n_samples)
    
    # Apply physical constraints
    synthetic_df = pd.DataFrame(synthetic, columns=numeric_cols)
    
    # Clip to physical ranges
    synthetic_df['Gravel_pct'] = synthetic_df['Gravel_pct'].clip(0, 95)
    synthetic_df['Sand_pct'] = synthetic_df['Sand_pct'].clip(0, 60)
    synthetic_df['Silt_pct'] = synthetic_df['Silt_pct'].clip(0, 80)
    synthetic_df['Clay_pct'] = synthetic_df['Clay_pct'].clip(0, 50)
    synthetic_df['Liquid_Limit'] = synthetic_df['Liquid_Limit'].clip(20, 100)
    synthetic_df['Plastic_Limit'] = synthetic_df['Plastic_Limit'].clip(15, 60)
    synthetic_df['Sp_Gravity'] = synthetic_df['Sp_Gravity'].clip(2.60, 2.95)
    synthetic_df['Moisture_pct'] = synthetic_df['Moisture_pct'].clip(5, 35)
    synthetic_df['Cohesion_kPa'] = synthetic_df['Cohesion_kPa'].clip(5, 80)
    synthetic_df['Friction_Angle_deg'] = synthetic_df['Friction_Angle_deg'].clip(15, 45)
    
    # Ensure LL > PL (Plasticity Index > 0)
    mask = synthetic_df['Liquid_Limit'] <= synthetic_df['Plastic_Limit']
    synthetic_df.loc[mask, 'Liquid_Limit'] = synthetic_df.loc[mask, 'Plastic_Limit'] + \
        np.random.uniform(5, 20, size=mask.sum())
    
    # Normalize grain size fractions to sum approximately to 100%
    grain_cols = ['Gravel_pct', 'Sand_pct', 'Silt_pct', 'Clay_pct']
    grain_sum = synthetic_df[grain_cols].sum(axis=1)
    for col in grain_cols:
        synthetic_df[col] = (synthetic_df[col] / grain_sum) * 100
    
    # Add metadata
    synthetic_df['Site'] = site_num
    synthetic_df['Parent_Rock'] = parent_rock
    synthetic_df['Sample'] = [f'MC-{parent_rock[0]}{site_num}-{i+1}' for i in range(n_samples)]
    
    return synthetic_df

# Generate synthetic samples
n_basalt = 120  # Maintain ~60% ratio
n_granite = 80  # Maintain ~40% ratio

print(f"\nGenerating {n_basalt} synthetic Basalt samples (Site 1 - Ratnagiri)...")
mc_basalt = generate_monte_carlo_samples(site1_data, n_basalt, 1, 'Basalt', seed_offset=0)

print(f"Generating {n_granite} synthetic Granite samples (Site 2 - Karwar)...")
mc_granite = generate_monte_carlo_samples(site2_data, n_granite, 2, 'Granite', seed_offset=100)

# Combine original + synthetic
augmented_data = pd.concat([original_data, mc_basalt, mc_granite], ignore_index=True)
augmented_data['Is_Original'] = [True]*10 + [False]*(n_basalt + n_granite)

# Calculate derived features
augmented_data['Plasticity_Index'] = augmented_data['Liquid_Limit'] - augmented_data['Plastic_Limit']
augmented_data['Fines_pct'] = augmented_data['Silt_pct'] + augmented_data['Clay_pct']

# Factor of Safety calculation
gamma = 18.0  # kN/m³
H = 12.0  # meters
alpha_deg = 45.0
alpha_rad = np.radians(alpha_deg)
ru_monsoon = 0.45

augmented_data['FoS_dry'] = (
    augmented_data['Cohesion_kPa'] / (gamma * H * np.sin(alpha_rad) * np.cos(alpha_rad)) +
    np.tan(np.radians(augmented_data['Friction_Angle_deg'])) / np.tan(alpha_rad)
)

augmented_data['FoS_saturated'] = (
    augmented_data['Cohesion_kPa'] / (gamma * H * np.sin(alpha_rad) * np.cos(alpha_rad)) +
    (1 - ru_monsoon) * np.tan(np.radians(augmented_data['Friction_Angle_deg'])) / np.tan(alpha_rad)
)

augmented_data['FoS'] = augmented_data['FoS_saturated']

print(f"\n--- Augmented Dataset Summary ---")
print(f"  Total samples: {len(augmented_data)}")
print(f"  Original: {augmented_data['Is_Original'].sum()}")
print(f"  Synthetic: {(~augmented_data['Is_Original']).sum()}")
print(f"  Basalt: {(augmented_data['Parent_Rock'] == 'Basalt').sum()}")
print(f"  Granite: {(augmented_data['Parent_Rock'] == 'Granite').sum()}")
print(f"\n  FoS Range: {augmented_data['FoS'].min():.4f} to {augmented_data['FoS'].max():.4f}")
print(f"  FoS Mean: {augmented_data['FoS'].mean():.4f} (±{augmented_data['FoS'].std():.4f})")
print(f"  Unstable (FoS < 1.0): {(augmented_data['FoS'] < 1.0).sum()} ({(augmented_data['FoS'] < 1.0).mean()*100:.1f}%)")
print(f"  Marginal (1.0 ≤ FoS < 1.4): {((augmented_data['FoS'] >= 1.0) & (augmented_data['FoS'] < 1.4)).sum()}")
print(f"  Stable (FoS ≥ 1.4): {(augmented_data['FoS'] >= 1.4).sum()}")

# Verify synthetic data preserves original distributions
print(f"\n--- Distribution Validation ---")
print(f"{'Parameter':<22} {'Orig Mean':<12} {'Synth Mean':<12} {'Orig Std':<10} {'Synth Std':<10}")
print("-" * 66)
key_params = ['Clay_pct', 'Moisture_pct', 'Cohesion_kPa', 'Friction_Angle_deg', 'Liquid_Limit']
for param in key_params:
    orig_mean = original_data[param].mean()
    synth_mean = augmented_data[~augmented_data['Is_Original']][param].mean()
    orig_std = original_data[param].std()
    synth_std = augmented_data[~augmented_data['Is_Original']][param].std()
    print(f"  {param:<20} {orig_mean:<12.2f} {synth_mean:<12.2f} {orig_std:<10.2f} {synth_std:<10.2f}")

# Save augmented data
augmented_data.to_csv('/home/ubuntu/ml_pipeline/data/01_augmented_dataset.csv', index=False)
original_data.to_csv('/home/ubuntu/ml_pipeline/data/01b_original_data.csv', index=False)
print("\n[SAVED] data/01_augmented_dataset.csv")
print("[SAVED] data/01b_original_data.csv")

# Distribution comparison plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
plot_params = ['Clay_pct', 'Cohesion_kPa', 'Friction_Angle_deg',
               'Moisture_pct', 'Liquid_Limit', 'FoS']
for ax, param in zip(axes.flat, plot_params):
    orig_vals = augmented_data[augmented_data['Is_Original']][param]
    synth_vals = augmented_data[~augmented_data['Is_Original']][param]
    ax.hist(synth_vals, bins=20, alpha=0.6, color='steelblue', label=f'Synthetic (n={len(synth_vals)})', density=True)
    ax.axvline(orig_vals.mean(), color='red', linewidth=2, linestyle='--', label=f'Original mean')
    for v in orig_vals:
        ax.axvline(v, color='red', linewidth=0.5, alpha=0.5)
    ax.set_xlabel(param, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title(f'{param} Distribution', fontsize=11)
plt.suptitle('Monte Carlo Augmentation: Distribution Validation\n(Red lines = original samples)', fontsize=13)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig0_mc_distribution_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] figures/fig0_mc_distribution_validation.png")

# ============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: DESCRIPTIVE STATISTICS (AUGMENTED DATASET)")
print("=" * 80)

feature_cols = ['Gravel_pct', 'Sand_pct', 'Silt_pct', 'Clay_pct',
                'Liquid_Limit', 'Plastic_Limit', 'Plasticity_Index',
                'Sp_Gravity', 'Moisture_pct', 'Cohesion_kPa',
                'Friction_Angle_deg', 'Fines_pct']

desc_stats = augmented_data[feature_cols].describe().T
desc_stats = desc_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_stats.columns = ['N', 'Mean', 'Std_Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max']
print(desc_stats.round(3).to_string())

desc_stats.to_csv('/home/ubuntu/ml_pipeline/data/02_descriptive_statistics.csv')
print("\n[SAVED] data/02_descriptive_statistics.csv")

# ============================================================================
# SECTION 4: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: DATA PREPROCESSING")
print("=" * 80)

X = augmented_data[feature_cols].copy()
y = augmented_data['FoS'].values

# 4.1 Outlier Detection
print("\n4.1 Outlier Detection (IQR 1.5x)")
outlier_log = []
for col in feature_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()
    if n_outliers > 0:
        outlier_log.append({'Feature': col, 'Lower': round(lower, 2),
                           'Upper': round(upper, 2), 'N_Outliers': n_outliers})
        X[col] = X[col].clip(lower, upper)
        print(f"  {col}: {n_outliers} outlier(s) capped")

if outlier_log:
    pd.DataFrame(outlier_log).to_csv('/home/ubuntu/ml_pipeline/data/03_outlier_log.csv', index=False)
else:
    print("  No outliers detected")

# 4.2 Label Encoding
print("\n4.2 Label Encoding")
le = LabelEncoder()
X['Site_encoded'] = le.fit_transform(augmented_data['Parent_Rock'])
print(f"  Parent_Rock: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4.3 StandardScaler
print("\n4.3 StandardScaler")
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X[feature_cols])
X_scaled = pd.DataFrame(X_scaled_array, columns=feature_cols)
X_scaled['Site_encoded'] = X['Site_encoded'].values

scaler_params = pd.DataFrame({'Feature': feature_cols, 'Mean': scaler.mean_, 'Std': scaler.scale_})
scaler_params.to_csv('/home/ubuntu/ml_pipeline/data/03b_scaler_parameters.csv', index=False)
X_scaled.to_csv('/home/ubuntu/ml_pipeline/data/04_preprocessed_scaled_data.csv', index=False)

print(f"  Scaled {len(feature_cols)} features to mean=0, std=1")
print(f"  Dataset shape: {X_scaled.shape}")
print("\n[SAVED] data/04_preprocessed_scaled_data.csv")

# ============================================================================
# SECTION 5: PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 80)

pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled[feature_cols])

pca_results = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(pca_full.explained_variance_))],
    'Eigenvalue': pca_full.explained_variance_,
    'Variance_Explained_pct': pca_full.explained_variance_ratio_ * 100,
    'Cumulative_Variance_pct': np.cumsum(pca_full.explained_variance_ratio_) * 100
})
print("\nEigenvalues and Variance Explained:")
print(pca_results.round(4).to_string(index=False))

# Kaiser criterion
n_kaiser = (pca_full.explained_variance_ > 1.0).sum()
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumvar >= 0.95) + 1 if any(cumvar >= 0.95) else len(feature_cols)
n_components = n_kaiser

print(f"\nComponent Selection:")
print(f"  Kaiser criterion: {n_kaiser} components")
print(f"  95% variance: {n_95} components")
print(f"  Selected: {n_components} components ({cumvar[n_components-1]*100:.2f}% variance)")

# Refit with selected components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled[feature_cols])

# Loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)],
                        index=feature_cols)
print(f"\nPCA Loadings:")
print(loadings.round(4).to_string())

print(f"\nPhysical Interpretation:")
for i in range(n_components):
    col = f'PC{i+1}'
    top = loadings[col].abs().nlargest(3)
    signs = ['+' if loadings[col][idx] > 0 else '-' for idx in top.index]
    print(f"  {col} ({pca.explained_variance_ratio_[i]*100:.1f}%): "
          f"{', '.join([f'{s}{idx}({abs(loadings[col][idx]):.3f})' for idx, s in zip(top.index, signs)])}")

loadings.to_csv('/home/ubuntu/ml_pipeline/data/05_pca_loadings.csv')
pca_results.to_csv('/home/ubuntu/ml_pipeline/data/06_pca_variance_explained.csv', index=False)
pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)]).to_csv(
    '/home/ubuntu/ml_pipeline/data/06b_pca_transformed_data.csv', index=False)

# Scree plot
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = range(1, len(pca_full.explained_variance_ratio_)+1)
ax.bar(x_pos, pca_full.explained_variance_ratio_*100, alpha=0.7, color='steelblue', label='Individual')
ax.plot(x_pos, cumvar*100, 'ro-', markersize=6, label='Cumulative')
ax.axhline(y=95, color='k', linestyle='--', alpha=0.5, label='95% threshold')
ax.axvline(x=n_components+0.5, color='green', linestyle='--', alpha=0.7, label=f'Selected: {n_components} PCs')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Variance Explained (%)', fontsize=12)
ax.set_title(f'PCA Scree Plot — Augmented Laterite Dataset (n={len(augmented_data)})', fontsize=13)
ax.legend()
ax.set_xticks(x_pos)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig1_pca_scree_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] figures/fig1_pca_scree_plot.png")

# ============================================================================
# SECTION 6: RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 80)

rf_rfe = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
rfe = RFE(estimator=rf_rfe, n_features_to_select=1, step=1)
rfe.fit(X_scaled[feature_cols], y)

rf_rfe.fit(X_scaled[feature_cols], y)

rfe_ranking = pd.DataFrame({
    'Feature': feature_cols,
    'RFE_Rank': rfe.ranking_,
    'RF_Importance': rf_rfe.feature_importances_
}).sort_values('RFE_Rank')

print("\nFeature Rankings (RFE + Random Forest Importance):")
print(rfe_ranking.to_string(index=False))

n_top = 5
top_features = rfe_ranking.nsmallest(n_top, 'RFE_Rank')['Feature'].tolist()
print(f"\nTop {n_top} features: {top_features}")

rfe_ranking.to_csv('/home/ubuntu/ml_pipeline/data/07_rfe_feature_rankings.csv', index=False)

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
rfe_sorted = rfe_ranking.sort_values('RF_Importance', ascending=True)
colors = ['#27ae60' if r <= n_top else '#bdc3c7' for r in rfe_sorted['RFE_Rank']]
bars = ax.barh(rfe_sorted['Feature'], rfe_sorted['RF_Importance'], color=colors, edgecolor='gray')
ax.set_xlabel('Random Forest Feature Importance', fontsize=12)
ax.set_title('Feature Importance for Factor of Safety Prediction\n(RFE with RF, 500 trees, n=210)', fontsize=13)
for bar, imp in zip(bars, rfe_sorted['RF_Importance']):
    ax.text(imp + 0.002, bar.get_y() + bar.get_height()/2, f'{imp:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] figures/fig2_feature_importance.png")

# ============================================================================
# SECTION 7: MODEL TRAINING AND CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: MODEL TRAINING AND 10-FOLD CROSS-VALIDATION")
print("=" * 80)

# Feature sets
X_all = X_scaled[feature_cols].values
X_pca_sel = X_pca[:, :n_components]
X_top5 = X_scaled[top_features].values

# 10-Fold CV (repeated 5 times for stability)
cv_10fold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_repeated = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

# Models
models = {
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                    max_depth=3, random_state=42),
    'SVM (RBF)': SVR(kernel='rbf', C=10.0, gamma='scale'),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5, weights='distance')
}

feature_sets = {
    'All Features (12)': X_all,
    f'PCA ({n_components} PCs)': X_pca_sel,
    'RFE Top-5': X_top5
}

print(f"\nEvaluation: 10-Fold CV (repeated 5x) on n={len(y)} samples")
print(f"\n{'Model':<20} {'Features':<18} {'CV_R²':<12} {'CV_R²_std':<10} {'CV_MSE':<12} {'Train_R²':<10}")
print("-" * 82)

all_results = []

for feat_name, X_feat in feature_sets.items():
    for model_name, model in models.items():
        # Repeated 10-fold CV
        r2_scores = cross_val_score(model, X_feat, y, cv=cv_repeated, scoring='r2')
        mse_scores = -cross_val_score(model, X_feat, y, cv=cv_repeated, scoring='neg_mean_squared_error')
        mae_scores = -cross_val_score(model, X_feat, y, cv=cv_repeated, scoring='neg_mean_absolute_error')
        
        # Also get predictions from single 10-fold for plotting
        y_pred_cv = cross_val_predict(model, X_feat, y, cv=cv_10fold)
        
        # Training performance
        model.fit(X_feat, y)
        train_r2 = model.score(X_feat, y)
        
        result = {
            'Model': model_name,
            'Features': feat_name,
            'CV_R2_mean': r2_scores.mean(),
            'CV_R2_std': r2_scores.std(),
            'CV_MSE_mean': mse_scores.mean(),
            'CV_MSE_std': mse_scores.std(),
            'CV_MAE_mean': mae_scores.mean(),
            'CV_MAE_std': mae_scores.std(),
            'Train_R2': train_r2,
            'CV_Predictions': y_pred_cv.tolist()
        }
        all_results.append(result)
        print(f"{model_name:<20} {feat_name:<18} {r2_scores.mean():<12.4f} {r2_scores.std():<10.4f} "
              f"{mse_scores.mean():<12.6f} {train_r2:<10.4f}")

results_df = pd.DataFrame(all_results)
results_save = results_df.drop(columns=['CV_Predictions'])
results_save.to_csv('/home/ubuntu/ml_pipeline/data/08_model_performance_cv.csv', index=False)
print(f"\n[SAVED] data/08_model_performance_cv.csv")

# Best results
print("\n\n--- Best Model per Feature Set ---")
for feat_name in feature_sets.keys():
    subset = results_df[results_df['Features'] == feat_name]
    best = subset.loc[subset['CV_R2_mean'].idxmax()]
    print(f"  {feat_name}: {best['Model']} (R²={best['CV_R2_mean']:.4f} ± {best['CV_R2_std']:.4f})")

best_idx = results_df['CV_R2_mean'].idxmax()
best_overall = results_df.loc[best_idx]
print(f"\n  OVERALL BEST: {best_overall['Model']} + {best_overall['Features']}")
print(f"    CV R² = {best_overall['CV_R2_mean']:.4f} ± {best_overall['CV_R2_std']:.4f}")
print(f"    CV MSE = {best_overall['CV_MSE_mean']:.6f}")
print(f"    CV MAE = {best_overall['CV_MAE_mean']:.6f}")
print(f"    Training R² = {best_overall['Train_R2']:.4f}")

# ============================================================================
# SECTION 8: HYPERPARAMETER TUNING (BEST MODEL)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: HYPERPARAMETER TUNING")
print("=" * 80)

# Tune the best model type
best_model_name = best_overall['Model']
best_feat_name = best_overall['Features']
best_X = feature_sets[best_feat_name]

print(f"\nTuning: {best_model_name} with {best_feat_name}")

if 'Random Forest' in best_model_name:
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base_model = RandomForestRegressor(random_state=42)
elif 'Gradient Boosting' in best_model_name:
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }
    base_model = GradientBoostingRegressor(random_state=42)
elif 'KNN' in best_model_name:
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    base_model = KNeighborsRegressor()
elif 'SVM' in best_model_name:
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }
    base_model = SVR()
else:
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    base_model = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(base_model, param_grid, cv=cv_10fold, scoring='r2',
                           n_jobs=-1, return_train_score=True)
grid_search.fit(best_X, y)

print(f"\n  Best parameters: {grid_search.best_params_}")
print(f"  Best CV R²: {grid_search.best_score_:.4f}")
print(f"  Training R²: {grid_search.best_estimator_.score(best_X, y):.4f}")

# Get tuned model predictions
tuned_model = grid_search.best_estimator_
y_pred_tuned = cross_val_predict(tuned_model, best_X, y, cv=cv_10fold)
tuned_r2 = r2_score(y, y_pred_tuned)
tuned_mse = mean_squared_error(y, y_pred_tuned)
tuned_mae = mean_absolute_error(y, y_pred_tuned)

print(f"\n  Tuned Model (10-fold CV predictions):")
print(f"    R² = {tuned_r2:.4f}")
print(f"    MSE = {tuned_mse:.6f}")
print(f"    MAE = {tuned_mae:.6f}")

# Save tuning results
tuning_results = pd.DataFrame(grid_search.cv_results_)
tuning_results.to_csv('/home/ubuntu/ml_pipeline/data/09_hyperparameter_tuning.csv', index=False)
print("\n[SAVED] data/09_hyperparameter_tuning.csv")

# ============================================================================
# SECTION 9: DECISION TREE PRUNING (INTERPRETABLE MODEL)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: DECISION TREE COST-COMPLEXITY PRUNING")
print("=" * 80)

# Use all features for interpretability
X_tree = X_scaled[feature_cols].values

dt_full = DecisionTreeRegressor(random_state=42)
path = dt_full.cost_complexity_pruning_path(X_tree, y)
ccp_alphas = path.ccp_alphas

print(f"\nPruning path: {len(ccp_alphas)} alpha values")

# Find optimal alpha via 10-fold CV
alpha_results = []
for alpha in ccp_alphas:
    dt_temp = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(dt_temp, X_tree, y, cv=cv_10fold, scoring='r2')
    dt_temp.fit(X_tree, y)
    alpha_results.append({
        'alpha': alpha, 'CV_R2_mean': scores.mean(), 'CV_R2_std': scores.std(),
        'depth': dt_temp.get_depth(), 'leaves': dt_temp.get_n_leaves(),
        'train_r2': dt_temp.score(X_tree, y)
    })

alpha_df = pd.DataFrame(alpha_results)
best_alpha_idx = alpha_df['CV_R2_mean'].idxmax()
best_alpha = alpha_df.loc[best_alpha_idx, 'alpha']

print(f"\nOptimal ccp_alpha: {best_alpha:.6f}")
print(f"  CV R²: {alpha_df.loc[best_alpha_idx, 'CV_R2_mean']:.4f} ± {alpha_df.loc[best_alpha_idx, 'CV_R2_std']:.4f}")
print(f"  Depth: {alpha_df.loc[best_alpha_idx, 'depth']}")
print(f"  Leaves: {alpha_df.loc[best_alpha_idx, 'leaves']}")

# Fit pruned tree
dt_pruned = DecisionTreeRegressor(ccp_alpha=best_alpha, random_state=42)
dt_pruned.fit(X_tree, y)

print(f"\nPruned Tree:")
print(f"  Depth: {dt_pruned.get_depth()}")
print(f"  Leaves: {dt_pruned.get_n_leaves()}")
print(f"  Training R²: {dt_pruned.score(X_tree, y):.4f}")

# Decision rules
tree_rules = export_text(dt_pruned, feature_names=feature_cols, decimals=4)
print(f"\nDecision Rules (scaled):")
print(tree_rules)

# Physical thresholds
print("\nPhysical Thresholds (original units):")
tree = dt_pruned.tree_
threshold_log = []
for node_id in range(tree.node_count):
    if tree.children_left[node_id] != tree.children_right[node_id]:
        feature_idx = tree.feature[node_id]
        threshold_scaled = tree.threshold[node_id]
        feature_name = feature_cols[feature_idx]
        threshold_original = threshold_scaled * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]
        n_samples_node = tree.n_node_samples[node_id]
        print(f"  Node {node_id}: if {feature_name} <= {threshold_original:.2f} "
              f"(n={n_samples_node})")
        threshold_log.append({
            'Node': node_id, 'Feature': feature_name,
            'Threshold_Scaled': threshold_scaled,
            'Threshold_Original': threshold_original,
            'N_Samples': n_samples_node
        })

# Leaf values
print("\nLeaf Predictions:")
for node_id in range(tree.node_count):
    if tree.children_left[node_id] == tree.children_right[node_id]:
        n_samples_node = tree.n_node_samples[node_id]
        fos_pred = tree.value[node_id][0][0]
        stability = "UNSTABLE" if fos_pred < 1.0 else "MARGINAL" if fos_pred < 1.4 else "STABLE"
        print(f"  Leaf {node_id}: FoS = {fos_pred:.4f} [{stability}] (n={n_samples_node})")

# Save
with open('/home/ubuntu/ml_pipeline/results/decision_tree_rules.txt', 'w') as f:
    f.write("PRUNED DECISION TREE FOR LATERITE SLOPE STABILITY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: n={len(augmented_data)} (10 original + {len(augmented_data)-10} Monte Carlo)\n")
    f.write(f"Optimal ccp_alpha: {best_alpha:.6f}\n")
    f.write(f"Depth: {dt_pruned.get_depth()}, Leaves: {dt_pruned.get_n_leaves()}\n")
    f.write(f"Training R²: {dt_pruned.score(X_tree, y):.4f}\n")
    f.write(f"CV R²: {alpha_df.loc[best_alpha_idx, 'CV_R2_mean']:.4f}\n\n")
    f.write("Rules (scaled features):\n")
    f.write(tree_rules + "\n\n")
    f.write("Physical Thresholds:\n")
    for t in threshold_log:
        f.write(f"  {t['Feature']} <= {t['Threshold_Original']:.2f}\n")

alpha_df.to_csv('/home/ubuntu/ml_pipeline/data/11_ccp_alpha_path.csv', index=False)
pd.DataFrame(threshold_log).to_csv('/home/ubuntu/ml_pipeline/data/11b_tree_thresholds.csv', index=False)

# Plot tree
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_pruned, feature_names=feature_cols, filled=True, rounded=True,
          ax=ax, fontsize=8, precision=3, impurity=True)
ax.set_title(f'Pruned Decision Tree for Slope Stability (FoS) Prediction\n'
             f'(α={best_alpha:.4f}, depth={dt_pruned.get_depth()}, '
             f'leaves={dt_pruned.get_n_leaves()}, CV R²={alpha_df.loc[best_alpha_idx, "CV_R2_mean"]:.3f})',
             fontsize=14)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig3_decision_tree_pruned.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SAVED] figures/fig3_decision_tree_pruned.png")

# CCP plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.errorbar(alpha_df['alpha'], alpha_df['CV_R2_mean'], yerr=alpha_df['CV_R2_std'],
             fmt='b-o', markersize=4, capsize=3)
ax1.axvline(x=best_alpha, color='r', linestyle='--', label=f'Optimal α={best_alpha:.4f}')
ax1.set_xlabel('Cost-Complexity Parameter (α)')
ax1.set_ylabel('10-Fold CV R²')
ax1.set_title('α vs. Cross-Validated R²')
ax1.legend()

ax2.plot(alpha_df['alpha'], alpha_df['depth'], 'g-s', markersize=4)
ax2.axvline(x=best_alpha, color='r', linestyle='--', label=f'Optimal α={best_alpha:.4f}')
ax2.set_xlabel('Cost-Complexity Parameter (α)')
ax2.set_ylabel('Tree Depth')
ax2.set_title('α vs. Tree Complexity')
ax2.legend()
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig4_ccp_alpha_selection.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] figures/fig4_ccp_alpha_selection.png")

# ============================================================================
# SECTION 10: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: VISUALIZATIONS")
print("=" * 80)

# Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² by feature set
ax1 = axes[0]
models_list = list(models.keys())
x = np.arange(len(models_list))
width = 0.25
for i, (feat_name, color) in enumerate(zip(feature_sets.keys(), ['#3498db', '#e74c3c', '#2ecc71'])):
    r2_vals = [results_df[(results_df['Model'] == m) & (results_df['Features'] == feat_name)]['CV_R2_mean'].values[0]
               for m in models_list]
    ax1.bar(x + i*width, r2_vals, width, label=feat_name, color=color, alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('10-Fold CV R² (mean)')
ax1.set_title('Model Comparison: CV R² by Feature Set')
ax1.set_xticks(x + width)
ax1.set_xticklabels([m.split('(')[0].strip() for m in models_list], rotation=15, ha='right')
ax1.legend(fontsize=8)
ax1.axhline(y=0, color='k', linewidth=0.5)

# Best per model (RFE Top-5)
ax2 = axes[1]
rfe_results = results_df[results_df['Features'] == 'RFE Top-5'].sort_values('CV_R2_mean', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rfe_results)))
ax2.barh(rfe_results['Model'], rfe_results['CV_R2_mean'], xerr=rfe_results['CV_R2_std'],
         color=colors, edgecolor='gray', capsize=3)
ax2.set_xlabel('10-Fold CV R² (± std)')
ax2.set_title('Model Comparison: RFE Top-5 Features')
ax2.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] figures/fig5_model_comparison.png")

# Actual vs Predicted (best model)
best_preds = np.array(best_overall['CV_Predictions'])
fig, ax = plt.subplots(figsize=(7, 7))
# Color by site
colors_scatter = ['#e74c3c' if s == 1 else '#3498db' for s in augmented_data['Site']]
ax.scatter(y, best_preds, c=colors_scatter, s=30, alpha=0.6, edgecolors='none')
# Highlight original samples
orig_mask = augmented_data['Is_Original'].values
ax.scatter(y[orig_mask], best_preds[orig_mask], c='black', s=100, marker='D',
           edgecolors='gold', linewidths=1.5, label='Original samples', zorder=5)
min_val = min(y.min(), best_preds.min()) - 0.05
max_val = max(y.max(), best_preds.max()) + 0.05
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Factor of Safety', fontsize=12)
ax.set_ylabel('Predicted Factor of Safety (CV)', fontsize=12)
ax.set_title(f'Actual vs. Predicted FoS\n{best_overall["Model"]} + {best_overall["Features"]} '
             f'(R²={best_overall["CV_R2_mean"]:.4f})', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
# Add legend for sites
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#e74c3c', label='Site 1 (Basalt)'),
                   Patch(facecolor='#3498db', label='Site 2 (Granite)'),
                   plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                              markeredgecolor='gold', markersize=10, label='Original samples')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig7_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] figures/fig7_actual_vs_predicted.png")

# Correlation heatmap
corr_features = feature_cols + ['FoS']
corr_matrix = augmented_data[corr_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax, annot_kws={'size': 8})
ax.set_title(f'Correlation Matrix — Augmented Laterite Dataset (n={len(augmented_data)})', fontsize=13)
plt.tight_layout()
plt.savefig('/home/ubuntu/ml_pipeline/figures/fig6_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SAVED] figures/fig6_correlation_heatmap.png")

# Correlation with FoS
print("\nCorrelation with FoS:")
fos_corr = corr_matrix['FoS'].drop('FoS').sort_values(key=abs, ascending=False)
for feat, corr in fos_corr.items():
    sig = "***" if abs(corr) > 0.5 else "**" if abs(corr) > 0.3 else "*" if abs(corr) > 0.2 else ""
    print(f"  {feat:<22}: r = {corr:+.4f} {sig}")

corr_matrix.to_csv('/home/ubuntu/ml_pipeline/data/10_correlation_matrix.csv')

# ============================================================================
# SECTION 11: PREPROCESSING IMPACT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: PREPROCESSING IMPACT ANALYSIS")
print("=" * 80)

print("\nR² Improvement from Preprocessing:")
print(f"{'Model':<20} {'Raw':<10} {'PCA':<10} {'RFE':<10} {'Best':<10} {'Δ(Raw→Best)':<12}")
print("-" * 72)
for model_name in models.keys():
    model_res = results_df[results_df['Model'] == model_name]
    raw = model_res[model_res['Features'] == 'All Features (12)']['CV_R2_mean'].values[0]
    pca_r2 = model_res[model_res['Features'] == f'PCA ({n_components} PCs)']['CV_R2_mean'].values[0]
    rfe_r2 = model_res[model_res['Features'] == 'RFE Top-5']['CV_R2_mean'].values[0]
    best_r2 = max(raw, pca_r2, rfe_r2)
    delta = best_r2 - raw
    print(f"  {model_name:<18} {raw:<10.4f} {pca_r2:<10.4f} {rfe_r2:<10.4f} {best_r2:<10.4f} {delta:+.4f}")

# ============================================================================
# SECTION 12: SAVE ALL PREDICTIONS AND METADATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 12: FINAL OUTPUTS")
print("=" * 80)

# Save predictions
predictions_data = augmented_data[['Sample', 'Site', 'Parent_Rock', 'Is_Original', 'FoS']].copy()
for _, row in results_df[results_df['Features'] == best_overall['Features']].iterrows():
    col_name = f"Pred_{row['Model'].replace(' ', '_').replace('(', '').replace(')', '')}"
    predictions_data[col_name] = row['CV_Predictions']
predictions_data.to_csv('/home/ubuntu/ml_pipeline/data/12_cv_predictions.csv', index=False)

# Metadata
metadata = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'python_version': '3.11',
    'dataset': {
        'original_samples': 10,
        'synthetic_samples': len(augmented_data) - 10,
        'total_samples': len(augmented_data),
        'augmentation_method': 'Monte Carlo simulation (multivariate normal)',
        'site1_basalt': int((augmented_data['Parent_Rock'] == 'Basalt').sum()),
        'site2_granite': int((augmented_data['Parent_Rock'] == 'Granite').sum())
    },
    'fos_parameters': {
        'method': 'Infinite slope with pore pressure ratio',
        'gamma_kN_m3': gamma, 'H_meters': H,
        'slope_angle_deg': alpha_deg, 'ru_monsoon': ru_monsoon
    },
    'pca': {
        'n_components': int(n_components),
        'cumulative_variance_pct': float(cumvar[n_components-1]*100)
    },
    'rfe_top5': top_features,
    'cross_validation': '10-Fold CV (repeated 5x)',
    'best_model': {
        'name': best_overall['Model'],
        'features': best_overall['Features'],
        'CV_R2': float(best_overall['CV_R2_mean']),
        'CV_R2_std': float(best_overall['CV_R2_std']),
        'CV_MSE': float(best_overall['CV_MSE_mean']),
        'Train_R2': float(best_overall['Train_R2'])
    },
    'tuned_model': {
        'best_params': grid_search.best_params_,
        'CV_R2': float(grid_search.best_score_),
        'aggregated_R2': float(tuned_r2),
        'aggregated_MSE': float(tuned_mse)
    },
    'decision_tree': {
        'ccp_alpha': float(best_alpha),
        'depth': int(dt_pruned.get_depth()),
        'leaves': int(dt_pruned.get_n_leaves()),
        'CV_R2': float(alpha_df.loc[best_alpha_idx, 'CV_R2_mean'])
    },
    'correlations_with_fos': {feat: float(corr) for feat, corr in fos_corr.items()}
}

with open('/home/ubuntu/ml_pipeline/results/analysis_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, default=str)

# Summary
print(f"\n{'='*60}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"  Dataset: {len(augmented_data)} samples (10 original + {len(augmented_data)-10} MC)")
print(f"  PCA: {n_components} components ({cumvar[n_components-1]*100:.1f}% variance)")
print(f"  RFE Top-5: {', '.join(top_features)}")
print(f"  Best Model: {best_overall['Model']} + {best_overall['Features']}")
print(f"    CV R² = {best_overall['CV_R2_mean']:.4f} ± {best_overall['CV_R2_std']:.4f}")
print(f"  Tuned Model R² = {tuned_r2:.4f}")
print(f"  Decision Tree CV R² = {alpha_df.loc[best_alpha_idx, 'CV_R2_mean']:.4f}")
print(f"{'='*60}")

with open('/home/ubuntu/ml_pipeline/results/analysis_summary.txt', 'w') as f:
    f.write(f"AUGMENTED ML ANALYSIS SUMMARY\n{'='*60}\n")
    f.write(f"Date: {metadata['analysis_date']}\n\n")
    f.write(f"Dataset: {len(augmented_data)} samples\n")
    f.write(f"  Original: 10 (6 Basalt + 4 Granite)\n")
    f.write(f"  Synthetic: {len(augmented_data)-10} (Monte Carlo, multivariate normal)\n\n")
    f.write(f"PCA: {n_components} components, {cumvar[n_components-1]*100:.2f}% variance\n")
    f.write(f"RFE Top-5: {', '.join(top_features)}\n\n")
    f.write(f"Best Model: {best_overall['Model']} + {best_overall['Features']}\n")
    f.write(f"  CV R² = {best_overall['CV_R2_mean']:.4f} ± {best_overall['CV_R2_std']:.4f}\n")
    f.write(f"  CV MSE = {best_overall['CV_MSE_mean']:.6f}\n")
    f.write(f"  Train R² = {best_overall['Train_R2']:.4f}\n\n")
    f.write(f"Tuned Model: {grid_search.best_params_}\n")
    f.write(f"  CV R² = {tuned_r2:.4f}, MSE = {tuned_mse:.6f}\n\n")
    f.write(f"Decision Tree (pruned, α={best_alpha:.4f}):\n")
    f.write(f"  Depth: {dt_pruned.get_depth()}, Leaves: {dt_pruned.get_n_leaves()}\n")
    f.write(f"  CV R² = {alpha_df.loc[best_alpha_idx, 'CV_R2_mean']:.4f}\n\n")
    f.write(f"Correlations with FoS:\n")
    for feat, corr in list(fos_corr.items())[:5]:
        f.write(f"  {feat}: r={corr:+.4f}\n")

print("\n[SAVED] All outputs to /home/ubuntu/ml_pipeline/")
print("=" * 80)
