# Comparative Evaluation of Machine Learning Models for Predicting Monsoon-Induced Slope Failures in Lateritic Terrain

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)

## Abstract

The stability of cut slopes in lateritic terrain presents a significant challenge for infrastructure projects, particularly during intense monsoon seasons. This study presents a machine learning framework to predict the Factor of Safety (FoS) of laterite slopes along the Konkan Railway alignment in India. To overcome the limitations of a small field dataset (n=10), a Monte Carlo simulation was employed to generate a physically constrained, augmented dataset (n=210) that preserves the multivariate statistical distributions of the original field samples. A comprehensive preprocessing pipeline including StandardScaler, Principal Component Analysis (PCA), and Recursive Feature Elimination (RFE) was applied. Five machine learning algorithms—Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)—were evaluated using repeated 10-fold cross-validation. The results demonstrate that Gradient Boosting combined with RFE feature selection achieved the highest predictive performance (R² = 0.929, MSE = 0.0004).

## Repository Structure

```
ml-slope-stability-laterite/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/                     # Original field measurements (n=10)
│   │   └── 01b_original_data.csv
│   ├── augmented/               # Monte Carlo augmented dataset (n=210)
│   │   └── 01_augmented_dataset.csv
│   └── processed/               # Analysis outputs
│       ├── 02_descriptive_statistics.csv
│       ├── 05_pca_loadings.csv
│       ├── 07_rfe_feature_rankings.csv
│       ├── 08_model_performance_cv.csv
│       ├── 09_hyperparameter_tuning.csv
│       └── 12_cv_predictions.csv
├── src/
│   └── run_analysis.py          # Complete reproducible pipeline
├── notebooks/
│   └── exploration.ipynb        # Interactive exploration notebook
├── figures/                     # Publication-quality figures (300 DPI)
│   ├── fig0_mc_distribution_validation.png
│   ├── fig1_pca_scree_plot.png
│   ├── fig2_feature_importance.png
│   ├── fig3_decision_tree_pruned.png
│   ├── fig4_ccp_alpha_selection.png
│   ├── fig5_model_comparison.png
│   ├── fig6_correlation_heatmap.png
│   └── fig7_actual_vs_predicted.png
├── paper/
│   └── paper/Comparative Evaluation of Machine Learning Models for Pre-dicting Monsoon-Induced Slope Failures in Lateritic Terrain A Data Augmentation Approach.docx
└── docs/
    └── METHODOLOGY.md           # Detailed methodology documentation
```

## Key Results

| Model | Features | CV R² | CV MSE | Train R² |
|-------|----------|-------|--------|----------|
| Gradient Boosting | RFE Top-5 | 0.929 ± 0.033 | 0.00036 | 0.997 |
| KNN (k=5) | RFE Top-5 | 0.906 ± 0.040 | 0.00048 | 1.000 |
| Random Forest | RFE Top-5 | 0.845 ± 0.067 | 0.00077 | 0.953 |
| Decision Tree | RFE Top-5 | 0.668 ± 0.264 | 0.00151 | 0.915 |
| SVM (RBF) | RFE Top-5 | 0.347 ± 0.322 | 0.00297 | 0.509 |

**Top-5 Features (RFE):** Cohesion, Friction Angle, Plastic Limit, Fines Percentage, Plasticity Index

## Quick Start

### Prerequisites

```bash
python >= 3.11
pip install -r requirements.txt
```

### Run the Full Analysis

```bash
# Clone the repository
git clone https://github.com/sushantwaidande/ml-slope-stability-laterite.git
cd ml-slope-stability-laterite

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python src/run_analysis.py
```

This will:
1. Load the original field data (n=10)
2. Generate Monte Carlo augmented samples (n=210)
3. Apply preprocessing (IQR outlier capping, StandardScaler)
4. Perform PCA and RFE feature selection
5. Train and evaluate 5 ML models with 10-fold CV (5 repeats)
6. Perform hyperparameter tuning via GridSearchCV
7. Generate all figures and output CSVs

### Reproducibility

All random seeds are fixed (`random_state=42`) for exact reproducibility. Running the script will regenerate all results, figures, and data files from scratch.

## Methodology

### Data Collection

Laterite soil samples were collected from two sites along the Konkan Railway alignment:
- **Site 1 (Ratnagiri):** Basalt parent rock, n=6 samples
- **Site 2 (Karwar):** Granite parent rock, n=4 samples

Laboratory tests determined 12 parameters: gravel, sand, silt, clay percentages; liquid limit; plastic limit; plasticity index; specific gravity; moisture content; cohesion; friction angle; and fines percentage.

### Factor of Safety Calculation

The target variable (FoS) was calculated using the infinite slope model under saturated monsoon conditions:

```
FoS = [c' / (γ × H × sin α × cos α)] + [(1 - ru) × tan φ' / tan α]
```

Where:
- c' = effective cohesion (kPa)
- γ = unit weight (18 kN/m³)
- H = cut height (12 m)
- α = slope angle (45°)
- ru = pore pressure ratio (0.45, representing monsoon saturation)

### Monte Carlo Data Augmentation

To overcome the small sample limitation, 200 synthetic samples were generated from multivariate normal distributions defined by the mean vectors and covariance matrices of the site-specific data. Physical constraints were enforced:
- Grain size fractions normalized to sum to 100%
- Liquid limit constrained to exceed plastic limit
- All parameters clipped to realistic geotechnical ranges

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{author2026slope,
  title={Comparative Evaluation of Machine Learning Models for Predicting Monsoon-Induced Slope Failures in Lateritic Terrain: A Data Augmentation Approach},
  author={[Author Name]},
  journal={Geosciences},
  volume={16},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research was conducted as part of the Doctor of Engineering program at The George Washington University, School of Engineering and Applied Science.
