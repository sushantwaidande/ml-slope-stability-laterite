# Methodology Documentation

## 1. Data Collection

### Study Area

The study focuses on cut slopes along the Konkan Railway alignment on the west coast of India. Two sites were selected based on differing parent rock geology:

- **Site 1 — Ratnagiri (Maharashtra):** Laterite derived from basalt parent rock. Six samples collected from cut slopes experiencing monsoon-induced failures.
- **Site 2 — Karwar (Karnataka):** Laterite derived from granite parent rock. Four samples collected from similar failure-prone cut slopes.

### Laboratory Testing Program

Each sample underwent the following tests per IS/ASTM standards:

| Test | Standard | Parameters Determined |
|------|----------|---------------------|
| Grain Size Analysis | IS 2720 Part 4 | Gravel %, Sand %, Silt %, Clay % |
| Atterberg Limits | IS 2720 Part 5 | Liquid Limit, Plastic Limit, Plasticity Index |
| Specific Gravity | IS 2720 Part 3 | Specific Gravity |
| Natural Moisture Content | IS 2720 Part 2 | Moisture Content % |
| Direct Shear Test | IS 2720 Part 13 | Cohesion (kPa), Friction Angle (degrees) |

### Derived Parameters

- **Fines Percentage** = Silt % + Clay %
- **Factor of Safety (FoS)** = Calculated using infinite slope model (see Section 2)

## 2. Factor of Safety Calculation

The infinite slope model was selected as appropriate for the shallow translational failures observed at both sites. Under saturated monsoon conditions:

```
FoS = [c' / (γ × H × sin α × cos α)] + [(1 - ru) × tan φ' / tan α]
```

### Parameters Used

| Parameter | Value | Justification |
|-----------|-------|---------------|
| γ (unit weight) | 18 kN/m³ | Average measured bulk density of laterite |
| H (cut height) | 12 m | Representative height of Konkan Railway cuts |
| α (slope angle) | 45° | Typical as-built slope angle |
| ru (pore pressure ratio) | 0.45 | Represents monsoon saturation conditions |

### FoS Interpretation

| FoS Range | Classification |
|-----------|---------------|
| < 1.0 | Unstable (failure expected) |
| 1.0 – 1.25 | Marginally stable |
| 1.25 – 1.5 | Moderately stable |
| > 1.5 | Stable |

## 3. Monte Carlo Data Augmentation

### Rationale

With only 10 field samples, ML models cannot generalize. Monte Carlo simulation generates physically realistic synthetic samples that preserve the statistical structure of the original data.

### Procedure

1. **Compute statistics:** Mean vector (μ) and covariance matrix (Σ) computed separately for each site.
2. **Generate samples:** 120 samples from Site 1 distribution, 80 from Site 2 distribution (preserving the 60:40 ratio).
3. **Enforce physical constraints:**
   - Grain size fractions (Gravel + Sand + Silt + Clay) normalized to sum to 100%
   - Liquid Limit > Plastic Limit (swap if violated)
   - Plasticity Index recalculated as LL - PL
   - All parameters clipped to physically realistic ranges
4. **Validate:** KS-tests confirm synthetic distributions are not statistically different from originals (p > 0.05).

### Physical Constraints Applied

| Parameter | Minimum | Maximum | Constraint Source |
|-----------|---------|---------|-------------------|
| Gravel % | 0 | 60 | Physical limit |
| Sand % | 5 | 70 | Observed range in laterites |
| Silt % | 5 | 50 | Observed range |
| Clay % | 5 | 50 | Observed range |
| Liquid Limit | 20 | 80 | IS classification |
| Plastic Limit | 10 | 50 | IS classification |
| Specific Gravity | 2.4 | 3.0 | Laterite range |
| Moisture Content | 5 | 35 | Field observations |
| Cohesion | 5 | 80 | Direct shear results |
| Friction Angle | 15 | 45 | Direct shear results |

## 4. Preprocessing Pipeline

### Step 1: Outlier Capping (IQR Method)

Values beyond 1.5 × IQR from Q1/Q3 are capped (not removed) to preserve sample size while reducing extreme influence.

### Step 2: Feature Scaling (StandardScaler)

All features normalized to zero mean and unit variance:
```
z = (x - μ) / σ
```

### Step 3: Principal Component Analysis (PCA)

- Kaiser criterion applied (eigenvalue > 1)
- 4 principal components retained, explaining 81.6% of total variance
- Used as an alternative feature set for model comparison

### Step 4: Recursive Feature Elimination (RFE)

- Base estimator: Random Forest (500 trees)
- Top 5 features selected: Cohesion, Friction Angle, Plastic Limit, Fines %, Plasticity Index
- Used as the primary feature set for model training

## 5. Machine Learning Models

### Models Evaluated

| Model | Key Hyperparameters | Tuning Range |
|-------|-------------------|--------------|
| Decision Tree | max_depth, ccp_alpha | 3-10, 0.001-0.1 |
| Random Forest | n_estimators, max_depth | 100-500, 5-20 |
| Gradient Boosting | n_estimators, learning_rate, max_depth | 100-500, 0.01-0.2, 3-7 |
| SVM (RBF) | C, gamma | 0.1-100, 0.001-1 |
| KNN | n_neighbors, weights | 3-15, uniform/distance |

### Evaluation Protocol

- **Primary metric:** 10-fold cross-validation, repeated 5 times (50 total folds)
- **Metrics:** R², MSE, MAE
- **Final model:** GridSearchCV on best algorithm (Gradient Boosting)

## 6. Reproducibility

All computations use `random_state=42`. Running `src/run_analysis.py` regenerates all results identically. Python version 3.11+ required with packages listed in `requirements.txt`.
