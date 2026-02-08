# Telecom Customer Churn Prediction with Machine Learning Pipeline

## Project Overview

Production-ready machine learning framework for predicting customer churn in telecommunications using scikit-learn pipelines, SMOTE resampling, and gradient boosting. Achieved 90.12% AUC with 86% reduction in false positives through systematic model comparison and imbalanced data handling.

## Key Features

* **Pipeline Architecture**: End-to-end sklearn pipeline integrating preprocessing, SMOTE, and model training
* **Imbalanced Data Handling**: SMOTE (Synthetic Minority Oversampling Technique) for class balance
* **Multi-Model Comparison**: Baseline Logistic Regression vs. XGBoost gradient boosting
* **86% False Positive Reduction**: From 110 to 15 false alarms (110→15)
* **Production-Ready**: Serializable pipeline for deployment and real-time predictions

## Technical Stack

### Core Technologies

* **Machine Learning**: scikit-learn 1.0+, XGBoost 1.5+, imbalanced-learn 0.9+
* **Data Processing**: pandas, numpy
* **Visualization**: matplotlib, seaborn
* **Pipeline Framework**: sklearn.pipeline, imblearn.pipeline
* **Environment**: Python 3.8+

## Model Architecture

```
Pipeline Configuration:
├── Preprocessing
│   ├── Feature Encoding (binary categorical)
│   └── Feature Selection (drop state, area_code)
├── SMOTE Resampling
│   ├── Strategy: Oversample minority class
│   └── Random State: 42
├── Model Training
│   ├── Baseline: Logistic Regression
│   └── Production: XGBoost
│       ├── n_estimators: 200
│       ├── max_depth: 6
│       ├── learning_rate: 0.05
│       └── eval_metric: AUC
└── Evaluation
    ├── ROC-AUC Score
    ├── Confusion Matrix
    └── Precision/Recall/F1
```

## Model Performance

### Forecast Accuracy Comparison

| Model | AUC | Precision (Churn) | Recall (Churn) | F1-Score | Improvement |
|-------|-----|-------------------|----------------|----------|-------------|
| Logistic Regression | 0.7917 | 0.29 | 0.77 | 0.42 | Baseline ✓ |
| **XGBoost** | **0.9012** | **0.75** | **0.77** | **0.76** | **+13.8%** ✓ |

### Performance Highlights

* **90.12% AUC**: Excellent discriminative ability between churners and non-churners
* **75% Precision**: 3 out of 4 churn predictions are correct
* **77% Recall**: Catching 77% of actual churners
* **86% FP Reduction**: Massive decrease in false alarms (110→15 customers)

## Model Performance

### Forecast Accuracy Comparison

| Model | AUC | Precision (Churn) | Recall (Churn) | F1-Score | Improvement |
|-------|-----|-------------------|----------------|----------|-------------|
| Logistic Regression | 0.7917 | 0.29 | 0.77 | 0.42 | Baseline ✓ |
| **XGBoost** | **0.9012** | **0.75** | **0.77** | **0.76** | **+13.8%** ✓ |

### Performance Highlights

* **90.12% AUC**: Excellent discriminative ability between churners and non-churners
* **75% Precision**: 3 out of 4 churn predictions are correct
* **77% Recall**: Catching 77% of actual churners
* **86% FP Reduction**: Massive decrease in false alarms (110→15 customers)

### Confusion Matrix (XGBoost)

```
              Predicted
              No Churn  Churn
Actual  
No Churn        350      15
Churn            14      46
```

**Key Metrics:**
- True Negatives: 350 (correctly identified loyal customers)
- False Positives: 15 (wrongly flagged as churners)
- False Negatives: 14 (missed actual churners)
- True Positives: 46 (correctly caught churners)

## Methodology

### 1. Data Collection & Preprocessing

```
Dataset: Telecom Customer Churn
- Total Samples: 4,250 customers
- Features: 13
- Target: Binary (churn: yes/no)
- Imbalance Ratio: ~14.5% churners

Preprocessing Steps:
├── Drop: state, area_code
├── Encode: international_plan, voice_mail_plan
└── Target: churn
```

### 2. Imbalanced Data Handling

**Problem:** Only 14.5% of customers churn: Model bias toward majority class

**Solution:** SMOTE (Synthetic Minority Oversampling Technique)

```
Before SMOTE:
├── No Churn: 3,287 samples (85.5%)
└── Churn: 538 samples (14.5%)

After SMOTE:
├── No Churn: 3,287 samples (50%)
└── Churn: 3,287 samples (50%)
```

**Critical Implementation:**
- SMOTE applied **only** to training set
- Test set remains imbalanced (realistic distribution)
- Prevents data leakage

### 3. Model Training & Comparison

**Baseline Model:**
```python
LogisticRegression(max_iter=500, random_state=42)
```

**Production Model:**
```python
XGBClassifier(
    n_estimators=200,      # Number of boosting rounds
    max_depth=6,           # Tree depth
    learning_rate=0.05,    # Shrinkage parameter
    random_state=42,
    eval_metric='auc'
)
```

**Pipeline Integration:**
```python
ImbPipeline([
    ('encode', FunctionTransformer(encode)),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(...))
])
```

## Key Insights

### Model Comparison

The XGBoost model demonstrates substantial improvements over Logistic Regression:

1. **AUC Improvement: +13.8%** (0.7917 → 0.9012)
   - Moves from "Good" to "Excellent" classification performance
   - Better ranking of churners vs. non-churners across all thresholds

2. **Precision Surge: +159%** (0.29 → 0.75)
   - Dramatically reduces wasted retention offers
   - From 1-in-3 accurate to 3-in-4 accurate predictions

3. **Maintained Recall: 77%**
   - Both models catch the same proportion of churners
   - XGBoost achieves this with far fewer false positives

### Business Impact

**False Positive Reduction: 110 → 15 (-86%)**

Assuming:
- Retention offer cost: $100 per customer
- Average customer value: $1,200/year
- Retention success rate: 30%

**Baseline Model (Logistic Regression):**
```
True Positives: 46 churners caught
False Positives: 110 unnecessary offers
Cost: (46 + 110) × $100 = $15,600
Revenue Saved: 46 × 0.30 × $1,200 = $16,560
Net Benefit: $960
```

**XGBoost Model:**
```
True Positives: 46 churners caught
False Positives: 15 unnecessary offers
Cost: (46 + 15) × $100 = $6,100
Revenue Saved: 46 × 0.30 × $1,200 = $16,560
Net Benefit: $10,460
```

**ROI Improvement: +989%** ($960 → $10,460)

## Contact & Links

**Author**: Nitin Vinayak  
**Email**: nitinvinayak.m@gmail.com  
**LinkedIn**: [linkedin.com/in/nitin-vinayak](https://linkedin.com/in/nitin-vinayak)
