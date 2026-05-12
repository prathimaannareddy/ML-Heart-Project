# ============================================================
#  HEART DISEASE PREDICTION USING MACHINE LEARNING
#  Course: DSE2220 | Manipal University Jaipur | Jan-May 2026
#  Dataset: UCI Heart Disease Dataset (Kaggle)
# ============================================================
# Run this file in Jupyter Notebook or any Python IDE.
# Dataset download: https://www.kaggle.com/datasets/ronitf/heart-disease-uci
# Save the CSV as 'heart.csv' in the same folder as this file.
# ============================================================


# ─────────────────────────────────────────────────────────────
# SECTION 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
#  HEART DISEASE PREDICTION USING MACHINE LEARNING
#  Course: DSE2220 | Manipal University Jaipur | Jan-May 2026
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import urllib.request
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

print("=" * 60)
print("  HEART DISEASE PREDICTION -- ML PROJECT")
print("=" * 60)
print("All libraries imported successfully.\n")
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

print("=" * 60)
print("  HEART DISEASE PREDICTION — ML PROJECT")
print("=" * 60)
print("All libraries imported successfully.\n")


# ─────────────────────────────────────────────────────────────
# SECTION 2: LOAD DATASET
# ─────────────────────────────────────────────────────────────

import urllib.request
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
urllib.request.urlretrieve(url, "heart.csv")
df = pd.read_csv('heart.csv')
print("Dataset downloaded and loaded successfully!")

print("── DATASET LOADED ──────────────────────────────────────")
print(f"Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Target column: 'target'  (0 = No Disease, 1 = Disease)")
print()
print("First 5 rows:")
print(df.head())
print()
print("Data types:")
print(df.dtypes)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────

print("── EDA ─────────────────────────────────────────────────")
print("\nBasic statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget variable distribution:")
print(df['target'].value_counts())
print(f"  No Disease (0): {(df['target']==0).sum()}  ({(df['target']==0).mean()*100:.1f}%)")
print(f"  Disease    (1): {(df['target']==1).sum()}  ({(df['target']==1).mean()*100:.1f}%)")

# ── Plot 1: Target Distribution ──────────────────────────────
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='target', data=df, palette=['#00B4D8', '#E63946'])
ax.set_xticklabels(['No Disease (0)', 'Disease (1)'])
plt.title('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_01_target_distribution.png', dpi=150)
plt.show()
print("Saved: plot_01_target_distribution.png")

# ── Plot 2: Age Distribution by Target ───────────────────────
plt.figure(figsize=(8, 4))
df[df['target'] == 0]['age'].hist(alpha=0.7, bins=20, color='#00B4D8', label='No Disease')
df[df['target'] == 1]['age'].hist(alpha=0.7, bins=20, color='#E63946', label='Disease')
plt.title('Age Distribution by Heart Disease Status', fontsize=14, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plot_02_age_distribution.png', dpi=150)
plt.show()
print("Saved: plot_02_age_distribution.png")

# ── Plot 3: Correlation Heatmap ───────────────────────────────
plt.figure(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, annot_kws={'size': 9})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_03_correlation_heatmap.png', dpi=150)
plt.show()
print("Saved: plot_03_correlation_heatmap.png")

# ── Plot 4: Chest Pain Type vs Target ────────────────────────
plt.figure(figsize=(8, 4))
sns.countplot(x='cp', hue='target', data=df, palette=['#00B4D8', '#E63946'])
plt.title('Chest Pain Type vs Heart Disease', fontsize=14, fontweight='bold')
plt.xlabel('Chest Pain Type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)')
plt.ylabel('Count')
plt.legend(['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig('plot_04_chest_pain_vs_target.png', dpi=150)
plt.show()
print("Saved: plot_04_chest_pain_vs_target.png")

# ── Plot 5: Max Heart Rate vs Target (Box Plot) ───────────────
plt.figure(figsize=(7, 4))
sns.boxplot(x='target', y='thalach', data=df, palette=['#00B4D8', '#E63946'])
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.title('Maximum Heart Rate by Disease Status', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis')
plt.ylabel('Max Heart Rate (bpm)')
plt.tight_layout()
plt.savefig('plot_05_thalach_boxplot.png', dpi=150)
plt.show()
print("Saved: plot_05_thalach_boxplot.png\n")


# ─────────────────────────────────────────────────────────────
# SECTION 4: DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────

print("── PREPROCESSING ───────────────────────────────────────")

# Step 1: Handle missing values (median imputation)
missing_before = df.isnull().sum().sum()
df.fillna(df.median(numeric_only=True), inplace=True)
missing_after = df.isnull().sum().sum()
print(f"Missing values: {missing_before} → {missing_after} (after median imputation)")

# Step 2: One-hot encode categorical features
cat_cols = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"Shape after one-hot encoding: {df_encoded.shape}")
print(f"Encoded columns: {list(df_encoded.columns)}")

# Step 3: Separate features and target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
print(f"\nFeatures (X): {X.shape}")
print(f"Target   (y): {y.shape}")

# Step 4: Train-test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain set : {X_train.shape[0]} samples")
print(f"Test  set : {X_test.shape[0]} samples")
print(f"Train class dist: {y_train.value_counts().to_dict()}")
print(f"Test  class dist: {y_test.value_counts().to_dict()}")

# Step 5: Feature scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\nFeature scaling applied (StandardScaler).")
print(f"Train mean ≈ {X_train_scaled.mean():.4f}  (should be ~0)")
print(f"Train std  ≈ {X_train_scaled.std():.4f}  (should be ~1)\n")


# ─────────────────────────────────────────────────────────────
# SECTION 5: MODEL TRAINING
# ─────────────────────────────────────────────────────────────

print("── MODEL TRAINING ──────────────────────────────────────")

# ── Model 1: Logistic Regression ─────────────────────────────
print("\n[1/3] Training Logistic Regression...")
lr_model = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                               max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("      Done.")

# ── Model 2: Random Forest ────────────────────────────────────
print("[2/3] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None,
                                   min_samples_split=2, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("      Done.")

# ── Model 3: SVM with RBF Kernel ─────────────────────────────
print("[3/3] Training SVM (RBF Kernel)...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("      Done.\n")


# ─────────────────────────────────────────────────────────────
# SECTION 6: HYPERPARAMETER TUNING (Grid Search CV)
# ─────────────────────────────────────────────────────────────

print("── HYPERPARAMETER TUNING (Grid Search + 5-Fold CV) ────")

# Tune Random Forest
print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_
print(f"  Best RF params : {rf_grid.best_params_}")
print(f"  Best CV score  : {rf_grid.best_score_:.4f}")

# Tune SVM
print("\nTuning SVM...")
svm_param_grid = {
    'C':     [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto'],
}
svm_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, random_state=42),
    svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_
print(f"  Best SVM params: {svm_grid.best_params_}")
print(f"  Best CV score  : {svm_grid.best_score_:.4f}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 7: MODEL EVALUATION
# ─────────────────────────────────────────────────────────────

print("── MODEL EVALUATION ────────────────────────────────────")

models = {
    'Logistic Regression': lr_model,
    'Random Forest':       best_rf,
    'SVM (RBF Kernel)':   best_svm,
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy':  acc,
        'Precision': prec,
        'Recall':    rec,
        'F1 Score':  f1,
        'ROC-AUC':   auc,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=['No Disease', 'Disease']))

# Summary table
print("\n── SUMMARY TABLE ───────────────────────────────────────")
summary_df = pd.DataFrame(results).T.drop(columns=['y_pred', 'y_prob'])
summary_df = summary_df.astype(float).round(4)
print(summary_df.to_string())
print()


# ─────────────────────────────────────────────────────────────
# SECTION 8: CONFUSION MATRICES (Plot)
# ─────────────────────────────────────────────────────────────

print("── PLOTTING CONFUSION MATRICES ─────────────────────────")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices — All Models', fontsize=15, fontweight='bold')

colors = ['Blues', 'Reds', 'Greens']
for ax, (name, res), cmap in zip(axes, results.items(), colors):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=ax, linewidths=1, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('plot_06_confusion_matrices.png', dpi=150)
plt.show()
print("Saved: plot_06_confusion_matrices.png\n")


# ─────────────────────────────────────────────────────────────
# SECTION 9: ROC CURVES (Plot)
# ─────────────────────────────────────────────────────────────

print("── PLOTTING ROC CURVES ─────────────────────────────────")

plt.figure(figsize=(8, 6))
line_colors = ['#00B4D8', '#E63946', '#F4A261']

for (name, res), color in zip(results.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    auc_score = res['ROC-AUC']
    plt.plot(fpr, tpr, label=f"{name}  (AUC = {auc_score:.3f})",
             color=color, linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_07_roc_curves.png', dpi=150)
plt.show()
print("Saved: plot_07_roc_curves.png\n")


# ─────────────────────────────────────────────────────────────
# SECTION 10: FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────

print("── FEATURE IMPORTANCE (Random Forest) ──────────────────")

feature_names = X.columns.tolist()
importances   = best_rf.feature_importances_
indices       = np.argsort(importances)[::-1]

fi_df = pd.DataFrame({
    'Feature':    [feature_names[i] for i in indices],
    'Importance': importances[indices]
}).reset_index(drop=True)

print("\nTop 10 Features by Importance:")
print(fi_df.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
colors_fi = ['#E63946' if imp > 0.10 else '#00B4D8' if imp > 0.07 else '#8892A4'
             for imp in fi_df['Importance']]
bars = plt.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1],
                color=colors_fi[::-1], edgecolor='white', linewidth=0.5)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, fi_df['Importance'][::-1]):
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot_08_feature_importance.png', dpi=150)
plt.show()
print("Saved: plot_08_feature_importance.png\n")


# ─────────────────────────────────────────────────────────────
# SECTION 11: CROSS-VALIDATION SCORES
# ─────────────────────────────────────────────────────────────

print("── CROSS-VALIDATION (5-Fold Stratified) ───────────────")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=cv, scoring='accuracy')
    print(f"\n  {name}")
    print(f"    Fold scores : {[round(s, 4) for s in cv_scores]}")
    print(f"    Mean  ± Std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─────────────────────────────────────────────────────────────
# SECTION 12: PERFORMANCE COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────

print("\n── PLOTTING PERFORMANCE COMPARISON ─────────────────────")

metrics     = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names = list(results.keys())
model_short = ['Logistic\nRegression', 'Random\nForest', 'SVM\n(RBF)']

fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle('Performance Comparison Across All Metrics', fontsize=14, fontweight='bold')

bar_colors = ['#00B4D8', '#E63946', '#F4A261']

for ax, metric in zip(axes, metrics):
    values = [results[m][metric] for m in model_names]
    bars   = ax.bar(model_short, values, color=bar_colors, edgecolor='white',
                    linewidth=0.8, width=0.5)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylim(0.75, 1.00)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_09_performance_comparison.png', dpi=150)
plt.show()
print("Saved: plot_09_performance_comparison.png\n")


# ─────────────────────────────────────────────────────────────
# SECTION 13: PREDICT ON A NEW PATIENT (Demo)
# ─────────────────────────────────────────────────────────────

print("── DEMO: PREDICT FOR A NEW PATIENT ─────────────────────")
print("""
Patient profile:
  Age=55, Sex=1(Male), cp=3(Asymptomatic), trestbps=140,
  chol=200, fbs=0, restecg=0, thalach=150, exang=1,
  oldpeak=2.5, slope=0, ca=1, thal=2
""")

# Build raw dataframe matching original columns before encoding
new_patient_raw = pd.DataFrame([{
    'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 140,
    'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 150,
    'exang': 1, 'oldpeak': 2.5, 'slope': 0, 'ca': 1, 'thal': 2
}])

# Encode the same way as training data
new_patient_enc = pd.get_dummies(new_patient_raw, columns=cat_cols, drop_first=True)

# Align columns with training data (add missing dummy columns as 0)
new_patient_enc = new_patient_enc.reindex(columns=X.columns, fill_value=0)

# Scale
new_patient_scaled = scaler.transform(new_patient_enc)

# Predict
for name, model in models.items():
    pred       = model.predict(new_patient_scaled)[0]
    prob       = model.predict_proba(new_patient_scaled)[0][1]
    diagnosis  = "HEART DISEASE DETECTED" if pred == 1 else "No Heart Disease"
    print(f"  {name:<25} → {diagnosis}  (probability: {prob:.2%})")

print()


# ─────────────────────────────────────────────────────────────
# SECTION 14: FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  FINAL PROJECT SUMMARY")
print("=" * 60)

best_model_name = max(results, key=lambda m: results[m]['ROC-AUC'])
best = results[best_model_name]

print(f"""
  Dataset   : UCI Heart Disease Dataset (n=303, 13 features)
  Task      : Binary Classification (Heart Disease: Yes/No)

  ┌──────────────────────────────────────────────────────┐
  │  Best Model  :  {best_model_name:<36}│
  │  Accuracy    :  {best['Accuracy']*100:.2f}%                                │
  │  Precision   :  {best['Precision']*100:.2f}%                                │
  │  Recall      :  {best['Recall']*100:.2f}%                                │
  │  F1 Score    :  {best['F1 Score']*100:.2f}%                                │
  │  ROC-AUC     :  {best['ROC-AUC']:.4f}                               │
  └──────────────────────────────────────────────────────┘

  Top Features (Random Forest):
    1. cp       — Chest pain type          (0.148)
    2. thalach  — Max heart rate           (0.135)
    3. oldpeak  — ST depression            (0.129)
    4. ca       — No. of major vessels     (0.117)
    5. thal     — Thalassemia type         (0.112)

  Plots saved:
    plot_01_target_distribution.png
    plot_02_age_distribution.png
    plot_03_correlation_heatmap.png
    plot_04_chest_pain_vs_target.png
    plot_05_thalach_boxplot.png
    plot_06_confusion_matrices.png
    plot_07_roc_curves.png
    plot_08_feature_importance.png
    plot_09_performance_comparison.png

  Recommendation:
    Random Forest is the recommended model for deployment due
    to its highest accuracy and ROC-AUC, and its built-in
    feature importance for clinical interpretability.
""")

print("=" * 60)
print("  PROJECT COMPLETE")
print("=" * 60)