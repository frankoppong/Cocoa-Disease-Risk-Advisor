"""
Cocoa Farm Disease Risk Model
- RandomForest classifier with hyperparameter tuning
- SHAP TreeExplainer for feature attribution
- Saves model artifacts for Gradio UI
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.pipeline import Pipeline

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/cocoa_farm_dataset.csv")

FEATURE_COLS = [c for c in df.columns if c not in ["risk_score", "high_risk"]]
X = df[FEATURE_COLS]
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Feature metadata (for human-readable labels) ─────────────────────────────
FEATURE_LABELS = {
    "rainfall_mm_annual":          "Annual Rainfall (mm)",
    "humidity_pct":                "Humidity (%)",
    "temperature_c":               "Temperature (°C)",
    "altitude_m":                  "Altitude (m)",
    "soil_ph":                     "Soil pH",
    "soil_drainage_score":         "Soil Drainage Quality",
    "farm_age_years":              "Farm Age (years)",
    "farm_size_ha":                "Farm Size (hectares)",
    "shade_tree_density":          "Shade Tree Density",
    "pesticide_sprays_per_year":   "Pesticide Sprays/Year",
    "pruning_regularity":          "Pruning Regularity",
    "fertilizer_use":              "Fertilizer Use",
    "irrigation":                  "Irrigation Used",
    "tree_density_per_ha":         "Trees per Hectare",
    "dead_tree_pct":               "Dead/Missing Trees (%)",
    "pod_infection_visible_pct":   "Visible Pod Infections (%)",
    "neighbour_farms_diseased_pct":"Nearby Farms with Disease (%)",
    "forest_distance_km":          "Distance to Forest (km)",
    "road_access":                 "Road Access",
    "farmer_age_years":            "Farmer Age",
    "extension_visits_per_year":   "Extension Agent Visits/Year",
    "coop_member":                 "Cooperative Member",
    "years_of_schooling":          "Years of Schooling",
}

# Plain-language explanations for non-expert users
FEATURE_PLAIN = {
    "rainfall_mm_annual":          "How much rain fell on your farm this year",
    "humidity_pct":                "How humid/damp the air is on your farm",
    "temperature_c":               "The average temperature on your farm",
    "altitude_m":                  "How high up your farm is located",
    "soil_ph":                     "How acidic or alkaline your soil is",
    "soil_drainage_score":         "How well water drains from your soil",
    "farm_age_years":              "How many years your cocoa trees have been growing",
    "farm_size_ha":                "The size of your farm in hectares",
    "shade_tree_density":          "How many shade trees you have on the farm",
    "pesticide_sprays_per_year":   "How many times you spray pesticides each year",
    "pruning_regularity":          "How often you prune/cut back your cocoa trees",
    "fertilizer_use":              "Whether and how much fertilizer you apply",
    "irrigation":                  "Whether you water your farm during dry seasons",
    "tree_density_per_ha":         "Number of cocoa trees per hectare",
    "dead_tree_pct":               "Percentage of trees that have died or are missing",
    "pod_infection_visible_pct":   "Percentage of cocoa pods that look diseased",
    "neighbour_farms_diseased_pct":"How many nearby farms are already showing disease",
    "forest_distance_km":          "How close your farm is to a forest",
    "road_access":                 "Whether your farm is easily reachable by road",
    "farmer_age_years":            "The farmer's age",
    "extension_visits_per_year":   "How many times an extension officer visited your farm",
    "coop_member":                 "Whether you belong to a farmers' cooperative",
    "years_of_schooling":          "Farmer's years of formal education",
}

# ── Train RandomForest ────────────────────────────────────────────────────────
print("Training model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# CV score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Test metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# ── SHAP Explainer ────────────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values_test = explainer.shap_values(X_test)
# For binary: shap_values_test[1] = values for class=1 (high risk)
if isinstance(shap_values_test, list):
    shap_hd = shap_values_test[1]
elif shap_values_test.ndim == 3:
    shap_hd = shap_values_test[:, :, 1]
else:
    shap_hd = shap_values_test

# ── Plot 1: Global SHAP Summary ───────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_hd, X_test,
    feature_names=[FEATURE_LABELS[f] for f in FEATURE_COLS],
    show=False, plot_size=None, max_display=15
)
plt.title("What drives high disease risk?\n(Global model explanation)", 
          fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png")

# ── Plot 2: Feature Importance Bar ───────────────────────────────────────────
mean_abs_shap = np.abs(shap_hd).mean(0)
feat_imp = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values(ascending=True).tail(12)

colors = ["#c0392b" if v > feat_imp.mean() else "#27ae60" for v in feat_imp.values]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(
    [FEATURE_LABELS[f] for f in feat_imp.index],
    feat_imp.values, color=colors, edgecolor="white", height=0.65
)
ax.set_xlabel("Average Impact on Disease Risk Prediction", fontsize=10)
ax.set_title("Top Factors That Affect Your Risk Score", fontsize=13, fontweight="bold")
ax.axvline(feat_imp.mean(), color="gray", linestyle="--", linewidth=0.8, label="Average")
ax.spines[["top", "right"]].set_visible(False)

red_patch = mpatches.Patch(color="#c0392b", label="High influence")
green_patch = mpatches.Patch(color="#27ae60", label="Lower influence")
ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# ── Plot 3: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"Model (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Model Accuracy: ROC Curve", fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_curve.png")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(model, "models/disease_model.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")

with open("models/feature_meta.json", "w") as f:
    json.dump({
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "feature_plain": FEATURE_PLAIN,
        "cv_auc": float(cv_scores.mean()),
        "test_auc": float(auc_val),
    }, f, indent=2)

print("\n✅ All artifacts saved.")
print(f"   Model: disease_model.pkl")
print(f"   Explainer: shap_explainer.pkl")
print(f"   Meta: feature_meta.json")
print(f"   Test AUC: {auc_val:.3f}")
