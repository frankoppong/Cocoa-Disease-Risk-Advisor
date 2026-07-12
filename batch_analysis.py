"""
Batch Farm Risk Analysis — for COCOBOD District Officers
Processes a CSV of farm records and outputs:
  - Risk scores for every farm
  - District-level SHAP summary
  - A ranked list of highest-risk farms needing urgent visits
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json, warnings
warnings.filterwarnings("ignore")
import shap

MODEL    = joblib.load("/home/claude/cocoa_risk/models/disease_model.pkl")
EXPLAINER = joblib.load("/home/claude/cocoa_risk/models/shap_explainer.pkl")
with open("/home/claude/cocoa_risk/models/feature_meta.json") as f:
    META = json.load(f)
FEATURE_COLS   = META["feature_cols"]
FEATURE_LABELS = META["feature_labels"]

# ── Load farm data ────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/cocoa_risk/data/cocoa_farm_dataset.csv")
X  = df[FEATURE_COLS]

# ── Score all farms ───────────────────────────────────────────────────────────
probs = MODEL.predict_proba(X)[:, 1]
df["risk_probability"] = probs.round(4)
df["risk_label"] = pd.cut(
    probs, bins=[0, 0.35, 0.5, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# ── SHAP for full dataset ─────────────────────────────────────────────────────
print("Computing SHAP for all farms (this may take a moment)...")
shap_vals_raw = EXPLAINER.shap_values(X)
if isinstance(shap_vals_raw, list):
    shap_all = shap_vals_raw[1]
elif shap_vals_raw.ndim == 3:
    shap_all = shap_vals_raw[:, :, 1]
else:
    shap_all = shap_vals_raw

mean_abs = np.abs(shap_all).mean(0)
feat_rank = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False)

# ── Priority list for field visits ───────────────────────────────────────────
high_risk_farms = (
    df[df["risk_label"] == "High Risk"]
    .sort_values("risk_probability", ascending=False)
    [["farm_age_years", "farm_size_ha", "risk_probability",
      "neighbour_farms_diseased_pct", "pod_infection_visible_pct",
      "pesticide_sprays_per_year"]]
    .head(20)
)

print(f"\n=== DISTRICT RISK SUMMARY ===")
print(df["risk_label"].value_counts())
print(f"\nAverage risk score: {probs.mean():.1%}")
print(f"\n=== TOP 10 FEATURES DRIVING DISTRICT RISK ===")
for feat, val in feat_rank.head(10).items():
    print(f"  {FEATURE_LABELS[feat]:<40} {val:.4f}")

print(f"\n=== TOP 20 HIGH-RISK FARMS (Prioritise for visits) ===")
print(high_risk_farms.to_string())

# ── Save enriched dataset ─────────────────────────────────────────────────────
df.to_csv("/home/claude/cocoa_risk/outputs/farms_with_risk_scores.csv", index=False)
high_risk_farms.to_csv("/home/claude/cocoa_risk/outputs/priority_visit_list.csv")
print("\n✅ Saved: farms_with_risk_scores.csv, priority_visit_list.csv")

# ── District SHAP bar chart ───────────────────────────────────────────────────
top10 = feat_rank.head(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
colors = ["#c0392b" if i >= 7 else "#e67e22" if i >= 4 else "#27ae60"
          for i in range(len(top10))]
ax.barh([FEATURE_LABELS[f] for f in top10.index], top10.values,
        color=colors, edgecolor="white", height=0.6)
ax.set_title("District Risk Drivers — What is causing disease across all farms?",
             fontweight="bold", fontsize=11)
ax.set_xlabel("Average SHAP impact on risk")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("/home/claude/cocoa_risk/outputs/district_risk_drivers.png", dpi=150, bbox_inches="tight")
print("✅ Saved: district_risk_drivers.png")
