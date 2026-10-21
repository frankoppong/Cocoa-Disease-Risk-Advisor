# 🌿 Cocoa Farm Disease Risk Advisor
> Predicting and explaining cocoa disease risk for smallholder farmers in Ghana  
> Built with Python · scikit-learn · SHAP · Gradio

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model AUC](https://img.shields.io/badge/Model%20AUC-0.821-brightgreen)
![Data](https://img.shields.io/badge/Data-Synthetic%20%2B%20FAO-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview
Ghana is the world's second-largest cocoa producer, with over 800,000 smallholder
farmers depending on cocoa for their livelihoods. Every year, diseases like
**Black Pod** (*Phytophthora palmivora*) and **Cocoa Swollen Shoot Virus (CSSV)**
destroy 30–40% of the national harvest, costing farmers hundreds of millions
of dollars in lost income.

The problem is not just detection. It is **explanation**.

Existing tools can tell a farmer their farm is high risk. None of them can tell
the farmer *why* — in language they understand, with actions they can take today.

This project was built to close that gap.

---

## 🎯 What This Project Does

This tool takes 23 farm-level inputs — weather, soil, management practices,
tree health, and neighbourhood conditions — and returns:

- A **risk level** (High / Medium / Low) with a probability score
- A **plain-language explanation** of exactly what is driving the risk
- A **visual waterfall chart** showing which factors are helping or hurting
- **Specific action steps** tailored to each risk driver
- **Twi language phrases** alongside English for Ghanaian farmers
- A **district-level batch tool** for COCOBOD extension officers

---

## 🖥️ Example Output
🔴 Your Farm Risk Level: HIGH RISK — 78% probability
(Ɔbaa tumi ba — Wɔhwɛ ho)

Your farm shows several warning signs that could lead to significant
crop loss. Please read the explanations below and take action as
soon as possible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ What is putting your farm at risk?

📌 Nearby Farms with Disease (%) — your value: 35%
Many farms around yours are already sick. Disease can spread by
wind, rain, and insects. Coordinate spraying with your neighbours
and report to your local COCOBOD officer.

📌 Annual Rainfall (mm) — your value: 1,600mm
Your area receives high rainfall — this creates perfect conditions
for Black Pod disease. Spray a copper-based fungicide regularly,
especially during and after heavy rains.

📌 Pesticide Sprays per Year — your value: 2
You are not spraying enough. COCOBOD recommends 4–6 sprays per
season. Under-spraying is one of the main reasons farms get Black Pod.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💪 What is protecting your farm?

✅ Cooperative Member — Yes
✅ Soil Drainage Quality — Score: 3 (drains well)


---

## 🧠 The Core Innovation: SHAP for Non-Experts

SHAP (SHapley Additive exPlanations) is a powerful technique for explaining
machine learning predictions. But raw SHAP output looks like this:

humidity_pct → SHAP value: +0.087
pesticide_sprays → SHAP value: -0.043



That means nothing to a farmer in the Ashanti Region.

This project translates every SHAP value through four layers:

| Layer | What it does |
|-------|-------------|
| **Plain language names** | `humidity_pct` becomes "How humid the air is on your farm" |
| **Direction-aware advice** | Positive SHAP triggers specific action recommendations |
| **Visual waterfall chart** | Bars labelled "increases risk" or "reduces risk" |
| **Cultural framing** | Twi phrases alongside English for local context |

The result is an explanation any farmer can understand and act on —
without needing to know what machine learning is.

---

## 📊 Data Sources

### 1. Agronomically-Grounded Synthetic Data (Training)

The model is trained on **2,000 simulated farm records** built from:

- **COCOBOD** agronomic guidelines — Ghana Cocoa Board official recommendations
- **FAO** cocoa disease literature — Black Pod and CSSV epidemiology
- **West African cocoa belt** field conditions — Ashanti, Brong-Ahafo, Western Region
- **PlantVillage** dataset — feature schema and disease class reference

The simulation is not arbitrary. Every feature distribution, risk weight,
and threshold was set to match documented agronomic reality. The model
is designed to be retrained on real farm records the moment they become
available — from COCOBOD district offices, MoFA Ghana, or primary field surveys.

> **Transparency:** This project uses synthetic training data, clearly
> documented as such. This is standard practice in agricultural ML research
> where farm-level records are difficult to access. The methodology is sound
> and publication-ready.

### 2. FAO FAOSTAT National Yield Data (Validation)

Real national cocoa yield data for Ghana (2000–2023) downloaded from
**faostat.fao.org**, used to:

- Validate that high-risk years in the model match real national yield drops
- Provide historical context inside the Gradio app
- Anchor the simulation to documented production trends

---

## 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Number of trees | 300 |
| Cross-validation ROC-AUC | 0.821 ± 0.022 |
| Test ROC-AUC | 0.786 |
| Test Accuracy | 73% |
| Training samples | 1,600 |
| Test samples | 400 |
| Features | 23 |
| Positive class (High Risk) | 35% of dataset |

### Top Risk Drivers by Mean SHAP Value

| Rank | Feature | SHAP Impact |
|------|---------|-------------|
| 1 | Annual Rainfall (mm) | 0.0899 |
| 2 | Soil Drainage Quality | 0.0689 |
| 3 | Nearby Farms with Disease (%) | 0.0474 |
| 4 | Cooperative Membership | 0.0350 |
| 5 | Pesticide Sprays per Year | 0.0341 |
| 6 | Farm Age (years) | 0.0280 |
| 7 | Humidity (%) | 0.0237 |
| 8 | Fertilizer Use | 0.0219 |

---

## 🏗️ Project Structure

```
cocoa_risk/
├── data/
│   ├── simulate_data.py            # Generates synthetic dataset
│   ├── cocoa_farm_dataset.csv      # 2,000 simulated farm records
│   └── faostat_ghana_cocoa.csv     # Real FAO national yield data
│
├── models/
│   ├── train_model.py              # Training pipeline
│   ├── disease_model.pkl           # Trained RandomForest model
│   ├── shap_explainer.pkl          # SHAP TreeExplainer
│   └── feature_meta.json           # Feature labels and metadata
│
├── outputs/
│   ├── shap_summary.png            # Global SHAP beeswarm plot
│   ├── feature_importance.png      # Mean SHAP bar chart
│   ├── roc_curve.png               # Model ROC curve
│   └── ghana_cocoa_yield_trend.png # FAO yield trend chart
│
├── app.py                          # Gradio farmer-facing UI
├── faostat_analysis.py             # FAO yield analysis script
├── batch_analysis.py


---

## ⚙️ How to Run It

### Requirements

scikit-learn,
shap,
gradio,
pandas,
numpy,
matplotlib,
seaborn,
joblib



Install everything at once:
```bash
pip install scikit-learn shap gradio pandas numpy matplotlib seaborn joblib
```

### Step by step

**1. Generate the training data**
```bash
python data/simulate_data.py
```

**2. Train the model**
```bash
python models/train_model.py
```

**3. Run the FAO yield analysis (optional)**
```bash
python faostat_analysis.py
```

**4. Launch the farmer app**
```bash
python app.py
```
Open **http://localhost:7860** in your browser.

**5. Run batch district analysis (optional)**
```bash
python batch_analysis.py
```

---

## 🚀 Deployment

### Hugging Face Spaces (free, public URL)
1. Create a free account at **huggingface.co**
2. Click **New Space** → choose **Gradio**
3. Upload all project files
4. Add a `requirements.txt` file with the package names above
5. Your app gets a public URL you can share instantly

### Offline tablet for field use
```bash
python app.py --server_name 0.0.0.0 --server_port 7860
```
Any device on the same WiFi network can open the app in a browser.

### Docker
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install scikit-learn shap gradio pandas numpy matplotlib seaborn joblib
CMD ["python", "app.py"]
```

---

## ⚠️ Limitations

- Training data is synthetic — the model has not yet been validated on
  real farm survey data
- National FAO figures cannot substitute for farm-level records
- Model accuracy of 73% reflects honest uncertainty — this is a
  decision-support tool, not a diagnosis
- Twi phrases are illustrative — a full translation requires
  native speaker review
- Always use alongside expert field visits, never as a replacement

---

## 🔮 Roadmap

| Feature | Status |
|---------|--------|
| Synthetic data training | ✅ Complete |
| SHAP plain-language explanations | ✅ Complete |
| FAO yield trend integration | ✅ Complete |
| Gradio farmer UI | ✅ Complete |
| District batch scoring tool | ✅ Complete |
| Real COCOBOD farm records | 🔄 Seeking access |
| PlantVillage image classifier for pod photos | 📋 Planned |
| Full Twi language translation | 📋 Planned |
| SMS risk alerts via Africa's Talking API | 📋 Planned |
| Offline Android app via ONNX export | 📋 Planned |
| Multi-region support (Côte d'Ivoire, Cameroon) | 📋 Planned |

---

## 👥 Who This Is For

**Farmers** — get a plain-language explanation of your farm's disease risk
and specific steps to reduce it, in English or Twi.

**Extension officers** — use the batch scoring tool to identify which farms
in your district need urgent visits, ranked by risk probability.

**Researchers** — the model architecture, SHAP explanation pipeline, and
simulation methodology are documented and reproducible. Ready to be
retrained on real data.



## 🙏 Acknowledgments

- **COCOBOD** — Ghana Cocoa Board agronomic guidelines
- **FAO FAOSTAT** — national yield data
- **PlantVillage** — disease dataset and feature reference
- **SHAP** — Lundberg & Lee, 2017 (*A Unified Approach to Interpreting
  Model Predictions*)
- **scikit-learn** — machine learning framework
- **Gradio** — farmer-facing UI framework
- **NASA POWER** — climate data reference

---

*Built to help Ghana's cocoa farmers protect their harvest.*  
*Wɔyɛ adwuma pa mma akuafo — Good work for farmers.*
