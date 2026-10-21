# 🌿 Cocoa Farm Disease Risk Advisor
> AI-powered disease risk prediction with plain-language SHAP explanations  
> Built for smallholder cocoa farmers in Ghana's Ashanti, Brong-Ahafo, and Western Regions

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model AUC](https://img.shields.io/badge/Model%20AUC-0.821-brightgreen)
![Data](https://img.shields.io/badge/Data-Synthetic%20%2B%20FAO-orange)

---

## 🌍 The Problem

Over 800,000 smallholder cocoa farmers in Ghana lose 30–40% of their yield
annually to diseases like Black Pod (*Phytophthora palmivora*) and Cocoa
Swollen Shoot Virus (CSSV). Existing prediction tools give a risk score —
but not an explanation a farmer can act on.

**This project answers the question every farmer is asking: "Why is MY farm high-risk?"**

---

## ✨ What Makes This Different

Most ML tools stop at a prediction. This one goes further:

| Standard tool | This tool |
|---|---|
| "Your risk score is 0.78" | "🔴 HIGH RISK — 78% chance of outbreak" |
| Raw feature importances | Plain-language explanations per factor |
| No guidance | Specific action steps for each risk driver |
| English only | English + Twi phrases for local context |
| No visuals | Waterfall chart showing what's helping vs hurting |

---

## 🖥️ Demo

Fill in your farm details → get an instant risk assessment with explanations:
🔴 Your Farm Risk Level: HIGH RISK — 78% probability
(Ɔbaa tumi ba — Wɔhwɛ ho)

⚠️ What is putting your farm at risk?

📌 Nearby Farms with Disease (%) — your value: 35%
“Many farms around yours are already sick. Disease spreads by wind,
rain, and insects. Coordinate spraying with neighbours and contact COCOBOD.”

📌 Annual Rainfall (mm) — your value: 1,600mm
“High rainfall creates perfect conditions for Black Pod. Spray
copper-based fungicide after every heavy rain.”

---

## 📊 Data

This project uses two data sources:

### 1. Agronomically-Grounded Synthetic Data
The model is trained on **2,000 simulated farm records** generated from:
- **COCOBOD** agronomic guidelines (Ghana Cocoa Board)
- **FAO** cocoa disease literature (Black Pod, CSSV epidemiology)
- **West African cocoa belt** documented field conditions
- **PlantVillage** feature schema reference

The simulation deliberately reflects real conditions in Ghana's cocoa belt —
feature distributions, disease logic, and risk thresholds are all grounded
in published agronomic research, not arbitrary numbers.

> **Transparency note:** This is synthetic data, clearly documented as such.
> The model architecture and explanation system are production-ready and
> designed to be retrained on real farm records the moment they become
> available — from COCOBOD district offices, MoFA Ghana, or primary
> field surveys.

### 2. FAO FAOSTAT National Yield Data
Real national cocoa yield data for Ghana (2000–2023) from
**faostat.fao.org**, used to:
- Validate that high-risk years in the model match real yield drops
- Provide national context inside the app
- Anchor the simulation to documented production trends

---

## 🏗️ Architecture
