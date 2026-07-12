---

## 🏗️ Architecture

---

## ⚙️ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cocoa-disease-risk-advisor.git
cd cocoa-disease-risk-advisor

# 2. Install dependencies
pip install scikit-learn shap gradio pandas numpy matplotlib seaborn joblib

# 3. Generate data and train model
python data/simulate_data.py
python models/train_model.py

# 4. Launch the app
python app.py
# Open http://localhost:7860 in your browser
```

---

## 🤖 Model Details

| Metric | Value |
|--------|-------|
| Algorithm | RandomForestClassifier (300 trees) |
| CV ROC-AUC | 0.821 ± 0.022 |
| Test ROC-AUC | 0.786 |
| Training samples | 1,600 |
| Features | 23 agronomic + socioeconomic |

### Top Risk Drivers (by mean SHAP value)
1. 🌧️ Annual Rainfall
2. 🌊 Soil Drainage Quality
3. 🏘️ Nearby Farms with Disease (%)
4. 💊 Pesticide Sprays per Year
5. ✂️ Pruning Regularity
6. 🌱 Farm Age
7. 💧 Humidity
8. 🌿 Fertilizer Use

---

## 🌱 Dataset

Simulated using domain knowledge from:
- **COCOBOD** agronomic guidelines (Ghana Cocoa Board)
- **FAO** cocoa disease literature
- **West African cocoa belt** field conditions
- **PlantVillage** feature schema reference

Key diseases modeled: Black Pod, CSSV, Capsid damage

To use real farm data, replace `cocoa_farm_dataset.csv` with your records
and re-run `train_model.py`.

---

## 🧠 How the Explanations Work

Raw SHAP values are meaningless to a farmer:

This system translates them into three layers:

**1. Plain language** — each feature has a farmer-friendly description  
**2. Direction-aware advice** — positive SHAP triggers specific action advice  
**3. Visual waterfall chart** — bars labelled "increases risk" or "reduces risk"

---

## 🚀 Deployment Options

**Hugging Face Spaces (free, shareable URL):**
- Upload all files
- Set `app.py` as entry point
- Add `requirements.txt` with the pip packages

**Offline tablet for field use:**
```bash
python app.py --server_name 0.0.0.0 --server_port 7860
```

**Docker:**
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

---

## 🔮 Extension Ideas

| Feature | Description |
|---------|-------------|
| 📸 Photo input | Add pod image classifier using PlantVillage CNN |
| 📱 SMS alerts | Send risk warnings via Africa's Talking API |
| 🗣️ Voice output | Text-to-speech in Twi/Akan via Google TTS |
| 📊 District dashboard | Aggregate SHAP across farms for COCOBOD officers |
| 📱 Offline mobile | Export to ONNX; run in Android without internet |

---

## 📄 License
Fran Oppong - free for agricultural and research use.

## 🙏 Acknowledgments
COCOBOD, FAO, PlantVillage, SHAP (Lundberg & Lee 2017), scikit-learn, Gradio.
