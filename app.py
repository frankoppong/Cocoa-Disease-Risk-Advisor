"""
🌿 Cocoa Farm Disease Risk Advisor
A farmer-facing Gradio app with SHAP explanations in plain language.

Design principles:
- Non-expert language (English + Twi phrases for Ghana context)
- Visual waterfall chart per farmer
- Traffic-light risk display
- Actionable recommendations based on SHAP top drivers
"""

import gradio as gr
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Load artifacts ────────────────────────────────────────────────────────────
MODEL = joblib.load("models/disease_model.pkl")
EXPLAINER = joblib.load("models/shap_explainer.pkl")
with open("models/feature_meta.json") as f:
    META = json.load(f)

FEATURE_COLS = META["feature_cols"]
FEATURE_LABELS = META["feature_labels"]
FEATURE_PLAIN = META["feature_plain"]

# ── Advice library (plain language, action-oriented) ─────────────────────────
ADVICE = {
    "rainfall_mm_annual": {
        "high": "Your area receives high rainfall — this creates perfect conditions for Black Pod disease. Make sure you spray a copper-based fungicide regularly, especially during and after heavy rains.",
        "low": "Your rainfall is within a safe range. Keep monitoring pod health during wet seasons."
    },
    "humidity_pct": {
        "high": "High humidity encourages fungal growth. Try to improve air flow by pruning overcrowded branches and removing infected pods quickly.",
        "low": "Humidity levels look okay. Continue good farm hygiene practices."
    },
    "pod_infection_visible_pct": {
        "high": "⚠️ You have visible disease on your pods right now! Remove and bury or burn infected pods immediately — this is the single most impactful action you can take today. Do not leave them on the ground.",
        "low": "Good — few visible infections spotted. Stay alert and check your trees weekly."
    },
    "neighbour_farms_diseased_pct": {
        "high": "Many farms around yours are already sick. Disease can spread by wind, rain, and insects. This is a high-priority risk — consider coordinating spraying with your neighbours and report to your local COCOBOD officer.",
        "low": "Your neighbourhood looks relatively healthy, which is good protection for your farm."
    },
    "pesticide_sprays_per_year": {
        "low": "You are not spraying enough. COCOBOD recommends at least 4–6 sprays per season with copper-based fungicides. Under-spraying is one of the main reasons farms get Black Pod.",
        "good": "Your spraying frequency looks reasonable. Make sure timing matches rainfall peaks."
    },
    "pruning_regularity": {
        "low": "You need to prune more regularly. Pruning opens up the canopy, reduces humidity inside the tree, and makes it harder for disease to spread. Prune at least twice a year.",
        "good": "Good pruning habits — keep it up."
    },
    "fertilizer_use": {
        "low": "Healthy, well-fed trees resist disease better. Consider using fertilizer — even a small amount of NPK can make a big difference to your yield and tree health.",
        "good": "You're applying fertilizer — well done. Healthy trees = stronger defence."
    },
    "dead_tree_pct": {
        "high": "A high percentage of dead or missing trees is a warning sign. Dead trees can harbour disease. Remove stumps where possible and replant with CSSVD-tolerant varieties available from COCOBOD.",
        "low": "Your tree survival rate looks good."
    },
    "extension_visits_per_year": {
        "low": "You haven't had many visits from an extension officer. Contact your local COCOBOD district office to request a visit — they can give you free advice and sometimes free inputs.",
        "good": "Regular extension visits help — make the most of each visit by asking about new disease threats."
    },
    "soil_drainage_score": {
        "low": "Poor drainage means waterlogged soil, which dramatically increases Black Pod risk. Simple trenches along the farm edges can help drain excess water.",
        "good": "Soil drainage seems adequate."
    },
    "shade_tree_density": {
        "low": "Having some shade trees (like plantain or Gliricidia) reduces temperature swings and helps with soil health. But too many can increase humidity — aim for moderate shade.",
        "high": "Too many shade trees may be trapping humidity. Thin them out selectively."
    },
}

def get_plain_advice(feature, shap_val, feature_value):
    """Return plain language advice based on SHAP direction and value."""
    advice_map = ADVICE.get(feature, {})
    if not advice_map:
        return None
    if shap_val > 0:  # increasing risk
        return advice_map.get("high")
    else:
        return advice_map.get("low") or advice_map.get("good")


def make_waterfall_chart(shap_vals, feature_vals, feature_names, base_val, pred_prob):
    """Build a clean, farmer-friendly SHAP waterfall chart."""
    # Sort by absolute SHAP value, show top 8
    idx = np.argsort(np.abs(shap_vals))[-8:]
    sv = shap_vals[idx]
    fv = feature_vals[idx]
    fn = [feature_names[i] for i in idx]

    # Reorder: most positive at top of chart (most risk-increasing)
    order = np.argsort(sv)
    sv, fv, fn = sv[order], fv[order], [fn[i] for i in order]

    colors = ["#e74c3c" if v > 0 else "#27ae60" for v in sv]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")

    bars = ax.barh(fn, sv, color=colors, edgecolor="white", height=0.55, linewidth=0.5)
    ax.axvline(0, color="#555", linewidth=1.2, zorder=3)

    # Value labels
    for bar, val, fval in zip(bars, sv, fv):
        x = bar.get_width()
        direction = "→ increases risk" if val > 0 else "→ reduces risk"
        label = f"{fval:.1f}  {direction}"
        ax.text(
            x + (0.003 if x >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left" if x >= 0 else "right",
            fontsize=8, color="#333"
        )

    # Risk badge
    risk_color = "#e74c3c" if pred_prob >= 0.5 else "#f39c12" if pred_prob >= 0.35 else "#27ae60"
    risk_text = "HIGH RISK" if pred_prob >= 0.5 else "MEDIUM RISK" if pred_prob >= 0.35 else "LOW RISK"
    risk_pct = int(pred_prob * 100)

    ax.set_title(
        f"Why your farm scored: {risk_text} ({risk_pct}% probability)\n"
        "Red bars = factors making things worse  |  Green bars = factors helping you",
        fontsize=10.5, fontweight="bold", color="#222",
        loc="left", pad=10
    )

    ax.set_xlabel("Impact on Disease Risk  (← lower risk  |  higher risk →)", fontsize=9, color="#555")
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Add legend
    red_p = mpatches.Patch(color="#e74c3c", label="Increasing your risk")
    green_p = mpatches.Patch(color="#27ae60", label="Protecting your farm")
    ax.legend(handles=[red_p, green_p], loc="lower right", fontsize=8,
              framealpha=0.8, edgecolor="#ddd")

    plt.tight_layout(pad=1.5)
    return fig


def predict_and_explain(
    rainfall, humidity, temperature, altitude,
    soil_ph, soil_drainage,
    farm_age, farm_size, shade_density,
    pesticide_sprays, pruning, fertilizer, irrigation,
    tree_density, dead_tree_pct, pod_infection_pct,
    neighbour_disease_pct, forest_distance, road_access,
    farmer_age, extension_visits, coop_member, schooling
):
    # Build input
    input_dict = {
        "rainfall_mm_annual": rainfall,
        "humidity_pct": humidity,
        "temperature_c": temperature,
        "altitude_m": altitude,
        "soil_ph": soil_ph,
        "soil_drainage_score": soil_drainage,
        "farm_age_years": farm_age,
        "farm_size_ha": farm_size,
        "shade_tree_density": shade_density,
        "pesticide_sprays_per_year": pesticide_sprays,
        "pruning_regularity": pruning,
        "fertilizer_use": fertilizer,
        "irrigation": int(irrigation),
        "tree_density_per_ha": tree_density,
        "dead_tree_pct": dead_tree_pct,
        "pod_infection_visible_pct": pod_infection_pct,
        "neighbour_farms_diseased_pct": neighbour_disease_pct,
        "forest_distance_km": forest_distance,
        "road_access": int(road_access),
        "farmer_age_years": farmer_age,
        "extension_visits_per_year": extension_visits,
        "coop_member": int(coop_member),
        "years_of_schooling": schooling,
    }

    X_input = pd.DataFrame([input_dict])[FEATURE_COLS]

    # Predict
    prob = MODEL.predict_proba(X_input)[0, 1]
    pred = int(prob >= 0.5)

    # SHAP
    shap_vals_raw = EXPLAINER.shap_values(X_input)
    if isinstance(shap_vals_raw, list):
        sv = shap_vals_raw[1][0]
    elif shap_vals_raw.ndim == 3:
        sv = shap_vals_raw[0, :, 1]
    else:
        sv = shap_vals_raw[0]

    base_val = EXPLAINER.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    fv = X_input.values[0]
    feature_names_plain = [FEATURE_PLAIN[f] for f in FEATURE_COLS]

    # ── Risk summary text ─────────────────────────────────────────────────────
    risk_pct = int(prob * 100)
    if prob >= 0.5:
        emoji = "🔴"
        level = "HIGH RISK"
        twi = "(Ɔbaa tumi ba — Wɔhwɛ ho)"  # "Danger may come — watch out"
        summary = f"""
{emoji} **Your Farm Risk Level: {level}** — {risk_pct}% probability of disease outbreak  
{twi}

Your farm shows several warning signs that could lead to significant crop loss. 
Please read the explanations below and take action as soon as possible.
"""
    elif prob >= 0.35:
        emoji = "🟡"
        level = "MEDIUM RISK"
        twi = "(Hwɛ wɔ so — Yɛrehwɛ)"  # "Watch carefully"
        summary = f"""
{emoji} **Your Farm Risk Level: {level}** — {risk_pct}% probability of disease outbreak  
{twi}

Your farm has some risk factors. With the right actions now, you can prevent serious losses.
"""
    else:
        emoji = "🟢"
        level = "LOW RISK"
        twi = "(Meda wo ase — W'afua aye)"  # "Thank you — your farm is well managed"
        summary = f"""
{emoji} **Your Farm Risk Level: {level}** — {risk_pct}% probability of disease outbreak  
{twi}

Your farm is in relatively good condition. Keep up your current practices.
"""

    # ── Top risk factors & advice ─────────────────────────────────────────────
    top_risk_idx = np.argsort(sv)[::-1][:5]  # top 5 risk-increasing factors
    advice_lines = []
    for i in top_risk_idx:
        if sv[i] > 0.01:  # only meaningful contributions
            feat = FEATURE_COLS[i]
            plain = FEATURE_PLAIN[feat]
            advice = get_plain_advice(feat, sv[i], fv[i])
            val_str = f"{fv[i]:.1f}" if isinstance(fv[i], float) else str(int(fv[i]))
            if advice:
                advice_lines.append(f"**📌 {FEATURE_LABELS[feat]}** (your value: {val_str})\n_{advice}_\n")

    top_protective_idx = np.argsort(sv)[:3]  # top 3 protective factors
    protective_lines = []
    for i in top_protective_idx:
        if sv[i] < -0.01:
            feat = FEATURE_COLS[i]
            val_str = f"{fv[i]:.1f}" if isinstance(fv[i], float) else str(int(fv[i]))
            protective_lines.append(f"✅ **{FEATURE_LABELS[feat]}** (your value: {val_str})")

    # Build full advice text
    advice_text = summary + "\n\n---\n\n"
    if advice_lines:
        advice_text += "## ⚠️ What is putting your farm at risk?\n\n"
        advice_text += "\n".join(advice_lines)
    if protective_lines:
        advice_text += "\n\n## 💪 What is protecting your farm?\n\n"
        advice_text += "\n".join(protective_lines)

    advice_text += f"""

---
**📞 Need help?**  
Contact your nearest COCOBOD district office or call the Farmer Helpline.  
_Model accuracy: {META['test_auc']*100:.0f}% (AUC-ROC). Always verify with a field visit._
"""

    # ── Waterfall chart ───────────────────────────────────────────────────────
    fig = make_waterfall_chart(sv, fv, feature_names_plain, base_val, prob)

    return advice_text, fig


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🌿 Cocoa Farm Disease Risk Advisor",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="yellow"),
    css="""
    .risk-header { font-size: 1.4em; font-weight: bold; }
    .section-label { font-weight: 600; color: #2d6a4f; margin-top: 1em; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🌿 Cocoa Farm Disease Risk Advisor
    ### *Designed by Frank Oppong — Built for Ghanaian Cocoa Farmers*
    
    Fill in details about your farm below. The system will tell you:
    - **Whether your farm is at risk** of cocoa disease
    - **Why** — explained in simple language
    - **What to do** — practical steps you can take
    
    *Ɛyɛ ade pa sɛ wohwɛ w'afuo so — It is good to check on your farm.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🌦️ Weather & Location")
            rainfall = gr.Slider(600, 2400, value=1400, step=10, label="Annual Rainfall (mm) — How much rain per year?")
            humidity = gr.Slider(50, 98, value=78, step=1, label="Humidity (%) — How damp is the air?")
            temperature = gr.Slider(20, 34, value=26, step=0.5, label="Temperature (°C)")
            altitude = gr.Slider(50, 700, value=250, step=10, label="Altitude (metres above sea level)")

            gr.Markdown("### 🌱 Soil")
            soil_ph = gr.Slider(4.5, 8.0, value=6.2, step=0.1, label="Soil pH (6.0–6.5 is ideal for cocoa)")
            soil_drainage = gr.Radio([0, 1, 2, 3], value=2,
                label="Soil Drainage (0=waterlogged, 3=drains well)")

            gr.Markdown("### 🌳 Your Farm")
            farm_age = gr.Slider(2, 45, value=12, step=1, label="Farm Age (years since planting)")
            farm_size = gr.Slider(0.5, 20, value=2.5, step=0.5, label="Farm Size (hectares)")
            shade_density = gr.Radio([0, 1, 2, 3], value=1,
                label="Shade Trees (0=none, 3=very dense)")
            tree_density = gr.Slider(400, 1400, value=900, step=50, label="Trees per Hectare")

        with gr.Column(scale=1):
            gr.Markdown("### 💊 Pest & Disease Management")
            pesticide_sprays = gr.Slider(0, 6, value=3, step=1, label="Pesticide Sprays per Year (COCOBOD recommends 4–6)")
            pruning = gr.Radio([0, 1, 2, 3], value=1,
                label="Pruning Regularity (0=never, 3=regular)")
            fertilizer = gr.Radio([0, 1, 2], value=1,
                label="Fertilizer Use (0=none, 1=some, 2=good)")
            irrigation = gr.Checkbox(label="Do you irrigate your farm?", value=False)

            gr.Markdown("### 🍫 Tree & Pod Health")
            dead_tree_pct = gr.Slider(0, 50, value=8, step=0.5, label="Dead or Missing Trees (% of your farm)")
            pod_infection_pct = gr.Slider(0, 40, value=5, step=0.5, label="Pods Showing Disease Signs (% right now)")

            gr.Markdown("### 🏘️ Neighbourhood & Farmer")
            neighbour_disease = gr.Slider(0, 60, value=20, step=1, label="Nearby Farms with Disease (%) — Estimate")
            forest_distance = gr.Slider(0.1, 30, value=4, step=0.1, label="Distance to Nearest Forest (km)")
            road_access = gr.Checkbox(label="Does your farm have road access?", value=True)
            farmer_age = gr.Slider(18, 80, value=45, step=1, label="Your Age (years)")
            extension_visits = gr.Slider(0, 8, value=2, step=1, label="Extension Officer Visits per Year")
            coop_member = gr.Checkbox(label="Are you a member of a farmers' cooperative?", value=False)
            schooling = gr.Slider(0, 16, value=8, step=1, label="Years of Formal Schooling")

    submit_btn = gr.Button("🔍 Check My Farm Risk", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("## 📋 Your Farm Assessment")

    with gr.Row():
        with gr.Column(scale=1):
            advice_output = gr.Markdown(label="Your Risk Explanation")
        with gr.Column(scale=1):
            chart_output = gr.Plot(label="What is driving your risk? (Visual Explanation)")

    submit_btn.click(
        fn=predict_and_explain,
        inputs=[
            rainfall, humidity, temperature, altitude,
            soil_ph, soil_drainage,
            farm_age, farm_size, shade_density,
            pesticide_sprays, pruning, fertilizer, irrigation,
            tree_density, dead_tree_pct, pod_infection_pct,
            neighbour_disease, forest_distance, road_access,
            farmer_age, extension_visits, coop_member, schooling
        ],
        outputs=[advice_output, chart_output]
    )

    gr.Markdown("""
    ---
    *Disclaimer: This tool is a decision-support aid, not a replacement for expert field visits.  
    Data source: Simulated from COCOBOD agronomic knowledge and FAO cocoa disease literature.  
    Model accuracy (AUC-ROC): 82.1% on held-out test data.*
    """)


if __name__ == "__main__":
    print("🚀 Launching Cocoa Disease Risk Advisor...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
