"""
Cocoa Farm Disease Risk Dataset Simulator
Based on Ghana COCOBOD agronomic knowledge, FAO cocoa disease literature,
and West African cocoa belt conditions (Ashanti, Brong-Ahafo, Western Region).

Key diseases modeled:
- Black Pod (Phytophthora palmivora) — most destructive, rainfall-driven
- Cocoa Swollen Shoot Virus (CSSV) — insect vector-driven
- Witches' Broom (Moniliophthora perniciosa) — rare in Ghana but emerging
- Capsid damage — predisposes to fungal infections
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000


def simulate_cocoa_farms(n=N):
    # ── Geographic & Environmental ──────────────────────────────────────────
    rainfall_mm = np.random.normal(1400, 300, n).clip(600, 2400)   # annual mm
    humidity_pct = np.random.normal(78, 10, n).clip(50, 98)        # %
    temperature_c = np.random.normal(26, 2, n).clip(20, 34)        # °C
    altitude_m = np.random.normal(250, 120, n).clip(50, 700)       # metres

    # ── Farm Management ─────────────────────────────────────────────────────
    farm_age_yrs = np.random.randint(2, 45, n)                     # years since planting
    shade_tree_density = np.random.randint(0, 4, n)                # 0=none … 3=dense
    farm_size_ha = np.random.exponential(2.5, n).clip(0.5, 20)     # ha (smallholder)
    pesticide_applications = np.random.randint(0, 6, n)            # times/year
    pruning_score = np.random.randint(0, 4, n)                     # 0=never … 3=regular
    fertilizer_use = np.random.randint(0, 3, n)                    # 0=none, 1=low, 2=good
    irrigation = np.random.choice([0, 1], n, p=[0.85, 0.15])       # binary

    # ── Tree Health Indicators ───────────────────────────────────────────────
    tree_density_per_ha = np.random.normal(900, 150, n).clip(400, 1400)
    dead_tree_pct = np.random.exponential(8, n).clip(0, 50)        # % dead/missing
    pod_infection_visible = np.random.beta(2, 6, n) * 40           # % pods showing symptoms

    # ── Soil ────────────────────────────────────────────────────────────────
    soil_ph = np.random.normal(6.2, 0.7, n).clip(4.5, 8.0)
    soil_drainage = np.random.randint(0, 4, n)                     # 0=waterlogged … 3=good

    # ── Neighbourhood / Landscape ───────────────────────────────────────────
    neighbour_disease_pct = np.random.beta(3, 5, n) * 60           # % nearby farms diseased
    forest_distance_km = np.random.exponential(4, n).clip(0.1, 30) # distance to forest edge
    road_access = np.random.choice([0, 1], n, p=[0.35, 0.65])

    # ── Farmer Socioeconomic ────────────────────────────────────────────────
    farmer_age = np.random.normal(47, 12, n).clip(18, 80)
    extension_visits_yr = np.random.poisson(1.5, n)                # visits/year
    coop_member = np.random.choice([0, 1], n, p=[0.45, 0.55])
    years_of_schooling = np.random.normal(8, 4, n).clip(0, 16)

    # ── Construct Disease Risk Score (domain-expert logic) ──────────────────
    risk = np.zeros(n)

    # Black Pod drivers (rainfall + humidity + drainage)
    risk += 0.25 * ((rainfall_mm - 600) / 1800)           # high rain → high risk
    risk += 0.20 * (humidity_pct / 100)
    risk += 0.10 * (1 - soil_drainage / 3)                # poor drainage bad
    risk += 0.05 * (1 - pruning_score / 3)                # no pruning → poor air circ

    # CSSV drivers (vectors, neighbouring farms)
    risk += 0.15 * (neighbour_disease_pct / 60)
    risk += 0.05 * (dead_tree_pct / 50)

    # Farm management protection
    risk -= 0.10 * (pesticide_applications / 6)
    risk -= 0.07 * (fertilizer_use / 2)
    risk -= 0.05 * (extension_visits_yr / 6).clip(0, 1)
    risk -= 0.04 * coop_member

    # Tree age (old trees more vulnerable)
    risk += 0.06 * ((farm_age_yrs - 2) / 43)

    # Pod visible symptoms directly push up
    risk += 0.08 * (pod_infection_visible / 40)

    # Temperature sweet-spot for Phytophthora (24–28 °C)
    temp_penalty = np.where((temperature_c >= 24) & (temperature_c <= 28), 0.05, 0.0)
    risk += temp_penalty

    # Add noise
    risk += np.random.normal(0, 0.08, n)
    risk = risk.clip(0, 1)

    # Binary label: top 35% = high risk
    threshold = np.percentile(risk, 65)
    high_risk = (risk >= threshold).astype(int)

    df = pd.DataFrame({
        # Environmental
        "rainfall_mm_annual": rainfall_mm.round(0),
        "humidity_pct": humidity_pct.round(1),
        "temperature_c": temperature_c.round(1),
        "altitude_m": altitude_m.round(0),
        "soil_ph": soil_ph.round(2),
        "soil_drainage_score": soil_drainage,          # 0–3

        # Farm management
        "farm_age_years": farm_age_yrs,
        "farm_size_ha": farm_size_ha.round(2),
        "shade_tree_density": shade_tree_density,      # 0–3
        "pesticide_sprays_per_year": pesticide_applications,
        "pruning_regularity": pruning_score,           # 0–3
        "fertilizer_use": fertilizer_use,              # 0–2
        "irrigation": irrigation,

        # Tree health
        "tree_density_per_ha": tree_density_per_ha.round(0),
        "dead_tree_pct": dead_tree_pct.round(1),
        "pod_infection_visible_pct": pod_infection_visible.round(1),

        # Neighbourhood
        "neighbour_farms_diseased_pct": neighbour_disease_pct.round(1),
        "forest_distance_km": forest_distance_km.round(1),
        "road_access": road_access,

        # Socioeconomic
        "farmer_age_years": farmer_age.round(0),
        "extension_visits_per_year": extension_visits_yr,
        "coop_member": coop_member,
        "years_of_schooling": years_of_schooling.round(0),

        # Target
        "risk_score": risk.round(4),
        "high_risk": high_risk,
    })

    return df


if __name__ == "__main__":
    df = simulate_cocoa_farms()
    df.to_csv("data/cocoa_farm_dataset.csv", index=False)
    print(f"Dataset created: {df.shape}")
    print(f"High risk farms: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")
    print(df.describe().round(2))
