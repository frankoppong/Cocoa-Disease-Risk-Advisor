import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/faostat_ghana_cocoa.csv")

# Filter for yield data only
yield_df = df[df["Element"] == "Yield"][["Year", "Value"]].copy()
yield_df.columns = ["Year", "Yield_hg_per_ha"]

# Convert to kg/ha for readability
yield_df["Yield_kg_per_ha"] = yield_df["Yield_hg_per_ha"] / 100

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(yield_df["Year"], yield_df["Yield_kg_per_ha"],
        marker="o", color="#27ae60", linewidth=2)
ax.fill_between(yield_df["Year"], yield_df["Yield_kg_per_ha"],
                alpha=0.1, color="#27ae60")
ax.set_title("Ghana Cocoa Yield 2000–2023\n(Lower yield = higher disease pressure nationally)",
             fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Yield (kg per hectare)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/ghana_cocoa_yield_trend.png", dpi=150)
plt.show()

print(yield_df.sort_values("Year").to_string(index=False))