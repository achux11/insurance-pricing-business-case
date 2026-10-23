"""
P&C Insurance Pricing & Risk Analysis - Segment Performance and Rate Change Impact
Covers: insurance pricing, claim frequency & severity, loss cost, loss ratio by geography (region) and
        segment, risk segmentation, rate change impact on premium / loss ratio / retention, profitability.

Data: freMTPL2 (French Motor Third-Party Liability) - ~678K auto policies + claim amounts, the standard
      public dataset for P&C pricing (OpenML ids 41214 frequency, 41215 severity).

Run in Google Colab (CPU is fine, ~3-5 minutes):
    !pip install -q scikit-learn pandas
    then paste this file and run it.
Paste the RESULTS block back to Claude.
"""
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance

# ---------- 1. Load policies + claims ----------
freq = fetch_openml(data_id=41214, as_frame=True).frame
sev = fetch_openml(data_id=41215, as_frame=True).frame
freq["IDpol"] = freq["IDpol"].astype(int)
sev["IDpol"] = sev["IDpol"].astype(int)
claims = sev.groupby("IDpol")["ClaimAmount"].sum().clip(upper=200_000)      # cap large losses at 200K
df = freq.set_index("IDpol").join(claims).fillna({"ClaimAmount": 0}).reset_index()
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df = df[df["Exposure"] > 0]

# ---------- 2. Rating segments ----------
df["DriverAgeBand"] = pd.cut(df["DrivAge"], [17, 25, 35, 50, 65, 120], labels=["18-25", "26-35", "36-50", "51-65", "66+"])
df["VehAgeBand"] = pd.cut(df["VehAge"], [-1, 2, 7, 15, 100], labels=["0-2", "3-7", "8-15", "16+"])
df["BonusMalusBand"] = pd.cut(df["BonusMalus"], [0, 50, 75, 100, 300], labels=["50", "51-75", "76-100", "100+"])
df["Density_log"] = np.log(df["Density"])
features = ["DriverAgeBand", "VehAgeBand", "BonusMalusBand", "VehBrand", "VehGas", "Area", "Region"]
X = pd.get_dummies(df[features], drop_first=True, dtype=float)
X["Density_log"] = df["Density_log"]; X["VehPower"] = df["VehPower"]

tr, te = train_test_split(df.index, test_size=0.25, random_state=42)

# ---------- 3. Frequency (Poisson) x Severity (Gamma) = pure premium (expected loss cost) ----------
y_freq = df["ClaimNb"] / df["Exposure"]
fm = PoissonRegressor(alpha=1e-4, max_iter=400).fit(X.loc[tr], y_freq.loc[tr], sample_weight=df.loc[tr, "Exposure"])
has = df.index[(df["ClaimAmount"] > 0) & df.index.isin(tr)]
avg_sev = df.loc[has, "ClaimAmount"] / df.loc[has, "ClaimNb"].clip(lower=1)
sm = GammaRegressor(alpha=10, max_iter=400).fit(X.loc[has], avg_sev, sample_weight=df.loc[has, "ClaimNb"].clip(lower=1))

df["pred_freq"] = fm.predict(X); df["pred_sev"] = sm.predict(X)
df["pure_premium"] = df["pred_freq"] * df["pred_sev"]                       # expected loss per policy-year
base_dev = mean_poisson_deviance(y_freq.loc[te], np.full(len(te), np.average(y_freq.loc[tr], weights=df.loc[tr, "Exposure"])),
                                 sample_weight=df.loc[te, "Exposure"])
model_dev = mean_poisson_deviance(y_freq.loc[te], fm.predict(X.loc[te]), sample_weight=df.loc[te, "Exposure"])

# Current (simulated) rating: a flat-ish book premium = portfolio average loss cost / 0.65 target loss ratio,
# lightly adjusted by Area only -> mispriced segments show up as high/low loss ratios.
area_rel = df.groupby("Area")["pure_premium"].mean() / df["pure_premium"].mean()
df["current_premium"] = (df["pure_premium"].mean() / 0.65) * df["Area"].map(area_rel).astype(float) * df["Exposure"]
df["incurred_loss"] = df["ClaimAmount"]

# ---------- 4. Segment & state (region) performance ----------
def perf(g):
    return pd.Series({"policies": len(g), "exposure_yrs": g["Exposure"].sum(),
                      "claim_freq": g["ClaimNb"].sum() / g["Exposure"].sum(),
                      "premium": g["current_premium"].sum(), "loss": g["incurred_loss"].sum(),
                      "loss_ratio": g["incurred_loss"].sum() / g["current_premium"].sum()})
region = df.groupby("Region", observed=True).apply(perf).sort_values("loss_ratio")
driver = df.groupby("DriverAgeBand", observed=True).apply(perf)
total_lr = df["incurred_loss"].sum() / df["current_premium"].sum()

# ---------- 5. Rate change impact: move each policy toward its risk-based price (capped +/-15%) ----------
target = df["pure_premium"] * df["Exposure"] / 0.65
change = (target / df["current_premium"] - 1).clip(-0.15, 0.15)
df["new_premium"] = df["current_premium"] * (1 + change)
# Simple retention assumption: each +1% rate increase lowers renewal probability by 0.5 pts (base 85%)
df["retention_p"] = (0.85 - 0.5 * change.clip(lower=0)).clip(0, 1)
kept = df["retention_p"]
new_premium = (df["new_premium"] * kept).sum(); new_loss = (df["incurred_loss"] * kept).sum()
old_premium = (df["current_premium"] * 0.85).sum(); old_loss = (df["incurred_loss"] * 0.85).sum()

pd.concat([region.assign(level="Region"), driver.assign(level="DriverAge")]).to_csv("segment_performance.csv")

print("\n================ RESULTS ================")
print(f"Policies: {len(df):,} | Exposure: {df['Exposure'].sum():,.0f} policy-years | Claims: {int(df['ClaimNb'].sum()):,} | Regions: {df['Region'].nunique()}")
print(f"Claim frequency model: Poisson deviance {model_dev:.4f} vs {base_dev:.4f} flat baseline "
      f"({100*(1-model_dev/base_dev):.1f}% better) on {len(te):,} holdout policies")
print(f"Portfolio loss ratio (current pricing): {total_lr:.1%}")
print("\nRegions - best and worst loss ratios:")
print(region[["policies", "claim_freq", "loss_ratio"]].round(3).iloc[[0, 1, 2, -3, -2, -1]].to_string())
print("\nDriver age segments:"); print(driver[["policies", "claim_freq", "loss_ratio"]].round(3).to_string())
print(f"\nRate change (risk-based, capped +/-15%): {(change > 0).mean():.1%} of policies up, {(change < 0).mean():.1%} down")
print(f"Expected loss ratio: {old_loss/old_premium:.1%} -> {new_loss/new_premium:.1%} | "
      f"premium {new_premium/old_premium-1:+.1%} | avg retention {kept.mean():.1%}")
print("Saved segment_performance.csv (load into Tableau / Excel for charts)")
