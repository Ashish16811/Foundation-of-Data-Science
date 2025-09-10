
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# ---------- Paths ----------
try:
    BASE = Path(__file__).resolve().parents[1]
except NameError:
    BASE = Path.cwd()
DATA = BASE / "data"
OUT  = BASE / "outputs"

# ---------- Output N ----------
OUT.mkdir(exist_ok=True)
nums = []
for p in OUT.iterdir():
    m = re.match(r"output\s+(\d+)$", p.name, flags=re.I)
    if p.is_dir() and m:
        nums.append(int(m.group(1)))
out_n = (max(nums) + 1) if nums else 1
RUN   = OUT / f"Output {out_n}"
FIG   = RUN / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORANGE = "#ff7f0e"   # tab:orange
BLUE   = "#1f77b4"   # tab:blue
BLACK  = "#000000"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "text.color": BLACK,
})

# ---------- 1) Load ----------
d1 = pd.read_csv(DATA / "dataset1.csv")
d2 = pd.read_csv(DATA / "dataset2.csv")

# ---------- 2) Clean ----------
to_dt = lambda s: pd.to_datetime(s, dayfirst=True, errors="coerce")
for c in ["start_time","rat_period_start","rat_period_end","sunset_time"]:
    if c in d1: d1[c] = to_dt(d1[c])
if "time" in d2: d2["time"] = to_dt(d2["time"])

if "habit" in d1:
    d1["habit"] = d1["habit"].astype(str).str.strip().replace({"nan":"unknown"}).fillna("unknown")

for c in ["bat_landing_to_food","seconds_after_rat_arrival","risk","reward","month","hours_after_sunset"]:
    if c in d1: d1[c] = pd.to_numeric(d1[c], errors="coerce")
for c in ["month","hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"]:
    if c in d2: d2[c] = pd.to_numeric(d2[c], errors="coerce")

if "season" not in d1:
    month_to_season = {12:"summer",1:"summer",2:"summer",3:"autumn",4:"autumn",5:"autumn",
                       6:"winter",7:"winter",8:"winter",9:"spring",10:"spring",11:"spring"}
    def norm_month(m):
        if pd.isna(m): return np.nan
        m = int(m)
        return {0:12,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11}.get(m, m)
    d1["season"] = d1["month"].apply(norm_month).map(month_to_season)
else:
    d1["season"] = d1["season"].astype(str).str.lower().str.strip()

# Labels + bins
if "risk"   in d1: d1["risk_label"]   = d1["risk"].map({0:"avoidance",  1:"risk_taking"})
if "reward" in d1: d1["reward_label"] = d1["reward"].map({0:"no_reward", 1:"reward"})
if "seconds_after_rat_arrival" in d1:
    d1["since_rat_bin"] = pd.cut(
        d1["seconds_after_rat_arrival"],
        bins=[-np.inf,0,60,180,600,np.inf],
        labels=["<=0s","1-60s","61-180s","181-600s",">600s"],
        include_lowest=True
    )

# ---------- 3) Descriptives ----------
desc = RUN / "descriptives.txt"
with open(desc, "w", encoding="utf-8") as f:
    f.write(f"Output folder: {RUN.name}\n")
    f.write(f"Dataset1 shape: {d1.shape}\nDataset2 shape: {d2.shape}\n\n")
    if "risk" in d1:
        f.write("Risk proportion:\n")
        f.write(d1["risk"].value_counts(normalize=True).rename({0:"avoidance",1:"risk_taking"}).to_string())
        f.write("\n\n")
    if "reward" in d1:
        f.write("Reward proportion:\n")
        f.write(d1["reward"].value_counts(normalize=True).rename({0:"no_reward",1:"reward"}).to_string())
        f.write("\n")

def save(figpath):
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

# a) Risk-taking proportion by time since rat arrival (binned)
if {"risk_label","since_rat_bin"}.issubset(d1.columns):
    tab = d1.groupby(["since_rat_bin","risk_label"]).size().unstack().fillna(0)
    # Ensure column order: avoidance (orange), risk_taking (blue)
    cols_order = [c for c in ["avoidance","risk_taking"] if c in tab.columns]
    tab = tab[cols_order]
    prop = (tab.T / tab.sum(axis=1)).T

    ax = prop.plot(kind="bar", color=[ORANGE, BLUE], edgecolor=BLACK)
    ax.set_title("Risk-taking proportion by time since rat arrival (binned)")
    ax.set_xlabel("Time since rat arrival (bin)")
    ax.set_ylabel("Proportion")
    ax.legend(cols_order, title="risk_label", frameon=False)
    save(FIG / "risk_by_since_rat_bin.png")

# b) Reward outcome by risk behaviour (counts)
if {"risk_label","reward_label"}.issubset(d1.columns):
    tab2 = d1.groupby(["risk_label","reward_label"]).size().unstack().fillna(0)
    # Ensure order: no_reward (orange), reward (blue)
    cols_order2 = [c for c in ["no_reward","reward"] if c in tab2.columns]
    tab2 = tab2[cols_order2]

    ax = tab2.plot(kind="bar", color=[ORANGE, BLUE], edgecolor=BLACK)
    ax.set_title("Reward outcome by risk behaviour")
    ax.set_xlabel("Risk behaviour")
    ax.set_ylabel("Count")
    ax.legend(cols_order2, title="reward_label", frameon=False)
    save(FIG / "reward_by_risk.png")

# c) Hours after sunset by risk (boxplot: black outlines, BLUE median)
if {"hours_after_sunset","risk_label"}.issubset(d1.columns):
    dat = [
        d1.loc[d1["risk_label"]=="avoidance","hours_after_sunset"].dropna(),
        d1.loc[d1["risk_label"]=="risk_taking","hours_after_sunset"].dropna()
    ]
    plt.figure()
    plt.boxplot(
        dat,
        tick_labels=["avoidance","risk_taking"],
        boxprops=dict(color=BLACK),
        whiskerprops=dict(color=BLACK),
        capprops=dict(color=BLACK),
        medianprops=dict(color=BLUE, linewidth=2)
    )
    plt.title("Hours after sunset by risk behaviour")
    plt.ylabel("Hours after sunset")
    save(FIG / "hours_after_sunset_by_risk.png")

# d) Scatter: bat landings vs rat arrivals (orange ×)
if {"bat_landing_number","rat_arrival_number"}.issubset(d2.columns):
    plt.figure()
    plt.scatter(
        d2["rat_arrival_number"], d2["bat_landing_number"],
        marker="x", c=ORANGE, linewidths=1.8
    )
    plt.title("Bat landings vs Rat arrivals (30-min periods)")
    plt.xlabel("Rat arrival number")
    plt.ylabel("Bat landing number")
    save(FIG / "scatter_batlandings_vs_ratarrivals.png")

# e) Scatter: bat landings vs rat minutes (orange ×)
if {"bat_landing_number","rat_minutes"}.issubset(d2.columns):
    plt.figure()
    plt.scatter(
        d2["rat_minutes"], d2["bat_landing_number"],
        marker="x", c=ORANGE, linewidths=1.8
    )
    plt.title("Bat landings vs Rat minutes (30-min periods)")
    plt.xlabel("Rat minutes")
    plt.ylabel("Bat landing number")
    save(FIG / "scatter_batlandings_vs_ratminutes.png")

# ---------- 5) Inferential ----------
infer = RUN / "inferential.txt"
lines = []

if {"risk_label","reward_label"}.issubset(d1.columns):
    ct = pd.crosstab(d1["risk_label"], d1["reward_label"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    lines += ["Chi-square: Risk vs Reward", ct.to_string(), f"chi2={chi2:.3f}, dof={dof}, p={p:.6f}", ""]

if {"risk_label","season"}.issubset(d1.columns):
    ct = pd.crosstab(d1["risk_label"], d1["season"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    lines += ["Chi-square: Risk vs Season", ct.to_string(), f"chi2={chi2:.3f}, dof={dof}, p={p:.6f}", ""]

if {"risk","seconds_after_rat_arrival","hours_after_sunset","season"}.issubset(d1.columns):
    df = d1[["risk","seconds_after_rat_arrival","hours_after_sunset","season"]].dropna()
    if len(df) > 50 and set(df["risk"].unique()) <= {0,1}:
        m = smf.logit("risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season)", data=df).fit(disp=False)
        lines += ["Logistic Regression: risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season)", str(m.summary())]

with open(infer, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Done. Outputs saved in: {RUN}")
print(f"   Figures → {FIG}")
print(f"   Files   → {desc.name}, {infer.name}")
