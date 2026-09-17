"""
bbb_step3.py — Blood-brain barrier, powering up the comparison.

Ryan Yeung. Step 3 of the BBB project.

Step 2 gave two results that were not what they appeared to be:

  - Fingerprint RF (0.907) vs 4-descriptor LR (0.860) LOOKED like a clear
    win, but a paired t-test over 5 seeds gave p = 0.083 with a 95%
    confidence interval of -0.010 to +0.103. It crossed zero. Not shown.

  - The similar/dissimilar split used the MEDIAN, giving two coarse halves.
    Quartiles show the trend properly.

This script fixes both:

  1. 20 seeds instead of 5. A power calculation on the observed effect size
     said ~10 seeds would give 80% power, so 20 is comfortable.

  2. Proper paired testing: t-test, Wilcoxon, and a bootstrap confidence
     interval, all on seed-by-seed differences.

  3. Similarity in QUARTILES, pooled across all seeds, with bootstrap
     confidence intervals on each.

  4. Everything written to CSV so you have the underlying numbers when you
     come to write this up, and two PNG figures.

Runtime: roughly 5 to 15 minutes. No GPU.

    conda activate bbb
    python bbb_step3.py
"""

import numpy as np
import pandas as pd

from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, DataStructs
from rdkit import RDLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from scipy import stats
import matplotlib
matplotlib.use("Agg")          # write files, don't try to open a window
import matplotlib.pyplot as plt

RDLogger.DisableLog("rdApp.*")

SEEDS = list(range(1, 21))     # 20 seeds
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(0)
DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD"]

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# CACHING
#
# Step 2 recomputed every fingerprint on every seed, which is 20x wasteful
# here. Each molecule's features never change, so compute once and store in
# a dictionary keyed by SMILES. This is the single biggest speed-up
# available and costs three small functions.
# ---------------------------------------------------------------------------
_fp_array_cache = {}
_fp_object_cache = {}
_desc_cache = {}


def build_caches(all_smiles):
    """Featurise every unique molecule in the dataset exactly once."""
    for smiles in set(all_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _fp_array_cache[smiles] = None
            _fp_object_cache[smiles] = None
            _desc_cache[smiles] = None
            continue
        _fp_array_cache[smiles] = fpgen.GetFingerprintAsNumPy(mol)
        _fp_object_cache[smiles] = fpgen.GetFingerprint(mol)
        _desc_cache[smiles] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
        ]


def prepare(sub_df):
    """Pull cached features for one split partition.

    Returns fingerprint matrix, descriptor matrix, fingerprint objects and
    labels, with unparseable molecules dropped consistently from all four.
    """
    fps_arr, fps_obj, descs, labels = [], [], [], []
    for smiles, label in zip(sub_df["Drug"], sub_df["Y"]):
        if _fp_array_cache[smiles] is None:
            continue
        fps_arr.append(_fp_array_cache[smiles])
        fps_obj.append(_fp_object_cache[smiles])
        descs.append(_desc_cache[smiles])
        labels.append(label)
    return (np.array(fps_arr), np.array(descs), fps_obj, np.array(labels))


def bootstrap_auroc_ci(y_true, y_prob, n=N_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap confidence interval for AUROC.

    Resample the data with replacement n times, recompute AUROC each time,
    and take the 2.5th and 97.5th percentiles. Makes no assumption that the
    sampling distribution is normal, which matters for a bounded statistic
    like AUROC sitting near 0.9.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = []
    for _ in range(n):
        idx = RNG.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:      # resample must contain both classes
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return (np.nan, np.nan)
    return (float(np.percentile(scores, 100 * alpha / 2)),
            float(np.percentile(scores, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
print("=" * 72)
print("LOADING AND CACHING")
print("=" * 72)

data = ADME(name="BBB_Martins")
full_df = data.get_data(format="df")
build_caches(full_df["Drug"].tolist())
print(f"\nCached features for {len(_fp_array_cache)} unique structures.")

per_seed = []          # one row per seed
pooled = []            # one row per test molecule per seed
coefficients = []

print(f"\nRunning {len(SEEDS)} seeds. This is the slow part.\n")

for seed in SEEDS:
    split = data.get_split(method="scaffold", seed=seed)
    Xtr, Dtr, FPtr, ytr = prepare(split["train"])
    Xte, Dte, FPte, yte = prepare(split["test"])

    # --- model 1: fingerprint random forest ---
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ).fit(Xtr, ytr)
    rf_prob = rf.predict_proba(Xte)[:, 1]
    rf_auroc = roc_auc_score(yte, rf_prob)

    # --- model 2: four-descriptor logistic regression ---
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(Dtr, ytr)
    lr_prob = lr.predict_proba(Dte)[:, 1]
    lr_auroc = roc_auc_score(yte, lr_prob)
    coefficients.append(lr.named_steps["logisticregression"].coef_[0])

    # --- nearest-training-neighbour similarity for every test molecule ---
    max_sim = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, FPtr)) for fp in FPte
    ])

    per_seed.append({
        "seed": seed,
        "n_train": len(ytr),
        "n_test": len(yte),
        "test_positive_rate": float(yte.mean()),
        "rf_auroc": rf_auroc,
        "lr_auroc": lr_auroc,
        "difference": rf_auroc - lr_auroc,
        "median_max_sim": float(np.median(max_sim)),
    })

    for prob_rf, prob_lr, truth, sim in zip(rf_prob, lr_prob, yte, max_sim):
        pooled.append({"seed": seed, "y": truth, "rf_prob": prob_rf,
                       "lr_prob": prob_lr, "max_sim": sim})

    print(f"  seed {seed:2d}:  RF {rf_auroc:.3f}   LR {lr_auroc:.3f}   "
          f"diff {rf_auroc - lr_auroc:+.3f}   median sim {np.median(max_sim):.3f}")

seed_df = pd.DataFrame(per_seed)
pooled_df = pd.DataFrame(pooled)
seed_df.to_csv("results_per_seed.csv", index=False)
pooled_df.to_csv("results_per_molecule.csv", index=False)


# ---------------------------------------------------------------------------
# PAIRED COMPARISON
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("IS THE FINGERPRINT MODEL ACTUALLY BETTER?")
print("=" * 72)

rf_scores = seed_df["rf_auroc"].values
lr_scores = seed_df["lr_auroc"].values
diffs = seed_df["difference"].values

print(f"\n  Fingerprint RF   {rf_scores.mean():.3f} +/- {rf_scores.std(ddof=1):.3f}")
print(f"  Descriptor LR    {lr_scores.mean():.3f} +/- {lr_scores.std(ddof=1):.3f}")
print(f"  Mean difference  {diffs.mean():+.3f}")

# Paired tests. PAIRED is the right choice because both models saw exactly
# the same split on each seed, so the seed-to-seed variation is shared and
# subtracting it out removes a large source of noise.
t_stat, t_p = stats.ttest_rel(rf_scores, lr_scores)
w_stat, w_p = stats.wilcoxon(rf_scores, lr_scores)

# Confidence interval on the mean difference.
ci_low, ci_high = stats.t.interval(
    0.95, len(diffs) - 1,
    loc=diffs.mean(),
    scale=stats.sem(diffs),
)

print(f"\n  Paired t-test    t = {t_stat:.3f},  p = {t_p:.4f}")
print(f"  Wilcoxon         p = {w_p:.4f}")
print(f"  95% CI of diff   {ci_low:+.3f} to {ci_high:+.3f}")

# Cohen's dz: mean difference divided by the SD of the differences. Tells
# you how large the effect is, separately from whether it is significant.
dz = diffs.mean() / diffs.std(ddof=1)
print(f"  Effect size dz   {dz:.3f}")

print()
if ci_low > 0:
    print("  VERDICT: the interval excludes zero. The fingerprint model is")
    print("  better, and you can now say so. Report the size of the gap, not")
    print("  just the p-value: a real but tiny gain is still a tiny gain.")
else:
    print("  VERDICT: the interval still includes zero. Even at 20 seeds you")
    print("  cannot claim the fingerprint model beats four descriptors. That")
    print("  is a genuine finding, not a failure. Say it plainly.")

mean_coefs = np.mean(coefficients, axis=0)
sd_coefs = np.std(coefficients, axis=0, ddof=1)
print("\n  Standardised logistic regression coefficients:")
for name, mean_c, sd_c in zip(DESCRIPTOR_NAMES, mean_coefs, sd_coefs):
    print(f"    {name:<6} {mean_c:+.3f} +/- {sd_c:.3f}")


# ---------------------------------------------------------------------------
# SIMILARITY QUARTILES
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PERFORMANCE BY CHEMICAL SIMILARITY (QUARTILES, POOLED)")
print("=" * 72)

# Quartile edges from the pooled distribution, so each bin holds a quarter
# of all test predictions across all seeds. Pooling gives roughly 8000
# predictions, enough for stable estimates within each bin.
# duplicates="drop" guards against the degenerate case where many molecules
# share the same similarity value, which would otherwise make two quartile
# edges identical and crash pd.qcut.
pooled_df["quartile"] = pd.qcut(pooled_df["max_sim"], q=4, duplicates="drop")
bins_present = list(pooled_df["quartile"].cat.categories)
nice_labels = ["Q1 least similar", "Q2", "Q3", "Q4 most similar"]
if len(bins_present) < 4:
    nice_labels = [f"bin {i + 1}" for i in range(len(bins_present))]
    print("\n  NOTE: fewer than 4 distinct bins. Similarities are heavily tied.")

edges = np.quantile(pooled_df["max_sim"], [0, 0.25, 0.5, 0.75, 1.0])
print("\n  Quartile edges (max Tanimoto): "
      + " / ".join(f"{e:.3f}" for e in edges) + "\n")

quartile_rows = []
for interval, label in zip(bins_present, nice_labels):
    grp = pooled_df[pooled_df["quartile"] == interval]
    if grp["y"].nunique() < 2:
        print(f"  {label:<18}  n={len(grp):5d}  single class, AUROC undefined")
        continue
    auroc = roc_auc_score(grp["y"], grp["rf_prob"])
    low, high = bootstrap_auroc_ci(grp["y"].values, grp["rf_prob"].values)
    quartile_rows.append({"quartile": label, "n": len(grp),
                          "positive_rate": grp["y"].mean(),
                          "auroc": auroc, "ci_low": low, "ci_high": high})
    print(f"  {label:<18}  n={len(grp):5d}  pos={grp['y'].mean():.2f}  "
          f"AUROC {auroc:.3f}  [{low:.3f}, {high:.3f}]")

quartile_df = pd.DataFrame(quartile_rows)
quartile_df.to_csv("results_by_quartile.csv", index=False)

q1, q4 = quartile_df.iloc[0], quartile_df.iloc[-1]
print(f"\n  Lowest to highest bin gap: {q4['auroc'] - q1['auroc']:+.3f}")
print("  If the Q1 and Q4 intervals do not overlap, the degradation on")
print("  unfamiliar chemistry is real and you can state it as a finding.")
print("\n  Watch the positive rate column too. If it shifts across quartiles,")
print("  part of the AUROC change is class balance, not model skill.")


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(lr_scores, rf_scores, s=45, alpha=0.75, edgecolor="black", linewidth=0.5)
lims = [min(lr_scores.min(), rf_scores.min()) - 0.02,
        max(lr_scores.max(), rf_scores.max()) + 0.02]
ax.plot(lims, lims, "--", color="grey", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("4-descriptor logistic regression AUROC")
ax.set_ylabel("2048-bit fingerprint RF AUROC")
ax.set_title(f"Paired by seed (n={len(SEEDS)})\npoints above the line favour fingerprints")
fig.tight_layout()
fig.savefig("fig_model_comparison.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(quartile_df))
err = np.vstack([quartile_df["auroc"] - quartile_df["ci_low"],
                 quartile_df["ci_high"] - quartile_df["auroc"]])
ax.bar(x, quartile_df["auroc"], yerr=err, capsize=5,
       color="#4c72b0", edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([q.replace(" ", "\n", 1) for q in quartile_df["quartile"]])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC (95% bootstrap CI)")
ax.set_title("Performance vs similarity to training set")
fig.tight_layout()
fig.savefig("fig_similarity_quartiles.png", dpi=150)

print("\n" + "=" * 72)
print("Written: results_per_seed.csv, results_per_molecule.csv,")
print("         results_by_quartile.csv, fig_model_comparison.png,")
print("         fig_similarity_quartiles.png")
print("=" * 72)