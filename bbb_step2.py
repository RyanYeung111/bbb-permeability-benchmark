"""
bbb_step2.py — Blood-brain barrier, interrogating the baseline.

Ryan Yeung. Step 2 of the BBB project.

Step 1 produced a number (AUROC 0.907). Step 2 asks whether that number
means what it appears to mean. Three questions, in order of how much they
could undermine the result:

  A. LABEL QUALITY
     Do any identical structures carry contradictory labels? If the same
     molecule appears as both 1 and 0, no model can get both right, and
     that puts a hard ceiling on achievable performance.

  B. IS THE MODEL LEARNING CHEMISTRY, OR JUST LIPINSKI?
     Compare the 2048-bit fingerprint random forest against a logistic
     regression using only four descriptors you already know from
     pharmacokinetics: MW, logP, TPSA, HBD. If four numbers get most of
     the way there, the fingerprint model is mostly re-deriving physchem
     intuition, and that is the more interesting finding.

  C. DOES IT GENERALISE TO NOVEL CHEMISTRY?
     For each test molecule, find its maximum Tanimoto similarity to any
     training molecule, then split the test set into "similar" and
     "dissimilar" halves and score each separately. A model that only
     works on close analogues is not useful for new chemical series.

Run exactly as step 1:
    conda activate bbb
    python bbb_step2.py

Takes roughly two to four minutes, mostly on the similarity calculation.
No GPU needed.
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

RDLogger.DisableLog("rdApp.*")

SEEDS = [1, 2, 3, 4, 5]
DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD"]

# One shared fingerprint generator, reused everywhere so all comparisons
# use identical featurisation.
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def fingerprint_array(smiles_list):
    """SMILES -> (n x 2048) numeric array, plus a mask of which parsed."""
    vectors, keep = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            keep.append(False)
            continue
        vectors.append(fpgen.GetFingerprintAsNumPy(mol))
        keep.append(True)
    return np.array(vectors), np.array(keep)


def descriptor_array(smiles_list):
    """SMILES -> (n x 4) array of MW, logP, TPSA, HBD, plus a parse mask."""
    rows, keep = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            keep.append(False)
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
        ])
        keep.append(True)
    return np.array(rows), np.array(keep)


def rdkit_fingerprints(smiles_list):
    """RDKit fingerprint OBJECTS (not arrays). Needed for Tanimoto, which
    operates on bit vectors rather than numpy arrays."""
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        out.append(None if mol is None else fpgen.GetFingerprint(mol))
    return out


def summarise(name, scores):
    """Print mean +/- std for a list of scores."""
    print(f"  {name:<34} {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


# ---------------------------------------------------------------------------
# A. LABEL QUALITY
# ---------------------------------------------------------------------------
print("=" * 70)
print("A. LABEL QUALITY: contradictory duplicates")
print("=" * 70)

data = ADME(name="BBB_Martins")
df = data.get_data(format="df")

# For each distinct SMILES, count how many DIFFERENT labels it carries.
# nunique() == 1 means consistent; == 2 means the same structure is
# labelled both 0 and 1 somewhere in the dataset.
labels_per_structure = df.groupby("Drug")["Y"].nunique()
conflicting = labels_per_structure[labels_per_structure > 1]

n_dupes = int(df["Drug"].duplicated().sum())
print(f"\nDuplicated SMILES entries:        {n_dupes}")
print(f"Structures with BOTH labels:      {len(conflicting)}")

if len(conflicting) > 0:
    print("\nConflicting structures:")
    for smiles in conflicting.index:
        subset = df[df["Drug"] == smiles]
        names = ", ".join(str(n) for n in subset["Drug_ID"].unique())
        print(f"  {smiles[:60]}")
        print(f"    labels: {sorted(subset['Y'].tolist())}   ids: {names}")
    n_rows = int(df["Drug"].isin(conflicting.index).sum())
    print(f"\n  {n_rows} rows are affected. No model can classify both sides")
    print("  of a contradiction correctly, so this caps achievable accuracy.")
else:
    print("\n  No contradictions. Duplicates are consistent, so they inflate")
    print("  the weight of certain structures but do not corrupt labels.")

print("\n  Worth knowing: identical SMILES share a scaffold, so duplicates")
print("  always land in the SAME split partition. They cannot leak across.")


# ---------------------------------------------------------------------------
# B. FINGERPRINTS vs FOUR DESCRIPTORS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("B. DOES 2048 BITS BEAT 4 DESCRIPTORS?")
print("=" * 70)
print("\nBoth models, identical splits, identical seeds.\n")

fp_scores, desc_scores, majority_scores = [], [], []
descriptor_coefficients = []

for seed in SEEDS:
    split = data.get_split(method="scaffold", seed=seed)
    train_df, test_df = split["train"], split["test"]

    # --- fingerprint random forest (the step 1 model) ---
    X_tr, m_tr = fingerprint_array(train_df["Drug"])
    y_tr = train_df["Y"].values[m_tr]
    X_te, m_te = fingerprint_array(test_df["Drug"])
    y_te = test_df["Y"].values[m_te]

    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ).fit(X_tr, y_tr)
    fp_scores.append(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))

    # --- four-descriptor logistic regression ---
    # StandardScaler puts each descriptor on the same scale first. Without
    # it, MW (hundreds) would dominate HBD (single digits) purely because
    # of units. make_pipeline fits the scaler on train only, which avoids
    # leaking test statistics into training.
    D_tr, dm_tr = descriptor_array(train_df["Drug"])
    dy_tr = train_df["Y"].values[dm_tr]
    D_te, dm_te = descriptor_array(test_df["Drug"])
    dy_te = test_df["Y"].values[dm_te]

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(D_tr, dy_tr)
    desc_scores.append(roc_auc_score(dy_te, logreg.predict_proba(D_te)[:, 1]))
    descriptor_coefficients.append(
        logreg.named_steps["logisticregression"].coef_[0]
    )

    # --- reference point: predict the majority class for everything ---
    # AUROC of a constant predictor is exactly 0.5 by definition. Printed
    # so the floor is explicit rather than assumed.
    majority_scores.append(0.5)

    print(f"  seed {seed}:  fingerprint RF {fp_scores[-1]:.3f}   "
          f"4-descriptor LR {desc_scores[-1]:.3f}")

print()
summarise("Fingerprint RF (2048 features)", fp_scores)
summarise("Descriptor LR (4 features)", desc_scores)
summarise("Constant predictor (floor)", majority_scores)

gap = np.mean(fp_scores) - np.mean(desc_scores)
print(f"\n  Gap: {gap:+.3f} AUROC for 2044 extra features.")
print("  Compare that gap to the seed-to-seed standard deviation above.")
print("  If the gap is smaller than the spread, it is not a real difference.")

# Average the logistic regression coefficients across seeds. Because the
# inputs were standardised, these are directly comparable to each other:
# a larger magnitude means that descriptor matters more. Sign tells you
# direction, positive meaning "pushes toward crossing the BBB".
mean_coefs = np.mean(descriptor_coefficients, axis=0)
print("\n  Standardised coefficients (mean across seeds):")
for name, coef in zip(DESCRIPTOR_NAMES, mean_coefs):
    direction = "favours crossing" if coef > 0 else "opposes crossing"
    print(f"    {name:<6} {coef:+.3f}   {direction}")
print("\n  Check these against what you were taught. logP positive and TPSA")
print("  negative is the expected pattern. If it is not, say so.")


# ---------------------------------------------------------------------------
# C. PERFORMANCE vs CHEMICAL NOVELTY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("C. DOES PERFORMANCE HOLD ON UNFAMILIAR CHEMISTRY?")
print("=" * 70)
print("\nTanimoto similarity: shared bits / total bits across two")
print("fingerprints. 1.0 is identical, 0.0 is nothing in common.\n")

near_scores, far_scores, medians = [], [], []

for seed in SEEDS:
    split = data.get_split(method="scaffold", seed=seed)
    train_df, test_df = split["train"], split["test"]

    train_fps = [f for f in rdkit_fingerprints(train_df["Drug"]) if f is not None]

    X_tr, m_tr = fingerprint_array(train_df["Drug"])
    y_tr = train_df["Y"].values[m_tr]
    X_te, m_te = fingerprint_array(test_df["Drug"])
    y_te = test_df["Y"].values[m_te]

    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ).fit(X_tr, y_tr)
    y_prob = rf.predict_proba(X_te)[:, 1]

    # For each test molecule, the single closest training molecule.
    test_fps = [f for f in rdkit_fingerprints(test_df["Drug"]) if f is not None]
    max_sim = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        for fp in test_fps
    ])

    # Split the test set at its own median similarity, so both halves are
    # the same size and the comparison is not confounded by sample count.
    cutoff = np.median(max_sim)
    medians.append(cutoff)
    near = max_sim >= cutoff
    far = ~near

    # A half can be single-class by chance, in which case AUROC is undefined.
    near_auroc = roc_auc_score(y_te[near], y_prob[near]) if len(set(y_te[near])) > 1 else np.nan
    far_auroc = roc_auc_score(y_te[far], y_prob[far]) if len(set(y_te[far])) > 1 else np.nan
    near_scores.append(near_auroc)
    far_scores.append(far_auroc)

    print(f"  seed {seed}:  median max-similarity {cutoff:.3f}   "
          f"similar half {near_auroc:.3f}   dissimilar half {far_auroc:.3f}")

print()
summarise("Similar half (above median)", np.array(near_scores)[~np.isnan(near_scores)])
summarise("Dissimilar half (below median)", np.array(far_scores)[~np.isnan(far_scores)])
print(f"\n  Median max-similarity across seeds: {np.mean(medians):.3f}")
print("\n  A large drop on the dissimilar half means the model leans on")
print("  close analogues and will degrade on genuinely new chemotypes.")
print("  A small drop is a real robustness claim you can defend.")

print("\n" + "=" * 70)
print("Record all three results. Together they say more about the dataset")
print("and the model than any single AUROC ever could.")
print("=" * 70)