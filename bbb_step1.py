"""
bbb_step1.py — Blood-brain barrier permeability, baseline model.

Ryan Yeung. Step 1 of the BBB project.

WHAT THIS DOES
  1. Downloads the BBB_Martins dataset (~2,000 molecules, labelled crosses / doesn't cross)
  2. Prints what's actually in it, so you look at the data before modelling it
  3. Turns each molecule into a numeric fingerprint RDKit can hand to scikit-learn
  4. Trains a random forest under a SCAFFOLD split, five times with different seeds
  5. Reports AUROC and AUPRC as mean +/- standard deviation

HOW TO RUN
  Open the Miniforge Prompt, then:
      conda activate bbb
      cd path\to\this\folder
      python bbb_step1.py

  The first run downloads the dataset (a few seconds) into a ./data folder.
  Later runs use the cached copy.

  Nothing here uses the GPU. It should finish in well under a minute.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# "import X" makes library X's tools available. "from X import Y" pulls in just Y.
# ---------------------------------------------------------------------------
import numpy as np                    # fast numerical arrays
import pandas as pd                   # spreadsheet-like tables (DataFrames)

from tdc.single_pred import ADME      # Therapeutics Data Commons, ADME task family
from rdkit import Chem                # core RDKit: parse and manipulate molecules
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# RDKit prints a lot of chemistry warnings. Silence them so the output stays readable.
RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# STEP 1 — LOAD THE DATA AND LOOK AT IT
#
# Never model a dataset you haven't inspected. Half the useful findings in a
# project like this come from this step, not from the model.
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: loading BBB_Martins")
print("=" * 70)

data = ADME(name="BBB_Martins")      # downloads on first run, caches after
df = data.get_data(format="df")      # the whole dataset as one pandas DataFrame

print(f"\nRows (molecules): {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst five rows:")
print(df.head())

# Class balance. `value_counts` counts how many of each distinct value.
counts = df["Y"].value_counts()
n_pos = int(counts.get(1, 0))
n_neg = int(counts.get(0, 0))
print(f"\nCrosses BBB (Y=1):       {n_pos}  ({100 * n_pos / len(df):.1f}%)")
print(f"Doesn't cross (Y=0):     {n_neg}  ({100 * n_neg / len(df):.1f}%)")
print("\n  ^ Note how lopsided this is. A model that blindly predicts 'crosses'")
print("    for every molecule would score that accuracy while being useless.")
print("    This is exactly why we report AUROC and AUPRC, not accuracy.")

# Duplicate structures are a common quiet problem in curated chemistry datasets.
n_dupes = int(df["Drug"].duplicated().sum())
print(f"\nDuplicate SMILES strings: {n_dupes}")


# ---------------------------------------------------------------------------
# STEP 2 — PHYSICOCHEMICAL SANITY CHECK
#
# This is pharmacy, not machine learning. Lipinski / CNS MPO intuition says
# BBB-penetrant molecules tend to be smaller, more lipophilic, and to have
# lower polar surface area and fewer hydrogen-bond donors. Check that the data
# behaves the way your degree says it should. If it doesn't, something is wrong
# with the data, not with pharmacology.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: physicochemical properties by class")
print("=" * 70)

def describe(smiles):
    """Compute four classic descriptors for one SMILES string.

    Returns None if RDKit can't parse the string, which does happen in real
    datasets and is worth counting rather than silently ignoring.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW":   Descriptors.MolWt(mol),        # molecular weight
        "logP": Descriptors.MolLogP(mol),      # lipophilicity
        "TPSA": Descriptors.TPSA(mol),         # topological polar surface area
        "HBD":  Descriptors.NumHDonors(mol),   # hydrogen-bond donors
    }

rows, n_failed = [], 0
for smiles, label in zip(df["Drug"], df["Y"]):
    d = describe(smiles)
    if d is None:
        n_failed += 1
        continue
    d["Y"] = label
    rows.append(d)

desc_df = pd.DataFrame(rows)
print(f"\nMolecules RDKit could not parse: {n_failed}")
print("\nMean descriptor values, split by class:")
print(desc_df.groupby("Y")[["MW", "logP", "TPSA", "HBD"]].mean().round(2))
print("\n  ^ Expect the Y=1 group to show lower MW, higher logP, lower TPSA,")
print("    fewer HBD. If that pattern is absent, stop and investigate.")


# ---------------------------------------------------------------------------
# STEP 3 — SCAFFOLDS
#
# The Bemis-Murcko scaffold strips side chains and keeps the ring systems plus
# the linkers between them. Aspirin and benzoic acid both reduce to c1ccccc1.
# The scaffold split groups molecules by this and never lets one scaffold group
# straddle train and test.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: scaffold structure of the dataset")
print("=" * 70)

scaffolds = []
for smiles in df["Drug"]:
    try:
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles))
    except Exception:
        scaffolds.append(None)

scaf_series = pd.Series([s for s in scaffolds if s is not None])
print(f"\nUnique scaffolds: {scaf_series.nunique()} across {len(scaf_series)} molecules")
print("\nFive most common scaffolds:")
print(scaf_series.value_counts().head())
print("\n  ^ The empty string '' is acyclic molecules, which have no ring system")
print("    and therefore no scaffold. They all fall into a single group.")


# ---------------------------------------------------------------------------
# STEP 4 — FINGERPRINTS
#
# scikit-learn needs numbers, not molecules. A Morgan (ECFP-style) fingerprint
# walks outward from each atom to a given radius, hashes each local environment
# it finds, and flips the corresponding bit in a fixed-length vector. radius=2
# with 2048 bits is the standard default and is roughly ECFP4.
#
# Result: each molecule becomes a 2048-length vector of 0s and 1s.
# ---------------------------------------------------------------------------
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurise(smiles_list):
    """Turn a list of SMILES into a (n_molecules x 2048) numeric array.

    Returns the array plus a boolean mask marking which inputs parsed, so the
    labels can be filtered to match.
    """
    vectors, keep = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            keep.append(False)
            continue
        vectors.append(fpgen.GetFingerprintAsNumPy(mol))
        keep.append(True)
    return np.array(vectors), np.array(keep)


# ---------------------------------------------------------------------------
# STEP 5 — TRAIN AND EVALUATE
#
# One run tells you almost nothing: change the seed and the number moves. So
# run five seeds and report mean +/- standard deviation. Anyone reading this
# who does research will look for the error bars first.
#
# METRICS
#   AUROC — probability a random positive is ranked above a random negative.
#           0.5 is chance, 1.0 is perfect.
#   AUPRC — better behaved than AUROC when classes are imbalanced, as here.
#           The chance baseline is the positive rate, not 0.5.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: random forest on Morgan fingerprints, scaffold split")
print("=" * 70)

auroc_scores, auprc_scores = [], []

for seed in [1, 2, 3, 4, 5]:
    # Scaffold split. Returns {'train': df, 'valid': df, 'test': df}.
    split = data.get_split(method="scaffold", seed=seed)
    train_df, test_df = split["train"], split["test"]

    X_train, mask_train = featurise(train_df["Drug"])
    y_train = train_df["Y"].values[mask_train]

    X_test, mask_test = featurise(test_df["Drug"])
    y_test = test_df["Y"].values[mask_test]

    # class_weight="balanced" tells the forest to weight the rare class more
    # heavily, compensating for the ~75/25 imbalance.
    # n_jobs=-1 uses every CPU core you have.
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # predict_proba returns a probability per class; column 1 is P(crosses).
    # Use probabilities, not hard 0/1 predictions — AUROC needs a ranking.
    y_prob = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    auroc_scores.append(auroc)
    auprc_scores.append(auprc)

    print(f"  seed {seed}:  train n={len(y_train):4d}  test n={len(y_test):4d}  "
          f"AUROC {auroc:.3f}  AUPRC {auprc:.3f}")

print("\n" + "-" * 70)
print(f"AUROC  {np.mean(auroc_scores):.3f} +/- {np.std(auroc_scores):.3f}")
print(f"AUPRC  {np.mean(auprc_scores):.3f} +/- {np.std(auprc_scores):.3f}")
print("-" * 70)
print("\nThis is your baseline. Every later model has to beat these numbers,")
print("under this same split, with error bars, or it isn't an improvement.")
print("\nWrite the two numbers down. That is the deliverable for step 1.")