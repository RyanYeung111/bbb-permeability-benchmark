"""
bbb_step4.py — Blood-brain barrier, three loose ends.

Ryan Yeung. Step 4 of the BBB project.

Step 3 left three questions open. None of them is a new model; all three
are checks on claims the report currently makes or wants to make.

  A. ARE THE FOUR DESCRIPTORS COLLINEAR?
     Step 3 reported standardised coefficients (TPSA -0.814, HBD -0.807,
     MW -0.222, logP +0.190) and it is tempting to read those as
     importance rankings. But HBD contributes directly to TPSA, and logP
     correlates negatively with both. Collinear predictors split shared
     variance unstably, so individual coefficients may not mean what they
     appear to. This section measures that directly, using a correlation
     matrix and variance inflation factors.

  B. WHICH NAMED MOLECULES DO THE MODELS GET WRONG?
     results_per_molecule.csv has predictions but no drug names, so the
     claims in section 4.4 rest entirely on the literature. Adding Drug_ID
     lets the report say how often the models actually misclassified
     loperamide and levodopa, which turns a literature argument into
     evidence from this experiment.

  C. WHY DID ONE TEST MOLECULE SHOW SIMILARITY 1.000?
     Step 3 found a maximum Tanimoto of exactly 1.000 despite scaffold
     splitting. Identical fingerprints usually imply identical scaffolds,
     which should put both molecules in the same partition. This prints
     every near-identical cross-partition pair so the cause can be seen.

Run exactly like the others:
    conda activate bbb
    python bbb_step4.py

Roughly 5 to 15 minutes. No GPU. Self-contained: it regenerates the
splits rather than reading step 3's output.
"""

import numpy as np
import pandas as pd

from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RDLogger.DisableLog("rdApp.*")

SEEDS = list(range(1, 21))
DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD"]
NEAR_IDENTICAL = 0.99          # threshold for "suspiciously similar"

# Molecules named in the report's discussion. Matching is done on the
# structure, not the name, so capitalisation and synonyms don't matter.
WATCHLIST = ["loperamide", "levodopa", "aspirin", "acetylsalicylate",
             "loratadine", "mequitazine", "miconazole", "atropine",
             "indomethacin", "trimetrexate", "methylprednisolone",
             "BRL53080"]

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_fp_arr, _fp_obj, _desc, _scaffold = {}, {}, {}, {}


def build_caches(smiles_list):
    for smiles in set(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _fp_arr[smiles] = _fp_obj[smiles] = _desc[smiles] = None
            _scaffold[smiles] = None
            continue
        _fp_arr[smiles] = fpgen.GetFingerprintAsNumPy(mol)
        _fp_obj[smiles] = fpgen.GetFingerprint(mol)
        _desc[smiles] = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                         Descriptors.TPSA(mol), Descriptors.NumHDonors(mol)]
        try:
            _scaffold[smiles] = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
        except Exception:
            _scaffold[smiles] = None


def prepare(sub_df):
    """Cached features for one partition, dropping unparseable molecules."""
    arrs, objs, descs, labels, ids, smis = [], [], [], [], [], []
    for smiles, label, drug_id in zip(sub_df["Drug"], sub_df["Y"], sub_df["Drug_ID"]):
        if _fp_arr[smiles] is None:
            continue
        arrs.append(_fp_arr[smiles])
        objs.append(_fp_obj[smiles])
        descs.append(_desc[smiles])
        labels.append(label)
        ids.append(drug_id)
        smis.append(smiles)
    return (np.array(arrs), np.array(descs), objs,
            np.array(labels), ids, smis)


print("=" * 72)
print("LOADING")
print("=" * 72)

data = ADME(name="BBB_Martins")
full_df = data.get_data(format="df")
build_caches(full_df["Drug"].tolist())
print(f"\nCached {len(_fp_arr)} unique structures.")


# ---------------------------------------------------------------------------
# A. DESCRIPTOR COLLINEARITY
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("A. ARE THE FOUR DESCRIPTORS INDEPENDENT?")
print("=" * 72)

desc_matrix = np.array([_desc[s] for s in full_df["Drug"] if _desc[s] is not None])
desc_df = pd.DataFrame(desc_matrix, columns=DESCRIPTOR_NAMES)

print("\nPearson correlation matrix:\n")
corr = desc_df.corr()
print(corr.round(3).to_string())

# Variance inflation factor. For each descriptor, regress it on the other
# three and record how much of its variance they explain (R-squared).
#   VIF = 1 / (1 - R^2)
# VIF of 1 means fully independent. Above about 5 is usually taken as
# problematic collinearity; above 10 as severe. The interpretation: VIF is
# the factor by which that coefficient's variance is inflated relative to
# a model where the predictors were uncorrelated.
print("\nVariance inflation factors:\n")
vif_rows = []
for i, name in enumerate(DESCRIPTOR_NAMES):
    others = [j for j in range(len(DESCRIPTOR_NAMES)) if j != i]
    r2 = LinearRegression().fit(desc_matrix[:, others],
                                desc_matrix[:, i]).score(desc_matrix[:, others],
                                                         desc_matrix[:, i])
    vif = np.inf if r2 >= 1 else 1 / (1 - r2)
    vif_rows.append({"descriptor": name, "r_squared": r2, "vif": vif})
    verdict = "severe" if vif > 10 else "problematic" if vif > 5 else "acceptable"
    print(f"  {name:<6} R^2 = {r2:.3f}   VIF = {vif:6.2f}   ({verdict})")

pd.DataFrame(vif_rows).to_csv("results_descriptor_collinearity.csv", index=False)

max_vif = max(r["vif"] for r in vif_rows)
print()
if max_vif > 5:
    print("  VERDICT: at least one descriptor is substantially explained by")
    print("  the others. Individual coefficient magnitudes should NOT be read")
    print("  as independent importance. Write section 4.1 as a caveat.")
else:
    print("  VERDICT: collinearity is within conventional limits. Coefficient")
    print("  magnitudes can be discussed, with the correlation matrix reported")
    print("  alongside them.")

# A single-descriptor comparison is a blunter but very readable check:
# if TPSA alone nearly matches all four together, the other three are
# adding little regardless of what their coefficients say.
print("\n  For context, how much does each descriptor achieve alone?")
print("  (fitted and scored on a single scaffold split, seed 1)")

from sklearn.metrics import roc_auc_score

split1 = data.get_split(method="scaffold", seed=1)
_, D_tr, _, y_tr, _, _ = prepare(split1["train"])
_, D_te, _, y_te, _, _ = prepare(split1["test"])

for i, name in enumerate(DESCRIPTOR_NAMES):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000,
                                             class_weight="balanced"))
    model.fit(D_tr[:, [i]], y_tr)
    auroc = roc_auc_score(y_te, model.predict_proba(D_te[:, [i]])[:, 1])
    print(f"    {name:<6} alone: AUROC {auroc:.3f}")

model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
model.fit(D_tr, y_tr)
print(f"    all four:    AUROC {roc_auc_score(y_te, model.predict_proba(D_te)[:, 1]):.3f}")


# ---------------------------------------------------------------------------
# B & C. PER-MOLECULE PREDICTIONS AND NEAREST NEIGHBOURS
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("B/C. PER-MOLECULE PREDICTIONS WITH NAMES AND NEAREST NEIGHBOURS")
print("=" * 72)
print(f"\nRunning {len(SEEDS)} splits again, this time recording identities.\n")

rows = []

for seed in SEEDS:
    split = data.get_split(method="scaffold", seed=seed)
    X_tr, D_tr, FP_tr, y_tr, id_tr, smi_tr = prepare(split["train"])
    X_te, D_te, FP_te, y_te, id_te, smi_te = prepare(split["test"])

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    rf_prob = rf.predict_proba(X_te)[:, 1]

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000,
                                          class_weight="balanced")).fit(D_tr, y_tr)
    lr_prob = lr.predict_proba(D_te)[:, 1]

    for k, fp in enumerate(FP_te):
        sims = DataStructs.BulkTanimotoSimilarity(fp, FP_tr)
        nearest = int(np.argmax(sims))
        rows.append({
            "seed": seed,
            "Drug_ID": id_te[k],
            "smiles": smi_te[k],
            "y_true": int(y_te[k]),
            "rf_prob": rf_prob[k],
            "lr_prob": lr_prob[k],
            "max_sim": sims[nearest],
            "nearest_train_id": id_tr[nearest],
            "nearest_train_smiles": smi_tr[nearest],
        })

    print(f"  seed {seed:2d} done")

per_mol = pd.DataFrame(rows)

# A prediction is "correct" if the probability falls on the right side of
# 0.5. Note this reintroduces a threshold, which AUROC avoided. It is used
# here only to talk about individual molecules, not to score the models.
per_mol["rf_correct"] = (per_mol["rf_prob"] >= 0.5).astype(int) == per_mol["y_true"]
per_mol["lr_correct"] = (per_mol["lr_prob"] >= 0.5).astype(int) == per_mol["y_true"]
per_mol.to_csv("results_per_molecule_named.csv", index=False)
print(f"\nWritten results_per_molecule_named.csv ({len(per_mol)} rows).")


# --- B. the watchlist ---
print("\n" + "-" * 72)
print("MOLECULES NAMED IN THE DISCUSSION")
print("-" * 72)

# Match on structure so that synonyms and capitalisation variants are
# picked up together. This is the same canonical-SMILES deduplication
# argued for in section 4.4, applied here.
watch_smiles = set()
for name in WATCHLIST:
    hits = full_df[full_df["Drug_ID"].astype(str).str.lower() == name.lower()]
    watch_smiles.update(hits["Drug"].tolist())

print(f"\n{'molecule':<24} {'label':>6} {'appears':>8} {'RF ok':>7} {'LR ok':>7}")
print("-" * 60)
for smiles in sorted(watch_smiles):
    sub = per_mol[per_mol["smiles"] == smiles]
    if len(sub) == 0:
        continue
    names = "/".join(sorted({str(n) for n in sub["Drug_ID"]}))
    labels = sorted(set(sub["y_true"]))
    label_str = "0/1" if len(labels) > 1 else str(labels[0])
    print(f"  {names[:22]:<22} {label_str:>6} {len(sub):>8} "
          f"{sub['rf_correct'].sum():>4}/{len(sub)} {sub['lr_correct'].sum():>4}/{len(sub)}")

print("\n  'appears' counts how often the molecule landed in a test set")
print("  across the 20 splits. A molecule can appear 0 times if its")
print("  scaffold group was always assigned to training.")
print("\n  Where a structure carries both labels, correctness is being")
print("  measured against contradictory targets, so a score near half is")
print("  the expected outcome. Say this rather than reporting it as error.")


# --- C. the similarity anomaly ---
print("\n" + "-" * 72)
print(f"NEAR-IDENTICAL PAIRS ACROSS THE SPLIT (Tanimoto >= {NEAR_IDENTICAL})")
print("-" * 72)

anomalies = per_mol[per_mol["max_sim"] >= NEAR_IDENTICAL].copy()
print(f"\n{len(anomalies)} of {len(per_mol)} test predictions "
      f"({100 * len(anomalies) / len(per_mol):.2f}%) had a near-identical "
      f"training neighbour.")

if len(anomalies) > 0:
    # Collapse to unique structure pairs; the same pair recurs across seeds.
    unique_pairs = anomalies.drop_duplicates(subset=["smiles", "nearest_train_smiles"])
    print(f"{len(unique_pairs)} unique pairs. Showing up to 10:\n")

    for _, row in unique_pairs.head(10).iterrows():
        same_smiles = row["smiles"] == row["nearest_train_smiles"]
        scaf_test = _scaffold.get(row["smiles"])
        scaf_train = _scaffold.get(row["nearest_train_smiles"])
        same_scaffold = scaf_test == scaf_train

        print(f"  Tanimoto {row['max_sim']:.4f}")
        print(f"    test  : {row['Drug_ID']}  {row['smiles'][:66]}")
        print(f"    train : {row['nearest_train_id']}  {row['nearest_train_smiles'][:66]}")
        print(f"    identical SMILES: {same_smiles}   identical scaffold: {same_scaffold}")
        if not same_smiles:
            print(f"    scaffolds: test '{str(scaf_test)[:40]}' vs "
                  f"train '{str(scaf_train)[:40]}'")
        print()

    unique_pairs.to_csv("results_similarity_anomalies.csv", index=False)
    print("  Written results_similarity_anomalies.csv")
    print("\n  How to read this:")
    print("   - identical SMILES + identical scaffold = a genuine split failure,")
    print("     worth reporting, since scaffold grouping should have prevented it")
    print("   - different SMILES, identical fingerprint = the fingerprint cannot")
    print("     distinguish them. Usually stereochemistry (Morgan fingerprints")
    print("     ignore chirality by default) or a difference outside radius 2.")
    print("   - different scaffolds = the two molecules differ in ring system but")
    print("     share every hashed environment. Check for salts and counterions.")
else:
    print("\n  None found at this threshold. Lower NEAR_IDENTICAL to inspect")
    print("  the most similar pairs regardless.")

print("\n" + "=" * 72)
print("Written: results_descriptor_collinearity.csv,")
print("         results_per_molecule_named.csv,")
print("         results_similarity_anomalies.csv (if any found)")
print("=" * 72)