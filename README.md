# Blood-Brain Barrier Permeability: A Re-evaluation of the BBB_Martins Benchmark

An independent computational project investigating how a single headline AUROC can obscure differences in molecular representation, chemical novelty, and label quality within a standard ADMET benchmark.

**Full report:** [report.pdf](report.pdf)

## Findings

1. Fingerprints improved performance, but the simpler model came surprisingly close.

A random forest using 2,048-bit Morgan fingerprints achieved an AUROC of 0.903 (SD 0.026), compared with 0.850 (SD 0.035) for logistic regression using just four standard molecular descriptors: molecular weight, logP, TPSA and hydrogen-bond donor count. Across 20 scaffold splits, the mean difference was 0.053 (95% CI 0.038–0.068, p < 0.0001). Despite using a much simpler representation, the four-descriptor model captured roughly 87% of the fingerprint model's improvement over chance.

2. Model performance varied substantially with the chemical novelty of the test compounds.

To investigate how well the model generalised to unfamiliar chemistry, 8,120 predictions were pooled and grouped according to each compound's maximum Tanimoto similarity to the training set.

| Quartile | Similarity | AUROC | 95% CI |
| --- | --- | --- | --- |
| Least similar | 0.04–0.31 | 0.773 | 0.749–0.795 |
| | 0.31–0.45 | 0.853 | 0.828–0.878 |
| | 0.45–0.60 | 0.924 | 0.907–0.939 |
| Most similar | 0.60–1.00 | 0.965 | 0.955–0.974 |

AUROC increased steadily across the four quartiles, with no overlap between their confidence intervals. The difference between the least and most similar groups was 0.192.

This pattern was not explained by class balance. In fact, the quartile with the highest AUROC had the lowest proportion of positive labels. A headline AUROC of around 0.90 therefore hides quite a wide variation in performance, depending on how familiar the test chemistry is.

3. Around 1% of the labels are contradictory, for two different reasons.

Ten structures appear with both positive and negative labels. In six cases, the issue appears to be a curation artefact. For example, Miconazole and miconazole were recorded as separate entries with opposite labels.

The remaining cases are less straightforward. Some reflect pharmacological behaviour that a binary label cannot fully capture. Loperamide can cross the endothelium but is subsequently effluxed by P-glycoprotein. Levodopa, meanwhile, is too polar for passive diffusion but can enter the brain through LAT1-mediated transport.

![AUROC by similarity quartile](fig_similarity_quartiles.png)

## Reproducing

```bash
conda env create -f environment.yml
conda activate bbb
python bbb_step1.py
python bbb_step2.py
python bbb_step3.py
python bbb_step4.py
```

Runs on CPU. Total runtime under 30 minutes on a consumer laptop.

If the pip installation fails on PyTDC, try installing it separately. Later versions of its dependency tree include a package without a Windows distribution:

```bash
pip install PyTDC==0.4.1 --no-deps
pip install fuzzywuzzy tqdm seaborn requests huggingface_hub
```

On Windows 11 with Smart App Control enabled, pip-installed packages with
compiled extensions may be blocked at import. Installing them from
conda-forge resolves this.

## Contents

| File | Purpose |
| --- | --- |
| `bbb_step1.py` | Data loading, inspection, baseline random forest under scaffold split |
| `bbb_step2.py` | Label quality, descriptor model, median similarity split |
| `bbb_step3.py` | Twenty-split evaluation, paired testing, quartile stratification, figures |
| `bbb_step4.py` | Descriptor collinearity, named per-molecule predictions, near-identical pairs |
| `results_per_seed.csv` | AUROC for both models, per seed, with median similarity |
| `results_per_molecule.csv` | Every test prediction with label and nearest-neighbour similarity |
| `results_per_molecule_named.csv` | As above, with entry identifier and nearest-neighbour identity |
| `results_by_quartile.csv` | Quartile AUROCs with bootstrap intervals |
| `results_descriptor_collinearity.csv` | Per-descriptor R² and variance inflation factors |
| `results_similarity_anomalies.csv` | Test–train pairs with Tanimoto similarity ≥ 0.99 |
| `environment.yml` | Pinned conda environment |
| `requirements.txt` | Pinned pip packages as installed for the analysis |

## Method notes

- **Scaffold splitting** via TDC's `get_split(method='scaffold')`, which
  assigns whole Bemis–Murcko scaffold groups to partitions. 70:10:20,
  1,421 train and 406 test molecules.
- **Deviation from the official TDC protocol:** the Benchmark Group fixes the
  test partition and reshuffles only train and validation. Here the entire
  split is resampled per seed, so these numbers are *not* directly comparable
  to leaderboard entries. The trade-off buys a measurement of sensitivity to
  scaffold assignment, which a fixed test partition cannot provide.
- **AUROC over accuracy** because the majority class is 76.4%. **AUROC over
  AUPRC** because the positive class is the majority, putting the AUPRC
  chance baseline at 0.764.
- **Paired testing** throughout: both models see identical splits at each
  seed, so split-to-split variance is differenced out.
- **No hyperparameter tuning** on either model. Both use library defaults.

## Known limitations

The analysis covers only one dataset and endpoint, with no external validation. Similarity was measured using the same fingerprint representation supplied to the random forest, so the results depend partly on that choice. The measure also has limitations of its own. For example, cyclopropane and cyclohexane receive a Tanimoto similarity of 1.000 because radius-2 environments cannot distinguish their ring sizes.

The four molecular descriptors are also correlated, with TPSA having a VIF of 10.09. This makes individual model coefficients difficult to interpret independently. Finally, duplicates were identified using exact SMILES matching without structure standardisation, meaning the analysis likely underestimates their actual number.

## Data

Martins et al. blood–brain barrier penetration dataset, accessed via
[Therapeutics Data Commons](https://tdcommons.ai/). 2,030 rows, 1,975 unique
structures, 76.4% positive.

## Author

Ryan Yeung — MPharm, UCL School of Pharmacy

An AI assistant was used for environment configuration, drafting the analysis
scripts, and manuscript review; see the acknowledgements section of the report
and [CLAUDE.md](CLAUDE.md).