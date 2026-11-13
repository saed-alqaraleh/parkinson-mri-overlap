# Sensitivity of Patient-Level Parkinson Disease MRI Evaluation to Training–Test Subject Overlap

**Reproducible research code, frozen study partitions, and derived evaluation evidence** for the controlled four-backbone Parkinson disease MRI sensitivity study.

Patient-level aggregation cannot establish independence if the model was developed using slices from an evaluated subject. This study evaluates a **combined intervention** that increases training–test subject overlap while preserving a diagnosis-matched training-slice budget. The findings describe **overlap sensitivity, not a fixed inflation factor**. A positive response for Swin-Tiny is **not established**. The software supports methodological research; it is **not a diagnostic or clinical system**.

## Study at a glance

- **Frozen study cohort:** 68 recorded subjects (48 PD, 20 non-PD), 42,147 MRI slice records, five fixed outer subject folds and independent inner validation.
- **Intervention:** nested 0%, 25%, 50%, 75%, 100% overlap; at most 50 slices and 20% per represented subject, with at least two test slices retained; equal-number, diagnosis-matched removal from the original training group.
- **Models and fits:** ResNet50V2 across five levels; VGG19, ConvNeXt-Tiny and Swin-Tiny at 0%/100%; **55 original fits + 40 second-seed fits = 95**.
- **Estimand:** five-fold mean paired Δ patient ROC-AUC (100% minus 0% overlap); diagnosis-stratified, paired subject-bootstrap conditional on the fitted models (**10,000 replicates, seed 20260910**).
- **Training seeds:** 42, 101, 202, 303, 404 by fold; repeat seeds add 100000, without regenerating any partition.

| Backbone | Original Δ | Second-seed Δ | Seed-averaged Δ |
|---|---:|---:|---:|
| ResNet50V2 | +0.075 | +0.135 | +0.105 |
| VGG19 | +0.160 | +0.181 | +0.170 |
| ConvNeXt-Tiny | +0.048 | +0.053 | +0.050 |
| Swin-Tiny | +0.023 | +0.019 | +0.021 |

All values above are fold-paired means, not estimates of universal inflation or general clinical diagnostic performance.

## Install and reproduce the saved evaluation

Use **Python 3.12** for numerical reproduction; no original MRI images or GPU are required. Run the following from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pytest -q
python -m scripts.reproduce_patient_analysis --output outputs/patient_metrics
python -m scripts.reproduce_full_statistics --out outputs/reproduced
python -m scripts.reproduce_tables --out outputs/tables
python -m scripts.regenerate_partitions --out outputs/reproduced/frozen_operations.csv.gz
python -m scripts.reproduce_figures --out outputs/figures
python scripts/reproduce_public_results.py --out outputs/derived
python scripts/verify_checksums.py
python scripts/check_release.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. The archived scores are sufficient to reproduce the conditional intervals, slope and omission analyses; the frozen source-relative paths are sufficient to validate the split and slice-replacement manifests without reading the images. All generated artifacts reside in the ignored `outputs/` directory.

## Reproduction map

| Manuscript result | Inputs | Regeneration command or source |
|---|---|---|
| **Table 3:** patient ROC-AUC, average precision, balanced accuracy, slice ROC-AUC | `data/original_run_metrics.csv`, `data/original_slice_metrics.csv`, patient scores | `python -m scripts.reproduce_tables`; `scripts/reproduce_full_statistics.py` |
| **Table 4:** original endpoint contrasts, conditional CIs, common-file comparison | `data/patient_scores_original*.csv` | `python -m scripts.reproduce_full_statistics` |
| **Table 5:** second-seed, seed-averaged and paired fold comparisons | `data/patient_scores_second_seed*.csv`; original scores | `python -m scripts.reproduce_full_statistics` |
| **Table 6:** content-audit counts, related identity representation diagnostics | `data/content_audit_*.csv`, `data/*representation*.csv` | `python -m scripts.reproduce_tables` and recorded summary CSVs |
| **Figure 2:** ResNet50V2 response against **realized**, not nominal, overlap | original fold-level AUCs and realized percentages | `python -m scripts.reproduce_figures` |
| **Figure 3:** fold-paired endpoint effects and original conditional intervals | original patient scores | `python -m scripts.reproduce_figures` |
| **Figure 4:** original, second-seed and seed-average estimates, each over five folds and after omitting fold 4 | both sets of patient scores | `python -m scripts.reproduce_figures` |
| **Slope:** +0.098, conditional 95% CI [0.017, 0.192] | graded original patient scores and realized overlap | `outputs/reproduced/graded_slope.csv` |
| **Fixed-threshold / common-file sensitivities** | original patient scores, common-file scores | `outputs/reproduced/fixed_threshold_metrics.csv`, `common_held_out_metrics.csv` |
| **Supplementary fold-level and identity diagnostics** | frozen run tables, `data/original_representation_runs.csv`, `data/diagnosis_matched_representation_summary.csv` | saved rows plus `outputs/reproduced/fold_endpoint_deltas.csv` |
| **Frozen partitions and replacement audit** | `data/source_relative_inventory.csv.gz`, roles and operation evidence | `python -m scripts.regenerate_partitions` |

Figures 2–4 are regenerated from archived numerical sources with the manuscript's panel structure, 0.5 reference line in Figure 2, and all three training-seed summaries in Figure 4. Figure 4B includes independently regenerated seed-averaged four-fold intervals. The released Table 5 second-seed and seed-averaged **point estimates reproduce exactly**; interval bounds use an independent 10,000-draw bootstrap sequence and may differ by up to **0.003** from printed Table 5 values. All Table 4 original-seed intervals and the reported original-seed slope interval reproduce at numerical precision. The six rounded bounds affected in Table 5 are recorded and regression-checked in `data/reference_table5_intervals.csv`.

## Data and implementation layout

```text
.
├── configs/                   Frozen model and intervention settings
├── data/                      Patient-score evidence, aggregate outputs, frozen manifests
├── docs/COMPUTE.md            Original framework and GPU configuration
├── pipelines/
│   ├── keras/                 Original and batched-augmentation Keras runners
│   ├── torchvision/src/       Original TorchVision implementation
│   ├── content_audit/         Pixel-level content audit source
│   └── protocol_checks.py    Partition validation
├── scripts/                   Statistics, tables, figures, partition checks, hashes
├── tests/                     Numerical, partition, source-integrity and CPU tests
├── .github/workflows/ci.yml  Analysis environment checks
├── DATA_ACCESS.md            Source dataset and frozen manifest instructions
├── DATA_CARD.md              Provenance, data fields and sharing considerations
├── CITATION.cff               Software citation metadata
└── SHA256SUMS.txt             Release integrity digests
```

### What is released

`data/patient_scores_original.csv` contains patient-level scores for the 55 original fits; `data/patient_scores_second_seed.csv` covers the 40 additional fits. The corresponding `*_common.csv` files contain scores restricted to the held-out image intersections. Subjects use consistent pseudonymous tokens for paired analyses. `data/frozen_fold_roles.csv`, `data/source_relative_inventory.csv.gz` and `data/frozen_operations.csv.gz` encode the frozen cohort and overlap assignments; the source images themselves are not redistributed. The path-bearing reconstructed operations were cross-checked against the recorded replacement totals and all independently archived 100%-overlap second-seed operations.

The original experiment's pseudonymized per-record tokens are preserved separately in `data/frozen_records.csv`, `injected_records.csv` and `replaced_training_records.csv`; those tokens are not falsely represented as a verified one-to-one file-path crosswalk. `data/original_representation_runs.csv` records the original diagnosis-unmatched exploratory measurements, while the diagnosis-matched representation summary refers to the second-seed Keras models.

## Full model reproduction

Obtain authorized NTUA MRI images as described in [DATA_ACCESS.md](DATA_ACCESS.md), and place the frozen 42,147 source-relative records under the indicated dataset root. Do not substitute the entire current NTUA dataset for the frozen cohort. Verify the cohort and five fold roles against `data/` before starting GPU training.

- **Original Keras:** use `pipelines/keras/runner.py` with `PDMRI_DATA_ROOT`, `PDMRI_OUTPUT_ROOT`, `PDMRI_OUTER_FOLD` and `PDMRI_SEED_OFFSET=0` or `100000`. The graded ResNet50V2 conditions are selected only for the original seed.
- **Batched Keras repeat:** for second-seed folds **2, 3 and 4**, use `pipelines/keras/runner_batched.py` with `PDMRI_OUTER_FOLD` in {2,3,4} and `PDMRI_SEED_OFFSET=100000`. This runner retains the executed sequence **decode → batch → augmentation → backbone preprocessing** and the recorded deterministic TensorFlow options. Folds 0–1 use the original-per-image augmentation path.
- **TorchVision:** from `pipelines/torchvision`, use `python -m src.run_experiment --help` and the frozen YAML settings plus the authorized input manifests. Do not replace the original TorchVision implementation with a different library training recipe.

Framework versions, pretrained-weight requirements, cache semantics and runtime information are in [docs/COMPUTE.md](docs/COMPUTE.md). Keras's **2,000-replicate per-run internal diagnostics** are distinct from the **10,000-replicate paired subject-bootstrap intervals** reported in the article and regenerated in `scripts/reproduce_full_statistics.py`.

## Citation, source rights and contact

The code is licensed under [MIT](LICENSE); that licence does **not** relicense the NTUA MRI source data. See [DATA_ACCESS.md](DATA_ACCESS.md) for the original dataset publication and access conditions, and [DATA_CARD.md](DATA_CARD.md) for the source-derived manifest and individual prediction records. For technical queries, use the repository's GitHub Issues tab. Cite the software using `CITATION.cff` and the final journal article once its bibliographic metadata is available.

```bibtex
@misc{ahmad2026parkinsonoverlap,
  title = {Sensitivity of Patient-Level Parkinson Disease MRI Evaluation to Training--Test Subject Overlap},
  author = {Ahmad, Baraa Alsyed and Alqaraleh, Saed},
  year = {2026},
  howpublished = {Research software, GitHub},
  url = {https://github.com/saed-alqaraleh/parkinson-mri-overlap-reproducibility}
}
```
