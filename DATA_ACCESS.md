# NTUA MRI source and frozen cohort

**Source dataset:** [NTUA Parkinson Dataset](https://github.com/ails-lab/ntua-parkinson-dataset). Its documentation describes **78 individuals** (55 PD, 23 controls), **43,087 MRI images** and **920 DaT scans**, and says that the images are available for **academic use**. The repository does not establish an explicit broad redistribution licence for individual files. Obtain the images through the source's permitted academic-access route and observe the associated conditions.

**Study subset:** the frozen experiment used **68 subjects** (48 PD, 20 non-PD), **42,147 MRI image records**. The precise cohort, outer-fold assignments and overlapping-image construction are distinct from simply using the complete public source collection.

## Reconstruction from an authorized local copy

1. Acquire the MRI collection from the original NTUA source; retain its folder organization under a local `Parkinson/` directory.
2. `data/source_relative_inventory.csv.gz` provides the frozen record list as paths **relative to `Parkinson/`**, along with anonymous subject token and diagnosis class. Resolve each relative path against the authorized local source root. Verify the 42,147 records and 68 unique subjects; stop on any missing or extra record.
3. `data/frozen_fold_roles.csv` records the five outer subject folds and independent validation roles. The role configuration is not regenerated from a training seed.
4. `data/frozen_operations.csv.gz` records every selected training-set injection and diagnosis-matched removal at 25%, 50%, 75% and 100%. Run `python -m scripts.regenerate_partitions` to regenerate the path-based operations from the frozen inventory and validate them against the original token-only counts and the independently saved repeat-run endpoint manifests.
5. The path-free experiment evidence is also preserved as `data/frozen_records.csv`, `data/injected_records.csv`, and `data/replaced_training_records.csv`. These original token identifiers are distinct from the relative paths; they are not a falsely reconstructed one-to-one file-token mapping.
6. Run the frozen pipeline with its recorded framework versions, pretrained weights, fold, overlap and training seed. Use the original intervention rather than a new random partition.

**Source attribution:** Tagaris, A., Kollias, D., Stafylopatis, A., Tagaris, G., and Kollias, S. (2018). *Machine Learning for Neurodegenerative Disorder Diagnosis—Survey of Practices and Launch of Benchmark Dataset*. International Journal on Artificial Intelligence Tools, 27(3), 1850011.

The repository contains **no MRI image pixels, fitted model weights, or patient names**. The source-relative paths and per-subject model scores are derived research records, not a grant of rights to disclose additional NTUA clinical metadata. Please assess any dataset-use and participant-privacy restrictions before disseminating these derived files outside the authorized research context.
