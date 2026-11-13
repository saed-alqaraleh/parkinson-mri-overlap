# Data card — frozen evaluation evidence

**Objective.** Measure how a combined training–test subject overlap intervention affects reported patient-level performance in a repeated-slice MRI design. These analyses evaluate methodology, not clinical diagnostic validity.

**Cohort and design.** The frozen NTUA inventory contains 68 subject identifiers (48 PD, 20 non-PD) and 42,147 MRI image records. Five fixed outer subject folds include independent validation. For each selected test subject, at most 50 slices and 20% of available slices enter the training pool, while at least two slices remain for testing. Diagnosis-matched training slices are removed to preserve the fixed budget. Original and repeat training seeds share the same partitions.

**Released fields.** Original and second-seed predictions from 95 fitted cases, subject class labels, anonymous stable subject keys, number of held-out slices, model probability, validation-selected operating threshold, architecture, fold and nominal overlap. Common-held-out score tables restrict evaluations to the intersection of held-out images. Source-relative MRI paths, frozen fold assignments, original token-only record inventory, injected/removed image selections and aggregated content-audit outcomes support reconstruction.

**Privacy.** The package contains no raw MRI, names, full scans, demographics, private Drive paths, credentials or pretrained checkpoints. Pseudonymous subject scores and source-relative paths may nevertheless be linkable to the openly described source dataset, which itself contains clinical metadata. They should not be treated as anonymized medical records or published without an appropriate data-use and privacy review.

**Image-content audit.** The cross-identifier exact-match finding concerns **one uniform all-zero frame** present under two recorded subject identifiers in the fold-4 training/validation partitions; this is a pixel-content match, not evidence that two identifiers belong to the same patient. Other high-similarity candidates remain unconfirmed scan matches. Exact image similarity is distinct from verification of true participant identity.

**Interpretation.** Effects vary across backbones and across training seeds; they represent overlap sensitivity under this combined intervention, **not a transferable inflation factor**. A positive Swin-Tiny response is not established. The 10,000-replicate intervals are conditional on the fitted models and recorded partitions, not estimates of variability over an unlimited number of training seeds.

**Provenance.** Inputs are frozen model outputs and source-relative path manifests, not synthetic experimental outcomes. Code is MIT licensed; the source MRI rights are separate and described in `DATA_ACCESS.md`.
