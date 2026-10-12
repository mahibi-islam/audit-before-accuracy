# Model card — ordinal AGA severity grading

**Model.** ResNet50 + CORN. ImageNet-pretrained backbone, two-stage transfer learning
(head-only @ lr 0.001, then fine-tune from `conv4_block1_1_conv` @ lr 1e-05,
BatchNorm frozen). Head: `corn`. Input 224x224 RGB, ImageNet
normalisation. Model selection on validation QWK. Confidence temperature-scaled (T = 1.868).

**Intended use.** Research prototype. A quantitative second opinion for dermatologist review and
for methods research on ordinal medical grading.

**Out of scope.** Autonomous diagnosis; treatment recommendation; any clinical deployment; any
use as a substitute for clinical judgement. Not a regulated medical device.

## Data
Source: `hair_norwood_hamilton`, Roboflow Universe (onuralpszr), CC BY 4.0,
https://universe.roboflow.com/onuralpszr/hair_norwood_hamilton

The published dataset contains 5,783 files, but they resolve to **1,447 independent
photographs**; the remainder are colour-augmented copies. The vendor train/valid/test split was
discarded after audit (§2) and replaced with a group-disjoint protocol using one image per group
(n = 1,441). Split manifest SHA-256: `dc94878c954130dc3aa7140dae1043de7f5660f27d41f910b7ec846a3f51e193`.

Audit findings, all reproduced in §2 of the notebook:
1. 53.2% of the vendor test set has a near-duplicate in vendor train.
2. Vendor augmentation is confounded with the label (stage 1: 0% augmented; stages 3–7: ~80%).
3. Source collection is confounded with the label (two collections contribute no stage-1 images).
4. Effective class imbalance is 37:1, not the ~7:1 the file counts suggest.

## Performance — locked test set, n = 289, evaluated once
| metric | value | 95% CI |
|---|---|---|
| QWK | 0.891 | [0.832, 0.927] |
| MAE | 0.311 | [0.253, 0.377] |
| macro_f1 | 0.641 | [0.533, 0.738] |
| balanced_acc | 0.652 | [0.546, 0.775] |
| accuracy | 0.720 | [0.671, 0.772] |
| within ±1 stage | 0.976 | [0.955, 0.993] |
| ECE (calibrated) | 0.097 | — |

Per-class performance: see `figures/11_confusion_locked.png` and the table in §11. Per-class
figures for stages with single-digit support are not interpretable in isolation; the pooled
cross-validated results (§10, `benchmark_cv.csv`) are the per-class evidence.

**Reference points.** Human inter-observer reliability on the Hamilton–Norwood scale is
ICC 0.63–0.68 (Guarrera et al., *Int. J. Trichology* 2009). A colour-only logistic regression —
which cannot see hair at all — scores QWK 0.722 on the *unaudited* split and
QWK 0.444 on the audited one.

## Known limitations
- **Small effective sample.** ~1,441 independent images; the rarest stage has
  19 images in total. Rare-stage metrics carry very wide CIs.
- **No patient identifiers.** Grouping is by photograph (filename + perceptual hash). If one
  patient was photographed in two separate sessions, those sessions may still straddle the split.
  Patient-level independence is assumed, not verified.
- **Irreversible preprocessing.** Images arrived already stretch-resized to 640×640 with EXIF
  stripped; original framing and aspect ratio cannot be recovered.
- **Single-view.** Mostly top-down/vertex photographs. The Norwood scale is defined on frontal
  *and* vertex patterns; frontal hairline recession is often not visible here.
- **Label subjectivity.** 6 groups carry internally conflicting labels, all adjacent-stage —
  consistent with the scale's known unreliability. These were excluded.
- **Population.** Predominantly one clinical/geographic source; no demographic metadata, so
  fairness across skin tone, age or sex cannot be assessed. Some images appear to be female scalps,
  which the Norwood scale does not cover.
- **Confounds persist.** The source-collection confound (§2.9) exists inside the audited data too;
  aggressive colour augmentation mitigates but does not eliminate it.

## Reproducibility
seed 1337 · TF 2.20.0 · split lock `dc94878c954130dc3aa7140dae1043de...`
Artefacts: `benchmark_cv.csv`, `leakage_ablation.csv`, `split_manifest.csv`,
`audit_manifest.csv`, `final_model.weights.h5`, `figures/`.
