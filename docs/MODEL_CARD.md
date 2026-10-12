# Model card: ordinal AGA severity grading (ResNet-50 + CORN)

## Model
ImageNet-pretrained ResNet-50, global average pooling, dropout 0.35, and a CORN ordinal head
(K−1 = 6 conditional logits). Training has two stages: (A) frozen trunk, head only, Adam 1e-3,
≤ 15 epochs; (B) fine-tuning from `conv4_block1_1_conv` at 1e-5, ≤ 35 epochs, with BatchNorm frozen.
Early stopping and checkpointing use validation QWK. Per-sample class weights are applied.
Augmentation lives inside the model and is aggressive on colour. Inputs are 224×224 RGB. The
prediction is `round(Σ_k P(y>k))`. Confidence is temperature-scaled (T = 1.868).

## Intended use
Research prototype. Supports methods research on ordinal medical grading and could serve as a
quantitative second opinion for dermatologist review.

**Out of scope:** autonomous diagnosis, treatment recommendation, any clinical deployment.
This is not a regulated medical device.

## Data
`hair_norwood_hamilton` v1 (Roboflow Universe, CC BY 4.0). The 5,783 files resolve to
1,447 independent photographs. Six groups with internally conflicting labels were excluded, which
leaves **1,441** images (stages 1–7: 721 / 468 / 104 / 57 / 36 / 19 / 36). From these, a stratified
20 % locked test set (n = 289) was drawn, with 1,152 development images.
Split manifest SHA-256: `dc94878c954130dc3aa7140dae1043de7f5660f27d41f910b7ec846a3f51e193`.

The vendor train/valid/test split was discarded. 53.2 % of its test set has a near-duplicate in
train, and both the augmentation marker and the source collection are confounded with the label.

## Performance: locked test set (n = 289, evaluated once)

| Metric | Value | 95 % CI |
|---|---|---|
| Quadratic-weighted κ | 0.891 | [0.832, 0.927] |
| MAE (stages) | 0.311 | [0.253, 0.377] |
| Macro-F1 | 0.641 | [0.533, 0.738] |
| Balanced accuracy | 0.652 | [0.546, 0.775] |
| Exact accuracy | 0.720 | [0.671, 0.772] |
| Within ±1 stage | 0.976 | [0.955, 0.993] |
| Linear-weighted κ / unweighted κ | 0.748 / 0.565 | — |
| ECE (exact / ±1 stage) | 0.097 / 0.019 | — |

Per-stage recall: I 0.792, II 0.670, III 0.429, IV 0.750, V 0.714, VI 0.333 (n = 3), VII 0.875.
For stages V–VII, use the pooled cross-validated results in the paper (Table II) rather than this
single hold-out.

**Reference points.** Human inter-observer reliability on this scale is ICC 0.63–0.68. A colour-only
logistic regression scores QWK 0.722 on the vendor split and 0.444 on the audited split. Trained on
the vendor split, the same model reports QWK 0.972 and 87.1 % accuracy, which overstates
generalisation.

**Triage use.** If only predictions with P(y = ŷ) ≥ 0.7 are kept, 49 % of cases remain, at 90.8 %
exact and 100 % adjacent accuracy.

## Limitations
- **Small effective sample.** There are 1,441 independent images. Stage VI has 19 in total and 3 in
  the test set.
- **No patient identifiers.** Independence holds at photograph level, not patient level. Table VI
  of the paper bounds the effect.
- **Irreversible preprocessing.** Images were stretch-resized to 640×640 and EXIF data was stripped.
- **Single view.** Images are mostly top-down or vertex photographs, so frontal recession is often
  not visible.
- **Population.** One dominant source, with no demographic metadata. Some images show female
  scalps, which the Norwood scale does not cover.
- **Explainability.** Grad-CAM is a qualitative attention audit, not causal evidence. The
  peripheral-attention ratio is a geometric proxy.
- **External validation** has not been performed.
