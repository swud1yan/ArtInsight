# ArtInsight Dataset Acquisition Plan

## Objective

ArtInsight must not produce publication-level results without real artwork images. This plan defines the dataset acquisition path required before SCI/IET experiments are run.

## Recommended Datasets

### 1. WikiArt

Source: WikiArt public artwork collection and widely used WikiArt-derived research snapshots.

Typical scale:

- Around 80K-85K paintings in common research snapshots.
- Common style-classification setups use roughly 25-27 style/category labels.
- Exact image/style/artist counts vary by snapshot and must be recomputed locally after download.

Category structure:

```text
WikiArt/
  Impressionism/
    Claude_Monet/
      image.jpg
  Cubism/
    Pablo_Picasso/
      image.jpg
```

Suitability:

- Primary dataset for PCA-based style analysis.
- Real painting images and art-movement labels align directly with ArtInsight's goal.
- Suitable for feature ablation: color histogram vs color+texture vs HOG/GLCM/LBP.

### 2. Kaggle Painter by Numbers

Source: Kaggle Painter by Numbers competition data.

Typical scale:

- Large artwork collection used for artist/style/genre metadata experiments.
- Related Kaggle/WikiArt-derived multitask painting datasets are about 100K paintings with artist/style/genre labels.
- Exact counts must be computed from downloaded CSV metadata and image files.

Category structure:

```text
PainterByNumbers/
  train_info.csv
  train/
    image_id.jpg
```

Expected metadata columns:

- filename or id
- artist
- style or genre
- title when available

Suitability:

- Strong secondary dataset for metadata-driven validation.
- Suitable for style/genre classification and retrieval experiments.
- Good for checking whether PCA projections remain meaningful across a larger metadata-rich dataset.

### 3. BAM

Source: Behance Artistic Media dataset.

Typical scale:

- Very large Behance-derived artistic-media dataset.
- Labels emphasize artistic media/content/emotion attributes.

Suitability:

- Recommended as optional robustness or domain-shift dataset.
- Less direct than WikiArt for classical style labels, but valuable for showing generalization beyond canonical painting datasets.

## Required Local Preparation

1. Download the selected real dataset according to its license and platform rules.
2. Place it under `data/datasets/`.
3. Keep raw data immutable; write derived manifests/features to separate outputs.
4. Run compliance check before any experiment:

   ```bash
   python experiments/pca_feature_comparison.py --dataset-root data/datasets/WikiArt
   ```

5. If the command reports `REQUIRED DOWNLOAD STEP` or `INSUFFICIENT_DATA`, do not run paper experiments.

## Validation Subset Protocol

Create a small real subset only from Level 1 data:

```text
data/datasets/WikiArt_validation/
  style_a/
  style_b/
  style_c/
```

Rules:

- Minimum 3 classes.
- Minimum 30 images per class for local validation.
- Record sampling seed and selected image list.
- Do not replace this subset with synthetic data.

## Debug Mode

Synthetic debug is allowed only for:

- pipeline smoke testing
- shape validation
- exception-path checks

Command:

```bash
python experiments/pca_feature_comparison.py --debug-synthetic
```

Output is written to `data/exports/*DEBUG_ONLY*` and must not be cited in the paper.
