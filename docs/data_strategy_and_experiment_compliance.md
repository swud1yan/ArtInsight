# ArtInsight Data Strategy and Experiment Compliance

## 1. Three-Level Data Architecture

### Level 1: Primary Dataset

Purpose: all SCI/IET paper-level quantitative results.

Required real art dataset candidates:

- WikiArt: large-scale artwork images with style labels.
- Kaggle Painter by Numbers: large artwork image collection with metadata.
- BAM: Behance Artistic Media dataset, suitable for broader visual-art studies.

Recommended primary datasets:

| Dataset | Typical scale | Category structure | Suitability for PCA-based style analysis |
| --- | ---: | --- | --- |
| WikiArt | Common research subsets report about 80K-85K paintings; exact count depends on snapshot. | Usually organized by style folders; artist/title can be inferred from path or metadata if available. | Best primary choice for style PCA because it contains canonical art movements and real painting images. |
| Kaggle Painter by Numbers | Kaggle-hosted competition data; related Kaggle/WikiArt-derived multitask datasets are around 100K paintings. Exact count must be recomputed after download. | CSV metadata commonly includes filename, artist, style/genre/title fields. | Strong for metadata-driven experiments and artist/style/genre ablation; loader supports CSV-driven labels. |
| BAM | Large-scale Behance Artistic Media dataset with contemporary artwork and rich attributes. | Attribute labels for media/content/emotion rather than only classical painting style. | Useful as a secondary robustness/domain-shift dataset, not the first choice for classical style labels. |

Important counting rule:

- The manuscript must report local, reproducible counts from the downloaded snapshot, not copied approximate counts.
- Generate and archive a manifest containing image path, dataset name, style label, artist/title when available, split id, and checksum if possible.

Status in the current workspace:

- `data/datasets/` does not contain a real art image dataset.
- REQUIRED DOWNLOAD STEP: download or prepare a real dataset before running paper experiments.

Minimum recommended structure:

```text
data/datasets/WikiArt/
  Impressionism/
    artist_or_other_subfolder/
      image_001.jpg
  Cubism/
    image_002.jpg
```

Minimum threshold used by the compliance checker:

- At least 3 style/classes.
- At least 30 real images per class for validation-scale experiments.
- Larger full-paper experiments should use substantially more data and report exact split details.

### Level 2: Validation Subset

Purpose: fast pipeline verification with real images.

Requirements:

- Must be sampled from Level 1 real art data.
- Must preserve real labels and dataset provenance.
- Can be smaller than the full dataset, but cannot be synthetic.
- Can be used to validate preprocessing, feature extraction, PCA fitting, KNN classification, retrieval, and report generation.

Publication status:

- May be reported only as a pilot or ablation subset if sampling rules are clearly documented.
- Main paper claims must be based on Level 1.

### Level 3: Synthetic Debug Data

Purpose:

- Pipeline smoke tests.
- Bug checks.
- Feature shape validation.
- PCA numerical sanity checks.

Strict limitation:

- DEBUG ONLY.
- Must not be used for paper accuracy, retrieval quality, PCA conclusion, feature ablation, or final comparison tables.
- Must be clearly labeled as non-publication data in any generated debug report.

## 2. Experiment Rules

Experiments requiring real data:

- Manual PCA vs sklearn PCA timing on art features.
- Manual PCA vs sklearn PCA reconstruction and projection comparison for paper tables.
- Color-only vs color+texture feature classification accuracy.
- Retrieval Top-N evaluation.
- Style-map clustering analysis.
- Any visual example used as evidence in the manuscript.

Experiments allowed with synthetic debug data:

- Unit tests.
- Shape and dtype checks.
- Exception-path checks.
- PCA algebra sanity checks where no performance or application claim is made.
- CI smoke tests.

Results allowed in paper:

- Results from Level 1 Primary Dataset.
- Carefully identified Level 2 real validation subset results, when reported as validation/pilot analysis.

Results prohibited in paper:

- Any Level 3 synthetic debug result.
- Any auto-generated fallback benchmark created because real data is missing.

## 3. Compliance Function

The project provides:

```python
check_dataset_compliance(dataset_root: Path | None, debug_synthetic: bool = False)
```

It returns:

- `VALID_REAL_DATA`: real dataset meets the configured threshold.
- `INSUFFICIENT_DATA`: missing path or too few real images/classes.
- `DEBUG_ONLY`: synthetic debug mode was explicitly requested.

The real dataset loader module is:

```python
experiments.art_dataset_loader.ArtDatasetLoader
```

Supported layouts:

- `DatasetFormat.WIKIART`: `root/style_name/.../image.jpg`
- `DatasetFormat.KAGGLE_PAINTER_BY_NUMBERS`: `root/train_info.csv` with image filename and style/genre columns.
- `DatasetFormat.AUTO`: infer Kaggle format when conventional metadata CSV files exist, otherwise use WikiArt folder layout.

Standardization:

- Resize to a fixed size, default `128x128`.
- Convert loaded images to RGB uint8 for a uniform research representation.
- Provide BGR uint8 view for OpenCV-based feature extraction.
- Provide optional float32 normalized image in `[0, 1]`.

## 4. SCI/IET Risk Assessment

The previous synthetic fallback behavior is not compliant with SCI/IET-style experimental standards if used as a main result. It would create the following review risks:

- Dataset validity challenge: reviewers can reject claims because the data do not represent real artworks.
- External validity failure: synthetic feature distributions cannot support conclusions about art styles.
- Reproducibility ambiguity: fallback behavior can silently change experiment meaning depending on local files.
- Misleading performance claims: PCA timing and classification accuracy on synthetic data do not establish real-world system quality.

Required correction:

- Experiments must stop when real data are missing.
- Synthetic data must require an explicit debug flag.
- Reports generated from synthetic debug mode must be labeled non-publication.
- Manuscript tables must cite real dataset source, class count, image count, train/test split, preprocessing, feature settings, PCA settings, and random seeds.

## 5. Current Project Status

Current data status:

- Primary Dataset: missing.
- Validation Subset: missing.
- Synthetic Debug Data: allowed only with explicit debug mode.

Current publication readiness:

- Code pipeline is ready for real data experiments.
- Paper-level quantitative claims are blocked until real art data are downloaded and registered.
