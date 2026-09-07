# PCA and Feature Comparison Experiment

Publication status: **PUBLICATION ELIGIBLE**

## Dataset

- Dataset: `paper_dataset`
- Samples: `4800`
- Classes: `6`
- Compliance status: `VALID_REAL_DATA`
- Compliance message: Real art dataset satisfies the minimum experiment threshold.
- Source note: Image dataset loaded from D:\PCA\data\paper_dataset

## PCA Implementation Comparison

| Metric | Manual PCA | sklearn PCA | Difference |
| --- | ---: | ---: | ---: |
| Median runtime | 36.682 ms | 63.454 ms | 0.58x manual/sklearn |
| Reconstruction MSE | 0.42030087 | 0.42030087 | 5.55e-17 |
| Explained-ratio max abs diff | 2.22e-16 | - | - |
| Component max abs diff after sign alignment | 6.12e-15 | - | - |
| Transformed-coordinate RMSE after sign alignment | 4.23e-14 | - | - |

## Feature Classification Comparison

| Rank | Feature set | Dimension | PCA k | Accuracy mean | Accuracy std |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `color_texture` | 226 | 12 | 40.94% | 1.59% |
| 2 | `color_hist` | 192 | 12 | 37.17% | 1.36% |

## Conclusion

The best feature representation in this run is `color_texture` with 40.94% mean accuracy.
Manual PCA and sklearn PCA produce numerically equivalent principal components after the expected sign alignment; manual PCA is faster on this medium-width covariance benchmark because the feature dimension is modest and the implementation uses a direct symmetric eigendecomposition. The manual implementation is retained for teaching, traceability, and step-by-step linear algebra explanation.
`color_texture` improves over `color_hist` by 3.77% absolute accuracy, showing that texture statistics provide complementary style information beyond color distribution.
