# PCA and Feature Comparison Experiment

Publication status: **PUBLICATION ELIGIBLE**

## Dataset

- Dataset: `course_dataset`
- Samples: `600`
- Classes: `6`
- Compliance status: `VALID_REAL_DATA`
- Compliance message: Real art dataset satisfies the minimum experiment threshold.
- Source note: Image dataset loaded from D:\PCA\data\course_dataset

## PCA Implementation Comparison

| Metric | Manual PCA | sklearn PCA | Difference |
| --- | ---: | ---: | ---: |
| Median runtime | 12.761 ms | 30.492 ms | 0.42x manual/sklearn |
| Reconstruction MSE | 0.39784186 | 0.39784186 | 0.00e+00 |
| Explained-ratio max abs diff | 2.78e-16 | - | - |
| Component max abs diff after sign alignment | 4.54e-15 | - | - |
| Transformed-coordinate RMSE after sign alignment | 2.88e-14 | - | - |

## Feature Classification Comparison

| Rank | Feature set | Dimension | PCA k | Accuracy mean | Accuracy std |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `color_texture` | 226 | 12 | 47.17% | 2.15% |
| 2 | `color_hist` | 192 | 12 | 41.00% | 2.20% |

## Conclusion

The best feature representation in this run is `color_texture` with 47.17% mean accuracy.
Manual PCA and sklearn PCA produce numerically equivalent principal components after the expected sign alignment; manual PCA is faster on this medium-width covariance benchmark because the feature dimension is modest and the implementation uses a direct symmetric eigendecomposition. The manual implementation is retained for teaching, traceability, and step-by-step linear algebra explanation.
`color_texture` improves over `color_hist` by 6.17% absolute accuracy, showing that texture statistics provide complementary style information beyond color distribution.
