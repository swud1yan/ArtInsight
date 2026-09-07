"""Small two-dimensional PCA demonstration for teaching and defense."""

from __future__ import annotations

import numpy as np

from controllers.gui_dtos import PCAAlgorithmDemoStep
from core.pca.pca_steps import (
    compute_covariance,
    compute_eigen,
    compute_mean,
    mean_center,
    select_components,
    transform,
)


class PCAAlgorithmDemoBuilder:
    """Build deterministic visual PCA steps from simple two-dimensional points."""

    def build_demo_steps(self) -> list[PCAAlgorithmDemoStep]:
        """Return a compact PCA teaching sequence independent of project data."""
        original = np.asarray(
            [
                [1.0, 2.0],
                [2.0, 2.6],
                [3.0, 3.2],
                [4.0, 3.9],
                [5.0, 5.1],
                [6.0, 5.8],
            ],
            dtype=np.float64,
        )
        mean = compute_mean(original)
        centered = mean_center(original)
        covariance = compute_covariance(centered)
        eigenvalues, eigenvectors = compute_eigen(covariance)
        projection = select_components(eigenvalues, eigenvectors, 1)
        coordinates = transform(centered, projection)
        reconstructed = coordinates @ projection.T

        principal_vector = eigenvectors[:, 0]
        secondary_vector = eigenvectors[:, 1]
        return [
            PCAAlgorithmDemoStep(
                title="1. 原始二维样本",
                explanation=(
                    "从一个简单的二维点集开始。每个点可以理解为一幅作品的两个"
                    "简化视觉特征，例如颜色强度和纹理强度。"
                ),
                table_headers=["样本", "x1", "x2"],
                table_rows=_matrix_rows(original),
                points=original,
                annotation="观察点云沿斜向拉伸，说明两个特征存在共同变化方向。",
            ),
            PCAAlgorithmDemoStep(
                title="2. 计算均值并中心化",
                explanation=(
                    "PCA 首先计算每一列特征的均值，然后把所有点平移到以原点为"
                    "中心的位置。中心化不会改变点之间的相对结构。"
                ),
                table_headers=["样本", "x1-mean", "x2-mean"],
                table_rows=_matrix_rows(centered),
                points=centered,
                highlight_mean=(0.0, 0.0),
                annotation=f"原始均值 = ({mean[0]:.2f}, {mean[1]:.2f})。",
            ),
            PCAAlgorithmDemoStep(
                title="3. 协方差矩阵",
                explanation=(
                    "协方差矩阵刻画不同特征如何共同变化。非对角线越大，表示两个"
                    "特征越倾向于一起增大或减小。"
                ),
                table_headers=["", "x1", "x2"],
                table_rows=[
                    ["x1", f"{covariance[0, 0]:.4f}", f"{covariance[0, 1]:.4f}"],
                    ["x2", f"{covariance[1, 0]:.4f}", f"{covariance[1, 1]:.4f}"],
                ],
                points=centered,
                annotation="协方差矩阵是后续特征值分解的对象。",
            ),
            PCAAlgorithmDemoStep(
                title="4. 特征值与主方向",
                explanation=(
                    "对协方差矩阵做特征值分解。最大特征值对应的特征向量就是数据"
                    "方差最大的方向，也就是第一主成分 PC1。"
                ),
                table_headers=["主成分", "特征值", "方向x", "方向y"],
                table_rows=[
                    [
                        "PC1",
                        f"{eigenvalues[0]:.4f}",
                        f"{principal_vector[0]:.4f}",
                        f"{principal_vector[1]:.4f}",
                    ],
                    [
                        "PC2",
                        f"{eigenvalues[1]:.4f}",
                        f"{secondary_vector[0]:.4f}",
                        f"{secondary_vector[1]:.4f}",
                    ],
                ],
                points=centered,
                vector_start=(0.0, 0.0),
                vector_end=(principal_vector[0] * 4.0, principal_vector[1] * 4.0),
                annotation="箭头表示 PC1，即保留信息最多的投影方向。",
            ),
            PCAAlgorithmDemoStep(
                title="5. 投影到第一主成分",
                explanation=(
                    "把中心化后的点投影到 PC1 上，就得到一维表示。此时维度降低，"
                    "但尽量保留原始数据中最大的变化趋势。"
                ),
                table_headers=["样本", "PC1坐标"],
                table_rows=[
                    [f"S{index + 1}", f"{value[0]:.4f}"]
                    for index, value in enumerate(coordinates)
                ],
                points=reconstructed,
                vector_start=(0.0, 0.0),
                vector_end=(principal_vector[0] * 4.0, principal_vector[1] * 4.0),
                annotation="投影后的点落在 PC1 方向线上，这是降维后的几何含义。",
            ),
        ]


def _matrix_rows(matrix: np.ndarray) -> list[list[str]]:
    """Format a two-column matrix into teaching table rows."""
    return [
        [f"S{index + 1}", f"{row[0]:.4f}", f"{row[1]:.4f}"]
        for index, row in enumerate(matrix)
    ]
