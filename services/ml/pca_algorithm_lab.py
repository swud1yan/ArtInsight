"""Compact PCA step previews for the algorithm laboratory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCALabStep:
    """One compact PCA teaching step."""

    title: str
    description: str
    column_headers: list[str]
    rows: list[list[str]]
    note: str


class PCAAlgorithmLabBuilder:
    """Create small matrix previews for step-by-step PCA teaching."""

    _MAX_ROWS = 6
    _MAX_COLUMNS = 6

    def build_steps(
        self,
        *,
        scaled_matrix: np.ndarray,
        prepared_matrix: np.ndarray,
        covariance_matrix: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        coordinates: np.ndarray,
        titles: list[str],
    ) -> list[PCALabStep]:
        """Build ordered teaching steps from already-computed PCA arrays."""
        return [
            self._matrix_step(
                title="1. 标准化后的特征矩阵 X",
                description="训练流程先用 StandardScaler 统一各特征尺度。",
                matrix=scaled_matrix,
                row_labels=titles,
                note="表格只展示前几行和前几列，完整矩阵仍保存在训练数据中。",
            ),
            self._matrix_step(
                title="2. 去均值矩阵 X_centered",
                description="PCA 对每一列减去均值，使每个特征围绕 0 波动。",
                matrix=prepared_matrix,
                row_labels=titles,
                note="后续协方差矩阵由该去均值矩阵计算得到。",
            ),
            self._matrix_step(
                title="3. 协方差矩阵 Cov(X)",
                description="协方差刻画不同特征共同变化的方向和强度。",
                matrix=covariance_matrix,
                row_labels=None,
                note="本项目使用样本协方差，分母为 n-1。",
            ),
            self._eigen_step(eigenvalues, eigenvectors),
            self._matrix_step(
                title="5. 投影结果 Z = X_centered W",
                description="选择前 K 个特征向量组成投影矩阵，将样本映射到主成分空间。",
                matrix=coordinates,
                row_labels=titles,
                note="PCA分析页的二维散点图使用这里的 PC1 与 PC2。",
            ),
        ]

    def _matrix_step(
        self,
        *,
        title: str,
        description: str,
        matrix: np.ndarray,
        row_labels: list[str] | None,
        note: str,
    ) -> PCALabStep:
        """Create a compact preview for a matrix-valued PCA step."""
        values = np.asarray(matrix, dtype=np.float64)
        rows = min(values.shape[0], self._MAX_ROWS)
        columns = min(values.shape[1], self._MAX_COLUMNS)
        headers = ["样本", *[f"C{index + 1}" for index in range(columns)]]
        preview_rows = []
        for row_index in range(rows):
            label = self._row_label(row_index, row_labels)
            preview_rows.append(
                [
                    label,
                    *[
                        self._format_number(values[row_index, col])
                        for col in range(columns)
                    ],
                ],
            )
        return PCALabStep(
            title=title,
            description=description,
            column_headers=headers,
            rows=preview_rows,
            note=note,
        )

    def _eigen_step(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> PCALabStep:
        """Create a compact eigenvalue/eigenvector preview."""
        values = np.asarray(eigenvalues, dtype=np.float64)
        vectors = np.asarray(eigenvectors, dtype=np.float64)
        rows = min(values.shape[0], self._MAX_ROWS)
        vector_columns = min(vectors.shape[0], self._MAX_COLUMNS)
        headers = [
            "主成分",
            "特征值",
            *[f"v{index + 1}" for index in range(vector_columns)],
        ]
        preview_rows = [
            [
                f"PC{index + 1}",
                self._format_number(values[index]),
                *[
                    self._format_number(vectors[col_index, index])
                    for col_index in range(vector_columns)
                ],
            ]
            for index in range(rows)
        ]
        return PCALabStep(
            title="4. 特征值与特征向量排序",
            description="特征值越大，表示对应主成分解释的数据方差越多。",
            column_headers=headers,
            rows=preview_rows,
            note="投影矩阵 W 由排序后的前 K 个特征向量列组成。",
        )

    def _row_label(self, index: int, row_labels: list[str] | None) -> str:
        """Return a compact row label for a preview table."""
        if row_labels is None or index >= len(row_labels):
            return str(index + 1)
        return row_labels[index][:24]

    def _format_number(self, value: float) -> str:
        """Format matrix values for teaching tables."""
        return f"{float(value):.4f}"
