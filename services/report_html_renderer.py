"""HTML rendering for ArtInsight analysis reports."""

from __future__ import annotations

from html import escape
from pathlib import Path

from services.report_models import ReportSnapshot


class ReportHtmlRenderer:
    """Render a report snapshot to an HTML file."""

    def write(self, snapshot: ReportSnapshot, output_path: Path) -> None:
        """Write a self-contained HTML report."""
        output_path.write_text(self.build_html(snapshot), encoding="utf-8")

    def build_html(self, snapshot: ReportSnapshot) -> str:
        """Build a self-contained HTML summary report."""
        style_rows = "".join(
            f"<tr><td>{escape(label)}</td><td>{count}</td></tr>"
            for label, count in snapshot.style_distribution.items()
        )
        dataset_rows = "".join(
            "<tr>"
            f"<td>{escape(row.name)}</td>"
            f"<td>{row.artwork_count}</td>"
            f"<td>{row.imported_at:%Y-%m-%d}</td>"
            f"<td>{escape(row.path)}</td>"
            "</tr>"
            for row in snapshot.datasets
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{escape(snapshot.report_name)}</title>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; color: #1f2a30; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
td, th {{ border: 1px solid #d9e1e2; padding: 8px 10px; }}
th {{ background: #eef4f4; }}
.metric {{ display: inline-block; margin-right: 24px; font-weight: 600; }}
</style>
</head>
<body>
<h1>{escape(snapshot.project_name)}</h1>
<p>{escape(snapshot.subtitle)}</p>
<p>报告名称：{escape(snapshot.report_name)} | 生成时间：{snapshot.created_at:%Y-%m-%d %H:%M}</p>
<p><span class="metric">作品：{snapshot.artwork_count}</span>
<span class="metric">风格：{snapshot.style_count}</span>
<span class="metric">特征：{snapshot.feature_count}</span></p>
<h2>数据集</h2>
<table><tr><th>名称</th><th>作品数</th><th>导入日期</th><th>路径</th></tr>{dataset_rows}</table>
<h2>风格分布</h2>
<table><tr><th>风格</th><th>作品数</th></tr>{style_rows}</table>
<h2>PCA结果</h2>
<p>保留主成分 K={snapshot.pca_summary.n_components}，
累计解释率 {snapshot.pca_summary.cumulative_at_k:.2%}。</p>
</body>
</html>
"""
