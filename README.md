# ArtInsight

ArtInsight 是“基于 PCA 的艺术风格分析与相似作品检索系统”。项目按优秀本科毕业设计的软件工程标准开发，采用 PySide6 桌面端、SQLite/SQLAlchemy 数据层、自研 PCA 算法与传统计算机视觉特征。

## 当前可用功能

- 首页：项目统计、模型状态、风格分布图。
- 数据集管理：选择文件夹导入、列表查看、确认删除，导入/删除使用后台任务避免界面卡顿。
- 图片上传：单图预览、快速分类、相似作品检索。
- 特征提取：预处理与多特征组合预览。
- PCA 分析：Scree Plot、累计贡献率、重建误差对比、算法实验室和主成分表格。
- 风格分类：预测风格、置信度、同类别样例。
- 相似检索：Top-N 缩略图网格与作品详情弹窗。
- 风格地图：PCA 二维散点图，悬停查看作品信息。
- 统计分析：风格/作者统计与 PCA 贡献率。
- 报告导出：支持 ReportLab PDF 报告和 HTML 预览报告，包含数据集、PCA 图表、统计图和自动结论。
- 设置：修改数据库路径、数据目录、模型目录、导出目录等配置。

## 运行方式

推荐在已安装项目依赖的 Python 3.12 环境中运行：

```bash
python main.py
```

也可以使用模块入口：

```bash
python -m app.main
```

核心依赖见 `pyproject.toml`，包括 `PySide6`、`numpy`、`opencv-python`、`scikit-image`、`scikit-learn`、`SQLAlchemy`、`matplotlib`、`joblib`、`reportlab`。

如需使用传统依赖文件安装，也可执行：

```bash
pip install -r requirements.txt
```

## 测试与质量检查

项目使用 Python 标准库 `unittest` 作为当前主测试框架：

```bash
python -m unittest discover -s tests -q
```

语法和导入快速检查：

```bash
python -m compileall app controllers core database models repositories services views visualization examples tests
```

测试覆盖重点包括：

- PCA 数学步骤、模型流程、可视化图表与解释数据。
- 图像预处理、颜色/纹理/形状特征提取。
- SQLite/SQLAlchemy 模型、Repository 和数据集管理。
- 训练、分类、相似检索与分页。
- PCA 风格地图、算法实验室、报告导出。
- 后端端到端流程：训练 → 分类 → 检索 → PDF 报告导出。

## 实验对比

PCA 与特征组合对比实验脚本位于 `experiments/pca_feature_comparison.py`。论文级实验必须使用真实艺术图像数据，当前不会在缺少真实数据时自动回退到 synthetic benchmark。

```bash
python experiments/pca_feature_comparison.py --dataset-root data/datasets/WikiArt
```

数据集目录应采用 `数据集/风格名/.../图片` 结构，或使用 Kaggle Painter by Numbers 的 `train_info.csv + train/` 结构。当前 `data/datasets/` 未内置真实数据，需先下载 WikiArt、Kaggle Painter by Numbers 或 BAM 等真实艺术数据集；数据获取方案见 `docs/dataset_acquisition_plan.md`。

仅用于 pipeline/shape/debug 检查时，可显式运行：

```bash
python experiments/pca_feature_comparison.py --debug-synthetic
```

Debug synthetic 输出会自动标记为 `DEBUG_ONLY`，严禁作为论文实验结果。完整数据策略与合规规范见 `docs/data_strategy_and_experiment_compliance.md`。

为降低误用风险，debug synthetic 报告默认写入 `data/exports/`，不会作为论文文档提交。

单次论文实验并导出 CSV/JSON/Markdown：

```bash
python -m experiments.paper_experiment_runner --experiments baseline,feature_ablation,pca_sensitivity --seed 42
```

汇总多个 seed 的 `metrics.csv`，并生成 mean/std 与显著性检验表：

```bash
python -m experiments.paper_experiment_summary --metrics-root data/exports/paper_experiments
```

GUI 中也可以直接使用这些论文结果：

1. 打开“算法实验中心”，设置实验类型、baseline、feature ablation、PCA 维度、Top-K、sample limit 和 Random Seed。
2. 点击“开始实验”，系统会把 300dpi 论文图保存到 `data/exports/paper_experiments/<experiment>_seed_<seed>/figures/`。
3. 实验完成后点击“导出论文结果”，生成 `metrics.csv`、`summary.json`、`report.md`。其中 `metrics.csv` 用于论文表格和统计汇总，`report.md` 可直接整理到实验章节，`figures/` 中的 PNG 可作为论文图。
4. 使用多个 seed 重复运行并导出后，点击“汇总论文结果”，生成 `aggregate_metrics.csv`、`significance_tests.csv` 和 `paper_experiment_summary.md`，用于 mean/std 表和显著性检验结果。

## 报告导出

报告导出页默认生成 PDF，适合答辩和归档；HTML 格式适合快速预览。PDF 使用 ReportLab 排版，并注册本机中文字体以保证中文显示质量。导出文件默认保存到 `data/exports/`。

## 工程结构

- `app/`：应用入口、启动装配、配置、日志和导航定义。
- `controllers/`：GUI 门面、页面适配器、DTO、查询协调。
- `views/`：PySide6 主窗口、页面、通用控件、表格模型、图表组件。
- `core/`：PCA、图像预处理、特征提取等算法模块。
- `services/`：数据集管理、训练、分类、检索等业务服务。
- `database/`、`models/`、`repositories/`：SQLAlchemy 数据访问层。
- `resources/`：QSS 样式与图标资源。
- `data/`：SQLite 数据库、模型、导出报告、缓存和数据集目录。
- `tests/`：算法、数据层和服务层测试。

## 最终验收与答辩

最终验收报告和答辩演示流程见 `docs/final_acceptance_report.md`。
