# 冻结评测数据集

- `Original.json`：原始数据源展开后的全部 23,421 个任务。
- `Test1.json`：Original 源顺序的第 1–500 个任务。
- `Test2.json`：Original 源顺序的第 501–600 个任务（100 题）。本轮把测试题目数量
  由 500 调整为 100，并与 Test1 完全不重叠，因此可复用同一套校验与报告流程。

顺序定义：按商品在原 JSON 数组中的顺序，再按该商品的 `instructions` 数组顺序展开。
每个清单固定源文件 SHA-256、任务 ID 算法版本、源序号和完整有序任务 ID 列表。
不包含 Gold ASIN、答案规格或凭据；源数据仍在被 Git 忽略的 `harness/data/`。
评测会核对源哈希和全部任务 ID，源数据或清单变化时拒绝继续。

旧 `all`/`test` 入口保留为 Original/Test1 的兼容选择；旧随机 test500 和 BM25 test10
结果不能再标为 Test1。历史目录保留原样。

从 `harness/` 运行以下命令可再次校验现有冻结文件（内容一致时不重写）：

```bash
python -m src.harness.benchmark.datasets
```

已创建的清单：Test1 = 第 1–500 题（500 题），Test2 = 第 501–600 题（100 题）。
再冻结另一组时显式给出名称、起点和数量：

```bash
python -m src.harness.benchmark.datasets --name Test3 --start 601 --count 500
```

该命令不会覆盖已存在的不同清单，源哈希或任务 ID 顺序变化时直接报错。
前 500 题可能存在品类和顺序偏差；Test1 用于诊断后，不应再作为独立最终留出集。
Test2 为 100 题时区间较窄，逐题方差更大，报告中的置信区间会相应变宽。
