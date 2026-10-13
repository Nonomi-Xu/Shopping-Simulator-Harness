# Shopping Simulator

本仓库根目录覆盖整个 `Shopping-Simulator/`，管理 BM25 基线和 Harness。
两个项目均为实体目录，不使用软链接。Git 历史和原有远程仓库保持连续。

```text
Shopping-Simulator/
  .git/
  .gitignore
  BM25/                       固定 Top-5 检索与单次 Agent 决策基线
  harness/                    Shopping Agent Runtime 与评测框架
    data/                     本地商品数据与索引（忽略）
  outputs/                    唯一默认结果根目录（忽略）
    bm25/evaluations/          BM25 批量评测；历史运行在 bm25/ 下
    rollouts/                 Harness 批量运行
    episodes/                 长程交互评测
    experiments/              配对实验
    reports/                  离线重建报告
  Shopping-Simulator-grpo/     独立本地项目，整个目录由 Git 忽略
```

安装、使用和测试分别见 [BM25 文档](BM25/README.md) 与 [Harness 文档](harness/README.md)。
两个 Python 包分别从各自目录安装，命令也可从各自目录执行。

默认结果路径由源码位置解析为绝对路径，与启动目录无关。每次批量运行使用
UTC 微秒时间和 UUID 创建独立目录，并返回完整的绝对 `output_dir`。
BM25 的 `--output-root`（兼容 `--output-dir`）和 `--resume-dir` 只接受绝对路径或
`~` 路径；使用默认位置时无需传入输出参数。Harness 的相对结果路径以此仓库根目录解析。

旧 BM25 和 Harness 结果已搬到顶层 `outputs/`，文件内容保留，恢复/回放时使用新位置。
GRPO 保留独立的本地文件与 Git 仓库，不作为子模块加入，不随本仓库提交或推送。
数据、完整评测结果、环境配置、临时文件和本地报告由 `.gitignore` 忽略。
BM25 的两份历史基准汇总保留在 `BM25/results/`，用于文档引用。

离线测试：

```bash
(cd BM25 && python -m unittest discover -s tests -v)
(cd harness && python -m unittest discover -s src/harness/test -p 'test_*.py' -v)          # 主线
(cd harness && python -m unittest discover -s src/harness/branch/test -p 'test_*.py' -v)  # 可选支线
```

## Original / Test1 与新评测

Original 为全部 23,421 题；Test1 冻结为原始任务顺序第 1–500 题。
清单与后续 Test2 的冻结方法见 [datasets](datasets/README.md)。
两种方法共用冻结 ID、源数据和商品索引；新 benchmark 逐题实时记录完整链路，
只在内存中保留并发任务的完整轨迹。运行与离线报告见
[评测说明](harness/docs/test1-benchmark.md)。
