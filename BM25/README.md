# BM25 Top-5 + Agent 购物推荐评测

这是一个从原 `harness` 中抽出的精简基线：

```text
公开购物需求
    ↓
固定多字段 BM25（只召回 Top 5）
    ↓
一次性提供 5 个商品的公开详情与规格
    ↓
Agent 只能选择其中 1 个 ASIN 及其可用规格
    ↓
确定性 Evaluator 输出 reward 与最终评测结果
```

它删除了原 Harness 中的 Session、反复搜索、逐商品详情工具、Skill 加载、多轮 Agent
循环等部分。模型只有一个终局工具 `select_product`，工具的 ASIN 枚举在每道题上动态
锁定为 BM25 返回的 5 个候选，因此无法扩大召回集。

## 评测口径

- BM25 字段、权重、NFKC 归一化、中英文词元和同分 ASIN 排序与 Harness v3 索引一致；
- `bm25()` 显式为第一列 `asin UNINDEXED` 传入 `0.0` 占位权重，保证标题到卖点的
  7 个业务权重与 FTS5 列位置严格对齐；
- 每个任务只运行一次检索，`LIMIT 5` 是代码常量，不接受命令行修改；
- Agent 可见标题、品牌、类别、价格、描述、属性、规格和规格组合，不可见 Gold ASIN；
- Agent 必须选择恰好一件商品，只能提交该候选中存在且可用的规格；
- 命中 Gold ASIN 且所需规格全对：`reward=1.0`；
- 命中 Gold ASIN 但规格缺失或错误：`reward=0.25`；
- 未命中：`reward=0.0`；越界或非法规格的结果记为无效；
- 汇总同时报告 `top5_recall` 和 `strict_success_rate`，用于区分检索瓶颈与 Agent 决策瓶颈。

## 数据

BM25 是 Shopping-Simulator 仓库根目录下的实体目录 `BM25/`，与 `harness/` 并列，默认只读使用仓库数据，不复制大文件：

```text
../harness/data/fine_items_eval_train_all.json
../harness/data/product_search.sqlite3
```

也可以在主命令前通过 `--products-path` 和 `--index-path` 指定其他兼容数据。

## 安装与测试

```bash
cd BM25
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

项目核心只使用 Python 标准库。

## 1. 先测 BM25 Top-5 召回率（不调用模型）

```bash
bm25-agent recall --split all
```

历史 Harness eval 子集（1,459 题）的实测 Top-5 recall 为
`1035 / 1459 = 70.9390%`，各目标排名分布保存在
[`results/bm25_top5_eval.json`](results/bm25_top5_eval.json)。这也是后续 Agent 严格成功率的
检索上限；剩余 424 题的目标商品不在候选集中，Agent 被设计为不能越界选择。

## 2. 运行单题完整评测

先在 `BM25/.env` 中配置：

```dotenv
DEEPSEEK_API_KEY=your_key
```

默认使用 DeepSeek 的 OpenAI 兼容直连地址
`https://api.deepseek.com/v1/chat/completions`。HTTPS 传输使用系统 `curl` 与系统证书库；
API Key 通过权限为 `0600` 的短期临时 header 文件传递，不出现在进程参数中。
请求显式关闭 V4 默认 Thinking Mode，以便强制模型只调用 `select_product`，而不是返回
无法确定性解析的自由文本。

获取一个可复现任务 ID：

```bash
python -c 'from bm25_agent.tasks import TaskStore; from bm25_agent.cli import DEFAULT_PRODUCTS; print(TaskStore(DEFAULT_PRODUCTS).sample_task_ids("test", 1, 42)[0])'
```

执行 BM25 Top-5、Agent 选择和最终评分：

```bash
bm25-agent run --task-id task_xxxxxxxxxxxxxxxxxxxxxxxx
```

标准输出包含精简候选列表、Agent 的最终选择以及完整 `evaluation`。

## 3. 批量评测并落盘

```bash
bm25-agent evaluate \
  --split test \
  --workers 4
```

Harness 与 BM25 Agent 统一使用 `all`（全量）和 `test`（通路测试），忽略原始
train/eval 标签。Test 使用固定整数 Seed 42 抽取 10 条任务（不足时取全部），
两边选出的任务 ID 一致，且 Test 仍包含在 all 中；检索始终使用完整商品库。
省略 `--count` 运行所选集合全部任务；指定时可用 `--seed` 从该集合抽样。
执行全量 Agent 检测：

```bash
bm25-agent evaluate --split all --workers 32
```

默认结果目录由代码定位仓库根目录后生成绝对路径：
`<Shopping-Simulator绝对路径>/outputs/bm25/evaluations/`，不依赖当前工作目录，
不使用软链接。通常省略 `--output-root` 即可。
显式指定 `--output-root`（旧名 `--output-dir`）和 `--resume-dir` 时，必须提供
绝对路径或以 `~` 开头的路径；相对路径会直接报错，避免保存到含糊的位置。
`--output-root` 是运行目录的父目录，`--resume-dir` 是已有运行目录本身。
CLI 在任务开始前向 stderr 打印绝对目录，结束时在 stdout 的 `output_dir` 返回相同位置。
历史目录可直接使用迁移后的绝对路径恢复，无须重命名。

每次运行会创建不可覆盖的 `all_<UTC微秒时间>_<UUID>` 或 `test_<UTC微秒时间>_<UUID>` 目录：

- `run_config.json`：恢复所需的任务集合、模型和数据路径；
- `summary.json`：Top-5 召回、严格成功率、平均有效 reward、失败原因和 Token 用量；
- `runs.jsonl`：每题的完整 Top-5 商品、Agent 选择和确定性评分，便于审计。

全量运行会在每题结束后立即追加并刷新 `runs.jsonl`，每 100 题输出一次进度。若进程中断，
使用首次输出目录继续运行；配置和任务集合必须与原运行完全一致：

```bash
bm25-agent evaluate \
  --split all \
  --workers 32 \
  --resume-dir /absolute/path/to/Shopping-Simulator/outputs/bm25/evaluations/all_<首次运行ID>
```

如果从 `BM25` 之外执行，或 `.env` 不在当前目录，可显式传入
`--env-file /absolute/path/to/.env`。

## 历史 Eval 子集结果

使用修复权重后的 Harness v3 BM25、`deepseek-v4-flash`、`temperature=0` 和固定 Top 5，
对旧划分的 1,459 道 eval 任务的实测结果为（不代表当前 all 全量结果）：

- BM25 Top-5 Recall：`1035 / 1459 = 70.9390%`；
- Agent 在目标已召回时选对 ASIN：`966 / 1035 = 93.3333%`；
- 严格成功：`888 / 1459 = 60.8636%`；
- 目标已召回条件下严格成功：`888 / 1035 = 85.7971%`；
- 平均有效 Reward：`0.625431`，Reward 覆盖率：`1449 / 1459 = 99.3146%`；
- 终局分布：严格成功 888、规格部分匹配 73、未命中 488、Gold 不可验证 6、Agent
  非法规格提交 4；
- 总 Token：4,427,794（输入 4,096,319，输出 331,475）。

可审计汇总见 [`results/bm25_agent_eval.json`](results/bm25_agent_eval.json)，完整逐题结果见
本地 `../outputs/bm25/eval_20260911T112102Z/runs.jsonl`（不提交 Git）。

`results/` 仅保留文中引用的两份小型历史基准汇总，属于版本化参考资料；新评测不会写入这里。
运行结果、缓存和环境配置均由根目录 `.gitignore` 忽略；BM25 源码以实体目录直接纳入 Git。
