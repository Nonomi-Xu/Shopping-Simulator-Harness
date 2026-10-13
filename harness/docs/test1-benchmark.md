# Original / Test1 与全链路评测

仓库顶层 `datasets/` 保存不可覆盖的 Original、Test1 清单。Original 为 23,421 题；
Test1 为按商品、instruction 原顺序展开后的第 1–500 题。两种方法读取同一清单，
验证原数据 SHA-256、任务 ID 顺序、索引哈希。固定整个商品库检索，不只检索前 500 个商品。

## 执行与恢复

以下命令从 `harness/` 执行。会调用 DeepSeek 并产生费用，凭据从本地 `src/.env` 加载。

```bash
python -m src.harness.benchmark.runner --dataset Test1 --workers 16
```

默认在顶层 `outputs/benchmarks/Test1_<UTC时间>_<UUID>/` 保存绝对路径输出。
可显式提供 `--run-dir /absolute/path`，再次运行相同命令会读取该目录已有任务记录，
跳过已完成任务，不会重复计费。中断中的单个任务没有可验证终局时，需要新建 attempt 重跑，
旧轨迹保留；它是任务级恢复，不是任意模型轮次的无损恢复。
成功、算法失败和 Provider 失败都是已完成观测，恢复不会偷偷重试失败题改变分母。
手动重复实验使用新目录。

`--limit 2` 只用于单独预检目录，不得当作正式 Test1 结果。正式目录配置禁止混入 limit 结果。
`--method bm25` / `--method harness` 可在相同配置目录分别执行；默认 both 顺序执行两种方法。
最多同时执行 workers 个任务，不是 1000 个任务一起并发。
Harness 遇到 HTTP 401/402/403 会停止派发新任务，并写入 `blocked.json`；未完成的批次
不会写 `completed.json`，也不能生成正式配对报告。307/308 仅跟随同域 HTTPS 跳转，保留 POST 请求体。

保留两种系统原配置：BM25 一次固定 Top-5 决策、Thinking disabled；Harness 单 Agent、
Thinking enabled、auto tool choice、最多 40 轮模型/40 动作/6 搜索/20 环境动作，最终最多推荐 5 件。
这些差异写入 `config.json`。`--harness-thinking disabled` 是另一个显式实验配置，不能覆盖原实验。
本轮主要观察现有系统，不在评测中途调整算法。

## 全链路记录

```text
config.json / dataset.json / provenance.json / source_hashes.json
bm25/ 或 harness/
  records.jsonl                     每题完成立即追加、flush、fsync 的紧凑索引
  completed.json                    最后写入，核对任务数量与 journal 哈希
  tasks/000001/
    record.json                     单题紧凑结果，供中断恢复
    attempt_01/
      events.jsonl                  Harness 带哈希链的完整实时事件
      timeline.jsonl                每个事件的 UTC 与经过秒数
      model_io.jsonl                实际模型请求体、响应、重试和错误；不记录 HTTP headers
      session.json                  工具调用参数/响应、完整会话与隔离评分
      result.json                   终局元数据；大字段引用同目录文件
      profile.json / plan.json      Agent/User/Task Profile 和最终计划
      final_memory.json             最终记忆、证据来源与引用
      evaluation.json               私有 Evaluator 结果，不传给模型
      summary.json                  终止原因、动作数和 Token
      run_manifest.json             完整写盘的最后标志
```

模型上下文、完整解析响应、计划更新、动作、观察和记忆更新均在 events 中。
`model_io.jsonl` 额外记录真实 HTTP 请求体和原始响应、每次重试的编号与错误状态。
记录的是 Provider 实际返回的可见内容，不声称能够获取模型未返回的内部状态。
BM25 的 `result.json` 包括候选、选择及评分；其模型 I/O 记录覆盖逻辑请求/最终响应。

每题完成立即落盘并释放完整 AgentRunResult；批量进程只保留轻量索引。
内存主要随并发任务数和单题轨迹大小增长，不随 500 个完整历史线性增长。
单题内为保持现有 Agent/回放 API 兼容，仍保留该题事件；不将整批装入一个超大 JSON。
评测产物含私有标签和模型输入输出，仅供离线分析，全部由根 `.gitignore` 忽略。

普通 `shopping-agent run --artifacts-dir /absolute/path` 也采用实时全链路记录。
Python API 可给 AgentRunner 传 `event_observer=RunAudit(...)`，给 DeepSeekProvider 传
`request_observer=audit.transport`，完成后调用 `audit.finish(result)`。

## 离线报告

```bash
python -m src.harness.benchmark.report --run-dir /absolute/path/to/Test1_run
```

若只重跑 Harness，使用新目录执行 `--method harness`，随后复用已完成的 BM25 基线：

```bash
python -m src.harness.benchmark.report --run-dir /absolute/path/to/Harness_rerun --bm25-run-dir /absolute/path/to/Original_completed_run
```

报告核验两批的冻结任务、数据、模型和基线结果哈希，记录 `comparison_sources.json`。
原 Harness 故障批次不会混入新统计；复用 BM25 不产生额外模型调用。

验证两条方法是否完整覆盖相同任务、配置和结果 journal 的哈希，再生成：

- `analysis/report.md`：新召回、准确率、成本、失败诊断与后续方向。
- `analysis/summary.json`：机器可读指标、配对 bootstrap 区间、McNemar 检验。
- `analysis/tasks.csv`：逐任务×方法对照表。
- `analysis/failures.json`：失败类型和审计路径。

逐题补充可追溯的离线规则标注（不调用模型）：

```bash
python -m src.harness.benchmark.annotations --run-dir /absolute/path/to/Harness_rerun --bm25-run-dir /absolute/path/to/Original_completed_run
```

生成 `analysis/annotated_report.md`、`annotations.jsonl`、`annotation_summary.json`，
包含每题需求、Gold、搜索目标排名、提交规格、方法对照标签和接口模型核对。
标注表示已观察到的失败阶段，并非人工语义审核结论。

主指标 Strict@1 要求首件 ASIN 与所需规格均正确且评分有效；另报 ASIN@1 和 Strict-any。
失败与不可评分任务保留在总任务分母中。召回必须从实际搜索候选确定，即使模型调用失败
也不能把 BM25 的已检索结果记为未召回。Harness 累计召回使用其多次搜索的并集；同时报告
首次搜索 Top-5 和详情观察覆盖。完整 Token 覆盖率独立报告，不把缺失 usage 当作零成本。
不同检索预算和 Thinking 模式意味着这是系统比较；后续需增加同预算消融。
