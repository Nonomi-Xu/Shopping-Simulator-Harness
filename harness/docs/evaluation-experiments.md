# Agent 多维配对评测

实验层在现有 AgentRunner / EpisodeRunner 上运行“固定任务 × 重复运行 × 配置组”矩阵，
保持原有商品检索、用户模拟、证据检查和 Evaluator 的评分含义。
支持更换模型、Provider 参数、Skill、单/多角色模式及上下文配置。
CLI 使用现有 DeepSeekProvider；Python API 可注入任意符合 ModelProvider 协议的工厂。

无需凭据的完整示例（3 个任务 × 3 次重复 × 2 个上下文配置）：

```bash
python -m examples.evaluation_demo --output-dir outputs/evaluation-demo
```

示例使用合成商品及脚本化 Provider，包含真实 BM25、Episode、记录写盘和报告重建。
示例 Token 是固定的合成值，结果用于验证工程链路，不代表真实模型质量或推理成本。

## 执行

从 harness 目录执行。以下 run 命令会调用模型，任务数 × repeat_seeds 数量 × variants 数量
就是计划运行次数。提供的示例默认最多 20 个任务 × 3 次重复 × 2 组 = 120 次运行。

```bash
python -m src.harness.cli experiments run \
  --config examples/evaluation-matrix.json \
  --split test --count 20 --output-dir outputs/evaluation-01
```

凭据仅从各组的 api_key_env 指定的环境变量读取，默认 DEEPSEEK_API_KEY。
不要在 JSON 中保存凭据。模型名、base_url、provider_parameters 可按组配置；
skill_dir 相对当前工作目录解析。实验全程串行，任务/重复块内随机化各组顺序，
减少固定执行顺序偏差；不评测并发负载。

要比较其他模型，新增或修改 variants 中的 model，必要时指定 base_url 和 api_key_env。
reference 指向唯一的参考组；每个其他组均与参考组比较。

使用已有 Episode 缓存时，把配置 mode 改为 episode，再运行：

```bash
python -m src.harness.cli experiments run \
  --config my-episode-matrix.json --cache data/episodes.sqlite3 \
  --split test --count 20 --output-dir outputs/episode-evaluation-01
```

每组 baseline 默认 long_horizon，可设 full_query。缓存存在同一任务多个版本时必须
显式提供 --episode-id；实验不会重新调用 GoalParser。显式 --task-id 或 --episode-id
选择的集合不受默认 count 截断。--count 0 选择该 split 的全部可用任务。

budget 控制所有组共享的最大模型轮次、动作数、环境调用数、搜索数、提问数；
这几个值覆盖 variant.runtime 中的对应值。其余 RuntimeConfig 选项按组生效。
Episode 会按 baseline 设置交互模式、progressive_disclosure，并开启结构化硬约束过滤。
manifest 保存每组最终有效配置，避免只比较表面配置。离线模式不支持 ask_user。

## 种子和可复现性

- selection_seed：固定任务采样和各组执行顺序；所有组使用完全相同的任务集合。
- repeat_seeds：固定重复运行配对标识。默认不发送给模型，重复运行代表重新调用服务。
- model_seed_parameter="seed"：仅在确认 Provider 接受 seed 时显式开启，把重复种子传给模型。
- statistics_seed：控制 bootstrap 与较大样本的随机化检验。
- EpisodeConfig.seed 不重新生成已冻结缓存的需求或披露内容。

temperature=0 或发送 seed 都不能保证外部模型逐位复现。相同任务的重复运行不是额外的独立任务。

## 指标

| 维度 | 指标及含义 |
|---|---|
| 结果 | strict_success、Success@1/@5、提交完成率、有效评分覆盖率、有效 reward 均值 |
| 资源 | 模型轮次、动作、环境调用、搜索、输入/输出/总 Token、运行墙钟时间、Agent 活跃时间 |
| 工具使用 | rejected_action_rate = 被拒绝的观察 / 全部观察 |
| 交互与状态 | Episode 原有约束满足率、验证覆盖率、遗忘率、遗漏率、违反率、重复问题率、过早推荐、信息增益及回答次数 |
| 稳定性 | 任务内重复运行标准差、成功结果发生变化的任务比例 |
| 失败诊断 | Provider/运行时失败、预算耗尽、无目标命中、规格不匹配、非法提交、不可评分等分类 |

未完成、Provider 失败、未提交等均留在成功率分母中。无有效评分的 reward 为 null，
单独报告 reward_valid。缺失 usage 或任一轮 usage 不完整时，Token 指标为 null，
不会把部分 Token 总和当成完整成本。Episode 特有指标在离线任务中为 null。
没有推荐时某些约束比例无定义，继续保留 null；不把它们解释成成功或零成本。
Token 是调用量，不是货币费用。每次墙钟时间包含重试及 Episode 控制器工作，
不包含共用数据预检和产物写盘；它与 Agent 的 active_seconds 分别报告。

## 统计口径

每个指标先对同一任务的重复运行取平均，再对任务取平均；任务权重相同。
置信区间采用按任务聚类的 percentile bootstrap，默认 95%、2000 次。
每组同时报告有效运行数/总运行数、有效任务数、P50/P95 和任务内波动。
P50/P95 是所有有效运行的经验分位数，不带 bootstrap 区间。

配对比较先对相同 task_id + repeat_seed 求“候选组 − 参考组”，再在任务内平均。
缺失指标只使用双方均有值的配对，并报告配对覆盖率；覆盖率不同可能产生选择偏差。
任务数小于 2 时区间及 p 值为 null。样本很少或取值全相同时区间可能退化，
不应据此推断普遍优越性。higher 指标正差值有利，lower 指标负差值有利；
answered_questions 只作上下文信息，不预设越多或越少越好。

同时提供双侧任务级配对符号随机化检验：最多 12 个任务时穷举，其余使用有固定种子的
Monte Carlo（次数与 bootstrap_samples 相同，使用 +1 修正）。检验依赖组标签可交换的假设，
不证明因果。所有可用“候选组 × 指标”检验共同进行 Holm 多重比较校正。
报告效果差值和区间，不能只按未校正 p 值挑选胜者。

## 产物和重新分析

```text
output/private/
  manifest.json       数据/索引/Skill 哈希、冻结任务、配置、有效配置、执行顺序
  records.jsonl       每次完成后立即追加并 flush 的紧凑记录
  runs/000001/...     原有单任务或 Episode 审计产物
  summary.json       所有指标、覆盖率、差值、区间和检验
  metrics.csv        每次运行的平面指标，可继续分析
  report.md          可阅读的比较报告
  completed.json     最后写入；记录 manifest 和 journal 哈希
```

实验输出整体属于私有评测数据，不可灌入 Actor 上下文。不覆盖任何旧输出目录。
运行中断会保留已完成记录，但本版不自动恢复中断实验；completed.json 缺失表示未完成。
哈希校验用于检测相对可信摘要的更改，不是数字签名。

无需 API Key、无需再次调用模型即可验证完整 journal 并重建统计报告：

```bash
python -m src.harness.cli experiments report \
  --run-dir outputs/evaluation-01 --output-dir outputs/evaluation-01-report
```

报告重建校验配置/journal 哈希和实际矩阵，不重新执行模型；逐任务事件链可继续使用
已有 replay 命令校验。实验 run 成功退出表示矩阵完成并写盘，不表示被评测 Agent 成功。
