# LangSmith 追踪与 LangGraph 编排

`AgentRunner`、购物工具、证据与提交校验、Session、Evaluator 和审计文件接口保持不变。
编排器现在有两个后端：`native` 保留原有循环，`langgraph` 用显式角色节点调度同一个
单回合执行方法。默认仍为 `native`，便于逐步迁移和对照。

## 安装和启用

在 `harness/` 目录执行：

```bash
python -m pip install -e '.[graph]'
```

只接入追踪、保留原编排时，可安装 `.[tracing]`。基础安装不依赖这两个 SDK。
本次验证使用 Python 3.12、LangGraph 1.2.15、LangSmith 0.14.8。

Python 入口：

```python
runner = AgentRunner(
    provider=provider,
    harness=harness,
    config=RuntimeConfig(agent_mode="multi", orchestration_backend="langgraph"),
)
result = runner.run(task_id)
```

现有 `run`、`rollout`、`chat`、`resume` 命令支持
`--orchestration-backend native|langgraph`。例如，在已有运行命令中增加：

```bash
shopping-agent run --task-id YOUR_TASK_ID \
  --agent-mode multi --orchestration-backend langgraph \
  --artifacts-dir outputs/graph-run
```

也可以在 RuntimeConfig JSON 中设置：

```json
{"agent_mode": "multi", "orchestration_backend": "langgraph"}
```

Episode 和实验入口继续通过各自的 RuntimeConfig 配置后端。
`resume` 默认读取 checkpoint 中的后端；不允许在恢复途中切换后端。
旧 checkpoint 没有该字段时按 `native` 解析。

## LangSmith 中看什么

追踪默认关闭。配置自己的凭据后启用：

```bash
export LANGSMITH_API_KEY='YOUR_LANGSMITH_API_KEY'
export LANGSMITH_PROJECT='shopping-simulator'
export LANGSMITH_TRACING=true
```

运行相同的 Python/CLI 入口即可。在 LangSmith 项目中查看：

- `shopping.run`：公开任务、会话 ID、后端、最终状态与计数。
- `shopping.planner` / `shopping.executor` / `shopping.observer`：每回合角色与预算。
- `shopping.model`：模型输入、工具定义、输出、Token 用量、耗时和调用错误。
- `shopping.tool.<name>`：工具参数、公开观察、拒绝原因与预算反馈。
- 使用图后端时，额外显示 `shopping-agent` 图及实际经过的角色节点。

恢复会创建新的根 trace，通过 `thread_id=session_id` 关联同一会话。
追踪启用后会向配置的 LangSmith 服务发送公开任务、提示词和工具数据。
不会序列化 Provider 实例/凭据、完整 SessionSnapshot、私有标签或 evaluation。
评分仍保存在本地 `evaluation.json`。遥测写入失败不会重试购物动作，也不会改变返回结果。
原 `event_observer` 和哈希事件链继续独立工作。

核心 AgentRunner 后端提供运行追踪。独立的本机 Agent Server/Studio 交互入口已经加入，
启动方式与 checkpoint 边界见 [本机 Studio 部署](studio-local.md)。
官方说明：[Tracing](https://docs.langchain.com/langsmith/annotate-code)、
[Studio](https://docs.langchain.com/langsmith/studio)。

## 状态和恢复边界

`graph.py` 创建 Planner、Executor、Observer 三个节点；单 Agent 模式只有 single 节点。
每个节点最多执行一个模型回合，再由条件边根据经过校验的角色交接结果选择下一个节点。
同一回合中的工具批次仍按原规则处理，拒绝、动作数和环境步数不变。

`ShoppingGraphState` 是公开观察视图，包含 task、plan、memory、budget、review、
pending_question 和不含评分的 outcome。它不持有第二份可独立修改的业务状态。
`_Run` 与 Session 继续拥有唯一的运行状态；`MultiAgentCoordinator` 保留角色权限与交接规则。
这次迁移聚焦调度和可观测性，没有把所有业务模块重写成图节点。

图未启用节点重试、缓存或独立 checkpointer，避免重放已经执行的搜索/提交。
`ask_user` 仍返回 `waiting_for_user`，审计 bundle 中的 checkpoint 保存 Session 和运行状态，
`AgentRunner.resume` 校验并恢复它，再启动图调度。它不提供任意崩溃点的自动恢复。
图步数不计入购物预算；实际墙钟耗时会受图和追踪开销影响，不能要求时间预算路径逐字一致。

## 同批轨迹对照

从 `harness/` 执行，路径示例中的输入 bundle 必须存在，输出目录必须是新目录：

```bash
python -m tools.compare_backends \
  --run-dir ../outputs/your-run/tasks/000001 \
  --run-dir ../outputs/your-run/tasks/000002 \
  --products-path data/fine_items_eval_train_all.json \
  --index-path data/product_search.sqlite3 \
  --output-dir ../outputs/backend-parity/new-comparison
```

工具先校验已有事件链和商品/索引 manifest，然后给两个后端提供完全相同的录制模型输出，
实际执行购物环境和 Evaluator。每条任务保留两套可回放审计 bundle，并生成 `report.json`。
它比较动作、观察、交接、预算、计数、Token、Session、评分、模型上下文、计划和记忆；
运行耗时、后端配置和因此变化的事件哈希不作为等价条件。不同 task/session ID 会被拒绝。
任何差异使命令返回非零状态。报告单独标记录制耗尽和历史提示词版本差异。
这验证调度语义，不衡量实时模型准确率；时间预算和交互恢复用专项测试覆盖。

2026-10-14 验证了现有 `Test2_100_20261014` 的前 20 条单 Agent 轨迹，20/20 一致，
没有录制耗尽；19 条提交、1 条达到动作上限，两种后端的评分完全一致。
另用改动前保存的原编排代码对这 20 条轨迹与图后端比较，20/20 一致。
多角色专项测试覆盖成功/失败评分、重规划、角色权限拒绝、批次拒绝、预算、模型异常，
以及澄清后通过新 Runner 恢复、避免重复提交和旧 checkpoint 兼容。

```bash
python -m pytest src/harness/test/test_graph_backend.py \
  src/harness/test/test_tracing.py src/harness/branch/test/test_graph_resume.py -q
```

测试追踪使用 SDK 本地上下文和拦截的传输，不需要 LangSmith 凭据或真实模型。
