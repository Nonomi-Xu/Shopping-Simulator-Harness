# 第三阶段：真实用户交互与公开商品来源（0.5.0）

Runtime v5 增加 `chat / resume / preferences`。原 `run / rollout` 继续执行离线
标注任务；交互任务使用 `InteractiveHarness`，不创建私有答案、不调用 Evaluator，
最终 `evaluation=null`。交互完成表示完成推荐，不代表获得某个基准 Reward。

## 运行流程

```text
公开 query → run_query → 搜索、检查、核验
                       ↓ ask_user
               waiting_for_user + checkpoint.json
                       ↓ 调用方 UserReply
              新需求版本 → update_plan → 推荐或再次澄清
```

`AskUser` 包含 question、information_gap、constraint_ids 和可选 options。
运行时返回唯一 question_id，立即停止模型调用；Session 保持 active。离线模式不注册
ask_user，不生成模拟回答。max_user_questions 默认 3，可配置为 1–20。

`resume(paused_result, UserReply(...))` 接受真实用户回答。默认保留既有硬约束；调用方
可以用 replace_constraint_ids 明确授权替换指定约束。模型没有删除硬约束的独立工具。
回复有稳定 message_id；新约束使用 source=user_message、source_message_id 和原文
source_quote。用户补充后 requirements_status 重新变为 unparsed，必须重新 update_plan。

原始 query 不变，所有用户补充保存在 task_profile.user_messages；revision 单调增加。
替换的约束对应判断被清除，所有旧数值选择检查失效；候选、已观察详情和未变约束判断
保留。模型是否正确理解用户补充仍需评估，引用存在并不证明语义理解正确。

动作、搜索、Token、环境步数与模型轮数跨恢复累计，不能通过多次恢复重置预算。
active_duration_seconds 跨恢复累计实际运行时间，排除等待用户和程序未运行的时间。
新需求版本重置重复动作检测，允许有理由重查价格或搜索；全局资源预算保持累计。

## 无 API Key 演示

在 harness 仓库根目录运行：

```bash
python -m examples.stage_three_demo --output-dir outputs/stage-three-demo
python -m src.harness.cli replay --run-dir outputs/stage-three-demo/waiting
python -m src.harness.cli replay --run-dir outputs/stage-three-demo/completed
```

演示用两个合成耳机和脚本化 Provider，先检查白色，再保存暂停状态，创建新的 Provider
和 Harness 从文件恢复，由调用方将颜色改成黑色，最后推荐黑色。全过程 9 个动作、
4 个环境调用；长期偏好仍是“通常喜欢白色”。它验证运行链路，不测真实模型推荐质量。

## CLI：真实模型 + 本地公开商品

沿用现有 DeepSeek 环境变量与 dotenv 加载方式。示例商品位于 examples/public-products.json。

```bash
shopping-agent chat \
  --query '想买一副耳机，预算300元，颜色还没决定' \
  --source-json examples/public-products.json --currency CNY \
  --artifacts-dir outputs/chat-01
```

模型可能直接完成，也可能返回 waiting_for_user；仅后者可以恢复：

```bash
shopping-agent resume \
  --run-dir outputs/chat-01 --reply '选择黑色，预算不变' \
  --source-json examples/public-products.json --currency CNY \
  --artifacts-dir outputs/chat-02
```

如果用户要改变已有硬约束，先从 checkpoint.json 的 task_profile.constraints 查看
实际 constraint_id，再添加 `--replace-constraint-id 实际ID`；支持重复该参数。该参数
表示调用方明确授权替换，不能由模型自动填入。未替换的硬约束仍受保护。

恢复默认使用 checkpoint 中的 RuntimeConfig、Profile、模型、轮数上限和 Provider 参数。
source-id/currency/source 配置须与启动时一致；若使用了自定义 --base-url 或 --skill-dir，
恢复时也须传入原值。凭证从当前环境读取，不写入 checkpoint。含脱敏 Provider 参数
的 checkpoint 必须重新显式传入原 --provider-parameters。

## Python API

```python
from pathlib import Path
from src.harness.agent import AgentRunner, RuntimeConfig, UserReply
from src.harness.branch.interactive import InteractiveHarness
from src.harness.branch.preferences import PreferenceStore
from src.harness.sources import JsonProductSource

# provider 使用调用方配置的 ModelProvider；可沿用 DeepSeekProvider。
store = PreferenceStore(Path('outputs/user-preferences.sqlite3'))
profile, revision = store.load('user-001')
with InteractiveHarness(JsonProductSource('examples/public-products.json', currency='CNY')) as env:
    runner = AgentRunner(provider=provider, harness=env, user_profile=profile,
                         config=RuntimeConfig(interaction_mode='interactive'))
    result = runner.run_query('我想买一副耳机')
    if result.status == 'waiting_for_user':
        # 实际应用应先展示 result.pending_question 并接收用户输入。
        reply_text = input(result.pending_question.question + '\n')
        result = runner.resume(result, UserReply(text=reply_text))
```

应用可对多次 waiting_for_user 重复接收回答。暂停结果是深复制的数据快照，不依赖
之前 _Run 对象。可通过 AgentRunResult.model_validate_json 加载 checkpoint，配合
同配置的新 Runner 和新的 InteractiveHarness 恢复。

## 长期偏好

`PreferenceStore` 是按 profile_id 隔离的 SQLite 存储。只支持调用方明确设置与删除；
任务选择不会自动学习为偏好，也不保存商品标签或任务记忆。加载得到 `(UserProfile, revision)`。
写入可携带 expected_revision，避免并发调用覆盖旧版本；写入使用 SQLite 事务。
每次修改记录 explicit_caller 来源、时间和修订号。

```bash
shopping-agent preferences set --db outputs/preferences.sqlite3 \
  --profile-id user-001 --name 颜色偏好 --value 通常喜欢白色 --expected-revision 0
shopping-agent preferences show --db outputs/preferences.sqlite3 --profile-id user-001
shopping-agent preferences history --db outputs/preferences.sqlite3 --profile-id user-001
shopping-agent preferences remove --db outputs/preferences.sqlite3 \
  --profile-id user-001 --name 颜色偏好 --expected-revision 1
```

chat 可传 `--preferences-db PATH --profile-id ID`，与 --user-profile 互斥。
恢复固定使用任务开始时的偏好快照；期间对长期偏好的修改作用于下次新任务，避免运行中
隐式改变推荐依据。删除偏好会从当前 Profile 移除，但修改历史仍保留，不是隐私数据清除接口。

## 外部商品来源

`ProductSource` 定义 manifest、search_page、get_product、close。InteractiveHarness
拥有并关闭该来源，每个环境使用一个明确的来源。未实现跨来源合并和 ID 消歧。

### JsonProductSource

接受公开 ProductDetail JSON 数组。私有顶层字段会被公开 schema 丢弃；未知字段不进入
索引、来源哈希或模型上下文。搜索使用确定性的词元重叠排序，随后按 ASIN 排序；保留
兼容字段 bm25_score=0，manifest 明确它不是 BM25。支持类别、品牌、价格范围、分页和
exclude_asins。恢复时要求公开数据快照哈希一致。

### HttpProductSource

这是可运行的通用商品网关客户端，尚未实现某个零售商专属接口。调用方配置 base_url；
服务需要实现以下只读业务协议：

```text
POST /search
  请求: SearchProductsAction JSON
  响应: {"items": [ProductSummary, ...], "page": SearchPage}

GET /products/{URL编码的ASIN}
  响应: ProductDetail JSON
```

search 请求虽然使用 POST，但网关必须作为只读查询处理。items 不得超出 limit，
rank 从 offset+1 开始；下一页 offset 必须向前推进。网关负责过滤和排序，客户端校验
Schema、返回数量、ID、rank、排除项和分页契约。每次响应记录公开内容哈希、来源 ID、
观察时间与调用方明确提供的币种。没有默认猜测币种，也不把快照价格称为当前结算价。

```bash
shopping-agent chat --query '寻找300元以内的耳机' \
  --source-url https://your-product-gateway.example \
  --source-id merchant-catalog --currency CNY \
  --source-key-env PRODUCT_SOURCE_API_KEY \
  --artifacts-dir outputs/external-chat
```

该域名是协议示例，需替换成实际网关；此版本没有调用真实零售商服务的质量验证。
客户端默认 15 秒超时、最多 2 MB 响应，不自动重试，不跟随重定向；错误不回显响应体
或凭证。认证值只在请求 header 中使用。JSON/HTTP 数据必须明确提供已知规格的
is_available；缺少顶层详情字段表示未观察，不能通过 Schema 默认值当成“没有规格”。

再次获取外部商品详情会清空该候选旧判断与数值核验，要求重新核验。历史观察继续保留，
但已被新观察改变的字段不能再用旧引文登记为当前证据。恢复本身不自动刷新远程数据，
模型可通过 get_product 刷新；跨观察合并字段会保留各自来源 ID。

## 审计与限制

等待产物额外包含 checkpoint.json。run_paused 保存状态摘要，事件支持
run_paused → run_resumed → user_message → plan_updated，最终仍通过 run_finished 结束。
恢复前校验事件链、配置、来源、Profile、任务、记忆、对话、预算和 checkpoint 摘要。
Manifest 最后落盘；无完整 Manifest 的目录不能作为恢复点。

同一个 InteractiveHarness 实例拒绝重复消费同一 checkpoint。独立进程仍可能各自加载
同一个文件，部署服务需要自行加任务锁或幂等控制。本版仅在明确等待用户时保存恢复点，
不提供任意动作中的崩溃恢复、跨机器调度、Web UI 或购买/支付操作。来源哈希与事件链
用于一致性检查，不是抵抗攻击者重写整个文件包的签名。已结束的推荐任务不能继续恢复，
应启动新 query。

交互模式沿用已检查商品的 verified / best_effort 提交协议；尚无独立的“无可推荐商品”
终局工具。需求语义理解、等价替代商品质量和真实用户满意度仍需独立评估。
