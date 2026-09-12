# 第二阶段：发现、比较与规格核验

包版本 0.4.0，Agent Runtime v4。新增能力在第一阶段需求解析、证据归档和提交前检查之上工作。
仍使用单模型、单轮一个工具调用；比较、核验和动作候选不发起额外模型请求。Evaluator v1 保持原评分。

## 搜索过滤与匹配片段

```json
{
  "query": "白色静音冰箱",
  "filters": {"category": "冰箱", "brand": "品牌名称", "max_price": 800, "include_unknown_price": true},
  "limit": 10,
  "offset": 0,
  "exclude_asins": []
}
```

category 是 NFKC/大小写归一化后的子串匹配，brand 是归一化精确匹配。
价格条件是商品价格区间与请求区间重叠：price_max >= min_price，price_min <= max_price。
缺少被比较价格字段时，include_unknown_price 默认保留商品；显式 false 才排除。
这只是召回条件，不能证明某个规格符合预算。筛选和排除在 SQL LIMIT 之前执行，值使用绑定参数。
不需要重建已有索引，原始无过滤查询的 BM25 顺序不变。

每个摘要额外返回最多三条 match_snippets，每条最多 180 字符，包含公开字段 JSON Pointer、
原文片段和 truncated 标记。片段用于发现，证据判断仍必须引用已读取的详情。
只有本页实际返回的 ASIN 会进入 surfaced 集合，分页探测用的额外一条不会泄露到 Session。

搜索仍返回摘要数组；SessionStep.search_page 和 AgentObservation.projection 提供 next_offset、
has_more。翻页时保持 query、filters、exclude_asins 一致，只更新 offset。
total_results 仍为 null。重复动作签名包含过滤、分页和排序后的排除列表，忽略 limit。
旧自定义 Catalog adapter 可以继续执行基础搜索；不支持扩展能力时明确拒绝，而非静默忽略参数。

## 按字段读取与证据完整性

```json
{"asin": "已搜索出的ASIN", "fields": ["attributes", "customization_options", "variant_combinations"]}
```

省略 fields 保持完整详情行为；指定时只返回选中字段和必需身份字段 asin/title/category。
不支持私有字段。归档 detail_fields 显式保存每次实际读取的字段，防止 Pydantic 默认值在
序列化或回放后把“未读取”伪装成空列表或空规格。

工作记忆合并同商品历次观察过的字段，并保留 field_sources；不会补入未观察的数据。
read_evidence 默认读取最近一次详情，读取较早字段时应使用 facts.field_sources 中的原观察 ID。
提交前若未观察规格字段，返回 selection_fields_unobserved 并提示补读。

## compare_products

```json
{"asins": ["A", "B"], "fields": ["attributes", "price_min", "customization_options"], "constraint_ids": ["color", "budget"]}
```

只允许本 Session 已检查的 2–5 个不同商品。最多 8 个公开字段、10 个约束。
未指定约束时按当前任务顺序展示前 10 个，并列出 omitted_constraint_ids。
响应包含字段差异、已有逐约束判断、未知字段及原始引用。未知字段不会被当成“相等”。
字段内容超过 200 个序列化字符时返回带标记的预览，可以 read_evidence 恢复。
超过 24,000 字符时明确拒绝并提示缩小比较范围。
比较不会更新约束判断，也不生成权威排名；语义结论仍是模型已有判断。

## verify_selection

```json
{"asin": "A", "selected_options": {"颜色": "黑色", "容量": "大"}, "max_price": 300, "budget_constraint_id": "budget"}
```

只读已观察商品，检查选项可用性、规格组合、未选择的轴以及所选价格。
完整选中且匹配可用组合、组合价格一致且已知时，使用组合价格；没有组合表且只有一个规格轴时，
可以使用该选项明确标注的价格。没有规格轴且商品上下限相等时，可以使用该单一价格。
多个规格轴无组合价格、选择不完整、缺少价格或相互冲突时，selected_price 为 null。
不会用商品最低价或相加独立轴价格来填空。没有组合表的多轴商品，组合可用性保持 unknown。

响应包括 selection_status、combination_status、issues、selected_price、price_basis、price_evidence
以及 budget_status。max_price 是模型从公开需求理解出的阈值，单位与目录价格一致；当前不解析币种、
运费、件数或折扣。关联 constraint_id 只验证该约束存在，不证明阈值准确概括用户原文。
数值比较是确定性的，语义判断仍需 record_evidence；工具不自动写入 satisfied。

检查保存在任务记忆；新详情使旧检查标记 stale。提交前对关联硬约束且选择一致的检查，使用最新
已观察字段重新计算。unknown/violated 必须修复，或通过 best_effort 如实披露。
重新检查同一商品/选择/约束时替换工作视图中的旧检查；所有原始事件仍保留。

## 轻量动作候选

ContextBuilder 每轮注入 action_proposals，默认最多 5 条；RuntimeConfig.action_proposal_limit
可以配置为 0–8，0 用于关闭和消融实验。候选根据公开任务、候选矩阵、缺失字段、历史搜索和
规格核验生成，包含工具参数、信息缺口、目标商品、证据来源和环境步数。

优先补读未观察字段、核验规格、比较可行候选、读取缺失证据；证据和规格就绪时建议提交。
未用的计划查询和可继续的分页提供探索路径。已尝试的规范化动作从完整参数候选中排除；
模型仍可在重复预算内主动重试。
需要模型填写理由、约束或所选规格的候选用 requires_model_input 标记为模板，不能原样执行。
候选不限制模型选择其他合法动作，也不改变工具可用性或强制增加模型调用。
上下文超预算时可删除建议，原始任务与必要状态保留。实际建议随 context_built 事件记录。

## 验证与演示

```bash
python -m unittest discover -s src/harness/test -p 'test_*.py' -v
python -m examples.comparison_demo --output-dir outputs/comparison-demo
python -m src.harness.cli replay --run-dir outputs/comparison-demo
```

示例使用合成目录与脚本化 Provider，执行九个动作、四次环境调用，覆盖按字段读取、比较、
所选规格价格、证据恢复、结构化推荐及审计回放。它验证工程行为，不证明真实模型推荐质量提高。
输出目录必须是新目录；不会覆盖已有运行。
