# 本机 LangSmith Studio

Studio 网页连接本机 Agent Server。服务只监听 `127.0.0.1:2024`，没有公网隧道。
这是本地开发部署，随本机进程运行；不是 LangSmith Cloud 或开机自启服务。

## 启动

在 `harness/` 目录使用 Python 3.11+：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[studio]'
.venv/bin/python -m tools.studio_server --background
```

已经创建 `.venv` 并安装依赖时，只需执行最后一行。省略 `--background` 可前台运行，
按 Ctrl-C 停止。启动器拒绝占用已有端口，支持 `--port` 和 `--env-file`。
后台日志和 PID 保存在仓库 `outputs/studio/server-2024.log`、`server-2024.json`。
后台模式仅报告进程已启动；可通过 `http://127.0.0.1:2024/ok` 确认服务就绪。
停止后台服务时，核对记录的 PID 仍为本服务后执行 `kill PID`；使用正常终止让本地状态落盘。

打开 [Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)，
选择 `shopping` 图。若网页显示登录或 Forbidden，请在已登录 LangSmith 的 Chrome/Edge
浏览器中打开上述链接；云端网页的登录状态与本机 API 是否运行是两回事。
[API 文档](http://127.0.0.1:2024/docs) 可直接在本机查看。

## 离线演示

默认 `SHOPPING_STUDIO_PROVIDER=demo`，使用两件合成耳机商品，不调用外部模型。
在 Graph 模式输入：

```json
{"query": "寻找白色耳机"}
```

能看到 initialize → planner → executor → observer 的循环，最后在 `recommendations`
中得到白色耳机。输入 `{"query":"寻找耳机"}` 会进入 `await_user`；在 Studio 的恢复操作中
传入 JSON 字符串 `"黑色"`，或 `{"text":"黑色"}`，继续完成推荐。
输入其他商品类别需要配置真实模型。演示 Token 数是固定的合成数据。

## 真实模型和商品目录

启动器读取 `harness/.env`（已被 Git 忽略），进程环境优先。也可使用 `--env-file` 指定文件。
`studio.env` 只设置演示和关闭追踪的默认值，不覆盖显式导出的这两项配置。

本地 Ollama 示例，需要先运行 Ollama 并准备相应模型：

```dotenv
SHOPPING_STUDIO_PROVIDER=ollama
SHOPPING_STUDIO_MODEL=your-installed-model
SHOPPING_STUDIO_BASE_URL=http://127.0.0.1:11434/v1
```

DeepSeek 示例：

```dotenv
SHOPPING_STUDIO_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key
SHOPPING_STUDIO_MODEL=deepseek-chat
```

真实模型模式默认读取已有 `harness/data/product_search.sqlite3` 的公开商品目录，
不加载私有评测任务。可用 `SHOPPING_STUDIO_INDEX` 指向另一个索引，或设置
`SHOPPING_STUDIO_SOURCE_JSON` 指向符合 ProductDetail 协议的公开 JSON 商品数组。
修改配置后重启服务并新建线程；旧线程拒绝在模型、来源或运行配置变化后恢复。

如需向 LangSmith 上传追踪，在私有 `.env` 中额外设置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=shopping-simulator
LANGSMITH_API_KEY=your-key
```

仅使用 Studio 本地图调试不需要启用追踪；默认不会上传提示词/商品数据到追踪服务。

## 运行状态与限制

入口在 `src/harness/branch/studio.py`，依赖方向保持“交互支线 → 主线”。
它复用 `AgentRunner` 配置和 `_Run` 的单回合业务逻辑。每个节点把公开 Session、计划、记忆、
完整事件和精确计数序列化进 graph state；下一节点恢复独立 Session，没有全局共享运行对象。
默认 Agent Server 保存线程/checkpoint 到本地 `.langgraph_api/`，该目录不提交 Git。

用户暂停由独立 `await_user` 节点调用 `interrupt`；回复前不执行模型或工具，恢复时不会
重新执行提问动作。支持节点边界恢复；崩溃发生在模型调用后、节点保存前时，重试仍可能再次
调用模型/读取商品，不承诺模型调用 exactly-once。当前 submit 仅记录推荐，不执行购物订单。

Studio 输入 schema 只接受 query。`runtime` 是恢复数据，`plan/memory/budget` 是展示视图，
请通过用户回复修改需求；任意手工改动内部恢复字段不属于受支持的恢复流程。
当前 Studio 只提供无私有标签的交互任务；离线基准评测仍通过原 CLI 执行。

验证覆盖角色图、与 AgentRunner 的动作/预算一致性、跨 graph 实例暂停恢复、并发会话隔离，
以及真实本机 HTTP API 的推荐和澄清恢复。参考
[官方本机 Studio 说明](https://docs.langchain.com/langsmith/quick-start-studio)。
