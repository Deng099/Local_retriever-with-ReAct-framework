# Local ReAct + RAG

一个面向学习与实验的小型 ReAct 框架。Agent 固定绑定 `search`、`visit`、`python`
三个工具；常驻检索服务直接持有本地 Qwen3 embedding 与 FAISS 索引，generation 和
extraction 通过 OpenAI-compatible API 调用。

## 结构

- `my_ReAct/`：Agent 主循环、LLM adapter、工具契约和运行时装配。
- `local_retrieval/`：本地 embedding、FAISS retriever、Visit 文档存储及 HTTP 服务。
- `DEPLOYMENT.md`：Linux 服务器安装、服务启动和验收命令。

工具观察统一返回：

```json
{"ok": true, "data": {}, "error": null, "metadata": {}}
```

一次 Agent 运行返回 `status`、`output` 和 `termination_reason`。对话 trajectory 是唯一
消息状态；供应商 continuation state 只在当次运行中显式传递。

## 安装与测试

在 Linux/WSL 的项目根目录执行：

```bash
python -m pip install -e '.[server]'
python -m unittest discover -v
```

完整的 API 配置和服务器启动方式见 `DEPLOYMENT.md`。安装后可以使用：

```bash
react-rag --help
react-rag-retrieval --help
```

## 设计边界

### 批跑配置与续跑

`python -m my_ReAct.run --input-tsv PATH --output-dir runs/experiment1`
默认使用 120000 字符的预算型 context view；完整 trajectory 仍然写盘。可用
`--context-chars N` 调整预算，或用旧的 `--context-turns 4` 明确切换为最近四轮窗口。
预算是字符数而非精确 token 数，需给模型输出和 provider tokenization 留余量。
逐题结果包含 `run_state`、context 配置、检索身份和 `experiment_id`，并原子写盘。
新运行还会在 trajectory 的 assistant/tool 消息中记录 `latency_seconds`，并在顶层
`timing` 汇总 LLM 与各工具耗时，便于区分模型 API、检索和 visit 的性能瓶颈。
目录内的 `experiment.json` 固定模型/API、prompt、输入数据、工具预算和 context
配置；相同配置可续跑，改变配置请使用新目录。旧版没有身份记录的结果目录也需
换目录，不会自动覆盖或认作当前实验。API key 不写入配置。

查看单个或一批 run 时可使用轻量检查命令；answers 和 relevance 都是可选项：

```bash
python -m my_ReAct.inspect_run runs/experiment1 \
  --answers data/answers.json \
  --relevance data/relevance.json \
  --query-id 770 \
  --trajectory
```

报告包含状态、停止原因、工具次数、token、LLM/tool 耗时、最终回答，以及在标签
可用时的 evidence/gold 文档 recall 和 ground truth。trajectory 默认截断旧观察，最终
回答完整保留；用 `--max-content-chars` 调整预览长度。

更新后先重启 retrieval 服务，它会在启动时计算 index/chunks/corpus 的 SHA-256，
通过 `/health` 返回给 runner。启动会多一次文件顺序读取；健康检查不重复计算。

检索服务使用 `--embedding-model PATH_OR_ID` 加载 Qwen3 embedding，默认 Hugging Face
ID 为 `Qwen/Qwen3-Embedding-0.6B`；服务器推荐用 `EMBEDDING_MODEL_PATH` 指向自己的
模型目录。`--embedding-device`/`EMBEDDING_DEVICE` 可指定 GPU。索引 manifest 会检查
模型名称、维度和向量数，防止 query embedding 与已有索引静默混用。

共享服务器建议设置 `RETRIEVER_API_KEY`。runner、retrieval eval 和 retrieval service
读取同一个变量，以 Bearer token 保护本机 HTTP 接口，避免同机其他用户调用 extraction
而消耗 generation API 配额。

检索评测入口：`python -m local_retrieval.retrieval_eval --help`；重新 editable
安装后也可使用 `react-rag-retrieval-eval`。其 K 为 chunk 排名位置，nDCG 为本地
诊断定义，不应直接作为官方文档级 nDCG 报分。

### RunState 与 ContextManager

参考 [Gizmo 的接口](https://github.com/namespace-ERI/Gizmo/blob/main/Gizmo/agents/base_agent.py)，
`agent.state` 记录本题已完成的模型调用 `step`（含最后回答轮）、`tool_rounds`、
`elapsed` 和 `stop_reason`，并持有显式 provider continuation。
Agent 的工具在 run 结束时关闭，因此每题创建新 Agent；重复调用同一 Agent 会报错。

可以在 `create_research_agent(..., context_manager=...)` 或 `ReActAgent` 构造时
传入一个处理器。批跑入口默认使用 `BudgetContextManager`；直接构造 Agent 时仍可选择不处理。
示例：

```python
from my_ReAct.context_manager import RecentTurnsContextManager

agent = create_research_agent(
    query, retriever, llm,
    context_manager=RecentTurnsContextManager(keep_recent_turns=4),
)
result = agent.run()
print(agent.state.step, agent.state.elapsed, agent.state.stop_reason)
```

自定义处理器实现 `process(messages, state)`，可选实现 `reset()`。
每次模型请求（包括预算耗尽后的回答轮）处理 trajectory 的深拷贝，原始
`agent.messages` 继续保存完整观察。`BudgetContextManager` 先压缩旧的 search snippet、
visit preview、extraction 和 Python 输出，再按预算移除最旧完整轮次，不拆散 tool call
与 output。压缩只影响发给模型的 view，不影响 `runs/` 审计记录，也不调用额外 LLM。
`RecentTurnsContextManager` 仍保留作为简单的兼容实现。
每次 run 还记录最后一次 view 的字符数、最大 view 字符数和移除轮次数，便于比较
benchmark 是否真正触发了裁剪。

配置处理器后，Responses 使用显式消息重放，不带 `previous_response_id`；
这样服务端旧历史不会绕过窗口。未配置时保持原有 continuation 行为。
显式重放只包含本框架保存的可见消息，不保留供应商隐藏 reasoning state，
因此开启窗口应作为独立实验配置验证。

### Visit 长文与 extraction

retrieval service 默认把每次 visit 的 inline 文本限制为 12000 字符；完整文档保留在服务端。
长文带 goal 且 extraction 已启用时，模型主要收到 extraction、来源和范围元数据，避免全文
重复进入上下文。可以用 `visit(reference, offset, limit)` 读取其他字符范围。extraction
缓存键为文档内容哈希、规范化 goal、模型和 prompt 版本，因此同一文档换 goal 会重新提取，
相同 goal 可复用缓存。

extraction 默认复用 generation 的 `OPENAI_MODEL`、`OPENAI_API_KEY` 和
`OPENAI_BASE_URL`；设置 `VISIT_EXTRACTION_MODEL`、`VISIT_EXTRACTION_BASE_URL` 或
`VISIT_EXTRACTION_API_KEY` 才覆盖对应项。设置 `VISIT_EXTRACTION_ENABLED=0` 可关闭。服务
健康检查会返回是否启用及使用的模型名，但不会返回 key。

当 generation/extraction 是本机或内网 endpoint 时设置 `OPENAI_TRUST_ENV=false`，避免系统
HTTP proxy 截获请求；独立 extraction 可用 `VISIT_EXTRACTION_TRUST_ENV` 覆盖。长文抽取输入、
输出字符上限可分别用 `VISIT_EXTRACTION_MAX_INPUT_CHARS` 和
`VISIT_EXTRACTION_MAX_OUTPUT_CHARS` 配置。

这是可信本地研究环境，不把 Python worker 当作恶意代码安全沙箱。Python 状态在一次
Agent run 内跨 tool call 保留，题目结束即回收；Visit 的网页缓存可跨进程重启复用。
live web 默认拒绝本机、内网和非全局地址。
