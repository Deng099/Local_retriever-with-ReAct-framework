# Linux 服务器部署

## 完整 BCP 数据准备（服务器直接下载）

以下命令在已激活的 `react` 环境、项目根目录运行。镜像与缓存只设置在当前终端：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data00/xingyi_deng/huggingface
hf download Tevatron/browsecomp-plus-corpus --repo-type dataset --local-dir /data00/xingyi_deng/data/bcp-full/raw
python -m local_retrieval.prepare_corpus --input-dir /data00/xingyi_deng/data/bcp-full/raw --output /data00/xingyi_deng/data/bcp-full/corpus.jsonl
python -m local_retrieval.chunk_corpus --input /data00/xingyi_deng/data/bcp-full/corpus.jsonl --output /data00/xingyi_deng/data/bcp-full/chunks.jsonl --tokenizer /data00/xingyi_deng/models/Qwen3-Embedding-0.6B
```

转换会检查重复/缺失 ID、读取全部分片并报告空文本；输出旁的 JSON 记录数量。切块默认
256 tokens、重叠 32 tokens，输出旁记录实际配置。已有最终输出会拒绝覆盖；中断留下的
`.tmp` 不是可用数据，重新运行会从头构建临时文件。

切块结束后先查看 `chunks.jsonl.json` 的 chunk 数。1024 维 float32 向量至少需要
`chunk_count × 4096` 字节 CPU RAM，另需留出 FAISS 构建及后续服务加载文本的内存。
确认实验室的 GPU 使用规则、获得可用 GPU 后再执行下面的编码命令；`cuda:0` 只是示例，
不是 GPU 分配授权：

```bash
python -m local_retrieval.build_index --chunks /data00/xingyi_deng/data/bcp-full/chunks.jsonl --index /data00/xingyi_deng/data/bcp-full/indexes/chunks_qwen3_0.6b.faiss --model-path /data00/xingyi_deng/models/Qwen3-Embedding-0.6B --device cuda:0 --batch-size 32
```

全程不调用 generation API。文本逐批读取；精确 FAISS 向量仍驻留 CPU RAM。当前没有编码
断点续建，进程中断后需重跑；最终索引只在写完后发布，不覆盖 5k 索引。后续检索服务
通过 `--corpus`、`--chunks`、`--index` 指向这些全量文件。

retrieval service 在服务器本地加载 Qwen3 embedding 和 FAISS；generation 与 extraction 通过 API 调用。无需安装或启动 vLLM。

## 1. 安装

推荐使用以下目录：

- 代码：`/data00/xingyi_deng/projects/Local_retriever-with-ReAct-framework`
- Python 环境：`/data00/xingyi_deng/envs/react`
- 数据：`/data00/xingyi_deng/data/react-rag`
- 模型：`/data00/xingyi_deng/models/Qwen3-Embedding-0.6B`
- 日志：`/data00/xingyi_deng/logs/react-rag`

先确认环境中的 Python；如果现有环境名称不同，只替换下面的 `REACT_PYTHON`：

```bash
export REACT_ROOT=/data00/xingyi_deng
export REACT_PROJECT=$REACT_ROOT/projects/Local_retriever-with-ReAct-framework
export REACT_PYTHON=$REACT_ROOT/envs/react/bin/python
cd "$REACT_PROJECT"
test -x "$REACT_PYTHON"
"$REACT_PYTHON" --version
"$REACT_PYTHON" -m pip install -e '.[server]'
"$REACT_PYTHON" -m unittest discover -v
```

如果需要重建 chunks/index，再安装：

```bash
"$REACT_PYTHON" -m pip install -e '.[indexing]'
```

如果需要重新下载/生成 BrowseComp-Plus query 子集，再安装：

```bash
"$REACT_PYTHON" -m pip install -e '.[benchmark-prep]'
```

## 2. 下载本地 embedding

把模型和 Hugging Face 缓存放在自己的磁盘目录，不占共享系统盘：

```bash
mkdir -p "$REACT_ROOT/models" "$REACT_ROOT/huggingface"
export HF_HOME=$REACT_ROOT/huggingface
"$REACT_ROOT/envs/react/bin/hf" download Qwen/Qwen3-Embedding-0.6B \
  --local-dir "$REACT_ROOT/models/Qwen3-Embedding-0.6B"
```

已有索引使用这个模型、1024 维、last-token pooling、L2 归一化和固定 query instruction；不要只替换模型目录而继续复用旧索引。

## 3. 配置 API 和本机服务令牌

复制示例配置并填写实际 endpoint：

```bash
cp .env.example .env
```

Visit extraction 默认复用 `OPENAI_MODEL`、`OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。如果暂时不需要 extraction，设置 `VISIT_EXTRACTION_ENABLED=0`；如果使用独立模型，再设置 `VISIT_EXTRACTION_*`。

远端 generation API 通常保留 `OPENAI_TRUST_ENV=true`；只有确认系统代理会错误截获请求时再设为 `false`。

`EMBEDDING_MODEL_PATH` 指向上一步下载的本地模型，`OPENAI_*` 只负责 generation/extraction。将 `RETRIEVER_API_KEY` 改成一段随机且仅自己知道的字符串；retrieval service 与 runner 都从同一 `.env` 读取它。

共享服务器上执行：

```bash
chmod 600 .env
```

如果目录权限尚未限制，确认 `/data00/xingyi_deng` 不允许其他用户读取 `.env`。不要把 `.env`、API key 或 retrieval token 上传到 Git。

## 4. 启动 retrieval service

确认以下四个文件已上传且彼此匹配：

- `local_retrieval/data/corpus_5k.jsonl`
- `local_retrieval/data/chunks_5k.jsonl`
- `local_retrieval/indexes/chunks_5k_qwen3_0.6b.faiss`
- `local_retrieval/indexes/chunks_5k_qwen3_0.6b.faiss.json`

然后启动：

```bash
react-rag-retrieval \
  --host 127.0.0.1 \
  --port 8002
```

也可以不用 entry point：

```bash
"$REACT_PYTHON" -m local_retrieval.retrieval_service \
  --index "$REACT_ROOT/data/react-rag/indexes/chunks_5k_qwen3_0.6b.faiss" \
  --chunks "$REACT_ROOT/data/react-rag/chunks_5k.jsonl" \
  --corpus "$REACT_ROOT/data/react-rag/corpus_5k.jsonl" \
  --host 127.0.0.1 \
  --port 8002
```

只监听 loopback 可以阻止服务器外部访问，但不能隔离同机用户，因此仍应设置 `RETRIEVER_API_KEY`。若 8002 已被占用，选择一个未占用端口，并同步修改 `RETRIEVER_BASE_URL`。

首次启动会把 embedding 权重加载到所选 GPU，并顺序计算数据文件哈希，因此会比普通重启慢。

## 5. 健康检查与运行

```bash
set -a; source .env; set +a
curl --fail -H "Authorization: Bearer $RETRIEVER_API_KEY" http://127.0.0.1:8002/health
"$REACT_PYTHON" -m my_ReAct.run \
  --input-tsv "$REACT_ROOT/data/react-rag/bcp/queries_5k_all_evidence.tsv" \
  --output-dir "$REACT_ROOT/logs/react-rag/server-smoke" \
  --limit 3 \
  --max-step 6
```

正式批跑前检查 `/health` 中的 embedding 模型名、vector count、dimension 和三个文件哈希。`runs/`、`.env`、cache、原始 parquet 和旧索引都不应包含在代码提交中。
