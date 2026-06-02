# RAG LangChain 实现项目

基于 LangChain 框架的高并发 RAG 实现，包含完整的检索优化流程。

## 项目结构

```
rag_langchain/
├── src/
│   ├── __init__.py           # 模块导出
│   ├── config.py             # 配置管理
│   ├── cache.py              # 三级缓存实现
│   └── pipeline.py           # RAG 管道核心（含检索优化）
├── scripts/
│   ├── build_index.py        # 建库脚本
│   ├── ask.py                # 问答脚本
│   └── test_llm.py           # LLM 测试脚本
├── data/
│   └── sample_docs/          # PDF 文档目录
├── .env                      # 环境配置
├── .env.example              # 配置模板
└── requirements.txt          # 依赖列表
```

## 核心特性

### 🔍 STEP 1: Query 分类器

轻量分类器，基于正则、领域词典、长度和疑问句特征，将查询分为三类：

| 类型 | 特征 | 适用检索方式 | 通道权重 |
|------|------|-------------|---------|
| **keyword** | 短文本、关键词查询 | BM25 | 向量:BM25 = 0.3:0.7 |
| **semantic** | 疑问句、长文本 | 向量检索 | 向量:BM25 = 0.7:0.3 |
| **balanced** | 混合特征 | 双路融合 | 向量:BM25 = 0.5:0.5 |

**领域关键词词典**：
- 保单相关：保单、保险、条款、合同、险种、保费、保额、理赔、赔付
- 健康相关：疾病、医疗、住院、手术、门诊、体检、健康、治疗
- 意外相关：意外、伤残、身故、烧伤、烫伤、骨折、意外事故
- 财务相关：缴费、缴费期、等待期、犹豫期、宽限期、现金价值、红利
- 责任相关：责任、免责、除外、不保、拒绝、不予赔付

---

### 📊 STEP 2: 纯 RRF 合并器

- **核心思想**：只看排名，不看分数
- **RRF 公式**：`score = sum(1 / (k + rank)) * weight`，其中 k=60
- **调参旋钮**：从"分数权重"改为"通道参与度"
- **优点**：尺度差问题直接消失，调参方向清晰

---

### 🏆 STEP 3: Cross-Encoder 精排

RRF 之后对 top 候选再做一次精排：

```
检索结果 → Cross-Encoder 评分 → 重排序 → 取 top N 进入 Prompt
```

**效果**：相关性更高的内容稳定排到前面，Prompt 噪声明显下降。

---

### 🎯 STEP 4: Bad Case 回归集

- 固定一组线上失败样例做回归集
- 按 query 类型分别调通道权重
- 不看平均，只看每一类的尾部是否被救回来
- **效果**：尾部 case 准确率明显回升，整体均值跟着上去

---

### 💾 三级缓存系统

| 层级 | 缓存内容 | TTL | 用途 |
|------|----------|-----|------|
| L1 | 问答结果 | 5分钟 | 快速响应重复查询 |
| L2 | 检索结果 | 30分钟 | 缓存检索上下文 |
| L3 | 向量嵌入 | 24小时 | 缓存文本向量 |

## 使用方式

### 1. 安装依赖

```bash
cd rag_langchain
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key
```

### 3. 构建索引

```bash
python scripts/build_index.py
```

### 4. 问答查询

```bash
# 关键词类型查询（自动路由到 BM25 为主）
python scripts/ask.py "等待期是多少天?"

# 语义类型查询（自动路由到向量检索为主）
python scripts/ask.py "什么是意外保险?"

# 平衡类型查询（双路融合）
python scripts/ask.py "意外医疗的理赔流程是什么?"
```

### 5. 查看分类信息

问答结果会显示分类信息：

```
Query Classification: keyword
Classification Scores: {'keyword': 0.8, 'semantic': 0.0, 'balanced': 0.2}
Channel Weights (vector:bm25): 0.30:0.70

Answer:
等待期是90天。
```

## 配置说明

```python
# 主要配置项
LLM_MODEL = "qwen-plus"              # LLM 模型
DASHSCOPE_API_KEY = "your-api-key"   # API Key
CHUNK_SIZE = 512                     # 文本切分大小
CHUNK_OVERLAP = 50                   # 切分重叠
RETRIEVAL_K = 10                     # 检索数量
RERANK_TOP_N = 5                     # 精排后保留数量
CACHE_ENABLED = True                 # 启用缓存
```

## 注意事项

1. **环境要求**：
   - Python 3.10+
   - DashScope API Key

2. **Embedding 模型**：
   - 使用 DashScope text-embedding-v2 API
   - 维度：1536
   - 批量处理：每次最多 25 条文本

3. **Cross-Encoder 精排**：
   - 默认使用 `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - 需要 sentence-transformers 库
   - 如遇 tensorflow/transformers 兼容性问题，会自动跳过精排

4. **缓存配置**：
   - 支持 Redis 作为分布式缓存
   - 未配置 Redis 时自动降级为内存缓存

## 与原项目对比

| 特性 | 原实现 | LangChain 实现 |
|------|--------|---------------|
| 文档加载 | 自定义 | PyPDFLoader |
| 文本切分 | 自定义 | RecursiveCharacterTextSplitter |
| 向量化 | 自定义 | DashScope text-embedding-v2 |
| 向量存储 | 自定义 | Chroma |
| 检索融合 | 自定义分数加权 | 纯 RRF 排名融合 |
| Query 分类 | 无 | 正则+词典+特征分类 |
| 精排 | 无 | Cross-Encoder |
| 回归集 | 无 | Bad Case 校准 |
| 缓存 | 无 | 三级缓存系统 |
| 并发支持 | 基础 | 异步接口支持 |

## API 服务

### 启动服务

```bash
# 启动 FastAPI 服务
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 接口列表

| 接口 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/query` | POST | RAG 查询 |
| `/batch_query` | POST | 批量查询 |
| `/stats` | GET | 服务统计 |

### 使用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 单条查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "等待期是多少天？"}'

# 批量查询
curl -X POST http://localhost:8000/batch_query \
  -H "Content-Type: application/json" \
  -d '["等待期是多少天？", "什么是意外保险？"]'
```

### 熔断机制

- 使用 `pybreaker` 实现断路器模式
- 连续失败 5 次触发熔断
- 熔断后 30 秒自动尝试恢复

## 待优化项

1. ✅ ~~修复 tensorflow/transformers 兼容性问题~~（已添加异常处理）
2. ✅ ~~添加 Query 分类器~~
3. ✅ ~~添加纯 RRF 合并器~~
4. ✅ ~~添加 Cross-Encoder 精排~~
5. ✅ ~~添加 Bad Case 回归集~~
6. ✅ ~~添加 API 服务封装（FastAPI）~~
7. ✅ ~~添加限流和熔断机制~~
8. 添加监控指标