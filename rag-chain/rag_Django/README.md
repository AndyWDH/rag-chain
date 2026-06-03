# RAG Django - 基于 Django + LangGraph 的智能问答系统

> 企业级 RAG 问答系统，集成 Query 分类、双路检索、RRF 融合、Cross-Encoder 精排和 SSE 流式输出。

---

## 🎯 项目特性

### 核心功能
| 功能 | 描述 |
|------|------|
| **Query 智能分类** | 基于正则、领域词典、长度和疑问句特征，自动分类为"偏关键词/偏语义/平衡"三类 |
| **双路检索** | 向量检索 + BM25 关键词检索，根据 Query 类型动态分配权重 |
| **RRF 结果融合** | 基于排名的融合算法，解决不同检索器分数尺度差异问题 |
| **Cross-Encoder 精排** | 对 top 候选进行相关性重排序，提升答案质量 |
| **SSE 流式输出** | 实时返回回答内容，提升用户体验 |
| **三级缓存系统** | L1 问答结果、L2 检索结果、L3 向量嵌入 |

### 技术栈
- **框架**: Django 5.0 + Django REST Framework
- **状态机**: LangGraph
- **向量数据库**: ChromaDB
- **前端**: Vue 3 + Tailwind CSS 3
- **LLM**: 阿里云 DashScope (Qwen)
- **Embedding**: DashScope Text Embedding

---

## 📁 项目结构

```
rag_Django/
├── apps/                    # 应用模块
│   ├── core/                # 核心基础模块
│   ├── documents/           # 文档管理（向量存储、BM25）
│   ├── retrieval/           # 检索核心（LangGraph状态机）
│   ├── chat/                # 对话接口（REST API + SSE）
│   └── analytics/           # 分析统计模块
├── config/                  # Django 配置
│   ├── settings.py          # 项目配置
│   ├── urls.py              # URL 路由
│   └── wsgi.py              # WSGI 入口
├── templates/               # 前端模板
│   └── index.html           # Vue 聊天界面
├── .env                     # 环境变量
├── db.sqlite3               # SQLite 数据库
└── manage.py                # Django 管理命令
```

---

## 🚀 快速开始

### 1. 环境要求
- Python >= 3.10
- Django >= 5.0
- DashScope API Key（申请地址：https://dashscope.console.aliyuncs.com/apiKey）

### 2. 安装依赖

```bash
cd rag_Django
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的 DashScope API Key：

```env
# DashScope API Key（必填）
DASHSCOPE_API_KEY=your-api-key-here

# LLM 模型配置
LLM_MODEL=qwen-plus

# 检索参数
RETRIEVAL_K=10
RERANK_TOP_N=5
RRF_K=60
```

### 4. 数据库迁移

```bash
python manage.py migrate
```

### 5. 启动服务

```bash
python manage.py runserver 0.0.0.0:8000
```

### 6. 访问服务

| 地址 | 说明 |
|------|------|
| http://localhost:8000/ | 前端聊天界面 |
| http://localhost:8000/admin/ | Django 管理后台 |
| http://localhost:8000/api/v1/ | REST API 接口 |

---

## 🔌 API 接口

### 查询接口

**POST** `/api/v1/chat/query/`

请求体：
```json
{
    "query": "犹豫期几天",
    "session_id": null,
    "use_cache": true
}
```

响应：
```json
{
    "answer": "15天。根据保险合同条款，投保人在签收保单后享有15天的犹豫期...",
    "sources": ["insurance_policy_v1.pdf"],
    "query_class": "keyword",
    "class_scores": {"keyword": 0.7, "semantic": 0.0, "balanced": 0.3},
    "channel_weights": [0.3, 0.7],
    "latency": 3.5,
    "cached": false
}
```

### 流式接口

**GET** `/api/v1/chat/query/stream/?query=犹豫期几天`

返回 SSE 事件流：
```
event: metadata
data: {"query_class": "keyword", "channel_weights": [0.3, 0.7]}

event: content
data: {"chunk": "15", "index": 0, "total": 3}

event: content
data: {"chunk": "天。", "index": 1, "total": 3}

event: finish
data: {"status": "completed"}
```

### 会话管理

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/chat/sessions/` | GET/POST | 获取/创建会话 |
| `/api/v1/chat/sessions/{id}/` | GET/PUT/DELETE | 会话详情/更新/删除 |
| `/api/v1/chat/sessions/{id}/messages/` | GET | 获取会话消息 |

---

## 🧠 检索流程

```
用户查询
    ↓
Query 分类器
    ├─→ 偏关键词 → BM25 权重 0.7
    ├─→ 偏语义   → 向量权重 0.7
    └─→ 平衡     → 权重各 0.5
    ↓
双路检索（向量 + BM25）
    ↓
RRF 合并（1/(k+rank)）
    ↓
Cross-Encoder 精排
    ↓
LLM 生成回答
    ↓
SSE 流式输出
```

---

## ⚙️ 配置说明

### 通道权重配置

在 `config/settings.py` 中配置不同 Query 类型的通道权重：

```python
CHANNEL_WEIGHTS = {
    'keyword': [0.3, 0.7],    # [向量权重, BM25权重]
    'semantic': [0.7, 0.3],
    'balanced': [0.5, 0.5],
}
```

### 检索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RETRIEVAL_K` | 10 | 每路检索返回文档数 |
| `RERANK_TOP_N` | 5 | 精排后保留文档数 |
| `RRF_K` | 60 | RRF 公式中的常数 k |

---

## 📊 前端界面

前端聊天界面已内置，访问 http://localhost:8000/ 即可使用：

**功能特性**：
- ✅ 实时消息气泡显示
- ✅ 打字状态指示器
- ✅ 查询分类标签（偏关键词/偏语义/平衡）
- ✅ 引用来源展示
- ✅ 示例问题快捷按钮
- ✅ 清空对话功能

---

## 🛠️ 开发指南

### 创建新的检索节点

在 `apps/retrieval/nodes.py` 中添加新节点：

```python
class CustomNode:
    def __init__(self):
        pass
    
    def process(self, state: Dict) -> Dict:
        # 处理逻辑
        return state
```

### 添加新的 API 接口

在 `apps/chat/views.py` 中添加新接口：

```python
class CustomViewSet(viewsets.ViewSet):
    def list(self, request):
        # 实现逻辑
        return Response(data)
```

---

## 📝 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**项目状态**：✅ 生产就绪