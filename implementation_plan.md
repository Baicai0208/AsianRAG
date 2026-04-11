# 高级 Chunking 策略升级

## 背景

当前 `chunking.py` 实现了三种**规则驱动**的分块策略：
- `fixed_size` — 固定词数滑动窗口
- `sentence` — 按句子边界累积
- `paragraph` — 按 `\n\n` 段落边界

这三种策略都是**纯启发式的**，只看字符/词数/标点符号，不理解文本的语义结构。问题是：
1. 一个完整的语义段落可能被切断（比如讨论"四川菜的调味特点"被切成了两个 chunk）
2. 不相关的内容可能被合并到同一个 chunk（仅因为字符数没满）
3. chunk 之间的 overlap 是机械的，不一定在语义自然的位置

## 提议新增两种高级策略

### 策略 4：Semantic Chunking（语义分块）

> **核心思想**：用 embedding 模型计算相邻句子的语义相似度，在相似度"骤降"的位置切分。切分点不靠字符数，而靠**意义的转折**。

**工作流程**：
1. 先将文档按句子分割
2. 对每个句子做 embedding（使用已有的 `BAAI/bge-small-en-v1.5`）
3. 计算相邻句子 embedding 的余弦相似度
4. 相似度低于阈值（或用百分位数动态阈值）的位置 = 语义断点
5. 在断点处切分，形成语义连贯的 chunk
6. 对超长 chunk 再做二次切分（fallback 到 sentence 策略）

**优势**：
- 每个 chunk 内部语义高度一致，检索时不会"答非所问"
- 自适应文本结构，不依赖任何固定窗口大小
- 特别适合维基百科这种多话题混编的长文本

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `breakpoint_percentile` | 85 | 相似度低于此百分位的位置作为切分点 |
| `max_chars` | 1500 | 单 chunk 上限，超过则二次切分 |
| `min_chars` | 100 | 过短 chunk 与邻居合并 |

---

### 策略 5：Late Chunking（延迟分块）

> **核心思想**：先对整篇文档做 embedding（利用 transformer 的长上下文注意力），然后再按句子/段落边界切分 token，对每个 chunk 内的 token 求均值得到 chunk embedding。这样每个 chunk 的向量**天然携带全文上下文**。

**工作流程**：
1. 将整篇文档输入 embedding 模型，获取每个 token 的上下文化 embedding（最后一层隐藏态）
2. 按句子边界将 token 序列切分成若干 span
3. 对每个 span 内的 token embeddings 做 mean pooling → 得到 chunk embedding
4. chunk 文本 + 预计算的 embedding 一起存储

**优势**：
- chunk 级 embedding 携带**全文上下文**，解决了传统短文本编码信息不足的问题
- 对代词消解（如 "It is then marinated..." 中的 "It" 指代前文食材）特别有效
- 论文实验在多个检索 benchmark 上优于 naive chunking + embedding

**但有一个重要约束**：
Late Chunking 的 embedding 过程和普通策略不同 —— 它不经过 `embedding.py` 的标准流程，而是在 chunking 阶段就同步产生 embedding。这需要对 `embedding.py` 做适配。

> [!IMPORTANT]
> Late Chunking 改变了"先分块再嵌入"的流水线顺序。对于这个策略，chunking 阶段就会输出 `chunked_corpus.json` **和** 预计算的 embedding，`embedding.py` 需要跳过重新编码、直接加载预计算向量。

**参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chars` | 800 | 按句子累积的 chunk 上限 |
| `overlap_sentences` | 1 | 句子重叠数 |

---

## Proposed Changes

### [MODIFY] [chunking.py](file:///home/baicai/NLP/Asian/notebooks/chunking.py)

主要改动：

1. **新增 `chunk_semantic()` 函数**：
   - 加载 `SentenceTransformer` 模型
   - 计算相邻句子的余弦相似度
   - 用百分位阈值找语义断点
   - 在断点处切分
   - 过短 chunk 合并，过长 chunk fallback 到 `chunk_sentence()`

2. **新增 `chunk_late()` 函数**：
   - 加载 `SentenceTransformer` 模型并使用 `model.encode()` 获取 token-level embeddings
   - 对文档做 tokenize，获取 token 到句子的映射
   - 按句子边界累积 span，对每个 span 做 mean pooling
   - 返回 chunk 文本 + 预计算向量

3. **更新 `STRATEGY_MAP`** 和 **CLI `--strategy` choices**

4. **更新 `run_chunking()`**：对 `late` 策略额外输出 `precomputed_embeddings.npy`

---

### [MODIFY] [embedding.py](file:///home/baicai/NLP/Asian/notebooks/embedding.py)

- 更新 `STRATEGIES` 列表，加入 `"semantic"` 和 `"late"`
- 对 `late` 策略：检测 `precomputed_embeddings.npy` 是否存在，如果有则跳过编码，直接使用预计算向量构建 FAISS 索引

---

### [MODIFY] [pipeline.py](file:///home/baicai/NLP/Asian/notebooks/pipeline.py)

- 更新 `--strategy` choices，加入 `"semantic"` 和 `"late"`

---

### [MODIFY] [retriever.py](file:///home/baicai/NLP/Asian/notebooks/retriever.py)

- 更新 `--strategy` choices，加入 `"semantic"` 和 `"late"`

---

## Open Questions

> [!IMPORTANT]
> **1. 是否两种策略都要加？** Semantic Chunking 实现相对简单、改动小；Late Chunking 更先进但改动流水线较大。你是想两种都加，还是只加其中一种？

> [!NOTE]
> **2. 关于 Late Chunking 的模型约束**：`bge-small-en-v1.5` 的最大 token 长度是 512。对于超过 512 tokens 的文档（你的语料平均 5700 字符 ≈ ~1400 tokens），需要做文档级滑动窗口或截断。我的计划是用滑动窗口拼接，你觉得可以吗？或者是否考虑换用更长上下文的模型？

> [!NOTE]
> **3. Semantic Chunking 的 `breakpoint_percentile` 参数**：85 是 LangChain 和多数论文的默认值，但根据语料特点可以调整。我会先用 85，然后在统计输出中显示 chunk 数量和长度分布供你调优。

---

## Verification Plan

### Automated Tests

```bash
# 1. 运行新策略的 chunking
cd notebooks
python chunking.py --strategy semantic
python chunking.py --strategy late
python chunking.py --strategy all

# 2. 构建 FAISS 索引
python embedding.py --strategy semantic
python embedding.py --strategy late

# 3. 测试检索
python retriever.py --strategy semantic --query "What are the main characteristics of Sichuan cuisine?"
python retriever.py --strategy late --query "What are the main characteristics of Sichuan cuisine?"

# 4. 跑 benchmark 并对比
python pipeline.py --strategy semantic --backend openrouter --api_model <模型>
python pipeline.py --strategy late --backend openrouter --api_model <模型>
python eval.py  # 对比各策略的 retrieval 和 generation metrics
```

### Manual Verification

- 对比 semantic/late 策略和 sentence 策略的 chunk 长度分布
- 抽样数个 chunk，人工确认语义连贯性是否改善
- 对比 benchmark 上的 Hit Rate / MRR / ROUGE 指标
