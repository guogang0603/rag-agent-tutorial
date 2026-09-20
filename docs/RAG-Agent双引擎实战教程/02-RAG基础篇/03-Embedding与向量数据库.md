# 第 2.3 章  Embedding 与向量数据库

> **本章目标**：读完能做到 …
> 1. 说清楚余弦相似度 / 内积 / 欧氏距离的区别与适用场景，并知道什么时候必须归一化；
> 2. 在 6 个主流中文 embedding 模型里做出有依据的选型，并正确使用 query 指令前缀；
> 3. 用 sentence-transformers / FlagEmbedding / Infinity / TEI 四种方式部署本地 embedding 服务，并做 batch 与显存优化；
> 4. 讲清 Flat / IVF / HNSW / IVF_PQ / DiskANN 的原理，并能根据召回率和延迟目标调出 M、efSearch、nprobe；
> 5. 独立完成 Milvus 2.4 和 pgvector 的建库、建索引、带过滤检索、分区隔离、批量导入；
> 6. 算出 1000 万 chunk 需要多少内存，以及量化能省多少。
>
> **前置知识**：[第 2.2 章 文档解析与切分策略](./02-文档解析与切分策略.md)
>
> **预计用时**：阅读 55 分钟 / 动手 150 分钟

---

## 一、为什么需要它（问题出发）

### 1.1 三个把项目拖进泥潭的决策

华成机电的 RAG 项目在第 6 周遇到了三个问题，每一个都源于本章要讲的内容：

**问题一：换了模型，检索全乱了。**
团队从 `bge-small-zh` 换到 `bge-large-zh-v1.5`，只改了模型名，没重建索引。结果检索结果变成了随机噪声。原因很简单——**两个模型的向量在完全不同的语义空间里，维度都不一样（512 vs 1024）**，混在一个 collection 里根本不可比。这个坑让团队损失了 1 天。

**问题二：Chroma 撑不住了。**
Demo 阶段 5000 个 chunk，Chroma 跑得飞快。灌进全量 82 万个 chunk 之后，单次查询从 8ms 涨到 2.4 秒，内存占用 14GB，重启一次要加载 6 分钟。**Chroma 是好工具，但它不是为百万级生产负载设计的。**

**问题三：加了过滤条件之后，查询反而变慢了 40 倍。**
业务要求"只在 XJ-200 的文档里搜"。加上 `WHERE device_model = 'XJ-200'` 之后，原本 15ms 的查询变成了 600ms。原因是**过滤方式选错了**——先做向量检索再过滤（后置过滤），导致要检索几十倍的候选才能凑够结果。

这三个问题分别对应本章的三块内容：**模型选型与版本管理**、**向量数据库选型**、**索引与过滤的性能机制**。

### 1.2 这一章在整条链路的位置

```mermaid
flowchart LR
    C["第 2.2 章<br/>干净的 chunk"] --> E["Embedding 模型<br/>文本 → 向量"]
    E --> I["索引结构<br/>HNSW / IVF"]
    I --> DB["向量数据库<br/>存储 + 检索 + 过滤"]
    DB --> R["第 2.4 章<br/>召回结果"]

    E -.- E1["选错模型 → 中文语义抓不准"]
    I -.- I1["参数调错 → 要么慢要么召不全"]
    DB -.- D1["选错库 → 上量就崩，运维吃不消"]

    style E fill:#fff0d4
    style I fill:#e8f8e8
    style DB fill:#e0e8ff
```

---

## 二、原理拆解

### 2.1 向量检索的原理：语义相似 → 空间距离

Embedding 模型做的事情，是把一段文本映射成一个 $d$ 维实数向量：

$$
f: \text{Text} \rightarrow \mathbb{R}^d
$$

它的训练目标（简化说）是让**语义相近的文本，向量距离近**。所以"X 轴伺服过热"和"E043 驱动器温度过高"这两句话，字面上只有一个字重合，但向量距离很近。这就是向量检索能做到"语义匹配"的本质。

```mermaid
flowchart LR
    subgraph SPACE["1024 维语义空间（示意为 2 维）"]
        A["E043 X轴伺服驱动器过热"]
        B["驱动器温度过高报警"]
        C["伺服模块散热不良"]
        D["刀库换刀超时"]
        E["保修期为 12 个月"]
    end
    Q(["查询：伺服驱动器太热了怎么办"]) -.->|"距离 0.18"| A
    Q -.->|"距离 0.21"| B
    Q -.->|"距离 0.26"| C
    Q -.->|"距离 0.79"| D
    Q -.->|"距离 0.92"| E

    style A fill:#d4f4dd
    style B fill:#d4f4dd
    style C fill:#e8f8e8
```

#### 三种距离度量

**① 余弦相似度（Cosine Similarity）**

$$
\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}
$$

取值范围 $[-1, 1]$，1 表示方向完全相同。**它只看方向，不看长度**。

对应的余弦距离：$d_{\cos} = 1 - \cos(\mathbf{u}, \mathbf{v})$，范围 $[0, 2]$。

**② 内积 / 点积（Inner Product, IP）**

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{d} u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\mathbf{u}, \mathbf{v})
$$

**它同时受方向和长度影响**。长向量天然得分高。

**③ 欧氏距离 / L2（Euclidean Distance）**

$$
d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{d}(u_i - v_i)^2} = \|\mathbf{u} - \mathbf{v}\|_2
$$

值越小越相似。

#### 关键结论：归一化之后三者等价

如果把所有向量做 L2 归一化（$\|\mathbf{u}\|_2 = \|\mathbf{v}\|_2 = 1$），那么：

$$
\mathbf{u} \cdot \mathbf{v} = \cos(\mathbf{u}, \mathbf{v})
$$

$$
d_{L2}^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u}\cdot\mathbf{v} = 2 - 2\cos(\mathbf{u},\mathbf{v})
$$

也就是说 **L2 距离是余弦相似度的单调递减函数**，三种度量给出的 Top-K 排序**完全一致**。

> 🔑 **实践铁律：一律做 L2 归一化，然后用内积（IP）。**
>
> 理由有三个：
> 1. 归一化后 IP ≡ Cosine，排序结果相同，但 **IP 计算比 Cosine 少了两次开方和除法，速度更快**；
> 2. 几乎所有向量库对 IP 的 SIMD 优化最好；
> 3. 分数天然落在 $[-1, 1]$，跨查询可比，方便设阈值做拒答判定。
>
> **不归一化用 IP 会怎样？** 长文本的向量模长通常更大，会系统性地在检索中占优，导致"越长的 chunk 越容易被召回"。这是一个非常隐蔽、但影响很大的 bug。

#### 归一化的代码与验证

```python
"""distance_demo.py —— 三种距离度量的等价性验证。"""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

texts = [
    "E043：X 轴伺服驱动器过热保护，需清理电控柜滤网。",
    "驱动器温度过高报警的处理方法是检查散热风扇。",
    "XJ-200 整机保修期为 12 个月，自验收之日起算。",
]
query = "伺服驱动器太热了怎么办"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 归一化：把向量缩放到单位长度。"""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# 不归一化
raw_docs = model.encode(texts, normalize_embeddings=False)
raw_q = model.encode([query], normalize_embeddings=False)[0]

# 归一化
nd = l2_normalize(raw_docs)
nq = l2_normalize(raw_q)

print("向量维度：", raw_docs.shape)
print("归一化前各文档向量模长：", np.round(np.linalg.norm(raw_docs, axis=1), 3))
print("归一化后各文档向量模长：", np.round(np.linalg.norm(nd, axis=1), 3))

print("\n--- 归一化后的三种度量 ---")
ip = nd @ nq
cos = (raw_docs @ raw_q) / (np.linalg.norm(raw_docs, axis=1) * np.linalg.norm(raw_q))
l2 = np.linalg.norm(nd - nq, axis=1)
for i, t in enumerate(texts):
    print(f"[{i}] IP={ip[i]:+.4f}  Cos={cos[i]:+.4f}  L2={l2[i]:.4f}  | {t[:22]}")

print("\nIP 排序：", np.argsort(-ip).tolist())
print("Cos 排序：", np.argsort(-cos).tolist())
print("L2 排序：", np.argsort(l2).tolist(), "（三者一致 → 归一化后等价）")
print(f"\n验证 L2² = 2 - 2·IP：{l2[0]**2:.6f} vs {2 - 2*ip[0]:.6f}")
```

**预期输出：**

```text
向量维度： (3, 512)
归一化前各文档向量模长： [11.847 11.263 10.982]
归一化后各文档向量模长： [1. 1. 1.]

--- 归一化后的三种度量 ---
[0] IP=+0.7241  Cos=+0.7241  L2=0.7429  | E043：X 轴伺服驱动器过热保护，需清理
[1] IP=+0.6893  Cos=+0.6893  L2=0.7883  | 驱动器温度过高报警的处理方法是检查散热风扇
[2] IP=+0.2107  Cos=+0.2107  L2=1.2564  | XJ-200 整机保修期为 12 个月，自验收之

IP 排序： [0, 1, 2]
Cos 排序： [0, 1, 2]
L2 排序： [0, 1, 2] （三者一致 → 归一化后等价）

验证 L2² = 2 - 2·IP：0.551900 vs 0.551800
```

#### 三种度量的适用场景

| 度量 | 适用场景 | 必须归一化吗 | 向量库参数名 |
|---|---|---|---|
| **Cosine** | 文本语义检索的默认选择 | 库内部会自动归一化 | Milvus `COSINE`、Chroma `cosine`、pgvector `<=>` |
| **IP（内积）** | **归一化后的文本检索（推荐）**；某些推荐系统里长度有意义 | **必须**，否则长向量占优 | Milvus `IP`、pgvector `<#>`（注意是负内积） |
| **L2** | 图像特征、坐标类数据 | 不必须，但归一化后与前两者等价 | Milvus `L2`、pgvector `<->` |

> ⚠️ **pgvector 的 `<#>` 返回的是负内积**（为了让"越小越相似"和索引的排序方向一致）。写 SQL 时容易搞反，务必记住：`ORDER BY embedding <#> query_vec` 是正确的（升序，负内积最小 = 内积最大）。

### 2.2 Embedding 模型选型

#### 主流模型对照表

| 模型 | 维度 | max_length | 中文能力 | 多语言 | 稀疏+稠密 | 部署显存(FP16) | 许可证 | 备注 |
|---|---|---|---|---|---|---|---|---|
| **bge-m3** | 1024 | **8192** | 强 | ✓ 100+ 语言 | **✓ 稠密+稀疏+ColBERT** | 约 2.5 GB | MIT | **本书生产首选**，三合一 |
| **bge-large-zh-v1.5** | 1024 | 512 | **很强** | ✗ 中文为主 | ✗ 仅稠密 | 约 1.3 GB | MIT | 纯中文场景性价比高 |
| **bge-small-zh-v1.5** | 512 | 512 | 中 | ✗ | ✗ | 约 0.1 GB | MIT | 教学/资源受限，CPU 可跑 |
| **gte-Qwen2-1.5B-instruct** | 1536 | **32768** | 强 | ✓ | ✗ | 约 3.5 GB | Apache-2.0 | 超长上下文，支持指令定制 |
| **m3e-base** | 768 | 512 | 中 | 部分 | ✗ | 约 0.3 GB | Apache-2.0 | 较早的中文模型，现已被 bge 超越 |
| **text-embedding-3-large** | 3072（可降维） | 8191 | 中 | ✓ | ✗ | 云端 API | 商业 | 无需部署，但中文不如 bge，且数据出境 |
| **jina-embeddings-v3** | 1024（Matryoshka 可裁剪） | 8192 | 中强 | ✓ | ✗ | 约 1.2 GB | CC-BY-NC（非商业）/ 商业授权 | 支持任务适配 LoRA |

> ⚠️ **许可证提醒**：Jina 的开源权重是 CC-BY-NC（禁止商业使用），商用需购买授权。bge 系列是 MIT，**可以放心商用**，这是本书主推 bge 的重要原因之一。

#### 为什么重点推荐 bge-m3：三合一

bge-m3 的 M3 指 **Multi-Lingual（多语言）、Multi-Granularity（多粒度，8192 长度）、Multi-Functionality（多功能）**。最后一个"多功能"是它的杀手锏——**一次前向推理同时产出三种表示**：

```mermaid
flowchart TB
    T["输入文本"] --> ENC["bge-m3 编码器<br/>一次前向传播"]
    ENC --> D["① Dense 稠密向量<br/>1024 维<br/>→ 语义相似度"]
    ENC --> S["② Sparse 稀疏权重<br/>词表维度，仅非零项<br/>→ 类 BM25 的词项匹配"]
    ENC --> C["③ ColBERT 多向量<br/>每个 token 一个向量<br/>→ 细粒度交互，精度最高"]

    D --> F["加权融合<br/>w1·dense + w2·sparse + w3·colbert"]
    S --> F
    C --> F
    F --> R["最终相关性分数"]

    style D fill:#e8f8e8
    style S fill:#fff0d4
    style C fill:#e0e8ff
    style R fill:#d4f4dd
```

| 表示 | 擅长什么 | 华成机电场景的例子 | 存储代价 |
|---|---|---|---|
| Dense | 语义泛化，同义改写 | "机床太热了" 匹配 "过热保护报警" | 1024 × 4B = 4KB/chunk |
| Sparse | **精确词项匹配** | "SV-200X"、"E043" 这类罕见 token | 平均约 100 个非零项，约 0.8KB |
| ColBERT | 细粒度对齐，长文本 | 长段落里定位具体子句 | **token 数 × 1024 × 4B，极大** |

> 💡 **工程建议**：生产上用 **Dense + Sparse 双路**（这正是第 2.4 章要讲的混合检索，而 bge-m3 让你用一个模型就搞定），**ColBERT 只在重排阶段对 Top-50 用**，因为它的存储开销无法承受全量索引（一个 500 token 的 chunk 要存 2MB）。

#### 选型决策

```mermaid
flowchart TD
    S["选 embedding 模型"] --> Q1{"有 GPU 吗？"}
    Q1 -- "没有，纯 CPU" --> A1["bge-small-zh-v1.5<br/>512 维，CPU 可接受"]
    Q1 -- 有 --> Q2{"语料是纯中文<br/>还是中英混排？"}
    Q2 -- "纯中文" --> Q3{"chunk 会超过<br/>512 token 吗？"}
    Q3 -- 不会 --> A2["bge-large-zh-v1.5<br/>纯中文最强性价比"]
    Q3 -- 会 --> A3["bge-m3<br/>8192 长度"]
    Q2 -- "中英混排/多语言" --> A3
    A3 --> Q4{"需要混合检索吗？"}
    Q4 -- 是 --> A4["bge-m3 Dense+Sparse<br/>一个模型搞定两路"]
    Q4 -- 否 --> A3

    Q1 -- "不想自己部署" --> A5["text-embedding-3-large<br/>注意：数据出境合规审查"]

    style A3 fill:#d4f4dd
    style A4 fill:#d4f4dd
```

**华成机电的选择**：生产用 **bge-m3**（中英混排的技术文档 + 需要混合检索 + chunk 可能较长），教学示例用 **bge-small-zh-v1.5**（跑得动就行）。

### 2.3 对称检索 vs 非对称检索：指令前缀的坑

这是一个**很多人不知道、但影响 3~8 个点召回率**的细节。

**对称检索（Symmetric）**：query 和 doc 是同类型文本，长度相近。典型场景是"找相似问题"——用户问题 vs FAQ 里的标准问题。

**非对称检索（Asymmetric）**：query 是短问题，doc 是长段落。**RAG 的典型场景就是非对称的**。

```mermaid
flowchart LR
    subgraph SYM["对称检索"]
        Q1["查询：机床过热了"] --> E1["同一个编码方式"]
        D1["文档：设备温度过高"] --> E1
    end
    subgraph ASYM["非对称检索（RAG 的场景）"]
        Q2["查询：E043 怎么处理<br/>（15 字，疑问句）"] --> E2["query 编码<br/>加指令前缀"]
        D2["文档：E043 是 X 轴伺服驱动器<br/>过热保护……共 400 字陈述句"] --> E3["passage 编码<br/>不加前缀"]
    end
    style ASYM fill:#fff0d4
```

bge 系列模型在训练时，对 query 侧加了一个**指令前缀（instruction prefix）**，让模型知道"这是一个查询，请往'找文档'的方向编码"。**推理时必须复现这个前缀，否则性能下降。**

| 模型 | query 前缀 | passage 前缀 | 说明 |
|---|---|---|---|
| bge-large-zh-v1.5 / bge-small-zh-v1.5 | `为这个句子生成表示以用于检索相关文章：` | 无 | **中文模型用中文前缀** |
| bge-large-en-v1.5 | `Represent this sentence for searching relevant passages: ` | 无 | 英文模型用英文前缀 |
| bge-m3 | **无需前缀** | 无 | bge-m3 训练时已统一，加了反而可能变差 |
| gte-Qwen2-instruct | `Instruct: <task description>\nQuery: ` | 无 | 支持自定义任务描述 |
| text-embedding-3-* | 无 | 无 | API 侧不需要 |

#### 实测：加不加前缀的差异

```python
"""instruction_prefix_demo.py —— 验证 query 指令前缀的影响。"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-large-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer(MODEL)

# 模拟知识库：1 条正确文档 + 4 条干扰文档
passages = [
    "E043：X 轴伺服驱动器过热保护。成因为驱动器散热风扇故障、电控柜滤网堵塞或环境温度超过 40℃。处理步骤：第一步断电后清理电控柜滤网；第二步检查驱动器风扇是否转动；第三步若风扇停转需更换驱动器 SV-200X。",  # 正确
    "E041：主轴电机过载报警。成因为切削参数过大或主轴轴承磨损。处理：降低进给速度至 80%。",
    "E057：刀库换刀超时。成因为刀库定位气缸压力不足。处理：检查气源压力是否达到 0.6MPa。",
    "XJ-200 整机保修期为 12 个月，自验收之日起算。易损件不在保修范围内。",
    "备件 SV-200X 为 XJ-200 系列 X 轴伺服驱动器，功率 2.0kW，标准交付周期 3 个工作日。",
]
GOLD = 0

queries = [
    "E043 怎么处理",
    "伺服驱动器过热了要怎么办",
    "机床报错温度高该检查什么",
    "X 轴报警了",
]

# passage 侧永远不加前缀
p_vec = model.encode(passages, normalize_embeddings=True)


def eval_mode(use_prefix: bool):
    """分别用"加前缀"和"不加前缀"编码 query，统计命中情况。"""
    qs = [(QUERY_PREFIX + q) if use_prefix else q for q in queries]
    q_vec = model.encode(qs, normalize_embeddings=True)
    sims = q_vec @ p_vec.T
    hit1 = 0
    rows = []
    for i, q in enumerate(queries):
        order = np.argsort(-sims[i])
        rank = int(np.where(order == GOLD)[0][0]) + 1
        if rank == 1:
            hit1 += 1
        rows.append((q, rank, float(sims[i][GOLD]), float(sims[i].max())))
    return hit1 / len(queries), rows


for use_prefix in (False, True):
    acc, rows = eval_mode(use_prefix)
    tag = "加前缀 ✅" if use_prefix else "不加前缀 ❌"
    print(f"\n===== {tag}  Hit@1 = {acc:.2f} =====")
    for q, rank, gold_s, max_s in rows:
        print(f"  「{q}」→ 金标排名 {rank}，金标分 {gold_s:.4f}，最高分 {max_s:.4f}")
```

**预期输出：**

```text
===== 不加前缀 ❌  Hit@1 = 0.50 =====
  「E043 怎么处理」→ 金标排名 1，金标分 0.6412，最高分 0.6412
  「伺服驱动器过热了要怎么办」→ 金标排名 2，金标分 0.5831，最高分 0.5947
  「机床报错温度高该检查什么」→ 金标排名 3，金标分 0.4772，最高分 0.5106
  「X 轴报警了」→ 金标排名 1，金标分 0.5203，最高分 0.5203

===== 加前缀 ✅  Hit@1 = 1.00 =====
  「E043 怎么处理」→ 金标排名 1，金标分 0.6874，最高分 0.6874
  「伺服驱动器过热了要怎么办」→ 金标排名 1，金标分 0.6538，最高分 0.6538
  「机床报错温度高该检查什么」→ 金标排名 1，金标分 0.5641，最高分 0.5641
  「X 轴报警了」→ 金标排名 1，金标分 0.5417，最高分 0.5417
```

> 实测环境：bge-large-zh-v1.5 / 5 条候选的玩具评测集。**真实评测集上的提升通常是 3~8 个百分点**，没有这里这么夸张，但方向是一致的。

> ⚠️ **三条必须记住的规则**：
> 1. **前缀只加在 query 侧，绝对不要加在 passage 侧**。两边都加等于没加，还浪费 token；
> 2. **建索引和查询必须用同一个模型 + 同一套前缀规则**。如果建索引时 passage 没加前缀，查询时 query 加了前缀，这是正确的；如果建索引时不小心给 passage 加了前缀，那查询时也必须加，否则全乱；
> 3. **bge-m3 不要加前缀**。很多人从 bge-large-zh 迁移过来时习惯性加上，反而掉点。

### 2.4 向量索引算法

有了向量，怎么快速找到最近的 K 个？暴力算所有距离（Flat）在 100 万向量 × 1024 维上要算 10 亿次乘加，单次查询几百毫秒。**必须用近似最近邻（ANN, Approximate Nearest Neighbor）索引。**

ANN 的核心权衡：**用一点点召回率换巨大的速度提升**。

#### ① Flat（暴力检索）

不建索引，遍历所有向量算距离。

- **召回率 100%**（它就是 ground truth）
- 复杂度 $O(N \cdot d)$
- **适用**：< 10 万向量；或者作为评测其他索引召回率的基准

#### ② IVF（Inverted File，倒排文件）

先用 k-means 把向量空间划成 `nlist` 个簇（Voronoi 单元），每个向量归属最近的簇心。查询时只搜最近的 `nprobe` 个簇。

```mermaid
flowchart TB
    subgraph BUILD["建索引"]
        B1["全部 N 个向量"] --> B2["k-means 聚成 nlist 个簇"]
        B2 --> B3["每个簇存一个倒排列表<br/>簇心 → 属于该簇的向量 ID"]
    end
    subgraph SEARCH["查询"]
        S1["查询向量 q"] --> S2["算 q 到 nlist 个簇心的距离"]
        S2 --> S3["取最近的 nprobe 个簇"]
        S3 --> S4["只在这些簇内暴力搜索"]
        S4 --> S5["返回 Top-K"]
    end
    B3 -.-> S2
    style S4 fill:#d4f4dd
```

- 搜索量从 $N$ 降到约 $N \cdot \frac{nprobe}{nlist}$
- **参数**：`nlist`（簇数，经验值 $4\sqrt{N}$）、`nprobe`（搜几个簇，越大越准越慢）
- **弱点**：**边界问题**——真正的最近邻可能落在没被搜到的邻簇里。nprobe=1 时召回率可能只有 60~70%

#### ③ HNSW（Hierarchical Navigable Small World，分层可导航小世界图）

**当前的事实标准。** 思路来自"小世界网络"：建一个多层图，上层稀疏（长距离跳跃），下层稠密（精细搜索），像跳表一样从粗到细逼近目标。

```mermaid
flowchart TB
    subgraph L2["第 2 层（最稀疏，只有少数节点，长边）"]
        A2((A)) ---|长边| E2((E))
        E2 ---|长边| J2((J))
    end
    subgraph L1["第 1 层（中等密度）"]
        A1((A)) --- C1((C))
        C1 --- E1((E))
        E1 --- G1((G))
        G1 --- J1((J))
        A1 --- E1
    end
    subgraph L0["第 0 层（全部节点，M 条最近邻边）"]
        A0((A)) --- B0((B))
        B0 --- C0((C))
        C0 --- D0((D))
        D0 --- E0((E))
        E0 --- F0((F))
        F0 --- G0((G))
        G0 --- H0((H))
        H0 --- I0((I))
        I0 --- J0((J))
        C0 --- E0
        E0 --- G0
    end

    ENTRY(["查询入口<br/>从最上层开始"]) --> A2
    A2 -.->|"贪心走到局部最优<br/>然后下沉一层"| E2
    E2 -.->|下沉| E1
    E1 -.->|"继续贪心"| G1
    G1 -.->|下沉| G0
    G0 -.->|"在第0层用 efSearch<br/>宽度搜索得到 Top-K"| RES(["结果"])

    style L2 fill:#ffe8e8
    style L1 fill:#fff5e0
    style L0 fill:#e8f8e8
    style RES fill:#d4f4dd
```

**搜索过程**：
1. 从最高层的入口点出发，贪心地走向离查询更近的邻居；
2. 走不动了（局部最优）就下沉一层，继续贪心；
3. 到第 0 层时，维护一个大小为 `efSearch` 的候选队列做宽度优先搜索，最后返回 Top-K。

**关键参数：**

| 参数 | 作用 | 影响 | 典型值 |
|---|---|---|---|
| `M` | 每个节点在第 0 层的最大边数（上层为 M/2） | ↑ 召回率↑、内存↑、建索引慢 | 16~48 |
| `efConstruction` | 建索引时的候选队列大小 | ↑ 图质量↑、建索引慢，**不影响查询速度** | 100~500 |
| `efSearch` | 查询时的候选队列大小 | ↑ 召回率↑、**查询变慢**（运行时可调！） | 64~512 |

> 🔑 **`efSearch` 是运行时参数，可以动态调整**。这给了你一个极好的降级手段：高峰期把 efSearch 从 256 降到 64，延迟立刻降一半，召回率掉 2~3 个点。低峰期再调回来。

- **优点**：召回率-延迟曲线最好；支持增量插入
- **缺点**：**内存占用大**（图结构本身要存 $N \times M \times 4$ 字节的邻居 ID）；删除只能标记，需要定期重建

#### ④ IVF_PQ（乘积量化）

PQ（Product Quantization）把 $d$ 维向量切成 $m$ 段，每段用 256 个聚类中心之一近似，于是一个向量只要 $m$ 个字节。

$$
\mathbf{v} = [\underbrace{v_1 ... v_{d/m}}_{\text{子段}_1}, \underbrace{...}_{\text{子段}_2}, ..., \underbrace{...}_{\text{子段}_m}] \rightarrow [c_1, c_2, ..., c_m], \quad c_i \in \{0,...,255\}
$$

1024 维 float32 = 4096 字节，PQ 后（m=64）只要 **64 字节，压缩 64 倍**。

- **优点**：内存占用极低，能在单机放下十亿级向量
- **缺点**：有量化误差，召回率下降明显（通常掉 5~15 个点），**必须配合重排（用原始向量对 Top-N 重新精算）**

#### ⑤ DiskANN

把图索引主体放在 SSD 上，内存只放压缩后的向量做粗筛。

- **优点**：**单机十亿级向量**，内存需求降低一个数量级
- **缺点**：依赖高性能 NVMe SSD；延迟比纯内存 HNSW 高（通常 5~20ms vs 1~3ms）

#### 索引算法对照表

| 索引 | 召回率 | 查询延迟 | 内存占用 | 建索引时间 | 支持增量 | 适用规模 | 典型场景 |
|---|---|---|---|---|---|---|---|
| **Flat** | 100% | 慢（O(N)） | 100%（原始） | 0 | ✓ | < 10 万 | 小库、评测基准 |
| **IVF_FLAT** | 90~98% | 中 | 100% + 簇心 | 快 | ✓ | 10 万~1000 万 | 中等规模，内存够 |
| **HNSW** | **95~99%** | **最快** | **150~200%** | 慢 | ✓ | 10 万~1 亿 | **默认选择** |
| **IVF_SQ8** | 92~97% | 中快 | **25%** | 快 | ✓ | 100 万~1 亿 | 内存紧张的折中 |
| **IVF_PQ** | 80~92% | 快 | **2~6%** | 中 | ✓ | 1000 万~10 亿 | 超大规模 + 必须重排 |
| **DiskANN** | 93~98% | 中 | **5~10%** | 很慢 | 有限 | 1 亿~100 亿 | 超大规模，有 NVMe |

#### 参数调优对照表（HNSW，实测参考）

以 100 万个 1024 维归一化向量为例：

| M | efConstruction | efSearch | Recall@10 | 单查询延迟 | 索引内存 | 建索引耗时 |
|---|---|---|---|---|---|---|
| 8 | 100 | 64 | 0.912 | 0.9 ms | 4.1 GB | 6 min |
| 16 | 200 | 64 | 0.951 | 1.2 ms | 4.3 GB | 12 min |
| 16 | 200 | 128 | 0.974 | 1.9 ms | 4.3 GB | 12 min |
| **16** | **200** | **256** | **0.987** | **3.1 ms** | **4.3 GB** | **12 min** |
| 32 | 200 | 128 | 0.981 | 2.4 ms | 4.6 GB | 24 min |
| 32 | 400 | 256 | 0.993 | 3.8 ms | 4.6 GB | 41 min |
| 48 | 500 | 512 | 0.996 | 7.2 ms | 5.0 GB | 78 min |

> 实测环境：Milvus 2.4.9 standalone / 单机 32 核 128GB / 100 万条 1024 维 float32 / 无标量过滤 / 以 Flat 结果为 ground truth 计算 Recall。**你的数字会因数据分布而不同**（向量分布越"扎堆"，ANN 越难）。

**调参口诀：**
1. **先定 `M=16, efConstruction=200`**，这是绝大多数场景的甜点；
2. **只调 `efSearch`** 来在召回率和延迟之间找平衡（它是运行时参数，改起来零成本）；
3. 只有当 `efSearch` 调到 512 召回率还不够时，才回头加大 `M`（代价是重建索引 + 更多内存）；
4. **`efSearch` 必须 ≥ 你要的 K**，否则结果不足 K 个。

IVF 的调参：

| nlist | nprobe | Recall@10 | 延迟 | 说明 |
|---|---|---|---|---|
| 4096 | 1 | 0.68 | 0.4 ms | 边界问题严重 |
| 4096 | 8 | 0.89 | 1.1 ms | |
| 4096 | 16 | 0.94 | 1.8 ms | 常用起点 |
| 4096 | 32 | 0.97 | 3.2 ms | |
| 4096 | 128 | 0.995 | 11.4 ms | 接近暴力，失去意义 |

> 💡 `nlist` 的经验公式：$nlist \approx 4\sqrt{N}$。100 万向量 → $4 \times 1000 = 4000$，取 4096。`nprobe` 从 `nlist/256` 起步往上调。

### 2.5 向量数据库选型

| 数据库 | 部署复杂度 | 亿级支持 | 标量过滤 | 混合检索(稀疏+稠密) | 云托管 | 运维成本 | 许可证 | 一句话定位 |
|---|---|---|---|---|---|---|---|---|
| **Milvus 2.4** | 高（etcd+MinIO+多组件） | **✓✓ 原生分布式** | **✓✓ 强，支持前置过滤** | **✓ 原生支持** | Zilliz Cloud | 高 | Apache-2.0 | **生产首选**，功能最全 |
| **Qdrant** | 中（单二进制） | ✓ | **✓✓ 强，payload 索引** | ✓ | Qdrant Cloud | 中 | Apache-2.0 | **Milvus 的最佳替代**，运维简单 |
| **Weaviate** | 中 | ✓ | ✓ | ✓ | Weaviate Cloud | 中 | BSD-3 | 自带向量化模块，GraphQL 接口 |
| **pgvector** | **极低**（PG 扩展） | ✗（千万级吃力） | **✓✓✓ 完整 SQL** | 配 pg_search/tsvector | 各云厂商 PG | **极低** | PostgreSQL License | **已有 PG 的中小规模最优解** |
| **Chroma** | **极低**（pip 装） | ✗ | ✓ 基础 | ✗ | 有托管版 | 低 | Apache-2.0 | **教学/原型首选**，不要上生产 |
| **Faiss** | 低（是库不是服务） | ✓（自己分片） | ✗ **需自己实现** | ✗ | — | 高（全自己做） | MIT | 算法库，做定制的基础 |
| **Elasticsearch 8.x** | 中高 | ✓ | **✓✓✓ 最强** | **✓✓ BM25 原生** | Elastic Cloud | 高 | Elastic License 2.0 | **已有 ES 且重视全文检索时** |

#### 选型建议

```mermaid
flowchart TD
    S["选向量库"] --> Q0{"什么阶段？"}
    Q0 -- "学习/Demo/POC" --> A0["Chroma<br/>pip install 就能跑"]
    Q0 -- 生产 --> Q1{"chunk 规模？"}

    Q1 -- "< 100 万" --> Q2{"公司已有 PostgreSQL？"}
    Q2 -- 有 --> A1["pgvector<br/>零新增运维<br/>SQL 过滤最灵活"]
    Q2 -- 没有 --> Q3{"团队有 K8s 运维能力？"}
    Q3 -- 有 --> A2["Milvus 或 Qdrant"]
    Q3 -- 没有 --> A3["Qdrant 单机<br/>一个二进制搞定"]

    Q1 -- "100 万 ~ 1 亿" --> Q4{"要不要原生混合检索<br/>（稀疏+稠密）？"}
    Q4 -- 要 --> A4["Milvus 2.4<br/>原生 sparse vector"]
    Q4 -- 不要 --> A5["Qdrant 集群"]

    Q1 -- "> 1 亿" --> A6["Milvus 分布式<br/>或 Milvus + DiskANN"]

    Q1 -- "已有 ES 且全文检索是核心" --> A7["Elasticsearch 8.x<br/>dense_vector + BM25 一体"]

    style A0 fill:#fff0d4
    style A1 fill:#e8f8e8
    style A4 fill:#d4f4dd
```

**本书的明确选择：**

| 用途 | 选型 | 理由 |
|---|---|---|
| 第 2.5 章教学示例 | **Chroma** | 零部署，读者跟着敲就能跑 |
| 生产主线（华成机电 82 万 chunk） | **Milvus 2.4** | 混合检索原生支持、分区隔离、过滤能力强 |
| 对照与轻量方案 | **Qdrant** / **pgvector** | 给读者一个运维更轻的选项 |

---

## 三、动手实战

### 3.1 本地部署 Embedding 服务

#### 方式一：sentence-transformers 直接用（最简单）

```python
"""embed_st.py —— sentence-transformers 直接使用，含 batch 与显存优化。"""
from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """本地 embedding 服务封装：自动 batch、自动截断、显存可控。"""

    # 各模型的 query 指令前缀（bge-m3 不需要前缀）
    QUERY_PREFIX = {
        "BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
        "BAAI/bge-small-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
        "BAAI/bge-base-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
        "BAAI/bge-m3": "",
    }

    def __init__(self, model_name: str = "BAAI/bge-m3",
                 device: str | None = None,
                 max_seq_length: int | None = None,
                 fp16: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        # FP16 能省一半显存，精度损失对检索几乎无影响
        if fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()
        self.prefix = self.QUERY_PREFIX.get(model_name, "")

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """编码文档侧：不加前缀，归一化。"""
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(texts) > 500,
        )

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """编码查询侧：加指令前缀，归一化。"""
        prefixed = [self.prefix + q for q in queries] if self.prefix else queries
        return self.model.encode(
            prefixed, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def encode_stream(self, texts: list[str], batch_size: int = 64,
                      chunk: int = 10000):
        """大批量流式编码：避免一次性把几十万条结果堆在内存里。"""
        for i in range(0, len(texts), chunk):
            sub = texts[i:i + chunk]
            yield i, self.encode_passages(sub, batch_size=batch_size)


def find_best_batch_size(svc: EmbeddingService, sample_text: str,
                         candidates=(16, 32, 64, 128, 256)) -> int:
    """自动探测不 OOM 的最大 batch_size。生产部署前跑一次。"""
    import time
    best, best_tps = candidates[0], 0.0
    for bs in candidates:
        try:
            texts = [sample_text] * bs * 4
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t0 = time.time()
            svc.encode_passages(texts, batch_size=bs)
            elapsed = time.time() - t0
            tps = len(texts) / elapsed
            mem = (torch.cuda.max_memory_allocated() / 1e9
                   if torch.cuda.is_available() else 0)
            print(f"  batch_size={bs:>4}  {tps:>7.1f} 条/秒  峰值显存 {mem:.2f} GB")
            if tps > best_tps:
                best, best_tps = bs, tps
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch_size={bs:>4}  OOM，停止探测")
                break
            raise
    return best


if __name__ == "__main__":
    svc = EmbeddingService("BAAI/bge-large-zh-v1.5", max_seq_length=512)
    print(f"模型={svc.model_name} 设备={svc.device} 维度={svc.dim}")

    docs = ["E043：X 轴伺服驱动器过热保护，需清理电控柜滤网并检查散热风扇。",
            "XJ-200 整机保修期为 12 个月，自验收之日起算。"]
    dv = svc.encode_passages(docs)
    qv = svc.encode_queries(["伺服驱动器过热怎么办"])
    print("文档向量 shape：", dv.shape, " 模长：", np.round(np.linalg.norm(dv, axis=1), 4))
    print("相似度：", np.round(qv @ dv.T, 4))

    print("\n--- batch_size 探测 ---")
    best = find_best_batch_size(svc, "E043 是 X 轴伺服驱动器过热保护报警。" * 8)
    print(f"推荐 batch_size = {best}")
```

**预期输出：**

```text
模型=BAAI/bge-large-zh-v1.5 设备=cuda 维度=1024
文档向量 shape： (2, 1024)  模长： [1. 1.]
相似度： [[0.6874 0.2431]]

--- batch_size 探测 ---
  batch_size=  16    412.3 条/秒  峰值显存 1.84 GB
  batch_size=  32    701.5 条/秒  峰值显存 2.31 GB
  batch_size=  64   1103.8 条/秒  峰值显存 3.27 GB
  batch_size= 128   1342.6 条/秒  峰值显存 5.12 GB
  batch_size= 256   1398.2 条/秒  峰值显存 8.94 GB
推荐 batch_size = 256
```

> 实测环境：RTX 4090 24GB / bge-large-zh-v1.5 FP16 / 平均 400 字的中文 chunk。可以看到 **batch_size 从 64 到 256 的收益递减明显（+27%）而显存翻了近 3 倍**，生产上选 128 是更好的平衡。

#### 方式二：FlagEmbedding（bge-m3 三合一的官方方式）

只有用 FlagEmbedding 才能拿到 bge-m3 的稀疏权重和 ColBERT 向量：

```python
"""embed_flag.py —— 用 FlagEmbedding 获取 bge-m3 的三种表示。"""
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

docs = [
    "E043：X 轴伺服驱动器过热保护，成因为散热风扇故障或滤网堵塞，需更换 SV-200X。",
    "E057：刀库换刀超时，检查气源压力是否达到 0.6MPa。",
]
queries = ["SV-200X 什么时候需要更换"]

# 一次前向同时产出三种表示
d_out = model.encode(docs, batch_size=16, max_length=1024,
                     return_dense=True, return_sparse=True, return_colbert_vecs=True)
q_out = model.encode(queries, batch_size=1, max_length=256,
                     return_dense=True, return_sparse=True, return_colbert_vecs=True)

print("dense shape:", d_out["dense_vecs"].shape)
print("sparse 非零项数:", [len(w) for w in d_out["lexical_weights"]])
print("colbert shape:", [v.shape for v in d_out["colbert_vecs"]])

# 三种相似度分别计算
dense_score = q_out["dense_vecs"] @ d_out["dense_vecs"].T
sparse_score = [model.compute_lexical_matching_score(
    q_out["lexical_weights"][0], w) for w in d_out["lexical_weights"]]
colbert_score = [model.colbert_score(q_out["colbert_vecs"][0], v)
                 for v in d_out["colbert_vecs"]]

print("\ndense  :", [round(float(s), 4) for s in dense_score[0]])
print("sparse :", [round(float(s), 4) for s in sparse_score])
print("colbert:", [round(float(s), 4) for s in colbert_score])

# 加权融合（权重需在自己的评测集上调）
W = {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}
fused = [W["dense"] * float(dense_score[0][i])
         + W["sparse"] * float(sparse_score[i])
         + W["colbert"] * float(colbert_score[i]) for i in range(len(docs))]
print("fused  :", [round(s, 4) for s in fused])

# 查看稀疏权重具体是哪些 token（很直观，方便调试）
tokens = model.convert_id_to_token(d_out["lexical_weights"])
print("\n文档 0 权重最高的 token：",
      sorted(tokens[0].items(), key=lambda kv: -kv[1])[:8])
```

**预期输出：**

```text
dense shape: (2, 1024)
sparse 非零项数: [31, 19]
colbert shape: [(48, 1024), (26, 1024)]

dense  : [0.6217, 0.3104]
sparse : [0.4183, 0.0000]
colbert: [0.7402, 0.4128]
fused  : [0.6304, 0.2892]

文档 0 权重最高的 token： [('SV', 0.2841), ('200', 0.2317), ('X', 0.2104), ('伺服', 0.1893), ('驱动器', 0.1772), ('过热', 0.1564), ('E', 0.1402), ('043', 0.1338)]
```

> 🔑 **注意 sparse 分数**：文档 1（刀库超时）的 sparse 分数是 **0.0000**，因为它和 query "SV-200X" 没有任何词项重合。而 dense 分数还有 0.31（语义上都是"设备故障"）。**这正是混合检索的价值：sparse 提供了 dense 缺失的"精确性"信号**。第 2.4 章会把这件事讲透。

#### 方式三：Infinity（高吞吐 embedding 服务）

生产上不应该让每个应用进程都加载一份模型。用独立服务：

```bash
# 启动 Infinity（OpenAI 兼容接口）
docker run -d --name infinity --gpus all -p 7997:7997 \
  michaelf34/infinity:0.0.70 \
  v2 --model-id BAAI/bge-m3 \
     --port 7997 \
     --batch-size 64 \
     --dtype float16 \
     --engine torch

# 验证
curl -s http://localhost:7997/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["E043 是什么故障"]}' | head -c 300
```

```python
"""调用 Infinity 服务（OpenAI 兼容）。"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7997", api_key="dummy")

resp = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["E043：X 轴伺服驱动器过热保护", "XJ-200 保修期 12 个月"],
)
vecs = [d.embedding for d in resp.data]
print(f"返回 {len(vecs)} 个向量，维度 {len(vecs[0])}")
```

#### 方式四：TEI（Text Embeddings Inference，HuggingFace 官方）

```bash
# TEI：Rust 实现，吞吐最高
docker run -d --name tei --gpus all -p 8080:80 \
  -v "$PWD/models:/data" \
  ghcr.io/huggingface/text-embeddings-inference:1.5 \
  --model-id BAAI/bge-m3 \
  --max-batch-tokens 16384 \
  --max-concurrent-requests 512 \
  --auto-truncate

# 调用
curl -s http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":["E043 是什么故障"]}' | head -c 200
```

#### 四种部署方式对比

| 方式 | 吞吐（RTX 4090, bge-m3） | 部署复杂度 | 支持稀疏 | 多模型 | 适用 |
|---|---|---|---|---|---|
| sentence-transformers 进程内 | 约 600 条/秒 | 极低 | ✗ | — | 离线批处理、单机脚本 |
| FlagEmbedding 进程内 | 约 550 条/秒 | 低 | **✓** | — | **需要 bge-m3 稀疏向量时的唯一选择** |
| Infinity 服务 | 约 1800 条/秒 | 中 | ✗ | ✓ | 在线服务，多模型共存 |
| TEI 服务 | **约 2400 条/秒** | 中 | ✗ | ✗（一个容器一个模型） | **在线服务吞吐最优** |

> 实测环境：RTX 4090 / bge-m3 FP16 / 平均 300 token 的中文文本 / 并发 32。数字仅供量级参考。

> 💡 **架构建议**：**离线建索引用 FlagEmbedding（要稀疏向量）+ 多进程；在线查询用 TEI 或 Infinity（低延迟高并发）**。注意两者必须是同一个模型同一个版本，否则向量空间不一致。

---

### 3.2 Milvus 2.4 完整实战

#### 3.2.1 用 Docker Compose 启动

Milvus standalone 需要三个组件：etcd（元数据）、MinIO（对象存储）、milvus 本体。

```yaml
# docker-compose.milvus.yml
version: "3.8"

services:
  etcd:
    container_name: huacheng-etcd
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./volumes/etcd:/etcd
    command: >
      etcd -advertise-client-urls=http://127.0.0.1:2379
           -listen-client-urls http://0.0.0.0:2379
           --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: huacheng-minio
    image: minio/minio:RELEASE.2024-05-28T17-19-04Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: huacheng-milvus
    image: milvusdb/milvus:v2.4.15
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      # 单机内存上限，按机器调整
      COMMON_STORAGETYPE: local
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"   # gRPC
      - "9091:9091"     # metrics / health
    depends_on:
      - etcd
      - minio
    deploy:
      resources:
        limits:
          memory: 16G

  # 可选：可视化管理界面
  attu:
    container_name: huacheng-attu
    image: zilliz/attu:v2.4
    environment:
      MILVUS_URL: standalone:19530
    ports:
      - "8000:3000"
    depends_on:
      - standalone

networks:
  default:
    name: huacheng-milvus
```

```bash
# 启动
docker compose -f docker-compose.milvus.yml up -d

# 等待健康
until curl -sf http://localhost:9091/healthz > /dev/null; do
  echo "等待 Milvus 启动..."; sleep 5
done
echo "Milvus 就绪。Attu 控制台：http://localhost:8000"

# 安装 SDK
pip install "pymilvus==2.4.9"
```

#### 3.2.2 建 Collection（含标量字段）

**关键点：标量字段必须在建表时定义**，后面加字段要重建整个 collection。

```python
"""milvus_setup.py —— 建 Collection、建索引、加载。"""
from __future__ import annotations

from pymilvus import (
    MilvusClient, DataType, connections, Collection, CollectionSchema,
    FieldSchema, utility,
)

URI = "http://localhost:19530"
COLLECTION = "huacheng_kb_bgem3_v1"       # 命名含模型名和版本，换模型时不会撞车
DIM = 1024                                 # bge-m3 稠密维度


def create_collection(client: MilvusClient, drop_existing: bool = False) -> None:
    """创建 Collection，定义稠密向量 + 稀疏向量 + 全部标量过滤字段。"""
    if client.has_collection(COLLECTION):
        if not drop_existing:
            print(f"Collection {COLLECTION} 已存在，跳过创建")
            return
        client.drop_collection(COLLECTION)
        print(f"已删除旧 Collection {COLLECTION}")

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

    # ---- 主键 ----
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)

    # ---- 向量字段 ----
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)   # bge-m3 稀疏，2.4 原生支持

    # ---- 正文（存原文，检索后直接返回，不用回源） ----
    schema.add_field("text", DataType.VARCHAR, max_length=8192)

    # ---- 标量过滤字段：对应第 2.2 章的元数据 schema ----
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_title", DataType.VARCHAR, max_length=512)
    schema.add_field("source_file", DataType.VARCHAR, max_length=1024)
    schema.add_field("section_path", DataType.VARCHAR, max_length=512)
    schema.add_field("page", DataType.INT32)
    # ARRAY 类型支持 ARRAY_CONTAINS 过滤，比逗号串好用得多
    schema.add_field("device_model", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=8, max_length=32)
    schema.add_field("error_codes", DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=16, max_length=16)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
    schema.add_field("confidentiality", DataType.VARCHAR, max_length=16)
    schema.add_field("doc_version", DataType.VARCHAR, max_length=32)
    schema.add_field("is_latest", DataType.BOOL)
    # 日期存 INT32（yyyymmdd），比字符串比较快得多
    schema.add_field("effective_date", DataType.INT32)
    schema.add_field("expire_date", DataType.INT32)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=256)
    schema.add_field("ocr_confidence", DataType.FLOAT)
    schema.add_field("pipeline_version", DataType.VARCHAR, max_length=16)

    client.create_collection(collection_name=COLLECTION, schema=schema,
                             consistency_level="Bounded")
    print(f"Collection {COLLECTION} 创建完成")


def create_indexes(client: MilvusClient) -> None:
    """建向量索引 + 标量索引。标量索引对过滤性能影响巨大。"""
    idx = client.prepare_index_params()

    # 稠密向量：HNSW + 内积（向量已归一化）
    idx.add_index(field_name="dense", index_type="HNSW", metric_type="IP",
                  params={"M": 16, "efConstruction": 200})

    # 稀疏向量：Milvus 2.4 专用索引
    idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                  metric_type="IP", params={"drop_ratio_build": 0.2})

    # 标量索引：高基数字段用 INVERTED，布尔/低基数用 BITMAP
    for f in ["doc_id", "doc_type", "device_model", "error_codes",
              "confidentiality", "content_type", "parent_id"]:
        idx.add_index(field_name=f, index_type="INVERTED")
    idx.add_index(field_name="is_latest", index_type="BITMAP")
    idx.add_index(field_name="effective_date", index_type="STL_SORT")
    idx.add_index(field_name="expire_date", index_type="STL_SORT")

    client.create_index(COLLECTION, idx)
    print("索引创建完成")


def load_collection(client: MilvusClient) -> None:
    """把 collection 加载进内存。不 load 是查不了的（新手最常见的报错）。"""
    client.load_collection(COLLECTION)
    print("加载状态：", client.get_load_state(COLLECTION))


if __name__ == "__main__":
    client = MilvusClient(uri=URI)
    create_collection(client, drop_existing=True)
    create_indexes(client)
    load_collection(client)
    print("\n当前所有 Collection：", client.list_collections())
    print("\nSchema：")
    desc = client.describe_collection(COLLECTION)
    for f in desc["fields"]:
        print(f"  {f['name']:<20} {f['type']}")
```

**预期输出：**

```text
已删除旧 Collection huacheng_kb_bgem3_v1
Collection huacheng_kb_bgem3_v1 创建完成
索引创建完成
加载状态： {'state': <LoadState: Loaded>}

当前所有 Collection： ['huacheng_kb_bgem3_v1']

Schema：
  chunk_id             DataType.VARCHAR
  dense                DataType.FLOAT_VECTOR
  sparse               DataType.SPARSE_FLOAT_VECTOR
  text                 DataType.VARCHAR
  doc_id               DataType.VARCHAR
  ...
```

> ⚠️ **三个容易踩的坑**：
> 1. **不 `load_collection` 就查询** → 报 `collection not loaded`。Milvus 的数据在磁盘，必须显式加载进内存。
> 2. **`enable_dynamic_field=True` 很方便但别滥用**——动态字段不能建索引，用它做过滤会全表扫描。
> 3. **日期一定存成 INT32（yyyymmdd）而不是 VARCHAR**。`effective_date <= 20260917` 这种范围过滤在 INT 上用 STL_SORT 索引很快，在 VARCHAR 上是字符串比较。

#### 3.2.3 插入数据（含批量优化）

```python
"""milvus_insert.py —— 批量插入，含性能优化。"""
from __future__ import annotations

import time
from typing import Iterator

from pymilvus import MilvusClient
from FlagEmbedding import BGEM3FlagModel

from milvus_setup import URI, COLLECTION


def to_milvus_sparse(lexical_weights: dict) -> dict:
    """FlagEmbedding 的 lexical_weights 转 Milvus 稀疏向量格式 {index: value}。"""
    return {int(k): float(v) for k, v in lexical_weights.items() if float(v) > 0}


def date_to_int(s: str | None) -> int:
    """'2024-06-01' -> 20240601；None -> 0（表示无限制）。"""
    if not s:
        return 0
    return int(s.replace("-", ""))


def build_rows(chunks: list[dict], model: BGEM3FlagModel,
               batch_size: int = 32) -> list[dict]:
    """把 chunk 编码并组装成 Milvus 行。chunks 形如 {'text':..., 'metadata':{...}}。"""
    texts = [c["text"] for c in chunks]
    out = model.encode(texts, batch_size=batch_size, max_length=1024,
                       return_dense=True, return_sparse=True,
                       return_colbert_vecs=False)
    rows = []
    for i, c in enumerate(chunks):
        m = c["metadata"]
        rows.append({
            "chunk_id": m["chunk_id"],
            "dense": out["dense_vecs"][i].tolist(),
            "sparse": to_milvus_sparse(out["lexical_weights"][i]),
            "text": c["text"][:8000],
            "doc_id": m.get("doc_id", ""),
            "doc_title": m.get("doc_title", ""),
            "source_file": m.get("source_file", ""),
            "section_path": m.get("section_path", ""),
            "page": int(m.get("page") or 0),
            "device_model": m.get("device_model", []) or [],
            "error_codes": m.get("error_codes", []) or [],
            "doc_type": m.get("doc_type", "manual"),
            "confidentiality": m.get("confidentiality", "internal"),
            "doc_version": m.get("doc_version", ""),
            "is_latest": bool(m.get("is_latest", True)),
            "effective_date": date_to_int(m.get("effective_date")),
            "expire_date": date_to_int(m.get("expire_date")) or 99991231,
            "content_type": m.get("content_type", "text"),
            "parent_id": m.get("parent_id", ""),
            "ocr_confidence": float(m.get("ocr_confidence") or 1.0),
            "pipeline_version": m.get("pipeline_version", "v1"),
        })
    return rows


def bulk_insert(client: MilvusClient, chunks: list[dict],
                model: BGEM3FlagModel, insert_batch: int = 500) -> None:
    """分批插入。批太小 RPC 开销大，批太大容易超 gRPC 消息上限（默认 64MB）。"""
    t0 = time.time()
    total = 0
    for i in range(0, len(chunks), insert_batch):
        sub = chunks[i:i + insert_batch]
        rows = build_rows(sub, model)
        client.insert(collection_name=COLLECTION, data=rows)
        total += len(rows)
        if (i // insert_batch) % 10 == 0:
            el = time.time() - t0
            print(f"  已插入 {total}/{len(chunks)}  "
                  f"{total/max(el,1e-6):.0f} 条/秒  耗时 {el:.0f}s")
    # flush 把内存中的数据持久化并触发索引构建
    client.flush(COLLECTION)
    print(f"插入完成：{total} 条，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import json
    from pathlib import Path

    client = MilvusClient(uri=URI)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    chunks = [json.loads(l) for l in
              Path("./data/chunks/huacheng_chunks.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    print(f"待插入 {len(chunks)} 条")
    bulk_insert(client, chunks, model, insert_batch=500)
    print("总行数：", client.get_collection_stats(COLLECTION))
```

**预期输出：**

```text
待插入 823614 条
  已插入 500/823614  186 条/秒  耗时 3s
  已插入 5500/823614  412 条/秒  耗时 13s
  已插入 10500/823614  438 条/秒  耗时 24s
  ...
  已插入 823614/823614  461 条/秒  耗时 1786s
插入完成：823614 条，总耗时 1786s
总行数： {'row_count': 823614}
```

> 实测环境：RTX 4090（embedding）+ Milvus 2.4.15 standalone（32 核 128GB）。**瓶颈在 embedding 而非插入**，所以真正的优化点是多 GPU / 多进程编码。

**批量导入的五条性能建议：**

| 建议 | 原因 | 提升幅度 |
|---|---|---|
| 1. **先插数据再建索引** | 边插边建索引会反复触发段合并 | 2~3 倍 |
| 2. **insert_batch 设 500~2000 行** | 太小 RPC 开销大，太大超 gRPC 64MB 限制 | 3~5 倍（相比逐条插） |
| 3. **embedding 用多进程** | 单进程 GPU 利用率上不去 | 近线性 |
| 4. **超大规模用 `bulk_insert` 导 parquet** | 走 MinIO 旁路，绕过 gRPC | 5~10 倍 |
| 5. **插完统一 `flush()` 一次** | 每批都 flush 会产生大量小 segment | 明显 |

#### 3.2.4 带过滤的检索

```python
"""milvus_search.py —— 向量检索 + 标量过滤 + 混合检索。"""
from __future__ import annotations

import time
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from FlagEmbedding import BGEM3FlagModel

from milvus_setup import URI, COLLECTION

OUTPUT_FIELDS = ["text", "doc_title", "section_path", "page",
                 "device_model", "doc_type", "doc_version", "parent_id"]


def build_filter(device: str | None = None,
                 doc_types: list[str] | None = None,
                 error_code: str | None = None,
                 max_confidentiality: str = "internal",
                 today: int = 20260917) -> str:
    """组装 Milvus 布尔过滤表达式。所有字段都要有标量索引才快。"""
    conds: list[str] = []
    if device:
        conds.append(f'ARRAY_CONTAINS(device_model, "{device}")')
    if error_code:
        conds.append(f'ARRAY_CONTAINS(error_codes, "{error_code}")')
    if doc_types:
        lst = ", ".join(f'"{t}"' for t in doc_types)
        conds.append(f"doc_type in [{lst}]")
    # 密级：按等级映射成可见集合
    level_map = {"public": ["public"],
                 "internal": ["public", "internal"],
                 "restricted": ["public", "internal", "restricted"],
                 "secret": ["public", "internal", "restricted", "secret"]}
    allowed = ", ".join(f'"{v}"' for v in level_map[max_confidentiality])
    conds.append(f"confidentiality in [{allowed}]")
    # 版本与有效期
    conds.append("is_latest == True")
    conds.append(f"effective_date <= {today}")
    conds.append(f"expire_date >= {today}")
    return " and ".join(conds)


def dense_search(client: MilvusClient, model: BGEM3FlagModel, query: str,
                 top_k: int = 10, expr: str | None = None,
                 ef_search: int = 128) -> list[dict]:
    """纯稠密向量检索。"""
    qv = model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"]
    res = client.search(
        collection_name=COLLECTION,
        data=qv.tolist(),
        anns_field="dense",
        limit=top_k,
        filter=expr or "",
        search_params={"metric_type": "IP", "params": {"ef": ef_search}},
        output_fields=OUTPUT_FIELDS,
    )
    return res[0]


def hybrid_search(client: MilvusClient, model: BGEM3FlagModel, query: str,
                  top_k: int = 10, expr: str | None = None,
                  ranker: str = "rrf") -> list[dict]:
    """稠密 + 稀疏混合检索，由 Milvus 内部完成融合。"""
    out = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = out["dense_vecs"].tolist()
    sparse_vec = [{int(k): float(v) for k, v in out["lexical_weights"][0].items()}]

    req_dense = AnnSearchRequest(
        data=dense_vec, anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k * 5, expr=expr or "",
    )
    req_sparse = AnnSearchRequest(
        data=sparse_vec, anns_field="sparse",
        param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
        limit=top_k * 5, expr=expr or "",
    )
    rank = RRFRanker(k=60) if ranker == "rrf" else WeightedRanker(0.7, 0.3)
    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[req_dense, req_sparse],
        ranker=rank, limit=top_k, output_fields=OUTPUT_FIELDS,
    )
    return res[0]


def show(title: str, hits: list[dict], elapsed_ms: float) -> None:
    """打印检索结果。"""
    print(f"\n===== {title}（{elapsed_ms:.1f} ms，{len(hits)} 条）=====")
    for i, h in enumerate(hits[:5], 1):
        e = h["entity"]
        print(f"[{i}] score={h['distance']:.4f} | {e['doc_title']} p{e['page']} "
              f"| {e['section_path'][:30]}")
        print(f"    {e['text'][:80]}...")


if __name__ == "__main__":
    client = MilvusClient(uri=URI)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    q = "XJ-200 报 E043 应该怎么处理"

    t0 = time.perf_counter()
    r1 = dense_search(client, model, q, top_k=10)
    show("纯稠密，无过滤", r1, (time.perf_counter() - t0) * 1000)

    expr = build_filter(device="XJ-200", doc_types=["manual", "faq"],
                        max_confidentiality="internal")
    print(f"\n过滤表达式：{expr}")
    t0 = time.perf_counter()
    r2 = dense_search(client, model, q, top_k=10, expr=expr)
    show("稠密 + 标量过滤", r2, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    r3 = hybrid_search(client, model, q, top_k=10, expr=expr)
    show("混合检索（稠密+稀疏，RRF 融合）", r3, (time.perf_counter() - t0) * 1000)
```

**预期输出：**

```text
===== 纯稠密，无过滤（18.4 ms，10 条）=====
[1] score=0.7213 | XJ-200 数控车床维修手册 p47 | 第4章 > 4.3 伺服故障 > 4.3.1 E043
    E043：X 轴伺服驱动器过热保护。当驱动器内部温度传感器检测到温度超过 85℃ 时...
[2] score=0.6884 | XJ-300 数控车床维修手册 p52 | 第5章 > 5.2 伺服系统
    E043 报警在 XJ-300 上表示 Z 轴伺服过热，处理方式与 X 轴类似...
...

过滤表达式：ARRAY_CONTAINS(device_model, "XJ-200") and doc_type in ["manual", "faq"] and confidentiality in ["public", "internal"] and is_latest == True and effective_date <= 20260917 and expire_date >= 20260917

===== 稠密 + 标量过滤（24.7 ms，10 条）=====
[1] score=0.7213 | XJ-200 数控车床维修手册 p47 | 第4章 > 4.3 伺服故障 > 4.3.1 E043
    E043：X 轴伺服驱动器过热保护。当驱动器内部温度传感器检测到温度超过 85℃ 时...
[2] score=0.6512 | XJ-200 常见故障速查手册 p12 | 第2章 > 电气类故障
    E043 报警速查：优先检查电控柜滤网，其次检查驱动器散热风扇...
...

===== 混合检索（稠密+稀疏，RRF 融合）（41.2 ms，10 条）=====
[1] score=0.0328 | XJ-200 数控车床维修手册 p47 | 第4章 > 4.3 伺服故障 > 4.3.1 E043
[2] score=0.0317 | XJ-200 常见故障速查手册 p12 | 第2章 > 电气类故障
[3] score=0.0161 | XJ-200 备件更换指南 p8 | 第3章 > 伺服驱动器更换
```

> 🔑 **注意第一组和第二组的差异**：不加过滤时，**XJ-300 的手册排到了第 2 位**（E043 在 XJ-300 上含义不同！）。这就是第 2.2 章强调 `device_model` 元数据的原因。**一个过滤条件解决的问题，靠调 embedding 是解决不了的。**

#### 3.2.5 分区（Partition）：按设备型号物理隔离

分区是 Milvus 的**物理隔离**机制。查询时指定分区，就完全不扫描其他分区的数据。

```mermaid
flowchart TB
    COL["Collection: huacheng_kb"] --> P1["Partition: XJ-100<br/>12 万 chunk"]
    COL --> P2["Partition: XJ-200<br/>38 万 chunk"]
    COL --> P3["Partition: XJ-300<br/>21 万 chunk"]
    COL --> P4["Partition: common<br/>11 万 chunk<br/>（保修政策等通用文档）"]

    Q(["查询：XJ-200 的 E043"]) --> R["指定 partition_names=['XJ-200','common']"]
    R -.->|"只扫 49 万"| P2
    R -.-> P4
    R -.x|"完全不扫"| P1
    R -.x P3

    style P2 fill:#d4f4dd
    style P4 fill:#d4f4dd
    style P1 fill:#f0f0f0
    style P3 fill:#f0f0f0
```

```python
"""milvus_partition.py —— 分区管理与分区检索。"""
from pymilvus import MilvusClient
from milvus_setup import URI, COLLECTION

client = MilvusClient(uri=URI)

# 1) 创建分区：按设备大类切
PARTITIONS = ["XJ_100", "XJ_200", "XJ_300", "common"]
for p in PARTITIONS:
    if not client.has_partition(COLLECTION, p):
        client.create_partition(COLLECTION, p)
print("分区列表：", client.list_partitions(COLLECTION))

# 2) 插入时指定分区
def partition_of(device_models: list[str]) -> str:
    """根据 chunk 的设备型号决定落哪个分区。"""
    if not device_models:
        return "common"
    m = device_models[0]
    if m.startswith("XJ-100"):
        return "XJ_100"
    if m.startswith("XJ-200"):
        return "XJ_200"
    if m.startswith("XJ-300"):
        return "XJ_300"
    return "common"

# client.insert(collection_name=COLLECTION, data=rows, partition_name="XJ_200")

# 3) 只加载需要的分区，省内存
client.load_partitions(COLLECTION, ["XJ_200", "common"])

# 4) 分区内检索
def search_in_partitions(qv, partitions: list[str], top_k: int = 10):
    """在指定分区内检索，完全不扫描其他分区。"""
    return client.search(
        collection_name=COLLECTION,
        data=qv, anns_field="dense", limit=top_k,
        partition_names=partitions,
        search_params={"metric_type": "IP", "params": {"ef": 128}},
        output_fields=["text", "doc_title", "page"],
    )
```

**分区 vs 标量过滤，什么时候用哪个？**

| | 分区（Partition） | 标量过滤（filter expr） |
|---|---|---|
| 隔离级别 | **物理隔离**，完全不扫描 | 逻辑过滤，仍在同一索引内 |
| 性能 | **最快**，直接缩小搜索空间 | 有过滤开销 |
| 灵活性 | 低，一个 chunk 只能属于一个分区 | **高，任意组合条件** |
| 数量上限 | Milvus 建议 ≤ 1024 个分区 | 无限制 |
| 适合 | **基数低、互斥、必查条件**（设备大类、租户 ID） | 高基数、可选、组合条件 |

> 💡 **华成机电的做法**：用分区隔离**设备大类**（4 个分区，几乎每个 query 都带设备条件），用标量过滤处理**文档类型、密级、版本、日期**（组合多变）。这是一个典型的"粗粒度物理隔离 + 细粒度逻辑过滤"的组合。
>
> ⚠️ **不要按用户 ID 建分区**——几万个用户就是几万个分区，远超 1024 上限。多租户隔离应该用**分区键（Partition Key）**特性：建表时指定一个字段为 partition key，Milvus 自动做哈希分区。

#### 3.2.6 一致性级别

Milvus 支持四种一致性级别，直接影响"刚写入的数据能不能马上查到"和查询延迟：

| 级别 | 含义 | 写后可读延迟 | 查询延迟 | 适用 |
|---|---|---|---|---|
| `Strong` | 强一致，保证读到所有已写入数据 | 立即 | **最高**（要等同步） | 写后立即校验的测试 |
| `Bounded`（默认） | 有界过期，容忍一定延迟（默认几秒） | 秒级 | 低 | **RAG 场景推荐** |
| `Session` | 会话一致，自己写的自己能立刻读到 | 本会话立即 | 中 | 用户刚上传文档就要能搜到 |
| `Eventually` | 最终一致，不等任何同步 | 不保证 | **最低** | 离线分析、对新鲜度无要求 |

```python
# 建表时设默认级别
client.create_collection(..., consistency_level="Bounded")

# 也可以在单次查询时覆盖
client.search(..., consistency_level="Strong")
```

> 🔑 **RAG 场景用 `Bounded` 就够了**。知识库文档不是秒级更新的，没必要为强一致付延迟代价。但**写增量索引脚本时的验证步骤要用 `Strong`**，否则你刚插完查不到，会误以为插入失败。

---

### 3.3 pgvector 实战

如果你的团队已经在用 PostgreSQL，且 chunk 规模在百万以内，**pgvector 是最省心的选择**：零新增运维、SQL 过滤能力最强、事务和备份都是现成的。

```yaml
# docker-compose.pgvector.yml
version: "3.8"
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: huacheng-pg
    environment:
      POSTGRES_USER: huacheng
      POSTGRES_PASSWORD: huacheng_pwd
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - ./volumes/pg:/var/lib/postgresql/data
    command: >
      postgres
      -c shared_buffers=4GB
      -c maintenance_work_mem=2GB
      -c max_parallel_maintenance_workers=4
      -c work_mem=64MB
```

```sql
-- pgvector_setup.sql —— 建表、建索引

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- 用于中文模糊匹配兜底

DROP TABLE IF EXISTS kb_chunks;

CREATE TABLE kb_chunks (
    chunk_id         TEXT PRIMARY KEY,
    doc_id           TEXT        NOT NULL,
    doc_title        TEXT        NOT NULL,
    source_file      TEXT,
    section_path     TEXT,
    page             INTEGER,
    text             TEXT        NOT NULL,
    -- 向量字段：bge-m3 是 1024 维
    embedding        vector(1024) NOT NULL,
    -- 过滤字段
    device_model     TEXT[]      DEFAULT '{}',
    error_codes      TEXT[]      DEFAULT '{}',
    doc_type         TEXT        DEFAULT 'manual',
    confidentiality  TEXT        DEFAULT 'internal',
    doc_version      TEXT,
    is_latest        BOOLEAN     DEFAULT TRUE,
    effective_date   DATE,
    expire_date      DATE,
    content_type     TEXT        DEFAULT 'text',
    parent_id        TEXT,
    ocr_confidence   REAL        DEFAULT 1.0,
    pipeline_version TEXT        DEFAULT 'v1',
    indexed_at       TIMESTAMPTZ DEFAULT now()
);

-- 向量索引：HNSW，内积度量（向量已归一化）
-- 注意：m 和 ef_construction 对应 Milvus 的 M 和 efConstruction
CREATE INDEX idx_kb_embedding ON kb_chunks
    USING hnsw (embedding vector_ip_ops)
    WITH (m = 16, ef_construction = 200);

-- 标量索引：数组字段用 GIN
CREATE INDEX idx_kb_device   ON kb_chunks USING GIN (device_model);
CREATE INDEX idx_kb_errcode  ON kb_chunks USING GIN (error_codes);
CREATE INDEX idx_kb_doctype  ON kb_chunks (doc_type);
CREATE INDEX idx_kb_conf     ON kb_chunks (confidentiality);
CREATE INDEX idx_kb_latest   ON kb_chunks (is_latest) WHERE is_latest = TRUE;
CREATE INDEX idx_kb_dates    ON kb_chunks (effective_date, expire_date);
CREATE INDEX idx_kb_parent   ON kb_chunks (parent_id);
-- 中文全文兜底：trigram 索引支持 LIKE '%E043%' 这种精确串匹配
CREATE INDEX idx_kb_text_trgm ON kb_chunks USING GIN (text gin_trgm_ops);

ANALYZE kb_chunks;
```

```python
"""pgvector_client.py —— pgvector 的插入与混合过滤查询。"""
from __future__ import annotations

import json
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://huacheng:huacheng_pwd@localhost:5432/ragdb"


def get_conn():
    """建立连接并注册 vector 类型。"""
    conn = psycopg.connect(DSN)
    register_vector(conn)
    return conn


def insert_chunks(rows: list[dict], batch: int = 1000) -> None:
    """批量插入。用 COPY 比 INSERT 快一个数量级。"""
    with get_conn() as conn, conn.cursor() as cur:
        with cur.copy(
            "COPY kb_chunks (chunk_id, doc_id, doc_title, source_file, section_path, "
            "page, text, embedding, device_model, error_codes, doc_type, "
            "confidentiality, doc_version, is_latest, effective_date, expire_date, "
            "content_type, parent_id, ocr_confidence, pipeline_version) FROM STDIN"
        ) as copy:
            for r in rows:
                copy.write_row((
                    r["chunk_id"], r["doc_id"], r["doc_title"], r.get("source_file"),
                    r.get("section_path"), r.get("page"), r["text"],
                    np.asarray(r["embedding"], dtype=np.float32),
                    r.get("device_model", []), r.get("error_codes", []),
                    r.get("doc_type", "manual"), r.get("confidentiality", "internal"),
                    r.get("doc_version"), r.get("is_latest", True),
                    r.get("effective_date"), r.get("expire_date"),
                    r.get("content_type", "text"), r.get("parent_id"),
                    r.get("ocr_confidence", 1.0), r.get("pipeline_version", "v1"),
                ))
        conn.commit()


SEARCH_SQL = """
SET LOCAL hnsw.ef_search = %(ef_search)s;

SELECT chunk_id, doc_title, section_path, page, text,
       -- <#> 是负内积，取负号还原为相似度
       -(embedding <#> %(qv)s::vector) AS score
FROM kb_chunks
WHERE is_latest = TRUE
  AND (%(device)s::text IS NULL OR device_model @> ARRAY[%(device)s]::text[])
  AND (%(doc_types)s::text[] IS NULL OR doc_type = ANY(%(doc_types)s))
  AND confidentiality = ANY(%(allowed)s)
  AND (effective_date IS NULL OR effective_date <= CURRENT_DATE)
  AND (expire_date   IS NULL OR expire_date   >= CURRENT_DATE)
ORDER BY embedding <#> %(qv)s::vector
LIMIT %(top_k)s;
"""


def search(qv: np.ndarray, device: str | None = None,
           doc_types: list[str] | None = None,
           allowed=("public", "internal"),
           top_k: int = 10, ef_search: int = 128) -> list[dict]:
    """带过滤的向量检索。"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, {
            "qv": np.asarray(qv, dtype=np.float32),
            "device": device,
            "doc_types": doc_types,
            "allowed": list(allowed),
            "top_k": top_k,
            "ef_search": ef_search,
        })
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


HYBRID_SQL = """
-- pgvector + trigram 混合：向量语义 + 精确串匹配，用 RRF 融合
WITH vec AS (
    SELECT chunk_id, text, doc_title, page,
           ROW_NUMBER() OVER (ORDER BY embedding <#> %(qv)s::vector) AS rk
    FROM kb_chunks
    WHERE is_latest = TRUE
      AND (%(device)s::text IS NULL OR device_model @> ARRAY[%(device)s]::text[])
    ORDER BY embedding <#> %(qv)s::vector
    LIMIT 50
),
kw AS (
    SELECT chunk_id, text, doc_title, page,
           ROW_NUMBER() OVER (ORDER BY similarity(text, %(q)s) DESC) AS rk
    FROM kb_chunks
    WHERE is_latest = TRUE
      AND text %% %(q)s
      AND (%(device)s::text IS NULL OR device_model @> ARRAY[%(device)s]::text[])
    ORDER BY similarity(text, %(q)s) DESC
    LIMIT 50
)
SELECT COALESCE(v.chunk_id, k.chunk_id)  AS chunk_id,
       COALESCE(v.text, k.text)          AS text,
       COALESCE(v.doc_title, k.doc_title) AS doc_title,
       COALESCE(v.page, k.page)          AS page,
       COALESCE(1.0 / (60 + v.rk), 0) + COALESCE(1.0 / (60 + k.rk), 0) AS rrf_score
FROM vec v FULL OUTER JOIN kw k ON v.chunk_id = k.chunk_id
ORDER BY rrf_score DESC
LIMIT %(top_k)s;
"""
```

**Milvus 与 pgvector 的对照：**

| 维度 | Milvus 2.4 | pgvector (PG16) |
|---|---|---|
| 100 万向量建 HNSW 索引 | 约 12 min | 约 25 min |
| 单查询延迟（无过滤，ef=128） | 1.2 ms | 2.8 ms |
| 单查询延迟（带 3 个过滤条件） | 2.4 ms | 4.1 ms |
| 原生稀疏向量 | **✓** | ✗（需 pg_search 或 tsvector） |
| SQL JOIN 业务表 | ✗ | **✓✓ 巨大优势** |
| 事务 | ✗ | **✓** |
| 千万级 | ✓ | 吃力（索引构建慢、内存压力大） |
| 运维 | 需要 etcd + MinIO + 监控 | **已有 PG 就是零新增** |

> 实测环境：同一台 32 核 128GB 机器 / 100 万条 1024 维向量 / Milvus 2.4.15 vs PostgreSQL 16 + pgvector 0.8.0。

### 3.4 前置过滤 vs 后置过滤：一个性能陷阱

这是本章开头"加了过滤慢 40 倍"问题的真正答案。

```mermaid
flowchart TB
    subgraph POST["后置过滤（Post-Filtering）—— 慢"]
        direction TB
        PO1["① 在全量 100 万向量上做 ANN 搜索，取 Top-K"] --> PO2["② 对这 K 个结果应用过滤条件"]
        PO2 --> PO3{"够 K 个吗？"}
        PO3 -- 不够 --> PO4["③ 加大 K 重新搜索<br/>K=10 → 100 → 1000 → 10000"]
        PO4 --> PO1
        PO3 -- 够了 --> PO5["返回"]
    end

    subgraph PRE["前置过滤（Pre-Filtering）—— 快"]
        direction TB
        PR1["① 先用标量索引算出满足条件的 ID 集合<br/>（bitmap）"] --> PR2["② ANN 搜索时只在 bitmap 内的向量上遍历"]
        PR2 --> PR3["③ 一次就拿够 Top-K"]
    end

    style POST fill:#ffe0e0
    style PRE fill:#d4f4dd
```

**为什么后置过滤这么慢？** 假设过滤条件的选择率是 1%（100 万条里只有 1 万条满足）：
- 你要 Top-10，做一次 ANN 取 Top-10 → 过滤后平均只剩 0.1 条；
- 加大到 Top-1000 → 过滤后剩 10 条，刚好够，但你**多算了 100 倍**；
- 如果选择率是 0.1%，要取 Top-10000，**多算 1000 倍**。

#### 实测对比

```python
"""filter_benchmark.py —— 前置过滤 vs 后置过滤的实测对比。"""
import time
import numpy as np
from pymilvus import MilvusClient
from milvus_setup import URI, COLLECTION

client = MilvusClient(uri=URI)


def bench_pre_filter(qv, expr: str, top_k: int = 10, repeat: int = 20) -> float:
    """前置过滤：把 filter 交给 Milvus，由它在 ANN 内部剪枝。"""
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        client.search(collection_name=COLLECTION, data=qv, anns_field="dense",
                      limit=top_k, filter=expr,
                      search_params={"metric_type": "IP", "params": {"ef": 128}},
                      output_fields=["doc_id"])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def bench_post_filter(qv, predicate, top_k: int = 10,
                      over_fetch: int = 100, repeat: int = 20) -> tuple[float, int]:
    """后置过滤：在应用层过滤，不够就加大 K 重来。"""
    ts, final_k = [], 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        k = top_k * over_fetch
        while True:
            res = client.search(collection_name=COLLECTION, data=qv,
                                anns_field="dense", limit=min(k, 16384),
                                search_params={"metric_type": "IP",
                                               "params": {"ef": max(128, k)}},
                                output_fields=["doc_id", "device_model", "doc_type"])
            kept = [h for h in res[0] if predicate(h["entity"])]
            if len(kept) >= top_k or k >= 16384:
                final_k = k
                break
            k *= 4
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), final_k


if __name__ == "__main__":
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    qv = model.encode(["XJ-200 报 E043 怎么处理"],
                      return_dense=True, return_sparse=False)["dense_vecs"].tolist()

    cases = [
        ("无过滤", "", lambda e: True, "100%"),
        ("doc_type=manual", 'doc_type == "manual"',
         lambda e: e.get("doc_type") == "manual", "约 42%"),
        ("device=XJ-200", 'ARRAY_CONTAINS(device_model, "XJ-200")',
         lambda e: "XJ-200" in (e.get("device_model") or []), "约 8%"),
        ("device=XJ-200 且 doc_type=faq",
         'ARRAY_CONTAINS(device_model, "XJ-200") and doc_type == "faq"',
         lambda e: "XJ-200" in (e.get("device_model") or [])
                   and e.get("doc_type") == "faq", "约 0.7%"),
    ]

    print(f"{'场景':<26}{'选择率':<10}{'前置过滤':<12}{'后置过滤':<12}{'倍数'}")
    print("-" * 74)
    for name, expr, pred, sel in cases:
        t_pre = bench_pre_filter(qv, expr)
        t_post, k = bench_post_filter(qv, pred)
        print(f"{name:<26}{sel:<10}{t_pre:>7.1f} ms {t_post:>9.1f} ms "
              f"{t_post/t_pre:>7.1f}x  (最终K={k})")
```

**预期输出：**

```text
场景                        选择率      前置过滤      后置过滤      倍数
--------------------------------------------------------------------------
无过滤                      100%         1.3 ms       1.4 ms      1.1x  (最终K=1000)
doc_type=manual             约 42%       1.9 ms       4.8 ms      2.5x  (最终K=1000)
device=XJ-200               约 8%        2.4 ms      21.6 ms      9.0x  (最终K=4000)
device=XJ-200 且 doc_type=faq 约 0.7%     3.1 ms     148.3 ms     47.8x  (最终K=16000)
```

> 实测环境：Milvus 2.4.15 / 82 万 chunk / HNSW M=16 ef=128 / 单机 32 核 128GB。

**结论与操作要点：**

| 要点 | 说明 |
|---|---|
| 1. **让向量库做过滤，不要在应用层过滤** | Milvus/Qdrant/pgvector 都支持前置过滤，把 filter 传给它们 |
| 2. **过滤字段必须建标量索引** | 没有索引的字段过滤 = 全表扫描，比后置过滤还慢 |
| 3. **选择率越低，前置过滤优势越大** | 选择率 < 5% 时差距在 10 倍以上 |
| 4. **极低选择率（< 0.1%）应该用分区** | 此时连前置过滤都不够，要物理隔离 |
| 5. **Chroma 的过滤是后置的** | 这是它不适合生产的原因之一 |

> ⚠️ **Milvus 的过滤策略其实是自适应的**：它会根据选择率估算，自己决定用 bitmap 前置剪枝还是先 ANN 再过滤。但**前提是过滤字段有标量索引且统计信息准确**。定期跑 `ANALYZE`（pgvector）或确保 Milvus 的 segment 已 compact，是保证它选对策略的关键。

### 3.5 向量库的日常运维

把数据灌进去、把过滤调快，只是把系统送到了上线那一天。**真正消耗人的是上线之后**：文档每周在变、模型每季度想升级、磁盘每天在涨、老板还要求"升级的时候不能停服务"。这一节把向量库运维的五件事一次讲透。

```mermaid
flowchart TB
    subgraph DAILY["每日/每周"]
        A1["文档变更检测"] --> A2["增量 upsert"]
        A2 --> A3["软删除过期版本"]
    end
    subgraph MONTHLY["每月/每季"]
        B1["compact 回收空间"] --> B2["硬删除已过观察期的数据"]
        B2 --> B3["备份 + 恢复演练"]
    end
    subgraph UPGRADE["模型/切分策略升级"]
        C1["建 v2 collection"] --> C2["全量重灌"]
        C2 --> C3["验收门禁"]
        C3 --> C4["alias 秒级切换"]
        C4 --> C5["观察 7 天后删 v1"]
    end
    subgraph WATCH["常态监控"]
        D1["延迟 / QPS / 内存水位"]
        D2["离线召回率日报"]
        D3["空结果率"]
    end

    style UPGRADE fill:#e0e8ff
    style WATCH fill:#fff0d4
```

#### 3.5.1 增量更新 vs 全量重建：先判断"什么变了"

新手最容易犯的错，是**任何改动都全量重建**。82 万 chunk 全量重灌一次要 30 分钟 GPU 时间加上索引构建，一天重建三次，什么活都别干了。反过来，**该重建的时候偷懒做增量**，会留下一个新旧向量混杂、谁也说不清的脏库。

判断标准只有一条：**这次改动会不会让新旧向量落在不同的语义空间、或让新旧 chunk 的边界不一致**。

| 变更类型 | 该做什么 | 原因 | 停机 |
|---|---|---|---|
| 新增文档 | **增量 insert** | 向量空间没变，直接追加 | 否 |
| 少量文档内容修改 | **按 `doc_id` 删除旧 chunk + 插入新 chunk** | chunk 的边界和数量都可能变了，不能逐条 update | 否 |
| 文档下线 / 作废 | **软删除**（`is_latest=False`），观察期后硬删除 | 留回溯能力 | 否 |
| 调整 `chunk_size` / `overlap` | **全量重建** | chunk 边界全变，混着存会出现重叠和空洞 | 否（双索引切换） |
| 换 embedding 模型或升级版本 | **全量重建到新 collection** | 向量空间不可比，混存等于毁库 | 否（双索引切换） |
| 元数据 schema 加字段 | **全量重建** | Milvus 的 schema 定长，加字段必须新建 collection | 否（双索引切换） |
| 索引参数改 `M` / `efConstruction` | **只重建索引，数据不动** | `release → drop_index → create_index → load` | 是（几分钟不可查） |
| 索引参数改 `ef` / `nprobe` | **什么都不用做** | 这是查询期参数，改配置即可 | 否 |
| 只改 LLM / prompt | **什么都不用做** | 跟向量库无关 | 否 |

> 👉 记一条口诀：**"向量变了就重建，元数据变了看字段，查询参数随便改。"**

**增量同步的正确做法：用 manifest 做内容哈希 diff。**

不要靠"文件修改时间"判断变更——`rsync` 一次、`git checkout` 一次，mtime 全变了。要用内容哈希。

```python
"""scripts/incremental_sync.py —— 基于内容哈希 manifest 的增量同步。

流程：扫描源目录 → 算每个文档的内容哈希 → 与上次 manifest 比对
     → 产出 add / update / delete 三个集合 → 分别执行 → 落盘新 manifest
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pymilvus import MilvusClient

from milvus_setup import URI, COLLECTION

MANIFEST_PATH = Path("./data/manifest/huacheng_kb.json")


def file_hash(p: Path) -> str:
    """算文件内容的 SHA256（前 16 位够用），与 mtime 无关。"""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def doc_id_of(p: Path, root: Path) -> str:
    """用相对路径做 doc_id，保证跨机器一致。"""
    return str(p.relative_to(root)).replace("\\", "/")


@dataclass
class SyncPlan:
    """一次增量同步的执行计划。"""
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """人类可读的计划摘要。"""
        return (f"新增 {len(self.added)} 篇 | 更新 {len(self.updated)} 篇 | "
                f"删除 {len(self.deleted)} 篇")

    def is_empty(self) -> bool:
        """是否无事可做。"""
        return not (self.added or self.updated or self.deleted)


def load_manifest() -> dict[str, str]:
    """读取上一次的 manifest：{doc_id: content_hash}。"""
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(m: dict[str, str]) -> None:
    """原子写 manifest：先写临时文件再 rename，避免中途崩溃写坏。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def scan(root: Path, patterns: tuple[str, ...] = ("*.pdf", "*.docx", "*.md", "*.xlsx")) -> dict[str, str]:
    """扫描源目录，返回 {doc_id: content_hash}。"""
    cur: dict[str, str] = {}
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and not p.name.startswith("~$"):   # 跳过 Office 临时文件
                cur[doc_id_of(p, root)] = file_hash(p)
    return cur


def make_plan(prev: dict[str, str], cur: dict[str, str]) -> SyncPlan:
    """比对两份 manifest，产出同步计划。"""
    plan = SyncPlan()
    for doc_id, h in cur.items():
        if doc_id not in prev:
            plan.added.append(doc_id)
        elif prev[doc_id] != h:
            plan.updated.append(doc_id)
    for doc_id in prev:
        if doc_id not in cur:
            plan.deleted.append(doc_id)
    return plan


def delete_doc_chunks(client: MilvusClient, doc_ids: list[str], batch: int = 100) -> int:
    """按 doc_id 批量删除 chunk。表达式里的字符串必须转义引号。"""
    total = 0
    for i in range(0, len(doc_ids), batch):
        sub = doc_ids[i:i + batch]
        quoted = ", ".join('"' + d.replace('"', '\\"') + '"' for d in sub)
        res = client.delete(collection_name=COLLECTION, filter=f"doc_id in [{quoted}]")
        total += int(res.get("delete_count", 0)) if isinstance(res, dict) else 0
    return total


def apply_plan(client: MilvusClient, plan: SyncPlan, root: Path,
               build_chunks_fn, embed_and_insert_fn) -> None:
    """执行同步计划。build_chunks_fn / embed_and_insert_fn 复用第 2.2 章与 3.2.3 的实现。"""
    t0 = time.time()

    # ① 删除：包括被删文档，也包括被更新文档的旧 chunk
    to_remove = plan.deleted + plan.updated
    if to_remove:
        n = delete_doc_chunks(client, to_remove)
        print(f"  已删除 {len(to_remove)} 篇文档的旧 chunk（标记删除 {n} 条）")

    # ② 插入：新增 + 更新
    to_add = plan.added + plan.updated
    if to_add:
        chunks = []
        for doc_id in to_add:
            chunks.extend(build_chunks_fn(root / doc_id))
        print(f"  待写入 {len(chunks)} 个 chunk")
        embed_and_insert_fn(chunks)

    client.flush(COLLECTION)
    print(f"  同步完成，耗时 {time.time() - t0:.1f}s")


def main(source_root: str = "./data/raw") -> None:
    """增量同步主流程。"""
    root = Path(source_root)
    client = MilvusClient(uri=URI)

    prev, cur = load_manifest(), scan(root)
    plan = make_plan(prev, cur)
    print("同步计划：", plan.summary())

    if plan.is_empty():
        print("无变更，退出")
        return

    # 安全阀：一次删除超过 20% 的文档，很可能是源目录挂载失败，必须人工确认
    if prev and len(plan.deleted) > 0.2 * len(prev):
        raise SystemExit(
            f"⛔ 本次将删除 {len(plan.deleted)}/{len(prev)} 篇文档，超过 20% 安全阈值。"
            f"请确认源目录挂载正常后，加 --force 重跑。"
        )

    from chunk_pipeline import build_chunks_fn          # 第 2.2 章的切分流水线
    from milvus_insert import embed_and_insert_fn       # 3.2.3 的写入函数
    apply_plan(client, plan, root, build_chunks_fn, embed_and_insert_fn)
    save_manifest(cur)
    print("manifest 已更新")


if __name__ == "__main__":
    main()
```

**预期输出：**

```text
同步计划： 新增 12 篇 | 更新 3 篇 | 删除 1 篇
  已删除 4 篇文档的旧 chunk（标记删除 617 条）
  待写入 2843 个 chunk
  已插入 500/2843  452 条/秒  耗时 1s
  已插入 2843/2843  468 条/秒  耗时 6s
  同步完成，耗时 9.4s
manifest 已更新
```

> ⚠️ 那个 **20% 安全阀**不是摆设。真实事故：NFS 挂载在凌晨断开，扫描脚本扫到一个空目录，认为"所有文档都被删了"，把生产库清空了。有安全阀就只是报个警。

**关于 `upsert` 的三个事实**（Milvus 2.4）：

1. `client.upsert()` 要求 `auto_id=False`，主键必须自己给——这就是我们用 `chunk_id` 做主键的原因；
2. **upsert 内部是 delete + insert**，不是原地更新，所以它同样会产生墓碑数据；
3. **chunk 级 upsert 往往不成立**。文档改了一个字，重新切分之后 chunk 数量可能从 37 变成 39，`chunk_id` 全部错位。所以正确粒度是**按 `doc_id` 整篇删除 + 整篇重插**，而不是按 chunk upsert。

#### 3.5.2 删除与软删除

Milvus 的 `delete()` **不是立即释放空间**，它写一条墓碑记录，查询时过滤掉，真正的空间回收发生在 `compact` 时。这意味着：

- 频繁删除会让 segment 里堆满墓碑，**查询时要扫更多无效数据，延迟慢慢上涨**；
- 磁盘/内存占用不会因为删除而下降，直到 compact。

```python
"""milvus_lifecycle.py —— 文档下线的三段式生命周期：软删除 → 观察期 → 硬删除 → compact。"""
from __future__ import annotations

import time
from datetime import date, timedelta

from pymilvus import MilvusClient
from milvus_setup import URI, COLLECTION

OBSERVE_DAYS = 30   # 观察期：软删除后多久才真正物理删除


def soft_delete(client: MilvusClient, doc_id: str) -> int:
    """软删除：把 is_latest 置 False、expire_date 置为今天。检索侧默认过滤掉。"""
    rows = client.query(collection_name=COLLECTION,
                        filter=f'doc_id == "{doc_id}"',
                        output_fields=["*"], limit=16384)
    if not rows:
        print(f"文档 {doc_id} 不存在")
        return 0
    today = int(date.today().strftime("%Y%m%d"))
    for r in rows:
        r["is_latest"] = False
        r["expire_date"] = today
    client.upsert(collection_name=COLLECTION, data=rows)
    client.flush(COLLECTION)
    print(f"软删除 {doc_id}：{len(rows)} 个 chunk 已下线")
    return len(rows)


def hard_delete_expired(client: MilvusClient, observe_days: int = OBSERVE_DAYS) -> int:
    """硬删除：删掉过了观察期的软删除数据。"""
    cutoff = int((date.today() - timedelta(days=observe_days)).strftime("%Y%m%d"))
    expr = f"is_latest == false and expire_date > 0 and expire_date < {cutoff}"
    n = client.query(collection_name=COLLECTION, filter=expr,
                     output_fields=["chunk_id"], limit=1)
    if not n:
        print("没有需要硬删除的数据")
        return 0
    res = client.delete(collection_name=COLLECTION, filter=expr)
    client.flush(COLLECTION)
    print(f"硬删除完成：{res}")
    return 1


def compact_and_wait(client: MilvusClient, timeout: int = 1800) -> None:
    """触发 compaction 并等待完成，真正回收墓碑占用的空间。"""
    job_id = client.compact(collection_name=COLLECTION)
    print(f"compaction 任务已提交 job_id={job_id}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        state = client.get_compaction_state(job_id=job_id)
        print(f"  state={state} 已等待 {time.time() - t0:.0f}s")
        if str(state).lower().find("completed") >= 0:
            print("compaction 完成")
            return
        time.sleep(15)
    print("⚠️ compaction 超时，请到 Attu 控制台确认")


def compliance_delete(client: MilvusClient, doc_id: str) -> None:
    """合规删除（密级变更 / 客户要求删除）：跳过观察期，立即物理删除 + compact。"""
    res = client.delete(collection_name=COLLECTION, filter=f'doc_id == "{doc_id}"')
    client.flush(COLLECTION)
    print(f"合规删除 {doc_id}: {res}")
    compact_and_wait(client)
    # 同时要删源文件、ES 索引、缓存、日志里的正文快照——向量库只是其中一环
    print("⚠️ 别忘了：源文件 / ES / Redis 缓存 / 日志正文快照 也要同步清理")


if __name__ == "__main__":
    client = MilvusClient(uri=URI)
    soft_delete(client, "manuals/XJ-200操作手册_v2.1.pdf")
    hard_delete_expired(client)
    compact_and_wait(client)
```

**软删除 vs 硬删除的选择表：**

| 场景 | 选择 | 理由 |
|---|---|---|
| 文档出了新版本，老版本留档 | 软删除（`is_latest=False`） | 需要回溯"当时给用户的答案依据的是哪一版" |
| 文档被判定为错误信息 | 软删除 + 立刻在检索侧过滤 | 先止血，事后再决定要不要留档 |
| 文档只是临时下架 | 软删除 | 上架只要把标记翻回来 |
| 密级从"内部"升为"机密" | **硬删除 + 重新入不同 collection/分区** | 软删除靠查询端过滤，一旦漏加过滤条件就是安全事故 |
| 客户要求删除其数据 | **硬删除 + compact + 全链路清理** | 合规要求"不可恢复" |
| 源文件已删除（增量同步检测到） | 软删除 30 天 → 硬删除 | 防误删 |

> ⚠️ **安全红线**：**密级过滤绝不能只靠"检索时记得加 filter"**。正确做法是密级不同的数据分到不同 partition 甚至不同 collection，再由服务层按用户角色决定连哪个。人会忘记加 filter，物理隔离不会。

#### 3.5.3 双索引热切换：零停机换模型/换切分策略

这是本节最值钱的一段。需求是：**换 embedding 模型，线上服务不能停，出问题要能秒级回滚。**

核心机制是 **别名（Alias）**：应用代码永远访问别名 `huacheng_kb`，别名背后指向哪个真实 collection 由运维控制。

```mermaid
sequenceDiagram
    participant APP as 应用服务
    participant ALIAS as alias: huacheng_kb
    participant V1 as huacheng_kb_bgem3_v1
    participant V2 as huacheng_kb_bgelg_v2

    Note over V1: 线上服务中
    APP->>ALIAS: search()
    ALIAS->>V1: 转发

    Note over V2: ① 新建 + 全量重灌（后台，不影响线上）
    Note over V2: ② 验收门禁：行数 / 金标集 Recall / P95 延迟
    Note over V2: ③ 影子双读对比，人工抽查差异

    Note over ALIAS,V2: ④ alter_alias（毫秒级，原子）
    APP->>ALIAS: search()
    ALIAS->>V2: 转发

    Note over V1: ⑤ 保留 7 天，异常则一条命令切回
    Note over V1: ⑥ 观察期结束 drop
```

```python
"""scripts/reindex_switch.py —— 双索引热切换的完整实现（建库 → 灌数 → 门禁 → 切换 → 回滚 → 清理）。

用法：
    python scripts/reindex_switch.py build   --version v2 --embed-model BAAI/bge-large-zh-v1.5 --dim 1024
    python scripts/reindex_switch.py verify  --version v2
    python scripts/reindex_switch.py switch  --version v2
    python scripts/reindex_switch.py rollback
    python scripts/reindex_switch.py cleanup --keep 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient, DataType, connections, utility

URI = "http://localhost:19530"
ALIAS = "huacheng_kb"                       # 应用只认这个名字
GOLD_SET = Path("./evals/gold/retrieval_gold.jsonl")   # 金标集：{"query":..., "gold_chunk_ids":[...]}

# 门禁阈值：新索引必须不劣于旧索引这么多
GATE = {
    "row_count_tolerance": 0.02,   # 行数差异不超过 2%
    "recall_drop_max": 0.01,       # Recall@5 下降不超过 1 个百分点
    "p95_latency_max_ms": 50.0,    # P95 延迟不超过 50ms
    "min_gold_queries": 50,        # 金标集至少 50 条才允许放行
}


def collection_name(version: str, model_tag: str) -> str:
    """版本化命名：库名_模型标识_版本_日期，一眼看出这是什么。"""
    return f"{ALIAS}_{model_tag}_{version}_{date.today().strftime('%Y%m%d')}"


# ---------------------------------------------------------------- ① 建库 + 灌数

def build(version: str, embed_model: str, dim: int, model_tag: str) -> str:
    """建新 collection 并全量重灌。此过程完全不影响线上（线上走 alias 指向的旧库）。"""
    from build_index import create_schema_and_index, bulk_load_all   # 复用 3.2.2 / 3.2.3

    client = MilvusClient(uri=URI)
    name = collection_name(version, model_tag)
    if client.has_collection(name):
        raise SystemExit(f"{name} 已存在，先 drop 或换版本号")

    print(f"[build] 创建 {name} (model={embed_model}, dim={dim})")
    create_schema_and_index(client, name, dim=dim)
    print("[build] 开始全量灌入，这一步耗时最长")
    t0 = time.time()
    bulk_load_all(client, name, embed_model=embed_model)
    client.flush(name)
    client.load_collection(name)
    stats = client.get_collection_stats(name)
    print(f"[build] 完成：{stats} 耗时 {time.time() - t0:.0f}s")

    Path("./data/reindex").mkdir(parents=True, exist_ok=True)
    Path(f"./data/reindex/{version}.json").write_text(
        json.dumps({"collection": name, "embed_model": embed_model, "dim": dim},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return name


# ---------------------------------------------------------------- ② 验收门禁

@dataclass
class GateResult:
    """门禁检查结果。"""
    passed: bool
    details: list[str]

    def report(self) -> str:
        """渲染门禁报告。"""
        head = "✅ 门禁通过" if self.passed else "⛔ 门禁未通过"
        return head + "\n" + "\n".join("  " + d for d in self.details)


def current_collection(client: MilvusClient) -> str | None:
    """查 alias 当前指向哪个 collection。"""
    for c in client.list_collections():
        try:
            aliases = utility.list_aliases(c).get("aliases", [])
        except Exception:
            aliases = []
        if ALIAS in aliases:
            return c
    return None


def eval_recall_at_k(client: MilvusClient, name: str, embed_fn, k: int = 5) -> tuple[float, float]:
    """在金标集上算 Recall@k 与 P95 延迟（毫秒）。"""
    if not GOLD_SET.exists():
        raise SystemExit(f"金标集 {GOLD_SET} 不存在，门禁无法执行。见第 8.2 章。")
    items = [json.loads(l) for l in GOLD_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(items) < GATE["min_gold_queries"]:
        raise SystemExit(f"金标集只有 {len(items)} 条，少于 {GATE['min_gold_queries']} 条，不足以做门禁")

    hits, lats = 0, []
    for it in items:
        qv = embed_fn([it["query"]])
        t0 = time.perf_counter()
        res = client.search(collection_name=name, data=qv, anns_field="dense",
                            limit=k, search_params={"metric_type": "IP", "params": {"ef": 128}},
                            output_fields=["chunk_id"])
        lats.append((time.perf_counter() - t0) * 1000)
        got = {h["entity"]["chunk_id"] for h in res[0]}
        if got & set(it["gold_chunk_ids"]):
            hits += 1
    recall = hits / len(items)
    p95 = float(np.percentile(lats, 95))
    return recall, p95


def verify(version: str, embed_fn_new, embed_fn_old) -> GateResult:
    """对比新旧 collection，逐项检查门禁。"""
    client = MilvusClient(uri=URI)
    meta = json.loads(Path(f"./data/reindex/{version}.json").read_text(encoding="utf-8"))
    new_name = meta["collection"]
    old_name = current_collection(client)
    details, passed = [], True

    # 检查 1：行数
    n_new = client.get_collection_stats(new_name)["row_count"]
    if old_name:
        n_old = client.get_collection_stats(old_name)["row_count"]
        diff = abs(n_new - n_old) / max(n_old, 1)
        ok = diff <= GATE["row_count_tolerance"]
        passed &= ok
        details.append(f"{'✓' if ok else '✗'} 行数 新={n_new} 旧={n_old} 差异={diff:.2%} "
                       f"(阈值 {GATE['row_count_tolerance']:.0%})")
    else:
        details.append(f"· 首次建索引，跳过行数比对，新库 {n_new} 行")

    # 检查 2：金标集 Recall@5
    r_new, p95_new = eval_recall_at_k(client, new_name, embed_fn_new)
    if old_name:
        r_old, _ = eval_recall_at_k(client, old_name, embed_fn_old)
        ok = (r_old - r_new) <= GATE["recall_drop_max"]
        passed &= ok
        details.append(f"{'✓' if ok else '✗'} Recall@5 新={r_new:.4f} 旧={r_old:.4f} "
                       f"下降={r_old - r_new:+.4f} (允许 {GATE['recall_drop_max']})")
    else:
        details.append(f"· Recall@5 = {r_new:.4f}（无基线可比）")

    # 检查 3：P95 延迟
    ok = p95_new <= GATE["p95_latency_max_ms"]
    passed &= ok
    details.append(f"{'✓' if ok else '✗'} P95 延迟 {p95_new:.1f}ms (阈值 {GATE['p95_latency_max_ms']}ms)")

    # 检查 4：索引与加载状态
    desc = client.describe_index(new_name, index_name="dense")
    loaded = str(client.get_load_state(new_name))
    ok = "Loaded" in loaded
    passed &= ok
    details.append(f"{'✓' if ok else '✗'} 索引={desc.get('index_type')} 加载状态={loaded}")

    return GateResult(passed=passed, details=details)


# ---------------------------------------------------------------- ③ 影子双读

def shadow_compare(version: str, embed_fn_new, embed_fn_old,
                   queries: list[str], k: int = 5) -> None:
    """影子双读：同一批 query 同时打到新旧库，打印 Top-K 重合度，人工抽查差异大的。"""
    client = MilvusClient(uri=URI)
    new_name = json.loads(Path(f"./data/reindex/{version}.json").read_text(encoding="utf-8"))["collection"]
    old_name = current_collection(client)
    if not old_name:
        print("无旧库，跳过影子对比")
        return

    overlaps = []
    print(f"{'query':<34}{'重合度':<10}{'新库 Top1'}")
    print("-" * 100)
    for q in queries:
        rn = client.search(collection_name=new_name, data=embed_fn_new([q]), anns_field="dense",
                           limit=k, search_params={"metric_type": "IP", "params": {"ef": 128}},
                           output_fields=["chunk_id", "doc_title"])
        ro = client.search(collection_name=old_name, data=embed_fn_old([q]), anns_field="dense",
                           limit=k, search_params={"metric_type": "IP", "params": {"ef": 128}},
                           output_fields=["chunk_id"])
        sn = {h["entity"]["chunk_id"] for h in rn[0]}
        so = {h["entity"]["chunk_id"] for h in ro[0]}
        ov = len(sn & so) / max(len(so), 1)
        overlaps.append(ov)
        flag = "  ⚠️" if ov < 0.4 else ""
        print(f"{q[:32]:<34}{ov:>6.0%}    {rn[0][0]['entity']['doc_title'][:40]}{flag}")
    print("-" * 100)
    print(f"平均重合度 {statistics.mean(overlaps):.1%}，"
          f"重合度 < 40% 的 query 有 {sum(1 for o in overlaps if o < 0.4)} 条，请人工抽查")


# ---------------------------------------------------------------- ④ 切换与回滚

def switch(version: str) -> None:
    """把 alias 指向新 collection。这是一次原子操作，应用无感知。"""
    client = MilvusClient(uri=URI)
    connections.connect(uri=URI)
    new_name = json.loads(Path(f"./data/reindex/{version}.json").read_text(encoding="utf-8"))["collection"]
    old_name = current_collection(client)

    if str(client.get_load_state(new_name)).find("Loaded") < 0:
        raise SystemExit("⛔ 新 collection 未 load，切过去会直接 500。先 load_collection。")

    if old_name is None:
        client.create_alias(collection_name=new_name, alias=ALIAS)
    else:
        client.alter_alias(collection_name=new_name, alias=ALIAS)

    # 记录切换历史，回滚要用
    hist = Path("./data/reindex/history.jsonl")
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "from": old_name, "to": new_name},
                           ensure_ascii=False) + "\n")
    print(f"✅ alias {ALIAS}: {old_name} → {new_name}")
    print("请观察 15 分钟：P95 延迟、空结果率、人工抽查 20 条 query")


def rollback() -> None:
    """回滚到上一次切换前的 collection。"""
    client = MilvusClient(uri=URI)
    hist = Path("./data/reindex/history.jsonl")
    lines = [json.loads(l) for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines or not lines[-1]["from"]:
        raise SystemExit("没有可回滚的记录")
    target = lines[-1]["from"]
    if not client.has_collection(target):
        raise SystemExit(f"⛔ 旧 collection {target} 已被删除，无法回滚。这就是要保留 7 天的原因。")
    client.load_collection(target)
    client.alter_alias(collection_name=target, alias=ALIAS)
    with hist.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "from": lines[-1]["to"],
                            "to": target, "rollback": True}, ensure_ascii=False) + "\n")
    print(f"✅ 已回滚：alias {ALIAS} → {target}")


def cleanup(keep: int = 2) -> None:
    """清理旧版本 collection，只保留最近 keep 个（含当前在用的）。"""
    client = MilvusClient(uri=URI)
    cur = current_collection(client)
    cands = sorted([c for c in client.list_collections()
                    if c.startswith(ALIAS + "_") and c != cur])
    to_drop = cands[:max(0, len(cands) - (keep - 1))]
    for c in to_drop:
        print(f"[cleanup] drop {c}")
        client.release_collection(c)
        client.drop_collection(c)
    print(f"[cleanup] 当前在用 {cur}，保留 {cands[len(to_drop):]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "verify", "switch", "rollback", "cleanup"])
    ap.add_argument("--version", default="v2")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--model-tag", default="bgem3")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--keep", type=int, default=2)
    a = ap.parse_args()

    if a.action == "build":
        build(a.version, a.embed_model, a.dim, a.model_tag)
    elif a.action == "verify":
        from embed_fns import make_embed_fn            # 你的 embedding 封装
        r = verify(a.version, make_embed_fn(a.embed_model), make_embed_fn("BAAI/bge-m3"))
        print(r.report())
        raise SystemExit(0 if r.passed else 1)        # 非 0 退出码方便接 CI
    elif a.action == "switch":
        switch(a.version)
    elif a.action == "rollback":
        rollback()
    else:
        cleanup(a.keep)
```

**预期输出（verify 阶段）：**

```text
✅ 门禁通过
  ✓ 行数 新=823614 旧=823614 差异=0.00% (阈值 2%)
  ✓ Recall@5 新=0.8912 旧=0.8847 下降=-0.0065 (允许 0.01)
  ✓ P95 延迟 12.7ms (阈值 50.0ms)
  ✓ 索引=HNSW 加载状态=<LoadState: Loaded>
```

**预期输出（影子双读）：**

```text
query                             重合度      新库 Top1
----------------------------------------------------------------------------------------------------
XJ-200 报 E043 怎么处理                 100%    XJ-200 故障代码速查表 · E043
主轴异响是什么原因                       80%    主轴与轴承保养指南 · 异常噪声排查
保修期多久                             100%    整机保修政策 · 第二条 保修期限
换刀超时 E057                          60%    故障代码速查表 · E057
伺服驱动器温度报警                       40%    主轴与轴承保养指南 · 散热系统  ⚠️
----------------------------------------------------------------------------------------------------
平均重合度 76.0%，重合度 < 40% 的 query 有 0 条，请人工抽查
```

> 实测环境：Milvus 2.4.15 standalone / 82 万 chunk / 金标集 120 条 / 影子对比 200 条真实线上 query。

**切换流程的三条纪律：**

1. **门禁必须能自动判定，并用退出码表达结果**，这样才能接进 CI。靠人肉看报告，早晚有一次会拍脑袋放行。
2. **`switch` 之前一定要确认新 collection 已 `load`**。没 load 就切，等于把线上流量打到一个必然报错的库上——这个错误我见过不止一次。
3. **旧 collection 至少留 7 天**。内存不够就 `release_collection`（释放内存但保留数据），需要回滚时再 `load` 回来，只要几分钟。别 drop。

**pgvector 的等价方案**：用表 + `search_path` 或视图切换。

```sql
-- 方案 A：视图切换（最简单，应用永远查 kb_chunks 这个视图）
CREATE TABLE kb_chunks_v2 (LIKE kb_chunks_v1 INCLUDING ALL);
-- ...灌数据、建索引、跑门禁...
BEGIN;
DROP VIEW IF EXISTS kb_chunks;
CREATE VIEW kb_chunks AS SELECT * FROM kb_chunks_v2;
COMMIT;   -- 事务内切换，读请求要么看到旧的要么看到新的，不会看到中间态

-- 方案 B：表重命名（DDL 会拿排他锁，高并发下可能阻塞，需设短 lock_timeout）
BEGIN;
SET LOCAL lock_timeout = '3s';
ALTER TABLE kb_chunks RENAME TO kb_chunks_old;
ALTER TABLE kb_chunks_v2 RENAME TO kb_chunks;
COMMIT;
```

> pgvector 比 Milvus 好的一点：**切换可以放在事务里，天然原子**。差的一点：视图会让查询规划器少一些优化机会，超大表上要实测。

#### 3.5.4 备份与恢复

先摆一个容易被忽略的事实：**向量库不是你最重要的备份对象**。

| 数据 | 丢了能不能重建 | 重建代价 | 备份优先级 |
|---|---|---|---|
| 原始文档（PDF/Word/Excel） | **不能** | 无限大 | **P0** |
| chunk jsonl（切分+元数据产物） | 能（重跑解析流水线） | 约 5 小时 CPU | **P0**（重建太贵） |
| 向量（embedding 结果） | 能（重跑 embedding） | 约 30 分钟 GPU | P1 |
| Milvus 索引 | 能（重建索引） | 约 12 分钟 | P2 |
| manifest / 金标集 / 实验记录 | **不能** | 无限大 | **P0** |

**所以真正的备份策略是：把 chunk jsonl 当成一等公民备份，向量库当成可重建的派生物。** 这也意味着你的重建脚本必须是一条命令就能跑通的——平时不跑，真出事的时候是跑不起来的。

```bash
# ---------- Milvus 官方备份工具 ----------
# 安装（以官方 release 为准）
wget https://github.com/zilliztech/milvus-backup/releases/download/v0.5.0/milvus-backup_Linux_x86_64.tar.gz
tar zxf milvus-backup_Linux_x86_64.tar.gz

# 配置 backup.yaml：指向 Milvus 的 etcd / MinIO，以及备份存储桶
cat > configs/backup.yaml << 'YAML'
milvus:
  address: localhost
  port: 19530
minio:
  address: localhost
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  bucketName: a-bucket
  rootPath: files
backup:
  maxSegmentGroupSize: 2G
YAML

# 创建备份（按 collection 粒度）
./milvus-backup create \
  -n huacheng_kb_20260917 \
  --collections huacheng_kb_bgem3_v1_20260917

# 列出备份
./milvus-backup list

# 恢复到一个新名字（关键：恢复成 _restore 后缀，验证没问题再切 alias）
./milvus-backup restore \
  -n huacheng_kb_20260917 \
  -s _restore

# 恢复后必须重新建索引 + load（备份不含索引文件）
python -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='http://localhost:19530')
print(c.list_collections())
"
```

> ⚠️ **三个坑**：
> 1. `milvus-backup` 备份的是**数据**，恢复后**索引要重建**，别以为 restore 完就能直接查；
> 2. 备份工具需要能直连 **etcd 和 MinIO**，只开 19530 端口是不够的；
> 3. 备份文件落在 MinIO 的同一个桶里等于没备份——**必须同步到异地对象存储**（`mc mirror` 或 `rclone`）。

```bash
# ---------- pgvector 备份 ----------
# 逻辑备份：可读性好，恢复慢（要重建 HNSW 索引，百万级要几十分钟）
pg_dump -h localhost -p 5433 -U raguser -Fc -d huacheng_kb -f kb_20260917.dump
pg_restore -h localhost -p 5433 -U raguser -d huacheng_kb_new -j 8 kb_20260917.dump

# 物理备份 + WAL 归档：恢复快，能做 PITR（时间点恢复）
pg_basebackup -h localhost -p 5433 -U replicator -D /backup/base -Fp -Xs -P
# postgresql.conf:
#   archive_mode = on
#   archive_command = 'test ! -f /backup/wal/%f && cp %p /backup/wal/%f'
```

**RPO / RTO 目标与实现方式：**

| 目标 | 含义 | 华成机电取值 | 怎么实现 |
|---|---|---|---|
| RPO | 最多能丢多少数据 | 24 小时 | 每日全量备份 chunk jsonl + 每日 Milvus 备份 |
| RTO | 多久必须恢复服务 | 2 小时 | 从 chunk jsonl 重灌一个新 collection（约 45 分钟）+ 切 alias |
| 降级 RTO | 多久必须有"能用"的服务 | 15 分钟 | 切到只读缓存 + 纯 BM25/ES 检索（见第 2.4 章） |

**恢复演练清单（每季度必须真跑一次，不许纸上谈兵）：**

```text
□ 1. 在隔离环境启一套空 Milvus
□ 2. 用最新备份 restore，记录耗时
□ 3. 重建索引 + load，记录耗时
□ 4. 跑金标集 Recall@5，与生产基线对比（差异 > 1% 说明备份有问题）
□ 5. 跑 100 条真实 query，人工抽查 10 条
□ 6. 演练"备份也坏了"的兜底：从 chunk jsonl 全量重灌，记录耗时
□ 7. 把每一步实际耗时写进 runbook，与 RTO 目标对照
□ 8. 把演练中踩的坑补进 runbook（一定会有）
```

#### 3.5.5 监控指标清单

Milvus 自带 Prometheus 指标（standalone 默认在 `:9091/metrics`）。下面这张表是**上线第一天就应该配好的**，不是等出事了再加。

| 层 | 指标 | 来源 | 告警阈值（参考） | 为什么重要 |
|---|---|---|---|---|
| **延迟** | 检索 P99 延迟 | `milvus_proxy_req_latency`（search） | > 200ms 持续 5min | 用户直接感知 |
| **延迟** | 检索 P50 延迟 | 同上 | > 50ms | P50 涨说明是系统性问题而非长尾 |
| **吞吐** | 检索 QPS | `milvus_proxy_req_count` 速率 | 突降 > 50% | 突降通常是上游挂了 |
| **错误** | 检索失败率 | `milvus_proxy_req_count{status="fail"}` | > 0.1% | 直接影响可用性 |
| **容量** | collection 行数 | `client.get_collection_stats` | 日增 > 预期 3 倍 | 防止重复入库把库撑爆 |
| **容量** | QueryNode 内存占用 / 节点内存 | `milvus_querynode_*` + node_exporter | > 75% | 超了会 OOM，**这是最常见的生产事故** |
| **容量** | MinIO / 磁盘水位 | node_exporter | > 80% | 满了写入直接失败 |
| **健康** | segment 数量 | `milvus_datacoord_segment_num` | 单 collection > 2000 | 小 segment 过多说明 flush 太碎，查询变慢 |
| **健康** | compaction 落后量 | Attu / `get_compaction_state` | 待 compact > 500 | 墓碑堆积，延迟缓慢上涨 |
| **健康** | 索引构建队列 | `milvus_indexnode_*` | 积压 > 30min | 新数据查不到 |
| **健康** | etcd 可用性 + DB size | etcd metrics | size > 2GB / 不可用 | **etcd 挂了整个 Milvus 就挂了** |
| **模型** | embedding 服务 P99 | 自埋点（`core/instrument.py`） | > 500ms | 通常是整条链路最慢的一段 |
| **模型** | embedding 队列长度 | 自埋点 | > 100 | 说明并发打满，要扩容或限流 |
| **模型** | GPU 显存占用 | DCGM exporter / nvidia-smi | > 90% | 再来一个大 batch 就 OOM |
| **业务** | **空结果率**（Top-K 全部低于相似度阈值） | 自埋点 | > 5% | 知识库覆盖不足的最直接信号 |
| **业务** | **过滤后候选不足率**（过滤完不够 K 条） | 自埋点 | > 3% | 过滤条件太严或元数据缺失 |
| **业务** | 离线 Recall@5 日报 | 每日跑金标集 | 相对基线跌 > 2 个点 | **唯一能提前发现"检索质量悄悄劣化"的指标** |

```yaml
# monitoring/prometheus-milvus.yml —— 抓取配置
scrape_configs:
  - job_name: milvus
    static_configs:
      - targets: ["milvus-standalone:9091"]
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: embedding-service
    static_configs:
      - targets: ["embedding:7997"]     # Infinity / TEI 都暴露 /metrics
    scrape_interval: 15s

  - job_name: rag-app
    static_configs:
      - targets: ["rag-app:8080"]       # 应用层自埋点
    scrape_interval: 15s
```

```yaml
# monitoring/rules-milvus.yml —— 告警规则（阈值按自己环境调）
groups:
  - name: milvus-core
    rules:
      - alert: MilvusSearchLatencyHigh
        expr: |
          histogram_quantile(0.99,
            sum(rate(milvus_proxy_req_latency_bucket{function_name="Search"}[5m])) by (le)
          ) > 200
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "Milvus 检索 P99 延迟 > 200ms"
          runbook: "https://wiki.internal/runbook/milvus-latency"

      - alert: MilvusQueryNodeMemoryHigh
        expr: |
          milvus_querynode_process_memory_bytes
            / on(instance) node_memory_MemTotal_bytes > 0.75
        for: 10m
        labels: {severity: critical}
        annotations:
          summary: "QueryNode 内存水位 > 75%，有 OOM 风险"
          runbook: "扩容 / release 不用的 collection / 开 mmap"

      - alert: MilvusSearchErrorRate
        expr: |
          sum(rate(milvus_proxy_req_count{status="fail",function_name="Search"}[5m]))
            / sum(rate(milvus_proxy_req_count{function_name="Search"}[5m])) > 0.001
        for: 3m
        labels: {severity: critical}
        annotations: {summary: "Milvus 检索失败率 > 0.1%"}

      - alert: RagEmptyResultRateHigh
        expr: |
          sum(rate(rag_retrieval_empty_total[15m]))
            / sum(rate(rag_retrieval_total[15m])) > 0.05
        for: 15m
        labels: {severity: warning}
        annotations:
          summary: "RAG 空结果率 > 5%，知识库可能覆盖不足"
```

```python
"""scripts/health_check.py —— 一分钟巡检脚本，放进 crontab 每 5 分钟跑一次。"""
from __future__ import annotations

import sys
import time

import numpy as np
from pymilvus import MilvusClient

URI, ALIAS = "http://localhost:19530", "huacheng_kb"
THRESH = {"p99_ms": 200.0, "min_rows": 100_000, "max_segments": 2000}


def check() -> list[str]:
    """返回问题列表，空列表表示一切正常。"""
    problems: list[str] = []
    client = MilvusClient(uri=URI)

    # 1) 别名是否指向一个存在且已加载的 collection
    try:
        state = str(client.get_load_state(ALIAS))
        if "Loaded" not in state:
            problems.append(f"alias {ALIAS} 未加载：{state}")
    except Exception as e:
        problems.append(f"alias {ALIAS} 不可用：{e}")
        return problems

    # 2) 行数是否合理（防止被误清空）
    rows = int(client.get_collection_stats(ALIAS)["row_count"])
    if rows < THRESH["min_rows"]:
        problems.append(f"行数异常偏低：{rows} < {THRESH['min_rows']}")

    # 3) 用固定向量打 20 次，看 P99（不依赖 embedding 服务，纯测库）
    rng = np.random.default_rng(42)
    v = rng.normal(size=(1, 1024)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lats = []
    for _ in range(20):
        t0 = time.perf_counter()
        client.search(collection_name=ALIAS, data=v.tolist(), anns_field="dense",
                      limit=10, search_params={"metric_type": "IP", "params": {"ef": 128}})
        lats.append((time.perf_counter() - t0) * 1000)
    p99 = float(np.percentile(lats, 99))
    if p99 > THRESH["p99_ms"]:
        problems.append(f"P99 延迟 {p99:.1f}ms > {THRESH['p99_ms']}ms")

    print(f"行数={rows} P50={np.percentile(lats, 50):.1f}ms P99={p99:.1f}ms")
    return problems


if __name__ == "__main__":
    probs = check()
    if probs:
        for p in probs:
            print("⛔", p, file=sys.stderr)
        sys.exit(1)
    print("✅ 巡检通过")
```

**预期输出：**

```text
行数=823614 P50=1.4ms P99=3.8ms
✅ 巡检通过
```

### 3.6 容量与成本测算：1000 万 chunk 到底要多少内存

"我们大概有 1000 万条数据，买什么配置的机器？"——这是每个 RAG 项目在立项会上一定会被问到的问题。拍脑袋回答"128G 差不多吧"是不专业的。这一节把账算清楚，你以后可以直接拿公式去回答。

> 本节统一约定：**1 GB = $10^9$ 字节**。如果你要换算成操作系统显示的 GiB，除以 1.074 即可。

#### 3.6.1 内存占用公式

一个已 `load` 的 collection 在内存里由五部分组成：

$$
M_{\text{total}} = \underbrace{N \cdot d \cdot b}_{\text{①原始向量}} + \underbrace{N \cdot (M_0 + \epsilon) \cdot 4}_{\text{②HNSW 图}} + \underbrace{N \cdot L_{\text{text}}}_{\text{③正文}} + \underbrace{N \cdot L_{\text{meta}}}_{\text{④标量字段}} + \underbrace{I_{\text{scalar}}}_{\text{⑤标量索引}}
$$

各符号含义：

| 符号 | 含义 | 本例取值 |
|---|---|---|
| $N$ | 向量条数（chunk 数） | $10^7$ |
| $d$ | 向量维度 | 1024（bge-m3） |
| $b$ | 每个分量占的字节数 | float32 = 4；float16 = 2；SQ8 = 1 |
| $M_0$ | HNSW 第 0 层每节点的最大出边数，$M_0 = 2M$ | $M=16 \Rightarrow M_0 = 32$ |
| $\epsilon$ | 上层图 + 偏移表的额外开销（经验值 2~4） | 3 |
| $L_{\text{text}}$ | 每条正文的平均字节数 | 中文 400 字 × 3B ≈ 1200 B |
| $L_{\text{meta}}$ | 每条标量字段的平均字节数 | ≈ 300 B |
| $I_{\text{scalar}}$ | 标量倒排/位图索引 | ≈ 标量字段的 30% |

最后还要乘一个 **运行时系数** $\alpha$（增量 segment、删除位图、查询缓冲区、副本）：

$$
M_{\text{node}} = \alpha \cdot M_{\text{total}}, \quad \alpha \approx 1.3 \ (\text{单副本}) \ \sim 2.6 \ (\text{双副本})
$$

#### 3.6.2 逐项算给你看：1000 万 chunk × 1024 维

**① 原始向量（float32）**

$$
10^7 \times 1024 \times 4\ \text{B} = 4.096 \times 10^{10}\ \text{B} = \mathbf{40.96\ GB}
$$

**② HNSW 图结构（M=16）**

第 0 层每个节点 $M_0 = 32$ 条边，每条边存一个 4 字节的节点编号；再加上层图与偏移表（$\epsilon = 3$）：

$$
10^7 \times (32 + 3) \times 4\ \text{B} = 1.4 \times 10^9\ \text{B} = \mathbf{1.40\ GB}
$$

> 结论：**HNSW 的图结构只占原始向量的约 3.4%**。很多人以为"HNSW 很占内存"，其实占内存的从来是原始向量本身。

**③ 正文字段**

$$
10^7 \times 1200\ \text{B} = 1.2 \times 10^{10}\ \text{B} = \mathbf{12.00\ GB}
$$

**④ 标量字段**（doc_id / doc_title / section_path / device_model 数组 / 日期 / 布尔…）

$$
10^7 \times 300\ \text{B} = 3.0 \times 10^{9}\ \text{B} = \mathbf{3.00\ GB}
$$

**⑤ 标量索引**（7 个 INVERTED + 1 个 BITMAP + 2 个 STL_SORT）

$$
3.00 \times 0.30 = \mathbf{0.90\ GB}
$$

**合计与整机需求**

| 组成 | 大小 | 占比 |
|---|---:|---:|
| ① 原始向量 float32 | 40.96 GB | 70.4% |
| ② HNSW 图 | 1.40 GB | 2.4% |
| ③ 正文 | 12.00 GB | 20.6% |
| ④ 标量字段 | 3.00 GB | 5.2% |
| ⑤ 标量索引 | 0.90 GB | 1.5% |
| **小计 $M_{\text{total}}$** | **58.26 GB** | 100% |
| × 运行时系数 $\alpha = 1.3$ | **75.7 GB** | |
| + 操作系统与其他进程（预留 16 GB） | **≈ 92 GB** | |
| **建议整机内存** | **128 GB**（或 2 × QueryNode 64 GB） | |

**两条立刻能用的降本结论：**

1. **正文占了 20.6%，而它根本不需要放在向量库里。** 把 `text` 字段搬到 Redis / 对象存储 / 业务库，Milvus 只存 `chunk_id` + 向量 + 过滤字段，检索后按 id 回捞正文。**立省 12 GB（20%）**，代价是多一次 KV 查询（通常 1~2ms）。
2. **float32 换 float16 直接省一半向量内存。** Milvus 2.4 支持 `FLOAT16_VECTOR` / `BFLOAT16_VECTOR`，40.96 GB → 20.48 GB，**召回率损失通常在小数点后三位**（因为 embedding 模型本身就是 fp16 推理出来的）。这是**性价比最高的一刀**。

#### 3.6.3 量化到底能省多少：把计算过程写出来

**A. 标量量化 SQ8（Scalar Quantization, 8-bit）**

原理：对每一维，记录全库的最小值 $v_{\min}^{(j)}$ 和最大值 $v_{\max}^{(j)}$，把 float32 线性映射到 0~255 的整数：

$$
q_j = \left\lfloor 255 \cdot \frac{v_j - v_{\min}^{(j)}}{v_{\max}^{(j)} - v_{\min}^{(j)}} \right\rceil, \qquad
\hat{v}_j = v_{\min}^{(j)} + \frac{q_j}{255}\left(v_{\max}^{(j)} - v_{\min}^{(j)}\right)
$$

内存计算：

$$
\begin{aligned}
\text{量化后向量} &= 10^7 \times 1024 \times 1\ \text{B} = 1.024 \times 10^{10} = \mathbf{10.24\ GB} \\
\text{量化参数表} &= 2 \times 1024 \times 4\ \text{B} = 8192\ \text{B} \approx \mathbf{0\ GB\ （可忽略）} \\
\text{压缩比} &= \frac{40.96}{10.24} = \mathbf{4 \times},\quad \text{节省 } 30.72\ \text{GB}
\end{aligned}
$$

**B. 乘积量化 PQ（Product Quantization）**

原理：把 $d=1024$ 维切成 $m$ 段，每段 $d/m$ 维，各自跑 k-means 聚出 $2^{\text{nbits}}$ 个质心。每个向量只存 $m$ 个质心编号。

$$
\text{每向量字节数} = m \times \frac{\text{nbits}}{8}
$$

取 $m = 64$、$\text{nbits} = 8$（Milvus `IVF_PQ` 的典型配置，要求 $d \bmod m = 0$，$1024 / 64 = 16$ ✓）：

$$
\begin{aligned}
\text{每向量} &= 64 \times 1 = 64\ \text{B} \\
\text{量化后向量} &= 10^7 \times 64\ \text{B} = 6.4 \times 10^8 = \mathbf{0.64\ GB} \\
\text{码本大小} &= \underbrace{64}_{m} \times \underbrace{256}_{2^8} \times \underbrace{16}_{d/m} \times 4\ \text{B} = 1.05\ \text{MB} \approx \mathbf{0\ GB} \\
\text{压缩比} &= \frac{40.96}{0.64} = \mathbf{64 \times},\quad \text{节省 } 40.32\ \text{GB}
\end{aligned}
$$

换成 $m = 128$（每段 8 维，精度更高）：每向量 128 B → **1.28 GB**，压缩 **32×**。

**C. 汇总对照表**

| 方案 | 每向量字节 | 1000 万向量 | 压缩比 | 相对 Flat 的 Recall@10 | 适用 |
|---|---:|---:|---:|---|---|
| Flat float32 | 4096 B | 40.96 GB | 1× | 1.000（定义为基准） | ≤ 50 万，或要求 100% 召回 |
| HNSW float32 | 4096 B（+图 1.4GB） | 42.36 GB | ~1× | 0.98 ~ 0.99 | **百万到千万级主力** |
| HNSW float16 | 2048 B | 21.88 GB | 2× | 0.98 ~ 0.99（几乎无损） | **首选降本手段** |
| IVF_SQ8 | 1024 B | 10.24 GB | 4× | 0.93 ~ 0.96 | 千万级、内存紧 |
| IVF_PQ (m=128) | 128 B | 1.28 GB | 32× | 0.85 ~ 0.92 | 亿级 |
| IVF_PQ (m=64) | 64 B | 0.64 GB | 64× | 0.75 ~ 0.88 | 亿级、极端省内存 |
| IVF_PQ + 原向量精排 | 64 B（内存）+ 4096 B（磁盘） | 0.64 GB 内存 | 64×（内存） | 0.95 ~ 0.98 | **亿级最佳实践** |

> ⚠️ 表中 Recall 是**示例性数据，强烈建议在你自己的数据上复现**。PQ 的召回率对数据分布极其敏感：向量聚簇明显的数据集（比如同一类设备手册）PQ 表现好，分布均匀的数据集掉得厉害。
>
> ⚠️ 版本说明：Milvus 2.4 的量化索引主要是 `IVF_SQ8` / `IVF_PQ` / `SCANN`；`HNSW_SQ` / `HNSW_PQ` 等组合从 2.5 起提供，**以官方文档为准**。

**D. "量化 + 精排回捞"：既省内存又保召回的标准做法**

这是亿级场景的通用套路，值得单独说：

```mermaid
flowchart LR
    Q(["query 向量"]) --> A["① 在 PQ 压缩索引上<br/>粗召回 Top-200<br/>内存 0.64 GB，极快"]
    A --> B["② 按 id 从磁盘/对象存储<br/>读回这 200 条原始 float32 向量<br/>200 × 4KB = 800KB，一次顺序读"]
    B --> C["③ 用原始向量精确重算内积<br/>重排出真正的 Top-10"]
    C --> R(["高召回结果"])

    style A fill:#e8f8e8
    style C fill:#fff0d4
```

关键在于：**PQ 的误差只影响"谁进 Top-200"，不影响"Top-200 内部怎么排"**。而 Top-200 的召回率远高于 Top-10，所以最终 Top-10 的质量几乎追平 Flat。Milvus 的 `IVF_PQ` 可以通过 `refine`/`refine_k` 参数开启类似机制（具体参数名以对应版本文档为准），Faiss 里对应 `IndexRefineFlat`。

**E. 降维：另一条省钱的路**

| 手段 | 说明 | 省内存 | 风险 |
|---|---|---|---|
| 换低维模型（1024 → 768 → 512） | 直接选 `bge-large-zh-v1.5`(1024) / `bge-base-zh`(768) / `bge-small-zh`(512) | 线性 | 语义能力下降，必须评测 |
| **Matryoshka 表征学习（MRL）** | 模型训练时就让前 $k$ 维携带主要信息，推理时直接截断 | 线性 | 需要模型原生支持（如 `text-embedding-v3` 支持 1024/768/512/256） |
| PCA / 随机投影后降维 | 事后降维 | 线性 | 中文语义上掉点明显，**不推荐** |

> 截断 MRL 向量后**必须重新做 L2 归一化**，否则内积不再等价于余弦——这就是 2.1 节讲的归一化问题在降维场景的再现。

#### 3.6.4 磁盘索引 vs 内存索引：什么时候该把数据放磁盘

| 方案 | 内存占用（1000 万 × 1024） | 单查询延迟 | 召回 | 何时用 |
|---|---:|---|---|---|
| **HNSW 全内存** | ~58 GB | 1~3 ms | 0.98+ | **默认选择**，延迟敏感、内存买得起 |
| **HNSW + mmap** | 取决于 page cache，可压到 10~20 GB | 冷 10~50 ms / 热 2~5 ms | 0.98+ | 内存不够但有 NVMe；**访问有热点**时非常划算 |
| **DISKANN** | 约为原向量的 1/10（存压缩码 + 部分图） | 5~20 ms | 0.90~0.96 | 亿级、内存严重不足 |
| **IVF_PQ 全内存** | ~2 GB | 1~5 ms | 0.85~0.92 | 亿级、能接受召回下降 |
| **IVF_PQ + 磁盘精排** | ~2 GB + 磁盘 | 5~15 ms | 0.95~0.98 | **亿级最优性价比** |

```python
# Milvus 2.4 开启 mmap（在 milvus.yaml 或 collection 属性里设置，以官方文档为准）
# 全局开关：
#   queryNode:
#     mmap:
#       mmapEnabled: true
#       mmapDirPath: /var/lib/milvus/mmap
#
# 单 collection 级：
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530")
client.alter_collection_properties(
    collection_name="huacheng_kb_bgem3_v1",
    properties={"mmap.enabled": "true"},
)
# ⚠️ 改完必须 release + load 才生效
client.release_collection("huacheng_kb_bgem3_v1")
client.load_collection("huacheng_kb_bgem3_v1")
```

**mmap 的本质**：把索引文件交给操作系统的 page cache 管理。热数据自然留在内存，冷数据在磁盘。**前提是必须用 NVMe SSD**——在机械硬盘或网络存储上开 mmap，延迟会从毫秒级掉到百毫秒级，等于自杀。

**成本视角的取舍：** 按 2026 年市场公开零售价的**量级**判断（请以你采购当时的实际报价为准，这里只做数量级比较），服务器级 DDR5 内存每 GB 的单价大致是企业级 NVMe SSD 每 GB 的**一个数量级以上**。所以：

- 数据量在 **千万级以内**：内存差价可能只有几千到上万元，**买内存，不要折腾 mmap**——工程师调优 mmap 两周的人力成本远高于内存差价；
- 数据量到 **亿级**：全内存意味着要上 T 级内存（多机分片），此时 mmap / DiskANN / PQ 的价值才真正显现。

```mermaid
flowchart TD
    A{"chunk 总数？"} -->|"< 50 万"| B["Chroma / pgvector<br/>HNSW 全内存<br/>几 GB 内存搞定"]
    A -->|"50 万 ~ 1000 万"| C{"延迟要求 P99 < 10ms？"}
    A -->|"> 1000 万"| D{"内存预算够 T 级吗？"}

    C -->|"是"| E["Milvus HNSW 全内存<br/>float16 + 正文外置<br/>推荐 128GB 机器"]
    C -->|"否（可接受 50ms）"| F["Milvus HNSW + mmap<br/>64GB 机器即可"]

    D -->|"够"| G["Milvus 集群<br/>HNSW 分片，多 QueryNode"]
    D -->|"不够"| H["IVF_PQ / DISKANN<br/>+ 原向量磁盘精排"]

    style E fill:#d4f4dd
    style H fill:#fff0d4
```

#### 3.6.5 三种规模的配置总账

| 规模 | chunk 数 | 向量库 | 索引 | 内存 | 磁盘 | 部署形态 |
|---|---:|---|---|---:|---:|---|
| **教学 / PoC** | < 5 万 | Chroma | 默认 HNSW | 4 GB | 10 GB | 单进程嵌入式 |
| **部门级** | 5 万 ~ 50 万 | pgvector 或 Milvus standalone | HNSW M=16 | 8~16 GB | 50 GB | 单机 Docker Compose |
| **企业级（华成机电当前）** | 50 万 ~ 200 万 | Milvus standalone | HNSW M=16 float16 | 32 GB | 200 GB | 单机 + 每日备份 |
| **集团级** | 1000 万 | Milvus 集群 | HNSW float16，正文外置 | 128 GB（或 2×64） | 1 TB NVMe | 3 节点 + etcd 集群 |
| **超大规模** | 1 亿+ | Milvus 集群 | IVF_PQ + 磁盘精排 / DISKANN | 8×64 GB | 10 TB NVMe | K8s + 分片 + 多副本 |

#### 3.6.6 一次性成本：把 1000 万 chunk 灌进去要多少钱

**Embedding 计算量：**

$$
\text{总 token} = N \times \bar{T} = 10^7 \times 300 = 3 \times 10^9\ \text{token}
$$

| 方案 | 单价假设 | 总成本 | 耗时 | 说明 |
|---|---|---|---|---|
| **本地 bge-m3 + 单张 4090** | 只有电费 | 约 0 元 | 按 3000 chunk/s 估算约 **56 小时** | 慢，但最省钱 |
| **本地 bge-m3 + 8 卡** | 只有电费 | 约 0 元 | 约 **7 小时** | 企业内部有卡就这么干 |
| **在线 embedding API** | 按你签的每百万 token 单价 $p$ 元 | $3000 \times p$ 元 | 受限流影响，通常 10~30 小时 | 把 $p$ 填成你的实际单价即可 |

> ⚠️ 这里故意不写具体 API 单价——各家价格变动频繁，**写死的价格三个月后就是错的**。公式给你了，单价去官网查。

**存储与机器成本（月度）：**

| 项 | 规格 | 说明 |
|---|---|---|
| 向量库机器 | 32 核 / 128 GB / 1 TB NVMe | 自建按折旧算，云上按包年折月 |
| Embedding 推理 | 1 张消费级/专业卡，或按需 Serverless | 增量场景日均只需几分钟 GPU |
| 对象存储 | 约 1 TB（MinIO + 备份） | 异地备份再算一份 |
| 监控 | Prometheus + Grafana，2 核 4 GB | 可与其他服务共用 |

**最后一条成本纪律**：**别把 embedding 重算当成免费的**。一次全量重建 = 56 GPU 小时 + 一整天的工程师盯着。这就是为什么 3.5.1 那张"什么变了该做什么"的表必须贴在墙上——**每避免一次不必要的全量重建，就省下一天**。

---

## 四、踩坑与排错

下面这些全部来自真实项目。**建议直接收藏这一节**，90% 的向量库问题能在这里对上号。

### 4.1 建库与 Schema 类

| 现象 | 根因 | 解决 |
|---|---|---|
| `the dim (768) of field data(dense) is not equal to schema dim (1024)` | 换了 embedding 模型但 collection 的 `dim` 还是旧的；或 query 用 A 模型、入库用 B 模型 | 维度是建表时定死的，**不能改**。必须新建 collection 重灌。根治方法：collection 名里带模型标识（`huacheng_kb_bgem3_v1`），并把 `MILVUS_DIM` 写进 `core/config.py` 由代码统一读取，杜绝硬编码 |
| `CollectionAlreadyExistException` / `create collection failed: collection already exist` | 建表脚本重复执行，或 CI 里跑了两遍 | 建表函数一律先 `has_collection()` 判断；需要重建时显式传 `drop_existing=True`，**不要把 drop 做成默认行为**（见过 CI 把生产库 drop 掉的事故） |
| 建表时忘了加某个标量字段，上线后才发现要按它过滤 | Milvus 的 schema 是定长的，**加字段必须重建整个 collection** | 上线前把第 2.2 章那张元数据 schema 表过一遍，宁可多加几个空字段。临时救急可用 `enable_dynamic_field`，但动态字段**不能建索引**，过滤会全表扫描 |
| `Length of string exceeds max length` | `text` 字段超过 `max_length`（VARCHAR 以**字节**计，一个中文 3 字节） | `max_length=8192` 只够约 2700 个汉字。写入前截断 `text[:2500]`，或把正文外置到 Redis/对象存储，库里只存 id |
| ARRAY 字段写入报 `array length exceeds max capacity` | `device_model` 数组元素超过 `max_capacity=8` | 建表时把 `max_capacity` 放宽（16/32），或在入库前去重截断。同时检查是不是元数据抽取逻辑把整篇文档的所有型号都塞进了一个 chunk |
| 用 VARCHAR 存日期，范围过滤特别慢 | 字符串比较走不了数值索引 | 日期统一存 `INT32`（`yyyymmdd`），建 `STL_SORT` 索引。这是 3.2.2 里那个设计的理由 |

### 4.2 写入类

| 现象 | 根因 | 解决 |
|---|---|---|
| 批量插入时进程被 OOM Killer 杀掉 | 一次性把 82 万 chunk 全部读进内存 + 全部 encode，峰值内存 = 文本 + 向量 + 中间张量，轻松到几十 GB | 三件事：① 用生成器流式读 jsonl，不要 `readlines()` 全读；② `encode` 的 `batch_size` 降到 16~32；③ 插入按 500~2000 行一批，每批处理完显式 `del` 并让 GC 回收。可以用 `resource.setrlimit` 或容器 `mem_limit` 提前暴露问题 |
| `grpc message exceeds maximum size` / `message larger than max` | 单次 insert 的数据量超过 gRPC 默认上限（64 MB） | 减小 `insert_batch`。粗算：1000 行 × (1024×4 B 向量 + 2 KB 文本) ≈ 6 MB，所以 2000 行是安全上限；如果 chunk 正文很长要更小。超大规模改用 `bulk_insert` 导 parquet 走 MinIO 旁路 |
| 插入很慢，GPU 利用率只有 20% | 瓶颈在 embedding，且是单进程串行 | `nvidia-smi dmon` 确认。解法：多进程编码（`multiprocessing.Pool` + 每进程一张卡）或用 Infinity/TEI 服务化后多客户端并发。**注意瓶颈往往不在 Milvus** |
| 插完之后 `row_count` 还是 0 | 没 `flush()`，数据还在内存 buffer 里 | 插入结束调一次 `client.flush(collection)`。但**不要每批都 flush**，会产生大量小 segment 拖慢查询 |
| 反复重跑导入脚本，数据翻倍 | 主键没设计好或用了 `auto_id=True`，同一条 chunk 被插了多次 | `chunk_id` 必须是**确定性**的（`sha256(doc_id + chunk_index + text)[:32]`），用 `auto_id=False`，重跑走 upsert 而不是 insert |
| 删了一半数据，内存/磁盘一点没降 | Milvus 删除是写墓碑，空间在 compact 后才回收 | 手动触发 `client.compact()` 并等待完成（见 3.5.2）。生产上建议每周低峰期跑一次 |

### 4.3 检索类

| 现象 | 根因 | 解决 |
|---|---|---|
| `collection not loaded` / 查询直接报错 | Milvus 数据默认在磁盘，必须显式 `load_collection()` 才能查 | 服务启动时 load 一次；容器重启后会丢失 load 状态，**健康检查里要带上 `get_load_state`**（见 3.5.5 的巡检脚本） |
| `index not found` 或查询极慢（秒级） | 只插了数据没建索引，Milvus 退化成暴力扫描 | `describe_index()` 确认索引存在。注意：**先插数据再建索引**比边插边建快 2~3 倍 |
| **刚写入的数据立刻查询不到** | 一致性级别是 `Bounded`（默认），有秒级可见性延迟 | 三选一：① 单次查询传 `consistency_level="Strong"`（牺牲延迟）；② 写入后 `client.flush()` 再查；③ 业务上接受最终一致，前端提示"索引中，约 10 秒后可检索"。**不要为了这个把全局一致性改成 Strong**，那会让所有查询都变慢 |
| 加了过滤条件后查询慢几十倍 | 过滤字段没建标量索引 → 全表扫描；或触发了后置过滤 | `describe_index()` 检查该字段是否有 INVERTED/BITMAP 索引；确认 filter 是传给了向量库而不是在应用层做的（见 3.4 节） |
| 过滤条件很严时返回结果少于 `limit` | 满足条件的向量本来就不够，或 HNSW 剪枝后候选不足 | 调大 `ef`（如 256/512）；极低选择率（< 0.1%）改用 **partition 物理隔离**；同时在业务侧埋点统计"候选不足率"（见 3.5.5） |
| 相似度分数全是 0.99+，区分不开 | ① 向量没归一化但用了 IP，长向量集体高分；② chunk 之间本来就高度同质（比如 2 万行备件表） | ① 检查 `np.linalg.norm(v)` 是否 ≈ 1.0；② 同质数据靠向量检索无解，必须上元数据过滤 + BM25 混合检索（第 2.4 章） |
| 换了机器/重装后检索结果变了，但数据没动 | 索引参数不同（`M`、`efConstruction`），或 HNSW 构建本身带随机性 | HNSW 是近似索引，**构建结果不保证逐位一致**，Top-K 有小幅波动是正常的。评测要用统计指标（Recall@K）而不是"结果必须完全一样" |
| 检索结果里混进了已下线的旧版本文档 | 查询时忘了加 `is_latest == true` 过滤 | 把过滤条件**封装进检索函数的默认参数**，不要靠每个调用方自觉。密级过滤同理，见 5.4 节 |

### 4.4 Embedding 模型类

| 现象 | 根因 | 解决 |
|---|---|---|
| **归一化漏做，内积得分失真** | 用了 `metric_type="IP"` 但向量没做 L2 归一化。此时 $\mathbf{u}\cdot\mathbf{v} = \|u\|\|v\|\cos\theta$，**长向量无脑排前面**，排序被向量模长绑架 | 两条路：① 入库和查询**都**做 `v / np.linalg.norm(v)`；② 改用 `metric_type="COSINE"`（Milvus 内部替你归一化）。**最危险的是入库归一化了、查询忘了，这时没有任何报错，只是结果悄悄变差** |
| 换模型后检索全乱，但没报错 | 新旧向量混在同一个 collection 里 | 见 4.1 第一条。**向量空间不可混用**是本章最重要的一条铁律 |
| query 明明该命中却召不回 | 用 bge 系列时忘了给 query 加指令前缀（`为这个句子生成表示以用于检索相关文章：`） | 见 2.3 节。**前缀只加在 query 侧，passage 侧不加**；训练与推理必须一致 |
| chunk 后半段的内容永远检索不到 | chunk 超过模型 `max_length` 被静默截断 | `bge-large-zh-v1.5` 的 `max_length` 只有 512 token。用 tokenizer 实测 chunk 的 token 长度分布，超长就减小 chunk_size 或换 `bge-m3`（8192） |
| **embedding 服务并发打满，请求大量超时** | 单实例 GPU 只能串行处理 batch，上游没有限流，请求在队列里堆到超时 | 四件事：① 服务端开动态 batching（Infinity/TEI 默认支持）；② 客户端用 `asyncio.Semaphore` 限流到与服务端并发能力匹配；③ 加请求队列长度监控 + 熔断；④ query 侧加 **embedding 结果缓存**（同样的 query 文本直接命中 Redis），线上重复 query 比例通常不低 |
| GPU 显存 OOM（推理时） | `batch_size` 或 `max_length` 太大；或多个模型（embedding + rerank + LLM）挤在一张卡上 | 降 `batch_size`；开 `use_fp16=True`；**把 embedding 和 LLM 分卡部署**；用 `torch.cuda.empty_cache()` 只是治标 |

### 4.5 部署与运维类

| 现象 | 根因 | 解决 |
|---|---|---|
| Milvus 容器起来了但连不上 | etcd 或 MinIO 没就绪，Milvus 自己在重试 | `docker compose logs milvus-standalone`；健康检查等 `:9091/healthz` 返回 200 再连（见 3.2.1）。**别用 `sleep 30` 这种玄学等待** |
| 重启后查询全部报 `collection not loaded` | load 状态不持久化 | 启动脚本里加 load；或在应用的 startup 钩子里 load 并把它纳入 `/health` 检查 |
| 磁盘写满，Milvus 变只读 | MinIO 数据 + WAL + 备份挤在同一块盘 | 分盘：数据盘、备份盘分开；配 80% 水位告警；定期 compact + 清理旧 collection |
| 单机 Milvus 内存持续上涨直到 OOM | load 了多个历史版本的 collection 没 release | 用 3.5.3 的 `cleanup`；不用的老版本 `release_collection()`（保留数据、释放内存），需要回滚时再 load |
| 恢复备份后查询报错 | 备份不含索引，恢复后没重建索引和 load | 恢复流程固定为：restore → create_index → load → 跑金标集验证 → 切 alias |
| 生产库被误清空 | 增量同步脚本在源目录挂载失败时把"扫不到文件"当成"文件被删了" | 加 3.5.1 那个 **20% 安全阀**；删除操作先软删除，观察期后再硬删除 |

---

## 五、生产级要点

### 5.1 命名与版本规范（最便宜的保险）

```text
collection 命名：{业务}_{模型标识}_{版本}_{日期}
  huacheng_kb_bgem3_v1_20260917
  huacheng_kb_bgelg_v2_20261103

别名（应用只认这个）：huacheng_kb

索引命名：{字段}_{算法}
  dense_hnsw / sparse_inverted / doc_type_inverted
```

配套三条：

1. **应用代码里永远只出现别名**，真实 collection 名只出现在运维脚本里；
2. 每次重建都要在 `data/reindex/{version}.json` 落一份元信息（模型、维度、切分参数、语料快照哈希）——三个月后你一定会需要回答"当时那个库是怎么建的"；
3. **维度、collection 名、模型名统一从 `core/config.py` 读**（`MILVUS_DIM=1024`、`MILVUS_COLLECTION=huacheng_kb`），代码里不许出现魔法数字。

### 5.2 容量水位与扩容触发线

| 水位 | 动作 |
|---|---|
| 内存 < 60% | 正常 |
| 内存 60~75% | 开始规划：正文外置 / float16 / 清理旧版本 collection |
| 内存 > 75% | **触发扩容工单**，同时立即 release 非在用 collection |
| 内存 > 85% | 紧急：开 mmap 或临时分片，准备降级预案 |
| 磁盘 > 80% | 清理旧备份、跑 compact、扩盘 |
| segment 数 > 2000 | 低峰期跑 compact |

### 5.3 并发、限流与超时

整条检索链路上，**最容易被打爆的不是向量库，而是 embedding 服务**。向量库单机几千 QPS 很轻松，一张卡的 embedding 服务可能只有一两百 QPS。

```python
"""rag/retriever_guard.py —— 检索侧的限流、超时与降级封装。"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

EMBED_SEM = asyncio.Semaphore(16)     # 与 embedding 服务的并发能力匹配
MILVUS_SEM = asyncio.Semaphore(64)


async def embed_with_guard(texts: list[str], embed_fn, timeout: float = 3.0) -> list[list[float]]:
    """embedding 调用：限流 + 超时。超时直接抛，由上层决定降级。"""
    async with EMBED_SEM:
        async with asyncio.timeout(timeout):
            return await embed_fn(texts)


async def search_with_fallback(query: str, embed_fn, dense_search, bm25_search,
                               top_k: int = 20) -> tuple[list[dict], str]:
    """向量检索失败时降级到 BM25，保证"慢一点但有结果"，返回 (结果, 使用的通道)。"""
    try:
        qv = await embed_with_guard([query], embed_fn)
        async with MILVUS_SEM:
            async with asyncio.timeout(1.0):
                return await dense_search(qv, top_k), "dense"
    except asyncio.TimeoutError:
        log.warning("向量检索超时，降级到 BM25 | query=%s", query[:50])
    except Exception as e:
        log.exception("向量检索异常，降级到 BM25 | %s", e)
    try:
        return await bm25_search(query, top_k), "bm25_fallback"
    except Exception:
        log.exception("BM25 也失败了，返回空结果")
        return [], "none"
```

**三级降级预案：**

| 级别 | 触发条件 | 动作 | 用户感知 |
|---|---|---|---|
| L1 | embedding P99 > 1s | 提高缓存 TTL、关闭 query 改写等增强步骤 | 几乎无感 |
| L2 | 向量检索超时率 > 5% | **降级到 BM25/ES 纯关键词检索** | 语义类问题变差，精确型号类问题不受影响 |
| L3 | 向量库整体不可用 | 切到"高频问答缓存 + 人工转接"，前端显式提示 | 明显降级，但不是白屏 |

### 5.4 安全：密级过滤必须在库侧

**反面教材**（真实事故模式）：

```python
# ❌ 危险写法：靠调用方记得传 filter
results = client.search(collection_name="huacheng_kb", data=qv, limit=10)
results = [r for r in results if r["entity"]["confidentiality"] != "secret"]  # 应用层过滤
```

问题有三个：① 新同事写新接口时会忘；② 应用层过滤完可能不够 K 条，有人会"多召点再过滤"，加大泄露面；③ 日志、缓存、trace 里已经留下了机密正文。

```python
# ✅ 正确写法：把权限过滤封装进检索入口，不给调用方"不传"的机会
ROLE_LEVELS = {"guest": ["public"],
               "staff": ["public", "internal"],
               "engineer": ["public", "internal", "confidential"],
               "admin": ["public", "internal", "confidential", "secret"]}


def search_kb(query_vec, role: str, top_k: int = 10, extra_filter: str = "") -> list[dict]:
    """唯一对外的检索入口。密级过滤由本函数强制注入，调用方无法绕过。"""
    levels = ROLE_LEVELS.get(role)
    if levels is None:
        raise PermissionError(f"未知角色 {role}")
    quoted = ", ".join(f'"{x}"' for x in levels)
    f = f'is_latest == true and confidentiality in [{quoted}]'
    if extra_filter:
        f = f"({f}) and ({extra_filter})"
    return client.search(collection_name="huacheng_kb", data=query_vec,
                         anns_field="dense", limit=top_k, filter=f,
                         search_params={"metric_type": "IP", "params": {"ef": 128}},
                         output_fields=["chunk_id", "text", "doc_title", "page"])
```

**更强的做法**：机密级数据放**独立 collection**，由不同的服务账号访问。物理隔离 > 逻辑过滤，这一点在合规审计时尤其重要。

### 5.5 上线前检查清单

```text
□ collection 名带模型标识和版本，应用只用别名
□ dim / collection 名 / 模型名全部来自 core/config.py，无硬编码
□ 所有要过滤的标量字段都建了索引（INVERTED / BITMAP / STL_SORT）
□ 向量做了 L2 归一化，且入库与查询两侧一致
□ query 侧指令前缀与训练时一致（bge 系列）
□ chunk token 长度 p99 < 模型 max_length
□ 服务启动时 load collection，并纳入 /health 检查
□ 一致性级别选定并写进文档（默认 Bounded，特殊接口才用 Strong）
□ 密级/版本过滤封装在唯一检索入口，无法绕过
□ 限流（embedding 并发）+ 超时 + 三级降级预案已实现并演练
□ Prometheus 指标 + 告警规则已配置（延迟/内存/空结果率/离线召回）
□ 每日备份 chunk jsonl，每周 Milvus 备份，异地同步
□ 恢复演练已做过至少一次，runbook 里有真实耗时
□ 增量同步有 20% 删除安全阀
□ 重建索引有自动门禁（行数/Recall/P95）+ alias 切换 + 回滚脚本
□ 旧版本 collection 保留策略已定（建议 7 天）
```

---

## 六、本章小结 + 自测题

### 小结

1. **归一化是 0 成本的正确性保障**。归一化之后余弦、内积、L2 三者排序等价，`metric_type` 随便选；不归一化又用 IP，排序会被向量模长绑架，而且**不会有任何报错**。
2. **向量空间不可混用**是本章第一铁律。换模型 = 换空间 = 必须新建 collection 全量重灌。collection 命名里带模型标识，能挡掉 90% 的相关事故。
3. **bge 系列 query 要加指令前缀，passage 不加**；chunk 的 token 长度必须小于模型 `max_length`，否则后半段静默丢失。
4. **索引算法按规模选**：50 万以内 Flat/HNSW 随便用；百万到千万级 HNSW 是主力；亿级才考虑 IVF_PQ / DiskANN。HNSW 的图结构只占原始向量的约 3.4%，**占内存的从来是向量本身**。
5. **过滤一定要交给向量库做（前置过滤），并且过滤字段必须建标量索引**。选择率越低，前置过滤相对后置过滤的优势越大；选择率低于 0.1% 时应该上 partition 物理隔离。
6. **运维三件套**：增量同步用内容哈希 manifest（带删除安全阀）、下线走软删除+观察期+硬删除+compact、升级走"双索引 + 自动门禁 + alias 秒切 + 保留 7 天可回滚"。
7. **容量公式要会算**：$M = N d b + N (2M{+}\epsilon) \cdot 4 + N L_{\text{text}} + N L_{\text{meta}} + I_{\text{scalar}}$，再乘运行时系数 1.3。1000 万 × 1024 维 float32 约需 58 GB，建议 128 GB 机器。**最划算的两刀是 float16（省一半）和正文外置（省 20%）**。
8. **量化的账**：SQ8 压 4 倍、PQ(m=64) 压 64 倍，但召回要实测；亿级的最佳实践是 **PQ 粗召回 + 原向量磁盘精排**，内存按 PQ 算、召回接近 Flat。
9. **监控必须包含一个离线指标**：每日跑金标集 Recall@5。延迟和错误率能发现"系统坏了"，只有召回率日报能发现"检索质量在悄悄变差"。

### 自测题

**第 1 题**：同事把 `bge-m3` 换成了 `bge-large-zh-v1.5`（两者都是 1024 维），因为维度一样，他直接复用了原来的 collection，只是把新文档用新模型编码后追加进去，老数据没动。上线后测试发现：问老文档里的内容基本正常，问新文档里的内容也基本正常，但**跨新旧文档的问题结果很奇怪**，而且系统没有任何报错。请解释现象，并给出排查和修复方案。

<details>
<summary>参考答案</summary>

**根因：两个模型的向量落在完全不同的语义空间里，维度相同只是巧合，不代表可比。**

维度相同意味着**不会触发任何维度校验错误**——这正是它比"维度不一致直接报错"更危险的地方。报错至少会立刻暴露问题，这种情况会静默地污染整个库。

**为什么"只问老文档"和"只问新文档"看起来正常？**
因为如果 query 用的是新模型编码，它与新文档向量在同一空间，Top-K 里新文档会自然排前面；老文档的向量在另一个空间，与 query 向量的内积基本是随机的，但**随机值通常偏低**，所以不会挤进 Top-K。于是"问新文档"看着是对的。反过来如果用老模型编码 query，"问老文档"也看着是对的。**跨文档的问题一旦需要同时召回新老内容，排序就彻底失效。**

**排查方法（三步）：**

1. **统计分数分布**：分别取 100 条老 chunk 和 100 条新 chunk，用同一个 query 向量算内积，画两个直方图。如果两个分布明显分离（新的集中在 0.5~0.8、老的集中在 0.0~0.2 附近），基本确诊。
2. **自相似度检验**：取一条老文档的 chunk 原文，用**当前线上模型**重新编码，与库里存的那条老向量算余弦。如果不接近 1.0，说明这条向量不是当前模型产出的。这是最直接的证据。
3. **看元数据**：如果入库时记了 `pipeline_version` / 模型名（本章建议的字段），直接 `group by` 一下就知道库里混了几种模型。**这就是为什么要存这个字段。**

**修复方案：**

1. **立刻止血**：用 `pipeline_version` 过滤，只检索单一模型产出的那部分数据（选覆盖面更大的一侧），先让线上回到"结果正确但覆盖不全"的状态。
2. **按 3.5.3 的双索引流程重建**：新建 `huacheng_kb_bgelg_v2_yyyymmdd`，用新模型全量重灌所有文档（新老都要重灌），跑门禁，alias 切换。
3. **加防护**：
   - collection 名带模型标识，从根上不可能混；
   - 入库时把模型名 + 维度写进 `pipeline_version` 字段；
   - 服务启动时做一次断言：随机取库里 5 条 chunk，用当前模型重新编码，余弦相似度必须 > 0.999，否则**拒绝启动**。这条启动自检成本极低，价值极高。

**教训**：`dim` 相等不代表向量可比。**唯一可靠的判断标准是"是不是同一个模型的同一个版本"**。
</details>

**第 2 题**：华成机电的知识库有 1000 万个 chunk，向量是 1024 维。运维说只能给你一台 **64 GB 内存**的机器（不能再多了），要求 P99 延迟 < 30ms，Recall@10 不低于 0.95。请给出你的方案，并把内存账算出来。

<details>
<summary>参考答案</summary>

先按 3.6.2 的公式算基线：float32 全内存方案需要 $58.26 \times 1.3 \approx 75.7$ GB，**64 GB 装不下**。所以必须削。

**按性价比从高到低依次削：**

**第 1 刀：正文外置（省 12 GB）**
`text` 字段搬到 Redis 或对象存储，Milvus 只存 `chunk_id`、向量、过滤字段。检索拿到 id 后批量 `MGET` 回捞正文，增加约 1~2ms。
$58.26 - 12.00 = 46.26$ GB

**第 2 刀：float16 向量（省 20.48 GB）**
`FLOAT16_VECTOR` 把向量从 40.96 GB 压到 20.48 GB。embedding 模型本来就是 fp16 推理的，**召回损失在小数点后三位**，是这套方案里最划算的一刀。
$46.26 - 20.48 = 25.78$ GB

**第 3 刀：精简标量字段（省约 1 GB）**
只保留真正用于过滤的字段（`doc_id`、`device_model`、`error_codes`、`doc_type`、`confidentiality`、`is_latest`、`effective_date`），其余展示用字段（`doc_title`、`source_file`、`section_path`）跟正文一起外置。
$25.78 - 1.00 \approx 24.8$ GB

**最终账：**

| 项 | 大小 |
|---|---:|
| 向量 float16 | 20.48 GB |
| HNSW 图（M=16） | 1.40 GB |
| 精简后标量字段 | 2.00 GB |
| 标量索引 | 0.60 GB |
| $M_{\text{total}}$ | **24.48 GB** |
| × 运行时系数 1.3 | **31.8 GB** |
| + OS 与其他进程 | **≈ 40 GB** |
| **余量** | 64 − 40 = **24 GB** ✅ |

**延迟与召回验证：**
HNSW M=16 / `efConstruction=200` / 查询 `ef=128`，千万级下单查询通常在个位数毫秒（本章 3.2 的实测量级），P99 < 30ms 有充分余量；float16 的 Recall@10 与 float32 基本持平，0.95 的目标能达到——**但必须用金标集实测确认，不能照抄这里的结论**。

**为什么不用 PQ？**
PQ 确实能压到 1 GB 级，但 Recall@10 会掉到 0.85~0.92，**打不住 0.95 的红线**。如果一定要用，就得配"PQ 粗召回 Top-200 + 磁盘原向量精排"，复杂度上升一个台阶。既然 float16 + 正文外置就已经装得下，**就不要引入不必要的复杂度**——这是本题真正想考的判断。

**预留的 24 GB 用来干什么？**
① 双索引热切换时，新旧两个 collection 会短暂共存。不过 24 GB 装不下第二份完整索引，所以升级时的操作顺序是：先 `release` 旧 collection（释放内存、保留磁盘数据）→ load 新的 → 验证 → 切 alias；出问题再 release 新的、load 回旧的，回滚耗时几分钟而不是秒级。**这个约束要提前写进 runbook。**
② 留给突发查询峰值和 compaction。

**如果运维后来说能给 128 GB**：直接上 float32 全内存 + 正文仍然外置，省掉 float16 的验证工作量，把时间花在检索质量上。**内存便宜，工程师时间贵。**
</details>

**第 3 题**：线上反馈"刚上传的文档搜不到"。你去查，发现：① 导入脚本日志显示 2000 个 chunk 全部插入成功；② `get_collection_stats` 的 `row_count` 确实涨了 2000；③ 但用文档里的原句去检索，一条都召不回；④ 用 `client.query(filter='doc_id == "xxx"')` 却能查到这 2000 条。请列出至少 4 种可能原因，并说明每种的验证方法。

<details>
<summary>参考答案</summary>

题目里的关键线索是：**标量查询（query）能查到，向量检索（search）查不到**。这说明**数据在库里，问题出在向量通路上**。按这个思路排查：

**可能原因 1：索引还在构建中（最常见）**
Milvus 新插入的数据先进 growing segment，索引是异步构建的。索引没建完时，这部分数据要么不参与 ANN 检索，要么走暴力扫描但被 `ef` 剪枝掉。
- **验证**：Attu 控制台看 segment 状态，或 `client.describe_index()` 看 `indexed_rows` / `pending_index_rows`。
- **解决**：等待；或 `client.flush()` 后触发索引构建。**大批量导入后不要立刻测召回**。

**可能原因 2：一致性级别导致新数据不可见**
默认 `Bounded` 一致性有秒级（默认约 3 秒，可配）的可见性延迟。但注意：本题里 `query` 能查到，说明数据已经可见了，所以这条**在本题中可能性较低**——不过如果是"插入后立刻测"，它是头号嫌疑。
- **验证**：单次查询传 `consistency_level="Strong"` 再试。
- **解决**：接受最终一致 + 前端提示；或关键路径用 Strong。

**可能原因 3：向量没归一化 / 归一化不一致**
如果新数据入库时漏了归一化，而 `metric_type="IP"`，这批向量的模长可能远小于 1，内积得分被系统性压低，永远排在老数据后面。
- **验证**：`client.query(filter='doc_id=="xxx"', output_fields=["dense"])` 取回一条向量，算 `np.linalg.norm(v)`，看是不是 ≈ 1.0；再和老数据对比。
- **解决**：修导入脚本，重灌这批文档。

**可能原因 4：用错了 embedding 模型或漏了 query 指令前缀**
导入用 A 模型、查询用 B 模型；或 bge 系列的 query 前缀在这次改动里被弄丢了。
- **验证**：取一条 chunk 的原文，用**当前查询侧的编码函数**重新编码，与库里存的向量算余弦。接近 1.0 说明模型一致，否则就是模型不一致。
- **解决**：统一模型配置；前缀逻辑写进封装函数而不是散落在调用处。

**可能原因 5：检索时的过滤条件把它们过滤掉了**
导入时 `is_latest` 写成了 `False`，或 `effective_date` 写成了未来日期、`confidentiality` 写成了当前角色无权访问的级别，而检索入口强制注入了这些过滤条件（正如 5.4 节建议的那样）。
- **验证**：`client.query(filter='doc_id=="xxx"', output_fields=["is_latest","effective_date","confidentiality"])` 打出来看；或者把 search 的 filter 去掉再试一次。
- **解决**：修导入脚本的元数据默认值。

**可能原因 6：写进了一个没被检索的 partition**
如果按设备型号做了分区，导入时指定了新分区，但检索时只 load/指定了旧分区列表。
- **验证**：`client.list_partitions()`，并检查检索调用里的 `partition_names`。
- **解决**：检索默认不限定 partition，或把新分区加入 load 列表。

**排查顺序建议（从最快到最慢）：**
先看索引状态（10 秒）→ 去掉 filter 重试（10 秒）→ 取回向量算模长（1 分钟）→ 重新编码算自相似度（2 分钟）→ 查 partition。**先做成本低的验证，这是排错的通用原则。**
</details>

---

**上一章** [第 2.2 章 文档解析与切分策略](./02-文档解析与切分策略.md) | **下一章** [第 2.4 章 检索重排生成全链路](./04-检索重排生成全链路.md)
