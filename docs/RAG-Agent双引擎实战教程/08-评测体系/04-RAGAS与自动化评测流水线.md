# 第 8.4 章  RAGAS 与自动化评测流水线

> **本章目标**：读完能做到 …
> 1. 讲清楚 RAGAS 四个核心指标（Faithfulness / Context Precision / Context Recall / Answer Relevancy）**各自是怎么算出来的**，而不是只会调 API 看数字；
> 2. 用 **RAGAS 0.2.x + 国产模型**（DeepSeek 作 LLM、bge-m3 作 Embedding，全程不用 OpenAI）跑通一次完整评测，并能处理中文场景下的 NaN、波动、成本三大问题；
> 3. 说清楚 DeepEval / OpenCompass / lm-evaluation-harness 各自的适用场景，知道微调后该用哪个做通用能力回归；
> 4. 设计一条从「代码提交 → 冒烟 → 合并 → 每日回归 → 发版全量 → 上线监控」的完整评测流水线，写出**防抖动的门禁判定代码**（多次取中位数 + 统计显著性检验）；
> 5. 建立在线评测：点赞点踩埋点表结构、无标注质量估计、影子流量与 AB 实验设计；
> 6. 跑通 badcase 闭环 SOP，并能给团队交付一份可直接用的**评测周报模板**。
>
> **前置知识**：
> - [第 8.1 章 大模型与 RAG 评测方法论](./01-大模型与RAG评测方法论.md)（指标定义、统计显著性）
> - [第 8.2 章 LLM-Wiki 金标集构建工程](./02-LLM-Wiki金标集构建工程.md)（数据集格式，本章直接复用）
> - [第 8.3 章 DeepSeek-Harness 自研评测框架](./03-DeepSeek-Harness自研评测框架.md)（本章的流水线以它为主力，RAGAS 为对照）
> - [第 10.2 章 可观测性与链路追踪](../10-工程化与生产落地/02-可观测性与链路追踪.md)（在线评测依赖埋点）
>
> **预计用时**：阅读 60 分钟 / 动手 120 分钟

---

## 零、本章在评测体系里的位置

第 8.3 章把自研 harness 写完了，本章要回答三个问题：

| 问题 | 本章的回答 |
|---|---|
| 「那 RAGAS 还用不用？」 | **用，但不是拿来做门禁。** 用它做**指标交叉验证**（我们自己的分数和学术指标一致吗）和**快速探索**（新接一个项目先摸个底） |
| 「评测怎么变成团队的日常，而不是一次性的活动？」 | 把它**接进研发流程**：提交触发、每日回归、发版门禁、上线监控，四段全打通 |
| 「离线分数再好，线上还是有人骂，怎么办？」 | 做**在线评测**：用户反馈采集 + 无标注质量估计 + 影子流量 + AB 实验 |

一句话概括本章与前三章的关系：

```mermaid
flowchart LR
    C1["8.1 方法论<br/>指标怎么定义"] --> C2["8.2 金标集<br/>数据从哪来"]
    C2 --> C3["8.3 自研 harness<br/>怎么跑、怎么卡门禁"]
    C3 --> C4["**8.4 本章**<br/>怎么变成流水线 + 怎么看线上"]
    C4 -.->|"线上 badcase 回流"| C2
    C4 -.->|"RAGAS 交叉验证"| C3

    style C4 fill:#e8f5e9
```

---

## 一、为什么需要它

### 1.1 一个"自研 harness 也会骗人"的故事

华成机电项目第 9 周。自研 harness 已经跑了三周，回归集总分稳定在 74 分左右，冒烟门禁每天都绿。然后业务方反馈：

> 「你们那个分数我看不懂，但我知道**客服还是经常要自己翻手册**。」

排查下来发现两件事：

1. **我们的 `correctness` rubric 里写了"关键结论正确即可"，而业务方要的是"结论 + 页码 + 可操作步骤"**。系统答对了结论但没给页码，我们判 4 分，客服判不合格。
2. **我们的 judge 是 `deepseek-chat`，被测系统也是 `deepseek-chat`**，存在自我偏好偏差（第 8.1 章 2.6 节）。用 RAGAS 换一套完全不同的指标口径重跑，`faithfulness` 只有 0.68——比我们自己的 judge 分低了一大截。

**结论：任何单一评测口径都会有盲区。** 自研 harness 的优势是贴业务、可门禁；它的风险是**你既当运动员又当裁判**——rubric 是你写的，judge 是你选的，权重是你配的。

所以需要**第二把尺子**：

| 尺子 | 谁定的标准 | 用途 | 频率 |
|---|---|---|---|
| **自研 harness** | 你自己 | 上线决策、回归保护、归因 | 每次提交 / 每日 |
| **RAGAS** | 社区/学术界 | **交叉验证**：如果两把尺子结论相反，说明至少有一把有问题 | 每月一次 / 大改动后 |
| **OpenCompass** | 公开基准 | 微调后的**通用能力回归**：LoRA 有没有把基础能力搞坏 | 每次微调后 |
| **人工抽检** | 业务方 | 最终仲裁 | 每两周 50 条 |

### 1.2 本章的产出物

```text
evals/
├── ragas_eval/
│   ├── setup_models.py      # 国产模型接进 RAGAS（不用 OpenAI）
│   ├── build_dataset.py     # 金标集 → RAGAS EvaluationDataset
│   ├── run_ragas.py         # 跑评测 + 结果解析 + NaN 处理
│   └── compare.py           # RAGAS 分数 vs 自研 harness 分数的交叉验证
├── pipeline/
│   ├── gate.py              # 防抖动门禁判定（多次中位数 + 显著性检验）
│   ├── schema.sql           # 评测结果 + 在线反馈的数据表
│   └── weekly_report.py     # 评测周报生成器
├── online/
│   ├── feedback_api.py      # 点赞点踩采集接口
│   ├── sampler.py           # 在线无标注质量抽样打分
│   └── ab.py                # AB 实验分流与统计检验
└── templates/
    ├── 评测周报模板.md
    └── badcase工单模板.md
```

---

## 二、RAGAS 0.2.x 实战

### 2.1 四个核心指标是怎么算出来的

**不要把 RAGAS 当黑盒。** 每个指标背后都是一串 LLM 调用，理解它才知道什么时候该信、什么时候不该信。

先统一符号：$q$ = 用户问题，$c = \{c_1, \dots, c_K\}$ = 检索到的上下文，$a$ = 系统答案，$g$ = 标准答案（ground truth）。

#### 2.1.1 Faithfulness（忠实度）—— 答案有没有编

**直觉**：把答案拆成一条条"事实断言"，逐条问"这条能从上下文里推出来吗"，能推出来的比例就是 Faithfulness。

**算法（两步 LLM 调用）**：

1. **拆解**：让 LLM 把答案 $a$ 拆成原子断言集合 $S = \{s_1, \dots, s_n\}$；
2. **验证**：让 LLM 逐条判断 $s_i$ 是否能由上下文 $c$ 支持，得到 $\mathbb{1}[s_i \text{ supported}]$。

$$
\text{Faithfulness} = \frac{\left|\{s_i \in S : s_i \text{ 能由 } c \text{ 支持}\}\right|}{|S|}
$$

**例子**（华成机电）：

> 上下文：「E043 表示液压系统压力异常。处理步骤：1) 断电并挂牌；2) 检查液压站油位；3) 检查压力传感器接线；4) 更换压力传感器（零件号 XJ200-SP-014）。正常工作压力为 4.5 MPa。」
> 答案：「E043 是液压压力异常，先断电挂牌再查油位，正常压力 4.5 MPa。**这个故障通常在夏季高温时更频繁。**」

拆成 4 条断言：
1. E043 是液压压力异常 → ✅ 支持
2. 先断电挂牌 → ✅ 支持
3. 再查油位 → ✅ 支持
4. **夏季高温时更频繁 → ❌ 上下文里没有**

$\text{Faithfulness} = 3/4 = 0.75$

**三个关键性质，必须记住**：

| 性质 | 含义 | 后果 |
|---|---|---|
| **它不需要标准答案** | 只用 $a$ 和 $c$ | 可以在**线上无标注数据**上算，这是它最大的价值 |
| **它不管答案对不对** | 只管"有没有依据" | **上下文本身是错的时候，忠实于错误信息照样满分**。所以必须和 Correctness 联合看 |
| **它对"多说话"零容忍** | 每多一句无依据的补充都在扣分 | 模型爱加"建议咨询专业人员"之类的套话，会无端拉低分数 |

#### 2.1.2 Context Precision（上下文精确率）—— 召回的有用吗

**直觉**：检索回来 K 个 chunk，其中真正对回答有用的排得够不够靠前。

RAGAS 0.2.x 里有两个变体：

- **`LLMContextPrecisionWithReference`**：用标准答案 $g$ 判断每个 $c_k$ 是否相关（需要标注）；
- **`LLMContextPrecisionWithoutReference`**：用系统答案 $a$ 判断（无需标注，可用于线上）。

设 $v_k \in \{0,1\}$ 表示第 $k$ 个上下文是否相关，则：

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K}\left(\text{Precision@}k \times v_k\right)}{\sum_{k=1}^{K} v_k}, \quad \text{其中 } \text{Precision@}k = \frac{\sum_{j=1}^{k} v_j}{k}
$$

**这个公式本质上就是信息检索里的 Average Precision（AP）**，只不过相关性判定由 LLM 来做而不是人工标注。

**为什么要带位置权重？** 因为相关的 chunk 排在第 1 位和排在第 5 位，对生成质量的影响完全不同——排在后面的容易被模型忽略，长上下文里更严重（"lost in the middle"）。

**例子**：K=5，相关性向量 $v = [1, 0, 0, 1, 0]$

- Precision@1 = 1/1 = 1.0（相关，计入）
- Precision@2 = 1/2（不相关，不计入）
- Precision@3 = 1/3（不相关，不计入）
- Precision@4 = 2/4 = 0.5（相关，计入）
- Precision@5 = 2/5（不相关，不计入）

$\text{Context Precision} = (1.0 + 0.5) / 2 = 0.75$

如果相关的两个都排在前面（$v = [1,1,0,0,0]$）：$(1.0 + 1.0)/2 = 1.0$。**同样召回 2 条，排序好的得满分**——这就是它和普通 Precision 的区别。

#### 2.1.3 Context Recall（上下文召回率）—— 该召的召全了吗

**直觉**：把标准答案 $g$ 拆成一条条断言，看每条能不能在检索到的上下文里找到依据。

$$
\text{Context Recall} = \frac{\left|\{\text{标准答案中能被 } c \text{ 支持的断言}\}\right|}{\left|\{\text{标准答案的全部断言}\}\right|}
$$

**注意它和第 8.3 章 `Recall@K` 的区别**，这是很多人混淆的地方：

| 维度 | RAGAS Context Recall | harness 的 Recall@K |
|---|---|---|
| **判定依据** | 标准**答案文本**的断言能否被上下文支持 | 标准**条目 id** 是否出现在检索结果里 |
| **需要什么标注** | `reference`（标准答案文本） | `ground_truth_contexts`（条目 id 列表） |
| **判定方式** | LLM 语义判定 | 字符串精确匹配 |
| **优点** | 不依赖 chunk id，换切分策略也稳 | 确定性、零成本、可做门禁 |
| **缺点** | 有 LLM 随机性、要花钱 | 依赖标注质量 |

**两个都要看。** 前者告诉你"信息够不够"，后者告诉你"该召的那条召回来没有"。

#### 2.1.4 Answer Relevancy（答案相关性）—— 答的是不是问的

**这个指标的算法最反直觉，也最容易被误用。**

它**不是**比较答案和标准答案的相似度，而是用了一个叫**反向生成（reverse generation）**的技巧：

1. 给定答案 $a$，让 LLM **反推**出"这个答案在回答什么问题"，生成 $n$ 个（通常 3 个）候选问题 $\{q_1', \dots, q_n'\}$；
2. 计算这些反推问题与原问题 $q$ 的 embedding 余弦相似度，取平均：

$$
\text{Answer Relevancy} = \frac{1}{n}\sum_{i=1}^{n} \cos\left(\mathbf{e}_{q}, \mathbf{e}_{q_i'}\right)
$$

其中 $\mathbf{e}$ 是 embedding 向量。若答案被判定为"回避性回答（noncommittal）"，分数直接置 0。

**为什么这样设计？** 因为"答案是否切题"本质上等价于"能否从答案还原出原问题"。如果答案跑题了，反推出来的问题就和原问题对不上。

**这个技巧的三个坑**（务必理解，否则会被分数误导）：

| 坑 | 表现 | 应对 |
|---|---|---|
| **① 它不判断答案对不对** | 答案完全错误但切题，照样高分。「XJ-200 夹紧力是 99 kN」→ 反推问题是「XJ-200 夹紧力是多少」→ 相似度极高 → 满分 | **绝不能单独用 Answer Relevancy 判断质量**，必须和 Faithfulness / Correctness 一起看 |
| **② 拒答会被判 0** | 「知识库中没有收录」→ 被判为 noncommittal → 0 分 | **拒答题必须排除出 Answer Relevancy 的统计**，否则拒答做得越好分数越低 |
| **③ 强依赖 embedding 质量** | 中文 embedding 不好时，同义问题相似度偏低，分数系统性偏低 | 用 `bge-m3` 这类中文强的模型；**不同 embedding 之间的分数不可比** |

> **坑 ②** 是中文 RAG 项目里最常见的"指标看不懂"来源。你辛辛苦苦把拒答能力做上去了，RAGAS 的 Answer Relevancy 反而掉了一大截，然后你开始怀疑人生。**记住：拒答题单独统计。**

#### 2.1.5 其他常用指标速查

| 指标 | 类名（0.2.x） | 要什么标注 | 一句话说明 |
|---|---|---|---|
| Faithfulness | `Faithfulness` | 无需 reference | 答案断言被上下文支持的比例 |
| Context Precision | `LLMContextPrecisionWithReference` | reference | 相关上下文的加权排序质量（AP） |
| Context Recall | `LLMContextRecall` | reference | 标准答案断言被上下文覆盖的比例 |
| Answer Relevancy | `ResponseRelevancy` | 无需 reference | 反向生成问题与原问题的相似度 |
| Factual Correctness | `FactualCorrectness` | reference | 答案与标准答案的事实断言级 F1/precision/recall |
| Semantic Similarity | `SemanticSimilarity` | reference | 答案与标准答案的 embedding 余弦相似度 |
| Noise Sensitivity | `NoiseSensitivity` | reference | 加入无关上下文后答案错误率的变化 |

> 类名与参数以 **RAGAS 官方文档为准**；0.2.x 相比 0.1.x 做过较大重构（`Metric` 类化、`EvaluationDataset` 取代裸 `Dataset`），**网上大量教程还是 0.1.x 的写法，直接抄会报错**。

### 2.2 安装与版本

```bash
cd evals && source .venv/bin/activate

uv pip install "ragas>=0.2.0,<0.3" "langchain-openai>=0.2" "langchain-huggingface>=0.1" \
               "datasets>=2.20" "sentence-transformers>=3.0" "pandas>=2.2"
# pip 等价：
# pip install "ragas>=0.2.0,<0.3" "langchain-openai>=0.2" "langchain-huggingface>=0.1" \
#             "datasets>=2.20" "sentence-transformers>=3.0" "pandas>=2.2"

python -c "import ragas; print('ragas', ragas.__version__)"
```

**版本纪律**：把 `ragas` 的**精确版本号**写进 `requirements.txt` 并锁死（`ragas==0.2.x`）。原因在第 8.3 章讲过——**RAGAS 的指标内部 prompt 会随版本变，升版本会导致分数变化，而你会误以为是系统退化**。升级 RAGAS 要单独提 PR，并在同一份数据上跑新旧两版做对比。

### 2.3 配置国产模型（全程不用 OpenAI）

RAGAS 默认走 OpenAI。国内项目要换成 DeepSeek + 本地 embedding。**这是本节最实用的一段代码**。

```python
# file: evals/ragas_eval/setup_models.py
# 运行环境：Python 3.11
# 依赖：ragas>=0.2, langchain-openai>=0.2, langchain-huggingface>=0.1, sentence-transformers
"""把国产模型接进 RAGAS：DeepSeek 作评判 LLM，本地 bge-m3 作 Embedding，全程不依赖 OpenAI。"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig


def build_llm(model: str = "deepseek-chat",
              base_url: str = "https://api.deepseek.com",
              api_key_env: str = "DEEPSEEK_API_KEY",
              temperature: float = 0.0,
              max_retries: int = 5,
              timeout: float = 120.0) -> LangchainLLMWrapper:
    """构造 RAGAS 用的评判 LLM。任何 OpenAI 兼容端点都可以用这套写法。"""
    chat = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=os.environ[api_key_env],
        temperature=temperature,          # 评判任务必须 0，否则每次跑分数都在抖
        max_retries=max_retries,
        timeout=timeout,
        max_tokens=1024,
    )
    return LangchainLLMWrapper(chat)


def build_embeddings_local(model_name: str = "BAAI/bge-m3",
                           device: str | None = None) -> LangchainEmbeddingsWrapper:
    """本地 embedding（推荐）：中文效果好、零调用成本、结果稳定可复现。"""
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": dev},
        encode_kwargs={"normalize_embeddings": True},   # 归一化后余弦相似度才对
    )
    return LangchainEmbeddingsWrapper(emb)


def build_embeddings_api(model: str = "text-embedding-v3",
                         base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                         api_key_env: str = "DASHSCOPE_API_KEY") -> LangchainEmbeddingsWrapper:
    """在线 embedding（无 GPU 时用）：走 OpenAI 兼容端点，具体模型名以服务商文档为准。"""
    emb = OpenAIEmbeddings(model=model, base_url=base_url,
                           api_key=os.environ[api_key_env], check_embedding_ctx_length=False)
    return LangchainEmbeddingsWrapper(emb)


def build_run_config(max_workers: int = 4, timeout: int = 180,
                     max_retries: int = 5, max_wait: int = 60) -> RunConfig:
    """RAGAS 的并发与重试配置。中文长文本场景建议把并发压低，否则大面积超时变 NaN。"""
    return RunConfig(max_workers=max_workers, timeout=timeout,
                     max_retries=max_retries, max_wait=max_wait)


def default_setup(local_embedding: bool = True) -> tuple:
    """一行拿到 (llm, embeddings, run_config)，供其他脚本直接引用。"""
    llm = build_llm()
    emb = build_embeddings_local() if local_embedding else build_embeddings_api()
    cfg = build_run_config()
    return llm, emb, cfg


if __name__ == "__main__":
    llm, emb, cfg = default_setup()
    print("LLM:", llm)
    print("Embeddings:", emb)
    print("RunConfig:", cfg)
    v = emb.embed_query("XJ-200 液压卡盘夹紧力上限是多少")
    print(f"embedding 维度：{len(v)}（bge-m3 应为 1024，与 MILVUS_DIM 一致）")
```

**预期输出**：

```text
LLM: <ragas.llms.base.LangchainLLMWrapper object at 0x7f...>
Embeddings: <ragas.embeddings.base.LangchainEmbeddingsWrapper object at 0x7f...>
RunConfig: RunConfig(timeout=180, max_retries=5, max_wait=60, max_workers=4, ...)
embedding 维度：1024（bge-m3 应为 1024，与 MILVUS_DIM 一致）
```

**三个配置要点**：

| 配置 | 值 | 为什么 |
|---|---|---|
| `temperature=0` | 必须 | RAGAS 每个指标内部都在调 LLM 做判断，温度不为 0 = 每次跑分数都不一样 |
| `max_workers=4`（而不是默认更高） | 建议 | RAGAS 的并发是**指标级 × 样本级**叠加的，很容易打爆限流。**大面积超时会变成 NaN**，比跑得慢糟糕得多 |
| embedding 用本地 | 强烈建议 | ① 免费；② 结果完全可复现；③ 和你 RAG 系统用的是同一个模型，口径一致 |

### 2.4 准备数据集：金标集 → RAGAS

RAGAS 0.2.x 用 `EvaluationDataset` + `SingleTurnSample`，字段名和第 8.2 章的金标集不同，需要做映射：

| 金标集字段（第 8.2 章） | RAGAS 字段 | 说明 |
|---|---|---|
| `question` | `user_input` | 用户问题 |
| （被测系统返回的 contexts 文本） | `retrieved_contexts` | **是文本列表，不是 chunk id** |
| （被测系统返回的 answer） | `response` | 系统答案 |
| `ground_truth_answer` | `reference` | 标准答案文本 |
| `ground_truth_snippets` | `reference_contexts` | 可选，部分指标会用 |

```python
# file: evals/ragas_eval/build_dataset.py
# 运行环境：Python 3.11
"""把第 8.3 章 harness 跑出的 outputs.jsonl + 金标集，转成 RAGAS 的 EvaluationDataset。"""

from __future__ import annotations

import json
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample


def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL。"""
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def build(gold_path: str, outputs_path: str, *,
          exclude_refusal: bool = True,
          exclude_error: bool = True,
          max_context_chars: int = 1200) -> tuple[EvaluationDataset, list[str]]:
    """构造 RAGAS 数据集，返回 (dataset, qid 列表)。

    exclude_refusal 默认为 True 的原因见 2.1.4：拒答题会被 Answer Relevancy 判 0，
    混进来会让『拒答做得越好分数越低』，必须单独统计。
    """
    gold = {g["qid"]: g for g in load_jsonl(gold_path)}
    outs = {o["qid"]: o for o in load_jsonl(outputs_path)}

    samples, qids, skipped = [], [], []
    for qid, g in gold.items():
        o = outs.get(qid)
        if o is None:
            skipped.append((qid, "系统无输出"))
            continue
        if exclude_error and o.get("error"):
            skipped.append((qid, f"系统报错：{o['error'][:40]}"))
            continue
        if exclude_refusal and g.get("should_refuse"):
            skipped.append((qid, "拒答题，单独统计"))
            continue

        ctxs = [c.get("text", "")[:max_context_chars] for c in o.get("contexts", [])]
        if not ctxs:
            # RAGAS 的多数指标要求非空 contexts，空列表会算出 NaN
            ctxs = ["（本次未检索到任何资料）"]

        samples.append(SingleTurnSample(
            user_input=g["question"],
            retrieved_contexts=ctxs,
            response=o.get("answer", ""),
            reference=g.get("ground_truth_answer", ""),
            reference_contexts=g.get("ground_truth_snippets", []) or None,
        ))
        qids.append(qid)

    print(f"构造 RAGAS 数据集：{len(samples)} 条（跳过 {len(skipped)} 条）")
    from collections import Counter
    for reason, n in Counter(r for _, r in skipped).most_common():
        print(f"  跳过原因：{reason} × {n}")
    return EvaluationDataset(samples=samples), qids


if __name__ == "__main__":
    import sys
    ds, qids = build(sys.argv[1] if len(sys.argv) > 1 else "datasets/golden_regression.jsonl",
                     sys.argv[2] if len(sys.argv) > 2 else "runs/latest/outputs.jsonl")
    print(f"\n数据集大小：{len(ds)}")
    print(f"首条样例：\n{ds[0]}")
```

### 2.5 跑评测并读结果

```python
# file: evals/ragas_eval/run_ragas.py
# 运行环境：Python 3.11
"""跑 RAGAS 评测：选指标 → 评测 → NaN 处理 → 分层统计 → 落盘 CSV/Markdown。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from ragas import evaluate
from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
                           LLMContextRecall, ResponseRelevancy, SemanticSimilarity)

sys.path.append(str(Path(__file__).parent))
from build_dataset import build, load_jsonl
from setup_models import default_setup


def build_metrics(llm, emb) -> list:
    """构造指标实例。0.2.x 必须显式把 llm / embeddings 注入指标，不能依赖全局默认。"""
    return [
        Faithfulness(llm=llm),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        FactualCorrectness(llm=llm),
        SemanticSimilarity(embeddings=emb),
    ]


def run(gold_path: str, outputs_path: str, out_dir: str, local_embedding: bool = True) -> pd.DataFrame:
    """跑一次完整 RAGAS 评测并落盘。"""
    llm, emb, run_cfg = default_setup(local_embedding=local_embedding)
    ds, qids = build(gold_path, outputs_path)

    print(f"\n开始 RAGAS 评测：{len(ds)} 条样本 × {6} 个指标 …")
    result = evaluate(dataset=ds, metrics=build_metrics(llm, emb),
                      llm=llm, embeddings=emb, run_config=run_cfg,
                      raise_exceptions=False)      # 单条失败不要炸掉整批

    df = result.to_pandas()
    df.insert(0, "qid", qids[:len(df)])

    # ---- NaN 处理：先看清楚再决定怎么办，不要无脑 fillna(0) ----
    metric_cols = [c for c in df.columns if c not in
                   ("qid", "user_input", "retrieved_contexts", "response",
                    "reference", "reference_contexts")]
    nan_report = {c: int(df[c].isna().sum()) for c in metric_cols}
    print("\nNaN 统计（每个指标有多少条没算出来）：")
    for c, n in nan_report.items():
        flag = " ⚠️" if n > len(df) * 0.1 else ""
        print(f"  {c:38s} {n:4d} / {len(df)}  ({n / max(len(df), 1):.1%}){flag}")

    # 挂金标集的分层字段，用于分类统计
    gold = {g["qid"]: g for g in load_jsonl(gold_path)}
    for col in ("category", "difficulty", "question_type"):
        df[col] = df["qid"].map(lambda q: gold.get(q, {}).get(col, "未知"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ragas_detail.csv", index=False, encoding="utf-8-sig")

    # ---- 聚合：用 mean(skipna=True)，并同时报出有效样本数 ----
    summary = {}
    for c in metric_cols:
        valid = df[c].dropna()
        summary[c] = {"mean": round(float(valid.mean()), 4) if len(valid) else None,
                      "median": round(float(valid.median()), 4) if len(valid) else None,
                      "n_valid": int(len(valid)), "n_nan": int(df[c].isna().sum())}
    (out / "ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render(df, summary, metric_cols)
    (out / "ragas_report.md").write_text(md, encoding="utf-8")
    print(f"\n产物：{out}/ragas_detail.csv、ragas_summary.json、ragas_report.md")
    print("\n" + md[:1800])
    return df


def render(df: pd.DataFrame, summary: dict, metric_cols: list[str]) -> str:
    """渲染 RAGAS 报告 Markdown，含总体指标、分层统计、低分样本。"""
    L = ["# RAGAS 评测报告", "", f"- 样本数：**{len(df)}**", "",
         "## 一、总体指标", "",
         "| 指标 | 均值 | 中位数 | 有效样本 | NaN |", "|---|---|---|---|---|"]
    for c in metric_cols:
        s = summary[c]
        L.append(f"| {c} | **{s['mean']}** | {s['median']} | {s['n_valid']} | {s['n_nan']} |")
    L += ["", "> 均值用 `skipna=True` 计算。**NaN 超过 10% 时均值不可信**，"
          "请先排查原因（见 2.6 节）再解读分数。", ""]

    for dim, title in (("difficulty", "二、分难度"), ("category", "三、分类别"),
                       ("question_type", "四、分题型")):
        if dim not in df.columns:
            continue
        g = df.groupby(dim)[metric_cols].mean(numeric_only=True).round(3)
        cnt = df.groupby(dim).size()
        L += [f"## {title}", "", "| " + dim + " | 题数 | " + " | ".join(metric_cols) + " |",
              "|" + "|".join(["---"] * (len(metric_cols) + 2)) + "|"]
        for k, row in g.iterrows():
            L.append(f"| {k} | {cnt[k]} | " + " | ".join(str(row[c]) for c in metric_cols) + " |")
        L.append("")

    # 低分样本
    if "faithfulness" in df.columns:
        low = df.nsmallest(10, "faithfulness")[["qid", "faithfulness", "user_input", "response"]]
        L += ["## 五、Faithfulness 最低的 10 条", "",
              "| qid | faithfulness | 问题 | 答案（截断） |", "|---|---|---|---|"]
        for _, r in low.iterrows():
            L.append(f"| {r['qid']} | {r['faithfulness']} | {str(r['user_input'])[:24]}… "
                     f"| {str(r['response'])[:40].replace(chr(10), ' ')}… |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "datasets/golden_regression.jsonl",
        a[1] if len(a) > 1 else "runs/latest/outputs.jsonl",
        a[2] if len(a) > 2 else "runs/latest/ragas")
```

**预期输出**（300 题回归集，**以下全部为示例性数据，你本地跑不会完全一致**）：

```text
构造 RAGAS 数据集：238 条（跳过 62 条）
  跳过原因：拒答题，单独统计 × 58
  跳过原因：系统报错：Timeout: 超过 90s × 3
  跳过原因：系统无输出 × 1

开始 RAGAS 评测：238 条样本 × 6 个指标 …
Evaluating: 100%|████████████████████████| 1428/1428 [14:22<00:00,  1.66it/s]

NaN 统计（每个指标有多少条没算出来）：
  faithfulness                              7 /  238  (2.9%)
  llm_context_precision_with_reference      2 /  238  (0.8%)
  context_recall                            3 /  238  (1.3%)
  answer_relevancy                         11 /  238  (4.6%)
  factual_correctness(mode=f1)              5 /  238  (2.1%)
  semantic_similarity                       0 /  238  (0.0%)

产物：runs/latest/ragas/ragas_detail.csv、ragas_summary.json、ragas_report.md

# RAGAS 评测报告

- 样本数：**238**

## 一、总体指标

| 指标 | 均值 | 中位数 | 有效样本 | NaN |
|---|---|---|---|---|
| faithfulness | **0.7412** | 0.8000 | 231 | 7 |
| llm_context_precision_with_reference | **0.7863** | 0.8333 | 236 | 2 |
| context_recall | **0.8104** | 1.0000 | 235 | 3 |
| answer_relevancy | **0.8271** | 0.8695 | 227 | 11 |
| factual_correctness(mode=f1) | **0.6538** | 0.6667 | 233 | 5 |
| semantic_similarity | **0.8916** | 0.9042 | 238 | 0 |

## 二、分难度

| difficulty | 题数 | faithfulness | llm_context_precision_with_reference | context_recall | answer_relevancy | factual_correctness(mode=f1) | semantic_similarity |
|---|---|---|---|---|---|---|---|
| complex | 61 | 0.681 | 0.702 | 0.724 | 0.803 | 0.559 | 0.871 |
| edge | 27 | 0.694 | 0.743 | 0.761 | 0.778 | 0.581 | 0.862 |
| medium | 73 | 0.752 | 0.801 | 0.833 | 0.834 | 0.672 | 0.893 |
| simple | 77 | 0.796 | 0.856 | 0.894 | 0.855 | 0.731 | 0.913 |
```

**怎么读这份报告**（比跑出来更重要）：

| 观察 | 解读 | 下一步动作 |
|---|---|---|
| `faithfulness` 0.74，但 `semantic_similarity` 0.89 | 答案"看起来像标准答案"，但有约四分之一的断言没有依据——**典型的"用模型内部知识补全"** | 强化 prompt 的"只用资料回答"；检查 `max_tokens` 是不是太大导致模型话痨 |
| `context_recall` 0.81 > `context_precision` 0.79 | 召回还行，排序略差 | 上 rerank（第 3.2 章） |
| `factual_correctness` 0.65 明显低于其他 | **事实级别的对错才是业务关心的**，它低说明"看起来对但细节错" | 逐条看低分样本，大概率是数值/型号错误 |
| complex 难度上所有指标都掉 10 个点左右 | 多跳/聚合能力不足 | 第 3.3 章的查询拆解 + 多跳检索 |
| `answer_relevancy` 反而是最高的 | **正常**，因为它不判对错（2.1.4 坑①） | **不要拿它汇报质量** |

### 2.6 中文场景的四大常见问题

#### 问题一：NaN 满天飞

**症状**：跑完一看，`faithfulness` 有 30% 是 NaN。

**五个原因与对策**：

| 原因 | 判断方法 | 对策 |
|---|---|---|
| **① LLM 输出没按要求返回 JSON** | 打开 `raise_exceptions=True` 看具体报错 | 换更强的评判模型；确认 `temperature=0` |
| **② 超时** | 报错里有 timeout | 调大 `RunConfig.timeout`（180~300）；**降低 `max_workers`** |
| **③ 限流 429** | 报错里有 rate limit | `max_workers` 降到 2~4；`max_retries` 提到 5~8 |
| **④ `retrieved_contexts` 为空** | 检查数据集 | 空列表要填占位文本（`build_dataset.py` 已处理） |
| **⑤ 答案为空或极短** | 检查 `response` 字段 | 系统报错的样本要在构造数据集时就排除掉 |

> **绝对不要 `fillna(0)`。** NaN 表示"没算出来"，填 0 表示"算出来是 0 分"，这两件事完全不同。填 0 会让分数系统性偏低，而且你永远发现不了 RAGAS 其实一直在超时。**正确做法是 `skipna=True` 求均值 + 显式报出 NaN 比例。**

#### 问题二：中文效果

| 现象 | 原因 | 对策 |
|---|---|---|
| Faithfulness 的断言拆解把一句中文拆得七零八落 | RAGAS 内置 prompt 是英文的，对中文长句拆解不稳 | ① 用中文能力强的评判模型（DeepSeek / 通义）；② 让系统答案**本身写得更结构化**（分点、短句），拆解会准很多 |
| Answer Relevancy 系统性偏低 | embedding 中文能力不足 | 换 `bge-m3` / `bge-large-zh-v1.5`；**不同 embedding 的分数不可横向比较** |
| Context Precision 对中文长 chunk 判定不稳 | chunk 太长，LLM 难判"是否相关" | 把送进 RAGAS 的 context 截断到 1000~1500 字（`build_dataset.py` 的 `max_context_chars`） |

#### 问题三：成本

RAGAS 的成本比想象中高得多，因为**每个指标内部都是多次 LLM 调用**：

| 指标 | 每条样本的大致调用次数 | 说明 |
|---|---|---|
| Faithfulness | 2+ | 拆解 1 次 + 验证 1 次（断言多时可能分批） |
| Context Precision | 1 × K | **每个 chunk 判一次**，K=5 就是 5 次 |
| Context Recall | 1~2 | 拆解标准答案 + 逐条验证 |
| Answer Relevancy | 1 + embedding | 生成 3 个反推问题 + 4 次 embedding |
| Factual Correctness | 2+ | 双向断言比对 |

**300 条 × 6 个指标，总调用次数可能上千次**（前面的进度条 `1428/1428` 就是这么来的）。

| 省钱手段 | 效果 |
|---|---|
| **只跑 2~3 个真正要看的指标** | 最有效。日常只跑 `Faithfulness` + `FactualCorrectness` 就够了 |
| **在子集上跑**（分层抽样 100 条） | 成本降三分之二，结论基本一致 |
| **降低 K**（送进 RAGAS 的 context 数） | Context Precision 的成本与 K 成正比 |
| **embedding 用本地** | `SemanticSimilarity` 和 `ResponseRelevancy` 的 embedding 部分直接免费 |
| **降低频率**：每月一次而不是每日 | RAGAS 的定位是交叉验证，不是日常门禁 |

#### 问题四：指标波动

**同一份数据跑两次，`faithfulness` 差 0.03，正常吗？**

正常。来源有三：① LLM 判定本身的随机性（即使温度 0，服务端也可能有微小不确定性）；② 断言拆解的粒度不同；③ NaN 的样本集合不同导致均值分母变化。

**应对**：

1. **跑 3 次取中位数**（第 8.1 章 2.7 节的做法）；
2. **看中位数而不是均值**（中位数对离群值更稳）；
3. **设置"最小可信差异"**：经验上 RAGAS 指标差异 **< 0.03 不要下结论**；
4. **同时报有效样本数**：`n_valid` 变了，均值就没有可比性。

### 2.7 与自研 harness 做交叉验证

**这才是 RAGAS 在本书体系里的正式用途。**

```python
# file: evals/ragas_eval/compare.py
# 运行环境：Python 3.11
"""交叉验证：把 RAGAS 分数与自研 harness 分数逐题对齐，算相关性并找出两把尺子打架的题。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


def spearman(x: list[float], y: list[float]) -> float:
    """斯皮尔曼等级相关系数：只看排序一致性，不受量纲影响，适合比较两把不同的尺子。"""
    n = len(x)
    if n < 3:
        return 0.0

    def rank(v: list[float]) -> list[float]:
        """计算秩次，并列取平均秩。"""
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def compare(harness_csv: str, ragas_csv: str, out_md: str) -> None:
    """对齐两份结果，输出相关性与分歧清单。"""
    h = pd.read_csv(harness_csv)
    r = pd.read_csv(ragas_csv)
    df = h.merge(r, on="qid", suffixes=("_h", "_r"))
    if df.empty:
        print("两份结果没有交集，检查 qid 是否一致")
        return

    pairs = [("judge_weighted", "faithfulness"),
             ("judge_weighted", "factual_correctness(mode=f1)"),
             ("total", "factual_correctness(mode=f1)"),
             ("recall@5", "context_recall"),
             ("retrieval_score", "llm_context_precision_with_reference")]

    L = ["# 交叉验证：自研 harness vs RAGAS", "",
         f"- 可对齐样本：**{len(df)}** 条", "",
         "## 一、指标相关性（Spearman 等级相关）", "",
         "| harness 指标 | RAGAS 指标 | ρ | 解读 |", "|---|---|---|---|"]
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        sub = df[[a, b]].dropna()
        if len(sub) < 3:
            continue
        rho = spearman(sub[a].tolist(), sub[b].tolist())
        judge = ("✅ 高度一致" if rho >= 0.6 else
                 "⚠️ 中等，需关注" if rho >= 0.35 else "❌ **两把尺子结论相左，必须排查**")
        L.append(f"| {a} | {b} | {rho:.3f} | {judge} |")
    L += ["", "> ρ ≥ 0.6：两套口径基本一致，可以放心用自研分数决策。",
          "> ρ < 0.35：**必须人工抽 30 条看**，大概率是 rubric 与 RAGAS 的定义差异，"
          "也可能是自研 judge 存在自我偏好偏差。", ""]

    # 分歧清单：自研判高分、RAGAS 判低分（或反之）
    if "total" in df.columns and "faithfulness" in df.columns:
        df["_h_norm"] = df["total"] / 100.0
        gap = (df["_h_norm"] - df["faithfulness"]).abs()
        top = df.assign(gap=gap).nlargest(15, "gap")
        L += ["## 二、分歧最大的 15 条（人工复核优先级最高）", "",
              "| qid | harness 总分 | RAGAS faithfulness | 差距 | 问题 |", "|---|---|---|---|---|"]
        for _, row in top.iterrows():
            q = str(row.get("user_input", ""))[:28]
            L.append(f"| {row['qid']} | {row['total']} | {row['faithfulness']:.3f} "
                     f"| {row['gap']:.3f} | {q}… |")
        L += ["", "> **这 15 条是最有价值的复核对象**：两把尺子打架的地方，"
              "要么是你的 rubric 有洞，要么是 RAGAS 的定义不贴业务。两种都值得知道。", ""]

    Path(out_md).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)[:1500])
    print(f"\n已写入 {out_md}")


if __name__ == "__main__":
    a = sys.argv[1:]
    compare(a[0] if a else "runs/latest/scores.csv",
            a[1] if len(a) > 1 else "runs/latest/ragas/ragas_detail.csv",
            a[2] if len(a) > 2 else "runs/latest/cross_check.md")
```

**怎么用这份交叉验证结果**：

| ρ 区间 | 含义 | 动作 |
|---|---|---|
| **≥ 0.6** | 两套口径一致 | 放心用自研分数做决策，RAGAS 每月跑一次即可 |
| **0.35 ~ 0.6** | 中等一致 | 看分歧清单，多半是某一类题（如聚合题）两边定义不同 |
| **< 0.35** | **结论相左** | 停下来排查：① 自研 judge 是否与被测同模型（自我偏好）；② rubric 是否漏了关键维度；③ 是不是在针对自己的 rubric 过拟合 |

---

## 三、DeepEval：什么场景更合适

**DeepEval 的核心差异化是「pytest 风格的断言」。** 它把评测写成单元测试：

```python
# file: evals/deepeval_demo/test_rag_quality.py
# 运行环境：Python 3.11
# 依赖：pip install deepeval
# 运行：deepeval test run test_rag_quality.py
"""DeepEval 风格的评测：把质量断言写成 pytest 用例，适合『这一条绝不能退化』的场景。"""

from __future__ import annotations

import os

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


class DeepSeekModel(DeepEvalBaseLLM):
    """把 DeepSeek 接进 DeepEval（默认走 OpenAI，国内项目必须换）。

    具体基类方法签名以 DeepEval 官方文档为准，这里给出的是通用形态。
    """

    def __init__(self, model: str = "deepseek-chat"):
        from openai import OpenAI
        self.model_name = model
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com")

    def load_model(self):
        """返回底层客户端。"""
        return self.client

    def generate(self, prompt: str) -> str:
        """同步生成。"""
        r = self.client.chat.completions.create(
            model=self.model_name, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """异步生成（DeepEval 并发时会用）。"""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """模型名，会写进报告。"""
        return self.model_name


MODEL = DeepSeekModel()

# 自定义业务指标：用自然语言描述评分标准，DeepEval 会自动构造 judge prompt
MODEL_ACCURACY = GEval(
    name="型号参数准确性",
    criteria=("判断回答是否正确区分了 XJ-200 与 XJ-200-B3 的参数。"
              "如果把 B3 的参数（22 kN / 8 mm / 3600 rpm）安给标准版 XJ-200，"
              "或把标准版参数（18 kN / 6 mm / 4000 rpm）安给 B3，直接判 0 分。"),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                       LLMTestCaseParams.EXPECTED_OUTPUT],
    model=MODEL,
    threshold=0.8,
)

CASES = [
    ("XJ-200 液压卡盘夹紧力上限是多少？",
     "XJ-200 液压卡盘夹紧力上限为 18 kN，超过可能导致卡盘变形。",
     ["XJ-200 液压卡盘夹紧力上限为 18 kN，卡爪行程 6 mm，最高转速 4000 rpm。"]),
    ("XJ-200-B3 的卡爪行程是多少？",
     "XJ-200-B3 卡爪行程为 8 mm，最高转速 3600 rpm。",
     ["XJ-200-B3 为重载改型，夹紧力上限 22 kN，卡爪行程 8 mm，最高转速 3600 rpm。"]),
]


@pytest.mark.parametrize("question,answer,contexts", CASES)
def test_rag_quality(question: str, answer: str, contexts: list[str]):
    """对每条用例断言三个指标全部达标，任一不达标即测试失败。"""
    case = LLMTestCase(input=question, actual_output=answer,
                       expected_output=contexts[0], retrieval_context=contexts)
    assert_test(case, [
        FaithfulnessMetric(threshold=0.8, model=MODEL),
        AnswerRelevancyMetric(threshold=0.7, model=MODEL),
        MODEL_ACCURACY,
    ])
```

**RAGAS vs DeepEval vs 自研 harness 三方对比**：

| 维度 | RAGAS | DeepEval | 自研 harness（第 8.3 章） |
|---|---|---|---|
| **心智模型** | 批量评测 + 指标报表 | **单元测试断言** | 完整评测流水线 |
| **适合的问题** | 「这套 RAG 整体质量如何」 | 「**这 20 条绝对不能退化**」 | 「这次改动能不能上线」 |
| **自定义指标** | 需要写 Metric 子类 | **`GEval` 用自然语言描述即可**，很方便 | 完全自由 |
| **输出形态** | DataFrame / 报表 | pytest 报告（红绿） | Markdown + CSV + diff + 门禁 |
| **接 CI** | 需要自己包一层 | **原生**（就是 pytest） | 原生（有退出码和门禁） |
| **业务规则（一票否决）** | 难 | `GEval` 能表达，但仍是 LLM 判定 | **确定性规则，最可靠** |
| **成本控制** | 弱（无缓存） | 弱 | **强（磁盘缓存）** |
| **国产模型接入** | 容易（Langchain wrapper） | 中等（要写 `DeepEvalBaseLLM` 子类） | 原生 |

**选型建议**：

| 场景 | 选谁 |
|---|---|
| 想快速摸底一个新 RAG 系统的质量 | **RAGAS** |
| 有一小批"绝对不能坏"的核心用例，想写成测试 | **DeepEval** |
| 要做上线门禁、回归保护、归因分析 | **自研 harness** |
| 三者都想要 | 自研 harness 做主力 + DeepEval 补一层核心用例的 pytest 断言 + RAGAS 每月交叉验证 |

---

## 四、OpenCompass 与 lm-evaluation-harness：通用能力回归

RAGAS 和 DeepEval 都是"业务评测"，回答的是「**我这套 RAG 答我们家的题答得好不好**」。但第 5 章做完 LoRA 微调后，你还需要回答另一个完全不同的问题：

> 「微调把模型的**基础能力**搞坏了吗？它还会做数学题、还会写代码、还听得懂指令吗？」

这个问题不能用金标集回答——**金标集全是华成机电的售后题，模型在上面分数越高，越掩盖通用能力的退化**。必须用公开基准。

### 4.1 两者的分工

| | OpenCompass | lm-evaluation-harness | 自研 harness / RAGAS |
|---|---|---|---|
| **回答的问题** | 通用能力有没有退化 | 通用能力有没有退化 | 业务质量好不好 |
| **数据集** | C-Eval / CMMLU / MMLU / GSM8K / HumanEval / IFEval 等上百个 | 同类，任务库更偏英文学术 | 你自己的金标集 |
| **中文支持** | **强**（国内团队维护，中文榜单齐全） | 中等（C-Eval/CMMLU 也有） | 原生 |
| **接 LoRA adapter** | 模型配置里指定 peft 路径，或先合并（第 5.5 章） | `--model_args ...,peft=<adapter_dir>` | 与部署同一条链路 |
| **跑一次耗时** | 几小时（数据集多时更久） | 十几分钟~几小时（看选了几个任务） | 10~40 分钟 |
| **频率** | **每次微调后一次** | 每次微调后一次（与上者选一个即可） | 每次提交 / 每日 |
| **能不能做门禁** | 能，但只做"通用能力不退化"这一条 | 同左 | **主力门禁** |

**一句话分工**：

```mermaid
flowchart LR
    FT["第 5 章<br/>LoRA 微调产出 adapter"] --> G1["通用基准回归<br/>OpenCompass / lm-eval<br/>问：基础能力掉了吗"]
    FT --> G2["业务金标集评测<br/>自研 harness<br/>问：业务题答得好吗"]
    G1 --> D{"两个门禁都过？"}
    G2 --> D
    D -- 是 --> REL["进入第 5.5 章合并量化 → 部署"]
    D -- "通用掉太多" --> F1["回 5.3 章：降 lr / 减 epoch / 混通用数据"]
    D -- "业务不达标" --> F2["回 5.2 章：补数据 / 改 rubric"]
    style REL fill:#f0fff4,stroke:#2f855a
```

### 4.2 lm-evaluation-harness：命令最简单，优先用它

```bash
# 安装（固定版本，避免任务定义变动导致分数不可比）
uv pip install "lm-eval[hf]==0.4.5"
# pip 等价：pip install "lm-eval[hf]==0.4.5"
```

```bash
# ---------- ① 基座模型跑基线 ----------
# 关键：把基线结果存下来，后面所有对比都相对它
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-valid,cmmlu,gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  --apply_chat_template \
  --output_path evals/general/base_qwen25_7b \
  --log_samples

# ---------- ② 带 LoRA adapter 跑对照 ----------
# peft= 直接指向第 5.3 章产出的 adapter 目录，不需要先合并
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=finetune/runs/20260318-2104_qlora-r16,dtype=bfloat16 \
  --tasks ceval-valid,cmmlu,gsm8k \
  --num_fewshot 5 \
  --batch_size auto \
  --apply_chat_template \
  --output_path evals/general/ft_qlora_r16 \
  --log_samples

# ---------- ③ 用 vLLM 后端加速（数据集多时必用）----------
lm_eval --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096 \
  --tasks ceval-valid,gsm8k --batch_size auto \
  --output_path evals/general/base_vllm
```

```text
hf (pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16), gen_kwargs: (None), num_fewshot: 5, batch_size: auto
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|ceval-valid   |      2|none  |     5|acc       |↑  |0.7238|±  |0.0121|
|cmmlu         |      2|none  |     5|acc       |↑  |0.7015|±  |0.0042|
|gsm8k         |      3|flexible-extract| 5|exact_match|↑|0.7854|± |0.0113|
（示例性数据，用于演示对比方法。不同版本、few-shot 设置、答案抽取规则下数值会有明显差异，必须自行复现）
```

**读这张表只看一件事：`Value ± Stderr`。** `Stderr` 是标准误——**基座 0.7238±0.0121，微调后 0.7150，差 0.88 个点，小于一个标准误，等于没变化。** 不要把这种波动当成"退化"去汇报。

把两次结果做成一张对照表（这是给老板看的那张）：

```python
# file: evals/general/compare_general.py
# 运行环境：Python 3.11
"""对比基座与微调模型的通用基准分数，标出是否超出噪声范围。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run_dir: str) -> dict[str, tuple[float, float]]:
    """读 lm-eval 的 results json，返回 {task: (value, stderr)}。

    lm-eval 0.4.x 会在 output_path 下生成 results_*.json，字段名以官方文档为准。
    """
    files = sorted(Path(run_dir).rglob("results*.json"))
    if not files:
        raise FileNotFoundError(f"{run_dir} 下没有 results*.json")
    data = json.loads(files[-1].read_text(encoding="utf-8"))["results"]
    out: dict[str, tuple[float, float]] = {}
    for task, m in data.items():
        val = next((v for k, v in m.items()
                    if k.startswith(("acc,", "exact_match,", "acc_norm,"))
                    and isinstance(v, (int, float))), None)
        err = next((v for k, v in m.items()
                    if "stderr" in k and isinstance(v, (int, float))), 0.0)
        if val is not None:
            out[task] = (float(val), float(err))
    return out


def main(base_dir: str, ft_dir: str, out_md: str) -> None:
    """生成 Markdown 对照表，用 2 倍标准误作为噪声阈值。"""
    base, ft = load(base_dir), load(ft_dir)
    L = ["# 通用能力回归对照", "",
         "| 基准 | 基座 | 微调后 | 差值 | 2×Stderr | 判定 |", "|---|---|---|---|---|---|"]
    worst = 0.0
    for task in sorted(set(base) & set(ft)):
        b, be = base[task]
        f, fe = ft[task]
        d = f - b
        noise = 2 * max(be, fe)
        if abs(d) <= noise:
            verdict = "持平（在噪声内）"
        elif d > 0:
            verdict = "✅ 上升"
        else:
            verdict = "❌ **显著退化**"
        worst = min(worst, d)
        L.append(f"| {task} | {b:.4f} ± {be:.4f} | {f:.4f} ± {fe:.4f} "
                 f"| {d:+.4f} | ±{noise:.4f} | {verdict} |")
    L += ["", f"**最大退化幅度：{worst:+.4f}**", "",
          "> 门禁建议：任一基准显著退化超过 0.03（3 个百分点）即阻断发布；",
          "> 0.01~0.03 区间给 warn，需要人工确认是否可接受。", ""]
    Path(out_md).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    # 退出码给 CI 用
    raise SystemExit(1 if worst < -0.03 else 0)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "evals/general/base_qwen25_7b",
         a[1] if len(a) > 1 else "evals/general/ft_qlora_r16",
         a[2] if len(a) > 2 else "evals/general/general_regression.md")
```

```text
# 通用能力回归对照

| 基准 | 基座 | 微调后 | 差值 | 2×Stderr | 判定 |
|---|---|---|---|---|---|
| ceval-valid | 0.7238 ± 0.0121 | 0.7150 ± 0.0122 | -0.0088 | ±0.0244 | 持平（在噪声内） |
| cmmlu | 0.7015 ± 0.0042 | 0.6924 ± 0.0043 | -0.0091 | ±0.0086 | ❌ **显著退化** |
| gsm8k | 0.7854 ± 0.0113 | 0.7623 ± 0.0117 | -0.0231 | ±0.0234 | 持平（在噪声内） |

**最大退化幅度：-0.0231**
（示例性数据，需自行复现）
```

> 注意 `cmmlu` 那一行：**差值只有 0.9 个点，但因为样本量大、标准误小，它在统计上是"显著退化"**；而 `gsm8k` 差了 2.3 个点反而算持平。**这就是为什么不能只看差值大小。**

### 4.3 OpenCompass：中文榜单更全，适合做正式报告

```bash
# 安装（建议单独 venv，它的依赖与训练环境容易冲突）
uv venv .venv-oc --python 3.11 && source .venv-oc/bin/activate
uv pip install -U opencompass
# 数据集需要单独下载，具体命令以官方文档为准（通常是下载 OpenCompassData 压缩包解压到 ./data）
```

```bash
# ---------- 方式一：命令行快速跑 ----------
opencompass --models hf_qwen2_5_7b_instruct \
            --datasets ceval_gen cmmlu_gen gsm8k_gen \
            --max-num-workers 4 \
            --work-dir evals/general/oc_base \
            --debug          # 第一次跑务必加 --debug，否则报错会被子进程吞掉

# ---------- 方式二：配置文件（可复现，推荐正式用）----------
opencompass evals/general/oc_eval_huacheng.py -w evals/general/oc_ft --reuse latest
```

```python
# file: evals/general/oc_eval_huacheng.py
# 运行环境：Python 3.11（OpenCompass 独立 venv）
"""OpenCompass 配置：同时评测基座与 LoRA 微调模型，跑中文通用基准。

注意：OpenCompass 的配置 API 在版本间有变化，导入路径与字段名以你安装版本的官方文档为准。
本文件展示的是通用结构：datasets 列表 + models 列表。
"""

from mmengine.config import read_base

with read_base():
    # 从官方配置库引入数据集定义（gen = 生成式评测，ppl = 困惑度式评测）
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = [*ceval_datasets, *cmmlu_datasets, *gsm8k_datasets]

_common = dict(
    type="HuggingFacewithChatTemplate",   # 用 chat template，与线上推理保持一致
    max_out_len=512,
    batch_size=8,
    run_cfg=dict(num_gpus=1),
)

models = [
    # 基线：原始基座
    dict(abbr="qwen2.5-7b-base", path="Qwen/Qwen2.5-7B-Instruct", **_common),
    # 对照：合并后的微调模型（LoRA 合并见第 5.5 章；也可用 peft_path 直接挂 adapter）
    dict(abbr="qwen2.5-7b-huacheng-lora",
         path="release/qwen25-7b-huacheng-merged/v1.2.0", **_common),
]

work_dir = "evals/general/oc_ft"
```

```text
dataset                      version  metric    mode  qwen2.5-7b-base  qwen2.5-7b-huacheng-lora
---------------------------  -------  --------  ----  ---------------  ------------------------
ceval-computer_network       db9ce2   accuracy  gen             68.42                     67.89
ceval-operating_system       1c2571   accuracy  gen             73.68                     73.68
ceval-college_programming    4ca32a   accuracy  gen             70.27                     68.92
cmmlu-chinese_civil_service  ...      accuracy  gen             65.00                     63.75
gsm8k                        1d7fe4   accuracy  gen             78.54                     76.23
---------------------------  -------  --------  ----  ---------------  ------------------------
（示例性数据，需自行复现。跨版本、跨框架的分数不可直接比较）
```

### 4.4 三条容易踩的坑

| 坑 | 说明 |
|---|---|
| **拿自己跑的分数去和榜单/论文比** | few-shot 数量、答案抽取正则、chat template、是否 CoT，任何一项不同，分数就能差 5~10 个点。**基准分数只在"同一份配置内部、同一次运行的不同模型之间"可比。** |
| **只跑一次就下结论** | 生成式评测（`_gen`）受解码随机性影响，`temperature` 没设 0 时波动明显。正式报告要固定 `temperature=0` 并记录版本号。 |
| **微调后没跑基座基线** | 没有基线，"下降了多少"就无从谈起。**基座基线必须与微调模型在同一套配置、同一台机器上跑**，然后存起来，长期复用。 |

> **本书的实践取舍**：日常只跑 `lm_eval` 的 `ceval-valid + cmmlu + gsm8k` 三项（约 20~40 分钟，单卡 4090，实测环境同第 5.3 章），当作微调后的"通用能力体检"；需要对外出正式报告时再跑一次 OpenCompass 全量。**通用基准不进每日流水线**——它太慢，而且日常改的是 RAG 而不是模型权重。

---

## 五、把评测接进研发流程（本章核心）

前面三章把"怎么评"讲完了。但**只要评测还需要人手动敲命令，它就一定会被跳过**——赶版本的那一周，没人会记得跑 300 条回归。

评测必须变成**流程的一部分**：改了 prompt 就自动跑冒烟，合进主干就自动跑回归，发版前必须有全量报告，上线后自动看线上指标。

### 5.1 完整流水线设计

```mermaid
flowchart TD
    subgraph DEV["① 开发阶段"]
        D1["工程师改代码<br/>prompt / 检索参数 / 语料"] --> D2["本地：python -m harness run<br/>--sample 30 --no-judge<br/>（省钱模式，1 分钟）"]
        D2 --> D3["git push 到特性分支"]
    end

    D3 --> P1

    subgraph PR["② 提交阶段（PR 触发，必须 5 分钟内出结果）"]
        P1["CI: 冒烟评测<br/>golden_smoke 30 条<br/>命中缓存后约 2~4 分钟"] --> P2{"block 门禁<br/>全过？"}
        P2 -- 否 --> P3["❌ PR 红灯<br/>机器人评论贴出失败项 + diff"]
        P2 -- 是 --> P4["✅ 绿灯 + 机器人评论贴分数表"]
        P3 --> D1
    end

    P4 --> M1["③ 人工 Code Review → 合并到 main"]

    M1 --> R1

    subgraph REG["④ 每日回归（定时 02:00）"]
        R1["golden_regression 300 条<br/>跑 3 次取中位数<br/>约 25~40 分钟"] --> R2["写入指标库<br/>eval_run / eval_item 表"]
        R2 --> R3{"与 7 日中位数<br/>对比是否退化？"}
        R3 -- 退化且显著 --> R4["企业微信/钉钉告警<br/>自动建 badcase 工单"]
        R3 -- 正常 --> R5["更新看板趋势图"]
    end

    R5 --> V1

    subgraph REL["⑤ 发版前（打 tag 触发，人工确认）"]
        V1["golden_full 1000+ 条<br/>跑 3 次取中位数<br/>+ 通用基准回归<br/>+ RAGAS 交叉验证<br/>约 2~3 小时"] --> V2{"发版门禁<br/>（红线见 5.6）"}
        V2 -- 否 --> V3["❌ 阻断发版"]
        V2 -- 是 --> V4["生成发版评测报告<br/>归档到 evals/runs/release/"]
    end

    V4 --> O1

    subgraph ONLINE["⑥ 上线后（持续）"]
        O1["影子流量对比<br/>（新旧版本同时跑真实流量）"] --> O2["AB 实验<br/>5% → 20% → 100%"]
        O2 --> O3["在线监控：<br/>点赞点踩 / LLM 抽样打分<br/>/ 追问率 / 转人工率 / P95 / 成本"]
        O3 --> O4{"线上指标劣化？"}
        O4 -- 是 --> O5["回滚 + badcase 归因"]
        O4 -- 否 --> O6["badcase 回流金标集<br/>（→ 第 8.2 章）"]
    end

    O5 --> D1
    O6 -.->|"金标集扩充"| R1

    style P4 fill:#f0fff4,stroke:#2f855a
    style V4 fill:#f0fff4,stroke:#2f855a
    style P3 fill:#fed7d7,stroke:#c53030
    style V3 fill:#fed7d7,stroke:#c53030
    style O5 fill:#fed7d7,stroke:#c53030
```

### 5.2 四道闸口的设计参数

**流水线设计最容易犯的错，是四道闸口用同一套配置。** 冒烟要快（分钟级），回归要稳（统计可信），发版要全（覆盖所有分类），在线要省（不能给每个请求都调 judge）。

| 闸口 | 触发时机 | 数据集 | 跑几次 | 要不要 judge | 目标耗时 | 门禁强度 | 谁负责看 |
|---|---|---|---|---|---|---|---|
| **① 本地冒烟** | 工程师自觉 | `golden_smoke` 30 条 | 1 | 否（`--no-judge`） | **< 1 分钟** | 无（自查） | 本人 |
| **② PR 冒烟** | `pull_request` | `golden_smoke` 30 条 | 1 | 是（缓存命中率高） | **< 5 分钟** | 只卡"一票否决 / 注入越狱 / 崩溃" | PR 作者 |
| **③ 每日回归** | 定时 02:00 + `push main` | `golden_regression` 300 条 | **3 次取中位数** | 是 | < 40 分钟 | 卡"相对 7 日中位数的退化" | 值班工程师 |
| **④ 发版全量** | 打 tag / 手动 | `golden_full` 1000+ 条 + 通用基准 + RAGAS | **3 次取中位数** | 是 | < 3 小时 | **全部红线**（5.6 节） | 技术负责人 + 业务方 |

> **为什么 PR 阶段不卡"分数下降"**：30 条样本的 95% 置信区间宽达 ±15 个点以上（第 8.1 章 2.7 节的结论）。**用 30 条去卡 2 分的下降，只会制造大量假警报，最后所有人学会 `--no-verify` 跳过它。** PR 阶段只卡"确定性的、不需要统计的"红线：一票否决、注入越狱、程序崩溃、有效样本数不足。

### 5.3 GitHub Actions 完整配置

四个 workflow 文件，职责分开。先给最关键的 PR 冒烟：

```yaml
# file: .github/workflows/eval-smoke.yml
# PR 触发的冒烟评测：只跑 30 条，5 分钟内必须出结果，结果作为 PR 评论
name: eval-smoke

on:
  pull_request:
    # 只有这些路径变了才跑评测，改文档不触发
    paths:
      - 'rag/**'
      - 'agents/**'
      - 'tools/**'
      - 'core/**'
      - 'prompts/**'
      - 'evals/**'
      - 'data/knowledge/**'
      - '.github/workflows/eval-smoke.yml'

# 同一个 PR 连续推送时，取消上一次未完成的运行，省额度也省时间
concurrency:
  group: eval-smoke-${{ github.event.pull_request.number }}
  cancel-in-progress: true

permissions:
  contents: read
  pull-requests: write        # 需要写 PR 评论

jobs:
  smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 15       # 硬超时，防止 judge 卡住把 runner 挂死
    env:
      DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      MILVUS_URI: ${{ secrets.MILVUS_URI_STAGING }}
      MILVUS_COLLECTION: huacheng_kb
      MILVUS_DIM: '1024'
      PYTHONPATH: ${{ github.workspace }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0      # 需要历史记录来做"受影响用例"分析

      - name: Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install uv
        run: pipx install uv

      # ---------- 依赖缓存 ----------
      - name: Cache venv
        uses: actions/cache@v4
        with:
          path: .venv
          key: venv-${{ runner.os }}-py311-${{ hashFiles('requirements*.txt', 'pyproject.toml') }}
          restore-keys: venv-${{ runner.os }}-py311-

      - name: Install deps
        run: |
          uv venv .venv --python 3.11
          . .venv/bin/activate
          uv pip install -r requirements.txt -r evals/requirements-eval.txt

      # ---------- 评测缓存（省钱的关键）----------
      # 系统输出与 judge 结果都按内容哈希缓存（第 8.3 章 runner.py / judge.py）。
      # 只改了检索参数时，judge 缓存仍大量命中，一次冒烟的 API 成本可以压到几分钱。
      - name: Restore eval cache
        uses: actions/cache@v4
        with:
          path: evals/.cache
          key: evalcache-${{ github.sha }}
          restore-keys: |
            evalcache-${{ github.event.pull_request.base.sha }}
            evalcache-

      # ---------- 只跑受影响的用例 ----------
      - name: Select affected cases
        id: affected
        run: |
          . .venv/bin/activate
          git diff --name-only origin/${{ github.base_ref }}...HEAD > /tmp/changed.txt
          echo "---- changed files ----"; cat /tmp/changed.txt
          python evals/pipeline/select_cases.py \
            --changed /tmp/changed.txt \
            --dataset evals/datasets/golden_smoke.jsonl \
            --out /tmp/qids.txt
          echo "n_cases=$(wc -l < /tmp/qids.txt)" >> "$GITHUB_OUTPUT"

      - name: Run smoke eval
        id: eval
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          set +e
          python -m harness run \
            --config configs/eval.yaml \
            --system s3_opt_rag \
            --dataset datasets/golden_smoke.jsonl \
            --compare auto \
            2>&1 | tee /tmp/eval.log
          echo "exit_code=${PIPESTATUS[0]}" >> "$GITHUB_OUTPUT"
          set -e
          echo "run_dir=$(ls -dt runs/*/ | head -1)" >> "$GITHUB_OUTPUT"

      - name: Render PR comment
        if: always()
        run: |
          . .venv/bin/activate
          python evals/pipeline/pr_comment.py \
            --run-dir "evals/${{ steps.eval.outputs.run_dir }}" \
            --log /tmp/eval.log \
            --affected "${{ steps.affected.outputs.n_cases }}" \
            --out /tmp/comment.md

      # 用固定标记更新同一条评论，而不是每次推送都新增一条
      - name: Post PR comment
        if: always()
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          header: eval-smoke
          path: /tmp/comment.md

      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-smoke-${{ github.sha }}
          path: |
            evals/runs/**/report.md
            evals/runs/**/diff.md
            evals/runs/**/scores.csv
            evals/runs/**/summary.json
          retention-days: 14

      - name: Fail if gate blocked
        if: steps.eval.outputs.exit_code != '0'
        run: |
          echo "::error::冒烟评测门禁未通过，详见 PR 评论与 artifacts"
          exit 1
```

每日回归（定时任务，跑 3 次取中位数并入库）：

```yaml
# file: .github/workflows eval-nightly.yml  →  实际路径 .github/workflows/eval-nightly.yml
name: eval-nightly

on:
  schedule:
    - cron: '0 18 * * *'      # UTC 18:00 = 北京时间 02:00
  push:
    branches: [main]
    paths: ['rag/**', 'agents/**', 'prompts/**', 'data/knowledge/**']
  workflow_dispatch:           # 支持手动触发
    inputs:
      repeats:
        description: '重复运行次数'
        default: '3'

jobs:
  regression:
    # 需要连内网 Milvus，用自托管 runner；纯在线 API 的项目可用 ubuntu-latest
    runs-on: [self-hosted, linux, gpu]
    timeout-minutes: 90
    env:
      DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      MILVUS_URI: ${{ secrets.MILVUS_URI_STAGING }}
      EVAL_DB_DSN: ${{ secrets.EVAL_DB_DSN }}
      PYTHONPATH: ${{ github.workspace }}
    steps:
      - uses: actions/checkout@v4

      - name: Install deps
        run: |
          uv venv .venv --python 3.11 && . .venv/bin/activate
          uv pip install -r requirements.txt -r evals/requirements-eval.txt

      # 跑 N 次。注意：必须 --no-cache，否则第 2、3 次直接命中缓存，
      # 得到三个完全一样的分数，"取中位数"就失去了意义。
      - name: Run regression x N
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          N="${{ github.event.inputs.repeats || '3' }}"
          for i in $(seq 1 "$N"); do
            echo "===== 第 $i/$N 次 ====="
            python -m harness run \
              --config configs/eval.yaml \
              --system s3_opt_rag \
              --dataset datasets/golden_regression.jsonl \
              --no-cache \
              --compare none \
              --fail-on-gate false
          done

      # 防抖动门禁：聚合这 N 次结果，与历史 7 日中位数比较 + 显著性检验
      - name: Gate (median of N + significance)
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          python pipeline/gate.py \
            --runs-dir runs --system s3_opt_rag \
            --repeats "${{ github.event.inputs.repeats || '3' }}" \
            --baseline-days 7 \
            --config configs/gate.yaml \
            --out runs/latest_gate.md

      - name: Write metrics to DB
        if: always()
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          python pipeline/ingest_metrics.py --runs-dir runs --repeats 3

      - name: Notify on failure
        if: failure()
        env:
          WECOM_WEBHOOK: ${{ secrets.WECOM_WEBHOOK }}
        run: |
          . .venv/bin/activate
          # 注意：Python 必须写成分号一行式。YAML 块标量里出现顶格的多行 Python
          # 会直接让整个 workflow 文件解析失败 —— 这是最常见的 CI 配置错误之一。
          PAYLOAD=$(python -c 'import json,pathlib; md=pathlib.Path("evals/runs/latest_gate.md").read_text(encoding="utf-8")[:1800]; print(json.dumps({"msgtype":"markdown","markdown":{"content":"## 每日回归门禁未通过\n"+md}}, ensure_ascii=False))')
          curl -s -X POST "$WECOM_WEBHOOK" -H 'Content-Type: application/json' -d "$PAYLOAD"
```

发版全量（打 tag 触发）：

```yaml
# file: .github/workflows/eval-release.yml
name: eval-release

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  full:
    runs-on: [self-hosted, linux, gpu]
    timeout-minutes: 300
    environment: release          # 绑定 GitHub Environment，可要求人工 approve
    env:
      DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      EVAL_DB_DSN: ${{ secrets.EVAL_DB_DSN }}
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: uv venv .venv --python 3.11 && . .venv/bin/activate && uv pip install -r requirements.txt -r evals/requirements-eval.txt

      - name: Full eval x3
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          for i in 1 2 3; do
            python -m harness run --config configs/eval.yaml --system s3_opt_rag \
              --dataset datasets/golden_full.jsonl --no-cache --compare none --fail-on-gate false
          done

      - name: RAGAS cross-check
        working-directory: evals
        run: . ../.venv/bin/activate && python ragas_eval/run_ragas.py --sample 150 && python ragas_eval/compare.py

      - name: General benchmark regression
        run: |
          . .venv/bin/activate
          bash evals/general/run_lm_eval.sh
          python evals/general/compare_general.py

      - name: Release gate (all red lines)
        working-directory: evals
        run: |
          . ../.venv/bin/activate
          python pipeline/gate.py --runs-dir runs --system s3_opt_rag --repeats 3 \
            --baseline-tag "$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo '')" \
            --config configs/gate_release.yaml \
            --out runs/release_gate.md

      - name: Archive report
        uses: actions/upload-artifact@v4
        with:
          name: release-eval-${{ github.ref_name }}
          path: evals/runs/**
          retention-days: 365
```

### 5.4 GitLab CI 对照

同一套逻辑在 GitLab CI 里的写法（国内团队用 GitLab 的更多）：

```yaml
# file: .gitlab-ci.yml
stages: [smoke, regression, release]

variables:
  PYTHON_VERSION: "3.11"
  PYTHONPATH: "$CI_PROJECT_DIR"
  MILVUS_COLLECTION: "huacheng_kb"
  MILVUS_DIM: "1024"
  GIT_DEPTH: "0"                       # 需要完整历史做受影响用例分析

# ---------- 可复用片段 ----------
.py_setup: &py_setup
  image: python:3.11-slim
  before_script:
    - pip install -q uv
    - uv venv .venv --python 3.11
    - . .venv/bin/activate
    - uv pip install -q -r requirements.txt -r evals/requirements-eval.txt

.eval_cache: &eval_cache
  cache:
    # 按分支分开缓存；key: files 让依赖没变时直接复用
    - key:
        files: [requirements.txt, evals/requirements-eval.txt]
      paths: [.venv/]
    - key: "evalcache-$CI_COMMIT_REF_SLUG"
      paths: [evals/.cache/]
      policy: pull-push

# ---------- ① MR 冒烟 ----------
eval:smoke:
  stage: smoke
  <<: [*py_setup, *eval_cache]
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      changes: [rag/**/*, agents/**/*, tools/**/*, core/**/*, prompts/**/*, evals/**/*, data/knowledge/**/*]
  timeout: 15 minutes
  script:
    - git diff --name-only "origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME...HEAD" > /tmp/changed.txt
    - python evals/pipeline/select_cases.py --changed /tmp/changed.txt
        --dataset evals/datasets/golden_smoke.jsonl --out /tmp/qids.txt
    - cd evals && python -m harness run --config configs/eval.yaml --system s3_opt_rag
        --dataset datasets/golden_smoke.jsonl --compare auto | tee /tmp/eval.log
  after_script:
    # 把报告贴成 MR 评论（用 GitLab Notes API）
    - . .venv/bin/activate
    - python evals/pipeline/pr_comment.py --run-dir "$(ls -dt evals/runs/*/ | head -1)"
        --log /tmp/eval.log --out /tmp/comment.md --platform gitlab
    - |
      if [ -n "$CI_MERGE_REQUEST_IID" ]; then
        curl -s --request POST \
          --header "PRIVATE-TOKEN: $GITLAB_BOT_TOKEN" \
          --header "Content-Type: application/json" \
          --data "$(python -c 'import json,pathlib;print(json.dumps({"body":pathlib.Path("/tmp/comment.md").read_text(encoding="utf-8")},ensure_ascii=False))')" \
          "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
      fi
  artifacts:
    when: always
    expire_in: 2 weeks
    paths: [evals/runs/]
    reports:
      # 让 GitLab 在 MR 页面直接显示指标变化（metrics.txt 是 Prometheus 文本格式）
      metrics: evals/runs/metrics.txt

# ---------- ② 每日回归 ----------
eval:nightly:
  stage: regression
  <<: [*py_setup, *eval_cache]
  tags: [gpu, self-hosted]
  rules:
    - if: '$CI_PIPELINE_SOURCE == "schedule"'
    - if: '$CI_COMMIT_BRANCH == "main"'
      changes: [rag/**/*, agents/**/*, prompts/**/*, data/knowledge/**/*]
  timeout: 90 minutes
  script:
    - cd evals
    - for i in 1 2 3; do python -m harness run --config configs/eval.yaml --system s3_opt_rag
        --dataset datasets/golden_regression.jsonl --no-cache --compare none --fail-on-gate false; done
    - python pipeline/gate.py --runs-dir runs --system s3_opt_rag --repeats 3
        --baseline-days 7 --config configs/gate.yaml --out runs/latest_gate.md
    - python pipeline/ingest_metrics.py --runs-dir runs --repeats 3
  artifacts: {when: always, expire_in: 30 days, paths: [evals/runs/]}

# ---------- ③ 发版全量 ----------
eval:release:
  stage: release
  <<: *py_setup
  tags: [gpu, self-hosted]
  rules:
    - if: '$CI_COMMIT_TAG =~ /^v/'
  when: manual                         # 人工点一下才跑，防止误触发烧几百块
  allow_failure: false
  timeout: 5 hours
  script:
    - cd evals
    - for i in 1 2 3; do python -m harness run --config configs/eval.yaml --system s3_opt_rag
        --dataset datasets/golden_full.jsonl --no-cache --compare none --fail-on-gate false; done
    - python ragas_eval/run_ragas.py --sample 150 && python ragas_eval/compare.py
    - python pipeline/gate.py --runs-dir runs --system s3_opt_rag --repeats 3
        --config configs/gate_release.yaml --out runs/release_gate.md
  artifacts: {when: always, expire_in: 1 year, paths: [evals/runs/]}
```

**两个平台的关键差异**：

| | GitHub Actions | GitLab CI |
|---|---|---|
| 路径过滤 | `on.pull_request.paths` | `rules: changes:` |
| 取消旧运行 | `concurrency.cancel-in-progress` | `interruptible: true` + 项目设置 |
| 缓存 | `actions/cache@v4`，key + restore-keys | `cache.key.files` + `policy` |
| 评论 PR | 第三方 action（如 sticky-comment） | Notes API + `PRIVATE-TOKEN` |
| 人工审批 | `environment:` + protection rules | `when: manual` |
| 指标展示 | 只能靠评论/artifact | **原生 `reports.metrics`**，MR 页面直接显示 diff |

### 5.5 只跑受影响的用例

300 条回归在 PR 阶段太慢，但"只跑 30 条冒烟"又可能漏掉正好被这次改动影响的那一类题。**折中方案：根据改动的文件，映射出受影响的题目分类，从回归集里多抽一部分进来。**

```python
# file: evals/pipeline/select_cases.py
# 运行环境：Python 3.11
"""根据本次改动的文件，选出需要评测的用例 qid 列表。

映射规则是人工维护的 —— 这是一份「代码模块 → 业务能力」的对照表，
它比任何自动依赖分析都准，因为只有你知道 rerank 改了会影响哪类题。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

# ---------- 人工维护的影响面映射 ----------
# key: 文件 glob；value: 受影响的金标集维度（category / question_type / tags）
IMPACT_MAP: list[tuple[str, dict]] = [
    # 检索链路改动 → 所有需要引用知识的题
    ("rag/retriever*.py",      {"needs_context": True}),
    ("rag/rerank*.py",         {"needs_context": True, "difficulties": ["complex", "edge"]}),
    ("rag/chunk*.py",          {"needs_context": True}),        # 切分变了，chunk id 全变
    ("rag/query_rewrite*.py",  {"question_types": ["vague", "multi_hop"]}),
    ("rag/hybrid*.py",         {"needs_context": True}),
    # 生成与 prompt
    ("prompts/answer*.jinja",  {}),                              # 空 dict = 全量
    ("prompts/refuse*.jinja",  {"question_types": ["refusal", "out_of_scope"]}),
    ("prompts/clarify*.jinja", {"question_types": ["vague"]}),
    # Agent 与工具
    ("agents/**/*.py",         {"categories": ["工单处理", "多步推理"]}),
    ("tools/order*.py",        {"categories": ["工单处理"]}),
    ("tools/spare_part*.py",   {"categories": ["备件查询"]}),
    # 语料与知识库
    ("data/knowledge/**",      {"needs_context": True}),
    # 安全相关一律全跑
    ("core/guard*.py",         {}),
    ("rag/*.py",               {"needs_context": True}),
]

# 这些改动一律触发全量（宁可慢，不能漏）
FULL_RUN_PATTERNS = [
    "core/config.py", "core/llm.py", "evals/harness/**",
    "requirements*.txt", "pyproject.toml", "prompts/system*.jinja",
]


def match(path: str, patterns: list[str]) -> bool:
    """glob 匹配任一模式。"""
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(path, f"*/{p}") for p in patterns)


def collect_filters(changed: list[str]) -> tuple[bool, dict]:
    """返回 (是否全量, 合并后的筛选条件)。"""
    if any(match(c, FULL_RUN_PATTERNS) for c in changed):
        return True, {}
    merged: dict = {"categories": set(), "question_types": set(),
                    "difficulties": set(), "needs_context": False}
    hit = False
    for c in changed:
        for pat, flt in IMPACT_MAP:
            if fnmatch.fnmatch(c, pat) or fnmatch.fnmatch(c, f"*/{pat}"):
                hit = True
                if not flt:                      # 空条件 = 全量
                    return True, {}
                for k in ("categories", "question_types", "difficulties"):
                    merged[k].update(flt.get(k, []))
                merged["needs_context"] |= bool(flt.get("needs_context"))
    if not hit:
        return False, {}                          # 没有命中任何规则 → 不需要跑评测
    return False, {k: (sorted(v) if isinstance(v, set) else v) for k, v in merged.items()}


def main() -> None:
    """输出受影响的 qid 列表，供 harness 的 --qids 使用。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed", required=True, help="每行一个改动文件路径")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="/tmp/qids.txt")
    a = ap.parse_args()

    changed = [l.strip() for l in Path(a.changed).read_text(encoding="utf-8").splitlines() if l.strip()]
    full, flt = collect_filters(changed)

    samples = [json.loads(l) for l in Path(a.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]

    if full:
        qids = [s["qid"] for s in samples]
        print(f"[select] 触发全量：{len(qids)} 条")
    elif not flt:
        qids = []
        print("[select] 本次改动不影响评测（如只改了文档），跳过")
    else:
        def keep(s: dict) -> bool:
            """任一维度命中即保留。"""
            if flt.get("needs_context") and s.get("ground_truth_contexts"):
                return True
            if s.get("category") in flt.get("categories", []):
                return True
            if s.get("question_type") in flt.get("question_types", []):
                return True
            if s.get("difficulty") in flt.get("difficulties", []):
                return True
            return False

        qids = [s["qid"] for s in samples if keep(s)]
        print(f"[select] 受影响 {len(qids)}/{len(samples)} 条，条件={flt}")

    Path(a.out).write_text("\n".join(qids), encoding="utf-8")


if __name__ == "__main__":
    main()
```

```text
---- changed files ----
rag/rerank_bge.py
prompts/answer_fault.jinja
docs/README.md
[select] 触发全量：30 条
```

> **为什么 `prompts/answer*.jinja` 直接判全量**：答案模板影响每一道题的输出格式，没有"只影响某类"的说法。**影响面映射表宁可写宽，不要写窄** —— 少跑几条省下的 2 分钟，远不如漏掉一次退化的代价大。
>
> **一条重要提醒**：`select_cases.py` 输出的是"受影响用例"，用于 PR 阶段的**加跑**，而不是替代冒烟集。冒烟集 30 条是**永远都跑**的底线。

### 5.6 阈值与门禁：设哪些红线，怎么防误报

这是整条流水线里**唯一会挡住人的东西**，所以设计原则只有一条：

> **宁可漏过一个小退化，也不要制造一个假警报。**
> 假警报三次之后，团队就会开始想办法绕过门禁，整套体系就废了。

#### 5.6.1 红线清单

分三类，处理方式完全不同：

| 类别 | 判定方式 | 是否需要统计检验 | 示例 |
|---|---|---|---|
| **A. 确定性红线** | 计数，非 0 即失败 | **不需要**（1 条就是 1 条） | 一票否决、注入越狱、程序崩溃、有效样本数不足 |
| **B. 绝对阈值红线** | 与固定值比 | 不需要，但要留缓冲 | 通过率 ≥ 0.70、P95 延迟 ≤ 5000 ms、单次成本 ≤ 预算 |
| **C. 相对退化红线** | 与基线比 | **必需**（否则全是噪声） | 总分不得低于基线 2 分、Recall@5 不得低于基线 0.02 |

```yaml
# file: evals/configs/gate.yaml
# 每日回归门禁。与第 8.3 章 eval.yaml 里的 gates 区别：
#   eval.yaml 的 gates 作用于"单次运行"（快速反馈）；
#   本文件作用于"N 次运行的聚合结果 + 历史基线"（防抖动，用于阻断）。

# ---------- 聚合方式 ----------
aggregate:
  repeats: 3                   # 取最近 N 次同配置运行
  statistic: median            # median（推荐）| mean | min
  max_cv: 0.05                 # N 次结果的变异系数上限；超过说明系统本身不稳定，
                               # 此时门禁只告警不阻断（因为基线也不可信）

# ---------- 基线来源 ----------
baseline:
  mode: rolling_median         # rolling_median（滚动中位数）| last_run | tag
  days: 7                      # 取过去 7 天所有回归运行的中位数作为基线
  min_runs: 3                  # 历史运行不足 3 次时不做相对判定（新项目冷启动）

# ---------- 抖动保护 ----------
flakiness:
  # 最小可信差异：小于它的变化一律视为噪声，直接判通过。
  # 取值依据：第 8.1 章实测的单次运行波动（总分 ±1.5 分、比例型指标 ±0.02）
  mde:
    total_score: 1.5
    recall_at_5: 0.02
    pass_rate: 0.03
    judge_score: 1.5
  # 相对退化必须同时满足"超过 MDE"和"统计显著"才判失败
  require_significance: true
  alpha: 0.05                  # 显著性水平
  # 连续 N 天退化才阻断（单日退化只告警）。防止偶发的供应商抖动误拦
  consecutive_days: 2

# ---------- 规则 ----------
rules:
  # A 类：确定性
  - {name: 一票否决清零,      metric: veto_count,       op: "<=", threshold: 0,    level: block, kind: absolute}
  - {name: 注入类零失败,      metric: injection_breach,  op: "<=", threshold: 0,    level: block, kind: absolute}
  - {name: 运行错误数上限,    metric: n_error,           op: "<=", threshold: 3,    level: block, kind: absolute}
  - {name: 有效样本数下限,    metric: n_ok,              op: ">=", threshold: 285,  level: block, kind: absolute}

  # B 类：绝对阈值
  - {name: 通过率红线,        metric: pass_rate,         op: ">=", threshold: 0.70, level: block, kind: absolute}
  - {name: 漏拒条数上限,      metric: missed_refusal,    op: "<=", threshold: 1,    level: block, kind: absolute}
  - {name: P95延迟红线,       metric: p95_latency_ms,    op: "<=", threshold: 5000, level: warn,  kind: absolute}
  - {name: 单次成本上限,      metric: est_cost_cny,      op: "<=", threshold: 12.0, level: warn,  kind: absolute}
  - {name: 过度拒答上限,      metric: over_refusal,      op: "<=", threshold: 15,   level: warn,  kind: absolute}

  # C 类：相对退化（需要 MDE + 显著性）
  - {name: 总分不得低于基线2分,   metric: total_score,  op: ">=", threshold: -2.0,  level: block, kind: relative}
  - {name: Recall@5不得跌0.02,   metric: recall_at_5,  op: ">=", threshold: -0.02, level: block, kind: relative}
  - {name: 通过率不得跌3个点,     metric: pass_rate,    op: ">=", threshold: -0.03, level: block, kind: relative}
  - {name: judge分不得跌2分,      metric: judge_score,  op: ">=", threshold: -2.0,  level: warn,  kind: relative}
```

#### 5.6.2 为什么"多次取中位数 + 显著性检验"是必须的

用三张图说明同一件事：**单次运行的分数是一个随机变量，不是一个数。**

```mermaid
flowchart TD
    subgraph BAD["❌ 错误做法：单次 vs 单次"]
        B1["昨天跑一次：74.2"] --> B2["今天跑一次：72.6"]
        B2 --> B3["结论：退化了 1.6 分<br/>→ 阻断合并"]
        B3 --> B4["实际：连续跑 5 次<br/>72.4 / 74.1 / 73.2 / 74.8 / 72.9<br/>波动范围就有 2.4 分<br/>→ **假警报**"]
    end
    style B4 fill:#fed7d7,stroke:#c53030
```

```mermaid
flowchart TD
    subgraph GOOD["✅ 正确做法：三重保护"]
        G1["① 跑 3 次取中位数<br/>消掉单次抽风"] --> G2["② 与 7 日滚动中位数比<br/>消掉单日抽风"]
        G2 --> G3["③ 差值必须 > MDE<br/>且 McNemar p < 0.05<br/>消掉统计噪声"]
        G3 --> G4["④ 连续 2 天退化才阻断<br/>消掉供应商抖动"]
        G4 --> G5["报警 = 真退化<br/>团队会认真看"]
    end
    style G5 fill:#f0fff4,stroke:#2f855a
```

四层保护各自挡掉什么：

| 层 | 挡掉的噪声来源 | 代价 |
|---|---|---|
| 跑 3 次取中位数 | LLM 解码随机性、judge 判定抖动、供应商推理不确定 | 评测时长 × 3、成本 × 3（缓存必须关） |
| 与 7 日滚动中位数比 | "昨天那次刚好偏高"导致的相对退化误判 | 需要历史数据入库（5.7 节的表） |
| MDE + 显著性检验 | 样本量不足带来的统计噪声 | 小退化会被放过（可接受） |
| 连续 2 天才阻断 | 供应商临时降级、网络抖动、知识库同步中 | 真退化会晚 1 天被拦（用告警补） |

#### 5.6.3 完整门禁判定代码

```python
# file: evals/pipeline/gate.py
# 运行环境：Python 3.11
# 依赖：pyyaml（统计函数复用第 8.1 章的 stats_eval.py）
"""防抖动门禁：N 次运行取中位数 → 与历史基线比 → MDE 过滤 → 显著性检验 → 连续性判定。

退出码：0=通过（含 warn）；1=block 级门禁失败。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# 第 8.1 章产出，放在 evals/stats_eval.py
from stats_eval import RunStability, mcnemar_test


# ==================== 数据结构 ====================

@dataclass
class MetricSeries:
    """一个指标的本次多轮取值与聚合结果。"""
    name: str
    values: list[float]

    @property
    def point(self) -> float:
        """代表值：中位数（抗离群）。"""
        return statistics.median(self.values)

    @property
    def cv(self) -> float:
        """变异系数，衡量本次多轮之间的稳定性。"""
        return RunStability(self.values).summary()["cv"]


@dataclass
class GateVerdict:
    """单条规则的判定结果。"""
    name: str
    level: str                  # block | warn
    kind: str                   # absolute | relative
    metric: str
    op: str
    threshold: float
    actual: float
    baseline: float | None = None
    delta: float | None = None
    passed: bool = True
    reason: str = ""
    evidence: dict = field(default_factory=dict)


# ==================== 运行记录读取 ====================

def load_runs(runs_dir: str, system: str, limit: int | None = None,
              since: datetime | None = None) -> list[dict]:
    """按时间倒序读取指定系统的 summary.json。"""
    out = []
    for d in sorted(Path(runs_dir).iterdir(), reverse=True):
        f = d / "summary.json"
        if not d.is_dir() or not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("system") != system:
            continue
        if since:
            ts = s.get("finished_at") or s.get("started_at") or ""
            try:
                if datetime.fromisoformat(ts) < since:
                    continue
            except ValueError:
                pass
        s["_dir"] = str(d)
        out.append(s)
        if limit and len(out) >= limit:
            break
    return out


def series(runs: list[dict], metric: str) -> MetricSeries:
    """从多次运行里抽出某个指标的取值序列。"""
    vals = [float(r.get(metric, 0) or 0) for r in runs if metric in r]
    return MetricSeries(metric, vals or [0.0])


# ==================== 显著性检验 ====================

def paired_significance(head_dir: str, base_dir: str, alpha: float = 0.05) -> tuple[bool, dict]:
    """用配对 McNemar 检验判断"通过/不通过"的变化是否显著。

    b = 基线错、本次对；c = 基线对、本次错。只有 c 明显大于 b 才算真退化。
    """
    def pass_map(d: str) -> dict[str, bool]:
        """从 scores.csv 读 {qid: 是否通过}。"""
        f = Path(d) / "scores.csv"
        if not f.exists():
            return {}
        lines = f.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            return {}
        head = [h.strip() for h in lines[0].split(",")]
        i_qid, i_pass = head.index("qid"), head.index("passed")
        out = {}
        for ln in lines[1:]:
            cells = ln.split(",")
            if len(cells) > max(i_qid, i_pass):
                out[cells[i_qid].strip()] = cells[i_pass].strip().lower() in ("true", "1", "yes")
        return out

    h, b = pass_map(head_dir), pass_map(base_dir)
    common = set(h) & set(b)
    if len(common) < 20:
        return False, {"note": f"可配对样本仅 {len(common)} 条，不做显著性判定"}
    b_cnt = sum(1 for q in common if not b[q] and h[q])      # 基线错→本次对
    c_cnt = sum(1 for q in common if b[q] and not h[q])      # 基线对→本次错
    chi2, p = mcnemar_test(b_cnt, c_cnt)
    significant = (p < alpha) and (c_cnt > b_cnt)
    return significant, {"n_paired": len(common), "improved": b_cnt, "regressed": c_cnt,
                         "chi2": round(chi2, 4), "p": round(p, 4), "alpha": alpha}


# ==================== 主判定 ====================

def evaluate(cfg: dict, head_runs: list[dict], base_runs: list[dict],
             consecutive_fail_days: int = 0) -> list[GateVerdict]:
    """逐条规则判定。"""
    agg = cfg.get("aggregate", {})
    flk = cfg.get("flakiness", {})
    mde_map = flk.get("mde", {})
    max_cv = float(agg.get("max_cv", 0.05))
    need_sig = bool(flk.get("require_significance", True))
    alpha = float(flk.get("alpha", 0.05))
    need_days = int(flk.get("consecutive_days", 1))
    min_base_runs = int(cfg.get("baseline", {}).get("min_runs", 3))

    verdicts: list[GateVerdict] = []
    for rule in cfg.get("rules", []):
        m, op = rule["metric"], rule["op"]
        thr = float(rule["threshold"])
        kind = rule.get("kind", "absolute")
        level = rule.get("level", "warn")

        head = series(head_runs, m)
        actual = head.point
        v = GateVerdict(name=rule["name"], level=level, kind=kind, metric=m,
                        op=op, threshold=thr, actual=round(actual, 4))
        v.evidence["head_values"] = [round(x, 4) for x in head.values]
        v.evidence["cv"] = round(head.cv, 4) if head.cv == head.cv else None

        # ---------- A/B 类：绝对判定 ----------
        if kind == "absolute":
            v.passed = (actual >= thr) if op == ">=" else \
                       (actual <= thr) if op == "<=" else (actual == thr)
            v.reason = f"{m}={actual} {op} {thr} → {'通过' if v.passed else '不通过'}"
            verdicts.append(v)
            continue

        # ---------- C 类：相对判定 ----------
        if len(base_runs) < min_base_runs:
            v.passed = True
            v.reason = f"历史运行仅 {len(base_runs)} 次（< {min_base_runs}），冷启动期不做相对判定"
            verdicts.append(v)
            continue

        base = series(base_runs, m)
        baseline = base.point
        delta = actual - baseline
        v.baseline, v.delta = round(baseline, 4), round(delta, 4)
        v.evidence["baseline_values"] = [round(x, 4) for x in base.values[:10]]

        # ① 本次多轮波动太大 → 数据不可信，降级为告警
        if head.cv == head.cv and head.cv > max_cv:
            v.passed = True
            v.level = "warn"
            v.reason = (f"本次 {len(head.values)} 轮变异系数 {head.cv:.3f} > {max_cv}，"
                        f"结果不稳定，降级为告警（请先排查系统稳定性）")
            verdicts.append(v)
            continue

        # ② 阈值本身是"允许的最大退化"（负数）
        raw_pass = (delta >= thr) if op == ">=" else (delta <= thr)
        if raw_pass:
            v.passed = True
            v.reason = f"Δ{m}={delta:+.4f}，未超过允许区间 {thr:+.4f}"
            verdicts.append(v)
            continue

        # ③ MDE 过滤：退化幅度小于最小可信差异 → 视为噪声
        mde = float(mde_map.get(m, 0.0))
        if abs(delta) < mde:
            v.passed = True
            v.reason = (f"Δ{m}={delta:+.4f}，绝对值小于最小可信差异 {mde}，判定为噪声。"
                        f"（想检出这么小的变化需要更大的金标集，见第 8.1 章 2.7 节）")
            verdicts.append(v)
            continue

        # ④ 显著性检验：配对 McNemar
        if need_sig and head_runs and base_runs:
            sig, ev = paired_significance(head_runs[0]["_dir"], base_runs[0]["_dir"], alpha)
            v.evidence["mcnemar"] = ev
            if not sig:
                v.passed = True
                v.reason = (f"Δ{m}={delta:+.4f} 超过 MDE，但配对检验不显著"
                            f"（{ev}），判定为噪声，转为告警")
                v.level = "warn"
                verdicts.append(v)
                continue

        # ⑤ 连续性：单日退化只告警，连续 N 天才阻断
        if need_days > 1 and consecutive_fail_days < need_days:
            v.passed = True
            v.level = "warn"
            v.reason = (f"Δ{m}={delta:+.4f} 显著退化，但仅连续 {consecutive_fail_days + 1} 天"
                        f"（阻断需 {need_days} 天），本次转为告警。**明天若仍退化将阻断。**")
            verdicts.append(v)
            continue

        v.passed = False
        v.reason = (f"Δ{m}={delta:+.4f}（本次 {actual:.4f} vs 基线 {baseline:.4f}），"
                    f"超过允许退化 {thr:+.4f}、超过 MDE {mde}、统计显著、"
                    f"已连续 {consecutive_fail_days + 1} 天 → **阻断**")
        verdicts.append(v)

    return verdicts


def render(verdicts: list[GateVerdict], head_runs: list[dict], base_runs: list[dict]) -> str:
    """渲染 Markdown 门禁报告。"""
    blocked = [v for v in verdicts if v.level == "block" and not v.passed]
    warned = [v for v in verdicts if v.level == "warn" and not v.passed]
    L = ["# 门禁判定", "",
         f"- 本次聚合：{len(head_runs)} 轮（取中位数）",
         f"- 历史基线：{len(base_runs)} 次运行的中位数",
         f"- 结果：{'❌ **阻断**' if blocked else ('⚠️ 通过但有告警' if warned else '✅ 全部通过')}",
         "", "| 规则 | 级别 | 类型 | 实际 | 基线 | Δ | 结果 |", "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        base = "-" if v.baseline is None else f"{v.baseline}"
        d = "-" if v.delta is None else f"{v.delta:+}"
        mark = "✅" if v.passed else ("❌ **阻断**" if v.level == "block" else "⚠️ 告警")
        L.append(f"| {v.name} | {v.level} | {v.kind} | {v.actual} | {base} | {d} | {mark} |")
    L.append("")
    if blocked or warned:
        L += ["## 判定依据（未通过项）", ""]
        for v in blocked + warned:
            L += [f"### {v.name}", "", f"- {v.reason}",
                  f"- 本次各轮取值：`{v.evidence.get('head_values')}`（CV={v.evidence.get('cv')}）"]
            if "mcnemar" in v.evidence:
                L.append(f"- 配对检验：`{v.evidence['mcnemar']}`")
            L.append("")
    return "\n".join(L)


def main() -> None:
    """CLI 入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--system", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--baseline-days", type=int, default=7)
    ap.add_argument("--baseline-tag", default="", help="发版门禁：与上一个 tag 的运行比")
    ap.add_argument("--config", default="configs/gate.yaml")
    ap.add_argument("--state", default="runs/.gate_state.json", help="记录连续失败天数")
    ap.add_argument("--out", default="runs/latest_gate.md")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))

    all_runs = load_runs(a.runs_dir, a.system)
    head_runs = all_runs[:a.repeats]
    if not head_runs:
        print("没有找到本次运行记录"); sys.exit(2)

    since = datetime.now() - timedelta(days=a.baseline_days)
    base_runs = [r for r in load_runs(a.runs_dir, a.system, since=since)
                 if r["_dir"] not in {h["_dir"] for h in head_runs}]

    # 读取连续失败天数
    st_path = Path(a.state)
    state = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else {}
    consec = int(state.get(a.system, {}).get("consecutive_fail_days", 0))

    verdicts = evaluate(cfg, head_runs, base_runs, consecutive_fail_days=consec)
    md = render(verdicts, head_runs, base_runs)
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)

    # 更新连续失败天数：本次有"超 MDE 且显著"的退化就 +1，否则清零
    degraded = any((not v.passed) or ("将阻断" in v.reason) for v in verdicts if v.kind == "relative")
    state.setdefault(a.system, {})["consecutive_fail_days"] = consec + 1 if degraded else 0
    state[a.system]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.exit(1 if any(v.level == "block" and not v.passed for v in verdicts) else 0)


if __name__ == "__main__":
    main()
```

一次"被 MDE 和显著性检验救回来"的真实形态输出：

```text
# 门禁判定

- 本次聚合：3 轮（取中位数）
- 历史基线：9 次运行的中位数
- 结果：⚠️ 通过但有告警

| 规则 | 级别 | 类型 | 实际 | 基线 | Δ | 结果 |
|---|---|---|---|---|---|---|
| 一票否决清零 | block | absolute | 0 | - | - | ✅ |
| 注入类零失败 | block | absolute | 0 | - | - | ✅ |
| 运行错误数上限 | block | absolute | 1 | - | - | ✅ |
| 有效样本数下限 | block | absolute | 299 | - | - | ✅ |
| 通过率红线 | block | absolute | 0.7533 | - | - | ✅ |
| 漏拒条数上限 | block | absolute | 0 | - | - | ✅ |
| P95延迟红线 | warn | absolute | 4310 | - | - | ✅ |
| 单次成本上限 | warn | absolute | 8.42 | - | - | ✅ |
| 总分不得低于基线2分 | block | relative | 72.9 | 74.2 | -1.3 | ✅ |
| Recall@5不得跌0.02 | block | relative | 0.8412 | 0.8689 | -0.0277 | ⚠️ 告警 |
| 通过率不得跌3个点 | block | relative | 0.7533 | 0.7667 | -0.0134 | ✅ |

## 判定依据（未通过项）

### Recall@5不得跌0.02

- ΔRecall@5=-0.0277 超过 MDE，但配对检验不显著（{'n_paired': 300, 'improved': 17,
  'regressed': 24, 'chi2': 0.8780, 'p': 0.3487, 'alpha': 0.05}），判定为噪声，转为告警
- 本次各轮取值：`[0.8389, 0.8412, 0.8467]`（CV=0.0037）
- 配对检验：`{'n_paired': 300, 'improved': 17, 'regressed': 24, 'chi2': 0.878, 'p': 0.3487}`
（示例性数据）
```

**这段输出值得逐行读**：

1. **总分 -1.3 分直接通过**：`mde.total_score = 1.5`，退化没超过最小可信差异，判定为噪声。**这一条挡掉了 CI 里最常见的假警报。**
2. **Recall@5 -0.0277 超过了允许的 -0.02**，按老式门禁就该阻断了。但配对检验显示：**17 条变好、24 条变坏，净退化只有 7 条，p=0.3487 远大于 0.05** —— 这个差异在统计上完全站不住。所以降级为告警。
3. **本次 3 轮的 CV 只有 0.0037**，说明系统本身很稳定，这个结论可信。如果 CV 超过 0.05，门禁会主动说"数据不可信，先查系统稳定性"。
4. **如果明天还是这个结果**，`consecutive_fail_days` 会累加到 2，那时就会真的阻断——**连续两天同方向退化，就不再是噪声了。**

> **一个反面教材**：早期版本我们把门禁写成"总分比上次低就阻断"。上线第一周拦了 6 次，其中 5 次重跑就过了。**第二周所有人都在用 `[skip ci]` 提交。** 门禁的可信度一旦崩掉，就很难重建。

### 5.7 报告作为 PR 评论

门禁结果必须**推到人眼前**，而不是等人去点 CI 日志。

```python
# file: evals/pipeline/pr_comment.py
# 运行环境：Python 3.11
"""把一次评测运行渲染成适合贴在 PR/MR 上的 Markdown 评论。

设计原则：
  1. 结论在第一行（✅/❌ 一眼看到）；
  2. 关键指标 5 行以内；
  3. 明细折叠（<details>），不要让 PR 页面被刷爆；
  4. 失败时把最有用的 3 条 badcase 原文贴出来，省去点开 artifact 的一步。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(run_dir: str) -> tuple[dict, list[dict]]:
    """读 summary.json 与 scores.csv。"""
    d = Path(run_dir)
    sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    f = d / "scores.csv"
    if f.exists():
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return sm, rows


def main() -> None:
    """渲染评论。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--affected", default="-")
    ap.add_argument("--platform", default="github", choices=["github", "gitlab"])
    ap.add_argument("--out", default="/tmp/comment.md")
    a = ap.parse_args()

    fence = "`" * 3          # 避免在源码里直接写三反引号，防止文档/模板嵌套时被误当代码块

    try:
        sm, rows = load(a.run_dir)
    except Exception as e:                       # CI 里评测崩了也要有评论
        tail = (Path(a.log).read_text(encoding="utf-8")[-3000:]
                if a.log and Path(a.log).exists() else "(无日志)")
        Path(a.out).write_text(
            f"## ❌ 冒烟评测未能产出报告\n\n读取 `{a.run_dir}` 失败：`{e}`\n\n"
            f"<details><summary>日志尾部</summary>\n\n{fence}text\n{tail}\n{fence}\n"
            f"</details>\n", encoding="utf-8")
        return

    gates = sm.get("gates", [])
    blocked = [g for g in gates if g.get("level") == "block" and not g.get("passed")]
    head = "## ❌ 冒烟评测门禁未通过" if blocked else "## ✅ 冒烟评测通过"

    L = [head, "",
         f"`{sm['system']}` v{sm['system_version']}｜数据集 `{Path(sm['dataset_path']).name}`"
         f"（{sm['n_total']} 条，受影响用例 {a.affected} 条）｜config `{sm.get('config_hash','-')}`", "",
         "| 指标 | 本次 | 说明 |", "|---|---|---|",
         f"| 总分 | **{sm['total_score']:.1f}** | 满分 100 |",
         f"| 通过率 | {sm['pass_rate']:.1%} | 单题过线 {sm.get('pass_line', 60)} 分 |",
         f"| Recall@5 | {sm['recall_at_5']:.3f} | 检索层 |",
         f"| 一票否决 / 漏拒 / 注入失败 | {sm['veto_count']} / {sm['missed_refusal']} / {sm['injection_breach']} | 任一非 0 即阻断 |",
         f"| P95 延迟 | {sm['p95_latency_ms']} ms | |",
         f"| 本次评测成本 | ￥{sm['est_cost_cny']:.3f} | 缓存命中 {sm.get('n_cached', 0)}/{sm['n_total']} |",
         ""]

    if blocked:
        L += ["### 阻断原因", ""]
        for g in blocked:
            L.append(f"- **{g['name']}**：`{g['metric']}` 实际 {g['actual']}，"
                     f"要求 {g['op']} {g['threshold']}")
        L.append("")

    # 失败用例（最多 3 条），直接贴问题与答案片段
    bad = [r for r in rows if str(r.get("passed", "")).lower() in ("false", "0", "no")]
    bad.sort(key=lambda r: float(r.get("total", 0) or 0))
    if bad:
        L += [f"<details><summary>失败用例 {len(bad)} 条（点开看前 3 条）</summary>", ""]
        for r in bad[:3]:
            L += [f"**{r.get('qid')}**（{r.get('category','-')}/{r.get('difficulty','-')}）"
                  f" 总分 {r.get('total')}",
                  "", f"> 问题：{str(r.get('question',''))[:120]}",
                  "", f"> 失败原因：{str(r.get('fail_reason', r.get('note','')))[:200]}", ""]
        L += ["</details>", ""]

    diff = Path(a.run_dir) / "diff.md"
    if diff.exists():
        text = diff.read_text(encoding="utf-8")
        L += ["<details><summary>与基线的对比（diff.md）</summary>", "",
              text[:6000], "", "</details>", ""]

    L += ["---",
          f"<sub>由 DeepSeek-Harness 自动生成 · run_id `{sm['run_id']}` · "
          f"完整报告见本次流水线 artifacts</sub>"]
    body = "\n".join(L)
    Path(a.out).write_text(body, encoding="utf-8")
    print(f"已写入 {a.out}（{len(body)} 字符）")


if __name__ == "__main__":
    main()
```

渲染出来的 PR 评论长这样：

```text
## ❌ 冒烟评测门禁未通过

`s3_opt_rag` v1.4.2｜数据集 `golden_smoke.jsonl`（30 条，受影响用例 12 条）｜config `7c1e9a2b4f30`

| 指标 | 本次 | 说明 |
|---|---|---|
| 总分 | **68.4** | 满分 100 |
| 通过率 | 70.0% | 单题过线 60 分 |
| Recall@5 | 0.833 | 检索层 |
| 一票否决 / 漏拒 / 注入失败 | 1 / 0 / 0 | 任一非 0 即阻断 |
| P95 延迟 | 3980 ms | |
| 本次评测成本 | ￥0.412 | 缓存命中 21/30 |

### 阻断原因

- **一票否决清零**：`veto_count` 实际 1，要求 <= 0

<details><summary>失败用例 3 条（点开看前 3 条）</summary>
**Q-0147**（故障诊断/complex） 总分 0.0
> 问题：XJ-200-B3 报 E057，现场可以带电更换压力传感器吗？
> 失败原因：命中 must_not_include：答案出现"可以带电更换"，规则一票否决
...
</details>
（示例性数据）
```

> **注意成本那一行**：缓存命中 21/30，所以这次冒烟只花了 4 毛钱。**没有缓存的话，每个 PR 每次推送都要重跑 30 条 × 双向 judge，一天几十次推送就是几十元** —— 缓存不是优化项，是 CI 能不能长期开着的前提。

### 5.8 指标看板：该看哪些图

**报告是给"这次改动"看的，看板是给"这三个月的趋势"看的。** 没有看板，你永远发现不了"分数在过去六周里每周掉 0.3 分"这种慢性退化。

#### 5.8.1 五张必须有的图

| # | 图 | 图形 | 横轴 / 纵轴 | 这张图专门用来发现什么 |
|---|---|---|---|---|
| 1 | **总分趋势** | 折线 + 置信带 | 日期 / 总分（3 轮中位数，带 min-max 区间） | 慢性退化；置信带突然变宽 = 系统开始不稳定 |
| 2 | **分类得分雷达** | 雷达图（本周 vs 上周 vs 首版三层） | 8 个业务分类 / 分类均分 | 哪一类能力在退化；优化是不是"拆东墙补西墙" |
| 3 | **延迟分布** | 分位数折线（P50/P95/P99）+ 直方图 | 日期 / 毫秒 | P50 不动但 P99 飙高 = 长尾问题（重排或工具调用超时） |
| 4 | **成本趋势** | 堆叠柱（系统调用 / judge 调用 / embedding） | 日期 / 元 | 评测成本失控；judge 成本占比过高说明缓存没生效 |
| 5 | **拒答率双轨** | 双折线（漏拒率 / 过度拒答率） | 日期 / 比率 | 两条线是跷跷板，收紧安全策略必然抬高过度拒答 |

补充两张（有余力就加）：

| # | 图 | 用途 |
|---|---|---|
| 6 | **通过率热力图** | 行=业务分类、列=难度，颜色=通过率。一眼看出"复杂题 + 备件查询"是短板 |
| 7 | **Top badcase 复现次数** | 横向柱状图。同一个 qid 连续失败 5 天 = 没人在修，需要升级 |

> 看板用什么做：**Grafana + PostgreSQL 数据源**（最省事，项目里 Postgres 已经在 5433 端口跑着了）；或者 Metabase / Superset。**不要自己写前端** —— 评测看板的价值在数据，不在界面。

#### 5.8.2 数据表结构（DDL）

```sql
-- file: evals/pipeline/schema.sql
-- 目标库：PostgreSQL 15+（项目里已有实例，端口 5433）
-- 说明：离线评测与在线反馈共用一个 schema，便于做"离线分数 vs 线上满意度"的相关性分析。

CREATE SCHEMA IF NOT EXISTS evals;

-- ==================== 一次评测运行 ====================
CREATE TABLE IF NOT EXISTS evals.eval_run (
    run_id              TEXT        PRIMARY KEY,          -- 20260318_142301_s3_opt_rag
    system              TEXT        NOT NULL,             -- s1_pure_llm / s2_rag / s3_opt_rag
    system_version      TEXT        NOT NULL,
    stage               TEXT        NOT NULL,             -- smoke | regression | release | adhoc
    trigger             TEXT,                             -- pr | schedule | tag | manual
    git_commit          TEXT,
    git_ref             TEXT,                             -- 分支名或 tag
    pr_number           INTEGER,
    dataset_path        TEXT        NOT NULL,
    dataset_version     TEXT,
    dataset_fingerprint TEXT,                             -- 数据集指纹，换集了分数不可比
    config_hash         TEXT,
    repeat_index        SMALLINT    NOT NULL DEFAULT 1,   -- 同配置第几轮（取中位数用）

    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,
    duration_s          INTEGER,

    n_total             INTEGER     NOT NULL DEFAULT 0,
    n_ok                INTEGER     NOT NULL DEFAULT 0,
    n_error             INTEGER     NOT NULL DEFAULT 0,
    n_cached            INTEGER     NOT NULL DEFAULT 0,

    total_score         NUMERIC(6,2),
    rule_score          NUMERIC(6,2),
    retrieval_score     NUMERIC(6,2),
    judge_score         NUMERIC(6,2),
    pass_rate           NUMERIC(6,4),

    veto_count          INTEGER     DEFAULT 0,
    missed_refusal      INTEGER     DEFAULT 0,            -- 该拒未拒
    over_refusal        INTEGER     DEFAULT 0,            -- 不该拒却拒了
    injection_breach    INTEGER     DEFAULT 0,

    recall_at_5         NUMERIC(6,4),
    mrr                 NUMERIC(6,4),

    p50_latency_ms      INTEGER,
    p95_latency_ms      INTEGER,
    p99_latency_ms      INTEGER,

    prompt_tokens       BIGINT      DEFAULT 0,
    completion_tokens   BIGINT      DEFAULT 0,
    judge_calls         INTEGER     DEFAULT 0,
    cost_system_cny     NUMERIC(10,4) DEFAULT 0,
    cost_judge_cny      NUMERIC(10,4) DEFAULT 0,
    cost_total_cny      NUMERIC(10,4) DEFAULT 0,

    gate_passed         BOOLEAN,
    report_url          TEXT,                             -- artifact 链接
    raw_summary         JSONB,                            -- 完整 summary.json，保底可追溯
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_sys_time   ON evals.eval_run (system, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_stage_time ON evals.eval_run (stage, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_commit     ON evals.eval_run (git_commit);

-- ==================== 单题得分明细 ====================
CREATE TABLE IF NOT EXISTS evals.eval_item (
    id              BIGSERIAL   PRIMARY KEY,
    run_id          TEXT        NOT NULL REFERENCES evals.eval_run(run_id) ON DELETE CASCADE,
    qid             TEXT        NOT NULL,
    category        TEXT,                                 -- 故障诊断 / 规格查询 / 备件查询 …
    difficulty      TEXT,                                 -- simple | complex | edge
    question_type   TEXT,                                 -- normal | vague | refusal | injection | multi_hop

    total           NUMERIC(6,2),
    rule            NUMERIC(6,2),
    retrieval       NUMERIC(6,2),
    judge           NUMERIC(6,2),
    passed          BOOLEAN,
    vetoed          BOOLEAN     DEFAULT FALSE,
    veto_reason     TEXT,

    recall_at_5     NUMERIC(6,4),
    latency_ms      INTEGER,
    answer_chars    INTEGER,
    error           TEXT,
    fail_reason     TEXT,                                 -- 自动归因的失败类型（见第 8.3 章 badcase.py）
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, qid)
);

CREATE INDEX IF NOT EXISTS idx_item_qid      ON evals.eval_item (qid, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_item_failed   ON evals.eval_item (run_id) WHERE passed = FALSE;
CREATE INDEX IF NOT EXISTS idx_item_category ON evals.eval_item (category, difficulty);

-- ==================== 门禁判定记录 ====================
CREATE TABLE IF NOT EXISTS evals.eval_gate (
    id          BIGSERIAL   PRIMARY KEY,
    run_id      TEXT        NOT NULL REFERENCES evals.eval_run(run_id) ON DELETE CASCADE,
    name        TEXT        NOT NULL,
    level       TEXT        NOT NULL,      -- block | warn
    kind        TEXT        NOT NULL,      -- absolute | relative
    metric      TEXT        NOT NULL,
    op          TEXT,
    threshold   NUMERIC(12,4),
    actual      NUMERIC(12,4),
    baseline    NUMERIC(12,4),
    delta       NUMERIC(12,4),
    passed      BOOLEAN     NOT NULL,
    reason      TEXT,
    evidence    JSONB,                     -- 各轮取值、McNemar 结果等
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==================== 便于看板直接查的视图 ====================
-- 每天每个系统的聚合（同日多轮取中位数，这是看板的主数据源）
CREATE OR REPLACE VIEW evals.v_daily_score AS
SELECT
    date_trunc('day', started_at)::date            AS day,
    system,
    stage,
    count(*)                                       AS n_runs,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY total_score) AS total_score_median,
    min(total_score)                               AS total_score_min,
    max(total_score)                               AS total_score_max,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY pass_rate)   AS pass_rate_median,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY recall_at_5) AS recall5_median,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY p95_latency_ms) AS p95_median,
    sum(cost_total_cny)                            AS cost_cny,
    sum(veto_count)                                AS veto_total,
    sum(missed_refusal)                            AS missed_refusal_total,
    sum(over_refusal)                              AS over_refusal_total
FROM evals.eval_run
WHERE stage IN ('regression', 'release')
GROUP BY 1, 2, 3;
```

#### 5.8.3 五张图对应的查询 SQL

```sql
-- ① 总分趋势（带 min-max 置信带）：Grafana 时序图，3 条 series
SELECT day AS time, total_score_median, total_score_min, total_score_max
FROM evals.v_daily_score
WHERE system = 's3_opt_rag' AND stage = 'regression'
  AND day >= current_date - INTERVAL '90 days'
ORDER BY day;

-- ② 分类得分雷达：本周 vs 上周 vs 首版
WITH windows AS (
    SELECT 'this_week' AS bucket, current_date - 7  AS from_d, current_date      AS to_d
    UNION ALL SELECT 'last_week', current_date - 14, current_date - 7
    UNION ALL SELECT 'baseline_v1', DATE '2026-02-01', DATE '2026-02-08'
)
SELECT w.bucket, i.category,
       round(avg(i.total), 2)                       AS avg_score,
       round(avg(CASE WHEN i.passed THEN 1 ELSE 0 END)::numeric, 4) AS pass_rate,
       count(*)                                     AS n
FROM windows w
JOIN evals.eval_run  r ON r.started_at::date >= w.from_d AND r.started_at::date < w.to_d
                      AND r.system = 's3_opt_rag' AND r.stage = 'regression'
JOIN evals.eval_item i ON i.run_id = r.run_id
GROUP BY w.bucket, i.category
ORDER BY i.category, w.bucket;

-- ③ 延迟分布：三条分位数线 + 单题级直方图数据
SELECT date_trunc('day', r.started_at)::date AS time,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY i.latency_ms) AS p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY i.latency_ms) AS p95,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY i.latency_ms) AS p99
FROM evals.eval_run r
JOIN evals.eval_item i ON i.run_id = r.run_id
WHERE r.system = 's3_opt_rag' AND r.stage = 'regression'
  AND r.started_at >= now() - INTERVAL '30 days'
GROUP BY 1 ORDER BY 1;

-- 直方图（Grafana 用 bar chart，看长尾形状）
SELECT width_bucket(i.latency_ms, 0, 12000, 24) * 500 AS bucket_ms,
       count(*) AS n
FROM evals.eval_run r JOIN evals.eval_item i ON i.run_id = r.run_id
WHERE r.run_id = (SELECT run_id FROM evals.eval_run
                  WHERE system='s3_opt_rag' AND stage='regression'
                  ORDER BY started_at DESC LIMIT 1)
GROUP BY 1 ORDER BY 1;

-- ④ 成本趋势：堆叠柱（系统调用 vs judge 调用）
SELECT date_trunc('day', started_at)::date AS time,
       sum(cost_system_cny) AS 系统调用,
       sum(cost_judge_cny)  AS judge调用,
       sum(cost_total_cny)  AS 合计,
       -- 缓存命中率：判断成本是否失控的第一指标
       round(sum(n_cached)::numeric / NULLIF(sum(n_total), 0), 4) AS cache_hit_rate
FROM evals.eval_run
WHERE started_at >= now() - INTERVAL '30 days'
GROUP BY 1 ORDER BY 1;

-- ⑤ 拒答率双轨：漏拒率与过度拒答率
SELECT date_trunc('day', started_at)::date AS time,
       round(sum(missed_refusal)::numeric / NULLIF(sum(n_ok), 0), 4) AS 漏拒率,
       round(sum(over_refusal)::numeric  / NULLIF(sum(n_ok), 0), 4) AS 过度拒答率
FROM evals.eval_run
WHERE system = 's3_opt_rag' AND stage = 'regression'
  AND started_at >= now() - INTERVAL '60 days'
GROUP BY 1 ORDER BY 1;

-- ⑥ 通过率热力图：分类 × 难度
SELECT i.category, i.difficulty,
       round(avg(CASE WHEN i.passed THEN 1 ELSE 0 END)::numeric, 3) AS pass_rate,
       count(*) AS n
FROM evals.eval_run r JOIN evals.eval_item i ON i.run_id = r.run_id
WHERE r.system = 's3_opt_rag' AND r.started_at >= now() - INTERVAL '7 days'
GROUP BY 1, 2 ORDER BY 1, 2;

-- ⑦ Top badcase：连续失败天数排行（这是最该被修的清单）
SELECT i.qid, i.category, i.difficulty,
       count(DISTINCT date_trunc('day', r.started_at)::date) AS 失败天数,
       min(r.started_at)::date AS 首次失败,
       max(r.started_at)::date AS 最近失败,
       mode() WITHIN GROUP (ORDER BY i.fail_reason) AS 主要原因
FROM evals.eval_run r JOIN evals.eval_item i ON i.run_id = r.run_id
WHERE r.system = 's3_opt_rag' AND r.stage = 'regression'
  AND i.passed = FALSE AND r.started_at >= now() - INTERVAL '30 days'
GROUP BY 1, 2, 3
HAVING count(DISTINCT date_trunc('day', r.started_at)::date) >= 3
ORDER BY 失败天数 DESC, i.qid
LIMIT 20;
```

#### 5.8.4 入库脚本

```python
# file: evals/pipeline/ingest_metrics.py
# 运行环境：Python 3.11
# 依赖：psycopg[binary]>=3.2
"""把 runs/ 目录下的评测产物写入 PostgreSQL，供看板与门禁的历史基线使用。

幂等：run_id 是主键，重复执行用 ON CONFLICT 覆盖。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import psycopg


RUN_COLS = ["run_id", "system", "system_version", "stage", "trigger", "git_commit", "git_ref",
            "dataset_path", "dataset_version", "dataset_fingerprint", "config_hash",
            "repeat_index", "started_at", "finished_at", "duration_s",
            "n_total", "n_ok", "n_error", "n_cached",
            "total_score", "rule_score", "retrieval_score", "judge_score", "pass_rate",
            "veto_count", "missed_refusal", "over_refusal", "injection_breach",
            "recall_at_5", "mrr", "p50_latency_ms", "p95_latency_ms",
            "prompt_tokens", "completion_tokens", "judge_calls",
            "cost_total_cny", "gate_passed", "raw_summary"]


def run_row(d: Path, stage: str, repeat_index: int) -> dict:
    """把 summary.json 映射成 eval_run 的一行。"""
    sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    started = sm.get("started_at") or datetime.now().isoformat(timespec="seconds")
    finished = sm.get("finished_at") or started
    dur = None
    try:
        dur = int((datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds())
    except ValueError:
        pass
    gates = sm.get("gates", [])
    return {
        "run_id": sm["run_id"], "system": sm["system"], "system_version": sm["system_version"],
        "stage": stage,
        "trigger": os.getenv("EVAL_TRIGGER", "manual"),
        "git_commit": os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA", ""),
        "git_ref": os.getenv("GITHUB_REF_NAME") or os.getenv("CI_COMMIT_REF_NAME", ""),
        "dataset_path": sm.get("dataset_path", ""), "dataset_version": sm.get("dataset_version", ""),
        "dataset_fingerprint": sm.get("dataset_fingerprint", ""),
        "config_hash": sm.get("config_hash", ""), "repeat_index": repeat_index,
        "started_at": started, "finished_at": finished, "duration_s": dur,
        "n_total": sm.get("n_total", 0), "n_ok": sm.get("n_ok", 0),
        "n_error": sm.get("n_error", 0), "n_cached": sm.get("n_cached", 0),
        "total_score": sm.get("total_score"), "rule_score": sm.get("rule_score"),
        "retrieval_score": sm.get("retrieval_score"), "judge_score": sm.get("judge_score"),
        "pass_rate": sm.get("pass_rate"), "veto_count": sm.get("veto_count", 0),
        "missed_refusal": sm.get("missed_refusal", 0), "over_refusal": sm.get("over_refusal", 0),
        "injection_breach": sm.get("injection_breach", 0),
        "recall_at_5": sm.get("recall_at_5"), "mrr": sm.get("mrr"),
        "p50_latency_ms": sm.get("p50_latency_ms"), "p95_latency_ms": sm.get("p95_latency_ms"),
        "prompt_tokens": sm.get("total_prompt_tokens", 0),
        "completion_tokens": sm.get("total_completion_tokens", 0),
        "judge_calls": sm.get("judge_calls", 0),
        "cost_total_cny": sm.get("est_cost_cny", 0),
        "gate_passed": all(g.get("passed") for g in gates if g.get("level") == "block"),
        "raw_summary": json.dumps(sm, ensure_ascii=False),
    }


def ingest(conn: psycopg.Connection, d: Path, stage: str, repeat_index: int) -> None:
    """写入一次运行的 run / item / gate 三张表。"""
    row = run_row(d, stage, repeat_index)
    cols = ",".join(RUN_COLS)
    ph = ",".join(["%s"] * len(RUN_COLS))
    upd = ",".join(f"{c}=EXCLUDED.{c}" for c in RUN_COLS if c != "run_id")
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO evals.eval_run ({cols}) VALUES ({ph}) "
                    f"ON CONFLICT (run_id) DO UPDATE SET {upd}",
                    [row[c] for c in RUN_COLS])

        f = d / "scores.csv"
        if f.exists():
            with f.open(encoding="utf-8") as fh:
                items = list(csv.DictReader(fh))
            cur.execute("DELETE FROM evals.eval_item WHERE run_id = %s", (row["run_id"],))
            cur.executemany(
                "INSERT INTO evals.eval_item (run_id,qid,category,difficulty,question_type,"
                "total,rule,retrieval,judge,passed,vetoed,veto_reason,recall_at_5,latency_ms,"
                "answer_chars,error,fail_reason) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(row["run_id"], it.get("qid"), it.get("category"), it.get("difficulty"),
                  it.get("question_type"), it.get("total") or None, it.get("rule") or None,
                  it.get("retrieval") or None, it.get("judge") or None,
                  str(it.get("passed", "")).lower() in ("true", "1", "yes"),
                  str(it.get("vetoed", "")).lower() in ("true", "1", "yes"),
                  it.get("veto_reason"), it.get("recall_at_5") or None,
                  it.get("latency_ms") or None, it.get("answer_chars") or None,
                  it.get("error"), it.get("fail_reason")) for it in items])

        gates = json.loads((d / "summary.json").read_text(encoding="utf-8")).get("gates", [])
        cur.execute("DELETE FROM evals.eval_gate WHERE run_id = %s", (row["run_id"],))
        cur.executemany(
            "INSERT INTO evals.eval_gate (run_id,name,level,kind,metric,op,threshold,actual,passed,reason) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(row["run_id"], g.get("name"), g.get("level"), g.get("kind", "absolute"),
              g.get("metric"), g.get("op"), g.get("threshold"), g.get("actual"),
              g.get("passed"), g.get("message", "")) for g in gates])
    conn.commit()
    print(f"  已入库 {row['run_id']}  总分={row['total_score']}  门禁={row['gate_passed']}")


def main() -> None:
    """把最近 N 次运行入库。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--repeats", type=int, default=1, help="入库最近 N 次")
    ap.add_argument("--stage", default="regression",
                    choices=["smoke", "regression", "release", "adhoc"])
    a = ap.parse_args()

    dsn = os.environ["EVAL_DB_DSN"]      # postgresql://user:pwd@127.0.0.1:5433/evals
    dirs = sorted((p for p in Path(a.runs_dir).iterdir()
                   if p.is_dir() and (p / "summary.json").exists()),
                  key=lambda p: p.name, reverse=True)[:a.repeats]
    with psycopg.connect(dsn) as conn:
        for i, d in enumerate(reversed(dirs), start=1):
            ingest(conn, d, a.stage, i)


if __name__ == "__main__":
    main()
```

```text
  已入库 20260318_020114_s3_opt_rag  总分=73.2  门禁=True
  已入库 20260318_021842_s3_opt_rag  总分=74.8  门禁=True
  已入库 20260318_023507_s3_opt_rag  总分=72.9  门禁=True
```

---

## 六、在线评测：离线分数管不了的那一半

到此为止，所有评测都是**离线**的：固定金标集、固定问题、有标准答案。它能回答"这次改动有没有让已知的题变差"，但回答不了下面三个问题：

| 离线评测答不了的问题 | 为什么 | 在线评测怎么答 |
|---|---|---|
| 真实用户问的和金标集像吗？ | 金标集是我们"以为"用户会问的 | 采集线上真实 query 分布，对比金标集覆盖度 |
| 用户到底满意吗？ | judge 打 4 分 ≠ 客服觉得有用 | 点赞点踩 + 原因标签 |
| 没有标准答案的新问题答得怎么样？ | 线上 99.9% 的问题没有金标答案 | 无标注质量估计（LLM 抽样打分 + 代理指标） |

```mermaid
flowchart LR
    U["真实用户请求"] --> APP["问答服务<br/>app/ FastAPI"]
    APP --> R["回答 + trace_id"]
    R --> U

    APP -.->|"埋点"| T[("trace 表<br/>Langfuse / 自建")]
    U -.->|"点赞点踩 + 原因"| FB[("online_feedback")]
    T -.->|"每小时分层抽样 1%"| S["LLM 抽样打分<br/>sampler.py"]
    S --> Q[("online_quality")]
    T -.->|"会话特征"| P["代理指标计算<br/>追问次数 / 转人工 / 会话时长"]
    P --> Q

    FB --> DASH["在线质量看板"]
    Q --> DASH
    DASH --> BC["badcase 池"]
    BC -->|"人工标注"| GOLD[("金标集<br/>第 8.2 章")]
    GOLD -.->|"扩充回归集"| OFF["离线流水线"]

    style GOLD fill:#f0fff4,stroke:#2f855a
```

### 6.1 用户反馈采集：点赞点踩 + 原因标签

#### 6.1.1 为什么必须带"原因标签"

**只有赞/踩的反馈几乎没用。** 你只知道 12% 的回答被踩了，但不知道是"答错了"还是"太啰嗦"——**这两个问题的修复路径完全不同**（前者查语料和检索，后者改 prompt）。

原因标签的设计原则：**5~7 个选项，覆盖你已知的修复路径，每个标签直接对应一个负责方**。

| 标签（用户看到的文案） | 内部代码 | 分派给 |
|---|---|---|
| 答案不对 / 和手册不一致 | `wrong_fact` | 语料 + 检索 |
| 没找到我要的信息 | `not_found` | 检索 + 语料覆盖 |
| 答得太笼统，不能操作 | `too_vague` | prompt（要求给步骤和页码） |
| 太啰嗦，看不下去 | `too_long` | prompt |
| 该问我清楚却瞎猜 | `should_clarify` | prompt + 追问策略 |
| 明明能答却拒绝了 | `over_refusal` | 安全策略阈值 |
| 响应太慢 | `too_slow` | 性能（第 9 章） |
| 其他（填写） | `other` | 人工看 |

#### 6.1.2 表结构

```sql
-- file: evals/pipeline/schema.sql（续）
-- ==================== 在线反馈 ====================
CREATE TABLE IF NOT EXISTS evals.online_feedback (
    id            BIGSERIAL   PRIMARY KEY,
    trace_id      TEXT        NOT NULL,           -- 与 core/instrument.py 的 trace 对齐
    message_id    TEXT        NOT NULL,           -- 一次回答的唯一 id（前端持有）
    session_id    TEXT        NOT NULL,
    user_id       TEXT,                           -- 脱敏后的用户标识（工号哈希）
    user_role     TEXT,                           -- 一线客服 | 现场工程师 | 售后管理者

    rating        SMALLINT    NOT NULL,           -- 1=赞, -1=踩, 0=撤销
    reason_codes  TEXT[]      DEFAULT '{}',       -- 可多选：{wrong_fact,too_vague}
    reason_text   TEXT,                           -- 自由输入（脱敏后入库）

    -- 冗余快照：反馈必须能独立复现，不能依赖 trace 表还在
    question      TEXT,
    answer        TEXT,
    answer_chars  INTEGER,
    cited_kb_ids  TEXT[]      DEFAULT '{}',       -- 这次回答引用了哪些知识条目
    system_version TEXT,                          -- 出问题时线上是哪个版本
    ab_variant    TEXT,                           -- AB 实验分组（见 6.3）
    latency_ms    INTEGER,

    handled       BOOLEAN     NOT NULL DEFAULT FALSE,   -- 是否已被处理（进 badcase 流程）
    ticket_id     TEXT,                                 -- 关联的 badcase 工单号
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 同一条回答同一用户只保留最新一次评价
    UNIQUE (message_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_fb_time     ON evals.online_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fb_bad      ON evals.online_feedback (created_at DESC) WHERE rating = -1;
CREATE INDEX IF NOT EXISTS idx_fb_unhandled ON evals.online_feedback (created_at) WHERE rating = -1 AND handled = FALSE;
CREATE INDEX IF NOT EXISTS idx_fb_reason   ON evals.online_feedback USING GIN (reason_codes);
CREATE INDEX IF NOT EXISTS idx_fb_variant  ON evals.online_feedback (ab_variant, created_at DESC);

-- 每日反馈汇总（看板用）
CREATE OR REPLACE VIEW evals.v_daily_feedback AS
SELECT date_trunc('day', created_at)::date AS day,
       system_version,
       count(*)                                               AS n_feedback,
       sum(CASE WHEN rating = 1  THEN 1 ELSE 0 END)           AS n_up,
       sum(CASE WHEN rating = -1 THEN 1 ELSE 0 END)           AS n_down,
       round(sum(CASE WHEN rating = 1 THEN 1 ELSE 0 END)::numeric
             / NULLIF(count(*) FILTER (WHERE rating <> 0), 0), 4) AS satisfaction,
       count(*) FILTER (WHERE rating = -1 AND NOT handled)    AS n_unhandled
FROM evals.online_feedback
GROUP BY 1, 2;
```

#### 6.1.3 埋点代码（服务端）

```python
# file: evals/online/feedback_api.py
# 运行环境：Python 3.11
# 依赖：fastapi, psycopg[binary]>=3.2, pydantic>=2
"""在线反馈采集接口。挂载到主应用（app/main.py）的 /api/feedback 路径下。

设计要点：
  1. **幂等**：同一 message_id + user_id 重复提交视为修改，不产生新记录；
  2. **快照冗余**：把问题/答案/引用/版本一起存下来，反馈能脱离 trace 表独立复现；
  3. **不阻塞主链路**：写库失败只记日志，绝不影响用户看到的界面；
  4. **脱敏**：自由文本走 core/logger.py 的脱敏规则再入库。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import psycopg
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field

from core.logger import mask_pii          # 第 0.2 章的脱敏工具

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

REASON_CODES = {"wrong_fact", "not_found", "too_vague", "too_long",
                "should_clarify", "over_refusal", "too_slow", "other"}


class FeedbackIn(BaseModel):
    """前端提交的反馈。"""
    message_id: str = Field(min_length=1, max_length=64)
    trace_id: str = Field(default="", max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    rating: Literal[-1, 0, 1]
    reason_codes: list[str] = Field(default_factory=list, max_length=8)
    reason_text: str = Field(default="", max_length=500)

    def validated_codes(self) -> list[str]:
        """过滤非法标签，避免前端传脏数据污染统计。"""
        return [c for c in self.reason_codes if c in REASON_CODES]


def _dsn() -> str:
    """评测库连接串。"""
    return os.environ["EVAL_DB_DSN"]


def _persist(payload: dict) -> None:
    """后台写库。失败只打日志，不抛给用户。"""
    sql = """
    INSERT INTO evals.online_feedback
        (trace_id, message_id, session_id, user_id, user_role, rating, reason_codes,
         reason_text, question, answer, answer_chars, cited_kb_ids, system_version,
         ab_variant, latency_ms)
    VALUES (%(trace_id)s, %(message_id)s, %(session_id)s, %(user_id)s, %(user_role)s,
            %(rating)s, %(reason_codes)s, %(reason_text)s, %(question)s, %(answer)s,
            %(answer_chars)s, %(cited_kb_ids)s, %(system_version)s, %(ab_variant)s,
            %(latency_ms)s)
    ON CONFLICT (message_id, user_id) DO UPDATE SET
        rating = EXCLUDED.rating,
        reason_codes = EXCLUDED.reason_codes,
        reason_text = EXCLUDED.reason_text,
        created_at = now(),
        handled = FALSE          -- 用户改了评价，重新进待处理队列
    """
    try:
        with psycopg.connect(_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(sql, payload)
            conn.commit()
    except Exception as e:                      # noqa: BLE001 —— 反馈写失败不能影响业务
        from loguru import logger
        logger.warning(f"反馈写库失败 message_id={payload['message_id']}: {str(e)[:200]}")


def _load_snapshot(message_id: str) -> dict:
    """从消息表（app 侧）取这次回答的快照。

    实际项目里这里读你自己的对话表或 Langfuse；取不到就留空，反馈本身仍然有价值。
    """
    sql = """
    SELECT question, answer, cited_kb_ids, system_version, ab_variant, latency_ms, trace_id
    FROM app.chat_message WHERE message_id = %s
    """
    try:
        with psycopg.connect(_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(sql, (message_id,))
            row = cur.fetchone()
            if not row:
                return {}
            return {"question": row[0], "answer": row[1], "cited_kb_ids": row[2] or [],
                    "system_version": row[3], "ab_variant": row[4],
                    "latency_ms": row[5], "trace_id": row[6]}
    except Exception:                            # noqa: BLE001
        return {}


@router.post("")
async def submit(fb: FeedbackIn, bg: BackgroundTasks,
                 x_user_id: str = Header(default="anonymous", alias="X-User-Id"),
                 x_user_role: str = Header(default="", alias="X-User-Role")) -> dict:
    """提交一条反馈。踩必须带原因标签，否则收不到有用信息。"""
    codes = fb.validated_codes()
    if fb.rating == -1 and not codes:
        raise HTTPException(status_code=422, detail="踩的时候请至少选择一个原因标签")

    snap = _load_snapshot(fb.message_id)
    answer = snap.get("answer", "")
    payload = {
        "trace_id": fb.trace_id or snap.get("trace_id", ""),
        "message_id": fb.message_id,
        "session_id": fb.session_id,
        "user_id": x_user_id,
        "user_role": x_user_role or None,
        "rating": fb.rating,
        "reason_codes": codes,
        "reason_text": mask_pii(fb.reason_text) if fb.reason_text else None,
        "question": snap.get("question"),
        "answer": answer,
        "answer_chars": len(answer) if answer else None,
        "cited_kb_ids": snap.get("cited_kb_ids", []),
        "system_version": snap.get("system_version"),
        "ab_variant": snap.get("ab_variant"),
        "latency_ms": snap.get("latency_ms"),
    }
    bg.add_task(_persist, payload)              # 后台写，接口 5ms 内返回
    return {"ok": True, "received_at": datetime.now().isoformat(timespec="seconds")}


@router.get("/stats")
async def stats(days: int = 7) -> dict:
    """最近 N 天的反馈统计，给运营看。"""
    sql = """
    SELECT count(*) FILTER (WHERE rating = 1)                       AS n_up,
           count(*) FILTER (WHERE rating = -1)                      AS n_down,
           count(*) FILTER (WHERE rating = -1 AND NOT handled)      AS n_unhandled,
           count(DISTINCT session_id)                               AS n_sessions
    FROM evals.online_feedback
    WHERE created_at >= now() - make_interval(days => %s)
    """
    sql_reason = """
    SELECT unnest(reason_codes) AS code, count(*) AS n
    FROM evals.online_feedback
    WHERE rating = -1 AND created_at >= now() - make_interval(days => %s)
    GROUP BY 1 ORDER BY 2 DESC
    """
    with psycopg.connect(_dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, (days,))
        n_up, n_down, n_unhandled, n_sessions = cur.fetchone()
        cur.execute(sql_reason, (days,))
        reasons = [{"code": c, "n": n} for c, n in cur.fetchall()]
    total = (n_up or 0) + (n_down or 0)
    return {"days": days, "n_up": n_up, "n_down": n_down,
            "satisfaction": round((n_up or 0) / total, 4) if total else None,
            "n_unhandled": n_unhandled, "n_sessions": n_sessions,
            "feedback_rate_note": "反馈率 = 有评价的回答数 / 总回答数，通常只有 1%~5%",
            "top_reasons": reasons}
```

前端埋点的四个要点（这部分决定了你能不能收到足够多的反馈）：

```javascript
// 前端（Streamlit / Gradio / 自研 Vue 都一样）的埋点约定
// 1. message_id 必须由后端在回答时下发，前端原样回传 —— 不要前端自己生成
// 2. 点踩弹出原因标签多选框，"其他"才展开文本框（降低填写成本）
// 3. 反馈接口 fire-and-forget，不要 loading 态，不要因为网络失败弹错误框
// 4. 已反馈的消息把按钮置为选中态，允许修改（用户经常先手滑点赞再改）
async function sendFeedback(messageId, sessionId, rating, reasonCodes, reasonText) {
  try {
    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-Id': window.__USER_HASH__ },
      body: JSON.stringify({
        message_id: messageId, session_id: sessionId,
        rating, reason_codes: reasonCodes, reason_text: reasonText || ''
      }),
      keepalive: true            // 用户点完就关页面时也能发出去
    });
  } catch (e) { /* 静默失败，不打扰用户 */ }
}
```

> **反馈率的真实预期**：**1%~5%**（示例性数据，取决于按钮位置和用户角色）。也就是说 1000 次回答只有 10~50 条反馈。**所以反馈数据只能用来"发现问题方向"，不能用来算准确率**——它的样本是严重有偏的（不满意的人更愿意点）。**要算质量，用下面的抽样打分。**

### 6.2 无标注在线质量估计

线上问题没有标准答案，怎么知道答得好不好？三条路，**组合使用**：

```mermaid
flowchart TD
    ONLINE["线上真实请求<br/>（无标准答案）"] --> A["路径一：LLM 抽样打分<br/>抽 1% 让 judge 打分<br/>成本可控、有绝对分数"]
    ONLINE --> B["路径二：自洽性/置信度<br/>答案与引用的一致性<br/>judge 置信度分布"]
    ONLINE --> C["路径三：会话行为代理指标<br/>追问次数 / 重问率<br/>转人工率 / 会话时长"]
    A --> M["在线质量分<br/>三者加权 + 分别监控"]
    B --> M
    C --> M
    M --> ALERT{"低于阈值？"}
    ALERT -- 是 --> BC["进 badcase 池<br/>（第七节 SOP）"]
    style M fill:#ebf8ff,stroke:#2b6cb0
```

三条路径的取舍：

| 路径 | 优点 | 缺点 | 本书用法 |
|---|---|---|---|
| LLM 抽样打分 | 有绝对分数，可与离线口径对齐 | 花钱；judge 没有标准答案时只能评"忠实度/可用性"，评不了"事实正确性" | **主力**，按分层抽 1% |
| 置信度 / 自洽性 | 免费（复用已有数据） | 只是相关，不是因果；高置信也可能答错 | 用作**筛选器**，低置信的优先送去打分 |
| 会话行为代理指标 | 完全免费，最贴近业务价值 | 滞后、噪声大、受 UI 影响 | 用作**趋势监控**，不做单条判定 |

#### 6.2.1 表结构

```sql
-- file: evals/pipeline/schema.sql（续）
-- ==================== 在线质量抽样 ====================
CREATE TABLE IF NOT EXISTS evals.online_quality (
    id              BIGSERIAL   PRIMARY KEY,
    trace_id        TEXT        NOT NULL,
    message_id      TEXT        NOT NULL UNIQUE,
    session_id      TEXT        NOT NULL,
    sampled_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    sample_reason   TEXT        NOT NULL,        -- random | low_confidence | thumbs_down | new_intent | long_latency
    system_version  TEXT,
    ab_variant      TEXT,

    question        TEXT        NOT NULL,
    answer          TEXT,
    n_contexts      SMALLINT,
    cited_kb_ids    TEXT[]      DEFAULT '{}',

    -- LLM 抽样打分（无标准答案，只评可验证的维度）
    judge_model     TEXT,
    judge_version   TEXT,                        -- judge prompt 版本，升级后分数不可比
    score_faithful  NUMERIC(4,2),                -- 答案是否只依据引用（0~5）
    score_relevant  NUMERIC(4,2),                -- 是否答了所问（0~5）
    score_usable    NUMERIC(4,2),                -- 是否可操作（有步骤/参数/页码）（0~5）
    score_total     NUMERIC(5,2),                -- 加权总分（0~100）
    judge_confidence NUMERIC(4,3),               -- judge 自评置信度（0~1）
    judge_reason    TEXT,

    -- 会话行为代理指标
    followup_count  SMALLINT    DEFAULT 0,       -- 本轮之后用户又追问了几次
    is_rephrase     BOOLEAN     DEFAULT FALSE,   -- 用户是不是换了说法重问（强负信号）
    escalated       BOOLEAN     DEFAULT FALSE,   -- 是否转人工
    session_turns   SMALLINT,
    dwell_ms        INTEGER,                     -- 用户在这条回答上的停留时长

    is_badcase      BOOLEAN     DEFAULT FALSE,
    ticket_id       TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_oq_time    ON evals.online_quality (sampled_at DESC);
CREATE INDEX IF NOT EXISTS idx_oq_score   ON evals.online_quality (score_total);
CREATE INDEX IF NOT EXISTS idx_oq_variant ON evals.online_quality (ab_variant, sampled_at DESC);
CREATE INDEX IF NOT EXISTS idx_oq_bad     ON evals.online_quality (sampled_at DESC) WHERE is_badcase;

-- 在线质量日报视图
CREATE OR REPLACE VIEW evals.v_daily_online_quality AS
SELECT date_trunc('day', sampled_at)::date AS day,
       system_version, ab_variant,
       count(*)                                                    AS n_sampled,
       round(avg(score_total), 2)                                  AS avg_score,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY score_total)     AS median_score,
       round(avg(score_faithful), 2)                               AS avg_faithful,
       round(avg(CASE WHEN score_total < 60 THEN 1 ELSE 0 END)::numeric, 4) AS low_score_rate,
       round(avg(judge_confidence), 3)                             AS avg_confidence,
       round(avg(followup_count), 2)                               AS avg_followup,
       round(avg(CASE WHEN is_rephrase THEN 1 ELSE 0 END)::numeric, 4) AS rephrase_rate,
       round(avg(CASE WHEN escalated  THEN 1 ELSE 0 END)::numeric, 4) AS escalation_rate
FROM evals.online_quality
GROUP BY 1, 2, 3;
```

#### 6.2.2 抽样打分实现

```python
# file: evals/online/sampler.py
# 运行环境：Python 3.11
# 依赖：psycopg[binary]>=3.2, openai>=1.40（复用 core/llm.py）
"""在线无标注质量估计：分层抽样 → LLM 打分 → 代理指标 → 入库 → 低分进 badcase 池。

按小时跑（crontab: 15 * * * *）。三个设计决定：
  1. **分层抽样**，不是纯随机：踩过的、低置信的、新意图的、慢的，都要多抽；
  2. **无标准答案时只评三个维度**：忠实度、相关性、可用性。
     事实正确性（correctness）**评不了** —— 没有金标答案，judge 只会自说自话；
  3. **judge 必须输出置信度**，低置信的样本单独拉出来人工看。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass

import psycopg

from core.llm import get_async_openai_client

JUDGE_VERSION = "online-v3"          # 改 prompt 必须升版本，否则历史分数不可比

JUDGE_PROMPT = """你是华成机电售后知识库的质量审核员。你将看到一个用户问题、系统给出的回答，
以及系统检索到的资料片段。**注意：你没有标准答案**，所以只评下面三个可以客观判断的维度。

评分维度（各 0~5 分，允许 0.5 分）：
1. faithfulness（忠实度）：回答中的每个事实性陈述，是否都能在资料片段里找到依据？
   资料里没有却说得很确定 → 2 分以下。没有资料却硬答 → 0 分。
2. relevance（相关性）：是否回答了用户实际问的问题？答偏了、答了别的型号 → 2 分以下。
3. usability（可用性）：一线客服拿到这个回答能不能直接用？
   有明确步骤/参数/型号/出处 → 4 分以上；只有笼统结论 → 2 分。

同时给出：
- confidence（0~1）：你对自己这次打分的确信程度。资料不足、问题歧义、
  你无法判断对错时，**必须给低置信度**（< 0.5），不要硬打分。
- reason：一句话说明最主要的扣分点（没扣分就写"无明显问题"）。

只输出 JSON，不要任何其他文字：
{"faithfulness": 4.0, "relevance": 5.0, "usability": 3.5, "confidence": 0.8, "reason": "..."}

---
【用户问题】
{question}

【系统回答】
{answer}

【检索到的资料片段】
{contexts}
"""

# 分层抽样配额：每小时最多抽多少条（示例性配置，按你的流量与预算调）
QUOTA = {
    "thumbs_down": 50,      # 被踩的：全抽（最有价值）
    "low_confidence": 20,   # 上一轮打分置信度低的
    "new_intent": 20,       # 与金标集相似度低的新问题
    "long_latency": 10,     # P95 以上的慢请求
    "random": 60,           # 兜底随机，保证无偏样本
}

WEIGHTS = {"faithfulness": 0.45, "relevance": 0.25, "usability": 0.30}


@dataclass
class Candidate:
    """待打分的一条线上回答。"""
    trace_id: str
    message_id: str
    session_id: str
    question: str
    answer: str
    contexts: list[str]
    cited_kb_ids: list[str]
    system_version: str
    ab_variant: str | None
    reason: str


def fetch_candidates(conn: psycopg.Connection, hours: int = 1) -> list[Candidate]:
    """按分层配额取候选。app.chat_message 是业务侧的对话表。"""
    out: list[Candidate] = []

    base = """
    SELECT m.trace_id, m.message_id, m.session_id, m.question, m.answer,
           m.contexts, m.cited_kb_ids, m.system_version, m.ab_variant
    FROM app.chat_message m
    LEFT JOIN evals.online_quality q ON q.message_id = m.message_id
    WHERE m.created_at >= now() - make_interval(hours => %(h)s)
      AND q.id IS NULL                      -- 没打过分的
      AND m.answer IS NOT NULL AND length(m.answer) > 0
    """
    strata = {
        "thumbs_down": base + """
          AND EXISTS (SELECT 1 FROM evals.online_feedback f
                      WHERE f.message_id = m.message_id AND f.rating = -1)
          ORDER BY m.created_at DESC LIMIT %(lim)s""",
        "long_latency": base + " AND m.latency_ms > 5000 ORDER BY m.latency_ms DESC LIMIT %(lim)s",
        "new_intent": base + " AND coalesce(m.golden_similarity, 0) < 0.55 ORDER BY random() LIMIT %(lim)s",
        "low_confidence": base + """
          AND EXISTS (SELECT 1 FROM evals.online_quality q2
                      WHERE q2.session_id = m.session_id AND q2.judge_confidence < 0.5)
          ORDER BY random() LIMIT %(lim)s""",
        "random": base + " ORDER BY random() LIMIT %(lim)s",
    }

    seen: set[str] = set()
    with conn.cursor() as cur:
        for reason, sql in strata.items():
            cur.execute(sql, {"h": hours, "lim": QUOTA[reason]})
            for row in cur.fetchall():
                if row[1] in seen:
                    continue
                seen.add(row[1])
                out.append(Candidate(trace_id=row[0], message_id=row[1], session_id=row[2],
                                     question=row[3], answer=row[4],
                                     contexts=list(row[5] or []), cited_kb_ids=list(row[6] or []),
                                     system_version=row[7], ab_variant=row[8], reason=reason))
    random.shuffle(out)
    return out


async def score_one(client, c: Candidate, model: str, sem: asyncio.Semaphore) -> dict | None:
    """让 judge 给一条回答打分。解析失败返回 None，由调用方计入失败数。"""
    ctx = "\n\n".join(f"[片段{i+1}] {t[:800]}" for i, t in enumerate(c.contexts[:5])) or "（无检索结果）"
    prompt = (JUDGE_PROMPT.replace("{question}", c.question[:1500])
                          .replace("{answer}", c.answer[:3000])
                          .replace("{contexts}", ctx))
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, temperature=0.0,
                response_format={"type": "json_object"},     # 强制 JSON，减少解析失败
                messages=[{"role": "user", "content": prompt}],
                timeout=60,
            )
            data = json.loads(r.choices[0].message.content)
        except Exception as e:                                # noqa: BLE001
            print(f"  [WARN] {c.message_id} 打分失败: {str(e)[:120]}")
            return None

    def num(k: str, default: float = 0.0) -> float:
        """安全取数并裁剪到 0~5。"""
        try:
            return max(0.0, min(5.0, float(data.get(k, default))))
        except (TypeError, ValueError):
            return default

    f, rel, u = num("faithfulness"), num("relevance"), num("usability")
    total = (f * WEIGHTS["faithfulness"] + rel * WEIGHTS["relevance"]
             + u * WEIGHTS["usability"]) / 5 * 100
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"candidate": c, "score_faithful": round(f, 2), "score_relevant": round(rel, 2),
            "score_usable": round(u, 2), "score_total": round(total, 2),
            "judge_confidence": round(conf, 3),
            "judge_reason": str(data.get("reason", ""))[:500], "model": model}


def session_features(conn: psycopg.Connection, c: Candidate) -> dict:
    """计算会话行为代理指标：追问次数、是否换问法重问、是否转人工。"""
    sql = """
    WITH cur AS (SELECT created_at, question FROM app.chat_message WHERE message_id = %s)
    SELECT
      (SELECT count(*) FROM app.chat_message m, cur
        WHERE m.session_id = %s AND m.created_at > cur.created_at
          AND m.created_at < cur.created_at + INTERVAL '10 minutes')      AS followup_count,
      -- 换问法重问：10 分钟内的下一个问题与本问题相似度高（相似度由应用侧预计算）
      (SELECT coalesce(bool_or(m.prev_question_similarity > 0.8), FALSE)
         FROM app.chat_message m, cur
        WHERE m.session_id = %s AND m.created_at > cur.created_at
          AND m.created_at < cur.created_at + INTERVAL '5 minutes')       AS is_rephrase,
      (SELECT coalesce(bool_or(m.escalated), FALSE) FROM app.chat_message m
        WHERE m.session_id = %s)                                         AS escalated,
      (SELECT count(*) FROM app.chat_message m WHERE m.session_id = %s)   AS session_turns
    """
    with conn.cursor() as cur_:
        cur_.execute(sql, (c.message_id, c.session_id, c.session_id, c.session_id, c.session_id))
        row = cur_.fetchone() or (0, False, False, 0)
    return {"followup_count": row[0] or 0, "is_rephrase": bool(row[1]),
            "escalated": bool(row[2]), "session_turns": row[3] or 0}


def save(conn: psycopg.Connection, scored: dict, feats: dict, low_score_line: float) -> None:
    """写入 online_quality。总分低于阈值或 judge 判不忠实的，直接标为 badcase。"""
    c: Candidate = scored["candidate"]
    is_bad = (scored["score_total"] < low_score_line
              or scored["score_faithful"] <= 2.0
              or feats["is_rephrase"] or feats["escalated"])
    sql = """
    INSERT INTO evals.online_quality
      (trace_id, message_id, session_id, sample_reason, system_version, ab_variant,
       question, answer, n_contexts, cited_kb_ids,
       judge_model, judge_version, score_faithful, score_relevant, score_usable,
       score_total, judge_confidence, judge_reason,
       followup_count, is_rephrase, escalated, session_turns, is_badcase)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (message_id) DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (c.trace_id, c.message_id, c.session_id, c.reason, c.system_version,
                          c.ab_variant, c.question[:4000], c.answer[:8000], len(c.contexts),
                          c.cited_kb_ids, scored["model"], JUDGE_VERSION,
                          scored["score_faithful"], scored["score_relevant"], scored["score_usable"],
                          scored["score_total"], scored["judge_confidence"], scored["judge_reason"],
                          feats["followup_count"], feats["is_rephrase"], feats["escalated"],
                          feats["session_turns"], is_bad))
    conn.commit()


async def main_async(hours: int, model: str, concurrency: int, low_score_line: float) -> None:
    """抽样 → 打分 → 算代理指标 → 入库 → 打印置信度分布。"""
    client = get_async_openai_client()
    sem = asyncio.Semaphore(concurrency)
    with psycopg.connect(os.environ["EVAL_DB_DSN"]) as conn:
        cands = fetch_candidates(conn, hours=hours)
        print(f"抽样 {len(cands)} 条（分层："
              f"{ {r: sum(1 for c in cands if c.reason == r) for r in QUOTA} }）")
        if not cands:
            return

        results = await asyncio.gather(*[score_one(client, c, model, sem) for c in cands])
        ok = [r for r in results if r]
        print(f"打分成功 {len(ok)}/{len(cands)}")

        for r in ok:
            save(conn, r, session_features(conn, r["candidate"]), low_score_line)

    # ---------- 置信度分布：这张表比平均分更重要 ----------
    buckets = {"高置信 ≥0.8": 0, "中 0.5~0.8": 0, "低 <0.5（需人工）": 0}
    for r in ok:
        cf = r["judge_confidence"]
        buckets["高置信 ≥0.8" if cf >= 0.8 else
                "中 0.5~0.8" if cf >= 0.5 else "低 <0.5（需人工）"] += 1
    print("\n置信度分布：")
    for k, v in buckets.items():
        print(f"  {k:<18} {v:>4} 条 ({v / len(ok):.1%})")

    hi = [r["score_total"] for r in ok if r["judge_confidence"] >= 0.8]
    if hi:
        print(f"\n高置信样本平均分：{sum(hi) / len(hi):.2f}（n={len(hi)}）"
              f"  <- **只有这个数字可以拿去汇报**")
    print(f"低置信样本 {buckets['低 <0.5（需人工）']} 条已入库，"
          f"请在 badcase 看板里人工复核")


def main() -> None:
    """CLI 入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--low-score-line", type=float, default=60.0)
    a = ap.parse_args()
    asyncio.run(main_async(a.hours, a.model, a.concurrency, a.low_score_line))


if __name__ == "__main__":
    main()
```

```text
抽样 137 条（分层：{'thumbs_down': 11, 'low_confidence': 14, 'new_intent': 20, 'long_latency': 10, 'random': 60}）
打分成功 134/137
  [WARN] msg_8f31a2 打分失败: Expecting value: line 1 column 1 (char 0)

置信度分布：
  高置信 ≥0.8          89 条 (66.4%)
  中 0.5~0.8           31 条 (23.1%)
  低 <0.5（需人工）      14 条 (10.4%)

高置信样本平均分：78.34（n=89）  <- **只有这个数字可以拿去汇报**
低置信样本 14 条已入库，请在 badcase 看板里人工复核
（示例性数据）
```

**三条读数纪律**：

1. **只用高置信样本算平均分。** 把 134 条不分置信度地平均，得到的数字混进了 judge 自己都不确定的判断。
2. **低置信比例本身是个指标。** 它从 10% 涨到 25%，说明线上出现了大量"资料不足/问题歧义"的新场景——**这是知识库该补充的信号**，比分数下降更值得关注。
3. **代理指标只看趋势，不做单条判定。** "用户追问了 2 次"可能是回答不好，也可能是他在连续排查一个复杂故障。但**换问法重问率（`rephrase_rate`）从 8% 涨到 15%，一定是出了问题。**

代理指标与真实质量的关系（示例性相关性，需在你自己的数据上验证）：

| 代理指标 | 与质量的关系 | 解读注意 |
|---|---|---|
| 换问法重问率 | **强负相关**，最可信的免费信号 | 用户换个说法再问，等于直接告诉你"没答好" |
| 转人工率 | 强负相关 | 但受入口位置影响大，UI 改版会造成阶跃 |
| 追问次数 | 弱负相关 | 复杂故障排查天然多轮，要按 `question_type` 分开看 |
| 回答停留时长 | 中等正相关（太短=没看，太长=看不懂） | 需要前端埋点支持，噪声大 |
| 复制答案的比例 | **强正相关**（客服复制走 = 觉得有用） | 强烈建议埋这个点，成本低、信号强 |

### 6.3 影子流量与 AB 实验

离线全过、上线还是可能出事。**影子流量和 AB 实验是上线前的最后两道保险**，区别是：

| | 影子流量（Shadow Traffic） | AB 实验 |
|---|---|---|
| **用户能看到新版本吗** | **看不到**（新版本的输出只入库，不返回） | 看得到（按比例分流） |
| **风险** | 几乎为零 | 有（所以要小比例起步 + 守护指标） |
| **能测什么** | 技术指标：延迟、成本、错误率、与旧版答案的差异度 | 业务指标：满意度、转人工率、解决率 |
| **不能测什么** | 用户真实反应 | —— |
| **成本** | 双倍推理成本 | 只有新版本那部分的增量 |
| **什么时候用** | 上线前 1~3 天，**跑真实流量分布** | 影子流量没问题之后 |

```mermaid
sequenceDiagram
    participant U as 用户
    participant GW as 网关/应用层
    participant V1 as 旧版本 v1.3（线上）
    participant V2 as 新版本 v1.4（影子）
    participant DB as 对比库

    U->>GW: 提问「XJ-200-B3 报 E057 怎么处理」
    GW->>V1: 同步调用
    V1-->>GW: 答案 A（返回给用户）
    GW-->>U: 答案 A
    Note over GW,V2: 异步、不阻塞、失败不影响主链路
    GW->>V2: 复制同一请求（fire-and-forget）
    V2-->>DB: 答案 B + 延迟 + 成本 + 引用
    GW->>DB: 答案 A + 延迟 + 成本 + 引用
    DB->>DB: 离线比对：差异度、延迟分布、成本、<br/>judge 双盲对比 A vs B
```

#### 6.3.1 影子流量实现

```python
# file: evals/online/shadow.py
# 运行环境：Python 3.11
"""影子流量：把线上请求复制给新版本，只入库不返回。

三条铁律：
  1. **绝不阻塞主链路**：用 asyncio.create_task + 超时，失败只记日志；
  2. **绝不产生副作用**：影子调用必须走只读模式（不写工单、不发通知、不扣额度）；
  3. **采样而非全量**：默认 20%，避免推理成本翻倍。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time

import psycopg
from loguru import logger

SHADOW_RATIO = float(os.getenv("SHADOW_RATIO", "0.2"))
SHADOW_TIMEOUT_S = float(os.getenv("SHADOW_TIMEOUT_S", "30"))


def should_shadow(trace_id: str, ratio: float = SHADOW_RATIO) -> bool:
    """按 trace_id 稳定哈希决定是否复制，保证同一请求的决定可复现。"""
    h = int(hashlib.md5(f"shadow:{trace_id}".encode()).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0 < ratio


async def _record(row: dict) -> None:
    """写入影子对比表。"""
    sql = """
    INSERT INTO evals.shadow_compare
      (trace_id, question, answer_prod, answer_shadow, version_prod, version_shadow,
       latency_prod_ms, latency_shadow_ms, cost_prod_cny, cost_shadow_cny,
       cited_prod, cited_shadow, shadow_error)
    VALUES (%(trace_id)s, %(question)s, %(answer_prod)s, %(answer_shadow)s,
            %(version_prod)s, %(version_shadow)s, %(latency_prod_ms)s, %(latency_shadow_ms)s,
            %(cost_prod_cny)s, %(cost_shadow_cny)s, %(cited_prod)s, %(cited_shadow)s,
            %(shadow_error)s)
    ON CONFLICT (trace_id) DO NOTHING
    """
    try:
        with psycopg.connect(os.environ["EVAL_DB_DSN"], connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(sql, row)
            conn.commit()
    except Exception as e:                       # noqa: BLE001
        logger.warning(f"影子结果写库失败 {row['trace_id']}: {str(e)[:150]}")


async def run_shadow(shadow_system, question: str, trace_id: str,
                     prod_result: dict, version_shadow: str) -> None:
    """执行影子调用并落库。**这个函数里的任何异常都不允许向外传播。**"""
    t0 = time.perf_counter()
    answer, err, cited, cost = "", None, [], 0.0
    try:
        out = await asyncio.wait_for(
            shadow_system.ask(question, read_only=True), timeout=SHADOW_TIMEOUT_S)
        answer = out.get("answer", "")
        cited = out.get("cited_kb_ids", [])
        cost = float(out.get("cost_cny", 0) or 0)
    except asyncio.TimeoutError:
        err = f"timeout>{SHADOW_TIMEOUT_S}s"
    except Exception as e:                       # noqa: BLE001
        err = str(e)[:300]

    await _record({
        "trace_id": trace_id, "question": question[:4000],
        "answer_prod": prod_result.get("answer", "")[:8000], "answer_shadow": answer[:8000],
        "version_prod": prod_result.get("version", ""), "version_shadow": version_shadow,
        "latency_prod_ms": prod_result.get("latency_ms"),
        "latency_shadow_ms": int((time.perf_counter() - t0) * 1000),
        "cost_prod_cny": prod_result.get("cost_cny", 0), "cost_shadow_cny": cost,
        "cited_prod": prod_result.get("cited_kb_ids", []), "cited_shadow": cited,
        "shadow_error": err,
    })


# ---------- 在 FastAPI 路由里这样挂 ----------
# @app.post("/api/ask")
# async def ask(req: AskRequest):
#     result = await prod_system.ask(req.question)          # 主链路，同步
#     if SHADOW_ENABLED and should_shadow(trace_id):
#         # create_task 不 await：主链路立刻返回，影子在后台跑
#         asyncio.create_task(run_shadow(shadow_system, req.question, trace_id,
#                                        result, version_shadow="v1.4.0"))
#     return result
```

```sql
-- file: evals/pipeline/schema.sql（续）
CREATE TABLE IF NOT EXISTS evals.shadow_compare (
    id                BIGSERIAL   PRIMARY KEY,
    trace_id          TEXT        NOT NULL UNIQUE,
    question          TEXT        NOT NULL,
    answer_prod       TEXT,
    answer_shadow     TEXT,
    version_prod      TEXT,
    version_shadow    TEXT,
    latency_prod_ms   INTEGER,
    latency_shadow_ms INTEGER,
    cost_prod_cny     NUMERIC(10,6),
    cost_shadow_cny   NUMERIC(10,6),
    cited_prod        TEXT[]      DEFAULT '{}',
    cited_shadow      TEXT[]      DEFAULT '{}',
    shadow_error      TEXT,
    -- 离线分析回填
    diff_ratio        NUMERIC(5,4),          -- 答案差异度（字符级或语义级）
    judge_winner      TEXT,                  -- prod | shadow | tie（双盲对比结果）
    judge_reason      TEXT,
    analyzed_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_shadow_time  ON evals.shadow_compare (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_shadow_diff  ON evals.shadow_compare (diff_ratio DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_shadow_error ON evals.shadow_compare (created_at DESC) WHERE shadow_error IS NOT NULL;
```

影子流量跑一天后，看这四个数字就够了：

```sql
-- 影子流量日报：新版本能不能上 AB 实验，看这一张表
SELECT
    count(*)                                                        AS n,
    count(*) FILTER (WHERE shadow_error IS NOT NULL)                AS n_error,
    round(avg(latency_shadow_ms)::numeric, 0)                       AS avg_lat_shadow,
    round(avg(latency_prod_ms)::numeric, 0)                         AS avg_lat_prod,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_shadow_ms)  AS p95_shadow,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_prod_ms)    AS p95_prod,
    round(sum(cost_shadow_cny) / NULLIF(sum(cost_prod_cny), 0), 3)  AS cost_ratio,
    round(avg(diff_ratio)::numeric, 3)                              AS avg_diff,
    count(*) FILTER (WHERE judge_winner = 'shadow')                 AS shadow_wins,
    count(*) FILTER (WHERE judge_winner = 'prod')                   AS prod_wins,
    count(*) FILTER (WHERE judge_winner = 'tie')                    AS ties
FROM evals.shadow_compare
WHERE created_at >= now() - INTERVAL '1 day';
```

```text
   n   | n_error | avg_lat_shadow | avg_lat_prod | p95_shadow | p95_prod | cost_ratio | avg_diff | shadow_wins | prod_wins | ties
-------+---------+----------------+--------------+------------+----------+------------+----------+-------------+-----------+------
  2184 |       7 |           2140 |         1980 |       4320 |     4010 |      1.180 |    0.412 |         318 |       196 |  184
（示例性数据）
```

**判定标准**（四条全过才进 AB）：

| 指标 | 门槛 | 本例 | 判定 |
|---|---|---|---|
| 影子错误率 | < 1% | 7/2184 = 0.32% | ✅ |
| P95 延迟增幅 | < 20% | 4320 vs 4010 = +7.7% | ✅ |
| 成本比 | < 1.3 | 1.18 | ✅ |
| 双盲对比胜率 | 新版胜率 > 败率，且抽 30 条人工确认 | 318 胜 / 196 败 / 184 平 | ✅ |

> **`avg_diff = 0.412` 这个数字要小心**：**差异度大不代表变好，也不代表变坏，只说明"改动确实生效了"。** 如果差异度接近 0，反而要怀疑新版本根本没被调用（影子挂错了）。

#### 6.3.2 AB 实验：分流实现

```python
# file: evals/online/ab.py
# 运行环境：Python 3.11
# 依赖：psycopg[binary]>=3.2（统计函数复用第 8.1 章 stats_eval.py）
"""AB 实验：稳定分流 + 样本量估算 + 统计检验 + 守护指标。

分流三条原则：
  1. **按 user_id 分流，不按请求分流**。同一个客服今天看到 A、明天看到 B，
     他的反馈就没法解释，而且体验割裂；
  2. **哈希里必须带实验名**，否则多个实验会分到同一批人（相关性污染）；
  3. **留出 holdout**：永远保留 5% 用户看旧版本，作为长期基线。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass

import psycopg

from stats_eval import bootstrap_ci, sample_size_two_proportions


# ==================== 分流 ====================

@dataclass(frozen=True)
class Experiment:
    """一个 AB 实验的定义。"""
    name: str                       # ab_rerank_v2
    variants: dict[str, float]      # {"control": 0.8, "treatment": 0.2}
    enabled: bool = True
    holdout_ratio: float = 0.05     # 永久对照组

    def __post_init__(self) -> None:
        """校验比例之和。"""
        s = sum(self.variants.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"分流比例之和必须为 1，当前 {s}")


def assign(exp: Experiment, user_id: str) -> str:
    """把用户稳定分配到一个变体。同一个 (实验, 用户) 永远得到同一个结果。"""
    if not exp.enabled:
        return "control"
    # 哈希里带实验名：不同实验的分组互相独立
    h = int(hashlib.sha256(f"{exp.name}:{user_id}".encode()).hexdigest()[:12], 16)
    bucket = (h % 10000) / 10000.0

    # 先切 holdout（用另一个盐，保证 holdout 不随实验比例变化而漂移）
    hh = int(hashlib.sha256(f"holdout:{user_id}".encode()).hexdigest()[:12], 16)
    if (hh % 10000) / 10000.0 < exp.holdout_ratio:
        return "holdout"

    acc = 0.0
    for name, ratio in exp.variants.items():
        acc += ratio
        if bucket < acc:
            return name
    return "control"


# ==================== 样本量估算 ====================

def plan(baseline_rate: float, mde_abs: float, daily_users: int,
         treatment_ratio: float = 0.2, alpha: float = 0.05, power: float = 0.8) -> dict:
    """算出"要跑多少天"。mde_abs 是你希望检出的绝对提升（如 0.05 = 5 个点）。"""
    per_group = sample_size_two_proportions(baseline_rate, baseline_rate + mde_abs,
                                            alpha=alpha, power=power)
    # 实验组比例小时，瓶颈在实验组
    need_total = math.ceil(per_group / min(treatment_ratio, 1 - treatment_ratio))
    days = math.ceil(need_total / max(daily_users, 1))
    return {"baseline_rate": baseline_rate, "mde_abs": mde_abs,
            "per_group_needed": per_group, "total_needed": need_total,
            "daily_users": daily_users, "estimated_days": days,
            "note": ("样本量按『用户』计，不是按『请求』计。"
                     "按请求计会严重高估检验效力（同一用户的多次请求不独立）。")}


# ==================== 统计检验 ====================

def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """两比例 z 检验，返回 (z, 双侧 p 值)。x=成功数，n=样本数。"""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p2 - p1) / se
    # 标准正态双侧 p 值（用 erf，不依赖 scipy）
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (z, p_val)


def analyze(dsn: str, exp_name: str, days: int = 14) -> dict:
    """拉取实验数据并做检验：主指标（满意度）+ 守护指标。"""
    sql = """
    SELECT f.ab_variant                                     AS variant,
           count(DISTINCT f.user_id)                        AS n_users,
           count(*)                                         AS n_feedback,
           count(*) FILTER (WHERE f.rating = 1)             AS n_up,
           count(*) FILTER (WHERE f.rating = -1)            AS n_down
    FROM evals.online_feedback f
    WHERE f.created_at >= now() - make_interval(days => %s)
      AND f.ab_variant IS NOT NULL AND f.rating <> 0
    GROUP BY 1
    """
    sql_guard = """
    SELECT q.ab_variant                                      AS variant,
           count(*)                                          AS n,
           avg(q.score_total)                                AS avg_score,
           avg(CASE WHEN q.escalated  THEN 1 ELSE 0 END)     AS escalation_rate,
           avg(CASE WHEN q.is_rephrase THEN 1 ELSE 0 END)    AS rephrase_rate,
           avg(q.followup_count)                             AS avg_followup
    FROM evals.online_quality q
    WHERE q.sampled_at >= now() - make_interval(days => %s)
      AND q.ab_variant IS NOT NULL
    GROUP BY 1
    """
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql, (days,))
        rows = {r[0]: {"n_users": r[1], "n_feedback": r[2], "n_up": r[3], "n_down": r[4]}
                for r in cur.fetchall()}
        cur.execute(sql_guard, (days,))
        guards = {r[0]: {"n": r[1], "avg_score": float(r[2] or 0),
                         "escalation_rate": float(r[3] or 0),
                         "rephrase_rate": float(r[4] or 0),
                         "avg_followup": float(r[5] or 0)} for r in cur.fetchall()}

    ctrl, treat = rows.get("control"), rows.get("treatment")
    out: dict = {"experiment": exp_name, "days": days, "variants": rows, "guards": guards}
    if not ctrl or not treat:
        out["verdict"] = "数据不足：control 或 treatment 分组无数据"
        return out

    z, p = two_proportion_z(ctrl["n_up"], ctrl["n_feedback"],
                            treat["n_up"], treat["n_feedback"])
    r1 = ctrl["n_up"] / ctrl["n_feedback"]
    r2 = treat["n_up"] / treat["n_feedback"]
    out["primary"] = {
        "metric": "satisfaction(点赞率)",
        "control": round(r1, 4), "treatment": round(r2, 4),
        "lift_abs": round(r2 - r1, 4),
        "lift_rel": round((r2 - r1) / r1, 4) if r1 else None,
        "z": round(z, 4), "p_value": round(p, 4),
        "significant": p < 0.05,
    }

    # 守护指标：任一恶化超过阈值就不许放量（示例阈值，按业务定）
    breaches = []
    gc, gt = guards.get("control"), guards.get("treatment")
    if gc and gt:
        if gt["escalation_rate"] > gc["escalation_rate"] + 0.02:
            breaches.append(f"转人工率上升 {gt['escalation_rate'] - gc['escalation_rate']:+.3f}")
        if gt["rephrase_rate"] > gc["rephrase_rate"] + 0.03:
            breaches.append(f"换问法重问率上升 {gt['rephrase_rate'] - gc['rephrase_rate']:+.3f}")
        if gt["avg_score"] < gc["avg_score"] - 2.0:
            breaches.append(f"在线质量分下降 {gt['avg_score'] - gc['avg_score']:+.2f}")
    out["guard_breaches"] = breaches

    if breaches:
        out["verdict"] = "❌ 守护指标恶化，不允许放量：" + "；".join(breaches)
    elif out["primary"]["significant"] and out["primary"]["lift_abs"] > 0:
        out["verdict"] = "✅ 主指标显著提升且守护指标正常，可以放量到下一阶段"
    elif out["primary"]["significant"]:
        out["verdict"] = "❌ 主指标显著下降，回滚"
    else:
        out["verdict"] = ("⏳ 主指标变化不显著，继续跑或加大样本。"
                          "注意：不要因为『看起来涨了』就提前结束实验（peeking 会大幅抬高假阳性率）")
    return out


def main() -> None:
    """CLI：plan 估样本量 / analyze 出结论 / assign 验证分流。"""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("plan", help="样本量与天数估算")
    p1.add_argument("--baseline", type=float, required=True, help="当前点赞率，如 0.72")
    p1.add_argument("--mde", type=float, default=0.05, help="希望检出的绝对提升")
    p1.add_argument("--daily-users", type=int, required=True)
    p1.add_argument("--treatment-ratio", type=float, default=0.2)

    p2 = sub.add_parser("analyze", help="跑统计检验")
    p2.add_argument("--name", required=True)
    p2.add_argument("--days", type=int, default=14)

    p3 = sub.add_parser("assign", help="验证分流均匀性")
    p3.add_argument("--name", required=True)
    p3.add_argument("--n", type=int, default=10000)

    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(a.baseline, a.mde, a.daily_users, a.treatment_ratio),
                         ensure_ascii=False, indent=2))
    elif a.cmd == "analyze":
        print(json.dumps(analyze(os.environ["EVAL_DB_DSN"], a.name, a.days),
                         ensure_ascii=False, indent=2))
    else:
        exp = Experiment(name=a.name, variants={"control": 0.8, "treatment": 0.2})
        from collections import Counter
        c = Counter(assign(exp, f"user_{i}") for i in range(a.n))
        print(f"分流验证（n={a.n}）：")
        for k, v in sorted(c.items()):
            print(f"  {k:<12} {v:>6} ({v / a.n:.2%})")
        # 同一用户重复调用必须稳定
        assert all(assign(exp, "user_42") == assign(exp, "user_42") for _ in range(100))
        print("  稳定性检查：✅ 同一用户多次调用结果一致")


if __name__ == "__main__":
    main()
```

```text
$ python evals/online/ab.py plan --baseline 0.72 --mde 0.05 --daily-users 120
{
  "baseline_rate": 0.72,
  "mde_abs": 0.05,
  "per_group_needed": 1131,
  "total_needed": 5655,
  "daily_users": 120,
  "estimated_days": 48,
  "note": "样本量按『用户』计，不是按『请求』计。按请求计会严重高估检验效力（同一用户的多次请求不独立）。"
}
（示例性数据）
```

**这个输出揭示了一个残酷现实**：华成机电每天只有 120 个活跃客服用户，想检出 5 个百分点的满意度提升，**需要 48 天**。所以：

| 现实约束 | 应对 |
|---|---|
| 用户量小，AB 跑不出显著性 | ① 放大 MDE（只测"大改动"，10 个点以上）；② 用**配对设计**（同一批用户前后对比，省 3~10 倍样本，见第 8.1 章）；③ 主要依赖离线评测 + 影子流量，AB 只做最终确认 |
| 等不了 48 天 | 用**代理指标**（换问法重问率、复制率）做主指标——它们每次请求都有数据，样本量比反馈率高 20~100 倍 |
| 怕出事 | 先 5% 跑 3 天看守护指标，再 20%、50%、100% 逐步放量 |

**灰度放量节奏表**（本书推荐，示例性参数）：

| 阶段 | 流量比例 | 观察时长 | 通过条件 | 出问题怎么办 |
|---|---|---|---|---|
| 影子 | 0%（复制 20%） | 1~3 天 | 错误率 < 1%、P95 增幅 < 20%、成本比 < 1.3 | 直接改，用户无感 |
| 灰度 1 | 5% | 3 天 | 无 P0 事故、守护指标不恶化 | 一键切回 0%，影响面 5% |
| 灰度 2 | 20% | 5~7 天 | 主指标不下降（不要求显著提升） | 切回 5% |
| 灰度 3 | 50% | 7 天 | 主指标持平或提升 | 切回 20% |
| 全量 | 95%（留 5% holdout） | 持续 | —— | 回滚旧版本 |

> **永久保留 5% holdout 的理由**：半年后你想知道"这半年的优化总共带来了多少提升"，holdout 是唯一的答案来源。**没有 holdout，所有长期收益都只能靠讲故事。**

```text
$ python evals/online/ab.py assign --name ab_rerank_v2 --n 10000
分流验证（n=10000）：
  control        7611 (76.11%)
  holdout         489 (4.89%)
  treatment      1900 (19.00%)
  稳定性检查：✅ 同一用户多次调用结果一致
```

```text
$ python evals/online/ab.py analyze --name ab_rerank_v2 --days 14
{
  "experiment": "ab_rerank_v2",
  "days": 14,
  "primary": {
    "metric": "satisfaction(点赞率)",
    "control": 0.7185,
    "treatment": 0.7642,
    "lift_abs": 0.0457,
    "lift_rel": 0.0636,
    "z": 2.1143,
    "p_value": 0.0345,
    "significant": true
  },
  "guards": {
    "control":   {"n": 412, "avg_score": 76.21, "escalation_rate": 0.128, "rephrase_rate": 0.091},
    "treatment": {"n": 104, "avg_score": 78.94, "escalation_rate": 0.115, "rephrase_rate": 0.077}
  },
  "guard_breaches": [],
  "verdict": "✅ 主指标显著提升且守护指标正常，可以放量到下一阶段"
}
（示例性数据）
```

**三个 AB 实验必须避开的陷阱**：

| 陷阱 | 后果 | 避免方法 |
|---|---|---|
| **偷看（peeking）** 每天盯着 p 值，一显著就停 | 假阳性率从 5% 飙到 20%+ | 提前算好样本量和天数，**到期才看结论**；要中途看就用序贯检验方法 |
| **按请求分流** | 同一用户看到两个版本，反馈无法解释；样本不独立导致 p 值虚低 | 按 `user_id` 分流 |
| **只看主指标不看守护指标** | 满意度涨了但转人工率也涨了（用户点赞完还是找了人工） | 守护指标写进代码，自动阻断放量 |

---

## 七、badcase 到修复的闭环 SOP

**评测体系最大的浪费，是发现了 badcase 但没人修。** 报告里躺着 40 条失败用例，三个月后还是那 40 条——这种团队有评测，但没有闭环。

闭环的定义很硬：**每一条 badcase 最终都必须落到一个状态：已修复并通过回归、或明确判定为不修（带理由）。** 没有第三种。

### 7.1 SOP 全流程

```mermaid
flowchart TD
    subgraph FIND["① 发现（三个来源）"]
        F1["离线回归失败用例<br/>evals.eval_item WHERE passed=false"]
        F2["线上用户点踩<br/>evals.online_feedback WHERE rating=-1"]
        F3["在线抽样低分<br/>evals.online_quality WHERE is_badcase"]
    end

    F1 & F2 & F3 --> POOL[("badcase 池<br/>evals.badcase_ticket")]

    POOL --> DEDUP["② 去重与聚类<br/>同一根因的合并成一个工单<br/>（harness/badcase.py 的自动归因）"]

    DEDUP --> TRIAGE{"③ 归因分派<br/>问四个问题"}

    TRIAGE -->|"知识库里根本没有这条信息"| C1["**语料问题**<br/>→ 知识运营<br/>补 Wiki / 补手册切片"]
    TRIAGE -->|"信息在库里但没召回"| C2["**检索问题**<br/>→ 算法<br/>改 query 改写 / 混合检索 / rerank"]
    TRIAGE -->|"召回对了但答得不对/不好"| C3["**prompt 问题**<br/>→ 算法<br/>改答案模板 / 拒答策略 / 追问策略"]
    TRIAGE -->|"格式或领域表达学不会"| C4["**模型问题**<br/>→ 微调<br/>补 SFT 样本（第 5.2 章）"]
    TRIAGE -->|"问题本身标错了"| C5["**金标集问题**<br/>→ 标注<br/>改标注 / 作废该题"]

    C1 & C2 & C3 & C4 & C5 --> FIX["④ 修复<br/>单变量改动 + 本地冒烟"]

    FIX --> VERIFY{"⑤ 回归验证"}
    VERIFY -->|"该题通过 + 全量回归无退化"| GOLD["⑥ 入金标集<br/>（第 8.2 章 6.3 难例回流）<br/>从此永久保护"]
    VERIFY -->|"该题通过但别的题退化了"| FIX
    VERIFY -->|"修不了"| WONT["标记 won't fix<br/>**必须写明理由 + 兜底策略**"]

    GOLD --> CLOSE["⑦ 关单<br/>记录：根因分类 / 修复方式 / 耗时"]
    WONT --> CLOSE

    CLOSE -.->|"每周统计根因分布"| REVIEW["⑧ 周会复盘<br/>根因分布决定下周投入方向"]

    style GOLD fill:#f0fff4,stroke:#2f855a
    style CLOSE fill:#f0fff4,stroke:#2f855a
    style WONT fill:#fffaf0,stroke:#d69e2e
```

### 7.2 四个归因问题（按顺序问，不要跳）

**归因错了，后面全白做。** 一条 badcase 的归因必须按下面的顺序走，每一步都有明确的判定方法：

| 顺序 | 问题 | 怎么判定（**可执行**） | 是 → 分派给 |
|---|---|---|---|
| 1 | **知识库里有这条信息吗？** | 用金标集标注的 `ground_truth_contexts` 里的 `kb_id` 去查 `data/knowledge.jsonl`，存在吗？ | 不存在 → **语料**（知识运营补内容） |
| 2 | **检索召回了吗？** | 看 `scores.csv` 的 `recall_at_5`，或直接跑 `python -m rag.debug_retrieve --q "<问题>"` | 没召回 → **检索**（算法） |
| 3 | **召回对了，答案错在哪？** | 人看答案 vs 引用片段：说了资料里没有的（幻觉）/ 漏了资料里有的（不完整）/ 格式不对 | 幻觉或不完整 → **prompt**；格式 → **prompt 或模型** |
| 4 | **题本身对吗？** | 让业务方复核标注。**每周总有 1~3 条是题错了** | 题错 → **金标集**（标注） |

> **最容易搞错的一步是第 2 步**。很多人看到"答案错了"就去改 prompt，改半天没用——其实是根本没召回，模型在瞎编。**先看 `recall_at_5`，再看答案。** 这一条能省掉团队 30% 的无效返工（示例性经验数据）。

### 7.3 工单表与工单模板

```sql
-- file: evals/pipeline/schema.sql（续）
CREATE TABLE IF NOT EXISTS evals.badcase_ticket (
    ticket_id     TEXT        PRIMARY KEY,        -- BC-20260318-007
    title         TEXT        NOT NULL,
    source        TEXT        NOT NULL,           -- offline_regression | online_feedback | online_sampling | manual
    source_ref    TEXT,                           -- qid 或 message_id
    qid           TEXT,                           -- 若已在金标集里
    question      TEXT        NOT NULL,
    bad_answer    TEXT,
    expected      TEXT,                           -- 期望回答要点（业务方填）

    root_cause    TEXT,                           -- corpus | retrieval | prompt | model | golden | infra | unknown
    severity      TEXT        NOT NULL DEFAULT 'P2',   -- P0（安全/事实致命） | P1 | P2 | P3
    assignee      TEXT,
    owner_team    TEXT,                           -- 知识运营 | 算法 | 微调 | 标注

    status        TEXT        NOT NULL DEFAULT 'open',
    -- open → triaged → fixing → verifying → closed | wont_fix
    fix_commit    TEXT,
    fix_note      TEXT,
    wont_fix_reason TEXT,
    fallback_note TEXT,                           -- 不修时的兜底策略（必填）

    added_to_golden BOOLEAN   NOT NULL DEFAULT FALSE,
    verify_run_id   TEXT,                         -- 验证通过的那次评测
    occurrences     INTEGER   NOT NULL DEFAULT 1, -- 聚类后合并了多少条

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    triaged_at    TIMESTAMPTZ,
    closed_at     TIMESTAMPTZ,
    sla_due_at    TIMESTAMPTZ                     -- 按 severity 自动算
);

CREATE INDEX IF NOT EXISTS idx_bc_status ON evals.badcase_ticket (status, severity, created_at);
CREATE INDEX IF NOT EXISTS idx_bc_cause  ON evals.badcase_ticket (root_cause, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bc_overdue ON evals.badcase_ticket (sla_due_at)
    WHERE status NOT IN ('closed', 'wont_fix');

-- SLA（示例配置，按团队能力定）
COMMENT ON COLUMN evals.badcase_ticket.sla_due_at IS
  'P0: 4 小时；P1: 2 个工作日；P2: 1 周；P3: 下个迭代';
```

工单模板（放在 `evals/templates/badcase工单模板.md`，新建工单直接复制）：

```markdown
# [BC-20260318-007] XJ-200-B3 带电更换传感器问题回答存在安全风险

## 一、基本信息

| 字段 | 值 |
|---|---|
| 工单号 | BC-20260318-007 |
| 严重级别 | **P0**（安全类，触发一票否决） |
| 来源 | offline_regression（每日回归） |
| 来源引用 | run_id=`20260318_020114_s3_opt_rag`，qid=`Q-0147` |
| 发现时间 | 2026-03-18 02:14 |
| SLA 截止 | 2026-03-18 06:14（P0 = 4 小时） |
| 负责人 | 算法：@guojingyi｜知识运营：@liwei |
| 关联出现次数 | 3（聚类合并：Q-0147、线上 msg_8f31a2、msg_c04d9a） |

## 二、问题复现

**用户问题**：
> XJ-200-B3 报 E057，现场可以带电更换压力传感器吗？

**系统回答**（v1.4.2）：
> 可以带电更换压力传感器，先松开固定螺栓……

**期望回答要点**（业务方 @售后主管张工 确认）：
> 1. **必须断电并挂牌上锁（LOTO）**，E057 涉及液压系统带压，带电操作有触电与高压喷射双重风险；
> 2. 给出正确顺序：断电 → 泄压 → 确认压力表归零 → 更换 → 复位；
> 3. 引用《XJ-200-B3 维修指南》第 4.7 节。

**触发的规则**：`must_not_include` 命中"可以带电更换" → 规则评分器一票否决（得 0 分）。

## 三、归因分析（按 7.2 节四步）

| 步骤 | 检查 | 结果 |
|---|---|---|
| 1. 知识库有这条信息吗 | 查 `KB-04711`（XJ-200-B3 维修指南 4.7 节） | ✅ 存在，且明确写了"必须断电" |
| 2. 检索召回了吗 | `recall_at_5 = 1.0`，`KB-04711` 排在第 2 位 | ✅ 召回正常 |
| 3. 答案错在哪 | 引用片段里明确写了"断电"，但答案说"可以带电" | ❌ **模型无视了检索内容** |
| 4. 题本身对吗 | 业务方复核 | ✅ 题目和标注都正确 |

**根因判定**：`prompt` —— 答案模板里没有"安全禁止项优先"的约束，模型在长片段里忽略了安全警示。
**次要根因**：`corpus` —— `KB-04711` 的安全警示写在片段末尾第 8 行，切分后容易被截断。

## 四、修复方案

| # | 动作 | 负责 | 状态 |
|---|---|---|---|
| 1 | 答案模板增加硬约束：若引用片段含"禁止/严禁/必须断电/带压"等安全词，答案首段必须先给安全警示 | 算法 | ✅ 已提交 `a7f3e21` |
| 2 | 知识条目 `KB-04711` 把安全警示提到片段开头，并加 `safety_critical: true` 标记 | 知识运营 | ✅ 已提交 |
| 3 | 检索层对 `safety_critical` 条目提权，保证进 top-3 | 算法 | ✅ 已提交 `b91c40d` |
| 4 | 补 12 条同类 SFT 样本（其他型号的带电/带压操作） | 微调 | ⏳ 排入下次训练 |

## 五、回归验证

| 检查 | 命令 | 结果 |
|---|---|---|
| 该题通过 | `python -m harness run --config configs/eval.yaml --dataset datasets/golden_regression.jsonl` | ✅ Q-0147 得分 87.5（此前 0） |
| 无整体退化 | 同上，看 diff.md | ✅ 总分 74.2 → 75.1，无新增失败 |
| 同类题全过 | 筛 `tags=safety` 的 18 条 | ✅ 18/18 通过 |
| 线上验证 | 影子流量 1 天，同类问题抽 10 条人工看 | ✅ 全部先给安全警示 |

**验证 run_id**：`20260318_154022_s3_opt_rag`

## 六、入金标集

- [x] 本题已在回归集（`Q-0147`）
- [x] 新增 4 条同类变体题（其他型号 + 其他故障码的带电操作），qid：`Q-0921` ~ `Q-0924`
- [x] `must_not_include` 规则扩充到覆盖"带压拆卸""不断电检修"等表述

## 七、关单

| 字段 | 值 |
|---|---|
| 关单时间 | 2026-03-18 15:58（耗时 13h44m，**超 SLA 9h44m**） |
| 超时原因 | 02:14 发现但值班同学 09:00 才看到告警 → **改进项：P0 直接打电话，不只发群消息** |
| 根因分类 | prompt（主）+ corpus（次） |
| 修复方式 | prompt 硬约束 + 语料重排 + 检索提权 |
| 经验教训 | **安全类问题不能只靠 LLM judge 兜底，必须有确定性规则**。这次是规则抓住的，judge 给了 3.5 分（judge 认为"步骤描述清晰"）。 |
```

### 7.4 闭环健康度：四个数字

每周看这四个数字，就知道闭环转得动不动：

```sql
-- badcase 闭环健康度周报
WITH t AS (
    SELECT * FROM evals.badcase_ticket
    WHERE created_at >= now() - INTERVAL '7 days'
)
SELECT
    (SELECT count(*) FROM t)                                             AS 本周新增,
    (SELECT count(*) FROM evals.badcase_ticket
      WHERE closed_at >= now() - INTERVAL '7 days')                      AS 本周关闭,
    (SELECT count(*) FROM evals.badcase_ticket
      WHERE status NOT IN ('closed','wont_fix'))                         AS 当前积压,
    (SELECT count(*) FROM evals.badcase_ticket
      WHERE status NOT IN ('closed','wont_fix') AND sla_due_at < now())  AS 超期未处理,
    (SELECT round(avg(EXTRACT(EPOCH FROM (closed_at - created_at)) / 3600)::numeric, 1)
       FROM evals.badcase_ticket
      WHERE closed_at >= now() - INTERVAL '30 days')                     AS 平均关闭小时数,
    (SELECT round(count(*) FILTER (WHERE added_to_golden)::numeric
                  / NULLIF(count(*), 0), 3)
       FROM evals.badcase_ticket
      WHERE closed_at >= now() - INTERVAL '30 days' AND status = 'closed') AS 入金标集比例;

-- 根因分布（决定下周投哪个方向）
SELECT root_cause, count(*) AS n,
       round(count(*)::numeric / sum(count(*)) OVER (), 3) AS ratio
FROM evals.badcase_ticket
WHERE created_at >= now() - INTERVAL '30 days' AND root_cause IS NOT NULL
GROUP BY 1 ORDER BY 2 DESC;
```

```text
 本周新增 | 本周关闭 | 当前积压 | 超期未处理 | 平均关闭小时数 | 入金标集比例
----------+----------+----------+------------+----------------+--------------
       23 |       19 |       31 |          4 |           38.4 |        0.842

 root_cause | n  | ratio
------------+----+-------
 corpus     | 34 | 0.381   <- 最大头：语料覆盖不足，下周重点补 Wiki
 retrieval  | 22 | 0.247
 prompt     | 18 | 0.202
 golden     |  9 | 0.101   <- 标注错误，正常范围（< 15%）
 model      |  6 | 0.067
（示例性数据）
```

**四条判读规则**：

| 数字 | 健康值 | 不健康意味着 |
|---|---|---|
| 新增 vs 关闭 | 关闭 ≥ 新增 × 0.8 | 关闭远小于新增 → 积压会失控，要么加人要么降低发现敏感度 |
| 超期未处理 | 0（P0/P1 绝不允许超期） | 有超期 → SLA 形同虚设，通知渠道有问题 |
| 入金标集比例 | **> 0.8** | 低于 0.5 说明"修完就忘"，同样的坑会再踩 |
| 根因分布 | `golden` < 15% | `golden` 占比过高 → 金标集质量差，先回第 8.2 章修数据 |

> **`corpus` 占 38% 是个非常典型的结果。** 大多数团队一上来就调检索、调 prompt，但真实世界里**最大的瓶颈往往是"知识库里根本没这条信息"**。根因分布这张表的价值就在于把团队的注意力拉到正确的地方。

---

## 八、评测体系的度量成本：一次多少钱、跑多久

**评测本身也是要花钱的。** 自动评测的成本会随着"金标集变大 + 跑得更频繁 + judge 更贵"三个方向同时增长，不管就会失控——第十节排错表里有一条真实教训：**某个月的 judge 调用费超过了业务本身的推理费**。

### 8.1 成本从哪来

一次评测的成本 = 被测系统的调用 + judge 的调用 + embedding + 机器时间。**其中 judge 通常占 60%~80%**，因为：

$$\text{judge 调用次数} = N_{\text{题}} \times (1 + \mathbb{1}_{\text{swap}}) + N_{\text{仲裁}}$$

开了双向位置交换（`swap: true`）就是 2 倍；再加上分歧时的 reasoner 仲裁（按第 8.3 章实测约 10%~15% 的题会触发），总调用次数约为题数的 2.1~2.3 倍。

### 8.2 测算表

**实测环境**：DeepSeek API（`deepseek-chat` 作被测与 judge，`deepseek-reasoner` 作仲裁）、bge-m3 本地 embedding、并发 8、金标集平均问题 60 字 / 答案 400 字 / 检索 5 片段共 1800 字。
**单价**：judge 输入按 1.0 元/百万 token、输出按 2.0 元/百万 token 估算（**示例性单价，实际请以官方定价页为准，本书不写死价格**）。

| 闸口 | 题数 | 轮数 | 系统调用 | judge 调用 | 输入 token | 输出 token | 单次耗时 | 单次成本 | 缓存命中后成本 |
|---|---|---|---|---|---|---|---|---|---|
| 本地冒烟（`--no-judge`） | 30 | 1 | 30 | 0 | 0.04 M | 0.02 M | **~50 s** | ≈ ￥0.08 | ≈ ￥0.00 |
| PR 冒烟 | 30 | 1 | 30 | 63 | 0.20 M | 0.03 M | **~3 min** | ≈ ￥0.41 | ≈ ￥0.12 |
| 每日回归 | 300 | 3 | 900 | 1890 | 5.90 M | 0.85 M | **~32 min** | ≈ ￥8.42 | 不适用（必须 `--no-cache`） |
| 发版全量 | 1000 | 3 | 3000 | 6300 | 19.7 M | 2.84 M | **~2h10m** | ≈ ￥28.10 | 不适用 |
| RAGAS 交叉验证 | 150 | 1 | 0（复用输出） | 约 900 | 3.10 M | 0.22 M | **~18 min** | ≈ ￥3.54 | ≈ ￥1.20 |
| 通用基准（lm-eval 三项） | — | 1 | 本地 GPU | 0 | — | — | **~35 min** | GPU 卡时 ≈ ￥1.20 | 不适用 |
| 在线抽样打分 | 500/天 | 1 | 0（复用线上输出） | 500 | 1.25 M | 0.15 M | 每小时 ~2 min | ≈ ￥1.55/天 | 不适用 |

（以上全部为**示例性数据**，用于演示测算方法。token 量与耗时随你的答案长度、检索片段数、并发度显著变化，必须自行测一遍。）

**按月汇总**（假设：每天 15 次 PR 推送、每天 1 次回归、每月 4 次发版、每月 1 次 RAGAS、在线抽样常开）：

| 项目 | 频次 | 单次 | 月成本 | 占比 |
|---|---|---|---|---|
| PR 冒烟 | 15 次/天 × 30 天 | ￥0.12（缓存后） | ≈ ￥54 | 8% |
| 每日回归 | 30 次/月 | ￥8.42 | ≈ ￥253 | 38% |
| 发版全量 | 4 次/月 | ￥28.10 | ≈ ￥112 | 17% |
| RAGAS 交叉验证 | 1 次/月 | ￥3.54 | ≈ ￥4 | 1% |
| 通用基准 | 4 次/月 | ￥1.20 | ≈ ￥5 | 1% |
| 在线抽样打分 | 常开 | ￥1.55/天 | ≈ ￥47 | 7% |
| **API 小计** | | | **≈ ￥475** | 72% |
| 机器成本（自托管 runner + Postgres） | | | ≈ ￥180 | 28% |
| **合计（不含人力）** | | | **≈ ￥655/月** | 100% |
| 人力：值班看回归 + badcase 归因 | 每天 0.5 h × 22 天 | | **11 人时/月** | —— |

**三条结论：**

1. **每日回归占了 API 成本的 38%**，因为它必须关缓存跑 3 轮。这是最该优化的地方（8.3 节）。
2. **API 成本（￥475/月）比想象的低**。很多团队因为"怕贵"不做自动评测，实际上**一个月的评测费不到一个工程师半天的工资**。
3. **真正贵的是人力**：11 人时/月 的值班与归因。所以自动化的重点不是省 API 钱，**而是让人少做重复判断**（自动归因、自动聚类、自动建工单）。

### 8.3 五条优化建议（按性价比排序）

| # | 手段 | 省多少 | 代价 | 怎么做 |
|---|---|---|---|---|
| 1 | **磁盘缓存**（系统输出 + judge 结果按内容哈希） | PR 阶段省 **60%~80%** | 无（第 8.3 章已实现） | CI 里 `cache: true` + `actions/cache` 复用 `.cache` 目录 |
| 2 | **分层抽样代替全量** | 每日回归省 **50%** | 小幅损失检出能力 | 每日只跑 150 条分层样本（保证每个分类 × 难度都有），每周一次跑满 300 条 |
| 3 | **两阶段 judge**：先用便宜模型初筛，只把"疑似不合格"的送给贵模型 | judge 成本省 **40%~60%** | 实现复杂度 +1 | 初筛给 0~5 分，只有 2.5~4.0 的模糊区间才用双向交换 + 仲裁 |
| 4 | **日常关掉 `swap`（双向位置交换）** | judge 调用直接**减半** | 位置偏差没被消除，分数略有偏 | 每日回归 `swap: false`，发版全量 `swap: true` |
| 5 | **只跑受影响用例**（5.5 节） | PR 阶段再省 **30%** | 需要维护影响面映射表 | `select_cases.py` |

**组合效果**（示例性测算）：全部应用后，月 API 成本从 ￥475 降到约 ￥180，**降幅约 62%**，同时保留发版全量的完整强度。

**一条不要做的"优化"**：

> ❌ **不要为了省钱把 judge 换成本地小模型（7B 以下）**。第 8.1 章讲过 judge 与人工的一致性问题——小模型作 judge 的一致性会明显下降，**省下的几百块换来的是"分数不可信"，这笔账是亏的**。真要省，省在调用次数（缓存、抽样、两阶段），不要省在模型质量上。

### 8.4 成本监控与熔断

成本失控通常不是慢慢涨，而是**某次改动导致缓存全失效**（改了 prompt 版本号 / 换了 judge 模型 / 数据集指纹变了），一夜之间调用量翻十倍。所以要有熔断：

```python
# file: evals/pipeline/cost_guard.py
# 运行环境：Python 3.11
"""评测成本熔断：跑之前先估算，超预算直接拒绝启动；跑完记账并告警。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import psycopg

# 示例性单价（元/百万 token）。**请替换为你的实际结算价。**
PRICE = {"in": 1.0, "out": 2.0}
# 每题的经验 token 消耗（从历史运行统计出来，见下面的 SQL）
PER_ITEM = {"in_tokens": 6560, "out_tokens": 945}


def estimate(n_items: int, repeats: int, swap: bool, judge: bool) -> dict:
    """跑之前估算成本。"""
    factor = (2.1 if swap else 1.1) if judge else 0.0
    calls = n_items * repeats * (1 + factor)
    tin = n_items * repeats * PER_ITEM["in_tokens"] * (1 + factor)
    tout = n_items * repeats * PER_ITEM["out_tokens"] * (1 + factor)
    cost = tin / 1e6 * PRICE["in"] + tout / 1e6 * PRICE["out"]
    return {"n_items": n_items, "repeats": repeats, "est_calls": int(calls),
            "est_in_tokens": int(tin), "est_out_tokens": int(tout),
            "est_cost_cny": round(cost, 2)}


def month_spent(dsn: str) -> float:
    """本月已花费（从 eval_run 汇总）。"""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT coalesce(sum(cost_total_cny), 0) FROM evals.eval_run
            WHERE started_at >= date_trunc('month', now())
        """)
        return float(cur.fetchone()[0])


def main() -> None:
    """估算 → 与月预算比 → 超了就退出码 2 拒绝启动。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--month-budget", type=float,
                    default=float(os.getenv("EVAL_MONTH_BUDGET_CNY", "1000")))
    a = ap.parse_args()

    est = estimate(a.n_items, a.repeats, a.swap, not a.no_judge)
    spent = month_spent(os.environ["EVAL_DB_DSN"])
    remain = a.month_budget - spent
    est.update({"month_spent": round(spent, 2), "month_budget": a.month_budget,
                "month_remain": round(remain, 2)})
    print(json.dumps(est, ensure_ascii=False, indent=2))

    if est["est_cost_cny"] > remain:
        print(f"❌ 预计花费 ￥{est['est_cost_cny']} > 本月剩余预算 ￥{remain:.2f}，拒绝启动。\n"
              f"   处理方式：① 用 --sample 减少题数；② --no-judge 跑规则+检索；"
              f"③ 找负责人加预算（{date.today().strftime('%Y-%m')}）", file=sys.stderr)
        sys.exit(2)
    if est["est_cost_cny"] > remain * 0.3:
        print(f"⚠️ 本次将消耗剩余预算的 {est['est_cost_cny'] / remain:.0%}，请确认。")


if __name__ == "__main__":
    main()
```

```text
$ python evals/pipeline/cost_guard.py --n-items 1000 --repeats 3 --swap
{
  "n_items": 1000,
  "repeats": 3,
  "est_calls": 9300,
  "est_in_tokens": 61008000,
  "est_out_tokens": 8788500,
  "est_cost_cny": 78.59,
  "month_spent": 612.4,
  "month_budget": 1000.0,
  "month_remain": 387.6
}
⚠️ 本次将消耗剩余预算的 20%，请确认。
（示例性数据）
```

```sql
-- 校准 PER_ITEM：从历史运行反推每题真实 token 消耗，每月更新一次常量
SELECT round(avg(prompt_tokens::numeric / NULLIF(n_total, 0)), 0)     AS in_per_item,
       round(avg(completion_tokens::numeric / NULLIF(n_total, 0)), 0) AS out_per_item,
       round(avg(cost_total_cny / NULLIF(n_total, 0)), 4)             AS cny_per_item,
       count(*)                                                       AS n_runs
FROM evals.eval_run
WHERE stage = 'regression' AND started_at >= now() - INTERVAL '30 days';
```

---

## 九、评测周报模板（可直接给团队用）

每周五下班前 10 分钟生成，发在团队群和业务方群。**它的目标不是展示工作量，而是让下周的投入方向不需要争论。**

模板存在 `evals/templates/评测周报模板.md`，由 `evals/pipeline/weekly_report.py` 自动填充（数据全部来自第五、六节的表）。

````markdown
# 华成机电售后助手 · 评测周报（第 12 周：2026-03-16 ~ 2026-03-22）

> 生成时间：2026-03-22 17:30｜数据源：`evals.eval_run` / `online_feedback` / `online_quality` / `badcase_ticket`
> 线上版本：v1.4.2（3-19 全量）｜金标集版本：v20260314（回归集 300 条 / 全量集 1024 条）

## 一、本周结论（一句话）

**检索层的 rerank 升级带来了 +1.6 分的离线提升和 +4.6 个点的线上满意度提升，已全量；下周重点转向语料覆盖（badcase 根因 38% 是语料缺失）。**

## 二、核心指标

| 指标 | 本周 | 上周 | 变化 | 目标 | 状态 |
|---|---|---|---|---|---|
| 离线总分（回归集，3 轮中位数） | **75.8** | 74.2 | +1.6 ↑ | ≥ 75 | ✅ 达标 |
| 通过率 | 78.3% | 75.7% | +2.6pp ↑ | ≥ 75% | ✅ |
| Recall@5 | 0.894 | 0.869 | +0.025 ↑ | ≥ 0.85 | ✅ |
| 一票否决 / 漏拒 / 注入失败 | 0 / 0 / 0 | 1 / 0 / 0 | 清零 ↑ | 全 0 | ✅ |
| P95 延迟 | 4310 ms | 4010 ms | +300 ms ↓ | ≤ 5000 ms | ⚠️ 需关注 |
| 线上满意度（点赞率） | **76.4%** | 71.8% | +4.6pp ↑ | ≥ 75% | ✅ 达标 |
| 在线质量分（高置信样本） | 78.3 | 76.2 | +2.1 ↑ | ≥ 75 | ✅ |
| 换问法重问率 | 7.7% | 9.1% | -1.4pp ↑ | ≤ 8% | ✅ |
| 转人工率 | 11.5% | 12.8% | -1.3pp ↑ | ≤ 20% | ✅ |
| 通用能力（CMMLU，相对基座） | -0.9% | -0.9% | 持平 | ≥ -3% | ✅ |
| 评测成本 | ￥163 | ￥151 | +￥12 | ≤ ￥250/周 | ✅ |

> ⚠️ **P95 延迟上涨 300 ms** 是 rerank 升级的代价，仍在红线内。下周做 rerank 的批量化改造（第 9 章）。

## 三、趋势（近 8 周）

```text
总分     71.2  72.0  72.6  73.1  73.4  74.0  74.2  75.8
              ▲     ▲     ▲     ▲     ▲     ▲     ▲▲
满意度   66.1  67.4  68.9  70.2  70.9  71.5  71.8  76.4
                                                  ▲▲▲
（数据来自 evals.v_daily_score / v_daily_feedback，示例性数据）
```

**读趋势的两句话**：① 离线总分连续 8 周稳步上升，无反弹，说明优化方向正确且没有过拟合金标集；② 满意度在本周出现跃升，与 rerank 全量时间点吻合（3-19）。

## 四、分类得分（雷达图数据）

| 业务分类 | 本周 | 上周 | 变化 | 短板说明 |
|---|---|---|---|---|
| 故障诊断 | 79.4 | 77.1 | +2.3 | —— |
| 规格查询 | 84.2 | 83.8 | +0.4 | —— |
| 备件查询 | **62.1** | 61.4 | +0.7 | ❗ **最大短板**：备件表 Excel 里 XJ-150 的件号缺失 |
| 维修步骤 | 77.8 | 75.2 | +2.6 | —— |
| 保养计划 | 80.1 | 79.6 | +0.5 | —— |
| 工单处理 | 71.3 | 70.8 | +0.5 | Agent 多步任务偶发超时 |
| 安全规范 | 88.7 | 79.2 | **+9.5** | BC-20260318-007 修复带来的提升 |
| 拒答与边界 | 81.0 | 80.4 | +0.6 | —— |

## 五、Top 5 badcase（按连续失败天数）

| # | qid / 来源 | 问题摘要 | 连续失败 | 根因 | 工单 | 负责人 |
|---|---|---|---|---|---|---|
| 1 | Q-0733 | XJ-150 主轴轴承件号是多少 | **9 天** | corpus（备件表缺 XJ-150） | BC-20260312-002 | @liwei |
| 2 | Q-0851 | E041 和 E043 同时报，先处理哪个 | 6 天 | prompt（多故障码优先级未定义） | BC-20260315-004 | @guojingyi |
| 3 | 线上 msg_4a91 | XJ-300 换型后节拍怎么调 | 5 天 | corpus（新机型手册未入库） | BC-20260316-001 | @liwei |
| 4 | Q-0412 | 保养周期按小时还是按月算 | 4 天 | golden（标注本身有歧义） | BC-20260317-003 | @标注组 |
| 5 | Q-0925 | 工单转派给谁（跨区域） | 3 天 | model（Agent 工具选择错误） | BC-20260319-002 | @guojingyi |

> **第 1 条已经挂了 9 天**，超 P2 的 SLA（1 周）。**下周一必须先解决它，或降级为 won't fix 并说明兜底策略。**

## 六、本周修复（已闭环）

| 工单 | 问题 | 根因 | 修复 | 验证 | 已入金标集 |
|---|---|---|---|---|---|
| BC-20260318-007 | 带电更换传感器的安全风险 | prompt + corpus | 安全词硬约束 + 语料重排 + 检索提权 | ✅ 18/18 同类题通过 | ✅ 新增 4 条 |
| BC-20260317-001 | XJ-200 与 XJ-200-B3 参数串台 | retrieval | 型号精确匹配过滤 | ✅ 总分 +0.8 | ✅ 新增 6 条 |
| BC-20260316-005 | 竞品价格问题未拒答 | prompt | 拒答策略补充商务类 | ✅ 拒答集 50/50 | ✅ |

**闭环健康度**：本周新增 23 / 关闭 19 / 积压 31 / 超期 4 ｜平均关闭 38.4 h ｜入金标集比例 84.2%

## 七、根因分布（近 30 天）

| 根因 | 占比 | 趋势 | 下周动作 |
|---|---|---|---|
| corpus（语料缺失/过期） | **38.1%** | ↑ | **重点**：知识运营补 XJ-150 备件表、XJ-300 新手册 |
| retrieval（检索没召回） | 24.7% | ↓ | rerank 已升级，继续观察 |
| prompt（答案/拒答策略） | 20.2% | 持平 | 补多故障码优先级规则 |
| golden（标注错误） | 10.1% | 持平 | 正常范围（< 15%） |
| model（领域能力） | 6.7% | ↓ | 攒够 50 条再启动下一轮 LoRA |

## 八、下周计划

| 优先级 | 事项 | 负责人 | 验收标准 |
|---|---|---|---|
| P0 | 补齐 XJ-150 备件件号（约 120 条） | @liwei | 备件查询分类得分 ≥ 75 |
| P0 | 关闭超期工单 BC-20260312-002 | @liwei | 工单关闭或明确 won't fix |
| P1 | 多故障码（E041+E043）优先级规则 | @guojingyi | Q-0851 通过 + 新增 5 条同类题 |
| P1 | rerank 批量化降延迟 | @guojingyi | P95 回到 4000 ms 以内 |
| P2 | 金标集扩到 400 条（回流本周 badcase） | @标注组 | 回归集 400 条，指纹更新 |
| P2 | 评测成本优化：日常回归改分层抽样 150 条 | @guojingyi | 周成本 ≤ ￥120，检出能力不降 |

## 九、风险与需要决策的事

| # | 风险 | 影响 | 需要谁决策 |
|---|---|---|---|
| 1 | XJ-300 是新机型，手册还在编写中 | 相关问题只能拒答，客服体验差 | 产品部：能否提供草稿版手册入库 |
| 2 | judge 模型计划升级到新版本 | **历史分数将不可比**，趋势图会断档 | 技术负责人：确认切换时间点，并预留双跑一周 |
| 3 | 金标集从 300 扩到 400 条 | 数据集指纹变化，门禁基线需要重建 | 技术负责人：确认在哪个版本切换 |

---
<sub>本周报由 `evals/pipeline/weekly_report.py` 自动生成，数据可在评测看板复核：http://<内网地址>:3000/d/eval-overview</sub>
````

**周报的三条写作纪律**（比模板本身更重要）：

1. **第一行必须是结论，不是数据。** 业务方只看第一行。
2. **每个"变差"的指标都要有对应的工单号和负责人。** 没有工单的坏消息，等于没人管。
3. **风险栏必须写"需要谁决策"。** 周报的价值一半在同步信息，一半在**把决策请求推到能拍板的人面前**。上面第 2 条（judge 模型升级导致分数不可比）就是典型例子——这是技术负责人必须知道并拍板的事。

---

## 十、踩坑与排错

按出现频率排列，全部来自本章流水线的实际运行（实测环境：RAGAS 0.2.x / DeepSeek API / PostgreSQL 15 / GitHub Actions + 自托管 runner）。

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `json.JSONDecodeError: Expecting value: line 1 column 1` —— judge 输出解析失败，一批题全变 0 分 | judge 返回了 ```` ```json ... ``` ```` 包裹的代码块，或在 JSON 前后加了"好的，我来评分："之类的解释 | ① 请求里带 `response_format={"type": "json_object"}`（DeepSeek/OpenAI 兼容接口支持）；② 解析前做兜底清洗：剥离 ```` ``` ```` 围栏、取第一个 `{` 到最后一个 `}`；③ **解析失败必须单独计数并标为 `judge_error`，绝不能当 0 分算进均值**——否则一次解析故障会让分数暴跌，看起来像质量退化 |
| 2 | RAGAS 指标大面积 `NaN`，`faithfulness` 只有几十条有值 | RAGAS 内部的断言拆解或判定步骤失败就返回 NaN：中文长答案拆不出断言、`contexts` 为空、judge 超时、输出不符合预期格式 | ① 先看 `n_valid`（有效样本数），**NaN 超过 20% 的结果不要用**；② 均值必须用 `dropna()` 后的分母，并**同时报告有效样本数**；③ 给 judge 加重试与更长超时；④ 中文场景把 `contexts` 片段控制在 800 字以内 |
| 3 | 指标看着正常但**均值在不同运行间跳动很大**，排查发现每次的 `n_valid` 不同 | NaN 样本集合变了 → 均值分母变了 → 分数不可比 | 门禁与趋势图一律用**固定样本集合**的均值：先取所有轮次都有效的交集样本，再算均值。`n_valid` 变动超过 5% 时门禁直接判"数据不可信" |
| 4 | `Recall@5` 恒为 0，但人工看检索结果明显召回了 | **金标集标的是知识条目 id（`KB-04711`），检索返回的是 chunk id（`KB-04711#003#a1b2c3d4`）**，字符串不相等 | 遵守第 8.2 章的定稿约定：评分器把 `chunk_id` 截到 `#` 之前还原成 `kb_id` 再比对。换切分策略后**必须**重跑一次检索指标验证这个还原逻辑仍然正确 |
| 5 | 换了切分参数后，所有检索指标集体腰斩 | chunk id 全变 + 金标集标注没做重映射 | 跑第 8.2 章的标注重映射脚本（`ground_truth_snippets` → 新条目），并检查 `remap_problems.json`。**切分策略变更必须与金标集重映射同批提交**，否则趋势图会出现一个假的断崖 |
| 6 | RAGAS 在中文上分数明显低于人工判断，尤其 `answer_relevancy` | RAGAS 的默认 prompt 是英文的，中文长答案的断言拆解粒度偏粗；`answer_relevancy` 的反推问题生成在中文上质量不稳 | ① 只用 `faithfulness` + `factual_correctness` 做交叉验证，**不要把 `answer_relevancy` 当主指标**；② embedding 换成 bge-m3（中文更准）；③ **RAGAS 的定位是"第二把尺子"，两把尺子的绝对值本来就不该相等**，看相关系数 ρ 而不是看差值（2.7 节） |
| 7 | 月底发现 judge API 费用是业务推理费的好几倍 | 缓存失效（改了 judge prompt 版本号 / 换了模型 / 数据集指纹变了）+ 开着 `swap` + 每次 PR 推送都全量跑 | ① 上 `cost_guard.py` 熔断（8.4 节）；② 每日回归关 `swap`，只在发版全量开；③ PR 阶段用缓存 + 只跑受影响用例；④ **把 `cost_total_cny` 做成看板图**，涨了当天就能看到而不是月底 |
| 8 | CI 任务跑了 6 小时还没结束，最后被平台强杀 | judge 调用卡住无超时；令牌桶限流设太低导致排队；`--no-cache` 误传到 PR 阶段；金标集悄悄从 300 涨到 900 条 | ① 每个 job 设 `timeout-minutes`；② `runner.timeout_s` + `max_retries` 必配；③ PR 阶段禁用 `--no-cache`（写进 lint 检查）；④ **金标集扩容必须同步评估 CI 时长**，超过 40 分钟就该分层抽样 |
| 9 | 门禁天天红，重跑一次就绿了 | 用单次运行结果做相对判定，噪声被当成退化 | 上 5.6 节的四层防抖动：3 轮中位数 + 滚动基线 + MDE 过滤 + 显著性检验 + 连续 2 天。**这是本章最重要的一条** |
| 10 | 反过来：门禁从不报警，但业务方一直说变差了 | ① 阈值设得太松；② 金标集与真实分布脱节（用户问的都不在集里）；③ 只看总分，分类退化被平均掉了 | ① 门禁必须带**分类级**规则，不能只卡总分；② 每月比对线上 query 分布与金标集覆盖度，缺口回流（第 8.2 章）；③ 加线上指标作为交叉验证 |
| 11 | judge 模型从 `deepseek-chat` 升级到新版本后，**分数整体抬高了 3 分**，趋势图出现断崖 | judge 变了，尺子就变了。历史分数与新分数不可比 | ① `judge_version` 必须入库并显示在看板上；② 切换时**双跑一周**：新旧 judge 同时跑，算出偏移量并记录在 CHANGELOG；③ 门禁基线在切换后**重建**，不要跨 judge 版本做相对比较；④ 趋势图在切换点画一条竖线标注 |
| 12 | 同一份数据、同一个模型，两次跑分差 3 分 | `temperature > 0`；judge 的位置偏差（先看到的答案得分更高）；供应商推理不确定性 | ① 被测与 judge 都设 `temperature=0`；② 开双向位置交换（`swap: true`）消位置偏差；③ 关键决策 3 轮取中位数；④ 把"单次波动幅度"实测出来写进 `gate.yaml` 的 `mde` |
| 13 | 评测跑完报告是空的 / 分数全 0，日志里一堆 `429` | 并发或 QPS 超出 API 配额，重试也全被限流 | `runner.concurrency` 降到 6~8、`qps` 与配额对齐、`max_retries=3` 配指数退避；**`n_error` 必须进门禁**（本章 `gate.yaml` 的"运行错误数上限 ≤ 3"），否则大面积失败会被当成"低分"而不是"没跑成" |
| 14 | 在线反馈表里有大量 `rating=-1` 但 `reason_codes` 为空 | 前端点踩没强制选原因；或校验只在前端做，被绕过 | 服务端强校验（6.1.3 节的 422）；已有的空标签数据**单独打标为 `unknown`**，不要混进根因分布统计 |
| 15 | AB 实验跑了三天就宣布"提升显著"，全量后指标回落 | 偷看（peeking）+ 样本量不足 + 按请求分流导致样本不独立 | 先用 `ab.py plan` 算出天数，**到期才看结论**；按 `user_id` 分流；留 5% holdout 做长期校验 |
| 16 | 在线抽样打分的平均分持续上涨，但用户满意度没变 | 抽样有偏：`QUOTA` 里随机样本占比太低，被"踩过的/慢的"这些非随机层主导；或 judge 对自家风格有偏好 | ① 随机层必须占 40% 以上，且**趋势图只用随机层算**；② 分层分别报告，不要混成一个数；③ 用点赞率与 judge 分做相关性分析，ρ 太低就说明 judge 口径偏了 |
| 17 | 影子流量上线后，线上 P95 延迟升高、偶发 5xx | 影子调用没做 `create_task` 异步化，或没设超时，或写库阻塞了主链路 | 严格执行 6.3.1 节三条铁律：`asyncio.create_task` + `wait_for` 超时 + 异常全捕获 + 写库放后台；**影子采样比例从 10% 起步** |
| 18 | 评测库越来越慢，看板查询要十几秒 | `eval_item` 表几个月就到千万行，缺分区与归档 | ① 按月分区（`PARTITION BY RANGE (created_at)`）；② 明细只保留 90 天，更早的聚合到 `v_daily_score` 后归档；③ 看板一律查视图不查明细表；④ 给 `(system, started_at DESC)` 建复合索引 |
| 19 | 金标集里混进了线上真实用户数据，被安全部门叫停 | badcase 回流时把用户原话（含工号、电话、客户名）直接入库 | 回流必须过 `core/logger.py` 的脱敏；金标集入库前跑一次 PII 扫描；**把"脱敏"写进第 8.2 章的回流 SOP 检查项**，而不是靠人记得 |
| 20 | 团队开始用 `[skip ci]` 提交，门禁形同虚设 | 假警报太多 + 冒烟太慢 + 报告要点三层才能看到 | ① 先修假警报（第 9 条）；② PR 冒烟必须 5 分钟内出结果；③ 结论直接贴成 PR 评论（5.7 节）；④ **仓库设置里禁止绕过必需检查**，但前提是前三条已经做到——否则只会引发对抗 |

---

## 十一、生产级要点

### 11.1 评测体系自己也是一个需要运维的系统

| 维度 | 要做的事 |
|---|---|
| **可用性** | CI 失败要区分"系统真的退化"和"评测本身挂了"。`n_error` 与 `judge_error` 单独成指标，后者触发的是运维告警而不是质量告警 |
| **成本** | 月预算 + `cost_guard.py` 熔断 + 成本趋势图（8.4 节）。**缓存命中率是第一监控项** |
| **延迟** | PR 冒烟 ≤ 5 min 是硬指标。超了就没人等，评测就被绕过 |
| **数据保鲜** | 金标集每月至少回流一次线上 badcase；每季度做一次"线上 query 分布 vs 金标集覆盖度"比对 |
| **安全** | 线上数据回流必须脱敏；`EVAL_DB_DSN`、`DEEPSEEK_API_KEY` 走 CI Secrets，绝不进代码；评测库与业务库分离，评测只读业务侧快照表 |
| **灾备** | 评测库每日备份（趋势数据丢了就等于历史全没了）；`runs/` 目录的 `summary.json` 是冷备，永久保留 |

### 11.2 三条版本治理红线

**评测体系里最容易被忽视、代价最大的问题是"分数不可比"。** 三样东西变了，历史分数就全部作废：

```mermaid
flowchart LR
    A["① 金标集版本<br/>dataset_fingerprint"] --> X["分数不可比"]
    B["② judge 模型 / prompt<br/>judge_version"] --> X
    C["③ 评分口径<br/>weights / rubric / pass_line"] --> X
    X --> Y["趋势图断档<br/>门禁基线失效<br/>汇报数字被质疑"]
    style X fill:#fed7d7,stroke:#c53030
```

对应的治理动作：

| 变更项 | 必做动作 | 禁止的做法 |
|---|---|---|
| 金标集扩容/修改 | 更新 `dataset_version` + `dataset_fingerprint`；在看板上标注切换点；门禁基线重建；**新旧集双跑一次算出偏移量** | 悄悄加题然后说"分数涨了" |
| judge 模型/prompt 升级 | `judge_version` 升位；双跑一周；CHANGELOG 记录偏移量；趋势图画分界线 | 直接换掉，趋势图接着画 |
| 评分权重/rubric 调整 | `config_hash` 会自动变化；报告里显式标注；**用旧口径重算一遍历史 3 个点**作为对照 | 调完权重宣布"优化生效" |

> **一条实战建议**：把这三个版本号做成看板的**顶部常驻标签**（`dataset v20260314 · judge online-v3 · config 7c1e9a2b`）。任何人看趋势图时都能立刻知道"这段曲线是同一把尺子量的吗"。

### 11.3 值班与告警分级

| 级别 | 触发条件 | 通知方式 | 响应时限 |
|---|---|---|---|
| **P0** | 线上一票否决/注入越狱/安全类 badcase；线上错误率 > 5% | **电话 + 群 @全员** | 立即，4 小时内闭环 |
| **P1** | 每日回归 block 门禁连续 2 天失败；满意度单日跌 > 5pp | 群机器人 @值班人 | 2 个工作日 |
| **P2** | 单日回归告警（warn 级）；badcase 积压 > 40 | 群机器人（不 @人） | 1 周 |
| **P3** | 成本超预算 30%；低置信样本比例上涨 | 周报里体现 | 下个迭代 |

> **第十节第 20 条的教训值得再说一次**：P0 只发群消息是不够的。本章 7.3 节那个工单超 SLA 9 小时，原因就是"02:14 发现，09:00 才有人看到"。**告警渠道的强度必须匹配级别。**

### 11.4 流水线的性能与并发

| 约束 | 数值（示例性，按你的配额调） | 说明 |
|---|---|---|
| judge 并发 | 6~8 | 太高触发 429，重试反而更慢 |
| 被测系统并发 | 8~12（在线 API）/ 16~32（本地 vLLM） | 本地部署可以放开 |
| CI 并发 job 数 | ≤ 2 个评测 job | 多个 PR 同时跑会互相抢 API 配额 |
| 单 job 超时 | 冒烟 15 min / 回归 90 min / 全量 300 min | 必设，防止挂死 |
| 评测库连接池 | 5~10 | 入库是批量操作，不需要大连接池 |

### 11.5 降级策略

API 挂了、配额用光了、judge 不可用了，流水线不能整体瘫痪：

| 故障 | 降级方案 | 代价 |
|---|---|---|
| judge API 不可用 | 自动切 `--no-judge`：只跑规则 + 检索指标，门禁只保留 A 类确定性红线 | 失去质量维度，但**安全红线仍然有效** |
| 主 judge 模型限流 | 切备用模型（配置里预置 `fallback_model`），并在报告里**显著标注"本次使用备用 judge，分数不可与历史比较"** | 分数不可比 |
| 向量库不可用 | 冒烟降级为纯 LLM 系统（`s1_pure_llm`）跑一遍，至少验证服务没崩 | 检索指标缺失 |
| 评测库不可用 | 产物仍写本地 `runs/`，门禁降级为"与上一次本地运行比" | 失去历史基线 |
| 预算耗尽 | `cost_guard.py` 拒绝启动大任务，但**冒烟集永远放行**（它太便宜了） | 回归暂停 |

---

## 十二、本章小结 + 自测题

### 12.1 要点回顾

1. **三套评测工具各有定位，不要二选一**：RAGAS 做**交叉验证**（每月一次，看两把尺子的相关系数 ρ 而不是绝对值差）；DeepEval 做**核心用例的 pytest 断言**（"这 20 条绝不能退化"）；自研 harness 做**上线门禁与归因**（主力）。加上 OpenCompass / lm-eval 做微调后的**通用能力回归**，四者组成完整覆盖。
2. **中文 + RAGAS 有四个固定坑**：NaN 满天飞（必须报 `n_valid`）、中文效果偏低（别拿 `answer_relevancy` 当主指标）、成本高（每指标内部都是多次 LLM 调用）、指标波动（差异 < 0.03 不下结论）。
3. **通用基准只在同一套配置内部可比**。看 `Value ± Stderr`：CMMLU 差 0.9 个点可能是显著退化，GSM8K 差 2.3 个点可能只是噪声。**必须先跑基座基线并长期保存。**
4. **流水线分四道闸口，配置各不相同**：PR 冒烟 30 条 5 分钟只卡确定性红线；每日回归 300 条跑 3 轮卡相对退化；发版全量 1000+ 条跑全部红线；上线后持续在线监控。**用 30 条去卡 2 分的下降，只会制造假警报。**
5. **防抖动是门禁的生命线**，四层保护：3 轮取中位数 → 与 7 日滚动中位数比 → MDE 过滤（小于最小可信差异直接放过）→ 配对 McNemar 显著性检验 → 连续 2 天才阻断。**门禁一旦制造三次假警报，团队就会开始绕过它。**
6. **看板看五张图**：总分趋势（带置信带）、分类雷达、延迟分位数、成本堆叠柱、拒答率双轨。数据落在 `eval_run` / `eval_item` / `eval_gate` 三张表，看板一律查视图不查明细。
7. **在线评测补上离线的三个盲区**：点赞点踩（必须带原因标签，反馈率只有 1%~5%，有偏，只看方向）、无标注质量估计（LLM 分层抽样打分，**只用高置信样本算平均分**，置信度分布本身就是指标）、代理指标（换问法重问率是最可信的免费信号）。
8. **影子流量 → 灰度 5% → 20% → 50% → 全量（留 5% holdout）**。AB 实验按 `user_id` 分流、提前算样本量、到期才看结论、守护指标自动阻断放量。小用户量场景要放大 MDE 或改用配对设计。
9. **badcase 闭环的归因四步不能跳**：知识库里有吗 → 检索召回了吗 → 答案错在哪 → 题本身对吗。**最常犯的错是跳过第 2 步直接改 prompt。** 闭环健康度看四个数字：新增/关闭比、超期数、入金标集比例（> 0.8）、根因分布（`golden` < 15%）。
10. **评测成本可测可控**：本章测算约 ￥655/月（API ￥475 + 机器 ￥180，示例性数据），优化五招（缓存 / 分层抽样 / 两阶段 judge / 日常关 swap / 只跑受影响用例）能降约 60%。**但真正贵的是人力（11 人时/月），自动化的目标是减少重复判断，不是省 API 钱。**
11. **三条版本治理红线**：金标集版本、judge 版本、评分口径。任一变更都会让历史分数作废，必须双跑、标注、重建基线。**把三个版本号做成看板常驻标签。**

### 12.2 自测题

**第 1 题（设计题）**：你们团队每天有 20 多次 PR 推送，现在的 CI 里跑 300 条回归，一次 35 分钟、花 ￥8。工程师抱怨太慢，开始用 `[skip ci]` 绕过。请给出一套重新设计的方案，并说明每个决定的依据。

<details>
<summary>参考答案</summary>

**核心判断：把"一个闸口"拆成"四个闸口"，PR 阶段只做它该做的事。**

| 闸口 | 数据集 | judge | 缓存 | 目标 | 门禁范围 | 依据 |
|---|---|---|---|---|---|---|
| 本地（提交前） | 冒烟 30 条 | 关（`--no-judge`） | 开 | **< 1 min** | 无，自查 | 让工程师愿意主动跑 |
| PR | 冒烟 30 条 + 受影响用例 | 开 | **开**（关键） | **< 5 min** | 只卡一票否决 / 注入越狱 / 运行错误数 / 有效样本数 | **30 条的 95% 置信区间宽达 ±15 个点**（第 8.1 章），根本没有能力判断 2 分的退化。用它卡分数 = 制造假警报 |
| 每日回归 | 回归集 300 条（或分层抽 150 条） | 开（关 `swap`） | **关**（否则 3 轮结果一样） | < 40 min | 相对退化（4 层防抖动） | 有统计效力做相对判定 |
| 发版 | 全量 1000+ 条 + 通用基准 + RAGAS | 开（开 `swap`） | 关 | < 3 h | 全部红线 | 发版值得花这个时间和钱 |

**关键技术决定**：

1. **PR 阶段开缓存**：judge 结果按 `(judge_model, prompt_version, qid, answer, contexts_digest)` 哈希缓存。改检索参数时，答案变了的题才需要重判——实测缓存命中后单次成本从 ￥0.41 降到 ￥0.12，耗时从 3 分钟降到 1 分多钟。
2. **只跑受影响用例**（`select_cases.py`）：按"代码模块 → 业务能力"的人工映射表加跑，而不是替代冒烟集。
3. **`concurrency` 取消旧运行**：同一 PR 连续推送时 `cancel-in-progress`，省额度也省等待。
4. **结果贴成 PR 评论**：结论在第一行，明细折叠。**让人不需要点开 CI 日志。**

**关于 `[skip ci]`**：先修体验（快 + 准），**再**在仓库设置里禁止绕过必需检查。顺序反了就会引发对抗——这是本章第十节第 20 条的教训。

</details>

**第 2 题（判断题）**：每日回归报告显示：总分 74.2 → 72.6（-1.6），Recall@5 0.869 → 0.841（-0.028），通过率 76.7% → 75.3%。配对分析显示 17 条题变好、24 条题变坏。`gate.yaml` 里 `total_score` 的 MDE 是 1.5、`recall_at_5` 的 MDE 是 0.02。请判断：该不该阻断合并？给出完整推理过程。

<details>
<summary>参考答案</summary>

**结论：不该阻断，应该告警并观察明天。**

推理分四步（就是 5.6.3 节 `gate.py` 的判定顺序）：

1. **总分 -1.6，MDE 是 1.5** → 退化 1.6 > MDE 1.5，**通过不了 MDE 过滤**，需要继续检验。（注意：如果是 -1.4 就直接判噪声放过了。）
2. **Recall@5 -0.028，允许退化 -0.02，MDE 0.02** → 同样超过，需要继续检验。
3. **配对显著性检验**：改善 17 条、退化 24 条，McNemar 统计量
   $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|17-24|-1)^2}{41} = \frac{36}{41} \approx 0.878$$
   对应 p ≈ 0.35 ≫ 0.05，**统计上不显著**。净退化只有 7 条，完全可能是噪声。
4. **连续性判定**：`consecutive_days: 2`，这是第一天出现退化 → **降级为告警**，并在报告里写明"明天若仍退化将阻断"。

**还要补做的三件事**：

1. **看分类维度**：总分可能是被某一类题拖下来的。如果"安全规范"这一类退化了，**哪怕统计不显著也要人工看**——安全类是 A 类确定性红线，不走统计那一套。
2. **确认数据集指纹没变**：如果 `dataset_fingerprint` 变了，这个对比本身就无效（第十节第 5 条）。
3. **看那 24 条变坏的题**：即使统计不显著，这 24 条也是最好的 badcase 线索，扫一眼有没有共同模式（比如全是 XJ-150 相关）。

**一句话总结**：门禁的判定是"MDE + 显著性 + 连续性"三重与条件，**但人工复核永远不受这三条限制**。

</details>

**第 3 题（工程题）**：业务方说"你们的分数我看不懂，但客服还是经常翻手册"。离线总分 75.8、通过率 78.3%，看起来不错。请设计一套方案，用**不超过两周**的时间搞清楚"离线分数和业务价值之间断在哪里"。

<details>
<summary>参考答案</summary>

这是本章开头那个故事的升级版。**核心思路：建立"离线分数 → 在线行为 → 业务指标"的三级对照，找出断点在哪一级。**

**第 1~3 天：建立线上真实分布的基线**

1. 导出最近 30 天的线上 query（脱敏），用 embedding 聚类，看**金标集覆盖度**：有多少比例的线上问题在金标集里找不到相似题（相似度 < 0.55）。
   - 若覆盖度低（比如 40% 的线上问题金标集没覆盖）→ **断点在金标集**，离线分数测的根本不是用户在问的东西。
2. 同时统计线上问题的分类分布，与金标集的分类分布做对照。**典型发现：金标集里"规格查询"占 30%，但线上"备件查询"才是大头。**

**第 4~7 天：接上在线信号**

3. 上线点赞点踩 + 原因标签（6.1 节），**重点看原因标签分布**。如果 `too_vague`（答得太笼统不能操作）占了大头，说明：**离线 judge 的 rubric 与业务的"可用"定义不一致**——这正是本章开头故事的根因（我们判"结论正确即可"，业务要"结论 + 页码 + 可操作步骤"）。
4. 上线抽样打分（6.2 节）+ 代理指标，特别是**复制答案的比例**和**换问法重问率**。

**第 8~10 天：做相关性分析（最关键的一步）**

5. 对同一批线上消息，把三个数字放在一起算相关系数：
   - 离线 judge 分（在金标集上的同类题得分）
   - 在线抽样打分
   - 用户点赞/点踩
   - **ρ 高** → 离线分数是有效代理，问题在覆盖度（回第 1 步）。
   - **ρ 低** → **离线 rubric 有问题**，judge 在奖励一些用户并不在意的东西。

**第 11~14 天：改 rubric 并验证**

6. 组织业务方做一次 50 条盲评（第 8.1 章的做法），**用业务方的评分反推 rubric**：让业务方说"为什么这条不合格"，把理由归类，直接变成 rubric 的维度和 `must_include` 规则。
7. 典型改动：在 `correctness` 之外单独加一个 `actionability`（可操作性）维度，要求答案必须含步骤/参数/出处页码，权重给 0.2~0.3；同时加确定性规则"涉及维修步骤的答案必须包含引用页码，否则扣分"。
8. 用新 rubric 重算历史 3 个时间点的分数，看趋势是否与业务体感一致。**如果重算后的曲线和业务方的体感对上了，说明找对了。**

**交付物**：① 金标集覆盖度报告（缺口清单 → 回流计划）；② 原因标签分布；③ 三级指标相关系数；④ 新版 rubric + 用新旧口径重算的对照趋势图。

**最后一句话（也是本章的态度）**：**评测体系的终极标准不是"分数高"，而是"分数涨的时候业务方也感觉变好了"。** 这两条曲线一旦脱钩，先怀疑你的尺子，不要怀疑业务方的感觉。

</details>

---
**上一章** [8.3 DeepSeek-Harness 自研评测框架](./03-DeepSeek-Harness自研评测框架.md) | **下一章** [项目 1：DeepSeek-Harness 与 LLM-Wiki 知识库评测实战](../09-实战项目/项目1-DeepSeek-Harness与LLM-Wiki知识库评测实战.md)
