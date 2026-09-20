# 附录 C  学习资源与论文清单

> **这份清单解决什么问题**：
> 大模型领域的信息量大到无法全读，而且噪声极高。本附录的目标不是"列得全"，而是**帮你把有限的时间花在回报最高的材料上**。
>
> **三条使用原则**：
> 1. **不要按顺序读完。** 按你当前遇到的问题去挑，带着问题读的效率是漫无目的读的 10 倍。
> 2. **论文看思路，不看数字。** 论文里的分数在你的数据上基本不可能复现（数据、超参、评测口径都不同），
>    有价值的是**方法思路**和**消融结论**。这也是本书坚持"所有数字标注实测环境"的原因。
> 3. **官方文档优先于任何二手材料。** 本领域的框架迭代极快，博客的半衰期大约三个月。

> **关于本清单的严谨性声明**：
> - 论文部分**只列作者确信存在的经典工作**，给出论文标题和核心贡献。
> - **不标注作者、年份、发表会议和引用数**——这类信息极易记错，而且对"要不要读"的判断没有帮助。
>   需要准确出处时，请用论文标题到 arXiv、Semantic Scholar 或 Google Scholar 检索，以检索结果为准。
> - 官方文档和开源项目**只给名称和查找路径描述，不给可能失效的深链接**。
> - 榜单分数、社区规模、课程价格等**变动信息一律不写具体数字**。

---

## 目录

| 章节 | 内容 |
|---|---|
| [§1 必读论文清单](#1-必读论文清单) | 六个主题分组，每篇给核心贡献 + 为什么值得读 + 对应本书章节 |
| [§2 官方文档清单](#2-官方文档清单) | 框架与工具的文档入口，以及"该看哪一部分" |
| [§3 开源项目清单](#3-开源项目清单) | 按用途分组，给定位和适用场景 |
| [§4 榜单与评测平台](#4-榜单与评测平台) | 各自测什么、怎么看、局限（尤其数据污染） |
| [§5 中文社区与信息源](#5-中文社区与信息源) | 怎么筛选高质量信息，怎么不被营销号带偏 |
| [§6 数据集资源](#6-数据集资源) | 中文问答/指令/评测数据集的类型与获取渠道 |
| [§7 学习路径推荐](#7-学习路径推荐) | 会用 / 会做 / 会优化 三个层次 |
| [§8 本书之后学什么](#8-本书之后学什么) | 多模态、推理模型、模型压缩、具身智能 |
| [§9 每周信息摄入模板](#9-每周信息摄入模板) | 用 3 小时跟上这个领域 |
| [§10 这份清单的保质期与更新节奏](#10-这份清单的保质期与更新节奏) | 哪部分耐用、哪部分半年就要重评 |
| [§11 必读论文补遗（第二批）](#11-必读论文补遗第二批续-1) | 溯源类 / 对照类 / 工程类论文，另加安全与长上下文两个主题 |
| [§12 开源项目补遗](#12-开源项目补遗续-3) | 文档解析与 OCR 单列，另加约束解码、网关缓存、prompt 管理、检索底层库 |
| [§13 每周模板完整表格版](#13-每周信息摄入模板完整表格版续-9) | 可打印总表 + 三种角色变体 + 月度季度节奏 |
| [§14 致读者](#14-致读者) | 关于持续学习与"怎么判断"的最后几句话 |

---

## 1. 必读论文清单

### 1.0 怎么读这份清单

论文分三个优先级，**不要平均用力**：

| 优先级 | 标记 | 含义 | 建议投入 |
|---|---|---|---|
| 必读 | ★★★ | 不读会影响你对整个领域的理解框架 | 精读，做笔记 |
| 推荐 | ★★ | 遇到对应问题时读，能直接指导实现 | 按需精读 |
| 了解 | ★ | 知道有这么个东西即可 | 读摘要和 Figure 1 |

**工程师读论文的正确姿势**（详见 [附录 B §12.4](B-术语中英对照表.md#124-读一篇论文的实用顺序)）：
摘要 → 第一张图 → 局限性 → 消融实验 → 方法。**别从头读到尾。**

---

### 1.1 Transformer 与基础模型

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Attention Is All You Need** | ★★★ | 提出 Transformer 架构，用自注意力完全替代循环结构 | 一切的起点。**不理解注意力就无法理解 KV Cache 为什么吃显存**，也就无法做容量规划 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** | ★★ | 双向编码器 + 掩码语言建模的预训练范式 | 理解 encoder-only 架构。**你用的所有 embedding 模型和 reranker 都是这个家族的后代** | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Language Models are Few-Shot Learners** | ★★ | 提出 GPT-3，展示了规模带来的 in-context learning 能力 | 理解"为什么 prompt 里给几个例子就能改变模型行为"，这是提示工程的理论基础 | [01.5](../01-大模型基础与技术选型/05-提示工程与结构化输出.md) |
| **Training language models to follow instructions with human feedback** | ★★★ | 提出 InstructGPT，确立"预训练 → SFT → RLHF"的三段式流程 | 这是当前所有 Instruct 模型的训练范式。**理解它才知道你的微调该插在哪一段** | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **LLaMA: Open and Efficient Foundation Language Models** | ★★ | 展示了用更多数据训练较小模型的可行性，开启开源大模型浪潮 | 理解开源模型的技术谱系。国产模型大多沿用了类似的架构选择 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **Scaling Laws for Neural Language Models** | ★★ | 给出模型性能与参数量/数据量/算力的幂律关系 | 理解"为什么大模型必须大"，以及**什么时候加参数已经不划算了** | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **Training Compute-Optimal Large Language Models** | ★ | 修正了缩放定律，指出此前的模型训练数据量普遍不足 | 理解为什么现在的 7B 模型比几年前的 13B 还强 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** | ★★ | 提出 RoPE 旋转位置编码 | RoPE 是长上下文外推的技术基础。**调 `rope_scaling` 参数之前必须理解它** | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** | ★★ | 提出分组查询注意力，在 MHA 和 MQA 之间取平衡 | **直接决定 KV Cache 的显存占用**，做显存估算必读 | [01.4](../01-大模型基础与技术选型/04-算力配置与成本测算.md) |
| **Fast Transformer Decoding: One Write-Head is All You Need** | ★ | 提出 MQA，所有 Q 头共享一组 KV | GQA 的前身，理解 KV 共享思路的来源 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** | ★ | 提出简化的 MoE 路由机制 | 理解 MoE 模型为什么"总参数大但推理便宜"，影响选型判断 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |

---

### 1.2 RAG

这是本书最核心的主题，论文也最值得细读。**建议按下表从上到下的顺序读**——它大致也是 RAG 技术的演进顺序。

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | ★★★ | 提出 RAG 框架，把参数化知识和非参数化知识结合 | **RAG 这个词的出处**。理解"为什么要检索"而不只是"怎么检索" | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **Dense Passage Retrieval for Open-Domain Question Answering** | ★★★ | 证明稠密向量检索可以超越 BM25，提出双塔训练范式 | **理解 Bi-Encoder 的训练方式和难负样本挖掘**，这是 embedding 微调的基础 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** | ★★ | 提出后期交互（late interaction），在精度和效率间取折中 | 理解 Bi-Encoder 和 Cross-Encoder 之间还有第三条路，bge-m3 的多向量模式受此启发 | [02.4](../02-RAG基础篇/04-检索重排生成全链路.md) |
| **REALM: Retrieval-Augmented Language Model Pre-Training** | ★ | 在预训练阶段就引入检索 | 了解"检索与生成联合训练"这条路线，虽然工程上很少这么做 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** | ★★ | 用孪生网络结构产出可比较的句向量 | **理解为什么不能直接拿 BERT 的 [CLS] 当句向量用**，这是个极常见的误解 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Text Embeddings by Weakly-Supervised Contrastive Pre-training** | ★★ | 提出 E5 系列，用大规模弱监督对比学习训练通用 embedding | 理解现代通用 embedding 模型的训练配方，以及 `query:` / `passage:` 前缀的由来 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation** | ★★★ | 提出 BGE-M3，同时支持稠密、稀疏、多向量三种检索模式 | **本书默认的 embedding 模型就是它**。读了才知道它的三种模式怎么配合用 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Lost in the Middle: How Language Models Use Long Contexts** | ★★★ | 实验证明模型对长上下文中间位置的信息利用最差 | **直接指导上下文组装策略**：重要的放头尾。也是"别把上下文塞满"的实证依据 | [06.3](../06-Agent智能体/03-记忆系统与上下文工程.md) |
| **Precise Zero-Shot Dense Retrieval without Relevance Labels** | ★★ | 提出 HyDE：先让模型生成假答案，用假答案去检索 | 一个反直觉但很有效的技巧，**零样本场景下收益明显** | [03.1](../03-RAG进阶与性能优化/01-查询理解与高级检索策略.md) |
| **Query2doc: Query Expansion with Large Language Models** | ★★ | 用 LLM 把短查询扩展成伪文档再检索 | 和 HyDE 思路相近，**两篇一起读能看清查询扩展这类方法的共性** | [03.1](../03-RAG进阶与性能优化/01-查询理解与高级检索策略.md) |
| **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models** | ★★ | 提出 Step-Back Prompting：先退一步问更抽象的问题 | 处理"太具体导致检索不到"的查询非常有效 | [03.1](../03-RAG进阶与性能优化/01-查询理解与高级检索策略.md) |
| **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** | ★★★ | 用特殊 token 让模型自己决定何时检索、并批判检索结果 | **理解"自适应检索"的思路**。即使不训练特殊 token，用 prompt 也能实现简化版 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **Corrective Retrieval Augmented Generation** | ★★★ | 提出 CRAG：用轻量评估器判断检索质量，差就触发纠正动作 | **工程上最容易落地的 RAG 改进之一**，本书用 LangGraph 完整实现了它 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions** | ★★ | 提出 IRCoT：把检索和思维链交错进行，解决多跳问题 | **多跳检索的标准方法**。单轮检索解决不了的问题看这篇 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **Active Retrieval Augmented Generation** | ★★ | 提出 FLARE：生成过程中按需主动触发检索 | 理解"检索时机"这个维度，长文生成场景有用 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval** | ★★ | 递归聚类 + 摘要构建多层次索引树 | **解决"问题需要跨多个文档的全局信息"这类场景**，是 GraphRAG 之外的另一条路 | [03.6](../03-RAG进阶与性能优化/06-GraphRAG与结构化知识融合.md) |
| **From Local to Global: A Graph RAG Approach to Query-Focused Summarization** | ★★★ | 提出 GraphRAG：抽实体建图，支持局部和全局两种查询 | **理解"向量检索解决不了什么问题"**。注意它的构建成本很高，读的时候重点看代价 | [03.6](../03-RAG进阶与性能优化/06-GraphRAG与结构化知识融合.md) |

---

### 1.3 Agent

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | ★★★ | 发现加入推理步骤示例能大幅提升复杂推理表现 | **Agent 的思想起点**。所有后续推理范式都是它的衍生 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Large Language Models are Zero-Shot Reasoners** | ★★ | 发现只加一句"让我们一步步思考"就能触发推理 | 理解 zero-shot CoT 的威力和局限，**成本最低的推理增强手段** | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Self-Consistency Improves Chain of Thought Reasoning in Language Models** | ★★★ | 采样多条推理路径后投票取多数 | **最简单粗暴但确实有效的可靠性提升手段**。代价是 token 成倍增长，读的时候算一下账 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **ReAct: Synergizing Reasoning and Acting in Language Models** | ★★★ | 提出 Thought-Action-Observation 交替循环 | **Agent 最经典的范式**，几乎所有 Agent 框架的默认实现 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Reflexion: Language Agents with Verbal Reinforcement Learning** | ★★ | 让 Agent 在失败后用自然语言反思并改进下次尝试 | 理解"自我改进"怎么在不训练的情况下实现 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | ★★ | 把推理组织成树并做搜索 | 理解推理的"广度"维度。**注意成本极高，企业场景要慎用** | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models** | ★★ | 先一次性规划所有工具调用再并行执行 | **显著降低 token 消耗**，是 ReAct 的成本优化版，工程价值高 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** | ★★ | 把复杂问题分解成从易到难的子问题依次解决 | **问题分解的方法论来源**，多跳检索和任务规划都用得上 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models** | ★★ | 先规划再执行的零样本提示法 | Plan-and-Execute 范式的理论依据 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Toolformer: Language Models Can Teach Themselves to Use Tools** | ★★ | 让模型自监督地学会何时调用 API | **理解工具调用能力的来源**。虽然现在都用 function calling，但这篇讲清了本质 | [06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face** | ★ | 用 LLM 作为控制器调度其他专用模型 | 理解"LLM 作为调度中枢"的思路，是多智能体的雏形 | [07.1](../07-多智能体协同/01-多智能体架构模式全解.md) |
| **Generative Agents: Interactive Simulacra of Human Behavior** | ★★ | 构建有记忆、反思、规划能力的仿真智能体 | **记忆系统设计的重要参考**（观察-反思-检索三层结构） | [06.3](../06-Agent智能体/03-记忆系统与上下文工程.md) |
| **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework** | ★★ | 提出以对话为核心的多智能体框架 | 理解多智能体的一种主流组织方式，对比 LangGraph 的状态机思路 | [07.1](../07-多智能体协同/01-多智能体架构模式全解.md) |
| **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework** | ★ | 把软件公司的 SOP 编码进多智能体协作 | **理解"给 Agent 加流程约束"的价值**——这正是企业落地需要的 | [07.1](../07-多智能体协同/01-多智能体架构模式全解.md) |
| **Voyager: An Open-Ended Embodied Agent with Large Language Models** | ★ | 在开放环境中持续学习并积累技能库 | 了解"技能库"这个概念，对设计可复用的工具集有启发 | [06.4](../06-Agent智能体/04-MCP协议与工具生态.md) |

---

### 1.4 微调与对齐

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **LoRA: Low-Rank Adaptation of Large Language Models** | ★★★ | 用低秩矩阵分解实现参数高效微调，可合并回原权重 | **本书微调部分的核心方法**。理解 $\Delta W = BA$ 才知道 rank 和 alpha 该怎么调 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | ★★★ | 4bit 量化基座 + LoRA，单卡微调大模型成为可能 | **消费级显卡跑微调的技术基础**。NF4、双重量化、分页优化器三个技巧都要理解 | [05.3](../05-微调LoRA与PEFT/03-LoRA-QLoRA实战.md) |
| **Parameter-Efficient Transfer Learning for NLP** | ★★ | 提出 Adapter，在层间插入小瓶颈网络 | PEFT 的开山之作，**理解 LoRA 相对它的改进在哪** | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **Prefix-Tuning: Optimizing Continuous Prompts for Generation** | ★ | 在输入前加可训练的连续前缀 | 了解 PEFT 家族的另一条路线 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **The Power of Scale for Parameter-Efficient Prompt Tuning** | ★ | 发现模型越大，prompt tuning 越接近全参微调 | 理解"规模改变方法有效性"这个重要现象 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **DoRA: Weight-Decomposed Low-Rank Adaptation** | ★★ | 把权重分解成幅度和方向分别适配 | LoRA 的改进版，**PEFT 库里可以一个参数开启，值得试** | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models** | ★ | 用移位稀疏注意力低成本扩展上下文 | 需要扩展模型上下文长度时读 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | ★★★ | 跳过奖励模型，直接用偏好对优化策略 | **RLHF 的工程简化版**，是多数企业唯一负担得起的对齐方法 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **Proximal Policy Optimization Algorithms** | ★ | 提出 PPO 算法 | RLHF 的底层算法。**不做 RLHF 的话读摘要即可** | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **Constitutional AI: Harmlessness from AI Feedback** | ★ | 用一套原则让 AI 自己监督自己的输出 | 理解"用 AI 反馈替代人类反馈"的思路，对做自动化数据质检有启发 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **Self-Instruct: Aligning Language Models with Self-Generated Instructions** | ★★★ | 用少量种子指令自举生成大规模指令数据 | **数据构造的核心方法**。本书的训练数据生成流水线就基于这个思路 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **LIMA: Less Is More for Alignment** | ★★★ | 用极少量高质量数据就能达到很好的对齐效果 | **对企业场景极其重要**：说明你不需要几十万条数据，几千条精标的就够了 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **Scaling Instruction-Finetuned Language Models** | ★ | 大规模多任务指令微调 | 理解指令微调的任务多样性为什么重要 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |

---

### 1.5 评测

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **RAGAS: Automated Evaluation of Retrieval Augmented Generation** | ★★★ | 提出无需标准答案的 RAG 评测指标体系 | **本书评测章的重要参考**。理解 faithfulness 和 answer relevancy 怎么算出来的 | [08.4](../08-评测体系/04-RAGAS与自动化评测流水线.md) |
| **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** | ★★★ | 系统研究了用 LLM 做评判的可行性与偏置 | **做 LLM-as-Judge 必读**。位置偏置、冗长偏置、自我增强偏置都出自这里 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment** | ★★ | 用 CoT + 概率加权提升 LLM 评判与人类的一致性 | 理解怎么让 judge 打分更稳定，**本书的 judge 设计借鉴了这个思路** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models** | ★★ | 构建跨领域的检索评测基准 | **理解"检索模型换个领域就不行了"这个现象**，是做领域 embedding 微调的理由 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **MTEB: Massive Text Embedding Benchmark** | ★★ | 大规模多任务 embedding 评测基准 | **选 embedding 模型时的主要参考**，但要注意看中文子集而不是总榜 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Benchmarking Large Language Models in Retrieval-Augmented Generation** | ★★ | 系统测试 LLM 在 RAG 场景下的四种能力（噪声鲁棒性、拒答、信息整合、反事实鲁棒性） | **RAG 失败模式的分类依据**，本书的 badcase 归因体系受此启发 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **Measuring Massive Multitask Language Understanding** | ★★ | 提出 MMLU 多学科知识评测 | 理解通用能力榜单测的是什么，**以及它和你的业务能力几乎无关** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models** | ★★ | 中文多学科评测基准 | 中文模型选型的常见参考，**同样要警惕数据污染** | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **CMMLU: Measuring Massive Multitask Language Understanding in Chinese** | ★ | 另一个中文多学科评测集 | 与 C-Eval 互为补充 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **TruthfulQA: Measuring How Models Mimic Human Falsehoods** | ★★ | 专门测试模型是否会复述常见的错误说法 | **理解幻觉的一种重要来源**：模型学会了人类的常见误解 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models** | ★★ | 构建幻觉检测评测集 | 做幻觉治理时的参考，理解幻觉的分类方式 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Holistic Evaluation of Language Models** | ★ | 提出多维度全面评测框架 HELM | **理解"单一分数无法刻画模型能力"这个立场**，多维评测的思想来源 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |

---

### 1.6 推理优化

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | ★★★ | 用分块计算和重计算减少 HBM 读写，加速注意力且省显存 | **理解"访存才是瓶颈"这个关键洞察**。它改变了整个推理优化的思路 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | ★★ | 改进并行度和工作划分，进一步提速 | 知道有第二代即可，工程上装最新版就行 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | ★★★ | 提出 PagedAttention，像操作系统分页一样管理 KV Cache | **vLLM 的核心论文**。理解它才知道为什么 vLLM 的吞吐能高那么多 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** | ★★ | 提出连续批处理（iteration-level scheduling） | **连续批处理的出处**，理解它和静态批处理的本质区别 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **SGLang: Efficient Execution of Structured Language Model Programs** | ★★ | 提出 RadixAttention 实现前缀 KV 复用，以及结构化生成的运行时 | **多轮对话和固定 system prompt 场景收益巨大**，是 vLLM 之外的重要选项 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Fast Inference from Transformers via Speculative Decoding** | ★★ | 用小模型草拟、大模型验证，加速生成 | 理解投机解码的原理与适用条件（**接受率低时反而更慢**） | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** | ★★ | 发现激活值中的离群特征，提出混合精度分解量化 | **理解"为什么大模型量化会崩"**，bitsandbytes 的理论基础 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | ★★ | 逐层最小化量化误差的 PTQ 方法 | 量化部署的主流方案之一 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | ★★★ | 根据激活分布保护重要权重通道 | **本书推荐的量化方案**。理解"不是所有权重同等重要"这个核心思想 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** | ★★ | 支持单机同时服务大量 LoRA adapter | **多租户/多领域微调模型部署的关键技术**，vLLM 的多 adapter 能力源于此 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** | ★ | 提出张量并行的经典切分方案 | 理解 TP 是怎么切的，**知道为什么 TP size 必须整除头数** | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** | ★★ | 把优化器状态、梯度、参数分片到多卡 | **多卡训练必读**。理解 stage 1/2/3 的区别才知道该选哪个 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |

---

### 1.7 如果只能读 10 篇

时间极其有限时，按这个顺序读，**它们覆盖了本书 80% 的技术决策依据**：

```text
1.  Attention Is All You Need                                  —— 一切的基础
2.  Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  —— RAG 的定义
3.  Dense Passage Retrieval for Open-Domain Question Answering  —— 检索怎么训出来的
4.  Lost in the Middle                                          —— 上下文该怎么组装
5.  Chain-of-Thought Prompting Elicits Reasoning                —— 推理能力从哪来
6.  ReAct                                                       —— Agent 的基本范式
7.  LoRA                                                        —— 微调怎么做
8.  Direct Preference Optimization                              —— 对齐怎么做
9.  Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena      —— 评测怎么做
10. Efficient Memory Management ... with PagedAttention          —— 推理怎么变快
```

---
## 2. 官方文档清单

> **本节不给具体链接**——框架文档的 URL 结构变动频繁，深链接的半衰期比博客还短。
> 给的是**名称 + 该看哪一部分**，你用搜索引擎搜「项目名 + docs」即可找到当前入口。
>
> **一条硬规矩**：遇到 API 用法问题，**先查官方文档，再查 GitHub issue，最后才看博客**。
> 顺序反过来会浪费大量时间在过期信息上。

### 2.1 核心框架

| 项目 | 文档入口找法 | 重点看哪部分 | 本书对应版本 |
|---|---|---|---|
| **LangChain** | 官网文档站，分 Python 和 JS 两套，注意选 Python | LCEL 章节、Integrations 索引、Migration guide | 0.3.x |
| **LangGraph** | LangChain 文档站下的独立子站 | Concepts（State/Node/Edge/Reducer）、Persistence、Human-in-the-loop | 0.2.x |
| **LlamaIndex** | 官网文档站 | Loading/Indexing/Querying 三段式、Node Postprocessor、Query Pipeline | 0.12.x |
| **Hugging Face Transformers** | HF 官网 docs 目录 | Generation strategies、Quantization、Chat templates、Trainer API | 4.44+ |
| **PEFT** | HF docs 下的 peft 子站 | Conceptual guides、LoRA 配置项说明、Model merging | 0.13.x |
| **TRL** | HF docs 下的 trl 子站 | SFTTrainer、DPOTrainer 的参数说明 | 最新 |
| **vLLM** | 官方文档站 | Engine Arguments（**逐个参数读一遍，值得**）、OpenAI Compatible Server、Quantization | 0.6.x |
| **Ollama** | 官方 GitHub README + docs 目录 | Modelfile 语法、API 端点、模型库 | 最新 |

**关于文档的一个技巧**：LangChain / LlamaIndex 这类快速迭代的框架，**文档站通常有版本切换器**。
确认你看的文档版本和你装的库版本一致，这能避免附录 A §5 里一半的问题。

### 2.2 数据与存储

| 项目 | 文档入口找法 | 重点看哪部分 | 本书对应版本 |
|---|---|---|---|
| **Milvus** | 官网 docs，注意选版本 | Schema 设计、Index 类型与参数、Search 参数、Alias 管理、运维章节 | 2.4 |
| **Attu** | Milvus 官网的可视化工具页 | 连接配置即可，它主要靠界面 | 端口 8000 |
| **Chroma** | 官网 docs | Collection API、Persist 配置 | 最新 |
| **Qdrant** | 官网 docs | Payload 过滤、Quantization、Snapshot | 最新 |
| **pgvector** | GitHub README（文档就在 README 里） | 索引类型（ivfflat/hnsw）、距离操作符、参数调优 | 最新 |
| **Elasticsearch** | Elastic 官网 docs，选 8.x | Mapping、Analyzer（**中文必看 IK 插件**）、Query DSL、kNN search | 8.x |
| **Neo4j** | 官网 docs | Cypher 语法、索引、APOC 库 | 最新 |

### 2.3 评测与可观测

| 项目 | 文档入口找法 | 重点看哪部分 |
|---|---|---|
| **RAGAS** | 官方文档站 | Metrics 定义（**每个指标的计算过程都要看懂**）、自定义 LLM/Embedding 配置 |
| **DeepEval** | 官方文档站 | Metrics、Pytest 集成 |
| **Langfuse** | 官方文档站 | Self-hosting（compose 配置）、SDK 集成、Prompt management、Datasets |
| **OpenTelemetry** | 官网 docs，选 Python SDK | Trace API、Instrumentation、Exporter 配置 |
| **Weights & Biases** | 官网 docs | Experiment tracking、Sweeps |
| **SwanLab** | 官方文档站 | 训练实验记录，国内访问更稳定 |

### 2.4 训练与部署

| 项目 | 文档入口找法 | 重点看哪部分 |
|---|---|---|
| **LLaMA-Factory** | GitHub README + examples 目录 | **examples 目录下的 yaml 是最好的文档**，比正文说明更实用 |
| **DeepSpeed** | 官网 docs | ZeRO 配置、Config JSON 说明 |
| **Accelerate** | HF docs 下的 accelerate 子站 | Config 文件生成、device_map 说明 |
| **bitsandbytes** | GitHub README + HF 的 quantization 文档 | 安装排错（**README 的 troubleshooting 一节很有用**） |
| **FastAPI** | 官网 docs | 异步、依赖注入、Lifespan、StreamingResponse |
| **Docker Compose** | Docker 官网 docs | Compose file reference（**v2 语法**）、Healthcheck、Deploy resources |

### 2.5 模型厂商 API 文档

| 厂商 | 文档入口找法 | 重点看哪部分 |
|---|---|---|
| **DeepSeek** | 开放平台官网的 API 文档 | 模型列表、参数限制、**推理模型的特殊约束**、计费规则、限流规则 |
| **通义千问（DashScope）** | 阿里云百炼平台文档 | OpenAI 兼容模式、模型列表、限流配额 |
| **OpenAI** | 官网 API reference | Chat Completions 的参数定义是**事实标准**，其他厂商都在对齐它 |

**重要提醒**：模型 API 的**限流规则、计费规则、参数支持情况**会变，且各家不同。
把它们抄进你自己的 `docs/llm-providers.md` 并定期核对，**不要依赖记忆**。
这也是附录 A 的 A11（推理模型参数不兼容）那类问题的根本预防手段。

---

## 3. 开源项目清单

按用途分组。**每个项目给一句话定位 + 什么时候该用它**，不给 star 数（会变，而且 star 数不代表适合你）。

### 3.1 推理与服务

| 项目 | 定位 | 什么时候用 | 什么时候别用 |
|---|---|---|---|
| **vLLM** | 高吞吐 LLM 推理引擎 | **生产部署的默认选择**，需要高并发 | 只有一两个请求的开发环境（启动慢） |
| **SGLang** | 面向结构化生成的推理引擎 | 多轮对话、大量共享前缀、结构化输出场景 | 简单的单轮问答（vLLM 够用） |
| **Ollama** | 本地模型的"docker" | **本地开发、快速试模型**、Mac 上跑 | 生产环境（吞吐不够） |
| **llama.cpp** | C++ 实现的轻量推理 | CPU 推理、边缘设备、Mac | 有 GPU 的服务器（用 vLLM） |
| **Text Generation Inference (TGI)** | HF 的推理服务 | 已经在 HF 生态里 | — |
| **LMDeploy** | 推理与压缩工具箱 | 需要 TurboMind 后端或特定量化 | — |
| **Xinference** | 多模型统一推理平台 | 需要同时托管 LLM + embedding + rerank | 只跑一个模型 |
| **LiteLLM** | 统一多家 LLM API 的代理层 | **需要在多个模型厂商间路由和降级** | 只用一家（直接用官方 SDK） |

### 3.2 RAG

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **LangChain** | 通用 LLM 应用编排框架 | **需要丰富集成和灵活编排**，本书主力 |
| **LlamaIndex** | 面向索引与检索的框架 | 检索策略复杂（多级索引、递归检索）时更顺手 |
| **Haystack** | 端到端 NLP/RAG 框架 | 偏好 pipeline 式抽象时 |
| **RAGFlow** | 开箱即用的 RAG 系统 | 需要快速给业务方看效果、重视文档解析质量 |
| **Dify** | 低代码 LLM 应用平台 | 需要让非技术同事也能配流程 |
| **FastGPT** | 知识库问答平台 | 类似 Dify，中文生态友好 |
| **Unstructured** | 多格式文档解析库 | 需要统一处理 PDF/Word/PPT/HTML |
| **MinerU** | 面向学术与复杂版面的 PDF 解析 | **扫描件、多栏、大量公式表格的 PDF** |
| **PaddleOCR / PP-Structure** | OCR 与版面分析 | 中文扫描件解析 |
| **FlagEmbedding** | BGE 系列模型的官方工具库 | **用 bge-m3 / bge-reranker 时直接用它**，比自己封装可靠 |
| **rank_bm25** | 纯 Python 的 BM25 实现 | 小规模、不想引入 ES 时（**注意中文要自己分词**） |
| **jieba** | 中文分词 | BM25 中文检索的必备预处理 |

### 3.3 Agent

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **LangGraph** | 基于状态机的 Agent 编排 | **本书主力**。需要循环、分支、中断恢复、人在回路 |
| **AutoGen** | 以对话为核心的多智能体框架 | 偏好"多个角色互相对话"的建模方式 |
| **CrewAI** | 面向角色分工的多智能体框架 | 任务能清晰拆成角色时 |
| **MetaGPT** | 把 SOP 编码进多智能体 | 有明确流程规范的场景 |
| **OpenAI Agents SDK / Swarm** | 轻量的 handoff 式多智能体 | 需要极简的交接模型 |
| **MCP SDK** | Model Context Protocol 的官方实现 | **要把工具做成可复用的标准服务时** |
| **smolagents** | 轻量 Agent 库 | 想看一个小而完整的 Agent 实现 |

### 3.4 微调

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **PEFT** | HF 的参数高效微调库 | **底层库，所有方案都依赖它** |
| **TRL** | HF 的 SFT/DPO/PPO 训练库 | 需要写自定义训练逻辑时 |
| **LLaMA-Factory** | 一站式微调工程化方案 | **本书推荐的工程化方案**。yaml 配置、支持多种方法、有 WebUI |
| **Unsloth** | 微调加速库 | 单卡训练想要更快更省显存 |
| **DeepSpeed** | 分布式训练优化 | 多卡训练、显存不够 |
| **Axolotl** | 配置驱动的微调框架 | LLaMA-Factory 的替代选择 |
| **ms-swift** | ModelScope 的微调框架 | 国内生态、需要适配国产模型的特殊需求 |

### 3.5 评测

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **RAGAS** | RAG 专用自动评测 | **快速得到 RAG 的基础指标**（需先做中文校准，见附录 A 的 V05） |
| **DeepEval** | 类 pytest 的 LLM 评测框架 | 想把评测写成测试用例进 CI |
| **OpenCompass** | 大模型全面评测平台 | 需要在标准 benchmark 上横评多个模型 |
| **lm-evaluation-harness** | 学术标准评测框架 | 需要和论文结果对齐 |
| **promptfoo** | prompt 的 A/B 评测工具 | 迭代 prompt 时做对比 |
| **Giskard** | LLM 质量与安全扫描 | 需要自动发现 badcase 和安全问题 |

**本书的立场**：**第三方评测库用于快速起步，最终要有自己的 harness。**
原因是企业场景的判分口径高度定制（什么算拒答、部分正确怎么算、型号答错扣多少），通用库无法表达。
本书在 [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) 从零实现了一套，代码量不大但完全可控。

### 3.6 可观测与运维

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **Langfuse** | LLM 应用可观测平台 | **本书主力**。可自托管、支持 trace/prompt 管理/数据集 |
| **Phoenix (Arize)** | LLM 可观测与评测 | 需要更强的分析能力 |
| **OpenTelemetry** | 通用可观测标准 | 要和公司已有的监控体系打通 |
| **Prometheus + Grafana** | 指标采集与看板 | 系统级指标监控 |
| **MLflow** | 实验与模型管理 | 需要模型版本管理和实验追踪 |

### 3.7 数据处理

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **Datasets (HF)** | 数据集加载与处理 | 训练数据流水线的标准工具 |
| **Data-Juicer** | 大规模数据清洗工具箱 | 预训练/大规模 SFT 数据治理 |
| **Label Studio** | 数据标注平台 | **构建金标集时需要多人协作标注** |
| **Argilla** | 面向 LLM 的数据标注与反馈 | 需要把线上 badcase 回流标注 |
| **pandas / polars** | 表格数据处理 | 工单 CSV、备件 Excel 的处理 |

---

## 4. 榜单与评测平台

> **看榜单的第一原则：榜单测的是通用能力，你要的是业务能力，两者相关性远低于直觉。**

### 4.1 主要榜单

| 榜单 | 测什么 | 怎么看 | 局限 |
|---|---|---|---|
| **Chatbot Arena（LMArena）** | 真人盲测投票的 Elo 排名 | **相对最可信的通用能力排名**，因为是人类偏好且题目不公开 | 偏好"讨喜"的回答；中文题量占比有限；不反映你的业务场景 |
| **MMLU / CMMLU / C-Eval** | 多学科知识选择题 | 看模型的知识广度 | **数据污染最严重的一类**。选择题也无法反映生成质量 |
| **MTEB** | 文本 embedding 的多任务表现 | **选 embedding 模型的主要参考**。务必看中文子集 | 榜单任务与你的领域可能差异很大 |
| **BEIR** | 检索模型的零样本跨领域表现 | 判断检索模型的泛化能力 | 以英文为主 |
| **OpenCompass 榜单** | 多维度综合评测 | 中文模型横评的常用参考 | 同样有污染风险 |
| **HumanEval / MBPP** | 代码生成能力 | 选代码模型时看 | 题目简单且已被大量训练数据覆盖 |
| **GSM8K / MATH** | 数学推理能力 | 判断推理能力 | 污染严重；且数学好不代表业务推理好 |

### 4.2 数据污染问题（必须理解）

**什么是数据污染**：评测集的题目（甚至答案）出现在了模型的训练数据里，模型是"背过"而不是"会做"。

**为什么普遍存在**：
1. 公开评测集在互联网上广泛传播，预训练爬虫无差别抓取
2. 评测集一旦公开，就会被有意无意地混进 SFT 数据
3. 部分团队会针对性地"刷榜"

**怎么判断一个分数可不可信**：

```text
□ 这个 benchmark 公开多久了？越久污染越可能严重
□ 模型的训练数据截止时间在 benchmark 发布之前吗？
□ 同一个模型在「公开 benchmark」和「私有/最新 benchmark」上的差距大吗？差距大是危险信号
□ 有没有做过污染检测（n-gram 重叠、困惑度异常）？
□ 这个分数是模型方自报的，还是第三方独立评测的？
```

**给工程师的实用结论**：

> **榜单只用来做初筛（把明显不行的排除掉），最终选型必须用你自己的业务数据评测。**
>
> 本书的做法是：用榜单选出 3~5 个候选 → 用 [08.2](../08-评测体系/02-LLM-Wiki金标集构建工程.md) 的方法构建自己的金标集
> → 用 [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) 的 harness 横评 → 按业务指标决策。
>
> **这套流程跑一遍的成本，远低于选错模型之后返工的成本。**

### 4.3 怎么构建"自己的榜单"

```text
1. 从真实业务日志里抽 200~300 条有代表性的问题（覆盖各类意图和难度）
2. 人工标注标准答案和判分要点（什么必须提到、什么绝对不能说）
3. 固定 prompt 和参数，横评候选模型
4. 同时记录：正确率、拒答率、平均延迟、单次成本、失败模式分布
5. 每次模型升级或方案变更，重跑一次，形成纵向对比
```

**关键点**：这个"私有榜单"**不要公开**，否则它也会被污染。

---
## 5. 中文社区与信息源

大模型领域的中文信息环境有个特点：**信息密度低、噪声高、时效性差**。
本节不推荐具体账号（会变质，也容易变成广告），而是给**可操作的判别方法**。

### 5.1 信息源的四个层级

| 层级 | 类型 | 可信度 | 时效 | 建议投入 |
|---|---|---|---|---|
| L1 | 论文原文、官方文档、官方仓库代码 | **最高** | 最新 | 60% |
| L2 | 框架维护者的博客、issue 讨论、release note | 高 | 最新 | 20% |
| L3 | 一线工程师的实践复盘（有代码、有数字、有踩坑） | 中高 | 滞后 1~3 月 | 15% |
| L4 | 二次加工的科普、"一文读懂"、榜单解读 | 低 | 滞后且失真 | 5% |

**大部分人的时间分配是反过来的**——80% 的时间在刷 L4。这是学习效率低的主要原因。

### 5.2 高质量内容的判别清单

看到一篇技术文章，用这 10 条快速判断要不要读下去。**命中 3 条以上红灯就可以关掉了。**

**绿灯（值得读）：**

```text
✓ 给了可运行的完整代码，不是片段
✓ 写清了版本号（python / 框架 / 模型）
✓ 给了实测环境（什么卡、多少数据、什么配置）
✓ 有"这个方法不适用于什么场景"的说明
✓ 有失败案例，不是只讲成功
✓ 数字有出处，或明确标注"我的环境下"
✓ 有对照组（优化前 vs 优化后，A 方案 vs B 方案）
✓ 承认了局限性
```

**红灯（可以关掉）：**

```text
✗ 通篇没有一行代码，全是概念图
✗ 出现"颠覆""革命""彻底解决"之类的词
✗ 精确百分比没有任何出处（"准确率提升 47.3%"）
✗ 代码里有 `# ...省略` 或明显跑不通
✗ 没有版本号，或者版本号是一年前的却说是"最新"
✗ 文末是课程/社群/咨询的推广
✗ 标题是"一文读懂"但内容是翻译的官方文档
✗ 用"业内人士透露""据说"作为论据
✗ 对比表里自家方案全绿、别家全红
✗ 讲某个技术但明显没用过（把参数名都写错了）
```

### 5.3 怎么避免被营销号带偏

**营销号的三个典型套路和破解方法：**

| 套路 | 表现 | 破解 |
|---|---|---|
| **偷换前提** | "RAG 已死，长上下文取代一切" | 问：成本呢？延迟呢？可溯源呢？**任何"X 取代 Y"的论断都要问代价** |
| **夸大普适性** | "这个 prompt 让准确率提升 3 倍" | 问：什么题集？基线多少？**3 倍从 10% 到 30% 也是 3 倍** |
| **制造焦虑** | "不会 XX 就会被淘汰" | 问：这个技术解决什么真实问题？**解决不了问题的技术不会成为门槛** |

**三个反制习惯：**

1. **看到结论先找前提。** 任何性能数字都依附于前提条件，前提没写清楚的结论直接忽略。
2. **看到新技术先问替代方案。** "这个问题不用新技术怎么解决？"——如果老办法能解决 80%，新技术的价值就有限。
3. **看到"最佳实践"先问上下文。** 100 人的团队和 5 人的团队，最佳实践完全不同。

### 5.4 值得建立的信息渠道（按类型，不推荐具体账号）

| 类型 | 怎么找 | 怎么用 |
|---|---|---|
| **arXiv 订阅** | 按 cs.CL / cs.IR / cs.AI 分类订阅，或用论文推荐工具按关键词订阅 | 每周扫标题，只看和当前问题相关的 |
| **框架的 Release Notes** | GitHub 仓库的 Releases 页面开启订阅 | **升级前必看**，破坏性变更都写在这里 |
| **框架的 GitHub Discussions / Issues** | 直接搜你的报错信息 | **比博客更接近真相**，维护者的回复权威 |
| **开源项目的 examples 目录** | 克隆仓库后直接看 | 常常比文档更实用（LLaMA-Factory 尤其明显） |
| **技术团队的工程博客** | 关注做基础设施的公司的工程博客 | 生产经验密度高 |
| **同行社群** | 行业群、公司内部技术分享 | **最快获得"这个坑我踩过"的信息** |

**一条建议**：**把"提问"也当作信息渠道。** 在框架的 GitHub Discussions 里提一个描述清楚的问题（带最小复现，见附录 A §10.4），
往往能得到比搜索半天更准确的答案。**描述清楚问题的过程本身，也解决了一部分问题。**

---

## 6. 数据集资源

> 本节讲**类型和获取渠道**，不列具体数据集的规模数字（会变），也不做许可证承诺——
> **使用任何数据集前，必须自己核对其许可证是否允许商用。**

### 6.1 数据集的四种类型

| 类型 | 用途 | 特征 | 本书对应 |
|---|---|---|---|
| **预训练语料** | 增量预训练（CPT） | 无标注纯文本，规模极大 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **指令数据** | SFT | (指令, 输入, 输出) 三元组 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **偏好数据** | DPO / RLHF | (指令, 好答案, 坏答案) | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **评测数据** | 评测 | 带标准答案和判分规则 | [08.2](../08-评测体系/02-LLM-Wiki金标集构建工程.md) |

### 6.2 中文数据集的获取渠道

| 渠道 | 特点 | 注意事项 |
|---|---|---|
| **Hugging Face Hub** | 数据集最全，搜索和预览方便 | 国内访问需配镜像端点（见附录 A 的 E09） |
| **ModelScope（魔搭）** | 国产模型和中文数据集丰富，访问稳定 | **国内团队优先** |
| **各高校/实验室的 GitHub 仓库** | 学术数据集的第一手来源 | 留意许可证 |
| **天池 / Kaggle 等竞赛平台** | 有贴近业务的真实数据 | 多数仅限非商用 |
| **政府与行业开放数据平台** | 权威数据 | 格式往往不友好，需要大量清洗 |

**搜索技巧**：在 HF / ModelScope 上按任务类型 + 语言筛选，例如
`task: question-answering` + `language: zh`，比直接搜关键词有效得多。

### 6.3 中文数据集的类型分布

| 类别 | 说明 | 典型用途 |
|---|---|---|
| **通用问答** | 开放域的问答对 | 评测通用能力、SFT 的通用配比部分 |
| **阅读理解** | 给定文段找答案 | **最接近 RAG 场景**，适合训练"基于资料回答"的能力 |
| **通用指令** | 多样化的指令-回答对 | SFT 的基础配比，防止领域微调后通用能力退化 |
| **多轮对话** | 上下文相关的对话 | 训练对话连贯性 |
| **推理与数学** | 需要多步推理的题目 | 评测推理能力 |
| **安全与价值观** | 敏感问题的合规回答 | 安全对齐 |
| **领域专业** | 医疗、法律、金融等垂域 | 参考其构造方式，而不是直接用 |

### 6.4 企业场景的真相：公开数据集帮不上太多忙

**这是本节最重要的一段。**

华成机电的售后场景里，没有任何公开数据集包含「XJ-200 报 E041 怎么处理」这类问题。
公开数据集在企业落地中的真实作用只有两个：

1. **提供通用配比数据**，防止领域微调后模型的通用能力塌陷（经验配比：领域数据 : 通用数据 ≈ 3:1 到 5:1，需按效果调整）
2. **提供格式参考**，看别人的指令数据是怎么组织的

**真正的数据必须自己造。** 本书给出的路径：

```text
业务数据（工单、手册、Wiki）
    ↓ 清洗与结构化          [05.2 数据集构造与清洗]
    ↓ 用 LLM 自动生成候选   [05.2 / 08.2]
    ↓ 人工质检与修正         [08.2 LLM-Wiki 金标集构建]
    ↓ 泄漏检查与配比         [05.2]
训练集 + 金标集
```

**对应章节**：[05.2 数据集构造与清洗](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) 和
[08.2 LLM-Wiki金标集构建工程](../08-评测体系/02-LLM-Wiki金标集构建工程.md)。

### 6.5 数据使用的合规检查清单

```text
□ 许可证允许你的使用方式吗？（研究 / 商用 / 衍生作品）
□ 数据里有 PII 吗？入库前做过脱敏吗？
□ 用业务数据训练，经过数据所有方（客户）的授权了吗？
□ 训练集和评测集之间做过泄漏检查吗？（n-gram 重叠 / 语义相似度）
□ 数据的来源和处理过程有记录吗？（可追溯是合规审计的要求）
□ 用第三方 API 处理敏感数据时，确认过数据不会被用于训练吗？
```

**最后一条尤其重要**：用在线 API 清洗/生成企业敏感数据前，**必须确认服务条款中的数据使用政策**。
很多企业的合规红线就在这里。

---

## 7. 学习路径推荐

三个层次不是线性的阶梯，而是**三种不同的能力定位**。想清楚自己要哪一种，比盲目往下学更重要。

### 7.1 层次一：会用（能把 LLM 接进业务）

**能力定义**：能独立完成一个能跑的 RAG 或 Agent demo，能调通 API，能解释每个参数的作用。

**目标产出**：一个能回答私域问题、有引用来源的问答系统。

| 资源类型 | 具体内容 |
|---|---|
| **本书章节** | [00 前置准备](../00-前置准备/) → [02 RAG 基础](../02-RAG基础篇/) → [06.1](../06-Agent智能体/01-Agent原理与思维链.md)、[06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **必读论文** | RAG、ReAct、Chain-of-Thought（各读摘要 + Figure 1 即可） |
| **官方文档** | LangChain 的 LCEL 部分、Milvus 的 Schema 与 Search、DeepSeek API 文档 |
| **动手项目** | 本书 [02.5 从0手写最小可用RAG](../02-RAG基础篇/05-从0手写一个最小可用RAG.md) |
| **验收标准** | 能说清检索为什么返回这几条、能改切分策略并观察效果变化 |

**常见卡点**：环境装不上（看附录 A §1、§2）、检索效果差（看 [03.1](../03-RAG进阶与性能优化/01-查询理解与高级检索策略.md)）。

**时间投入参考**：每天 2~3 小时，约 1~2 周。

---

### 7.2 层次二：会做（能把系统交付上线）

**能力定义**：能设计架构、能建评测体系、能处理生产问题、能对技术决策给出理由。

**目标产出**：一个通过验收、有监控、有降级、可持续迭代的生产系统。

| 资源类型 | 具体内容 |
|---|---|
| **本书章节** | 层次一 + [03 RAG 进阶](../03-RAG进阶与性能优化/) + [04 LangChain/LangGraph](../04-LangChain与工程框架/) + [08 评测](../08-评测体系/) + [10 工程化](../10-工程化与生产落地/) |
| **必读论文** | DPR、Lost in the Middle、CRAG、Self-RAG、RAGAS、LLM-as-Judge |
| **官方文档** | LangGraph 全部概念、vLLM Engine Arguments 逐参数、Langfuse 自托管 |
| **动手项目** | 本书 [项目 1](../09-实战项目/项目1-DeepSeek-Harness与LLM-Wiki知识库评测实战.md) + [项目 2](../09-实战项目/项目2-RAG与Agent双引擎智能决策问答系统.md) |
| **验收标准** | 能拿出评测报告证明系统好在哪、能画出线上故障的排查路径 |

**这一层的分水岭是评测**。很多人卡在"感觉效果还行但说不清好在哪"，本质是没有评测体系。
**先把 [08 评测体系](../08-评测体系/) 读完，再回头做优化，效率会高得多。**

**时间投入参考**：每天 2~3 小时，约 1~2 个月。

---

### 7.3 层次三：会优化（能在约束下持续提升）

**能力定义**：能定位瓶颈、能量化收益、能在成本/延迟/质量的三角约束下做权衡、能做微调。

**目标产出**：可量化的性能、成本、准确率改进，以及支撑这些改进的方法论。

| 资源类型 | 具体内容 |
|---|---|
| **本书章节** | 层次二 + [03.4 延迟优化](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) + [03.5 准确率优化](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md) + [05 微调](../05-微调LoRA与PEFT/) + [07 多智能体](../07-多智能体协同/) |
| **必读论文** | LoRA、QLoRA、DPO、Self-Instruct、LIMA、FlashAttention、PagedAttention、AWQ |
| **官方文档** | PEFT 概念指南、TRL 的 Trainer、DeepSpeed ZeRO 配置、vLLM 量化与多 adapter |
| **动手项目** | 本书 [项目 3](../09-实战项目/项目3-LangChain与LoRA与Agent三位一体智能应用.md) + [项目 4](../09-实战项目/项目4-RAG性能突围-查询速度提升10倍.md) |
| **验收标准** | 能拿出"基线 → 每项优化的收益/代价 → 最终效果"的完整归因表 |

**这一层的核心是"归因"**：不是做了 11 项优化然后总体快了 10 倍，而是**说得清每一项贡献了多少、代价是什么**。
这也是本书 [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) 反复强调的。

**时间投入参考**：每周 10 小时，约 3 个月。

---

### 7.4 三个层次的对照

| 维度 | 会用 | 会做 | 会优化 |
|---|---|---|---|
| 关心的问题 | "怎么跑起来" | "怎么上线不出事" | "怎么更快更准更便宜" |
| 核心能力 | 调通 | 验收 + 兜底 | 量化 + 归因 |
| 时间尺度 | 天 | 周~月 | 月~季 |
| 最容易缺的 | 环境和基础概念 | **评测体系** | **测量与归因方法** |
| 本书重点章节 | 00、02、06 | 03、04、08、10 | 03.4/3.5、05、07 |

**一个常见误区**：跳过"会做"直接学"会优化"。
**没有评测体系的优化是自欺欺人**——你无法知道自己是不是把系统改坏了。

---

## 8. 本书之后学什么

本书覆盖的是**文本 RAG + Agent 的工程落地**。这是当前企业需求最集中的方向，但不是全部。
以下四个方向按"与本书的衔接紧密程度"排序。

### 8.1 多模态（与本书衔接最紧）

**为什么值得学**：企业的真实数据里有大量图片——设备照片、故障截图、电路图、扫描的纸质单据。
华成机电的场景里，**现场工程师拍一张照片问"这是什么故障"** 是极高频的需求，纯文本 RAG 完全处理不了。

**入门路径**：

```text
1. 理解视觉编码器怎么接进 LLM（图像 → patch → 投影 → 当作 token 送进 LLM）
2. 跑通一个开源 VLM 的推理（Qwen-VL 系列是中文场景的常见选择）
3. 做多模态 RAG：图片向量化 + 图文混合检索
4. 做文档理解：版面分析 + 图表理解（这一条对企业最实用）
```

**关键概念**：Vision Encoder、Visual Token、图文对齐、多模态 embedding、OCR-free 文档理解。

**与本书的衔接点**：
[02.2 文档解析与切分策略](../02-RAG基础篇/02-文档解析与切分策略.md) 里的 PDF 解析问题，
在多模态方案下有完全不同的解法——不再提取文字，而是直接让 VLM 理解整页。

---

### 8.2 强化学习与推理模型

**为什么值得学**：推理模型（会先输出长思维链再回答）在复杂推理任务上有明显优势，
而它们的训练依赖强化学习。理解这条技术路线，才能判断**什么任务值得用推理模型、什么任务用它是浪费**。

**入门路径**：

```text
1. 补强化学习基础概念（策略、价值、奖励、优势函数）
2. 理解 PPO 和它在 RLHF 里的角色
3. 理解 DPO 为什么能绕过奖励模型（本书 05.4 已覆盖）
4. 理解基于可验证奖励的训练思路（数学、代码这类有明确对错的任务）
5. 实践：用推理模型做复杂排障，对比普通模型的效果与成本
```

**关键概念**：Policy、Reward Model、KL 惩罚、Advantage、过程监督 vs 结果监督、测试时计算扩展。

**工程视角的提醒**：推理模型的**输出 token 数常常是普通模型的数倍**，延迟和成本都显著更高。
在做选型时要算清这笔账（方法见 [01.4 算力配置与成本测算](../01-大模型基础与技术选型/04-算力配置与成本测算.md)）。

---

### 8.3 模型压缩与端侧部署

**为什么值得学**：企业私有化部署的算力预算往往很紧，
而"能不能在一张消费级卡上跑起来"经常是项目能否立项的决定性因素。

**入门路径**：

```text
1. 吃透量化（本书 05.5 已覆盖 AWQ/GPTQ/GGUF）
2. 学蒸馏：用大模型的输出训小模型（企业场景常用于把通用能力压缩到专用小模型）
3. 了解剪枝与结构化稀疏
4. 学推理引擎的底层优化：算子融合、图优化、编译（TensorRT-LLM 这类）
5. 端侧：移动端和边缘设备的部署方案
```

**关键概念**：PTQ vs QAT、KV Cache 量化、蒸馏温度、结构化剪枝、算子融合、Continuous Batching 调优。

**与本书的衔接点**：
[05.5 模型合并量化与部署](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) 是这个方向的起点。

---

### 8.4 具身智能与 Agent 的物理世界延伸

**为什么值得学**：Agent 从"调 API"走向"操作真实设备"是一个明确的趋势。
制造业场景里，**从"告诉工程师怎么修"到"直接控制设备做诊断"** 是一个巨大的价值跃迁。

**入门路径**：

```text
1. 理解 Agent 与环境交互的抽象（感知 → 决策 → 执行 → 反馈）
2. 学习机器人领域的基础概念（运动规划、抓取、仿真）
3. 理解视觉-语言-动作模型的思路
4. 从仿真环境开始实践，不要一上来碰真机
```

**关键概念**：Embodied AI、World Model、Sim-to-Real、VLA（Vision-Language-Action）、技能库。

**现实提醒**：这个方向**离企业软件落地还有距离**，硬件门槛也高。
如果你的目标是近期职业发展，前三个方向的回报更直接。

---

### 8.5 一个横向建议：补工程基础

**这是回报率最高但最容易被忽视的方向。**

大模型应用最终是一个**分布式系统 + 数据系统**，而不是一个算法问题。本书反复出现的问题
——连接池、并发、缓存、可观测、优雅停机、幂等——**都不是大模型特有的，是后端工程的经典问题**。

| 补什么 | 为什么 | 对应本书 |
|---|---|---|
| 分布式系统基础 | 多智能体的一致性、重试、幂等本质是分布式问题 | [07.3](../07-多智能体协同/03-通信-任务分解-冲突消解.md) |
| 数据库与索引原理 | 理解 HNSW / 倒排索引 / 缓存的本质 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| 高并发服务设计 | 限流、熔断、降级、连接池 | [10.1](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md) |
| 可观测性工程 | 指标/日志/链路的设计 | [10.2](../10-工程化与生产落地/02-可观测性与链路追踪.md) |
| 统计基础 | 看懂置信区间、显著性、抽样 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |

**一个观察**：能把大模型系统做上生产的人，**通常是工程能力强的人补了大模型知识，而不是反过来**。

---

## 9. 每周信息摄入模板

> **目标**：每周投入 3 小时，做到"不掉队"——不需要什么都知道，但遇到问题时知道该去哪找。
>
> **核心原则**：**广度用扫，深度用挖。** 大部分信息扫过即可，只对与当前工作相关的做深入。

### 9.1 时间分配

| 时段 | 时长 | 做什么 | 目标 |
|---|---|---|---|
| **周一 · 扫更新** | 30 min | 扫框架 release note + 订阅的论文标题 | 知道"发生了什么" |
| **周三 · 挖一篇** | 60 min | 精读一篇与当前工作相关的论文或工程博客 | 解决"当前的问题" |
| **周五 · 动手** | 60 min | 跑通一个新东西（新参数、新工具、新方法） | 变成"自己的能力" |
| **周日 · 复盘** | 30 min | 整理本周笔记，更新自己的报错手册和术语表 | 沉淀，防遗忘 |

### 9.2 周一 · 扫更新（30 分钟）

```text
□ 扫 GitHub 订阅的仓库 Release（10 min）
    重点：langchain / langgraph / vllm / transformers / peft / milvus
    只看两件事：有没有破坏性变更？有没有解决我遇到过的问题？

□ 扫 arXiv 订阅的标题（15 min）
    只读标题，命中关键词才点开摘要
    我的关键词：RAG / retrieval / agent / LoRA / inference / evaluation
    判断标准：这篇论文解决的问题，我遇到过吗？

□ 扫一眼榜单变化（5 min）
    只看有没有新的开源模型值得试，不看具体分数
```

**输出**：一个"待挖清单"，记下 1~3 条值得深入的。

### 9.3 周三 · 挖一篇（60 分钟）

从周一的待挖清单里选 **一篇**（不要贪多），按附录 B §12.4 的顺序精读：

```text
摘要（5 min）→ Figure 1（5 min）→ 局限性（5 min）
→ 消融实验（15 min）→ 方法（20 min）→ 写笔记（10 min）
```

**笔记模板**（三句话，超过三句说明没读懂）：

```markdown
## <论文标题>
- **解决什么问题**：（一句话）
- **核心做法**：（一句话，最好能画成一个箭头流程）
- **对我有什么用**：（一句话。如果答不上来，说明这篇不该读）
- **可以直接抄的**：（哪个具体组件/技巧能用到我的系统里）
```

### 9.4 周五 · 动手（60 分钟）

**规则：必须产出可运行的代码，不能只是看。**

选题来源（按优先级）：

```text
1. 本周工作中遇到但没时间深究的问题   ← 优先级最高
2. 周三读的论文里可以直接抄的组件
3. 框架 release note 里的新特性
4. 本书里还没跑过的章节
```

**记录模板**：

```markdown
## [日期] 试了什么
- **动机**：为什么试这个
- **做法**：关键代码/配置（贴上去，别只写描述）
- **结果**：有效 / 无效 / 部分有效，附上对比数字和实测环境
- **结论**：什么场景下该用、什么场景下不该用
```

**重点**：**无效的尝试同样要记录**。"我试过 X，在 Y 场景下没用，因为 Z"——
这种知识的价值不低于成功案例，而且几乎没人会写在博客里。

### 9.5 周日 · 复盘（30 分钟）

```text
□ 更新 docs/troubleshooting.md（本周解决的非平凡问题，见附录 A §10.8）
□ 更新 docs/glossary.md（本周遇到的新术语）
□ 回顾本周笔记，标记"还没搞懂的"
□ 更新下周的待挖清单
```

### 9.6 关于这个模板的三条说明

1. **不要追求覆盖率。** 这个领域每周的新论文数量远超任何人的阅读能力。
   **正确的心态是"我知道去哪找"，而不是"我什么都知道"。**

2. **允许跳过。** 忙的那一周就只做周一的 30 分钟扫更新。
   **持续 6 个月的低强度输入，远胜于突击一个月然后放弃。**

3. **动手的权重最高。** 如果只能做一件事，做周五的那 60 分钟。
   读 10 篇论文不如把 1 个方法真正跑通——**这个领域的知识几乎全部是实践性知识。**

### 9.7 一个季度的检查点

每三个月问自己四个问题：

```text
1. 这三个月，我解决了哪些三个月前解决不了的问题？
   （答不上来说明在原地打转）

2. 我的 troubleshooting.md 增加了几条？
   （这是最真实的成长度量）

3. 有没有哪个技术，我从"听说过"变成了"用过并知道它的边界"？
   （知道边界比知道用法重要）

4. 我现在能给别人讲清楚哪个技术决策的理由？
   （能讲清楚才算真的懂）
```

**第 4 条是本书从头到尾的立场**：
大模型落地的难点从来不在于调通一个 API，而在于**你能不能证明它做得好、说得清为什么、并且在成本和延迟的约束下持续做好**。

学习资源的价值，最终也体现在这一点上。

---

## 10. 这份清单的保质期与更新节奏

这份清单里，不同部分的保质期差异巨大：

| 内容 | 保质期 | 建议 |
|---|---|---|
| **经典论文**（§1） | 长 | 核心思想多年有效，可放心投入 |
| **判别方法与学习路径**（§5、§7、§9） | 长 | 方法论比具体工具更耐用 |
| **官方文档入口**（§2） | 中 | 入口稳定，具体页面会变，所以本文只给入口 |
| **开源项目**（§3） | 短 | **半年后请重新评估**，新项目层出不穷 |
| **榜单**（§4） | 极短 | 分数每周都在变，但**看榜单的方法**不变 |

**所以本附录的真正价值不在于"列了什么"，而在于"怎么判断"**：
怎么判断一篇论文值不值得读、一个项目适不适合你、一个数字可不可信、一条信息该不该信。

**工具会过时，判断力不会。**


---

## 11. 必读论文补遗（第二批，续 §1）

### 11.0 这一批是什么

§1 给的是"**不读会影响理解框架**"的那批。这一节补的是另外三类：

| 类别 | 特征 | 怎么用 |
|---|---|---|
| **溯源类** | 某个现在天天用的东西的出处（HNSW、BPE、RRF、蒸馏） | **遇到参数调不动时去读**，读了才知道参数在物理上意味着什么 |
| **对照类** | 和 §1 里的经典观点唱反调的工作（"涌现是幻觉吗""模型能自我纠错吗"） | **做技术决策前读**，避免只听到一面之词 |
| **工程类** | 大型系统论文和技术报告（PD 分离、KV 中心架构、模型技术报告） | **做架构设计时读**，它们的"代价"章节最有价值 |

**纪律重申**（和 §1 一致）：本节**只给论文标题和核心贡献，不标作者、年份、会议、引用数**。
需要准确出处时用标题去 arXiv / Semantic Scholar 检索，**以检索结果为准**。
表格里"对应本书"列的链接是本书展开讲该主题的章节。

---

### 11.1 Transformer 与基础模型（续 §1.1）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Sequence to Sequence Learning with Neural Networks** | ★ | 确立 encoder-decoder 的序列到序列范式 | 理解 Transformer 之前的世界长什么样，**知道它替代了什么** | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Neural Machine Translation by Jointly Learning to Align and Translate** | ★ | 首次把注意力机制引入序列模型 | "注意力"这个词的真正出处，**理解它最初是为了解决什么问题** | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Efficient Estimation of Word Representations in Vector Space** | ★★ | 提出 word2vec，用向量表示词义 | **"用向量表示语义"这个思想的起点**，理解 embedding 的本质从这里开始 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Neural Machine Translation of Rare Words with Subword Units** | ★★ | 提出 BPE 子词切分 | **理解为什么中文一个字可能是 1~3 个 token**，这直接关系到你的成本计算 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing** | ★ | 语言无关的分词实现 | 处理多语言和中文分词异常时有用 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Improving Language Understanding by Generative Pre-Training** | ★ | 提出"生成式预训练 + 判别式微调"范式（GPT-1） | 理解 decoder-only 路线的起点 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Language Models are Unsupervised Multitask Learners** | ★★ | 提出 GPT-2，展示无监督预训练模型的多任务能力 | **"不微调也能做任务"这个观念的转折点** | [01.5](../01-大模型基础与技术选型/05-提示工程与结构化输出.md) |
| **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** | ★ | 提出 T5，把所有 NLP 任务统一成文本到文本 | 理解"任务统一化"思想，**它是 prompt 工程能成立的前提** | [01.5](../01-大模型基础与技术选型/05-提示工程与结构化输出.md) |
| **Layer Normalization** | ★ | 提出层归一化 | 训练不稳定时的必备背景知识 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Root Mean Square Layer Normalization** | ★ | 提出 RMSNorm，去掉均值中心化 | **现代开源模型的默认归一化**，读模型结构代码时会遇到 | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **GLU Variants Improve Transformer** | ★ | 系统比较门控线性单元的各种变体 | SwiGLU 的出处，**读模型配置里的 `hidden_act` 时用得上** | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** | ★ | 提出 ALiBi，用线性偏置实现长度外推 | 理解长度外推的另一条路线（与 RoPE 缩放对比） | [01.1](../01-大模型基础与技术选型/01-大模型原理速览.md) |
| **YaRN: Efficient Context Window Extension of Large Language Models** | ★★ | 改进 RoPE 插值策略，低成本扩展上下文窗口 | **想把 8K 模型扩到 32K 时的标准方案**，配 `rope_scaling` 前必读 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Emergent Abilities of Large Language Models** | ★★ | 提出"涌现能力"：某些能力在规模跨过阈值后突然出现 | 理解"为什么小模型怎么调都做不到某些事" | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **Are Emergent Abilities of Large Language Models a Mirage?** | ★★ | 论证部分"涌现"可能来自评测指标的不连续性 | **必须和上一篇一起读**。这是本书"看数字先看口径"立场的最好案例 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **Llama 2: Open Foundation and Fine-Tuned Chat Models** | ★★ | 开源基座 + chat 版本的完整训练与安全对齐报告 | **技术报告的范本**，它的数据配比和安全章节工程价值很高 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **Mistral 7B** | ★ | 小模型 + 滑动窗口注意力 + 分组查询注意力 | 理解"小模型怎么把长上下文做便宜" | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Mixtral of Experts** | ★★ | 开源稀疏 MoE 模型的完整实现细节 | **理解 MoE 的显存与吞吐特性**：总参数大但激活参数小，选型时容易算错这笔账 | [01.4](../01-大模型基础与技术选型/04-算力配置与成本测算.md) |
| **Qwen2 Technical Report / Qwen2.5 Technical Report** | ★★★ | 本书默认本地模型的官方技术报告 | **你要部署的就是它**。词表、上下文长度、chat template、许可证都在里面 | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **DeepSeek-V3 Technical Report** | ★★ | 大规模 MoE 的训练与推理工程实践 | 本书在线模型的技术背景，**它的工程优化章节值得单独读** | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** | ★★★ | 用强化学习激发长思维链推理能力 | **理解推理模型的来源与代价**。本书用它做复杂排障，成本账见对应章节 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **GLM-130B: An Open Bilingual Pre-trained Model** | ★ | 中英双语大模型的训练工程报告 | 国产模型技术谱系的一环，**训练稳定性一节很实用** | [01.2](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md) |
| **Baichuan 2: Open Large-scale Language Models** | ★ | 中文大模型的数据与训练细节 | 想了解中文语料清洗做法时读 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |

> **读技术报告的一个技巧**：模型技术报告里，**"数据"和"局限"两节的信息密度远高于评测表格**。
> 评测表格是给榜单看的，数据和局限是给用它的人看的。

---

### 11.2 RAG：检索侧的经典工作（续 §1.2）

这一组是**"你每天在用但可能不知道出处"**的那批。读它们的收益是：**参数调不动的时候知道该往哪调。**

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs** | ★★★ | 提出 HNSW 图索引 | **`M` 和 `ef_construction` 到底在控制什么，只有读了这篇才真懂**。调向量库参数前必读 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Product Quantization for Nearest Neighbor Search** | ★★ | 提出乘积量化压缩向量 | 理解 IVF_PQ 的精度损失从哪来，**内存吃紧时的决策依据** | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Billion-scale Similarity Search with GPUs** | ★ | Faiss 的 GPU 实现与工程设计 | 数据量到亿级时读，**理解索引的工程边界** | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods** | ★★★ | 提出 RRF 排名融合 | **本书混合检索的默认融合方法**，公式极简但效果极稳，读它只要 10 分钟 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **Unsupervised Dense Information Retrieval with Contrastive Learning** | ★★ | 提出 Contriever，无监督训练稠密检索器 | 理解"没有标注数据也能训检索器"，**冷启动阶段的思路来源** | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval** | ★★ | 用索引动态挖掘难负样本 | **难负样本挖掘的核心方法**，做 embedding 微调时直接用得上 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering** | ★★ | 跨批次负样本 + 去噪 + 数据增强的训练配方 | **中文检索训练最常被参考的工程配方之一** | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking** | ★★ | 学习式稀疏检索：用模型生成带权重的词项 | **理解"稀疏检索不等于 BM25"**，bge-m3 的稀疏模式与此同源 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **Document Expansion by Query Prediction** | ★★ | 给文档预生成可能的查询再入索引 | **一个便宜且有效的召回增强手段**：离线做，在线零成本 | [03.1](../03-RAG进阶与性能优化/01-查询理解与高级检索策略.md) |
| **Passage Re-ranking with BERT** | ★★ | 用 Cross-Encoder 做重排的最简范式 | **重排这件事的出处**，读了才知道 rerank 为什么必须在线算 | [02.4](../02-RAG基础篇/04-检索重排生成全链路.md) |
| **Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents** | ★★ | 用 LLM 做列表式重排 | 理解"用 LLM 当 reranker"的可行性与**延迟代价**，本书给了折中方案 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **Text and Code Embeddings by Contrastive Pre-Training** | ★ | 大规模对比学习训练通用 embedding | 理解通用 embedding 的训练范式 | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Towards General Text Embeddings with Multi-stage Contrastive Learning** | ★★ | 提出 GTE 系列，多阶段对比训练 | **本书备选 embedding 之一的技术背景** | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **Dense X Retrieval: What Retrieval Granularity Should We Use?** | ★★ | 系统比较句子/段落/命题三种检索粒度 | **直接回答"切多大"这个最高频问题**，实验设计值得学 | [02.2](../02-RAG基础篇/02-文档解析与切分策略.md) |
| **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering** | ★★ | 提出 FiD：编码器分别编码多个片段，解码器融合 | 理解"多片段融合"的另一种架构，**和把片段拼进 prompt 的差别在哪** | [02.4](../02-RAG基础篇/04-检索重排生成全链路.md) |
| **Atlas: Few-shot Learning with Retrieval Augmented Language Models** | ★ | 检索器与生成器联合训练的少样本方案 | 了解"联合训练"路线的代价 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **Improving Language Models by Retrieving from Trillions of Tokens** | ★ | 万亿级 token 检索库 + 分块交叉注意力 | 理解"检索规模到极限"会怎样，**工程上难复现但思路有启发** | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **Generalization through Memorization: Nearest Neighbor Language Models** | ★ | 在输出层用最近邻检索插值预测分布 | 理解"检索可以发生在模型内部"这一维度 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **REPLUG: Retrieval-Augmented Black-Box Language Models** | ★★ | 把 LLM 当黑盒，只训检索器去适配它 | **企业最现实的姿势**：模型动不了，那就优化检索器 | [03.2](../03-RAG进阶与性能优化/02-混合检索与重排序精调.md) |
| **In-Context Retrieval-Augmented Language Models** | ★ | 系统研究"仅靠上下文注入"能带来多少提升 | 给"不改模型只改 prompt"这条路线一个量化的上限参考 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |

> **这一组的读法建议**：**HNSW、RRF、Passage Re-ranking with BERT 三篇优先**，它们对应你每天在调的三个旋钮。
> 其余按遇到的问题去挑。

---

### 11.3 RAG：上下文、压缩与综述（续 §1.2）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Retrieval-Augmented Generation for Large Language Models: A Survey** | ★★★ | 系统梳理 Naive / Advanced / Modular RAG 的演进与技术图谱 | **入门 RAG 论文的最佳索引**，先读它再按图去挑具体论文，效率最高 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **Seven Failure Points When Engineering a Retrieval Augmented Generation System** | ★★★ | 从真实项目中总结 RAG 的七类失败点 | **工程师最该读的一篇 RAG 论文**，几乎每一条本书都单独讲过解法 | [03.5](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md) |
| **Searching for Best Practices in Retrieval-Augmented Generation** | ★★ | 对 RAG 各环节做大规模消融，给出组合建议 | **看它的消融表**：哪个环节值得投入、哪个环节收益很小 | [03.5](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md) |
| **The Power of Noise: Redefining Retrieval for RAG Systems** | ★★ | 发现无关文档有时反而提升效果，重新审视"只要精准"的假设 | **反直觉但重要**：说明 top-k 不是越准越好，要实测 | [03.5](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md) |
| **Large Language Models Can Be Easily Distracted by Irrelevant Context** | ★★ | 量化无关上下文对推理的干扰 | **和上一篇一起读**，你会发现"噪声有害还是有益"取决于任务类型 | [03.5](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md) |
| **RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation** | ★★ | 抽取式 + 生成式两种上下文压缩器 | **同时降延迟和降成本的手段**，长上下文场景必看 | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models** | ★★ | 用小模型判断 token 重要性来压缩 prompt | 工程上可直接套用，**注意压缩比过高会伤害引用溯源** | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **Learning to Filter Context for Retrieval-Augmented Generation** | ★ | 训练一个过滤器决定哪些片段该留下 | 和 CRAG 思路互补，**都属于"检索后置质检"** | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity** | ★★ | 按问题复杂度动态选择"不检索 / 单次检索 / 多跳检索" | **成本与效果平衡的关键思路**，简单问题不该走多跳 | [03.3](../03-RAG进阶与性能优化/03-复杂问题拆解与多跳检索.md) |
| **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models** | ★★ | 用图 + 个性化 PageRank 做知识关联检索 | GraphRAG 的轻量替代路线，**构建成本明显更低** | [03.6](../03-RAG进阶与性能优化/06-GraphRAG与结构化知识融合.md) |
| **LightRAG: Simple and Fast Retrieval-Augmented Generation** | ★★ | 双层图检索，兼顾局部与全局且成本可控 | **想要 GraphRAG 的效果但预算不够时先看它** | [03.6](../03-RAG进阶与性能优化/06-GraphRAG与结构化知识融合.md) |

> **本书的立场提醒**：这一组里有互相矛盾的结论（噪声有害 / 噪声有益、压缩有效 / 压缩伤溯源）。
> **这不是学界混乱，而是结论依赖任务类型和数据分布。** 正确的做法是：把它们当成候选假设，**在你自己的金标集上验证**。
> 方法见 [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md)。

---

### 11.4 Agent：工具、记忆与评测（续 §1.3）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **WebGPT: Browser-assisted question-answering with human feedback** | ★ | 让模型学会用浏览器检索并给出引用 | **"带引用回答"这件事的早期系统实践** | [06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **Gorilla: Large Language Model Connected with Massive APIs** | ★★ | 微调模型以准确调用大量 API，并缓解 API 幻觉 | **工具多到几十上百个时必读**，它讲清了"工具选错"的根因 | [06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** | ★★ | 大规模工具调用数据集与训练方法 | 理解工具调用能力怎么被"训"出来，**以及为什么工具描述质量决定一切** | [06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** | ★★ | 工具调用能力的评测基准与失败分类 | **设计自己的工具调用评测时直接借它的分类体系** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **AgentBench: Evaluating LLMs as Agents** | ★★ | 多环境的 Agent 能力评测 | 理解"Agent 能力"该怎么拆开来测，**不要只看最终成功率** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** | ★★ | 用真实仓库 issue 构建端到端任务评测 | **"用真实任务而非人造题目评测"的最佳示范**，本书金标集构建受此影响 | [08.2](../08-评测体系/02-LLM-Wiki金标集构建工程.md) |
| **WebArena: A Realistic Web Environment for Building Autonomous Agents** | ★ | 可复现的真实网页环境 | 了解 Agent 评测环境的工程难度 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **MemGPT: Towards LLMs as Operating Systems** | ★★★ | 借操作系统的分页思想管理有限上下文与外部存储 | **本书记忆系统设计的重要参考**：把上下文当"内存"、外部存储当"磁盘" | [06.3](../06-Agent智能体/03-记忆系统与上下文工程.md) |
| **Self-Refine: Iterative Refinement with Self-Feedback** | ★★ | 让模型自己提意见再自己改 | **最容易实现的质量提升手段**，但要算清 token 成本 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **Chain-of-Verification Reduces Hallucination in Large Language Models** | ★★ | 生成答案后自动生成验证问题并逐条核对 | **幻觉治理里性价比很高的一招**，可直接落到 RAG 的生成后置校验 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Large Language Models Cannot Self-Correct Reasoning Yet** | ★★★ | 论证模型在没有外部反馈时的自我纠错能力有限 | **必须和 Self-Refine / Reflexion 对照读**。结论是：自我纠错要有**外部信号**（检索结果、工具返回、测试用例）才靠得住 | [06.5](../06-Agent智能体/05-手写一个生产级Agent.md) |
| **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** | ★ | 把推理组织成图，允许合并与回溯 | 了解 CoT → ToT → GoT 的这条"结构复杂化"路线**及其成本曲线** | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |
| **CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society** | ★ | 角色扮演式的双智能体协作 | 多智能体"对话式"路线的早期代表 | [07.1](../07-多智能体协同/01-多智能体架构模式全解.md) |
| **Executable Code Actions Elicit Better LLM Agents** | ★★ | 用可执行代码作为动作空间替代 JSON 调用 | **一个重要的工程选择**：动作用代码表达比用 JSON 更紧凑，但沙箱要求更高 | [06.2](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| **The Rise and Potential of Large Language Model Based Agents: A Survey** | ★★ | Agent 领域的系统综述与分类框架 | **入门 Agent 论文的索引**，先读它再按需深挖 | [06.1](../06-Agent智能体/01-Agent原理与思维链.md) |

> **这一组最该读的两篇**：**MemGPT**（记忆怎么设计）和 **Large Language Models Cannot Self-Correct Reasoning Yet**（自我纠错的边界）。
> 前者给你架构，后者防止你对 Agent 抱有不切实际的期待——**而后者在企业项目里省下的返工成本，通常更大**。

---

### 11.5 微调与对齐（续 §1.4）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models** | ★ | 只训 bias 项也能有不错效果 | **PEFT 的下限探索**，理解"到底需要训多少参数" | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **Towards a Unified View of Parameter-Efficient Transfer Learning** | ★★ | 把 Adapter / Prefix / LoRA 统一到同一个数学框架下 | **一篇顶三篇**：读完就不会再混淆 PEFT 家族的各成员 | [05.1](../05-微调LoRA与PEFT/01-微调原理与PEFT家族全解.md) |
| **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** | ★★ | 让不同层自适应分配秩预算 | **"秩该设多少"这个问题的一种自动答案**，显存紧时值得试 | [05.3](../05-微调LoRA与PEFT/03-LoRA-QLoRA实战.md) |
| **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning** | ★★★ | 论证少样本微调在效果和成本上都优于把示例塞进 prompt | **企业最该读的一篇 PEFT 论文**：它回答了"该微调还是该堆 few-shot" | [05.6](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md) |
| **LoRA+: Efficient Low Rank Adaptation of Large Models** | ★ | 给 LoRA 的 A、B 两个矩阵设不同学习率 | 一个改一行配置就能试的小优化 | [05.3](../05-微调LoRA与PEFT/03-LoRA-QLoRA实战.md) |
| **WizardLM: Empowering Large Language Models to Follow Complex Instructions** | ★★★ | 提出 Evol-Instruct：把简单指令自动进化成复杂指令 | **本书数据构造流水线的重要参考**，解决"业务问题都太简单"的数据缺口 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning** | ★★★ | 从复杂度、质量、多样性三个维度自动筛选指令数据 | **和 LIMA 配套读**：LIMA 说少而精有效，这篇说"精"该怎么自动判定 | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning** | ★★★ | 系统量化持续微调中的能力遗忘 | **领域微调前必读**。它是本书"领域数据 : 通用数据要配比"这条建议的依据 | [05.6](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md) |
| **ORPO: Monolithic Preference Optimization without Reference Model** | ★★ | 把 SFT 和偏好优化合成一步，不需要参考模型 | **显存受限时的对齐方案**，比 DPO 又省一个模型副本 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **SimPO: Simple Preference Optimization with a Reference-Free Reward** | ★★ | 用长度归一化的平均对数似然作为隐式奖励 | DPO 的简化改进，**注意它对长度偏置的处理** | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **KTO: Model Alignment as Prospect Theoretic Optimization** | ★★ | 只需"好/坏"单条标注，不需要成对偏好数据 | **企业场景数据形态的救星**：线上点赞点踩就是天然的 KTO 数据 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **Zephyr: Direct Distillation of LM Alignment** | ★ | 用蒸馏数据 + DPO 做低成本对齐的完整配方 | 一个可照抄的小规模对齐流水线 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **Training Verifiers to Solve Math Word Problems** | ★ | 提出 GSM8K，并用验证器给候选答案打分 | **"生成多个候选再验证"这个范式的源头**，对排障场景有直接启发 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **Let's Verify Step by Step** | ★★ | 过程监督（对每一步打分）优于结果监督 | **理解推理模型训练的关键思路**，也解释了为什么"只看最终答案"的评测会漏掉问题 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |

> **本组的实用结论**（本书在 [05.6](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md) 展开）：
> **数据质量 > 数据量 > 方法选择 > 超参**。绝大多数微调翻车是数据问题，不是方法问题。

---

### 11.6 评测与幻觉（续 §1.5）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **BLEU: a Method for Automatic Evaluation of Machine Translation** | ★ | 基于 n-gram 重叠的自动评测指标 | **知道它的局限比知道它的公式重要**：换个说法就扣分，不适合开放问答 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **ROUGE: A Package for Automatic Evaluation of Summaries** | ★ | 面向摘要的召回型重叠指标 | 同上，**现在主要用来做回归监控而不是绝对判分** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **BERTScore: Evaluating Text Generation with BERT** | ★★ | 用上下文向量的相似度替代字面重叠 | 比 BLEU/ROUGE 合理，**但对中文和领域词仍不敏感**，本书用它做辅助信号 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems** | ★★ | 训练轻量判别器做 RAG 自动评测，并给置信区间 | **和 RAGAS 对照读**：一个用 LLM judge，一个训小模型，代价与稳定性不同 | [08.4](../08-评测体系/04-RAGAS与自动化评测流水线.md) |
| **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models** | ★★★ | 多次采样看一致性来检测幻觉，无需外部知识 | **无金标集时的幻觉检测手段**，工程上极容易实现 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation** | ★★★ | 把长答案拆成原子事实逐条核验 | **本书忠实度判分的核心思路**：整段判对错太粗，拆到句子级才可操作 | [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) |
| **Survey of Hallucination in Natural Language Generation** | ★★ | 幻觉的系统分类（内在 / 外在） | **给团队统一"什么算幻觉"的口径**，验收争议往往源于这里没定义清 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models** | ★★ | LLM 幻觉的成因、检测与缓解综述 | 做幻觉治理方案时的索引 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Large Language Models are not Fair Evaluators** | ★★★ | 揭示 LLM 评判的位置偏置并给出缓解方法 | **做 LLM-as-Judge 必读**。本书的 judge 做位置交换正是出于此 | [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) |
| **Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators** | ★★ | 量化并校正自动评测对长回答的偏爱 | **解释了"为什么啰嗦的答案总能赢"**，判分口径设计时要处理 | [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) |
| **Rethinking Benchmark and Contamination for Language Models with Rephrased Samples** | ★★★ | 证明改写后的评测样本能绕过常规污染检测 | **§4.2 数据污染那一节的直接依据**，看榜单前读一遍 | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark** | ★ | 加难度、加干扰项，缓解原版饱和与污染 | 理解"榜单为什么需要不断被重做" | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |

---

### 11.7 推理优化与压缩（续 §1.6）

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Efficiently Scaling Transformer Inference** | ★★ | 系统分析推理的并行切分与延迟/成本帕累托边界 | **做部署拓扑设计时的理论框架**，讲清了"为什么切法决定延迟" | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | ★★ | 把激活的量化难度"迁移"到权重上 | **W8A8 量化的关键思路**，与 AWQ 互为对照 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** | ★★ | 提出分块预填充，把 decode 搭在 prefill 的空档上跑 | **RAG 场景收益最大的一类优化**（prompt 长、prefill 重） | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | ★★ | 无停顿调度：兼顾吞吐与尾延迟 | **理解 vLLM 那些调度参数背后的取舍** | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | ★★ | 把 prefill 和 decode 拆到不同资源池 | **TTFT 和 TPOT 有独立 SLO 时的架构答案** | [10.1](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md) |
| **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** | ★ | 两阶段分离部署的另一套实现与成本分析 | 和 DistServe 对照读，**看它的成本模型** | [10.4](../10-工程化与生产落地/04-成本优化与容量规划.md) |
| **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | ★★ | 以 KV Cache 为中心组织存储与调度 | **长上下文 + 高复用场景的架构参考**，对多轮客服场景有直接启发 | [10.1](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md) |
| **Efficient Streaming Language Models with Attention Sinks** | ★★ | 发现"注意力汇聚点"现象，实现极长流式生成 | **理解为什么随便丢弃开头的 token 会让模型崩掉** | [06.3](../06-Agent智能体/03-记忆系统与上下文工程.md) |
| **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models** | ★★ | 只保留重要 token 的 KV，压缩缓存 | KV Cache 吃不下时的一条路，**注意对长程依赖的影响** | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | ★★ | 加多个解码头一次预测多个 token | 投机解码的"无需草稿模型"变体 | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | ★★ | 在特征层做投机，提高接受率 | **投机解码想真正提速，接受率是关键**，这篇讲清了怎么提 | [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) |
| **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** | ★ | 面向新架构 GPU 的异步与低精度优化 | 知道有这一代即可，**装最新版本就享受到了** | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Reducing Activation Recomputation in Large Transformer Models** | ★★ | 选择性重计算：只重算划算的部分 | **梯度检查点为什么能省显存、代价是多少**，训练调参必读 | [05.4](../05-微调LoRA与PEFT/04-LLaMA-Factory工程化训练.md) |
| **Distilling the Knowledge in a Neural Network** | ★★ | 提出知识蒸馏与软标签 | **蒸馏的源头**，企业想把大模型能力压到小模型上时的第一篇 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter** | ★ | 蒸馏在预训练模型上的经典实践 | 理解蒸馏的实际收益量级 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers** | ★ | 蒸馏注意力分布而非仅输出 | **小型 embedding / rerank 模型的常见来源** | [02.3](../02-RAG基础篇/03-Embedding与向量数据库.md) |
| **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot** | ★★ | 一次性剪枝大模型且精度损失可控 | 剪枝路线的代表作，**读它的"为什么不用重训"** | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **A Simple and Effective Pruning Approach for Large Language Models** | ★★ | 用权重与激活的乘积作为剪枝指标，无需二阶信息 | **极简且好复现**，是剪枝入门的最佳一篇 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |
| **LLM-Pruner: On the Structural Pruning of Large Language Models** | ★ | 结构化剪枝 + 少量数据恢复训练 | 需要真实减小模型体积（而非稀疏掩码）时读 | [05.5](../05-微调LoRA与PEFT/05-模型合并量化与部署.md) |

> **一个对工程最有用的判断**：这一组里，**分块预填充、前缀缓存、PD 分离**三项是**改配置或改架构就能拿到收益**的；
> 而**剪枝、蒸馏**需要重新训练和重新评测，**投入产出比在企业场景通常更差**。优先级应该按这个顺序排。

---

### 11.8 安全、提示注入与数据泄漏（本书 §1 未单列的主题）

这一组对应本书 [10.3 安全合规与幻觉治理](../10-工程化与生产落地/03-安全合规与幻觉治理.md)。
**只要你的 Agent 有写权限或能访问内网系统，这一组就不是"可选阅读"。**

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Ignore Previous Prompt: Attack Techniques For Language Models** | ★★★ | 系统化描述提示注入的攻击手法 | **提示注入这个词的常见出处**，读完你会重新设计 system prompt | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** | ★★★ | 提出间接注入：恶意指令藏在被检索的文档里 | **RAG 系统最该警惕的攻击面**。你的知识库如果允许用户上传文档，必读 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Prompt Injection Attack against LLM-integrated Applications** | ★★ | 对真实应用做注入攻击的实测与分类 | **看它的攻击成功率章节**，能校准你对"加一句防御提示就够了"的乐观 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Jailbroken: How Does LLM Safety Training Fail?** | ★★ | 分析越狱成功的两类根因（能力与目标不匹配） | 理解**为什么单靠对齐训练防不住越狱**，必须有外层防护 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Universal and Transferable Adversarial Attacks on Aligned Language Models** | ★★ | 自动搜索出可跨模型迁移的攻击后缀 | **说明"人工穷举 badcase"不可能覆盖全部风险**，需要自动化红队 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Red Teaming Language Models with Language Models** | ★★ | 用模型自动生成攻击用例 | **可直接落地成 CI 里的安全回归测试** | [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) |
| **Extracting Training Data from Large Language Models** | ★★★ | 证明可从模型中提取出训练数据原文 | **用企业敏感数据微调前必读**。它是"敏感数据不该进训练集"的硬依据 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |
| **Poisoning Web-Scale Training Datasets is Practical** | ★★ | 证明大规模训练数据投毒在现实中可行 | 理解数据来源可信度为什么重要，**对接外部语料时要有审核** | [05.2](../05-微调LoRA与PEFT/02-数据集构造与清洗.md) |
| **Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations** | ★★ | 用专用模型做输入输出安全分类 | **护栏层的可落地方案**，比在主模型 prompt 里写规则更可靠 | [10.3](../10-工程化与生产落地/03-安全合规与幻觉治理.md) |

> **本书最想强调的一条**：RAG 把"被检索的文档"变成了 prompt 的一部分，
> **所以文档上传通道就是攻击通道**。间接注入那一篇如果只读一篇安全论文，就读它。

---

### 11.9 长上下文：它和 RAG 是什么关系

这是被营销号消费最多的话题（"长上下文取代 RAG"），所以单列一组，**用论文而不是口号来判断**。

| 论文 | 优先级 | 核心贡献 | 为什么值得读 | 对应本书 |
|---|---|---|---|---|
| **Retrieval meets Long Context Large Language Models** | ★★★ | 系统对比"长上下文直接塞"与"检索 + 较短上下文" | **回答"要不要放弃 RAG"的关键实证**：两者可叠加，且检索在成本上有明显优势 | [02.1](../02-RAG基础篇/01-RAG原理与整体架构.md) |
| **LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding** | ★★ | 中英双语长上下文评测基准 | **中文长上下文能力的少数可参考基准之一** | [08.1](../08-评测体系/01-大模型与RAG评测方法论.md) |
| **RULER: What's the Real Context Size of Your Long-Context Language Models?** | ★★★ | 用可控难度的合成任务测"有效上下文长度" | **理解"标称 128K，有效可能只有几万"**，是本书辨析"上下文长度 vs 有效长度"的依据 | [06.3](../06-Agent智能体/03-记忆系统与上下文工程.md) |
| **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** | ★ | 非均匀位置插值实现超长窗口扩展 | 了解上下文扩展的技术上限在哪 | [01.3](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md) |
| **Ring Attention with Blockwise Transformers for Near-Infinite Context** | ★ | 跨设备分块传递实现超长序列注意力 | 了解长上下文的算力代价是怎么被分摊的 | [01.4](../01-大模型基础与技术选型/04-算力配置与成本测算.md) |

**读完这一组你应该能回答三个问题**（这也是被问到"RAG 会不会被取代"时的标准答复）：

```text
1. 长上下文能不能替代检索？
   → 在小语料、可一次装入、且不在意成本的场景可以；企业知识库（几十万到上百万片段）不行。

2. 长上下文和 RAG 冲突吗？
   → 不冲突。检索负责"从海量里挑出相关的"，长上下文负责"挑出来之后能装得下"。二者叠加效果最好。

3. 为什么企业更需要 RAG？
   → 三个原因：成本（token 按量计费）、可溯源（必须给出引用）、可更新（改文档不用改模型）。
```

**这三条也是本书 [02.1 RAG 原理与整体架构](../02-RAG基础篇/01-RAG原理与整体架构.md) 一开头就讲清的立场。**

---

### 11.10 补一句：技术报告与工程博客怎么读

论文之外，还有两类材料信息密度很高，但**读法完全不同**：

| 类型 | 典型例子 | 该看什么 | 该警惕什么 |
|---|---|---|---|
| **模型技术报告** | 各家模型的 Technical Report | 数据配比、上下文长度、chat template、许可证、**局限性一节** | 评测表格（有污染与挑选口径的空间） |
| **推理/框架的设计文档** | vLLM、SGLang、Milvus 的 design doc 与 RFC | **为什么这么设计、放弃了什么方案** | 性能数字的测试环境（往往是理想配置） |
| **公司工程博客** | 做基础设施的团队写的实践复盘 | 踩坑、监控指标、容量数据 | 隐含的规模前提（他们的量级可能和你差 100 倍） |
| **框架 Release Notes** | GitHub Releases | 破坏性变更、默认值改动 | **默认值静默改动是最容易踩的坑** |

**一个具体建议**：把你依赖的 5 个核心组件（本书的组合是 vLLM / LangGraph / Milvus / transformers / PEFT）
的 Release Notes 订阅起来，**每次升级前读一遍破坏性变更**。这件事的投入产出比，高于读任何一篇论文。

---

## 12. 开源项目补遗（续 §3）

§3 按"推理 / RAG / Agent / 微调 / 评测 / 可观测 / 数据处理"分了七组。
这一节补两件事：**把文档解析单独拎出来**（它在企业 RAG 里的工作量常常占一半），
以及补齐 §3 没覆盖的四类工具（约束解码、网关与缓存、prompt 管理、检索底层库）。

**口径不变**：只给定位和适用场景，**不给 star 数、不给版本号承诺、不给"最好用"这种判断**。

### 12.1 文档解析与 OCR（企业 RAG 的真实工作量所在）

> **为什么单列一节**：本书 [02.2 文档解析与切分策略](../02-RAG基础篇/02-文档解析与切分策略.md) 反复强调过一个事实——
> **企业 RAG 项目里，文档解析的工作量常常超过检索和生成部分的总和**。
> 华成机电的五类数据源（PDF 手册、Word 指南、Excel 备件表、工单 CSV、Wiki）里，前三类都绕不开解析问题。

| 项目 | 定位 | 什么时候用 | 什么时候别用 |
|---|---|---|---|
| **Unstructured** | 多格式统一解析（PDF/Word/PPT/HTML/邮件） | **格式杂、想先用一个库覆盖 80%** | 对版面精度要求极高的扫描件 |
| **MinerU** | 面向复杂版面的 PDF 解析（多栏、公式、表格） | **扫描件、学术排版、大量图表的手册** | 结构简单的电子版 PDF（太重） |
| **PaddleOCR / PP-Structure** | 中文 OCR 与版面分析 | **中文扫描件、图片里的铭牌与参数表** | 纯文本 PDF（不需要 OCR） |
| **pdfplumber** | 按坐标精确提取 PDF 文本与表格 | 需要**自己写规则**处理固定模板的文档 | 版面不固定的文档 |
| **PyMuPDF** | 高性能 PDF 读写与渲染 | 需要**快**、需要抽图片、需要按页渲染 | 复杂表格结构还原 |
| **python-docx / openpyxl** | Word / Excel 的结构化读写 | **备件表、维修指南这类原生 Office 文件** | 老版本 `.doc` / `.xls`（需先转换） |
| **markitdown** | 把多种格式统一转成 Markdown | 想让所有文档在入库前变成同一种中间格式 | 需要保留精确版面坐标时 |
| **Camelot / Tabula** | PDF 表格抽取 | **表格是主要信息载体时**（参数表、扭矩表） | 无框线的复杂表格（效果不稳） |
| **Tesseract** | 通用开源 OCR 引擎 | 英文为主、需要完全离线 | 中文复杂版面（建议用 PaddleOCR） |
| **VLM 直接读页面** | 用视觉语言模型直接理解整页图像 | **版面极复杂、传统解析全都失败时的兜底** | 大批量文档（成本高，需算账） |

**解析方案的选择顺序**（本书建议，也是 [02.2](../02-RAG基础篇/02-文档解析与切分策略.md) 的落地流程）：

```text
1. 先判断文档是"电子版"还是"扫描版"（有没有文本层）
     ↓ 电子版
2. 用轻量库直接抽（PyMuPDF / python-docx / openpyxl），成本最低
     ↓ 版面复杂、表格丢失
3. 用专业解析（MinerU / Unstructured / Camelot）
     ↓ 扫描版
4. 走 OCR（PaddleOCR），并单独评估 OCR 错误率
     ↓ 以上全部效果不达标、且文档量不大
5. 用 VLM 逐页理解（**必须先算清单页成本 × 页数**）
```

> **一条实践经验**：**解析质量要单独评测，不要混在 RAG 端到端指标里看。**
> 做法是抽 30~50 页有代表性的文档，人工核对解析结果，统计"关键信息丢失率"（表格丢了、参数错了、段落串行了）。
> 这一步的投入通常在一两天，但能避免后面几周在"检索不到"上瞎调参数——
> **因为信息在解析阶段就已经丢了，检索再怎么优化都找不回来。**

### 12.2 结构化输出与约束解码

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **Pydantic** | 数据校验与模型定义 | **本书全程使用**，工具入参、配置、输出 schema 都靠它 |
| **Outlines** | 基于正则/语法的约束解码 | 需要**保证**输出合法 JSON 或符合特定语法时 |
| **XGrammar / lm-format-enforcer** | 推理引擎侧的语法约束 | 与 vLLM 等引擎配合，在服务端强制结构 |
| **Instructor** | 把 LLM 输出直接映射成 Pydantic 对象 | 想少写解析与重试代码时 |
| **jsonschema** | JSON Schema 校验 | 与模型无关的通用校验兜底 |

> **本书立场**（见 [01.5](../01-大模型基础与技术选型/05-提示工程与结构化输出.md)）：
> **能用引擎侧约束解码就别只靠 prompt 求。** prompt 里写"请输出 JSON"是概率性的，约束解码是确定性的。
> 但要注意：约束解码**只保证格式合法，不保证内容正确**。这两件事必须分开测。

### 12.3 网关、缓存与成本控制

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **LiteLLM** | 多厂商 API 统一网关 | **需要在多家模型间路由、降级、统一计费** |
| **One API / New API 类网关** | 自建 API 网关与配额管理 | 需要在公司内部给多个团队分发额度 |
| **GPTCache** | 语义缓存 | **重复问法多的客服场景**（本书场景命中率可观，但必须自己实测） |
| **Redis** | 通用缓存与会话存储 | 精确缓存、会话状态、限流计数 |
| **Nginx / APISIX** | 反向代理与限流 | 服务入口的通用流量治理 |

> **语义缓存的一个坑**：语义缓存会把"XJ-200 报 E041"和"XJ-300 报 E041"判成相似问题。
> **型号/故障码这类关键实体必须参与缓存 key 的精确匹配**，否则会出现"答得很快但答错型号"的严重问题。
> 具体实现见 [03.4](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) 和 [10.4](../10-工程化与生产落地/04-成本优化与容量规划.md)。

### 12.4 Prompt 管理与实验

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **Langfuse Prompt Management** | prompt 版本化与线上下发 | **本书主力**，改 prompt 不用发版 |
| **promptfoo** | prompt 的批量对比评测 | 迭代 prompt 时做 A/B |
| **Jinja2** | 模板引擎 | 复杂 prompt 拼装（条件、循环、片段复用） |
| **Git + 纯文本文件** | 最朴素的 prompt 管理 | **团队小于 5 人时，这就够了** |

> **不要过度工程**：prompt 管理的核心需求只有三条——**能版本化、能回滚、能知道线上正在用哪一版**。
> 满足这三条的最简方案就是好方案。

### 12.5 向量检索底层库

| 项目 | 定位 | 什么时候用 |
|---|---|---|
| **Faiss** | 向量索引库（非服务） | **想搞清楚索引参数到底在做什么**；小规模嵌入式场景 |
| **hnswlib** | HNSW 的轻量实现 | 单机、百万级以下、不想装数据库 |
| **ScaNN** | 高效 ANN 库 | 对召回-延迟曲线有极致要求时 |
| **Annoy** | 静态索引库 | 索引只建一次、只读场景 |
| **Milvus / Qdrant / Chroma / pgvector** | 向量数据库（见 §2.2） | **需要持久化、过滤、多租户、运维能力时** |

> **选择的分界线很清晰**：只要你需要**标量过滤、增量更新、权限隔离、备份恢复**中的任何一项，
> 就该用向量数据库而不是向量索引库。辨析见 [附录 B §9.17](B-术语中英对照表.md#917-向量索引-vs-向量数据库)。

### 12.6 "要不要引入这个开源项目"的判别清单

开源项目的引入成本，往往在引入后三个月才显现。**决定前过一遍这 10 条：**

```text
□ 最近 3 个月有提交吗？（不是看 star，是看 commit）
□ issue 有人回吗？平均多久回？
□ 有没有 release 节奏，还是只有 main 分支？
□ 许可证允许你的用法吗？（**商用场景必查，尤其 AGPL 和自定义协议**）
□ 依赖重不重？会不会和你现有的 torch / transformers 版本冲突？
□ 文档能不能支撑你独立跑通 quickstart？跑不通就是危险信号
□ 有没有中文场景的使用者？（中文分词、中文向量的适配常是暗坑）
□ 核心能力是不是可替换的？（万一它停更，换掉要改多少代码）
□ 它解决的问题，你自己写要多久？**少于 3 天的，倾向自己写**
□ 它有没有把你锁进某个特定的数据格式或部署方式？
```

**本书的通用建议**：**框架层尽量薄，核心链路尽量自己掌握。**
本书之所以在 [02.5](../02-RAG基础篇/05-从0手写一个最小可用RAG.md) 和 [08.3](../08-评测体系/03-DeepSeek-Harness自研评测框架.md) 坚持手写一遍，
原因就是：**能自己写出来的部分，出问题时你能修；只会调 API 的部分，出问题时你只能等。**

---

## 13. 每周信息摄入模板（完整表格版，续 §9）

§9 给了四个时段的动作说明。这一节给的是**可以直接打印贴在显示器边上的表格**，
以及**三种角色的变体**——因为 RAG 工程师、Agent 工程师和微调工程师该关注的更新源并不相同。

### 13.1 一周总表（可直接打印）

| 时段 | 时长 | 看什么 / 做什么 | 具体动作 | 产出物 | 这一格失败的信号 |
|---|---|---|---|---|---|
| **周一 · 扫更新** | 30 min | 框架 Release Notes + arXiv 标题 + 新开源模型 | 只看两件事：**有没有破坏性变更？有没有解决我遇到过的问题？** | 一个 1~3 条的"待挖清单" | 收藏了 20 篇但一条都没标注为什么要读 |
| **周三 · 挖一篇** | 60 min | 从待挖清单里挑**一篇** | 按 [附录 B §12.4](B-术语中英对照表.md#124-读一篇论文的实用顺序) 的顺序：摘要→图1→局限→消融→方法 | 三句话笔记（问题 / 做法 / 对我有什么用） | 读了 3 篇，每篇都只读了摘要 |
| **周五 · 动手** | 60 min | 跑通一个新东西 | **必须产出可运行代码**：新参数、新工具、周三论文里的一个组件 | 一段能跑的代码 + 对比数字（含实测环境） | 只看了教程没敲代码 |
| **周日 · 复盘** | 30 min | 沉淀与整理 | 更新 `docs/troubleshooting.md` 和 `docs/glossary.md`，标记"还没搞懂的" | 两个文档的增量 + 下周待挖清单 | 三周没更新过 troubleshooting |
| **合计** | **3 h/周** | — | **广度用扫，深度用挖，能力用做** | 每周 4 份小产出 | 连续两周一个产出都没有 |

**为什么是这四格**：它们分别对应**知道 → 理解 → 会用 → 记住**这四个环节。
**跳过任何一格，链条就断了**：只扫不挖是浮光掠影，只挖不做是纸上谈兵，只做不复盘是反复踩同一个坑。

### 13.2 三种角色的变体

同样是 3 小时，不同岗位该把注意力放在不同地方。

| 时段 | RAG 工程师 | Agent 工程师 | 微调 / 推理工程师 |
|---|---|---|---|
| **周一扫什么** | 向量库 release、embedding/rerank 新模型、检索类论文标题 | Agent 框架 release、MCP 生态新服务、工具调用类论文 | 推理引擎 release、量化方案、PEFT 新方法 |
| **周三挖什么** | 检索策略、切分粒度、重排、上下文压缩 | 记忆架构、多智能体拓扑、工具设计与失败处理 | 训练稳定性、量化精度、调度与吞吐 |
| **周五做什么** | 换一个 embedding 跑一次金标集横评 | 给一个现有工具加上幂等与 dry-run | 换一组 vLLM 参数压一次基准 |
| **必须盯的指标** | 召回率、Context Precision、引用正确率 | 任务成功率、平均步数、工具错误率 | TTFT / TPOT / 吞吐 / 显存占用 |
| **最容易忽略的** | **解析质量**（信息在入库前就丢了） | **成本**（一次任务几十次调用） | **业务指标**（模型快了但答得更差） |

> **一条跨角色的提醒**：三种角色的"最容易忽略的"那一栏，恰好是对方的日常。
> **所以团队里每个月做一次交叉复盘，收益往往高于各自多读几篇论文。**

### 13.3 月度与季度节奏

周维度解决"不掉队"，但**能力跃升发生在月和季的尺度上**。

| 周期 | 动作 | 时长 | 产出 |
|---|---|---|---|
| **每月一次** | 把本月的"还没搞懂的"清单挑一条彻底搞懂（读源码 / 做实验） | 半天 | 一篇能讲给同事听的内部分享 |
| **每月一次** | 重跑一遍金标集，对比上月，确认没有静默回归 | 1 小时 | 一张纵向对比表 |
| **每季一次** | 重新评估一次技术选型（模型、向量库、框架是否还是最优） | 一天 | 一份选型复核结论（**维持原状也是结论**） |
| **每季一次** | 回答 §9.7 的四个检查点问题 | 半小时 | 一段自评 |
| **每半年一次** | 重读本附录 §3 和 §12 的开源项目清单，标记"已过时 / 新候选" | 2 小时 | 更新后的技术雷达 |

### 13.4 这个模板什么时候该改

模板是给人用的，不是给人守的。**出现下面任一情况，就该改模板而不是硬撑：**

| 信号 | 说明 | 建议调整 |
|---|---|---|
| 连续两周一个产出都没有 | 强度超出当前生活负荷 | **只保留周一的 30 分钟**，其余暂停，等忙完再恢复 |
| 待挖清单越堆越长 | 扫的门槛太低 | 提高筛选标准：**只留"我这个月真会遇到"的** |
| 周五总在跑教程 | 选题脱离实际工作 | 强制规则：**选题必须来自本周工作中的真实卡点** |
| 笔记从来没被回看过 | 笔记写成了摘要而不是结论 | 改成三句话模板，**第三句"对我有什么用"必须填** |
| 已经能带新人了 | 你的瓶颈变了 | 把周五的动手换成**给团队做分享或写内部文档**——输出是更高阶的学习 |

---

## 14. 致读者

写到这里，这本书能给的东西差不多都给完了：一套可复现的代码、一条从 POC 到上线的路径、
一堆踩过的坑，以及这份清单。

但最后想留给你的，不是这些。

**是一种对数字的态度。**

这本书从第一章起就在做一件看起来很"不解风情"的事：
凡是遇到性能数字，就一定要追问"实测环境是什么""基线是多少""前提成立吗"。
海报上那些「查询速度提升 10 倍」「性能提升 300%」「99.9% 准确率」，本书没有回避，
而是老老实实把它们拆成了**基线 → 每一项优化的收益与代价 → 最终实测**——
因为这些数字确实能达到，但**只在特定前提下达到**，而那些前提，恰恰是最容易被省略的部分。

省略前提的数字是口号；带着前提的数字才是结论。**能分辨这两者，就是这本书最想给你的东西。**

同样的态度也适用于这份清单本身。这里列的论文会被后续工作超越，开源项目会停更，
榜单会被刷爆，官方文档会改版。**半年后重读这份清单，你一定会发现过时的条目。**
那时候真正还有用的，不是某个项目名，而是 §12.6 那 10 条判别清单、§5.2 的红绿灯、§4.2 判断分数可信度的那几个问题——
**是"怎么判断"，不是"知道什么"。**

关于持续学习，还想说三件很实在的事：

**第一，不要用追新代替理解。**
这个领域每周都有新东西，永远追不完，而且大部分半年后就没人提了。
真正让你在三年后还有竞争力的，是 Transformer 怎么算、检索为什么会漏、显存怎么估、评测口径怎么定这些**不怎么变的东西**。
新方法层出不穷，但它们几乎都是在这些老问题上做权衡——**理解了权衡，看新方法只需要十分钟。**

**第二，动手的权重永远最高。**
读十篇论文的收益，不如把一个方法真正跑通、跑崩、再修好。
这个领域的知识几乎全是实践性知识：`gpu_memory_utilization` 调到多少会 OOM、
中文切片多长召回最好、judge 的位置偏置有多严重——**这些都写在别人的博客里，但只有你自己跑过才是你的。**

**第三，允许自己慢。**
本书有五十多个文件、几十万字，不是为了让你一个月读完的。
按 §7 的三个层次走，"会用"一两周，"会做"一两个月，"会优化"三个月——**这是正常速度**。
持续六个月的低强度输入，远胜于突击一个月然后放弃。

最后回到本书从头到尾的那句话：

> **大模型落地的难点，从来不在于调通一个 API，
> 而在于你能不能证明它做得好、说得清为什么、并且在成本和延迟的约束下持续做好。**

模型会换代，框架会重写，这份清单会过时。
**但"拿证据说话"这件事不会过时。**

祝你在自己的项目里，能拿出一张让所有人信服的**基线 → 优化 → 实测**对比表。

那时候，你就不需要这份清单了。

---

**上一章** [附录 B-术语中英对照表](B-术语中英对照表.md) | **返回** [本书目录](../README.md)
