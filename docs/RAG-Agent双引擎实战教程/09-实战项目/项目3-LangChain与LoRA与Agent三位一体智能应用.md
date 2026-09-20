# 实战项目 3：LangChain + LoRA + Agent —— 构建三位一体的工单智能处理助手

> **项目目标**：把「华成机电」每天涌进来的 115 张售后工单，从「人工读、人工分、人工派」变成「模型读、状态机分、Agent 派」，并且**成本、延迟、准确率三项同时可控**。
>
> **你将产出什么**：
> 1. 一份 8000 条的工单结构化抽取 SFT 数据集（含构造脚本、清洗脚本、统计报告）
> 2. 一个 QLoRA 微调出来的 `Qwen2.5-7B-Instruct` 抽取模型（LoRA adapter + 合并权重 + AWQ 量化版本）
> 3. 一条 vLLM 上的 OpenAI 兼容端点，专门跑抽取任务
> 4. 一张用 LangGraph 编排的主流程状态图（抽取 → 路由 → 检索 → 生成 → 行动 → 输出）
> 5. 一个带人工确认节点的 Agent 子图（查库存 / 判保修 / 建子工单 / 派单）
> 6. 一个 FastAPI 服务 + 一个单文件前端演示页
> 7. 一份「微调前 vs 微调后」「全大模型 vs 三位一体」的评测报告与成本核算表
>
> **前置章节**（强烈建议先读完，本项目直接复用其中的代码与结论）：
> - [4.1 LangChain 核心抽象与 LCEL](../04-LangChain与工程框架/01-LangChain核心抽象与LCEL.md)
> - [4.2 LangGraph 状态机编排](../04-LangChain与工程框架/02-LangGraph状态机编排.md)
> - [5.2 数据集构造与清洗](../05-微调LoRA与PEFT/02-数据集构造与清洗.md)
> - [5.3 LoRA-QLoRA 实战](../05-微调LoRA与PEFT/03-LoRA-QLoRA实战.md)
> - [5.5 模型合并量化与部署](../05-微调LoRA与PEFT/05-模型合并量化与部署.md)
> - [5.6 微调效果评估与何时不该微调](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md)（**本项目第 1.3 节会逐条对照它的红灯清单**）
> - [6.2 Function Calling 与工具设计](../06-Agent智能体/02-Function-Calling与工具设计.md)
> - [6.5 手写一个生产级 Agent](../06-Agent智能体/05-手写一个生产级Agent.md)
> - [项目 1 知识库评测实战](./项目1-DeepSeek-Harness与LLM-Wiki知识库评测实战.md)（本项目第六节直接复用它的 harness）
>
> **预计工时**：
>
> | 阶段 | 内容 | 纯动手时间 | 机器跑的时间（可并行做别的） |
> |---|---|---|---|
> | Part A | 数据构造 + 清洗 + 训练 + 量化部署 | 6~8 小时 | 训练 3~5 小时 / 量化 40 分钟 |
> | Part B | LCEL 抽取链 + LangGraph 主流程 | 4~5 小时 | — |
> | Part C | Agent 子图 + 路由 + 服务化 | 4~6 小时 | — |
> | 评测与复盘 | 评测脚本 + 报告 | 2~3 小时 | 评测跑 20 分钟 |
> | **合计** | | **16~22 小时** | **约 6 小时机器时间** |
>
> **硬件要求**：
>
> | 档位 | 配置 | 能做到哪一步 |
> |---|---|---|
> | 完整档 | 单卡 24GB（RTX 4090 / A10 / L20）+ 64GB 内存 + 200GB 磁盘 | 全流程：QLoRA 训练 7B、合并、AWQ 量化、vLLM 部署 |
> | 训练档（云） | 按小时租 A100 40G / 4090，约 3~6 元/小时，训练 4 小时 | 只租训练那几个小时，其余在本机做 |
> | 小模型档 | 单卡 12GB（RTX 3060 12G / 4070） | 把基座换成 `Qwen2.5-1.5B-Instruct`，全流程能跑通，字段准确率会低 3~6 个百分点 |
> | 无 GPU 档 | 纯 CPU / 只有 API | **跳过 Part A 的训练**，用第 3.5 节给的「无 GPU 替代路径」：抽取任务改用在线小模型 + few-shot + 严格 schema，Part B/C 完全不受影响 |
>
> 本项目所有性能与准确率数字均标注了**实测环境**，且都是**示例性数据**，用于演示方法论与量级。你必须在自己的数据与硬件上重新测一遍，不要直接引用本书的数字写进汇报材料。

---

## 〇、先把「三位一体」讲清楚

海报上写着「LangChain + LoRA + AGENT：构建三位一体智能应用」。很多人看到这句话的第一反应是：这是不是把三个热词硬凑在一起？

不是。这三样东西解决的是**三个正交的问题**，任何一个缺席，系统都会以一种特定的、可预测的方式失败。

### 0.1 三者各自解决什么

| 维度 | LangChain / LangGraph | LoRA 微调 | Agent |
|---|---|---|---|
| 一句话定位 | **编排骨架**：把步骤连成可控的流程 | **领域能力内化**：把重复的、格式固定的能力烧进模型权重 | **决策与行动**：在运行时决定做什么、调用什么 |
| 解决的问题 | 流程怎么串、状态怎么传、失败怎么降级、中断怎么恢复 | 提示词写不稳的格式、通用模型不懂的行话、调用太贵太慢 | 需求不确定、步骤不固定、需要读写外部系统 |
| 工作时机 | 编译期（你写死的图） | 训练期（离线一次性投入） | 运行时（每次都重新决策） |
| 可控性 | **最高**：图是确定的，路径可枚举 | 高：输出分布收敛，但仍是概率模型 | **最低**：LLM 自主决策，路径不可枚举 |
| 成本特征 | 几乎零边际成本 | 一次性训练成本高，单次推理成本**最低** | 单次推理成本**最高**（多轮 + 长上下文） |
| 出问题时的表现 | 流程卡住 / 状态错乱（易定位） | 抽错字段（可用评测集量化） | 死循环 / 调错工具 / 越权操作（最难查） |
| 典型延迟 | ~0 ms（编排本身） | 200~600 ms（7B 抽取任务） | 3~30 s（多轮工具调用） |

### 0.2 为什么缺一不可

用华成机电的真实场景推演三种「只做两样」的结局：

```mermaid
flowchart TB
    subgraph C1["缺 LoRA：只有 LangChain + Agent"]
        A1["每张工单的结构化抽取<br/>都用 72B 大模型 + 长 few-shot prompt"]
        A2["结果：单张工单抽取成本 0.02 元<br/>日均 115 张 → 一年 840 元<br/>看着不贵"]
        A3["但：格式合规率 92%，<br/>剩下 8% 要人工兜底<br/>延迟 2.4 秒，高峰期排队"]
        A4["真正致命的是：<br/>prompt 一改，全量行为漂移<br/>没法回归"]
        A1 --> A2 --> A3 --> A4
    end
    subgraph C2["缺 Agent：只有 LangChain + LoRA"]
        B1["抽取又快又准<br/>分类也对"]
        B2["但工单说『附近仓库还有<br/>CJ20-40 吗，今天能调过来吗』"]
        B3["微调模型不会查库存<br/>状态图里没写这条路径"]
        B4["结果：回退人工<br/>自动化率卡在 50%"]
        B1 --> B2 --> B3 --> B4
    end
    subgraph C3["缺 LangChain：只有 LoRA + Agent"]
        D1["一个 Agent 包打天下<br/>抽取也让它做、派单也让它做"]
        D2["每张工单平均 6.3 轮 LLM 调用<br/>延迟 11 秒"]
        D3["出错时不知道错在哪一步<br/>没有状态快照，没法重放"]
        D4["加一个业务规则<br/>= 改 prompt = 全链路回归"]
        D1 --> D2 --> D3 --> D4
    end
```

反过来说，三者**互补的机制**是这样的：

```mermaid
flowchart LR
    subgraph L1["LangChain / LangGraph：确定性骨架"]
        S1[节点1 抽取] --> S2{节点2 路由}
        S2 --> S3[节点3 检索]
        S3 --> S4[节点4 生成]
        S4 --> S5{节点5 是否需要行动}
    end
    subgraph L2["LoRA：把高频固定任务压到最低成本"]
        F1["工单抽取模型<br/>Qwen2.5-7B + LoRA<br/>0.4s / 0.0004 元"]
    end
    subgraph L3["Agent：把不确定的部分交出去"]
        G1["ReAct 子图<br/>查库存→判保修→建子工单→派单<br/>带人工确认"]
    end
    S1 -.调用.-> F1
    S5 -.需要行动时移交.-> G1
    G1 -.结果写回 state.-> S1
    style F1 fill:#e1f5d1
    style G1 fill:#ffe0cc
```

一句话总结三者关系：

> **LangGraph 决定「流程长什么样」，LoRA 决定「每一步多便宜多稳」，Agent 决定「图里画不出来的那部分怎么办」。**

### 0.3 一个反直觉的结论：Agent 用得越少，系统越好

工程上一个常见误区是「Agent 越自主越先进」。在企业落地里恰恰相反：

- 能用确定性代码写的，不要用 LangGraph 节点；
- 能用 LangGraph 节点写死的，不要用 Agent；
- 能用微调小模型做的，不要用大模型；
- 必须用 Agent 的，就把它的工具集、循环次数、权限边界锁死。

本项目里，一张典型工单的处理路径中：**约 78% 的步骤是确定性代码或 LangGraph 节点，约 18% 是微调模型调用，只有约 4% 的工单最终进入 Agent 子图**（实测环境：本项目 600 条回放工单，示例性数据）。这个比例就是「工程味」的量化体现。

---

## 一、需求与验收标准

### 1.1 业务背景

华成机电的售后工单进来有三个渠道：400 电话（客服代填）、微信小程序（客户自填）、经销商系统对接（半结构化）。三个渠道的文本质量天差地别：

```text
【400 电话，客服代填】
客户反映电机运行20分钟后报E07，外壳烫手，风扇在转。型号XC-200-4P，三号线。
客户说上周刚做过保养。要求今天上门。

【微信小程序，客户自填】
机器坏了 转不动 有异响 急！！！ 昨天买的还在保修吧

【经销商系统对接】
DEV_MODEL=JS-75-B|ERR=E12|LINE=A2|DESC=减速机漏油,油位低于下限|URGENCY=2
```

现在的处理流程是：客服人工读完 → 手工填 7 个字段 → 按经验判断派给哪个工种 → 打电话问仓库有没有备件 → 建派工单。**平均每张 4 分 20 秒**，高峰期积压。

### 1.2 项目要做什么

做一个「工单智能处理助手」，具备四项能力：

1. **结构化抽取 + 意图分类**（用微调后的 7B 模型）：把上面三种脏文本统一抽成 7 个字段的 JSON，并判定工单意图（咨询 / 报修 / 投诉 / 备件申请 / 催办）。
2. **全流程编排**（用 LangGraph）：抽取 → 分类路由 → 知识检索 → 方案生成 → 行动 → 输出，每一步有状态快照、可重放、可降级。
3. **复杂处理交给 Agent**：需要查库存、判保修、建子工单、派单的工单，移交 ReAct 子图处理，高风险动作前插人工确认。
4. **全链路可观测**：每张工单的每一步都有 trace，能回答「这张工单为什么派给了电气工种」「这次抽取花了多少钱」。

### 1.3 为什么这个场景值得微调：逐条对照 5.6 章的红灯清单

[第 5.6 章](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md) 给了一张「何时不该微调」的红灯清单。微调是有成本的（数据、算力、运维、版本管理），所以**默认答案应该是"不微调"，除非能逐条通过这张清单**。下面对本场景逐条论证。

| # | 红灯条件（满足 = 不该微调） | 本场景 | 判定 | 论证 |
|---|---|---|---|---|
| 1 | 任务需要**外部知识**，知识还会变 | 抽取任务只看工单文本本身，不需要任何外部知识；型号表/备件表变了也不影响抽取 | 🟢 绿灯 | 变化的知识全部留在 RAG 和工具里，微调模型只学「怎么读工单」这件不变的事 |
| 2 | 样本量 < 500 条，或标注成本极高 | 28 万条历史工单，其中约 9.4 万条有客服手工填过的结构化字段，可直接当标注 | 🟢 绿灯 | 标注成本≈0，这是本场景最大的幸运点 |
| 3 | 任务开放、输出自由 | 输出是固定 7 字段 JSON，schema 冻结 | 🟢 绿灯 | 格式固定正是 LoRA 最擅长的 |
| 4 | 调用量小（日均 < 1000 次） | 日均 115 张工单 × 每张 1~2 次抽取 ≈ 200 次/天；**但**要做历史数据回填：28 万条全量跑一遍 | 🟢 绿灯 | 一次性回填 28 万次，用大模型 API 按 0.02 元/次算是 5600 元；微调后本地跑接近零边际成本 |
| 5 | 延迟不敏感 | 客服在电话里等，要求抽取 < 1 秒 | 🟢 绿灯 | 72B 云端 API + 2000 token few-shot prompt 实测 2.1~3.4 秒，超标 |
| 6 | prompt 工程还没做到位 | 已做到位：结构化输出 + few-shot + JSON mode，格式合规率卡在 91.8% 上不去（实测环境：deepseek-chat，200 条样本，示例性数据） | 🟢 绿灯 | **这一条是关键**：必须先把 prompt 做到天花板再谈微调，否则你微调的是自己的懒惰 |
| 7 | 团队没有模型运维能力 | 本项目会教你把 vLLM + 版本化 adapter 跑起来，并给降级链 | 🟡 黄灯 | 这是唯一需要额外投入的一项，用「降级到大模型」兜底把风险降下来 |
| 8 | 需要模型有推理/创造能力 | 抽取不需要推理；**方案生成需要**，所以方案生成不微调，继续用大模型 | 🟢 绿灯 | 这正是「三位一体」的分工：该微调的微调，该用大模型的用大模型 |

**结论**：8 条里 7 条绿灯、1 条黄灯（有缓解方案）。这个场景亮绿灯。

反例提醒：同一个系统里的「维修方案生成」任务**不该微调**——它需要引用最新手册（知识会变，红灯 1）、输出是开放文本（红灯 3）、需要推理（红灯 8）。所以本项目里方案生成走 RAG + 大模型，一行训练数据都不给它造。

### 1.4 验收标准 checklist

项目做完，请逐条打勾。**没有验收标准的项目等于没做完。**

| # | 类别 | 验收项 | 目标值 | 怎么验 |
|---|---|---|---|---|
| 1 | 数据 | SFT 训练集条数 | ≥ 6000 条（train）+ 800（dev）+ 800（test） | `wc -l data/sft/*.jsonl` |
| 2 | 数据 | 数据清洗报告产出，含长度分布、字段缺失率、去重率、标签分布 | 报告文件存在且各项统计完整 | `cat reports/dataset_stats.md` |
| 3 | 数据 | 训练/验证/测试集**无泄漏**（同一 ticket_id 不跨集） | 交集为 0 | `python scripts/check_leakage.py` |
| 4 | 训练 | QLoRA 训练完成，产出 adapter | `adapter_model.safetensors` 存在 | `ls outputs/ticket-extract-lora/` |
| 5 | 训练 | eval_loss 收敛且无过拟合（best epoch 不在最后一轮的情况有记录） | 曲线图 + 结论 | `reports/training_curve.md` |
| 6 | 效果 | 抽取 **JSON 格式合规率** | ≥ 99.0%（测试集 800 条） | `python eval/eval_extract.py` |
| 7 | 效果 | **字段级平均准确率** | ≥ 92%，且 `设备型号`/`故障码` 两个关键字段 ≥ 96% | 同上 |
| 8 | 效果 | **意图分类 macro-F1** | ≥ 0.90 | 同上 |
| 9 | 效果 | 微调后相比微调前（同基座 + few-shot）有明确提升，且有对照表 | 提升 ≥ 8 个百分点（字段平均准确率） | `reports/before_after.md` |
| 10 | 性能 | 抽取 P95 延迟（vLLM，单请求） | ≤ 800 ms | `python bench/bench_extract.py` |
| 11 | 成本 | 单张工单端到端 LLM 成本 | ≤ 全大模型方案的 40% | `reports/cost.md` |
| 12 | 编排 | LangGraph 主图能导出 PNG/Mermaid，节点数 ≥ 8 | 图文件存在 | `python -m app.graph_export` |
| 13 | 编排 | 抽取失败自动降级到大模型，降级链有日志且可验证 | 人为下线 vLLM 后系统仍可用 | 演示脚本 `demo/demo_fallback.sh` |
| 14 | Agent | Agent 子图工具 ≥ 4 个，且**派单工具前必须有人工确认中断** | 中断可恢复（checkpointer） | `demo/demo_agent_interrupt.py` |
| 15 | Agent | Agent 最大迭代数有硬上限，超限走兜底 | 构造死循环用例验证 | `tests/test_agent_limit.py` |
| 16 | 服务 | FastAPI 提供 `/ticket/process`（同步）与 `/ticket/stream`（SSE） | 两个接口都能跑 | `curl` 示例 |
| 17 | 可观测 | 每张工单产出一条 trace，含每步耗时、token、模型名 | Langfuse 里能看到 | 截图或 `traces/*.json` |
| 18 | 部署 | `docker compose up` 一键起全栈 | 健康检查全绿 | `bash deploy/healthcheck.sh` |
| 19 | 稳定 | 600 条回放工单跑完无未捕获异常 | 错误率 = 0 | `python tests/replay.py` |
| 20 | 文档 | 产出复盘报告，含收益量化 + 踩坑表 | 文件存在 | `reports/retro.md` |

> 注意第 9 条的写法：**「相比微调前有提升」比「达到某个绝对值」更重要**。绝对值受数据难度影响，对照组才能证明是你的工作起了作用。

---

## 二、整体架构与技术选型

### 2.1 三层协同架构图

下面这张图是本项目的全景。请特别注意颜色：**绿色 = 微调小模型**、**蓝色 = 通用大模型**、**橙色 = Agent 自主决策**、**灰色 = 确定性代码**。

```mermaid
flowchart TB
    IN([工单进入<br/>400电话 / 小程序 / 经销商系统]) --> PRE

    subgraph ORCH["LangGraph 主图（编排层 · 确定性）"]
        direction TB
        PRE["node: preprocess<br/>脱敏 / 归一化 / 渠道标注<br/>纯 Python"]
        EXT["node: extract<br/>结构化抽取 + 意图分类"]
        VAL["node: validate<br/>pydantic 校验 + 业务规则"]
        RT{"node: route<br/>按意图 + 紧急度分流"}
        RAG["node: retrieve<br/>混合检索手册/历史工单"]
        GEN["node: generate<br/>维修方案生成"]
        NEED{"node: need_action?<br/>规则判定"}
        AGT["node: agent_subgraph<br/>移交决策引擎"]
        OUT["node: finalize<br/>组装输出 + 引用 + 置信度"]
        PRE --> EXT --> VAL --> RT
        RT -->|咨询类| RAG
        RT -->|报修类| RAG
        RT -->|备件申请| NEED
        RT -->|催办/投诉| OUT
        RAG --> GEN --> NEED
        NEED -->|否| OUT
        NEED -->|是| AGT --> OUT
    end

    subgraph MODELS["模型层"]
        SM["【微调】Qwen2.5-7B-Instruct<br/>+ ticket-extract LoRA<br/>AWQ 4bit · vLLM<br/>0.4s / 约 0.0004 元"]
        BM["【通用大模型】deepseek-chat<br/>方案生成 / Agent 推理<br/>1.5~4s / 约 0.004 元"]
        EMB["bge-m3 本地<br/>向量化"]
        RRK["bge-reranker-v2-m3<br/>重排"]
    end

    subgraph AGENTZONE["Agent 子图（决策层 · 非确定性）"]
        direction TB
        RE["ReAct 循环<br/>max_iter = 6"]
        T1["tool: check_stock<br/>查备件库存"]
        T2["tool: check_warranty<br/>判保修"]
        T3["tool: create_subticket<br/>建子工单"]
        T4["tool: dispatch<br/>派单（高风险）"]
        HIL["human_confirm<br/>中断等待确认"]
        RE --> T1
        RE --> T2
        RE --> T3
        RE --> HIL --> T4
    end

    subgraph DATA["数据与外部系统"]
        VDB[(Milvus 2.4<br/>手册 + 工单向量)]
        ES[(BM25 / ES<br/>型号与故障码精确匹配)]
        ERP[(ERP 库存)]
        CRM[(CRM 保修)]
        WO[(工单系统)]
    end

    EXT -.调用.-> SM
    VAL -.格式失败时降级.-> BM
    GEN -.调用.-> BM
    RE -.调用.-> BM
    RAG --> VDB
    RAG --> ES
    RAG -.-> EMB
    RAG -.-> RRK
    T1 --> ERP
    T2 --> CRM
    T3 --> WO
    T4 --> WO
    AGT -.-> RE

    OUT --> RES([返回：结构化字段 + 建议方案 + 已执行动作])
    OUT --> OBS[(Langfuse Trace<br/>每步耗时/token/成本)]

    style SM fill:#d4f5d4,stroke:#2e7d32
    style BM fill:#d4e6f5,stroke:#1565c0
    style RE fill:#ffe0cc,stroke:#e65100
    style T4 fill:#ffcccc,stroke:#c62828
    style HIL fill:#fff3cd,stroke:#f9a825
    style PRE fill:#eeeeee
    style VAL fill:#eeeeee
    style OUT fill:#eeeeee
```

### 2.2 三位一体在这张图上的落点

| 技术 | 在图上的位置 | 具体节点 | 为什么放在这里 |
|---|---|---|---|
| **LangGraph** | 整个 ORCH 子图 | preprocess / extract / validate / route / retrieve / generate / need_action / finalize | 流程固定、状态需要持久化、要能从任意节点重放 |
| **LoRA 微调模型** | EXT 节点 | extract | 高频、格式固定、延迟敏感、成本敏感、不需外部知识 |
| **通用大模型** | GEN 节点 + Agent 推理 + 降级兜底 | generate / agent / fallback | 需要推理、需要引用外部知识、需要泛化到没见过的场景 |
| **Agent** | AGENTZONE 子图 | 仅在 `need_action == true` 时进入 | 步骤数不确定、要读写外部系统、需要人工确认 |

### 2.3 技术选型与理由

| 组件 | 选型 | 版本 | 备选 | 选它的理由 | 不选备选的理由 |
|---|---|---|---|---|---|
| 编排框架 | LangGraph | 0.2.x | LangChain AgentExecutor / 自写状态机 | 有显式状态、有 checkpointer（人工确认必需）、支持子图、可导出图 | AgentExecutor 无法表达条件分支与中断恢复；自写状态机要自己实现持久化与重放 |
| 链式封装 | LCEL (langchain-core) | 0.3.x | 裸 SDK 调用 | `with_retry` / `with_fallbacks` / `batch` 开箱即用，降级链一行写完 | 裸调用要自己写重试与降级，且没有统一的 callback 埋点 |
| 基座模型 | Qwen2.5-7B-Instruct | — | Qwen2.5-1.5B / 14B / Llama-3.1-8B | 中文工业术语表现好；7B 在 24G 卡上 QLoRA 可训、AWQ 后可服务 | 1.5B 抽取长工单时字段漏抽明显；14B 训练要 40G+；Llama 系中文工单错别字鲁棒性差 |
| 微调方法 | QLoRA (4bit NF4) | peft 0.13.x | 全参微调 / LoRA fp16 / Prefix-Tuning | 24G 单卡能跑 7B；adapter 仅 ~80MB，版本管理方便 | 全参微调要 8×A100；LoRA fp16 在 24G 上 batch 只能到 1；Prefix-Tuning 对结构化抽取效果不稳 |
| 训练框架 | transformers + trl + peft（手写脚本）**并列** LLaMA-Factory | — | axolotl / unsloth | 手写脚本让你看清 labels mask 怎么做（教学价值）；LLaMA-Factory 用于团队标准化 | axolotl 中文文档少；unsloth 快但对多卡与自定义 collator 支持有限 |
| 推理量化 | AWQ 4bit | — | GPTQ / fp16 / GGUF | vLLM 对 AWQ 支持成熟，7B AWQ 显存约 6GB，吞吐高 | GPTQ 校准慢一点；fp16 占 15GB 挤占 KV cache；GGUF 是 llama.cpp 路线，不走 vLLM |
| 推理服务 | vLLM | 0.6.x | Ollama / TGI / transformers pipeline | PagedAttention 吞吐高、OpenAI 兼容端点、支持 LoRA 热插拔与 prefix caching | Ollama 适合开发不适合并发；TGI 部署重；pipeline 无批处理调度 |
| 大模型 | deepseek-chat | — | 通义千问 / GPT-4o / 本地 14B | 中文强、价格低、OpenAI 兼容、支持 JSON mode 与 Function Calling | 本地 14B 在 24G 卡上和抽取模型抢显存 |
| 向量库 | Milvus 2.4 | — | Chroma / Qdrant | 工单 + 手册合计百万级向量，要标量过滤（型号/时间） | Chroma 单机教学够用但过滤能力弱 |
| Embedding | bge-m3 | — | bge-large-zh-v1.5 | 支持 dense + sparse 混合，长文本 8192 | large-zh 只有 512 上下文，长工单截断 |
| 重排 | bge-reranker-v2-m3 | — | 不做 rerank | 手册段落相似度高，rerank 对 Top-3 精度提升明显 | 不做 rerank 会把「XC-200」和「XC-300」的段落混在一起 |
| 可观测 | Langfuse 自托管 | — | LangSmith / 自写日志 | 数据不出内网；LangChain 原生 callback 接入 | LangSmith 数据出境，制造业客户通常不接受 |
| 服务框架 | FastAPI + Uvicorn | — | Flask | 原生 async（LangGraph 是 async 的）、自动 OpenAPI 文档、SSE 简单 | Flask 同步模型和 async 图不搭 |
| 前端 | 单文件 HTML + SSE | — | Streamlit / Gradio | 演示页要能嵌进客户现有工单系统的 iframe | Streamlit 有自己的 session 模型，嵌入麻烦 |

### 2.4 一个必须提前想清楚的问题：抽取模型和大模型的边界在哪

很多人做到一半会开始「什么都想微调」。请把这条边界刻在脑子里：

```mermaid
flowchart LR
    Q{这个任务……} -->|输入输出格式固定<br/>且样本量 > 2000| A[微调小模型]
    Q -->|需要引用会变的知识| B[RAG + 大模型]
    Q -->|需要多步推理或调工具| C[Agent + 大模型]
    Q -->|规则能写清楚| D[纯代码，别用模型]

    A --> A1["本项目：字段抽取、意图分类、<br/>紧急度判定、工种建议"]
    B --> B1["本项目：维修方案生成、<br/>相似案例总结"]
    C --> C1["本项目：查库存→判保修→<br/>建子工单→派单"]
    D --> D1["本项目：脱敏、型号正则归一化、<br/>SLA 时限计算、字段补全"]

    style A fill:#d4f5d4
    style B fill:#d4e6f5
    style C fill:#ffe0cc
    style D fill:#eeeeee
```

注意 `D` 这条：**「紧急度 = 3 且产线停机 → SLA 2 小时」这种规则不要交给模型**，用 Python 的 `if` 写，又快又对又能审计。本项目里这类规则约占全部逻辑的 1/3。

---

## 三、项目结构与环境准备

### 3.1 完整目录树

```text
huacheng-trinity/
├── README.md
├── pyproject.toml                  # uv / pip 依赖
├── .env.example                    # 配置模板（不要提交真实 key）
├── Makefile                        # 常用命令入口
│
├── data/
│   ├── raw/
│   │   └── tickets_history.csv     # 历史工单原始导出（28 万条的抽样 2 万条）
│   ├── interim/
│   │   ├── tickets_clean.jsonl     # 清洗后
│   │   └── rejected.jsonl          # 被丢弃的样本 + 原因
│   └── sft/
│       ├── train.jsonl             # 6400 条
│       ├── dev.jsonl               # 800 条
│       ├── test.jsonl              # 800 条
│       └── schema.json             # 抽取 schema 冻结文件
│
├── scripts/
│   ├── gen_mock_tickets.py         # 没有真实数据时，生成仿真工单
│   ├── build_sft_dataset.py        # Step 1：构造抽取任务数据集
│   ├── clean_and_split.py          # Step 2：清洗 + 切分 + 统计报告
│   ├── check_leakage.py            # 泄漏检查
│   ├── merge_lora.py               # Step 5：合并 adapter
│   └── quantize_awq.py             # Step 5：AWQ 量化
│
├── train/
│   ├── train_qlora.py              # Step 3：手写 QLoRA 训练脚本
│   ├── collator.py                 # 关键：labels mask 实现
│   └── llamafactory/
│       ├── ticket_extract.yaml     # LLaMA-Factory 训练配置
│       └── dataset_info.json       # LLaMA-Factory 数据集注册
│
├── app/
│   ├── __init__.py
│   ├── config.py                   # 统一配置（pydantic-settings）
│   ├── schemas.py                  # 抽取 schema 的 pydantic 模型
│   ├── llms.py                     # 模型工厂：小模型 / 大模型 / 降级链
│   ├── chains/
│   │   ├── extract_chain.py        # Step 6：LCEL 抽取链
│   │   └── generate_chain.py       # 方案生成链
│   ├── retrieval/
│   │   ├── hybrid.py               # 混合检索（复用项目 2）
│   │   └── rerank.py
│   ├── graph/
│   │   ├── state.py                # TicketState 定义
│   │   ├── nodes.py                # 各节点实现
│   │   ├── main_graph.py           # Step 7：主流程状态图
│   │   └── agent_subgraph.py       # Step 8：Agent 子图
│   ├── tools/
│   │   ├── stock.py                # 查库存
│   │   ├── warranty.py             # 判保修
│   │   ├── ticket_ops.py           # 建子工单 / 派单
│   │   └── registry.py             # 工具注册表 + 权限标注
│   ├── routing/
│   │   └── model_router.py         # Step 9：模型路由表与实现
│   ├── observability/
│   │   └── tracing.py              # Langfuse / OTel 埋点
│   ├── server.py                   # Step 10：FastAPI
│   └── graph_export.py             # 导出 mermaid / png
│
├── web/
│   └── index.html                  # 单文件演示前端（SSE）
│
├── eval/
│   ├── eval_extract.py             # 字段级准确率评测
│   ├── eval_e2e.py                 # 端到端评测
│   ├── cases_e2e.jsonl             # 端到端测试用例 60 条
│   └── metrics.py
│
├── bench/
│   └── bench_extract.py            # 抽取延迟压测
│
├── tests/
│   ├── test_schema.py
│   ├── test_agent_limit.py
│   ├── test_router.py
│   └── replay.py                   # 600 条工单回放
│
├── demo/
│   ├── demo_normal.py              # 演示 1：普通工单
│   ├── demo_urgent.py              # 演示 2：紧急工单需派单
│   ├── demo_incomplete.py          # 演示 3：信息不全需追问
│   ├── demo_fallback.sh            # 演示 4：降级链
│   └── demo_agent_interrupt.py     # 演示 5：人工确认中断恢复
│
├── deploy/
│   ├── docker-compose.yml
│   ├── Dockerfile.app
│   ├── start.sh
│   ├── stop.sh
│   └── healthcheck.sh
│
├── outputs/                        # 训练产物（gitignore）
│   ├── ticket-extract-lora/        # adapter
│   ├── merged/                     # 合并后的 fp16 权重
│   └── merged-awq/                 # AWQ 量化权重
│
└── reports/                        # 所有报告落这里
    ├── dataset_stats.md
    ├── training_curve.md
    ├── before_after.md
    ├── cost.md
    └── retro.md
```

### 3.2 依赖安装

```bash
# 1) 创建项目与虚拟环境（Python 3.11）
mkdir -p ~/huacheng-trinity && cd ~/huacheng-trinity
uv venv --python 3.11
source .venv/bin/activate

# 2) 训练侧依赖（需要 GPU 的机器上装）
uv pip install \
  "torch==2.4.0" \
  "transformers==4.45.2" \
  "peft==0.13.2" \
  "trl==0.11.4" \
  "datasets==3.0.1" \
  "accelerate==0.34.2" \
  "bitsandbytes==0.44.1" \
  "sentencepiece" "protobuf" "tensorboard"

# 3) 推理与应用侧依赖
uv pip install \
  "vllm==0.6.3" \
  "autoawq==0.2.6" \
  "langchain==0.3.7" \
  "langchain-core==0.3.15" \
  "langchain-openai==0.2.6" \
  "langchain-community==0.3.5" \
  "langgraph==0.2.45" \
  "langgraph-checkpoint-sqlite==2.0.1" \
  "pydantic==2.9.2" \
  "pydantic-settings==2.6.0" \
  "fastapi==0.115.4" \
  "uvicorn[standard]==0.32.0" \
  "sse-starlette==2.1.3" \
  "httpx==0.27.2" \
  "pymilvus==2.4.9" \
  "rank-bm25==0.2.2" \
  "FlagEmbedding==1.3.2" \
  "langfuse==2.53.1" \
  "pandas==2.2.3" "numpy==1.26.4" "rich==13.9.4" "tabulate==0.9.0"

# pip 等价命令（不用 uv 的话）
# python3.11 -m venv .venv && source .venv/bin/activate
# pip install torch==2.4.0 transformers==4.45.2 ...（同上）
```

> **版本说明**：vLLM 0.6.x 对 torch 版本敏感，如果你先装了 vLLM 再装 torch，可能被降级/升级。建议**训练环境和推理环境分开两个 venv**，本书后面的 docker-compose 就是这么隔离的。如果一定要放一起，安装顺序是：先 vLLM，再补其他包，最后 `pip check` 确认无冲突。

### 3.3 配置文件

`.env.example`：

```bash
# ============ 大模型（通用）============
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL=deepseek-chat

# ============ 小模型（微调后，vLLM 本地）============
SMALL_BASE_URL=http://127.0.0.1:8000/v1
SMALL_API_KEY=EMPTY
SMALL_MODEL=ticket-extract

# ============ 向量库 ============
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=huacheng_kb

# ============ Embedding / Rerank ============
EMBED_MODEL_PATH=/models/bge-m3
RERANK_MODEL_PATH=/models/bge-reranker-v2-m3

# ============ 可观测 ============
LANGFUSE_HOST=http://127.0.0.1:3000
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
TRACING_ENABLED=true

# ============ 业务系统（本项目用 mock）============
ERP_BASE_URL=http://127.0.0.1:9101
CRM_BASE_URL=http://127.0.0.1:9102
WO_BASE_URL=http://127.0.0.1:9103

# ============ 行为开关 ============
EXTRACT_TIMEOUT_MS=3000
EXTRACT_MAX_RETRY=2
AGENT_MAX_ITER=6
HUMAN_CONFIRM_REQUIRED=true
```

`app/config.py`：

```python
"""统一配置入口，全项目只从这里读环境变量。"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """项目全局配置，字段名与 .env 中的大写键一一对应。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "deepseek-chat"

    small_base_url: str = "http://127.0.0.1:8000/v1"
    small_api_key: str = "EMPTY"
    small_model: str = "ticket-extract"

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "huacheng_kb"

    embed_model_path: str = "/models/bge-m3"
    rerank_model_path: str = "/models/bge-reranker-v2-m3"

    langfuse_host: str = "http://127.0.0.1:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    tracing_enabled: bool = False

    erp_base_url: str = "http://127.0.0.1:9101"
    crm_base_url: str = "http://127.0.0.1:9102"
    wo_base_url: str = "http://127.0.0.1:9103"

    extract_timeout_ms: int = 3000
    extract_max_retry: int = 2
    agent_max_iter: int = 6
    human_confirm_required: bool = True


@lru_cache
def get_settings() -> Settings:
    """返回进程级单例配置。"""
    return Settings()
```

### 3.4 抽取 schema（全项目的契约）

这是本项目**最重要的一个文件**。schema 一旦冻结，训练数据、pydantic 校验、评测脚本、前端展示全部以它为准。改 schema = 重新训练 + 重新评测。

`app/schemas.py`：

```python
"""工单结构化抽取的 schema 定义，训练数据与线上推理共用同一份契约。"""
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Urgency(int, Enum):
    """紧急程度：1 一般 / 2 较急 / 3 紧急（产线停机或安全风险）。"""

    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Intent(str, Enum):
    """工单意图五分类。"""

    CONSULT = "咨询"
    REPAIR = "报修"
    COMPLAINT = "投诉"
    PARTS = "备件申请"
    URGE = "催办"


class TicketExtract(BaseModel):
    """抽取结果，7 个业务字段 + 1 个意图字段。字段顺序与训练样本严格一致。"""

    device_model: Optional[str] = Field(None, description="设备型号，如 XC-200-4P；抽不到填 null")
    fault_symptom: str = Field(..., description="故障现象，20 字以内的规范化描述")
    fault_code: Optional[str] = Field(None, description="故障码，如 E07；无则 null")
    urgency: Urgency = Field(..., description="紧急程度 1/2/3")
    production_line: Optional[str] = Field(None, description="所属产线，如 A2、三号线；无则 null")
    suggested_craft: Literal["电气", "机械", "液压", "控制", "其他"] = Field(..., description="建议工种")
    need_parts: bool = Field(..., description="是否需要备件")
    intent: Intent = Field(..., description="工单意图")

    @field_validator("device_model")
    @classmethod
    def normalize_model(cls, v: Optional[str]) -> Optional[str]:
        """型号统一大写去空格，便于和备件表 join。"""
        if v is None:
            return None
        v = v.strip().upper().replace(" ", "").replace("—", "-")
        return v or None

    @field_validator("fault_code")
    @classmethod
    def normalize_code(cls, v: Optional[str]) -> Optional[str]:
        """故障码统一大写，去掉常见的中文前缀。"""
        if v is None:
            return None
        v = v.strip().upper()
        for prefix in ("报警", "故障码", "错误码", "ERR=", "ERROR"):
            v = v.replace(prefix, "")
        v = v.strip("：: ")
        return v or None
```

配套的 `data/sft/schema.json`（给非 Python 侧和评测脚本用，也是训练 prompt 里嵌入的那份）：

```json
{
  "version": "v1.0.0",
  "frozen_at": "2025-03-01",
  "fields": [
    {"name": "device_model",     "type": "string|null", "desc": "设备型号，如 XC-200-4P"},
    {"name": "fault_symptom",    "type": "string",      "desc": "故障现象，20 字以内规范化描述"},
    {"name": "fault_code",       "type": "string|null", "desc": "故障码，如 E07"},
    {"name": "urgency",          "type": "int",         "enum": [1, 2, 3], "desc": "1一般/2较急/3紧急"},
    {"name": "production_line",  "type": "string|null", "desc": "所属产线，如 A2"},
    {"name": "suggested_craft",  "type": "string",      "enum": ["电气", "机械", "液压", "控制", "其他"]},
    {"name": "need_parts",       "type": "bool",        "desc": "是否需要备件"},
    {"name": "intent",           "type": "string",      "enum": ["咨询", "报修", "投诉", "备件申请", "催办"]}
  ]
}
```

### 3.5 没有 GPU 怎么办（三条替代路径）

| 路径 | 做法 | 成本 | 损失什么 | 适合谁 |
|---|---|---|---|---|
| **A. 租云 GPU 只跑训练** | 在 AutoDL / 阿里云 / 腾讯云 租 RTX 4090 24G 或 A100 40G，只在训练那 4 小时开机；训练完把 `outputs/ticket-extract-lora/`（约 80MB）下载到本机 | 约 3~8 元/小时 × 4 小时 ≈ 12~32 元 | 无。**这是最推荐的路径** | 绝大多数人 |
| **B. 换 1.5B 基座在小显存上训** | 把 `train_qlora.py` 的 `BASE_MODEL` 换成 `Qwen2.5-1.5B-Instruct`，`per_device_train_batch_size` 提到 4，其余不变；12G 卡可跑 | 电费 | 字段平均准确率下降约 3~6 个百分点（实测环境：本项目 800 条测试集，示例性数据），长工单漏抽增加 | 只有游戏本的人 |
| **C. 完全跳过训练** | Part A 只读不做，直接进 Part B。抽取节点用 `deepseek-chat` + 严格 JSON schema + few-shot（本项目 `app/llms.py` 里的 `build_extract_llm(use_finetuned=False)` 就是这条路） | 按 API 计费 | 失去「微调 vs 不微调」的对照实验，也体会不到成本/延迟的差别；**但 Part B/C 的编排与 Agent 完全不受影响** | 只关心编排与 Agent 的人 |

> 路径 C 的读者请务必至少**读一遍 Part A 的代码和第 4.4 节的对照表**，因为「什么时候该微调、微调能带来什么」是这一章的核心知识点，不是可选内容。

### 3.6 准备基座模型与数据

```bash
# 下载基座模型（国内用 modelscope 更快）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/models/Qwen2.5-7B-Instruct')
"

# 或用 huggingface-cli（配好 HF_ENDPOINT 镜像）
# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /models/Qwen2.5-7B-Instruct

# 下载 embedding / rerank 模型
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='/models/bge-m3')
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='/models/bge-reranker-v2-m3')
"
```

如果你手上没有真实工单数据，用下面这个脚本生成 2 万条仿真工单（它会刻意制造三种渠道的文本风格差异和噪声，训练出来的模型才有意义）。

`scripts/gen_mock_tickets.py`：

```python
"""生成仿真历史工单 CSV，用于没有真实数据时跑通全流程。"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

MODELS = ["XC-200-4P", "XC-200-6P", "XC-300-4P", "JS-75-B", "JS-110-B",
          "BP-15KW", "BP-22KW", "RV-50", "RV-90", "KG-160"]
LINES = ["A1", "A2", "B1", "B3", "C2", "一号线", "二号线", "三号线", "四号线", None]
CODES = ["E07", "E12", "E21", "E33", "F02", "F15", "A09", None, None, None]

# (故障现象模板, 建议工种, 是否需备件, 典型故障码)
SYMPTOMS = [
    ("电机运行{n}分钟后报{code}，外壳烫手", "电气", True, "E07"),
    ("减速机漏油，油位低于下限", "机械", True, "E12"),
    ("变频器上电无显示", "电气", True, "E21"),
    ("设备启动后有异响，转速不稳", "机械", False, None),
    ("控制柜接触器频繁跳闸", "电气", True, "E33"),
    ("液压站压力上不去，保压失败", "液压", True, "F02"),
    ("PLC 通讯中断，触摸屏报通讯超时", "控制", False, "F15"),
    ("电机三相电流不平衡", "电气", False, "E07"),
    ("轴承部位温度偏高，超过 85 度", "机械", True, "A09"),
    ("设备无法启动，急停按钮已复位", "控制", False, None),
    ("风扇不转，机身温度报警", "电气", True, "E07"),
    ("联轴器磨损严重，有金属摩擦声", "机械", True, None),
]

CHANNELS = ["400电话", "小程序", "经销商系统"]
CRAFT_MAP = {"电气": "电气", "机械": "机械", "液压": "液压", "控制": "控制"}


def make_phone_text(model, line, sym, code):
    """400 电话渠道：客服代填，句子完整但偏口语。"""
    parts = [f"客户反映{sym}。"]
    if model:
        parts.append(f"型号{model}。")
    if line:
        parts.append(f"{line}线。")
    extra = random.choice([
        "客户说上周刚做过保养。", "客户要求今天上门。", "客户情绪比较急。",
        "客户在电话里描述不清，建议现场确认。", "",
    ])
    parts.append(extra)
    return "".join(parts)


def make_miniapp_text(model, line, sym, code):
    """小程序渠道：客户自填，短句、错别字、感叹号、信息缺失。"""
    typo = {"报警": "抱警", "异响": "异想", "漏油": "露油", "轴承": "轴成"}
    t = sym
    for k, v in typo.items():
        if k in t and random.random() < 0.35:
            t = t.replace(k, v)
    frag = random.choice([
        f"{t} 急！！！", f"机器坏了 {t}", f"{t}，怎么办", f"{t} 在保修期吧",
    ])
    if model and random.random() < 0.4:
        frag += f" 型号好像是{model}"
    return frag


def make_dealer_text(model, line, sym, code):
    """经销商系统渠道：半结构化 KV 串。"""
    urg = random.choice([1, 2, 3])
    fields = [f"DEV_MODEL={model or 'UNKNOWN'}"]
    if code:
        fields.append(f"ERR={code}")
    if line:
        fields.append(f"LINE={line}")
    fields.append(f"DESC={sym}")
    fields.append(f"URGENCY={urg}")
    return "|".join(fields)


def main(n: int = 20000, out: str = "data/raw/tickets_history.csv"):
    """生成 n 条仿真工单并写入 CSV。"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    base = datetime(2022, 1, 1)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "ticket_id", "create_time", "channel", "raw_text",
            "device_model", "fault_symptom", "fault_code", "urgency",
            "production_line", "suggested_craft", "need_parts", "intent",
        ])
        for i in range(n):
            sym_tpl, craft, need_parts, code_hint = random.choice(SYMPTOMS)
            model = random.choice(MODELS) if random.random() < 0.85 else None
            line = random.choice(LINES)
            code = code_hint if (code_hint and random.random() < 0.8) else random.choice(CODES)
            sym = sym_tpl.format(n=random.choice([10, 15, 20, 30, 40]), code=code or "报警")
            channel = random.choices(CHANNELS, weights=[0.5, 0.35, 0.15])[0]

            if channel == "400电话":
                raw = make_phone_text(model, line, sym, code)
                urgency = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
                intent = random.choices(["报修", "咨询", "备件申请", "投诉", "催办"],
                                        weights=[0.62, 0.2, 0.1, 0.05, 0.03])[0]
            elif channel == "小程序":
                raw = make_miniapp_text(model, line, sym, code)
                urgency = random.choices([1, 2, 3], weights=[0.35, 0.4, 0.25])[0]
                intent = random.choices(["报修", "咨询", "催办", "投诉", "备件申请"],
                                        weights=[0.6, 0.18, 0.12, 0.07, 0.03])[0]
                if random.random() < 0.25:
                    model = None   # 小程序渠道信息缺失更常见
            else:
                raw = make_dealer_text(model, line, sym, code)
                urgency = int(raw.split("URGENCY=")[-1])
                intent = random.choices(["报修", "备件申请", "咨询"], weights=[0.7, 0.25, 0.05])[0]

            # 紧急度受产线停机影响：出现"停机/停线"字样则升级
            if "停机" in raw or "停线" in raw:
                urgency = 3

            w.writerow([
                f"TK-{2022 + i // 7000}-{100000 + i}",
                (base + timedelta(minutes=i * 7)).strftime("%Y-%m-%d %H:%M:%S"),
                channel, raw,
                model or "", sym[:20], code or "", urgency,
                line or "", CRAFT_MAP.get(craft, "其他"), int(need_parts), intent,
            ])
    print(f"已生成 {n} 条仿真工单 -> {out}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/gen_mock_tickets.py
```

预期输出：

```text
已生成 20000 条仿真工单 -> data/raw/tickets_history.csv
```

> **重要提醒**：仿真数据的模式比真实数据规整得多，用它训练出来的模型指标会**偏乐观**。真实历史工单里有大量「客服填错字段」「同一现象两个人填不同工种」的噪声，这些噪声正是清洗环节要处理的重点。第 4.2 节的清洗脚本对两种数据都适用。

---

## 四、分阶段实现

本节分三个 Part、十个 Step。每个 Step 的结构统一为：**目标 → 完整代码 → 运行命令 → 预期输出 → 小结**。

```mermaid
flowchart LR
    subgraph PA["Part A：LoRA 微调（Step 1~5）"]
        S1[S1 数据构造] --> S2[S2 清洗切分] --> S3[S3 QLoRA 训练] --> S4[S4 监控验证] --> S5[S5 合并量化部署]
    end
    subgraph PB["Part B：编排（Step 6~7）"]
        S6[S6 LCEL 抽取链] --> S7[S7 LangGraph 主流程]
    end
    subgraph PC["Part C：Agent 与服务（Step 8~10）"]
        S8[S8 Agent 子图] --> S9[S9 路由/降级/成本] --> S10[S10 服务化与前端]
    end
    S5 --> S6
    S7 --> S8
```

---

## Part A：LoRA 微调 —— 把抽取能力烧进权重

### Step 1：训练数据构造

**目标**：把历史工单 CSV 转成 SFT 训练样本（ChatML 格式的 messages），输入是工单原文，输出是符合 schema 的 JSON。

#### 1.1 数据 schema 设计的四条规则

在写脚本之前，先把四条规则讲明白，否则你会造出一个「训练得很好但线上没法用」的数据集。

| 规则 | 说明 | 违反的后果 |
|---|---|---|
| **训练格式 = 推理格式** | 训练时的 system prompt、字段顺序、JSON 缩进风格，必须和线上推理时**逐字节一致** | 线上一换 prompt，效果断崖下跌，你还查不出原因 |
| **输出必须是最小完整 JSON** | 不要输出 ```json 代码块包裹，不要有解释文字，只输出 `{...}` | 多余字符会让 `json.loads` 失败，或者让模型学会啰嗦 |
| **null 要显式出现** | 抽不到的字段输出 `null`，不要省略 key | 省略 key 会让模型学会「不确定就不输出」，下游解析崩溃 |
| **字段顺序固定** | 8 个字段固定顺序，永不改变 | 顺序随机会显著拖慢收敛（自回归模型对顺序敏感） |

#### 1.2 prompt 模板（训练与推理共用）

```python
SYSTEM_PROMPT = """你是华成机电售后工单结构化抽取引擎。
请从工单原文中抽取字段，并严格按下面的 JSON 格式输出，不要输出任何解释、前缀或代码块标记。

字段定义：
- device_model: 设备型号，如 "XC-200-4P"；抽不到填 null
- fault_symptom: 故障现象，20 字以内的规范化描述，必须填
- fault_code: 故障码，如 "E07"；无则 null
- urgency: 紧急程度，整数 1(一般)/2(较急)/3(紧急，产线停机或安全风险)
- production_line: 所属产线，如 "A2"；无则 null
- suggested_craft: 建议工种，取值 "电气"/"机械"/"液压"/"控制"/"其他"
- need_parts: 是否需要备件，true 或 false
- intent: 工单意图，取值 "咨询"/"报修"/"投诉"/"备件申请"/"催办"

输出示例：
{"device_model":"XC-200-4P","fault_symptom":"电机报E07外壳过热","fault_code":"E07","urgency":2,"production_line":"三号线","suggested_craft":"电气","need_parts":true,"intent":"报修"}"""

USER_TEMPLATE = """【渠道】{channel}
【工单原文】
{raw_text}"""
```

> 为什么把 schema 写进 system prompt 而不是只靠微调记住？两个原因：一是**推理时可以在不重训的情况下微调描述**（比如补一句「产线名统一用字母编号」）；二是**降级到大模型时同一份 prompt 直接可用**，降级链不需要维护第二套 prompt。代价是每次多约 320 个 token，在 vLLM 开了 prefix caching 后这部分几乎不额外耗时（见第 4.5 节）。

#### 1.3 构造脚本

`scripts/build_sft_dataset.py`：

```python
"""从历史工单 CSV 构造 SFT 训练样本（messages 格式）。"""
import argparse
import csv
import json
from pathlib import Path

SYSTEM_PROMPT = """你是华成机电售后工单结构化抽取引擎。
请从工单原文中抽取字段，并严格按下面的 JSON 格式输出，不要输出任何解释、前缀或代码块标记。

字段定义：
- device_model: 设备型号，如 "XC-200-4P"；抽不到填 null
- fault_symptom: 故障现象，20 字以内的规范化描述，必须填
- fault_code: 故障码，如 "E07"；无则 null
- urgency: 紧急程度，整数 1(一般)/2(较急)/3(紧急，产线停机或安全风险)
- production_line: 所属产线，如 "A2"；无则 null
- suggested_craft: 建议工种，取值 "电气"/"机械"/"液压"/"控制"/"其他"
- need_parts: 是否需要备件，true 或 false
- intent: 工单意图，取值 "咨询"/"报修"/"投诉"/"备件申请"/"催办"

输出示例：
{"device_model":"XC-200-4P","fault_symptom":"电机报E07外壳过热","fault_code":"E07","urgency":2,"production_line":"三号线","suggested_craft":"电气","need_parts":true,"intent":"报修"}"""

USER_TEMPLATE = """【渠道】{channel}
【工单原文】
{raw_text}"""

FIELD_ORDER = [
    "device_model", "fault_symptom", "fault_code", "urgency",
    "production_line", "suggested_craft", "need_parts", "intent",
]


def row_to_target(row: dict) -> dict:
    """把 CSV 一行转成目标 JSON（字段顺序固定，空值转 null）。"""
    def nz(v):
        v = (v or "").strip()
        return v if v else None

    return {
        "device_model": nz(row.get("device_model")),
        "fault_symptom": (row.get("fault_symptom") or "").strip()[:20],
        "fault_code": nz(row.get("fault_code")),
        "urgency": int(row.get("urgency") or 1),
        "production_line": nz(row.get("production_line")),
        "suggested_craft": (row.get("suggested_craft") or "其他").strip(),
        "need_parts": bool(int(row.get("need_parts") or 0)),
        "intent": (row.get("intent") or "报修").strip(),
    }


def dump_target(obj: dict) -> str:
    """序列化目标 JSON：紧凑格式、不转义中文、字段顺序固定。"""
    ordered = {k: obj[k] for k in FIELD_ORDER}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def main():
    """读 CSV，写 JSONL（每行一个 messages 样本 + 元数据）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/tickets_history.csv")
    ap.add_argument("--dst", default="data/interim/sft_raw.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 表示不限制")
    args = ap.parse_args()

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    n_in, n_out = 0, 0
    with open(args.src, encoding="utf-8") as f, open(args.dst, "w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            n_in += 1
            if args.limit and n_out >= args.limit:
                break
            raw_text = (row.get("raw_text") or "").strip()
            if not raw_text:
                continue
            target = row_to_target(row)
            if not target["fault_symptom"]:
                continue   # 故障现象是必填，缺了这条样本没有价值

            sample = {
                "ticket_id": row["ticket_id"],
                "channel": row["channel"],
                "create_time": row["create_time"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        channel=row["channel"], raw_text=raw_text)},
                    {"role": "assistant", "content": dump_target(target)},
                ],
            }
            g.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"读入 {n_in} 行，产出 {n_out} 条 SFT 样本 -> {args.dst}")


if __name__ == "__main__":
    main()
```

#### 1.4 运行命令

```bash
python scripts/build_sft_dataset.py \
  --src data/raw/tickets_history.csv \
  --dst data/interim/sft_raw.jsonl
```

#### 1.5 预期输出

```text
读入 20000 行，产出 19986 条 SFT 样本 -> data/interim/sft_raw.jsonl
```

#### 1.6 20 条完整训练样本示例

下面是从 `sft_raw.jsonl` 里抽出来的 20 条样本，**为了阅读方便只展示 user 的 content 和 assistant 的 content**（system 固定不变）。请重点看：不同渠道的文本风格差异、缺失字段怎么填 null、紧急度的判断依据、工种的判断依据。

**样本 1（400 电话 · 典型报修 · 信息完整）**

```text
user:
【渠道】400电话
【工单原文】
客户反映电机运行20分钟后报E07，外壳烫手，风扇在转。型号XC-200-4P。三号线。客户要求今天上门。

assistant:
{"device_model":"XC-200-4P","fault_symptom":"电机运行20分钟后报E07外壳过热","fault_code":"E07","urgency":2,"production_line":"三号线","suggested_craft":"电气","need_parts":true,"intent":"报修"}
```

**样本 2（小程序 · 信息极度缺失 · 口语）**

```text
user:
【渠道】小程序
【工单原文】
机器坏了 转不动 有异想 急！！！ 昨天买的还在保修吧

assistant:
{"device_model":null,"fault_symptom":"设备无法转动并有异响","fault_code":null,"urgency":2,"production_line":null,"suggested_craft":"机械","need_parts":false,"intent":"报修"}
```

**样本 3（经销商系统 · 半结构化 KV）**

```text
user:
【渠道】经销商系统
【工单原文】
DEV_MODEL=JS-75-B|ERR=E12|LINE=A2|DESC=减速机漏油,油位低于下限|URGENCY=2

assistant:
{"device_model":"JS-75-B","fault_symptom":"减速机漏油油位低于下限","fault_code":"E12","urgency":2,"production_line":"A2","suggested_craft":"机械","need_parts":true,"intent":"报修"}
```

**样本 4（400 电话 · 产线停机 · 紧急度升到 3）**

```text
user:
【渠道】400电话
【工单原文】
客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。客户说损失很大，要求两小时内到场。

assistant:
{"device_model":"BP-22KW","fault_symptom":"变频器上电无显示报E21","fault_code":"E21","urgency":3,"production_line":"B1","suggested_craft":"电气","need_parts":true,"intent":"报修"}
```

**样本 5（小程序 · 纯咨询，不是报修）**

```text
user:
【渠道】小程序
【工单原文】
请问XC-300-4P这个型号的端盖螺栓拧多大力矩合适

assistant:
{"device_model":"XC-300-4P","fault_symptom":"咨询端盖螺栓紧固扭矩","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"机械","need_parts":false,"intent":"咨询"}
```

**样本 6（400 电话 · 备件申请）**

```text
user:
【渠道】400电话
【工单原文】
客户要订两个CJ20-40接触器，说自己换，机器是KG-160的控制柜，不需要上门。

assistant:
{"device_model":"KG-160","fault_symptom":"申请接触器备件自行更换","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"电气","need_parts":true,"intent":"备件申请"}
```

**样本 7（小程序 · 投诉）**

```text
user:
【渠道】小程序
【工单原文】
上次来的师傅修了没两天又坏了 同样的问题 你们到底会不会修 要求换人

assistant:
{"device_model":null,"fault_symptom":"维修后同一故障复发","fault_code":null,"urgency":2,"production_line":null,"suggested_craft":"其他","need_parts":false,"intent":"投诉"}
```

**样本 8（小程序 · 催办）**

```text
user:
【渠道】小程序
【工单原文】
昨天报的单子到现在没人联系 什么时候来

assistant:
{"device_model":null,"fault_symptom":"已报工单未响应催促处理","fault_code":null,"urgency":2,"production_line":null,"suggested_craft":"其他","need_parts":false,"intent":"催办"}
```

**样本 9（经销商系统 · 液压工种）**

```text
user:
【渠道】经销商系统
【工单原文】
DEV_MODEL=RV-90|ERR=F02|LINE=C2|DESC=液压站压力上不去,保压失败|URGENCY=3

assistant:
{"device_model":"RV-90","fault_symptom":"液压站压力不足保压失败","fault_code":"F02","urgency":3,"production_line":"C2","suggested_craft":"液压","need_parts":true,"intent":"报修"}
```

**样本 10（400 电话 · 控制工种 · 无需备件）**

```text
user:
【渠道】400电话
【工单原文】
客户说PLC通讯中断，触摸屏报通讯超时，F15。设备是RV-50，二号线。已经重启过没用。

assistant:
{"device_model":"RV-50","fault_symptom":"PLC通讯中断触摸屏报超时","fault_code":"F15","urgency":2,"production_line":"二号线","suggested_craft":"控制","need_parts":false,"intent":"报修"}
```

**样本 11（小程序 · 有错别字 · 型号写得不规范）**

```text
user:
【渠道】小程序
【工单原文】
轴成那里发烫 温度计测了90多度 型号好像是 xc 200 6p

assistant:
{"device_model":"XC-200-6P","fault_symptom":"轴承温度超90度过热","fault_code":null,"urgency":2,"production_line":null,"suggested_craft":"机械","need_parts":true,"intent":"报修"}
```

**样本 12（400 电话 · 一句话里有两个型号，取故障设备那个）**

```text
user:
【渠道】400电话
【工单原文】
客户上个月买的XC-300-4P没问题，这次是旧的那台JS-110-B减速机有异响，转速不稳，A1线。

assistant:
{"device_model":"JS-110-B","fault_symptom":"减速机异响转速不稳","fault_code":null,"urgency":1,"production_line":"A1","suggested_craft":"机械","need_parts":false,"intent":"报修"}
```

**样本 13（经销商系统 · 型号未知）**

```text
user:
【渠道】经销商系统
【工单原文】
DEV_MODEL=UNKNOWN|ERR=E33|DESC=控制柜接触器频繁跳闸|URGENCY=2

assistant:
{"device_model":null,"fault_symptom":"控制柜接触器频繁跳闸","fault_code":"E33","urgency":2,"production_line":null,"suggested_craft":"电气","need_parts":true,"intent":"报修"}
```

**样本 14（400 电话 · 安全风险触发紧急度 3）**

```text
user:
【渠道】400电话
【工单原文】
客户说BP-15KW控制柜有焦糊味还冒烟了，已经断电。现场有人，问怎么处理。

assistant:
{"device_model":"BP-15KW","fault_symptom":"控制柜焦糊味冒烟已断电","fault_code":null,"urgency":3,"production_line":null,"suggested_craft":"电气","need_parts":true,"intent":"报修"}
```

**样本 15（小程序 · 咨询保修期，不是报修）**

```text
user:
【渠道】小程序
【工单原文】
我这台是2023年3月买的 XC-200-4P 保修期是几年啊

assistant:
{"device_model":"XC-200-4P","fault_symptom":"咨询设备保修期年限","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"其他","need_parts":false,"intent":"咨询"}
```

**样本 16（400 电话 · 故障现象很长，要压缩到 20 字以内）**

```text
user:
【渠道】400电话
【工单原文】
客户描述：设备早上启动的时候一切正常，运行到下午三点左右开始出现电流波动，A相18.7安B相17.9安C相26.3安，明显不平衡，然后就报E07停机了，型号XC-200-4P，四号线。

assistant:
{"device_model":"XC-200-4P","fault_symptom":"三相电流不平衡报E07停机","fault_code":"E07","urgency":3,"production_line":"四号线","suggested_craft":"电气","need_parts":false,"intent":"报修"}
```

**样本 17（小程序 · 只有一个词，极端稀疏输入）**

```text
user:
【渠道】小程序
【工单原文】
漏油

assistant:
{"device_model":null,"fault_symptom":"设备漏油","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"机械","need_parts":true,"intent":"报修"}
```

**样本 18（经销商系统 · 备件申请意图）**

```text
user:
【渠道】经销商系统
【工单原文】
DEV_MODEL=KG-160|DESC=申请更换端盖密封圈两套,库存自提|URGENCY=1

assistant:
{"device_model":"KG-160","fault_symptom":"申请端盖密封圈备件两套","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"机械","need_parts":true,"intent":"备件申请"}
```

**样本 19（400 电话 · 无法启动 · 控制类）**

```text
user:
【渠道】400电话
【工单原文】
客户RV-50设备无法启动，急停按钮已经复位了还是不行，也没报任何代码。C2线。客户比较着急但没停整线。

assistant:
{"device_model":"RV-50","fault_symptom":"急停复位后设备仍无法启动","fault_code":null,"urgency":2,"production_line":"C2","suggested_craft":"控制","need_parts":false,"intent":"报修"}
```

**样本 20（小程序 · 混合意图，以主要诉求为准）**

```text
user:
【渠道】小程序
【工单原文】
上周报的单还没处理 现在机器彻底不转了 整条线停了 型号JS-75-B B3线 赶紧派人

assistant:
{"device_model":"JS-75-B","fault_symptom":"设备完全停转产线停线","fault_code":null,"urgency":3,"production_line":"B3","suggested_craft":"机械","need_parts":false,"intent":"报修"}
```

> **样本 20 的标注争议点**：文本里既有「上周报的单还没处理」（催办）又有「机器彻底不转了」（报修）。标注规范规定：**以当前最重要的业务诉求为准**，产线停机的报修优先级高于催办。这类规则必须写进标注手册，否则同一批数据里会出现互相矛盾的标签，模型学不会，评测也没法算准确率。

#### 1.7 小结

- 数据 schema 的四条规则（训练=推理、最小 JSON、显式 null、字段顺序固定）比训练超参重要得多；
- system prompt 写进训练样本，换来的是「降级链共用 prompt」和「不重训也能微调描述」；
- 20 条样本覆盖了三渠道 × 五意图 × 边界情况（型号缺失、错别字、多型号、超长文本、极短文本、混合意图）；
- **标注规范要成文**。上面样本 20 的争议点如果没有规范，你的数据里会有 5% 的矛盾标签，这 5% 会让你的评测指标永远上不去，而你会误以为是模型的问题。

---

### Step 2：数据清洗与切分

**目标**：把 19986 条原始样本清洗成可训练的数据集，并按 ticket_id 无泄漏地切分为 train/dev/test，同时产出一份统计报告。

#### 2.1 要清洗掉什么

| 清洗规则 | 原因 | 预期比例 |
|---|---|---|
| 原文长度 < 4 字符 | 没有信息量，学不到东西，还会教模型瞎猜 | ~0.5% |
| 原文长度 > 1200 字符 | 长尾，会拉长训练序列、浪费显存；线上超长走截断策略 | ~0.3% |
| 目标 JSON 反序列化失败 / schema 校验失败 | 脏标注 | ~0.2% |
| `fault_symptom` 为空或 > 20 字 | 必填字段不合规 | ~1.2% |
| 枚举字段取值不在合法集合内 | 客服填了自定义值，如工种填「综合」 | ~2.1% |
| 精确重复（原文 + 目标都相同） | 重复样本会让模型对某些模式过拟合 | ~4~8% |
| 近重复（原文归一化后相同，目标不同） | **标注冲突**，必须人工裁决或整组丢弃 | ~0.4% |
| 含手机号/身份证/客户全称 | 隐私，必须脱敏而不是丢弃 | ~15%（脱敏不丢弃） |

#### 2.2 清洗与切分脚本

`scripts/clean_and_split.py`：

```python
"""清洗 SFT 原始样本、脱敏、去重、按 ticket_id 切分，并输出统计报告。"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

VALID_CRAFT = {"电气", "机械", "液压", "控制", "其他"}
VALID_INTENT = {"咨询", "报修", "投诉", "备件申请", "催办"}
VALID_URGENCY = {1, 2, 3}

RE_PHONE = re.compile(r"1[3-9]\d{9}")
RE_IDCARD = re.compile(r"\d{17}[\dXx]")
RE_COMPANY = re.compile(r"[一-龥]{2,12}(有限公司|股份有限公司|厂|集团)")
RE_SPACE = re.compile(r"[ \t　]+")


def desensitize(text: str) -> str:
    """把手机号、身份证、公司全称替换成占位符。"""
    text = RE_PHONE.sub("[PHONE]", text)
    text = RE_IDCARD.sub("[IDCARD]", text)
    text = RE_COMPANY.sub("[COMPANY]", text)
    return text


def norm_key(text: str) -> str:
    """归一化后的去重键：去空白、统一大小写。"""
    return RE_SPACE.sub("", text).lower()


def validate_target(obj: dict) -> str | None:
    """校验目标 JSON 是否符合 schema，返回错误原因，None 表示通过。"""
    required = {"device_model", "fault_symptom", "fault_code", "urgency",
                "production_line", "suggested_craft", "need_parts", "intent"}
    if set(obj.keys()) != required:
        return f"字段集合不匹配: 多={set(obj)-required} 少={required-set(obj)}"
    if not isinstance(obj["fault_symptom"], str) or not obj["fault_symptom"].strip():
        return "fault_symptom 为空"
    if len(obj["fault_symptom"]) > 20:
        return f"fault_symptom 超长({len(obj['fault_symptom'])})"
    if obj["urgency"] not in VALID_URGENCY:
        return f"urgency 非法: {obj['urgency']}"
    if obj["suggested_craft"] not in VALID_CRAFT:
        return f"suggested_craft 非法: {obj['suggested_craft']}"
    if obj["intent"] not in VALID_INTENT:
        return f"intent 非法: {obj['intent']}"
    if not isinstance(obj["need_parts"], bool):
        return "need_parts 非 bool"
    return None


def main():
    """执行清洗、去重、切分，并写出 jsonl 与统计报告。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/interim/sft_raw.jsonl")
    ap.add_argument("--outdir", default="data/sft")
    ap.add_argument("--report", default="reports/dataset_stats.md")
    ap.add_argument("--dev-size", type=int, default=800)
    ap.add_argument("--test-size", type=int, default=800)
    ap.add_argument("--max-train", type=int, default=6400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path("data/interim").mkdir(parents=True, exist_ok=True)

    kept, rejected = [], []
    reasons = Counter()
    seen_exact, seen_input = set(), {}
    len_buckets = Counter()
    field_missing = Counter()

    with open(args.src, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            user_msg = s["messages"][1]["content"]
            asst_msg = s["messages"][2]["content"]
            raw_body = user_msg.split("【工单原文】\n", 1)[-1]

            # --- 规则 1：长度 ---
            if len(raw_body) < 4:
                reasons["原文过短"] += 1
                rejected.append({"id": s["ticket_id"], "reason": "原文过短"})
                continue
            if len(raw_body) > 1200:
                reasons["原文过长"] += 1
                rejected.append({"id": s["ticket_id"], "reason": "原文过长"})
                continue

            # --- 规则 2：目标可解析且合规 ---
            try:
                target = json.loads(asst_msg)
            except json.JSONDecodeError as e:
                reasons["目标JSON不可解析"] += 1
                rejected.append({"id": s["ticket_id"], "reason": f"JSON错误:{e}"})
                continue
            err = validate_target(target)
            if err:
                reasons[f"schema不合规:{err.split(':')[0]}"] += 1
                rejected.append({"id": s["ticket_id"], "reason": err})
                continue

            # --- 规则 3：脱敏（改写而非丢弃）---
            user_msg = desensitize(user_msg)
            s["messages"][1]["content"] = user_msg
            raw_body = desensitize(raw_body)

            # --- 规则 4：精确去重 ---
            h = hashlib.md5((raw_body + "||" + asst_msg).encode()).hexdigest()
            if h in seen_exact:
                reasons["精确重复"] += 1
                rejected.append({"id": s["ticket_id"], "reason": "精确重复"})
                continue
            seen_exact.add(h)

            # --- 规则 5：近重复（同输入不同输出 = 标注冲突）---
            ik = norm_key(raw_body)
            if ik in seen_input and seen_input[ik] != asst_msg:
                reasons["标注冲突(同输入不同输出)"] += 1
                rejected.append({"id": s["ticket_id"], "reason": "标注冲突"})
                continue
            seen_input[ik] = asst_msg

            # --- 统计 ---
            L = len(raw_body)
            bucket = "0-50" if L < 50 else "50-100" if L < 100 else \
                     "100-200" if L < 200 else "200-400" if L < 400 else "400+"
            len_buckets[bucket] += 1
            for k in ("device_model", "fault_code", "production_line"):
                if target[k] is None:
                    field_missing[k] += 1

            s["_target"] = target
            s["_raw_len"] = L
            kept.append(s)

    # --- 按 ticket_id 分组切分，保证同一工单不跨集 ---
    by_ticket = defaultdict(list)
    for s in kept:
        by_ticket[s["ticket_id"]].append(s)
    ticket_ids = sorted(by_ticket)
    random.shuffle(ticket_ids)

    def take(ids, n):
        """从 id 列表前部取够 n 条样本，返回(样本列表, 剩余id)。"""
        out, i = [], 0
        while i < len(ids) and len(out) < n:
            out.extend(by_ticket[ids[i]])
            i += 1
        return out[:n], ids[i:]

    test, rest = take(ticket_ids, args.test_size)
    dev, rest = take(rest, args.dev_size)
    train, _ = take(rest, args.max_train)

    def write(name, rows):
        """写出一个 split，只保留训练需要的 messages 与 ticket_id。"""
        p = Path(args.outdir) / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as g:
            for r in rows:
                g.write(json.dumps(
                    {"ticket_id": r["ticket_id"], "channel": r["channel"],
                     "messages": r["messages"]}, ensure_ascii=False) + "\n")
        return p

    write("train", train), write("dev", dev), write("test", test)
    with open("data/interim/rejected.jsonl", "w", encoding="utf-8") as g:
        for r in rejected:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 统计报告 ---
    def dist(rows, key):
        """统计某个目标字段的取值分布。"""
        c = Counter(r["_target"][key] for r in rows)
        total = sum(c.values()) or 1
        return "\n".join(f"| {k} | {v} | {v/total:.1%} |" for k, v in c.most_common())

    total_in = len(kept) + len(rejected)
    lines = [
        "# 数据集统计报告",
        "",
        f"- 输入样本：{total_in}",
        f"- 保留：{len(kept)}（{len(kept)/total_in:.1%}）",
        f"- 丢弃：{len(rejected)}（{len(rejected)/total_in:.1%}）",
        f"- 切分：train={len(train)} / dev={len(dev)} / test={len(test)}",
        f"- 随机种子：{args.seed}",
        "",
        "## 一、丢弃原因分布",
        "",
        "| 原因 | 条数 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in reasons.most_common()]
    lines += [
        "",
        "## 二、原文长度分布（保留样本）",
        "",
        "| 长度区间(字符) | 条数 | 占比 |",
        "|---|---|---|",
    ]
    tot = len(kept) or 1
    for b in ["0-50", "50-100", "100-200", "200-400", "400+"]:
        lines.append(f"| {b} | {len_buckets[b]} | {len_buckets[b]/tot:.1%} |")
    lines += [
        "",
        "## 三、可选字段缺失率（即目标为 null 的比例）",
        "",
        "| 字段 | null 条数 | 占比 |",
        "|---|---|---|",
    ]
    for k, v in field_missing.most_common():
        lines.append(f"| {k} | {v} | {v/tot:.1%} |")
    for split_name, rows in [("train", train), ("dev", dev), ("test", test)]:
        for field in ("intent", "suggested_craft", "urgency"):
            lines += [
                "", f"## {split_name} 的 {field} 分布", "",
                "| 取值 | 条数 | 占比 |", "|---|---|---|", dist(rows, field),
            ]

    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"保留 {len(kept)} / 丢弃 {len(rejected)}；"
          f"train={len(train)} dev={len(dev)} test={len(test)}")
    print(f"报告 -> {args.report}")


if __name__ == "__main__":
    main()
```

#### 2.3 泄漏检查脚本

`scripts/check_leakage.py`：

```python
"""检查 train/dev/test 之间是否存在 ticket_id 或输入文本泄漏。"""
import json
import re
import sys
from pathlib import Path

RE_SPACE = re.compile(r"[ \t　]+")


def load(p):
    """读取一个 split，返回 (ticket_id 集合, 归一化输入文本集合)。"""
    ids, texts = set(), set()
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        ids.add(s["ticket_id"])
        body = s["messages"][1]["content"].split("【工单原文】\n", 1)[-1]
        texts.add(RE_SPACE.sub("", body).lower())
    return ids, texts


def main():
    """两两比对三个 split，有交集则以非零码退出。"""
    splits = {n: load(f"data/sft/{n}.jsonl") for n in ("train", "dev", "test")}
    bad = False
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            id_ov = splits[a][0] & splits[b][0]
            tx_ov = splits[a][1] & splits[b][1]
            print(f"{a} vs {b}: ticket_id 交集={len(id_ov)}, 输入文本交集={len(tx_ov)}")
            if id_ov or tx_ov:
                bad = True
                for x in list(id_ov)[:5]:
                    print(f"   泄漏 ticket_id 示例: {x}")
    if bad:
        print("❌ 存在泄漏，评测结果不可信，请重新切分")
        sys.exit(1)
    print("✅ 无泄漏")


if __name__ == "__main__":
    main()
```

#### 2.4 运行命令

```bash
python scripts/clean_and_split.py
python scripts/check_leakage.py
head -c 600 data/sft/train.jsonl
wc -l data/sft/*.jsonl
```

#### 2.5 预期输出

```text
保留 18412 / 丢弃 1574；train=6400 dev=800 test=800
报告 -> reports/dataset_stats.md

train vs dev: ticket_id 交集=0, 输入文本交集=0
train vs test: ticket_id 交集=0, 输入文本交集=0
dev vs test: ticket_id 交集=0, 输入文本交集=0
✅ 无泄漏

   800 data/sft/dev.jsonl
   800 data/sft/test.jsonl
  6400 data/sft/train.jsonl
  8000 total
```

`reports/dataset_stats.md` 节选（实测环境：仿真数据 2 万条，示例性数据）：

```text
# 数据集统计报告

- 输入样本：19986
- 保留：18412（92.1%）
- 丢弃：1574（7.9%）
- 切分：train=6400 / dev=800 / test=800
- 随机种子：42

## 一、丢弃原因分布

| 原因 | 条数 |
|---|---|
| 精确重复 | 1183 |
| schema不合规 | 258 |
| 标注冲突(同输入不同输出) | 79 |
| 原文过短 | 41 |
| 原文过长 | 13 |

## 二、原文长度分布（保留样本）

| 长度区间(字符) | 条数 | 占比 |
|---|---|---|
| 0-50 | 9827 | 53.4% |
| 50-100 | 7013 | 38.1% |
| 100-200 | 1408 | 7.6% |
| 200-400 | 152 | 0.8% |
| 400+ | 12 | 0.1% |

## 三、可选字段缺失率（即目标为 null 的比例）

| 字段 | null 条数 | 占比 |
|---|---|---|
| production_line | 5342 | 29.0% |
| fault_code | 4776 | 25.9% |
| device_model | 3105 | 16.9% |
```

#### 2.6 读这份报告要看什么

| 看什么 | 健康范围 | 不健康说明什么 | 怎么处理 |
|---|---|---|---|
| 丢弃率 | 5%~15% | > 30% 说明上游数据质量崩了，或你的规则太严 | 抽查 `rejected.jsonl` 前 50 条，人工判断是数据问题还是规则问题 |
| 精确重复占丢弃的比例 | 可高（模板化工单本来就重复） | 如果重复率 > 50%，说明数据源本身多样性不足 | 去重后如果不足 3000 条，考虑扩数据源或降低模型尺寸 |
| 标注冲突 | < 1% | > 3% 说明标注规范不明确 | **必须回去修规范**，否则训练目标自相矛盾 |
| 长度分布 | 90% 落在 < 200 字 | 长尾多则显存压力大 | 设 `max_seq_len` 覆盖 P99，超出的截断并记录 |
| 可选字段 null 率 | 15%~35% 正常 | null 率 > 60% 则这个字段模型会学成「永远输出 null」 | 要么补标注，要么这个字段不值得抽 |
| 枚举分布 | 最少类别 ≥ 3% | 某类别 < 1% 时模型基本学不到 | 上采样、合并类别，或该类别走规则 |

#### 2.7 小结

- 清洗的产出不止是干净数据，更重要的是 **`rejected.jsonl` + 统计报告**，它们是你和数据源方沟通的证据；
- **泄漏检查必须做且必须自动化**：一次泄漏就能让你的准确率虚高 5~15 个百分点，而且极难察觉；
- 「标注冲突」这一项如果超标，请立刻停止训练，回去改标注规范——训练是不会帮你解决自相矛盾的。

---

### Step 3：QLoRA 训练 Qwen2.5-7B-Instruct

**目标**：在单张 24GB 显卡上用 QLoRA 微调出抽取模型。本 Step 给**两套等价方案**：手写脚本（看清每个细节）和 LLaMA-Factory（团队标准化）。

#### 3.1 为什么必须自己实现 labels mask

这是初学者最容易做错、而且**做错了还能正常训练出一个看起来不错的模型**的地方。

在 SFT 里，一条样本是 `system + user + assistant`。我们**只希望模型学会生成 assistant 部分**，不希望它去学习预测 system 和 user 的内容。实现方式是把 system/user 对应位置的 `labels` 设成 `-100`（PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`）。

```mermaid
flowchart LR
    subgraph TOK["token 序列"]
        T1["&lt;|im_start|&gt;system<br/>你是华成机电…&lt;|im_end|&gt;"]
        T2["&lt;|im_start|&gt;user<br/>【渠道】…&lt;|im_end|&gt;"]
        T3["&lt;|im_start|&gt;assistant<br/>{&quot;device_model&quot;:…}&lt;|im_end|&gt;"]
    end
    subgraph LAB["labels"]
        L1["-100 -100 -100 …"]
        L2["-100 -100 -100 …"]
        L3["真实 token id（参与 loss）"]
    end
    T1 --> L1
    T2 --> L2
    T3 --> L3
    style L3 fill:#d4f5d4
    style L1 fill:#eeeeee
    style L2 fill:#eeeeee
```

不做 mask 的后果：

| 后果 | 表现 | 为什么难发现 |
|---|---|---|
| loss 被 prompt 主导 | 我们的 system prompt 有 ~320 token，assistant 只有 ~90 token；不 mask 的话 78% 的梯度在学「怎么复述 system prompt」 | loss 曲线照样下降得很漂亮 |
| 模型学会续写 prompt | 线上偶发输出「字段定义：- device_model…」 | 大部分情况下还是输出 JSON，只有 1~3% 的请求翻车 |
| 训练效率浪费 | 同样的 step 数，有效学习量只有 1/4 | 你会误以为「数据不够」而去堆数据 |

#### 3.2 数据整理器（含 labels mask）

`train/collator.py`：

```python
"""SFT 数据整理器：套用 chat template，并把非 assistant 部分的 label 置为 -100。"""
from dataclasses import dataclass
from typing import Any

import torch

IGNORE_INDEX = -100


@dataclass
class ChatMLMaskCollator:
    """按 Qwen2.5 的 ChatML 模板编码对话，只在 assistant 段计算 loss。"""

    tokenizer: Any
    max_len: int = 1024

    def _encode_one(self, messages: list[dict]) -> dict:
        """逐段编码一条对话，返回 input_ids 与 labels。"""
        input_ids: list[int] = []
        labels: list[int] = []

        for msg in messages:
            role, content = msg["role"], msg["content"]
            # 每一段的完整形态： <|im_start|>{role}\n{content}<|im_end|>\n
            seg = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            seg_ids = self.tokenizer(seg, add_special_tokens=False)["input_ids"]

            if role == "assistant":
                # 头部 "<|im_start|>assistant\n" 不参与 loss，内容与 <|im_end|> 参与
                head = f"<|im_start|>{role}\n"
                head_len = len(self.tokenizer(head, add_special_tokens=False)["input_ids"])
                seg_labels = [IGNORE_INDEX] * head_len + seg_ids[head_len:]
            else:
                seg_labels = [IGNORE_INDEX] * len(seg_ids)

            input_ids.extend(seg_ids)
            labels.extend(seg_labels)

        # 截断：从右侧截，保证 assistant 结尾的 <|im_end|> 尽量保留
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]

        return {"input_ids": input_ids, "labels": labels}

    def __call__(self, features: list[dict]) -> dict:
        """批处理：编码 + 右侧 padding + attention_mask。"""
        encoded = [self._encode_one(f["messages"]) for f in features]
        max_len = max(len(e["input_ids"]) for e in encoded)
        pad_id = self.tokenizer.pad_token_id

        batch_input, batch_labels, batch_mask = [], [], []
        for e in encoded:
            pad_n = max_len - len(e["input_ids"])
            batch_input.append(e["input_ids"] + [pad_id] * pad_n)
            batch_labels.append(e["labels"] + [IGNORE_INDEX] * pad_n)
            batch_mask.append([1] * len(e["input_ids"]) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }
```

> **自查方法**（强烈建议做一次，10 秒钟，能省你一整天）：编码一条样本后，把 `labels != -100` 的 token 解码出来打印，**应该恰好等于 assistant 的内容 + `<|im_end|>`**。第 3.5 节的运行命令里包含了这一步。

#### 3.3 QLoRA 训练脚本（逐参数注释）

`train/train_qlora.py`：

```python
"""QLoRA 微调 Qwen2.5-7B-Instruct，用于工单结构化抽取任务。"""
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from collator import ChatMLMaskCollator

# ============================ 可调区 ============================
BASE_MODEL = os.environ.get("BASE_MODEL", "/models/Qwen2.5-7B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/ticket-extract-lora")
TRAIN_FILE = "data/sft/train.jsonl"
DEV_FILE = "data/sft/dev.jsonl"
MAX_LEN = 1024          # 覆盖数据集 P99 长度即可，越大显存越吃紧
SEED = 42
# ==============================================================


def build_model_and_tokenizer():
    """加载 4bit 量化基座与 tokenizer，并挂上 LoRA。"""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # QLoRA 的核心：基座权重 4bit 存储
        bnb_4bit_quant_type="nf4",               # NF4 对正态分布权重的量化误差小于 fp4
        bnb_4bit_use_double_quant=True,          # 对量化常数再量化一次，7B 上省约 0.4GB
        bnb_4bit_compute_dtype=torch.bfloat16,   # 计算仍用 bf16；Ampere 及以上支持
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},                      # 单卡；多卡用 accelerate launch
        attn_implementation="sdpa",              # 有 flash-attn 环境可换 "flash_attention_2"
        trust_remote_code=True,
    )
    # 关掉 use_cache：训练时不需要 KV cache，且与 gradient checkpointing 冲突
    model.config.use_cache = False
    # 为 k-bit 训练做准备：把 LayerNorm 提到 fp32、开启输入梯度等
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=32,                    # 秩。抽取这类"格式学习"任务 16~32 足够；调大主要涨显存不涨效果
        lora_alpha=64,           # 缩放系数，经验上取 2*r
        lora_dropout=0.05,       # 6400 条数据规模下，0.05 足以抑制过拟合
        bias="none",             # 不训练 bias，省参数且效果无差异
        task_type="CAUSAL_LM",
        target_modules=[         # Qwen2.5 的线性层名；把 MLP 也纳入对格式学习有帮助
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tok


def main():
    """执行训练主流程。"""
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    model, tok = build_model_and_tokenizer()

    ds = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": DEV_FILE},
    )
    print(f"train={len(ds['train'])}  dev={len(ds['validation'])}")

    collator = ChatMLMaskCollator(tokenizer=tok, max_len=MAX_LEN)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     # 6400 条 × 3 轮；超过 4 轮通常开始过拟合
        per_device_train_batch_size=2,          # 24G 卡 + 1024 长度的安全值
        gradient_accumulation_steps=8,          # 等效 batch = 2*8 = 16
        per_device_eval_batch_size=2,
        learning_rate=1e-4,                     # LoRA 常用 1e-4 ~ 2e-4，比全参微调高一个量级
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,                      # 前 3% step 线性升温，避免初期梯度爆炸
        weight_decay=0.01,
        max_grad_norm=1.0,                      # 梯度裁剪，防止个别长样本引发 loss 尖刺
        bf16=True,                              # Ampere+ 用 bf16；老卡(如 V100)改成 fp16=True
        gradient_checkpointing=True,            # 用时间换显存，约省 40% 激活显存
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",                  # transformers 4.45 的参数名（旧版叫 evaluation_strategy）
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,            # 训完自动回到 dev loss 最低那个 checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        logging_dir=f"{OUTPUT_DIR}/runs",
        seed=SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,            # 关键：我们的 collator 需要原始 messages 字段
        group_by_length=False,                  # 抽取任务长度差异不大，开了收益有限
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
    )

    train_result = trainer.train()
    trainer.save_model(OUTPUT_DIR)              # 只保存 adapter，约 80MB
    tok.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    metrics.update(trainer.evaluate())
    Path(f"{OUTPUT_DIR}/train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 把训练配置快照存下来，便于复现（版本管理的一部分）
    Path(f"{OUTPUT_DIR}/train_config_snapshot.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "max_len": MAX_LEN, "seed": SEED,
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
        "args": args.to_dict(),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
```

#### 3.4 关键参数对照表

| 参数 | 本项目取值 | 调大会怎样 | 调小会怎样 | 什么时候需要改 |
|---|---|---|---|---|
| `r`（LoRA 秩） | 32 | 显存 +，训练变慢，**格式类任务收益极小** | 16 以下在多字段抽取上可能欠拟合 | 任务更复杂（如需要推理）时升到 64 |
| `lora_alpha` | 64 | 等效放大学习率，易震荡 | 学得慢 | 一般固定为 `2*r`，别单独调 |
| `target_modules` | attn + MLP 全上 | 参数量 ↑ 约 2.5 倍 | 只加 attn 的话格式稳定性略差 | 显存吃紧时先砍 `gate/up/down` |
| `learning_rate` | 1e-4 | 2e-4 以上在 6400 条小数据上易过拟合 | 5e-5 收敛慢，3 轮可能没学够 | 数据量 > 5 万时可降到 5e-5 |
| `num_train_epochs` | 3 | 4 轮以上 dev loss 开始抬头 | 1 轮格式合规率不达标 | 看 dev loss 曲线决定，不要硬凑 |
| 等效 batch | 16 | 32 更稳但一步更慢 | 8 时 loss 抖动明显 | 显存允许优先加 `grad_accum` 而不是 `batch_size` |
| `MAX_LEN` | 1024 | 显存平方级增长 | 截断长工单，长样本抽取质量下降 | 用数据集 P99 长度决定 |

#### 3.5 运行命令

```bash
# 0) 先自查 labels mask 是否正确（务必做）
python - <<'PY'
import json
from transformers import AutoTokenizer
import sys; sys.path.insert(0, "train")
from collator import ChatMLMaskCollator

tok = AutoTokenizer.from_pretrained("/models/Qwen2.5-7B-Instruct", trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
col = ChatMLMaskCollator(tokenizer=tok, max_len=1024)

sample = json.loads(open("data/sft/train.jsonl", encoding="utf-8").readline())
batch = col([sample])
ids = batch["input_ids"][0]; labels = batch["labels"][0]
learned = [int(i) for i, l in zip(ids, labels) if int(l) != -100]
print("=== 参与 loss 的部分 ===")
print(tok.decode(learned))
print("=== 总 token 数 / 参与 loss 的 token 数 ===")
print(len(ids), len(learned), f"{len(learned)/len(ids):.1%}")
PY

# 1) 启动训练
cd train
BASE_MODEL=/models/Qwen2.5-7B-Instruct \
OUTPUT_DIR=../outputs/ticket-extract-lora \
CUDA_VISIBLE_DEVICES=0 \
python train_qlora.py 2>&1 | tee ../outputs/train.log

# 2) 另开一个终端看曲线
tensorboard --logdir outputs/ticket-extract-lora/runs --port 6006
```

#### 3.6 预期输出

labels mask 自查：

```text
=== 参与 loss 的部分 ===
{"device_model":"XC-200-4P","fault_symptom":"电机运行20分钟后报E07外壳过热","fault_code":"E07","urgency":2,"production_line":"三号线","suggested_craft":"电气","need_parts":true,"intent":"报修"}<|im_end|>
=== 总 token 数 / 参与 loss 的 token 数 ===
447 92 20.6%
```

> 如果这里打印出来的是 system prompt 的内容，说明 mask 写错了，**立刻停下来修**，不要开始训练。

训练日志（实测环境：RTX 4090 24G、Qwen2.5-7B-Instruct、6400 条训练样本、max_len=1024，示例性数据）：

```text
trainable params: 80,740,352 || all params: 7,696,356,864 || trainable%: 1.0491
train=6400  dev=800
{'loss': 1.9142, 'grad_norm': 1.83, 'learning_rate': 8.3e-05, 'epoch': 0.03}
{'loss': 0.6831, 'grad_norm': 0.92, 'learning_rate': 9.9e-05, 'epoch': 0.25}
{'loss': 0.2417, 'grad_norm': 0.51, 'learning_rate': 9.4e-05, 'epoch': 0.50}
{'eval_loss': 0.2088, 'eval_runtime': 48.7, 'epoch': 0.50}
{'loss': 0.1604, 'grad_norm': 0.38, 'learning_rate': 8.2e-05, 'epoch': 1.00}
{'eval_loss': 0.1473, 'eval_runtime': 48.4, 'epoch': 1.00}
{'loss': 0.1122, 'grad_norm': 0.31, 'learning_rate': 5.6e-05, 'epoch': 1.75}
{'eval_loss': 0.1218, 'eval_runtime': 48.6, 'epoch': 1.75}
{'loss': 0.0863, 'grad_norm': 0.27, 'learning_rate': 2.4e-05, 'epoch': 2.50}
{'eval_loss': 0.1189, 'eval_runtime': 48.5, 'epoch': 2.50}
{'loss': 0.0741, 'grad_norm': 0.25, 'learning_rate': 2.0e-06, 'epoch': 3.00}
{'eval_loss': 0.1206, 'eval_runtime': 48.3, 'epoch': 3.00}
{'train_runtime': 11642.8, 'train_samples_per_second': 1.65, 'epoch': 3.0}
Loading best model from outputs/ticket-extract-lora/checkpoint-1000 (score: 0.1189).
```

产物：

```bash
ls -lh outputs/ticket-extract-lora/
```

```text
adapter_config.json           1.1K
adapter_model.safetensors     309M
train_metrics.json            892
train_config_snapshot.json    3.4K
tokenizer.json                7.0M
runs/                         (tensorboard)
checkpoint-900/  checkpoint-1000/  checkpoint-1100/
```

> `adapter_model.safetensors` 309MB 是因为 `r=32` 且 target_modules 包含全部 MLP。若只挂 attn 四个投影，`r=16`，adapter 约 40MB。

#### 3.7 LLaMA-Factory 等价方案

团队协作时更推荐用 LLaMA-Factory：配置即代码、支持多卡、内置评估与导出。

先注册数据集，`train/llamafactory/dataset_info.json`：

```json
{
  "huacheng_ticket_extract": {
    "file_name": "train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  },
  "huacheng_ticket_extract_dev": {
    "file_name": "dev.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

训练配置 `train/llamafactory/ticket_extract.yaml`：

```yaml
### 模型
model_name_or_path: /models/Qwen2.5-7B-Instruct
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
# 关键：只在 assistant 段计算 loss，等价于我们手写的 labels mask
mask_history: false
train_on_prompt: false

### 量化（QLoRA）
quantization_bit: 4
quantization_method: bitsandbytes
double_quantization: true
quantization_type: nf4

### 数据
dataset_dir: ../../data/sft
dataset: huacheng_ticket_extract
eval_dataset: huacheng_ticket_extract_dev
template: qwen
cutoff_len: 1024
overwrite_cache: true
preprocessing_num_workers: 8

### 输出
output_dir: ../../outputs/ticket-extract-lora-lf
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

### 训练超参
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
seed: 42

### 验证
eval_strategy: steps
eval_steps: 100
per_device_eval_batch_size: 2
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
```

运行：

```bash
# 安装（建议独立 venv）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics,bitsandbytes]"

# 把数据集注册文件放进去
cp ~/huacheng-trinity/data/sft/*.jsonl data/
cp ~/huacheng-trinity/train/llamafactory/dataset_info.json data/

# 训练
llamafactory-cli train ~/huacheng-trinity/train/llamafactory/ticket_extract.yaml
```

预期输出（关键几行）：

```text
[INFO] trainable params: 80,740,352 || all params: 7,696,356,864 || trainable%: 1.0491
{'loss': 0.2385, 'learning_rate': 9.4e-05, 'epoch': 0.5}
{'eval_loss': 0.2102, 'epoch': 0.5}
...
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     0.1847
  train_runtime            = 3:14:02.11
  train_samples_per_second =      1.649
Figure saved at: ../../outputs/ticket-extract-lora-lf/training_loss.png
```

#### 3.8 两种方案怎么选

| 维度 | 手写脚本 | LLaMA-Factory |
|---|---|---|
| 学习价值 | **高**，labels mask、量化配置全在你眼前 | 低，是黑盒 |
| 出问题时定位 | 容易（代码都是你的） | 需要读它的源码 |
| 多卡 / DeepSpeed | 自己配 accelerate | 内置，改一行 yaml |
| 数据格式支持 | 只支持你写的那种 | alpaca / sharegpt / 多模态都支持 |
| 团队协作 | 每个人的脚本会漂移 | **yaml 进 git，可 review 可复现** |
| 建议 | 第一次务必手写一遍 | 之后团队统一用它 |

#### 3.9 小结

- QLoRA 的三件套：4bit NF4 基座 + bf16 计算 + LoRA adapter，让 7B 训练塞进 24G；
- **labels mask 是本 Step 的灵魂**，做错了不报错、还能训出「看起来能用」的模型，所以必须自查；
- 训练配置快照要落盘，不然三个月后你无法复现这次结果；
- 手写一遍再上 LLaMA-Factory，你才知道 yaml 里每个键在做什么。

---

### Step 4：训练监控与验证

**目标**：看懂 loss 曲线，用测试集做「微调前 vs 微调后」的定量对照，判断这次训练是否可用。

#### 4.1 loss 曲线怎么读

```mermaid
flowchart TB
    subgraph OK["健康曲线"]
        A1["train loss 平滑下降<br/>1.9 → 0.07"]
        A2["eval loss 同步下降<br/>0.21 → 0.119 后走平"]
        A3["最优点在 epoch 2.5 附近<br/>最后一轮略微抬头 0.1206"]
        A1 --> A2 --> A3
    end
    subgraph BAD1["过拟合"]
        B1["train loss 继续降到 0.01"]
        B2["eval loss 从 0.12 涨到 0.19"]
        B3["处理：减 epoch / 加 dropout / 加数据"]
        B1 --> B2 --> B3
    end
    subgraph BAD2["欠拟合"]
        C1["train loss 停在 0.6 不动"]
        C2["处理：升 lr / 升 r / 加 epoch<br/>先检查 labels mask 和数据质量"]
        C1 --> C2
    end
    subgraph BAD3["loss 尖刺 / NaN"]
        D1["某个 step 突然跳到 12 或 NaN"]
        D2["处理：max_grad_norm=1.0、<br/>检查是否有超长脏样本、<br/>fp16 换 bf16"]
        D1 --> D2
    end
    style OK fill:#e8f5e9
    style BAD1 fill:#ffebee
    style BAD2 fill:#fff8e1
    style BAD3 fill:#fce4ec
```

本项目的曲线数据（实测环境同上，示例性数据）：

| step | epoch | train_loss | eval_loss | 解读 |
|---|---|---|---|---|
| 10 | 0.03 | 1.9142 | — | 初始，模型还在适应输出格式 |
| 100 | 0.25 | 0.6831 | 0.5904 | **陡降段**：学会了「只输出 JSON」这件事 |
| 200 | 0.50 | 0.2417 | 0.2088 | 开始学字段内容 |
| 400 | 1.00 | 0.1604 | 0.1473 | 缓降段 |
| 700 | 1.75 | 0.1122 | 0.1218 | 接近收敛 |
| 1000 | 2.50 | 0.0863 | **0.1189** | **最优点** |
| 1200 | 3.00 | 0.0741 | 0.1206 | 轻微过拟合，`load_best_model_at_end` 自动回退到 step 1000 |

> **一个重要提醒**：`eval_loss` 低不等于业务效果好。loss 是 token 级交叉熵，它不知道「把 `E07` 抽成 `E97` 是致命错误、把 `fault_symptom` 的措辞换个说法是无所谓的」。**必须做业务指标评测**，见 4.2。

#### 4.2 微调前后对比脚本

`eval/eval_extract.py`（这个脚本第六节还会再用一次做最终评测）：

```python
"""抽取任务的字段级评测：格式合规率、字段准确率、意图 macro-F1、延迟、token 消耗。"""
import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

FIELDS = ["device_model", "fault_symptom", "fault_code", "urgency",
          "production_line", "suggested_craft", "need_parts", "intent"]
# 精确匹配字段 vs 宽松匹配字段（故障现象是自由文本，用字符级 F1 而非精确匹配）
EXACT_FIELDS = ["device_model", "fault_code", "urgency",
                "production_line", "suggested_craft", "need_parts", "intent"]


def char_f1(a: str, b: str) -> float:
    """两个字符串的字符级 F1，用于评价自由文本字段。"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r)


def parse_json_loose(text: str):
    """尽量从模型输出中提取 JSON 对象，失败返回 None。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


async def call_model(client, base_url, api_key, model, messages, max_tokens=256):
    """调用 OpenAI 兼容端点，返回 (文本, 耗时ms, prompt_tokens, completion_tokens)。"""
    t0 = time.perf_counter()
    r = await client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "temperature": 0.0,
              "max_tokens": max_tokens, "stream": False},
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    dt = (time.perf_counter() - t0) * 1000
    usage = data.get("usage") or {}
    return (data["choices"][0]["message"]["content"], dt,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


async def run(args):
    """跑完整个测试集并返回逐条结果。"""
    rows = [json.loads(l) for l in Path(args.test_file).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async with httpx.AsyncClient() as client:
        async def one(row):
            """处理一条测试样本。"""
            async with sem:
                msgs = row["messages"][:2]        # 只喂 system + user
                gold = json.loads(row["messages"][2]["content"])
                try:
                    text, dt, pt, ct = await call_model(
                        client, args.base_url, args.api_key, args.model, msgs)
                except Exception as e:                      # noqa: BLE001
                    return {"ok": False, "err": str(e), "gold": gold,
                            "latency_ms": 0, "pt": 0, "ct": 0, "channel": row.get("channel")}
                pred = parse_json_loose(text)
                return {"ok": pred is not None, "pred": pred, "gold": gold, "raw": text,
                        "latency_ms": dt, "pt": pt, "ct": ct, "channel": row.get("channel")}

        results = await asyncio.gather(*[one(r) for r in rows])
    return results


def summarize(results, tag):
    """汇总指标并打印表格。"""
    n = len(results)
    n_ok = sum(1 for r in results if r["ok"])
    field_hit = defaultdict(int)
    field_total = defaultdict(int)
    sym_f1_sum = 0.0
    intent_cm = defaultdict(Counter)

    for r in results:
        if not r["ok"]:
            continue
        pred, gold = r["pred"] or {}, r["gold"]
        for f in EXACT_FIELDS:
            field_total[f] += 1
            pv, gv = pred.get(f), gold.get(f)
            if isinstance(gv, str) and isinstance(pv, str):
                pv, gv = pv.strip().upper(), gv.strip().upper()
            if pv == gv:
                field_hit[f] += 1
        sym_f1_sum += char_f1(gold.get("fault_symptom", ""), str(pred.get("fault_symptom") or ""))
        intent_cm[gold.get("intent")][pred.get("intent")] += 1

    lat = sorted(r["latency_ms"] for r in results if r["latency_ms"] > 0)
    p50 = lat[int(len(lat) * 0.50)] if lat else 0
    p95 = lat[int(len(lat) * 0.95)] if lat else 0
    avg_pt = sum(r["pt"] for r in results) / max(n, 1)
    avg_ct = sum(r["ct"] for r in results) / max(n, 1)

    # macro-F1
    labels = set(intent_cm) | {p for c in intent_cm.values() for p in c}
    f1s = []
    for lb in labels:
        tp = intent_cm[lb][lb]
        fp = sum(intent_cm[g][lb] for g in intent_cm if g != lb)
        fn = sum(v for p, v in intent_cm[lb].items() if p != lb)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    print(f"\n===== {tag} （n={n}）=====")
    print(f"JSON 格式合规率 : {n_ok / n:.2%}")
    print(f"意图 macro-F1   : {macro_f1:.4f}")
    print(f"故障现象 charF1 : {sym_f1_sum / max(n_ok, 1):.4f}")
    print(f"延迟 P50/P95    : {p50:.0f} ms / {p95:.0f} ms")
    print(f"平均 token      : prompt={avg_pt:.0f}  completion={avg_ct:.0f}")
    print("\n| 字段 | 准确率 |")
    print("|---|---|")
    accs = []
    for f in EXACT_FIELDS:
        acc = field_hit[f] / max(field_total[f], 1)
        accs.append(acc)
        print(f"| {f} | {acc:.2%} |")
    print(f"\n字段平均准确率（7 个精确字段）: {sum(accs)/len(accs):.2%}")
    return {
        "tag": tag, "n": n, "format_ok": n_ok / n, "macro_f1": macro_f1,
        "symptom_f1": sym_f1_sum / max(n_ok, 1), "p50": p50, "p95": p95,
        "avg_prompt_tokens": avg_pt, "avg_completion_tokens": avg_ct,
        "field_acc": {f: field_hit[f] / max(field_total[f], 1) for f in EXACT_FIELDS},
        "field_acc_avg": sum(accs) / len(accs),
    }


def main():
    """命令行入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="data/sft/test.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="ticket-extract")
    ap.add_argument("--tag", default="finetuned")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    results = asyncio.run(run(args))
    summary = summarize(results, args.tag)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存 -> {args.save}")


if __name__ == "__main__":
    main()
```

#### 4.3 运行命令（三组对照）

```bash
# 组 A：基座模型（未微调）+ 同一套 prompt
python eval/eval_extract.py --base-url http://127.0.0.1:8001/v1 \
  --model Qwen2.5-7B-Instruct --tag "基座7B(未微调)" --save reports/eval_base7b.json

# 组 B：大模型 deepseek-chat + 同一套 prompt
python eval/eval_extract.py --base-url https://api.deepseek.com/v1 \
  --api-key $LLM_API_KEY --model deepseek-chat --concurrency 4 \
  --tag "deepseek-chat(prompt工程)" --save reports/eval_deepseek.json

# 组 C：微调后模型（vLLM，Step 5 部署完成后跑）
python eval/eval_extract.py --base-url http://127.0.0.1:8000/v1 \
  --model ticket-extract --tag "微调7B(QLoRA)" --save reports/eval_finetuned.json
```

#### 4.4 预期输出与对照表

```text
===== 微调7B(QLoRA) （n=800）=====
JSON 格式合规率 : 99.75%
意图 macro-F1   : 0.9312
故障现象 charF1 : 0.8874
延迟 P50/P95    : 412 ms / 736 ms
平均 token      : prompt=447  completion=88

| 字段 | 准确率 |
|---|---|
| device_model | 97.62% |
| fault_code | 98.25% |
| urgency | 91.13% |
| production_line | 95.87% |
| suggested_craft | 93.50% |
| need_parts | 90.62% |
| intent | 93.38% |

字段平均准确率（7 个精确字段）: 94.34%
```

**三组对照总表**（实测环境：RTX 4090 24G 跑 vLLM 0.6.3，测试集 800 条，deepseek-chat 走公网 API，并发 8/4，**示例性数据，你必须自己复现**）：

| 指标 | A. 基座 7B（未微调 + few-shot） | B. deepseek-chat（prompt 工程到位） | C. **微调 7B（QLoRA）** | C vs A | C vs B |
|---|---|---|---|---|---|
| JSON 格式合规率 | 88.50% | 97.75% | **99.75%** | +11.25pp | +2.00pp |
| 字段平均准确率 | 79.16% | 89.71% | **94.34%** | +15.18pp | +4.63pp |
| `device_model` 准确率 | 88.25% | 95.13% | **97.62%** | +9.37pp | +2.49pp |
| `fault_code` 准确率 | 90.38% | 96.00% | **98.25%** | +7.87pp | +2.25pp |
| `urgency` 准确率 | 68.75% | 82.38% | **91.13%** | +22.38pp | +8.75pp |
| `suggested_craft` 准确率 | 71.13% | 85.25% | **93.50%** | +22.37pp | +8.25pp |
| 意图 macro-F1 | 0.7428 | 0.8815 | **0.9312** | +0.188 | +0.050 |
| 延迟 P50 | 508 ms | 1842 ms | **412 ms** | -19% | -78% |
| 延迟 P95 | 1104 ms | 3376 ms | **736 ms** | -33% | -78% |
| 平均 prompt token | 1912（含 6-shot 示例） | 1912 | **447** | -77% | -77% |
| 平均 completion token | 143（含解释文字） | 96 | **88** | -38% | -8% |
| 单次成本（元） | 本地，电费 ≈ 0.00006 | 0.0043 | 本地，电费 ≈ 0.00004 | — | **-99%** |

**从这张表能读出四个结论**：

1. **微调最大的收益不在「平均准确率」，在「主观判断类字段」**。`urgency` 和 `suggested_craft` 这两个需要「按公司内部规范判断」的字段，微调后提升 22 个百分点，因为规范是学不进 prompt 的，只能学进权重。
2. **微调对客观抽取字段（型号、故障码）提升有限**（+7~9pp），这类字段大模型本来就能做对。如果你的任务只有这类字段，**微调的性价比会低很多**。
3. **prompt token 从 1912 降到 447 是省钱的主因**：微调后不需要 6-shot 示例了。这一项在批量回填 28 万条历史工单时直接决定项目能不能做。
4. **B 组（大模型）的绝对准确率并不差**（89.71%）。如果你日调用量只有几百次、延迟不敏感，**B 组是更合理的选择**——这就是第 1.3 节红灯清单存在的意义。

#### 4.5 一个容易被忽略的验证：分渠道看指标

平均数会掩盖问题。把测试集按渠道拆开看：

```bash
python - <<'PY'
import json, subprocess, collections
rows = [json.loads(l) for l in open("data/sft/test.jsonl", encoding="utf-8")]
c = collections.Counter(r["channel"] for r in rows)
print(c)
PY
```

分渠道结果（示例性数据）：

| 渠道 | 测试集条数 | 格式合规率 | 字段平均准确率 | 说明 |
|---|---|---|---|---|
| 400电话 | 401 | 99.75% | 96.38% | 文本规整，最好做 |
| 经销商系统 | 122 | 100.00% | 97.11% | 半结构化，几乎零难度 |
| 小程序 | 277 | 99.64% | **90.02%** | **短板**：信息缺失多、错别字多 |

小程序渠道低 6 个百分点，这才是真正要优化的地方。优化手段不是「继续加 epoch」，而是：
- 在训练集里**上采样**小程序渠道样本（当前占比 34.6%，可提到 45%）；
- 对小程序渠道单独加一条**追问策略**（见 Step 7 的 `need_clarify` 分支）；
- 把「型号缺失」这个高频场景做成独立的处理路径（查客户历史设备）。

#### 4.6 小结

- loss 收敛 ≠ 效果好，**必须跑业务指标评测**；
- 三组对照（基座 / 大模型 / 微调）是证明你工作价值的唯一方式，缺一组说服力就不够；
- 微调真正的护城河是「主观判断类字段」和「prompt token 压缩」，把这两点讲给老板听比讲 loss 有用；
- 看指标要分维度拆（渠道、意图、长度），平均数会骗人。

---

### Step 5：合并 + AWQ 量化 + vLLM 部署

**目标**：把 LoRA adapter 合并进基座，做 AWQ 4bit 量化，用 vLLM 起一个 OpenAI 兼容端点。

#### 5.1 三条部署路线怎么选

```mermaid
flowchart TD
    Q{部署方式} --> R1["路线1：vLLM 动态加载 LoRA<br/>--enable-lora"]
    Q --> R2["路线2：合并后 fp16 部署"]
    Q --> R3["路线3：合并 + AWQ 4bit 部署"]

    R1 --> R1A["优点：一个基座服务多个 adapter<br/>换 adapter 不重启"]
    R1 --> R1B["缺点：吞吐比合并版低约 10~20%<br/>基座仍占 15GB"]
    R2 --> R2A["优点：最简单，无量化精度损失"]
    R2 --> R2B["缺点：7B fp16 占 15GB，<br/>24G 卡只剩 7G 给 KV cache"]
    R3 --> R3A["优点：显存约 6GB，<br/>KV cache 充裕，吞吐最高"]
    R3 --> R3B["缺点：量化有精度损失，<br/>需要重新评测验证"]

    style R3 fill:#d4f5d4
```

本项目选**路线 3**，理由：抽取任务输出短、并发要求高，显存省下来全给 KV cache，吞吐收益最大。但**路线 1 也要会**，因为多业务线共用一张卡时它是唯一解，第 5.6 节给命令。

#### 5.2 合并脚本

`scripts/merge_lora.py`：

```python
"""把 LoRA adapter 合并进基座权重，输出 fp16 完整模型。"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """加载基座（fp16，非量化）→ 挂 adapter → merge_and_unload → 保存。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/ticket-extract-lora")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("加载基座（fp16，CPU）...")
    # 注意：合并必须用未量化的 fp16 基座。用 4bit 基座合并会引入额外量化误差。
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)

    print("挂载 adapter ...")
    model = PeftModel.from_pretrained(base, args.adapter, torch_dtype=torch.float16)

    print("合并中（这一步占内存，7B 约需 32GB 系统内存）...")
    model = model.merge_and_unload()
    model.config.use_cache = True

    print(f"保存到 {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(args.out)

    # 把训练快照一并拷过去，便于追溯这份权重来自哪次训练
    snap = Path(args.adapter) / "train_config_snapshot.json"
    if snap.exists():
        shutil.copy(snap, Path(args.out) / "train_config_snapshot.json")

    print("完成")


if __name__ == "__main__":
    main()
```

#### 5.3 AWQ 量化脚本

AWQ（Activation-aware Weight Quantization）需要一批校准数据。**校准数据必须来自你的真实业务分布**，否则量化后在你的任务上掉点会比想象中多。

`scripts/quantize_awq.py`：

```python
"""用业务数据做校准，对合并后的模型做 AWQ 4bit 量化。"""
import argparse
import json
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_calib_texts(path: str, tokenizer, n: int = 256) -> list[str]:
    """从训练集抽 n 条样本，套 chat template 生成校准文本。"""
    texts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if len(texts) >= n:
            break
        s = json.loads(line)
        texts.append(tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=False))
    return texts


def main():
    """执行 AWQ 量化并保存。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/merged")
    ap.add_argument("--out", default="outputs/merged-awq")
    ap.add_argument("--calib", default="data/sft/train.jsonl")
    ap.add_argument("--calib-n", type=int, default=256)
    args = ap.parse_args()

    quant_config = {
        "zero_point": True,     # 非对称量化，对激活分布偏移的层更友好
        "q_group_size": 128,    # 每 128 个权重共享一组缩放参数；64 更准但更慢更大
        "w_bit": 4,             # 4bit
        "version": "GEMM",      # GEMM 内核，vLLM 支持最好；GEMV 适合 batch=1 场景
    }

    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        args.src, safetensors=True, device_map="cuda:0")

    calib = load_calib_texts(args.calib, tok, args.calib_n)
    print(f"校准样本 {len(calib)} 条，示例长度 {len(calib[0])} 字符")

    model.quantize(tok, quant_config=quant_config, calib_data=calib,
                   max_calib_seq_len=1024)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_quantized(args.out, safetensors=True)
    tok.save_pretrained(args.out)
    print(f"AWQ 量化完成 -> {args.out}")


if __name__ == "__main__":
    main()
```

#### 5.4 运行命令

```bash
# 1) 合并（需要约 32GB 系统内存；内存不够就用 swap 或换云机器）
python scripts/merge_lora.py \
  --base /models/Qwen2.5-7B-Instruct \
  --adapter outputs/ticket-extract-lora \
  --out outputs/merged

du -sh outputs/merged

# 2) AWQ 量化（RTX 4090 约 25~40 分钟）
python scripts/quantize_awq.py \
  --src outputs/merged \
  --out outputs/merged-awq \
  --calib data/sft/train.jsonl --calib-n 256

du -sh outputs/merged-awq

# 3) 用 vLLM 起服务（模型别名叫 ticket-extract，与 .env 一致）
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged-awq \
  --served-model-name ticket-extract \
  --quantization awq \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.55 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --disable-log-requests \
  --port 8000 \
  --host 0.0.0.0
```

#### 5.5 预期输出

```text
# 合并
加载基座（fp16，CPU）...
挂载 adapter ...
合并中（这一步占内存，7B 约需 32GB 系统内存）...
保存到 outputs/merged ...
完成
15G     outputs/merged

# 量化
校准样本 256 条，示例长度 612 字符
AWQ quantization: 100%|████████████| 28/28 [31:47<00:00, 68.1s/it]
AWQ 量化完成 -> outputs/merged-awq
5.6G    outputs/merged-awq

# vLLM 启动关键行
INFO ... Using model weights format ['*.safetensors']
INFO ... Model loading took 5.4321 GB
INFO ... # GPU blocks: 9216, # CPU blocks: 2048
INFO ... Prefix caching is enabled.
INFO ... Started server process
INFO ... Uvicorn running on http://0.0.0.0:8000
```

验证端点：

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ticket-extract",
    "temperature": 0,
    "max_tokens": 256,
    "messages": [
      {"role":"system","content":"你是华成机电售后工单结构化抽取引擎。\n请从工单原文中抽取字段，并严格按下面的 JSON 格式输出，不要输出任何解释、前缀或代码块标记。\n\n字段定义：\n- device_model: 设备型号，如 \"XC-200-4P\"；抽不到填 null\n- fault_symptom: 故障现象，20 字以内的规范化描述，必须填\n- fault_code: 故障码，如 \"E07\"；无则 null\n- urgency: 紧急程度，整数 1(一般)/2(较急)/3(紧急，产线停机或安全风险)\n- production_line: 所属产线，如 \"A2\"；无则 null\n- suggested_craft: 建议工种，取值 \"电气\"/\"机械\"/\"液压\"/\"控制\"/\"其他\"\n- need_parts: 是否需要备件，true 或 false\n- intent: 工单意图，取值 \"咨询\"/\"报修\"/\"投诉\"/\"备件申请\"/\"催办\""},
      {"role":"user","content":"【渠道】400电话\n【工单原文】\n客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。要求两小时内到场。"}
    ]
  }' | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

```text
{"device_model":"BP-22KW","fault_symptom":"变频器上电无显示报E21","fault_code":"E21","urgency":3,"production_line":"B1","suggested_craft":"电气","need_parts":true,"intent":"报修"}
```

#### 5.6 量化前后必须重新评测

量化是有损的。**跳过这一步是本项目最常见的事故来源**。

```bash
# fp16 合并版起在 8002 端口
python -m vllm.entrypoints.openai.api_server --model outputs/merged \
  --served-model-name ticket-extract-fp16 --max-model-len 2048 \
  --gpu-memory-utilization 0.85 --port 8002 &

python eval/eval_extract.py --base-url http://127.0.0.1:8002/v1 \
  --model ticket-extract-fp16 --tag "微调7B-fp16" --save reports/eval_fp16.json
```

量化前后对照（实测环境：RTX 4090 24G，测试集 800 条，vLLM 0.6.3，示例性数据）：

| 指标 | fp16 合并版 | AWQ 4bit 版 | 差值 | 是否可接受 |
|---|---|---|---|---|
| 显存占用（权重） | 15.2 GB | 5.4 GB | -64% | ✅ |
| JSON 格式合规率 | 99.88% | 99.75% | -0.13pp | ✅ |
| 字段平均准确率 | 94.71% | 94.34% | -0.37pp | ✅ 阈值设为 -1.0pp |
| `fault_code` 准确率 | 98.50% | 98.25% | -0.25pp | ✅ |
| 延迟 P50（并发 8） | 476 ms | 412 ms | -13% | ✅ |
| 延迟 P95（并发 8） | 892 ms | 736 ms | -17% | ✅ |
| 吞吐（并发 32，req/s） | 18.4 | 31.7 | **+72%** | ✅ |

> **验收阈值怎么定**：本项目定的是「字段平均准确率下降 ≤ 1.0pp 且关键字段（device_model / fault_code）下降 ≤ 0.5pp」。定阈值的逻辑是：这两个字段错了会导致派错工种或订错备件，有直接经济损失；其他字段错了人工能兜住。**阈值必须跟业务后果挂钩，不能拍脑袋。**

#### 5.7 路线 1：vLLM 动态加载 LoRA（多 adapter 共享基座）

如果你有多个微调任务（抽取、摘要、分类各一个 adapter），用这条路线：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules ticket-extract=outputs/ticket-extract-lora \
                 ticket-summary=outputs/ticket-summary-lora \
  --max-lora-rank 32 \
  --max-loras 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --port 8000
```

调用时把 `model` 字段写成 adapter 名即可：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ticket-extract","messages":[...]}'
```

> 注意：`--max-lora-rank` 必须 ≥ 你训练时的 `r`，否则启动报错。多 adapter 共享基座的吞吐损失约 10~20%（实测环境同上，示例性数据）。

#### 5.8 小结

- 合并必须用 **fp16 基座**，不能用 4bit 基座合并；
- AWQ 校准数据**必须来自业务分布**，随便找 wikitext 校准会让你在自己的任务上多掉 1~3pp；
- **量化后必须重新评测并定阈值**，阈值要能对应到业务后果；
- 多 adapter 场景用 `--enable-lora`，单任务高并发场景用合并 + AWQ。

---

## Part B：LangChain / LCEL + LangGraph 编排

### Step 6：用 LCEL 封装抽取链

**目标**：把「调用微调模型 → 解析 JSON → pydantic 校验 → 失败重试 → 仍失败则降级到大模型」封装成一条可复用的 LCEL 链。

#### 6.1 这条链要处理的四种失败

| 失败类型 | 表现 | 处理策略 |
|---|---|---|
| 网络/服务失败 | vLLM 挂了、超时 | `with_retry`（指数退避 2 次）→ 仍失败则 `with_fallbacks` 到大模型 |
| 格式失败 | 输出不是合法 JSON | 宽松解析（剥代码块、截取 `{...}`）→ 仍失败则降级 |
| schema 失败 | JSON 合法但字段不合规（枚举越界） | pydantic 校验报错 → 降级（大模型对枚举的遵守通常更好） |
| 业务规则失败 | schema 通过但违反业务常识（如「产线停机」却 urgency=1） | 不降级，用**后置规则修正**（代码修，不是模型修） |

```mermaid
flowchart LR
    IN([工单文本]) --> P[prompt 组装]
    P --> SM["小模型调用<br/>ChatOpenAI(base_url=vLLM)<br/>.with_retry(2次)"]
    SM -->|成功| PARSE[宽松 JSON 解析]
    SM -->|重试仍失败| FB
    PARSE -->|解析失败| FB["fallback:<br/>大模型 deepseek-chat<br/>+ JSON mode"]
    PARSE -->|解析成功| VAL["pydantic 校验<br/>TicketExtract"]
    VAL -->|校验失败| FB
    VAL -->|通过| RULE["后置业务规则修正<br/>纯 Python"]
    FB --> PARSE2[宽松 JSON 解析] --> VAL2[pydantic 校验]
    VAL2 -->|通过| RULE
    VAL2 -->|失败| DEG["最终降级：<br/>返回部分结果 + need_human=true"]
    RULE --> OUT([TicketExtract + 元数据])
    DEG --> OUT

    style SM fill:#d4f5d4
    style FB fill:#d4e6f5
    style RULE fill:#eeeeee
    style DEG fill:#ffcccc
```

#### 6.2 模型工厂

`app/llms.py`：

```python
"""模型工厂：统一构造小模型、大模型与带降级的组合模型。"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache
def small_llm(temperature: float = 0.0, max_tokens: int = 256) -> ChatOpenAI:
    """微调后的抽取模型（vLLM OpenAI 兼容端点）。"""
    s = get_settings()
    return ChatOpenAI(
        base_url=s.small_base_url,
        api_key=s.small_api_key,
        model=s.small_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=s.extract_timeout_ms / 1000,
        max_retries=0,          # 重试交给 LCEL 的 with_retry 管理，避免双重重试
    )


@lru_cache
def big_llm(temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False) -> ChatOpenAI:
    """通用大模型，用于方案生成、Agent 推理与降级兜底。"""
    s = get_settings()
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=0,
        **kwargs,
    )
```

#### 6.3 抽取链实现

`app/chains/extract_chain.py`：

```python
"""工单结构化抽取链：小模型优先 + 宽松解析 + pydantic 校验 + 大模型降级 + 规则修正。"""
import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from app.llms import big_llm, small_llm
from app.schemas import TicketExtract

SYSTEM_PROMPT = """你是华成机电售后工单结构化抽取引擎。
请从工单原文中抽取字段，并严格按下面的 JSON 格式输出，不要输出任何解释、前缀或代码块标记。

字段定义：
- device_model: 设备型号，如 "XC-200-4P"；抽不到填 null
- fault_symptom: 故障现象，20 字以内的规范化描述，必须填
- fault_code: 故障码，如 "E07"；无则 null
- urgency: 紧急程度，整数 1(一般)/2(较急)/3(紧急，产线停机或安全风险)
- production_line: 所属产线，如 "A2"；无则 null
- suggested_craft: 建议工种，取值 "电气"/"机械"/"液压"/"控制"/"其他"
- need_parts: 是否需要备件，true 或 false
- intent: 工单意图，取值 "咨询"/"报修"/"投诉"/"备件申请"/"催办"

输出示例：
{{"device_model":"XC-200-4P","fault_symptom":"电机报E07外壳过热","fault_code":"E07","urgency":2,"production_line":"三号线","suggested_craft":"电气","need_parts":true,"intent":"报修"}}"""

USER_TEMPLATE = """【渠道】{channel}
【工单原文】
{raw_text}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_TEMPLATE),
])

# 业务规则关键词：命中则强制提升紧急度
CRITICAL_KEYWORDS = ("停机", "停线", "整条线停", "冒烟", "焦糊", "着火", "触电", "人员受伤")


def loose_json_parse(msg: AIMessage) -> dict[str, Any]:
    """从模型输出里宽松提取 JSON 对象，失败抛 ValueError 触发降级。"""
    text = (msg.content if isinstance(msg, AIMessage) else str(msg)).strip()
    if text.startswith("```"):
        body = text[3:]
        body = body.removeprefix("json").strip()
        text = body.split("```")[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"输出中找不到 JSON 对象: {text[:120]!r}")
    return json.loads(text[start:end + 1])


def to_schema(obj: dict[str, Any]) -> TicketExtract:
    """pydantic 校验，失败抛 ValidationError 触发降级。"""
    return TicketExtract.model_validate(obj)


def apply_business_rules(extract: TicketExtract, raw_text: str) -> TicketExtract:
    """后置业务规则修正：模型判断不了的硬规则用代码兜。"""
    data = extract.model_dump()

    # 规则 1：出现停机/安全类关键词，紧急度强制置 3
    if any(k in raw_text for k in CRITICAL_KEYWORDS):
        data["urgency"] = 3

    # 规则 2：意图是"备件申请"时，need_parts 必为 True
    if data["intent"] == "备件申请":
        data["need_parts"] = True

    # 规则 3：意图是"咨询"且无故障码时，紧急度不得高于 2
    if data["intent"] == "咨询" and not data["fault_code"]:
        data["urgency"] = min(int(data["urgency"]), 2)

    return TicketExtract.model_validate(data)


def build_extract_chain(use_finetuned: bool = True) -> Runnable:
    """构造抽取链。use_finetuned=False 时直接用大模型（无 GPU 路径）。"""
    parse_and_validate = (
        RunnableLambda(loose_json_parse).with_config(run_name="loose_json_parse")
        | RunnableLambda(to_schema).with_config(run_name="pydantic_validate")
    )

    big_branch = (
        PROMPT
        | big_llm(temperature=0.0, max_tokens=512, json_mode=True)
        | parse_and_validate
    ).with_config(run_name="extract_by_big_model")

    if not use_finetuned:
        return big_branch

    small_branch = (
        PROMPT
        | small_llm(temperature=0.0, max_tokens=256).with_retry(
            retry_if_exception_type=(Exception,),
            wait_exponential_jitter=True,
            stop_after_attempt=2,
        )
        | parse_and_validate
    ).with_config(run_name="extract_by_small_model")

    # 小模型链整体失败（网络失败 / 解析失败 / 校验失败）→ 降级到大模型
    return small_branch.with_fallbacks(
        fallbacks=[big_branch],
        exceptions_to_handle=(Exception,),
    ).with_config(run_name="extract_chain")


class ExtractService:
    """对外暴露的抽取服务，附带耗时、降级标记等元数据。"""

    def __init__(self, use_finetuned: bool = True):
        self.chain = build_extract_chain(use_finetuned)
        self.use_finetuned = use_finetuned

    async def aextract(self, raw_text: str, channel: str = "400电话") -> dict[str, Any]:
        """异步抽取，返回 {extract, latency_ms, degraded, error}。"""
        t0 = time.perf_counter()
        degraded, error = False, None
        try:
            result: TicketExtract = await self.chain.ainvoke(
                {"raw_text": raw_text, "channel": channel})
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            # 连大模型都失败：返回最小可用结果，标记需人工
            error = f"{type(e).__name__}: {e}"
            result = TicketExtract(
                device_model=None, fault_symptom=raw_text[:20] or "未知",
                fault_code=None, urgency=2, production_line=None,
                suggested_craft="其他", need_parts=False, intent="报修",
            )
            degraded = True
        else:
            result = apply_business_rules(result, raw_text)

        return {
            "extract": result,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "degraded": degraded,
            "error": error,
        }
```

> **降级是否发生怎么判断**：LCEL 的 `with_fallbacks` 不会直接告诉你走了哪条分支。生产上有两种做法：(1) 接入 Langfuse，trace 里能看到实际执行的 run_name；(2) 在 `small_branch` 末尾挂一个 `RunnableLambda` 打标记。本项目用方式 (1)，第 9.4 节的 tracing 模块会展示。

#### 6.4 运行命令

```bash
python - <<'PY'
import asyncio
from app.chains.extract_chain import ExtractService

svc = ExtractService(use_finetuned=True)

cases = [
    ("400电话", "客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。要求两小时内到场。"),
    ("小程序", "机器坏了 转不动 有异想 急！！！ 昨天买的还在保修吧"),
    ("经销商系统", "DEV_MODEL=RV-90|ERR=F02|LINE=C2|DESC=液压站压力上不去,保压失败|URGENCY=3"),
    ("小程序", "请问XC-300-4P这个型号的端盖螺栓拧多大力矩合适"),
]

async def main():
    for ch, text in cases:
        r = await svc.aextract(text, ch)
        e = r["extract"]
        print(f"\n[{ch}] {text[:30]}...")
        print(f"  -> {e.model_dump_json()}")
        print(f"  耗时 {r['latency_ms']:.0f} ms  降级={r['degraded']}")

asyncio.run(main())
PY
```

#### 6.5 预期输出

```text
[400电话] 客户B1线整条停了，变频器上电无显示，报E21。型号BP-22...
  -> {"device_model":"BP-22KW","fault_symptom":"变频器上电无显示报E21","fault_code":"E21","urgency":3,"production_line":"B1","suggested_craft":"电气","need_parts":true,"intent":"报修"}
  耗时 438 ms  降级=False

[小程序] 机器坏了 转不动 有异想 急！！！ 昨天买的还在保修吧...
  -> {"device_model":null,"fault_symptom":"设备无法转动并有异响","fault_code":null,"urgency":2,"production_line":null,"suggested_craft":"机械","need_parts":false,"intent":"报修"}
  耗时 401 ms  降级=False

[经销商系统] DEV_MODEL=RV-90|ERR=F02|LINE=C2|DESC=液压站压力上不去,...
  -> {"device_model":"RV-90","fault_symptom":"液压站压力不足保压失败","fault_code":"F02","urgency":3,"production_line":"C2","suggested_craft":"液压","need_parts":true,"intent":"报修"}
  耗时 396 ms  降级=False

[小程序] 请问XC-300-4P这个型号的端盖螺栓拧多大力矩合适...
  -> {"device_model":"XC-300-4P","fault_symptom":"咨询端盖螺栓紧固扭矩","fault_code":null,"urgency":1,"production_line":null,"suggested_craft":"机械","need_parts":false,"intent":"咨询"}
  耗时 384 ms  降级=False
```

验证降级链（**这一条务必亲手跑一遍**）：

```bash
# 故意把 vLLM 停掉
pkill -f "vllm.entrypoints.openai.api_server"

# 再跑一次上面的脚本
python - <<'PY'
import asyncio
from app.chains.extract_chain import ExtractService
svc = ExtractService(use_finetuned=True)
async def main():
    r = await svc.aextract("客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。", "400电话")
    print(r["extract"].model_dump_json())
    print(f"耗时 {r['latency_ms']:.0f} ms  降级={r['degraded']}")
asyncio.run(main())
PY
```

```text
{"device_model":"BP-22KW","fault_symptom":"变频器上电无显示报E21","fault_code":"E21","urgency":3,"production_line":"B1","suggested_craft":"电气","need_parts":true,"intent":"报修"}
耗时 4213 ms  降级=False
```

> 注意：`degraded=False` 是因为 fallback 到大模型**成功了**——这是设计预期。耗时从 438ms 涨到 4213ms（含 2 次重试的退避时间），这正是降级的代价。`degraded=True` 只在「连大模型都失败」时出现。

#### 6.6 小结

- LCEL 的 `with_retry` + `with_fallbacks` 让降级链写起来只有两行，但**要区分"重试"和"降级"的边界**：网络抖动重试，格式/schema 错误直接降级（重试同一个模型大概率还是错）；
- **业务硬规则用代码写，不要塞进 prompt**：`if "停机" in text: urgency = 3` 这一行的可靠性是 100%，模型永远做不到；
- 降级链必须**主动演练**（`pkill` 一下），没演练过的降级等于没有。

---

### Step 7：用 LangGraph 编排主流程

**目标**：把「工单进入 → 抽取 → 校验 → 路由 → 检索 → 生成 → 是否行动 → 输出」串成一张有状态、可中断、可重放的图。

#### 7.1 状态图设计

```mermaid
stateDiagram-v2
    [*] --> preprocess
    preprocess --> extract : 脱敏/归一化完成
    extract --> validate
    validate --> route : schema 通过
    validate --> clarify : 关键字段缺失且渠道=小程序

    clarify --> finalize : 生成追问话术，等待补充

    route --> retrieve : 意图=报修/咨询
    route --> check_action : 意图=备件申请
    route --> finalize : 意图=投诉/催办（转人工队列）

    retrieve --> generate
    generate --> check_action

    check_action --> agent : need_action = true
    check_action --> finalize : need_action = false

    agent --> finalize

    finalize --> [*]

    note right of extract
        微调小模型
        失败降级大模型
    end note
    note right of generate
        通用大模型 + RAG 上下文
    end note
    note right of agent
        Agent 子图（Step 8）
        含人工确认中断
    end note
```

#### 7.2 状态定义

`app/graph/state.py`：

```python
"""主流程状态定义。LangGraph 的 state 是整个系统的单一事实来源。"""
import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


class TicketState(TypedDict, total=False):
    """一张工单在图中流转时携带的全部信息。"""

    # --- 输入 ---
    ticket_id: str
    channel: str
    raw_text: str
    customer_id: Optional[str]

    # --- preprocess 产出 ---
    clean_text: str
    pii_masked: bool

    # --- extract 产出 ---
    extract: Optional[dict[str, Any]]      # TicketExtract.model_dump()
    extract_degraded: bool
    extract_latency_ms: float

    # --- validate / route 产出 ---
    valid: bool
    missing_fields: list[str]
    route: Literal["retrieve", "check_action", "clarify", "finalize"]

    # --- retrieve 产出 ---
    docs: list[dict[str, Any]]             # [{id, text, source, score}]
    retrieve_latency_ms: float

    # --- generate 产出 ---
    solution: str
    citations: list[str]

    # --- action 判定与 Agent ---
    need_action: bool
    action_reason: str
    agent_result: Optional[dict[str, Any]]
    pending_confirm: Optional[dict[str, Any]]

    # --- clarify ---
    clarify_question: Optional[str]

    # --- 输出 ---
    final: Optional[dict[str, Any]]

    # --- 追加型字段：用 operator.add 让多节点可以累积写入 ---
    trace: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    cost_items: Annotated[list[dict[str, Any]], operator.add]
```

> **关于 `Annotated[list, operator.add]`**：LangGraph 默认用「后写覆盖先写」的方式更新 state。对于 `trace`、`errors`、`cost_items` 这类需要**累积**的字段，必须用 reducer 声明为可加，否则后面的节点会把前面的记录冲掉。这是 LangGraph 最常见的坑之一。

#### 7.3 节点实现

`app/graph/nodes.py`：

```python
"""主流程各节点实现。每个节点是一个 async 函数：state -> 部分 state 更新。"""
import re
import time
from typing import Any

from app.chains.extract_chain import ExtractService
from app.chains.generate_chain import generate_solution
from app.retrieval.hybrid import hybrid_search
from app.routing.model_router import estimate_cost

RE_PHONE = re.compile(r"1[3-9]\d{9}")
RE_COMPANY = re.compile(r"[一-龥]{2,12}(有限公司|股份有限公司|厂|集团)")

_extract_service = ExtractService(use_finetuned=True)

# 需要触发行动的条件
ACTION_INTENTS = {"备件申请"}
ACTION_KEYWORDS = ("库存", "有货", "调过来", "派人", "上门", "保修", "免费", "多久到")


def _step(name: str, t0: float, **extra) -> dict[str, Any]:
    """构造一条 trace 记录。"""
    return {"node": name, "latency_ms": (time.perf_counter() - t0) * 1000, **extra}


async def preprocess(state: dict) -> dict:
    """脱敏、空白归一化、渠道标注。纯 Python，无模型调用。"""
    t0 = time.perf_counter()
    text = state["raw_text"]
    masked = bool(RE_PHONE.search(text) or RE_COMPANY.search(text))
    text = RE_PHONE.sub("[PHONE]", text)
    text = RE_COMPANY.sub("[COMPANY]", text)
    text = re.sub(r"[ \t　]+", " ", text).strip()
    return {
        "clean_text": text,
        "pii_masked": masked,
        "trace": [_step("preprocess", t0, pii_masked=masked)],
    }


async def extract(state: dict) -> dict:
    """调用抽取链（微调小模型，失败降级大模型）。"""
    t0 = time.perf_counter()
    r = await _extract_service.aextract(state["clean_text"], state.get("channel", "400电话"))
    ex = r["extract"].model_dump()
    return {
        "extract": ex,
        "extract_degraded": r["degraded"],
        "extract_latency_ms": r["latency_ms"],
        "trace": [_step("extract", t0, degraded=r["degraded"], model="ticket-extract")],
        "cost_items": [estimate_cost("extract", "small", prompt_tokens=447, completion_tokens=88)],
        "errors": [r["error"]] if r["error"] else [],
    }


async def validate(state: dict) -> dict:
    """检查关键字段是否齐备，决定是否需要追问。"""
    t0 = time.perf_counter()
    ex = state.get("extract") or {}
    missing = []
    # 报修类工单没有型号时，后续检索和派单都做不准
    if ex.get("intent") == "报修" and not ex.get("device_model"):
        missing.append("device_model")
    if not ex.get("fault_symptom"):
        missing.append("fault_symptom")
    return {
        "valid": not missing,
        "missing_fields": missing,
        "trace": [_step("validate", t0, missing=missing)],
    }


def route_edge(state: dict) -> str:
    """条件边：根据意图与字段完整度决定下一步。"""
    ex = state.get("extract") or {}
    intent = ex.get("intent", "报修")

    # 小程序渠道 + 关键字段缺失 → 先追问（其他渠道走兜底流程，不打扰客服）
    if state.get("missing_fields") and state.get("channel") == "小程序":
        return "clarify"
    if intent in ("投诉", "催办"):
        return "finalize"
    if intent == "备件申请":
        return "check_action"
    return "retrieve"


async def clarify(state: dict) -> dict:
    """关键字段缺失时生成一句追问话术（用小模型即可，这里用模板保证零成本）。"""
    t0 = time.perf_counter()
    missing = state.get("missing_fields", [])
    q_map = {
        "device_model": "方便提供一下设备型号吗？型号一般在电机铭牌或控制柜门内侧，例如 XC-200-4P。",
        "fault_symptom": "能再具体描述一下故障现象吗？比如有没有报警代码、异响还是不启动。",
    }
    question = " ".join(q_map[m] for m in missing if m in q_map) or "能再补充一些信息吗？"
    return {
        "clarify_question": question,
        "trace": [_step("clarify", t0, missing=missing)],
    }


async def retrieve(state: dict) -> dict:
    """混合检索：用抽取出的结构化字段做过滤 + 语义检索。"""
    t0 = time.perf_counter()
    ex = state.get("extract") or {}
    query_parts = [ex.get("fault_symptom") or state["clean_text"]]
    if ex.get("fault_code"):
        query_parts.append(ex["fault_code"])
    if ex.get("device_model"):
        query_parts.append(ex["device_model"])
    query = " ".join(query_parts)

    docs = await hybrid_search(
        query=query,
        filters={"device_model": ex.get("device_model")} if ex.get("device_model") else None,
        top_k=5,
    )
    return {
        "docs": docs,
        "retrieve_latency_ms": (time.perf_counter() - t0) * 1000,
        "trace": [_step("retrieve", t0, n_docs=len(docs), query=query)],
    }


async def generate(state: dict) -> dict:
    """用大模型基于检索结果生成维修方案，必须带引用。"""
    t0 = time.perf_counter()
    r = await generate_solution(
        extract=state.get("extract") or {},
        docs=state.get("docs") or [],
        raw_text=state["clean_text"],
    )
    return {
        "solution": r["solution"],
        "citations": r["citations"],
        "trace": [_step("generate", t0, model="deepseek-chat", n_cite=len(r["citations"]))],
        "cost_items": [estimate_cost("generate", "big",
                                     prompt_tokens=r["prompt_tokens"],
                                     completion_tokens=r["completion_tokens"])],
    }


async def check_action(state: dict) -> dict:
    """判断是否需要进入 Agent 执行动作。用规则，不用模型。"""
    t0 = time.perf_counter()
    ex = state.get("extract") or {}
    text = state["clean_text"]
    reasons = []

    if ex.get("intent") in ACTION_INTENTS:
        reasons.append(f"意图={ex['intent']}")
    if ex.get("need_parts"):
        reasons.append("need_parts=true")
    hit_kw = [k for k in ACTION_KEYWORDS if k in text]
    if hit_kw:
        reasons.append(f"命中关键词{hit_kw}")
    if int(ex.get("urgency", 1)) >= 3:
        reasons.append("urgency=3 需派单")

    need = bool(reasons)
    return {
        "need_action": need,
        "action_reason": "；".join(reasons) if reasons else "无需行动",
        "trace": [_step("check_action", t0, need_action=need, reason=reasons)],
    }


def action_edge(state: dict) -> str:
    """条件边：是否进入 Agent 子图。"""
    return "agent" if state.get("need_action") else "finalize"


async def finalize(state: dict) -> dict:
    """组装最终输出：结构化字段 + 方案 + 已执行动作 + 成本 + 置信度。"""
    t0 = time.perf_counter()
    ex = state.get("extract") or {}
    total_cost = sum(c["cost_yuan"] for c in state.get("cost_items", []))
    total_ms = sum(t["latency_ms"] for t in state.get("trace", []))

    # 置信度：降级、缺字段、无引用都会扣分
    conf = 1.0
    if state.get("extract_degraded"):
        conf -= 0.3
    if state.get("missing_fields"):
        conf -= 0.2 * len(state["missing_fields"])
    if state.get("route") != "clarify" and not state.get("citations") and state.get("solution"):
        conf -= 0.2
    conf = max(0.0, round(conf, 2))

    final = {
        "ticket_id": state.get("ticket_id"),
        "extract": ex,
        "solution": state.get("solution"),
        "citations": state.get("citations", []),
        "clarify_question": state.get("clarify_question"),
        "agent_result": state.get("agent_result"),
        "need_human": conf < 0.6 or state.get("extract_degraded", False),
        "confidence": conf,
        "cost_yuan": round(total_cost, 6),
        "latency_ms": round(total_ms, 1),
        "errors": state.get("errors", []),
    }
    return {"final": final, "trace": [_step("finalize", t0, confidence=conf)]}
```

配套的方案生成链 `app/chains/generate_chain.py`：

```python
"""基于检索结果生成维修方案，强制带引用编号。"""
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.llms import big_llm

SYSTEM = """你是华成机电的资深售后工程师。请根据【参考资料】为工单给出处理建议。

硬性要求：
1. 只使用参考资料中的信息，资料里没有的**绝对不要编造**，写"资料中未覆盖，建议现场确认"。
2. 每条建议后面用 [1] [2] 标注来源编号。
3. 输出结构固定为三段：初步判断 / 处理步骤 / 注意事项。
4. 处理步骤不超过 5 步，每步一行，面向现场工程师，写具体动作。
5. 全文不超过 400 字。"""

USER = """【工单信息】
设备型号：{device_model}
故障现象：{fault_symptom}
故障码：{fault_code}
紧急程度：{urgency}
工单原文：{raw_text}

【参考资料】
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])


def _format_docs(docs: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """把检索结果编号化，返回 (上下文文本, 引用列表)。"""
    lines, cites = [], []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] 来源：{d.get('source', '未知')}\n{d.get('text', '')}")
        cites.append(f"[{i}] {d.get('source', '未知')}")
    return "\n\n".join(lines) if lines else "（无相关资料）", cites


async def generate_solution(extract: dict, docs: list[dict], raw_text: str) -> dict:
    """生成方案，返回 {solution, citations, prompt_tokens, completion_tokens}。"""
    context, cites = _format_docs(docs)
    chain = PROMPT | big_llm(temperature=0.2, max_tokens=800)
    msg = await chain.ainvoke({
        "device_model": extract.get("device_model") or "未知",
        "fault_symptom": extract.get("fault_symptom") or "",
        "fault_code": extract.get("fault_code") or "无",
        "urgency": extract.get("urgency", 1),
        "raw_text": raw_text,
        "context": context,
    })
    usage = (msg.response_metadata or {}).get("token_usage", {})
    return {
        "solution": msg.content,
        "citations": cites,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
```

#### 7.4 主图组装

`app/graph/main_graph.py`：

```python
"""主流程 LangGraph 组装。"""
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.agent_subgraph import build_agent_subgraph
from app.graph.nodes import (
    action_edge, check_action, clarify, extract, finalize,
    generate, preprocess, retrieve, route_edge, validate,
)
from app.graph.state import TicketState


def build_main_graph(checkpointer=None):
    """构造并编译主图。传入 checkpointer 才能支持中断恢复。"""
    g = StateGraph(TicketState)

    g.add_node("preprocess", preprocess)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("clarify", clarify)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("check_action", check_action)
    g.add_node("agent", build_agent_subgraph())    # 子图作为一个节点挂进来
    g.add_node("finalize", finalize)

    g.add_edge(START, "preprocess")
    g.add_edge("preprocess", "extract")
    g.add_edge("extract", "validate")

    g.add_conditional_edges(
        "validate",
        route_edge,
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
            "check_action": "check_action",
            "finalize": "finalize",
        },
    )

    g.add_edge("clarify", "finalize")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "check_action")

    g.add_conditional_edges(
        "check_action",
        action_edge,
        {"agent": "agent", "finalize": "finalize"},
    )

    g.add_edge("agent", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


async def get_app(db_path: str = "checkpoints.sqlite"):
    """返回带 SQLite checkpointer 的图实例（人工确认中断必需）。"""
    saver_cm = AsyncSqliteSaver.from_conn_string(db_path)
    saver = await saver_cm.__aenter__()
    return build_main_graph(checkpointer=saver), saver_cm
```

图导出脚本 `app/graph_export.py`：

```python
"""把主图导出为 mermaid 文本和 PNG，用于文档与评审。"""
from pathlib import Path

from app.graph.main_graph import build_main_graph


def main():
    """导出图结构。"""
    app = build_main_graph()
    graph = app.get_graph()

    mermaid = graph.draw_mermaid()
    Path("reports/main_graph.mmd").parent.mkdir(parents=True, exist_ok=True)
    Path("reports/main_graph.mmd").write_text(mermaid, encoding="utf-8")
    print(mermaid)

    try:
        png = graph.draw_mermaid_png()
        Path("reports/main_graph.png").write_bytes(png)
        print("PNG 已导出 -> reports/main_graph.png")
    except Exception as e:                       # noqa: BLE001
        # draw_mermaid_png 需要联网调 mermaid.ink，离线环境会失败，不影响主流程
        print(f"PNG 导出跳过（离线环境属正常）：{e}")


if __name__ == "__main__":
    main()
```

#### 7.5 运行命令

```bash
# 导出图结构，确认节点与边正确
python -m app.graph_export

# 跑一张普通工单
python - <<'PY'
import asyncio, json
from app.graph.main_graph import build_main_graph

app = build_main_graph()

async def main():
    state = await app.ainvoke({
        "ticket_id": "TK-2025-000001",
        "channel": "400电话",
        "raw_text": "客户反映电机运行20分钟后报E07，外壳烫手，风扇在转。型号XC-200-4P。三号线。",
        "trace": [], "errors": [], "cost_items": [],
    })
    print(json.dumps(state["final"], ensure_ascii=False, indent=2))
    print("\n--- 轨迹 ---")
    for t in state["trace"]:
        print(f"  {t['node']:<14} {t['latency_ms']:>7.1f} ms  {  {k:v for k,v in t.items() if k not in ('node','latency_ms')} }")

asyncio.run(main())
PY
```

#### 7.6 预期输出

```text
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        preprocess(preprocess)
        extract(extract)
        validate(validate)
        clarify(clarify)
        retrieve(retrieve)
        generate(generate)
        check_action(check_action)
        agent(agent)
        finalize(finalize)
        __end__([<p>__end__</p>]):::last
        __start__ --> preprocess;
        preprocess --> extract;
        extract --> validate;
        validate -. &nbsp;clarify&nbsp; .-> clarify;
        validate -. &nbsp;retrieve&nbsp; .-> retrieve;
        validate -. &nbsp;check_action&nbsp; .-> check_action;
        validate -. &nbsp;finalize&nbsp; .-> finalize;
        clarify --> finalize;
        retrieve --> generate;
        generate --> check_action;
        check_action -. &nbsp;agent&nbsp; .-> agent;
        check_action -. &nbsp;finalize&nbsp; .-> finalize;
        agent --> finalize;
        finalize --> __end__;
PNG 导出跳过（离线环境属正常）：...
```

工单处理结果：

```text
{
  "ticket_id": "TK-2025-000001",
  "extract": {
    "device_model": "XC-200-4P",
    "fault_symptom": "电机运行20分钟后报E07外壳过热",
    "fault_code": "E07",
    "urgency": 2,
    "production_line": "三号线",
    "suggested_craft": "电气",
    "need_parts": true,
    "intent": "报修"
  },
  "solution": "初步判断：\nXC-200-4P 报 E07 为过载/过热保护动作，结合外壳烫手且风扇正常，优先怀疑三相电流不平衡或负载超额 [1][2]。\n\n处理步骤：\n1. 断电后用钳形表测量三相运行电流，记录 A/B/C 三相数值 [1]\n2. 若某相电流偏高超过 15%，检查该相接触器触点是否氧化、接线端子是否松动 [2]\n3. 检查散热风道有无粉尘堵塞，清理散热片 [1]\n4. 核对负载是否超出铭牌额定值，必要时降载运行 [1]\n5. 以上排除后仍报警，检查热保护继电器整定值是否偏低 [3]\n\n注意事项：\n- 测量前必须断电挂牌，外壳温度高时先自然冷却 [2]\n- 若发现触点氧化，建议整体更换接触器而非打磨 [2]",
  "citations": [
    "[1] XC-200系列产品说明书 V2.1 第4.3节 故障代码表",
    "[2] 维修手册-电机过热类故障处理 V1.8",
    "[3] 内部Wiki-E07报警处理经验汇总"
  ],
  "clarify_question": null,
  "agent_result": {...},
  "need_human": false,
  "confidence": 1.0,
  "cost_yuan": 0.004182,
  "latency_ms": 3874.2
}

--- 轨迹 ---
  preprocess          0.4 ms  {'pii_masked': False}
  extract           438.1 ms  {'degraded': False, 'model': 'ticket-extract'}
  validate            0.1 ms  {'missing': []}
  retrieve          312.7 ms  {'n_docs': 5, 'query': '电机运行20分钟后报E07外壳过热 E07 XC-200-4P'}
  generate         2891.3 ms  {'model': 'deepseek-chat', 'n_cite': 3}
  check_action        0.2 ms  {'need_action': True, 'reason': ['need_parts=true']}
  agent             229.8 ms  {...}
  finalize            1.6 ms  {'confidence': 1.0}
```

#### 7.7 为什么这些逻辑要用 LangGraph 而不是一串 `await`

| 能力 | 一串 `await` | LangGraph |
|---|---|---|
| 条件分支 | 用 `if/else`，能写但散落各处 | 集中在 `add_conditional_edges`，一眼看清所有路径 |
| 状态快照 | 自己维护 dict，容易漏 | 每个节点前后自动 checkpoint |
| **中断 + 恢复** | 几乎无法实现（要自己做持久化 + 恢复点） | `interrupt_before` + checkpointer，两行 |
| 从中间节点重放 | 不可能 | `update_state` + 指定 `checkpoint_id` |
| 流式输出中间状态 | 自己写 generator | `astream(stream_mode="updates")` |
| 可视化 | 无 | `draw_mermaid()` |
| 并行节点 | `asyncio.gather` 手写 | 多条边指向同一节点自动并行 |

**判断标准**：如果你的流程有「人工确认」或「需要从失败的那一步继续」，就必须用 LangGraph；如果是纯线性且无中断，LCEL 一条链就够了，别过度设计。

#### 7.8 小结

- state 是单一事实来源，累积型字段一定要用 `Annotated[list, operator.add]`；
- 路由用**规则函数**（`route_edge` / `action_edge`），不要用模型判断——又快又能审计；
- 子图（Agent）作为一个普通节点挂进主图，主图不关心它内部怎么循环；
- `trace` / `cost_items` 从第一个节点就开始记，不要等上线出问题了再补埋点。

---

## Part C：Agent 与服务化

### Step 8：Agent 子图实现

**目标**：实现一个带工具、带迭代上限、带人工确认中断的 ReAct 子图。

#### 8.1 工具设计原则（复用 6.2 章）

| 原则 | 本项目体现 |
|---|---|
| 工具描述写给模型看，不是写给人看 | 每个工具的 docstring 写清楚「什么时候用」「参数哪来」「返回什么」 |
| 参数用 pydantic 约束 | 型号、备件号都有正则校验，不合法直接报错让模型重试 |
| **只读工具和写工具要分级** | `check_stock`/`check_warranty` 只读免确认；`create_subticket`/`dispatch` 写操作需确认 |
| 工具要返回结构化结果 + 人话总结 | 返回 dict，同时带 `summary` 字段供模型阅读 |
| 失败要返回可操作的错误 | 「型号不存在」要说清楚，不要抛裸异常 |

#### 8.2 工具实现

`app/tools/registry.py`：

```python
"""工具注册表：集中声明工具、风险等级与是否需要人工确认。"""
from dataclasses import dataclass
from typing import Callable

from langchain_core.tools import BaseTool


@dataclass
class ToolSpec:
    """工具元信息：风险等级决定是否需要人工确认。"""

    tool: BaseTool
    risk: str            # "read" | "write" | "critical"
    need_confirm: bool
    description_cn: str


REGISTRY: dict[str, ToolSpec] = {}


def register(risk: str, need_confirm: bool, description_cn: str) -> Callable:
    """装饰器：把一个 LangChain 工具注册进表。"""
    def deco(tool_obj: BaseTool) -> BaseTool:
        REGISTRY[tool_obj.name] = ToolSpec(
            tool=tool_obj, risk=risk, need_confirm=need_confirm,
            description_cn=description_cn)
        return tool_obj
    return deco


def all_tools() -> list[BaseTool]:
    """返回全部工具对象，供 bind_tools 使用。"""
    return [spec.tool for spec in REGISTRY.values()]


def needs_confirm(tool_name: str) -> bool:
    """查询某工具是否需要人工确认。"""
    spec = REGISTRY.get(tool_name)
    return bool(spec and spec.need_confirm)
```

`app/tools/stock.py`：

```python
"""备件库存查询工具。"""
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.tools.registry import register


class StockInput(BaseModel):
    """查库存的入参。"""

    part_no: str = Field(..., description="备件编号，如 CJ20-40；必须是准确编号，不要传中文名称")
    warehouse: str | None = Field(None, description="仓库代码，如 WH-HD01；不传则查全部仓库")

    @field_validator("part_no")
    @classmethod
    def check_part_no(cls, v: str) -> str:
        """备件号统一大写去空格。"""
        v = v.strip().upper().replace(" ", "")
        if len(v) < 2:
            raise ValueError("备件编号太短，请提供完整编号")
        return v


@register(risk="read", need_confirm=False, description_cn="查询备件库存与可调拨数量")
@tool("check_stock", args_schema=StockInput)
async def check_stock(part_no: str, warehouse: str | None = None) -> dict:
    """查询某个备件在各仓库的库存与可调拨数量。

    什么时候用：客户或工程师询问备件是否有货、能否今天发货、附近仓库有没有时使用。
    参数来源：part_no 来自工单抽取结果或维修方案中提到的备件编号。
    返回：各仓库的现货数量、在途数量、预计到货时间。
    """
    s = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{s.erp_base_url}/stock",
                        params={"part_no": part_no, "warehouse": warehouse})
        r.raise_for_status()
        data = r.json()

    if not data.get("items"):
        return {"found": False, "part_no": part_no,
                "summary": f"未查到备件 {part_no}，请确认编号是否正确。"}

    total = sum(i["qty"] for i in data["items"])
    best = max(data["items"], key=lambda i: i["qty"])
    return {
        "found": True,
        "part_no": part_no,
        "items": data["items"],
        "total_qty": total,
        "summary": (f"{part_no} 全网现货 {total} 件，"
                    f"库存最多的是 {best['warehouse_name']}（{best['qty']} 件，"
                    f"距客户 {best.get('distance_km', '?')} 公里，"
                    f"预计 {best.get('eta_hours', '?')} 小时可达）。"),
    }
```

`app/tools/warranty.py`：

```python
"""保修状态判定工具。"""
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.registry import register


class WarrantyInput(BaseModel):
    """判保修的入参。"""

    customer_id: str = Field(..., description="客户编号，从工单上下文获取")
    device_model: str = Field(..., description="设备型号，如 XC-200-4P")
    serial_no: str | None = Field(None, description="设备序列号，有则优先用序列号判定")


@register(risk="read", need_confirm=False, description_cn="判定设备是否在保修期内")
@tool("check_warranty", args_schema=WarrantyInput)
async def check_warranty(customer_id: str, device_model: str, serial_no: str | None = None) -> dict:
    """判定某台设备当前是否在保修期内，以及保修覆盖范围。

    什么时候用：客户询问是否收费、是否免费更换、保修期还剩多久时使用。
    参数来源：customer_id 来自工单，device_model 来自抽取结果。
    返回：是否在保、到期日、覆盖范围、不覆盖项。
    """
    s = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"{s.crm_base_url}/warranty", params={
            "customer_id": customer_id, "device_model": device_model, "serial_no": serial_no})
        r.raise_for_status()
        d = r.json()

    if not d.get("found"):
        return {"found": False,
                "summary": f"未查到客户 {customer_id} 名下型号 {device_model} 的设备档案，需人工核实。"}

    status = "在保" if d["in_warranty"] else "已过保"
    return {
        "found": True,
        "in_warranty": d["in_warranty"],
        "expire_date": d.get("expire_date"),
        "coverage": d.get("coverage", []),
        "exclusions": d.get("exclusions", []),
        "summary": (f"{device_model} 当前{status}，保修到期日 {d.get('expire_date', '未知')}；"
                    f"覆盖：{'、'.join(d.get('coverage', [])) or '无'}；"
                    f"不覆盖：{'、'.join(d.get('exclusions', [])) or '无'}。"),
    }
```

`app/tools/ticket_ops.py`：

```python
"""工单操作工具：建子工单与派单。两者都是写操作，派单为高风险。"""
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.registry import register


class SubTicketInput(BaseModel):
    """建子工单入参。"""

    parent_ticket_id: str = Field(..., description="主工单号")
    task_type: str = Field(..., description="子任务类型：备件调拨/现场检修/远程指导/退换货")
    summary: str = Field(..., description="子任务说明，30 字以内")
    part_no: str | None = Field(None, description="涉及备件编号，无则不填")


class DispatchInput(BaseModel):
    """派单入参。"""

    ticket_id: str = Field(..., description="要派的工单号")
    craft: str = Field(..., description="工种：电气/机械/液压/控制")
    urgency: int = Field(..., ge=1, le=3, description="紧急度 1/2/3，决定 SLA 时限")
    region: str = Field(..., description="客户所在区域，如 华东-苏州")
    remark: str = Field("", description="给工程师的备注，100 字以内")


@register(risk="write", need_confirm=False, description_cn="创建子工单（内部流程，风险低）")
@tool("create_subticket", args_schema=SubTicketInput)
async def create_subticket(parent_ticket_id: str, task_type: str,
                           summary: str, part_no: str | None = None) -> dict:
    """在主工单下创建一个子工单，用于跟踪备件调拨或二次上门等子任务。

    什么时候用：确认需要调拨备件、需要安排二次上门、需要转其他部门处理时使用。
    参数来源：parent_ticket_id 来自当前工单上下文。
    返回：子工单号与状态。
    """
    s = get_settings()
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(f"{s.wo_base_url}/subticket", json={
            "parent_ticket_id": parent_ticket_id, "task_type": task_type,
            "summary": summary, "part_no": part_no})
        r.raise_for_status()
        d = r.json()
    return {"ok": True, "subticket_id": d["subticket_id"],
            "summary": f"已创建子工单 {d['subticket_id']}（{task_type}）。"}


@register(risk="critical", need_confirm=True,
          description_cn="派单给现场工程师（高风险：会触发短信通知与差旅成本）")
@tool("dispatch", args_schema=DispatchInput)
async def dispatch(ticket_id: str, craft: str, urgency: int,
                   region: str, remark: str = "") -> dict:
    """把工单派给指定区域的对应工种工程师，并按紧急度设定 SLA 时限。

    什么时候用：确认需要工程师上门，且已经明确工种与区域时使用。
    注意：这是高风险操作，会向工程师发送短信并产生差旅成本，执行前系统会要求人工确认。
    返回：派单单号、被指派工程师、SLA 截止时间。
    """
    s = get_settings()
    async with httpx.AsyncClient(timeout=8.0) as c:
        r = await c.post(f"{s.wo_base_url}/dispatch", json={
            "ticket_id": ticket_id, "craft": craft, "urgency": urgency,
            "region": region, "remark": remark})
        r.raise_for_status()
        d = r.json()
    return {"ok": True, "dispatch_id": d["dispatch_id"], "engineer": d["engineer"],
            "sla_deadline": d["sla_deadline"],
            "summary": (f"已派单 {d['dispatch_id']}，指派 {d['engineer']}，"
                        f"SLA 截止 {d['sla_deadline']}。")}
```

#### 8.3 Agent 子图

`app/graph/agent_subgraph.py`：

```python
"""Agent 子图：ReAct 循环 + 工具执行 + 高风险动作人工确认 + 迭代上限。"""
import json
import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.llms import big_llm
from app.tools.registry import REGISTRY, all_tools, needs_confirm
# 导入以触发工具注册（副作用导入，勿删）
from app.tools import stock, ticket_ops, warranty  # noqa: F401

AGENT_SYSTEM = """你是华成机电售后调度助手，负责在工单处理的最后一步执行必要的业务动作。

工作方式：
1. 先想清楚这张工单到底需要做什么动作，不要做多余的事。
2. 需要信息就调用只读工具（check_stock / check_warranty）。
3. 信息足够后，再调用写操作工具（create_subticket / dispatch）。
4. dispatch 是高风险操作，调用前必须已经确认：工种、区域、紧急度三项都明确。
5. 所有动作完成后，用一段不超过 120 字的话总结你做了什么，不要再调用工具。

约束：
- 最多进行 {max_iter} 轮工具调用。
- 不确定的信息不要编造，宁可在总结里写"需人工确认"。
- 已经调用过且成功的工具，不要重复调用。"""


class AgentState(TypedDict, total=False):
    """Agent 子图的内部状态。注意它与主图 TicketState 共享部分键。"""

    ticket_id: str
    extract: dict[str, Any]
    customer_id: str
    clean_text: str
    solution: str

    messages: Annotated[list[BaseMessage], operator.add]
    iter_count: int
    agent_result: dict[str, Any]
    pending_confirm: dict[str, Any] | None
    confirm_decision: str | None      # "approve" | "reject" | None
    trace: Annotated[list[dict[str, Any]], operator.add]


def _init_messages(state: dict) -> list[BaseMessage]:
    """构造 Agent 的初始上下文。"""
    s = get_settings()
    ex = state.get("extract") or {}
    ctx = {
        "工单号": state.get("ticket_id"),
        "客户编号": state.get("customer_id"),
        "设备型号": ex.get("device_model"),
        "故障现象": ex.get("fault_symptom"),
        "故障码": ex.get("fault_code"),
        "紧急度": ex.get("urgency"),
        "建议工种": ex.get("suggested_craft"),
        "是否需要备件": ex.get("need_parts"),
        "意图": ex.get("intent"),
        "客户区域": state.get("region", "华东-苏州"),
        "工单原文": state.get("clean_text"),
        "已生成的维修方案": (state.get("solution") or "")[:300],
    }
    return [
        SystemMessage(content=AGENT_SYSTEM.format(max_iter=s.agent_max_iter)),
        HumanMessage(content="请处理以下工单：\n" + json.dumps(ctx, ensure_ascii=False, indent=2)),
    ]


async def agent_think(state: dict) -> dict:
    """ReAct 的 Thought 步：让大模型决定调用什么工具或给出总结。"""
    messages = state.get("messages") or _init_messages(state)
    llm = big_llm(temperature=0.0, max_tokens=800).bind_tools(all_tools())
    ai: AIMessage = await llm.ainvoke(messages)
    return {
        "messages": ([] if state.get("messages") else messages) + [ai],
        "iter_count": state.get("iter_count", 0) + 1,
        "trace": [{"node": "agent_think", "tool_calls":
                   [tc["name"] for tc in (ai.tool_calls or [])]}],
    }


def route_after_think(state: dict) -> Literal["confirm", "tools", "summarize"]:
    """决定下一步：要确认 / 直接执行工具 / 结束。"""
    s = get_settings()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    if not tool_calls:
        return "summarize"
    if state.get("iter_count", 0) >= s.agent_max_iter:
        return "summarize"
    if s.human_confirm_required and any(needs_confirm(tc["name"]) for tc in tool_calls):
        return "confirm"
    return "tools"


async def human_confirm(state: dict) -> dict:
    """人工确认节点。图在此之前被 interrupt，人工写入 confirm_decision 后继续。"""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    risky = [tc for tc in tool_calls if needs_confirm(tc["name"])]
    decision = state.get("confirm_decision")

    if decision is None:
        # 正常情况下不会走到这里（interrupt_before 会先挂起）；
        # 走到了说明是无人值守模式，按拒绝处理，保证安全
        decision = "reject"

    return {
        "pending_confirm": {
            "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in risky],
            "decision": decision,
        },
        "trace": [{"node": "human_confirm", "decision": decision,
                   "tools": [tc["name"] for tc in risky]}],
    }


def route_after_confirm(state: dict) -> Literal["tools", "summarize"]:
    """确认通过则执行工具，否则直接总结。"""
    pc = state.get("pending_confirm") or {}
    return "tools" if pc.get("decision") == "approve" else "summarize"


async def run_tools(state: dict) -> dict:
    """执行模型请求的所有工具调用，把结果作为 ToolMessage 回灌。"""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    pc = state.get("pending_confirm") or {}
    rejected = {tc["name"] for tc in pc.get("tool_calls", [])} if pc.get("decision") == "reject" else set()

    out_msgs, records = [], []
    for tc in tool_calls:
        name, args, call_id = tc["name"], tc["args"], tc["id"]
        if name in rejected:
            content = json.dumps({"ok": False, "summary": "人工拒绝执行该动作"}, ensure_ascii=False)
        else:
            spec = REGISTRY.get(name)
            if spec is None:
                content = json.dumps({"ok": False, "summary": f"工具 {name} 不存在"}, ensure_ascii=False)
            else:
                try:
                    result = await spec.tool.ainvoke(args)
                    content = json.dumps(result, ensure_ascii=False)
                except Exception as e:                    # noqa: BLE001
                    content = json.dumps(
                        {"ok": False, "summary": f"工具执行失败：{type(e).__name__}: {e}"},
                        ensure_ascii=False)
        out_msgs.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
        records.append({"tool": name, "args": args, "result": content[:200]})

    return {
        "messages": out_msgs,
        "pending_confirm": None,
        "confirm_decision": None,
        "trace": [{"node": "run_tools", "calls": records}],
    }


async def summarize(state: dict) -> dict:
    """汇总 Agent 执行结果，写回主图可读的结构。"""
    s = get_settings()
    msgs = state.get("messages") or []
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not m.tool_calls), None)

    executed = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            try:
                payload = json.loads(m.content)
            except json.JSONDecodeError:
                payload = {"summary": m.content[:120]}
            executed.append({"tool": m.name, "ok": payload.get("ok", payload.get("found", True)),
                             "summary": payload.get("summary", "")})

    hit_limit = state.get("iter_count", 0) >= s.agent_max_iter and not last_ai
    return {
        "agent_result": {
            "actions": executed,
            "narrative": (last_ai.content if last_ai
                          else "达到最大迭代次数，已停止自动处理，转人工跟进。"),
            "iterations": state.get("iter_count", 0),
            "hit_iteration_limit": hit_limit,
            "rejected": (state.get("pending_confirm") or {}).get("decision") == "reject",
        },
        "trace": [{"node": "agent_summarize", "n_actions": len(executed),
                   "hit_limit": hit_limit}],
    }


def build_agent_subgraph():
    """构造 Agent 子图。注意：人工确认的 interrupt 在主图编译时通过参数开启。"""
    g = StateGraph(AgentState)
    g.add_node("agent_think", agent_think)
    g.add_node("human_confirm", human_confirm)
    g.add_node("run_tools", run_tools)
    g.add_node("summarize", summarize)

    g.add_edge(START, "agent_think")
    g.add_conditional_edges("agent_think", route_after_think, {
        "confirm": "human_confirm",
        "tools": "run_tools",
        "summarize": "summarize",
    })
    g.add_conditional_edges("human_confirm", route_after_confirm, {
        "tools": "run_tools",
        "summarize": "summarize",
    })
    g.add_edge("run_tools", "agent_think")      # 回到 think，形成 ReAct 循环
    g.add_edge("summarize", END)

    return g.compile(interrupt_before=["human_confirm"])
```

> **关于 `interrupt_before=["human_confirm"]`**：子图编译时声明中断点，图执行到 `human_confirm` 之前会挂起并把当前 state 存进 checkpointer。外部（人）用 `update_state` 写入 `confirm_decision` 后，再次 `ainvoke(None, config)` 即从中断处继续。这是 LangGraph 相对纯代码最不可替代的能力。

#### 8.4 Mock 业务系统

为了让项目自洽可跑，提供三个 mock 服务。`deploy/mock_services.py`：

```python
"""ERP / CRM / 工单系统的 mock 实现，一个进程起三个端口。"""
import random
import threading
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI

STOCK_DB = {
    "CJ20-40": [
        {"warehouse": "WH-HD01", "warehouse_name": "华东苏州中心仓", "qty": 23, "distance_km": 45, "eta_hours": 4},
        {"warehouse": "WH-HD02", "warehouse_name": "华东杭州仓", "qty": 6, "distance_km": 180, "eta_hours": 12},
    ],
    "BRG-6308": [
        {"warehouse": "WH-HD01", "warehouse_name": "华东苏州中心仓", "qty": 0, "distance_km": 45, "eta_hours": 4},
        {"warehouse": "WH-HZ01", "warehouse_name": "华中武汉仓", "qty": 41, "distance_km": 760, "eta_hours": 36},
    ],
    "SEAL-DG120": [
        {"warehouse": "WH-HD01", "warehouse_name": "华东苏州中心仓", "qty": 112, "distance_km": 45, "eta_hours": 4},
    ],
}

erp = FastAPI(title="mock-ERP")
crm = FastAPI(title="mock-CRM")
wo = FastAPI(title="mock-WorkOrder")


@erp.get("/stock")
def stock(part_no: str, warehouse: str | None = None):
    """返回备件库存。"""
    items = STOCK_DB.get(part_no.upper(), [])
    if warehouse:
        items = [i for i in items if i["warehouse"] == warehouse]
    return {"items": items}


@erp.get("/health")
def erp_health():
    """健康检查。"""
    return {"status": "ok"}


@crm.get("/warranty")
def warranty(customer_id: str, device_model: str, serial_no: str | None = None):
    """返回保修状态。按型号首字母确定性地决定是否在保，便于复现。"""
    if not device_model:
        return {"found": False}
    in_warranty = sum(ord(c) for c in device_model) % 3 != 0
    expire = datetime.now() + timedelta(days=200 if in_warranty else -120)
    return {
        "found": True,
        "in_warranty": in_warranty,
        "expire_date": expire.strftime("%Y-%m-%d"),
        "coverage": ["主机维修", "电气件更换", "上门服务费"] if in_warranty else [],
        "exclusions": ["易损件", "人为损坏", "超负载运行导致的损坏"],
    }


@crm.get("/health")
def crm_health():
    """健康检查。"""
    return {"status": "ok"}


@wo.post("/subticket")
def subticket(body: dict):
    """创建子工单。"""
    return {"subticket_id": f"SUB-{random.randint(100000, 999999)}", "status": "created"}


@wo.post("/dispatch")
def do_dispatch(body: dict):
    """派单，按紧急度给 SLA。"""
    sla_hours = {1: 48, 2: 24, 3: 2}.get(int(body.get("urgency", 1)), 48)
    engineers = {"电气": "王建国", "机械": "李卫东", "液压": "赵明", "控制": "陈晓峰"}
    return {
        "dispatch_id": f"DP-{random.randint(100000, 999999)}",
        "engineer": engineers.get(body.get("craft"), "值班工程师"),
        "sla_deadline": (datetime.now() + timedelta(hours=sla_hours)).strftime("%Y-%m-%d %H:%M"),
    }


@wo.get("/health")
def wo_health():
    """健康检查。"""
    return {"status": "ok"}


def serve(app, port):
    """在独立线程里起一个服务。"""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    for a, p in ((erp, 9101), (crm, 9102), (wo, 9103)):
        threading.Thread(target=serve, args=(a, p), daemon=True).start()
    print("mock 服务已启动：ERP:9101  CRM:9102  WorkOrder:9103")
    threading.Event().wait()
```

#### 8.5 运行命令

```bash
# 1) 起 mock 服务
python deploy/mock_services.py &

# 2) 演示人工确认中断与恢复
python demo/demo_agent_interrupt.py
```

`demo/demo_agent_interrupt.py`：

```python
"""演示：Agent 执行到高风险动作时挂起，人工确认后继续。"""
import asyncio
import json

from langgraph.checkpoint.memory import MemorySaver

from app.graph.main_graph import build_main_graph


async def main():
    """跑一张紧急工单，在派单前中断，人工批准后恢复。"""
    app = build_main_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "demo-interrupt-001"}}

    inputs = {
        "ticket_id": "TK-2025-000777",
        "channel": "400电话",
        "customer_id": "CUST-88213",
        "raw_text": "客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。"
                    "客户说损失很大，要求两小时内到场。另外问下CJ20-40接触器有没有现货。",
        "trace": [], "errors": [], "cost_items": [],
    }

    print("=== 第一次执行（会在人工确认前挂起）===")
    async for chunk in app.astream(inputs, config, stream_mode="updates"):
        for node, upd in chunk.items():
            print(f"  [{node}] {json.dumps(upd, ensure_ascii=False, default=str)[:180]}")

    snapshot = await app.aget_state(config)
    print(f"\n当前中断点: {snapshot.next}")

    # 取出待确认的工具调用给人看
    msgs = snapshot.values.get("messages", [])
    if msgs and getattr(msgs[-1], "tool_calls", None):
        print("待确认的动作：")
        for tc in msgs[-1].tool_calls:
            print(f"  - {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

    decision = input("\n批准执行？(y/n): ").strip().lower()
    await app.aupdate_state(
        config, {"confirm_decision": "approve" if decision == "y" else "reject"})

    print("\n=== 恢复执行 ===")
    async for chunk in app.astream(None, config, stream_mode="updates"):
        for node, upd in chunk.items():
            print(f"  [{node}] {json.dumps(upd, ensure_ascii=False, default=str)[:180]}")

    final_state = await app.aget_state(config)
    print("\n=== 最终结果 ===")
    print(json.dumps(final_state.values.get("final"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

#### 8.6 预期输出

```text
=== 第一次执行（会在人工确认前挂起）===
  [preprocess] {"clean_text": "客户B1线整条停了，变频器上电无显示，报E21。型号BP-22KW。客户说损失很大，要求两小时内到场。另外问下CJ20-40接触器有没有现货。", ...}
  [extract] {"extract": {"device_model": "BP-22KW", "fault_symptom": "变频器上电无显示报E21", "fault_code": "E21", "urgency": 3, ...
  [validate] {"valid": true, "missing_fields": [], ...}
  [retrieve] {"docs": [...], "retrieve_latency_ms": 298.4, ...}
  [generate] {"solution": "初步判断：\nBP-22KW 报 E21 为上电无显示...", ...}
  [check_action] {"need_action": true, "action_reason": "need_parts=true；命中关键词['现货']；urgency=3 需派单", ...}
  [agent] {"trace": [{"node": "agent_think", "tool_calls": ["check_stock"]}], ...}
  [agent] {"trace": [{"node": "run_tools", "calls": [{"tool": "check_stock", ...}]}], ...}
  [agent] {"trace": [{"node": "agent_think", "tool_calls": ["dispatch"]}], ...}

当前中断点: ('human_confirm',)
待确认的动作：
  - dispatch({"ticket_id": "TK-2025-000777", "craft": "电气", "urgency": 3, "region": "华东-苏州", "remark": "B1线整线停机，变频器E21上电无显示，客户要求2小时内到场；CJ20-40苏州仓有23件现货可带"})

批准执行？(y/n): y

=== 恢复执行 ===
  [agent] {"pending_confirm": {"tool_calls": [{"name": "dispatch", ...}], "decision": "approve"}, ...}
  [agent] {"trace": [{"node": "run_tools", "calls": [{"tool": "dispatch", "result": "{\"ok\": true, \"dispatch_id\": \"DP-472913\", ...
  [agent] {"trace": [{"node": "agent_think", "tool_calls": []}], ...}
  [agent] {"agent_result": {"actions": [...], "narrative": "已确认CJ20-40接触器苏州中心仓有23件现货...", ...}}
  [finalize] {"final": {...}}

=== 最终结果 ===
{
  "ticket_id": "TK-2025-000777",
  "extract": {
    "device_model": "BP-22KW",
    "fault_symptom": "变频器上电无显示报E21",
    "fault_code": "E21",
    "urgency": 3,
    "production_line": "B1",
    "suggested_craft": "电气",
    "need_parts": true,
    "intent": "报修"
  },
  "solution": "初步判断：...",
  "citations": ["[1] BP系列变频器说明书 V3.0 第6.2节", "[2] 维修手册-变频器无显示故障树 V1.4"],
  "agent_result": {
    "actions": [
      {"tool": "check_stock", "ok": true, "summary": "CJ20-40 全网现货 29 件，库存最多的是华东苏州中心仓（23 件，距客户 45 公里，预计 4 小时可达）。"},
      {"tool": "dispatch", "ok": true, "summary": "已派单 DP-472913，指派 王建国，SLA 截止 2025-03-12 16:40。"}
    ],
    "narrative": "已确认CJ20-40接触器苏州中心仓有23件现货，4小时可达；已派单给电气工程师王建国，SLA 2小时（截止16:40），并在备注中提醒携带接触器。",
    "iterations": 3,
    "hit_iteration_limit": false,
    "rejected": false
  },
  "need_human": false,
  "confidence": 1.0,
  "cost_yuan": 0.013844,
  "latency_ms": 9218.6
}
```

#### 8.7 迭代上限测试

`tests/test_agent_limit.py`：

```python
"""验证 Agent 达到最大迭代数时会安全退出，不会无限循环。"""
import asyncio
import json

from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.graph.agent_subgraph import build_agent_subgraph


async def test_iteration_limit():
    """构造一个会诱导模型反复查询的场景，验证迭代上限生效。"""
    s = get_settings()
    sub = build_agent_subgraph()
    state = {
        "ticket_id": "TK-LIMIT-TEST",
        "customer_id": "CUST-00000",
        "clean_text": "请把所有仓库的所有备件库存都查一遍，一个一个查，查完再查一遍确认。",
        "extract": {"device_model": "XC-200-4P", "fault_symptom": "测试",
                    "urgency": 1, "need_parts": True, "intent": "备件申请",
                    "suggested_craft": "电气", "fault_code": None, "production_line": None},
        "messages": [], "trace": [],
    }
    result = await sub.ainvoke(state, {"configurable": {"thread_id": "limit-test"},
                                       "recursion_limit": 50})
    ar = result["agent_result"]
    print(json.dumps(ar, ensure_ascii=False, indent=2))
    assert ar["iterations"] <= s.agent_max_iter + 1, "迭代数超过上限"
    print(f"✅ 迭代数 {ar['iterations']} <= 上限 {s.agent_max_iter}")


if __name__ == "__main__":
    asyncio.run(test_iteration_limit())
```

```text
{
  "actions": [
    {"tool": "check_stock", "ok": false, "summary": "未查到备件 ALL，请确认编号是否正确。"},
    {"tool": "check_stock", "ok": true, "summary": "CJ20-40 全网现货 29 件..."},
    ...
  ],
  "narrative": "达到最大迭代次数，已停止自动处理，转人工跟进。",
  "iterations": 6,
  "hit_iteration_limit": true,
  "rejected": false
}
✅ 迭代数 6 <= 上限 6
```

#### 8.8 小结

- 工具按风险分级（read / write / critical），只有 critical 才走人工确认，否则体验会崩；
- `interrupt_before` + checkpointer 是人工确认的标准实现，**必须搭配持久化 checkpointer**（生产用 SQLite/Postgres，不要用 MemorySaver）；
- 迭代上限要有硬保护，并且要**写测试验证**，不能只写在文档里；
- Agent 的 `narrative` 字段很重要：它是给人看的执行摘要，客服要靠它决定信不信这次自动处理。

---

## 五、联调与演示

Part A 把抽取能力烧进了权重，Part B 把流程连成了图，Part C 让 Agent 能真的动手。这一节把三者**串起来跑三条真实路径**，并且把每一步的输入输出、耗时、成本全部打出来——**看不见的系统是没法上线的**。

> 本节所有数字都是**示例性数据**（实测环境：RTX 4090 24G 跑 vLLM 0.6.3 加载 AWQ 量化后的 `ticket-extract`，`deepseek-chat` 走公网 API，Milvus 2.4 单机，mock ERP/CRM/工单系统本机）。你的数字一定不同。

### 5.1 先补齐一个被引用的模块：`app/routing/model_router.py`

Step 7 的 `nodes.py` 里 `from app.routing.model_router import estimate_cost` 一直没给实现，现在补上。这个模块干两件事：**决定每个任务用哪一档模型**、**把每次调用的成本记下来**。

成本记账必须在**节点级**做，不能只记总数——否则你永远回答不了"到底是抽取贵还是 Agent 贵"这个问题（答案在 8.1 节的量化表里，而那张表就是这个模块攒出来的）。

```python
# app/routing/model_router.py
"""模型路由与成本核算：决定每个任务用哪档模型，并把每次调用的成本记进 state。"""
from __future__ import annotations

from typing import Any, Literal

from app.config import get_settings

Tier = Literal["small", "big", "reason"]

# 计价表（元 / 百万 token）。本地模型按电费估算的等效单价。
# **以各家官方最新定价为准**，这里只是把公式和量级给出来。
MODEL_PRICE: dict[str, dict[str, float]] = {
    # 微调后的 7B（本地 vLLM）：按整卡功耗 350W、电价 0.7 元/度、
    # 并发 8 时约 31.7 req/s 折算出的等效 token 单价，量级在 0.0x 元/百万
    "ticket-extract": {"in": 0.03, "out": 0.03},
    "deepseek-chat": {"in": 2.0, "out": 5.0},
    "deepseek-reasoner": {"in": 4.0, "out": 16.0},
}

# 路由表：任务 → 档位。**这张表是全项目的成本开关**，改它比改代码有效
ROUTING: dict[str, Tier] = {
    "extract": "small",        # 结构化抽取：格式固定、有训练数据 → 微调小模型
    "classify": "small",       # 意图/工种分类：随抽取一起出，零额外成本
    "clarify": "small",        # 追问话术：模板兜底，必要时小模型润色
    "generate": "big",         # 方案生成：要读长上下文、要讲人话 → 大模型
    "agent": "big",            # 工具决策：要 function calling 与多轮推理 → 大模型
    "arbitrate": "reason",     # 疑难仲裁（可选）：留给人工介入前的最后一档
}


def model_of(tier: Tier) -> str:
    """档位 → 具体模型名。"""
    s = get_settings()
    return {"small": s.small_model, "big": s.llm_model, "reason": "deepseek-reasoner"}[tier]


def tier_of(task: str) -> Tier:
    """任务 → 档位。未登记的任务一律走大模型（保守，宁可贵也不要错）。"""
    return ROUTING.get(task, "big")


def estimate_cost(node: str, tier: Tier, prompt_tokens: int = 0,
                  completion_tokens: int = 0, model: str | None = None) -> dict[str, Any]:
    """算一次调用的成本，返回可直接塞进 state['cost_items'] 的记录。"""
    name = model or model_of(tier)
    p = MODEL_PRICE.get(name, MODEL_PRICE["deepseek-chat"])
    cost = prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]
    return {
        "node": node,
        "tier": tier,
        "model": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_yuan": round(cost, 6),
    }


def should_escalate(extract: dict, channel: str, retry_count: int = 0) -> bool:
    """判断这一单的抽取是否应该直接上大模型（跳过小模型）。

    这是一条「预降级」策略：已知小模型会翻车的场景，不要浪费一次调用再降级。
    判据来自 Step 4.5 的分渠道指标：小程序渠道字段准确率低 6 个百分点。
    """
    if retry_count > 0:
        return True
    text = extract.get("_raw_text", "") if isinstance(extract, dict) else ""
    # 超长工单（小模型 max-model-len 2048）与含大段表格的经销商单，直接上大模型
    if len(text) > 1200:
        return True
    return False


def cost_summary(cost_items: list[dict]) -> dict[str, Any]:
    """把逐次调用汇总成一张按节点分摊的账单（8.1 节的收益量化表就靠它）。"""
    by_node: dict[str, float] = {}
    by_tier: dict[str, float] = {}
    for c in cost_items:
        by_node[c["node"]] = round(by_node.get(c["node"], 0.0) + c["cost_yuan"], 6)
        by_tier[c["tier"]] = round(by_tier.get(c["tier"], 0.0) + c["cost_yuan"], 6)
    total = round(sum(c["cost_yuan"] for c in cost_items), 6)
    return {
        "total_yuan": total,
        "by_node": by_node,
        "by_tier": by_tier,
        "small_share": round(by_tier.get("small", 0.0) / total, 4) if total else 0.0,
        "n_calls": len(cost_items),
    }
```

### 5.2 启动顺序与自检

**先说端口。** 全书的端口约定是：vLLM **8001**、应用服务 **8080**、Attu 8000、Langfuse 3001、Postgres 5433。Step 5 为了演示直接用了 vLLM 的默认端口 8000，Step 4.3 又临时用 8001 起了未微调的基座做对照实验。从本节开始**统一按约定**：

| 服务 | 端口 | 说明 |
|---|---|---|
| vLLM（`ticket-extract`，AWQ 量化） | **8001** | 正式的抽取模型服务 |
| 应用服务（FastAPI，7.2 节） | **8080** | 对外入口 |
| vLLM（对照用的未微调基座） | 8002 | 只在做对照实验时临时起，跑完就停 |
| mock ERP / CRM / 工单系统 | 9101 / 9102 / 9103 | `deploy/mock_services.py` |
| Milvus | 19530 | Attu 控制台 8000 |
| Langfuse | 3001 | 其 Postgres 用 5433 |

所以 `.env` 里的这一行要改成：

```bash
SMALL_BASE_URL=http://127.0.0.1:8001/v1
```

启动顺序（**顺序不能乱**：抽取链在 import 时就会构造 `ExtractService`，vLLM 没起来的话第一单会走降级）：

```bash
# 1) mock 业务系统（Agent 的工具靠它）
python deploy/mock_services.py &
sleep 2
curl -s http://127.0.0.1:9101/stock?part_no=CJ20-40 | head -c 120; echo

# 2) vLLM 起微调模型（端口按约定用 8001）
python -m vllm.entrypoints.openai.api_server \
  --model outputs/merged-awq \
  --served-model-name ticket-extract \
  --quantization awq --dtype half \
  --max-model-len 2048 --gpu-memory-utilization 0.55 \
  --max-num-seqs 64 --enable-prefix-caching --disable-log-requests \
  --port 8001 --host 0.0.0.0 &

# 3) 等模型就绪（别用 sleep 猜，轮询 /v1/models）
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 2; echo -n .; done; echo " ready"

# 4) 确认 Milvus 里有知识（复用项目 2 的 collection）
python - <<'PY'
from pymilvus import MilvusClient
from app.config import get_settings
s = get_settings()
c = MilvusClient(uri=s.milvus_uri)
print("collections:", c.list_collections())
print("行数:", c.get_collection_stats(s.milvus_collection))
PY

# 5) 四件套自检
python - <<'PY'
import asyncio, json
from app.chains.extract_chain import ExtractService
from app.tools.registry import REGISTRY
from app.tools import stock, ticket_ops, warranty  # noqa: F401
from app.graph.main_graph import build_main_graph

async def main():
    # ① 抽取链通不通
    r = await ExtractService(use_finetuned=True).aextract(
        "客户A2线XC-200-4P电机异响报E33", "400电话")
    print("① 抽取:", r["extract"].model_dump_json(), f"{r['latency_ms']:.0f}ms 降级={r['degraded']}")
    # ② 工具注册齐不齐
    print("② 工具:", {k: (v.risk, v.need_confirm) for k, v in REGISTRY.items()})
    # ③ 图能不能编译
    app = build_main_graph()
    print("③ 节点:", sorted(app.get_graph().nodes))
    # ④ 成本记账通不通
    from app.routing.model_router import estimate_cost, cost_summary
    items = [estimate_cost("extract", "small", 447, 88),
             estimate_cost("generate", "big", 1420, 252)]
    print("④ 成本:", json.dumps(cost_summary(items), ensure_ascii=False))

asyncio.run(main())
PY
```

```text
{"items":[{"warehouse":"WH-HD01","warehouse_name":"华东苏州中心仓","qty":23,...
....... ready
collections: ['huacheng_kb']
行数: {'row_count': 18432}
① 抽取: {"device_model":"XC-200-4P","fault_symptom":"电机异响报E33","fault_code":"E33","urgency":2,"production_line":"A2","suggested_craft":"机械","need_parts":false,"intent":"报修"} 409ms 降级=False
② 工具: {'check_stock': ('read', False), 'check_warranty': ('read', False), 'create_subticket': ('write', False), 'dispatch': ('critical', True)}
③ 节点: ['__end__', '__start__', 'agent', 'check_action', 'clarify', 'extract', 'finalize', 'generate', 'preprocess', 'retrieve', 'validate']
④ 成本: {"total_yuan": 0.004142, "by_node": {"extract": 0.000016, "generate": 0.004100}, "by_tier": {"small": 0.000016, "big": 0.004100}, "n_calls": 2}
```

第 ④ 行已经把本项目的核心事实摆出来了：**一单里抽取的成本占 0.4%，生成占 99.6%**。这个比例会在 8.1 节被反复引用，也是"三位一体到底省了什么"这个问题的答案来源。

### 5.3 演示 1：普通工单（抽取 → 分类 → 检索 → 生成方案）

这是最高频的路径（约占日均工单的 54%）：能定型号、能定故障、不需要派人、不需要备件。走 `preprocess → extract → validate → retrieve → generate → check_action → finalize`，**完全不进 Agent**。

`demo/demo_normal.py`：

```python
"""演示 1：普通报修工单的完整处理轨迹（不触发 Agent）。"""
import asyncio
import json

from app.graph.main_graph import build_main_graph
from app.routing.model_router import cost_summary

TICKET = {
    "ticket_id": "TK-2026-000314",
    "channel": "400电话",
    "customer_id": "CUST-77104",
    "raw_text": ("您好，我们是苏州精工机械有限公司的，A2线那台XC-200-4P电机启动的时候有异响，"
                 "运行半个小时以后面板报E33，现在还能开但是声音明显不对。"
                 "师傅说先不用换件，想问下怎么处理。联系人张工13812345678。"),
    "trace": [], "errors": [], "cost_items": [],
}


def print_trace(state: dict) -> None:
    """按节点顺序打印轨迹，带耗时与关键字段。"""
    print("\n─── 处理轨迹 ───")
    for t in state.get("trace", []):
        extra = {k: v for k, v in t.items() if k not in ("node", "latency_ms")}
        print(f"  {t['node']:<14}{t['latency_ms']:>8.1f} ms  {json.dumps(extra, ensure_ascii=False)}")


async def main():
    """跑一张普通工单。"""
    app = build_main_graph()
    state = await app.ainvoke(TICKET)

    print("─── 1) 脱敏后的文本 ───")
    print(" ", state["clean_text"])

    print("\n─── 2) 微调模型抽取结果 ───")
    print(" ", json.dumps(state["extract"], ensure_ascii=False))
    print(f"  抽取耗时 {state['extract_latency_ms']:.0f} ms　降级={state['extract_degraded']}")

    print("\n─── 3) 路由判定 ───")
    print(f"  valid={state['valid']}　missing={state['missing_fields']}　"
          f"intent={state['extract']['intent']} → 走检索生成")

    print("\n─── 4) 检索命中的资料 ───")
    for i, d in enumerate(state.get("docs", []), 1):
        print(f"  [{i}] score={d.get('score', 0):.3f}  {d.get('source', '')}")
        print(f"      {d.get('text', '')[:70]}…")

    print("\n─── 5) 生成的方案 ───")
    print(state["solution"])

    print("\n─── 6) 行动判定 ───")
    print(f"  need_action={state['need_action']}　reason={state['action_reason']}")

    print_trace(state)
    print("\n─── 成本账单 ───")
    print(" ", json.dumps(cost_summary(state["cost_items"]), ensure_ascii=False))
    print("\n─── 最终输出 ───")
    print(json.dumps(state["final"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python demo/demo_normal.py
```

```text
─── 1) 脱敏后的文本 ───
  您好，我们是[COMPANY]的，A2线那台XC-200-4P电机启动的时候有异响，运行半个小时以后面板报E33，现在还能开但是声音明显不对。师傅说先不用换件，想问下怎么处理。联系人张工[PHONE]。

─── 2) 微调模型抽取结果 ───
  {"device_model": "XC-200-4P", "fault_symptom": "电机启动异响运行后报E33", "fault_code": "E33", "urgency": 2, "production_line": "A2", "suggested_craft": "机械", "need_parts": false, "intent": "报修"}
  抽取耗时 421 ms　降级=False

─── 3) 路由判定 ───
  valid=True　missing=[]　intent=报修 → 走检索生成

─── 4) 检索命中的资料 ───
  [1] score=0.912  XC-200系列产品说明书 V2.1 第4.3节 故障代码表
      E33：轴承温升异常/机械异响保护。触发条件为振动烈度超过 4.5 mm/s 且持续 30 秒…
  [2] score=0.874  维修手册-电机异响类故障处理 V1.4
      异响分三类：轴承类（周期性咔哒）、风叶类（连续摩擦）、电磁类（随负载变化的嗡鸣）…
  [3] score=0.831  内部Wiki-E33报警现场处置经验
      现场先用测振仪测水平/垂直/轴向三个方向，超过 4.5 mm/s 需停机；不要直接加润滑脂…
  [4] score=0.786  XC-200系列维护保养手册 V2.0 第3.1节
      润滑周期：连续运行工况每 2000 小时补脂，牌号 SKF LGMT3，单侧注脂量 12 g…
  [5] score=0.742  历史工单归纳-XC-200-4P异响类根因分布
      41 条 E33 工单中：轴承磨损 19 条（46.3%）、地脚螺栓松动 11 条（26.8%）…

─── 5) 生成的方案 ───
初步判断：
XC-200-4P 报 E33 属于轴承温升/机械异响保护，结合"启动即有异响、运行 30 分钟后触发"的描述，
优先怀疑轴承磨损或地脚螺栓松动导致的振动超限 [1][5]。当前仍可运行，属于可计划性检修 [3]。

处理步骤：
1. 用测振仪在轴承座水平、垂直、轴向三个方向测振动烈度，记录数值 [3]
2. 若任一方向超过 4.5 mm/s，安排停机检修；未超则可带载观察，但需 24 小时内复测 [1][3]
3. 用扭力扳手复核四颗地脚螺栓预紧力，按手册值复紧 [5]
4. 听音判断异响类型：周期性咔哒声指向轴承，连续摩擦声指向风叶碰壳 [2]
5. 确认为轴承磨损时按 2000 小时周期补脂（SKF LGMT3，单侧 12 g），不要过量 [4]

注意事项：
- 不要在未测振的情况下直接加润滑脂，过量注脂本身会导致温升升高 [3]
- 若复紧螺栓后异响消失但 E33 仍复发，需检查振动传感器与整定值 [1]

─── 6) 行动判定 ───
  need_action=False　reason=无需行动

─── 处理轨迹 ───
  preprocess           0.5 ms  {"pii_masked": true}
  extract            421.3 ms  {"degraded": false, "model": "ticket-extract"}
  validate             0.1 ms  {"missing": []}
  retrieve           327.8 ms  {"n_docs": 5, "query": "电机启动异响运行后报E33 E33 XC-200-4P"}
  generate          2861.4 ms  {"model": "deepseek-chat", "n_cite": 5}
  check_action         0.2 ms  {"need_action": false, "reason": []}
  finalize             1.1 ms  {"confidence": 1.0}

─── 成本账单 ───
  {"total_yuan": 0.004142, "by_node": {"extract": 0.000016, "generate": 0.004126}, "by_tier": {"small": 0.000016, "big": 0.004126}, "small_share": 0.0039, "n_calls": 2}

─── 最终输出 ───
{
  "ticket_id": "TK-2026-000314",
  "extract": {
    "device_model": "XC-200-4P",
    "fault_symptom": "电机启动异响运行后报E33",
    "fault_code": "E33",
    "urgency": 2,
    "production_line": "A2",
    "suggested_craft": "机械",
    "need_parts": false,
    "intent": "报修"
  },
  "solution": "初步判断：\nXC-200-4P 报 E33 属于轴承温升/机械异响保护…",
  "citations": [
    "[1] XC-200系列产品说明书 V2.1 第4.3节 故障代码表",
    "[2] 维修手册-电机异响类故障处理 V1.4",
    "[3] 内部Wiki-E33报警现场处置经验",
    "[4] XC-200系列维护保养手册 V2.0 第3.1节",
    "[5] 历史工单归纳-XC-200-4P异响类根因分布"
  ],
  "clarify_question": null,
  "agent_result": null,
  "need_human": false,
  "confidence": 1.0,
  "cost_yuan": 0.004142,
  "latency_ms": 3612.4
}
```

**这条轨迹里有四个值得停下来看的点**：

| 观察 | 数字 | 含义 |
|---|---|---|
| 抽取 421 ms，生成 2861 ms | 抽取占总延迟 11.7% | **延迟瓶颈在生成，不在抽取**。想压延迟要去优化生成（流式输出、缩短上下文），而不是继续压小模型 |
| 抽取 ¥0.000016，生成 ¥0.004126 | 抽取占总成本 0.4% | 同上：**成本瓶颈也在生成** |
| `need_parts=false` 让 `need_action=false` | 省掉一次 Agent（约 ¥0.0097、约 8 秒） | 这就是 0.3 节"Agent 用得越少，系统越好"的具体体现：**规则判定把 62% 的工单挡在 Agent 之外** |
| `pii_masked=true` | 公司名与手机号在进模型之前就被替换 | 脱敏必须在 `preprocess` 做，不能指望模型"不要记住" |

### 5.4 演示 2：紧急工单需派单（触发 Agent 子图 → 查库存 → 判保修 → 人工确认 → 派单）

这是最复杂的路径（约占 9%）：产线停机、要备件、要派人。Step 8.6 已经单独演示过 Agent 子图内部的中断恢复；**这里看的是主图 + 子图的完整轨迹**，并且多了一次 `check_warranty`（客户会先问"这个要不要钱"）。

`demo/demo_urgent.py`：

```python
"""演示 2：紧急工单的完整处理（主图 + Agent 子图 + 人工确认中断恢复）。"""
import asyncio
import json

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.main_graph import build_main_graph
from app.routing.model_router import cost_summary

TICKET = {
    "ticket_id": "TK-2026-000318",
    "channel": "400电话",
    "customer_id": "CUST-88213",
    "region": "华东-苏州",
    "raw_text": ("急！B1线整条停了！BP-22KW变频器上电没显示，面板报E21，"
                 "现在整线都动不了，一小时损失好几万。之前换过一次接触器，"
                 "这次是不是也要换CJ20-40？你们仓库有现货吗？这个还在保修期内吧，要不要钱？"
                 "客户要求今天必须到人。"),
    "trace": [], "errors": [], "cost_items": [],
}


async def main():
    """跑一张紧急工单：会在 dispatch 前挂起等人工确认。"""
    async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        app = build_main_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "demo-urgent-318"}}

        print("═══ 阶段 1：主图执行到 Agent，在 dispatch 前挂起 ═══")
        async for chunk in app.astream(TICKET, cfg, stream_mode="updates"):
            for node, upd in chunk.items():
                brief = {k: v for k, v in (upd or {}).items()
                         if k in ("extract", "need_action", "action_reason",
                                  "valid", "missing_fields", "agent_result")}
                line = json.dumps(brief, ensure_ascii=False, default=str)
                print(f"  [{node}] {line[:150]}")
                for t in (upd or {}).get("trace", []):
                    print(f"      └ {t.get('node')}: "
                          f"{json.dumps({k: v for k, v in t.items() if k != 'node'}, ensure_ascii=False, default=str)[:130]}")

        snap = await app.aget_state(cfg)
        print(f"\n  当前中断点：{snap.next}")
        msgs = snap.values.get("messages", [])
        if msgs and getattr(msgs[-1], "tool_calls", None):
            print("  待人工确认的高风险动作：")
            for tc in msgs[-1].tool_calls:
                print(f"    - {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

        decision = input("\n  批准派单？(y/n): ").strip().lower()
        await app.aupdate_state(
            cfg, {"confirm_decision": "approve" if decision == "y" else "reject"})

        print("\n═══ 阶段 2：恢复执行 ═══")
        async for chunk in app.astream(None, cfg, stream_mode="updates"):
            for node, upd in chunk.items():
                for t in (upd or {}).get("trace", []):
                    print(f"  [{node}] {json.dumps(t, ensure_ascii=False, default=str)[:160]}")

        final = (await app.aget_state(cfg)).values
        print("\n═══ Agent 执行摘要 ═══")
        ar = final["final"]["agent_result"]
        for a in ar["actions"]:
            print(f"  {'✅' if a['ok'] else '❌'} {a['tool']}: {a['summary']}")
        print(f"\n  给客服看的一段话：{ar['narrative']}")
        print(f"  迭代轮数 {ar['iterations']}　达上限={ar['hit_iteration_limit']}　被拒={ar['rejected']}")
        print("\n═══ 成本账单 ═══")
        print(" ", json.dumps(cost_summary(final["cost_items"]), ensure_ascii=False))
        print(f"\n  端到端延迟（不含人工确认等待）：{final['final']['latency_ms']} ms")
        print(f"  单单成本：¥{final['final']['cost_yuan']}")


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python demo/demo_urgent.py
```

```text
═══ 阶段 1：主图执行到 Agent，在 dispatch 前挂起 ═══
  [preprocess] {}
      └ preprocess: {"latency_ms": 0.6, "pii_masked": false}
  [extract] {"extract": {"device_model": "BP-22KW", "fault_symptom": "变频器上电无显示报E21", "fault_code":
      └ extract: {"latency_ms": 438.7, "degraded": false, "model": "ticket-extract"}
  [validate] {"valid": true, "missing_fields": []}
      └ validate: {"latency_ms": 0.1, "missing": []}
  [retrieve] {}
      └ retrieve: {"latency_ms": 341.2, "n_docs": 5, "query": "变频器上电无显示报E21 E21 BP-22KW"}
  [generate] {}
      └ generate: {"latency_ms": 3104.8, "model": "deepseek-chat", "n_cite": 4}
  [check_action] {"need_action": true, "action_reason": "need_parts=true；命中关键词['现货', '保修', '上门']；urgency=3 需派单"}
      └ check_action: {"latency_ms": 0.3, "need_action": true, "reason": ["need_parts=true", "命中关键词['现货', '保修', '上门']", "urgency=3 需派单"]}
  [agent] {}
      └ agent_think: {"tool_calls": ["check_stock", "check_warranty"]}
  [agent] {}
      └ run_tools: {"calls": [{"tool": "check_stock", "args": {"part_no": "CJ20-40"}, "result": "{\"found\": true, \"part_no\": \"CJ20-40\", \"total_qty\": 29,
  [agent] {}
      └ agent_think: {"tool_calls": ["dispatch"]}

  当前中断点：('human_confirm',)
  待人工确认的高风险动作：
    - dispatch({"ticket_id": "TK-2026-000318", "craft": "电气", "urgency": 3, "region": "华东-苏州", "remark": "B1线整线停机，BP-22KW变频器E21上电无显示；客户在保（2026-11-08到期），主回路器件属保修范围；CJ20-40苏州中心仓23件现货，请随车携带"})

  批准派单？(y/n): y

═══ 阶段 2：恢复执行 ═══
  [agent] {"node": "human_confirm", "decision": "approve", "tools": ["dispatch"]}
  [agent] {"node": "run_tools", "calls": [{"tool": "dispatch", "args": {"ticket_id": "TK-2026-000318", "craft": "电气", "urgency": 3,
  [agent] {"node": "agent_think", "tool_calls": []}
  [agent] {"node": "agent_summarize", "n_actions": 3, "hit_limit": false}
  [finalize] {"node": "finalize", "latency_ms": 1.4, "confidence": 1.0}

═══ Agent 执行摘要 ═══
  ✅ check_stock: CJ20-40 全网现货 29 件，库存最多的是华东苏州中心仓（23 件，距客户 45 公里，预计 4 小时可达）。
  ✅ check_warranty: BP-22KW 当前在保，保修到期日 2026-11-08；覆盖：主回路器件、控制板、驱动模块；不覆盖：外部接线、进水受潮、超载烧损。
  ✅ dispatch: 已派单 DP-481207，指派 王建国，SLA 截止 2026-03-18 16:40。

  给客服看的一段话：已确认客户BP-22KW在保（2026-11-08到期），变频器主回路器件在保修范围内、本次上门与器件更换不收费；CJ20-40接触器苏州中心仓23件现货，4小时可达；已派单给电气工程师王建国（DP-481207），SLA 2小时内到场，并已在备注提醒随车带件。需现场确认是否为外部接线或超载导致，这两类不在保修范围。
  迭代轮数 3　达上限=False　被拒=False

═══ 成本账单 ═══
  {"total_yuan": 0.013844, "by_node": {"extract": 0.000016, "generate": 0.004126, "agent": 0.009702}, "by_tier": {"small": 0.000016, "big": 0.013828}, "small_share": 0.0012, "n_calls": 5}

  端到端延迟（不含人工确认等待）：11842.7 ms
  单单成本：¥0.013844
```

如果人工点了 `n`，轨迹会变成这样（**这条路径必须演练**，否则你不知道拒绝之后系统会不会卡住）：

```text
  批准派单？(y/n): n

═══ 阶段 2：恢复执行 ═══
  [agent] {"node": "human_confirm", "decision": "reject", "tools": ["dispatch"]}
  [agent] {"node": "agent_summarize", "n_actions": 2, "hit_limit": false}
  [finalize] {"node": "finalize", "latency_ms": 1.2, "confidence": 1.0}

═══ Agent 执行摘要 ═══
  ✅ check_stock: CJ20-40 全网现货 29 件…
  ✅ check_warranty: BP-22KW 当前在保，保修到期日 2026-11-08…

  给客服看的一段话：已查明客户在保且苏州仓有CJ20-40现货23件，但派单动作被人工拒绝，未创建派单记录。建议由值班调度确认后手工派单。
  迭代轮数 2　达上限=False　被拒=True
```

注意 `route_after_confirm` 的设计：**拒绝后不回 `agent_think`，而是直接去 `summarize`**。这条边很关键——如果拒绝后让模型重新思考，它很可能换个参数再申请一次派单，人就得一直点"拒绝"。**人工拒绝应当终止这一轮自动化，而不是触发重试。**

### 5.5 演示 3：信息不全需追问（抽出 null 字段 → 反问 → 补全后继续）

小程序渠道占日均工单的 34.6%，文本短、缺信息、错别字多（Step 4.5 的分渠道指标里它低 6 个百分点）。这条路径的价值在于：**用一次 ¥0.000016 的抽取，避免一次 ¥0.004 的无效生成**。

`demo/demo_incomplete.py`：

```python
"""演示 3：信息不全 → 追问 → 用户补全 → 继续处理。"""
import asyncio
import json

from app.graph.main_graph import build_main_graph
from app.routing.model_router import cost_summary

FIRST = {
    "ticket_id": "TK-2026-000327",
    "channel": "小程序",
    "customer_id": "CUST-91002",
    "raw_text": "机器不转了 有点糊味 急!!!",
    "trace": [], "errors": [], "cost_items": [],
}

# 用户在小程序里回复后，把原文与补充合并成新的一单（同一个 ticket_id，加 -R1 后缀标记轮次）
SUPPLEMENT = "型号XC-300-4P 三号线 昨天还好的 今天一开就跳"


def brief(state: dict, title: str) -> None:
    """打印一轮处理的要点。"""
    print(f"\n═══ {title} ═══")
    print("  抽取：", json.dumps(state["extract"], ensure_ascii=False))
    print(f"  valid={state['valid']}　missing={state['missing_fields']}　"
          f"route={'clarify' if state.get('clarify_question') else '正常流程'}")
    if state.get("clarify_question"):
        print(f"  ⇢ 反问用户：{state['clarify_question']}")
    if state.get("solution"):
        print(f"  ⇢ 方案（截断）：{state['solution'][:120]}…")
    print("  轨迹：", " → ".join(f"{t['node']}({t['latency_ms']:.0f}ms)"
                                 for t in state["trace"]))
    print("  成本：", json.dumps(cost_summary(state["cost_items"]), ensure_ascii=False))
    print("  最终：", json.dumps({k: state["final"][k] for k in
                                  ("need_human", "confidence", "cost_yuan", "clarify_question")},
                                 ensure_ascii=False))


async def main():
    """先跑信息不全的一轮，再跑补全后的一轮。"""
    app = build_main_graph()

    s1 = await app.ainvoke(FIRST)
    brief(s1, "第 1 轮：信息不全，触发追问")

    second = dict(FIRST)
    second["ticket_id"] = "TK-2026-000327-R1"
    second["raw_text"] = FIRST["raw_text"] + "\n【用户补充】" + SUPPLEMENT
    second["trace"], second["errors"], second["cost_items"] = [], [], []
    s2 = await app.ainvoke(second)
    brief(s2, "第 2 轮：用户补全后，正常走完全流程")

    c1 = cost_summary(s1["cost_items"])["total_yuan"]
    c2 = cost_summary(s2["cost_items"])["total_yuan"]
    print(f"\n两轮合计成本 ¥{c1 + c2:.6f}；"
          f"若不追问、直接拿'机器不转了'去检索生成，"
          f"一次无效生成也要 ¥0.004 左右，且答案没法用。")


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python demo/demo_incomplete.py
```

```text
═══ 第 1 轮：信息不全，触发追问 ═══
  抽取： {"device_model": null, "fault_symptom": "设备无法转动并有焦糊味", "fault_code": null, "urgency": 3, "production_line": null, "suggested_craft": "电气", "need_parts": false, "intent": "报修"}
  valid=False　missing=['device_model']　route=clarify
  ⇢ 反问用户：方便提供一下设备型号吗？型号一般在电机铭牌或控制柜门内侧，例如 XC-200-4P。
  轨迹： preprocess(0ms) → extract(392ms) → validate(0ms) → clarify(0ms) → finalize(1ms)
  成本： {"total_yuan": 1.6e-05, "by_node": {"extract": 1.6e-05}, "by_tier": {"small": 1.6e-05}, "small_share": 1.0, "n_calls": 1}
  最终： {"need_human": false, "confidence": 0.8, "cost_yuan": 1.6e-05, "clarify_question": "方便提供一下设备型号吗？型号一般在电机铭牌或控制柜门内侧，例如 XC-200-4P。"}

═══ 第 2 轮：用户补全后，正常走完全流程 ═══
  抽取： {"device_model": "XC-300-4P", "fault_symptom": "设备无法启动并有焦糊味", "fault_code": null, "urgency": 3, "production_line": "三号线", "suggested_craft": "电气", "need_parts": false, "intent": "报修"}
  valid=True　missing=[]　route=正常流程
  ⇢ 方案（截断）：初步判断：\nXC-300-4P 上电即跳闸并伴有焦糊味，属于绕组或电缆绝缘击穿的典型征兆，必须立即停止送电尝试 [1][3]…
  轨迹： preprocess(0ms) → extract(404ms) → validate(0ms) → retrieve(298ms) → generate(2903ms) → check_action(0ms) → finalize(1ms)
  成本： {"total_yuan": 0.004142, "by_node": {"extract": 1.6e-05, "generate": 0.004126}, "by_tier": {"small": 1.6e-05, "big": 0.004126}, "small_share": 0.0039, "n_calls": 2}
  最终： {"need_human": false, "confidence": 1.0, "cost_yuan": 0.004142, "clarify_question": null}

两轮合计成本 ¥0.004158；若不追问、直接拿'机器不转了'去检索生成，一次无效生成也要 ¥0.004 左右，且答案没法用。
```

**这条路径里有三个设计细节，都不显然**：

1. **"有点糊味"被抽成了 `urgency=3`**，不是因为模型聪明，而是 Step 6.3 的后置业务规则 `CRITICAL_KEYWORDS` 里有"焦糊"。`apply_business_rules` 在 schema 校验之后跑，**这条规则的可靠性是 100%，模型永远做不到**。安全相关的判断一律用代码兜。
2. **`confidence` 从 1.0 降到 0.8**（`finalize` 里 `missing_fields` 每个扣 0.2）。confidence 不是模型给的，是**流程算出来的**——降级、缺字段、无引用三种情况各自扣分。这个数字直接决定前端要不要显示"建议人工复核"。
3. **追问话术用模板而不是模型生成**。`clarify` 节点的成本是 0，而且话术里带了"型号一般在电机铭牌或控制柜门内侧"这种**领域知识**——这种固定话术让模型生成是浪费钱，而且不稳定。

### 5.6 三条路径横向对比

```mermaid
flowchart LR
    S([工单进入]) --> P[preprocess<br/>0.5ms ¥0]
    P --> E[extract<br/>~420ms ¥0.000016]
    E --> V{validate}
    V -->|关键字段缺失<br/>且渠道=小程序| C[clarify<br/>0ms ¥0]
    C --> F3[finalize]
    V -->|字段齐备| R[retrieve<br/>~320ms ¥0]
    R --> G[generate<br/>~2900ms ¥0.004126]
    G --> A{check_action}
    A -->|62% 的工单| F1[finalize<br/>合计 ¥0.004142]
    A -->|38% 的工单| AG[Agent 子图<br/>~8000ms ¥0.009702<br/>含人工确认]
    AG --> F2[finalize<br/>合计 ¥0.013844]

    style E fill:#d4f5d4
    style C fill:#fff3e0
    style AG fill:#ffe0e0
    style G fill:#d4e6f5
```

| 维度 | 演示 1 普通工单 | 演示 2 紧急派单 | 演示 3 信息不全 |
|---|---|---|---|
| 走过的节点 | 7 个 | 8 个（含 Agent 子图内部 6 步） | 5 个（第 1 轮）/ 7 个（第 2 轮） |
| 模型调用次数 | 2（小 1 + 大 1） | 5（小 1 + 大 4） | 1（第 1 轮）/ 2（第 2 轮） |
| 端到端延迟 | 3,612 ms | 11,843 ms（不含人工等待） | 452 ms / 3,606 ms |
| 单单成本 | ¥0.004142 | ¥0.013844 | ¥0.000016 / ¥0.004142 |
| 小模型成本占比 | 0.39% | 0.12% | 100% / 0.39% |
| 是否需要人 | 否 | **是（dispatch 前必须确认）** | 是（等用户补信息，但不占客服工时） |
| 日均占比（示例性数据） | 54% | 9% | 37% |

**按日均 115 张工单加权的平均成本**：

$$
\bar{c} = 0.54 \times 0.004142 + 0.09 \times 0.013844 + 0.37 \times \left(\frac{0.000016 + 0.004142}{1}\right) \times 0.72 \approx 0.0046\ \text{元/单}
$$

（演示 3 里约 72% 的用户会补全信息进入第 2 轮，其余流失）→ **日均 ¥0.53，月均约 ¥16**。这个数字在 8.1 节会和"全部用大模型"的方案做对比，也是判断这套架构值不值得的关键输入。

### 5.7 本节小结

- 端口按全书约定统一：vLLM **8001**、应用 **8080**；对照实验的基座临时起在 8002，跑完就停；
- 三条路径的成本结构完全一致：**生成占 99%+，抽取占 0.4%**——这决定了优化方向（压生成，别压抽取）；
- `check_action` 用规则把 62% 的工单挡在 Agent 之外，这是整套系统能控制成本与延迟的最主要原因；
- "焦糊味 → urgency=3"靠的是 `CRITICAL_KEYWORDS` 代码规则，不是模型；**安全相关判断永远不要交给概率模型**；
- 人工拒绝派单后要**直接终止**（去 `summarize`），不能回 `agent_think`——否则模型会换个参数再申请一次；
- 追问路径的价值是"用 ¥0.000016 避免 ¥0.004 的无效生成"，而且它把"信息不全"这个**业务问题**显式化了，而不是让模型硬猜。

---

## 六、评测

三位一体系统的评测**不能只有一个数字**。抽取错了、检索错了、Agent 调错工具，这三种失败的修法完全不同，所以评测必须分层：

```mermaid
flowchart TB
    L1["第 1 层：字段级评测（本节 6.2）<br/>对象：微调抽取模型<br/>数据：data/sft/test.jsonl 800 条<br/>指标：逐字段准确率 / 格式合规率 / null 误判率 / 混淆矩阵<br/>频率：每次训练后 + 每次量化后 + 每周回归"]
    L2["第 2 层：端到端评测（本节 6.4）<br/>对象：整条主图（抽取+检索+生成+Agent）<br/>数据：eval/cases_e2e.jsonl 60 条<br/>指标：复用项目 1 的 harness（规则分 / judge 分 / 检索分）<br/>频率：每次 PR + 每日"]
    L3["第 3 层：线上抽检（8.5 节优化清单）<br/>对象：真实流量<br/>数据：每天随机抽 30 单<br/>指标：人工标注的字段准确率 + 客服采纳率<br/>频率：每日"]
    L1 -->|字段准了才谈得上流程| L2 -->|离线好不等于线上好| L3
    style L1 fill:#d4f5d4
    style L2 fill:#d4e6f5
    style L3 fill:#fff3e0
```

Step 4.2 的 `eval/eval_extract.py` 已经给出了第 1 层的骨架（格式合规率、字段准确率、意图 macro-F1、延迟、token）。本节把它**补齐成可以进门禁的版本**，补的是三样它没有的东西：

| 缺什么 | 为什么必须有 |
|---|---|
| **逐字段的模糊匹配** | `XC-200` vs `XC-200-4P` 是"型号族对了、规格错了"，和抽成 `BP-22KW` 是两种完全不同的错误，按精确匹配一律算错会让你看不到真实情况 |
| **null 误判率** | 这是可选字段最重要的指标，分两种：**编造率**（gold 是 null，模型编了一个）和**漏抽率**（gold 有值，模型填了 null）。编造率高会让下游派错工种，漏抽率高会触发多余的追问 |
| **混淆矩阵** | `urgency` 把 2 判成 3 只是多派了一次车，把 3 判成 1 会漏掉产线停机。**平均准确率掩盖方向性错误**，必须看矩阵 |

### 6.1 评测数据与口径

```bash
wc -l data/sft/test.jsonl eval/cases_e2e.jsonl
```

```text
   800 data/sft/test.jsonl
    60 eval/cases_e2e.jsonl
   860 total
```

口径约定（写进 `reports/eval_protocol.md`，改口径必须升版本）：

| 字段 | 匹配方式 | 说明 |
|---|---|---|
| `device_model` | 精确（归一化后）+ **型号族**模糊 | 归一化 = 大写、去空格、破折号统一；型号族 = 第一个 `-` 之前的部分，如 `XC-200-4P` → `XC` + `200` |
| `fault_code` | 精确（归一化后） | `e21` / `E21` / `报警E21` 都归一到 `E21`；**这个字段不允许模糊**，错一个字符就是错 |
| `urgency` | 精确 + 相邻容忍（±1）分别统计 | 相邻错误（2↔3）与跨级错误（1↔3）分开看 |
| `production_line` | 精确 + 中文数字归一 | `三号线` / `3号线` / `3线` 视为同一个 |
| `suggested_craft` | 精确（枚举） | 出现枚举外的值 = 格式不合规 |
| `need_parts` | 精确（布尔） | — |
| `intent` | 精确（枚举） + macro-F1 | 五分类，类别不均衡所以看 macro |
| `fault_symptom` | 字符级 F1（`char_f1`） | 自由文本，不做精确匹配 |

### 6.2 `eval/eval_fields.py`——字段级评测（完整代码）

这个脚本**复用** `eval_extract.py` 的模型调用与宽松解析逻辑，不重复实现。

```python
# eval/eval_fields.py
"""抽取任务的字段级深度评测：逐字段精确/模糊匹配、格式合规率、null 误判率、混淆矩阵。

复用 eval/eval_extract.py 的 call_model / parse_json_loose / char_f1 / run，
只扩展指标与报告部分。

用法：
    python eval/eval_fields.py --base-url http://127.0.0.1:8001/v1 \
        --model ticket-extract --tag "微调7B(AWQ)" \
        --save reports/eval_fields_finetuned.json \
        --md reports/eval_fields_finetuned.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas import TicketExtract
from eval.eval_extract import char_f1, run

# ---------------- 字段口径 ----------------
NULLABLE_FIELDS = ["device_model", "fault_code", "production_line"]
ENUM_FIELDS = {
    "suggested_craft": ["电气", "机械", "液压", "控制", "其他"],
    "intent": ["咨询", "报修", "投诉", "备件申请", "催办"],
    "urgency": [1, 2, 3],
}
EXACT_FIELDS = ["device_model", "fault_code", "urgency", "production_line",
                "suggested_craft", "need_parts", "intent"]
ALL_FIELDS = EXACT_FIELDS + ["fault_symptom"]

CN_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
          "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}


def norm_model(v: Any) -> str | None:
    """型号归一：大写、去空格、破折号统一。与 app/schemas.py 的 validator 保持一致。"""
    if v is None:
        return None
    s = str(v).strip().upper().replace(" ", "")
    s = re.sub(r"[—－–_]", "-", s)
    return s or None


def model_family(v: Any) -> str | None:
    """型号族：取字母前缀 + 第一段数字，如 XC-200-4P → XC200、BP-22KW → BP22。"""
    s = norm_model(v)
    if not s:
        return None
    m = re.match(r"([A-Z]+)-?(\d+)", s)
    return f"{m.group(1)}{m.group(2)}" if m else s


def norm_code(v: Any) -> str | None:
    """故障码归一：大写、去中文前缀与冒号。"""
    if v is None:
        return None
    s = str(v).strip().upper()
    for p in ("报警", "故障码", "错误码", "ERR=", "ERROR", "报", "CODE"):
        s = s.replace(p, "")
    s = s.strip("：: -")
    return s or None


def norm_line(v: Any) -> str | None:
    """产线归一：中文数字转阿拉伯、去掉"号线/线/#"等后缀词。"""
    if v is None:
        return None
    s = str(v).strip().upper().replace(" ", "")
    for cn, ar in CN_NUM.items():
        s = s.replace(cn, ar)
    s = re.sub(r"(号线|号|线|#|LINE)", "", s)
    return s or None


NORMALIZERS = {
    "device_model": norm_model,
    "fault_code": norm_code,
    "production_line": norm_line,
}


def norm_value(field: str, v: Any) -> Any:
    """按字段取归一化后的值。"""
    if field in NORMALIZERS:
        return NORMALIZERS[field](v)
    if field == "urgency":
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    if field == "need_parts":
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "是")
        return bool(v) if v is not None else None
    return str(v).strip() if v is not None else None


# ---------------- 格式合规 ----------------
def check_format(pred: dict | None) -> dict[str, Any]:
    """检查一条预测的格式合规性：能解析 / 键齐全 / 无多余键 / 枚举合法 / schema 通过。"""
    if pred is None:
        return {"parsable": False, "keys_ok": False, "enum_ok": False,
                "schema_ok": False, "extra_keys": [], "missing_keys": ALL_FIELDS,
                "enum_violations": []}
    missing = [f for f in ALL_FIELDS if f not in pred]
    extra = [k for k in pred if k not in ALL_FIELDS]
    violations = []
    for f, allowed in ENUM_FIELDS.items():
        if f in pred and pred[f] is not None:
            v = int(pred[f]) if f == "urgency" and str(pred[f]).isdigit() else pred[f]
            if v not in allowed:
                violations.append(f"{f}={pred[f]!r}")
    try:
        TicketExtract.model_validate(pred)
        schema_ok, schema_err = True, ""
    except ValidationError as e:
        schema_ok, schema_err = False, str(e).splitlines()[0][:80]
    return {"parsable": True, "keys_ok": not missing and not extra,
            "enum_ok": not violations, "schema_ok": schema_ok,
            "extra_keys": extra, "missing_keys": missing,
            "enum_violations": violations, "schema_err": schema_err}


# ---------------- 逐字段比对 ----------------
def compare_field(field: str, gold: Any, pred: Any) -> str:
    """返回该字段的比对结果标签。

    exact      精确一致（含两者皆 null）
    family     型号族一致但规格不同（仅 device_model）
    adjacent   相邻级别错误（仅 urgency，差 1 级）
    fabricated gold 为 null 但模型给了值（编造）
    omitted    gold 有值但模型给了 null（漏抽）
    wrong      其它错误
    """
    g, p = norm_value(field, gold), norm_value(field, pred)
    if g == p:
        return "exact"
    if g is None and p is not None:
        return "fabricated"
    if g is not None and p is None:
        return "omitted"
    if field == "device_model" and model_family(gold) == model_family(pred):
        return "family"
    if field == "urgency" and isinstance(g, int) and isinstance(p, int) and abs(g - p) == 1:
        return "adjacent"
    return "wrong"


def confusion(rows: list[dict], field: str, labels: list) -> dict:
    """构造某个枚举字段的混淆矩阵与逐类 P/R/F1。行 = gold，列 = pred。"""
    cm = {g: Counter() for g in labels}
    other = Counter()
    for r in rows:
        if not r["ok"]:
            continue
        g = norm_value(field, r["gold"].get(field))
        p = norm_value(field, (r["pred"] or {}).get(field))
        if g in cm:
            cm[g][p if p in labels else "非法"] += 1
        else:
            other[g] += 1
    stats = {}
    for lb in labels:
        tp = cm[lb][lb]
        fn = sum(v for k, v in cm[lb].items() if k != lb)
        fp = sum(cm[g][lb] for g in labels if g != lb)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        stats[lb] = {"support": tp + fn, "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    macro_f1 = sum(s["f1"] for s in stats.values()) / len(labels) if labels else 0.0
    return {"matrix": {g: dict(c) for g, c in cm.items()}, "stats": stats,
            "macro_f1": macro_f1, "gold_out_of_enum": dict(other)}


def summarize_fields(rows: list[dict]) -> dict[str, Any]:
    """汇总所有字段级指标。"""
    n = len(rows)
    fmt = [check_format(r.get("pred")) for r in rows]
    fmt_stat = {
        "parsable": sum(1 for f in fmt if f["parsable"]) / n,
        "keys_ok": sum(1 for f in fmt if f["keys_ok"]) / n,
        "enum_ok": sum(1 for f in fmt if f["enum_ok"]) / n,
        "schema_ok": sum(1 for f in fmt if f["schema_ok"]) / n,
        "enum_violation_samples": [v for f in fmt for v in f["enum_violations"]][:10],
        "extra_key_samples": sorted({k for f in fmt for k in f["extra_keys"]})[:10],
    }

    per_field: dict[str, Counter] = {f: Counter() for f in EXACT_FIELDS}
    for r in rows:
        if not r["ok"]:
            for f in EXACT_FIELDS:
                per_field[f]["unparsable"] += 1
            continue
        for f in EXACT_FIELDS:
            per_field[f][compare_field(f, r["gold"].get(f), (r["pred"] or {}).get(f))] += 1

    field_table = {}
    for f, c in per_field.items():
        total = sum(c.values())
        strict = c["exact"] / total
        loose = (c["exact"] + c["family"] + c["adjacent"]) / total
        field_table[f] = {
            "n": total, "strict_acc": strict, "loose_acc": loose,
            "exact": c["exact"], "family": c["family"], "adjacent": c["adjacent"],
            "fabricated": c["fabricated"], "omitted": c["omitted"],
            "wrong": c["wrong"], "unparsable": c["unparsable"],
        }

    # null 误判率：只在可空字段上算，分母分别是「gold 为 null 的条数」与「gold 有值的条数」
    null_stat = {}
    for f in NULLABLE_FIELDS:
        g_null = [r for r in rows if r["ok"] and norm_value(f, r["gold"].get(f)) is None]
        g_val = [r for r in rows if r["ok"] and norm_value(f, r["gold"].get(f)) is not None]
        fab = sum(1 for r in g_null if norm_value(f, (r["pred"] or {}).get(f)) is not None)
        omi = sum(1 for r in g_val if norm_value(f, (r["pred"] or {}).get(f)) is None)
        null_stat[f] = {
            "gold_null_n": len(g_null), "gold_value_n": len(g_val),
            "fabricate_rate": fab / len(g_null) if g_null else 0.0,
            "omit_rate": omi / len(g_val) if g_val else 0.0,
            "fabricated": fab, "omitted": omi,
        }

    sym = [char_f1(r["gold"].get("fault_symptom", ""),
                   str((r["pred"] or {}).get("fault_symptom") or ""))
           for r in rows if r["ok"]]
    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] > 0)

    # 整条记录完全正确的比例：这是业务最关心的「这一单不用人碰」
    full_ok = sum(1 for r in rows if r["ok"]
                  and all(compare_field(f, r["gold"].get(f), (r["pred"] or {}).get(f)) == "exact"
                          for f in EXACT_FIELDS))

    return {
        "n": n,
        "format": fmt_stat,
        "fields": field_table,
        "null": null_stat,
        "symptom_char_f1": sum(sym) / len(sym) if sym else 0.0,
        "record_exact_rate": full_ok / n,
        "field_strict_avg": sum(v["strict_acc"] for v in field_table.values()) / len(field_table),
        "field_loose_avg": sum(v["loose_acc"] for v in field_table.values()) / len(field_table),
        "p50_ms": lat[int(len(lat) * 0.50)] if lat else 0,
        "p95_ms": lat[int(len(lat) * 0.95)] if lat else 0,
        "avg_prompt_tokens": sum(r["pt"] for r in rows) / n,
        "avg_completion_tokens": sum(r["ct"] for r in rows) / n,
        "confusion": {
            "urgency": confusion(rows, "urgency", [1, 2, 3]),
            "intent": confusion(rows, "intent", ENUM_FIELDS["intent"]),
            "suggested_craft": confusion(rows, "suggested_craft", ENUM_FIELDS["suggested_craft"]),
        },
        "by_channel": by_channel(rows),
    }


def by_channel(rows: list[dict]) -> dict[str, Any]:
    """分渠道拆指标（Step 4.5 的结论：小程序渠道是短板）。"""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get("channel") or "未知"].append(r)
    out = {}
    for ch, rs in buckets.items():
        ok = [r for r in rs if r["ok"]]
        if not ok:
            out[ch] = {"n": len(rs), "format_ok": 0.0, "field_strict_avg": 0.0}
            continue
        accs = []
        for f in EXACT_FIELDS:
            hit = sum(1 for r in ok
                      if compare_field(f, r["gold"].get(f), (r["pred"] or {}).get(f)) == "exact")
            accs.append(hit / len(ok))
        out[ch] = {"n": len(rs), "format_ok": len(ok) / len(rs),
                   "field_strict_avg": sum(accs) / len(accs)}
    return out


# ---------------- 报告渲染 ----------------
def _tbl(headers: list[str], rows: list[list]) -> str:
    """渲染 Markdown 表格。"""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_md(s: dict, tag: str) -> str:
    """渲染字段级评测报告。"""
    L = [f"# 抽取字段级评测报告 · {tag}", "",
         f"- 测试集：{s['n']} 条", f"- 延迟 P50/P95：{s['p50_ms']:.0f} / {s['p95_ms']:.0f} ms",
         f"- 平均 token：prompt {s['avg_prompt_tokens']:.0f} / completion {s['avg_completion_tokens']:.0f}",
         "", "## 一、总览", "",
         _tbl(["指标", "值", "说明"], [
             ["JSON 可解析率", f"{s['format']['parsable']:.2%}", "能提取出 JSON 对象"],
             ["键集合正确率", f"{s['format']['keys_ok']:.2%}", "8 个键不多不少"],
             ["枚举合法率", f"{s['format']['enum_ok']:.2%}", "urgency/craft/intent 取值在枚举内"],
             ["**schema 通过率**", f"**{s['format']['schema_ok']:.2%}**", "pydantic 校验通过，可直接进下游"],
             ["**整单完全正确率**", f"**{s['record_exact_rate']:.2%}**", "7 个精确字段全对，这一单不用人碰"],
             ["字段严格准确率（均）", f"{s['field_strict_avg']:.2%}", "7 字段精确匹配平均"],
             ["字段宽松准确率（均）", f"{s['field_loose_avg']:.2%}", "型号族/相邻紧急度算对"],
             ["故障现象 charF1", f"{s['symptom_char_f1']:.4f}", "自由文本字段"],
         ]), "", "## 二、逐字段明细", "",
         _tbl(["字段", "n", "严格准确率", "宽松准确率", "精确", "型号族对", "相邻级", "编造", "漏抽", "其它错", "不可解析"],
              [[f, v["n"], f"{v['strict_acc']:.2%}", f"{v['loose_acc']:.2%}", v["exact"],
                v["family"] or "—", v["adjacent"] or "—", v["fabricated"], v["omitted"],
                v["wrong"], v["unparsable"]]
               for f, v in s["fields"].items()]),
         "", "## 三、null 误判率（可空字段）", "",
         _tbl(["字段", "gold 为 null 的条数", "**编造率**", "编造数", "gold 有值的条数", "**漏抽率**", "漏抽数"],
              [[f, v["gold_null_n"], f"**{v['fabricate_rate']:.2%}**", v["fabricated"],
                v["gold_value_n"], f"**{v['omit_rate']:.2%}**", v["omitted"]]
               for f, v in s["null"].items()]),
         "", "> 编造率高 → 下游会按错误型号订件/派工种；漏抽率高 → 触发多余追问，影响体验。两者要分开治。"]

    for i, (field, title) in enumerate((("urgency", "紧急度"), ("intent", "意图"),
                                        ("suggested_craft", "建议工种")), 1):
        c = s["confusion"][field]
        labels = list(c["matrix"].keys())
        cols = labels + ["非法"]
        L += ["", f"## 四.{i} {title}混淆矩阵（行=标准答案，列=模型预测）", "",
              _tbl([f"gold ↓ / pred →"] + [str(x) for x in cols] + ["召回率"],
                   [[str(g)] + [c["matrix"][g].get(p, 0) for p in cols]
                    + [f"{c['stats'][g]['recall']:.2%}"] for g in labels]),
              "",
              _tbl(["类别", "support", "precision", "recall", "F1"],
                   [[str(lb), v["support"], f"{v['precision']:.2%}", f"{v['recall']:.2%}",
                     f"{v['f1']:.4f}"] for lb, v in c["stats"].items()]
                   + [["**macro-F1**", "", "", "", f"**{c['macro_f1']:.4f}**"]])]

    L += ["", "## 五、分渠道", "",
          _tbl(["渠道", "条数", "schema 通过率", "字段严格准确率（均）"],
               [[ch, v["n"], f"{v['format_ok']:.2%}", f"{v['field_strict_avg']:.2%}"]
                for ch, v in sorted(s["by_channel"].items(), key=lambda x: -x[1]["n"])])]
    if s["format"]["enum_violation_samples"]:
        L += ["", "## 六、格式异常样例", "",
              "```text", *s["format"]["enum_violation_samples"], "```"]
    L += ["", "---", "",
          "> 本报告的数字口径见 `reports/eval_protocol.md`。**改口径必须升版本并重跑历史模型**，"
          "否则不同版本的报告不可比。"]
    return "\n".join(L)


def main():
    """命令行入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="data/sft/test.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="ticket-extract")
    ap.add_argument("--tag", default="finetuned")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    rows = asyncio.run(run(args))
    s = summarize_fields(rows)

    print(f"\n===== {args.tag}（n={s['n']}）=====")
    print(f"schema 通过率      : {s['format']['schema_ok']:.2%}")
    print(f"整单完全正确率     : {s['record_exact_rate']:.2%}")
    print(f"字段严格/宽松准确率: {s['field_strict_avg']:.2%} / {s['field_loose_avg']:.2%}")
    print(f"故障现象 charF1    : {s['symptom_char_f1']:.4f}")
    print(f"延迟 P50/P95       : {s['p50_ms']:.0f} / {s['p95_ms']:.0f} ms")
    print("\nnull 误判：")
    for f, v in s["null"].items():
        print(f"  {f:<16} 编造率 {v['fabricate_rate']:.2%}（{v['fabricated']}/{v['gold_null_n']}）"
              f"　漏抽率 {v['omit_rate']:.2%}（{v['omitted']}/{v['gold_value_n']}）")

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(render_md(s, args.tag), encoding="utf-8")
        print(f"\nMarkdown 报告 -> {args.md}")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")
        print(f"JSON 结果 -> {args.save}")


if __name__ == "__main__":
    main()
```

### 6.3 运行与输出

```bash
# 组 C：微调 + AWQ（端口 8001）
python eval/eval_fields.py --base-url http://127.0.0.1:8001/v1 \
  --model ticket-extract --tag "微调7B(QLoRA+AWQ)" \
  --save reports/eval_fields_finetuned.json \
  --md reports/eval_fields_finetuned.md

# 组 B：大模型直接抽取（同一套 prompt、同一份测试集）
python eval/eval_fields.py --base-url https://api.deepseek.com/v1 \
  --api-key "$LLM_API_KEY" --model deepseek-chat --concurrency 4 \
  --tag "deepseek-chat(prompt工程)" \
  --save reports/eval_fields_deepseek.json \
  --md reports/eval_fields_deepseek.md
```

```text
===== 微调7B(QLoRA+AWQ)（n=800）=====
schema 通过率      : 99.50%
整单完全正确率     : 78.75%
字段严格/宽松准确率: 94.34% / 96.71%
故障现象 charF1    : 0.8874
延迟 P50/P95       : 412 / 736 ms

null 误判：
  device_model     编造率 2.35%（4/170）　漏抽率 1.75%（11/630）
  fault_code       编造率 1.42%（6/422）　漏抽率 2.12%（8/378）
  production_line  编造率 3.11%（9/289）　漏抽率 3.33%（17/511）

Markdown 报告 -> reports/eval_fields_finetuned.md
JSON 结果 -> reports/eval_fields_finetuned.json
```

生成的 `reports/eval_fields_finetuned.md` 全文（**示例性数据**）：

````markdown
# 抽取字段级评测报告 · 微调7B(QLoRA+AWQ)

- 测试集：800 条
- 延迟 P50/P95：412 / 736 ms
- 平均 token：prompt 447 / completion 88

## 一、总览

| 指标 | 值 | 说明 |
|---|---|---|
| JSON 可解析率 | 99.75% | 能提取出 JSON 对象 |
| 键集合正确率 | 99.63% | 8 个键不多不少 |
| 枚举合法率 | 99.63% | urgency/craft/intent 取值在枚举内 |
| **schema 通过率** | **99.50%** | pydantic 校验通过，可直接进下游 |
| **整单完全正确率** | **78.75%** | 7 个精确字段全对，这一单不用人碰 |
| 字段严格准确率（均） | 94.34% | 7 字段精确匹配平均 |
| 字段宽松准确率（均） | 96.71% | 型号族/相邻紧急度算对 |
| 故障现象 charF1 | 0.8874 | 自由文本字段 |

## 二、逐字段明细

| 字段 | n | 严格准确率 | 宽松准确率 | 精确 | 型号族对 | 相邻级 | 编造 | 漏抽 | 其它错 | 不可解析 |
|---|---|---|---|---|---|---|---|---|---|---|
| device_model | 800 | 97.63% | 98.75% | 781 | 9 | — | 4 | 11 | 3 | 2 |
| fault_code | 800 | 98.25% | 98.25% | 786 | — | — | 6 | 8 | 0 | 2 |
| urgency | 800 | 91.13% | 98.13% | 729 | — | 56 | 0 | 0 | 13 | 2 |
| production_line | 800 | 95.88% | 95.88% | 767 | — | — | 9 | 17 | 5 | 2 |
| suggested_craft | 800 | 93.50% | 93.50% | 748 | — | — | 0 | 0 | 50 | 2 |
| need_parts | 800 | 90.63% | 90.63% | 725 | — | — | 0 | 0 | 73 | 2 |
| intent | 800 | 93.38% | 93.38% | 747 | — | — | 0 | 0 | 51 | 2 |

## 三、null 误判率（可空字段）

| 字段 | gold 为 null 的条数 | **编造率** | 编造数 | gold 有值的条数 | **漏抽率** | 漏抽数 |
|---|---|---|---|---|---|---|
| device_model | 170 | **2.35%** | 4 | 630 | **1.75%** | 11 |
| fault_code | 422 | **1.42%** | 6 | 378 | **2.12%** | 8 |
| production_line | 289 | **3.11%** | 9 | 511 | **3.33%** | 17 |

> 编造率高 → 下游会按错误型号订件/派工种；漏抽率高 → 触发多余追问，影响体验。两者要分开治。

## 四.1 紧急度混淆矩阵（行=标准答案，列=模型预测）

| gold ↓ / pred → | 1 | 2 | 3 | 非法 | 召回率 |
|---|---|---|---|---|---|
| 1 | 194 | 22 | 2 | 0 | 88.99% |
| 2 | 19 | 342 | 17 | 0 | 90.48% |
| 3 | 3 | 15 | 184 | 0 | 91.09% |

| 类别 | support | precision | recall | F1 |
|---|---|---|---|---|
| 1 | 218 | 89.81% | 88.99% | 0.8940 |
| 2 | 378 | 90.24% | 90.48% | 0.9036 |
| 3 | 202 | 90.64% | 91.09% | 0.9086 |
| **macro-F1** | | | | **0.9021** |

## 四.2 意图混淆矩阵（行=标准答案，列=模型预测）

| gold ↓ / pred → | 咨询 | 报修 | 投诉 | 备件申请 | 催办 | 非法 | 召回率 |
|---|---|---|---|---|---|---|---|
| 咨询 | 138 | 9 | 0 | 2 | 0 | 0 | 92.62% |
| 报修 | 7 | 396 | 3 | 5 | 2 | 0 | 95.88% |
| 投诉 | 1 | 6 | 42 | 0 | 4 | 0 | 79.25% |
| 备件申请 | 3 | 4 | 0 | 111 | 1 | 0 | 93.28% |
| 催办 | 0 | 2 | 5 | 1 | 60 | 0 | 88.24% |

| 类别 | support | precision | recall | F1 |
|---|---|---|---|---|
| 咨询 | 149 | 92.62% | 92.62% | 0.9262 |
| 报修 | 413 | 94.96% | 95.88% | 0.9542 |
| 投诉 | 53 | 84.00% | 79.25% | 0.8155 |
| 备件申请 | 119 | 93.28% | 93.28% | 0.9328 |
| 催办 | 68 | 89.55% | 88.24% | 0.8889 |
| **macro-F1** | | | | **0.9035** |

## 四.3 建议工种混淆矩阵（行=标准答案，列=模型预测）

| gold ↓ / pred → | 电气 | 机械 | 液压 | 控制 | 其他 | 非法 | 召回率 |
|---|---|---|---|---|---|---|---|
| 电气 | 312 | 8 | 1 | 14 | 2 | 0 | 92.58% |
| 机械 | 6 | 221 | 7 | 1 | 3 | 0 | 92.86% |
| 液压 | 1 | 5 | 96 | 0 | 1 | 0 | 93.20% |
| 控制 | 11 | 1 | 0 | 84 | 2 | 0 | 85.71% |
| 其他 | 3 | 4 | 1 | 2 | 35 | 3 | 72.92% |

| 类别 | support | precision | recall | F1 |
|---|---|---|---|---|
| 电气 | 337 | 93.69% | 92.58% | 0.9313 |
| 机械 | 238 | 92.47% | 92.86% | 0.9266 |
| 液压 | 103 | 91.43% | 93.20% | 0.9231 |
| 控制 | 98 | 83.17% | 85.71% | 0.8442 |
| 其他 | 45 | 81.40% | 72.92% | 0.7692 |
| **macro-F1** | | | | **0.8789** |

## 五、分渠道

| 渠道 | 条数 | schema 通过率 | 字段严格准确率（均） |
|---|---|---|---|
| 400电话 | 401 | 99.75% | 96.38% |
| 小程序 | 277 | 99.28% | 90.02% |
| 经销商系统 | 122 | 100.00% | 97.11% |

## 六、格式异常样例

```text
urgency='紧急'
suggested_craft='电气/控制'
intent='咨询报修'
```

---

> 本报告的数字口径见 `reports/eval_protocol.md`。**改口径必须升版本并重跑历史模型**，否则不同版本的报告不可比。
````

**这份报告比 Step 4.4 那张表多告诉了我们五件事**：

| 新发现 | 数据 | 该做什么 |
|---|---|---|
| **整单完全正确率只有 78.75%** | 7 字段全对的比例 | 字段平均 94.34% 听起来很美，但业务关心的是"这一单能不能直接用"。**汇报要用 78.75% 这个数**，它才是"人工兜底工作量"的直接来源 |
| `urgency` 的错误 **92% 是相邻级** | 56 相邻 / 13 跨级 | 严格准确率 91.13% 但宽松准确率 98.13%。1↔2 混淆无所谓，**真正危险的是 3 被判成 1（只有 3 例）**，这 3 例要逐条看 |
| `need_parts` 是最差字段（90.63%） | 73 条纯错 | 这个字段的标注本身就模糊（"可能要换件"算不算？），**先去修标注口径，别急着加训练数据** |
| `suggested_craft` 的错误集中在"控制 ↔ 电气"（11+14=25 例） | 混淆矩阵 | 这两个工种在业务上本来就有重叠。解法不是训练，而是**下游派单时把"控制"归到电气组**，或在标注规范里给出明确判据 |
| `其他` 类工种 F1 只有 0.77，且有 3 条枚举非法 | 45 条 support | 小类别样本太少。**要么上采样，要么干脆合并掉这个类别** |

### 6.4 端到端评测：复用项目 1 的 harness

字段准了不代表整条链路好用。端到端评测直接**复用项目 1 的 DeepSeek-Harness**——只要写一个适配器，把三位一体系统包成 `BaseSystem` 就行。这正是项目 1 里 `adapter.py` 那套抽象的价值：**新增一个被测系统 = 一个类 + 一行 yaml**。

`eval/adapters/s4_trinity.py`（放在评测仓库里，不放业务仓库）：

```python
# eval/adapters/s4_trinity.py
"""把三位一体系统包成项目 1 harness 的被测系统。

依赖：本文件跑在评测仓库（huacheng-eval）里，通过 HTTP 调用业务服务（huacheng-trinity:8080），
而不是直接 import 业务代码——这样评测与业务可以独立部署、独立发版。
"""
from __future__ import annotations

import os
import time

import httpx

from goldset.schema import GoldItem
from harness.adapter import BaseSystem
from harness.types import RetrievedChunk, SystemOutput


class TrinitySystem(BaseSystem):
    """三位一体工单助手（LangChain + LoRA + Agent）。"""

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.base = self.params.get("base_url") or os.environ.get(
            "TRINITY_BASE_URL", "http://127.0.0.1:8080")
        self.timeout = float(self.params.get("timeout_s", 60))
        self.client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        """建连接并确认服务健康。"""
        self.client = httpx.AsyncClient(base_url=self.base, timeout=self.timeout)
        r = await self.client.get("/health")
        r.raise_for_status()
        info = r.json()
        # 把被测服务的版本写进 system_version，报告里才能追溯是哪个版本的模型
        self.version = (f"{info.get('app_version', '?')}"
                        f"+adapter{info.get('adapter_version', '?')}")

    async def teardown(self) -> None:
        """关连接。"""
        if self.client:
            await self.client.aclose()

    async def answer(self, item: GoldItem) -> SystemOutput:
        """把一道金标题当成一张工单投进去，把结果映射成 SystemOutput。"""
        t0 = time.perf_counter()
        try:
            r = await self.client.post("/api/ticket", json={
                "ticket_id": f"EVAL-{item.qid}",
                "channel": self.params.get("channel", "400电话"),
                "customer_id": "CUST-EVAL-0001",
                "raw_text": item.question,
                # 评测模式：禁用写操作工具，人工确认自动拒绝，避免污染业务数据
                "eval_mode": True,
            })
            r.raise_for_status()
            d = r.json()
        except Exception as e:                       # noqa: BLE001
            return SystemOutput(qid=item.qid, system=self.name,
                                system_version=self.version, answer="",
                                error=f"{type(e).__name__}: {e}",
                                latency_ms=int((time.perf_counter() - t0) * 1000))

        # 答案 = 方案正文；追问场景把追问话术当答案（拒答类题目靠它拿分）
        answer = d.get("solution") or d.get("clarify_question") or ""
        if d.get("agent_result", {}).get("narrative"):
            answer += "\n\n【已执行动作】" + d["agent_result"]["narrative"]

        ctxs = [
            RetrievedChunk(chunk_id=c.get("id", f"c{i}"),
                           item_id=c.get("item_id") or c.get("source", ""),
                           text=c.get("text", ""), score=float(c.get("score", 0)),
                           rank=i + 1, source=c.get("source", ""))
            for i, c in enumerate(d.get("docs") or [])
        ]
        usage = d.get("usage") or {}
        return SystemOutput(
            qid=item.qid, system=self.name, system_version=self.version,
            answer=answer, contexts=ctxs,
            latency_ms=int(d.get("latency_ms") or (time.perf_counter() - t0) * 1000),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            extra={
                "extract": d.get("extract"),
                "extract_degraded": d.get("extract_degraded"),
                "need_human": d.get("need_human"),
                "confidence": d.get("confidence"),
                "cost_yuan": d.get("cost_yuan"),
                "agent_actions": [a.get("tool") for a in
                                  (d.get("agent_result") or {}).get("actions", [])],
            },
        )
```

在评测仓库的 `configs/systems.yaml` 里加一段：

```yaml
s4_trinity:
  impl: eval.adapters.s4_trinity:TrinitySystem
  version: "auto"              # 实际版本由 /health 返回，启动时覆盖
  params:
    base_url: http://127.0.0.1:8080
    timeout_s: 90
    channel: 400电话
```

60 条端到端用例转成 harness 的金标格式：

```python
# eval/cases_to_goldset.py
"""把 eval/cases_e2e.jsonl 转成项目 1 harness 的 GoldItem JSONL。

cases_e2e.jsonl 每行形如：
{"id":"E2E-001","channel":"400电话","text":"...","expect":{"key_points":[...],
 "must_include":[...],"must_not_include":[...],"gold_docs":["DOC-XC200-4.3"],
 "type":"simple","difficulty":"easy","expect_tools":["check_stock"]}}
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("eval/cases_e2e.jsonl")
DST = Path("eval/goldset_e2e.jsonl")

QTYPE_MAP = {"simple": "simple", "multihop": "multihop", "compare": "compare",
             "aggregate": "aggregate", "refusal": "refusal", "adversarial": "adversarial"}


def main():
    """转换并做基本校验。"""
    out, n_bad = [], 0
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        e = c["expect"]
        qtype = QTYPE_MAP[e.get("type", "simple")]
        answerable = qtype != "refusal"
        gold_docs = e.get("gold_docs", [])
        if answerable and not gold_docs:
            n_bad += 1
            print(f"跳过 {c['id']}：可答题必须给 gold_docs（否则检索指标算不出来）")
            continue
        out.append({
            "qid": f"G-E2E-{c['id'][-3:]}",
            "question": c["text"],
            "qtype": qtype,
            "difficulty": e.get("difficulty", "medium"),
            "answerable": answerable,
            "reference_answer": e["reference_answer"],
            "key_points": e.get("key_points", []),
            "must_include": e.get("must_include", []),
            "must_not_include": e.get("must_not_include", []),
            "gold_item_ids": [] if qtype == "refusal" else gold_docs,
            "product_line": e.get("device_model"),
            "error_code": e.get("fault_code"),
            "tags": (["expect_tools:" + ",".join(e["expect_tools"])]
                     if e.get("expect_tools") else []),
            "source_items": gold_docs,
            "created_by": "manual",
            "reviewed_by": e.get("reviewed_by", "lijun@huacheng.example"),
            "version": "e2e-v1",
        })
    DST.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
    print(f"已写出 {DST}：{len(out)} 条（跳过 {n_bad} 条）")


if __name__ == "__main__":
    main()
```

跑端到端评测：

```bash
# 在业务仓库起服务（7.2 节的 FastAPI，端口 8080）
uvicorn app.server:app --host 0.0.0.0 --port 8080 &

# 在评测仓库跑 harness
python eval/cases_to_goldset.py
harness run --system s4_trinity --config configs/config_e2e.yaml
```

```text
──────────────────────────────────────── s4_trinity ────────────────────────────────────────
数据集：60 题，分布 {'simple': 21, 'multihop': 14, 'compare': 6, 'aggregate': 5, 'refusal': 8, 'adversarial': 6}
[setup] TrinitySystem v1.2.0+adapterticket-extract-v1.3.0
评测 s4_trinity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60/60 0:04:18 0:00:00
跑批完成：缓存命中 0 / 实际调用 60
LLM-Judge 打分中（含位置交换去偏，调用量 ×2）…
judge 进度 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 120/120 0:00:41 0:00:00
judge 仲裁触发 4 题（swap_delta > 3.0），改用 deepseek-reasoner 重判
报告已生成：reports/20260318-153044-s4_trinity/report.md（总分 79.86）
```

端到端结果摘要（**示例性数据**，与项目 1 的三个系统放在一起看）：

| 系统 | 加权总分 | 规则分 | Judge | 检索分 | 通过率 | 违规率 | P50 延迟 | 单题成本 |
|---|---|---|---|---|---|---|---|---|
| s1_pure_llm（纯大模型，无检索） | 48.61 | 41.28 | 53.47 | — | 31.0% | 28.7% | 2,180 ms | ¥0.0053 |
| s2_basic_rag（基础 RAG） | 67.34 | 63.05 | 69.82 | 61.73 | 62.3% | 12.3% | 2,860 ms | ¥0.0060 |
| s3_opt_rag（优化 RAG） | 81.42 | 78.94 | 83.61 | 79.85 | 84.0% | 4.3% | 3,180 ms | ¥0.0066 |
| **s4_trinity（三位一体）** | **79.86** | 77.12 | 81.94 | 78.43 | 81.7% | 5.0% | **3,612 ms** | **¥0.0046** |

> 两点必须诚实说明：
> 1. **s4 的总分略低于 s3（79.86 vs 81.42）**，而且用的是不同的测试集（60 条端到端用例 vs 300 条金标集），**这两个数字严格来说不可直接比**。放在一起只是为了看量级。
> 2. s4 的价值不在分数，而在**它多做了 s3 做不到的事**：抽取出结构化字段、按意图路由、触发工具动作、在信息不全时主动追问。这些能力在"问答式"金标集里根本得不到分。**这就是为什么第 3 层（线上抽检）不可省略**——离线指标无法度量"帮客服省了多少事"。

### 6.5 核心对照表：微调小模型 vs 大模型直接抽取

这是本项目最需要给老板看的一张表。三项对比，一项都不能少。

**实测环境**：RTX 4090 24G 跑 vLLM 0.6.3（AWQ 4bit），测试集 800 条，`deepseek-chat` 走公网 API，抽取并发 8（大模型并发 4，受限流约束），计价按 `app/routing/model_router.py` 的 `MODEL_PRICE`（**以各家官方最新定价为准**）。**以下全部为示例性数据，必须自行复现。**

| 维度 | 指标 | A. 大模型直接抽取<br/>`deepseek-chat` + 6-shot | B. **微调小模型**<br/>Qwen2.5-7B + QLoRA + AWQ | B vs A |
|---|---|---|---|---|
| **准确率** | schema 通过率 | 97.75% | **99.50%** | +1.75pp |
| | 整单完全正确率 | 71.13% | **78.75%** | +7.62pp |
| | 字段严格准确率（均） | 89.71% | **94.34%** | +4.63pp |
| | `device_model` | 95.13% | **97.63%** | +2.50pp |
| | `fault_code` | 96.00% | **98.25%** | +2.25pp |
| | `urgency`（主观判断类） | 82.38% | **91.13%** | **+8.75pp** |
| | `suggested_craft`（主观判断类） | 85.25% | **93.50%** | **+8.25pp** |
| | 意图 macro-F1 | 0.8815 | **0.9035** | +0.022 |
| | `device_model` 编造率 | 5.88% | **2.35%** | -3.53pp |
| **延迟** | P50 | 1,842 ms | **412 ms** | **-78%** |
| | P95 | 3,376 ms | **736 ms** | **-78%** |
| | 吞吐（并发 32） | 受 API 限流，约 6 req/s | **31.7 req/s** | **+428%** |
| **成本** | 平均 prompt token | 1,912（含 6-shot） | **447** | -77% |
| | 平均 completion token | 96 | **88** | -8% |
| | 单次抽取成本 | ¥0.004304 | **¥0.000016**（电费等效） | **-99.6%** |
| | 日均 115 单的抽取月成本 | ¥14.85 | **¥0.06** | -¥14.79 |
| | 回填 28 万条历史工单的抽取成本 | **¥1,205** | **¥4.5 + 约 2.5 小时 GPU** | **-99.6%** |
| **其它** | 一次性训练成本 | ¥0 | 云 4090 约 4 小时 ≈ ¥12~32 + 约 20 人时数据工程 | — |
| | 数据不出内网 | ❌ 工单要发到公网 | ✅ 全程内网 | 合规收益 |
| | 行为可回归 | ❌ 改 prompt 全量漂移 | ✅ 版本化 adapter + 固定 test.jsonl | 工程收益 |

**读这张表的正确方式**（这几句比表本身重要）：

1. **微调的准确率收益集中在"主观判断类字段"**：`urgency` +8.75pp、`suggested_craft` +8.25pp，而客观字段（型号、故障码）只有 +2pp 多。原因在 Step 4.4 已经说过：**公司内部的判断规范学不进 prompt，只能学进权重**。如果你的任务只有客观字段，微调的性价比会低很多。
2. **延迟收益是确定的、且与调用量无关**：412 ms vs 1842 ms。这一项不受调用量影响，是**本地小模型的结构性优势**（不过公网、prompt 短 77%）。
3. **成本收益几乎完全取决于调用量**。日均 115 单时一个月只省 ¥14.79——**这点钱买不来任何东西**，连一次会议的时间成本都不够。真正让成本账算得过来的是两个场景：
   - **批量回填**：28 万条历史工单，省 ¥1,200 和大量等待时间；
   - **未来放量**：如果工单量涨到日均 2,000 单（放开经销商自助报修），月成本从 ¥258 降到 ¥1。
4. **别忽略那两行"其它"**：数据不出内网（合规）和行为可回归（工程），在很多企业里**比省钱重要得多**。第 8.4 节会给一张清单，帮你判断这些收益在你的场景里值不值。

### 6.6 本节小结

- 评测必须分三层：字段级（管抽取）、端到端（管流程）、线上抽检（管真实价值），**任何一层单独看都会得出错误结论**；
- 字段级评测要补三样东西：**逐字段模糊匹配**（区分"型号族对了"和"完全抽错"）、**null 误判率**（编造率 vs 漏抽率要分开治）、**混淆矩阵**（平均准确率掩盖方向性错误）；
- **整单完全正确率（78.75%）才是业务指标**，字段平均准确率（94.34%）是技术指标，汇报时别混用；
- `urgency` 的错误 92% 是相邻级、`suggested_craft` 的错误集中在"控制↔电气"——这两类问题的解法是**改标注口径和下游规则，不是加训练数据**；
- 端到端评测直接复用项目 1 的 harness，只写一个 90 行的适配器；评测通过 HTTP 调服务而不是 import 业务代码，**评测与业务可以独立发版**；
- 微调的三项收益：准确率 +4.63pp（集中在主观字段）、延迟 -78%（结构性优势）、成本 -99.6%（**但绝对金额取决于调用量**）。这张表的诚实版本在 8.1 和 8.4 节。

---

## 七、部署

这一节把三位一体系统装进容器跑起来。和普通 Web 服务比，它多了三个麻烦：

| 麻烦 | 具体表现 | 本节的解法 |
|---|---|---|
| **要带一块 GPU** | vLLM 要独占显存，不能和应用挤在一个容器里 | 训练环境 / 推理环境 / 应用环境**三个镜像分开**，compose 里 vLLM 单独一个 service 并声明 GPU |
| **要分发一个模型 adapter** | adapter 80MB、合并后 15GB、量化后 5.6GB，不能进 git、不能进镜像 | 7.5 节的 manifest + 对象存储 + 版本目录 + 软链切换 |
| **有状态（人工确认中断）** | checkpointer 里存着未完成的会话，重启不能丢 | checkpointer 用挂载出来的 SQLite（生产上换 Postgres） |

### 7.1 部署拓扑与端口

```mermaid
flowchart TB
    U["客服 / 工程师浏览器"] -->|HTTP 8080| APP
    CI["评测仓库 harness<br/>s4_trinity 适配器"] -->|HTTP 8080| APP

    subgraph HOST["一台 24G 显存的机器（docker compose）"]
        APP["app<br/>FastAPI + LangGraph<br/>:8080"]
        VLLM["vllm-extract<br/>ticket-extract (AWQ)<br/>容器内 :8000 → 宿主机 :8001<br/>GPU 独占 ~6GB"]
        MOCK["mock-services<br/>ERP :9101 / CRM :9102 / 工单 :9103"]
        MILVUS["milvus-standalone<br/>:19530"]
        ETCD["etcd"]
        MINIO["minio<br/>:9000 / 控制台 :9001"]
        ATTU["attu<br/>:8000"]
        LF["langfuse<br/>:3001"]
        PG["postgres<br/>:5433"]
        CKPT[("checkpoints.sqlite<br/>挂载卷")]
    end

    APP -->|OpenAI 兼容| VLLM
    APP -->|公网| DS["deepseek-chat API"]
    APP --> MILVUS
    APP --> MOCK
    APP --> LF
    APP --- CKPT
    MILVUS --- ETCD
    MILVUS --- MINIO
    ATTU --> MILVUS
    LF --> PG

    style VLLM fill:#d4f5d4
    style APP fill:#d4e6f5
    style DS fill:#fff3e0
```

| 服务 | 容器内端口 | 宿主机端口 | 为什么是这个端口 |
|---|---|---|---|
| app（FastAPI） | 8080 | **8080** | 全书约定：应用服务 8080 |
| vllm-extract | 8000（vLLM 默认） | **8001** | 全书约定：vLLM 对外 8001。容器内保持默认 8000，映射时改 |
| attu（Milvus 控制台） | 3000 | **8000** | 全书约定：Attu 8000 |
| langfuse | 3000 | **3001** | 全书约定：Langfuse 3001 |
| postgres（langfuse 用） | 5432 | **5433** | 全书约定：Postgres 5433，避开本机已有的 5432 |
| milvus | 19530 / 9091 | 19530 / 9091 | Milvus 默认 |
| minio | 9000 / 9001 | 9000 / 9001 | Milvus 的对象存储后端 |
| mock 业务系统 | 9101-9103 | 9101-9103 | 与 `.env` 一致 |

> **容器内外端口不一致要写进文档**，否则下一个人一定会踩。记住规则：**`.env` 里给宿主机用的地址写 8001，compose 里给容器间调用的地址写 `http://vllm-extract:8000/v1`**（容器网络内走容器端口，不走映射端口）。

### 7.2 `app/server.py`——FastAPI 服务

服务要提供 5 个能力：健康检查（给 compose 和评测用）、同步处理一单、流式处理一单（给前端看进度）、人工确认恢复、以及查会话状态。

```python
# app/server.py
"""三位一体工单助手的 HTTP 服务（端口 8080）。"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.graph.main_graph import build_main_graph
from app.routing.model_router import cost_summary

APP_VERSION = os.environ.get("APP_VERSION", "1.2.0")
ADAPTER_MANIFEST = os.environ.get("ADAPTER_MANIFEST", "/models/current/manifest.json")
CKPT_PATH = os.environ.get("CHECKPOINT_PATH", "/data/checkpoints.sqlite")

_state: dict[str, Any] = {}


def adapter_info() -> dict[str, Any]:
    """读 adapter manifest（7.5 节生成），拿不到就返回 unknown。"""
    p = Path(ADAPTER_MANIFEST)
    if not p.exists():
        return {"adapter_version": "unknown", "base_model": "unknown"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"adapter_version": "invalid-manifest", "base_model": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建图与 checkpointer，关闭时释放。"""
    Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
    saver_cm = AsyncSqliteSaver.from_conn_string(CKPT_PATH)
    saver = await saver_cm.__aenter__()
    _state["saver_cm"] = saver_cm
    _state["graph"] = build_main_graph(checkpointer=saver)
    _state["adapter"] = adapter_info()
    _state["started_at"] = time.time()
    _state["counters"] = {"tickets": 0, "clarify": 0, "agent": 0, "degraded": 0,
                          "errors": 0, "cost_yuan": 0.0}
    yield
    await _state["saver_cm"].__aexit__(None, None, None)


app = FastAPI(title="华成机电 · 三位一体工单助手", version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------- 请求 / 响应模型 ----------------
class TicketIn(BaseModel):
    """一张待处理的工单。"""

    ticket_id: str = Field(..., description="工单号，评测时用 EVAL- 前缀")
    channel: Literal["400电话", "小程序", "经销商系统", "邮件"] = "400电话"
    raw_text: str = Field(..., min_length=2, max_length=4000)
    customer_id: Optional[str] = None
    region: str = "华东-苏州"
    eval_mode: bool = Field(False, description="评测模式：自动拒绝所有高风险写操作")


class ConfirmIn(BaseModel):
    """人工确认的决定。"""

    thread_id: str
    decision: Literal["approve", "reject"]
    operator: str = "unknown"
    note: str = ""


def _thread(ticket_id: str) -> dict:
    """一张工单一个 thread_id，便于中断恢复与重放。"""
    return {"configurable": {"thread_id": f"t-{ticket_id}"}}


def _inputs(t: TicketIn) -> dict:
    """把请求转成图的初始 state。"""
    d = {
        "ticket_id": t.ticket_id, "channel": t.channel, "raw_text": t.raw_text,
        "customer_id": t.customer_id, "region": t.region,
        "trace": [], "errors": [], "cost_items": [],
    }
    if t.eval_mode:
        # 评测模式：提前写入 reject，让 human_confirm 直接走「拒绝」分支，
        # 从而绝不会真的派单/建单，污染业务数据
        d["confirm_decision"] = "reject"
    return d


def _pack(values: dict) -> dict:
    """把最终 state 打包成响应体。"""
    final = values.get("final") or {}
    cost = cost_summary(values.get("cost_items") or [])
    return {
        **final,
        "docs": [{"id": d.get("id"), "item_id": d.get("item_id") or d.get("source"),
                  "source": d.get("source"), "score": d.get("score"),
                  "text": (d.get("text") or "")[:500]}
                 for d in (values.get("docs") or [])],
        "extract_degraded": values.get("extract_degraded", False),
        "usage": {
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in values.get("cost_items", [])),
            "completion_tokens": sum(c.get("completion_tokens", 0)
                                     for c in values.get("cost_items", [])),
            "by_node": cost["by_node"], "small_share": cost["small_share"],
        },
        "trace": values.get("trace", []),
        "app_version": APP_VERSION,
        "adapter_version": _state["adapter"].get("adapter_version"),
    }


def _bump(values: dict) -> None:
    """更新进程内计数器（给 /metrics 用）。"""
    c = _state["counters"]
    c["tickets"] += 1
    if values.get("clarify_question"):
        c["clarify"] += 1
    if values.get("agent_result"):
        c["agent"] += 1
    if values.get("extract_degraded"):
        c["degraded"] += 1
    if values.get("errors"):
        c["errors"] += 1
    c["cost_yuan"] = round(c["cost_yuan"] + float((values.get("final") or {}).get("cost_yuan", 0)), 6)


# ---------------- 接口 ----------------
@app.get("/health")
async def health():
    """健康检查：给 compose、评测适配器、看板用。"""
    s = get_settings()
    small_ok, milvus_ok = False, False
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{s.small_base_url.rstrip('/')}/models")
            small_ok = r.status_code == 200
    except Exception:                          # noqa: BLE001
        small_ok = False
    try:
        from pymilvus import MilvusClient
        MilvusClient(uri=s.milvus_uri).list_collections()
        milvus_ok = True
    except Exception:                          # noqa: BLE001
        milvus_ok = False

    ok = small_ok and milvus_ok
    return {
        "status": "ok" if ok else "degraded",
        "app_version": APP_VERSION,
        "adapter_version": _state["adapter"].get("adapter_version"),
        "base_model": _state["adapter"].get("base_model"),
        "small_model_ready": small_ok,
        "milvus_ready": milvus_ok,
        "uptime_s": round(time.time() - _state["started_at"], 1),
    }


@app.get("/metrics")
async def metrics():
    """极简指标（生产上换成 Prometheus exporter）。"""
    c = dict(_state["counters"])
    n = max(1, c["tickets"])
    return {**c,
            "clarify_rate": round(c["clarify"] / n, 4),
            "agent_rate": round(c["agent"] / n, 4),
            "degraded_rate": round(c["degraded"] / n, 4),
            "avg_cost_yuan": round(c["cost_yuan"] / n, 6)}


@app.post("/api/extract")
async def api_extract(t: TicketIn):
    """只跑抽取（给批量回填历史工单用，不走检索与生成）。"""
    from app.chains.extract_chain import ExtractService
    svc = ExtractService(use_finetuned=True)
    r = await svc.aextract(t.raw_text, t.channel)
    return {"ticket_id": t.ticket_id, "extract": r["extract"].model_dump(),
            "latency_ms": round(r["latency_ms"], 1), "degraded": r["degraded"],
            "error": r["error"], "adapter_version": _state["adapter"].get("adapter_version")}


@app.post("/api/ticket")
async def api_ticket(t: TicketIn):
    """同步处理一张工单。若中途需要人工确认，返回 pending 状态与 thread_id。"""
    graph, cfg = _state["graph"], _thread(t.ticket_id)
    try:
        await graph.ainvoke(_inputs(t), cfg)
    except Exception as e:                     # noqa: BLE001
        _state["counters"]["errors"] += 1
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    snap = await graph.aget_state(cfg)
    if snap.next:                              # 图挂起了，等人工确认
        msgs = snap.values.get("messages", [])
        pending = [{"name": tc["name"], "args": tc["args"]}
                   for tc in (getattr(msgs[-1], "tool_calls", None) or [])] if msgs else []
        return {"status": "pending_confirm", "thread_id": cfg["configurable"]["thread_id"],
                "interrupt_at": list(snap.next), "pending_actions": pending,
                "extract": snap.values.get("extract"),
                "solution": snap.values.get("solution"),
                "app_version": APP_VERSION}
    _bump(snap.values)
    return {"status": "done", "thread_id": cfg["configurable"]["thread_id"], **_pack(snap.values)}


@app.post("/api/confirm")
async def api_confirm(c: ConfirmIn):
    """人工确认后恢复执行。"""
    graph = _state["graph"]
    cfg = {"configurable": {"thread_id": c.thread_id}}
    snap = await graph.aget_state(cfg)
    if not snap.next:
        raise HTTPException(status_code=409, detail="该会话没有待确认的动作（可能已处理或已过期）")

    await graph.aupdate_state(cfg, {"confirm_decision": c.decision})
    await graph.ainvoke(None, cfg)
    final_snap = await graph.aget_state(cfg)
    _bump(final_snap.values)
    # 审计日志：谁在什么时候批了什么，必须落盘
    Path("/data/audit.log").open("a", encoding="utf-8").write(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "thread_id": c.thread_id,
        "decision": c.decision, "operator": c.operator, "note": c.note,
        "actions": [a.get("tool") for a in
                    ((final_snap.values.get("agent_result") or {}).get("actions") or [])],
    }, ensure_ascii=False) + "\n")
    return {"status": "done", "thread_id": c.thread_id, **_pack(final_snap.values)}


@app.get("/api/threads/{thread_id}")
async def api_thread(thread_id: str):
    """查会话状态（前端轮询 / 排查问题用）。"""
    snap = await _state["graph"].aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"thread_id": thread_id, "next": list(snap.next),
            "extract": snap.values.get("extract"), "final": snap.values.get("final"),
            "trace": snap.values.get("trace", [])}


@app.post("/api/ticket/stream")
async def api_ticket_stream(t: TicketIn):
    """流式处理：每个节点完成就推一条 SSE，前端能看到进度。"""
    graph, cfg = _state["graph"], _thread(t.ticket_id)

    async def gen():
        """逐节点推送更新。"""
        try:
            async for chunk in graph.astream(_inputs(t), cfg, stream_mode="updates"):
                for node, upd in chunk.items():
                    payload = {"node": node}
                    for k in ("extract", "clarify_question", "need_action", "action_reason",
                              "valid", "missing_fields", "solution", "citations"):
                        if upd and k in upd:
                            payload[k] = upd[k]
                    if upd and upd.get("trace"):
                        payload["steps"] = upd["trace"]
                    if upd and upd.get("docs"):
                        payload["n_docs"] = len(upd["docs"])
                    yield {"event": "node", "data": json.dumps(payload, ensure_ascii=False,
                                                               default=str)}
                    await asyncio.sleep(0)     # 让出事件循环，保证及时 flush
            snap = await graph.aget_state(cfg)
            if snap.next:
                msgs = snap.values.get("messages", [])
                pending = [{"name": tc["name"], "args": tc["args"]}
                           for tc in (getattr(msgs[-1], "tool_calls", None) or [])] if msgs else []
                yield {"event": "pending_confirm",
                       "data": json.dumps({"thread_id": cfg["configurable"]["thread_id"],
                                           "pending_actions": pending},
                                          ensure_ascii=False, default=str)}
            else:
                _bump(snap.values)
                yield {"event": "done", "data": json.dumps(_pack(snap.values),
                                                           ensure_ascii=False, default=str)}
        except Exception as e:                 # noqa: BLE001
            _state["counters"]["errors"] += 1
            yield {"event": "error", "data": json.dumps({"detail": f"{type(e).__name__}: {e}"},
                                                        ensure_ascii=False)}

    return EventSourceResponse(gen())


@app.get("/")
async def index():
    """演示前端。"""
    p = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if not p.exists():
        return {"msg": "前端文件不存在，直接用 /docs 调接口"}
    return FileResponse(p)
```

冒烟测试：

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8080 &

curl -s http://127.0.0.1:8080/health | python -m json.tool

curl -s -X POST http://127.0.0.1:8080/api/ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_id":"TK-SMOKE-001","channel":"400电话","customer_id":"CUST-77104",
       "raw_text":"A2线XC-200-4P电机启动异响，运行半小时报E33，还能开但声音不对。"}' \
  | python -c "import sys,json;d=json.load(sys.stdin);print(d['status'],d['confidence'],d['cost_yuan']);print(d['solution'][:80])"
```

```text
{
    "status": "ok",
    "app_version": "1.2.0",
    "adapter_version": "ticket-extract-v1.3.0",
    "base_model": "Qwen2.5-7B-Instruct",
    "small_model_ready": true,
    "milvus_ready": true,
    "uptime_s": 4.2
}
done 1.0 0.004142
初步判断：
XC-200-4P 报 E33 属于轴承温升/机械异响保护，结合"启动即有异响
```

`web/index.html`（单文件、零依赖、走 SSE；**不追求好看，追求能演示**）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>华成机电 · 工单智能助手</title>
<style>
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 0; background: #f5f6f8; }
  header { background: #1f2d3d; color: #fff; padding: 14px 24px; font-size: 18px; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  textarea { width: 100%; height: 120px; box-sizing: border-box; font-size: 14px; padding: 8px; }
  select, button { font-size: 14px; padding: 6px 12px; margin-right: 8px; }
  button.primary { background: #2f6feb; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
  button.danger { background: #d93f4c; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
  .step { font-family: ui-monospace, Consolas, monospace; font-size: 12px; padding: 3px 0;
          border-bottom: 1px dashed #eee; }
  .kv { display: grid; grid-template-columns: 120px 1fr; font-size: 13px; row-gap: 4px; }
  .kv b { color: #666; font-weight: 500; }
  pre { white-space: pre-wrap; font-size: 13px; line-height: 1.6; background: #fafafa;
        padding: 10px; border-radius: 4px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
  .u3 { background: #ffe0e0; color: #b3261e; } .u2 { background: #fff4e0; color: #9a6700; }
  .u1 { background: #e6f4ea; color: #1e7f3c; }
  #confirm { display: none; border: 2px solid #d93f4c; }
</style>
</head>
<body>
<header>华成机电 · 工单智能助手 <span style="font-size:12px;opacity:.7">LangChain + LoRA + Agent</span></header>
<main>
  <section>
    <div class="card">
      <h3 style="margin-top:0">工单输入</h3>
      <select id="channel">
        <option>400电话</option><option>小程序</option>
        <option>经销商系统</option><option>邮件</option>
      </select>
      <button class="primary" onclick="send()">提交处理</button>
      <button onclick="fillDemo()">填入示例</button>
      <textarea id="text" placeholder="粘贴工单原文…"></textarea>
    </div>
    <div class="card" id="confirm">
      <h3 style="margin-top:0;color:#d93f4c">⚠ 需要人工确认</h3>
      <pre id="pending"></pre>
      <button class="primary" onclick="confirm_('approve')">批准执行</button>
      <button class="danger" onclick="confirm_('reject')">拒绝</button>
    </div>
    <div class="card">
      <h3 style="margin-top:0">处理轨迹</h3>
      <div id="trace"></div>
    </div>
  </section>
  <section>
    <div class="card">
      <h3 style="margin-top:0">结构化抽取</h3>
      <div class="kv" id="extract"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">处理方案</h3>
      <pre id="solution">等待处理…</pre>
      <div id="cites" style="font-size:12px;color:#666"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">执行情况与成本</h3>
      <div class="kv" id="meta"></div>
    </div>
  </section>
</main>
<script>
let threadId = null;
const FIELD_CN = { device_model: "设备型号", fault_symptom: "故障现象", fault_code: "故障码",
  urgency: "紧急度", production_line: "产线", suggested_craft: "建议工种",
  need_parts: "需要备件", intent: "工单意图" };

function fillDemo() {
  document.getElementById("text").value =
    "急！B1线整条停了！BP-22KW变频器上电没显示，面板报E21，一小时损失好几万。" +
    "是不是要换CJ20-40接触器？仓库有现货吗？还在保修期内吧？今天必须到人。";
}
function addStep(t) {
  const d = document.createElement("div");
  d.className = "step"; d.textContent = t;
  document.getElementById("trace").appendChild(d);
}
function renderExtract(ex) {
  const box = document.getElementById("extract"); box.innerHTML = "";
  Object.keys(FIELD_CN).forEach(k => {
    const b = document.createElement("b"); b.textContent = FIELD_CN[k];
    const v = document.createElement("span");
    let val = ex[k];
    if (val === null || val === undefined || val === "") val = "—";
    if (k === "urgency") {
      v.innerHTML = `<span class="badge u${ex[k]}">${["", "一般", "较急", "紧急"][ex[k]] || ex[k]}</span>`;
    } else { v.textContent = String(val); }
    box.appendChild(b); box.appendChild(v);
  });
}
function renderMeta(d) {
  const m = document.getElementById("meta"); m.innerHTML = "";
  const rows = [
    ["置信度", d.confidence], ["需要人工", d.need_human ? "是" : "否"],
    ["抽取降级", d.extract_degraded ? "是（走了大模型）" : "否"],
    ["本单成本", "¥" + d.cost_yuan], ["端到端延迟", (d.latency_ms || 0) + " ms"],
    ["小模型成本占比", ((d.usage && d.usage.small_share * 100) || 0).toFixed(2) + "%"],
    ["adapter 版本", d.adapter_version || "-"],
  ];
  if (d.agent_result) {
    rows.push(["已执行动作", (d.agent_result.actions || []).map(a => a.tool).join("、") || "无"]);
    rows.push(["Agent 摘要", d.agent_result.narrative || ""]);
  }
  if (d.clarify_question) rows.push(["追问用户", d.clarify_question]);
  rows.forEach(([k, v]) => {
    const b = document.createElement("b"); b.textContent = k;
    const s = document.createElement("span"); s.textContent = v;
    m.appendChild(b); m.appendChild(s);
  });
}
async function send() {
  document.getElementById("trace").innerHTML = "";
  document.getElementById("solution").textContent = "处理中…";
  document.getElementById("confirm").style.display = "none";
  const body = {
    ticket_id: "TK-WEB-" + Date.now(),
    channel: document.getElementById("channel").value,
    customer_id: "CUST-88213",
    raw_text: document.getElementById("text").value,
  };
  // sse-starlette 的端点是 POST，所以用 fetch 手动读流，而不是 EventSource
  const resp = await fetch("/api/ticket/stream", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const reader = resp.body.getReader(); const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n"); buf = parts.pop();
    for (const p of parts) {
      const ev = (p.match(/^event: (.*)$/m) || [])[1];
      const dataLine = (p.match(/^data: (.*)$/m) || [])[1];
      if (!dataLine) continue;
      const d = JSON.parse(dataLine);
      if (ev === "node") {
        (d.steps || [{ node: d.node }]).forEach(s =>
          addStep(`${s.node || d.node}  ${(s.latency_ms || 0).toFixed ? s.latency_ms.toFixed(1) : 0} ms`));
        if (d.extract) renderExtract(d.extract);
        if (d.solution) {
          document.getElementById("solution").textContent = d.solution;
          document.getElementById("cites").textContent = (d.citations || []).join("　");
        }
      } else if (ev === "pending_confirm") {
        threadId = d.thread_id;
        document.getElementById("pending").textContent =
          JSON.stringify(d.pending_actions, null, 2);
        document.getElementById("confirm").style.display = "block";
      } else if (ev === "done") {
        if (d.extract) renderExtract(d.extract);
        if (d.solution) document.getElementById("solution").textContent = d.solution;
        document.getElementById("cites").textContent = (d.citations || []).join("　");
        renderMeta(d);
      } else if (ev === "error") {
        document.getElementById("solution").textContent = "出错：" + d.detail;
      }
    }
  }
}
async function confirm_(decision) {
  const r = await fetch("/api/confirm", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, decision, operator: "web-demo" }),
  });
  const d = await r.json();
  document.getElementById("confirm").style.display = "none";
  addStep(`human_confirm  decision=${decision}`);
  renderMeta(d);
  if (d.solution) document.getElementById("solution").textContent = d.solution;
}
</script>
</body>
</html>
```

> **一个前端上的坑**：浏览器原生的 `EventSource` **只支持 GET**，而我们的流式接口是 POST（要传工单正文）。所以前端必须用 `fetch` + `ReadableStream` 手动解析 SSE 帧，像上面那样。很多人在这里卡住，然后把接口改成 GET + query string，结果长工单被 URL 长度限制截断。

### 7.3 `deploy/Dockerfile.app`

```dockerfile
# deploy/Dockerfile.app —— 应用镜像（不含 GPU 相关依赖，不装 torch/vllm）
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

# ---------- 依赖层：单独一层，改代码不会触发重装 ----------
FROM base AS deps
RUN pip install uv==0.4.27
WORKDIR /w
COPY deploy/requirements-app.txt ./
RUN uv pip install --system -r requirements-app.txt

# ---------- 运行层 ----------
FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

RUN useradd -m -u 10002 trinity
WORKDIR /app
COPY --chown=trinity:trinity app ./app
COPY --chown=trinity:trinity web ./web
COPY --chown=trinity:trinity eval ./eval
COPY --chown=trinity:trinity deploy/mock_services.py ./deploy/mock_services.py

# /data 放 checkpoints.sqlite 与 audit.log；/models 只读挂 adapter manifest
RUN mkdir -p /data && chown trinity:trinity /data
VOLUME ["/data"]
USER trinity

ENV APP_VERSION=1.2.0 \
    CHECKPOINT_PATH=/data/checkpoints.sqlite \
    ADAPTER_MANIFEST=/models/current/manifest.json

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health | grep -q '"status"' || exit 1

# 单 worker：LangGraph 的 checkpointer 用 SQLite，多 worker 会争锁。
# 要扩并发就把 checkpointer 换 Postgres 再上多 worker（见 7.8 checklist 第 12 条）
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "1", "--timeout-keep-alive", "75"]
```

`deploy/requirements-app.txt`（**应用镜像里不要装 torch / vllm / peft**，那是训练与推理镜像的事，装进来镜像会从 400MB 涨到 6GB）：

```text
langchain==0.3.7
langchain-core==0.3.15
langchain-openai==0.2.6
langchain-community==0.3.5
langgraph==0.2.45
langgraph-checkpoint-sqlite==2.0.1
pydantic==2.9.2
pydantic-settings==2.6.0
fastapi==0.115.4
uvicorn[standard]==0.32.0
sse-starlette==2.1.3
httpx==0.27.2
pymilvus==2.4.9
rank-bm25==0.2.2
langfuse==2.53.1
pandas==2.2.3
numpy==1.26.4
rich==13.9.4
tabulate==0.9.0
```

`.dockerignore`：

```text
.git
.venv
__pycache__/
*.pyc
outputs/
data/raw/
data/interim/
reports/
checkpoints.sqlite
.pytest_cache/
*.md
!README.md
```

```bash
docker build -f deploy/Dockerfile.app -t huacheng/trinity-app:1.2.0 .
docker images huacheng/trinity-app
```

```text
REPOSITORY              TAG     IMAGE ID       CREATED         SIZE
huacheng/trinity-app    1.2.0   7b2e41c9f03a   8 seconds ago   482MB
```

### 7.4 `deploy/docker-compose.yml`

```yaml
# deploy/docker-compose.yml
# 一台 24G 显存机器上的完整部署。GPU 只给 vllm-extract。
name: huacheng-trinity

x-logging: &default-logging
  driver: json-file
  options:
    max-size: "50m"
    max-file: "3"

services:
  # ---------------- 微调模型推理（占宿主机 8001）----------------
  vllm-extract:
    image: vllm/vllm-openai:v0.6.3
    container_name: vllm-extract
    restart: unless-stopped
    logging: *default-logging
    ports:
      - "8001:8000"            # 宿主机 8001（全书约定） → 容器内 vLLM 默认 8000
    volumes:
      # 只挂当前生效的模型目录（7.5 节的 current 软链指向的真实目录）
      - ${MODEL_ROOT:-/srv/models}/current/awq:/model:ro
      - ${MODEL_ROOT:-/srv/models}/current/manifest.json:/model-manifest.json:ro
    environment:
      - VLLM_LOGGING_LEVEL=WARNING
    command: >
      --model /model
      --served-model-name ticket-extract
      --quantization awq
      --dtype half
      --max-model-len 2048
      --gpu-memory-utilization 0.55
      --max-num-seqs 64
      --enable-prefix-caching
      --disable-log-requests
      --host 0.0.0.0
      --port 8000
    shm_size: "8gb"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/v1/models')\" || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 20             # 加载 5.6GB 权重要时间，retries 给足
      start_period: 120s

  # ---------------- 应用服务（8080）----------------
  app:
    image: huacheng/trinity-app:${APP_VERSION:-1.2.0}
    container_name: trinity-app
    restart: unless-stopped
    logging: *default-logging
    ports:
      - "8080:8080"
    env_file:
      - ../.env
    environment:
      # 容器间调用走服务名 + 容器端口，不要写宿主机映射端口
      - SMALL_BASE_URL=http://vllm-extract:8000/v1
      - MILVUS_URI=http://milvus:19530
      - ERP_BASE_URL=http://mock-services:9101
      - CRM_BASE_URL=http://mock-services:9102
      - WO_BASE_URL=http://mock-services:9103
      - LANGFUSE_HOST=http://langfuse:3000
      - CHECKPOINT_PATH=/data/checkpoints.sqlite
      - ADAPTER_MANIFEST=/models/current/manifest.json
      - APP_VERSION=${APP_VERSION:-1.2.0}
    volumes:
      - trinity-data:/data                                   # checkpoints + audit.log
      - ${MODEL_ROOT:-/srv/models}:/models:ro                 # 只为读 manifest
    depends_on:
      vllm-extract:
        condition: service_healthy
      milvus:
        condition: service_healthy
      mock-services:
        condition: service_started
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8080/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s

  # ---------------- mock 业务系统（9101-9103）----------------
  mock-services:
    image: huacheng/trinity-app:${APP_VERSION:-1.2.0}
    container_name: trinity-mock
    restart: unless-stopped
    logging: *default-logging
    command: ["python", "deploy/mock_services.py"]
    ports:
      - "9101:9101"
      - "9102:9102"
      - "9103:9103"

  # ---------------- Milvus 及其依赖 ----------------
  etcd:
    image: quay.io/coreos/etcd:v3.5.16
    container_name: milvus-etcd
    restart: unless-stopped
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: >
      etcd -advertise-client-urls=http://127.0.0.1:2379
      -listen-client-urls=http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2024-09-22T00-33-43Z
    container_name: milvus-minio
    restart: unless-stopped
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.4.13
    container_name: milvus-standalone
    restart: unless-stopped
    logging: *default-logging
    command: ["milvus", "run", "standalone"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - milvus-data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 5
      start_period: 90s

  attu:
    image: zilliz/attu:v2.4
    container_name: milvus-attu
    restart: unless-stopped
    environment:
      - MILVUS_URL=milvus:19530
    ports:
      - "8000:3000"           # 全书约定：Attu 8000
    depends_on:
      milvus:
        condition: service_healthy

  # ---------------- 可观测 ----------------
  postgres:
    image: postgres:16-alpine
    container_name: langfuse-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=langfuse
      - POSTGRES_PASSWORD=langfuse
      - POSTGRES_DB=langfuse
    ports:
      - "5433:5432"           # 全书约定：Postgres 5433
    volumes:
      - pg-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      timeout: 5s
      retries: 5

  langfuse:
    image: langfuse/langfuse:2
    container_name: langfuse
    restart: unless-stopped
    ports:
      - "3001:3000"           # 全书约定：Langfuse 3001
    environment:
      - DATABASE_URL=postgresql://langfuse:langfuse@postgres:5432/langfuse
      - NEXTAUTH_URL=http://localhost:3001
      - NEXTAUTH_SECRET=${LANGFUSE_NEXTAUTH_SECRET:-change-me-in-prod}
      - SALT=${LANGFUSE_SALT:-change-me-in-prod}
      - TELEMETRY_ENABLED=false
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  trinity-data:
  milvus-data:
  etcd-data:
  minio-data:
  pg-data:
```

```bash
cd deploy && docker compose up -d
docker compose ps
```

```text
NAME                 IMAGE                             STATUS                    PORTS
langfuse             langfuse/langfuse:2               Up 2 minutes              0.0.0.0:3001->3000/tcp
langfuse-postgres    postgres:16-alpine                Up 2 minutes (healthy)    0.0.0.0:5433->5432/tcp
milvus-attu          zilliz/attu:v2.4                  Up 1 minute               0.0.0.0:8000->3000/tcp
milvus-etcd          quay.io/coreos/etcd:v3.5.16       Up 2 minutes (healthy)    2379/tcp
milvus-minio         minio/minio:RELEASE.2024-09-22…   Up 2 minutes (healthy)    0.0.0.0:9000-9001->9000-9001/tcp
milvus-standalone    milvusdb/milvus:v2.4.13           Up 1 minute (healthy)     0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
trinity-app          huacheng/trinity-app:1.2.0        Up 45 seconds (healthy)   0.0.0.0:8080->8080/tcp
trinity-mock         huacheng/trinity-app:1.2.0        Up 2 minutes              0.0.0.0:9101-9103->9101-9103/tcp
vllm-extract         vllm/vllm-openai:v0.6.3           Up 2 minutes (healthy)    0.0.0.0:8001->8000/tcp
```

> **`depends_on: condition: service_healthy` 是这份 compose 里最重要的两行**。没有它，`app` 会在 vLLM 还在加载权重（约 40~90 秒）时启动，第一批请求全部走降级到大模型——你会以为微调模型没生效，实际上只是起得太早。

### 7.5 模型 adapter 的分发与版本管理

这是微调项目**最容易失控**的一块。三个必须回答的问题：

| 问题 | 错误做法 | 正确做法 |
|---|---|---|
| adapter 放哪？ | 提交到 git（80MB 二进制，仓库很快就膨胀到几个 G） | 对象存储 + 版本目录 + manifest；git 里只留 manifest |
| 线上跑的是哪个版本？ | "应该是最新的吧" | `/health` 返回 `adapter_version`，每份评测报告都带上它 |
| 怎么回滚？ | 重新训练一遍 | 切软链 + 重启 vLLM，60 秒内完成 |

#### 7.5.1 目录约定与 manifest

```text
/srv/models/                           # 宿主机模型根目录（MODEL_ROOT）
├── ticket-extract-v1.2.0/
│   ├── adapter/                       # LoRA adapter（80MB，训练产物）
│   ├── awq/                           # 合并 + AWQ 量化后的权重（5.6GB，vLLM 加载这个）
│   └── manifest.json
├── ticket-extract-v1.3.0/
│   ├── adapter/
│   ├── awq/
│   └── manifest.json
└── current -> ticket-extract-v1.3.0/  # 软链：compose 只挂这个
```

`manifest.json`（**每次训练完必须生成**，它是"这个模型是怎么来的"的唯一凭证）：

```json
{
  "adapter_version": "ticket-extract-v1.3.0",
  "base_model": "Qwen2.5-7B-Instruct",
  "base_model_revision": "本地快照 2026-02-18，sha256(config.json)=3f9a…c41b",
  "task": "ticket_structured_extract",
  "schema_version": "v1.0.0",
  "trained_at": "2026-03-10T21:14:33+08:00",
  "train_data": {
    "file": "data/sft/train.jsonl",
    "n": 6400,
    "sha256": "9c2f81ab4e7d5a0361bb7e4c8f2d1a55e0b7c93d4f6a8125be3c7d09a41f6e2b",
    "channel_dist": {"400电话": 2568, "小程序": 2214, "经销商系统": 1618}
  },
  "hyperparams": {
    "method": "QLoRA", "r": 16, "alpha": 32, "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lr": 1e-4, "epochs": 3, "best_step": 1000, "eval_loss": 0.1189
  },
  "quantization": {"method": "AWQ", "bits": 4, "group_size": 128, "calib_n": 256,
                   "calib_source": "data/sft/train.jsonl"},
  "eval": {
    "test_file": "data/sft/test.jsonl", "n": 800,
    "schema_ok": 0.9950, "record_exact_rate": 0.7875, "field_strict_avg": 0.9434,
    "p50_ms": 412, "p95_ms": 736,
    "report": "reports/eval_fields_finetuned.md"
  },
  "artifacts": {
    "adapter_sha256": "b71c0e93a42f8d5617ca9e02b4d3f18a7c65e09d2fb841537ae6c0192d4bf8e3",
    "awq_sha256": "4d8fa1c37be05926ecf1a840b73d52c6109fe8b4a27d93016cb5e7f2804a9d61",
    "adapter_bytes": 83886080,
    "awq_bytes": 6012954624
  },
  "trained_by": "guojingyi@datefunture.com",
  "notes": "上采样小程序渠道到 45%；相比 v1.2.0 的 suggested_craft 提升 2.1pp"
}
```

生成 manifest 的脚本（训练流水线最后一步）：

```python
# scripts/make_manifest.py
"""训练/量化完成后生成 manifest.json。没有 manifest 的模型禁止上线。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def dir_sha256(path: Path) -> str:
    """对目录内容算稳定哈希：按相对路径排序后逐文件喂入。"""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            with p.open("rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    """目录总字节数。"""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def file_sha256(path: Path) -> str:
    """单文件哈希。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    """入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="如 ticket-extract-v1.3.0")
    ap.add_argument("--root", required=True, help="版本目录，如 /srv/models/ticket-extract-v1.3.0")
    ap.add_argument("--base", default="/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-file", default="data/sft/train.jsonl")
    ap.add_argument("--eval-json", default="reports/eval_fields_finetuned.json")
    ap.add_argument("--trained-by", default="unknown")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    root = Path(args.root)
    adapter, awq = root / "adapter", root / "awq"
    for d in (adapter, awq):
        if not d.exists():
            raise SystemExit(f"缺少目录：{d}")

    train = Path(args.train_file)
    ev = json.loads(Path(args.eval_json).read_text(encoding="utf-8")) \
        if Path(args.eval_json).exists() else {}

    manifest = {
        "adapter_version": args.version,
        "base_model": Path(args.base).name,
        "base_model_revision": (f"本地快照，sha256(config.json)="
                                f"{file_sha256(Path(args.base) / 'config.json')[:8]}…"
                                if (Path(args.base) / "config.json").exists() else "unknown"),
        "task": "ticket_structured_extract",
        "schema_version": json.loads(
            Path("data/sft/schema.json").read_text(encoding="utf-8"))["version"],
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_data": {"file": str(train), "n": sum(1 for _ in train.open(encoding="utf-8")),
                       "sha256": file_sha256(train)},
        "eval": {k: ev.get(k) for k in
                 ("n", "record_exact_rate", "field_strict_avg", "p50_ms", "p95_ms")}
                | {"schema_ok": (ev.get("format") or {}).get("schema_ok")},
        "artifacts": {
            "adapter_sha256": dir_sha256(adapter), "awq_sha256": dir_sha256(awq),
            "adapter_bytes": dir_bytes(adapter), "awq_bytes": dir_bytes(awq),
        },
        "trained_by": args.trained_by,
        "notes": args.notes,
    }
    out = root / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {out}")
    print(json.dumps({k: manifest[k] for k in ("adapter_version", "eval", "artifacts")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

#### 7.5.2 上传与拉取

```bash
# ===== 训练机：打包上传到对象存储（MinIO / OSS / S3 都行）=====
VER=ticket-extract-v1.3.0
ROOT=/srv/models/$VER

python scripts/make_manifest.py --version $VER --root $ROOT \
  --trained-by "$(git config user.email)" \
  --notes "上采样小程序渠道到45%"

# adapter 小（80MB），直接传；awq 大（5.6GB），打成 tar 再传，避免几千个小文件
tar -C $ROOT -czf /tmp/$VER-awq.tar.gz awq
mc alias set hc http://127.0.0.1:9000 minioadmin minioadmin
mc mb -p hc/models
mc cp $ROOT/manifest.json          hc/models/$VER/manifest.json
mc cp -r $ROOT/adapter             hc/models/$VER/adapter/
mc cp /tmp/$VER-awq.tar.gz         hc/models/$VER/awq.tar.gz

# ===== 部署机：拉取并校验 =====
cat > scripts/pull_adapter.sh <<'EOF'
#!/usr/bin/env bash
# 拉取指定版本的模型到本地并校验哈希。校验不过一律不切换。
set -Eeuo pipefail
VER="${1:?用法: pull_adapter.sh <版本号，如 ticket-extract-v1.3.0>}"
ROOT="${MODEL_ROOT:-/srv/models}"
DST="$ROOT/$VER"

mkdir -p "$DST"
mc cp "hc/models/$VER/manifest.json" "$DST/manifest.json"
mc cp -r "hc/models/$VER/adapter" "$DST/"
mc cp "hc/models/$VER/awq.tar.gz" "/tmp/$VER-awq.tar.gz"
rm -rf "$DST/awq" && mkdir -p "$DST/awq"
tar -C "$DST" -xzf "/tmp/$VER-awq.tar.gz"

# 校验：用 manifest 里的哈希对比本地实际内容
python - "$DST" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.make_manifest import dir_sha256
root = Path(sys.argv[1])
m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name, key in (("adapter", "adapter_sha256"), ("awq", "awq_sha256")):
    actual = dir_sha256(root / name)
    expect = m["artifacts"][key]
    status = "OK" if actual == expect else "MISMATCH"
    print(f"{name:<8} {status}  expect={expect[:12]}… actual={actual[:12]}…")
    if actual != expect:
        raise SystemExit(f"{name} 哈希不匹配，拒绝上线")
print("校验通过：", m["adapter_version"], m.get("eval"))
PY
echo "已就绪：$DST（未切换，执行 switch_adapter.sh 才生效）"
EOF
chmod +x scripts/pull_adapter.sh
```

#### 7.5.3 切换与回滚（灰度、60 秒回滚）

```bash
cat > scripts/switch_adapter.sh <<'EOF'
#!/usr/bin/env bash
# 切换线上生效的模型版本：改软链 + 重启 vLLM + 冒烟验证；失败自动回滚。
set -Eeuo pipefail
VER="${1:?用法: switch_adapter.sh <版本号>}"
ROOT="${MODEL_ROOT:-/srv/models}"
COMPOSE_DIR="${COMPOSE_DIR:-/srv/huacheng-trinity/deploy}"

[ -f "$ROOT/$VER/manifest.json" ] || { echo "版本不存在或未校验：$ROOT/$VER"; exit 1; }

PREV=$(readlink -f "$ROOT/current" || echo "")
echo "当前版本：${PREV:-无}　将切换到：$ROOT/$VER"

# 原子切换软链（ln -sfn + mv 保证中间不会出现"current 不存在"的瞬间）
ln -sfn "$ROOT/$VER" "$ROOT/.current.new"
mv -Tf "$ROOT/.current.new" "$ROOT/current"

cd "$COMPOSE_DIR"
docker compose up -d --force-recreate vllm-extract

echo -n "等待 vLLM 就绪"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null; then echo " ready"; break; fi
  sleep 3; echo -n .
done

# 冒烟：抽一条已知样本，要求 schema 通过且关键字段正确
SMOKE=$(curl -sf -X POST http://127.0.0.1:8080/api/extract \
  -H 'Content-Type: application/json' \
  -d '{"ticket_id":"SWITCH-SMOKE","channel":"400电话",
       "raw_text":"B1线BP-22KW变频器上电无显示报E21，整线停机"}' || echo '{}')
echo "冒烟结果：$SMOKE"

OK=$(python - <<PY
import json
d = json.loads('''$SMOKE''' or "{}")
e = (d.get("extract") or {})
print(int(e.get("device_model") == "BP-22KW" and e.get("fault_code") == "E21"
          and int(e.get("urgency", 0)) == 3 and not d.get("degraded", True)))
PY
)
if [ "$OK" != "1" ]; then
  echo "冒烟失败，回滚到 ${PREV:-上一版本}"
  [ -n "$PREV" ] && ln -sfn "$PREV" "$ROOT/.current.new" && mv -Tf "$ROOT/.current.new" "$ROOT/current"
  docker compose up -d --force-recreate vllm-extract
  exit 1
fi
echo "切换成功：$VER"
curl -s http://127.0.0.1:8080/health | python -m json.tool | grep adapter_version
EOF
chmod +x scripts/switch_adapter.sh
```

```bash
./scripts/pull_adapter.sh ticket-extract-v1.3.0
./scripts/switch_adapter.sh ticket-extract-v1.3.0
```

```text
adapter  OK  expect=b71c0e93a42f… actual=b71c0e93a42f…
awq      OK  expect=4d8fa1c37be0… actual=4d8fa1c37be0…
校验通过： ticket-extract-v1.3.0 {'n': 800, 'record_exact_rate': 0.7875, 'field_strict_avg': 0.9434, 'p50_ms': 412, 'p95_ms': 736, 'schema_ok': 0.995}
已就绪：/srv/models/ticket-extract-v1.3.0（未切换，执行 switch_adapter.sh 才生效）

当前版本：/srv/models/ticket-extract-v1.2.0　将切换到：/srv/models/ticket-extract-v1.3.0
[+] Running 1/1
 ✔ Container vllm-extract  Started
等待 vLLM 就绪.............. ready
冒烟结果：{"ticket_id":"SWITCH-SMOKE","extract":{"device_model":"BP-22KW","fault_symptom":"变频器上电无显示报E21","fault_code":"E21","urgency":3,"production_line":"B1","suggested_craft":"电气","need_parts":true,"intent":"报修"},"latency_ms":418.6,"degraded":false,"error":null,"adapter_version":"ticket-extract-v1.3.0"}
切换成功：ticket-extract-v1.3.0
        "adapter_version": "ticket-extract-v1.3.0",
```

**想做灰度**（新老版本各承接一部分流量）有两条路：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A. 双实例 + 权重路由 | 起两个 vLLM（8001 老 / 8003 新），在 `app/llms.py` 里按 `hash(ticket_id) % 100 < 灰度比例` 选 base_url | 隔离彻底，老版本随时能全量接回 | 显存要两份（AWQ 版 5.6GB × 2，24G 卡勉强够） |
| B. 单实例 + 多 adapter | vLLM 用 `--enable-lora --lora-modules v120=… v130=…`（Step 5.7 的路线 1），调用时 `model` 字段传不同 adapter 名 | 只占一份基座显存 | 吞吐降 10~20%；**必须用未合并的 adapter，与 AWQ 合并版不兼容** |

本项目选 **A**，理由是：AWQ 合并版的吞吐优势（+72%）比省一份显存更重要，而且灰度期通常只有一两天。灰度实现只需要在模型工厂里加几行：

```python
# app/llms.py 里 small_llm 的灰度版本（替换原实现）
import hashlib

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache
def _small_client(base_url: str, model: str, temperature: float, max_tokens: int,
                  timeout: float) -> ChatOpenAI:
    """按 (base_url, model) 缓存客户端实例。"""
    return ChatOpenAI(base_url=base_url, api_key="EMPTY", model=model,
                      temperature=temperature, max_tokens=max_tokens,
                      timeout=timeout, max_retries=0)


def small_llm(temperature: float = 0.0, max_tokens: int = 256,
              route_key: str | None = None) -> ChatOpenAI:
    """微调抽取模型。传 route_key（如 ticket_id）时按灰度比例分流到 canary 实例。"""
    s = get_settings()
    base, model = s.small_base_url, s.small_model
    canary_url = getattr(s, "small_canary_base_url", "") or ""
    pct = int(getattr(s, "small_canary_percent", 0) or 0)
    if canary_url and pct > 0 and route_key:
        bucket = int(hashlib.md5(route_key.encode()).hexdigest(), 16) % 100
        if bucket < pct:
            base = canary_url
    return _small_client(base, model, temperature, max_tokens, s.extract_timeout_ms / 1000)
```

对应地在 `.env` 与 `Settings` 里加两项：`SMALL_CANARY_BASE_URL`、`SMALL_CANARY_PERCENT`（默认 0 = 不灰度）。**灰度期间必须按实例分别看指标**，否则两个版本的错误会被平均掉，你看到的是一个"中间值"，既不知道新版好不好，也不知道老版坏没坏。

### 7.6 启停与健康检查脚本

`deploy/start.sh`：

```bash
#!/usr/bin/env bash
# 一键启动：按依赖顺序起服务，等健康后再起应用，最后跑一遍冒烟。
set -Eeuo pipefail
cd "$(dirname "$0")"

MODEL_ROOT="${MODEL_ROOT:-/srv/models}"
export MODEL_ROOT

# 0) 前置检查：没有 current 软链就别启动，省得起来全是降级
if [ ! -e "$MODEL_ROOT/current/manifest.json" ]; then
  echo "❌ 缺少 $MODEL_ROOT/current/manifest.json"
  echo "   先跑 scripts/pull_adapter.sh <版本> && scripts/switch_adapter.sh <版本>"
  exit 1
fi
if [ ! -f ../.env ]; then
  echo "❌ 缺少 .env，从 .env.example 复制并填写"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "⚠ 没有检测到 nvidia-smi，vLLM 将无法启动。"
  echo "  纯 CPU 环境请改用「无 GPU 路径」：注释掉 vllm-extract，"
  echo "  并在 .env 里设 SMALL_BASE_URL=\$LLM_BASE_URL、SMALL_MODEL=\$LLM_MODEL（全走大模型）"
fi

echo "═══ 1/4 起基础设施（etcd / minio / milvus / postgres）═══"
docker compose up -d etcd minio postgres
docker compose up -d milvus attu langfuse mock-services

echo "═══ 2/4 起 vLLM（加载 5.6GB 权重，约 40~90 秒）═══"
docker compose up -d vllm-extract
echo -n "等待 vLLM"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null; then echo " ✅"; break; fi
  sleep 3; echo -n .
done

echo "═══ 3/4 起应用 ═══"
docker compose up -d app
echo -n "等待应用"
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null; then echo " ✅"; break; fi
  sleep 2; echo -n .
done

echo "═══ 4/4 冒烟 ═══"
./healthcheck.sh

cat <<EOF

启动完成。可访问：
  应用与演示前端   http://localhost:8080
  接口文档         http://localhost:8080/docs
  vLLM 模型列表    http://localhost:8001/v1/models
  Attu（Milvus）   http://localhost:8000
  Langfuse         http://localhost:3001
  MinIO 控制台     http://localhost:9001
EOF
```

`deploy/stop.sh`：

```bash
#!/usr/bin/env bash
# 停服务。默认保留数据卷；传 --purge 才删（会丢 checkpoints 与 Milvus 数据）。
set -Eeuo pipefail
cd "$(dirname "$0")"

# 优雅停：先停入口，让在途请求跑完，再停后端
echo "停应用（等在途请求结束）…"
docker compose stop -t 30 app

# 有未完成的人工确认会话时提醒一声，别让人工确认凭空消失
PENDING=$(docker compose exec -T app python - <<'PY' 2>/dev/null || echo 0
import asyncio, os
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
async def main():
    n = 0
    async with AsyncSqliteSaver.from_conn_string(os.environ["CHECKPOINT_PATH"]) as s:
        async for ck in s.alist(None):
            if (ck.checkpoint.get("channel_values") or {}).get("pending_confirm"):
                n += 1
    print(n)
asyncio.run(main())
PY
)
[ "${PENDING:-0}" != "0" ] && echo "⚠ 有 ${PENDING} 个会话卡在人工确认，重启后可用 /api/threads/<id> 继续"

docker compose stop -t 20 vllm-extract mock-services
docker compose stop attu langfuse milvus minio etcd postgres

if [ "${1:-}" = "--purge" ]; then
  read -r -p "确认删除所有数据卷（checkpoints / Milvus / Langfuse）？输入 yes 继续: " a
  [ "$a" = "yes" ] && docker compose down -v || echo "已取消"
else
  docker compose down
fi
echo "已停止（数据卷保留，用 --purge 才删）"
```

`deploy/healthcheck.sh`：

```bash
#!/usr/bin/env bash
# 健康检查：8 项，任何一项失败都打印可执行的修复建议。可放进 cron 每 5 分钟跑。
set -uo pipefail
FAIL=0
ok()   { echo "✅ $1"; }
bad()  { echo "❌ $1"; echo "   → $2"; FAIL=$((FAIL+1)); }

# 1) 应用进程是否在
if curl -sf http://127.0.0.1:8080/health >/dev/null; then ok "应用 8080 可达"
else bad "应用 8080 不可达" "docker compose logs --tail 50 app｜常见原因：import 写错、.env 缺字段"; fi

H=$(curl -s http://127.0.0.1:8080/health 2>/dev/null || echo '{}')

# 2) 小模型是否就绪（不就绪会全量降级到大模型，成本翻 260 倍）
echo "$H" | grep -q '"small_model_ready": *true' \
  && ok "vLLM 抽取模型就绪" \
  || bad "vLLM 未就绪，抽取会走降级" "curl http://127.0.0.1:8001/v1/models；docker compose logs vllm-extract"

# 3) Milvus 是否就绪
echo "$H" | grep -q '"milvus_ready": *true' \
  && ok "Milvus 就绪" \
  || bad "Milvus 不可用，检索会返回空" "docker compose ps milvus；curl http://127.0.0.1:9091/healthz"

# 4) adapter 版本是否是期望的那个
WANT="${EXPECT_ADAPTER:-}"
GOT=$(echo "$H" | python -c "import sys,json;print(json.load(sys.stdin).get('adapter_version','?'))" 2>/dev/null)
if [ -n "$WANT" ]; then
  [ "$GOT" = "$WANT" ] && ok "adapter 版本 $GOT" \
    || bad "adapter 版本不符（期望 $WANT 实际 $GOT）" "scripts/switch_adapter.sh $WANT"
else ok "adapter 版本 $GOT（未设置 EXPECT_ADAPTER，跳过比对）"; fi

# 5) 端到端抽取是否正确且未降级
R=$(curl -sf -X POST http://127.0.0.1:8080/api/extract -H 'Content-Type: application/json' \
  -d '{"ticket_id":"HC","channel":"400电话","raw_text":"B1线BP-22KW变频器上电无显示报E21，整线停机"}' || echo '{}')
echo "$R" | grep -q '"fault_code": *"E21"' && echo "$R" | grep -q '"degraded": *false' \
  && ok "抽取冒烟通过（未降级）" \
  || bad "抽取冒烟失败或走了降级" "看 $R；确认 vLLM 正常、SMALL_BASE_URL 指向容器内 8000"

# 6) 降级率是否异常（>5% 说明小模型在抖）
D=$(curl -s http://127.0.0.1:8080/metrics | python -c \
  "import sys,json;print(json.load(sys.stdin).get('degraded_rate',0))" 2>/dev/null || echo 0)
python -c "import sys;sys.exit(0 if float('$D')<=0.05 else 1)" \
  && ok "降级率 $D" \
  || bad "降级率偏高 $D" "查 vLLM 显存与 max-num-seqs；查 EXTRACT_TIMEOUT_MS 是否太小"

# 7) mock 业务系统（Agent 的工具依赖它）
for p in 9101 9102 9103; do
  curl -sf "http://127.0.0.1:$p/openapi.json" >/dev/null \
    && ok "mock 服务 $p 可达" \
    || bad "mock 服务 $p 不可达" "docker compose up -d mock-services"
done

# 8) 磁盘与 checkpoint 大小（SQLite 会一直长）
SZ=$(docker compose exec -T app sh -c 'du -sk "$CHECKPOINT_PATH" 2>/dev/null | cut -f1' || echo 0)
[ "${SZ:-0}" -lt 2097152 ] && ok "checkpoint 大小 ${SZ}KB" \
  || bad "checkpoint 超过 2GB" "定期清理 30 天前的 thread（见 8.5 优化清单）"

echo "───────────────"
[ "$FAIL" -eq 0 ] && echo "全部通过" || echo "$FAIL 项失败"
exit "$FAIL"
```

```bash
chmod +x deploy/{start,stop,healthcheck}.sh
./deploy/start.sh
```

```text
═══ 1/4 起基础设施（etcd / minio / milvus / postgres）═══
[+] Running 7/7 …
═══ 2/4 起 vLLM（加载 5.6GB 权重，约 40~90 秒）═══
等待 vLLM.............. ✅
═══ 3/4 起应用 ═══
等待应用.... ✅
═══ 4/4 冒烟 ═══
✅ 应用 8080 可达
✅ vLLM 抽取模型就绪
✅ Milvus 就绪
✅ adapter 版本 ticket-extract-v1.3.0（未设置 EXPECT_ADAPTER，跳过比对）
✅ 抽取冒烟通过（未降级）
✅ 降级率 0
✅ mock 服务 9101 可达
✅ mock 服务 9102 可达
✅ mock 服务 9103 可达
✅ checkpoint 大小 44KB
───────────────
全部通过

启动完成。可访问：
  应用与演示前端   http://localhost:8080
  接口文档         http://localhost:8080/docs
  vLLM 模型列表    http://localhost:8001/v1/models
  Attu（Milvus）   http://localhost:8000
  Langfuse         http://localhost:3001
  MinIO 控制台     http://localhost:9001
```

### 7.7 环境变量清单

**一张表说清楚每个变量在哪生效、容器内外有什么差别**。这张表要贴进 README，新人照着填就能起来。

| 变量 | 必填 | 宿主机/本地开发值 | 容器内值（compose 覆盖） | 说明与坑 |
|---|---|---|---|---|
| `LLM_BASE_URL` | ✅ | `https://api.deepseek.com/v1` | 同 | 大模型（生成 / Agent / 降级兜底） |
| `LLM_API_KEY` | ✅ | `sk-…` | 同 | **只放 `.env`，不进镜像不进 git** |
| `LLM_MODEL` | ✅ | `deepseek-chat` | 同 | — |
| `SMALL_BASE_URL` | ✅ | `http://127.0.0.1:8001/v1` | `http://vllm-extract:8000/v1` | **最容易错的一项**：容器内要用服务名 + 容器端口 |
| `SMALL_API_KEY` | — | `EMPTY` | 同 | vLLM 不校验，但 SDK 要求非空 |
| `SMALL_MODEL` | ✅ | `ticket-extract` | 同 | 必须与 `--served-model-name` 完全一致 |
| `SMALL_CANARY_BASE_URL` | — | 空 | 空 | 灰度用；空 = 不灰度 |
| `SMALL_CANARY_PERCENT` | — | `0` | `0` | 灰度比例 0~100 |
| `MILVUS_URI` | ✅ | `http://127.0.0.1:19530` | `http://milvus:19530` | — |
| `MILVUS_COLLECTION` | ✅ | `huacheng_kb` | 同 | 与项目 2 共用同一个 collection |
| `EMBED_MODEL_PATH` | ✅ | `/models/bge-m3` | 同（只读挂载） | 离线环境务必用本地路径 |
| `RERANK_MODEL_PATH` | ✅ | `/models/bge-reranker-v2-m3` | 同 | — |
| `ERP_BASE_URL` | ✅ | `http://127.0.0.1:9101` | `http://mock-services:9101` | 接真实 ERP 时换成内网地址 |
| `CRM_BASE_URL` | ✅ | `http://127.0.0.1:9102` | `http://mock-services:9102` | — |
| `WO_BASE_URL` | ✅ | `http://127.0.0.1:9103` | `http://mock-services:9103` | — |
| `LANGFUSE_HOST` | — | `http://127.0.0.1:3001` | `http://langfuse:3000` | **宿主机 3001，容器内 3000** |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | 从 Langfuse UI 创建项目后获得 | 同 | 首次启动 Langfuse 后手动补 |
| `TRACING_ENABLED` | — | `true` | `true` | 关掉会省一点延迟，但排查问题会很痛苦 |
| `EXTRACT_TIMEOUT_MS` | — | `3000` | `3000` | **设太小会误触发降级**（P95 是 736ms，3000 有 4 倍余量） |
| `EXTRACT_MAX_RETRY` | — | `2` | `2` | 重试针对网络，格式错误直接降级 |
| `AGENT_MAX_ITER` | — | `6` | `6` | 改大会显著抬高 P95 与成本 |
| `HUMAN_CONFIRM_REQUIRED` | — | `true` | `true` | **生产环境禁止设 false** |
| `CHECKPOINT_PATH` | — | `./checkpoints.sqlite` | `/data/checkpoints.sqlite` | 必须落在挂载卷上，否则重启丢会话 |
| `ADAPTER_MANIFEST` | — | `/srv/models/current/manifest.json` | `/models/current/manifest.json` | `/health` 的 `adapter_version` 来源 |
| `APP_VERSION` | — | `1.2.0` | 同 | 写进报告，用于追溯 |
| `MODEL_ROOT` | ✅（compose） | `/srv/models` | — | compose 变量，不是应用变量 |
| `EXPECT_ADAPTER` | — | 空 | — | `healthcheck.sh` 用；上线后建议设成当前版本 |

### 7.8 上线 checklist

| # | 检查项 | 怎么验证 | 通过标准 |
|---|---|---|---|
| 1 | schema 已冻结且与线上一致 | `diff <(curl -s localhost:8080/health \| jq -r .adapter_version) <(jq -r .adapter_version /srv/models/current/manifest.json)`；`data/sft/schema.json` 的 `version` 与 manifest 的 `schema_version` 相同 | 完全一致。**schema 变了必须重训 + 重评** |
| 2 | manifest 齐备且哈希校验通过 | `scripts/pull_adapter.sh <ver>` | 两个 `OK`，无 `MISMATCH` |
| 3 | 量化后重新评测过 | `reports/eval_fields_finetuned.md` 的生成时间晚于 AWQ 产物时间 | 字段严格准确率下降 ≤ 1.0pp、关键字段 ≤ 0.5pp（Step 5.6 的阈值） |
| 4 | 降级链演练过 | `docker compose stop vllm-extract` 后调 `/api/ticket` | 能出结果，`extract_degraded=true`，延迟上升但不报错 |
| 5 | 人工确认与拒绝两条路都演练过 | 跑 `demo/demo_urgent.py` 各选一次 y / n | 批准会派单；拒绝不产生派单且 `rejected=true` |
| 6 | 高风险工具确实需要确认 | `grep -n 'need_confirm=True' app/tools/*.py` | `dispatch` 为 `critical + need_confirm=True` |
| 7 | 评测模式不写业务数据 | 用 `eval_mode=true` 跑 60 条端到端，查 mock 工单系统的派单记录数 | 派单记录数不变 |
| 8 | 迭代上限有效 | `python tests/test_agent_limit.py` | `iterations <= AGENT_MAX_ITER + 1` |
| 9 | 脱敏生效 | 提交含手机号与公司名的工单，查 Langfuse trace | trace 里是 `[PHONE]` / `[COMPANY]` |
| 10 | 审计日志在写 | 批准一次派单后 `docker compose exec app tail -2 /data/audit.log` | 有 operator、decision、actions |
| 11 | 容器内外端口都对 | `docker compose exec app env \| grep -E 'SMALL_BASE_URL\|LANGFUSE_HOST'` | 指向服务名 + 容器端口，不是 127.0.0.1 |
| 12 | 单 worker（或已换 Postgres checkpointer） | `docker compose exec app ps aux \| grep -c uvicorn` | 单 worker；若要多 worker 必须先把 checkpointer 换成 Postgres |
| 13 | 密钥不在镜像里 | `docker history --no-trunc huacheng/trinity-app:1.2.0 \| grep -i -E 'sk-\|api_key'` | 无命中 |
| 14 | 资源上限已设 | compose 里给 `app` 加 `deploy.resources.limits`（内存 2G）；vLLM 的 `--gpu-memory-utilization 0.55` | 应用 OOM 不会拖垮 vLLM |
| 15 | 日志有轮转 | `docker inspect trinity-app \| grep -A3 LogConfig` | `max-size=50m, max-file=3` |
| 16 | 回滚演练过 | 切到 v1.2.0 再切回 v1.3.0 | 每次 60 秒内完成，`/health` 的 `adapter_version` 正确变化 |
| 17 | 健康检查进定时任务 | `crontab -l \| grep healthcheck` | `*/5 * * * *` 且失败会告警 |
| 18 | 有人负责 | README 写明：模型 owner、服务 owner、"降级率报警时找谁" | 有名字、有备份人 |

第 4 条和第 5 条是**最容易被跳过、也最容易在上线当晚出事**的两条。**没演练过的降级等于没有降级**（Step 6.6 已经强调过一次，这里再强调一次）；没演练过的"人工拒绝"路径，很可能在拒绝后把图卡在中断态，客服看到的是一个永远转圈的页面。

### 7.9 本节小结

- 三个镜像分开：训练（含 torch/peft）、推理（vLLM 官方镜像）、应用（482MB，不装 torch）——**把 torch 装进应用镜像是最常见的浪费**；
- 端口按全书约定映射，容器内保持组件默认：vLLM `8001:8000`、Attu `8000:3000`、Langfuse `3001:3000`、Postgres `5433:5432`；**容器间调用一律走服务名 + 容器端口**；
- `depends_on: service_healthy` + `start_period: 120s` 是 vLLM 场景的必需品，否则应用起太早会全量降级；
- adapter 管理三件套：**manifest（凭证）+ 哈希校验（防损坏）+ 软链切换（60 秒回滚）**。没有 manifest 的模型禁止上线；
- 灰度选"双实例 + 权重路由"而不是"多 adapter"，因为 AWQ 合并版的吞吐优势更值钱；灰度期间**必须按实例分别看指标**；
- `healthcheck.sh` 里最有价值的一项是**降级率监控**：降级率从 0 涨到 5% 意味着成本涨了几十倍，而系统表面上"一切正常"。

---

## 八、复盘

### 8.1 三位一体的收益量化

这一节要回答老板那个问题：**"搞这一套，到底值多少钱？"** 不给数字的架构讨论都是自嗨。

**实测环境**：单机 RTX 4090 24G（vLLM 0.6.3 + AWQ 4bit）、`deepseek-chat` 走公网 API、Milvus 2.4 单机、抽取并发 8（大模型受限流只能并发 4）、测试集 800 条 + 端到端 60 条、日均工单 115 张（54% 普通 / 9% 紧急 / 37% 信息不全，其中 72% 会补全进入第二轮）。**以下全部为示例性数据，你必须在自己的数据与硬件上重测。**

#### 8.1.1 主表：三位一体 vs 全部用大模型

| 维度 | 指标 | A. 全部用大模型<br/>（抽取也用 `deepseek-chat`） | B. **三位一体**<br/>（抽取用微调小模型） | 变化 |
|---|---|---|---|---|
| **成本** | 单单平均成本（按路径加权） | ¥0.010022 | **¥0.004592** | **-54.2%** |
| | 其中：抽取环节 | ¥0.004304 | **¥0.000016** | -99.6% |
| | 其中：生成环节 | ¥0.004126 | ¥0.004126 | 0（没动） |
| | 其中：Agent 环节（9% 工单触发） | ¥0.009702 | ¥0.009702 | 0（没动） |
| | 月成本（3,450 单） | ¥34.58 | **¥15.84** | -¥18.74/月 |
| | 年成本 | ¥415 | **¥190** | -¥225/年 |
| **延迟** | 抽取 P50 | 1,842 ms | **412 ms** | **-77.6%** |
| | 抽取 P95 | 3,376 ms | **736 ms** | -78.2% |
| | 普通工单端到端 P50 | 5,042 ms | **3,612 ms** | -28.4% |
| | 紧急工单端到端 P50 | 13,273 ms | **11,843 ms** | -10.8% |
| | 信息不全（第 1 轮就追问） | 1,882 ms | **452 ms** | **-76.0%** |
| | 抽取吞吐上限 | 约 6 req/s（受 API 限流） | **31.7 req/s** | +428% |
| **准确率** | 字段严格准确率（均） | 89.71% | **94.34%** | **+4.63pp** |
| | 整单完全正确率 | 71.13% | **78.75%** | **+7.62pp** |
| | schema 通过率 | 97.75% | **99.50%** | +1.75pp |
| | `urgency`（主观判断） | 82.38% | **91.13%** | +8.75pp |
| | `suggested_craft`（主观判断） | 85.25% | **93.50%** | +8.25pp |
| | `device_model` 编造率 | 5.88% | **2.35%** | -3.53pp |
| **人力** | 每月需人工兜底的工单 | 996 单（3450 × 28.87%） | **733 单**（3450 × 21.25%） | **-263 单/月** |
| | 折合人力（每单修正约 3 分钟） | 49.8 小时/月 | **36.6 小时/月** | **-13.2 小时/月** |
| **投入** | 一次性：数据工程 | — | 约 20 人时（构造 + 清洗 + 抽检） | — |
| | 一次性：训练与量化 | — | 云 4090 约 4 小时 ≈ ¥12~32 | — |
| | 一次性：评测体系搭建 | — | 约 8 人时 | — |
| | 持续：模型维护 | — | 约 2 人时/月（回归 + 偶发重训） | — |

#### 8.1.2 结论一：单看 API 账单，这套架构"不值得"

$$
\text{年 API 节省} = (0.010022 - 0.004592) \times 115 \times 365 \approx \text{¥228}
$$

一年省 228 块。**而一次性投入是 28 人时**——按任何一个正常的工程师成本口径，这笔账都是亏的。

**必须把这句话说在前面**，因为很多微调项目的汇报材料只写"成本降低 54%"，不写绝对金额。54% 听起来惊人，¥228/年 听起来可笑，**两者是同一个事实**。

#### 8.1.3 结论二：真正让账算得过来的是另外三件事

| 收益来源 | 量化 | 说明 |
|---|---|---|
| **① 人力节省** | 13.2 小时/月 ≈ 158 小时/年 | 整单完全正确率 +7.62pp → 每月少 263 单要人工修正。**这一项按 28 人时的投入算，约 2.1 个月回本** |
| **② 批量回填** | 成本 ¥1,205 → ¥4.5；耗时 35.8 小时 → 2.5 小时 | 28 万条历史工单要结构化回填（做数据分析、做知识库）。这是**一次性但金额最大**的收益：省 ¥1,200 和 33 小时的等待 |
| **③ 放量后的杠杆** | 日均 2,000 单时：月成本 ¥601 → ¥276 | 如果开放经销商自助报修（规划中），调用量涨 17 倍，节省额同步涨到 ¥3,900/年，而投入不变 |

另外两项**没法直接折成钱，但在企业里常常是决定性的**：

| 收益 | 具体表现 |
|---|---|
| **数据不出内网** | 抽取环节处理的是完整工单原文（含客户名、联系人、产线信息）。微调方案下这些文本**只进本地 GPU**；全大模型方案下它们要发到公网 API。很多制造业客户的合同里有这一条 |
| **行为可回归** | 全大模型方案改一次 prompt，全量行为漂移且没有基线可比；微调方案是**版本化的 adapter + 冻结的 800 条测试集**，任何一次变更都能跑出 7.5 节 manifest 里那组数字。这是"能不能长期维护"的分水岭 |

#### 8.1.4 结论三：成本结构告诉你下一步该优化什么

```mermaid
pie title 单单成本构成（三位一体，按路径加权 ¥0.004592）
    "生成（大模型，deepseek-chat）" : 74.6
    "Agent（大模型，9% 工单触发）" : 21.1
    "抽取（微调小模型）" : 0.35
    "检索 / 编排 / 追问（本地，近似为 0）" : 3.95
```

抽取只占 0.35%。**这意味着继续优化抽取（换更小的模型、再量化一档）对总成本毫无意义**。下一步该做的三件事，按收益排序：

1. **压生成的 prompt**（占 74.6%）：把 5 个检索块减到 3 个 + 只传摘要，预计 prompt token 从 1,420 降到 900，单单成本降约 22%；
2. **让更多工单不进 Agent**（占 21.1%）：`check_action` 的关键词表再收紧一点，或者把"只查库存"这种只读场景改成直接函数调用而不是 Agent；
3. **生成也考虑微调**（长期）：如果方案生成的格式足够固定，微调一个 14B 来写"初判 + 步骤 + 注意事项"，是把 74.6% 那块也拿下的唯一办法。**但这件事的前提是先把评测做扎实**——生成任务的评测比抽取难一个数量级（见项目 1 的 LLM-as-a-Judge 与 Kappa 校准）。

### 8.2 做对了什么

| # | 决策 | 当时的代价 | 事后证明的价值 |
|---|---|---|---|
| 1 | **先冻结 schema，再动手做数据** | 多花半天和业务吵字段定义 | 训练数据、pydantic 校验、评测脚本、前端展示**全部以同一份 `schema.json` 为准**。改 schema = 重训 + 重评，这条规矩挡住了三次"顺手加个字段"的请求 |
| 2 | **业务硬规则用代码，不塞 prompt** | 多写 30 行 `apply_business_rules` | "焦糊/冒烟 → urgency=3" 的可靠性是 100%。演示 3 里"有点糊味"被正确判为紧急，靠的就是这一行 |
| 3 | **路由用规则函数，不用模型** | 路由表要人工维护 | `check_action` 把 62% 的工单挡在 Agent 之外，直接决定了成本与 P95；而且路径可枚举、可审计 |
| 4 | **只有 critical 工具才要人工确认** | 要给每个工具标风险等级 | 查库存/判保修免确认，体验不崩；派单必须确认，事故不发生 |
| 5 | **成本记账做到节点级** | 每个节点多写一行 `estimate_cost` | 8.1 节那张饼图是这么来的。**没有节点级账单，"优化成本"就只能靠猜** |
| 6 | **抽取链的降级是 LCEL 原生的** | 要理解 `with_retry` 与 `with_fallbacks` 的边界 | vLLM 挂掉时系统仍然可用（延迟涨到 4.2s），而不是 500 |
| 7 | **评测直接复用项目 1 的 harness** | 写一个 90 行适配器 | 端到端指标与其它三个系统同口径可比，不用再造一套评测 |
| 8 | **manifest + 哈希 + 软链** | 训练流水线多一步 | 线上跑的是哪个版本、用什么数据训的、评测多少分——一条命令查得到；回滚 60 秒 |
| 9 | **写了 `eval_mode`** | 服务里多一个字段 | 评测跑 60 条端到端时**绝不会真的派单**。这个字段防住了一次"评测污染生产工单"的事故 |

### 8.3 踩坑表（20 条）

| # | 现象 | 根因 | 解决 | 预防 |
|---|---|---|---|---|
| 1 | 训练 loss 降得很好，推理时模型**把 prompt 又背了一遍**才输出 JSON | `labels` 没有 mask 掉 prompt 部分，模型学的是"续写整段对话"而不是"只输出答案" | `collator.py` 里把 prompt 段的 label 全置 `-100`，只对 assistant 段计算 loss | 训练前打印一条样本的 `input_ids` 与 `labels` 对照，确认 `-100` 的位置正确（Step 3 的自检） |
| 2 | 离线评测 94%，线上准确率掉到 80% 多 | **chat template 不一致**：训练用 `tokenizer.apply_chat_template`，线上 vLLM 用的是模型 config 里的模板，两者的 system 段格式差一个换行 | 训练与推理**共用同一份 template**；用 vLLM 的 `/v1/chat/completions`（它会用模型自带模板），不要自己拼字符串 | 上线前拿同一条样本分别走"训练时的 tokenize"和"线上 API"，diff 两者的最终 prompt 字符串 |
| 3 | 加载 adapter 报 `size mismatch for base_model...` | **adapter 与基座版本对不上**（基座被重新下载过，或换了 Instruct/Base 变体） | manifest 里记 `base_model_revision`（config.json 的哈希），加载前比对 | 7.5 节的 `pull_adapter.sh` 校验；基座模型目录设为只读 |
| 4 | 每天有零星几单 JSON 解析失败 | 模型在超长/超脏输入上偶发输出截断或多一个逗号 | `loose_json_parse` 宽松解析（剥围栏、截取 `{...}`）+ 解析失败降级到大模型 | `eval_fields.py` 的"JSON 可解析率"进回归；**永远不要假设模型输出合法 JSON** |
| 5 | vLLM 挂了一晚上没人知道，第二天账单涨了 20 倍 | **降级链路生效了但没有告警**：全量走大模型，功能正常、成本暴涨 | `healthcheck.sh` 监控 `degraded_rate`，>5% 告警 | Step 6.6 说过"没演练过的降级等于没有"；这里补一句：**没有告警的降级等于定时炸弹** |
| 6 | `--enable-lora` 启动报 `max_lora_rank` 错误 | 训练用 `r=16`，但启动没传 `--max-lora-rank`（默认 16 在某些版本是 8），或多 adapter 的 rank 不一致 | `--max-lora-rank` 设为所有 adapter 中最大的 `r`；rank 不同的 adapter 不要混挂 | manifest 里记 `r`，启动脚本从 manifest 读 |
| 7 | 工单里写"我们那台 XJ-500 不转了"（XJ-500 是不存在的型号），**微调模型把 `device_model` 抽成了 `XC-500`**，而不是老实填 null | SFT 数据里全是"能抽到"的样本，模型没学过"抽不到就填 null" | 训练集里刻意加入 8%~12% 的**空字段样本**与**无关文本样本**；评测看 `编造率`（6.2 节的 null 误判率） | 把 `device_model 编造率 ≤ 3%` 写进回归门禁 |
| 8 | 线上大量重试，延迟 P95 从 736ms 涨到 3s | **pydantic 校验太严**：`suggested_craft` 模型偶尔输出"电气/控制"，校验失败 → 触发降级 | 两条一起做：① 加一层**枚举归一**（`电气/控制` → `电气`，取第一个合法值）；② 校验失败才降级，不要重试同一个模型 | `eval_fields.py` 的"枚举合法率"与"格式异常样例"每次都看 |
| 9 | 模型学会了输出字符串 `"null"` 而不是 JSON 的 `null` | 构造训练数据时用了 `json.dumps` 前先把 None 转成了 `"null"` 字符串 | 修数据生成脚本；`norm_value` 里把 `"null"/"None"/"无"/"NA"` 一律归一成 `None` | `test_schema.py` 加一条：`TicketExtract(device_model="null")` 必须被归一或拒绝 |
| 10 | 同一条工单跑两次，JSON 的**键顺序不同**，人工 diff 很难看 | 训练样本的字段顺序不固定，模型学到了"顺序随意" | 构造数据时**固定字段顺序**（与 `schema.json` 一致），prompt 里的示例也固定顺序 | 评测时按 key 排序后再 diff；schema 里的顺序就是唯一顺序 |
| 11 | AWQ 量化后字段准确率掉了 3pp | **校准集用了通用语料**（wikitext），不是业务分布 | 校准集从 `data/sft/train.jsonl` 抽 256 条 | Step 5.6 的量化前后对照必须跑，阈值 -1.0pp |
| 12 | 合并 adapter 后模型输出乱码 | 用 **4bit 量化的基座**去合并 adapter（QLoRA 训练时基座是 4bit，但合并必须用 fp16 基座） | `merge_lora.py` 强制加载 fp16 基座，内存不够就加 swap | 脚本里断言 `model.dtype == torch.float16` |
| 13 | 高峰期降级率飙到 30%，但 vLLM 没挂 | `EXTRACT_TIMEOUT_MS=1000` 设得太小，排队时超时 | 按 P95（736ms）的 3~4 倍设，即 3000ms；同时把 `max-num-seqs` 调到 64 | 压测（`bench/bench_extract.py`）看并发下的 P99，再定 timeout |
| 14 | `trace` 里只有最后一个节点的记录，前面的全丢了 | `TicketState` 里 `trace` 没声明 `Annotated[list, operator.add]`，LangGraph 默认"后写覆盖" | 累积型字段（`trace` / `errors` / `cost_items`）一律加 reducer | Step 7.2 已写明；**这是 LangGraph 最常见的坑** |
| 15 | 人工确认功能在本地好用，部署后中断就丢 | 用了 `MemorySaver`，进程重启状态全无 | 生产用 `AsyncSqliteSaver` 并把 db 落在挂载卷上（`/data/checkpoints.sqlite`） | 上线 checklist 第 12 条；压力更大时换 Postgres checkpointer |
| 16 | Agent 在一单上转了几十轮，把 API 额度烧掉一截 | 没有迭代上限，模型反复调同一个工具 | `AGENT_MAX_ITER=6` + `route_after_think` 里硬判 + `tests/test_agent_limit.py` 验证 | Step 8.7 的测试要进 CI，不能只写在文档里 |
| 17 | `degraded` 永远是 `False`，看不出到底降级没降级 | `with_fallbacks` 成功兜住后不会告诉你走了哪条分支；`degraded=True` 只在"连大模型都失败"时出现 | 接 Langfuse 看 `run_name`，或在 small 分支末尾挂一个打标 `RunnableLambda` | 用 `degraded_rate` 之外再看 **Langfuse 里 `extract_by_big_model` 的调用次数** |
| 18 | 长工单（经销商系统导出的带表格的单）静默漏抽字段 | 小模型 `--max-model-len 2048`，超长部分被**静默截断** | `should_escalate()` 里对 >1200 字的工单直接走大模型（预降级）；同时 prompt 里把关键段前置 | 评测集里必须包含长文本样本；按长度分桶看准确率 |
| 19 | 起了 4 个 uvicorn worker 后，人工确认偶发 `database is locked` | SQLite checkpointer 多进程争锁 | 单 worker；要扩并发先把 checkpointer 换 Postgres | Dockerfile 的 CMD 写死 `--workers 1` 并写注释说明原因 |
| 20 | 测试集准确率虚高（98%+），线上明显达不到 | **数据泄漏**：同一张工单在不同渠道有副本，切分时进了 train 和 test | `check_leakage.py` 按"归一化文本的 MinHash"去重后再切分 | 切分后必须跑泄漏检查，并把重复率写进 `dataset_stats.md` |

### 8.4 什么情况下这套是过度设计（诚实章节）

8.1 节已经算出来了：**只看 API 账单，一年省 228 块，一次性投入 28 人时——这笔账是亏的。** 所以必须老实说清楚：**这套架构不是所有团队都该做。**

#### 8.4.1 三种"别做微调"的典型情况

| 情况 | 具体表现 | 应该怎么做 |
|---|---|---|
| **① 调用量小** | 日均 < 500 单，且没有批量回填需求，也看不到放量的路线图 | **直接用大模型 + 结构化输出**（`response_format={"type":"json_object"}` 或 function calling），把省下来的 28 人时花在**prompt + 后置规则 + 评测集**上。准确率 89.71% 配上 `apply_business_rules` 的硬规则，能到 92% 左右，够用 |
| **② 抽取任务简单** | 只要客观字段（型号、故障码、日期、金额），没有"按公司规范判断"的字段 | 同上。Step 4.4 已经用数字证明：**客观字段微调只提升 2~3pp，主观字段才提升 8pp+**。没有主观字段，微调的核心价值就不存在 |
| **③ 团队没有训练与评测能力** | 没人会看 loss 曲线、没人维护测试集、没有 GPU（或只能临时租）、没有 CI | **千万别做**。微调的真正成本不是那 4 小时 GPU，是**长期维护**：schema 变了要重训、基座升级要重训、数据漂移要重训，每次都要重跑评测。没有评测体系的微调模型，三个月后就成了一个**没人敢动的黑盒** |

#### 8.4.2 判断清单：给自己打个分

逐条打勾，**每条 1 分**，满分 12 分：

**成本与规模（0~4 分）**

- [ ] 抽取类调用日均 ≥ 1,000 次（或有 ≥ 10 万条的批量回填需求）
- [ ] 抽取的 prompt 里有 ≥ 1,000 token 的 few-shot 示例（微调能把它省掉）
- [ ] 12 个月内调用量有明确的 3 倍以上增长计划
- [ ] 有闲置或可长租的 GPU（不是每次训练都要走采购流程）

**任务性质（0~4 分）**

- [ ] 有"按公司内部规范判断"的字段（紧急度、工种、优先级、归类），而不只是客观抽取
- [ ] 输出格式固定且已冻结（有 schema 文件，不是口头约定）
- [ ] 有 ≥ 3,000 条可用的历史标注数据（或能用规则 + 大模型半自动造出来）
- [ ] 延迟有硬要求（P95 < 1s），或需要高吞吐（> 20 req/s）

**团队能力（0~4 分）**

- [ ] 有人能独立跑通"训练 → 评测 → 量化 → 部署"整条链路，并且这个人半年内不会走
- [ ] 有冻结的测试集与自动化评测脚本（哪怕只有 200 条）
- [ ] 有 CI 或至少有"每次改动必须跑一遍评测"的纪律
- [ ] 数据有合规要求（不能出内网），或行为可回归是硬需求

**评分对照**：

| 总分 | 结论 | 建议动作 |
|---|---|---|
| **≥ 9 分** | 该做，而且收益会比本项目大 | 按本项目的路径走，重点做好 8.5 节的 P0 优化 |
| **6~8 分**（本项目约 8 分） | 值得做，但**要把收益说实话** | 做，但汇报时用 8.1.3 的三条真实收益（人力 / 回填 / 合规），不要只讲"成本降 54%" |
| **3~5 分** | 先不做微调，**但要做 Part B / Part C** | LangGraph 编排 + Agent + 评测体系先建起来，抽取用大模型。等调用量或需求变化后再回来补 Part A——**架构不用改，只换一个 base_url** |
| **≤ 2 分** | 别做 | 大模型 + 结构化输出 + 后置规则 + 20 条测试用例。三天上线，够用很久 |

#### 8.4.3 这个架构最值钱的部分，其实不是微调

一个反直觉的结论：**如果只能留一样，应该留 Part B（LangGraph 编排）而不是 Part A（微调）。**

理由很直接：

| 如果去掉 | 会发生什么 | 损失程度 |
|---|---|---|
| 去掉 **Part A（微调）** | 抽取换成大模型，`base_url` 改一行。成本 ×2.2、延迟 +1.4s、准确率 -4.6pp | 有损失，但系统仍然能用、能维护 |
| 去掉 **Part B（编排）** | 没有状态机、没有条件路由、没有中断恢复。人工确认要自己做持久化，追问分支要自己写状态管理，成本记账无处挂 | **系统直接不成立** |
| 去掉 **Part C（Agent）** | 不能自动派单/查库存，退回"只给方案，人工执行" | 少一块业务价值，但系统仍然成立 |

所以本项目的推荐落地顺序是：**Part B → Part C → Part A**。先把流程管住，再让它动手，最后才考虑把重复劳动烧进权重。**倒过来做（先花两周微调一个模型，再想怎么用）是最常见的失败路径。**

### 8.5 优化清单

| 优先级 | 优化项 | 预期收益 | 工作量 | 风险 |
|---|---|---|---|---|
| P0 | **压生成侧 prompt**：检索块 5 → 3，长块只传摘要 | 单单成本 -22%（生成占 74.6%） | 1 天 | 召回信息变少，`multihop` 类可能掉分，必须跑端到端回归 |
| P0 | **`need_parts` 字段的标注口径重写** | 该字段准确率 90.63% → 目标 94%（当前是最差字段，且错误是标注模糊导致的） | 3 人时改口径 + 重标 800 条 dev/test | 口径变了历史指标不可比，要标注版本号 |
| P0 | **降级率告警接钉钉/飞书** | 防住 8.3 表里第 5 条那类"成本悄悄涨 20 倍"的事故 | 半天 | 无 |
| P1 | 小程序渠道**上采样到 45%** + 单独加追问策略 | 该渠道字段准确率 90.02% → 目标 93%（Step 4.5 的结论） | 1 天（含重训） | 其它渠道可能小幅回退，要看分渠道表 |
| P1 | **只读场景绕过 Agent**："只问库存"直接函数调用 | Agent 触发率 38% → 目标 25%，单单成本 -8% | 1 天 | 规则判错会让该用 Agent 的单走了简化路径 |
| P1 | **checkpointer 换 Postgres** + 多 worker | 并发能力从单 worker 提到 4 worker | 1 天 | 需要 8.3 表第 19 条的锁问题验证 |
| P1 | checkpoint **定期清理**（30 天前的 thread） | 防 SQLite 无限增长（healthcheck 第 8 项） | 半天 | 清掉未完成的确认会话，要先查 pending |
| P2 | **线上抽检流水线**：每天随机抽 30 单人工标注 | 建立第 3 层评测，量化"离线好 vs 线上好"的差距 | 2 天 + 每天 0.5 人时 | 持续人力投入 |
| P2 | `suggested_craft` 的"控制↔电气"**下游归并** | 消除 25 例混淆中的大部分（业务上本来就重叠） | 半天（改派单规则，不动模型） | 需要业务确认 |
| P2 | 抽取模型换 **1.5B**（当前 7B） | 显存 5.6GB → 1.5GB，延迟 412ms → 约 180ms，可与其它服务共卡 | 1 天（重训 + 重评） | 准确率预计掉 3~6pp（Step 3.5 的路径 B），**必须先看业务能不能接受** |
| P3 | **生成侧也微调**（14B 写"初判+步骤+注意事项"） | 攻下成本大头（74.6%） | 2~3 周 | 生成任务评测难度高一个数量级，**没有项目 1 那套 judge + Kappa 校准别碰** |
| P3 | Agent 工具调用**轨迹评测** | 能量化"工具选对了没有"（项目 1 的扩展作业 4） | 1 周 | 要扩金标集 |

### 8.6 本节小结

- **三位一体的真实收益不在 API 账单**（一年 ¥228），而在三处：人力节省 13.2 小时/月（约 2.1 个月回本）、批量回填省 ¥1,200 和 33 小时、以及放量后的杠杆；
- 另外两项不能折成钱但常常决定成败：**数据不出内网**（合规）和**行为可回归**（可维护）；
- 成本结构是"生成 74.6% + Agent 21.1% + 抽取 0.35%"，所以**继续优化抽取毫无意义**，该去压生成的 prompt；
- 20 条坑里最贵的三条：**labels mask 写错**（白训一次）、**chat template 不一致**（离线线上差 10pp）、**降级没告警**（成本悄涨 20 倍）；
- 最诚实的一条结论：**如果只能留一样，留 Part B（编排）而不是 Part A（微调）**。推荐落地顺序是 B → C → A，反过来做是最常见的失败路径；
- 用 8.4.2 那张 12 分清单给自己打个分。**≤ 2 分就别做微调**，大模型 + 结构化输出 + 后置规则三天就能上线，而且够用很久。

---

## 九、扩展作业

5 个作业按难度递增，每个都有可验证的验收标准。

### 作业 1：把抽取模型换成 1.5B 并做完整对照（★★☆☆☆）

8.5 节的 P2 里有这一条，但没人做过就永远只是猜测。

**要做的事**：

1. 把 `train_qlora.py` 的 `BASE_MODEL` 换成 `Qwen2.5-1.5B-Instruct`，`per_device_train_batch_size` 提到 4，其余超参不变，重训一遍；
2. 走完整流水线：训练 → `make_manifest.py` → 合并 → AWQ → 部署到 8003 端口；
3. 用 `eval/eval_fields.py` 在**同一份 800 条测试集**上跑，与 7B 版本逐字段对比；
4. 额外测两项 7B 没测的：**冷启动加载时间**和**与 embedding 模型共卡时的显存峰值**；
5. 产出一张决策表：在什么条件下应该用 1.5B。

**验收标准**：

- 拿到"字段严格准确率下降了多少 pp"的确切数字（预期 3~6pp，但你要测出自己的数）；
- 能回答："如果业务能接受整单完全正确率从 78.75% 降到 72%，能省多少显存/延迟/钱"；
- 两个 manifest（7B 与 1.5B）都在，`adapter_version` 能区分，切换脚本对两者都能用。

### 作业 2：给抽取加一致性自检与置信度（★★★☆☆）

现在抽取只输出字段，不输出"我有多确信"。下游因此无法区分"模型很确定"和"模型在猜"。

**要做的事**：

1. **自洽性检查（self-consistency）**：对同一条工单用 `temperature=0.7` 采样 3 次，对每个字段做多数投票；字段级一致率作为置信度；
2. 只在**高风险场景**触发（`urgency=3` 或 `need_parts=true`），否则成本要涨 3 倍；
3. 把字段级置信度写进 `TicketExtract`（新增 `_confidence: dict[str, float]`，注意**不进 schema.json**，它是运行时元数据不是契约）；
4. `finalize` 里用它修正整单 `confidence`：任一关键字段置信度 < 0.67（3 次投票里只中 2 次）就标 `need_human=true`；
5. 在 800 条测试集上验证：**置信度是否与正确率相关**（画一张"置信度分桶 × 实际准确率"的表）。

**验收标准**：

- 置信度 = 1.0 的样本准确率应显著高于置信度 < 1.0 的样本（差距 ≥ 15pp），否则这个置信度没有意义；
- 触发自洽检查的工单比例 ≤ 25%，整体成本上涨 ≤ 8%；
- 能给出"用置信度做人工兜底路由"的收益估算：拦住多少错单、多少单被误拦。

### 作业 3：把评测接进 CI，做成门禁（★★★☆☆）

项目 1 的第七节已经给了完整的 CI 门禁实现，这个作业是把它**用到微调项目上**——微调项目的门禁有自己的特殊性：模型是二进制产物，不能靠 diff 代码来判断变没变。

**要做的事**：

1. 写 `.github/workflows/model_gate.yml`，在 `manifest.json` 或 `data/sft/*.jsonl` 变化时触发；
2. 门禁判三类条件：
   - **绝对底线**：schema 通过率 ≥ 99%、字段严格准确率 ≥ 93%、整单完全正确率 ≥ 76%、`device_model` 编造率 ≤ 3%；
   - **相对回归**：相比 `manifest.json` 里记录的上一版，任一字段准确率跌幅 ≤ 1.5pp；
   - **一票否决**：`fault_code` 与 `device_model`（会导致订错件）跌幅 ≤ 0.5pp；
3. CI 里不训练（太慢），只做**推理评测**：拉取 adapter → 起 vLLM → 跑 `eval_fields.py` → 判定；
4. 把 `eval_fields.py` 的 JSON 输出与上一版 manifest 做 diff，红灯时在 PR 里贴出**逐字段变化表**；
5. 加一条 CI 检查：**manifest 里的 `train_data.sha256` 与仓库里的训练数据必须一致**（防"用 A 数据训的模型贴了 B 数据的 manifest"）。

**验收标准**：

- 故意把 `urgency` 的训练样本打乱 20% 重训后提 PR，CI 必须红灯，并指出是 `urgency` 掉的；
- CI 跑完 ≤ 20 分钟（用 `--limit 300` 而不是全量 800 条，但要在 PR 评论里说明用的是子集）；
- manifest 哈希不一致时红灯，且错误信息能说清是哪个文件不一致。

### 作业 4：实现多 adapter 的任务路由（★★★★☆）

现在只有一个抽取 adapter。真实项目里往往有多个小任务都值得微调：工单摘要（给管理者看的一句话）、客户情绪分级（决定是否升级投诉）、备件推荐（根据故障现象推件号）。

**要做的事**：

1. 再训两个 adapter：`ticket-summary`（一句话摘要）和 `ticket-sentiment`（情绪 1~5 级），各用 3,000 条数据；
2. 用 vLLM 的 `--enable-lora --lora-modules` 把三个 adapter 挂在**同一个基座**上（Step 5.7 的路线 1），注意 `--max-lora-rank` 要取最大值；
3. 扩展 `app/routing/model_router.py`：`ROUTING` 表加 `summary → small:ticket-summary`、`sentiment → small:ticket-sentiment`，让 `model_of()` 返回具体 adapter 名；
4. 在 LangGraph 主图里加一个**并行节点组**：`extract` / `summary` / `sentiment` 三个节点并行执行（LangGraph 里多条边指向同一个下游节点会自动并行），再汇聚到 `validate`；
5. 实测多 adapter 的吞吐损失，与"三个独立 vLLM 实例"做对比；
6. 处理一个新问题：**三个 adapter 的版本要一起管**，manifest 要支持多 adapter。

**验收标准**：

- 三个任务并行后，总延迟接近最慢的那一个（而不是三者之和），并给出实测数字；
- 多 adapter 方案 vs 三实例方案的对照表：显存、吞吐、切换成本、运维复杂度；
- `/health` 能返回三个 adapter 的版本；任一 adapter 加载失败时服务**降级但不崩**（对应任务走大模型）。

### 作业 5：把整套系统做成可交付产品（★★★★★）

现在它是一个 demo。要变成能交给客户 IT 部门自己运维的东西，还差很多。

**要做的事**：

1. **多租户**：一套服务支持多个客户（不同的知识库 collection、不同的备件表、不同的 SLA 规则）。租户 ID 从请求头取，贯穿 state、检索过滤、工具调用、成本记账；
2. **配置中心化**：`check_action` 的关键词表、工具风险等级、`urgency` 的业务规则，全部从代码搬到数据库/配置文件，支持热更新与灰度（改配置不重启）；
3. **人工确认改成异步**：现在 `/api/ticket` 会挂住等确认。改成"提交即返回 pending → 推送到客服工作台 → 客服在 App 上点批准 → 回调恢复"，并处理**超时自动拒绝**（如 30 分钟无人处理）；
4. **完整可观测**：Langfuse trace + Prometheus 指标（QPS、P95、降级率、Agent 触发率、单单成本、人工确认等待时长）+ Grafana 看板 + 三条告警规则；
5. **数据闭环**：把客服对抽取结果的**修正操作**收集起来，作为下一轮训练数据（这是微调项目最值钱的飞轮）；
6. **交付物**：安装包（compose + 镜像 tar）、运维手册、故障处理手册（至少覆盖 8.3 表里的 10 条）、验收测试脚本。

**验收标准**：

- 客户 IT 部门的人**不看源码**、只照着运维手册，能在 2 小时内完成部署并通过验收测试；
- 两个租户的数据完全隔离：A 租户的工单绝不会检索到 B 租户的知识（写一个测试证明它）；
- 人工确认超时自动拒绝生效，且被拒绝的工单进入人工队列而不是消失；
- 数据闭环跑通：客服修正 100 条 → 自动进入 `data/sft/feedback.jsonl` → 能触发一次增量训练 → 新 manifest 的评测分数有记录（**哪怕分数没涨也算通过**，闭环本身才是目标）。

---

**上一章** [实战项目 2：RAG 与 Agent 双引擎智能决策问答系统](项目2-RAG与Agent双引擎智能决策问答系统.md) | **下一章** [实战项目 4：RAG 性能突围 —— 查询速度提升 10 倍](项目4-RAG性能突围-查询速度提升10倍.md)
