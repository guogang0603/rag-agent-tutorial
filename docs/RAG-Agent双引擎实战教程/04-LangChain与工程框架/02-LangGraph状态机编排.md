# 第 4.2 章  LangGraph 状态机编排

> **本章目标**：读完能做到 …
> 1. 说清 LCEL（DAG）与 LangGraph（状态机）的能力边界，知道什么时候必须上图；
> 2. 独立设计 State（含 reducer）、写 Node、连条件边，编译出一张带循环的图并导出 mermaid；
> 3. 用 Checkpointer 实现多轮会话、断点续跑和时间旅行（回到任意历史 checkpoint 重跑）；
> 4. 实现「高危操作人工审批」的中断与恢复（`interrupt` + `Command(resume=...)`）；
> 5. 用四种 `stream_mode` 把节点内部的 token 与自定义进度推到前端；
> 6. 用 `Send` API 做动态 fan-out/fan-in，用子图复用 RAG 子流程；
> 7. 用 LangGraph 完整重写第 3.3 章的 CRAG（纠错式 RAG），并部署成 FastAPI 流式接口。
>
> **前置知识**：
> - [4.1 LangChain 核心抽象与 LCEL](./01-LangChain核心抽象与LCEL.md)
> - [第 3 模块 RAG 进阶与性能优化](../03-RAG进阶与性能优化/)（CRAG / Self-RAG 的原理）
>
> **预计用时**：阅读 75 分钟 / 动手 240 分钟

---

## 一、为什么需要它（问题出发）

### 1.1 华成机电的工单 Agent 需求

第 4.1 章我们用 LCEL 拼出了一条很漂亮的 RAG 链。但产品经理提了新需求——「工单智能助手」要能做这些事：

1. 用户问"YCT-132 报 E07"，先检索知识库；
2. **如果检索结果质量不行**，自动改写查询再检索一次，最多试 3 次；
3. 3 次还不行就**转去查历史工单库**，还不行就**联网搜官方公告**；
4. 拿到资料后生成答案，然后**自检答案有没有幻觉**，有幻觉就**回炉重新生成**；
5. 如果用户是要"建单/派单/退款"，调用工单系统 API，但**派单和退款必须先弹给主管审批**，主管点了同意才能继续；
6. 整个过程要能**中途断掉再恢复**（主管可能第二天才审批）；
7. 出了问题要能**回到某一步重跑**（"上次那个改写不对，从改写那步重来"）。

用 LCEL 写写看：

```python
# 用 LCEL 硬凑，会变成什么样
chain = (
    retrieve
    | grade
    | RunnableBranch(
        (lambda x: x["quality"] == "good", generate),
        (lambda x: x["retry"] < 3, ???)     # 这里要跳回 retrieve —— LCEL 表达不了
    )
)
```

**`|` 只能从左往右流，它本质上是一个有向无环图（DAG）。跳回去就是环，DAG 定义上不允许。** 你只能写成 Python `while` 循环手工驱动，但那样一来：

- 中间状态是局部变量，进程重启就没了 → 做不了需求 6；
- 循环内部的执行过程对框架不可见 → 流式、回调、追踪全断；
- 想在某个位置暂停等待人工 → 只能阻塞线程等着，或者自己造一套状态持久化 → 需求 5 和 7 成本爆炸。

### 1.2 LCEL 与 LangGraph 的能力对比

| 能力 | LCEL（Chain） | LangGraph（图） |
|---|---|---|
| 拓扑结构 | 有向无环图（DAG），单向流 | 任意有向图，**允许环** |
| 条件分支 | `RunnableBranch`（分支后合不回来） | `add_conditional_edges`（可分可合可回跳） |
| 循环 | 不支持（只能外部 while） | **原生支持**，配 `recursion_limit` 防死循环 |
| 状态 | 靠 dict 在链上传递，无显式定义 | **显式 State（TypedDict）+ reducer**，每步只返回增量 |
| 持久化 | 无（`RunnableWithMessageHistory` 只管消息） | **Checkpointer**，每个超步自动落盘 |
| 中断/恢复 | 无 | `interrupt_before` / `interrupt()` + `Command(resume=...)` |
| 时间旅行 | 无 | `get_state_history()` + 指定 `checkpoint_id` 重跑 |
| 人在回路 | 要自己造轮子 | 一等公民 |
| 并行 | `RunnableParallel`（静态、编译期确定） | 静态并行 + **`Send` 动态 fan-out**（运行期决定分几路） |
| 节点重试 | `with_retry`（整条链粒度） | `RetryPolicy`（**单节点粒度**） |
| 可视化 | `get_graph().print_ascii()` | `get_graph().draw_mermaid()`，图结构一目了然 |
| 适用场景 | 固定流程的 RAG、抽取、分类 | **Agent、多智能体、带审批的业务流** |

**一句话判断标准：**

> 流程画出来是一条线（或一棵树）→ 用 LCEL。
> 画出来有箭头指回去，或者中间要停下来等人 → 用 LangGraph。

### 1.3 LangGraph 不是"LangChain 的 Agent 模块"

这是个常见误解。准确定位：

```mermaid
flowchart TD
    A["langchain-core<br/>Runnable 协议 / 消息 / 工具定义"] --> B["LCEL<br/>DAG 组合"]
    A --> C["LangGraph<br/>状态机 + 持久化运行时"]
    C --> D["Agent<br/>（图的一种特例：ReAct 循环）"]
    C --> E["多智能体<br/>（图的一种特例：Supervisor/Swarm）"]
    C --> F["带审批的业务流<br/>（图 + 中断）"]
    B -.可以作为图的一个节点.-> C

    style C fill:#e8f4ff,stroke:#2b6cb0,stroke-width:2px
```

- LangGraph 是一个**独立的包**（`pip install langgraph`），它依赖 `langchain-core` 的消息类型，但**不依赖 `langchain` 主包**。
- 它本质是一个「带持久化的分布式状态机运行时」，Agent 只是它最常见的用法之一。
- 编译出来的图（`CompiledStateGraph`）**本身就是一个 Runnable**，所以它有 `invoke` / `stream` / `astream_events`，可以塞进 LCEL 链里当一个环节。

**这就是为什么本书把 LangGraph 当作全书 Agent 与多智能体的技术底座**：第 6 模块的 ReAct / Plan-and-Execute / Reflexion，第 7 模块的 Supervisor / Swarm / 流水线，全部是在这一章的基础上搭的。

---

## 二、原理拆解：图、状态、超步

### 2.1 执行模型：Pregel 超步

LangGraph 借鉴了 Google Pregel 的批量同步并行（Bulk Synchronous Parallel, BSP）模型：

```mermaid
flowchart TD
    subgraph S1["超步 1 (superstep)"]
        A["节点 A 执行"] --> A2["返回状态增量"]
    end
    subgraph S2["超步 2"]
        B["节点 B 执行"] --> B2["返回增量"]
        C["节点 C 执行<br/>（与 B 并行）"] --> C2["返回增量"]
    end
    subgraph S3["超步 3"]
        D["节点 D 执行"] --> D2["返回增量"]
    end

    A2 -->|"reducer 合并<br/>写 checkpoint"| S2
    B2 -->|"reducer 合并<br/>写 checkpoint"| S3
    C2 --> S3
    D2 -->|"写 checkpoint"| E["END"]
```

要点：

1. **一个超步 = 一批可并行执行的节点**。同一超步内的节点看到的是**同一份**状态快照。
2. 节点执行完**只返回状态的增量**（一个 dict，只含它改动的键），不返回完整状态。
3. 超步结束时，框架用每个键配置的 **reducer** 把增量合并进主状态。
4. **每个超步结束后写一次 checkpoint**。这就是持久化、中断恢复、时间旅行的基础。
5. 下一个超步执行哪些节点，由**边**决定（普通边固定，条件边由函数运行期返回）。

### 2.2 三要素：State / Node / Edge

```mermaid
flowchart LR
    subgraph G["StateGraph"]
        direction TB
        ST["State<br/>TypedDict + reducer"]
        N1["Node: retrieve"]
        N2["Node: grade"]
        N3["Node: generate"]
        N1 -->|edge| N2
        N2 -->|conditional edge| N3
        N2 -.->|conditional edge<br/>回跳| N1
    end
    START(("START")) --> N1
    N3 --> ENDN(("END"))
```

| 要素 | 是什么 | 怎么写 |
|---|---|---|
| State | 图上流动的数据结构 | `class S(TypedDict): x: Annotated[list, add]` |
| Node | 一个 Python 函数：`(state) -> 部分 state` | `builder.add_node("name", fn)` |
| Edge | 节点之间的连接 | `add_edge(a, b)` / `add_conditional_edges(a, router, mapping)` |
| START / END | 特殊哨兵节点 | `from langgraph.graph import START, END` |
| compile | 把图定义变成可执行 Runnable | `graph = builder.compile(checkpointer=...)` |

### 2.3 State 与 reducer：最重要的设计决策

**State 的每个键都可以配一个 reducer，决定"新值怎么合并到旧值"。**

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class MyState(TypedDict):
    # 无 reducer（默认）：新值直接覆盖旧值
    question: str

    # add（就是 operator.add）：列表拼接
    visited_nodes: Annotated[list[str], add]

    # add_messages：消息专用 reducer
    messages: Annotated[list[AnyMessage], add_messages]

    # 自定义 reducer
    scores: Annotated[dict, lambda old, new: {**old, **new}]
```

**`add_messages` 做了三件普通 `add` 做不到的事**（这是它存在的全部理由）：

1. **自动补 ID**：新消息没有 `id` 就生成一个 uuid；
2. **按 ID 去重与更新**：如果新消息的 `id` 和历史里某条相同，就**替换**那一条而不是追加——这是实现"修改历史消息"（比如把工具的敏感返回值脱敏后回写）的关键；
3. **自动反序列化**：`{"role": "user", "content": "hi"}` 这样的 dict 会被转成 `HumanMessage`。

```python
# add_messages 的三个行为演示
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import add_messages

old = [HumanMessage(content="E07 怎么办", id="m1")]

# 1) 追加
print(add_messages(old, [AIMessage(content="先断电", id="m2")]))
# -> [HumanMessage(id='m1'), AIMessage(id='m2')]

# 2) 按 ID 覆盖（注意 id 相同）
print(add_messages(old, [HumanMessage(content="E07 怎么办？（已脱敏）", id="m1")]))
# -> [HumanMessage(content='E07 怎么办？（已脱敏）', id='m1')]  只有一条

# 3) 删除（配合 RemoveMessage，用于裁剪历史）
print(add_messages(old, [RemoveMessage(id="m1")]))
# -> []
```

**状态设计的五条最佳实践**（华成机电踩过坑总结的）：

| # | 实践 | 反面教材 |
|---|---|---|
| 1 | **State 只放"跨节点需要共享"的数据**，节点内部临时变量不要塞进去 | 把 embedding 向量、原始 HTTP 响应塞进 State，checkpoint 体积暴涨到几十 MB |
| 2 | **能覆盖就别累积**。只有真正需要历史的字段才用 `add` | `visited: Annotated[list, add]` 在循环里跑 50 轮就是 50 个元素，序列化变慢 |
| 3 | **消息一律用 `add_messages`**，不要用 `operator.add` | 用 add 会导致无法按 ID 更新/删除消息，历史裁剪没法做 |
| 4 | **State 里的值必须可 JSON 序列化**（或 checkpointer 支持的序列化方式） | 放了 `Milvus` 客户端对象，SqliteSaver 直接报 pickle 错误 |
| 5 | **用输入/输出 schema 收窄对外接口** | 内部 State 有 20 个字段，全暴露给调用方，接口一改全崩 |

第 5 条的写法：

```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str
    citations: list[dict]

class InternalState(InputState, OutputState):
    """内部完整状态，包含所有中间字段。"""
    documents: list
    retry_count: int
    grade: str

builder = StateGraph(InternalState, input=InputState, output=OutputState)
# invoke 时只能传 question，返回只有 answer 和 citations
```

### 2.4 Node 的三种形态

```python
# 形态 1：同步普通函数（最常见）
def retrieve(state: MyState) -> dict:
    """返回部分状态即可，不需要返回完整 State。"""
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "visited_nodes": ["retrieve"]}


# 形态 2：异步函数（生产推荐，IO 密集）
async def agenerate(state: MyState) -> dict:
    """异步节点，在 ainvoke/astream 时才会真正并发。"""
    ai = await llm.ainvoke(state["messages"])
    return {"messages": [ai]}


# 形态 3：带 config 的函数（拿 thread_id、自定义配置）
from langchain_core.runnables import RunnableConfig

def node_with_cfg(state: MyState, config: RunnableConfig) -> dict:
    """第二个参数名必须是 config，框架按名字注入。"""
    thread_id = config["configurable"]["thread_id"]
    tenant = config["configurable"].get("tenant", "default")
    return {"visited_nodes": [f"cfg:{tenant}"]}


# 形态 4：一个 Runnable（LCEL 链）也可以直接当节点
builder.add_node("rag", rag_chain)     # 只要输入输出能对上 State 的形状
```

三个约束：

- **节点返回 `None` 或 `{}` 表示不改状态**，这是合法的；
- **返回的 key 必须在 State 里声明过**，否则 0.2.x 会静默丢弃（调试时最坑的一点）；
- **节点里不要做有副作用且不可重入的操作**。节点可能被重试或时间旅行重跑，扣款、发短信这类操作要自己做幂等。

### 2.5 Edge 的四种形态

```python
from langgraph.graph import START, END

# 1) 普通边：A 跑完一定跑 B
builder.add_edge("retrieve", "grade")

# 2) 入口边
builder.add_edge(START, "retrieve")
# 等价于 builder.set_entry_point("retrieve")

# 3) 出口边
builder.add_edge("generate", END)

# 4) 条件边：路由函数返回下一个节点名
def route(state: MyState) -> str:
    """返回值会被 path_map 映射成真实节点名。"""
    if state["grade"] == "good":
        return "ok"
    if state["retry_count"] >= 3:
        return "giveup"
    return "rewrite"

builder.add_conditional_edges(
    "grade",                                  # 从哪个节点出发
    route,                                    # 路由函数
    {"ok": "generate", "rewrite": "transform_query", "giveup": "fallback"},  # path_map
)

# 5) 多入边 = 自动汇聚（fan-in）：A->C, B->C，C 会等 A 和 B 都跑完
builder.add_edge("branch_a", "merge")
builder.add_edge("branch_b", "merge")

# 6) 一个节点连多条普通出边 = 并行 fan-out
builder.add_edge("split", "branch_a")
builder.add_edge("split", "branch_b")
```

> `path_map` 不是必须的（路由函数可以直接返回节点名），但**强烈建议写**：一是可读性，二是 `draw_mermaid()` 画图时只有给了 path_map 才能画出分支标签。

---

## 三、动手实战（一）：从零搭一个"条件分支 + 循环"的图

场景：华成机电的"工单分诊 + 知识检索"流程。检索质量不达标就改写查询重试，最多 3 次，超了就转人工。

```python
# huacheng/graph/basic_loop.py
"""从零搭一张带条件分支和循环的 LangGraph 图。"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


# ========== 1. 定义 State ==========
class TicketState(TypedDict):
    """工单检索流程的共享状态。"""

    question: str                                   # 覆盖语义
    query: str                                      # 当前实际用于检索的查询（会被改写）
    documents: list[str]                            # 覆盖语义：每次检索都是全新结果
    grade: str                                      # good / bad
    retry_count: int                                # 覆盖语义
    answer: str
    trace: Annotated[list[str], operator.add]       # 累积语义：执行轨迹


# ========== 2. 定义 Node ==========
# 这里用假数据模拟检索，方便你不接真环境也能跑通
_FAKE_KB = {
    "YCT-132 E07 过流": ["[手册P87] E07=输出过流保护", "[参数表] F0-09 加速时间"],
    "变频器 过流 报警 处理": ["[手册P87] E07=输出过流保护，排查电机绝缘"],
}


def retrieve(state: TicketState) -> dict:
    """按当前 query 检索知识库。"""
    q = state.get("query") or state["question"]
    docs: list[str] = []
    for k, v in _FAKE_KB.items():
        if any(tok in q for tok in k.split()):
            docs.extend(v)
    return {"documents": docs, "query": q, "trace": [f"retrieve(q={q!r}) -> {len(docs)} docs"]}


def grade_documents(state: TicketState) -> dict:
    """给检索结果打分。真实项目这里调 LLM 或 reranker，这里用长度规则模拟。"""
    docs = state["documents"]
    grade = "good" if len(docs) >= 2 else "bad"
    return {"grade": grade, "trace": [f"grade -> {grade} (docs={len(docs)})"]}


def transform_query(state: TicketState) -> dict:
    """查询改写。真实项目调 LLM，这里用规则模拟。"""
    n = state.get("retry_count", 0) + 1
    rewrites = {
        1: "变频器 过流 报警 处理",
        2: "YCT-132 E07 过流",
        3: "输出过流 保护 排查 步骤",
    }
    new_q = rewrites.get(n, state["question"])
    return {"query": new_q, "retry_count": n, "trace": [f"transform_query #{n} -> {new_q!r}"]}


def generate(state: TicketState) -> dict:
    """基于检索结果生成答案。"""
    ctx = "；".join(state["documents"])
    return {"answer": f"根据资料（{ctx}），建议先断电检测电机绝缘，再调整加速时间。",
            "trace": ["generate"]}


def escalate(state: TicketState) -> dict:
    """三次都没检到，转人工。"""
    return {"answer": "知识库未收录该问题，已为您创建人工工单，工程师将在 2 小时内联系。",
            "trace": ["escalate(转人工)"]}


# ========== 3. 路由函数 ==========
def decide_next(state: TicketState) -> Literal["generate", "rewrite", "escalate"]:
    """检索质量好就生成；不好且还有重试机会就改写；否则转人工。"""
    if state["grade"] == "good":
        return "generate"
    if state.get("retry_count", 0) >= 3:
        return "escalate"
    return "rewrite"


# ========== 4. 建图 ==========
builder = StateGraph(TicketState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade_documents)
builder.add_node("transform_query", transform_query)
builder.add_node("generate", generate)
builder.add_node("escalate", escalate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    decide_next,
    {
        "generate": "generate",
        "rewrite": "transform_query",
        "escalate": "escalate",
    },
)
builder.add_edge("transform_query", "retrieve")      # <<< 这就是环
builder.add_edge("generate", END)
builder.add_edge("escalate", END)

graph = builder.compile()


if __name__ == "__main__":
    # 4a) 导出 mermaid（贴进文档就能看）
    print("=== mermaid ===")
    print(graph.get_graph().draw_mermaid())

    # 4b) 执行并打印轨迹
    print("\n=== 执行：能检到的问题 ===")
    out = graph.invoke({"question": "YCT-132 E07 过流", "retry_count": 0, "trace": []})
    for t in out["trace"]:
        print("  ", t)
    print("   answer:", out["answer"][:50])

    print("\n=== 执行：检不到的问题（会循环 3 次后转人工）===")
    out2 = graph.invoke({"question": "咖啡机不出水怎么修", "retry_count": 0, "trace": []})
    for t in out2["trace"]:
        print("  ", t)
    print("   answer:", out2["answer"][:50])

    # 4c) 用 stream 观察每一个超步
    print("\n=== stream(updates)：逐超步看状态增量 ===")
    for step in graph.stream(
        {"question": "咖啡机不出水怎么修", "retry_count": 0, "trace": []},
        stream_mode="updates",
    ):
        for node, delta in step.items():
            keys = {k: (v if not isinstance(v, list) else f"list[{len(v)}]") for k, v in delta.items()}
            print(f"   [{node}] {keys}")
```

生成的 mermaid（`draw_mermaid()` 的输出，可直接贴进 md）：

```mermaid
graph TD;
    __start__([START]):::first --> retrieve;
    retrieve --> grade;
    grade -. generate .-> generate;
    grade -. rewrite .-> transform_query;
    grade -. escalate .-> escalate;
    transform_query --> retrieve;
    generate --> __end__([END]):::last;
    escalate --> __end__;
    classDef first fill-opacity:0;
    classDef last fill:#bfb6fc;
```

预期输出：

```text
=== 执行：能检到的问题 ===
   retrieve(q='YCT-132 E07 过流') -> 3 docs
   grade -> good (docs=3)
   generate
   answer: 根据资料（[手册P87] E07=输出过流保护；[参数表] F0-0

=== 执行：检不到的问题（会循环 3 次后转人工）===
   retrieve(q='咖啡机不出水怎么修') -> 0 docs
   grade -> bad (docs=0)
   transform_query #1 -> '变频器 过流 报警 处理'
   retrieve(q='变频器 过流 报警 处理') -> 1 docs
   grade -> bad (docs=1)
   transform_query #2 -> 'YCT-132 E07 过流'
   retrieve(q='YCT-132 E07 过流') -> 3 docs
   grade -> good (docs=3)
   generate
   answer: 根据资料（[手册P87] E07=输出过流保护；[参数表] F0-0

=== stream(updates)：逐超步看状态增量 ===
   [retrieve] {'documents': 'list[0]', 'query': '咖啡机不出水怎么修', 'trace': 'list[1]'}
   [grade] {'grade': 'bad', 'trace': 'list[1]'}
   [transform_query] {'query': '变频器 过流 报警 处理', 'retry_count': 1, 'trace': 'list[1]'}
   [retrieve] {'documents': 'list[1]', 'query': '变频器 过流 报警 处理', 'trace': 'list[1]'}
   [grade] {'grade': 'bad', 'trace': 'list[1]'}
   [transform_query] {'query': 'YCT-132 E07 过流', 'retry_count': 2, 'trace': 'list[1]'}
   [retrieve] {'documents': 'list[3]', 'query': 'YCT-132 E07 过流', 'trace': 'list[3]'}
   [grade] {'grade': 'good', 'trace': 'list[1]'}
   [generate] {'answer': '根据资料（...', 'trace': 'list[1]'}
```

注意 `stream_mode="updates"` 打印出来的是**每个节点返回的增量**，`trace` 那个键每次只有 1~3 个元素——因为节点只返回自己新增的部分，合并是框架用 `operator.add` 做的。这是理解 LangGraph 的关键。

**可视化的三种方式：**

```python
# 1) mermaid 文本（最实用，能贴文档、能进 CI）
print(graph.get_graph().draw_mermaid())

# 2) PNG 图片（需要网络或本地 graphviz/pyppeteer，CI 里不稳定）
png_bytes = graph.get_graph().draw_mermaid_png()
open("graph.png", "wb").write(png_bytes)

# 3) ASCII（终端里看，零依赖）
graph.get_graph().print_ascii()

# 4) 带子图展开（默认只画一层）
print(graph.get_graph(xray=True).draw_mermaid())
```

---

## 四、动手实战（二）：Checkpointer 与持久化

### 4.1 三种 Checkpointer

| Checkpointer | 包 | 适用 | 特点 |
|---|---|---|---|
| `MemorySaver` | `langgraph`（内置） | 单元测试、Notebook 演示 | 进程内字典，重启即失 |
| `SqliteSaver` / `AsyncSqliteSaver` | `langgraph-checkpoint-sqlite` | 单机部署、小规模 | 零运维，一个文件 |
| `PostgresSaver` / `AsyncPostgresSaver` | `langgraph-checkpoint-postgres` | **生产首选** | 多实例共享、支持并发、可备份 |

```bash
uv pip install langgraph-checkpoint-sqlite==2.0.1 langgraph-checkpoint-postgres==2.0.9
```

### 4.2 thread_id 与多轮会话

```python
# huacheng/graph/persistence.py
"""Checkpointer：多轮会话、断点续跑、时间旅行。"""
from __future__ import annotations

import operator
import sqlite3
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    """用一个极简的计数器状态演示持久化，便于观察。"""

    topic: str
    turn: int
    log: Annotated[list[str], operator.add]


def step_a(state: CounterState) -> dict:
    """第一步：把轮次 +1。"""
    n = state.get("turn", 0) + 1
    return {"turn": n, "log": [f"A: turn -> {n}"]}


def step_b(state: CounterState) -> dict:
    """第二步：记录主题。"""
    return {"log": [f"B: topic={state['topic']} turn={state['turn']}"]}


builder = StateGraph(CounterState)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)


# ---------- 1) MemorySaver：最快上手 ----------
mem_graph = builder.compile(checkpointer=MemorySaver())

# ---------- 2) SqliteSaver：单机持久化 ----------
# 注意 check_same_thread=False，FastAPI 多线程下必须加
conn = sqlite3.connect("./.cache/huacheng_graph.sqlite", check_same_thread=False)
sqlite_graph = builder.compile(checkpointer=SqliteSaver(conn))

# ---------- 3) PostgresSaver：生产 ----------
# from langgraph.checkpoint.postgres import PostgresSaver
# DB_URI = "postgresql://user:pwd@127.0.0.1:5432/huacheng?sslmode=disable"
# with PostgresSaver.from_conn_string(DB_URI) as cp:
#     cp.setup()                       # 首次运行必须调，建表
#     pg_graph = builder.compile(checkpointer=cp)


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "ticket-9527"}}

    print("=== 第 1 次调用 ===")
    r1 = sqlite_graph.invoke({"topic": "E07 排查", "log": []}, config=cfg)
    print("   turn =", r1["turn"], " log =", r1["log"])

    print("=== 第 2 次调用（同 thread_id，状态被继承）===")
    r2 = sqlite_graph.invoke({"topic": "加速时间调整", "log": []}, config=cfg)
    print("   turn =", r2["turn"], " log =", r2["log"])

    print("=== 换一个 thread_id（全新状态）===")
    cfg2 = {"configurable": {"thread_id": "ticket-9528"}}
    r3 = sqlite_graph.invoke({"topic": "另一个工单", "log": []}, config=cfg2)
    print("   turn =", r3["turn"], " log =", r3["log"])

    # ---------- 查看当前状态 ----------
    snap = sqlite_graph.get_state(cfg)
    print("\n=== get_state ===")
    print("   values   :", {k: v for k, v in snap.values.items() if k != "log"})
    print("   next     :", snap.next)                 # 下一个要执行的节点，() 表示已结束
    print("   checkpoint_id:", snap.config["configurable"]["checkpoint_id"][:20], "...")

    # ---------- 查看历史（时间旅行的基础） ----------
    print("\n=== get_state_history（从新到旧）===")
    history = list(sqlite_graph.get_state_history(cfg))
    for i, s in enumerate(history):
        print(f"   [{i}] next={s.next} turn={s.values.get('turn')} "
              f"ckpt={s.config['configurable']['checkpoint_id'][-12:]}")
```

预期输出：

```text
=== 第 1 次调用 ===
   turn = 1  log = ['A: turn -> 1', 'B: topic=E07 排查 turn=1']
=== 第 2 次调用（同 thread_id，状态被继承）===
   turn = 2  log = ['A: turn -> 1', 'B: topic=E07 排查 turn=1', 'A: turn -> 2', 'B: topic=加速时间调整 turn=2']
=== 换一个 thread_id（全新状态）===
   turn = 1  log = ['A: turn -> 1', 'B: topic=另一个工单 turn=1']

=== get_state ===
   values   : {'topic': '加速时间调整', 'turn': 2}
   next     : ()
   checkpoint_id: 1ef9a4c2-8b31-6d40-8 ...

=== get_state_history（从新到旧）===
   [0] next=() turn=2 ckpt=8b316d408003
   [1] next=('b',) turn=2 ckpt=8b2f1a903001
   [2] next=('a',) turn=1 ckpt=8b2d04812000
   [3] next=() turn=1 ckpt=8a91f2b0c004
   ...
```

三个关键观察：

1. **`thread_id` 就是会话 ID**。同一个 thread 的状态会被继承，`turn` 从 1 变成 2，`log` 因为是 `add` reducer 所以一直累积。
2. **`topic` 被覆盖了**（无 reducer 的默认行为），而 `log` 被追加了。这就是 reducer 的效果。
3. **每个超步产生一个 checkpoint**，`get_state_history` 能拿到全部历史，这是时间旅行的基础。

### 4.3 时间旅行：回到某个 checkpoint 重跑

场景：主管看到 Agent 第 2 步的查询改写不对，想"从改写那一步重来，但换个查询"。

```python
# huacheng/graph/time_travel.py
"""时间旅行：回到历史 checkpoint、修改状态、分叉重跑。"""
from __future__ import annotations

from huacheng.graph.basic_loop import builder as loop_builder
from langgraph.checkpoint.memory import MemorySaver

graph = loop_builder.compile(checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "tt-001"}}

# ---------- 1) 正常跑一遍 ----------
graph.invoke({"question": "咖啡机不出水怎么修", "retry_count": 0, "trace": []}, config=cfg)

# ---------- 2) 列出历史 ----------
history = list(graph.get_state_history(cfg))
print("=== 历史 checkpoint（从新到旧）===")
for i, s in enumerate(history):
    print(f"[{i}] next={str(s.next):>22} query={s.values.get('query')!r:>28} "
          f"retry={s.values.get('retry_count')}")

# ---------- 3) 挑一个点回溯 ----------
# 找到"即将执行 transform_query"的那个 checkpoint
target = next(s for s in history if s.next == ("transform_query",))
print(f"\n选中 checkpoint: {target.config['configurable']['checkpoint_id'][-12:]}")

# ---------- 4a) 原样重跑（invoke 第一个参数传 None 表示"从该 checkpoint 继续"）----------
print("\n=== 原样重跑 ===")
replay = graph.invoke(None, config=target.config)
for t in replay["trace"][-4:]:
    print("  ", t)

# ---------- 4b) 改状态后分叉重跑 ----------
print("\n=== 改状态后分叉重跑（人工指定更好的查询）===")
forked_cfg = graph.update_state(
    target.config,
    {"query": "YCT-132 E07 过流", "retry_count": 1},
    # as_node 指定"假装这个更新是由哪个节点产生的"，影响接下来走哪条边
    as_node="transform_query",
)
print("   分叉出的新 checkpoint:", forked_cfg["configurable"]["checkpoint_id"][-12:])
out = graph.invoke(None, config=forked_cfg)
for t in out["trace"][-4:]:
    print("  ", t)
print("   answer:", out["answer"][:44])
```

预期输出：

```text
=== 历史 checkpoint（从新到旧）===
[0] next=                     () query=          'YCT-132 E07 过流' retry=2
[1] next=           ('generate',) query=          'YCT-132 E07 过流' retry=2
[2] next=              ('grade',) query=          'YCT-132 E07 过流' retry=2
[3] next=           ('retrieve',) query=          'YCT-132 E07 过流' retry=2
[4] next=    ('transform_query',) query=      '变频器 过流 报警 处理' retry=1
[5] next=              ('grade',) query=      '变频器 过流 报警 处理' retry=1
...

选中 checkpoint: 8b2f1a903001

=== 原样重跑 ===
   transform_query #2 -> 'YCT-132 E07 过流'
   retrieve(q='YCT-132 E07 过流') -> 3 docs
   grade -> good (docs=3)
   generate

=== 改状态后分叉重跑（人工指定更好的查询）===
   分叉出的新 checkpoint: 9c4e7b210007
   retrieve(q='YCT-132 E07 过流') -> 3 docs
   grade -> good (docs=3)
   generate
   answer: 根据资料（[手册P87] E07=输出过流保护；[参数表]
```

**时间旅行的三个生产用途：**

1. **调试**：Agent 走错了，不用从头重跑（省 token、省时间），直接回到出错前一步改状态继续。
2. **人工纠偏**：主管在审批界面直接编辑 Agent 的中间结论，然后让它继续。
3. **A/B 对比**：同一个 checkpoint 分叉出多条路径，比较不同 prompt / 模型的效果——这是第 8 模块评测体系的一个实用技巧。

> **`update_state` 的 `as_node` 参数很关键**：它决定更新完之后图从哪条边继续走。不传的话，LangGraph 会尝试推断（通常是最后执行的节点），推断不出来会报错。

---

## 五、动手实战（三）：中断与人在回路（Human-in-the-loop）

华成机电的硬性规定：**Agent 可以自动建单、查库存、查物流，但"派单给工程师"和"退款"这两个动作必须主管点头。**

LangGraph 提供两代中断机制，都要配 checkpointer 才能用：

| 机制 | 写法 | 特点 | 适用 |
|---|---|---|---|
| 静态中断 | `compile(interrupt_before=["node"])` | 编译期确定在哪停 | 固定审批点，简单 |
| 动态中断 | 节点里调 `interrupt(payload)` | 运行期按条件决定停不停，还能带出数据给人看 | **推荐**，灵活 |

> 版本提示：`interrupt()` + `Command(resume=...)` 是 langgraph 0.2.57 之后的能力，导入路径为 `from langgraph.types import interrupt, Command`。更早的版本只有 `interrupt_before/after`。具体可用性以官方文档为准。

### 5.1 静态中断：`interrupt_before`

```python
# huacheng/graph/hitl_static.py
"""静态中断：编译期指定审批节点。"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class OpState(TypedDict):
    """工单操作流程的状态。"""

    ticket_id: str
    action: str                     # create / assign / refund / query
    engineer: str
    amount: float
    approved: bool
    result: str
    log: Annotated[list[str], operator.add]


def plan(state: OpState) -> dict:
    """规划要执行的动作（真实项目由 LLM 决定）。"""
    return {"log": [f"plan: 准备执行 {state['action']}，工单 {state['ticket_id']}"]}


def execute_assign(state: OpState) -> dict:
    """真正调用工单系统 API 派单。"""
    return {"result": f"已派单给 {state['engineer']}",
            "log": [f"execute: assign -> {state['engineer']}"]}


builder = StateGraph(OpState)
builder.add_node("plan", plan)
builder.add_node("execute_assign", execute_assign)
builder.add_edge(START, "plan")
builder.add_edge("plan", "execute_assign")
builder.add_edge("execute_assign", END)

# 关键：在 execute_assign 之前停下来
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_assign"],
    # interrupt_after=["plan"],     # 也可以在某节点之后停
)


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "op-001"}}
    inp = {"ticket_id": "WO-20250318-0031", "action": "assign",
           "engineer": "张工", "amount": 0.0, "approved": False, "log": []}

    print("=== 第一次 invoke：会在 execute_assign 前停住 ===")
    out = graph.invoke(inp, config=cfg)
    print("   返回:", out["log"])

    snap = graph.get_state(cfg)
    print("   next =", snap.next, "  <- 说明停在这里等着")
    print("   待执行动作:", snap.values["action"], "->", snap.values["engineer"])

    # ---- 人工审批界面：主管看到上面的信息，决定是否放行 ----
    approve = True
    if approve:
        print("\n=== 主管同意，继续执行（invoke 传 None）===")
        out = graph.invoke(None, config=cfg)
        print("   结果:", out["result"])
    else:
        print("\n=== 主管拒绝：改写状态后走别的路 ===")
        graph.update_state(cfg, {"result": "主管驳回，未派单", "approved": False})
```

预期输出：

```text
=== 第一次 invoke：会在 execute_assign 前停住 ===
   返回: ['plan: 准备执行 assign，工单 WO-20250318-0031']
   next = ('execute_assign',)   <- 说明停在这里等着
   待执行动作: assign -> 张工

=== 主管同意，继续执行（invoke 传 None）===
   结果: 已派单给 张工
```

### 5.2 动态中断：`interrupt()` + `Command(resume=...)`

静态中断的问题：它**无条件**在那个节点前停。但华成机电的规则是"退款金额 > 500 元才要审批"，小额自动过。这就需要动态中断。

```python
# huacheng/graph/hitl_dynamic.py
"""动态中断：运行期按业务规则决定是否需要人工审批，并把待审信息带给人看。"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

APPROVAL_THRESHOLD = 500.0          # 退款超过这个金额要审批


class OpState(TypedDict):
    """工单操作状态。"""

    ticket_id: str
    action: Literal["create", "assign", "refund", "query"]
    engineer: str
    amount: float
    result: str
    audit: Annotated[list[dict], operator.add]     # 审批留痕
    log: Annotated[list[str], operator.add]


def plan(state: OpState) -> dict:
    """规划阶段：真实项目是 LLM 输出的 tool_call。"""
    return {"log": [f"plan: {state['action']} on {state['ticket_id']}"]}


def risk_gate(state: OpState) -> dict:
    """风险闸门：高危动作调 interrupt() 挂起，等人工放行。"""
    action, amount = state["action"], state.get("amount", 0.0)

    need_approval = (action == "assign") or (action == "refund" and amount > APPROVAL_THRESHOLD)
    if not need_approval:
        return {"log": [f"risk_gate: {action} 低风险，自动放行"]}

    # interrupt 的参数会作为"待审信息"返回给调用方（前端展示给主管看）
    decision: dict[str, Any] = interrupt({
        "type": "approval_required",
        "ticket_id": state["ticket_id"],
        "action": action,
        "engineer": state.get("engineer"),
        "amount": amount,
        "reason": f"{action} 属于高危操作（阈值 {APPROVAL_THRESHOLD} 元）",
        "options": ["approve", "reject", "modify"],
    })
    # ↑ 执行到这里会抛出 GraphInterrupt，图挂起；恢复时从这行继续，decision 就是 resume 传的值

    if decision.get("decision") == "reject":
        return Command(
            goto="rejected",
            update={"audit": [decision], "log": [f"risk_gate: 被 {decision.get('by')} 驳回"]},
        )
    if decision.get("decision") == "modify":
        return Command(
            goto="execute",
            update={
                "engineer": decision.get("engineer", state.get("engineer")),
                "amount": float(decision.get("amount", amount)),
                "audit": [decision],
                "log": ["risk_gate: 人工修改后放行"],
            },
        )
    return {"audit": [decision], "log": [f"risk_gate: 被 {decision.get('by')} 批准"]}


def execute(state: OpState) -> dict:
    """真正调用外部系统。注意：必须做幂等，节点可能被重跑。"""
    a = state["action"]
    if a == "assign":
        res = f"已派单给 {state['engineer']}"
    elif a == "refund":
        res = f"已退款 {state['amount']:.2f} 元"
    else:
        res = f"已执行 {a}"
    return {"result": res, "log": [f"execute: {res}"]}


def rejected(state: OpState) -> dict:
    """被驳回的分支。"""
    return {"result": "操作被主管驳回，未执行", "log": ["rejected"]}


builder = StateGraph(OpState)
builder.add_node("plan", plan)
builder.add_node("risk_gate", risk_gate)
builder.add_node("execute", execute)
builder.add_node("rejected", rejected)

builder.add_edge(START, "plan")
builder.add_edge("plan", "risk_gate")
builder.add_edge("risk_gate", "execute")     # 默认路径；Command(goto=...) 可覆盖
builder.add_edge("execute", END)
builder.add_edge("rejected", END)

graph = builder.compile(checkpointer=MemorySaver())


def _run(title: str, inp: dict, thread: str, resume_payload: dict | None = None):
    """跑一个场景，演示挂起与恢复的完整流程。"""
    cfg = {"configurable": {"thread_id": thread}}
    print(f"\n===== {title} =====")

    result = graph.invoke(inp, config=cfg)
    # 挂起时，返回值里带 __interrupt__ 字段
    if "__interrupt__" in result:
        itr = result["__interrupt__"][0]
        print("  [挂起] 待审信息 =", itr.value)
        if resume_payload is None:
            print("  （无人审批，保持挂起）")
            return
        print("  [恢复] 主管决定 =", resume_payload)
        result = graph.invoke(Command(resume=resume_payload), config=cfg)

    print("  [完成] result =", result.get("result"))
    print("  [日志]", result.get("log"))


if __name__ == "__main__":
    base = {"ticket_id": "WO-0001", "engineer": "", "amount": 0.0,
            "result": "", "audit": [], "log": []}

    # 场景 1：低风险，不中断
    _run("场景1 小额退款 300 元（自动放行）",
         {**base, "ticket_id": "WO-0001", "action": "refund", "amount": 300.0}, "t1")

    # 场景 2：高风险，审批通过
    _run("场景2 退款 1200 元（需审批，通过）",
         {**base, "ticket_id": "WO-0002", "action": "refund", "amount": 1200.0}, "t2",
         resume_payload={"decision": "approve", "by": "李主管", "at": "2025-03-18 10:21"})

    # 场景 3：派单，主管驳回
    _run("场景3 派单给张工（需审批，驳回）",
         {**base, "ticket_id": "WO-0003", "action": "assign", "engineer": "张工"}, "t3",
         resume_payload={"decision": "reject", "by": "李主管", "note": "张工已排满"})

    # 场景 4：派单，主管改人后放行
    _run("场景4 派单（需审批，改派给王工）",
         {**base, "ticket_id": "WO-0004", "action": "assign", "engineer": "张工"}, "t4",
         resume_payload={"decision": "modify", "by": "李主管", "engineer": "王工"})

    # 场景 5：挂起后进程"重启"，第二天再恢复（演示持久性）
    _run("场景5 挂起不处理", {**base, "ticket_id": "WO-0005", "action": "assign",
                            "engineer": "赵工"}, "t5")
    print("\n  —— 假装过了一天，进程重启（同一个 checkpointer 实例即可演示）——")
    snap = graph.get_state({"configurable": {"thread_id": "t5"}})
    print("  挂起点 next =", snap.next)
    final = graph.invoke(Command(resume={"decision": "approve", "by": "王总"}),
                         config={"configurable": {"thread_id": "t5"}})
    print("  [次日恢复] result =", final.get("result"))
```

预期输出：

```text
===== 场景1 小额退款 300 元（自动放行） =====
  [完成] result = 已退款 300.00 元
  [日志] ['plan: refund on WO-0001', 'risk_gate: refund 低风险，自动放行', 'execute: 已退款 300.00 元']

===== 场景2 退款 1200 元（需审批，通过） =====
  [挂起] 待审信息 = {'type': 'approval_required', 'ticket_id': 'WO-0002', 'action': 'refund', 'engineer': '', 'amount': 1200.0, 'reason': 'refund 属于高危操作（阈值 500.0 元）', 'options': ['approve', 'reject', 'modify']}
  [恢复] 主管决定 = {'decision': 'approve', 'by': '李主管', 'at': '2025-03-18 10:21'}
  [完成] result = 已退款 1200.00 元
  [日志] ['plan: refund on WO-0002', 'risk_gate: 被 李主管 批准', 'execute: 已退款 1200.00 元']

===== 场景3 派单给张工（需审批，驳回） =====
  [挂起] 待审信息 = {'type': 'approval_required', 'ticket_id': 'WO-0003', 'action': 'assign', ...}
  [恢复] 主管决定 = {'decision': 'reject', 'by': '李主管', 'note': '张工已排满'}
  [完成] result = 操作被主管驳回，未执行

===== 场景4 派单（需审批，改派给王工） =====
  [挂起] 待审信息 = {...'engineer': '张工'...}
  [恢复] 主管决定 = {'decision': 'modify', 'by': '李主管', 'engineer': '王工'}
  [完成] result = 已派单给 王工

===== 场景5 挂起不处理 =====
  [挂起] 待审信息 = {...'engineer': '赵工'...}
  （无人审批，保持挂起）

  —— 假装过了一天，进程重启（同一个 checkpointer 实例即可演示）——
  挂起点 next = ('risk_gate',)
  [次日恢复] result = 已派单给 赵工
```

### 5.3 中断机制的关键细节

```mermaid
sequenceDiagram
    participant U as 前端
    participant G as LangGraph
    participant CP as Checkpointer(PG)
    participant S as 主管审批台

    U->>G: invoke({action:"refund", amount:1200}, thread_id=t2)
    G->>G: plan 节点执行
    G->>CP: 写 checkpoint
    G->>G: risk_gate 节点 -> interrupt(payload)
    G->>CP: 写 checkpoint（标记 pending interrupt）
    G-->>U: 返回 {"__interrupt__": [{value: payload}]}
    U-->>S: 推送待审卡片（可离线，可过很久）
    Note over G,CP: 进程可以重启，状态在库里
    S->>U: 点击「同意」
    U->>G: invoke(Command(resume={"decision":"approve"}), thread_id=t2)
    G->>CP: 读 checkpoint 恢复状态
    G->>G: 从 risk_gate 节点头部重跑，interrupt() 直接返回 resume 值
    G->>G: execute 节点执行
    G-->>U: 返回最终结果
```

**五个必须知道的细节：**

1. **恢复时节点是从头重跑的**，不是从 `interrupt()` 那一行继续。也就是说 `interrupt()` 之前的代码会**再执行一遍**。所以 `interrupt()` 前面不要放有副作用的代码（发短信、扣款、写数据库）。
2. 一个节点里可以调多次 `interrupt()`，恢复时按顺序匹配；但**不要在循环里动态改变 interrupt 的调用顺序**，会错位。
3. **必须配 checkpointer**，否则 `interrupt()` 会报错。
4. **必须传 `thread_id`**，否则不知道恢复哪个会话。
5. 挂起时 `invoke` 的返回值里有 `__interrupt__` 键（一个 `Interrupt` 对象列表，`.value` 是你传的 payload）。用 `graph.get_state(cfg).tasks` 也能查到 pending 的中断信息。

### 5.4 `Command` 的三种用法

`Command` 是 0.2.x 后期引入的统一控制原语，节点返回它可以同时做"更新状态"和"决定下一步去哪"：

```python
from langgraph.types import Command

# 1) 只跳转
return Command(goto="rejected")

# 2) 更新状态 + 跳转（省掉一条条件边，代码更内聚）
return Command(update={"engineer": "王工"}, goto="execute")

# 3) 跳到父图的节点（子图里用，见第七节）
return Command(update={...}, goto="supervisor", graph=Command.PARENT)

# 4) 作为 invoke 的输入：恢复中断
graph.invoke(Command(resume={"decision": "approve"}), config=cfg)

# 5) 恢复并同时改状态
graph.invoke(Command(resume={...}, update={"amount": 800.0}), config=cfg)
```

> 用 `Command(goto=...)` 时，如果目标节点不在该节点的静态出边里，需要在 `add_node` 时声明可达节点：`builder.add_node("risk_gate", risk_gate)` 之后仍要保证 `rejected`、`execute` 是图里已有的节点。部分版本要求用 `add_node(..., destinations=("execute", "rejected"))` 来让 `draw_mermaid` 画出虚线边——不影响运行，只影响可视化。具体以官方文档为准。

---

## 六、动手实战（四）：流式

### 6.1 四种 `stream_mode`

| mode | 每次 yield 什么 | 用途 |
|---|---|---|
| `"values"` | 每个超步后的**完整状态** | 调试、前端需要全量状态时 |
| `"updates"` | 每个节点返回的**增量** `{node_name: delta}` | 显示"正在执行 XX 步骤" |
| `"messages"` | `(AIMessageChunk, metadata)` 元组，**节点内 LLM 的 token** | 打字机效果 |
| `"custom"` | 节点里用 `StreamWriter` 主动写的任意数据 | 自定义进度条、中间产物 |
| `"debug"` | 详细的任务级事件 | 深度排错 |

**可以同时订阅多个**：`stream_mode=["updates", "messages"]`，此时 yield 的是 `(mode, chunk)` 二元组。

```python
# huacheng/graph/streaming_demo.py
"""LangGraph 的四种流式模式，以及怎么把它们变成 SSE 推给前端。"""
from __future__ import annotations

import asyncio
import json
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from huacheng.core.llm import get_chat_model

llm = get_chat_model("deepseek", temperature=0.0, streaming=True)


class ChatState(TypedDict):
    """带消息流的问答状态。"""

    question: str
    documents: list[str]
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]


def retrieve(state: ChatState, writer: StreamWriter) -> dict:
    """检索节点。writer 参数由框架按名字注入，用于 custom 流。"""
    writer({"stage": "retrieving", "progress": 0.1, "text": "正在检索知识库…"})
    docs = ["[手册P87] E07 = 输出侧过流保护",
            "[参数表] F0-09 加速时间，出厂值 3.0s",
            "[工单WO-20240912] 同型号故障最终定位为电机轴承卡死"]
    writer({"stage": "retrieved", "progress": 0.4,
            "text": f"命中 {len(docs)} 条资料", "sources": docs})
    return {"documents": docs, "log": [f"retrieve:{len(docs)}"]}


def generate(state: ChatState, writer: StreamWriter) -> dict:
    """生成节点。节点内部的 llm 调用会被 stream_mode='messages' 捕获。"""
    writer({"stage": "generating", "progress": 0.5, "text": "正在生成答案…"})
    ctx = "\n".join(f"[{i}] {d}" for i, d in enumerate(state["documents"], 1))
    msgs = [
        SystemMessage(content="你是华成机电售后助手，依据资料回答并标注引用编号，150 字以内。"),
        HumanMessage(content=f"【资料】\n{ctx}\n\n【问题】{state['question']}"),
    ]
    ai = llm.invoke(msgs)
    writer({"stage": "done", "progress": 1.0, "text": "完成"})
    return {"messages": [ai], "log": ["generate"]}


builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile(checkpointer=MemorySaver())


def demo_modes():
    """依次演示四种 stream_mode。"""
    cfg = {"configurable": {"thread_id": "st-1"}}
    inp = {"question": "E07 报警怎么处理", "documents": [], "messages": [], "log": []}

    print("=== mode=updates（每个节点的增量）===")
    for chunk in graph.stream(inp, config={"configurable": {"thread_id": "st-a"}},
                              stream_mode="updates"):
        for node, delta in chunk.items():
            print(f"   [{node}] keys={list(delta.keys())}")

    print("\n=== mode=values（每步完整状态）===")
    for chunk in graph.stream(inp, config={"configurable": {"thread_id": "st-b"}},
                              stream_mode="values"):
        print(f"   docs={len(chunk.get('documents', []))} msgs={len(chunk.get('messages', []))}")

    print("\n=== mode=messages（节点内 LLM token）===")
    print("   ", end="")
    for msg_chunk, meta in graph.stream(inp, config={"configurable": {"thread_id": "st-c"}},
                                        stream_mode="messages"):
        if msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
    print()

    print("\n=== mode=custom（节点主动写的进度）===")
    for item in graph.stream(inp, config={"configurable": {"thread_id": "st-d"}},
                             stream_mode="custom"):
        print(f"   {item['progress']:>4.0%} {item['stage']:<12} {item['text']}")

    print("\n=== 多模式组合 ===")
    for mode, chunk in graph.stream(inp, config={"configurable": {"thread_id": "st-e"}},
                                    stream_mode=["custom", "messages", "updates"]):
        if mode == "custom":
            print(f"   [custom] {chunk['stage']}")
        elif mode == "messages":
            pass    # token，太多不打印
        elif mode == "updates":
            print(f"   [updates] {list(chunk.keys())}")
```

预期输出：

```text
=== mode=updates（每个节点的增量）===
   [retrieve] keys=['documents', 'log']
   [generate] keys=['messages', 'log']

=== mode=values（每步完整状态）===
   docs=0 msgs=0
   docs=3 msgs=0
   docs=3 msgs=1

=== mode=messages（节点内 LLM token）===
   E07 为变频器输出侧过流保护 [1]。排查顺序：1) 断电测量电机三相绝缘…

=== mode=custom（节点主动写的进度）===
    10% retrieving    正在检索知识库…
    40% retrieved      命中 3 条资料
    50% generating     正在生成答案…
   100% done           完成

=== 多模式组合 ===
   [custom] retrieving
   [custom] retrieved
   [updates] ['retrieve']
   [custom] generating
   [custom] done
   [updates] ['generate']
```

### 6.2 把节点内 token 流到前端（完整 SSE 实现）

```python
# huacheng/graph/sse.py
"""把 LangGraph 的多模式流转成 SSE 帧，给前端做完整的进度 + 打字机体验。"""
from __future__ import annotations

import json
from typing import AsyncIterator

from huacheng.graph.streaming_demo import graph


def _sse(event: str, data: dict) -> str:
    """标准 SSE 帧：event 行 + data 行 + 空行。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_answer(question: str, thread_id: str) -> AsyncIterator[str]:
    """把图的执行过程转成 SSE 流。"""
    cfg = {"configurable": {"thread_id": thread_id}}
    inp = {"question": question, "documents": [], "messages": [], "log": []}

    yield _sse("start", {"thread_id": thread_id})

    async for mode, chunk in graph.astream(
        inp, config=cfg,
        stream_mode=["custom", "messages", "updates"],
        subgraphs=True,           # 子图内部的事件也一起吐出来（见第七节）
    ):
        if mode == "custom":
            yield _sse("progress", chunk)

        elif mode == "messages":
            msg_chunk, meta = chunk
            # meta 里有 langgraph_node，可以据此区分是哪个节点在说话
            if getattr(msg_chunk, "content", ""):
                yield _sse("token", {
                    "node": meta.get("langgraph_node"),
                    "text": msg_chunk.content,
                })

        elif mode == "updates":
            for node, delta in chunk.items():
                if node == "__interrupt__":
                    yield _sse("approval_required", {"payload": delta[0].value})
                else:
                    yield _sse("node_done", {"node": node, "keys": list(delta.keys())})

    final = graph.get_state(cfg).values
    yield _sse("done", {
        "answer": final["messages"][-1].content if final.get("messages") else "",
        "sources": final.get("documents", []),
    })
```

> 注意：开了 `subgraphs=True` 之后，`astream` yield 的元组会多一层命名空间前缀（`(namespace, mode, chunk)`）。上面代码为了简洁按两元组解包，实际使用时按你的 langgraph 版本确认返回结构，**以官方文档为准**；稳妥写法是先 `print(chunk)` 看一眼结构再解包。

### 6.3 `astream_events` 也能用

编译后的图是 Runnable，所以 4.1 章学的 `astream_events(version="v2")` 一样可用，而且能拿到更细的事件（工具调用、解析器等）。

**选型建议：**

| 需求 | 推荐 |
|---|---|
| 只要 token | `stream_mode="messages"` |
| 要节点级进度 | `stream_mode="updates"` |
| 要自定义业务进度（"正在查库存 3/7"） | `stream_mode="custom"` + `StreamWriter` |
| 要工具调用、检索细节等全部事件 | `astream_events(version="v2")` |
| 要最高性能（事件少） | `stream_mode`，`astream_events` 开销更大 |

---

## 七、动手实战（五）：子图、并行与 Send

### 7.1 子图 Subgraph：把 RAG 封装成可复用组件

```python
# huacheng/graph/subgraph.py
"""把一条 RAG 流程封装成子图，在父图里复用；演示两种状态映射方式。"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


# ============ 子图：RAG 检索与生成 ============
class RagState(TypedDict):
    """子图自己的状态。"""

    query: str
    docs: list[str]
    draft: str
    rag_log: Annotated[list[str], operator.add]


def sub_retrieve(state: RagState) -> dict:
    """子图节点：检索。"""
    docs = [f"[doc] 关于「{state['query']}」的资料 A", f"[doc] 资料 B"]
    return {"docs": docs, "rag_log": [f"sub.retrieve({state['query']})"]}


def sub_generate(state: RagState) -> dict:
    """子图节点：生成草稿。"""
    return {"draft": f"基于 {len(state['docs'])} 条资料的回答：…",
            "rag_log": ["sub.generate"]}


rag_builder = StateGraph(RagState)
rag_builder.add_node("retrieve", sub_retrieve)
rag_builder.add_node("generate", sub_generate)
rag_builder.add_edge(START, "retrieve")
rag_builder.add_edge("retrieve", "generate")
rag_builder.add_edge("generate", END)
rag_subgraph = rag_builder.compile()


# ============ 父图 ============
class MainState(TypedDict):
    """父图状态。注意 query/docs/draft 与子图同名 —— 这是共享 schema 的前提。"""

    question: str
    query: str
    docs: list[str]
    draft: str
    final: str
    rag_log: Annotated[list[str], operator.add]
    main_log: Annotated[list[str], operator.add]


def preprocess(state: MainState) -> dict:
    """父图节点：预处理，把 question 变成 query。"""
    return {"query": state["question"].strip(), "main_log": ["preprocess"]}


def postprocess(state: MainState) -> dict:
    """父图节点：给草稿加免责声明。"""
    return {"final": state["draft"] + "\n（本回答由 AI 生成，操作前请确认现场安全）",
            "main_log": ["postprocess"]}


main_builder = StateGraph(MainState)
main_builder.add_node("preprocess", preprocess)
# 方式一：状态 schema 有共享键，直接把编译好的子图当节点加
main_builder.add_node("rag", rag_subgraph)
main_builder.add_node("postprocess", postprocess)
main_builder.add_edge(START, "preprocess")
main_builder.add_edge("preprocess", "rag")
main_builder.add_edge("rag", "postprocess")
main_builder.add_edge("postprocess", END)
main_graph = main_builder.compile()


# ============ 方式二：状态 schema 不共享时，用包装函数做映射 ============
class OtherState(TypedDict):
    """一个字段名完全不同的父状态。"""

    user_input: str
    ai_output: str
    steps: Annotated[list[str], operator.add]


def call_rag_with_mapping(state: OtherState) -> dict:
    """包装函数：父状态 -> 子状态 -> 父状态，显式映射，解耦最彻底。"""
    sub_in = {"query": state["user_input"], "docs": [], "draft": "", "rag_log": []}
    sub_out = rag_subgraph.invoke(sub_in)
    return {"ai_output": sub_out["draft"], "steps": ["rag(mapped)"] + sub_out["rag_log"]}


other_builder = StateGraph(OtherState)
other_builder.add_node("rag", call_rag_with_mapping)
other_builder.add_edge(START, "rag")
other_builder.add_edge("rag", END)
other_graph = other_builder.compile()


if __name__ == "__main__":
    print("=== 方式一：共享 schema ===")
    out = main_graph.invoke({"question": "E07 怎么处理", "query": "", "docs": [],
                             "draft": "", "final": "", "rag_log": [], "main_log": []})
    print("   main_log =", out["main_log"])
    print("   rag_log  =", out["rag_log"])
    print("   final    =", out["final"][:40])

    print("\n=== 看到子图内部的执行过程（subgraphs=True）===")
    for ns, chunk in main_graph.stream(
        {"question": "E07 怎么处理", "query": "", "docs": [], "draft": "",
         "final": "", "rag_log": [], "main_log": []},
        stream_mode="updates", subgraphs=True,
    ):
        print(f"   ns={ns or '(root)'} -> {list(chunk.keys())}")

    print("\n=== 方式二：schema 映射 ===")
    out2 = other_graph.invoke({"user_input": "E07 怎么处理", "ai_output": "", "steps": []})
    print("   steps =", out2["steps"])

    print("\n=== 展开子图的 mermaid（xray=True）===")
    print(main_graph.get_graph(xray=True).draw_mermaid())
```

预期输出：

```text
=== 方式一：共享 schema ===
   main_log = ['preprocess', 'postprocess']
   rag_log  = ['sub.retrieve(E07 怎么处理)', 'sub.generate']
   final    = 基于 2 条资料的回答：…
（本回答由 AI 生成，操作前请确

=== 看到子图内部的执行过程（subgraphs=True）===
   ns=(root) -> ['preprocess']
   ns=('rag:9f2c...',) -> ['retrieve']
   ns=('rag:9f2c...',) -> ['generate']
   ns=(root) -> ['rag']
   ns=(root) -> ['postprocess']

=== 方式二：schema 映射 ===
   steps = ['rag(mapped)', 'sub.retrieve(E07 怎么处理)', 'sub.generate']
```

**两种方式怎么选：**

| | 直接嵌入（共享 schema） | 包装函数映射 |
|---|---|---|
| 写法 | `add_node("rag", subgraph)` | `add_node("rag", wrapper_fn)` |
| 状态传递 | 同名键自动流通 | 手工映射 |
| 耦合度 | 高（改子图 schema 影响父图） | 低 |
| 流式穿透 | 支持（`subgraphs=True`） | **不支持**（wrapper 里是 invoke） |
| 中断穿透 | 支持 | 不支持 |
| 推荐场景 | 同一个团队维护的紧密流程 | 跨团队复用、需要适配旧接口 |

> 华成机电的实践：**Agent 内部的 RAG 子流程用直接嵌入**（要流式和中断），**调用其他团队的图用包装函数**（接口稳定性优先）。

### 7.2 静态并行（fan-out / fan-in）

```python
# huacheng/graph/parallel.py
"""静态并行：一个节点连多条出边就是并行；多条入边自动汇聚。"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class MultiSourceState(TypedDict):
    """多路召回的状态。注意 sources 必须用 add reducer，否则并行写会冲突。"""

    question: str
    sources: Annotated[list[dict], operator.add]
    merged: list[dict]
    timing: Annotated[list[str], operator.add]


def _make_source(name: str, delay: float, n: int):
    """生成一个模拟检索节点。"""
    def _node(state: MultiSourceState) -> dict:
        t0 = time.perf_counter()
        time.sleep(delay)
        docs = [{"src": name, "text": f"{name}#{i}", "score": 0.9 - i * 0.05} for i in range(n)]
        return {"sources": docs, "timing": [f"{name}:{(time.perf_counter()-t0)*1000:.0f}ms"]}
    return _node


def merge(state: MultiSourceState) -> dict:
    """汇聚节点：三路结果都到齐了才会执行。"""
    ranked = sorted(state["sources"], key=lambda d: -d["score"])[:5]
    return {"merged": ranked, "timing": [f"merge:{len(state['sources'])}->{len(ranked)}"]}


b = StateGraph(MultiSourceState)
b.add_node("kb", _make_source("知识库", 0.8, 4))
b.add_node("ticket", _make_source("历史工单", 1.2, 3))
b.add_node("web", _make_source("官网公告", 0.5, 2))
b.add_node("merge", merge)

# fan-out：START 直接连三个节点，它们在同一个超步并行执行
b.add_edge(START, "kb")
b.add_edge(START, "ticket")
b.add_edge(START, "web")
# fan-in：三条边汇到 merge，merge 会等全部完成
b.add_edge("kb", "merge")
b.add_edge("ticket", "merge")
b.add_edge("web", "merge")
b.add_edge("merge", END)

par_graph = b.compile()

if __name__ == "__main__":
    t0 = time.perf_counter()
    out = par_graph.invoke({"question": "E07", "sources": [], "merged": [], "timing": []})
    print(f"总耗时 {(time.perf_counter()-t0)*1000:.0f}ms（串行需约 2500ms）")
    print("各路耗时:", out["timing"])
    print("融合后 top:", [d["text"] for d in out["merged"]])
    print("\nmermaid:")
    print(par_graph.get_graph().draw_mermaid())
```

预期输出：

```text
总耗时 1214ms（串行需约 2500ms）
各路耗时: ['官网公告:501ms', '知识库:802ms', '历史工单:1203ms', 'merge:9->5']
融合后 top: ['知识库#0', '历史工单#0', '官网公告#0', '知识库#1', '历史工单#1']
```

**并行的两条铁律：**

1. **并行分支要写的 State 键必须有 reducer**，否则多个节点同时写同一个键会抛 `InvalidUpdateError`。上面 `sources` 用了 `operator.add` 才没炸。
2. **并行分支是同一个超步**，它们看到的是**同一份**输入状态，互相看不到对方的结果。要串联就别并行。

### 7.3 动态 fan-out：`Send` API

静态并行的分支数在编译期就定死了。但真实需求是："把用户的复合问题拆成 N 个子问题，N 是运行期才知道的，每个子问题并行检索"。这时用 `Send`。

```python
# huacheng/graph/send_api.py
"""Send API：运行期动态决定并行多少路（map-reduce 模式）。"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class MapReduceState(TypedDict):
    """父状态：持有子问题列表和汇总结果。"""

    question: str
    sub_questions: list[str]
    sub_answers: Annotated[list[dict], operator.add]     # 必须有 reducer
    final_answer: str


class SubTaskState(TypedDict):
    """子任务的状态：Send 发过去的就是这个形状。"""

    sub_question: str
    index: int


def decompose(state: MapReduceState) -> dict:
    """问题分解：真实项目调 LLM，这里用规则模拟。"""
    q = state["question"]
    subs = [s.strip() for s in q.replace("；", ";").split(";") if s.strip()]
    if len(subs) == 1:
        subs = [f"{q} 的原因", f"{q} 的排查步骤", f"{q} 的预防措施"]
    return {"sub_questions": subs}


def fan_out(state: MapReduceState) -> list[Send]:
    """条件边返回 Send 列表：有几个子问题就并行几路。"""
    return [
        Send("solve_sub", {"sub_question": sq, "index": i})
        for i, sq in enumerate(state["sub_questions"])
    ]


def solve_sub(state: SubTaskState) -> dict:
    """子任务节点：注意它的 state 是 Send 传过来的 SubTaskState，不是父状态。"""
    time.sleep(0.4)
    return {"sub_answers": [{
        "index": state["index"],
        "q": state["sub_question"],
        "a": f"针对「{state['sub_question']}」的回答…",
    }]}


def reduce_answers(state: MapReduceState) -> dict:
    """汇总节点：所有 Send 出去的分支都完成后执行。"""
    ordered = sorted(state["sub_answers"], key=lambda d: d["index"])
    body = "\n".join(f"{i+1}. {d['q']}：{d['a']}" for i, d in enumerate(ordered))
    return {"final_answer": f"综合回答：\n{body}"}


b = StateGraph(MapReduceState)
b.add_node("decompose", decompose)
b.add_node("solve_sub", solve_sub)
b.add_node("reduce", reduce_answers)

b.add_edge(START, "decompose")
# 关键：条件边的函数返回 Send 列表，path_map 写上可能到达的节点
b.add_conditional_edges("decompose", fan_out, ["solve_sub"])
b.add_edge("solve_sub", "reduce")
b.add_edge("reduce", END)

mr_graph = b.compile()

if __name__ == "__main__":
    t0 = time.perf_counter()
    out = mr_graph.invoke({
        "question": "变频器报E07是什么原因；怎么排查；怎么预防；要换哪些备件",
        "sub_questions": [], "sub_answers": [], "final_answer": "",
    })
    print(f"并行 {len(out['sub_questions'])} 路，耗时 {(time.perf_counter()-t0)*1000:.0f}ms "
          f"（串行需约 {len(out['sub_questions'])*400}ms）")
    print(out["final_answer"])

    print("\n=== 观察并行执行 ===")
    for chunk in mr_graph.stream(
        {"question": "E07 怎么办", "sub_questions": [], "sub_answers": [], "final_answer": ""},
        stream_mode="updates",
    ):
        print("  ", {k: (f"{len(v['sub_answers'])} answers" if k == "solve_sub" else "...")
                     for k, v in chunk.items()})
```

预期输出：

```text
并行 4 路，耗时 412ms（串行需约 1600ms）
综合回答：
1. 变频器报E07是什么原因：针对「变频器报E07是什么原因」的回答…
2. 怎么排查：针对「怎么排查」的回答…
3. 怎么预防：针对「怎么预防」的回答…
4. 要换哪些备件：针对「要换哪些备件」的回答…

=== 观察并行执行 ===
   {'decompose': '...'}
   {'solve_sub': '1 answers'}
   {'solve_sub': '1 answers'}
   {'solve_sub': '1 answers'}
   {'reduce': '...'}
```

**`Send` 的四个要点：**

1. `Send(node_name, payload)` 里的 `payload` **直接成为目标节点的 state**，它不是父状态的一部分。
2. 目标节点返回的增量会按 **父状态的 reducer** 合并回去，所以 `sub_answers` 必须有 reducer。
3. `Send` 只能在**条件边的路由函数**里返回（返回一个 list）。
4. **并发数由 `max_concurrency` 控制**：`config={"max_concurrency": 5}`，否则 100 个子问题会同时打爆 API 配额。

---

## 八、错误处理

### 8.1 节点级重试 `RetryPolicy`

```python
# huacheng/graph/error_handling.py
"""节点重试、超时、异常兜底、递归上限。"""
from __future__ import annotations

import operator
import random
from typing import Annotated, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class RobustState(TypedDict):
    """带错误信息的状态。"""

    question: str
    docs: list[str]
    answer: str
    errors: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]


_attempt = {"n": 0}


def flaky_retrieve(state: RobustState) -> dict:
    """一个会随机失败的检索节点，用来验证重试。"""
    _attempt["n"] += 1
    if _attempt["n"] < 3:
        raise httpx.ConnectTimeout(f"模拟网络超时（第 {_attempt['n']} 次）")
    return {"docs": ["[手册] E07 = 输出过流"], "log": [f"retrieve 成功（第 {_attempt['n']} 次尝试）"]}


def generate(state: RobustState) -> dict:
    """生成节点：内部自己 try 住，失败也不让整个图崩。"""
    try:
        if not state["docs"]:
            raise ValueError("无可用资料")
        return {"answer": f"根据 {state['docs'][0]} 作答", "log": ["generate 成功"]}
    except Exception as e:
        return {"answer": "", "errors": [f"generate: {e}"], "log": ["generate 失败"]}


def fallback(state: RobustState) -> dict:
    """兜底节点：任何上游失败最终都走到这里，保证用户有响应。"""
    return {"answer": "系统繁忙，已为您转接人工工程师，工单号稍后推送。",
            "log": ["fallback 兜底"]}


def route_after_generate(state: RobustState) -> str:
    """有答案走结束，没答案走兜底。"""
    return "end" if state.get("answer") else "fallback"


b = StateGraph(RobustState)

# ---- 节点级重试策略 ----
b.add_node(
    "retrieve",
    flaky_retrieve,
    retry=RetryPolicy(
        max_attempts=4,
        initial_interval=0.5,        # 首次等待 0.5s
        backoff_factor=2.0,          # 指数退避：0.5, 1, 2, 4
        max_interval=8.0,
        jitter=True,                 # 加抖动，避免惊群
        retry_on=(httpx.ConnectTimeout, httpx.ReadTimeout, ConnectionError),
    ),
)
b.add_node("generate", generate)
b.add_node("fallback", fallback)

b.add_edge(START, "retrieve")
b.add_edge("retrieve", "generate")
b.add_conditional_edges("generate", route_after_generate,
                        {"end": END, "fallback": "fallback"})
b.add_edge("fallback", END)

robust_graph = b.compile()


if __name__ == "__main__":
    out = robust_graph.invoke(
        {"question": "E07", "docs": [], "answer": "", "errors": [], "log": []},
        config={"recursion_limit": 25},     # 防死循环：最多 25 个超步
    )
    print("log   :", out["log"])
    print("errors:", out["errors"])
    print("answer:", out["answer"])
```

预期输出：

```text
log   : ['retrieve 成功（第 3 次尝试）', 'generate 成功']
errors: []
answer: 根据 [手册] E07 = 输出过流 作答
```

> `add_node(..., retry=RetryPolicy(...))` 是 langgraph 0.2.x 的参数名；较新的版本改为 `retry_policy=`，且支持传列表。请按你安装的版本确认，**以官方文档为准**。

### 8.2 超时

LangGraph 本身没有"节点级超时"参数，三种做法：

```python
# 做法 1（推荐）：在节点内部用 asyncio.wait_for
import asyncio

async def timed_retrieve(state: RobustState) -> dict:
    """给外部调用加超时，超时就返回空结果而不是卡死整个图。"""
    try:
        docs = await asyncio.wait_for(retriever.ainvoke(state["question"]), timeout=3.0)
        return {"docs": docs}
    except asyncio.TimeoutError:
        return {"docs": [], "errors": ["retrieve 超时 3s，已降级"]}


# 做法 2：底层组件自带超时（ChatOpenAI(timeout=60)、httpx.Timeout）
# 做法 3：整图超时，在调用方包一层
result = await asyncio.wait_for(graph.ainvoke(inp, config=cfg), timeout=60.0)
```

### 8.3 递归上限 `recursion_limit`

**这是防止 Agent 死循环烧钱的最后一道保险。**

```python
from langgraph.errors import GraphRecursionError

try:
    out = graph.invoke(inp, config={"recursion_limit": 30, **cfg})
except GraphRecursionError:
    # 说明超步数超了，通常是条件边写错导致无限循环
    out = {"answer": "任务过于复杂，已转人工处理"}
    # 同时打点告警，这是需要人看的信号
```

| 图类型 | 建议 `recursion_limit` |
|---|---|
| 固定流程 RAG | 15 |
| CRAG（最多 3 次改写） | 25 |
| ReAct Agent（最多 10 轮工具） | 40 |
| 多智能体 Supervisor | 60~100 |

计算方法：`recursion_limit` 数的是**超步数**，不是节点数。一轮 ReAct（agent → tools → agent）算 2~3 个超步。

### 8.4 异常传播规则

| 情况 | 行为 |
|---|---|
| 节点抛异常，无 `RetryPolicy` | 整个图立即失败，异常向上抛给调用方 |
| 节点抛异常，有 `RetryPolicy` 且异常类型匹配 | 按策略重试，用尽后抛出 |
| 节点内部 try/except 后返回正常 dict | 图继续执行（**推荐的兜底模式**） |
| 并行分支中一个失败 | 整个超步失败，其他分支结果丢弃 |
| 有 checkpointer | 失败前的 checkpoint 已落盘，可以从失败点恢复 |

**华成机电的错误处理规范（写进 code review checklist）：**

1. 所有调用外部系统（LLM / 向量库 / 工单 API）的节点，**必须**配 `RetryPolicy` 且显式指定 `retry_on`；
2. 所有节点的业务逻辑异常**必须**在节点内 try 住，转成状态里的 `errors` 字段，不要让图崩；
3. 每张图**必须**有 fallback 节点和 `recursion_limit`；
4. 有副作用的节点（扣款、发短信、改工单状态）**必须**幂等（用 `thread_id + node + 业务主键` 做幂等键）。

---

## 九、压轴实战：用 LangGraph 重写 CRAG

第 3.3 章我们用 Python 的 while 循环实现了 CRAG（Corrective RAG，纠错式 RAG）。现在用 LangGraph 重写，你会看到代码量没怎么增加，但白拿了持久化、流式、可视化、时间旅行和人工介入。

### 9.1 CRAG 的流程设计

```mermaid
flowchart TD
    START([START]) --> RET["retrieve<br/>混合检索知识库"]
    RET --> GRD["grade_documents<br/>逐篇判定 relevant/irrelevant"]
    GRD --> DEC{"decide_next<br/>相关文档数 / 重试次数"}

    DEC -->|"相关数 >= 2"| GEN["generate<br/>带引用编号生成"]
    DEC -->|"相关数 < 2 且 retry < 3"| TRQ["transform_query<br/>LLM 改写查询"]
    DEC -->|"retry >= 3"| WEB["web_search<br/>兜底：搜官网/公告"]

    TRQ --> RET
    WEB --> GEN

    GEN --> HAL{"grade_generation<br/>幻觉检测 + 回答有用性"}
    HAL -->|"有幻觉，regen < 2"| GEN
    HAL -->|"答非所问，retry < 3"| TRQ
    HAL -->|"通过"| OK["finalize<br/>拼引用清单"]
    HAL -->|"重试用尽"| ESC["escalate<br/>转人工工单"]

    OK --> ENDN([END])
    ESC --> ENDN

    style GEN fill:#e8f4ff,stroke:#2b6cb0
    style ESC fill:#fff3cd,stroke:#856404
```

### 9.2 完整实现

```python
# huacheng/graph/crag.py
"""CRAG（纠错式 RAG）的 LangGraph 实现 —— 本章压轴实战。"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, StreamWriter
from pydantic import BaseModel, Field

from huacheng.core.llm import get_chat_model

llm = get_chat_model("deepseek", temperature=0.0, streaming=True)
llm_json = get_chat_model("deepseek", temperature=0.0, streaming=False)

MAX_QUERY_RETRY = 3          # 查询改写次数上限
MAX_REGEN = 2                # 因幻觉重新生成的次数上限
MIN_RELEVANT_DOCS = 2        # 认为"检索够用"的最低相关文档数


# ==================== 1. State ====================
class CragState(TypedDict):
    """CRAG 的完整状态。"""

    # 输入
    question: str
    # 过程
    query: str
    documents: list[Document]
    relevant_docs: list[Document]
    generation: str
    query_retry: int
    regen_count: int
    used_web: bool
    route: str
    # 输出
    answer: str
    citations: list[dict]
    # 轨迹（累积）
    trace: Annotated[list[str], operator.add]


# ==================== 2. 结构化评判器 ====================
class DocGrade(BaseModel):
    """单篇文档的相关性判定。"""

    relevant: bool = Field(description="该文档是否能帮助回答问题")
    reason: str = Field(description="判定理由，20 字以内")


class GenGrade(BaseModel):
    """生成结果的双重判定。"""

    grounded: bool = Field(description="回答是否完全基于给定资料，没有编造")
    useful: bool = Field(description="回答是否切实解决了用户的问题")
    reason: str = Field(description="判定理由，30 字以内")


doc_grader = (
    ChatPromptTemplate.from_messages([
        ("system", "你是华成机电知识库的文档相关性评判员。判断文档能否帮助回答用户问题。"
                   "宁可放过也不要错杀：只要有部分信息相关就算相关。"),
        ("human", "【文档】\n{document}\n\n【问题】\n{question}"),
    ])
    | llm_json.with_structured_output(DocGrade)
)

gen_grader = (
    ChatPromptTemplate.from_messages([
        ("system", "你是回答质量评判员。判断两点：\n"
                   "1) grounded：回答的每一个事实是否都能在资料中找到依据（有编造就是 false）；\n"
                   "2) useful：回答是否真正解决了用户问题（答非所问、过于笼统都是 false）。"),
        ("human", "【资料】\n{context}\n\n【问题】\n{question}\n\n【回答】\n{generation}"),
    ])
    | llm_json.with_structured_output(GenGrade)
)

query_rewriter = (
    ChatPromptTemplate.from_messages([
        ("system", "你是华成机电售后知识库的检索查询优化器。用户的原始问题检索效果不好，"
                   "请改写成更适合向量检索的查询：补全设备型号、用规范术语替换口语、"
                   "补充故障代码。只输出改写后的查询，不要解释。"),
        ("human", "原始问题：{question}\n已尝试过的查询：{tried}\n改写后的查询："),
    ])
    | llm_json
    | StrOutputParser()
)

answer_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是华成机电售后知识助手。严格依据资料回答。\n"
                   "1) 每个结论句末标注来源编号 [n]；\n"
                   "2) 资料没有的绝不编造；\n"
                   "3) 涉及带电操作，第一句必须是安全提示；\n"
                   "4) 步骤类用有序列表，总长不超过 350 字。"),
        ("human", "【资料】\n{context}\n\n【问题】\n{question}"),
    ])
    | llm
    | StrOutputParser()
)


def _format_ctx(docs: list[Document]) -> str:
    """把文档列表格式化成带编号的上下文。"""
    if not docs:
        return "（无资料）"
    return "\n\n".join(
        f"[{i}] （{d.metadata.get('source', '未知')}）\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


# ==================== 3. 节点 ====================
def retrieve(state: CragState, writer: StreamWriter) -> dict:
    """检索节点：用当前 query 走混合检索。"""
    from huacheng.rag.retriever_combos import production_retriever

    q = state.get("query") or state["question"]
    writer({"stage": "retrieve", "text": f"检索：{q}"})
    docs = production_retriever.invoke(q)
    return {"documents": docs, "query": q,
            "trace": [f"retrieve(q={q!r}) -> {len(docs)} docs"]}


def grade_documents(state: CragState, writer: StreamWriter) -> dict:
    """逐篇判定相关性。文档少时串行，多时可以换成 batch 并发。"""
    writer({"stage": "grade_docs", "text": f"评判 {len(state['documents'])} 篇文档相关性"})
    keep: list[Document] = []
    detail: list[str] = []
    # 用 batch 并发评判，比 for 循环快数倍
    inputs = [{"document": d.page_content[:1500], "question": state["question"]}
              for d in state["documents"]]
    grades = doc_grader.batch(inputs, config={"max_concurrency": 6})
    for d, g in zip(state["documents"], grades):
        if g.relevant:
            keep.append(d)
        detail.append(f"{'Y' if g.relevant else 'N'}:{d.metadata.get('source', '?')[:14]}")
    return {"relevant_docs": keep,
            "trace": [f"grade_documents -> {len(keep)}/{len(state['documents'])} 相关 | "
                      + " ".join(detail[:6])]}


def transform_query(state: CragState, writer: StreamWriter) -> dict:
    """查询改写：把已尝试过的查询告诉 LLM，避免改出同一个。"""
    n = state.get("query_retry", 0) + 1
    writer({"stage": "transform", "text": f"第 {n} 次改写查询"})
    tried = state.get("query", state["question"])
    new_q = query_rewriter.invoke({"question": state["question"], "tried": tried}).strip()
    return {"query": new_q, "query_retry": n,
            "trace": [f"transform_query #{n} -> {new_q!r}"]}


def web_search(state: CragState, writer: StreamWriter) -> dict:
    """兜底检索：查官网公告 / 外部知识。这里给出接口形状，实际接你们的搜索服务。"""
    writer({"stage": "web", "text": "知识库未覆盖，转查官网公告"})
    # 真实实现示例（择一）：
    #   from langchain_community.tools import DuckDuckGoSearchResults
    #   results = DuckDuckGoSearchResults().invoke(state["question"])
    # 或调用企业内部搜索 API。此处用占位数据保证示例可跑。
    web_docs = [Document(
        page_content=f"（官网公告检索结果占位）关于「{state['question']}」的公开资料。",
        metadata={"source": "华成机电官网公告", "url": "https://example.com"},
    )]
    merged = state.get("relevant_docs", []) + web_docs
    return {"relevant_docs": merged, "used_web": True,
            "trace": [f"web_search -> +{len(web_docs)} docs"]}


def generate(state: CragState, writer: StreamWriter) -> dict:
    """生成节点：内部的 LLM token 可以被 stream_mode='messages' 捕获。"""
    n = state.get("regen_count", 0)
    writer({"stage": "generate", "text": "生成答案" + (f"（第 {n+1} 次）" if n else "")})
    ctx = _format_ctx(state["relevant_docs"])
    text = answer_chain.invoke({"context": ctx, "question": state["question"]})
    return {"generation": text, "trace": [f"generate #{n+1}, len={len(text)}"]}


def grade_generation(state: CragState, writer: StreamWriter) -> dict:
    """幻觉检测 + 有用性检测，结果写进 route 供路由函数使用。"""
    writer({"stage": "grade_gen", "text": "检测幻觉与回答有用性"})
    g: GenGrade = gen_grader.invoke({
        "context": _format_ctx(state["relevant_docs"]),
        "question": state["question"],
        "generation": state["generation"],
    })
    if not g.grounded:
        route = "regenerate" if state.get("regen_count", 0) < MAX_REGEN else "escalate"
    elif not g.useful:
        route = "transform" if state.get("query_retry", 0) < MAX_QUERY_RETRY else "escalate"
    else:
        route = "finalize"
    return {"route": route,
            "regen_count": state.get("regen_count", 0) + (1 if route == "regenerate" else 0),
            "trace": [f"grade_generation grounded={g.grounded} useful={g.useful} "
                      f"-> {route} ({g.reason})"]}


def finalize(state: CragState, writer: StreamWriter) -> dict:
    """拼装最终答案与引用清单。"""
    writer({"stage": "finalize", "text": "整理引用"})
    cites = [{
        "index": i,
        "source": d.metadata.get("source", ""),
        "page": d.metadata.get("page"),
        "score": d.metadata.get("score"),
        "snippet": d.page_content[:120],
    } for i, d in enumerate(state["relevant_docs"], 1)]
    suffix = "\n\n（部分信息来自官网公告，请以最新手册为准）" if state.get("used_web") else ""
    return {"answer": state["generation"] + suffix, "citations": cites, "trace": ["finalize"]}


def escalate(state: CragState, writer: StreamWriter) -> dict:
    """转人工：AI 搞不定时的体面退出。"""
    writer({"stage": "escalate", "text": "转人工"})
    return {
        "answer": "抱歉，知识库中没有足够依据回答这个问题。已为您创建人工工单，"
                  "资深工程师将在 2 小时内联系您。",
        "citations": [],
        "trace": ["escalate(转人工)"],
    }


# ==================== 4. 路由函数 ====================
def decide_after_grade(state: CragState) -> Literal["generate", "transform", "web"]:
    """检索质量路由。"""
    n = len(state.get("relevant_docs", []))
    if n >= MIN_RELEVANT_DOCS:
        return "generate"
    if state.get("query_retry", 0) < MAX_QUERY_RETRY:
        return "transform"
    return "web"


def decide_after_generation(state: CragState) -> Literal["finalize", "regenerate",
                                                          "transform", "escalate"]:
    """生成质量路由，直接读上一节点写好的 route。"""
    return state["route"]


# ==================== 5. 建图 ====================
def build_crag_graph(checkpointer=None):
    """构建并编译 CRAG 图。"""
    b = StateGraph(CragState)

    net_retry = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True)

    b.add_node("retrieve", retrieve, retry=net_retry)
    b.add_node("grade_documents", grade_documents, retry=net_retry)
    b.add_node("transform_query", transform_query, retry=net_retry)
    b.add_node("web_search", web_search, retry=net_retry)
    b.add_node("generate", generate, retry=net_retry)
    b.add_node("grade_generation", grade_generation, retry=net_retry)
    b.add_node("finalize", finalize)
    b.add_node("escalate", escalate)

    b.add_edge(START, "retrieve")
    b.add_edge("retrieve", "grade_documents")
    b.add_conditional_edges("grade_documents", decide_after_grade, {
        "generate": "generate",
        "transform": "transform_query",
        "web": "web_search",
    })
    b.add_edge("transform_query", "retrieve")        # 环 1
    b.add_edge("web_search", "generate")
    b.add_edge("generate", "grade_generation")
    b.add_conditional_edges("grade_generation", decide_after_generation, {
        "finalize": "finalize",
        "regenerate": "generate",                    # 环 2
        "transform": "transform_query",              # 环 3
        "escalate": "escalate",
    })
    b.add_edge("finalize", END)
    b.add_edge("escalate", END)

    return b.compile(checkpointer=checkpointer)


crag_graph = build_crag_graph(checkpointer=MemorySaver())


# ==================== 6. 运行 ====================
def run(question: str, thread_id: str = "crag-demo") -> dict:
    """跑一次 CRAG，打印执行轨迹。"""
    init: CragState = {
        "question": question, "query": "", "documents": [], "relevant_docs": [],
        "generation": "", "query_retry": 0, "regen_count": 0, "used_web": False,
        "route": "", "answer": "", "citations": [], "trace": [],
    }
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}

    print(f"\n{'='*66}\n问题：{question}\n{'='*66}")
    for item in crag_graph.stream(init, config=cfg, stream_mode="custom"):
        print(f"  ▸ [{item['stage']:<14}] {item['text']}")

    final = crag_graph.get_state(cfg).values
    print("\n--- 执行轨迹 ---")
    for t in final["trace"]:
        print("  ", t)
    print("\n--- 答案 ---")
    print(final["answer"])
    if final["citations"]:
        print("\n--- 引用 ---")
        for c in final["citations"][:5]:
            print(f"  [{c['index']}] {c['source']}  score={c['score']}")
    return final


if __name__ == "__main__":
    print("=== 图结构 ===")
    print(crag_graph.get_graph().draw_mermaid())

    run("YCT-132 变频器启动 2 秒就报 E07，怎么处理？", "crag-1")
    run("我们车间的咖啡机不出水了怎么办", "crag-2")
```

### 9.3 执行轨迹（实测输出）

**场景 A：知识库能覆盖的问题**

```text
==================================================================
问题：YCT-132 变频器启动 2 秒就报 E07，怎么处理？
==================================================================
  ▸ [retrieve      ] 检索：YCT-132 变频器启动 2 秒就报 E07，怎么处理？
  ▸ [grade_docs    ] 评判 8 篇文档相关性
  ▸ [generate      ] 生成答案
  ▸ [grade_gen     ] 检测幻觉与回答有用性
  ▸ [finalize      ] 整理引用

--- 执行轨迹 ---
   retrieve(q='YCT-132 变频器启动 2 秒就报 E07，怎么处理？') -> 8 docs
   grade_documents -> 5/8 相关 | Y:YCT系列变频器使用手 Y:YCT系列变频器参数表 Y:故障代码速查表.md N:安装尺寸图.pdf Y:电机绝缘检测作业指 Y:WO-20240912-0118
   generate #1, len=287
   grade_generation grounded=True useful=True -> finalize (回答基于资料且步骤明确)
   finalize

--- 答案 ---
安全提示：排查前请断开主电源，等待直流母线电压放电至 36V 以下再操作。
1. 断电后用兆欧表测量电机三相绕组对地绝缘电阻，正常应大于 5MΩ [1][4]。
2. 检查变频器输出侧电缆有无破皮、受潮、端子松动 [1]。
3. 将加速时间参数 F0-09 由出厂值 3.0s 调整为 10s，重新试机 [2]。
4. 若仍报警，读取 F8-01~F8-05 故障记录中的跳闸电流，超过额定值 200% 说明为负载侧问题 [3]。
5. 同型号历史工单 WO-20240912-0118 的最终结论为电机轴承卡死导致堵转，可一并排查 [5]。

--- 引用 ---
  [1] YCT系列变频器使用手册_v3.2.pdf  score=0.8912
  [2] YCT系列变频器参数表_v3.2.pdf  score=0.8433
  [3] 故障代码速查表.md  score=0.8107
  [4] 电机绝缘检测作业指导书.docx  score=0.7788
  [5] WO-20240912-0118  score=0.7512
```

**场景 B：知识库完全不覆盖的问题（触发全部纠错路径）**

```text
==================================================================
问题：我们车间的咖啡机不出水了怎么办
==================================================================
  ▸ [retrieve      ] 检索：我们车间的咖啡机不出水了怎么办
  ▸ [grade_docs    ] 评判 8 篇文档相关性
  ▸ [transform     ] 第 1 次改写查询
  ▸ [retrieve      ] 检索：咖啡机 出水故障 排查
  ▸ [grade_docs    ] 评判 6 篇文档相关性
  ▸ [transform     ] 第 2 次改写查询
  ▸ [retrieve      ] 检索：设备 供水系统 不出水 管路堵塞 检修
  ▸ [grade_docs    ] 评判 8 篇文档相关性
  ▸ [transform     ] 第 3 次改写查询
  ▸ [retrieve      ] 检索：车间辅助设备 供水 故障 维修流程
  ▸ [grade_docs    ] 评判 7 篇文档相关性
  ▸ [web           ] 知识库未覆盖，转查官网公告
  ▸ [generate      ] 生成答案
  ▸ [grade_gen     ] 检测幻觉与回答有用性
  ▸ [escalate      ] 转人工

--- 执行轨迹 ---
   retrieve(q='我们车间的咖啡机不出水了怎么办') -> 8 docs
   grade_documents -> 0/8 相关 | N:YCT系列变频器使用手 N:安装尺寸图.pdf ...
   transform_query #1 -> '咖啡机 出水故障 排查'
   retrieve(q='咖啡机 出水故障 排查') -> 6 docs
   grade_documents -> 0/6 相关 | ...
   transform_query #2 -> '设备 供水系统 不出水 管路堵塞 检修'
   retrieve(q='设备 供水系统 不出水 管路堵塞 检修') -> 8 docs
   grade_documents -> 1/8 相关 | ...
   transform_query #3 -> '车间辅助设备 供水 故障 维修流程'
   retrieve(q='车间辅助设备 供水 故障 维修流程') -> 7 docs
   grade_documents -> 1/7 相关 | ...
   web_search -> +1 docs
   generate #1, len=132
   grade_generation grounded=False useful=False -> escalate (回答内容无法从资料中得到支持)
   escalate(转人工)

--- 答案 ---
抱歉，知识库中没有足够依据回答这个问题。已为您创建人工工单，资深工程师将在 2 小时内联系您。
```

**这份输出是本章最有价值的东西**：它证明了 CRAG 的纠错回路真的在工作——三次改写、一次兜底检索、一次幻觉拦截，最后体面地转人工，而不是硬编一段假答案给用户。

### 9.4 相比第 3.3 章的 while 循环版，多拿到了什么

| 能力 | while 循环版 | LangGraph 版 |
|---|---|---|
| 中间状态持久化 | 无，进程挂了从头来 | 每超步落盘，可恢复 |
| 流式进度 | 只能自己造 | `stream_mode="custom"` 四行搞定 |
| 可视化 | 靠注释和想象 | `draw_mermaid()` 一行导出 |
| 时间旅行调试 | 无 | 回到任意 checkpoint 重跑 |
| 人工介入 | 无 | 加一个 `interrupt()` 即可 |
| 节点级重试 | 手写 try/except | `RetryPolicy` 声明式 |
| 并发评判文档 | 手写 ThreadPool | `doc_grader.batch(...)` |
| 代码行数 | 约 180 行 | 约 260 行（多的是 State 和图定义） |

多写 80 行，换来上面 7 项能力。这笔账在生产环境是划算的。

---

## 十、预构建组件 `create_react_agent`

LangGraph 提供了一批开箱即用的图，最常用的是 ReAct Agent。

```python
# huacheng/graph/prebuilt_agent.py
"""用 create_react_agent 十行搭一个工单 Agent。"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from huacheng.core.llm import get_chat_model


@tool
def search_knowledge_base(query: str) -> str:
    """检索华成机电售后知识库，返回相关的手册片段与故障处理方法。"""
    from huacheng.rag.retriever_combos import production_retriever
    docs = production_retriever.invoke(query)
    return "\n\n".join(f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs, 1)) or "无结果"


@tool
def query_part_stock(part_no: str) -> str:
    """查询备件库存。part_no 为备件编码，例如 BRG-6205-2RS。"""
    fake = {"BRG-6205-2RS": 37, "FAN-120-24V": 4, "IGBT-FS50R": 0}
    n = fake.get(part_no.upper())
    return f"备件 {part_no} 库存 {n} 件" if n is not None else f"未找到备件 {part_no}"


@tool
def create_ticket(customer: str, device_model: str, description: str) -> str:
    """创建售后工单。返回工单号。注意：此操作有副作用，需保证幂等。"""
    return f"工单已创建：WO-20250318-{abs(hash(description)) % 10000:04d}"


agent = create_react_agent(
    model=get_chat_model("deepseek", temperature=0.0),
    tools=[search_knowledge_base, query_part_stock, create_ticket],
    prompt=("你是华成机电售后助手。先查知识库再回答；需要备件时查库存；"
            "用户明确要求报修时才创建工单。回答用中文，简洁清晰。"),
    checkpointer=MemorySaver(),
    # interrupt_before=["tools"],     # 想让每次工具调用都人工确认就打开
)


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "agent-1"}, "recursion_limit": 30}

    print("=== 图结构 ===")
    print(agent.get_graph().draw_mermaid())

    for q in ["YCT-132 报 E07 怎么办？顺便看看轴承 BRG-6205-2RS 有没有货",
              "那帮我报修吧，客户是苏州华东纺织，设备 YCT-132"]:
        print(f"\n>>> {q}")
        for chunk in agent.stream({"messages": [("user", q)]}, config=cfg,
                                  stream_mode="updates"):
            for node, delta in chunk.items():
                for m in delta.get("messages", []):
                    if getattr(m, "tool_calls", None):
                        for tc in m.tool_calls:
                            print(f"   [调用工具] {tc['name']}({tc['args']})")
                    elif m.type == "tool":
                        print(f"   [工具返回] {str(m.content)[:70]}")
                    elif m.content:
                        print(f"   [回答] {m.content[:150]}")
```

预期输出：

```text
=== 图结构 ===
graph TD;
    __start__([START]):::first --> agent;
    agent -. tools .-> tools;
    agent -. __end__ .-> __end__([END]):::last;
    tools --> agent;

>>> YCT-132 报 E07 怎么办？顺便看看轴承 BRG-6205-2RS 有没有货
   [调用工具] search_knowledge_base({'query': 'YCT-132 E07 故障处理'})
   [调用工具] query_part_stock({'part_no': 'BRG-6205-2RS'})
   [工具返回] [1] E07 = 输出侧过流保护。排查顺序：1) 测量电机三相绝缘…
   [工具返回] 备件 BRG-6205-2RS 库存 37 件
   [回答] E07 为输出侧过流保护。建议：1) 断电测电机三相绝缘；2) 检查输出电缆；3) 将加速时间 F0-09 调到 10s。轴承 BRG-6205-2RS 当前库存 37 件，可以直接安排更换。

>>> 那帮我报修吧，客户是苏州华东纺织，设备 YCT-132
   [调用工具] create_ticket({'customer': '苏州华东纺织', 'device_model': 'YCT-132', 'description': 'YCT-132 启动报 E07 输出过流，疑似轴承卡死'})
   [工具返回] 工单已创建：WO-20250318-4471
   [回答] 已为您创建工单 WO-20250318-4471，工程师会尽快联系。
```

### 什么时候用 `create_react_agent`，什么时候自己写图

| 情况 | 建议 |
|---|---|
| 单个 Agent，工具少于 10 个，逻辑就是"想 → 调工具 → 再想" | **用 `create_react_agent`**，省时间 |
| 演示、PoC、内部工具 | 用预构建 |
| 需要在工具调用前做风险闸门 / 人工审批 | 可以用 `interrupt_before=["tools"]`，但复杂规则要自己写图 |
| 需要固定的业务流程（先做 A 再做 B，不能乱来） | **自己写图**，ReAct 的自由度反而是风险 |
| 需要 Plan-and-Execute、Reflexion 等特定 CoT 范式 | **自己写图**（第 6 模块详讲） |
| 多智能体协同（Supervisor / Swarm） | **自己写图**（第 7 模块详讲） |
| 需要精细控制上下文（哪些消息进模型、怎么裁剪） | 自己写图 |

> 一个折中做法：先用 `create_react_agent` 快速验证业务价值，验证通过后再拆成自定义图上生产。华成机电就是这么干的——PoC 阶段 2 天出 demo，生产化用了 3 周。

---

## 十一、部署：把图暴露成 FastAPI 接口

```python
# huacheng/api/server.py
"""把 LangGraph 图暴露成 FastAPI 服务：普通接口 + SSE 流式 + 会话 + 审批。"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from huacheng.graph.crag import build_crag_graph

DB_URI = os.environ.get("PG_URI", "postgresql://postgres:postgres@127.0.0.1:5432/huacheng")

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建 checkpointer 和图，关闭时释放连接。"""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as cp:
        await cp.setup()                      # 首次运行建表，幂等
        _state["graph"] = build_crag_graph(checkpointer=cp)
        yield
    _state.clear()


app = FastAPI(title="华成机电售后助手", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    """问答请求。"""

    question: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(description="会话 ID，同一会话用同一个值")
    user_id: str = ""


class ResumeRequest(BaseModel):
    """审批恢复请求。"""

    thread_id: str
    decision: str = Field(description="approve / reject / modify")
    payload: dict = Field(default_factory=dict)


def _init_state(question: str) -> dict:
    """构造 CRAG 的初始状态。"""
    return {
        "question": question, "query": "", "documents": [], "relevant_docs": [],
        "generation": "", "query_retry": 0, "regen_count": 0, "used_web": False,
        "route": "", "answer": "", "citations": [], "trace": [],
    }


def _cfg(thread_id: str, user_id: str = "") -> dict:
    """构造运行配置。"""
    return {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 30,
        "max_concurrency": 6,
        "tags": ["prod", "crag"],
        "run_name": "华成CRAG",
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    """非流式问答：一次性返回答案与引用。"""
    graph = _state["graph"]
    try:
        out = await graph.ainvoke(_init_state(req.question), config=_cfg(req.thread_id, req.user_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    if "__interrupt__" in out:
        return {"status": "approval_required",
                "thread_id": req.thread_id,
                "payload": out["__interrupt__"][0].value}
    return {"status": "ok", "thread_id": req.thread_id,
            "answer": out.get("answer", ""), "citations": out.get("citations", []),
            "trace": out.get("trace", [])}


def _sse(event: str, data: dict) -> str:
    """SSE 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _event_stream(question: str, thread_id: str, user_id: str) -> AsyncIterator[str]:
    """把图执行过程转成 SSE。"""
    graph = _state["graph"]
    cfg = _cfg(thread_id, user_id)
    yield _sse("start", {"thread_id": thread_id})
    try:
        async for mode, chunk in graph.astream(
            _init_state(question), config=cfg, stream_mode=["custom", "messages", "updates"]
        ):
            if mode == "custom":
                yield _sse("progress", chunk)
            elif mode == "messages":
                msg, meta = chunk
                if getattr(msg, "content", ""):
                    yield _sse("token", {"text": msg.content,
                                         "node": meta.get("langgraph_node")})
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    yield _sse("approval_required",
                               {"payload": chunk["__interrupt__"][0].value})
                    return
        snap = await graph.aget_state(cfg)
        yield _sse("done", {"answer": snap.values.get("answer", ""),
                            "citations": snap.values.get("citations", [])})
    except Exception as e:
        yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest):
    """流式问答：SSE。"""
    return StreamingResponse(
        _event_stream(req.question, req.thread_id, req.user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/resume")
async def resume(req: ResumeRequest) -> dict:
    """主管审批后恢复执行。"""
    graph = _state["graph"]
    payload = {"decision": req.decision, **req.payload}
    out = await graph.ainvoke(Command(resume=payload), config=_cfg(req.thread_id))
    return {"status": "ok", "answer": out.get("answer", "")}


@app.get("/api/state/{thread_id}")
async def get_state(thread_id: str) -> dict:
    """查会话当前状态（调试 / 审批台用）。"""
    snap = await _state["graph"].aget_state(_cfg(thread_id))
    return {
        "next": list(snap.next),
        "values": {k: v for k, v in snap.values.items()
                   if k in ("question", "query", "answer", "query_retry", "trace")},
        "checkpoint_id": snap.config["configurable"].get("checkpoint_id"),
    }


@app.get("/api/history/{thread_id}")
async def get_history(thread_id: str, limit: int = 20) -> dict:
    """查历史 checkpoint（时间旅行用）。"""
    items = []
    async for s in _state["graph"].aget_state_history(_cfg(thread_id), limit=limit):
        items.append({
            "checkpoint_id": s.config["configurable"].get("checkpoint_id"),
            "next": list(s.next),
            "query": s.values.get("query"),
            "retry": s.values.get("query_retry"),
        })
    return {"thread_id": thread_id, "checkpoints": items}


@app.get("/healthz")
async def healthz() -> dict:
    """健康检查。"""
    return {"status": "ok", "graph_loaded": "graph" in _state}
```

启动与验证：

```bash
uv pip install "fastapi==0.115.6" "uvicorn[standard]==0.34.0" \
  "langgraph-checkpoint-postgres==2.0.9" "psycopg[binary,pool]==3.2.3"

uvicorn huacheng.api.server:app --host 0.0.0.0 --port 8080 --workers 4
```

```bash
# 1) 普通问答
curl -s -X POST http://127.0.0.1:8080/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"YCT-132 报 E07 怎么处理","thread_id":"t-1001","user_id":"agent_07"}' | jq

# 2) 流式
curl -N -X POST http://127.0.0.1:8080/api/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"YCT-132 报 E07 怎么处理","thread_id":"t-1002"}'

# 3) 查状态 / 历史
curl -s http://127.0.0.1:8080/api/state/t-1001 | jq
curl -s http://127.0.0.1:8080/api/history/t-1001 | jq
```

流式接口的预期输出：

```text
event: start
data: {"thread_id": "t-1002"}

event: progress
data: {"stage": "retrieve", "text": "检索：YCT-132 报 E07 怎么处理"}

event: progress
data: {"stage": "grade_docs", "text": "评判 8 篇文档相关性"}

event: progress
data: {"stage": "generate", "text": "生成答案"}

event: token
data: {"text": "安全", "node": "generate"}

event: token
data: {"text": "提示", "node": "generate"}

...

event: done
data: {"answer": "安全提示：排查前请断开主电源…", "citations": [...]}
```

**部署注意事项：**

1. **checkpointer 用 async 版本**（`AsyncPostgresSaver`），配 async 路由，否则会阻塞事件循环。
2. **`cp.setup()` 只需首次跑**，但它是幂等的，放在 lifespan 里没问题。生产建议做成独立的迁移脚本。
3. **`thread_id` 必须由调用方管理**，不要在服务端随机生成——否则前端没法续接会话。建议格式 `{tenant}:{user_id}:{conversation_id}`。
4. **Nginx 反代要关缓冲**：`proxy_buffering off;`，否则 SSE 会被攒起来一次性发出。
5. **多 worker 时状态是共享的**（因为在 Postgres），这是用 PostgresSaver 而不是 SqliteSaver 的关键理由。
6. 生产还应加：鉴权中间件、请求限流、超时中断、trace_id 透传、Langfuse 回调（见第 10 模块）。

---

## 十二、踩坑与排错

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | 节点返回的键在后续节点里读不到 | 该键没在 State 的 TypedDict 里声明，被静默丢弃 | 所有节点会写的键必须在 State 里声明；调试时用 `stream_mode="values"` 看完整状态 |
| 2 | `InvalidUpdateError: At key 'x': Can receive only one value per step` | 两个并行节点同时写同一个无 reducer 的键 | 给该键加 reducer（`operator.add` 或自定义），或改成串行 |
| 3 | `GraphRecursionError` | 条件边逻辑有问题导致无限循环；或 `recursion_limit` 设太小 | 打印 `stream_mode="updates"` 看循环在哪两个节点之间；确认计数器字段确实在递增（**常见错误：用了 `add` reducer 的计数器永远在累加，判断条件永远不成立**） |
| 4 | `interrupt()` 报错说需要 checkpointer | 编译时没传 checkpointer | `builder.compile(checkpointer=MemorySaver())` |
| 5 | 恢复中断后，`interrupt()` 前面的代码又执行了一遍（重复发了短信） | 恢复时节点从头重跑，这是设计如此 | 把副作用操作移到 `interrupt()` 之后；或用幂等键去重 |
| 6 | 多轮对话状态没继承 | 每次传了不同的 `thread_id`，或根本没传 | `config={"configurable": {"thread_id": ...}}` 必须传且稳定 |
| 7 | `SqliteSaver` 在 FastAPI 下报 `SQLite objects created in a thread can only be used in that same thread` | sqlite3 默认线程绑定 | `sqlite3.connect(path, check_same_thread=False)`；更好的做法是换 PostgresSaver |
| 8 | checkpoint 写入报 pickle / 序列化错误 | State 里放了不可序列化对象（数据库连接、模型实例、文件句柄） | State 只放纯数据；对象通过闭包或全局单例访问 |
| 9 | checkpoint 表体积暴涨，几天几 GB | State 里放了完整文档内容 / 向量 | 只存 doc_id，内容用时再查；定期清理过期 thread（写个定时任务按 `created_at` 删） |
| 10 | `stream_mode="messages"` 什么都不吐 | 节点里的模型没开 streaming；或用的是 `with_structured_output` 包过的模型 | `ChatOpenAI(streaming=True)`；结构化输出的调用天然不逐 token 流 |
| 11 | `StreamWriter` 注入失败，报参数错误 | 参数名必须是 `writer`，且要有类型注解 | `def node(state: S, writer: StreamWriter) -> dict` |
| 12 | 子图内部的进度在父图流里看不到 | 没开 `subgraphs=True` | `graph.astream(..., subgraphs=True)`，注意返回值多一层命名空间 |
| 13 | `Send` 出去的节点报 KeyError | 目标节点读的是父状态的键，但 Send 只给了 payload | Send 的 payload 要包含目标节点需要的全部字段 |
| 14 | 并行 `Send` 100 路打爆 API 配额 | 没限并发 | `config={"max_concurrency": 5}` |
| 15 | `update_state` 报"无法推断 as_node" | 图有多个可能的前驱节点 | 显式指定 `as_node="节点名"` |
| 16 | 时间旅行重跑后，副作用执行了两次 | 节点不幂等 | 有副作用的节点用 `thread_id + checkpoint_id + 业务键` 做幂等判断 |
| 17 | 升级 langgraph 后 `from langgraph.constants import Send` 报错 | `Send` / `Command` / `interrupt` 的导入路径在版本间有调整 | 统一从 `langgraph.types` 导；版本差异以官方文档为准 |
| 18 | `add_node(..., retry=...)` 报未知参数 | 新版本参数名改成 `retry_policy` | 按安装版本确认参数名 |
| 19 | `draw_mermaid()` 画出来的条件边没有标签 | `add_conditional_edges` 没传 `path_map` | 传 dict 形式的 path_map |
| 20 | Agent 每次都重新检索同样的内容，token 爆炸 | messages 无限增长，全量重发 | 用 `trim_messages` 或 `RemoveMessage` 裁剪历史；把长工具返回摘要后再入 state |

---

## 十三、生产级要点

### 13.1 成本控制

| 手段 | 说明 |
|---|---|
| 给循环设硬上限 | `MAX_QUERY_RETRY` / `MAX_REGEN` / `recursion_limit` 三重保险 |
| 评判器用小模型 | 文档相关性评判这类任务用 7B 本地模型就够，不必用在线大模型 |
| 评判器 batch 并发 | `doc_grader.batch(...)` 比 for 循环快，且可以只对 top-K 评判 |
| 状态里只存 doc_id | 减少 checkpoint 体积和序列化开销 |
| 在图上挂 token 统计回调 | 复用 4.1 章的 `HuachengObserver`，按 `thread_id` 归集成本 |

华成机电实测（2024Q4，日均 1200 次 CRAG 调用，实测环境：DeepSeek API + 单卡 A10 本地评判模型）：

- 把文档评判从 `deepseek-chat` 换成本地 `Qwen2.5-7B-Instruct`，**月成本从约 3400 元降到约 900 元**，评判一致率（与原模型比）约 93%；
- 把 `MAX_QUERY_RETRY` 从 5 降到 3，**平均 token 消耗降 22%**，答案质量（人工抽检 200 条）无显著变化。

### 13.2 延迟

CRAG 的延迟是"节点数 × 每节点耗时"的累加，最坏路径可能是最好路径的 5 倍。应对：

1. **首字节优先**：先把检索进度流给前端（`stream_mode="custom"`），用户不会觉得在干等；
2. **评判并发化**：`batch(config={"max_concurrency": 6})`；
3. **短路优化**：第一次检索就很好（相关文档 ≥ 4）时跳过评判，直接生成；
4. **设 SLA 超时**：整图 `asyncio.wait_for(..., timeout=25)`，超了直接返回"已转人工"。

### 13.3 并发与水平扩展

```mermaid
flowchart LR
    LB["Nginx<br/>proxy_buffering off"] --> A1["FastAPI 实例 1"]
    LB --> A2["FastAPI 实例 2"]
    LB --> A3["FastAPI 实例 3"]
    A1 --> PG[("PostgreSQL<br/>checkpoints")]
    A2 --> PG
    A3 --> PG
    A1 --> MV[("Milvus 2.4")]
    A2 --> MV
    A3 --> MV
    A1 --> LLM["DeepSeek API /<br/>本地 vLLM"]
    A2 --> LLM
    A3 --> LLM
```

- **状态在 Postgres，实例无状态**，可以随意扩缩容；
- 同一个 `thread_id` 的并发请求要做**幂等/加锁**（Postgres 有行锁，但业务层最好也加分布式锁，避免两个请求同时恢复同一个中断）；
- checkpoint 表要**定期归档**：`DELETE FROM checkpoints WHERE created_at < now() - interval '30 days'`（表名以实际 schema 为准）。

### 13.4 监控指标

| 指标 | 含义 | 告警线 |
|---|---|---|
| 每次调用的平均超步数 | Agent 绕圈程度 | > 12 说明流程设计有问题 |
| `escalate` 命中率 | 转人工比例 | > 15% 说明知识库缺口大 |
| `transform_query` 触发率 | 首次检索质量 | > 40% 说明检索或切分要优化 |
| 幻觉拦截率 | `grounded=False` 占比 | > 10% 说明 prompt 或模型要调 |
| `GraphRecursionError` 次数 | 死循环 | > 0 立即排查 |
| 中断挂起超过 24h 的 thread 数 | 审批积压 | > 20 通知运营 |
| checkpoint 表大小增长速率 | 状态膨胀 | 日增 > 1GB 排查 State 设计 |

### 13.5 降级预案

1. **CRAG 降级为普通 RAG**：开关一开，跳过评判和改写，直接检索 + 生成（延迟降 60%，质量降一点）；
2. **评判模型降级**：本地模型挂了，跳过评判默认全部相关；
3. **兜底检索降级**：外部搜索不可用直接转人工；
4. **整体降级**：图服务挂了，网关直接返回 FAQ 匹配结果。

---

## 十四、本章小结 + 自测题

### 14.1 要点回顾

1. **LCEL 是 DAG，LangGraph 是状态机**。画出来有回跳箭头、或者中间要停下来等人，就必须用图。
2. **三要素**：State（TypedDict + reducer，消息用 `add_messages`）、Node（函数，返回增量）、Edge（普通边 / 条件边 / START / END）。执行模型是 Pregel 超步，每个超步落一次 checkpoint。
3. **Checkpointer 是 LangGraph 最大的差异化能力**：多轮会话（`thread_id`）、断点续跑、时间旅行（`get_state_history` + 指定 `checkpoint_id` + `update_state` 分叉）全靠它。生产用 `PostgresSaver`。
4. **人在回路是一等公民**：`interrupt()` 挂起 + `Command(resume=...)` 恢复，进程可以重启、可以隔天审批。注意恢复时节点从头重跑，副作用要幂等。
5. **四种 `stream_mode`**：`values`（全量）、`updates`（增量）、`messages`（token）、`custom`（自定义进度）。可以组合订阅。
6. **`Send` 做动态 fan-out**，子图做流程复用（共享 schema 直接嵌入，跨团队用包装函数映射）。
7. **错误处理四件套**：节点级 `RetryPolicy`、节点内 try 转 errors 字段、fallback 兜底节点、`recursion_limit` 硬上限。
8. **CRAG 实战**证明了：多写约 80 行代码，换来持久化、流式、可视化、时间旅行、人工介入、节点重试六项能力。

### 14.2 自测题

**第 1 题**：下面这段 CRAG 的重试计数器为什么会导致死循环？怎么改？

```python
class S(TypedDict):
    retry: Annotated[int, operator.add]

def transform(state: S) -> dict:
    return {"retry": state["retry"] + 1}

def route(state: S) -> str:
    return "give_up" if state["retry"] >= 3 else "retry"
```

<details>
<summary>参考答案</summary>

问题在 reducer 用错了。`Annotated[int, operator.add]` 的语义是"新值**加到**旧值上"，而节点返回的是 `state["retry"] + 1`，所以：

- 第 1 轮：旧值 0，返回 1，合并后 = 0 + 1 = 1（看起来对）
- 第 2 轮：旧值 1，返回 2，合并后 = 1 + 2 = 3
- 第 3 轮：旧值 3，返回 4，合并后 = 3 + 4 = 7

计数器虽然会增长（这个例子里恰好还能停），但语义完全是错的；如果节点返回的是 `{"retry": 1}` 配 `operator.add`，那也对；两种写法混用才是灾难。更隐蔽的错误是反过来——用了累加 reducer 但节点返回固定值，或者用了覆盖但节点返回增量，都会让判断条件永远不成立。

两种正确写法，**任选其一，不要混**：

```python
# 写法 A：覆盖语义（推荐，最直观）
class S(TypedDict):
    retry: int                                  # 无 reducer

def transform(state: S) -> dict:
    return {"retry": state.get("retry", 0) + 1}   # 节点返回"新的完整值"

# 写法 B：累加语义
class S(TypedDict):
    retry: Annotated[int, operator.add]

def transform(state: S) -> dict:
    return {"retry": 1}                           # 节点只返回"增量"
```

排查技巧：`stream_mode="values"` 打印每步的完整状态，一眼就能看出计数器的实际值和你以为的不一样。
</details>

**第 2 题**：华成机电要求"派单操作必须主管审批，且审批可能隔天才处理，中间服务会重启"。用本章内容设计这个流程，并说明为什么某些方案不行。

<details>
<summary>参考答案</summary>

**可行方案：**

1. checkpointer 用 `AsyncPostgresSaver`（不能用 `MemorySaver`——重启就丢；不能用 `SqliteSaver`——多实例不共享）；
2. 在执行派单的节点**之前**加一个 `risk_gate` 节点，里面调 `interrupt({...待审信息...})`；
3. `invoke` 返回值里检测 `__interrupt__`，把 payload 推给审批台，接口立即返回（不要阻塞等待）；
4. 主管审批后，调用 `/api/resume` 接口，用**同一个 `thread_id`** 调 `graph.ainvoke(Command(resume={"decision": "approve", "by": "李主管"}), config=cfg)`；
5. 执行派单的节点做幂等：以 `ticket_id + action` 为幂等键，重复执行直接返回已有结果。

**不行的方案及原因：**

| 方案 | 为什么不行 |
|---|---|
| 在节点里 `while not approved: sleep(10)` 轮询 | 阻塞线程/协程，重启即丢，撑不住并发 |
| `MemorySaver` + 长连接 | 重启丢状态；多实例时请求可能落到没有状态的实例上 |
| 自己在 Redis 存一个"待审状态机" | 就是在重新实现 checkpointer，还要处理状态与代码版本不一致的问题 |
| 用 `interrupt_before=["execute"]` 静态中断 | 能用，但无法表达"退款 > 500 才审批"这类条件；且无法把待审详情带给前端 |

**额外注意**：`interrupt()` 之前不要放发短信、写日志到工单系统之类的副作用代码，因为恢复时这个节点会从头重跑一遍。
</details>

**第 3 题**：CRAG 图跑起来后报 `GraphRecursionError`，`recursion_limit=25`。给出三条具体的排查步骤。

<details>
<summary>参考答案</summary>

1. **先看循环发生在哪两个节点之间**：

```python
for chunk in graph.stream(inp, config=cfg, stream_mode="updates"):
    print(list(chunk.keys()))
# 如果反复打印 ['generate'] ['grade_generation'] ['generate'] ['grade_generation'] ...
# 说明幻觉检测一直不通过但 regen_count 没有正确递增
```

2. **打印关键计数器的实际值**，验证它真的在变（第 1 题的坑）：

```python
for chunk in graph.stream(inp, config=cfg, stream_mode="values"):
    print({k: chunk.get(k) for k in ("query_retry", "regen_count", "route")})
```

3. **检查路由函数的所有返回分支是否都在 `path_map` 里**，以及有没有"两个分支互相跳"的情况：

```python
print(graph.get_graph().draw_mermaid())    # 肉眼找环
# 常见错误：grade_generation 返回 "transform"，transform_query 又回到 retrieve，
# 但 query_retry 的上限判断写在了 decide_after_grade 里，
# decide_after_generation 没做上限判断 -> 无限循环
```

**根治手段**：给每条回跳路径都配独立的计数器和上限，并在路由函数的**每一个**返回分支前都做上限判断。再加一个"总超步数"的软上限（在 State 里放 `step_count`，超过就强制走 escalate），这样即使某个计数器写错了，也不会真的打到 `recursion_limit`。
</details>

---

**上一章** [4.1 LangChain 核心抽象与 LCEL](./01-LangChain核心抽象与LCEL.md) | **下一章** [4.3 LlamaIndex 与框架选型对比](./03-LlamaIndex与框架选型对比.md)
