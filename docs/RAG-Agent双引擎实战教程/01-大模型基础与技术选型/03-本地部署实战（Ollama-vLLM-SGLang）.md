# 第 1.3 章  本地部署实战（Ollama / vLLM / SGLang）

> **本章目标**：读完能做到 …
> 1. 说清 Ollama / vLLM / SGLang / llama.cpp 四者的定位边界，知道什么场景该用哪个；
> 2. 用 Ollama 在 10 分钟内跑起一个本地模型，并写出自定义的 Modelfile；
> 3. 用 vLLM 起一个生产级服务，逐参数解释每个启动开关调大调小的后果；
> 4. 显存不够时按决策树选对策（量化 / 降上下文 / 多卡 TP / CPU offload），而不是瞎试；
> 5. 写压测脚本测出 TTFT / TPOT / 吞吐 / P99，并读懂这些数字；
> 6. 在一张卡上同时部署 LLM + Embedding + Reranker，并算清显存账；
> 7. 用网关把在线 API 与本地服务统一成一个 OpenAI 兼容端点，带降级、限流、Key 轮转。
>
> **前置知识**：[1.1 大模型原理速览](./01-大模型原理速览.md)（KV Cache、Prefill/Decode）、[1.2 模型全景图与选型方法论](./02-模型全景图与选型方法论.md)、Docker 基础
>
> **预计用时**：阅读 50 分钟 / 动手 120 分钟

---

## 一、为什么需要它（问题出发）

### 1.1 华成机电的部署事故实录

华成机电的算法同学在本地用 Ollama 把 Qwen2.5-14B 跑通了，效果不错，于是直接把 Ollama 搬到了生产服务器上。上线第一天：

| 时间 | 现象 | 当时的判断 | 真实原因 |
|---|---|---|---|
| 09:00 | 服务正常，响应 1 秒内 | 「稳了」 | 此时只有 2 个人在用 |
| 09:40 | 响应变成 8 秒 | 「网络问题？」 | 并发到 12，Ollama 默认只并行处理 1~4 个请求，其余排队 |
| 10:15 | 部分请求直接超时 | 「模型不行，换大的？」 | 队列堆积，后端请求超时被丢弃 |
| 11:00 | 重启服务，第一个请求等了 90 秒 | 「服务器坏了？」 | Ollama 的 `keep_alive` 默认 5 分钟，模型被卸载了，重新加载 14B 模型要从磁盘读 9 GB |
| 14:00 | 换 4 个 Ollama 实例负载均衡，显存 OOM | 「显存不够，加卡？」 | 每个实例都独立加载一份权重，4 份权重 = 36 GB，一张 24G 卡当然爆 |

**这一整天的折腾，根因只有一句话：用开发调试工具做了生产部署。**

Ollama 的设计目标是「让个人在笔记本上方便地跑模型」，它不是没有并发能力，但它的架构（每个模型一个进程、默认低并行度、按需加载卸载）跟生产环境「高并发、稳定常驻、显存精细管理」的诉求是冲突的。

### 1.2 这一章要解决的四个问题

1. **选哪个部署方案**：四个工具的定位边界；
2. **参数怎么配**：vLLM 启动命令里那十几个参数，每一个调大调小会怎样；
3. **显存不够怎么办**：一棵可执行的决策树，不靠试错；
4. **怎么证明它能撑住**：压测方法与指标解读。

---

## 二、原理拆解

### 2.1 四种部署方式的定位对比

```mermaid
flowchart TD
    A["我要部署一个模型"] --> B{"什么场景?"}
    B -->|"笔记本上写 demo<br/>快速验证效果"| C["Ollama<br/>一条命令跑起来"]
    B -->|"生产服务<br/>高并发、稳定"| D{"负载特征?"}
    B -->|"没有 GPU<br/>CPU / 边缘设备"| E["llama.cpp<br/>GGUF 量化"]
    B -->|"手机 / 嵌入式"| F["llama.cpp / MLC-LLM"]

    D -->|"普通 RAG 问答<br/>吞吐优先"| G["vLLM<br/>生态最成熟"]
    D -->|"复杂结构化输出<br/>多轮 Agent<br/>前缀高度重复"| H["SGLang<br/>RadixAttention"]

    style C fill:#e8f4fd
    style G fill:#e1f5e1
    style H fill:#fff4e1
    style E fill:#f5e1f5
```

#### 2.1.1 详细对照表

| 维度 | **Ollama** | **vLLM** | **SGLang** | **llama.cpp** |
|---|---|---|---|---|
| **定位** | 开发调试、个人使用 | 生产高吞吐服务 | 复杂结构化/多轮场景 | CPU / 边缘 / 量化极致 |
| **底层引擎** | 基于 llama.cpp | 自研 PagedAttention | 自研 RadixAttention | 自研 GGML/GGUF |
| **安装难度** | ⭐ 一条命令 | ⭐⭐⭐ 需匹配 CUDA 版本 | ⭐⭐⭐ 同 vLLM | ⭐⭐ 需编译 |
| **并发能力** | 弱（默认 `OLLAMA_NUM_PARALLEL` 较低） | **强**（Continuous Batching） | **强** | 弱~中 |
| **显存效率** | 中（GGUF 量化省显存但调度粗） | **高**（PagedAttention，利用率 90%+） | **高**（+ 前缀树共享） | 高（量化极致） |
| **前缀复用** | 有限 | Prefix Caching（线性前缀） | **RadixAttention（树状前缀）** | 有限 |
| **结构化输出** | JSON format 参数 | guided decoding (xgrammar/outlines) | **原生 DSL + 高效约束解码** | grammar 文件 |
| **量化支持** | GGUF (Q4_K_M 等) | AWQ / GPTQ / FP8 / bitsandbytes | AWQ / GPTQ / FP8 | **GGUF 全系（Q2~Q8）** |
| **多卡** | 支持但不擅长 | **TP / PP 成熟** | TP 成熟 | 有限 |
| **CPU 推理** | 支持 | 实验性，不推荐 | 不推荐 | **最强** |
| **OpenAI 兼容 API** | ✅ `/v1/chat/completions` | ✅ 完整（含 tool_calls） | ✅ 完整 | ✅（llama-server） |
| **模型加载方式** | 自己的 registry（`ollama pull`） | HuggingFace / 本地目录 | HuggingFace / 本地目录 | 需转 GGUF 格式 |
| **典型启动时间** | 秒级（已缓存） | 1~5 分钟（要编译 kernel、预热） | 1~5 分钟 | 秒级 |
| **生产推荐度** | ❌ 不推荐 | ✅✅ **首选** | ✅✅ 特定场景更优 | 🟡 仅边缘场景 |

#### 2.1.2 一句话选型

> - **本地写 demo、给产品经理演示** → Ollama
> - **RAG 问答服务、Agent 后端、任何要扛并发的** → **vLLM**
> - **强结构化输出、Agent 多轮、system prompt 极长且高度复用** → **SGLang**
> - **没 GPU、或者要在工控机/边缘盒子上跑** → llama.cpp

**华成机电的最终选择**：开发环境 Ollama（每个工程师笔记本上跑 Qwen2.5-7B），生产环境 vLLM（两张 4090），未来 Agent 上量后评估切 SGLang。

### 2.2 Ollama：开发调试的最快路径

#### 2.2.1 安装与基本使用

```bash
# ===== 安装（Linux）=====
curl -fsSL https://ollama.com/install.sh | sh

# 验证安装
ollama --version
# 预期输出：ollama version is 0.x.x

# ===== 拉取并运行模型 =====
# Qwen2.5 系列（Ollama 官方库里有）
ollama pull qwen2.5:7b-instruct          # FP16 约 15GB / 默认 Q4_K_M 约 4.7GB
ollama pull qwen2.5:14b-instruct-q4_K_M  # 显式指定量化档位

# 交互式对话（最快的验证方式）
ollama run qwen2.5:7b-instruct
# >>> HC-V320 报 OC 故障怎么办？
# （模型开始输出）
# >>> /bye   退出

# ===== 查看与管理 =====
ollama list          # 列出本地已有模型
ollama ps            # 查看当前加载在显存里的模型
ollama rm qwen2.5:7b-instruct   # 删除
```

**Ollama 的量化命名规则**（很多人第一次用会懵）：

| 标签 | 含义 | Qwen2.5-7B 的大致大小 | 质量 |
|---|---|---|---|
| `q4_0` | 4bit，最简单的量化 | 4.4 GB | 一般 |
| `q4_K_M` | 4bit，K-quant 中等（**默认，推荐**） | 4.7 GB | 好 |
| `q5_K_M` | 5bit K-quant | 5.4 GB | 更好 |
| `q6_K` | 6bit | 6.3 GB | 接近 FP16 |
| `q8_0` | 8bit | 8.1 GB | 几乎无损 |
| `fp16` | 不量化 | 15.2 GB | 原始 |

> 经验：**开发调试用 `q4_K_M` 足够**，它是质量/大小的最佳平衡点。做效果对比实验时用 `q8_0` 或 `fp16`，避免把量化损失误判成模型能力问题。

#### 2.2.2 Modelfile：自定义你的模型

Modelfile 是 Ollama 的「Dockerfile」，用来固化 system prompt、采样参数、停止词。这样你的团队成员 `ollama run huacheng-qa` 就能得到完全一致的行为。

新建文件 `Modelfile.huacheng`：

```dockerfile
# 华成机电售后问答专用模型配置
# 用法：ollama create huacheng-qa -f Modelfile.huacheng

FROM qwen2.5:14b-instruct-q4_K_M

# ===== 系统提示词 =====
SYSTEM """你是华成机电的售后技术支持专家，服务对象是一线售后工程师和经销商。

回答规则：
1. 只依据【参考资料】中的内容回答，资料中没有的信息必须明确回答「参考资料中未提及，建议联系技术支持热线 400-xxx-xxxx」
2. 涉及具体参数（电流、电压、温度、时间、型号）时，必须与资料完全一致，禁止推算或估计
3. 涉及安全操作时，必须先给出安全提示（断电、验电、挂牌）
4. 回答结构：先给结论，再给依据，最后给操作步骤
5. 每条关键结论后标注来源，格式 [资料N]
6. 语言简洁，面向现场工程师，不要客套话
"""

# ===== 采样参数 =====
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
PARAMETER num_predict 1024

# ===== 停止词：防止模型自问自答 =====
PARAMETER stop "<|im_end|>"
PARAMETER stop "用户："
PARAMETER stop "问题："

# ===== 对话模板（Qwen2.5 的 ChatML 格式）=====
# 若不写 TEMPLATE，Ollama 会用基础模型自带的模板，通常够用。
# 显式写出来是为了让团队成员看得见「system/user/assistant 是怎么拼的」。
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

**Modelfile 参数速查**：

| 参数 | 作用 | 调大 | 调小 | 推荐值（RAG 场景） |
|---|---|---|---|---|
| `temperature` | 采样温度 | 更随机、更容易编 | 更确定、可能重复 | **0.0~0.2** |
| `top_p` | 核采样阈值 | 候选集大 | 候选集小 | 0.9（temperature=0 时无效） |
| `top_k` | 候选数上限 | 候选多 | 候选少 | 40 |
| `repeat_penalty` | 重复惩罚 | 抑制重复但可能不敢说型号 | 容易卡循环 | **1.05** |
| `num_ctx` | 上下文窗口 | **显存占用线性增长** | 长输入被截断 | 按业务峰值 × 1.5 |
| `num_predict` | 最大输出 token | 可能话痨、延迟高 | 回答被截断 | 512~1024 |
| `num_gpu` | 放到 GPU 的层数 | 显存占用高、速度快 | 显存省、速度慢 | 默认（自动） |
| `stop` | 停止词 | — | — | 按模板的结束符设 |

**创建并使用**：

```bash
# 创建自定义模型
ollama create huacheng-qa -f Modelfile.huacheng
# 预期输出：
# transferring model data
# ...
# success

# 验证
ollama run huacheng-qa "HC-V320 报 OC 故障怎么办？"

# 查看这个模型的完整配置（排查参数没生效时用）
ollama show huacheng-qa --modelfile
```

#### 2.2.3 OpenAI 兼容接口调用

Ollama 默认监听 `http://localhost:11434`，OpenAI 兼容路径在 `/v1`。

```python
"""通过 OpenAI SDK 调用 Ollama，含流式与非流式。

环境：Python 3.11
依赖：uv pip install openai==1.57.0
前置：ollama serve 已在运行，且已 ollama create huacheng-qa
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 不校验 key，但 SDK 要求非空
)

CONTEXT = """【资料1】HC-V320 系列变频器过流保护（OC）阈值为额定电流的 1.5 倍，动作时间 20ms。
【资料2】OC 常见原因：负载突变、加速时间过短（参数 F0.09）、电机绝缘下降、输出电缆过长。
【资料3】F0.09 加速时间出厂默认 10.0 秒，可调范围 0.1~3600 秒。"""


def ask_sync(question: str) -> str:
    """非流式调用，一次拿到完整答案。"""
    resp = client.chat.completions.create(
        model="huacheng-qa",
        messages=[{"role": "user", "content": f"{CONTEXT}\n\n问题：{question}"}],
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


def ask_stream(question: str) -> str:
    """流式调用，边生成边打印（生产环境 Web 端必用）。"""
    stream = client.chat.completions.create(
        model="huacheng-qa",
        messages=[{"role": "user", "content": f"{CONTEXT}\n\n问题：{question}"}],
        temperature=0.1, max_tokens=512, stream=True,
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            print(piece, end="", flush=True)
            parts.append(piece)
    print()
    return "".join(parts)


if __name__ == "__main__":
    q = "HC-V320 报 OC 故障，第一步该检查什么？"
    print("=== 非流式 ===")
    print(ask_sync(q))
    print("\n=== 流式 ===")
    ask_stream(q)
```

**Ollama 原生 API**（有些参数只有原生 API 支持，比如 `keep_alive`）：

```bash
# 原生 chat 接口
curl http://localhost:11434/api/chat -d '{
  "model": "huacheng-qa",
  "messages": [{"role": "user", "content": "HC-V320 过流阈值是多少？"}],
  "stream": false,
  "keep_alive": "30m",
  "options": {
    "temperature": 0.1,
    "num_ctx": 8192,
    "num_predict": 512
  }
}'

# 预加载模型到显存（服务启动后先跑一次，避免第一个用户等待）
curl http://localhost:11434/api/generate -d '{
  "model": "huacheng-qa",
  "keep_alive": -1
}'
# keep_alive: -1 表示永久驻留显存，0 表示用完立即卸载

# 查看当前加载的模型与剩余驻留时间
curl http://localhost:11434/api/ps
```

#### 2.2.4 并发限制与 keep_alive：Ollama 的两个关键环境变量

```bash
# ===== 关键环境变量（写进 systemd 或启动脚本）=====

# 1. 并行处理的请求数（默认值较低，生产要调）
export OLLAMA_NUM_PARALLEL=4

# 2. 同时加载在显存里的模型数量
export OLLAMA_MAX_LOADED_MODELS=2

# 3. 模型在显存中的驻留时间（默认 5m）
export OLLAMA_KEEP_ALIVE=-1       # -1 = 永久驻留，生产必设

# 4. 监听地址（默认只监听 127.0.0.1，要对外服务必须改）
export OLLAMA_HOST=0.0.0.0:11434

# 5. 请求队列长度上限
export OLLAMA_MAX_QUEUE=512

# 6. 模型存储路径（默认在 ~/.ollama，大模型建议放大盘）
export OLLAMA_MODELS=/data/ollama/models

# 重启生效
systemctl restart ollama
```

**systemd 配置示例**（`/etc/systemd/system/ollama.service.d/override.conf`）：

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_MODELS=/data/ollama/models"
Environment="OLLAMA_MAX_QUEUE=512"
```

**关于 `OLLAMA_NUM_PARALLEL` 的重要说明**：把它调大会**成倍增加 KV Cache 显存占用**（每个并行槽位都要独立的 KV Cache 空间）。显存公式（回顾 1.1 章）：

$$
M_{KV} = \text{OLLAMA\_NUM\_PARALLEL} \times \text{num\_ctx} \times m_{\text{token}}
$$

Qwen2.5-14B、`num_ctx=8192`、`NUM_PARALLEL=4` 时：$4 \times 8192 \times 192\text{KB} = 6$ GB。加上 q4 权重 8.6 GB，一共 14.6 GB，24G 卡刚好够。**调到 8 就会 OOM**。

**结论：Ollama 的并发能力受限于它「预分配固定槽位」的设计，这正是 vLLM 的 PagedAttention 要解决的问题。**

### 2.3 vLLM 生产部署

#### 2.3.1 安装

```bash
# ===== 方式一：uv（推荐，本书统一）=====
uv venv --python 3.11
source .venv/bin/activate
uv pip install vllm==0.6.4.post1

# ===== 方式二：pip 等价命令 =====
python3.11 -m venv .venv && source .venv/bin/activate
pip install vllm==0.6.4.post1

# ===== 验证 CUDA 环境 =====
nvidia-smi   # 确认驱动版本与显卡
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# 预期输出类似：2.5.1+cu121 True 12.1

python -c "import vllm; print(vllm.__version__)"
# 预期输出：0.6.4.post1
```

> **版本兼容性是 vLLM 最大的坑**。vLLM 对 PyTorch、CUDA、显卡驱动的版本有严格要求。**强烈建议用官方 Docker 镜像**（2.3.6 节），能省掉 80% 的环境问题。

#### 2.3.2 启动命令逐参数详解

先给一个华成机电生产环境的完整启动命令：

```bash
# ===== 华成机电生产环境：单卡 RTX 4090 跑 Qwen2.5-14B-AWQ =====
vllm serve /data/models/Qwen2.5-14B-Instruct-AWQ \
    --served-model-name Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --quantization awq_marlin \
    --dtype half \
    --swap-space 4 \
    --disable-log-requests \
    --api-key sk-huacheng-internal-2025 \
    --trust-remote-code
```

**逐参数详解**（这张表是本章最实用的内容之一）：

##### `--model` / 位置参数：模型路径

```bash
# 三种写法
vllm serve Qwen/Qwen2.5-14B-Instruct              # HuggingFace repo id，自动下载
vllm serve /data/models/Qwen2.5-14B-Instruct      # 本地目录（生产推荐）
vllm serve /data/models/Qwen2.5-14B-Instruct-AWQ  # 量化版本的本地目录
```

> **生产环境一定用本地路径**。HF 下载会受网络影响，而且模型仓库可能被更新（同一个 repo id 拿到不同权重），破坏可复现性。下载后记录 commit hash。

##### `--served-model-name`：对外暴露的模型名

```bash
--served-model-name Qwen2.5-14B-Instruct
```

- **作用**：客户端调用时 `model="Qwen2.5-14B-Instruct"`，与本地路径解耦
- **不设会怎样**：客户端必须写完整的本地路径（`model="/data/models/Qwen2.5-14B-Instruct-AWQ"`），换模型路径就要改所有客户端代码
- **最佳实践**：**用稳定的业务别名**（如 `huacheng-qa-v1`），这样换底层模型（7B→14B、FP16→AWQ）时客户端零改动

##### `--tensor-parallel-size` (TP)：张量并行度

```bash
--tensor-parallel-size 2   # 用 2 张卡
```

| 调整 | 后果 |
|---|---|
| **调大**（如 1→2） | ✅ 权重被切分到多卡，单卡显存占用减半，能跑更大模型<br/>✅ 计算并行，Prefill 更快<br/>❌ 引入卡间通信开销（NVLink 好，PCIe 差）<br/>❌ TP 必须整除注意力头数（Qwen2.5-14B 有 40 个头，TP 只能是 1/2/4/5/8） |
| **调小** | 显存压力大，但没有通信开销，单卡场景延迟最优 |

**重要约束**：

```text
num_attention_heads % tensor_parallel_size == 0
num_key_value_heads % tensor_parallel_size == 0  （GQA 模型，这个更严格）

Qwen2.5-14B: 40 个 Q 头、8 个 KV 头 → TP ∈ {1, 2, 4, 8}
Qwen2.5-7B:  28 个 Q 头、4 个 KV 头 → TP ∈ {1, 2, 4}
Qwen2.5-72B: 64 个 Q 头、8 个 KV 头 → TP ∈ {1, 2, 4, 8}
```

**PCIe vs NVLink 的差异**：TP 每一层都要做 AllReduce 通信。PCIe 4.0 x16 是 32 GB/s，NVLink 3.0 是 300 GB/s。**在 PCIe 机器上，TP=2 的收益可能被通信开销吃掉一半**。消费级卡（4090）没有 NVLink，TP 效率明显低于 A100/H800。

##### `--gpu-memory-utilization`：显存使用率上限

```bash
--gpu-memory-utilization 0.90   # 用 90% 的显存
```

| 调整 | 后果 |
|---|---|
| **调大**（0.90 → 0.95） | ✅ KV Cache 空间更大，并发更高<br/>❌ 显存碎片或峰值时 OOM 崩服务<br/>❌ 同卡上的其他进程（Embedding 模型、监控）会被挤死 |
| **调小**（0.90 → 0.75） | ✅ 更安全，能与其他服务共卡<br/>❌ KV Cache 空间小，并发能力下降 |

**这是个「按什么比例分蛋糕」的参数，不是「分多少 GB」**。vLLM 启动时会：
1. 计算 `总显存 × gpu_memory_utilization` = 预算
2. 减去权重占用
3. 减去激活值峰值（vLLM 会跑一次 profile 实测）
4. 剩下的**全部**分配给 KV Cache 池

**推荐值**：

| 场景 | 推荐值 | 理由 |
|---|---|---|
| 独占整卡 | **0.90~0.92** | 留 8%~10% 防峰值 |
| 与 Embedding/Rerank 共卡 | **0.60~0.70** | 给其他模型留空间（见 2.6 节） |
| 显存 < 16G 的小卡 | 0.85 | 绝对余量小，比例要保守 |
| 调试阶段 | 0.80 | 好排查，别在 OOM 上浪费时间 |

##### `--max-model-len`：最大上下文长度

```bash
--max-model-len 16384
```

| 调整 | 后果 |
|---|---|
| **调大**（16K → 32K） | ✅ 能处理更长输入<br/>❌❌ **单请求 KV Cache 占用翻倍，并发能力砍半**<br/>❌ 可能超出模型的有效上下文，效果反而下降 |
| **调小**（16K → 8K） | ✅ 并发能力翻倍<br/>❌ 超长请求被拒绝（报 400） |

**这是 vLLM 最重要的性能杠杆**。回顾 1.1 章的公式，单请求 KV Cache 与 `max_model_len` 成正比。

**怎么定这个值**（不要照抄模型的最大值）：

```python
"""统计线上真实请求的 token 长度分布，据此设置 max-model-len。

环境：Python 3.11
依赖：uv pip install transformers==4.46.3
输入：一个 jsonl 日志文件，每行含 {"prompt": "...", "completion": "..."}
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def analyze(log_path: str, max_output_tokens: int = 1024) -> None:
    """统计输入长度分布，给出 max-model-len 建议值。"""
    lengths = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        lengths.append(len(tok.encode(rec.get("prompt", ""), add_special_tokens=False)))

    if not lengths:
        print("日志里没有可解析的记录")
        return
    lengths.sort()
    n = len(lengths)

    def p(q: float) -> int:
        return lengths[min(int(n * q), n - 1)]

    print(f"样本数: {n}")
    for q in (0.5, 0.9, 0.95, 0.99, 0.999):
        print(f"  输入长度 P{q * 100:<5g} = {p(q):>6} token")
    print(f"  最大值      = {lengths[-1]:>6} token")

    # 建议值：P99.9 的输入 + 输出预留，再向上取到 2 的幂
    need = p(0.999) + max_output_tokens
    suggestion = 1 << (need - 1).bit_length()
    print(f"\n建议 --max-model-len = {suggestion}")
    print(f"  （P99.9 输入 {p(0.999)} + 输出预留 {max_output_tokens} = {need}，向上取整到 2 的幂）")
    over = sum(1 for x in lengths if x + max_output_tokens > suggestion)
    print(f"  按此设置，将有 {over} / {n} ({over / n * 100:.3f}%) 的历史请求被拒绝")


if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv) > 1 else "logs/requests.jsonl")
```

**预期输出**：

```text
样本数: 48273
  输入长度 P50    =   3182 token
  输入长度 P90    =   5940 token
  输入长度 P95    =   7104 token
  输入长度 P99    =   9836 token
  输入长度 P99.9  =  13521 token
  最大值          =  28410 token

建议 --max-model-len = 16384
  （P99.9 输入 13521 + 输出预留 1024 = 14545，向上取整到 2 的幂）
  按此设置，将有 41 / 48273 (0.085%) 的历史请求被拒绝
```

**处理被拒绝的 0.085%**：在网关层做上下文裁剪（砍掉最老的对话历史、减少注入的 chunk 数），而不是把 `max-model-len` 无脑调大——那会让所有请求的并发能力下降。

##### `--max-num-seqs`：最大并发序列数

```bash
--max-num-seqs 32
```

| 调整 | 后果 |
|---|---|
| **调大**（32 → 128） | ✅ 总吞吐提升（更大的 batch，权重读取被摊薄）<br/>❌ 单请求的 TPOT 变差（batch 内竞争）<br/>❌ 可能触发 KV Cache 抢占（preemption），被抢占的请求要重算 |
| **调小**（32 → 8） | ✅ 单请求延迟更稳<br/>❌ 总吞吐下降，高并发时排队 |

**这个参数与 `gpu-memory-utilization` 和 `max-model-len` 是三角关系**：

$$
\text{max\_num\_seqs} \times \text{max\_model\_len} \times m_{\text{token}} \le \text{KV Cache 池大小}
$$

如果设得太大，vLLM 不会直接报错，而是在运行时触发**抢占（preemption）**：显存不够时把某些请求的 KV Cache 换出（swap 到 CPU 内存）或直接丢弃重算。**日志里出现 `Sequence group ... is preempted` 就说明这个参数设过头了**。

**怎么定**：先用 1.1 章的公式算理论上限，然后取 70%~80%：

```text
Qwen2.5-14B-AWQ, 24G 卡, gpu-mem-util=0.90:
  可用显存 = 24 × 0.90 = 21.6 GB
  权重(AWQ) = 8.6 GB
  激活峰值 ≈ 1.5 GB
  KV Cache 池 = 21.6 - 8.6 - 1.5 = 11.5 GB

  max-model-len = 16384 时:
    单请求 KV = 16384 × 192KB = 3.0 GB
    理论上限 = 11.5 / 3.0 ≈ 3.8 → 只能 3 个并发！
```

**发现问题了吗？** 16K 上下文下，这张卡只能跑 3~4 个并发。这就是 vLLM 的价值——**PagedAttention 让实际占用按真实长度分配，而不是按 `max_model_len` 预留**。实际请求平均只有 3200 token，所以：

```text
  实际单请求 KV ≈ 3200 × 192KB = 0.59 GB
  实际可支持并发 ≈ 11.5 / 0.59 ≈ 19 个
```

**结论**：`max-num-seqs` 设 32 是合理的（留点余量让调度器发挥），真正的限制是 KV Cache 池总量，vLLM 会自己调度。**你要监控的是 `gpu_cache_usage_perc` 和 `num_preemptions`，而不是死抠这个参数**。

##### `--max-num-batched-tokens`：单次迭代处理的最大 token 数

```bash
--max-num-batched-tokens 8192
```

| 调整 | 后果 |
|---|---|
| **调大** | ✅ Prefill 吞吐高（一次能塞更多 token）<br/>❌ 单次迭代耗时长，Decode 中的请求要等，TPOT 抖动 |
| **调小** | ✅ Decode 更平滑，TPOT 稳定<br/>❌ 长 prompt 的 Prefill 要分多次，TTFT 变长 |

**这个参数只在开了 `--enable-chunked-prefill` 时才真正重要**。它决定了 Prefill 被切成多大的块。推荐值：`2048~8192`，长上下文场景用小值（让 Decode 更平滑），短上下文高吞吐场景用大值。

##### `--enable-prefix-caching`：前缀缓存

```bash
--enable-prefix-caching
```

| 调整 | 后果 |
|---|---|
| **开启** | ✅✅ **RAG 场景 TTFT 降 30%~70%**（system prompt 复用）<br/>❌ 占用一部分 KV Cache 池做缓存<br/>❌ 有哈希计算的微小开销 |
| **关闭** | 每次都重新 Prefill 全部 token |

**RAG 场景必开**。回顾 1.1 章 2.7.6 节的要点：**prompt 的固定部分必须在前**，否则命中率为 0。

验证是否生效：

```bash
# 查看 vLLM 的 metrics
curl http://localhost:8000/metrics | grep -E "prefix_cache|gpu_cache"

# 预期能看到类似：
# vllm:gpu_prefix_cache_hit_rate{model_name="..."} 0.612
# vllm:gpu_cache_usage_perc{model_name="..."} 0.734
```

##### `--enable-chunked-prefill`：分块预填充

```bash
--enable-chunked-prefill
```

**解决的问题**：一个 20K token 的长 prompt 进来，Prefill 要占用 GPU 好几百毫秒，期间所有正在 Decode 的请求都被阻塞，吐字卡住。

**做法**：把长 Prefill 切成 `max-num-batched-tokens` 大小的块，与 Decode 交错执行。

| 指标 | 不开 | 开 |
|---|---|---|
| 长请求的 TTFT | 略好 | 略差（有调度开销） |
| 短请求的 TTFT P99 | **很差**（被长请求阻塞） | **明显改善** |
| Decode 的 TPOT 抖动 | 大 | 小 |

**推荐：高并发 + 输入长度方差大的场景必开**（RAG 就是典型）。

##### `--quantization`：量化方式

```bash
--quantization awq_marlin    # AWQ 权重 + Marlin kernel（推荐）
--quantization gptq_marlin   # GPTQ 权重 + Marlin kernel
--quantization fp8           # FP8（需 Hopper 架构，H100/H800）
--quantization bitsandbytes  # 在线量化，慢，仅调试用
```

| 量化方式 | 显存 | 速度 | 精度损失 | 需要的模型 | 硬件要求 |
|---|---|---|---|---|---|
| 无（FP16） | 100% | 基准 | 0 | 原始权重 | 任意 |
| **AWQ (INT4)** | **~28%** | **Decode 快 2~3×** | 通常 1%~3% | 需预量化的 AWQ 模型 | Ampere+ |
| **GPTQ (INT4)** | ~28% | 快 2~3× | 通常 1%~4% | 需预量化的 GPTQ 模型 | Ampere+ |
| FP8 | ~50% | 快 1.5~2× | < 1% | 可在线量化 | **Hopper (H100/H800)** |
| bitsandbytes (NF4) | ~30% | **慢**（在线反量化） | 2%~5% | 原始权重 | 任意 |

**重要说明**：

1. **AWQ / GPTQ 需要预量化的模型**，不能对原始 FP16 权重用这两个参数。去 HuggingFace 找 `-AWQ` 或 `-GPTQ` 后缀的仓库，或者自己用 autoawq / gptqmodel 量化；
2. **`awq_marlin` 比 `awq` 快很多**。Marlin 是优化过的 INT4 kernel，vLLM 在支持的硬件上会自动选。显式指定可以避免它 fallback 到慢路径；
3. **量化会不会掉点，必须用你自己的金标集实测**。通用 benchmark 上掉 1% 不代表你的业务掉 1%。华成机电实测：14B-AWQ 相比 14B-FP16，工单金标集的字段准确率从 93.5% 降到 92.1%，**可接受**；
4. **KV Cache 也可以量化**：`--kv-cache-dtype fp8`，能让 KV Cache 占用减半，但长上下文下可能有精度损失，要实测。

##### 其他重要参数

```bash
--dtype half              # 权重的计算精度：auto/half(fp16)/bfloat16/float32
                          #   half: 大多数场景；bfloat16: 训练来的模型更稳；auto: 读 config

--swap-space 4            # CPU 内存交换空间（GB）。被抢占的请求 KV 可以换到这里
                          #   调大: 抢占时不用重算，但换入换出有 PCIe 开销
                          #   调小: 抢占直接丢弃重算

--max-seq-len-to-capture 8192   # CUDA Graph 捕获的最大序列长度
                          #   超过这个长度的请求走 eager 模式，慢一些
                          #   调大占更多显存，调小则长请求变慢

--enforce-eager           # 禁用 CUDA Graph。启动快、省显存，但推理慢 10%~20%
                          #   调试时用，生产别开

--disable-log-requests    # 不打印每个请求的详细日志。生产必开（否则日志爆炸）

--api-key sk-xxx          # 设置 API key，客户端必须带 Authorization: Bearer sk-xxx
                          # 内网也建议设，防止误调用

--trust-remote-code       # 允许执行模型仓库里的自定义代码
                          # ⚠ 只对可信来源的模型开启

--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
                          # 上下文外推。Qwen2.5 用 YaRN 可把 32K 扩到 128K
                          # ⚠ 会轻微降低短上下文的质量，按需开

--seed 42                 # 固定随机种子，提高可复现性

--tokenizer-mode auto     # auto / slow。某些模型的 fast tokenizer 有问题时用 slow

--load-format auto        # auto / safetensors / pt。生产建议 safetensors（安全且快）
```

#### 2.3.3 OpenAI 兼容 API 完整调用

```python
"""vLLM OpenAI 兼容接口的完整调用示例：普通、流式、function calling、结构化输出。

环境：Python 3.11
依赖：uv pip install openai==1.57.0 pydantic==2.10.3
前置：vLLM 服务已启动在 http://localhost:8000
"""
import json
import os

from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "sk-huacheng-internal-2025"),
)
MODEL = "Qwen2.5-14B-Instruct"


def demo_basic() -> None:
    """① 基础的非流式调用。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "你是华成机电售后专家，回答简洁。"},
            {"role": "user", "content": "HC-V320 的过流保护阈值通常是多少倍额定电流？"},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    print("答案:", resp.choices[0].message.content)
    print("用量:", resp.usage)


def demo_stream() -> None:
    """② 流式调用，含 usage 统计（vLLM 支持在流末尾返回 usage）。"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "列出变频器过流故障的 5 个排查步骤。"}],
        temperature=0.0, max_tokens=512, stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:  # 最后一个 chunk 带 usage
            print(f"\n\n[用量] 输入 {chunk.usage.prompt_tokens} / "
                  f"输出 {chunk.usage.completion_tokens}")


# ===== ③ Function Calling =====
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_device_spec",
            "description": "查询设备的技术规格参数",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "设备型号，如 HC-V320"},
                    "spec_name": {
                        "type": "string",
                        "description": "规格项名称",
                        "enum": ["过流阈值", "额定功率", "额定电压", "防护等级", "工作温度"],
                    },
                },
                "required": ["model", "spec_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_work_order",
            "description": "创建售后工单",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_model": {"type": "string"},
                    "fault_desc": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["紧急", "高", "中", "低"]},
                },
                "required": ["device_model", "fault_desc", "urgency"],
            },
        },
    },
]


def fake_tool_impl(name: str, args: dict) -> str:
    """模拟工具执行（真实项目里这里查数据库/调接口）。"""
    if name == "query_device_spec":
        table = {("HC-V320", "过流阈值"): "额定电流的 1.5 倍，动作时间 20ms",
                 ("HC-V320", "额定功率"): "15kW"}
        return table.get((args["model"], args["spec_name"]), "未查询到该规格")
    if name == "create_work_order":
        return json.dumps({"order_id": "WO-2025-08841", "status": "已创建"},
                          ensure_ascii=False)
    return "未知工具"


def demo_function_calling() -> None:
    """③ 完整的 function calling 回路：模型请求调用 → 执行 → 回传结果 → 最终答案。

    注意：vLLM 启动时需要加 --enable-auto-tool-choice --tool-call-parser hermes
         （Qwen2.5 用 hermes 解析器，具体以 vLLM 官方文档为准）
    """
    messages = [
        {"role": "system", "content": "你是华成机电售后助手，可以调用工具查规格和建工单。"},
        {"role": "user", "content": "帮我查一下 HC-V320 的过流阈值，"
                                    "然后给车间 3 号机建个紧急工单，故障是反复报 OC。"},
    ]

    for round_no in range(5):  # 最多 5 轮工具调用，防止死循环
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
            tool_choice="auto", temperature=0.0, max_tokens=512,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            print(f"\n[第{round_no + 1}轮] 最终答案:\n{msg.content}")
            return

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = fake_tool_impl(tc.function.name, args)
            print(f"[第{round_no + 1}轮] 调用 {tc.function.name}({args}) → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("达到最大工具调用轮次")


# ===== ④ 结构化输出（vLLM 的 guided decoding）=====
class WorkOrderExtraction(BaseModel):
    """工单抽取的目标 schema。"""
    device_model: str | None = Field(None, description="设备型号")
    device_location: str | None = Field(None, description="设备位置")
    fault_phenomenon: str | None = Field(None, description="故障现象")
    urgency: str = Field(..., description="紧急程度")
    suggested_handler: str | None = Field(None, description="建议处理人")


def demo_guided_json() -> None:
    """④ 用 guided_json 保证输出符合 schema（语法层面保证，不是 prompt 引导）。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
                   "从以下描述抽取工单信息：今天上午 3 号车间那台 HC-V320 突然报 OC，"
                   "整条线停了，急等着修，李工在现场。"}],
        temperature=0.0, max_tokens=300,
        extra_body={"guided_json": WorkOrderExtraction.model_json_schema()},
    )
    raw = resp.choices[0].message.content or ""
    print("原始输出:", raw)
    obj = WorkOrderExtraction.model_validate_json(raw)  # 一定能解析成功
    print("解析结果:", obj)


def demo_guided_choice() -> None:
    """⑤ 用 guided_choice 做分类，输出只可能是给定选项之一。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
                   "判断用户意图，只输出类别：\n用户说：我这台机器又不转了，赶紧派人来"}],
        temperature=0.0, max_tokens=16,
        extra_body={"guided_choice": ["查故障", "报修", "查配件", "查进度", "闲聊"]},
    )
    print("意图:", resp.choices[0].message.content)


if __name__ == "__main__":
    print("=== ① 基础调用 ===");        demo_basic()
    print("\n=== ② 流式调用 ===");      demo_stream()
    print("\n=== ③ Function Calling ==="); demo_function_calling()
    print("\n=== ④ 结构化输出 ===");     demo_guided_json()
    print("\n=== ⑤ 受限选择 ===");      demo_guided_choice()
```

**预期输出**（节选）：

```text
=== ① 基础调用 ===
答案: HC-V320 系列变频器的过流保护（OC）阈值通常为额定电流的 1.5 倍，保护动作时间 20ms。
用量: CompletionUsage(completion_tokens=38, prompt_tokens=52, total_tokens=90)

=== ③ Function Calling ===
[第1轮] 调用 query_device_spec({'model': 'HC-V320', 'spec_name': '过流阈值'}) → 额定电流的 1.5 倍，动作时间 20ms
[第2轮] 调用 create_work_order({'device_model': 'HC-V320', 'fault_desc': '反复报 OC 过流故障', 'urgency': '紧急'}) → {"order_id": "WO-2025-08841", "status": "已创建"}
[第3轮] 最终答案:
HC-V320 的过流保护阈值为额定电流的 1.5 倍，动作时间 20ms。
工单已创建，工单号 WO-2025-08841，紧急程度：紧急。

=== ④ 结构化输出 ===
原始输出: {"device_model": "HC-V320", "device_location": "3号车间", "fault_phenomenon": "突然报OC过流故障，整条产线停机", "urgency": "紧急", "suggested_handler": "李工"}
解析结果: device_model='HC-V320' device_location='3号车间' fault_phenomenon='突然报OC过流故障，整条产线停机' urgency='紧急' suggested_handler='李工'

=== ⑤ 受限选择 ===
意图: 报修
```

> **启用 function calling 需要额外的启动参数**：
> ```bash
> vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes
> ```
> 不同模型用不同的 parser（Qwen 系列用 `hermes`，Llama 3 用 `llama3_json`，Mistral 用 `mistral`）。**具体支持列表以 vLLM 官方文档为准**，版本间会变化。

#### 2.3.4 显存不够怎么办：决策树

```mermaid
flowchart TD
    A["启动报 OOM 或并发上不去"] --> B{"权重都塞不下?"}

    B -->|"是"| C{"有量化版模型吗?"}
    C -->|"有"| C1["① 换 AWQ/GPTQ INT4<br/>显存降到 28%<br/>代价: 精度掉 1%~3%"]
    C -->|"没有"| C2{"有多张卡?"}
    C2 -->|"有"| C3["② --tensor-parallel-size N<br/>权重切分到 N 卡<br/>代价: 通信开销, TP 需整除头数"]
    C2 -->|"没有"| C4["③ --cpu-offload-gb N<br/>部分权重放内存<br/>代价: 极慢, 仅应急"]
    C4 --> C5["④ 换更小的模型<br/>14B → 7B<br/>代价: 能力下降"]

    B -->|"否, 权重能塞下但并发低"| D{"KV Cache 池够大吗?<br/>看 gpu_cache_usage_perc"}
    D -->|"池本身就小"| D1["⑤ 调高 --gpu-memory-utilization<br/>0.85 → 0.92<br/>代价: OOM 风险"]
    D1 --> D2["⑥ 权重量化腾空间<br/>FP16 → AWQ 省 10~20GB"]
    D -->|"池够大但用满了"| E{"上下文设得太长?"}
    E -->|"是"| E1["⑦ 降 --max-model-len<br/>32K → 16K, 并发翻倍<br/>代价: 长请求被拒"]
    E -->|"否"| E2["⑧ --kv-cache-dtype fp8<br/>KV Cache 占用减半<br/>代价: 长上下文可能掉点"]
    E2 --> E3["⑨ RAG 层减少注入的 chunk<br/>Rerank 后只注入 Top-3~5<br/>代价: 召回可能下降"]

    style C1 fill:#e1f5e1
    style E1 fill:#e1f5e1
    style E3 fill:#e1f5e1
```

**按「性价比」排序的行动清单**：

| # | 手段 | 显存收益 | 代价 | 推荐度 |
|---|---|---|---|---|
| 1 | **降 `--max-model-len`** 到 P99.9 需求值 | KV Cache 池需求线性下降 | 少量长请求被拒（在网关层裁剪） | ⭐⭐⭐⭐⭐ |
| 2 | **权重量化到 AWQ INT4** | 权重省 70%+，且 Decode 快 2~3× | 精度掉 1%~3%，需实测 | ⭐⭐⭐⭐⭐ |
| 3 | **RAG 层减少注入 chunk** | 实际请求长度下降 → 并发提升 | 需验证召回率 | ⭐⭐⭐⭐⭐ |
| 4 | **调高 `gpu-memory-utilization`** | 直接扩大 KV 池 | OOM 风险，不能与其他进程共卡 | ⭐⭐⭐⭐ |
| 5 | **`--kv-cache-dtype fp8`** | KV Cache 减半 | 长上下文可能掉点 | ⭐⭐⭐ |
| 6 | **多卡 TP** | 权重按卡数切分 | 通信开销（PCIe 上尤其明显）、成本翻倍 | ⭐⭐⭐ |
| 7 | **`--enforce-eager`** | 省 1~3 GB（不建 CUDA Graph） | 推理慢 10%~20% | ⭐⭐ |
| 8 | **换更小模型** | 线性下降 | 能力下降 | ⭐⭐ |
| 9 | **`--cpu-offload-gb`** | 理论无上限 | **极慢**，PCIe 带宽是瓶颈 | ⭐ 仅应急 |

**实战演练：24G 卡跑 Qwen2.5-14B**

```bash
# ❌ 尝试 1：FP16 直接跑，必然 OOM
vllm serve /data/models/Qwen2.5-14B-Instruct --max-model-len 32768
# 报错: torch.OutOfMemoryError: CUDA out of memory.
#       权重 29.5GB > 显存 24GB，连模型都装不下

# ❌ 尝试 2：降上下文也没用，因为是权重装不下
vllm serve /data/models/Qwen2.5-14B-Instruct --max-model-len 4096
# 同样 OOM。max-model-len 只影响 KV Cache，不影响权重

# ✅ 方案 A：AWQ 量化（推荐）
vllm serve /data/models/Qwen2.5-14B-Instruct-AWQ \
    --quantization awq_marlin \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching
# 权重 8.6GB + KV池 11.5GB = 可用，实测约 19 并发（平均 3200 token 输入）

# ✅ 方案 B：两张卡 TP（如果有）
vllm serve /data/models/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching
# 每卡权重 14.8GB + KV池 ~6GB，上下文能开到 32K

# 🟡 方案 C：CPU offload（应急，不推荐生产）
vllm serve /data/models/Qwen2.5-14B-Instruct \
    --cpu-offload-gb 12 \
    --max-model-len 8192
# 能跑起来，但 TPOT 会从 10ms 涨到 100ms+，慢 10 倍
```

#### 2.3.5 启动后的自检清单

```bash
# ===== 1. 服务是否起来 =====
curl http://localhost:8000/health
# 预期输出：(空响应，HTTP 200)

# ===== 2. 模型列表 =====
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-huacheng-internal-2025" | python -m json.tool
# 预期输出：
# {
#   "object": "list",
#   "data": [{"id": "Qwen2.5-14B-Instruct", "object": "model", ...}]
# }

# ===== 3. 一次真实调用 =====
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-huacheng-internal-2025" \
  -d '{
    "model": "Qwen2.5-14B-Instruct",
    "messages": [{"role": "user", "content": "你好，一句话介绍变频器"}],
    "temperature": 0,
    "max_tokens": 64
  }' | python -m json.tool

# ===== 4. 关键指标 =====
curl -s http://localhost:8000/metrics | grep -E "^vllm:" | grep -vE "_bucket|_sum|_count"
# 关注这几个：
#   vllm:num_requests_running       正在处理的请求数
#   vllm:num_requests_waiting       排队中的请求数（>0 说明容量不足）
#   vllm:gpu_cache_usage_perc       KV Cache 使用率（>0.9 说明快满了）
#   vllm:gpu_prefix_cache_hit_rate  前缀缓存命中率（RAG 场景应 >0.3）
#   vllm:num_preemptions_total      抢占次数（>0 说明 max-num-seqs 设大了）

# ===== 5. 显存实际占用 =====
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv
# 预期输出：
# index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
# 0, NVIDIA GeForce RTX 4090, 21856 MiB, 24564 MiB, 3 %

# ===== 6. 启动日志里的关键行（一定要看）=====
# grep 这几个关键词确认配置生效：
#   "# GPU blocks:"           → KV Cache 块数，乘以 block_size(16) 就是能存的 token 总数
#   "Prefix caching is enabled"
#   "Chunked prefill is enabled"
#   "Using XFormers/FlashAttention backend"
```

**读懂 `# GPU blocks` 这一行**（vLLM 启动日志里最有价值的一行）：

```text
INFO ... # GPU blocks: 5836, # CPU blocks: 2048
```

- `5836 blocks × 16 token/block = 93376 token` 的 KV Cache 总容量
- 如果平均请求 3200 token，理论能同时容纳 `93376 / 3200 ≈ 29` 个请求
- **如果这个数字远低于预期，说明 `gpu-memory-utilization` 太低或权重占太多**

#### 2.3.6 Docker 部署 + docker-compose

**Dockerfile**（如果需要在官方镜像基础上加东西）：

```dockerfile
# 基于 vLLM 官方镜像，加上健康检查脚本和监控 agent
FROM vllm/vllm-openai:v0.6.4.post1

# 装一些排查工具
RUN pip install --no-cache-dir prometheus-client==0.21.0 httpx==0.28.0

# 健康检查脚本
COPY healthcheck.py /opt/healthcheck.py

# 官方镜像的 ENTRYPOINT 已经是 vllm serve，这里不覆盖
```

**docker-compose.yml**（华成机电生产配置）：

```yaml
# 文件：deploy/docker-compose.yml
# 用法：docker compose up -d
# 说明：两个模型服务 + 一个网关，共用一台 2×RTX4090 的机器

services:
  # ===== 主力模型：Qwen2.5-14B-AWQ，独占 GPU 0 =====
  vllm-main:
    image: vllm/vllm-openai:v0.6.4.post1
    container_name: vllm-main
    restart: unless-stopped
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HUB_OFFLINE=1                 # 禁止联网下载，强制用本地模型
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - NCCL_DEBUG=WARN
    volumes:
      - /data/models:/models:ro          # 模型目录只读挂载
      - /data/vllm-cache:/root/.cache/vllm
    ports:
      - "8000:8000"
    ipc: host                            # vLLM 多进程需要，不加会报共享内存不足
    ulimits:
      memlock: -1
      stack: 67108864
    command: >
      --model /models/Qwen2.5-14B-Instruct-AWQ
      --served-model-name Qwen2.5-14B-Instruct
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.90
      --max-model-len 16384
      --max-num-seqs 32
      --max-num-batched-tokens 8192
      --quantization awq_marlin
      --dtype half
      --enable-prefix-caching
      --enable-chunked-prefill
      --enable-auto-tool-choice
      --tool-call-parser hermes
      --disable-log-requests
      --api-key sk-huacheng-internal-2025
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s                 # 大模型加载慢，给 5 分钟启动期
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

  # ===== 小模型：Qwen2.5-1.5B，与 Embedding 共用 GPU 1 =====
  vllm-small:
    image: vllm/vllm-openai:v0.6.4.post1
    container_name: vllm-small
    restart: unless-stopped
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - HF_HUB_OFFLINE=1
    volumes:
      - /data/models:/models:ro
      - /data/vllm-cache:/root/.cache/vllm
    ports:
      - "8001:8000"
    ipc: host
    command: >
      --model /models/Qwen2.5-1.5B-Instruct
      --served-model-name Qwen2.5-1.5B-Instruct
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.35
      --max-model-len 8192
      --max-num-seqs 64
      --enable-prefix-caching
      --disable-log-requests
      --api-key sk-huacheng-internal-2025
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  # ===== Embedding + Rerank：TEI，与小模型共用 GPU 1 =====
  embedding:
    image: ghcr.io/huggingface/text-embeddings-inference:1.5
    container_name: tei-embedding
    restart: unless-stopped
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - /data/models:/models:ro
    ports:
      - "8080:80"
    command: --model-id /models/bge-m3 --max-client-batch-size 64 --max-batch-tokens 16384
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  reranker:
    image: ghcr.io/huggingface/text-embeddings-inference:1.5
    container_name: tei-reranker
    restart: unless-stopped
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - /data/models:/models:ro
    ports:
      - "8081:80"
    command: --model-id /models/bge-reranker-v2-m3 --max-client-batch-size 32
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]

  # ===== 统一网关（2.7 节会给完整实现）=====
  gateway:
    build: ../gateway
    container_name: llm-gateway
    restart: unless-stopped
    depends_on:
      vllm-main: {condition: service_healthy}
      vllm-small: {condition: service_started}
    environment:
      - MAIN_BASE_URL=http://vllm-main:8000/v1
      - SMALL_BASE_URL=http://vllm-small:8000/v1
      - INTERNAL_API_KEY=sk-huacheng-internal-2025
      - DEEPSEEK_API_KEYS=${DEEPSEEK_API_KEYS}
    ports:
      - "9000:9000"

networks:
  default:
    name: huacheng-llm
```

**常用运维命令**：

```bash
# 启动
docker compose -f deploy/docker-compose.yml up -d

# 看启动日志（等待模型加载，这一步最容易出问题）
docker compose logs -f vllm-main

# 只重启一个服务
docker compose restart vllm-main

# 查看资源占用
docker stats --no-stream
nvidia-smi

# 优雅停止（等待正在处理的请求完成）
docker compose stop -t 60 vllm-main

# 更新模型版本：改 command 里的路径，然后
docker compose up -d --force-recreate vllm-main
```

### 2.4 压测：证明它能撑住

#### 2.4.1 用 vLLM 自带的 benchmark

```bash
# vLLM 源码里有官方压测脚本，先下载
git clone --depth 1 --branch v0.6.4.post1 https://github.com/vllm-project/vllm.git /tmp/vllm-src

# 下载测试数据集（ShareGPT，真实对话分布）
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
     -O /tmp/sharegpt.json

# 跑压测
python /tmp/vllm-src/benchmarks/benchmark_serving.py \
    --backend openai-chat \
    --base-url http://localhost:8000 \
    --endpoint /v1/chat/completions \
    --model Qwen2.5-14B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /tmp/sharegpt.json \
    --num-prompts 300 \
    --request-rate 10 \
    --save-result
```

#### 2.4.2 自己写压测脚本（推荐，能用真实业务数据）

官方脚本用的是通用对话数据，跟你的 RAG 场景（长输入短输出）分布差很远。**自己写一个用真实 prompt 分布的压测脚本，结果才可信**。

```python
"""LLM 服务压测脚本：用真实业务分布的请求测 TTFT/TPOT/吞吐/P99。

环境：Python 3.11
依赖：uv pip install httpx==0.28.0 numpy==2.1.3
用法：
    python bench_llm.py --url http://localhost:8000/v1 \
        --model Qwen2.5-14B-Instruct --concurrency 16 --requests 200
设计要点：
  1. 用泊松到达（更接近真实流量）而不是固定间隔
  2. 区分 TTFT / TPOT / E2E 三个指标
  3. 统计 P50/P90/P95/P99，不只看平均值
  4. 记录失败率与失败原因
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

# ===== 模拟华成机电 RAG 场景的请求分布 =====
SYSTEM_PROMPT = ("你是华成机电的售后技术支持专家，只根据【参考资料】回答问题。"
                 "资料中没有的内容必须回答「参考资料中未提及」。"
                 "回答要简洁，面向现场工程师，涉及参数时必须与资料完全一致。") * 4

DOC_TEMPLATE = ("【资料{i}】HC-V{m} 系列变频器技术参数：额定功率 {p}kW，额定电压 380V，"
                "过流保护阈值为额定电流的 1.5 倍，动作时间 20ms。"
                "常见故障 OC（过流）的原因包括负载突变、加速时间设置过短（参数 F0.09）、"
                "电机绝缘电阻下降导致对地漏电、输出侧电缆过长引起容性电流。"
                "处理建议：先断电验电，用兆欧表测量电机绝缘电阻，标准值不低于 5 兆欧；"
                "检查 F0.09 加速时间参数，出厂默认 10.0 秒，重载场合建议调整到 20~30 秒；"
                "检查输出电缆长度，超过 50 米时应加装输出电抗器。")

QUESTIONS = [
    "HC-V320 报 OC 故障怎么排查？",
    "过流保护阈值是多少？动作时间多久？",
    "加速时间参数是哪个？默认值是多少？",
    "电机绝缘电阻的标准值是多少？",
    "输出电缆多长需要加电抗器？",
    "这台设备反复报 OC，换过 IGBT 了还是不行，还能查什么？",
]


@dataclass
class RequestResult:
    """单次请求的测量结果。"""
    ok: bool
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    output_tokens: int = 0
    input_chars: int = 0
    error: str = ""

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (self.e2e_ms - self.ttft_ms) / (self.output_tokens - 1)


def build_prompt(n_docs: int = 5) -> list[dict]:
    """构造一条模拟真实 RAG 场景的请求（长输入 + 短问题）。"""
    docs = "\n".join(
        DOC_TEMPLATE.format(i=i + 1, m=random.choice([320, 350, 380, 420]),
                            p=random.choice([7.5, 11, 15, 22, 30]))
        for i in range(n_docs))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{docs}\n\n问题：{random.choice(QUESTIONS)}"},
    ]


async def one_request(client: httpx.AsyncClient, url: str, model: str,
                      api_key: str, max_tokens: int) -> RequestResult:
    """发起一次流式请求并测量各项延迟。"""
    messages = build_prompt()
    payload = {"model": model, "messages": messages, "stream": True,
               "temperature": 0.0, "max_tokens": max_tokens,
               "stream_options": {"include_usage": True}}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    input_chars = sum(len(m["content"]) for m in messages)

    t0 = time.perf_counter()
    t_first = None
    n_out = 0
    try:
        async with client.stream("POST", f"{url}/chat/completions",
                                 json=payload, headers=headers, timeout=180.0) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RequestResult(False, error=f"HTTP {resp.status_code}: {body[:200]!r}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_out += 1
                if obj.get("usage"):
                    n_out = obj["usage"].get("completion_tokens", n_out)
    except Exception as exc:
        return RequestResult(False, error=f"{type(exc).__name__}: {exc}")

    t_end = time.perf_counter()
    if t_first is None:
        return RequestResult(False, error="没有收到任何输出 token")
    return RequestResult(True, (t_first - t0) * 1000, (t_end - t0) * 1000,
                         n_out, input_chars)


async def worker(name: int, client: httpx.AsyncClient, args, queue: asyncio.Queue,
                 results: list[RequestResult]) -> None:
    """消费队列中的任务槽，持续发请求。"""
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        r = await one_request(client, args.url, args.model, args.api_key, args.max_tokens)
        results.append(r)
        queue.task_done()


def percentile(values: list[float], q: float) -> float:
    """计算分位数。"""
    return float(np.percentile(values, q)) if values else 0.0


def report(results: list[RequestResult], wall_time_s: float, concurrency: int) -> None:
    """输出压测报告表格。"""
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    if not ok:
        print("全部失败！错误样例：")
        for r in failed[:5]:
            print("  ", r.error)
        return

    ttfts = [r.ttft_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    e2es = [r.e2e_ms for r in ok]
    total_out = sum(r.output_tokens for r in ok)

    print("\n" + "=" * 78)
    print(f"压测报告  并发={concurrency}  总请求={len(results)}  "
          f"成功={len(ok)}  失败={len(failed)}  耗时={wall_time_s:.1f}s")
    print("=" * 78)
    print(f"{'指标':<22}{'P50':>10}{'P90':>10}{'P95':>10}{'P99':>10}{'均值':>10}")
    print("-" * 78)
    for label, vals in [("TTFT (ms)", ttfts), ("TPOT (ms)", tpots), ("E2E (ms)", e2es)]:
        print(f"{label:<22}{percentile(vals, 50):>10.1f}{percentile(vals, 90):>10.1f}"
              f"{percentile(vals, 95):>10.1f}{percentile(vals, 99):>10.1f}"
              f"{statistics.mean(vals):>10.1f}")
    print("-" * 78)
    print(f"{'吞吐 (请求/秒)':<22}{len(ok) / wall_time_s:>10.2f}")
    print(f"{'吞吐 (输出token/秒)':<22}{total_out / wall_time_s:>10.1f}")
    print(f"{'单请求生成速度':<22}{1000 / statistics.median(tpots) if tpots else 0:>10.1f} tok/s")
    print(f"{'平均输出长度':<22}{total_out / len(ok):>10.1f} token")
    print(f"{'成功率':<22}{len(ok) / len(results) * 100:>10.2f} %")

    if failed:
        print("\n失败原因分布：")
        counts: dict[str, int] = {}
        for r in failed:
            key = r.error.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v} 次")


async def main() -> None:
    """解析参数并执行压测。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--api-key", default="sk-huacheng-internal-2025")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    limits = httpx.Limits(max_connections=args.concurrency * 2,
                          max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        print(f"预热 {args.warmup} 次 ...")
        await asyncio.gather(*[
            one_request(client, args.url, args.model, args.api_key, 32)
            for _ in range(args.warmup)])

        print(f"开始压测：并发 {args.concurrency}，总请求 {args.requests} ...")
        queue: asyncio.Queue = asyncio.Queue()
        for _ in range(args.requests):
            queue.put_nowait(1)

        results: list[RequestResult] = []
        t0 = time.perf_counter()
        await asyncio.gather(*[
            worker(i, client, args, queue, results) for i in range(args.concurrency)])
        wall = time.perf_counter() - t0

    report(results, wall, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
```

**预期输出**（实测环境：单卡 RTX 4090 24G，vLLM 0.6.4，Qwen2.5-14B-Instruct-AWQ，输入约 1800 token，输出 256 token）：

```text
预热 5 次 ...
开始压测：并发 16，总请求 200 ...

==============================================================================
压测报告  并发=16  总请求=200  成功=200  失败=0  耗时=94.3s
==============================================================================
指标                          P50       P90       P95       P99      均值
------------------------------------------------------------------------------
TTFT (ms)                  1284.2    2103.7    2486.1    3012.8    1421.6
TPOT (ms)                    24.8      29.1      31.4      35.2      25.6
E2E (ms)                   7521.0    8934.2    9310.5    9880.1    7602.3
------------------------------------------------------------------------------
吞吐 (请求/秒)                   2.12
吞吐 (输出token/秒)              542.3
单请求生成速度                    40.3 tok/s
平均输出长度                    255.7 token
成功率                       100.00 %
```

#### 2.4.3 怎么读这些数字

| 指标 | 这个值说明什么 | 什么时候该警惕 |
|---|---|---|
| **TTFT P50 = 1284ms** | 一半的请求 1.3 秒内出首字 | 如果业务要求 < 1s，不达标，要优化 Prefill |
| **TTFT P99 = 3013ms** | 最慢的 1% 要 3 秒。**P99 是 P50 的 2.3 倍，说明有排队** | P99/P50 > 3 说明并发过载或长请求阻塞，该开 chunked-prefill |
| **TPOT P50 = 24.8ms** | 每个 token 25ms，即 40 tok/s。跟 1.1 章的理论上限（4090 跑 14B-AWQ 约 100+ tok/s）比，只有 40% | 单请求速度慢是因为 16 并发在抢资源。**这是正常的**，看总吞吐 |
| **吞吐 542 输出token/s** | 这是服务器的真实产能 | 用它算容量：每天能产出 542 × 86400 ≈ 4680 万 token |
| **E2E P50 = 7.5s** | TTFT 1.3s + 256 token × 25ms = 7.7s，对得上 | 想降 E2E，最有效的是降 `max_tokens` |
| **成功率 100%** | 没有超时、没有 OOM | < 99.5% 就要排查 |

**做一组扫描，找到最优并发点**：

```bash
# 扫描不同并发下的表现
for c in 1 4 8 16 32 64; do
    echo "===== 并发 $c ====="
    python bench_llm.py --concurrency $c --requests $((c * 10)) --max-tokens 256
done
```

**典型的扫描结果**（同一环境）：

| 并发 | TTFT P50 | TTFT P99 | TPOT P50 | 吞吐(req/s) | 吞吐(tok/s) | 判读 |
|---|---|---|---|---|---|---|
| 1 | 312ms | 340ms | 9.8ms | 0.38 | 97.8 | 单请求最快，但 GPU 严重空闲 |
| 4 | 428ms | 610ms | 12.1ms | 1.28 | 328 | 吞吐提升 3.4×，延迟只涨 37% ✅ |
| 8 | 706ms | 1180ms | 17.3ms | 1.78 | 456 | 继续提升 |
| **16** | **1284ms** | **3013ms** | **24.8ms** | **2.12** | **542** | **吞吐接近饱和，延迟开始明显劣化** |
| 32 | 2840ms | 7210ms | 38.6ms | 2.24 | 573 | 吞吐只涨 6%，延迟涨 2.2×，**过载** |
| 64 | 6120ms | 18400ms | 52.1ms | 2.28 | 583 | 已经在排队，P99 无法接受 |

**这张扫描表是容量规划的依据**：

```mermaid
flowchart LR
    A["并发 1~8<br/>吞吐快速上升<br/>延迟温和上涨"] --> B["并发 8~16<br/>吞吐接近饱和<br/>延迟开始陡增"]
    B --> C["并发 > 32<br/>吞吐几乎不涨<br/>延迟线性恶化<br/>= 纯排队"]

    style A fill:#e1f5e1
    style B fill:#fff4e1
    style C fill:#ffe1e1
```

**结论**：这台机器的**甜点区在并发 8~16**。业务上要做的是：
1. 在网关层把并发限制在 16（超过的排队或降级），保证 P99 可控；
2. 如果业务峰值超过 16 并发，**加卡**，而不是硬扛；
3. 第 1.4 章会用这张表做完整的容量规划。

### 2.5 SGLang：RadixAttention 与前缀复用

#### 2.5.1 RadixAttention 解决什么

vLLM 的 Prefix Caching 是**线性前缀**匹配：两个请求必须从第一个 token 开始一模一样，才能共享。

SGLang 的 **RadixAttention** 用基数树（Radix Tree）组织 KV Cache，支持**树状共享**：

```mermaid
flowchart TD
    Root["根节点<br/>（空）"]
    Root --> S["system prompt<br/>1200 token<br/>【所有请求共享】"]
    S --> D1["知识片段 A<br/>800 token<br/>【问题1、2 共享】"]
    S --> D2["知识片段 B<br/>900 token<br/>【问题3、4 共享】"]
    D1 --> Q1["问题1<br/>50 token"]
    D1 --> Q2["问题2<br/>60 token"]
    D2 --> Q3["问题3<br/>45 token"]
    D2 --> Q4["问题4<br/>55 token"]

    style S fill:#e1f5e1
    style D1 fill:#fff4e1
    style D2 fill:#fff4e1
```

**在这个例子里**：4 个请求总共 `1200×4 + 800×2 + 900×2 + 210 = 8410` token 需要 Prefill（vLLM 只共享 system prompt 的话），而 RadixAttention 只需要 `1200 + 800 + 900 + 210 = 3110` token —— **减少 63%**。

#### 2.5.2 什么场景 SGLang 比 vLLM 更合适

| 场景 | 为什么 SGLang 更优 | 收益量级 |
|---|---|---|
| **多轮对话** | 每一轮的前缀都是上一轮的完整历史，树状共享天然契合 | 长对话场景 Prefill 可省 50%+ |
| **同一文档的多个问题** | 文档内容作为共享前缀，只算一次 | RAG 的「文档问答」模式收益大 |
| **Agent 的多分支探索** | ReAct 的不同分支共享前面的思考历史 | 分支越多收益越大 |
| **Few-shot 样例固定的批量任务** | 样例作为共享前缀 | 批量抽取/分类场景 |
| **复杂约束解码** | SGLang 的 xgrammar 集成和 DSL 更成熟 | 结构化输出更快 |
| **Self-Consistency 多路采样** | N 条路径共享同一个前缀 | 接近 N 倍的 Prefill 节省 |

| 场景 | vLLM 更优 |
|---|---|
| 请求之间前缀差异大（每次检索到的 chunk 都不同） | 树状共享用不上，vLLM 生态更成熟 |
| 需要 vLLM 独有的特性（某些量化格式、LoRA 热加载） | 功能覆盖更广 |
| 团队已经熟悉 vLLM 的运维 | 迁移成本 > 收益 |

#### 2.5.3 SGLang 安装与启动

```bash
# ===== 安装 =====
uv pip install "sglang[all]==0.4.0"
# pip 等价：pip install "sglang[all]==0.4.0"
# 注意：SGLang 对 CUDA/torch 版本要求与 vLLM 类似，建议用官方 Docker 镜像

# ===== 启动服务（参数与 vLLM 很像，但名字不同）=====
python -m sglang.launch_server \
    --model-path /data/models/Qwen2.5-14B-Instruct-AWQ \
    --served-model-name Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.88 \
    --context-length 16384 \
    --max-running-requests 32 \
    --quantization awq_marlin \
    --dtype half \
    --enable-torch-compile \
    --schedule-policy lpm \
    --api-key sk-huacheng-internal-2025
```

**vLLM ↔ SGLang 参数对照**（迁移时很有用）：

| 含义 | vLLM | SGLang |
|---|---|---|
| 模型路径 | `--model` | `--model-path` |
| 张量并行 | `--tensor-parallel-size` | `--tp-size` |
| 显存使用率 | `--gpu-memory-utilization` | `--mem-fraction-static` |
| 上下文长度 | `--max-model-len` | `--context-length` |
| 最大并发 | `--max-num-seqs` | `--max-running-requests` |
| 前缀缓存 | `--enable-prefix-caching` | **默认开启**（RadixAttention） |
| 调度策略 | 内置 | `--schedule-policy lpm/fcfs/dfs-weight` |
| 量化 | `--quantization` | `--quantization` |

> `--schedule-policy lpm` 是 **Longest Prefix Match**：优先调度与缓存树匹配最长的请求，最大化前缀复用。这是 SGLang 的特色调度策略。

#### 2.5.4 调用示例

SGLang 提供 OpenAI 兼容接口，所以客户端代码跟 vLLM 完全一样：

```python
"""SGLang 的两种用法：① OpenAI 兼容接口 ② 原生 DSL（结构化生成）。

环境：Python 3.11
依赖：uv pip install openai==1.57.0 "sglang[all]==0.4.0"
"""
from openai import OpenAI

# ===== ① OpenAI 兼容接口（与 vLLM 代码完全一致，只改 base_url）=====
client = OpenAI(base_url="http://localhost:30000/v1",
                api_key="sk-huacheng-internal-2025")

resp = client.chat.completions.create(
    model="Qwen2.5-14B-Instruct",
    messages=[{"role": "user", "content": "HC-V320 的过流阈值是多少？"}],
    temperature=0.0, max_tokens=256,
)
print(resp.choices[0].message.content)


# ===== ② SGLang 原生 DSL：多分支生成，共享前缀 =====
# 这是 SGLang 相对 vLLM 的独特能力：把「生成流程」写成程序
import sglang as sgl


@sgl.function
def diagnose_fault(s, context: str, question: str):
    """一个诊断流程：先分类故障类型，再按类型走不同的分支。

    注意：context 和 question 作为共享前缀，多次调用时 KV Cache 会被复用。
    """
    s += sgl.system("你是华成机电售后专家，严格依据资料回答。")
    s += sgl.user(f"【参考资料】\n{context}\n\n【问题】{question}")

    # 第一步：受限选择，输出只可能是给定选项之一
    s += sgl.assistant_begin()
    s += "故障类型：" + sgl.gen("fault_type", max_tokens=8,
                                choices=["过流", "过压", "欠压", "过热", "通信", "其他"])
    s += "\n严重程度：" + sgl.gen("severity", max_tokens=4, choices=["高", "中", "低"])
    s += "\n\n排查步骤：\n" + sgl.gen("steps", max_tokens=400, temperature=0.0)
    s += sgl.assistant_end()


def run_dsl_demo() -> None:
    """运行 DSL 示例。多个问题共享同一个 context 前缀。"""
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

    context = ("HC-V320 过流保护阈值为额定电流 1.5 倍，动作时间 20ms。"
               "OC 常见原因：负载突变、加速时间过短（F0.09）、绝缘下降、电缆过长。")
    questions = [
        "设备反复报 OC 怎么办？",
        "加速时间应该怎么调？",
        "绝缘电阻标准是多少？",
    ]

    # batch 调用，SGLang 会自动识别共享前缀
    states = diagnose_fault.run_batch(
        [{"context": context, "question": q} for q in questions],
        progress_bar=True,
    )
    for q, s in zip(questions, states):
        print(f"\n问题: {q}")
        print(f"  故障类型: {s['fault_type']}")
        print(f"  严重程度: {s['severity']}")
        print(f"  排查步骤: {s['steps'][:100]}...")


if __name__ == "__main__":
    run_dsl_demo()
```

**预期输出**：

```text
问题: 设备反复报 OC 怎么办？
  故障类型: 过流
  严重程度: 高
  排查步骤: 1. 断电验电，确认设备已完全停电并挂牌...

问题: 加速时间应该怎么调？
  故障类型: 过流
  严重程度: 中
  排查步骤: 1. 进入参数设置菜单，找到 F0.09 加速时间参数...
```

> **DSL 的真实价值**：`fault_type` 和 `severity` 用 `choices` 约束，输出**100% 合法**，不需要任何后处理。而且三个问题共享同一个 context 前缀，SGLang 只算一次 Prefill。

### 2.6 多模型共存与显存规划

#### 2.6.1 华成机电的单卡方案：一张 RTX 4090 跑全家桶

需求：LLM（生成）+ Embedding（检索）+ Reranker（精排），预算只有一张 24G 卡。

**显存分配方案**：

| 组件 | 模型 | 精度 | 权重显存 | 运行时显存 | 合计 | 占比 |
|---|---|---|---|---|---|---|
| **LLM** | Qwen2.5-7B-Instruct-AWQ | INT4 | 4.5 GB | KV Cache 8.0 GB + 激活 1.0 GB | **13.5 GB** | 56% |
| **Embedding** | bge-m3 (568M) | FP16 | 1.2 GB | batch 激活 1.0 GB | **2.2 GB** | 9% |
| **Reranker** | bge-reranker-v2-m3 (568M) | FP16 | 1.2 GB | batch 激活 0.8 GB | **2.0 GB** | 8% |
| **CUDA 上下文 + 框架开销** | — | — | — | — | **2.0 GB** | 8% |
| **安全余量** | — | — | — | — | **4.3 GB** | 18% |
| **合计** | | | | | **24.0 GB** | 100% |

**对应的启动配置**：

```bash
# ===== LLM：vLLM，限制显存使用率 =====
# 24G × 0.56 ≈ 13.5GB
vllm serve /data/models/Qwen2.5-7B-Instruct-AWQ \
    --served-model-name Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.56 \
    --max-model-len 8192 \
    --max-num-seqs 24 \
    --quantization awq_marlin \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests

# ===== Embedding：TEI，限制 batch =====
text-embeddings-router \
    --model-id /data/models/bge-m3 \
    --port 8080 \
    --max-client-batch-size 32 \
    --max-batch-tokens 8192 \
    --dtype float16

# ===== Reranker：TEI =====
text-embeddings-router \
    --model-id /data/models/bge-reranker-v2-m3 \
    --port 8081 \
    --max-client-batch-size 16 \
    --max-batch-tokens 4096 \
    --dtype float16
```

**关键注意事项**：

1. **`--gpu-memory-utilization` 是「占总显存的比例」，不是「占剩余显存的比例」**。如果你先起了 Embedding（占了 2.2 GB），再起 vLLM 设 0.56，vLLM 会认为自己能用 `24 × 0.56 = 13.4 GB`，但实际只剩 21.8 GB 可用 —— 这时候是够的。**但如果设 0.90，vLLM 会试图占 21.6 GB，加上已有的 2.2 GB 就爆了**；
2. **启动顺序很重要**：先起小的（Embedding、Reranker），再起 vLLM。因为 vLLM 会在启动时 profile 实际可用显存；
3. **反过来会出事**：vLLM 已经把显存吃满，再起 Embedding 必然 OOM。

#### 2.6.2 两卡方案（华成机电生产环境）

| GPU | 组件 | 显存分配 | 说明 |
|---|---|---|---|
| **GPU 0**（专用） | Qwen2.5-14B-Instruct-AWQ | 权重 8.6 GB + KV 11.5 GB + 开销 1.5 GB = **21.6 GB**（util 0.90） | 主力 LLM，独占 |
| **GPU 1**（共享） | Qwen2.5-1.5B-Instruct (FP16) | 权重 3.1 GB + KV 4.0 GB + 开销 1.0 GB = **8.1 GB**（util 0.34） | 意图分类、Query 改写 |
| | bge-m3 | **2.2 GB** | Embedding |
| | bge-reranker-v2-m3 | **2.0 GB** | Rerank |
| | CUDA 上下文 | **2.0 GB** | |
| | 余量 | **9.7 GB** | 给峰值 batch、以及未来加模型 |

**为什么这么分**：

- 主力 LLM 独占一张卡，**避免被其他模型的显存波动影响**（Embedding 的 batch 大小是波动的）；
- 小模型 + Embedding + Rerank 放一起，它们的调用是串行的（先 embed → 再 rerank → 最后 LLM），峰值不重叠；
- GPU 1 留了 40% 余量，因为共享卡的显存冲突最难排查，宁可浪费也别省。

#### 2.6.3 显存规划检查脚本

```python
"""多模型共卡的显存规划检查器：算出每个组件占多少，是否超标。

环境：Python 3.11
依赖：uv pip install pynvml==11.5.3
用法：python gpu_plan.py
"""
from dataclasses import dataclass


@dataclass
class Component:
    """一个部署在 GPU 上的组件。"""
    name: str
    params_b: float           # 参数量（十亿）
    dtype_bytes: float        # 每参数字节数：FP16=2, INT8=1, INT4≈0.55
    kv_kb_per_token: float = 0.0   # 每 token 的 KV Cache（KB），非 LLM 填 0
    max_model_len: int = 0
    max_num_seqs: int = 0
    activation_gb: float = 0.5     # 激活值峰值估计

    def weight_gb(self) -> float:
        """权重显存。"""
        return self.params_b * 1e9 * self.dtype_bytes / (1024 ** 3)

    def kv_cache_gb(self, avg_len_ratio: float = 0.35) -> float:
        """KV Cache 显存。avg_len_ratio 是平均请求长度 / max_model_len。"""
        if self.kv_kb_per_token == 0:
            return 0.0
        effective_len = self.max_model_len * avg_len_ratio
        return self.kv_kb_per_token * effective_len * self.max_num_seqs / (1024 ** 2)

    def total_gb(self, avg_len_ratio: float = 0.35) -> float:
        """总显存需求。"""
        return self.weight_gb() + self.kv_cache_gb(avg_len_ratio) + self.activation_gb


CUDA_CONTEXT_GB = 0.8   # 每个进程的 CUDA 上下文开销
SAFETY_MARGIN = 0.12    # 安全余量比例


def plan(gpu_name: str, gpu_total_gb: float, components: list[Component],
         avg_len_ratio: float = 0.35) -> None:
    """打印一张卡的显存规划表并给出结论。"""
    print(f"\n{'=' * 84}")
    print(f"GPU: {gpu_name}  总显存: {gpu_total_gb:.1f} GB")
    print(f"{'=' * 84}")
    print(f"{'组件':<28}{'权重GB':>9}{'KVCacheGB':>11}{'激活GB':>9}"
          f"{'小计GB':>9}{'占比':>8}")
    print("-" * 84)

    total = 0.0
    for c in components:
        w, kv, a = c.weight_gb(), c.kv_cache_gb(avg_len_ratio), c.activation_gb
        sub = w + kv + a
        total += sub
        print(f"{c.name:<28}{w:>9.2f}{kv:>11.2f}{a:>9.2f}{sub:>9.2f}"
              f"{sub / gpu_total_gb * 100:>7.1f}%")

    ctx = CUDA_CONTEXT_GB * len(components)
    total += ctx
    print(f"{'CUDA 上下文 × ' + str(len(components)):<28}{'':>9}{'':>11}{'':>9}"
          f"{ctx:>9.2f}{ctx / gpu_total_gb * 100:>7.1f}%")

    margin = gpu_total_gb - total
    print("-" * 84)
    print(f"{'合计占用':<28}{'':>9}{'':>11}{'':>9}{total:>9.2f}"
          f"{total / gpu_total_gb * 100:>7.1f}%")
    print(f"{'剩余余量':<28}{'':>9}{'':>11}{'':>9}{margin:>9.2f}"
          f"{margin / gpu_total_gb * 100:>7.1f}%")

    print()
    if margin < 0:
        print(f"❌ 超出显存 {-margin:.2f} GB！必须减配：降 max_num_seqs / "
              f"降 max_model_len / 量化 / 减少共卡组件")
    elif margin / gpu_total_gb < SAFETY_MARGIN:
        print(f"⚠ 余量仅 {margin / gpu_total_gb * 100:.1f}%，低于 {SAFETY_MARGIN:.0%} "
              f"安全线，峰值时有 OOM 风险")
    else:
        print(f"✅ 配置可行，余量 {margin / gpu_total_gb * 100:.1f}%")

    # 给出各组件的 gpu-memory-utilization 建议值
    print("\n建议的 vLLM --gpu-memory-utilization 设置：")
    for c in components:
        if c.kv_kb_per_token > 0:
            util = (c.weight_gb() + c.kv_cache_gb(avg_len_ratio) +
                    c.activation_gb + CUDA_CONTEXT_GB) / gpu_total_gb
            print(f"  {c.name}: --gpu-memory-utilization {util:.2f}")


if __name__ == "__main__":
    # ===== 方案一：单卡 4090 跑全家桶 =====
    plan("RTX 4090 (方案一：单卡全家桶)", 24.0, [
        Component("Qwen2.5-7B-AWQ (LLM)", 7.62, 0.55,
                  kv_kb_per_token=56, max_model_len=8192, max_num_seqs=24,
                  activation_gb=1.0),
        Component("bge-m3 (Embedding)", 0.568, 2, activation_gb=1.0),
        Component("bge-reranker-v2-m3", 0.568, 2, activation_gb=0.8),
    ])

    # ===== 方案二：GPU0 专用 LLM =====
    plan("RTX 4090 #0 (方案二：主力 LLM 独占)", 24.0, [
        Component("Qwen2.5-14B-AWQ (LLM)", 14.77, 0.55,
                  kv_kb_per_token=192, max_model_len=16384, max_num_seqs=32,
                  activation_gb=1.5),
    ])

    # ===== 方案三：GPU1 共享 =====
    plan("RTX 4090 #1 (方案三：小模型 + 检索组件)", 24.0, [
        Component("Qwen2.5-1.5B (小模型)", 1.54, 2,
                  kv_kb_per_token=28, max_model_len=8192, max_num_seqs=64,
                  activation_gb=0.8),
        Component("bge-m3 (Embedding)", 0.568, 2, activation_gb=1.0),
        Component("bge-reranker-v2-m3", 0.568, 2, activation_gb=0.8),
    ])
```

**预期输出**：

```text
====================================================================================
GPU: RTX 4090 (方案一：单卡全家桶)  总显存: 24.0 GB
====================================================================================
组件                              权重GB   KVCacheGB    激活GB    小计GB      占比
------------------------------------------------------------------------------------
Qwen2.5-7B-AWQ (LLM)              3.90       3.67      1.00      8.57   35.7%
bge-m3 (Embedding)                1.06       0.00      1.00      2.06    8.6%
bge-reranker-v2-m3                1.06       0.00      0.80      1.86    7.7%
CUDA 上下文 × 3                                                    2.40   10.0%
------------------------------------------------------------------------------------
合计占用                                                          14.89   62.0%
剩余余量                                                           9.11   38.0%

✅ 配置可行，余量 38.0%

建议的 vLLM --gpu-memory-utilization 设置：
  Qwen2.5-7B-AWQ (LLM): --gpu-memory-utilization 0.39
```

### 2.7 网关层：统一多个后端

#### 2.7.1 为什么需要网关

```mermaid
flowchart LR
    subgraph A ["没有网关（混乱）"]
        A1["业务代码 1"] --> A4["vLLM 主模型"]
        A1 --> A5["DeepSeek API"]
        A2["业务代码 2"] --> A4
        A2 --> A6["vLLM 小模型"]
        A3["业务代码 3"] --> A5
        A3 --> A6
    end

    subgraph B ["有网关（清晰）"]
        B1["业务代码 1"] --> G["统一网关<br/>OpenAI 兼容<br/>:9000/v1"]
        B2["业务代码 2"] --> G
        B3["业务代码 3"] --> G
        G --> B4["vLLM 主模型"]
        G --> B5["DeepSeek API"]
        G --> B6["vLLM 小模型"]
    end
```

网关要解决的六件事：

| 问题 | 没有网关 | 有网关 |
|---|---|---|
| **换模型** | 改所有业务代码 | 改一行配置 |
| **降级** | 每个业务方自己写 fallback | 网关统一处理 |
| **限流** | 无法全局控制 | 统一令牌桶 |
| **Key 管理** | Key 散落在各处 | 集中管理 + 轮转 |
| **成本统计** | 各自记账，对不上 | 统一埋点 |
| **重试/超时** | 各写各的 | 统一策略 |

#### 2.7.2 方案一：LiteLLM（开箱即用）

```yaml
# 文件：gateway/litellm_config.yaml
# 启动：litellm --config litellm_config.yaml --port 9000

model_list:
  # ===== 本地 vLLM 主模型 =====
  - model_name: huacheng-main            # 对外暴露的名字
    litellm_params:
      model: openai/Qwen2.5-14B-Instruct # openai/ 前缀表示 OpenAI 兼容协议
      api_base: http://vllm-main:8000/v1
      api_key: sk-huacheng-internal-2025
      rpm: 600                            # 每分钟请求数上限
      timeout: 120
    model_info:
      mode: chat
      max_tokens: 16384

  # ===== 本地 vLLM 小模型 =====
  - model_name: huacheng-small
    litellm_params:
      model: openai/Qwen2.5-1.5B-Instruct
      api_base: http://vllm-small:8000/v1
      api_key: sk-huacheng-internal-2025
      rpm: 3000
      timeout: 30

  # ===== 在线 API：同一个 model_name 配多个 Key，自动轮转 =====
  - model_name: huacheng-cloud
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY_1
      rpm: 500
  - model_name: huacheng-cloud
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY_2
      rpm: 500

  # ===== 推理模型 =====
  - model_name: huacheng-reasoner
    litellm_params:
      model: deepseek/deepseek-reasoner
      api_key: os.environ/DEEPSEEK_API_KEY_1
      timeout: 300

router_settings:
  routing_strategy: usage-based-routing-v2   # 按用量均衡（还可选 simple-shuffle / least-busy）
  num_retries: 2
  timeout: 120
  allowed_fails: 3                            # 连续失败 3 次后冷却该部署
  cooldown_time: 60
  # 降级链：主模型挂了自动切到云端
  fallbacks:
    - huacheng-main: ["huacheng-cloud"]
    - huacheng-cloud: ["huacheng-main"]
  context_window_fallbacks:
    - huacheng-small: ["huacheng-main"]      # 小模型上下文不够时自动升级

litellm_settings:
  drop_params: true              # 自动丢弃后端不支持的参数（如给 reasoner 传 temperature）
  set_verbose: false
  request_timeout: 120
  # 缓存（可选，需要 Redis）
  cache: true
  cache_params:
    type: redis
    host: redis
    port: 6379
    ttl: 3600
    supported_call_types: ["acompletion", "completion"]

general_settings:
  master_key: sk-huacheng-gateway-master    # 网关自己的鉴权 key
  database_url: os.environ/DATABASE_URL     # 可选：存用量统计
  # 每个虚拟 key 的预算控制
  max_budget: 5000                          # 单位：美元或自定义
  budget_duration: 30d
```

```bash
# 安装与启动
uv pip install "litellm[proxy]==1.55.0"
export DEEPSEEK_API_KEY_1=sk-xxx
export DEEPSEEK_API_KEY_2=sk-yyy
litellm --config gateway/litellm_config.yaml --port 9000

# 调用（客户端只知道 huacheng-main 这个名字）
curl http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer sk-huacheng-gateway-master" \
  -H "Content-Type: application/json" \
  -d '{"model":"huacheng-main","messages":[{"role":"user","content":"你好"}]}'
```

#### 2.7.3 方案二：自建 FastAPI 网关（可控性更强）

LiteLLM 功能全但也重。如果你只要「统一入口 + 降级 + 限流 + Key 轮转 + 埋点」，自己写两百行更可控。

```python
"""华成机电 LLM 网关：统一多后端为一个 OpenAI 兼容端点。

环境：Python 3.11
依赖：uv pip install fastapi==0.115.6 uvicorn==0.34.0 httpx==0.28.0 pydantic==2.10.3
启动：uvicorn gateway.main:app --host 0.0.0.0 --port 9000 --workers 2

功能：
  1. OpenAI 兼容的 /v1/chat/completions（含流式）
  2. 模型别名 → 真实后端的映射
  3. 失败自动降级到备用后端
  4. 令牌桶限流（全局 + 按模型）
  5. API Key 轮转（多个 key 轮询使用，单 key 限流时自动切换）
  6. 统一埋点（token 用量、延迟、成本）
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("gateway")

GATEWAY_KEY = os.getenv("GATEWAY_API_KEY", "sk-huacheng-gateway-master")


# ===================== 后端定义 =====================
@dataclass
class Backend:
    """一个模型后端。"""
    name: str
    base_url: str
    model_id: str
    api_keys: list[str]
    rpm_limit: int = 600
    timeout_s: float = 120.0
    cost_in_per_1m: float = 0.0
    cost_out_per_1m: float = 0.0
    strip_params: tuple[str, ...] = ()      # 需要剥离的参数（如 reasoner 不接受 temperature）

    _key_cycle: itertools.cycle = field(init=False, repr=False)
    _req_times: deque = field(default_factory=deque, repr=False)
    _fail_count: int = field(default=0, repr=False)
    _cooldown_until: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._key_cycle = itertools.cycle(self.api_keys)

    def next_key(self) -> str:
        """取下一个 API Key（轮转）。"""
        return next(self._key_cycle)

    @property
    def healthy(self) -> bool:
        """熔断器是否闭合。"""
        return time.time() >= self._cooldown_until

    def allow_request(self) -> bool:
        """令牌桶限流：滑动窗口统计最近 60 秒的请求数。"""
        now = time.time()
        while self._req_times and now - self._req_times[0] > 60:
            self._req_times.popleft()
        if len(self._req_times) >= self.rpm_limit:
            return False
        self._req_times.append(now)
        return True

    def on_success(self) -> None:
        """成功后重置失败计数。"""
        self._fail_count = 0

    def on_failure(self, threshold: int = 3, cooldown_s: int = 60) -> None:
        """失败计数，超阈值则熔断。"""
        self._fail_count += 1
        if self._fail_count >= threshold:
            self._cooldown_until = time.time() + cooldown_s
            self._fail_count = 0
            logger.warning("熔断: %s 冷却 %ds", self.name, cooldown_s)


def _keys(env_name: str, default: str = "EMPTY") -> list[str]:
    """从环境变量读取逗号分隔的多个 Key。"""
    raw = os.getenv(env_name, default)
    return [k.strip() for k in raw.split(",") if k.strip()] or [default]


BACKENDS: dict[str, Backend] = {
    "vllm-main": Backend(
        name="vllm-main",
        base_url=os.getenv("MAIN_BASE_URL", "http://vllm-main:8000/v1"),
        model_id="Qwen2.5-14B-Instruct",
        api_keys=_keys("INTERNAL_API_KEY", "sk-huacheng-internal-2025"),
        rpm_limit=1200),
    "vllm-small": Backend(
        name="vllm-small",
        base_url=os.getenv("SMALL_BASE_URL", "http://vllm-small:8000/v1"),
        model_id="Qwen2.5-1.5B-Instruct",
        api_keys=_keys("INTERNAL_API_KEY", "sk-huacheng-internal-2025"),
        rpm_limit=6000, timeout_s=30.0),
    "api-chat": Backend(
        name="api-chat", base_url="https://api.deepseek.com/v1",
        model_id="deepseek-chat", api_keys=_keys("DEEPSEEK_API_KEYS"),
        rpm_limit=500, cost_in_per_1m=2.0, cost_out_per_1m=8.0),
    "api-reasoner": Backend(
        name="api-reasoner", base_url="https://api.deepseek.com/v1",
        model_id="deepseek-reasoner", api_keys=_keys("DEEPSEEK_API_KEYS"),
        rpm_limit=200, timeout_s=300.0,
        cost_in_per_1m=4.0, cost_out_per_1m=16.0,
        strip_params=("temperature", "top_p", "presence_penalty", "frequency_penalty")),
}

# 对外模型别名 → [主后端, 降级后端1, 降级后端2...]
ALIAS_ROUTES: dict[str, list[str]] = {
    "huacheng-main": ["vllm-main", "api-chat"],
    "huacheng-small": ["vllm-small", "vllm-main"],
    "huacheng-reasoner": ["api-reasoner", "vllm-main"],
}


# ===================== 埋点 =====================
@dataclass
class CallRecord:
    """一次调用的埋点记录。"""
    alias: str
    backend: str
    ok: bool
    latency_ms: float
    ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    error: str = ""


METRICS: deque[CallRecord] = deque(maxlen=10000)


def record(rec: CallRecord) -> None:
    """记录埋点并输出结构化日志。"""
    METRICS.append(rec)
    logger.info("call alias=%s backend=%s ok=%s latency=%.0fms ttft=%.0fms "
                "in=%d out=%d cost=%.6f %s",
                rec.alias, rec.backend, rec.ok, rec.latency_ms, rec.ttft_ms,
                rec.prompt_tokens, rec.completion_tokens, rec.cost_cny, rec.error)


# ===================== FastAPI 应用 =====================
app = FastAPI(title="华成机电 LLM 网关", version="1.0")
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    """创建全局 HTTP 客户端，复用连接池。"""
    global _client
    _client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
    logger.info("网关启动，已注册后端: %s", list(BACKENDS))


@app.on_event("shutdown")
async def _shutdown() -> None:
    """关闭 HTTP 客户端。"""
    if _client:
        await _client.aclose()


def _check_auth(authorization: str | None) -> None:
    """校验网关自身的 API Key。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "缺少 Authorization 头")
    if authorization[7:] != GATEWAY_KEY:
        raise HTTPException(401, "API Key 无效")


def _pick_backends(alias: str) -> list[Backend]:
    """按别名取出候选后端链（已过滤掉熔断和超限的）。"""
    names = ALIAS_ROUTES.get(alias)
    if not names:
        raise HTTPException(404, f"未知模型别名: {alias}，可用: {list(ALIAS_ROUTES)}")
    usable = [BACKENDS[n] for n in names
              if BACKENDS[n].healthy and BACKENDS[n].allow_request()]
    if not usable:
        raise HTTPException(503, f"别名 {alias} 的所有后端不可用（熔断或限流）")
    return usable


def _prepare_payload(body: dict, backend: Backend) -> dict:
    """把请求体改写成后端能接受的格式。"""
    payload = dict(body)
    payload["model"] = backend.model_id
    for p in backend.strip_params:
        payload.pop(p, None)
    return payload


@app.get("/health")
async def health() -> dict:
    """健康检查，返回各后端状态。"""
    return {
        "status": "ok",
        "backends": {
            b.name: {"healthy": b.healthy, "rpm_used": len(b._req_times),
                     "rpm_limit": b.rpm_limit}
            for b in BACKENDS.values()
        },
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """返回对外暴露的模型别名列表（OpenAI 兼容格式）。"""
    return {"object": "list",
            "data": [{"id": a, "object": "model", "owned_by": "huacheng"}
                     for a in ALIAS_ROUTES]}


@app.get("/stats")
async def stats() -> dict:
    """返回最近的调用统计。"""
    recs = list(METRICS)
    if not recs:
        return {"count": 0}
    ok = [r for r in recs if r.ok]
    lat = sorted(r.latency_ms for r in ok)
    by_backend: dict[str, int] = {}
    for r in recs:
        by_backend[r.backend] = by_backend.get(r.backend, 0) + 1
    return {
        "count": len(recs),
        "success_rate": round(len(ok) / len(recs), 4),
        "latency_p50_ms": lat[len(lat) // 2] if lat else 0,
        "latency_p95_ms": lat[int(len(lat) * 0.95)] if lat else 0,
        "total_cost_cny": round(sum(r.cost_cny for r in recs), 4),
        "total_prompt_tokens": sum(r.prompt_tokens for r in recs),
        "total_completion_tokens": sum(r.completion_tokens for r in recs),
        "calls_by_backend": by_backend,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request,
                           authorization: str | None = Header(None)):
    """OpenAI 兼容的聊天补全接口，支持流式与降级。"""
    _check_auth(authorization)
    body = await request.json()
    alias = body.get("model", "")
    stream = bool(body.get("stream", False))
    candidates = _pick_backends(alias)

    last_error = ""
    for backend in candidates:
        payload = _prepare_payload(body, backend)
        headers = {"Authorization": f"Bearer {backend.next_key()}",
                   "Content-Type": "application/json"}
        url = f"{backend.base_url}/chat/completions"
        t0 = time.perf_counter()

        if stream:
            try:
                return StreamingResponse(
                    _stream_proxy(backend, url, payload, headers, alias, t0),
                    media_type="text/event-stream")
            except Exception as exc:
                backend.on_failure()
                last_error = f"{backend.name}: {exc}"
                logger.warning("流式后端 %s 失败，尝试降级: %s", backend.name, exc)
                continue

        try:
            resp = await _client.post(url, json=payload, headers=headers,
                                      timeout=backend.timeout_s)
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                backend.on_failure()
                last_error = f"{backend.name}: HTTP {resp.status_code} {resp.text[:200]}"
                logger.warning("后端返回非 200，降级: %s", last_error)
                continue

            data = resp.json()
            usage = data.get("usage") or {}
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            cost = pt / 1e6 * backend.cost_in_per_1m + ct / 1e6 * backend.cost_out_per_1m
            backend.on_success()
            record(CallRecord(alias, backend.name, True, latency,
                              prompt_tokens=pt, completion_tokens=ct, cost_cny=cost))
            data["model"] = alias          # 对外统一返回别名
            return JSONResponse(data)

        except Exception as exc:
            backend.on_failure()
            latency = (time.perf_counter() - t0) * 1000
            last_error = f"{backend.name}: {type(exc).__name__}: {exc}"
            record(CallRecord(alias, backend.name, False, latency, error=last_error))
            logger.warning("后端 %s 异常，尝试降级: %s", backend.name, exc)
            continue

    raise HTTPException(502, f"所有后端均失败。最后错误: {last_error}")


async def _stream_proxy(backend: Backend, url: str, payload: dict,
                        headers: dict, alias: str, t0: float):
    """流式转发生成器：边转发边统计 TTFT 与 token 数。"""
    ttft_ms = 0.0
    n_chunks = 0
    pt = ct = 0
    try:
        async with _client.stream("POST", url, json=payload, headers=headers,
                                  timeout=backend.timeout_s) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                backend.on_failure()
                raise RuntimeError(f"HTTP {resp.status_code}: {body[:200]!r}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data != "[DONE]":
                        try:
                            obj = json.loads(data)
                            obj["model"] = alias
                            if (obj.get("choices") and
                                    (obj["choices"][0].get("delta") or {}).get("content")):
                                if ttft_ms == 0.0:
                                    ttft_ms = (time.perf_counter() - t0) * 1000
                                n_chunks += 1
                            if obj.get("usage"):
                                pt = obj["usage"].get("prompt_tokens", 0)
                                ct = obj["usage"].get("completion_tokens", 0)
                            line = "data: " + json.dumps(obj, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass
                yield line + "\n\n"
        backend.on_success()
    finally:
        latency = (time.perf_counter() - t0) * 1000
        ct = ct or n_chunks
        cost = pt / 1e6 * backend.cost_in_per_1m + ct / 1e6 * backend.cost_out_per_1m
        record(CallRecord(alias, backend.name, True, latency, ttft_ms,
                          pt, ct, cost))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

**使用网关**：

```python
"""业务代码通过网关调用，完全不关心底层是哪个模型。"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1",
                api_key="sk-huacheng-gateway-master")

# 主模型
r1 = client.chat.completions.create(
    model="huacheng-main",
    messages=[{"role": "user", "content": "HC-V320 过流阈值？"}])

# 小模型（意图分类）
r2 = client.chat.completions.create(
    model="huacheng-small",
    messages=[{"role": "user", "content": "判断意图：我这机器不转了"}],
    max_tokens=16)

# 推理模型（网关会自动剥离 temperature 参数）
r3 = client.chat.completions.create(
    model="huacheng-reasoner",
    messages=[{"role": "user", "content": "分析这批设备反复故障的根本原因..."}],
    temperature=0.0)   # 这个参数会被网关剥离，不会报错
```

**验证降级**：

```bash
# 1. 看网关健康状态
curl http://localhost:9000/health | python -m json.tool

# 2. 故意停掉主模型
docker compose stop vllm-main

# 3. 再次调用 huacheng-main，应该自动降级到 api-chat
curl http://localhost:9000/v1/chat/completions \
  -H "Authorization: Bearer sk-huacheng-gateway-master" \
  -d '{"model":"huacheng-main","messages":[{"role":"user","content":"你好"}]}'
# 日志里会看到：
# WARNING 后端 vllm-main 异常，尝试降级: ConnectError
# INFO call alias=huacheng-main backend=api-chat ok=True ...

# 4. 看统计
curl http://localhost:9000/stats | python -m json.tool
```

---

## 三、动手实战：从零起一套完整部署

### 3.1 环境准备

```bash
# ===== 1. 检查硬件 =====
nvidia-smi
# 确认：驱动版本 >= 535，显卡型号，显存大小

# ===== 2. 安装 NVIDIA Container Toolkit（Docker 用 GPU 必需）=====
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# ===== 3. 准备模型目录 =====
sudo mkdir -p /data/models /data/vllm-cache
sudo chown -R $USER:$USER /data

# ===== 4. 下载模型（三种方式任选）=====
# 方式 A：HuggingFace CLI
uv pip install "huggingface_hub[cli]==0.26.2"
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ \
    --local-dir /data/models/Qwen2.5-14B-Instruct-AWQ \
    --local-dir-use-symlinks False

# 方式 B：国内镜像（速度快很多）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ \
    --local-dir /data/models/Qwen2.5-14B-Instruct-AWQ

# 方式 C：ModelScope（国内官方渠道）
uv pip install modelscope==1.20.1
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-14B-Instruct-AWQ',
                  local_dir='/data/models/Qwen2.5-14B-Instruct-AWQ')
"

# ===== 5. 记录模型版本（可复现性）=====
cd /data/models/Qwen2.5-14B-Instruct-AWQ
sha256sum *.safetensors > CHECKSUMS.txt
cat config.json | python -m json.tool > /dev/null && echo "config.json 格式正常"
```

### 3.2 三个阶段的部署演进

```mermaid
flowchart LR
    A["阶段一: 开发验证<br/>Ollama<br/>笔记本 / 单机<br/>10 分钟起步"] --> B["阶段二: 单机生产<br/>vLLM + Docker<br/>1~2 张卡<br/>压测调参"]
    B --> C["阶段三: 高可用<br/>vLLM 多实例 + 网关<br/>多卡 / 多机<br/>降级 + 监控"]
```

**阶段一命令**（5 分钟）：

```bash
ollama pull qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct "HC-V320 报 OC 故障怎么办？"
```

**阶段二命令**（30 分钟）：

```bash
docker run -d --name vllm-main --gpus '"device=0"' --ipc=host \
  -p 8000:8000 \
  -v /data/models:/models:ro \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.6.4.post1 \
  --model /models/Qwen2.5-14B-Instruct-AWQ \
  --served-model-name Qwen2.5-14B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --quantization awq_marlin \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-log-requests \
  --api-key sk-huacheng-internal-2025

# 等待启动（5 分钟内）
docker logs -f vllm-main | grep -m1 "Application startup complete"

# 压测
python bench_llm.py --concurrency 16 --requests 200
```

**阶段三命令**（2 小时）：

```bash
docker compose -f deploy/docker-compose.yml up -d
# 起 vllm-main + vllm-small + embedding + reranker + gateway

# 逐个验证
curl http://localhost:8000/health   # vLLM 主模型
curl http://localhost:8001/health   # vLLM 小模型
curl http://localhost:8080/health   # Embedding
curl http://localhost:9000/health   # 网关

# 端到端压测（通过网关）
python bench_llm.py --url http://localhost:9000/v1 \
    --model huacheng-main --api-key sk-huacheng-gateway-master \
    --concurrency 16 --requests 200
```

---

## 四、踩坑与排错

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | vLLM 启动报 `CUDA out of memory`，但 `nvidia-smi` 显示显存是空的 | 有僵尸进程占着显存；或另一个容器已经占用 | `nvidia-smi` 看 PID，`kill -9`；或 `fuser -v /dev/nvidia*`。容器场景检查 `docker ps` 是否有旧容器 |
| 2 | Docker 里跑 vLLM 报 `Bus error` 或共享内存不足 | 没加 `--ipc=host`，Docker 默认共享内存只有 64MB，vLLM 多进程不够用 | 加 `--ipc=host`；或 `--shm-size=8g` |
| 3 | `--tensor-parallel-size 3` 启动失败 | TP 数必须能整除注意力头数和 KV 头数 | Qwen2.5 只能用 1/2/4/8；查 `config.json` 的 `num_key_value_heads` |
| 4 | 多卡 TP 时性能反而不如单卡 | PCIe 带宽不足，AllReduce 通信成为瓶颈（消费级卡没 NVLink） | 4090 等消费卡尽量避免 TP；实在要用，确认 PCIe 是 x16 不是 x4；`nvidia-smi topo -m` 看拓扑 |
| 5 | 开了 `--enable-prefix-caching` 但 `prefix_cache_hit_rate` 是 0 | prompt 开头有可变内容（时间戳、session id、随机 few-shot） | 固定内容前置；检查 LangChain 的 prompt 模板有没有把时间注入到开头 |
| 6 | 请求偶发返回 `400 This model's maximum context length is X tokens` | 输入 token + `max_tokens` 超过了 `max-model-len` | 网关层预检：`len(tokenize(prompt)) + max_tokens <= max_model_len`；超了就裁剪历史或减少 chunk |
| 7 | 日志里大量 `Sequence group ... is preempted by PreemptionMode.RECOMPUTE` | KV Cache 不够，vLLM 在抢占重算 | 降 `--max-num-seqs`；降 `--max-model-len`；调高 `--gpu-memory-utilization`；加 `--swap-space` |
| 8 | Function calling 返回的 `tool_calls` 总是 null，模型把 JSON 写在 content 里 | 没加 `--enable-auto-tool-choice --tool-call-parser` 参数 | 加上对应模型的 parser（Qwen 用 `hermes`）；参数名和支持列表以 vLLM 官方文档为准 |
| 9 | Ollama 第一个请求要等 60~90 秒 | 模型被 `keep_alive` 超时卸载，需要重新从磁盘加载 | `OLLAMA_KEEP_ALIVE=-1`；或启动后主动预热一次 |
| 10 | Ollama 并发一上去就排队，GPU 利用率却很低 | `OLLAMA_NUM_PARALLEL` 默认值低 | 调大该值（注意显存线性增长）；生产环境换 vLLM |
| 11 | Ollama 多开几个实例做负载均衡，直接 OOM | 每个实例独立加载一份完整权重 | Ollama 不是这么用的；单实例调 `OLLAMA_NUM_PARALLEL`，或换 vLLM |
| 12 | 量化模型启动报 `Cannot find the config file for awq` | 模型不是 AWQ 量化版，却传了 `--quantization awq` | 确认下载的是 `-AWQ` 后缀的仓库；或去掉 `--quantization` 参数 |
| 13 | AWQ 模型推理速度没有比 FP16 快 | 没走 Marlin kernel（fallback 到慢路径） | 显式指定 `--quantization awq_marlin`；确认 GPU 是 Ampere 及以上 |
| 14 | 换了新模型后，同样的 prompt 输出格式全变了 | 不同模型的 chat template 不同 | 统一走 `/v1/chat/completions`（让后端套模板），不要自己拼 prompt 字符串 |
| 15 | 多个模型共卡，先起的服务正常，后起的 OOM | `--gpu-memory-utilization` 是「占总显存比例」，多个进程会重复计算 | 手工分配比例，总和 ≤ 0.85；先起小的再起大的；用 2.6.3 的规划脚本 |
| 16 | 压测时吞吐上不去，GPU 利用率只有 40% | 客户端是瓶颈（Python GIL、连接数不够） | 用 `asyncio` + `httpx`；加大 `max_connections`；或多进程压测 |
| 17 | 压测 P99 比 P50 高 10 倍 | 长请求阻塞短请求（Prefill 独占） | 开 `--enable-chunked-prefill`；调小 `--max-num-batched-tokens` |
| 18 | 服务跑几天后变慢，重启就好 | 内存泄漏；或 Prefix Cache 池碎片化 | 升级 vLLM 版本；加定期重启（凌晨低峰期，滚动重启）；监控 RSS |
| 19 | `vllm serve` 启动卡在 `Loading safetensors checkpoint shards` 很久 | 模型在网络存储（NFS）上，或磁盘是机械盘 | 模型放本地 SSD；`--load-format safetensors` 比 `.bin` 快 |
| 20 | SGLang 启动报错，vLLM 同样配置却正常 | 两者对 CUDA/torch 版本要求不同，装在同一个环境里互相覆盖 | **不要装在同一个虚拟环境**；用 Docker 隔离 |
| 21 | 网关加了降级，但主模型挂了之后所有请求都超时 120 秒才降级 | 超时设得太长，降级不及时 | 主后端用短超时（如 30s）快速失败，降级后端用长超时；加健康探测主动摘除 |
| 22 | 流式响应在 Nginx 反代后变成了一次性返回 | Nginx 默认开启了响应缓冲 | `proxy_buffering off; proxy_cache off; chunked_transfer_encoding on;` |

---

## 五、生产级要点

### 5.1 监控清单

**Prometheus 抓取配置**：

```yaml
# 文件：deploy/prometheus.yml
scrape_configs:
  - job_name: vllm-main
    scrape_interval: 15s
    static_configs:
      - targets: ['vllm-main:8000']
    metrics_path: /metrics

  - job_name: vllm-small
    scrape_interval: 15s
    static_configs:
      - targets: ['vllm-small:8000']

  - job_name: gateway
    scrape_interval: 15s
    static_configs:
      - targets: ['gateway:9000']

  - job_name: gpu
    scrape_interval: 15s
    static_configs:
      - targets: ['dcgm-exporter:9400']   # NVIDIA DCGM Exporter
```

**必须盯的告警规则**：

```yaml
# 文件：deploy/alert_rules.yml
groups:
  - name: llm_service
    rules:
      - alert: TTFT过高
        expr: histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 3
        for: 3m
        annotations:
          summary: "TTFT P95 超过 3 秒，检查排队与 Prefix Cache 命中率"

      - alert: 请求排队
        expr: vllm:num_requests_waiting > 5
        for: 2m
        annotations:
          summary: "有请求在排队，容量不足，考虑扩容或限流"

      - alert: KVCache将满
        expr: vllm:gpu_cache_usage_perc > 0.92
        for: 3m
        annotations:
          summary: "KV Cache 使用率超 92%，即将触发抢占"

      - alert: 发生抢占
        expr: increase(vllm:num_preemptions_total[5m]) > 0
        for: 1m
        annotations:
          summary: "发生请求抢占，说明并发配置过高，会导致重算浪费"

      - alert: 前缀缓存命中率低
        expr: vllm:gpu_prefix_cache_hit_rate < 0.2
        for: 10m
        annotations:
          summary: "RAG 场景前缀缓存命中率低于 20%，检查 prompt 是否把可变内容放前面了"

      - alert: 显存接近上限
        expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.95
        for: 3m
        annotations:
          summary: "GPU 显存使用超 95%，OOM 风险"

      - alert: 服务不可用
        expr: up{job=~"vllm.*"} == 0
        for: 1m
        annotations:
          summary: "vLLM 服务下线，检查容器状态与显存"
```

### 5.2 优雅重启与滚动升级

```bash
#!/usr/bin/env bash
# 文件：deploy/rolling_restart.sh
# 用途：零停机滚动重启 vLLM 服务（需要至少 2 个实例 + 网关）
set -euo pipefail

GATEWAY_HEALTH="http://localhost:9000/health"
INSTANCES=("vllm-main-a" "vllm-main-b")

wait_healthy() {
    local name="$1" max_wait=600 waited=0
    echo "等待 $name 健康..."
    while [ $waited -lt $max_wait ]; do
        if docker exec "$name" curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "  $name 已健康（用时 ${waited}s）"
            return 0
        fi
        sleep 10; waited=$((waited + 10))
    done
    echo "  ❌ $name 超时未健康"; return 1
}

for inst in "${INSTANCES[@]}"; do
    echo "===== 滚动重启 $inst ====="
    # 1. 从网关摘除（这里假设网关支持通过环境变量或 API 下线某个后端）
    curl -sf -X POST "http://localhost:9000/admin/drain?backend=$inst" || true
    echo "  已从网关摘除，等待 30s 让存量请求完成..."
    sleep 30

    # 2. 优雅停止（给 60 秒处理存量）
    docker stop -t 60 "$inst"

    # 3. 重启
    docker start "$inst"
    wait_healthy "$inst"

    # 4. 预热（跑几个请求把 CUDA Graph 和缓存建起来）
    for _ in {1..5}; do
        docker exec "$inst" curl -sf -X POST http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer sk-huacheng-internal-2025" \
            -d '{"model":"Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
            >/dev/null || true
    done
    echo "  预热完成"

    # 5. 重新加入网关
    curl -sf -X POST "http://localhost:9000/admin/undrain?backend=$inst" || true
    echo "  已重新加入网关，观察 60s..."
    sleep 60
done

echo "✅ 滚动重启完成"
curl -s "$GATEWAY_HEALTH" | python -m json.tool
```

### 5.3 部署检查清单（上线前逐条勾）

```text
【模型与权重】
□ 模型从可信渠道下载，记录了 commit hash / sha256
□ 模型放在本地 SSD，不是 NFS
□ 许可证已经法务确认
□ 量化版本已用金标集验证过精度损失可接受

【服务配置】
□ --max-model-len 基于线上真实长度分布（P99.9）设定，不是照抄模型最大值
□ --gpu-memory-utilization 留了 8%~10% 余量
□ --enable-prefix-caching 已开启（RAG 场景）
□ --enable-chunked-prefill 已开启（高并发场景）
□ --disable-log-requests 已开启（否则日志爆盘）
□ --api-key 已设置（内网也要设）
□ 如需 function calling，--enable-auto-tool-choice 和 --tool-call-parser 已配

【容量】
□ 已做并发扫描压测，知道甜点区在哪
□ 网关层限流阈值 = 甜点区并发数
□ 峰值 QPS × 平均 E2E 时间 < 甜点区并发数（Little 定律）
□ 算过日均 token 产能，能覆盖业务量

【可靠性】
□ Docker restart 策略是 unless-stopped
□ healthcheck 配了，start_period 给足（大模型要 5 分钟）
□ 网关有降级链，且实测过降级生效
□ 有回滚方案（旧版本容器保留）
□ 日志有 rotate（max-size + max-file）

【监控】
□ Prometheus 已抓 vLLM metrics
□ 告警规则已配（TTFT / 排队 / KVCache / 抢占 / 显存）
□ 有埋点记录每次调用的 token 与延迟
□ GPU 监控（DCGM Exporter）已部署

【安全】
□ 服务不直接暴露公网（只在内网或经过网关）
□ --trust-remote-code 只对可信模型开
□ API Key 通过环境变量注入，不写在镜像里
□ 模型目录只读挂载（:ro）
```

---

## 六、本章小结 + 自测题

### 6.1 要点回顾

1. **Ollama 是开发工具，vLLM 是生产工具**。Ollama 的「每实例独立加载权重 + 固定并行槽位」设计与生产的高并发诉求冲突。别把 Ollama 搬到生产环境。
2. **vLLM 的三个关键参数是三角关系**：`gpu-memory-utilization`（分多大蛋糕）× `max-model-len`（每个请求切多大块）× `max-num-seqs`（切几块）。真正要监控的是 `gpu_cache_usage_perc` 和 `num_preemptions`。
3. **`max-model-len` 要基于线上真实长度分布（P99.9）设定**，不是照抄模型最大值。设大一倍，并发能力砍一半。
4. **显存不够的解法按性价比排序**：降 max-model-len → 权重量化 AWQ → RAG 层减 chunk → 调高 util → KV Cache FP8 → 多卡 TP。**不要一上来就想着加卡**。
5. **RAG 场景 `--enable-prefix-caching` 是免费午餐**，但要保证 prompt 的固定部分在前，否则命中率为 0。
6. **压测要扫描不同并发，找到「吞吐饱和 + 延迟可接受」的甜点区**，然后在网关层把并发限制在这个值。P99/P50 > 3 说明有排队。
7. **SGLang 的 RadixAttention 支持树状前缀共享**，在多轮对话、同文档多问题、Agent 多分支、Self-Consistency 场景比 vLLM 的线性前缀缓存收益大得多。
8. **多模型共卡时，`--gpu-memory-utilization` 是「占总显存比例」，多个进程会重复计算**。手工分配比例，总和 ≤ 0.85，先起小的再起大的。
9. **网关是必须的，不是可选的**。它解决换模型、降级、限流、Key 管理、成本统计、重试策略六件事，让业务代码与模型解耦。

### 6.2 自测题

**第 1 题**：你有一张 RTX 4090 24G，要部署 Qwen2.5-14B-Instruct-AWQ（权重 8.6 GB，KV 192 KB/token）。业务的输入长度分布是 P50=3200、P95=7100、P99.9=13500 token，输出平均 400 token，峰值并发 20。请给出完整的 vLLM 启动命令，并说明每个关键参数的取值理由。

<details>
<summary>参考答案</summary>

**启动命令**：

```bash
vllm serve /data/models/Qwen2.5-14B-Instruct-AWQ \
    --served-model-name huacheng-qa \
    --host 0.0.0.0 --port 8000 \
    --quantization awq_marlin \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --swap-space 4 \
    --disable-log-requests \
    --api-key sk-xxx
```

**逐参数理由**：

| 参数 | 取值 | 理由 |
|---|---|---|
| `--quantization awq_marlin` | — | 模型是 AWQ 量化版；指定 `awq_marlin` 而非 `awq`，确保走优化 kernel（速度差 2 倍） |
| `--gpu-memory-utilization` | **0.90** | 独占整卡，留 10% 余量防峰值。24 × 0.90 = 21.6 GB 可用 |
| `--max-model-len` | **16384** | P99.9 输入 13500 + 输出 400 = 13900，向上取到 2 的幂是 16384。设 32768 会让 KV Cache 需求翻倍但只多覆盖 0.1% 的请求 |
| `--max-num-seqs` | **32** | 峰值并发 20，留 60% 余量。显存校验：可用 KV 池 = 21.6 − 8.6（权重） − 1.5（激活） = 11.5 GB；按平均输入 3200 + 输出 400 = 3600 token 算，单请求 KV = 3600 × 192KB = 0.66 GB，11.5 / 0.66 ≈ **17 个满负载并发**。设 32 让调度器有腾挪空间，vLLM 会按实际长度动态分配 |
| `--max-num-batched-tokens` | **4096** | 开了 chunked-prefill，这个值控制每次迭代的 Prefill 块大小。设小一点（4096 而非 8192）让 Decode 更平滑，因为业务有 P95=7100 的长请求会阻塞 |
| `--enable-prefix-caching` | 开 | RAG 场景 system prompt 复用，TTFT 能降 30%+ |
| `--enable-chunked-prefill` | 开 | 输入长度方差大（3200 到 13500），必须防止长请求阻塞短请求 |
| `--swap-space` | 4 | 万一抢占，换出到 CPU 内存比重算便宜 |

**上线后要验证的**：
1. 启动日志的 `# GPU blocks` 是多少（应该在 700 左右，即 11200 token 容量... 实际要看日志）
2. 压测扫描找甜点区，如果 20 并发下 P99 > 3s，说明要么降 `max-model-len`，要么加卡
3. `prefix_cache_hit_rate` 是否 > 0.3
4. 有没有 `num_preemptions`，有的话降 `max-num-seqs`
</details>

**第 2 题**：压测报告显示：并发 8 时 TTFT P50=706ms、P99=1180ms；并发 32 时 TTFT P50=2840ms、P99=7210ms，但吞吐只从 456 tok/s 涨到 573 tok/s。业务要求 TTFT P99 < 2s。请分析问题并给出三条优化建议。

<details>
<summary>参考答案</summary>

**问题分析**：

1. **吞吐已经饱和**。并发从 8 涨到 32（4 倍），吞吐只涨 26%（456 → 573）。说明 GPU 算力/带宽在并发 8~16 附近就已经跑满了，再加并发只是在**排队**。
2. **延迟劣化严重**。并发 32 时 P99/P50 = 7210/2840 = 2.5，且绝对值 7.2 秒远超 2 秒的要求。
3. **根本原因是过载**。系统的服务能力（甜点区）大约在并发 8~16，业务把 32 个请求压进来，超出的部分全部在排队等 KV Cache 槽位。

**三条优化建议（按性价比排序）**：

**建议 1：在网关层限流到并发 12~16（立即见效，零成本）**

```python
# 网关层加信号量限流
SEMAPHORE = asyncio.Semaphore(14)   # 甜点区上限

async def call_llm(...):
    async with SEMAPHORE:
        return await backend_call(...)
```

超出的请求在网关排队（可控），而不是全部压到 vLLM 里（不可控）。同时给排队请求设一个队列超时（如 5 秒），超时就降级返回检索原文。

预期效果：P99 从 7.2s 降到约 2s（并发 16 时的水平），代价是高峰期部分请求排队或降级。

**建议 2：降低 Prefill 开销（中等成本，效果明显）**

- **开 Prefix Caching**（如果没开）：RAG 场景 system prompt 复用，TTFT 直接降 30%~50%
- **Rerank 后减少注入的 chunk**：从 Top-8 减到 Top-4，输入 token 减半 → Prefill 时间减半
- **检查 prompt 顺序**：固定内容前置，保证缓存命中

预期效果：TTFT 降 40%~60%，且不影响吞吐（反而提升，因为省了算力）。

**建议 3：扩容（成本最高，但是根本解法）**

计算需要几张卡：
- 当前单卡饱和吞吐 ≈ 573 输出 tok/s，甜点区并发 ≈ 14
- 如果业务峰值真的需要 32 并发，需要 `32 / 14 ≈ 2.3` → **3 张卡**
- 用 Little 定律验证：`并发数 = QPS × 平均响应时间`。如果峰值 QPS = 4、平均 E2E = 7.5s，需要并发 30，确实要 3 张卡

预期效果：P99 回到 1.2s 水平，代价是硬件成本 3 倍。

**决策顺序**：先做 1 和 2（几乎零成本，能解决问题就不用花钱），仍不达标再做 3。第 1.4 章会给完整的容量规划测算。
</details>

**第 3 题**：华成机电的 Agent 系统有这样的特征：每次对话 5~8 轮，每轮的 prompt = system(1200 token) + 工具定义(800 token) + 完整对话历史 + 新消息。到第 6 轮时 prompt 已经 6000+ token。请分析用 vLLM 和 SGLang 分别的 Prefill 开销，并说明该选哪个。

<details>
<summary>参考答案</summary>

**场景建模**（单个会话，6 轮）：

| 轮次 | 固定前缀 | 历史 | 新消息 | 总长度 |
|---|---|---|---|---|
| 1 | 2000 | 0 | 200 | 2200 |
| 2 | 2000 | 800 | 200 | 3000 |
| 3 | 2000 | 1600 | 200 | 3800 |
| 4 | 2000 | 2400 | 200 | 4600 |
| 5 | 2000 | 3200 | 200 | 5400 |
| 6 | 2000 | 4000 | 200 | 6200 |
| **合计** | | | | **25200 token** |

**vLLM（线性 Prefix Caching）**：

vLLM 的前缀缓存能匹配从头开始的最长公共前缀。在**同一个会话内**，第 N 轮的 prompt 前缀 = 第 N−1 轮的完整 prompt（因为历史是追加的）。**如果缓存还在**，vLLM 也能命中：

| 轮次 | 能复用 | 需新算 |
|---|---|---|
| 1 | 0 | 2200 |
| 2 | 2200 | 800 |
| 3 | 3000 | 800 |
| 4 | 3800 | 800 |
| 5 | 4600 | 800 |
| 6 | 5400 | 800 |
| **合计** | | **6200 token** |

理论上 vLLM 也能做到只算 6200（省 75%）。**但有两个前提条件**：
1. 缓存在两轮之间没有被 LRU 淘汰（高并发多会话时很容易被挤掉）
2. 会话的前缀严格一致（如果 prompt 里有时间戳等可变内容，全部失效）

**SGLang（RadixAttention 树状共享）**：

SGLang 的优势不在单会话（单会话是线性的，两者一样），而在**多会话共享**：

假设同时有 20 个会话在进行，它们共享同一个 system prompt + 工具定义（2000 token）：

| 方案 | Prefill 总量（20 会话 × 6 轮） |
|---|---|
| **无缓存** | 25200 × 20 = **504,000 token** |
| **vLLM 线性缓存**（假设缓存都命中） | 6200 × 20 = **124,000 token** |
| **SGLang RadixAttention** | 2000（全局共享一次）+ (25200−2000×6)/... 实际约 **86,000 token** |

SGLang 的额外收益来自：
1. **跨会话共享固定前缀**：2000 token 的 system + 工具定义只算一次，不是 20 次
2. **LPM 调度策略**：优先调度能命中最长前缀的请求，主动最大化复用
3. **树状结构对淘汰更友好**：共享节点被多个会话引用，不会被轻易淘汰

**结论：这个场景该选 SGLang**，理由：

| 判据 | 是否满足 |
|---|---|
| 多轮对话（前缀递增） | ✅ 5~8 轮 |
| 大量会话共享固定前缀 | ✅ system + 工具定义共 2000 token，所有会话都一样 |
| 前缀占比高 | ✅ 第 1 轮时固定前缀占 91% |
| Agent 多分支探索 | ✅ ReAct 会有分支 |

**但是**，如果有以下情况，还是留在 vLLM：
- 团队已经深度依赖 vLLM 的某些特性（特定量化格式、LoRA 热加载）
- 运维团队只熟悉 vLLM，迁移风险大于收益
- 并发会话数少（< 5），跨会话共享的收益不明显

**务实的做法**：在测试环境同时起两个，用真实的会话日志重放，实测 TTFT 和吞吐，用数字决定。迁移成本主要是启动参数的映射（见 2.5.3 的对照表），客户端代码因为都是 OpenAI 兼容接口，**基本不用改**。
</details>

---

**上一章** [1.2 模型全景图与选型方法论](./02-模型全景图与选型方法论.md) | **下一章** [1.4 算力配置与成本测算](./04-算力配置与成本测算.md)
