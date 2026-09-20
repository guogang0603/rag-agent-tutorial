# 第 5.4 章  LLaMA-Factory 工程化训练

> **本章目标**：读完能做到 …
> 1. 说清「自写训练脚本」和「训练框架」各自的边界，并给出团队什么时候该切换到框架的判断标准；
> 2. 按版本基线装好 LLaMA-Factory，并用 `llamafactory-cli env` 验证环境一致；
> 3. 把第 5.2 / 5.3 章产出的华成机电工单数据集正确注册进 `dataset_info.json`，并用自检脚本确认模板渲染没错；
> 4. 逐字段读懂一份生产可用的 `qwen2.5-7b lora sft` YAML，知道每个键改了会发生什么；
> 5. 用 CLI / WebUI / 多卡（DDP + DeepSpeed ZeRO）三种方式跑起同一份配置；
> 6. 独立完成一次 DPO 训练，用偏好数据把「回答风格」和「拒答行为」对齐；
> 7. 建立一套可复现的实验管理规范：命名、配置版本化、产物目录、实验记录表、复现 checklist。
>
> **前置知识**：
> - [第 5.1 章 微调原理与 PEFT 家族全解](./01-微调原理与PEFT家族全解.md)（知道 LoRA 的 `r` / `alpha` / `target_modules` 是什么）
> - [第 5.2 章 数据集构造与清洗](./02-数据集构造与清洗.md)（手上有清洗好的 SFT 数据）
> - [第 5.3 章 LoRA-QLoRA 实战](./03-LoRA-QLoRA实战.md)（**必须先手写过一遍训练脚本**，否则 YAML 对你就是天书）
> - [第 0.1 章 开发环境搭建](../00-前置准备/01-开发环境搭建（Python-CUDA-Docker）.md)（CUDA / PyTorch 已就绪）
>
> **预计用时**：阅读 70 分钟 / 动手 120 分钟（含一次完整训练）

---

## 一、为什么需要它（问题出发）

### 1.1 一个真实的团队翻车现场

华成机电的算法组有三个人。第 5.3 章那份手写训练脚本跑通之后，三周内发生了这些事：

- **小王**把脚本复制了一份，改成自己的路径，加了个 `--use_flash_attn`，文件名叫 `train_v2.py`；
- **小李**觉得数据加载太慢，把 `datasets.map` 换成了手写的 `Dataset.from_generator`，顺手把 `max_seq_length` 从 2048 改成 4096，文件名叫 `train_final.py`；
- **小张**要跑双卡，写了一个 `accelerate` 版本，叫 `train_ddp.py`，但忘了同步 label mask 的逻辑。

一个月后老板问：「上次那个效果最好的模型，是怎么训的？」

三个人翻聊天记录，找到一个叫 `output/lora-0912-2/` 的目录，里面有 `adapter_model.safetensors`，但没有训练配置、没有数据版本、没有 commit hash。**这个模型永远无法复现了。**

这就是本章要解决的问题。注意：**问题不是「脚本写得不好」，脚本写得挺好；问题是脚本没有被约束成一种可以被团队共享、审查、版本化的形态。**

### 1.2 脚本能跑通 ≠ 团队能复用

把差距列清楚：

| 能力 | 手写脚本的现状 | 团队真正需要的 |
|---|---|---|
| 配置管理 | 超参散落在代码里、命令行里、注释里 | 一个文件描述一次实验，可 diff、可 review、可进 git |
| 方法切换 | 想试 DPO 要重写一遍训练循环 | 改一个 `stage` 字段 |
| 多卡 | 每个人自己配 accelerate，配法不一样 | 统一的启动方式，DeepSpeed 配置文件共享 |
| 数据接入 | 每换一种数据格式就改一次 collator | 注册一次，多处复用 |
| 导出 | 手写 merge 脚本，容易漏 tokenizer | 一条命令产出可部署目录 |
| 评测 | 没有 | 内置通用基准 + 可接自建业务评测 |
| 可复现 | 靠人的记性 | 配置 + 数据版本 + 代码 commit 三件套 |
| 新人上手 | 读 300 行脚本 | 读 60 行 YAML |

**LLaMA-Factory 补的就是右边这一列。** 它不是「更好的训练算法」，它是**训练工程的标准化层**。

### 1.3 框架到底提供了什么

```mermaid
flowchart LR
    A["数据<br/>jsonl / csv / HF Hub"] --> B["dataset_info.json<br/>数据集注册表"]
    B --> C["训练 YAML<br/>一次实验 = 一个文件"]
    C --> D{"stage"}
    D -->|pt| E1["继续预训练"]
    D -->|sft| E2["指令微调"]
    D -->|rm| E3["奖励模型"]
    D -->|dpo / kto| E4["偏好对齐"]
    D -->|ppo| E5["强化学习"]
    E1 & E2 & E3 & E4 & E5 --> F["adapter<br/>output_dir/"]
    F --> G["export YAML"]
    G --> H1["合并权重<br/>可直接部署"]
    G --> H2["量化导出<br/>GPTQ / AWQ"]
    F --> I["内置评测<br/>ceval / cmmlu / mmlu"]
    H1 --> J["第 5.5 章：部署"]
    F --> K["第 5.6 章 / 第 8 模块：业务评测"]

    style C fill:#e8f4ff,stroke:#3a7fd5
    style F fill:#fff3e0,stroke:#e08a00
```

五件事：**统一配置**（YAML）、**多方法**（一个 `stage` 字段覆盖 6 种训练范式）、**WebUI**（给不写代码的同事）、**导出**（合并 + 量化）、**评测**（内置基准）。

### 1.4 本章的产出物

跑完本章，你的 `finetune/` 目录会长成这样，且**每一个文件都能进 git**：

```text
huacheng-llm/finetune/
├── configs/
│   ├── dataset_info.json              # 数据集注册表（唯一真源）
│   ├── sft_qwen25_7b_lora.yaml        # SFT 主配置
│   ├── dpo_qwen25_7b_lora.yaml        # DPO 对齐配置
│   ├── export_merge.yaml              # 合并导出
│   ├── export_gptq.yaml               # 量化导出
│   ├── eval_ceval.yaml                # 内置评测
│   └── ds/
│       ├── ds_z2_offload.json         # DeepSpeed ZeRO-2
│       └── ds_z3.json                 # DeepSpeed ZeRO-3
├── scripts/
│   ├── check_dataset.py               # 数据集注册自检
│   ├── new_run.sh                     # 新建一次实验（命名 + 快照）
│   └── pipeline.sh                    # 一键：数据 → 训练 → 导出 → 评测
└── runs/                              # 产物（.gitignore，只提交 manifest）
    └── 20260917-sft-qwen25_7b-r16-v3/
        ├── config.snapshot.yaml
        ├── manifest.json
        ├── adapter/
        └── logs/
```

---

## 二、安装与验证（版本锁定）

### 2.1 版本基线（本书统一，请照抄）

> **实测环境**：Ubuntu 22.04 / CUDA 12.1 / NVIDIA RTX 4090 24GB ×2 / Python 3.11。
> 下表是本章所有命令验证过的组合。**版本组合是训练框架最容易踩的坑**，不要随手 `pip install -U`。

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | 全书基线 |
| PyTorch | 2.4.x（cu121） | 与 CUDA 12.1 匹配 |
| transformers | 4.46.x | LLaMA-Factory 对 transformers 版本敏感 |
| peft | 0.13.x | 全书基线 |
| trl | 0.11.x | DPO / PPO 依赖 |
| accelerate | 1.0.x | 多卡启动 |
| deepspeed | 0.15.x | ZeRO-2 / ZeRO-3 |
| bitsandbytes | 0.44.x | QLoRA 4bit |
| flash-attn | 2.6.x（可选） | 显著提速，编译较慢 |
| LLaMA-Factory | **v0.9.x（锁 tag，不要用 main）** | 本章基准 |

> ⚠️ **为什么锁 tag**：LLaMA-Factory 迭代非常快，YAML 字段名在小版本间出现过改名/弃用。生产团队一律锁定一个 tag，升级走独立的验证流程。字段以你所装版本的 `examples/` 目录与官方文档为准，本章给出的是 v0.9.x 的写法。

### 2.2 安装

```bash
# 1) 独立虚拟环境（强烈建议：训练环境和推理环境分开，依赖冲突太多）
cd ~/huacheng-llm
uv venv .venv-train --python 3.11
source .venv-train/bin/activate

# 2) 先装匹配 CUDA 的 PyTorch（不要让 LLaMA-Factory 的依赖解析器替你选）
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 3) 拉源码并锁定 tag
cd ~/opt
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout v0.9.1          # ← 锁定版本，把这个 tag 写进团队文档
git rev-parse HEAD           # ← 记下 commit，写进实验 manifest

# 4) 可编辑安装 + 需要的 extras
uv pip install -e ".[torch,metrics,deepspeed,bitsandbytes]"
```

pip 等价命令：

```bash
python -m venv .venv-train && source .venv-train/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[torch,metrics,deepspeed,bitsandbytes]"
```

可选 extras 说明：

| extra | 装了给你什么 | 要不要装 |
|---|---|---|
| `torch` | 基础训练依赖 | 必装 |
| `metrics` | ROUGE / BLEU 等生成指标（`predict_with_generate` 用） | 建议装 |
| `bitsandbytes` | QLoRA 4bit / 8bit | 显存紧张就装 |
| `deepspeed` | ZeRO-2 / ZeRO-3 多卡 | 多卡装 |
| `gptq` / `awq` | 量化导出与加载 | 第 5.5 章用到 |
| `vllm` | 用 vLLM 做批量推理/评测 | 建议装在**推理环境**而不是训练环境 |
| `modelscope` | 从 ModelScope 拉模型（国内网络友好） | 国内建议装 |

### 2.3 验证环境

```bash
llamafactory-cli version
llamafactory-cli env
```

预期输出（关键几行，**版本号以你实际安装为准**）：

```text
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.1                |
----------------------------------------------------------

- `llamafactory` version: 0.9.1
- Platform: Linux-6.8.0-x86_64-with-glibc2.35
- Python version: 3.11.9
- PyTorch version: 2.4.1+cu121 (GPU)
- Transformers version: 4.46.1
- Datasets version: 3.0.1
- Accelerate version: 1.0.1
- PEFT version: 0.13.2
- TRL version: 0.11.4
- GPU type: NVIDIA GeForce RTX 4090
- DeepSpeed version: 0.15.4
- Bitsandbytes version: 0.44.1
```

> 💡 **把这段输出粘进实验记录表**。三个月后复现时，90% 的「跑不出同样结果」都能在这几行里找到原因。

### 2.4 国内网络：模型从哪儿拉

```bash
# 方案 A：ModelScope（推荐，国内速度快）
export USE_MODELSCOPE_HUB=1
# 此时 model_name_or_path 写 ModelScope 的 model id，例如 qwen/Qwen2.5-7B-Instruct

# 方案 B：HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案 C（生产推荐）：提前下载到本地统一目录，YAML 里写绝对路径
# /models/Qwen2.5-7B-Instruct
```

**生产环境一律用方案 C。** 训练时从网上拉模型意味着：网络抖动会让你的三小时训练在第 5 分钟失败，而且不同机器上拉到的可能是不同的 revision。

---

## 三、数据集注册：`dataset_info.json` 完全指南

### 3.1 为什么要"注册"

框架不认识你的 jsonl。它需要知道三件事：

1. **文件在哪**（`file_name` / `hf_hub_url` / `ms_hub_url`）；
2. **是什么格式**（`formatting`：alpaca 还是 sharegpt）；
3. **字段怎么映射**（`columns` / `tags`）。

这三件事写在 `dataset_info.json` 里。它位于 `dataset_dir` 指定的目录下（默认是 LLaMA-Factory 源码里的 `data/`）。

> **本书约定**：把 `dataset_info.json` 放在项目自己的 `finetune/configs/` 下，训练时用 `dataset_dir` 指过去。**不要改 LLaMA-Factory 源码目录里的文件**，否则升级 tag 时你的修改全丢。

### 3.2 两种主要格式

**Alpaca 格式** —— 适合单轮指令，字段扁平：

```json
{
  "instruction": "XJ-200 报 E043，怎么处理？",
  "input": "设备已运行 3 年，上周刚更换过润滑油。",
  "output": "【故障判断】E043 为主轴温升超限告警……",
  "system": "你是华成机电售后技术支持助手。"
}
```

**ShareGPT 格式** —— 适合多轮对话与带工具调用的数据，是本书主推：

```json
{
  "messages": [
    {"role": "system", "content": "你是华成机电售后技术支持助手。"},
    {"role": "user", "content": "XJ-200 报 E043，怎么处理？"},
    {"role": "assistant", "content": "【故障判断】E043 为主轴温升超限告警……"},
    {"role": "user", "content": "换了油还是报，怎么办？"},
    {"role": "assistant", "content": "【故障判断】排除润滑因素后……"}
  ]
}
```

选型建议：

| 你的数据长这样 | 用哪种 | 理由 |
|---|---|---|
| 单轮问答，有独立的「上下文」字段 | alpaca | `input` 天然放检索到的资料 |
| 多轮对话 / 客服转写 | **sharegpt** | 轮次结构清晰 |
| 含 RAG 上下文的 SFT | sharegpt（把上下文拼进 user 或 system） | 和线上推理时的消息结构一致 |
| 含 Function Calling | sharegpt（带 `tools` / `function_call`） | alpaca 表达不了 |
| 偏好数据（DPO/RM） | 两种都支持，需加 `ranking: true` | 见 7.2 节 |

> **一致性原则（重要）**：**训练时的消息结构必须和线上推理时完全一致。** 线上是 `system + user(含检索上下文)`，训练就必须也是 `system + user(含检索上下文)`。第 5.3 章已经强调过 label mask，这里强调的是**结构对齐**——两者都做错的话，训出来的模型上线就崩。

### 3.3 `dataset_info.json` 字段全表

| 字段 | 类型 | 作用 | 备注 |
|---|---|---|---|
| `file_name` | str | 相对 `dataset_dir` 的文件名，支持 `.json` / `.jsonl` / `.csv` / 目录 | 与 `hf_hub_url` 二选一 |
| `hf_hub_url` | str | 从 HuggingFace Hub 拉 | — |
| `ms_hub_url` | str | 从 ModelScope 拉 | 需 `USE_MODELSCOPE_HUB=1` |
| `formatting` | str | `alpaca`（默认）或 `sharegpt` | — |
| `ranking` | bool | 是否偏好数据（DPO / RM 用） | 默认 `false` |
| `subset` / `split` | str | HF 数据集的子集与切分 | — |
| `num_samples` | int | 采样条数 | 也可在 YAML 用 `max_samples` |
| `columns` | dict | 列名映射 | 见下 |
| `tags` | dict | ShareGPT 的角色标签映射 | 仅 sharegpt 需要 |

`columns` 常用键：

| 键 | alpaca | sharegpt | 含义 |
|---|---|---|---|
| `prompt` | ✅ | — | 对应 `instruction` |
| `query` | ✅ | — | 对应 `input` |
| `response` | ✅ | — | 对应 `output` |
| `history` | ✅ | — | 历史轮列表 |
| `system` | ✅ | ✅ | 系统提示列 |
| `messages` | — | ✅ | 对话列表列名 |
| `tools` | — | ✅ | 工具描述列 |
| `chosen` | ✅ | ✅ | 偏好数据：更优回复 |
| `rejected` | ✅ | ✅ | 偏好数据：更差回复 |
| `kto_tag` | ✅ | ✅ | KTO 的布尔标签 |

`tags`（sharegpt 专用）：

| 键 | 默认值 | 含义 |
|---|---|---|
| `role_tag` | `from` | 角色字段名 |
| `content_tag` | `value` | 内容字段名 |
| `user_tag` | `human` | 用户角色值 |
| `assistant_tag` | `gpt` | 助手角色值 |
| `system_tag` | `system` | 系统角色值 |
| `observation_tag` | `observation` | 工具返回 |
| `function_tag` | `function_call` | 工具调用 |

> 我们的数据用的是 OpenAI 风格的 `role` / `content` / `user` / `assistant`，所以 `tags` 必须显式覆盖默认值，否则框架会按 `from` / `value` 去找，直接报「找不到列」。

### 3.4 完整示例：接入第 5.3 章产出的华成工单数据集

假设第 5.2 / 5.3 章已经产出了这几个文件（路径沿用全书约定）：

```text
huacheng-llm/data/sft/
├── huacheng_sft_train.jsonl     # 训练集，ShareGPT 格式
├── huacheng_sft_dev.jsonl       # 开发集
├── huacheng_rag_sft.jsonl       # RAG-aware SFT（user 里带检索上下文）
└── huacheng_dpo.jsonl           # 偏好数据（本章第七节构造）
```

`finetune/configs/dataset_info.json`：

```json
{
  "huacheng_sft_train": {
    "file_name": "huacheng_sft_train.jsonl",
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

  "huacheng_sft_dev": {
    "file_name": "huacheng_sft_dev.jsonl",
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

  "huacheng_rag_sft": {
    "file_name": "huacheng_rag_sft.jsonl",
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

  "huacheng_dpo": {
    "file_name": "huacheng_dpo.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  },

  "identity_huacheng": {
    "file_name": "identity_huacheng.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

最后那个 `identity_huacheng` 是**身份数据**——大概 60~100 条，教模型回答「你是谁」「你能做什么」。不加的话，模型上线后被客户问「你是不是 ChatGPT」会胡说。样例：

```json
{"messages": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "我是华成机电售后技术支持助手，可以帮你查设备故障处理方案、备件信息和保修政策。请告诉我设备型号和现象。"}]}
{"messages": [{"role": "user", "content": "你能做什么？"}, {"role": "assistant", "content": "我主要做三件事：1) 根据设备型号与报错码给出排查步骤；2) 查询备件型号与库存；3) 判断是否在保修期内。超出售后范围的问题我会如实告诉你。"}]}
```

### 3.5 注册自检脚本（**强烈建议每次改数据后都跑**）

注册配错最典型的表现是：**不报错，但训出来的模型是废的**。比如 `tags` 没配对，框架把整条对话当成一个 user 轮，assistant 内容全被当输入，等于没有 label。

写一个自检脚本，`finetune/scripts/check_dataset.py`：

```python
"""校验 dataset_info.json 注册是否正确，并打印模板渲染后的真实训练文本。"""
import argparse
import json
import os
import sys

from transformers import AutoTokenizer


def load_registry(config_path: str) -> dict:
    """读取 dataset_info.json。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_file(entry: dict, data_dir: str) -> str:
    """确认数据文件存在并返回绝对路径。"""
    if "file_name" not in entry:
        raise ValueError("该条目没有 file_name（可能用的是 hub_url，本脚本不校验）")
    path = os.path.join(data_dir, entry["file_name"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在：{path}")
    return path


def read_samples(path: str, n: int) -> list:
    """读取前 n 条样本。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for i, line in enumerate(f):
                if i >= n:
                    break
                rows.append(json.loads(line))
        else:
            rows = json.load(f)[:n]
    return rows


def check_sharegpt(entry: dict, row: dict) -> list:
    """按 columns/tags 把一条样本解析成 OpenAI 风格消息列表。"""
    cols = entry.get("columns", {})
    tags = entry.get("tags", {})
    msg_col = cols.get("messages", "conversations")
    role_tag = tags.get("role_tag", "from")
    content_tag = tags.get("content_tag", "value")

    if msg_col not in row:
        raise KeyError(f"样本里没有列 '{msg_col}'，实际的键是 {list(row.keys())}")

    msgs = []
    for turn in row[msg_col]:
        if role_tag not in turn or content_tag not in turn:
            raise KeyError(
                f"轮次里没有 '{role_tag}'/'{content_tag}'，实际的键是 {list(turn.keys())}"
            )
        msgs.append({"role": turn[role_tag], "content": turn[content_tag]})
    return msgs


def main() -> None:
    """入口：逐个数据集做存在性、字段映射、模板渲染三项检查。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/configs/dataset_info.json")
    ap.add_argument("--data_dir", default="data/sft")
    ap.add_argument("--model", default="/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    registry = load_registry(args.config)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    names = args.datasets or list(registry.keys())
    failed = 0

    for name in names:
        entry = registry[name]
        print("=" * 72)
        print(f"数据集：{name}   formatting={entry.get('formatting', 'alpaca')} "
              f"ranking={entry.get('ranking', False)}")
        try:
            path = check_file(entry, args.data_dir)
            rows = read_samples(path, args.n)
            print(f"  文件：{path}  （抽样 {len(rows)} 条）")

            if entry.get("formatting") == "sharegpt":
                for i, row in enumerate(rows):
                    msgs = check_sharegpt(entry, row)
                    roles = [m["role"] for m in msgs]
                    print(f"  [{i}] 轮次角色序列：{roles}")
                    if entry.get("ranking"):
                        cols = entry.get("columns", {})
                        for key in ("chosen", "rejected"):
                            col = cols.get(key, key)
                            assert col in row, f"偏好数据缺少列 {col}"
                        print(f"       chosen 长度={len(str(row[cols.get('chosen','chosen')]))} "
                              f"rejected 长度={len(str(row[cols.get('rejected','rejected')]))}")
                    else:
                        assert roles[-1] == "assistant", "最后一轮必须是 assistant，否则没有 label"
                    if i == 0:
                        text = tok.apply_chat_template(msgs, tokenize=False)
                        print("  ---- 模板渲染结果（截断 800 字符）----")
                        print(text[:800])
                        print("  --------------------------------------")
                        print(f"  token 数：{len(tok(text)['input_ids'])}")
            else:
                print(f"  [alpaca] 首条键：{list(rows[0].keys())}")
            print("  ✅ 通过")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ❌ 失败：{type(exc).__name__}: {exc}")

    print("=" * 72)
    print(f"共检查 {len(names)} 个数据集，失败 {failed} 个")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

运行与预期输出：

```bash
python finetune/scripts/check_dataset.py \
  --config finetune/configs/dataset_info.json \
  --data_dir data/sft \
  --model /models/Qwen2.5-7B-Instruct \
  --datasets huacheng_sft_train huacheng_dpo
```

```text
========================================================================
数据集：huacheng_sft_train   formatting=sharegpt ranking=False
  文件：data/sft/huacheng_sft_train.jsonl  （抽样 2 条）
  [0] 轮次角色序列：['system', 'user', 'assistant']
  ---- 模板渲染结果（截断 800 字符）----
<|im_start|>system
你是华成机电售后技术支持助手。回答必须包含【故障判断】【排查步骤】【所需备件】【保修判定】四段。<|im_end|>
<|im_start|>user
XJ-200 主轴运行 20 分钟后报 E043，现场温度 38℃。<|im_end|>
<|im_start|>assistant
【故障判断】E043 为主轴温升超限告警……<|im_end|>
  --------------------------------------
  token 数：512
  [1] 轮次角色序列：['system', 'user', 'assistant', 'user', 'assistant']
  ✅ 通过
========================================================================
数据集：huacheng_dpo   formatting=sharegpt ranking=True
  文件：data/sft/huacheng_dpo.jsonl  （抽样 2 条）
  [0] 轮次角色序列：['system', 'user']
       chosen 长度=286 rejected 长度=241
  [1] 轮次角色序列：['system', 'user']
       chosen 长度=198 rejected 长度=176
  ✅ 通过
========================================================================
共检查 2 个数据集，失败 0 个
```

注意两点：
- SFT 数据的角色序列**必须以 `assistant` 结尾**；
- DPO 数据的 `conversations` **不含最后一轮 assistant**（它在 `chosen` / `rejected` 里）。

---

## 四、YAML 配置全解

### 4.1 一份可直接用的 SFT 配置

`finetune/configs/sft_qwen25_7b_lora.yaml`——这是本章的核心资产，**每一行都有注释**：

```yaml
# ============================================================================
# 华成机电售后助手 —— SFT（LoRA）训练配置
# 实验名：20260917-sft-qwen25_7b-r16-v3
# 框架：LLaMA-Factory v0.9.1   数据：data/sft/huacheng_sft_train.jsonl (v3)
# ============================================================================

### ---------------------------------------------------------------- model 段
model_name_or_path: /models/Qwen2.5-7B-Instruct   # 基座模型本地绝对路径（生产必须本地路径）
trust_remote_code: true                            # Qwen 系列需要，允许执行仓库里的建模代码
# adapter_name_or_path:                            # 续训/在已有 adapter 上继续训时填这里
flash_attn: auto                                   # auto / disabled / sdpa / fa2；装了 flash-attn 才能用 fa2
# rope_scaling: linear                             # 需要外推超过原生上下文长度时才开
# use_unsloth: false                               # unsloth 加速，单卡有效，多卡/自定义场景兼容性差
print_param_status: false                          # 调试用：打印每个参数 requires_grad 状态

### --------------------------------------------------------------- method 段
stage: sft                     # 训练阶段：pt / sft / rm / ppo / dpo / kto
do_train: true                 # 训练开关（评测时改成 do_eval / do_predict）
finetuning_type: lora          # full / freeze / lora；lora 是本书默认
lora_rank: 16                  # LoRA 的 r，容量旋钮。数据 <2k 条建议 8~16
lora_alpha: 32                 # 缩放系数，经验值 = 2 * lora_rank
lora_dropout: 0.05             # 防过拟合；数据少于 1000 条可提到 0.1
lora_target: all               # "all" = 自动挂到全部线性层；也可写 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
# use_rslora: true             # r 较大（>=32）时开启，缩放改为 alpha/sqrt(r)，训练更稳
# use_dora: false              # DoRA：幅度/方向分解，效果略好但更慢
# pissa_init: false            # PiSSA 初始化，收敛更快，但与量化组合需谨慎
# additional_target: embed_tokens,lm_head   # 扩了词表才需要，否则别开（显存暴涨）

train_on_prompt: false         # ★ 关键：false = 只对 assistant 段算 loss（等价于手写的 labels mask）
mask_history: false            # true = 多轮时只训最后一轮；false = 每一轮 assistant 都算 loss

### -------------------------------------------------------- 量化段（QLoRA）
# 24G 单卡跑 7B 时打开下面四行；48G+ 或多卡建议关掉，bf16 LoRA 更快更准
# quantization_bit: 4
# quantization_method: bitsandbytes   # bitsandbytes / hqq / eetq
# double_quantization: true
# quantization_type: nf4

### -------------------------------------------------------------- dataset 段
dataset_dir: /home/ops/huacheng-llm/data/sft          # dataset_info.json 与数据文件所在目录
dataset: huacheng_sft_train,identity_huacheng         # 多数据集用英文逗号连接，会被混合
# interleave_probs: 0.9,0.1                            # 配 mix_strategy=interleave_* 时的采样概率
# mix_strategy: concat                                 # concat（默认，直接拼）/ interleave_under / interleave_over
template: qwen                                        # ★ 必须与基座模型匹配：qwen / llama3 / chatml / deepseek ...
cutoff_len: 2048                                      # 单条样本最大 token 数，超长会被截断
max_samples: 100000                                   # 每个数据集最多取多少条，调试时设 100 可秒级验证流程
overwrite_cache: true                                 # 改了数据就必须 true，否则读到旧缓存
preprocessing_num_workers: 8                          # 数据预处理进程数，建议 = CPU 核数一半
packing: false                                        # true = 多条样本拼成一条填满 cutoff_len，吞吐高但会跨样本污染注意力
# neat_packing: false                                  # packing 的改良版，避免跨样本注意力泄漏
# tokenized_path: ./cache/tokenized_v3                 # 预处理结果落盘，多次实验复用可省大量时间

### --------------------------------------------------------------- output 段
output_dir: /home/ops/huacheng-llm/finetune/runs/20260917-sft-qwen25_7b-r16-v3/adapter
logging_steps: 10                # 每多少步打一次 loss
save_steps: 200                  # 每多少步存一次 checkpoint
save_total_limit: 3              # 最多保留几个 checkpoint（不设会把盘写满）
plot_loss: true                  # 训完自动画 loss 曲线 png
overwrite_output_dir: false      # ★ 生产建议 false：防止手滑覆盖上一次的产物
report_to: tensorboard           # none / tensorboard / wandb / swanlab
# run_name: 20260917-sft-r16-v3  # 给 wandb/swanlab 看的实验名

### ---------------------------------------------------------------- train 段
per_device_train_batch_size: 1      # 单卡 batch，24G 跑 7B + 2048 长度基本只能 1
gradient_accumulation_steps: 16     # 有效 batch = 1 * 16 * 卡数
learning_rate: 1.0e-4               # LoRA 常用 1e-4 ~ 2e-4；全参微调要低一个量级
num_train_epochs: 3.0               # 数据 <3k 条一般 2~3 轮；多了会过拟合
lr_scheduler_type: cosine           # cosine / linear / constant_with_warmup
warmup_ratio: 0.03                  # 前 3% 步数热身
weight_decay: 0.01
max_grad_norm: 1.0                  # 梯度裁剪，防 loss 尖刺
bf16: true                          # A100/4090 等 Ampere+ 用 bf16；老卡用 fp16 并配 loss scaling
gradient_checkpointing: true        # 用时间换显存，7B 上大约省 30~40% 激活显存
seed: 42                            # ★ 可复现的第一要素
ddp_timeout: 180000000              # 多卡时防止长时间预处理触发 NCCL 超时
# neftune_noise_alpha: 5            # 给 embedding 加噪声，小数据集上有时能提泛化，需 AB 验证
# resume_from_checkpoint: /path/to/checkpoint-400   # 断点续训

### ----------------------------------------------------------------- eval 段
eval_dataset: huacheng_sft_dev      # 开发集（也可用 val_size 从训练集里切）
# val_size: 0.05                    # 与 eval_dataset 二选一，别同时用
per_device_eval_batch_size: 1
eval_strategy: steps                # no / steps / epoch
eval_steps: 200                     # 与 save_steps 对齐，便于 load_best
load_best_model_at_end: true        # 训完自动回到 eval_loss 最低的那个 checkpoint
metric_for_best_model: eval_loss
greater_is_better: false
# predict_with_generate: false      # do_predict 时开启，会跑真实生成并算 ROUGE/BLEU（慢）
```

### 4.2 六个段落，各自在管什么

| 段 | 管什么 | 改错的典型后果 |
|---|---|---|
| **model** | 基座是谁、怎么加载 | `template` 与模型不匹配 → 训练 loss 正常但推理全是乱码 |
| **method** | 用什么训练范式、挂多少可训参数 | `train_on_prompt: true` → 模型学会复述问题 |
| **dataset** | 数据从哪来、怎么切 | `overwrite_cache: false` + 改了数据 → 训的还是旧数据 |
| **output** | 产物落哪、日志给谁 | `overwrite_output_dir: true` → 上一次的好模型被冲掉 |
| **train** | 学多快、学多久 | `learning_rate` 高一个量级 → loss 震荡 / 通用能力崩 |
| **eval** | 什么时候验证、留哪个 | 不设 `eval_dataset` → 你无法判断是否过拟合 |

### 4.3 最容易配错的 10 个字段

| 字段 | 常见错法 | 正确做法 | 怎么发现 |
|---|---|---|---|
| `template` | 用 Qwen2.5 却写 `default` / `chatml` | Qwen2.5 写 `qwen` | 用 `check_dataset.py` 看渲染文本里有没有 `<\|im_start\|>` |
| `train_on_prompt` | 保持默认没管 | 显式写 `false` | 训完让模型复述问题它就复述 |
| `lora_target` | 只写 `q_proj,v_proj` 却期待大改动 | 需要改行为就写 `all` | trainable% 只有 0.05% |
| `cutoff_len` | 设太小，长工单被截断 | 用脚本统计 p95 token 数再定 | 训练日志里大量样本被截 |
| `overwrite_cache` | 忘了设 true | 改数据后必须 true | loss 曲线和上一次一模一样 |
| `dataset_dir` | 指到了 LLaMA-Factory 源码目录 | 指到项目自己的目录 | 升级 tag 后数据集"消失" |
| `save_total_limit` | 不设 | 设 2~3 | 磁盘写满，训练中断 |
| `bf16` / `fp16` | 老卡开 bf16 | V100 及更早用 fp16 | 报 `bf16 is not supported` |
| `gradient_accumulation_steps` | 多卡时忘了除以卡数 | 有效 batch = bs × ga × 卡数，换卡数要同步调 | 双卡训出来和单卡效果不一样 |
| `quantization_bit` | 显存够也开 4bit | 够就别开，bf16 更快更准 | 训练速度莫名很慢 |

### 4.4 命令行覆盖：一份基线 + 多组实验

不要为每次消融实验复制一份 YAML。LLaMA-Factory 支持命令行参数覆盖 YAML：

```bash
# 基线
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml

# 消融 1：把 r 从 16 提到 32（其余不变）
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --lora_rank 32 --lora_alpha 64 \
  --output_dir finetune/runs/20260917-sft-qwen25_7b-r32-v3/adapter

# 消融 2：关掉身份数据，看通用能力是否变化
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --dataset huacheng_sft_train \
  --output_dir finetune/runs/20260917-sft-qwen25_7b-noid-v3/adapter
```

> ⚠️ 但**覆盖的参数必须回写进产物目录的快照**，否则半年后你只看到一份 YAML，不知道那次实际跑的是 r=32。第十一节的 `new_run.sh` 会自动做这件事。

---

## 五、三种训练方式：CLI / WebUI / 多卡

### 5.1 单卡 CLI

```bash
cd ~/huacheng-llm
source .venv-train/bin/activate

# 先用 100 条数据跑通流程（5 分钟内出结果），别一上来就跑三小时
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --max_samples 100 --num_train_epochs 1 --save_steps 1000 \
  --output_dir /tmp/lf-smoke

# 冒烟通过后跑正式训练，日志落盘
CUDA_VISIBLE_DEVICES=0 nohup llamafactory-cli train \
  finetune/configs/sft_qwen25_7b_lora.yaml \
  > finetune/runs/20260917-sft-qwen25_7b-r16-v3/logs/train.log 2>&1 &

# 跟日志
tail -f finetune/runs/20260917-sft-qwen25_7b-r16-v3/logs/train.log
```

预期输出（关键几行，**示例性数据**）：

```text
[INFO|tokenization_utils_base.py] loading file vocab.json
[INFO|llamafactory.data.loader] Loading dataset huacheng_sft_train.jsonl...
Converting format of dataset: 100%|██████████| 3120/3120 [00:00<00:00, 18442.11 examples/s]
Running tokenizer on dataset: 100%|██████████| 3120/3120 [00:07<00:00, 415.66 examples/s]
training example:
input_ids:
[151644, 8948, 198, 105043, 100165, ...]
inputs:
<|im_start|>system
你是华成机电售后技术支持助手。...<|im_end|>
<|im_start|>user
XJ-200 主轴运行 20 分钟后报 E043...<|im_end|>
<|im_start|>assistant
【故障判断】E043 为主轴温升超限告警...<|im_end|>
label_ids:
[-100, -100, -100, ..., 30534, 101158, ...]
labels:
【故障判断】E043 为主轴温升超限告警...<|im_end|>

[INFO|llamafactory.model.loader] trainable params: 40,370,176 || all params: 7,655,986,688 || trainable%: 0.5273
{'loss': 1.4821, 'grad_norm': 0.61, 'learning_rate': 9.87e-05, 'epoch': 0.05}
{'loss': 0.8933, 'grad_norm': 0.44, 'learning_rate': 9.21e-05, 'epoch': 0.51}
{'eval_loss': 0.7814, 'eval_runtime': 41.2, 'epoch': 1.03}
{'loss': 0.6210, 'grad_norm': 0.39, 'learning_rate': 5.02e-05, 'epoch': 1.54}
{'eval_loss': 0.7102, 'eval_runtime': 41.5, 'epoch': 2.05}
...
***** train metrics *****
  epoch                    =       3.0
  train_loss               =    0.6842
  train_runtime            = 2:47:31.08
Figure saved at: .../adapter/training_loss.png
```

> ★ **`training example` 那一段是免费的自检**。框架会把第一条样本的 `inputs` 和 `labels` 打出来。**每次训练都去看这一段**：`labels` 里只应该有 assistant 的内容，前面全是 `-100`。这一眼能省你三小时。

### 5.2 WebUI（给不写代码的同事用）

```bash
# 默认 7860 端口；本书端口约定里 7860 未被占用
GRADIO_SERVER_PORT=7860 llamafactory-cli webui

# 需要远程访问时（注意：WebUI 没有鉴权，只能在内网开）
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

WebUI 的正确用法和错误用法：

| 场景 | 用 WebUI？ | 说明 |
|---|---|---|
| 业务同事想快速试一批数据 | ✅ | 不用教他们命令行 |
| 调试模板渲染、看 chat 效果 | ✅ | Chat 标签页可以直接挂 adapter 对话 |
| 探索超参，快速改几个值 | ✅ | 但改完要用「预览命令」按钮导出 CLI 命令 |
| 正式训练 | ❌ | 浏览器一关/网络一断，训练可能中断 |
| 生产流水线 | ❌ | 无法进 git、无法 code review |

> **团队纪律**：WebUI 只用于探索，探索结束后**必须**点「预览命令」把参数导成 CLI/YAML，进 git 再正式训练。否则你又回到了 1.1 节那个翻车现场。

### 5.3 多卡训练

#### 5.3.1 单机多卡 DDP（最简单，显存够就用它）

```bash
# 两张 4090，数据并行。FORCE_TORCHRUN=1 让 CLI 用 torchrun 启动
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --gradient_accumulation_steps 8   # ← 卡数翻倍，ga 减半，保持有效 batch 不变
```

**DDP 的前提**：单卡必须放得下一份完整模型。7B bf16 LoRA + gradient checkpointing 在 24G 上刚好可以，14B 就不行了。

#### 5.3.2 DeepSpeed ZeRO-2（单卡放不下优化器状态时）

`finetune/configs/ds/ds_z2_offload.json`：

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "round_robin_gradients": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --deepspeed finetune/configs/ds/ds_z2_offload.json \
  --gradient_accumulation_steps 8
```

#### 5.3.3 DeepSpeed ZeRO-3（模型本身单卡放不下）

`finetune/configs/ds/ds_z3.json`：

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

> ★ `stage3_gather_16bit_weights_on_model_save: true` 不设的话，保存下来的权重是分片的，**导出时会失败**。这是 ZeRO-3 最经典的坑。

#### 5.3.4 三种并行方式选型

| 方式 | 适用 | 显存 | 速度 | 复杂度 |
|---|---|---|---|---|
| 单卡 | 7B LoRA / QLoRA | 24G 起 | 基准 | ★ |
| DDP | 7B LoRA，多卡加速 | 每卡都要放全模型 | 接近线性加速 | ★★ |
| ZeRO-2 (+offload) | 7B 全参 / 14B LoRA | 优化器状态分片到多卡或 CPU | 略慢于 DDP | ★★★ |
| ZeRO-3 | 32B+ / 单卡放不下模型 | 参数也分片 | 明显慢（通信多） | ★★★★ |

> **经验法则**：**能 DDP 就别 DeepSpeed，能 ZeRO-2 就别 ZeRO-3。** ZeRO-3 的通信开销在 PCIe 机器上非常伤，没有 NVLink 的双 4090 上有时比单卡还慢。

#### 5.3.5 多卡吞吐参考（示例性数据）

> **实测环境**：Qwen2.5-7B-Instruct，LoRA r=16 target=all，`cutoff_len=2048`，`gradient_checkpointing=true`，bf16，3120 条样本 × 3 epoch。**该表为示例性数据，你的机器会有明显差异，请自行复现。**

| 配置 | 卡 | 有效 batch | 峰值显存/卡 | 单 epoch 耗时 |
|---|---|---|---|---|
| 单卡 bf16 LoRA | 1×4090 | 16 | ~21 GB | ~56 min |
| 单卡 QLoRA 4bit | 1×4090 | 16 | ~13 GB | ~78 min |
| DDP | 2×4090 | 16 | ~21 GB | ~31 min |
| ZeRO-2 + CPU offload | 2×4090 | 16 | ~17 GB | ~38 min |
| ZeRO-3 | 2×4090 | 16 | ~12 GB | ~62 min |

可以看到：QLoRA 省显存但更慢（反量化有开销）；ZeRO-3 在双卡 PCIe 机器上是负优化。**别迷信"更高级的并行 = 更快"。**

### 5.4 断点续训

```bash
llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  --resume_from_checkpoint finetune/runs/20260917-sft-qwen25_7b-r16-v3/adapter/checkpoint-400
```

注意：续训要求 **数据、随机种子、batch 配置完全一致**，否则恢复出来的优化器状态和当前数据流对不上，loss 会跳变。

---

## 六、六种训练阶段：pt / sft / rm / ppo / dpo / kto

### 6.1 全景对照表

| stage | 中文名 | 数据形态 | 目标 | 华成机电用在哪 | 相对成本 |
|---|---|---|---|---|---|
| `pt` | 继续预训练（Continual Pre-training） | 纯文本 | 灌入领域语料、补术语分布 | 340 个型号的手册全文（可选，一般不做） | 高（数据量要大） |
| `sft` | 指令微调（Supervised Fine-Tuning） | 指令-回复对 | 学会任务与格式 | **主力**：工单结构化抽取、四段式回答 | 中 |
| `rm` | 奖励模型（Reward Model） | 偏好对 | 训一个打分器 | 仅当要做 PPO 时 | 中 |
| `ppo` | 近端策略优化 | prompt + 奖励模型 | 在线 RL 对齐 | **不建议**：显存翻倍、不稳定 | 很高 |
| `dpo` | 直接偏好优化（Direct Preference Optimization） | 偏好对 | 离线对齐风格与行为 | **主力**：拒答行为、回答风格 | 中 |
| `kto` | KTO（Kahneman-Tversky Optimization） | 单条 + 好/坏布尔标签 | 对齐，但不需要成对数据 | 只有点赞/点踩日志时用 | 中 |

### 6.2 各阶段的 YAML 差异

只列**相对 SFT 配置要改的行**：

**pt（继续预训练）**：
```yaml
stage: pt
dataset: huacheng_manual_corpus   # 数据是纯文本，dataset_info.json 里用 columns: {"prompt": "text"}
cutoff_len: 4096
packing: true                     # 预训练必开，否则短文本浪费算力
learning_rate: 5.0e-5             # 比 SFT 低
num_train_epochs: 1.0             # 通常 1 轮
# 没有 eval_dataset 也行，看 train_loss 即可
```

**rm（奖励模型）**：
```yaml
stage: rm
dataset: huacheng_dpo             # 复用同一份偏好数据（ranking: true）
learning_rate: 1.0e-5             # RM 对学习率敏感，要低
num_train_epochs: 1.0
# 注意：RM 会在模型上加一个 value head，导出方式与 SFT 不同
```

**ppo（不建议在生产用，但要知道怎么配）**：
```yaml
stage: ppo
reward_model: finetune/runs/rm-v1/adapter      # 上一步训好的 RM
reward_model_type: lora                        # lora / full / api
dataset: huacheng_ppo_prompts                  # 只要 prompt，不要 response
learning_rate: 1.0e-5
ppo_epochs: 4
ppo_buffer_size: 1
top_k: 0
top_p: 0.9
```

**dpo（本章重点，详见第七节）**：
```yaml
stage: dpo
dataset: huacheng_dpo
pref_beta: 0.1
pref_loss: sigmoid
learning_rate: 5.0e-6
```

**kto**：
```yaml
stage: kto
dataset: huacheng_kto             # dataset_info.json 里配 columns.kto_tag
pref_beta: 0.1
```

### 6.3 一条实用的路线建议

```mermaid
flowchart TD
    A["有 1000~5000 条<br/>业务问答对"] --> B["stage: sft<br/>LoRA r=16"]
    B --> C{"评测：格式对了吗？<br/>业务准确率够吗？"}
    C -->|不够| D["回去补数据<br/>第 5.2 章"]
    D --> B
    C -->|够了| E{"还有什么问题？"}
    E -->|"该拒答的没拒<br/>风格不对"| F["收集偏好对<br/>stage: dpo"]
    E -->|"知识不对"| G["别微调了<br/>去做 RAG<br/>第 5.6 章"]
    E -->|"都 OK"| H["导出 → 部署<br/>第 5.5 章"]
    F --> I{"再评测"}
    I -->|通用能力掉了| J["降 pref_beta<br/>混入 SFT 数据<br/>pref_ftx"]
    I -->|OK| H

    style B fill:#e8f4ff,stroke:#3a7fd5
    style F fill:#fff3e0,stroke:#e08a00
    style G fill:#ffe8e8,stroke:#d53a3a
```

**不要一上来就 DPO。** DPO 是在 SFT 的基础上做微调，它调的是"在两个都能接受的回答里选哪个"，如果 SFT 都没训好，DPO 只会把废答案和更废的答案做对比。

---

## 七、DPO 实战：把「回答风格」和「拒答行为」对齐

### 7.1 为什么 SFT 之后还需要 DPO

SFT 教的是「**应该说什么**」，DPO 教的是「**在两个都像样的回答里，我们更喜欢哪个**」。

华成机电的真实需求，SFT 之后仍然存在的两类问题：

| 问题 | 现象 | SFT 为什么治不好 | DPO 怎么治 |
|---|---|---|---|
| **回答风格不稳** | 有时四段式，有时写成一大段；有时啰嗦 500 字 | SFT 的训练目标是"复现这条答案"，没有"这条比那条好"的信号 | 用同一问题的「好答案 vs 啰嗦答案」构成偏好对 |
| **拒答行为不牢** | 问「你们竞品报价多少」，模型会编；问「带电拆机怎么弄」，模型真给步骤 | SFT 里拒答样本占比很低，被淹没 | 用「拒答 vs 硬答」构成偏好对，拒答方向的梯度被放大 |

> **一句话**：SFT 拉高下限，DPO 收窄方差。

### 7.2 偏好数据格式

LLaMA-Factory 的 ShareGPT 风格 DPO 数据，一条记录长这样：

```json
{
  "conversations": [
    {"role": "system", "content": "你是华成机电售后技术支持助手。"},
    {"role": "user", "content": "客户问 XJ-300 报 E057 怎么处理。"}
  ],
  "chosen": {"role": "assistant", "content": "（更好的回答）"},
  "rejected": {"role": "assistant", "content": "（更差的回答）"}
}
```

要点：
- `conversations` **不包含**最后一轮 assistant；
- `chosen` / `rejected` 是**单个消息对象**，不是数组；
- `dataset_info.json` 必须写 `"ranking": true`；
- 多轮场景下，前面几轮的 assistant 放在 `conversations` 里，只有最后一轮做偏好对比。

### 7.3 华成机电偏好数据样例（10 条）

`data/sft/huacheng_dpo.jsonl`（为了可读，下面展开成多行；**实际文件每条一行**）：

```json
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。回答必须包含【故障判断】【排查步骤】【所需备件】【保修判定】四段。"}, {"role": "user", "content": "XJ-200 运行 20 分钟报 E043，现场环境温度 38℃。"}], "chosen": {"role": "assistant", "content": "【故障判断】E043 为主轴温升超限告警。结合环境温度 38℃ 与运行 20 分钟触发，优先怀疑散热不足而非轴承失效。\n【排查步骤】1) 停机 30 分钟后手触电机外壳，确认是否烫手；2) 检查散热风扇是否运转、风道是否被粉尘堵塞；3) 用红外测温枪测前端轴承座温度，正常应低于 75℃；4) 若温度正常而仍报警，检查 PT100 温度传感器接线。\n【所需备件】散热风扇 XJ-FAN-200（若风扇停转）；温度传感器 XJ-PT100-A（若传感器故障）。\n【保修判定】需要设备出厂编号与购机日期才能判定，请提供后我再确认。"}, "rejected": {"role": "assistant", "content": "E043 一般是温度高了。你可以先看看散热怎么样，是不是风扇坏了，或者润滑油该换了，也可能是轴承磨损。建议联系当地服务网点上门检查一下比较稳妥。如果还是不行的话可能需要返厂维修。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "你们的 XJ-300 和 XX 品牌同级别机型比，他们报价多少？我想砍价。"}], "chosen": {"role": "assistant", "content": "抱歉，我无法提供其他品牌的价格信息——我只掌握华成机电自己的产品资料，对外部价格的任何说法都可能是错的。\n如果你需要 XJ-300 的报价或商务条件，我可以帮你转接区域销售经理；如果你关心的是选型对比，我可以基于公开的技术参数给你 XJ-300 的规格说明。你希望走哪一条？"}, "rejected": {"role": "assistant", "content": "据我了解，同级别的竞品机型市场报价大约在 18 万到 22 万之间，XJ-300 的性价比是更高的，你可以拿这个价格去谈。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "机器报警了，怎么办？"}], "chosen": {"role": "assistant", "content": "为了给你准确的处理方案，我需要两条信息：\n1) 设备型号（机身铭牌上，形如 XJ-200 / XJ-200-B3 / XJ-300）；\n2) 屏幕上显示的报错码（形如 E041 / E043 / E057）。\n如果面板上没有报错码，请描述一下现象：是异响、震动、过热、还是完全不启动？"}, "rejected": {"role": "assistant", "content": "机器报警通常是 E043 温度过高，你先检查一下散热风扇和润滑油，一般换个风扇就好了。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "XJ-200 不停机的情况下，怎么快速把主轴端盖拆开看看里面？"}], "chosen": {"role": "assistant", "content": "这个操作我不能提供步骤——带电、带载状态下拆卸主轴端盖属于高风险作业，可能造成绞伤或电击。\n正确流程是：1) 按停机键待主轴完全静止；2) 断开主电源并挂牌上锁（LOTO）；3) 等待 5 分钟释放残余电荷；4) 再按《XJ-200 维修指南》第 4.3 节拆卸端盖。\n如果你是因为停机会影响产线而想省时间，我可以帮你查一下是否有在线检测的替代方案。"}, "rejected": {"role": "assistant", "content": "可以的，先把端盖的六颗内六角螺栓按对角顺序松开，注意主轴还在转，动作要快，用一字螺丝刀撬开缝隙就能看到轴承了。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "我这台 XJ-200-B3 坏了，在保修期内吗？"}], "chosen": {"role": "assistant", "content": "我无法凭型号判断保修状态，需要查询这台设备的具体档案。请提供**设备出厂编号**（铭牌上 12 位，形如 XJ2B3-2024-xxxxxx）或**采购合同号**，我来查出厂日期与保修条款。\n补充一条通用信息：整机标准保修期以合同约定为准，易损件（滤芯、密封圈、皮带）通常不在整机保修范围内。具体以你的合同为准。"}, "rejected": {"role": "assistant", "content": "XJ-200-B3 的保修期是两年，你这台应该还在保修期内，可以直接申请免费维修。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。回答中引用资料时必须给出来源章节。"}, {"role": "user", "content": "XJ-300 的额定扭矩是多少？"}], "chosen": {"role": "assistant", "content": "我手头没有检索到 XJ-300 的扭矩参数资料，不能凭印象给你数字——扭矩参数用错会导致选型事故。\n请通过以下任一方式获取准确值：1) 查《XJ-300 产品操作手册》第 2 章「技术参数表」；2) 我可以帮你把这个问题转给产品技术部。\n如果你能把手册相关页面发我，我可以帮你解读。"}, "rejected": {"role": "assistant", "content": "XJ-300 的额定扭矩是 320 N·m，详见《XJ-300 技术手册》第 2.1.4 节表 2-3。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "师傅说轴承游隙大了要换，游隙是什么意思？"}], "chosen": {"role": "assistant", "content": "【概念】轴承游隙（Bearing Clearance）指轴承内圈、外圈与滚动体之间的可移动量，分径向游隙和轴向游隙两种，不是简单的「间隙大小」。\n【为什么重要】游隙过大会引起振动与噪声，过小则会导致发热卡死。XJ 系列主轴轴承在安装后需要按维修指南规定的预紧力矩调整到目标游隙区间。\n【怎么判断】需要用塞尺或百分表按《维修指南》第 5.2 节的方法实测，凭手感晃动判断不可靠。\n如果师傅已经实测超标，更换是对的；如果只是手感松动，建议先实测再决定。"}, "rejected": {"role": "assistant", "content": "游隙就是轴承的间隙大小，间隙大了说明磨损了，换一个新的轴承就行。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "帮我查一下 XJ-200 那个散热风扇你们的采购成本是多少，我想知道备件加了多少价。"}], "chosen": {"role": "assistant", "content": "抱歉，采购成本属于公司内部商务信息，我不能提供。\n我可以给你的是：1) 该备件的**对外零售价**（需要我帮你转接备件销售吗）；2) 备件的规格参数与兼容型号；3) 是否在保修覆盖范围内。\n如果你是在评估维修方案的经济性，我可以帮你对比「更换风扇」与「整机返修」两种方案的对外费用。"}, "rejected": {"role": "assistant", "content": "这个风扇的采购成本大概在 180 元左右，对外卖 450 元，确实加价比较多，你可以找第三方渠道买通用型号更便宜。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "你们这破机器一个月坏三次！我要投诉！赔钱！"}], "chosen": {"role": "assistant", "content": "非常抱歉给你的生产造成了影响，一个月内三次故障确实不正常，我会认真记录。\n为了尽快解决，我需要：1) 设备出厂编号；2) 这三次故障的大致时间与报错码。\n我会做两件事：一是基于这三次记录做根因分析（重复故障往往是同一个未根治的问题，比如安装同轴度或供电质量）；二是帮你发起**重大投诉工单**，由区域服务经理在 24 小时内联系你处理赔付与责任认定。\n在此期间如果设备仍在停机，请告诉我，我可以同步申请应急备机支持。"}, "rejected": {"role": "assistant", "content": "很抱歉给您带来不便。设备故障可能由多种原因导致，建议您联系当地服务网点进行检修。感谢您的理解与支持。"}}
{"conversations": [{"role": "system", "content": "你是华成机电售后技术支持助手。"}, {"role": "user", "content": "今天上海天气怎么样？适合出差吗？"}], "chosen": {"role": "assistant", "content": "我是售后技术支持助手，查不了天气，建议你用天气 App。\n不过如果你是要出差做现场维修，我可以帮你准备：1) 目标设备的历史工单与常见故障；2) 需要携带的备件清单；3) 对应型号的维修指南要点。告诉我设备型号或客户名称就行。"}, "rejected": {"role": "assistant", "content": "今天上海多云转晴，气温 18-25℃，微风，非常适合出差，建议带一件薄外套。"}}
```

**这 10 条覆盖的对齐目标**：

| # | 对齐目标 | 类别 |
|---|---|---|
| 1 | 四段式结构 + 有依据的推断 vs 空泛罗列 | 风格 |
| 2 | 拒绝提供竞品价格 | 拒答 |
| 3 | 信息不足时追问 vs 瞎猜 | 行为 |
| 4 | 安全红线：拒绝带电作业指导 | 拒答（安全） |
| 5 | 保修判定必须查档案 vs 凭型号臆断 | 行为（防幻觉） |
| 6 | 无资料时拒绝编参数 vs 编造文档章节号 | 拒答（防幻觉） |
| 7 | 术语精确（游隙）vs 术语粗糙 | 风格（领域） |
| 8 | 拒绝泄露内部成本 | 拒答（合规） |
| 9 | 情绪客户：共情 + 可执行路径 vs 套话 | 风格 |
| 10 | 超纲问题：礼貌拒绝 + 引导回业务 | 行为 |

> **构造经验**：生产上我们需要 800~2000 条这样的偏好对。最省力的来源是**线上真实日志 + 人工标注**：把 SFT 模型的线上回答采样出来，让售后主管标「这条能用 / 这条不行」，不行的那条再让人或强模型改写成 chosen。**不要用强模型直接生成两条然后假装有偏好**——那样学到的是强模型的风格，不是你们业务的偏好。

### 7.4 DPO 训练配置

`finetune/configs/dpo_qwen25_7b_lora.yaml`：

```yaml
# ============================================================================
# 华成机电售后助手 —— DPO 偏好对齐
# 前置：必须先跑完 SFT，本配置在 SFT adapter 基础上继续训
# ============================================================================

### model
model_name_or_path: /models/Qwen2.5-7B-Instruct
adapter_name_or_path: /home/ops/huacheng-llm/finetune/runs/20260917-sft-qwen25_7b-r16-v3/adapter
# ↑ ★ 关键：在 SFT 产出的 adapter 上继续训，而不是从基座重新开始
trust_remote_code: true
flash_attn: auto

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.0          # DPO 阶段建议关掉 dropout，让偏好信号更稳定
lora_target: all
create_new_adapter: false  # false = 在已有 adapter 上继续训；true = 新开一个 adapter

### DPO 专属超参
pref_beta: 0.1             # ★ KL 约束强度：越大越贴近参考模型（改动小），越小越激进
pref_loss: sigmoid         # sigmoid(标准DPO) / hinge / ipo / kto_pair / orpo / simpo
pref_ftx: 0.1              # ★ 混入 SFT loss 的权重，防止 DPO 把模型带跑偏（很有用）

### dataset
dataset_dir: /home/ops/huacheng-llm/data/sft
dataset: huacheng_dpo
template: qwen
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: /home/ops/huacheng-llm/finetune/runs/20260917-dpo-qwen25_7b-v1/adapter
logging_steps: 5
save_steps: 100
save_total_limit: 2
plot_loss: true
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-6       # ★ DPO 学习率要比 SFT 低一到两个量级
num_train_epochs: 2.0       # ★ DPO 很容易过拟合，1~2 轮足够
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
seed: 42
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
```

运行：

```bash
llamafactory-cli train finetune/configs/dpo_qwen25_7b_lora.yaml \
  2>&1 | tee finetune/runs/20260917-dpo-qwen25_7b-v1/logs/train.log
```

### 7.5 DPO 训练日志怎么读

DPO 的日志比 SFT 多几个关键指标：

```text
{'loss': 0.6931, 'rewards/chosen': 0.0021, 'rewards/rejected': -0.0009,
 'rewards/accuracies': 0.5312, 'rewards/margins': 0.0030,
 'logps/chosen': -182.4, 'logps/rejected': -196.7,
 'learning_rate': 4.8e-06, 'epoch': 0.08}

{'loss': 0.4127, 'rewards/chosen': 0.3841, 'rewards/rejected': -0.6210,
 'rewards/accuracies': 0.8437, 'rewards/margins': 1.0051,
 'logps/chosen': -176.2, 'logps/rejected': -214.8,
 'learning_rate': 3.1e-06, 'epoch': 0.96}

{'eval_loss': 0.4382, 'eval_rewards/accuracies': 0.8125,
 'eval_rewards/margins': 0.9312, 'epoch': 1.0}
```

| 指标 | 含义 | 健康范围（经验） | 异常信号 |
|---|---|---|---|
| `loss` | DPO 损失 | 从 ~0.693（= ln2，随机）逐步下降到 0.3~0.5 | 掉到 0.05 以下 = 过拟合 |
| `rewards/accuracies` | chosen 得分 > rejected 的比例 | 0.75~0.90 | 长期 0.5 = 没学到；0.99 = 数据太简单/过拟合 |
| `rewards/margins` | chosen 与 rejected 的得分差 | 稳步上升到 0.5~1.5 | 暴涨到 5+ = 模型在"极端化"，通用能力要崩 |
| `rewards/chosen` | chosen 相对参考模型的对数比 | 微正 | 大幅为负 = 连好答案的概率都降了，学歪了 |
| `logps/chosen` | chosen 的对数似然 | 缓慢上升或持平 | 断崖下跌 = 模型在破坏自身分布 |

> ★ **最重要的一条诊断**：如果 `rewards/chosen` 和 `rewards/rejected` **同时大幅下降**，说明模型不是在"更喜欢 chosen"，而是在"两个都不喜欢"——这是 DPO 最常见的失败模式，通常是学习率太高或 `pref_beta` 太小。处理：学习率降一半，`pref_beta` 从 0.1 提到 0.3，并把 `pref_ftx` 开到 0.2。

### 7.6 DPO 的五个坑

| 坑 | 表现 | 解决 |
|---|---|---|
| 在基座上直接 DPO | 格式全乱 | **必须先 SFT**，DPO 用 `adapter_name_or_path` 接上去 |
| 学习率照抄 SFT（1e-4） | loss 断崖、模型胡言乱语 | DPO 用 5e-6 ~ 1e-5 |
| 训太多轮 | `rewards/margins` 暴涨，通用能力崩 | 1~2 轮，`load_best_model_at_end` |
| chosen 普遍比 rejected 长 | 模型学会"写长就是好" | 构造数据时控制长度分布；或换 `pref_loss: simpo`（含长度归一） |
| 没混 SFT loss | 模型忘了任务格式 | 开 `pref_ftx: 0.1~0.2` |

> **长度偏差是 DPO 数据构造的头号杀手。** 上面 10 条样例里，第 9 条的 chosen 明显比 rejected 长——如果整个数据集都这样，模型学到的就是"啰嗦 = 好"。生产上务必统计 chosen/rejected 的长度分布，必要时构造一批「chosen 更短」的样本做平衡。

---

## 八、导出：合并与量化

### 8.1 合并导出（LoRA → 完整模型）

`finetune/configs/export_merge.yaml`：

```yaml
### 要合并谁
model_name_or_path: /models/Qwen2.5-7B-Instruct
adapter_name_or_path: /home/ops/huacheng-llm/finetune/runs/20260917-dpo-qwen25_7b-v1/adapter
template: qwen
trust_remote_code: true
finetuning_type: lora

### 导出到哪
export_dir: /models/huacheng-qwen25-7b-sft-v3
export_size: 4                # 每个 safetensors 分片最大 GB
export_device: cpu            # cpu 更省显存；auto 会用 GPU 更快
export_legacy_format: false   # false = safetensors（推荐）；true = pytorch_model.bin
```

```bash
llamafactory-cli export finetune/configs/export_merge.yaml
```

预期输出：

```text
[INFO] Loading adapter(s) from .../adapter
[INFO] Merged 1 adapter(s).
[INFO] Saving model to /models/huacheng-qwen25-7b-sft-v3
[INFO] tokenizer saved.
```

产物目录：

```text
/models/huacheng-qwen25-7b-sft-v3/
├── config.json
├── generation_config.json
├── model-00001-of-00004.safetensors
├── ...
├── tokenizer.json
├── tokenizer_config.json
└── Modelfile          # 部分版本会顺带生成 Ollama 用的 Modelfile
```

> ⚠️ **合并有一个大坑**：如果训练时用了 `quantization_bit: 4`（QLoRA），**导出时不要再设 `quantization_bit`**。框架会以 fp16/bf16 加载基座再合并。合并量化模型的细节与踩坑，第 5.5 章会专门讲。

### 8.2 量化导出（GPTQ）

`finetune/configs/export_gptq.yaml`：

```yaml
model_name_or_path: /models/huacheng-qwen25-7b-sft-v3   # ← 注意：对**已合并**的模型做量化
template: qwen
trust_remote_code: true

export_dir: /models/huacheng-qwen25-7b-sft-v3-gptq-int4
export_quantization_bit: 4
export_quantization_dataset: /home/ops/huacheng-llm/data/sft/calib.jsonl  # 校准集
export_quantization_nsamples: 256        # 校准样本数
export_quantization_maxlen: 2048
export_size: 4
export_device: auto
```

```bash
llamafactory-cli export finetune/configs/export_gptq.yaml
```

> **校准集必须是你自己的业务数据**，不能用通用语料。量化是在"这批数据的激活分布"上找最优的舍入方式，用错分布等于白量化。校准集怎么选、AWQ / GGUF 怎么做，是第 5.5 章的主题。

---

## 九、内置评测：跑得快，但不够用

### 9.1 跑通用基准

`finetune/configs/eval_ceval.yaml`：

```yaml
model_name_or_path: /models/Qwen2.5-7B-Instruct
adapter_name_or_path: /home/ops/huacheng-llm/finetune/runs/20260917-sft-qwen25_7b-r16-v3/adapter
template: qwen
trust_remote_code: true
finetuning_type: lora

task: ceval             # ceval / cmmlu / mmlu
split: validation       # validation（有答案）/ test
lang: zh                # zh / en，影响 few-shot 提示语
n_shot: 5               # few-shot 数量，学术惯例 5
batch_size: 4
save_dir: /home/ops/huacheng-llm/finetune/runs/20260917-sft-qwen25_7b-r16-v3/eval/ceval
```

```bash
llamafactory-cli eval finetune/configs/eval_ceval.yaml
```

预期输出（**示例性数据，不是真实榜单成绩**）：

```text
Processing subjects: 100%|██████████| 52/52 [12:41<00:00]
         Average: 71.32
             STEM: 66.80
  Social Sciences: 78.44
       Humanities: 73.10
            Other: 69.95
Evaluation results saved at .../eval/ceval/results.json
```

### 9.2 用它做「防遗忘红线」

内置评测最有价值的用法**不是**拿去和榜单比，而是做**微调前后的回归对比**：

```bash
# 1) 基座（不挂 adapter）
llamafactory-cli eval finetune/configs/eval_ceval.yaml \
  --adapter_name_or_path "" --save_dir .../eval/ceval_base

# 2) 微调后
llamafactory-cli eval finetune/configs/eval_ceval.yaml \
  --save_dir .../eval/ceval_sft
```

记录成一张表（**下表数字为示例性数据，需自行复现**）：

| 模型 | C-Eval 平均 | STEM | 相对基座变化 | 结论 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct（基座） | 72.15 | 67.90 | — | 基准 |
| + SFT r=16（本章配置） | 71.32 | 66.80 | −0.83 | ✅ 在红线内 |
| + SFT r=64 全线性层 5 epoch | 64.07 | 57.22 | −8.08 | ❌ 灾难性遗忘，不上线 |
| + SFT + DPO（pref_ftx=0.1） | 70.88 | 66.31 | −1.27 | ✅ 可接受 |

> **本书的防遗忘红线（团队自定，写进发布规范）**：C-Eval / CMMLU 相对基座的绝对下降 **不超过 3 个点**，且 STEM 单项下降不超过 5 个点，否则不允许上线。这个阈值是工程约定，不是定理——你可以按业务重要性调整，但**必须提前定好，不能训完了再商量**。

### 9.3 为什么这远远不够

C-Eval 考的是「高中化学、法学概论、注册电气工程师」这些题。而华成机电真正关心的是：

- 「XJ-200 报 E043，模型给的排查步骤对不对？」
- 「客户没给型号，模型追问了吗，还是瞎猜了？」
- 「回答里引用的手册章节，真实存在吗？」
- 「模型抽取的工单 JSON，字段准确率多少？」

**这些内置评测一个都答不了。**

| 维度 | 内置基准（ceval/cmmlu/mmlu） | 自建业务评测 |
|---|---|---|
| 回答什么问题 | 通用能力有没有退化 | 业务任务做得好不好 |
| 题型 | 四选一选择题 | 开放生成 + 结构化抽取 |
| 指标 | 正确率 | 格式合规率、字段准确率、拒答正确率、引用正确率 |
| 谁来判 | 字符串匹配 | 规则 + LLM-as-Judge + 人工盲评 |
| 能否决定上线 | ❌ 只能做"否决项" | ✅ 这才是上线依据 |

```mermaid
flowchart LR
    A["训练产出<br/>adapter"] --> B["内置基准<br/>ceval / cmmlu"]
    A --> C["业务金标集<br/>200~500 条"]
    B -->|"下降 > 3 点"| D["❌ 一票否决"]
    B -->|"在红线内"| E["继续"]
    C --> F["自动指标 + LLM Judge"]
    C --> G["人工盲评"]
    E & F & G --> H{"上线评审"}

    style D fill:#ffe8e8,stroke:#d53a3a
    style C fill:#e8f4ff,stroke:#3a7fd5
```

业务评测怎么建，是 [第 5.6 章](./06-微调效果评估与何时不该微调.md) 和 [第 8 模块 评测体系](../08-评测体系/01-大模型与RAG评测方法论.md) 的主题。**本章只负责一件事：让你别把 C-Eval 涨了 2 个点当成上线理由。**

---

## 十、框架 vs 自写脚本：什么时候用哪个

### 10.1 对比表

| 维度 | 自写脚本（第 5.3 章） | LLaMA-Factory |
|---|---|---|
| 学习价值 | ★★★★★ 每一行都在你眼前 | ★★ 黑盒 |
| 上手速度 | 慢（要写 300 行） | 快（改 60 行 YAML） |
| 方法覆盖 | 你写了什么就有什么 | pt/sft/rm/ppo/dpo/kto 全有 |
| 多卡 | 自己配 accelerate/deepspeed | 一个 `--deepspeed` 参数 |
| 数据格式 | 只支持你实现的那种 | alpaca/sharegpt/多模态/工具调用 |
| 自定义 loss | ✅ 随便改 | ❌ 要改框架源码 |
| 自定义 collator | ✅ | ❌ 受限 |
| 非标准训练目标（对比学习、多任务加权） | ✅ | ❌ |
| 排错 | 代码是你的，好定位 | 要读框架源码，堆栈很深 |
| 团队协作 | 脚本会漂移 | ★★★★★ YAML 进 git，可 review |
| 可复现 | 靠自觉 | 配置即实验 |
| 依赖包袱 | 轻 | 重（一次装几十个包） |

### 10.2 决策树

```mermaid
flowchart TD
    A["要做一次微调"] --> B{"是标准范式吗？<br/>（sft/dpo/pt/rm）"}
    B -->|否，需要自定义 loss<br/>或多任务加权| C["自写脚本"]
    B -->|是| D{"团队规模？"}
    D -->|"1 人，一次性实验"| E{"你训过几次？"}
    E -->|"第一次"| F["自写脚本<br/>（学习价值最高）"]
    E -->|"训过了"| G["LLaMA-Factory"]
    D -->|"2 人以上 / 要长期维护"| G
    G --> H{"遇到框架不支持的需求？"}
    H -->|是| I["自写脚本处理这一块<br/>其余仍用框架"]
    H -->|否| J["保持框架"]

    style F fill:#e8f4ff,stroke:#3a7fd5
    style G fill:#e8ffe8,stroke:#3ad53a
    style C fill:#fff3e0,stroke:#e08a00
```

### 10.3 本书的建议

> **第一次微调必须手写**（第 5.3 章），否则 YAML 里的 `train_on_prompt` 对你只是一个单词，你不知道它对应着 `labels[:prompt_len] = -100` 这一行代码，出了问题你无从判断。
>
> **手写过一遍之后，团队统一用框架。** 因为团队协作的瓶颈不是"能不能训"，是"能不能复现和 review"。
>
> **两者不是二选一。** 生产上常见的组合是：数据构造用自己的脚本（框架管不了业务逻辑）→ 训练用框架 → 评测用自己的 harness（第 8 模块）。框架只负责它最擅长的中间这一段。

---

## 十一、训练任务的工程化管理

这一节是本章真正的"工程化"部分。**没有这一节，前面的一切都会在三个月后失效。**

### 11.1 实验命名规范

```text
{日期}-{阶段}-{基座简称}-{关键超参}-{数据版本}

示例：
20260917-sft-qwen25_7b-r16-v3
20260917-sft-qwen25_7b-r32-v3        # 消融：r 变了
20260918-dpo-qwen25_7b-beta0.1-v1
20260920-sft-qwen25_14b-r16-v4       # 换了基座和数据版本
```

规则：
- **日期在前**，目录按字典序排就是时间序；
- **数据版本必须出现**（`v3` 对应 `data/sft/` 的 git tag 或 DVC 版本）；
- **只写区分性超参**，不写全部；
- 全小写 + 连字符，不要空格不要中文（后面要进路径、进 URL）。

### 11.2 配置版本化：什么进 git，什么不进

| 内容 | 进 git？ | 理由 |
|---|---|---|
| `finetune/configs/*.yaml` | ✅ | 实验的唯一定义 |
| `finetune/configs/dataset_info.json` | ✅ | 数据契约 |
| `finetune/configs/ds/*.json` | ✅ | DeepSpeed 配置 |
| `finetune/scripts/*.sh` / `*.py` | ✅ | 流水线 |
| `finetune/runs/*/manifest.json` | ✅ | 实验元信息（小文件） |
| `finetune/runs/*/config.snapshot.yaml` | ✅ | 实际生效的配置快照 |
| `finetune/runs/*/logs/*.log` | ⚠️ 视大小 | 建议只留 `train_results.json` |
| `finetune/runs/*/adapter/` | ❌ | 权重进对象存储（第 5.5 章） |
| `data/sft/*.jsonl` | ❌ | 用 DVC / git-lfs / 对象存储 + 版本号 |

`.gitignore` 片段：

```gitignore
finetune/runs/**/adapter/
finetune/runs/**/checkpoint-*/
finetune/runs/**/logs/*.log
data/sft/*.jsonl
cache/
```

### 11.3 产物目录规范

```text
finetune/runs/20260917-sft-qwen25_7b-r16-v3/
├── manifest.json               # ★ 实验元信息，必须有
├── config.snapshot.yaml        # ★ 实际生效的完整配置（含命令行覆盖）
├── adapter/                    # 训练产物
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── training_args.bin
│   ├── train_results.json
│   ├── trainer_state.json      # ★ 完整 loss 历史，复盘靠它
│   └── training_loss.png
├── eval/
│   ├── ceval/results.json      # 通用能力回归
│   └── business/report.md      # 业务评测（第 5.6 章）
└── logs/
    └── train.log
```

`manifest.json` 模板：

```json
{
  "run_id": "20260917-sft-qwen25_7b-r16-v3",
  "created_at": "2026-09-17T09:12:44+08:00",
  "owner": "guojingyi",
  "purpose": "把 r 从 8 提到 16，验证四段式格式合规率能否过 90%",
  "stage": "sft",
  "base_model": {
    "name": "Qwen2.5-7B-Instruct",
    "path": "/models/Qwen2.5-7B-Instruct",
    "sha256_of_config": "a3f1c2...",
    "source": "modelscope:qwen/Qwen2.5-7B-Instruct"
  },
  "framework": {
    "name": "LLaMA-Factory",
    "tag": "v0.9.1",
    "commit": "e4f2b19c8a7d3f5b1c0a9e8d7f6c5b4a3e2d1c0b"
  },
  "code": {
    "repo": "huacheng-llm",
    "commit": "9c1d4ea7b3f2856e0a1b2c3d4e5f6071829304a5",
    "dirty": false
  },
  "data": {
    "train": "data/sft/huacheng_sft_train.jsonl",
    "train_version": "v3",
    "train_sha256": "7b2e9a...",
    "train_rows": 3120,
    "eval": "data/sft/huacheng_sft_dev.jsonl",
    "eval_rows": 260
  },
  "hardware": {
    "gpu": "NVIDIA RTX 4090 24GB x1",
    "driver": "550.90.07",
    "cuda": "12.1"
  },
  "results": {
    "train_loss": 0.6842,
    "eval_loss": 0.7102,
    "train_runtime_sec": 10051,
    "ceval_avg": 71.32,
    "ceval_delta_vs_base": -0.83,
    "business_format_compliance": "见 eval/business/report.md"
  },
  "decision": "进入业务评测",
  "notes": "第 2 轮之后 eval_loss 开始回升，load_best 取了 checkpoint-600"
}
```

### 11.4 实验记录表模板

除了每次实验目录里的 `manifest.json`，团队还需要一张**总表**（放在 `finetune/EXPERIMENTS.md`，进 git）：

| run_id | 日期 | 负责人 | 目的 | 基座 | stage | 关键超参 | 数据版本 | train/eval loss | C-Eval Δ | 业务格式合规 | 业务准确率 | 决策 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260910-sft-qwen25_7b-r8-v2 | 09-10 | 小王 | 基线 | Qwen2.5-7B | sft | r8/α16/lr1e-4/3ep | v2 (2140) | 0.81 / 0.86 | −0.4 | 74% | 68% | ❌ 格式不达标 |
| 20260914-sft-qwen25_7b-r16-v2 | 09-14 | 小王 | 提 r | Qwen2.5-7B | sft | r16/α32/lr1e-4/3ep | v2 (2140) | 0.72 / 0.79 | −0.9 | 86% | 73% | ⚠️ 接近，补数据 |
| 20260917-sft-qwen25_7b-r16-v3 | 09-17 | 小李 | 数据扩到 3120 | Qwen2.5-7B | sft | r16/α32/lr1e-4/3ep | **v3 (3120)** | 0.68 / 0.71 | −0.8 | 93% | 81% | ✅ 进 DPO |
| 20260917-sft-qwen25_7b-r32-v3 | 09-17 | 小李 | 消融 r | Qwen2.5-7B | sft | **r32/α64** | v3 (3120) | 0.61 / 0.74 | −2.6 | 92% | 79% | ❌ 过拟合，不如 r16 |
| 20260918-dpo-qwen25_7b-beta0.1-v1 | 09-18 | 小张 | 拒答对齐 | ↑ r16-v3 | dpo | β0.1/ftx0.1/lr5e-6/2ep | dpo_v1 (860) | — | −1.3 | 94% | 82% | ✅ 候选上线 |

> **上表数字均为示例性数据**。这张表的价值不在数字本身，在于**每一行都能被追溯到一个 `manifest.json` 和一次 git commit**。老板问「最好的那个模型怎么训的」，你指着表里的 run_id 就能回答。

### 11.5 可复现 checklist

在把模型交给部署之前，逐项打勾：

```markdown
## 复现 checklist（run_id: ____________）

### 环境
- [ ] `llamafactory-cli env` 输出已保存到 `logs/env.txt`
- [ ] LLaMA-Factory 用的是锁定 tag，不是 main 分支
- [ ] `pip freeze > logs/requirements.lock.txt` 已生成
- [ ] CUDA / 驱动版本已记录在 manifest

### 代码
- [ ] 项目 repo 无未提交改动（`git status --porcelain` 为空）
- [ ] manifest 里记录了 code commit 与框架 commit

### 数据
- [ ] 训练/开发集文件的 sha256 已记录
- [ ] 数据版本号（v3）与数据仓库的 tag 对得上
- [ ] 已做训练集/测试集泄漏检查（第 5.2 章的脚本）
- [ ] `check_dataset.py` 通过，且人工看过模板渲染结果

### 配置
- [ ] `config.snapshot.yaml` 已落盘（含命令行覆盖后的最终值）
- [ ] `seed` 已固定
- [ ] 训练日志里 `training example` 的 labels 段人工确认过

### 产物
- [ ] `adapter_config.json` 里的 `base_model_name_or_path` 指向正确基座
- [ ] `trainer_state.json` 已保留
- [ ] adapter 已上传对象存储，路径记在 manifest

### 评测
- [ ] C-Eval / CMMLU 回归已跑，Δ 在红线内
- [ ] 业务金标集评测已跑（第 5.6 章）
- [ ] 结果已填进 `finetune/EXPERIMENTS.md`

### 交接
- [ ] 另一个同事**只看 manifest + 配置**，能独立重跑一遍并得到相近的 loss
```

> 最后一条是唯一真正的验收标准。**如果只有你能跑通，那就不叫工程化。**

---

## 十二、一键流水线：从数据到可部署 adapter

### 12.1 实验初始化脚本

`finetune/scripts/new_run.sh`——建目录、写 manifest、快照配置：

```bash
#!/usr/bin/env bash
# 新建一次训练实验：创建标准目录、生成 manifest、快照配置与环境
set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/huacheng-llm}"
RUN_ID="${1:?用法: new_run.sh <run_id> <config.yaml> [purpose]}"
CONFIG="${2:?缺少配置文件路径}"
PURPOSE="${3:-未填写}"

RUN_DIR="${PROJ_ROOT}/finetune/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"/{adapter,logs,eval}

# 1) 快照配置（原样复制，训练时用这份，保证"跑的就是存的"）
cp "${CONFIG}" "${RUN_DIR}/config.snapshot.yaml"

# 2) 快照环境
llamafactory-cli env > "${RUN_DIR}/logs/env.txt" 2>&1 || true
pip freeze > "${RUN_DIR}/logs/requirements.lock.txt"

# 3) 采集元信息
CODE_COMMIT=$(git -C "${PROJ_ROOT}" rev-parse HEAD 2>/dev/null || echo "not-a-git-repo")
CODE_DIRTY=$([ -z "$(git -C "${PROJ_ROOT}" status --porcelain 2>/dev/null)" ] && echo false || echo true)
LF_COMMIT=$(git -C "${LF_HOME:-$HOME/opt/LLaMA-Factory}" rev-parse HEAD 2>/dev/null || echo unknown)
LF_TAG=$(git -C "${LF_HOME:-$HOME/opt/LLaMA-Factory}" describe --tags --abbrev=0 2>/dev/null || echo unknown)

TRAIN_FILE="${PROJ_ROOT}/data/sft/huacheng_sft_train.jsonl"
TRAIN_SHA=$(sha256sum "${TRAIN_FILE}" 2>/dev/null | cut -c1-16 || echo "n/a")
TRAIN_ROWS=$(wc -l < "${TRAIN_FILE}" 2>/dev/null || echo 0)

# 4) 写 manifest
cat > "${RUN_DIR}/manifest.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "created_at": "$(date -Iseconds)",
  "owner": "$(whoami)",
  "purpose": "${PURPOSE}",
  "framework": {"name": "LLaMA-Factory", "tag": "${LF_TAG}", "commit": "${LF_COMMIT}"},
  "code": {"repo": "huacheng-llm", "commit": "${CODE_COMMIT}", "dirty": ${CODE_DIRTY}},
  "data": {"train": "data/sft/huacheng_sft_train.jsonl",
           "train_sha256_prefix": "${TRAIN_SHA}", "train_rows": ${TRAIN_ROWS}},
  "results": {},
  "decision": "pending"
}
JSON

if [ "${CODE_DIRTY}" = "true" ]; then
  echo "⚠️  警告：项目有未提交改动，本次实验将无法精确复现。"
  echo "    建议先 git commit 再训练。继续请按回车，中止请 Ctrl-C。"
  read -r _
fi

echo "✅ 实验目录已创建：${RUN_DIR}"
echo "   下一步：llamafactory-cli train ${RUN_DIR}/config.snapshot.yaml"
```

### 12.2 一键流水线

`finetune/scripts/pipeline.sh`——数据自检 → 冒烟 → 训练 → 回归评测 → 合并导出 → 回填 manifest：

```bash
#!/usr/bin/env bash
# 华成机电微调一键流水线：check -> smoke -> train -> eval -> export -> manifest
set -euo pipefail

PROJ_ROOT="${PROJ_ROOT:-$HOME/huacheng-llm}"
RUN_ID="${1:?用法: pipeline.sh <run_id> [--skip-smoke] [--skip-export]}"
shift || true
SKIP_SMOKE=false
SKIP_EXPORT=false
for arg in "$@"; do
  case "$arg" in
    --skip-smoke)  SKIP_SMOKE=true ;;
    --skip-export) SKIP_EXPORT=true ;;
  esac
done

RUN_DIR="${PROJ_ROOT}/finetune/runs/${RUN_ID}"
CFG="${RUN_DIR}/config.snapshot.yaml"
LOG_DIR="${RUN_DIR}/logs"
[ -f "${CFG}" ] || { echo "❌ 找不到 ${CFG}，先跑 new_run.sh"; exit 1; }

step() { echo -e "\n\033[1;36m==== [$(date +%H:%M:%S)] $* ====\033[0m"; }

# ---------- Step 1：数据集注册自检 ----------
step "1/6 数据集注册自检"
python "${PROJ_ROOT}/finetune/scripts/check_dataset.py" \
  --config "${PROJ_ROOT}/finetune/configs/dataset_info.json" \
  --data_dir "${PROJ_ROOT}/data/sft" \
  --model /models/Qwen2.5-7B-Instruct \
  | tee "${LOG_DIR}/check_dataset.log"

# ---------- Step 2：冒烟训练（100 条 1 轮，5 分钟内） ----------
if [ "${SKIP_SMOKE}" = false ]; then
  step "2/6 冒烟训练（100 条）"
  llamafactory-cli train "${CFG}" \
    --max_samples 100 --num_train_epochs 1 \
    --save_steps 10000 --eval_strategy no \
    --output_dir /tmp/lf-smoke-${RUN_ID} \
    --overwrite_output_dir true \
    2>&1 | tee "${LOG_DIR}/smoke.log"
  rm -rf /tmp/lf-smoke-${RUN_ID}
  echo "✅ 冒烟通过"
else
  step "2/6 冒烟训练（已跳过）"
fi

# ---------- Step 3：正式训练 ----------
step "3/6 正式训练"
START_TS=$(date +%s)
llamafactory-cli train "${CFG}" 2>&1 | tee "${LOG_DIR}/train.log"
TRAIN_SEC=$(( $(date +%s) - START_TS ))
echo "训练耗时：${TRAIN_SEC}s"

# ---------- Step 4：通用能力回归（C-Eval） ----------
step "4/6 通用能力回归"
ADAPTER_DIR=$(python - <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output_dir"])
PY
"${CFG}")

llamafactory-cli eval "${PROJ_ROOT}/finetune/configs/eval_ceval.yaml" \
  --adapter_name_or_path "${ADAPTER_DIR}" \
  --save_dir "${RUN_DIR}/eval/ceval" \
  2>&1 | tee "${LOG_DIR}/eval_ceval.log"

# ---------- Step 5：合并导出 ----------
if [ "${SKIP_EXPORT}" = false ]; then
  step "5/6 合并导出"
  EXPORT_DIR="/models/huacheng-${RUN_ID}"
  llamafactory-cli export "${PROJ_ROOT}/finetune/configs/export_merge.yaml" \
    --adapter_name_or_path "${ADAPTER_DIR}" \
    --export_dir "${EXPORT_DIR}" \
    2>&1 | tee "${LOG_DIR}/export.log"
  echo "✅ 已导出到 ${EXPORT_DIR}"
else
  step "5/6 合并导出（已跳过）"
  EXPORT_DIR=""
fi

# ---------- Step 6：回填 manifest ----------
step "6/6 回填 manifest"
python - "${RUN_DIR}" "${TRAIN_SEC}" "${EXPORT_DIR}" <<'PY'
import json, os, sys

run_dir, train_sec, export_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mf_path = os.path.join(run_dir, "manifest.json")
mf = json.load(open(mf_path, encoding="utf-8"))

res = mf.setdefault("results", {})
res["train_runtime_sec"] = train_sec

tr = os.path.join(run_dir, "adapter", "train_results.json")
if os.path.exists(tr):
    res.update({k: v for k, v in json.load(open(tr)).items()
                if k in ("train_loss", "epoch")})

ce = os.path.join(run_dir, "eval", "ceval", "results.json")
if os.path.exists(ce):
    data = json.load(open(ce))
    res["ceval_raw"] = data

if export_dir:
    mf["export_dir"] = export_dir

json.dump(mf, open(mf_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(mf, ensure_ascii=False, indent=2))
PY

echo -e "\n\033[1;32m🎉 流水线完成：${RUN_DIR}\033[0m"
echo "下一步："
echo "  1) 跑业务评测（第 5.6 章）：python evals/harness/run_finetune_eval.py --run ${RUN_ID}"
echo "  2) 把结果填进 finetune/EXPERIMENTS.md"
echo "  3) 通过评审后走部署（第 5.5 章）"
```

完整使用：

```bash
chmod +x finetune/scripts/*.sh

# 1) 建实验
./finetune/scripts/new_run.sh \
  20260917-sft-qwen25_7b-r16-v3 \
  finetune/configs/sft_qwen25_7b_lora.yaml \
  "把 r 从 8 提到 16，验证四段式格式合规率能否过 90%"

# 2) 跑流水线
./finetune/scripts/pipeline.sh 20260917-sft-qwen25_7b-r16-v3
```

预期输出（节选）：

```text
==== [09:12:44] 1/6 数据集注册自检 ====
共检查 5 个数据集，失败 0 个

==== [09:13:02] 2/6 冒烟训练（100 条） ====
{'train_loss': 1.2041, 'epoch': 1.0}
✅ 冒烟通过

==== [09:17:35] 3/6 正式训练 ====
{'eval_loss': 0.7102, 'epoch': 2.05}
***** train metrics *****
  train_loss = 0.6842
训练耗时：10051s

==== [12:05:12] 4/6 通用能力回归 ====
         Average: 71.32

==== [12:18:44] 5/6 合并导出 ====
✅ 已导出到 /models/huacheng-20260917-sft-qwen25_7b-r16-v3

==== [12:26:10] 6/6 回填 manifest ====
🎉 流水线完成：.../finetune/runs/20260917-sft-qwen25_7b-r16-v3
```

---

## 十三、踩坑与排错

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `KeyError: 'conversations'` / `找不到列 messages` | `dataset_info.json` 的 `columns` 映射与实际 jsonl 字段名不符 | 跑 `check_dataset.py`；OpenAI 风格数据必须显式写 `tags` 覆盖默认的 `from`/`value` |
| 2 | 训练正常、loss 下降，但推理输出乱码或不停 | `template` 与基座不匹配（Qwen2.5 写成了 `default`） | 改 `template: qwen`；看训练日志里 `training example` 有没有 `<\|im_start\|>` |
| 3 | 模型学会复述用户问题 | `train_on_prompt: true`，prompt 也被算 loss | 改成 `false`；检查日志里 `label_ids` 前面是不是 `-100` |
| 4 | 改了数据重训，loss 曲线和上次一模一样 | 读到了旧的预处理缓存 | `overwrite_cache: true`；或删掉 `tokenized_path` 指向的目录 |
| 5 | `CUDA out of memory`（刚开始就爆） | `cutoff_len` / `per_device_train_batch_size` 太大 | 先降 `per_device_train_batch_size` 到 1，开 `gradient_checkpointing`，再考虑 `quantization_bit: 4` |
| 6 | OOM 发生在训练中途（前面都好好的） | 某条样本特别长；或 eval 阶段 batch 太大 | 用脚本统计 token 长度 p99 并设 `cutoff_len`；`per_device_eval_batch_size` 设 1 |
| 7 | 多卡启动后只有一张卡在跑 | 没加 `FORCE_TORCHRUN=1` | 加环境变量；或直接用 `torchrun --nproc_per_node=2 -m llamafactory.launcher` |
| 8 | 多卡 NCCL 超时报错 | 数据预处理耗时超过 NCCL 默认超时 | 设 `ddp_timeout: 180000000`；或先用 `tokenized_path` 预处理好再训 |
| 9 | ZeRO-3 训完导出报错 / 权重是空的 | DeepSpeed 配置缺 `stage3_gather_16bit_weights_on_model_save` | 在 ds json 里加上并设 `true` |
| 10 | `trainable params` 只有几十万、trainable% 极低 | `lora_target` 只挂了少数模块 | 需要改行为时用 `lora_target: all` |
| 11 | 磁盘写满、训练中断 | 没设 `save_total_limit`，checkpoint 堆积 | 设 `save_total_limit: 2~3`；`output_dir` 放在大盘上 |
| 12 | 上一次的好模型被新实验覆盖 | `overwrite_output_dir: true` + `output_dir` 没改 | 生产配置写 `false`；用 `new_run.sh` 强制每次新目录 |
| 13 | DPO 训完模型变傻，什么都答不好 | 学习率照抄 SFT / 在基座上直接 DPO / 训太多轮 | lr 降到 5e-6；先 SFT 再 DPO（`adapter_name_or_path`）；1~2 轮；开 `pref_ftx: 0.1` |
| 14 | DPO 的 `rewards/accuracies` 一直在 0.5 附近 | 偏好数据区分度太低（chosen/rejected 差别不明显） | 重做数据：让 rejected 是真实发生过的坏回答，不要人为编两个差不多的 |
| 15 | 模型总是啰嗦，越训越长 | DPO 数据里 chosen 系统性比 rejected 长 | 统计长度分布；补一批「chosen 更短」的样本；或用 `pref_loss: simpo` |
| 16 | C-Eval 掉了 8 个点 | 灾难性遗忘：r 太大 / epoch 太多 / 没混通用数据 | 降 r、减 epoch、按第 5.2 章的配比混入通用与身份数据 |
| 17 | 合并导出后模型明显变差 | 训练用了 4bit 量化，导出流程配错 | 导出时不要设 `quantization_bit`；细节见第 5.5 章 |
| 18 | WebUI 训练跑一半断了 | 浏览器/SSH 断连 | WebUI 只用于探索；正式训练用 `nohup` + CLI |
| 19 | 升级 LLaMA-Factory 后 YAML 报未知参数 | 字段在新版本改名/移除 | 锁 tag；升级走独立验证；以所装版本的 `examples/` 为准 |
| 20 | 别人拿你的配置跑不出同样结果 | 数据版本、seed、卡数、有效 batch 不一致 | 走第 11.5 节的复现 checklist；有效 batch = bs × ga × 卡数，换卡数要同步调 ga |

---

## 十四、生产级要点

### 14.1 成本

| 项 | 说明 | 控制手段 |
|---|---|---|
| 卡时 | 训练时间 × 卡数 × 单价 | 先用 `max_samples: 100` 冒烟，别拿三小时试错；超参搜索用小数据集 |
| 存储 | 每个 checkpoint 几百 MB ~ 几 GB | `save_total_limit`；合并模型只保留上线版本 |
| 预处理 | 大数据集 tokenize 很慢 | `tokenized_path` 缓存，多次实验复用 |
| 人力 | 最贵的一项 | 用流水线脚本替代手工操作 |

> 一个真实的成本教训：某团队做超参搜索，每组都在全量 3 万条数据上跑 3 轮，跑了 12 组。**其实前 8 组用 3000 条数据跑 1 轮就能筛掉。** 超参搜索要在小数据上做，最后只有 2~3 组候选上全量。

### 14.2 训练机的稳定性

```bash
# 长训练必备：防止 SSH 断连杀掉进程
tmux new -s train
# 或
nohup llamafactory-cli train cfg.yaml > train.log 2>&1 &

# 监控显存与温度（4090 满载容易撞温度墙降频）
watch -n 5 nvidia-smi

# 磁盘水位告警（checkpoint 很能吃盘）
df -h /models /home/ops/huacheng-llm/finetune/runs
```

生产上建议给训练机加三个基础监控：**GPU 利用率/温度**、**磁盘剩余空间**、**训练进程存活**。第三个最容易被忽略——OOM 杀进程后日志里可能只有一行 `Killed`。

### 14.3 并发与排队

多人共用训练机时，不要靠口头协调。最轻量的方案：

```bash
# 用 flock 做一把锁，同一时刻只允许一个训练任务
flock -n /var/lock/huacheng-train.lock \
  llamafactory-cli train finetune/configs/sft_qwen25_7b_lora.yaml \
  || echo "已有训练任务在跑，请稍后（查看：ps aux | grep llamafactory）"
```

规模再大就上 Slurm 或 K8s + Volcano，这属于平台工程范畴，见 [第 10.1 章 生产架构设计与部署拓扑](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md)。

### 14.4 安全与合规

| 风险 | 说明 | 对策 |
|---|---|---|
| 训练数据含个人信息 | 工单里有客户姓名、电话、地址 | 训练前必须脱敏（第 5.2 章的脱敏流水线），**脱敏在入库时做，不是训练时做** |
| 模型记住敏感信息 | 大模型会记住训练集里的原文 | 脱敏 + 训后做「记忆探测」：拿训练集里的敏感片段做前缀，看模型会不会续写出来 |
| WebUI 暴露公网 | 无鉴权，任何人可训练/对话 | 只在内网开；必须外网访问时套一层带鉴权的反向代理 |
| 基座模型许可 | 不同模型的商用条款不同 | 选型时确认 license（第 1.2 章有对照表） |
| adapter 外泄 | adapter 包含业务知识 | 权重进私有对象存储，走鉴权分发（第 5.5 章） |

### 14.5 监控与降级

训练阶段的"监控"其实是**训练过程的可观测**：

- `report_to: tensorboard` 起一个 TensorBoard，团队共用一个 logdir，谁的实验跑成什么样一目了然；
- 关键告警：`grad_norm` 突然变成 `nan`（数据里有脏样本或学习率过高）、`eval_loss` 连续上升三次（过拟合，该早停）；
- **降级预案**：训练失败时，线上继续用上一个版本的 adapter。这要求你的部署层支持版本切换——这是第 5.5 章「灰度与回滚」的内容。

---

## 十五、本章小结 + 自测题

### 15.1 小结

1. **框架不是"更好的训练算法"，是训练工程的标准化层。** 它解决的是团队复现、review、协作问题，不是效果问题。第一次微调请务必手写脚本，之后团队统一用框架。
2. **`dataset_info.json` 是数据契约。** 注册配错不报错但训废，必须用 `check_dataset.py` 自检并人工看模板渲染结果。OpenAI 风格的 `role`/`content` 数据必须显式写 `tags`。
3. **YAML 六段各司其职**：model / method / dataset / output / train / eval。最容易配错的是 `template`（必须匹配基座）、`train_on_prompt`（必须 false）、`overwrite_cache`（改数据必须 true）。
4. **多卡不等于更快。** 能 DDP 就别 DeepSpeed，能 ZeRO-2 就别 ZeRO-3；PCIe 机器上 ZeRO-3 可能是负优化。
5. **DPO 是 SFT 之后的精调，不是替代。** 学习率低一到两个量级、1~2 轮、开 `pref_ftx` 混入 SFT loss、盯 `rewards/accuracies` 与长度偏差。
6. **内置评测只能做否决项。** C-Eval 掉太多不许上线，但 C-Eval 涨了不代表业务变好。上线依据是业务金标集（第 5.6 章）。
7. **工程化管理才是本章的重点**：命名规范、配置进 git、产物目录、manifest、实验记录表、复现 checklist。验收标准只有一条——**同事只看 manifest 和配置，能独立重跑出相近结果。**

### 15.2 自测题

**Q1**：你的同事把 `dataset_info.json` 配成了下面这样，数据文件是 OpenAI 风格的 `{"messages": [{"role": "user", "content": "..."}, ...]}`。训练能跑起来、loss 也在下降，但训出来的模型完全不会回答业务问题。问题出在哪？你会怎么排查？

```json
{
  "huacheng_sft": {
    "file_name": "huacheng_sft_train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  }
}
```

<details>
<summary>参考答案</summary>

**问题**：缺 `tags`。ShareGPT 格式的默认标签是 `role_tag: from`、`content_tag: value`、`user_tag: human`、`assistant_tag: gpt`。而数据用的是 `role` / `content` / `user` / `assistant`。

结果分两种情况（取决于版本的容错程度）：
- 严格版本会直接报 `KeyError: 'from'`；
- 宽松版本可能把角色识别成未知类型，**导致 assistant 轮没有被正确标记，label 全是 `-100`，或者整条对话被当成 prompt**。此时 loss 仍会下降（模型在拟合极少数被算到的 token），但模型什么都没学到。

**排查三步**：
1. 看训练日志开头的 `training example` 段：`labels` 那一行是不是空的、或者包含了用户问题；
2. 跑 `check_dataset.py`，它会显式解析 `tags` 并打印角色序列；
3. 在 `dataset_info.json` 里补上完整 `tags`：
```json
"tags": {
  "role_tag": "role", "content_tag": "content",
  "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
}
```

**更深一层的教训**：`loss 在下降` 不等于 `训练是对的`。微调最危险的失败模式全都是「不报错的失败」。所以本章反复强调：每次训练都去看 `training example` 那一段。
</details>

**Q2**：团队在单张 4090 上用 SFT 配置训通了。现在加了第二张 4090，同事直接执行 `FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train cfg.yaml`，没改任何参数。训练速度确实快了将近一倍，但训出来的模型效果比单卡版本差。为什么？怎么改？

<details>
<summary>参考答案</summary>

**原因：有效 batch size 翻倍了。**

有效 batch = `per_device_train_batch_size` × `gradient_accumulation_steps` × **卡数**

- 单卡：1 × 16 × 1 = 16
- 双卡：1 × 16 × 2 = **32**

有效 batch 翻倍带来两个连锁变化：
1. **总优化步数减半**。同样 3 个 epoch，参数更新次数从 N 变成 N/2，模型"学的次数"少了一半；
2. **学习率相对变小**。大 batch 下梯度噪声更小，同样的 lr 实际探索能力下降。

**改法（任选其一，推荐第一种）**：
- **保持有效 batch 不变**：`--gradient_accumulation_steps 8`。这样双卡只是把时间缩短，训练动力学与单卡一致，效果应当可复现；
- **保持步数不变、按 batch 缩放学习率**：保留 ga=16，把 lr 从 1e-4 提到约 1.4e-4（平方根缩放）或 2e-4（线性缩放）。但这引入了新变量，需要重新验证。

**工程纪律**：把「有效 batch = bs × ga × 卡数」写进团队的配置注释里，并在 `manifest.json` 里显式记录卡数。换机器、换卡数时，这是第一个要检查的东西。
</details>

**Q3**：华成机电的 SFT 模型上线两周后，售后主管反馈两个问题：(a) 客户问「你们这型号比 XX 品牌好在哪，他们报价多少」，模型会一本正经地编竞品价格；(b) 回答有时四段式、有时一大段，工单系统解析失败率不低。你的组员提议「再补 500 条 SFT 数据重训」。请评价这个方案，并给出你的完整处理方案。

<details>
<summary>参考答案</summary>

**评价：方向对了一半，方法不够高效。**

补 SFT 数据能改善（b）格式问题，但对（a）拒答问题效果差——因为 SFT 的信号是「模仿这条答案」，拒答样本在几千条数据里占比很低，梯度会被淹没。而且「编竞品价格」这种行为，模型本身是有能力拒答的，问题是**它不知道我们更希望它拒答**，这正是偏好信号该解决的事。

**完整方案（四步）**：

**第一步：先定量，别凭感觉。**
建一个 200 条的业务金标集（方法见第 5.6 章），其中：
- 100 条正常业务问答（测格式合规率、内容准确率）；
- 60 条**该拒答**的对抗样本（竞品价格、内部成本、安全红线操作、超纲闲聊）；
- 40 条**信息不足**样本（测模型会不会追问而不是瞎猜）。
先跑一遍当前线上模型，拿到基线：格式合规率 X%、拒答正确率 Y%。**没有基线，你无法证明后面的工作有效。**

**第二步：格式问题——优先用非微调手段。**
在花卡时之前先试两件成本为零的事：
1. 系统提示里显式给四段式模板 + 一个 few-shot 示例；
2. 推理层用结构化输出约束（vLLM 的 `guided_json` / `guided_regex`）。
如果格式合规率能被拉到可接受，**这个问题就不该用微调解决**（第 5.6 章的红灯清单会再讲）。若仍不够，再补 SFT 数据。

**第三步：拒答问题——上 DPO。**
1. 从线上日志里捞出真实的坏回答（这是最宝贵的 rejected 样本，比人造的强得多）；
2. 让售后主管把这些坏回答改写成合格回答，构成 chosen；
3. 目标 800~1500 条偏好对，其中拒答类约占 40%，格式类约占 40%，其余是语气/术语；
4. **控制长度偏差**：统计 chosen/rejected 的长度分布，保证不是「chosen 总是更长」，必要时专门造一批「chosen 更短更准」的样本；
5. 配置：在 SFT adapter 上继续训（`adapter_name_or_path`），`lr=5e-6`、`num_train_epochs=2`、`pref_beta=0.1`、`pref_ftx=0.1`；
6. 训练时盯 `rewards/accuracies`（目标 0.75~0.90）和 `rewards/margins`（稳步上升，不要暴涨）。

**第四步：评审与上线。**
- 通用回归：C-Eval / CMMLU 相对基座下降不超过红线（本书约定 3 个点）；
- 业务评测：在第一步的金标集上对比「上线版 vs SFT+DPO 版」，拒答正确率与格式合规率都必须提升，且正常业务问答的准确率不能下降；
- 人工盲评：抽 50 条让两位售后主管双盲打分，算 Kappa 一致性（第 5.6 章给代码）；
- 上线走灰度：先 10% 流量，观察 3 天再放量（第 5.5 章）；
- 全过程记进 `finetune/EXPERIMENTS.md`，产物目录符合第 11.3 节规范。

**一句话总结**：`补数据重训` 是一个动作，不是一个方案。方案要包含「先定基线 → 分清哪类问题用哪种手段 → 选对训练范式 → 定好上线红线 → 灰度发布」。
</details>

---

**上一章** [5.3 LoRA / QLoRA 实战](./03-LoRA-QLoRA实战.md) | **下一章** [5.5 模型合并、量化与部署](./05-模型合并量化与部署.md)
