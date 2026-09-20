# 附录 A  常见报错速查表

> **怎么用这份表**：遇到报错，先复制 traceback **最后一行**的关键片段，用 `Ctrl+F` 在本文搜。
> 搜不到就去掉路径、变量名、数字，只留报错类型和固定措辞再搜一次（例如把
> `the dim (768) of field data(vector) is not equal to schema dim (1024)` 缩成 `is not equal to schema dim`）。
> 每条给四段：**出现场景 / 根因 / 解决 / 预防**。解决段一律给可直接执行的命令或可直接粘贴的代码。

> **本表的环境基线**（与全书一致，见 [README 技术栈基线](../README.md)）：
> Python 3.11、CUDA 12.1、PyTorch 2.4.x、transformers 4.44+、langchain 0.3.x、langgraph 0.2.x、
> llama-index 0.12.x、vllm 0.6.x、peft 0.13.x、ragas 0.2.x、Milvus 2.4、Elasticsearch 8.x。
> **端口约定**：Attu `8000`、Langfuse `3001`、Postgres `5433`、vLLM `8001`、应用服务 `8080`。
> **环境变量**：`DEEPSEEK_API_KEY`、`MILVUS_URI`、`MILVUS_COLLECTION=huacheng_kb`、`MILVUS_DIM=1024`、`ES_INDEX`、`LANGFUSE_HOST`。
>
> 本表内所有耗时、显存、体积数字均为**示例性数据**，取自作者单机环境（RTX 4090 24G / Ubuntu 22.04 / 64G 内存），
> 你的机器上数值必然不同，请以方法而非数字为准。

---

## 按症状快速跳转

先用"你看到的现象"定位分组，再在组内 `Ctrl+F` 搜具体报错。

| 你看到的现象 | 大概率属于 | 跳转 |
|---|---|---|
| `pip install` / `uv pip install` 卡住、编译报错、找不到 wheel | 环境与安装 | [§1](#1-环境与安装) |
| `torch.cuda.is_available()` 是 `False`，或 GPU 明明在却用不上 | 环境与安装 | [§1](#1-环境与安装) |
| 模型下载 401 / 连接超时 / 下到一半断 | 环境与安装 | [§1](#1-环境与安装) |
| `docker compose up` 后容器反复重启、`Exited (1)` / `Exited (137)` | Docker 与服务 | [§2](#2-docker-与服务) |
| `bind: address already in use` / 端口冲突 | Docker 与服务 | [§2](#2-docker-与服务) |
| Milvus / Elasticsearch / Langfuse 连不上 | Docker 与服务 | [§2](#2-docker-与服务) |
| `CUDA out of memory` / 显存不够 / 显存忽高忽低 | 模型加载与推理 | [§3](#3-模型加载与推理) |
| 模型加载时 `trust_remote_code` / `config.json` / tokenizer 报错 | 模型加载与推理 | [§3](#3-模型加载与推理) |
| vLLM 启不来、启起来又挂、多卡报错 | 模型加载与推理 | [§3](#3-模型加载与推理) |
| 写向量报维度不匹配 / collection 已存在 | RAG 链路 | [§4](#4-rag-链路) |
| 检索**不报错但返回空**、过滤条件一加就没结果 | RAG 链路 | [§4](#4-rag-链路) |
| 中文乱码、PDF 解析出来是空的、分词明显不对 | RAG 链路 | [§4](#4-rag-链路) |
| `ImportError: cannot import name ...` 从 langchain 里导不出东西 | LangChain / LangGraph | [§5](#5-langchain--langgraph) |
| 图跑不动、递归超限、checkpoint 报 `thread_id` | LangChain / LangGraph | [§5](#5-langchain--langgraph) |
| 结构化输出解析失败、工具 schema 被拒、流式拿不到 token | LangChain / LangGraph | [§5](#5-langchain--langgraph) |
| 训练 loss 是 `nan` / 一直不降 / 训完模型变傻 | 微调 | [§6](#6-微调) |
| adapter 加载不上、合并后输出乱 | 微调 | [§6](#6-微调) |
| 调 API 报 429 / 401 / 上下文超长 / 流式断流 | API 调用 | [§7](#7-api-调用) |
| judge 输出解析不了、指标算出来是 `NaN`、评测账单爆炸 | 评测 | [§8](#8-评测) |
| 上线后 P99 抖、内存只涨不降、并发一高结果串味 | 生产 | [§9](#9-生产) |
| 以上都不是 / 完全没头绪 | 通用方法论 | [§10](#10-排错通用方法论) |

**分组条目数**：环境与安装 14 条、Docker 与服务 14 条、模型加载与推理 15 条、RAG 链路 15 条、
LangChain/LangGraph 13 条、微调 14 条、API 调用 12 条、评测 8 条、生产 9 条，合计 **114 条**。

---

## 1. 环境与安装

这一组的坑有一个共同特征：**报错发生的位置离真正的原因很远**。
`import torch` 时报的错，根子往往在三天前装驱动的那一步。所以本组的"预防"比"解决"更值钱。

### E01 · `CUDA error: no kernel image is available for execution on the device`

**出现场景**：装完 PyTorch，第一次 `model.to("cuda")` 或跑第一个 forward 时炸。常见于用了较新的显卡（如 RTX 40 系、H100）配了较老的 torch。

**根因**：PyTorch 的二进制 wheel 是按 **compute capability（算力架构）** 预编译的。你的显卡架构不在这个 wheel 编译进去的架构列表里，于是找不到可执行的 kernel。不是驱动坏了，是**版本组合不对**。

**解决**：先确认自己的算力号，再装对应 CUDA 版本的 torch。

```bash
# 1. 看显卡算力号（4090 = 8.9，A100 = 8.0，V100 = 7.0，T4 = 7.5）
nvidia-smi --query-gpu=name,compute_cap --format=csv

# 2. 看驱动支持的最高 CUDA 版本（右上角 CUDA Version）
nvidia-smi

# 3. 卸掉装错的 torch，按官方索引重装（本书基线 CUDA 12.1）
uv pip uninstall torch torchvision torchaudio
uv pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip 等价命令
# pip uninstall -y torch torchvision torchaudio
# pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 验证：应输出 True 和你的卡名
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**预防**：把"驱动 CUDA 版本 / torch cu 版本 / 显卡算力"这三项写进项目 `README` 的环境章节，新人入职照抄。本书 [00-前置准备/01-开发环境搭建](../00-前置准备/01-开发环境搭建（Python-CUDA-Docker）.md) 里给的版本矩阵就是干这个用的。

---

### E02 · `AssertionError: Torch not compiled with CUDA enabled`

**出现场景**：`torch.cuda.is_available()` 返回 `False`，但 `nvidia-smi` 一切正常。

**根因**：装成了 **CPU 版 torch**。最常见的触发方式是：某个依赖（比如 `sentence-transformers`、`ragas`、某个工具包）在解析依赖时把 `torch` 从默认 PyPI 源又拉了一遍，覆盖掉了你之前从 `download.pytorch.org` 装的 GPU 版。从 PyPI 装的 `torch` 在 Linux 上默认带 CUDA，但如果 resolver 选了 `+cpu` 的变体或者你在 `requirements.txt` 里裸写了 `torch==2.4.1`，就可能中招。

**解决**：

```bash
# 看版本号后缀：带 +cpu 就是 CPU 版，带 +cu121 或纯 2.4.1 且 torch.version.cuda 非 None 才是 GPU 版
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 重装 GPU 版，并锁死不让后续安装覆盖
uv pip install --force-reinstall torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

在 `pyproject.toml` 里显式声明索引（uv 支持），避免被覆盖：

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu121" }

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
```

**预防**：装完任何一批新依赖，都跑一次环境自检脚本。本书在前置准备章给过 `scripts/check_env.py`，核心就三行：打印 torch 版本与 CUDA 版本、打印可见 GPU、跑一次 1024×1024 矩阵乘法计时。养成"装完就自检"的肌肉记忆。

---

### E03 · `RuntimeError: The NVIDIA driver on your system is too old`

**出现场景**：在公司老服务器上装了新版 torch。

**根因**：CUDA runtime（torch 自带）需要不低于某个版本的驱动。CUDA 12.x 通常要求驱动 ≥ 525（Linux）。驱动比 runtime 老就会拒绝启动。

**解决**：两条路，二选一。

```bash
# 路线 A：升驱动（需要 root，且要停掉所有用 GPU 的进程）
nvidia-smi --query-gpu=driver_version --format=csv
sudo apt-get install -y nvidia-driver-550   # 版本按发行版实际可用的来
sudo reboot

# 路线 B：降 torch 到匹配驱动的 CUDA 版本（没有 root 权限时唯一选择）
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

**预防**：生产机的驱动版本由运维统一管理并写进资产表。开发机上用 Docker（`nvidia/cuda:12.1.1-runtime-ubuntu22.04` 这类基础镜像）隔离 CUDA runtime，宿主机只管驱动，能消掉一大半这类问题。

---

### E04 · `CUDA Setup failed despite GPU being available` / `libbitsandbytes_cpu.so: undefined symbol`

**出现场景**：准备跑 QLoRA，`import bitsandbytes` 或 `BitsAndBytesConfig(load_in_4bit=True)` 时炸。

**根因**：bitsandbytes 需要在安装时定位到 CUDA 运行时库（`libcudart.so`）。定位失败就退化成 CPU 版共享库，而 CPU 版里没有 4bit 量化那些符号，于是 `undefined symbol`。典型诱因：容器里只有 runtime 没有 toolkit、`LD_LIBRARY_PATH` 没包含 CUDA 目录、或者装了 conda 版 cudatoolkit 与系统版打架。

**解决**：

```bash
# 1. 先看 bitsandbytes 自己的诊断输出，它会打印它找到的 CUDA 路径
python -m bitsandbytes

# 2. 确认 libcudart 在哪
find / -name "libcudart.so*" 2>/dev/null | head

# 3. 补上环境变量（路径按上一步实际结果改）
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. 版本太老的 bitsandbytes 对新卡支持不好，升到 0.43+ 再试
uv pip install -U "bitsandbytes>=0.43.0"
```

**预防**：微调环境统一用镜像交付（`FROM nvidia/cuda:12.1.1-devel-ubuntu22.04`，注意是 **devel** 不是 runtime），别让每个人自己拼 CUDA 环境。

---

### E05 · `ModuleNotFoundError: No module named 'torch'`（在装 flash-attn 时）

**出现场景**：`pip install flash-attn`，日志里 `Preparing metadata (setup.py)` 阶段就失败，说找不到 torch —— 可你明明装了 torch。

**根因**：flash-attn 的 `setup.py` 在构建阶段要 `import torch` 来决定编译参数，而 pip 默认在**隔离的构建环境**里执行 setup.py，那个环境里没有 torch。

**解决**：关掉构建隔离。

```bash
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation
# pip 等价：pip install flash-attn --no-build-isolation
```

**预防**：优先用官方预编译 wheel，别自己编。到 flash-attention 仓库的 Releases 页面按 `torch 版本 + CUDA 版本 + Python 版本 + cxx11abi` 四个维度挑对应的 `.whl` 直接装，编译 40 分钟变成下载 1 分钟。

---

### E06 · `c++: fatal error: Killed signal terminated program cc1plus`

**出现场景**：编译 flash-attn / deepspeed / 某些自定义算子时，编译到一半进程被杀。

**根因**：**内存不够**，不是磁盘也不是 CUDA。flash-attn 会并行编译几十个 CUDA kernel，每个 `nvcc` 进程吃 2~4 GB，默认按 CPU 核数并行，16 核机器瞬间吃掉 50GB+，OOM killer 出手。

**解决**：限制并行编译数。

```bash
# 把编译并行度压到 4（16G 内存建议 2）
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

# 确认是不是被 OOM killer 杀的
dmesg | grep -i "killed process" | tail -5
```

**预防**：还是那句话——**能下 wheel 就别编译**。确实要编译时先 `free -g` 看可用内存，按"可用 GB / 3"设 `MAX_JOBS`。

---

### E07 · `ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443)`

**出现场景**：国内网络装大包（torch、nvidia-cudnn 之类几百 MB 的），下到 30% 卡死。

**根因**：默认 PyPI 源跨境，大文件超时。

**解决**：换国内镜像源并调大超时。

```bash
# 临时用
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout 120 langchain

# 永久配置（pip）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 120

# uv 用环境变量，写进 ~/.bashrc
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

**预防**：注意 **PyTorch 的 cu 版 wheel 不在国内镜像的常规索引里**，需要保留 `--index-url https://download.pytorch.org/whl/cu121` 单独装，或者用镜像站提供的 pytorch 专用路径。别把全局源改了就以为万事大吉。

---

### E08 · `401 Client Error: Unauthorized for url: https://huggingface.co/...`

**出现场景**：下载 Qwen2.5、Llama 系列或任何 gated 模型时。

**根因**：两种。一是模型是 gated 的，需要先在网页上同意协议；二是你压根没登录，或者 token 过期/权限不足（只有 `read` 权限的 token 访问不了私有库）。

**解决**：

```bash
# 登录（token 在 HF 网站 Settings -> Access Tokens 生成，选 read 权限）
huggingface-cli login
# 或者用环境变量，适合 CI
export HF_TOKEN=hf_xxxxxxxxxxxx

# 验证身份
huggingface-cli whoami
```

代码里显式传：

```python
from transformers import AutoModelForCausalLM
import os

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    token=os.environ["HF_TOKEN"],
)
```

**预防**：国内团队直接改用 ModelScope 拉国产模型，绕开 gated 流程：

```bash
uv pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

---

### E09 · `OSError: We couldn't connect to 'https://huggingface.co' to load this file`

**出现场景**：明明模型已经下到本地缓存了，重跑一次却报连不上。

**根因**：transformers 默认每次加载都要去 Hub 校验 etag（看有没有新版本）。网络不通时，如果本地缓存不完整或者你传的是模型 ID 而不是本地路径，就会直接失败。

**解决**：

```bash
# 方案 1：强制离线模式，只用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 方案 2：走国内镜像端点
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

生产环境一律传**本地绝对路径**而不是模型 ID：

```python
MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"   # 不是 "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
```

**预防**：把模型当作**制品**管理——下载一次、校验 sha256、推进内网对象存储、部署时从内网拉。生产机不应该有公网出口，也就不该依赖 Hub 可达。

---

### E10 · `ProxyError` / 本地服务明明起着却 `Connection refused`

**出现场景**：设了 `http_proxy` 翻墙下模型，然后连本机 Milvus（`localhost:19530`）或 vLLM（`localhost:8001`）就连不上了。

**根因**：`requests` / `httpx` 会把**所有**请求（包括 localhost）都扔给代理，代理当然连不上你本机的服务。

**解决**：配 `no_proxy` 白名单。

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# 关键：本地与内网地址不走代理
export no_proxy="localhost,127.0.0.1,::1,*.internal,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export NO_PROXY="$no_proxy"
```

Docker 容器里访问宿主机服务同理，注意容器里的 `localhost` 是容器自己：

```yaml
# compose 里用服务名互访，不要用 localhost
environment:
  MILVUS_URI: http://milvus-standalone:19530
  NO_PROXY: localhost,127.0.0.1,milvus-standalone,elasticsearch,postgres
```

**预防**：把代理只加在"下载"这一步（`HF_ENDPOINT` + 临时 `https_proxy`），跑服务时不开代理。写两个 shell 函数 `proxy_on` / `proxy_off` 切换。

---

### E11 · `OSError: [Errno 28] No space left on device`

**出现场景**：训练跑到第 3 个 epoch 存 checkpoint 时挂；或者 Milvus 容器突然进入只读。

**根因**：磁盘满。大模型项目的磁盘黑洞排行：HF 缓存（`~/.cache/huggingface`，几个 7B 模型就 60GB+）、训练 checkpoint（每个 epoch 存一份全量的话，7B 全参微调一份 28GB）、Docker 镜像与日志、Milvus 的 MinIO 数据。

**解决**：

```bash
# 1. 定位大户
df -h
du -h --max-depth=1 ~/.cache /var/lib/docker /data 2>/dev/null | sort -hr | head -20

# 2. 清 HF 缓存里没用的 revision（交互式，安全）
huggingface-cli delete-cache

# 3. 清 Docker（注意 -a 会删掉所有没在用的镜像）
docker system df
docker system prune -a --volumes

# 4. 训练时限制 checkpoint 数量
```

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output/lora-qwen25-7b",
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,          # 只保留最近 2 个，老的自动删
    load_best_model_at_end=True,
)
```

**预防**：把 HF 缓存和训练输出都指到大盘，并配磁盘告警（> 80% 报警）。

```bash
export HF_HOME=/data/hf-cache
export HF_HUB_CACHE=/data/hf-cache/hub
```

---

### E12 · uv 装的包 conda 环境里 import 不到（`ModuleNotFoundError` 但明明装过）

**出现场景**：`conda activate rag`，然后 `uv pip install langchain`，再 `python -c "import langchain"` 报找不到。

**根因**：`uv pip install` 默认装到 **uv 自己发现的虚拟环境**（当前目录的 `.venv`），不认 conda 的激活状态。于是包装进了 `.venv`，你用的却是 conda 的 python。

**解决**：二选一，**不要混用**。

```bash
# 方案 A（推荐）：纯 uv，彻底不用 conda
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
which python   # 必须指向 .venv/bin/python

# 方案 B：非要用 conda，就让 uv 明确指向 conda 的解释器
conda activate rag
uv pip install --python "$(which python)" langchain
```

**预防**：团队里定一条规矩——**一个项目只用一种环境管理器**，在 `README` 第一行写清楚。判断当前用的是谁：

```bash
python -c "import sys; print(sys.prefix); print(sys.executable)"
```

---

### E13 · `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

**出现场景**：装完 `sentence-transformers` 或某个老库，import 时炸。

**根因**：`huggingface_hub` 在演进中移除了一些老 API（如 `cached_download`），而依赖它的老版本库还在调。典型的**上游破坏性变更 + 下游没跟上**。

**解决**：要么升下游，要么钉住上游。

```bash
# 首选：升级依赖它的库
uv pip install -U sentence-transformers

# 兜底：钉住 huggingface_hub 到兼容版本（查下游库的 setup.py 要求）
uv pip install "huggingface_hub==0.25.2"

# 查谁在依赖它
uv pip show huggingface_hub
python -m pipdeptree -r -p huggingface_hub   # 需先装 pipdeptree
```

**预防**：用 lockfile。`uv lock` / `uv export --format requirements-txt > requirements.lock` 生成锁文件提交到仓库，部署时 `uv pip sync requirements.lock`，从根上杜绝"今天能跑明天不能跑"。

---

### E14 · `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`

**出现场景**：在较老的 Linux 发行版（CentOS 7、Ubuntu 18.04）上 import `torch`、`onnxruntime` 或 Milvus 客户端。

**根因**：新编译的 C++ 扩展需要新版 libstdc++，系统自带的太老。conda 环境里也常见——conda 装了自己的 libstdc++ 但版本比 wheel 要求的低。

**解决**：

```bash
# 看系统有哪些 GLIBCXX 版本
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5

# conda 环境下直接升 conda 自己的那份（最常见的修法）
conda install -c conda-forge libstdcxx-ng -y

# 系统级（Ubuntu）
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && sudo apt update && sudo apt install -y libstdc++6
```

**预防**：老系统上一律跑容器，别在宿主机装 Python 环境。`nvidia/cuda:12.1.1-runtime-ubuntu22.04` 这类镜像自带够新的 glibc / libstdc++。

---

## 2. Docker 与服务

本书依赖的服务栈：Milvus（含 etcd + MinIO）、Elasticsearch、Postgres、Langfuse、vLLM。
这一组的排查套路高度统一：**看容器状态 → 看容器日志 → 看端口 → 看挂载权限 → 看资源限制**。

```bash
# 这四条命令是本组所有问题的起手式，背下来
docker compose ps                       # 谁没起来
docker compose logs --tail=200 <服务名>  # 为什么没起来
ss -lntp | grep -E '19530|9200|5433|3001|8000|8001|8080'   # 端口被谁占了
docker stats --no-stream                # 谁在吃内存
```

### D01 · Milvus standalone 反复重启，日志里 `failed to connect etcd` / `context deadline exceeded`

**出现场景**：`docker compose up -d` 之后 `docker compose ps` 显示 milvus 状态在 `Restarting` 之间循环。

**根因**：Milvus standalone 依赖 etcd（元数据）和 MinIO（对象存储）。这两个没起来或起得比 Milvus 慢，Milvus 连不上就退出，然后被 restart policy 拉起来，循环往复。**依赖顺序问题**。

**解决**：先确认依赖，再加健康检查。

```bash
docker compose ps                         # 看 etcd / minio 是不是 healthy
docker compose logs --tail=100 etcd
docker compose logs --tail=100 minio
```

在 compose 里用 `depends_on: condition: service_healthy` 而不是裸 `depends_on`：

```yaml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 10s
      timeout: 5s
      retries: 6

  milvus-standalone:
    image: milvusdb/milvus:v2.4.15
    command: ["milvus", "run", "standalone"]
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
```

**预防**：给应用侧加连接重试（本书 `core/` 里的连接封装都带指数退避），别假设"compose up 完就能连"。启动脚本里加一句 `until` 轮询等待。

---

### D02 · `Error starting userland proxy: listen tcp4 0.0.0.0:2379: bind: address already in use`

**出现场景**：机器上装过 k3s、或者之前跑过另一套 compose，再起 Milvus 时 etcd 端口冲突。

**根因**：2379/2380 是 etcd 的标准端口，被别的 etcd 实例（k3s 内置、Kubernetes 控制面）占了。

**解决**：先查谁占了，再决定停它还是改自己。

```bash
sudo ss -lntp | grep 2379
# 输出里 users:(("etcd",pid=1234,...)) 就是占用者

# 选项 A：如果是遗留容器，停掉
docker ps -a | grep etcd
docker rm -f <container_id>

# 选项 B：不对外暴露 etcd 端口（推荐——Milvus 走容器网络访问，根本不需要映射到宿主机）
```

```yaml
services:
  etcd:
    # 删掉 ports 映射即可，同一 compose 网络内用服务名 etcd:2379 互访
    # ports:
    #   - "2379:2379"
    expose:
      - "2379"
```

**预防**：**只暴露真正需要从宿主机访问的端口**。本书 compose 里对外暴露的只有 Milvus 19530、Attu 8000、ES 9200、Postgres 5433、Langfuse 3001，其余全部只在容器网络内可见。

---

### D03 · `max virtual memory areas vm.max_map_count [65530] is too low, increase to at least [262144]`

**出现场景**：Elasticsearch 8.x 容器启动后立刻 `Exited (78)`。

**根因**：ES 用 mmap 读索引文件，需要比 Linux 默认更大的 `vm.max_map_count`。这是**宿主机内核参数**，在容器里改不了。

**解决**：

```bash
# 临时生效（重启宿主机后失效）
sudo sysctl -w vm.max_map_count=262144

# 永久生效
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 验证
sysctl vm.max_map_count
```

WSL2 下需要在 `%USERPROFILE%\.wslconfig` 里配 `kernelCommandLine = sysctl.vm.max_map_count=262144`，然后 `wsl --shutdown` 重启。

**预防**：把这条写进项目的 `scripts/prepare_host.sh`，和 `docker compose up` 放在同一个 make 目标里：

```bash
# scripts/prepare_host.sh
set -e
current=$(sysctl -n vm.max_map_count)
if [ "$current" -lt 262144 ]; then
  echo "[prepare] vm.max_map_count=$current 过低，正在调整..."
  sudo sysctl -w vm.max_map_count=262144
fi
```

---

### D04 · `memory locking requested for elasticsearch process but memory is not locked`

**出现场景**：ES 启动日志里的警告，或直接启动失败。

**根因**：ES 配了 `bootstrap.memory_lock: true`（防止堆被 swap 出去），但容器没有 `IPC_LOCK` 能力或 `memlock` ulimit 太小。

**解决**：在 compose 里补齐两项。

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.15.0
    environment:
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"     # 堆不要超过物理内存一半，且不超过 31g
      - xpack.security.enabled=false      # 教学环境关掉，生产必须开
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536
    ports:
      - "9200:9200"
```

**预防**：教学/开发环境其实可以直接 `bootstrap.memory_lock=false`，省事。生产环境则应该在宿主机上关掉 swap（`swapoff -a`）而不是依赖 memlock。

---

### D05 · `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**出现场景**：`docker run --gpus all ...` 或 compose 里配了 GPU 的服务（如 vLLM 容器）起不来。

**根因**：宿主机没装 NVIDIA Container Toolkit，Docker 不知道怎么把 GPU 透给容器。

**解决**：

```bash
# Ubuntu 安装 nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 让 docker 认识它
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证：容器里能看到 nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

compose 写法（Compose v2）：

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.6.3
    ports:
      - "8001:8000"          # 容器内 8000 -> 宿主机 8001，符合本书端口约定
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**预防**：把 `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi` 作为环境自检脚本的一项，装完机器立刻验一次。

---

### D06 · `WARN[0000] the attribute 'version' is obsolete` / `Unsupported config option for services`

**出现场景**：跑别人给的 `docker-compose.yml`，Compose v2 报警告或直接报不认识的字段。

**根因**：Compose 从 v1（Python 版，命令是 `docker-compose`）演进到 v2（Go 版，命令是 `docker compose`），顶层 `version:` 字段已废弃，且部分字段语义变了（比如 GPU 从 `runtime: nvidia` 改成 `deploy.resources`）。

**解决**：

```bash
# 确认你用的是哪个
docker compose version    # v2，输出类似 Docker Compose version v2.29.x
docker-compose version    # v1，如果还有的话

# 校验 compose 文件（会把最终生效的配置打印出来，非常适合排查）
docker compose config
```

改造要点：删掉顶层 `version:`；`runtime: nvidia` 换成上面 D05 的 `deploy.resources` 写法；`links` 换成 `networks`。

**预防**：项目里统一用 `docker compose`（带空格），并在 `Makefile` 里封装，避免有人手滑用了 v1。

---

### D07 · `AccessDeniedException[/usr/share/elasticsearch/data/nodes]` / Milvus 写卷 `Permission denied`

**出现场景**：用 bind mount 把宿主机目录挂给容器，容器里的进程写不了。

**根因**：容器内进程用的是镜像里定义的 uid（ES 是 1000，Milvus 是 root 但 MinIO 不是），宿主机目录的 owner 对不上。SELinux / AppArmor 也可能掺一脚。

**解决**：

```bash
# ES 的数据目录给 uid=1000
sudo mkdir -p ./volumes/es-data
sudo chown -R 1000:1000 ./volumes/es-data

# Milvus 的三个卷
sudo mkdir -p ./volumes/{etcd,minio,milvus}
sudo chown -R $(id -u):$(id -g) ./volumes

# SELinux 系统（CentOS/RHEL）加 :z 后缀
# volumes:
#   - ./volumes/es-data:/usr/share/elasticsearch/data:z
```

**预防**：优先用 **named volume** 而不是 bind mount，Docker 会自动处理权限：

```yaml
volumes:
  es-data:
  milvus-data:

services:
  elasticsearch:
    volumes:
      - es-data:/usr/share/elasticsearch/data
```

只有确实需要在宿主机直接看文件时才用 bind mount。

---

### D08 · `bind: address already in use`（端口 8000 / 8001 / 3001 / 5433 / 8080）

**出现场景**：起 Attu、vLLM、Langfuse、Postgres 或应用服务时报端口占用。

**根因**：本书的端口约定是为了**避开常见默认端口的冲突**才这么定的，但你机器上可能还跑着别的东西。

**解决**：按下表逐一确认。

| 端口 | 本书用途 | 常见冲突来源 | 处理 |
|---|---|---|---|
| 8000 | Attu（Milvus 控制台） | 别的 uvicorn/django 默认端口 | 改 Attu 映射或停掉对方 |
| 8001 | vLLM OpenAI 兼容端点 | 少见 | — |
| 3001 | Langfuse Web | 前端 dev server | Langfuse 改 `3002`，同步改 `LANGFUSE_HOST` |
| 5433 | Postgres（Langfuse 用） | 宿主机自带 Postgres 用 5432，本书特意错开 | 若 5433 也被占，改 5434 |
| 8080 | 本书应用服务 | Tomcat、Jenkins、各种管理台 | 改应用 `--port` |
| 19530 | Milvus gRPC | 少见 | — |
| 9200 | Elasticsearch | 少见 | — |

```bash
# 查占用
sudo ss -lntp | grep ':8000'
sudo lsof -i :8000          # 备选

# 端口统一走环境变量，改一处生效
# .env
APP_PORT=8080
ATTU_PORT=8000
LANGFUSE_PORT=3001
```

```yaml
services:
  attu:
    image: zilliz/attu:v2.4
    ports:
      - "${ATTU_PORT:-8000}:3000"
```

**预防**：所有端口写进 `.env` 并在项目 `README` 里列一张表。新人克隆仓库第一件事是跑 `scripts/check_ports.sh` 扫一遍冲突。

---

### D09 · Milvus 日志 `Fail to connect to MinIO/S3` / `The Access Key Id you provided does not exist`

**出现场景**：Milvus 起来了但一写数据就报错，或干脆起不来。

**根因**：Milvus 的对象存储配置（endpoint / accessKey / secretKey / bucket）和 MinIO 容器的实际配置对不上。常见于抄了不同版本的 compose 片段拼在一起。

**解决**：三处必须一致。

```yaml
services:
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus-standalone:
    image: milvusdb/milvus:v2.4.15
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000          # 注意是服务名，不是 localhost
      MINIO_ACCESS_KEY_ID: minioadmin    # 必须和上面一致
      MINIO_SECRET_ACCESS_KEY: minioadmin
```

```bash
# 验证 MinIO 可达
docker compose exec milvus-standalone curl -sf http://minio:9000/minio/health/live && echo OK
```

**预防**：不要东拼西凑 compose 片段。从 Milvus 官方仓库拉当前版本对应的 `milvus-standalone-docker-compose.yml` 作为基底，只在上面加自己的服务。

---

### D10 · `could not connect to server: Connection refused (0.0.0.0:5433)`

**出现场景**：Langfuse 容器启动失败，日志里 Prisma 连不上数据库；或者你自己写脚本连 Postgres 失败。

**根因**：本书把 Postgres 映射到宿主机 **5433**（避开系统自带的 5432），但容器**内部**监听的仍是 5432。Langfuse 容器和 Postgres 容器在同一个网络里，应该用 `postgres:5432` 而不是 `localhost:5433`。搞混"容器内地址"和"宿主机地址"是这类问题的头号原因。

**解决**：

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    ports:
      - "5433:5432"        # 宿主机 5433 -> 容器 5432
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langfuse"]
      interval: 10s
      retries: 5

  langfuse:
    image: langfuse/langfuse:2
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      # 容器之间用服务名 + 容器内端口
      DATABASE_URL: postgresql://langfuse:langfuse@postgres:5432/langfuse
      NEXTAUTH_URL: http://localhost:3001
      NEXTAUTH_SECRET: change-me-in-production
      SALT: change-me-too
    ports:
      - "3001:3000"
```

宿主机上的脚本才用 5433：

```bash
psql "postgresql://langfuse:langfuse@127.0.0.1:5433/langfuse" -c "select 1;"
```

**预防**：在 `.env` 里用两组变量名区分：`PG_PORT_HOST=5433` 和 `PG_PORT_INTERNAL=5432`，写 compose 时看名字就不会用错。

---

### D11 · Langfuse 登录后一直跳回登录页 / `NEXTAUTH_URL` 相关报错

**出现场景**：Langfuse 起来了，页面能打开，但登录后立刻被踢回去。

**根因**：NextAuth 要求 `NEXTAUTH_URL` 和你**浏览器里实际访问的地址**完全一致（协议 + 主机 + 端口）。你从 `http://192.168.1.10:3001` 访问，但 `NEXTAUTH_URL` 写的是 `http://localhost:3001`，cookie domain 对不上，session 就建不起来。

**解决**：

```yaml
environment:
  NEXTAUTH_URL: http://192.168.1.10:3001    # 改成实际访问地址
  NEXTAUTH_SECRET: <openssl rand -base64 32 生成>
  SALT: <另一个随机串>
```

```bash
# 生成随机密钥
openssl rand -base64 32

# 改完必须重建容器（改 env 不会自动生效）
docker compose up -d --force-recreate langfuse
```

**预防**：把 `LANGFUSE_HOST`（SDK 侧用）和 `NEXTAUTH_URL`（服务端用）都从同一个 `.env` 变量派生，避免两边写不一致：

```bash
# .env
LANGFUSE_HOST=http://192.168.1.10:3001
```

```yaml
environment:
  NEXTAUTH_URL: ${LANGFUSE_HOST}
```

---

### D12 · 容器状态 `Exited (137)`

**出现场景**：ES、Milvus 或训练容器跑着跑着突然没了，`docker compose ps` 显示 `Exited (137)`。

**根因**：137 = 128 + 9，即被 SIGKILL 杀掉。几乎总是 **OOM**：要么触发了容器的 `mem_limit`，要么触发了宿主机的 OOM killer。

**解决**：

```bash
# 确认是不是 OOM
docker inspect <container> --format '{{.State.OOMKilled}}'    # true 就是
dmesg | grep -i "out of memory" | tail -5

# 看实时内存占用
docker stats --no-stream
```

对症下药：ES 调小 `ES_JAVA_OPTS`；Milvus 调小 `queryNode` 的缓存；训练调小 batch。同时给容器设**合理**的限制（设了才有告警依据）：

```yaml
services:
  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    deploy:
      resources:
        limits:
          memory: 4G        # 堆 2G + 堆外，留一倍余量
```

**预防**：给每个容器都设 `limits.memory`，并接上监控告警。没有限制的容器会在你最不希望的时候把宿主机拖死，连带杀掉别的服务。

---

### D13 · `MilvusException: (code=1, message=can't find collection: huacheng_kb)`

**出现场景**：明明昨天建过 collection，今天重启完就没了。

**根因**：`docker compose down -v` 把数据卷删了；或者用的是 **Milvus Lite / 内存模式**；或者连到了另一个 Milvus 实例（`MILVUS_URI` 指错）。

**解决**：

```python
from pymilvus import connections, utility

connections.connect(alias="default", uri="http://localhost:19530")
print("当前实例上的 collection:", utility.list_collections())
```

```bash
# 确认自己连的是哪个实例
echo $MILVUS_URI
docker compose ps milvus-standalone

# down 的时候不要带 -v，除非你真想清库
docker compose down          # 保留卷
docker compose down -v       # 删卷，数据没了
```

**预防**：建索引的脚本做成**幂等**的，随时可以重跑：

```python
if utility.has_collection(COLLECTION_NAME):
    print(f"{COLLECTION_NAME} 已存在，跳过创建")
else:
    create_collection()
```

并且把"重建索引"做成一条 `make reindex` 命令，数据没了就重跑一遍，而不是手忙脚乱。

---

### D14 · `failed to solve: ... net/http: TLS handshake timeout`（拉镜像超时）

**出现场景**：`docker compose pull` 拉 milvus / elasticsearch / vllm 镜像时卡死。

**根因**：Docker Hub 或 quay.io 在国内访问不稳。

**解决**：配镜像加速器。

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'JSON'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://mirror.ccs.tencentyun.com"
  ],
  "max-concurrent-downloads": 3,
  "log-driver": "json-file",
  "log-opts": {"max-size": "100m", "max-file": "3"}
}
JSON
sudo systemctl daemon-reload && sudo systemctl restart docker
docker info | grep -A3 "Registry Mirrors"
```

镜像源随时可能变动，以上只是示例，用之前先确认可用性。实在拉不动就在能联网的机器上 `docker save` 成 tar 包，拷到内网 `docker load`。

**预防**：生产环境搭私有 registry（Harbor），所有镜像先同步进来再用，顺带解决了供应链安全问题。注意上面 `daemon.json` 里的 `log-opts` —— 不加这个，容器日志会无限增长直到打爆磁盘（见 E11）。

---

## 3. 模型加载与推理

这一组最容易让人怀疑人生：同一段代码，昨天能跑今天不能；A 机器能跑 B 机器不能。
核心心法：**显存问题看占用曲线，加载问题看文件完整性，输出异常看 chat template。**

```bash
# 本组起手式：实时盯显存
watch -n 1 nvidia-smi

# 或者只看关键列
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -l 1
```

### M01 · `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB`

**出现场景**：加载模型时、推理长文本时、训练 batch 调大时。

**根因**：显存不够。要算清楚账：**权重 + KV Cache + 激活值 + 框架开销**。以 Qwen2.5-7B 为例，BF16 权重约 15GB，24G 卡装下权重后只剩约 8GB 给 KV Cache 和激活，长上下文场景很容易超。

显存估算公式（推理）：

$$
\text{显存} \approx \underbrace{P \times B_w}_{\text{权重}} + \underbrace{2 \times L \times H_{kv} \times d_h \times S \times N \times B_{kv}}_{\text{KV Cache}} + \text{激活与开销}
$$

其中 $P$ 为参数量，$B_w$ 为权重字节数（BF16 = 2，INT4 ≈ 0.5），$L$ 为层数，$H_{kv}$ 为 KV 头数（GQA 下远小于注意力头数），$d_h$ 为头维度，$S$ 为序列长度，$N$ 为并发数，$B_{kv}$ 为 KV 精度字节数。

**解决**：按代价从低到高逐条试。

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 1) 换精度：BF16 是底线，别用 FP32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
)

# 2) 4bit 量化（QLoRA / 显存紧张时推理）
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
```

```bash
# 3) vLLM 侧：限制上下文长度和显存占用比例
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16
```

```python
# 4) 训练侧：梯度检查点 + 梯度累积换显存
args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,     # 等效 batch 16，显存按 1 算
    gradient_checkpointing=True,        # 用约 30% 的速度换约 60% 的激活显存
    bf16=True,
    optim="paged_adamw_8bit",           # 优化器状态也能省
)
```

**预防**：上线前跑一次**容量压测**：固定并发数，把输入长度从 512 逐步拉到 `max_model_len`，记录显存峰值曲线，取 85% 水位作为上限。本书 [01-大模型基础与技术选型/04-算力配置与成本测算](../01-大模型基础与技术选型/04-算力配置与成本测算.md) 给了完整的容量规划脚本。

---

### M02 · `... 22.00 GiB reserved in total by PyTorch. If reserved memory is >> allocated memory try setting max_split_size_mb`

**出现场景**：训练跑了几百步之后突然 OOM，但 `nvidia-smi` 显示还有空闲显存。

**根因**：**显存碎片**。PyTorch 的 caching allocator 会缓存释放的块，如果请求的连续块大小和缓存块尺寸不匹配，就会出现"总量够但找不到连续空间"的情况。变长序列训练（每个 batch 长度不同）最容易触发。

**解决**：

```bash
# 换用 expandable_segments，对碎片有明显改善
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 老版本 torch 用这个
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

```python
# 代码侧：在评估/切换阶段主动清缓存
import torch, gc

def free_gpu():
    """释放未被引用的显存缓存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# 打印显存诊断，区分"已分配"和"已保留"
print(f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB  reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
```

**预防**：训练时开启 **length grouping**（`group_by_length=True`），让相近长度的样本进同一批，显著减少碎片；或者直接用固定长度 packing。

---

### M03 · `ValueError: Loading this model requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option trust_remote_code=True`

**出现场景**：加载 ChatGLM、部分 Qwen 老版本、InternLM 等自定义架构模型时。

**根因**：模型仓库里带了自定义的建模代码（`modeling_*.py`），transformers 不敢自动执行，要你显式授权。

**解决**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
)
```

vLLM：`--trust-remote-code`。

**预防**：**这不是一个可以闭眼开的开关**——它等于允许执行模型仓库里的任意 Python 代码。生产环境的做法是：把模型下到本地后，人工 review 一遍 `modeling_*.py` 和 `configuration_*.py`，确认无害，然后把模型固化进内网制品库，之后所有部署都从内网加载。Qwen2.5 系列已经被 transformers 原生支持，**不需要** `trust_remote_code`，如果你在加载 Qwen2.5 时被要求开这个开关，八成是 transformers 版本太旧（见 M15）。

---

### M04 · `ValueError: Asking to pad but the tokenizer does not have a padding token`

**出现场景**：批量推理或训练时对多条样本做 padding。

**根因**：很多 decoder-only 模型（尤其基座模型）的 tokenizer 没定义 `pad_token`，因为自回归生成时本来就不需要 padding。

**解决**：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 生成任务必须左 padding，否则生成会从 pad 后面接着写，输出全是垃圾
tokenizer.padding_side = "left"      # 推理
# tokenizer.padding_side = "right"   # 训练（SFT）
```

如果非要加一个真正的新 token（不推荐，会改词表大小）：

```python
tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
model.resize_token_embeddings(len(tokenizer))   # 必须同步改 embedding 尺寸，否则索引越界
```

**预防**：把 tokenizer 初始化封装成一个函数，全项目只有一处设置 pad_token 和 padding_side，杜绝"训练用右 padding、推理忘了改左 padding"这种隐蔽 bug——它不报错，只是让模型输出变差，最难查。

---

### M05 · `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`

**出现场景**：对着一个 **base**（基座，非 Instruct）模型调 `tokenizer.apply_chat_template()`。

**根因**：base 模型没经过对话微调，自然没有 chat template。

**解决**：

```python
# 确认有没有
print(tokenizer.chat_template)

# 方案 A：换用 Instruct 版本（绝大多数情况的正确答案）
# Qwen2.5-7B  -> Qwen2.5-7B-Instruct

# 方案 B：确实要在 base 上做 SFT，自己定义模板（下面是 ChatML 风格，与 Qwen 系列一致）
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
```

验证渲染结果（**永远要打出来看一眼**）：

```python
msgs = [{"role": "system", "content": "你是华成机电售后助手"},
        {"role": "user", "content": "XJ-200 报 E041 怎么处理？"}]
print(repr(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
```

**预防**：训练和推理**必须用同一份 chat template**。把模板渲染结果作为单元测试的断言对象，模板一改测试就红。

---

### M06 · `ValueError: The model's max seq len (32768) is larger than the maximum number of tokens that can be stored in KV cache (12336)`

**出现场景**：vLLM 启动时报错退出。

**根因**：vLLM 启动时会预分配 KV Cache 显存池。模型声称支持 32K 上下文，但你的显存装不下 32K 的 KV Cache，vLLM 拒绝启动而不是等到运行时才崩。这是个**好设计**。

**解决**：三个旋钮。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct \
  --port 8001 \
  --max-model-len 8192 \            # 旋钮 1：降上下文（最直接）
  --gpu-memory-utilization 0.92 \   # 旋钮 2：提高显存利用率上限（有风险，别超 0.95）
  --kv-cache-dtype fp8 \            # 旋钮 3：KV Cache 用 FP8，省一半（需 Ada/Hopper 架构）
  --enforce-eager                    # 关掉 CUDA graph，省几百 MB，代价是慢一点
```

**预防**：`max_model_len` 应该由**业务真实需求**决定而不是照抄模型上限。本书 RAG 场景下，上下文预算是"系统提示 + 检索到的 5~8 个 chunk + 对话历史"，算下来 8K 足够，硬开 32K 只是白白吃掉 KV Cache 空间、降低并发。详见 [06-Agent智能体/03-记忆系统与上下文工程](../06-Agent智能体/03-记忆系统与上下文工程.md) 的上下文预算分配。

---

### M07 · `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0`

**出现场景**：在 V100（7.0）、T4（7.5）这类老卡上跑 vLLM 或 BF16 训练。

**根因**：BF16 需要 Ampere（8.0）及以上架构的硬件支持。

**解决**：

```bash
# vLLM 上显式指定 fp16
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct --port 8001 --dtype float16
```

```python
# transformers 侧
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
```

**注意**：FP16 的数值范围比 BF16 窄得多，训练时更容易溢出出 NaN（见 F01、F11）。老卡上做训练要配 loss scaling（`fp16=True` 时 Trainer 会自动开 GradScaler），且学习率要更保守。

**预防**：在选型阶段就把卡的算力号记进机器资产表。本书 [01-大模型基础与技术选型/04-算力配置与成本测算](../01-大模型基础与技术选型/04-算力配置与成本测算.md) 里有卡型对照表。

---

### M08 · `AsyncEngineDeadError: Background loop has errored already` / `Engine loop has died`

**出现场景**：vLLM 服务跑了一段时间后，所有请求都返回 500，服务进程还活着但已经是僵尸。

**根因**：vLLM 的后台引擎线程崩了（最常见是 OOM，其次是某个 CUDA kernel 出错），但 HTTP 层还在接请求。一旦引擎死了，进程无法自愈，必须重启。

**解决**：

```bash
# 1. 看真正的错误——它在引擎崩掉的那一刻，不在你现在看到的 500 里
docker compose logs vllm --tail=500 | grep -B 30 "died\|Traceback"

# 2. 重启
docker compose restart vllm
```

生产上必须配健康检查 + 自动重启：

```yaml
services:
  vllm:
    image: vllm/vllm-openai:v0.6.3
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 300s      # 模型加载慢，给足启动时间，否则会被反复重启
```

**预防**：把 `gpu_memory_utilization` 留出余量（0.85 而不是 0.95），并在网关层做**熔断**：连续 N 次失败就把这个后端摘掉，流量切到备用模型（本书的模型网关设计见 [01-大模型基础与技术选型/03-本地部署实战](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md)）。

---

### M09 · `ValueError: Unknown quantization method: awq` / 量化模型加载后输出乱码

**出现场景**：加载 AWQ / GPTQ 量化模型。

**根因**：缺少对应的量化后端库；或者用 transformers 直接加载了一个为 vLLM 准备的量化格式；或者 GGUF 格式被误当成 HF 格式加载（GGUF 是 llama.cpp 的格式，transformers 只有有限支持）。

**解决**：

```bash
# AWQ 推理后端
uv pip install autoawq
# GPTQ 推理后端
uv pip install optimum gptqmodel
```

```bash
# vLLM 加载量化模型：显式指定 quantization
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct-AWQ \
  --port 8001 \
  --quantization awq \
  --dtype float16          # AWQ 通常配 fp16
```

格式对照速查：

| 格式 | 主要用途 | 加载方 | 备注 |
|---|---|---|---|
| AWQ | GPU 推理 | vLLM / autoawq | 精度损失小，速度快 |
| GPTQ | GPU 推理 | vLLM / gptqmodel | 生态成熟 |
| GGUF | CPU / Mac 推理 | llama.cpp / Ollama | **不要**用 transformers 加载 |
| bitsandbytes NF4 | 训练（QLoRA） | transformers + peft | 推理慢，别用于生产服务 |

**预防**：模型资产库里的每个量化模型都记清楚"格式 + 目标推理引擎"，别混用。详见 [05-微调LoRA与PEFT/05-模型合并量化与部署](../05-微调LoRA与PEFT/05-模型合并量化与部署.md)。

---

### M10 · `ValueError: Total number of attention heads (28) must be divisible by tensor parallel size (3)`

**出现场景**：多卡跑 vLLM，随手设了 `--tensor-parallel-size 3`。

**根因**：张量并行是把注意力头切开分给各卡，头数必须能被 TP size 整除。Qwen2.5-7B 是 28 个头，能被 1、2、4、7、14、28 整除，但不能被 3 整除。

**解决**：

```bash
# 查模型的头数
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('/data/models/Qwen2.5-7B-Instruct')
print('num_attention_heads =', c.num_attention_heads)
print('num_key_value_heads =', c.num_key_value_heads)
print('num_hidden_layers   =', c.num_hidden_layers)
"

# TP size 取 2 或 4
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct --port 8001 --tensor-parallel-size 2
```

**预防**：TP size 只在 **1 / 2 / 4 / 8** 里选。三卡机器的正确用法是 TP=1 起三个副本做数据并行（吞吐更高），而不是硬凑 TP=3。

---

### M11 · `NCCL error: unhandled system error` / `torch.distributed ... Timed out (default 1800000ms)`

**出现场景**：多卡训练或多卡 vLLM 启动时卡住，半小时后超时退出。

**根因**：多卡通信初始化失败。常见原因：容器没开 `--shm-size`（共享内存默认只有 64MB，NCCL 不够用）、P2P 不可用、防火墙挡了、或者某张卡上有别的进程占着。

**解决**：

```bash
# 1. 打开 NCCL 调试日志，它会明确告诉你卡在哪一步
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 2. 容器共享内存要给够
docker run --gpus all --shm-size=16g --ipc=host ...
```

```yaml
services:
  train:
    shm_size: '16gb'
    ipc: host
```

```bash
# 3. 单机多卡 P2P 有问题时禁掉（会慢，但能跑起来）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1       # 没有 InfiniBand 时

# 4. 确认卡上没有别的进程
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**预防**：多卡任务启动前统一跑一遍 `torchrun --nproc_per_node=N -m torch.distributed.run` 的 all-reduce 冒烟测试，5 秒就能验出通信是否正常，比训练跑半小时才超时强得多。

---

### M12 · `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

**出现场景**：手写推理循环、或者用 `device_map="auto"` 加载后又手动移动了某些张量。

**根因**：输入张量在 CPU，模型在 GPU（或者模型被 `device_map` 切到了多张卡，你把输入固定送到了 cuda:0）。

**解决**：

```python
# 正确做法：把输入送到模型第一层所在的设备
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=512)

# device_map 切分了多卡时，用 hf_device_map 查每层在哪
print(model.hf_device_map)
```

**预防**：不要在同一份代码里既用 `device_map="auto"` 又手动 `.to("cuda:0")`。要么全交给 accelerate，要么全自己管。生产推理一律走 vLLM，根本不用操心这个。

---

### M13 · `OSError: /data/models/xxx does not appear to have a file named config.json`

**出现场景**：从 HF 或 ModelScope 下完模型后加载失败。

**根因**：下载不完整（断网中途停了）、或者路径指到了上一级目录/snapshot 目录、或者 `git clone` 时没装 git-lfs 导致下下来的是几 KB 的指针文件。

**解决**：

```bash
# 1. 看目录里到底有什么
ls -lh /data/models/Qwen2.5-7B-Instruct/

# 正常应该有：config.json、tokenizer.json、tokenizer_config.json、
# generation_config.json、model-0000x-of-0000y.safetensors、model.safetensors.index.json

# 2. 检查是不是 lfs 指针（文件只有 100 多字节就是）
find /data/models -name "*.safetensors" -size -1M -exec ls -lh {} \;

# 3. 用 huggingface-cli 重下（它支持断点续传，比 git clone 可靠得多）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/Qwen2.5-7B-Instruct \
  --resume-download
```

**预防**：下载完做一次**完整性校验**，把它写成脚本：

```python
import json
from pathlib import Path

def verify_model_dir(path: str) -> None:
    """校验 HF 模型目录的必需文件与分片完整性"""
    p = Path(path)
    required = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required if not (p / f).exists()]
    assert not missing, f"缺少文件: {missing}"
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        shards = set(json.loads(idx.read_text())["weight_map"].values())
        lost = [s for s in shards if not (p / s).exists()]
        assert not lost, f"缺少权重分片: {lost}"
    print(f"{path} 校验通过，共 {len(list(p.glob('*.safetensors')))} 个分片")
```

---

### M14 · `RuntimeError: probability tensor contains either inf, nan or element < 0`

**出现场景**：`model.generate()` 采样时报错，往往在生成到几十上百个 token 之后。

**根因**：logits 里出现了 NaN/Inf。常见诱因：FP16 数值溢出（尤其老卡上跑本该用 BF16 的模型）、采样参数不合法（`temperature=0` 配 `do_sample=True`）、或者模型权重本身就坏了（量化转换出错、微调炸了）。

**解决**：

```python
# 1) 贪心解码时不要开 do_sample
out = model.generate(**inputs, do_sample=False, max_new_tokens=512)

# 2) 要采样就给合法参数
out = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,      # 必须 > 0
    top_p=0.9,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

# 3) 换 BF16（如果卡支持）
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

# 4) 排查权重是否已损坏
import torch
bad = [n for n, p in model.named_parameters() if torch.isnan(p).any() or torch.isinf(p).any()]
print("异常参数:", bad[:10])
```

**预防**：微调产出的模型在合并/导出之后，**第一件事是跑一遍权重 NaN 扫描**，再跑 20 条冒烟样例，通过了才进模型库。

---

### M15 · `KeyError: 'qwen2'` / `The checkpoint you are trying to load has model type 'qwen2' but Transformers does not recognize this architecture`

**出现场景**：加载 Qwen2/Qwen2.5 时报不认识的架构。

**根因**：transformers 版本太旧，还没有内置该架构的支持。

**解决**：

```bash
uv pip install -U "transformers>=4.44.0"
python -c "import transformers; print(transformers.__version__)"
```

架构支持的版本门槛（示例，具体以官方 release note 为准）：

| 模型架构 | 需要的 transformers 版本 |
|---|---|
| `qwen2` / `qwen2.5` | 4.37+（建议 4.44+） |
| `qwen2_vl` | 4.45+ |
| `llama3` | 4.40+ |

**预防**：新模型出来先查它 `config.json` 里的 `model_type`，然后到 transformers 的 release note 里确认从哪个版本开始支持。别一上来就加 `trust_remote_code=True` 绕过——那只是把问题从"加载失败"变成"加载了但行为诡异"。

---
## 4. RAG 链路

RAG 的坑有个恶劣特点：**大部分不报错**。检索质量差、召回为空、中文切得稀碎，程序都跑得好好的，
只是答案变成一本正经的胡说八道。所以本组里有相当一部分"错误信息"其实是**症状描述**而非异常文本。

排查 RAG 链路的标准顺序（从后往前，别一上来就怀疑模型）：

```
答案不对
  ↓ 把送进 LLM 的完整 prompt 打出来
上下文里有正确答案吗？
  ├─ 有 → 是生成问题：prompt 写法 / 模型能力 / 上下文太长被"迷失在中间"
  └─ 没有 → 往前一步：rerank 后的 top-k 里有吗？
        ├─ 有 → rerank 把它排下去了：看 rerank 分数分布
        └─ 没有 → 再往前：召回的 top-50 里有吗？
              ├─ 有 → 召回够了，是排序问题
              └─ 没有 → 是检索问题：查询改写 / embedding / 分块 / 这段内容压根没入库
```

### R01 · `the dim (768) of field data(vector) is not equal to schema dim (1024)`

**出现场景**：往 Milvus 写向量时。

**根因**：建 collection 时声明的维度和实际 embedding 模型输出的维度不一致。本书统一用 `bge-m3`，维度 **1024**，对应 `MILVUS_DIM=1024`。如果中途换成 `bge-large-zh-v1.5`（1024，一致）没事，但换成 `text2vec-base`（768）或 `bge-small`（512）就会撞上。

**解决**：先查实际维度，再决定是改 schema 还是换模型。

```python
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-m3")
print("实际维度:", m.get_sentence_embedding_dimension())    # 1024
```

```python
from pymilvus import utility, Collection

col = Collection("huacheng_kb")
for f in col.schema.fields:
    if f.dtype.name.endswith("VECTOR"):
        print("schema 维度:", f.params.get("dim"))
```

维度必须变的话，**只能重建 collection 并全量重灌**，没有在线改维度这回事：

```python
if utility.has_collection("huacheng_kb"):
    utility.drop_collection("huacheng_kb")
# 然后按新维度重建 + 重新灌数据
```

**预防**：把维度写进配置而不是硬编码，并在服务启动时做一次断言：

```python
# core/config.py 已有 settings，这里只做校验
assert embedder.get_sentence_embedding_dimension() == settings.milvus_dim, (
    f"embedding 维度 {embedder.get_sentence_embedding_dimension()} "
    f"与 MILVUS_DIM={settings.milvus_dim} 不符，拒绝启动"
)
```

**启动时炸，好过灌了 50 万条之后才发现。**

---

### R02 · `CollectionAlreadyExistException` / collection 存在但字段对不上

**出现场景**：重跑建库脚本。

**根因**：脚本没做幂等处理；或者做了 `has_collection` 判断但 schema 已经改了，老 collection 的字段与新代码期望的不符，于是后面插入时报字段不存在。

**解决**：建库脚本要支持三种模式。

```python
from pymilvus import utility, Collection, CollectionSchema, FieldSchema, DataType

def ensure_collection(name: str, dim: int, mode: str = "reuse") -> Collection:
    """确保 collection 存在。mode: reuse(复用) / recreate(删重建) / strict(schema 不符就报错)"""
    exists = utility.has_collection(name)
    if exists and mode == "recreate":
        utility.drop_collection(name)
        exists = False
    if exists:
        col = Collection(name)
        got = {f.name for f in col.schema.fields}
        want = {"pk", "vector", "text", "doc_id", "source", "model_no", "error_code"}
        if not want.issubset(got):
            msg = f"collection {name} schema 不符，缺少字段 {want - got}"
            if mode == "strict":
                raise RuntimeError(msg)
            print("[warn]", msg)
        return col
    fields = [
        FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema("text", DataType.VARCHAR, max_length=4096),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=128),
        FieldSchema("source", DataType.VARCHAR, max_length=256),
        FieldSchema("model_no", DataType.VARCHAR, max_length=64),
        FieldSchema("error_code", DataType.VARCHAR, max_length=32),
    ]
    schema = CollectionSchema(fields, description="华成机电售后知识库")
    return Collection(name, schema)
```

**预防**：**collection 名字带版本号**：`huacheng_kb_v1`、`huacheng_kb_v2`，用别名（alias）指向当前生效的那个。这样重建索引可以后台做完再切别名，实现零停机更新（详见 [10-工程化与生产落地/01-生产架构设计与部署拓扑](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md)）。

---

### R03 · 检索返回空列表（不报错），或 `collection not loaded`

**出现场景**：数据明明插进去了，`search` 却返回 `[]`。

**根因**：Milvus 的数据要**先 flush 落盘、建索引、再 load 进内存**才能被检索。少任何一步都会出现"有数据但搜不到"。

**解决**：

```python
from pymilvus import Collection, utility

col = Collection("huacheng_kb")

# 1. 确认真的有数据
col.flush()
print("实体数:", col.num_entities)          # 0 就是根本没插进去

# 2. 确认索引建了
print("索引:", col.indexes)
if not col.indexes:
    col.create_index(
        field_name="vector",
        index_params={"index_type": "HNSW", "metric_type": "COSINE",
                      "params": {"M": 16, "efConstruction": 200}},
    )
    utility.wait_for_index_building_complete("huacheng_kb")

# 3. load 进内存（这一步最常被忘）
col.load()
print("load 进度:", utility.load_state("huacheng_kb"))
```

标准的"写入后"三连：`flush()` → `create_index()` → `load()`。

**预防**：封装一个 `after_insert()` 帮助函数，所有灌数据的脚本结尾都调它。另外在健康检查接口里加一项：随便发一个查询，断言返回条数 > 0，这样索引没 load 会在健康检查就暴露，而不是等用户来投诉。

---

### R04 · 加了 metadata 过滤条件就搜不到东西

**出现场景**：`expr="model_no == 'XJ-200'"` 一加上，结果从 10 条变 0 条。

**根因**：候选有四个。① 字段值里有不可见字符或大小写差异（存的是 `XJ-200 `，带尾空格）；② 类型不匹配（字段是 VARCHAR，你按数字比较）；③ 表达式语法错（Milvus 的 `expr` 有自己的语法，`and` 要写 `&&` 或 `and`，字符串必须单/双引号包住）；④ **过滤把候选集砍得太狠**，HNSW 在过滤模式下可能扫不到足够的点。

**解决**：分步验证，先不带向量只做标量查询。

```python
# 第一步：只按标量查，确认数据里到底存的是什么
rows = col.query(expr='model_no == "XJ-200"', output_fields=["model_no", "doc_id"], limit=5)
print(len(rows), rows[:2])

# 查不到？把该字段所有取值抽样看看
rows = col.query(expr="pk > 0", output_fields=["model_no"], limit=100)
print(sorted({r["model_no"] for r in rows}))    # 看真实取值，很可能是 'XJ200' 或 'XJ-200 '
```

```python
# 第二步：确认语法。几种常用写法
expr = 'model_no == "XJ-200"'
expr = 'model_no in ["XJ-200", "XJ-200-B3"]'
expr = 'error_code == "E041" and source == "manual"'
expr = 'model_no like "XJ-200%"'      # 前缀匹配
```

```python
# 第三步：过滤太狠时，加大 ef 让图索引多走几步
res = col.search(
    data=[query_vec], anns_field="vector", param={"metric_type": "COSINE", "params": {"ef": 256}},
    limit=10, expr=expr, output_fields=["text", "doc_id"],
)
```

**预防**：入库时对所有用于过滤的字段做**规范化**（`strip()` + 统一大小写 + 统一分隔符），并把规范化函数同时用在查询侧。本书在 [02-RAG基础篇/02-文档解析与切分策略](../02-RAG基础篇/02-文档解析与切分策略.md) 的元数据设计一节强调过：**元数据的取值域要可枚举、可测试**。

---

### R05 · `Token indices sequence length is longer than the specified maximum sequence length for this model (1024 > 512). Running this sequence through the model will result in indexing errors`

**出现场景**：调用 embedding 模型或 reranker 时的警告。

**根因**：chunk 比 embedding 模型的最大输入长度还长，超出部分被**静默截断**。`bge-large-zh-v1.5` 最大 512 token，`bge-m3` 最大 8192。截断意味着 chunk 后半段的内容压根没进向量，检索时自然找不到。

**解决**：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

def check_chunk_lengths(chunks: list[str], max_len: int = 8192) -> None:
    """统计切块的 token 长度分布，超长的直接报出来"""
    lens = [len(tok.encode(c)) for c in chunks]
    lens.sort()
    n = len(lens)
    print(f"共 {n} 块  p50={lens[n//2]}  p95={lens[int(n*0.95)]}  max={lens[-1]}")
    over = [i for i, l in enumerate(lens) if l > max_len]
    if over:
        print(f"[警告] {len(over)} 块超过 {max_len} token，将被截断")
```

切分时按 **token** 而不是字符计数：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tok,
    chunk_size=512,          # 这里的单位是 token
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
```

**预防**：把"chunk token 长度分布"作为**入库前的必检项**，p95 超过模型上限 80% 就报警。这个检查 5 行代码，能挡掉 RAG 里最隐蔽的一类质量损失。

---

### R06 · `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 12: invalid start byte`

**出现场景**：读取从 Windows 导出的 CSV 工单、或者老系统导出的 txt。

**根因**：文件不是 UTF-8。国内业务系统导出的文件常见 GBK / GB18030 / UTF-8-BOM 三种。

**解决**：

```python
from pathlib import Path
import chardet     # uv pip install chardet

def read_text_auto(path: str | Path) -> str:
    """自动识别编码读取文本，优先按常见中文编码尝试"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    guess = chardet.detect(raw)
    return raw.decode(guess["encoding"] or "utf-8", errors="replace")
```

```python
import pandas as pd
# 读 CSV 工单，gb18030 是 gbk 的超集，能多覆盖一批生僻字
df = pd.read_csv("data/raw/tickets.csv", encoding="gb18030")
```

```bash
# 命令行确认编码 / 批量转码
file -i data/raw/tickets.csv
iconv -f GB18030 -t UTF-8 data/raw/tickets.csv -o data/raw/tickets_utf8.csv
```

**预防**：数据接入层统一做一次**编码归一化**，落地成 UTF-8 之后再进后续流水线。别让编码问题渗透到下游每一个脚本里。

---

### R07 · PDF 解析出来是空字符串 / 只有页码和页眉

**出现场景**：解析华成机电的产品操作手册 PDF。

**根因**：PDF 是**扫描件**（整页就是一张图），没有文本层。或者是用特殊字体嵌入的矢量图，文本提取出来是乱码。

**解决**：先判断有没有文本层，再决定走哪条路。

```python
import fitz     # PyMuPDF, uv pip install pymupdf

def diagnose_pdf(path: str) -> dict:
    """判断 PDF 是文本型还是扫描型"""
    doc = fitz.open(path)
    stats = {"pages": len(doc), "text_chars": 0, "images": 0}
    for page in doc:
        stats["text_chars"] += len(page.get_text().strip())
        stats["images"] += len(page.get_images())
    stats["avg_chars_per_page"] = stats["text_chars"] / max(stats["pages"], 1)
    # 经验阈值：每页平均不足 50 个字符，基本可判定为扫描件
    stats["likely_scanned"] = stats["avg_chars_per_page"] < 50
    return stats

print(diagnose_pdf("data/raw/XJ-200-操作手册.pdf"))
```

扫描件走 OCR：

```bash
# 方案 A：ocrmypdf，给 PDF 加文本层，原版式保留，后续按普通 PDF 处理
sudo apt install -y ocrmypdf tesseract-ocr-chi-sim
ocrmypdf -l chi_sim+eng --force-ocr input.pdf output.pdf
```

版式复杂（多栏、大量表格）的手册，建议用版面分析类工具（如 MinerU、PP-Structure 这类开源方案）而不是纯文本提取。本书 [02-RAG基础篇/02-文档解析与切分策略](../02-RAG基础篇/02-文档解析与切分策略.md) 有解析工具对比表。

**预防**：建立**入库前质检门禁**——每份文档解析后统计字符数、表格数、图片数，低于阈值的进人工复核队列，不允许静默入库。RAG 系统里"某份关键手册其实根本没入库成功"是最常见的事故。

---

### R08 · Reranker 打分全都差不多 / 长 chunk 被 rerank 截断

**出现场景**：接了 `bge-reranker-v2-m3`，但排序结果和不排几乎一样。

**根因**：Cross-Encoder 的输入是 `[query, passage]` 拼接后送进模型，有最大长度限制。passage 太长会被截断，模型只看到前半段。另一种可能是你把归一化前的原始分数当成概率在用。

**解决**：

```python
from FlagEmbedding import FlagReranker      # uv pip install FlagEmbedding

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

def rerank(query: str, passages: list[str], top_k: int = 5, max_len: int = 1024):
    """对候选段落重排，返回 (原索引, 分数) 列表，按分数降序"""
    # 关键：截断策略——保留段落开头（通常信息密度最高），而不是让模型默认从右侧截
    trimmed = [p[: max_len * 2] for p in passages]     # 中文约 2 字符/token，粗略换算
    pairs = [[query, p] for p in trimmed]
    scores = reranker.compute_score(pairs, normalize=True)   # normalize=True 得到 0~1
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, scores[i]) for i in order[:top_k]]
```

**分数分布诊断**（判断 reranker 是否真的在工作）：

```python
scores = [s for _, s in rerank(q, passages, top_k=len(passages))]
print(f"max={max(scores):.3f} min={min(scores):.3f} 极差={max(scores)-min(scores):.3f}")
# 极差小于 0.1 说明 reranker 没有区分度：要么截断了，要么候选质量本来就都差不多
```

**预防**：用**父子块**结构——子块（短，约 256 token）用于检索和 rerank，命中后取其父块（长，约 1500 token）送给 LLM。这样 rerank 不会被截断，LLM 又能拿到完整上下文。详见 [02-RAG基础篇/02-文档解析与切分策略](../02-RAG基础篇/02-文档解析与切分策略.md)。

---

### R09 · BM25 检索中文效果极差（搜"E041 故障"什么都搜不到）

**出现场景**：用 `rank_bm25` 或 ES 默认分析器做中文全文检索。

**根因**：`rank_bm25` 需要你自己传**已分词的 token 列表**，很多人直接 `text.split()`，中文句子按空格切等于没切，整句变成一个 token，自然匹配不上。ES 的 `standard` 分析器对中文是**逐字切分**，虽然能搜到但相关性很差。

**解决**：

```python
import jieba                                  # uv pip install jieba
from rank_bm25 import BM25Okapi               # uv pip install rank_bm25

# 加载领域词典，保证型号和故障码不被切碎
jieba.load_userdict("data/dict/huacheng_terms.txt")   # 每行: XJ-200-B3 10 n

def tokenize_zh(text: str) -> list[str]:
    """中文分词 + 过滤空白，用于 BM25 语料构建与查询"""
    return [t for t in jieba.lcut(text) if t.strip()]

corpus_tokens = [tokenize_zh(doc) for doc in corpus]
bm25 = BM25Okapi(corpus_tokens)
scores = bm25.get_scores(tokenize_zh("XJ-200 报 E041 怎么处理"))
```

ES 侧装 IK 分词器并显式指定：

```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "zh_analyzer": { "type": "custom", "tokenizer": "ik_max_word" }
      }
    }
  },
  "mappings": {
    "properties": {
      "text":     { "type": "text", "analyzer": "zh_analyzer", "search_analyzer": "ik_smart" },
      "model_no": { "type": "keyword" },
      "error_code": { "type": "keyword" }
    }
  }
}
```

**预防**：**分词结果一定要打出来人工看一眼**：

```python
print(tokenize_zh("XJ-200-B3 主轴过载报 E041"))
# 期望: ['XJ-200-B3', '主轴', '过载', '报', 'E041']
# 如果是 ['XJ', '-', '200', '-', 'B3', ...]，说明词典没生效
```

型号、故障码这类**强标识符**应该同时用 `keyword` 字段做精确匹配兜底，不能只靠分词。本书 [02-RAG基础篇/04-检索重排生成全链路](../02-RAG基础篇/04-检索重排生成全链路.md) 详细讲了这套混合策略。

---

### R10 · `MilvusException: (code=1, message=fail to search on all shard leaders)` / `search request timeout`

**出现场景**：数据量上去之后检索开始超时。

**根因**：常见三种。① collection 没 load 完就来查了；② `ef`/`nprobe` 设太大导致单次检索太慢；③ QueryNode 内存不够，数据被反复换入换出。

**解决**：

```python
from pymilvus import utility

# 1. 看 load 状态与进度
print(utility.load_state("huacheng_kb"))
print(utility.loading_progress("huacheng_kb"))
```

```python
# 2. 调小检索参数看是否立刻恢复（HNSW 的 ef 是最主要的延迟旋钮）
search_params = {"metric_type": "COSINE", "params": {"ef": 64}}   # 从 256 降到 64 试试
res = col.search([qv], "vector", search_params, limit=10, timeout=10)
```

```bash
# 3. 看 QueryNode 内存水位
docker stats --no-stream | grep milvus
docker compose logs milvus-standalone --tail=200 | grep -i "memory\|evict"
```

**预防**：`ef` 与召回率的关系要**实测出一条曲线**再定值，不要拍脑袋。本书 [02-RAG基础篇/03-Embedding与向量数据库](../02-RAG基础篇/03-Embedding与向量数据库.md) 给了参数扫描脚本：固定测试集，遍历 `ef ∈ {32, 64, 128, 256, 512}`，画出「召回率 vs P99 延迟」曲线，选拐点。

---

### R11 · `IndexError: list index out of range`（取 top-k 时）

**出现场景**：检索结果不足 k 条，代码却硬取 `results[k-1]`。

**根因**：库里数据少于 k、或者过滤条件太严、或者 `limit` 和实际返回条数不一致。

**解决**：

```python
# 永远按实际长度取，不要假设一定有 k 条
hits = res[0]
top = list(hits)[: min(top_k, len(hits))]
if not top:
    logger.warning("检索返回为空 query=%s expr=%s", query, expr)
    return {"answer": "知识库里没有找到相关资料，建议转人工。", "sources": []}
```

**预防**：把"检索为空"当成一条**正常业务分支**来设计，而不是异常。RAG 系统必须有明确的拒答路径——这也是准确率工程的一部分，详见 [03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径](../03-RAG进阶与性能优化/05-准确率优化-从70到95的工程路径.md)。

---

### R12 · 相似度分数看起来很奇怪（负数、大于 1、或者全是 0.99）

**出现场景**：打日志时发现召回分数不符合直觉。

**根因**：**度量类型（metric_type）与向量是否归一化不匹配**。三种度量的取值域完全不同：

| metric_type | 含义 | 取值范围 | 越大越相似？ | 要求归一化？ |
|---|---|---|---|---|
| `COSINE` | 余弦相似度 | [-1, 1] | 是 | 库内部处理，不强制 |
| `IP` | 内积 | 无界 | 是 | **必须**，否则等价于受模长影响 |
| `L2` | 欧氏距离 | [0, +∞) | **否**（越小越相似） | 否 |

bge 系列官方推荐**归一化后用内积（等价于余弦）**。如果你用了 `IP` 却没归一化，长文本因为模长大就会天然排前面——这是个非常隐蔽的质量杀手。

**解决**：

```python
import numpy as np
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("BAAI/bge-m3")
vecs = m.encode(texts, normalize_embeddings=True)     # 关键参数
print("模长:", np.linalg.norm(vecs, axis=1)[:5])       # 应该全是 1.0
```

建索引与检索必须用**同一个** metric：

```python
index_params = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
search_params = {"metric_type": "COSINE", "params": {"ef": 128}}   # 必须一致
```

**预防**：全是 0.99 通常不是 bug 而是**特征坍缩**——文档太同质（比如一堆模板化工单），embedding 区分不开。这时候应该去改分块策略或换模型，而不是调检索参数。判断方法：随机抽 100 对**无关**文档算相似度，如果均值也在 0.9 以上，就是坍缩。

---

### R13 · ES `search_phase_execution_exception` / `No mapping found for [score] in order to sort`

**出现场景**：做混合检索时查 ES 那一路挂了。

**根因**：查询里引用了 mapping 里不存在的字段；或者对 `text` 类型字段做排序/聚合（ES 默认不允许，要 `keyword` 子字段）。

**解决**：

```bash
# 看真实 mapping
curl -s "http://localhost:9200/${ES_INDEX}/_mapping?pretty"

# 带 explain 跑查询，ES 会告诉你每一步的打分细节
curl -s -X POST "http://localhost:9200/${ES_INDEX}/_search?explain=true" \
  -H 'Content-Type: application/json' -d '{"query":{"match":{"text":"E041"}},"size":1}' | head -50
```

正确的 mapping 设计（text 用于检索，keyword 用于过滤和排序）：

```json
{
  "properties": {
    "text": {
      "type": "text",
      "analyzer": "zh_analyzer",
      "fields": { "raw": { "type": "keyword", "ignore_above": 256 } }
    },
    "model_no":   { "type": "keyword" },
    "error_code": { "type": "keyword" },
    "updated_at": { "type": "date" }
  }
}
```

**预防**：**mapping 用显式模板，关掉 dynamic mapping**（`"dynamic": "strict"`）。让"写入了一个没定义的字段"这件事在写入时就失败，而不是在查询时才莫名其妙。

---

### R14 · `the length (1873) of str field exceeds max length (1000)`

**出现场景**：往 Milvus 插入 chunk 文本时。

**根因**：`VARCHAR` 字段的 `max_length` 单位是**字节**（UTF-8 下一个中文字符占 3 字节），不是字符数。声明 1000 的字段实际只能存约 333 个中文字。

**解决**：

```python
# 按字节算，给足余量
FieldSchema("text", DataType.VARCHAR, max_length=8192)     # 约 2700 个中文字

# 插入前主动校验并截断（保底，不要指望永远算得准）
def fit_varchar(s: str, max_bytes: int = 8192) -> str:
    """把字符串截断到指定字节数以内，不切断多字节字符"""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")
```

**预防**：更好的做法是**正文不存在向量库里**。向量库只存 `chunk_id` + 向量 + 少量过滤字段，正文存在对象存储或关系库里，检索命中后按 id 回捞。这样向量库轻、备份快、改正文不用重建索引。这是本书生产架构的标准做法。

---

### R15 · Embedding 批量请求 `413 Request Entity Too Large` / 批量编码时 OOM

**出现场景**：一次性把 5000 条 chunk 丢给 embedding 服务或本地模型。

**根因**：HTTP body 超过服务端限制；或者本地 GPU 编码时 batch × seq_len 太大爆显存。

**解决**：分批 + 限流 + 重试。

```python
import asyncio
from typing import Iterable

async def embed_in_batches(texts: list[str], batch_size: int = 32, concurrency: int = 4) -> list[list[float]]:
    """分批并发编码，带并发上限，返回顺序与输入一致"""
    sem = asyncio.Semaphore(concurrency)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    async def run(idx: int, batch: list[str]):
        async with sem:
            return idx, await embed_api(batch)      # 你的 embedding 调用

    results = await asyncio.gather(*[run(i, b) for i, b in enumerate(batches)])
    ordered: list[list[float]] = []
    for _, vecs in sorted(results, key=lambda x: x[0]):
        ordered.extend(vecs)
    return ordered
```

本地模型侧直接用 `batch_size` 参数并按显存调：

```python
vecs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
```

**预防**：灌库脚本要能**断点续传**——每处理完一批就把进度写进本地状态文件，挂了重跑从断点继续。28 万条工单灌到 90% 挂掉再从头来一遍的痛苦，经历一次就够了。

---

## 5. LangChain / LangGraph

这一组的坑高度集中在两个来源：**版本演进**（LangChain 0.1 → 0.2 → 0.3 拆包重构）和 **抽象泄漏**（LCEL 的类型约束、异步传染、状态合并语义）。

本书基线：`langchain 0.3.x` + `langgraph 0.2.x`。

```bash
# 本组起手式：确认所有 langchain 相关包的版本，版本不一致是最常见的病根
uv pip list | grep -i "lang\|openai\|pydantic"
```

### L01 · `ImportError: cannot import name 'LLMChain' from 'langchain'`

**出现场景**：跑网上抄来的 2023 年的教程代码。

**根因**：LangChain 0.1 之后做了包拆分：核心抽象在 `langchain-core`，社区集成在 `langchain-community`，各家模型在 `langchain-openai` / `langchain-deepseek` 等独立包里。`LLMChain` 这类老 API 已被 LCEL 取代。

**解决**：按新写法重写，而不是找老版本。

```python
# 老写法（不要再用）
# from langchain import LLMChain, PromptTemplate
# chain = LLMChain(llm=llm, prompt=prompt)
# chain.run(question="...")

# 新写法：LCEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    temperature=0.1,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是华成机电的售后知识助手，只依据提供的资料回答。"),
    ("human", "资料：\n{context}\n\n问题：{question}"),
])
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"context": "...", "question": "XJ-200 报 E041 怎么办？"}))
```

**预防**：看教程先看日期和版本号。判断一段 LangChain 代码是不是过时，最快的标志：**有没有用 `|` 管道符**。没有的基本都是 0.1 时代的写法。

---

### L02 · `ModuleNotFoundError: No module named 'langchain_community'`

**出现场景**：用 `BM25Retriever`、各种 loader、各种向量库集成时。

**根因**：这些集成都被移到了 `langchain-community`，装 `langchain` 不会自动带上。

**解决**：按需装，装哪个包看 import 路径的第一段。

```bash
uv pip install langchain langchain-core langchain-community langchain-openai langchain-text-splitters
uv pip install langchain-milvus langchain-elasticsearch   # 本书用到的向量/全文后端
uv pip install langgraph langgraph-checkpoint-sqlite
```

对照表：

| import 路径 | 需要装的包 |
|---|---|
| `langchain_core.*` | `langchain-core` |
| `langchain.chains.*` / `langchain.agents.*` | `langchain` |
| `langchain_community.*` | `langchain-community` |
| `langchain_openai.*` | `langchain-openai` |
| `langchain_text_splitters.*` | `langchain-text-splitters` |
| `langgraph.*` | `langgraph` |

**预防**：把依赖固化进 `pyproject.toml` 并用 lockfile。**不要**在 notebook 里临时 `!pip install`，那样的环境无法复现。

---

### L03 · `TypeError: Expected a Runnable, callable or dict. Instead got an unsupported type: <class 'str'>`

**出现场景**：拼 LCEL 管道时。

**根因**：管道里混进了非 Runnable 的东西。最常见是把一个普通函数、一个字符串或一个已经 `invoke` 过的结果放进了 `|` 链条。

**解决**：用 `RunnableLambda` 包裹普通函数，用 `RunnablePassthrough` 传递原始输入。

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

def format_docs(docs) -> str:
    """把检索到的文档拼成带编号的上下文串，供引用使用"""
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

chain = (
    {
        "context": retriever | RunnableLambda(format_docs),   # 普通函数必须包一层
        "question": RunnablePassthrough(),                    # 原样透传输入
    }
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke("XJ-200 报 E041 怎么办？")
```

调试技巧——**逐段 invoke，看每一段的输入输出类型**：

```python
step1 = {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
out1 = step1.invoke("XJ-200 报 E041 怎么办？")
print(type(out1), out1.keys())      # 应该是 dict，键是 context/question

out2 = prompt.invoke(out1)
print(type(out2))                    # ChatPromptValue
print(out2.to_string()[:500])        # 把最终 prompt 打出来看
```

**预防**：给链条的每一段加类型注解，用 `chain.get_input_schema()` / `chain.get_output_schema()` 打印契约。链条长了就拆成命名的子链，分段测试。

---

### L04 · `InvalidUpdateError: Expected dict, got <class 'str'>`（LangGraph 节点返回值）

**出现场景**：写 LangGraph 节点函数。

**根因**：LangGraph 的节点必须返回一个 **state 的部分更新字典**，不能返回裸值。

**解决**：

```python
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    question: str
    documents: Annotated[list, operator.add]     # add reducer：多次写入会累加
    answer: str
    retry_count: int

# 错误写法
# def retrieve(state: AgentState):
#     return retriever.invoke(state["question"])       # 返回了 list，不是 dict

# 正确写法：返回要更新的字段
def retrieve(state: AgentState) -> dict:
    """检索节点：按 question 检索，把结果写入 documents"""
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def generate(state: AgentState) -> dict:
    """生成节点：基于 documents 产出 answer"""
    ctx = "\n\n".join(d.page_content for d in state["documents"])
    return {"answer": chain.invoke({"context": ctx, "question": state["question"]})}
```

**预防**：节点函数一律写成 `def node(state: XxxState) -> dict:`，返回类型注解写死 `dict`，配合 mypy 能直接在 CI 里拦住。

---

### L05 · `RuntimeError: asyncio.run() cannot be called from a running event loop` / `This event loop is already running`

**出现场景**：在 Jupyter 里调 `asyncio.run()`；或者在 FastAPI 的 `async def` 接口里调了同步版的 `chain.invoke()` 而链内部又是异步的。

**根因**：事件循环嵌套。Jupyter 本身跑在事件循环里，`asyncio.run()` 会再开一个。

**解决**：

```python
# Jupyter 里：直接 await，不要 asyncio.run
result = await chain.ainvoke({"question": "..."})

# 实在需要在同步上下文里跑异步代码（脚本、测试）
import nest_asyncio
nest_asyncio.apply()        # 只在 notebook/脚本用，生产代码别加
```

FastAPI 里的正确姿势——**async 接口全程用 a 系列方法**：

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/ask")
async def ask(q: str):
    # 全链路异步：ainvoke / astream / abatch
    return {"answer": await chain.ainvoke({"question": q})}

@app.post("/ask-sync-lib")
async def ask_sync_lib(q: str):
    # 确实要调同步阻塞函数（比如某个只有同步 API 的库），扔到线程池
    import anyio
    return {"answer": await anyio.to_thread.run_sync(blocking_call, q)}
```

**预防**：**一个项目选定一种范式**。本书服务端全走异步：FastAPI 接口 `async def`，链路用 `ainvoke`/`astream`，阻塞库用 `anyio.to_thread` 隔离。混着写迟早出事——最典型的症状是"压测时 QPS 上不去且 CPU 只跑满一个核"，那是同步阻塞卡死了事件循环。

---

### L06 · `GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition`

**出现场景**：Agent 图跑起来后在几个节点之间来回循环。

**根因**：两种。① 确实需要更多步（复杂任务）；② **逻辑 bug 导致死循环**——条件边的判断条件永远不满足退出条件。95% 的情况是第二种，**不要一上来就调大 limit**。

**解决**：先看它到底在循环什么。

```python
# 1. 打印执行轨迹，看是哪两个节点在死循环
for chunk in graph.stream({"question": "..."}, stream_mode="updates"):
    print(list(chunk.keys()), "->", {k: str(v)[:80] for k, v in chunk.items()})
```

```python
# 2. 给状态加硬性计数器，从代码层面保证一定能退出
def should_retry(state: AgentState) -> str:
    """条件边：决定是重试检索还是直接生成"""
    if state.get("retry_count", 0) >= 3:
        return "generate"          # 到上限强制走生成，不再重试
    if not state["documents"]:
        return "rewrite_query"
    return "generate"

def rewrite_query(state: AgentState) -> dict:
    """查询改写节点，同时递增重试计数"""
    new_q = rewriter.invoke(state["question"])
    return {"question": new_q, "retry_count": state.get("retry_count", 0) + 1}
```

```python
# 3. 确认逻辑没问题后再调限额
result = graph.invoke({"question": "..."}, config={"recursion_limit": 50})
```

**预防**：**每个可能形成环的路径上都必须有一个单调递增的计数器和一个硬上限**。这是本书 Agent 章反复强调的护栏第一条，见 [06-Agent智能体/05-手写一个生产级Agent](../06-Agent智能体/05-手写一个生产级Agent.md)。

---

### L07 · `ValueError: Checkpointer requires one or more of the following 'configurable' keys: thread_id`

**出现场景**：给图配了 checkpointer 之后调用 `invoke`。

**根因**：checkpointer 需要知道"这次调用属于哪个会话"才能存取状态，`thread_id` 就是会话标识。

**解决**：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("data/checkpoints.sqlite") as memory:
    graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "ticket-20250917-001"}}
    result = graph.invoke({"question": "XJ-200 报 E041 怎么办？"}, config=config)

    # 同一个 thread_id 下继续对话，自动带上历史状态
    result2 = graph.invoke({"question": "那 E043 呢？"}, config=config)

    # 查看某个会话的状态快照与历史
    print(graph.get_state(config).values)
    for snap in graph.get_state_history(config):
        print(snap.config["configurable"]["checkpoint_id"], snap.next)
```

生产环境用 Postgres 而不是 SQLite：

```python
from langgraph.checkpoint.postgres import PostgresSaver
# 注意端口：宿主机访问用 5433（本书约定），容器内互访用 5432
DB_URI = "postgresql://langfuse:langfuse@127.0.0.1:5433/agent_state?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as memory:
    memory.setup()          # 首次运行建表
    graph = builder.compile(checkpointer=memory)
```

**预防**：把 `thread_id` 的生成规则定死（本书用 `工单号` 或 `会话 UUID`），封装成一个 `build_config(session_id)` 函数，全项目只有一处构造 config。

---

### L08 · `OutputParserException: Failed to parse ... Got: Expecting value: line 1 column 1 (char 0)`

**出现场景**：要求模型输出 JSON，然后用 parser 解析。

**根因**：模型输出被 markdown 代码块包裹（` ```json ... ``` `）、或者前后带了解释性文字、或者中文引号混进来了。小模型和推理模型（会先输出思考过程）尤其容易这样。

**解决**：三层防御。

```python
# 第一层：优先用原生结构化输出（模型支持时最可靠）
from pydantic import BaseModel, Field
from typing import Literal

class TicketExtraction(BaseModel):
    """从工单描述中抽取的结构化信息"""
    model_no: str = Field(description="设备型号，如 XJ-200")
    error_code: str | None = Field(default=None, description="故障码，如 E041")
    severity: Literal["低", "中", "高"] = Field(description="严重程度")

structured_llm = llm.with_structured_output(TicketExtraction)
res = structured_llm.invoke("XJ-200-B3 主轴报 E041，产线已停")
```

```python
# 第二层：不支持原生结构化输出时，用 robust parser 兜底
import json, re

def parse_json_robust(text: str) -> dict:
    """从可能带 markdown 包裹或前后缀文字的文本中提取第一个合法 JSON 对象"""
    text = text.strip()
    # 去掉 markdown 代码围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 直接试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 括号配对扫描，提取第一个完整对象
    start = text.find("{")
    if start == -1:
        raise ValueError(f"文本中未找到 JSON 对象: {text[:200]}")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"JSON 括号不闭合: {text[:200]}")
```

```python
# 第三层：解析失败时带着错误信息让模型重试一次
from langchain_core.runnables import RunnableLambda

def with_retry_on_parse_error(llm, prompt, parser, max_retry: int = 1):
    """解析失败时把错误回灌给模型让它自我修正"""
    def _run(inputs: dict):
        msg = prompt.invoke(inputs)
        for attempt in range(max_retry + 1):
            raw = llm.invoke(msg).content
            try:
                return parser(raw)
            except Exception as e:
                if attempt == max_retry:
                    raise
                msg = prompt.invoke({**inputs, "error_hint": f"上次输出解析失败：{e}，请只输出合法 JSON。"})
    return RunnableLambda(_run)
```

**预防**：prompt 里明确写 "只输出 JSON，不要加任何解释文字，不要用 markdown 代码块"，并把 `temperature` 调到 0~0.1。同时**永远保留原始输出的日志**，解析失败时你需要看到模型到底说了什么。详见 [01-大模型基础与技术选型/05-提示工程与结构化输出](../01-大模型基础与技术选型/05-提示工程与结构化输出.md)。

---

### L09 · `BadRequestError: Invalid schema for function 'query_kb': ...` / 工具一直不被调用

**出现场景**：给模型绑定工具做 function calling。

**根因**：生成的 JSON Schema 不合法。常见于：用了模型侧不支持的 JSON Schema 特性（如 `allOf`、复杂 `$ref`、`Optional` 嵌套联合类型）；参数没写 description；函数名带了非法字符（只允许字母数字下划线连字符）。

**解决**：工具定义保持**扁平、简单、每个字段都有描述**。

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class QueryKBInput(BaseModel):
    """知识库检索工具的入参"""
    query: str = Field(description="检索关键词，使用用户原话中的设备型号和故障码")
    model_no: str | None = Field(default=None, description="设备型号过滤，如 XJ-200，不确定就留空")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数，默认 5")

@tool("query_kb", args_schema=QueryKBInput)
def query_kb(query: str, model_no: str | None = None, top_k: int = 5) -> str:
    """检索华成机电售后知识库，返回与故障现象相关的手册片段和历史工单。
    当用户询问设备故障、维修步骤、备件信息时使用本工具。"""
    ...

# 关键：把生成的 schema 打出来看，这是排查的唯一有效手段
import json
print(json.dumps(query_kb.args_schema.model_json_schema(), ensure_ascii=False, indent=2))
```

**工具不被调用**（不报错但模型就是不用）通常不是 schema 问题，而是**描述写得不好**：

| 写法 | 效果 |
|---|---|
| `"""查询知识库"""` | 差。模型不知道什么时候该用 |
| `"""检索华成机电售后知识库……当用户询问设备故障、维修步骤、备件信息时使用本工具。"""` | 好。说清了「做什么」和「何时用」 |

**预防**：工具的 docstring 必须包含三段：**功能**、**何时使用**、**何时不要使用**。本书 [06-Agent智能体/02-Function-Calling与工具设计](../06-Agent智能体/02-Function-Calling与工具设计.md) 有 Bad/Good 对照的完整案例。

---

### L10 · `astream_events` / `astream` 拿不到流式 token

**出现场景**：前端接了 SSE，但一直等到最后才一次性收到全部内容。

**根因**：四种可能。① 模型对象没开 `streaming=True`（部分集成需要）；② 链条里有**非流式的阻塞环节**（比如中间加了个 `RunnableLambda` 做全量后处理，它必须等上游结束）；③ `astream_events` 的 `version` 参数不对；④ FastAPI 侧没用 `StreamingResponse`，或者中间有 Nginx 在缓冲。

**解决**：

```python
# 1. 先确认模型本身能流
async for chunk in llm.astream("你好"):
    print(chunk.content, end="", flush=True)

# 2. 用 astream_events 定位是哪一段不流（注意 version）
async for ev in chain.astream_events({"question": "..."}, version="v2"):
    kind = ev["event"]
    if kind == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="", flush=True)
    elif kind in ("on_retriever_end", "on_chain_start"):
        print(f"\n[{kind}] {ev['name']}")
```

```python
# 3. FastAPI 侧正确的 SSE 写法
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@app.post("/stream")
async def stream(q: str):
    async def gen():
        async for ev in chain.astream_events({"question": q}, version="v2"):
            if ev["event"] == "on_chat_model_stream":
                tok = ev["data"]["chunk"].content
                if tok:
                    yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},   # 关掉 Nginx 缓冲
    )
```

**预防**：链条设计时就把"必须等全量"的环节（如全文改写、JSON 解析）放在**流式输出之前**，别放在中间。需要边流边处理的场景（如流式引用标注），要用增量算法而不是全量重算。

---

### L11 · `ValidationError: 1 validation error for ... value is not a valid dict` / pydantic v1 与 v2 混用

**出现场景**：LangChain 0.3 之后全面用 pydantic v2，但你的项目里还有用 `pydantic.v1` 的老代码。

**根因**：`BaseModel` 来自不同命名空间时互不认识。

**解决**：

```python
# 确认版本
import pydantic
print(pydantic.VERSION)      # 应为 2.x

# 统一从 pydantic 顶层导入，不要用 pydantic.v1
from pydantic import BaseModel, Field, field_validator     # v2 写法

# v1 -> v2 常见改名对照
# @validator            -> @field_validator
# class Config          -> model_config = ConfigDict(...)
# .dict()               -> .model_dump()
# .json()               -> .model_dump_json()
# .parse_obj()          -> .model_validate()
# Field(..., regex=)    -> Field(..., pattern=)
```

**预防**：项目根的 `pyproject.toml` 里钉住 `pydantic>=2.7,<3`，并在 CI 里跑一次 `grep -rn "pydantic.v1" .` 作为门禁。

---

### L12 · `InvalidUpdateError: At key 'documents': Can receive only one value per step. Use an Annotated key to handle multiple values`

**出现场景**：LangGraph 里两个并行节点同时写同一个 state 字段。

**根因**：LangGraph 默认的更新语义是"覆盖"，同一步内两个节点都写同一个 key 就无法决定用谁的。需要显式声明 **reducer**。

**解决**：

```python
from typing import TypedDict, Annotated
import operator

def merge_unique(left: list, right: list) -> list:
    """合并两路检索结果并按 doc_id 去重，保持原有顺序"""
    seen, out = set(), []
    for d in list(left) + list(right):
        key = d.metadata.get("doc_id") if hasattr(d, "metadata") else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

class RAGState(TypedDict):
    question: str
    documents: Annotated[list, merge_unique]     # 自定义 reducer：并行写入自动合并去重
    logs: Annotated[list[str], operator.add]     # 内置 reducer：追加
    answer: str                                   # 无 reducer：覆盖语义，不能并行写
```

**预防**：设计 state 时就问一句"这个字段会不会被多个节点同时写"。混合检索（向量 + BM25 并行）是最典型的场景，本书 [07-多智能体协同/02-LangGraph实现Supervisor与Swarm](../07-多智能体协同/02-LangGraph实现Supervisor与Swarm.md) 里的并行模式详细讲了 reducer 的设计。

---

### L13 · Langfuse 里看不到 trace / trace 断成好几截

**出现场景**：接了 Langfuse，但链路图不完整，或者根本没数据。

**根因**：① callback handler 没传下去（子链、工具调用里没带）；② 用了 `@traceable` 但在异步上下文里没正确传播；③ 进程退出太快，缓冲区还没 flush；④ `LANGFUSE_HOST` 配错（注意端口 3001）。

**解决**：

```python
import os
from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ.get("LANGFUSE_HOST", "http://localhost:3001"),
)

# 关键：通过 config 传，它会自动向下传播到整条链和所有子链
result = chain.invoke(
    {"question": "..."},
    config={
        "callbacks": [handler],
        "metadata": {"user_id": "engineer-042", "ticket_id": "T-20250917-001"},
        "run_name": "rag_answer",
    },
)

# 脚本/短任务结束前必须 flush，否则数据还在缓冲区里进程就没了
handler.flush()
```

```bash
# 检查连通性
curl -sf "${LANGFUSE_HOST}/api/public/health" && echo "Langfuse OK"
```

**预防**：把 handler 的构造和 config 组装封装进 `core/instrument.py`（本书已有该模块），业务代码只调 `traced_invoke(chain, inputs, **ctx)`，杜绝有人忘了传 callbacks。

---

### L14 · `NotImplementedError` / 自定义 Runnable 只能同步调不能异步调

**出现场景**：自己继承 `Runnable` 写了个组件，同步 `invoke` 正常，`ainvoke` 报未实现。

**根因**：`Runnable` 基类的 `ainvoke` 默认会把 `invoke` 扔线程池执行，但如果你重写了某些方法而没实现对应的异步版本，就会缺。自定义 Retriever 尤其常见（实现了 `_get_relevant_documents` 但没实现 `_aget_relevant_documents`）。

**解决**：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun

class HybridRetriever(BaseRetriever):
    """向量 + BM25 混合检索，同步异步双实现"""
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        dense = self._dense_search(query)
        sparse = self._bm25_search(query)
        return self._rrf_fuse(dense, sparse)[: self.top_k]

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> list[Document]:
        import asyncio
        dense, sparse = await asyncio.gather(self._adense_search(query), self._abm25_search(query))
        return self._rrf_fuse(dense, sparse)[: self.top_k]
```

**预防**：自定义组件写完，**同步和异步各跑一遍单测**。CI 里对每个自定义 Runnable 都跑 `invoke` / `ainvoke` / `batch` / `stream` 四件套的冒烟测试。

---
## 6. 微调

微调的坑分两类：**会报错的**（好办，照着改）和**不报错但把模型练废的**（难办，要靠评测发现）。
本组特别留意后者——F04、F09、F13 都属于"程序跑完了，loss 曲线也好看，模型却学歪了"。

**微调排错的黄金法则：先用 20 条样本过拟合一遍。**
20 条数据、50 步、学习率调大 10 倍，如果 loss 降不到接近 0，说明流水线本身有 bug（labels 错、模板错、梯度没通），
这时候扩大数据集只是在浪费 GPU 时间。

### F01 · loss 变成 `nan`

**出现场景**：训练跑了几十步之后 loss 突然变 nan，之后再也回不来。

**根因**：数值溢出。诱因排序：① FP16 + 大学习率（最常见）；② 数据里有异常样本（超长、空文本、全是特殊 token）；③ 没做梯度裁剪；④ Adam 的 `eps` 太小配 FP16。

**解决**：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output/lora-qwen25-7b",
    bf16=True,                    # 优先 BF16，它的数值范围和 FP32 一样宽，几乎不会溢出
    fp16=False,
    learning_rate=1e-4,           # LoRA 常用 1e-4 ~ 2e-4；全参微调要降到 1e-5 量级
    warmup_ratio=0.03,            # 一定要有 warmup
    max_grad_norm=1.0,            # 梯度裁剪
    lr_scheduler_type="cosine",
    logging_steps=10,
)
```

```python
# 定位是哪条样本引发的：加一个回调，nan 出现时把当前 batch 打出来
from transformers import TrainerCallback

class NanDetector(TrainerCallback):
    """检测到 loss 为 nan 时立刻打印上下文并中止"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            import math
            if math.isnan(logs["loss"]) or math.isinf(logs["loss"]):
                print(f"[NaN] step={state.global_step} logs={logs}")
                control.should_training_stop = True
```

```python
# 数据侧体检：把异常样本挑出来
bad = [i for i, ex in enumerate(dataset)
       if not ex["output"].strip() or len(ex["input"]) > 20000]
print(f"异常样本 {len(bad)} 条: {bad[:20]}")
```

**预防**：训练脚本的第一步永远是**数据体检**（空值、超长、重复、标签缺失），第二步是**20 条过拟合冒烟**。这两步加起来 10 分钟，能省下无数次半夜起来看 nan 的经历。

---

### F02 · loss 一直不降，或者降到某个值就平了

**出现场景**：训练正常跑完，但 loss 从 2.1 降到 1.9 就不动了。

**根因**：候选很多，按可能性排序：① 学习率太小；② LoRA rank 太小 / `target_modules` 只挂了一小部分层；③ 数据量太少或质量太差（任务模式学不出来）；④ 数据里的模式本来就是模型已经会的（loss 降不下去是因为已经接近下界）；⑤ 梯度没有真的在更新（见 F10）。

**解决**：按顺序排查。

```python
# 1. 确认可训练参数量。LoRA 正常占比在 0.1% ~ 1% 之间，太小就是挂得不够
model.print_trainable_parameters()
# 期望输出类似: trainable params: 20,185,088 || all params: 7,635,801,600 || trainable%: 0.2643
```

```python
from peft import LoraConfig

# 2. 挂全所有线性层，而不是只挂 q_proj/v_proj
lora_config = LoraConfig(
    r=32,                       # 效果不好优先提 r（16 -> 32 -> 64）
    lora_alpha=64,              # 惯例 alpha = 2 * r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

```bash
# 3. 学习率扫描：跑 4 组各 100 步，看哪条 loss 曲线下降最快
for lr in 5e-5 1e-4 2e-4 5e-4; do
  python finetune/train.py --learning_rate $lr --max_steps 100 --output_dir ./output/lr_$lr
done
```

**预防**：loss 绝对值没有意义，**要看的是相对下降幅度和验证集指标**。SFT 的经验区间：起始 loss 在 1.5~2.5，训练 2~3 个 epoch 后降到 0.5~1.0 属于正常；降到 0.1 以下基本是过拟合了。但最终判断标准永远是**下游任务评测**，不是 loss。见 [05-微调LoRA与PEFT/06-微调效果评估与何时不该微调](../05-微调LoRA与PEFT/06-微调效果评估与何时不该微调.md)。

---

### F03 · 训练时 `CUDA out of memory`（显存不够训 7B）

**出现场景**：24G 卡上想 LoRA 微调 Qwen2.5-7B。

**根因**：训练显存 = 权重 + 梯度 + 优化器状态 + 激活值。LoRA 已经把梯度和优化器状态砍掉 99%，剩下的大头是**权重**和**激活值**。

**解决**：按"省显存收益 / 速度损失"排序逐条上。

| 手段 | 省多少 | 慢多少 | 配置 |
|---|---|---|---|
| BF16 权重 | 相对 FP32 省一半 | 不慢 | `bf16=True` |
| 4bit 量化（QLoRA） | 权重再省约 3/4 | 慢约 30% | `BitsAndBytesConfig(load_in_4bit=True)` |
| 梯度累积 | 按比例省激活 | 不慢（等效 batch 不变） | `gradient_accumulation_steps=16` |
| 梯度检查点 | 省约 60% 激活 | 慢约 30% | `gradient_checkpointing=True` |
| 8bit 优化器 | 省优化器状态 | 几乎不慢 | `optim="paged_adamw_8bit"` |
| 缩短 `max_seq_length` | 线性相关 | 不慢 | 按数据 p95 长度定 |

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "/data/models/Qwen2.5-7B-Instruct", quantization_config=bnb,
    device_map={"": 0}, torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)
model.config.use_cache = False          # 训练时必须关，否则和梯度检查点冲突且白占显存

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    bf16=True,
    max_steps=1000,
    output_dir="./output/qlora-qwen25-7b",
)
```

**预防**：训练前先跑 `max_steps=5` 看显存峰值，确认留有 15% 余量再跑全量。别等跑了 6 小时在第 3 个 epoch OOM。

---

### F04 · loss 一开始就是 0，或者训完模型什么也没学会（labels 全是 -100）

**出现场景**：自己写 collator 或数据处理逻辑。

**根因**：`-100` 是 PyTorch 交叉熵的 `ignore_index`，被标成 -100 的位置不参与 loss 计算。SFT 里我们**只对 assistant 回复部分算 loss**，prompt 部分要 mask 成 -100。如果 mask 逻辑写错，把全部位置都 mask 了，loss 恒为 0 且模型学不到任何东西。

**解决**：写一个诊断函数，**训练前必看**。

```python
def inspect_labels(example: dict, tokenizer) -> None:
    """打印一条样本的 input_ids / labels 对照，确认 mask 范围正确"""
    ids, labels = example["input_ids"], example["labels"]
    n_valid = sum(1 for l in labels if l != -100)
    print(f"总长度={len(ids)}  参与 loss 的 token 数={n_valid}  占比={n_valid/len(ids):.1%}")
    assert n_valid > 0, "labels 全是 -100，loss 无法计算"

    print("\n--- 被 mask 的部分（不算 loss，应为 system+user）---")
    print(tokenizer.decode([i for i, l in zip(ids, labels) if l == -100]))
    print("\n--- 参与 loss 的部分（应为 assistant 回复）---")
    print(tokenizer.decode([i for i, l in zip(ids, labels) if l != -100]))

inspect_labels(train_dataset[0], tokenizer)
```

正确的 mask 构造（ChatML 风格）：

```python
def build_sft_example(sample: dict, tokenizer, max_len: int = 2048) -> dict:
    """构造 SFT 样本：prompt 部分 mask 为 -100，只对 assistant 回复算 loss"""
    messages = [
        {"role": "system", "content": "你是华成机电的售后知识助手。"},
        {"role": "user", "content": sample["instruction"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    answer_ids = tokenizer(sample["output"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}
```

**预防**：把 `inspect_labels` 作为训练脚本的**强制第一步**，打印 3 条样本，人工确认后才开始训。这一步能挡住微调里最昂贵的错误。

---

### F05 · `KeyError: 'instruction'` / `ValueError: Unable to create tensor, you should probably activate truncation and/or padding`

**出现场景**：换了个数据集或换了个训练框架。

**根因**：数据集字段名与代码期望的不一致（alpaca 格式是 `instruction/input/output`，sharegpt 格式是 `conversations`，ChatML 是 `messages`）；或者样本长度不一致又没 padding。

**解决**：

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data/sft/huacheng_sft.jsonl", split="train")
print(ds.column_names)            # 先看真实字段名
print(ds[0])                       # 再看一条样本长什么样

# 字段名对不上就改名，不要改代码
ds = ds.rename_columns({"question": "instruction", "answer": "output"})
```

三种常见 SFT 数据格式对照：

```json
// alpaca
{"instruction": "XJ-200 报 E041 怎么处理？", "input": "", "output": "E041 表示主轴过载……"}

// sharegpt
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

// openai / chatml
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

padding 用官方 collator，别自己拼：

```python
from transformers import DataCollatorForSeq2Seq

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt",
)
```

**预防**：数据生产流水线的最后一步输出**统一格式**（本书统一用 ChatML 的 `messages` 格式），所有训练脚本只认这一种。格式转换在数据侧做一次，不要在每个训练脚本里各写一遍。见 [05-微调LoRA与PEFT/02-数据集构造与清洗](../05-微调LoRA与PEFT/02-数据集构造与清洗.md)。

---

### F06 · `ValueError: Can't find 'adapter_config.json' at '/path/to/adapter'`

**出现场景**：加载训练好的 LoRA adapter。

**根因**：路径指错了（指到了 `output_dir` 而不是具体的 `checkpoint-xxx`）；或者训练时用的是 `model.save_pretrained()` 但保存的是 base model 的包装对象而不是 peft model。

**解决**：

```bash
# 看 adapter 目录里应该有什么
ls -lh ./output/qlora-qwen25-7b/checkpoint-1000/
# 期望: adapter_config.json  adapter_model.safetensors  README.md
#       （可能还有 tokenizer 相关文件）
```

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base = AutoModelForCausalLM.from_pretrained(
    "/data/models/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",
)
model = PeftModel.from_pretrained(base, "./output/qlora-qwen25-7b/checkpoint-1000")
model.eval()

# 确认 adapter 真的挂上了
print(model.peft_config)
print([n for n, _ in model.named_parameters() if "lora" in n][:5])
```

保存时确保存的是 peft model：

```python
trainer.model.save_pretrained("./output/final-adapter")     # peft model 的 save
tokenizer.save_pretrained("./output/final-adapter")          # tokenizer 一起存，避免后面对不上
```

**预防**：adapter 目录里同时存一份 `meta.json`，记录 base 模型路径、base 模型的 sha256、训练数据版本、训练参数。加载时校验 base 是否匹配——**用错 base 加载 adapter 不会报错，只会让模型变傻**。

---

### F07 · 合并 adapter 后模型输出重复、乱码或者完全变了一个模型

**出现场景**：`merge_and_unload()` 导出模型后推理。

**根因**：① 合并时 base 用了量化版本（4bit 量化的权重合并回去会有明显精度损失）；② dtype 不一致（adapter 是 fp32，base 是 bf16）；③ 推理时用的 chat template 和训练时不一样；④ 合并的 base 和训练时的 base 不是同一个。

**解决**：合并必须用**全精度的原始 base**，不能用量化版。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/data/models/Qwen2.5-7B-Instruct"
ADAPTER = "./output/qlora-qwen25-7b/checkpoint-1000"
MERGED = "/data/models/Qwen2.5-7B-huacheng-v1"

# 关键：不要加 quantization_config，用 bf16 全精度加载
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER, torch_dtype=torch.bfloat16)
merged = model.merge_and_unload()
merged = merged.to(torch.bfloat16)

merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)     # tokenizer 必须一起存
print("合并完成:", MERGED)
```

合并后**必须跑冒烟测试**：

```python
def smoke_test(model_path: str, n: int = 20) -> None:
    """合并后冒烟：跑 n 条样例，检查是否有重复、空输出、乱码"""
    import re
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    for q in SMOKE_QUESTIONS[:n]:
        msgs = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(m.device)
        out = tok.decode(m.generate(ids, max_new_tokens=256, do_sample=False)[0][ids.shape[-1]:],
                         skip_special_tokens=True)
        assert out.strip(), f"空输出: {q}"
        # 简单的复读检测：某个 8 字子串重复超过 5 次
        assert not re.search(r"(.{8,})\1{4,}", out), f"疑似复读: {q} -> {out[:200]}"
        print(f"[OK] {q[:30]} -> {out[:80]}")
```

**预防**：合并、量化、部署三步之间**每一步都跑同一套冒烟集**，比对输出。输出在哪一步变样，问题就在哪一步。

---

### F08 · DeepSpeed `AssertionError: Check batch related parameters. train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size`

**出现场景**：用 DeepSpeed ZeRO 做多卡训练。

**根因**：DeepSpeed 的 batch 参数有严格的等式约束：

$$
\text{train\_batch\_size} = \text{micro\_batch\_per\_gpu} \times \text{gradient\_accumulation\_steps} \times \text{world\_size}
$$

而 HF Trainer 也有自己的一套参数，两边冲突。

**解决**：让 DeepSpeed 配置里的相关值全写 `"auto"`，交给 Trainer 推导。

```json
{
  "bf16": { "enabled": "auto" },
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "allgather_partitions": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

```bash
deepspeed --num_gpus=2 finetune/train.py \
  --deepspeed configs/ds_zero2.json \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16 True
```

**ZeRO stage 选择**：stage 2（切分优化器状态和梯度）适合 LoRA 和中等规模；stage 3（连权重也切）适合全参微调大模型，但通信开销大、速度慢。**LoRA 场景通常 stage 2 就够，别无脑上 stage 3。**

**预防**：多卡配置先在 `--max_steps 10` 下验证一遍能跑通，再放全量。

---

### F09 · 训完模型答非所问 / 学会了一堆奇怪的格式（chat template 用错）

**出现场景**：微调后的模型在评测里比 base 还差。

**根因**：**训练时的对话模板和推理时的不一致**。比如训练时手工拼了 `### Instruction:\n...\n### Response:\n`（Alpaca 风格），推理时却用 `tokenizer.apply_chat_template`（ChatML 风格）。模型学到的是前者的模式，遇到后者就懵了。

这是微调里**最贵的一个坑**：它不报任何错，只是让你的几十小时 GPU 时间打水漂。

**解决**：训练和推理共用**同一个函数**构造输入。

```python
# finetune/template.py —— 全项目唯一的模板构造入口
SYSTEM_PROMPT = "你是华成机电的售后知识助手，依据提供的资料准确回答设备故障问题。"

def build_messages(question: str, context: str | None = None) -> list[dict]:
    """构造对话消息列表，训练与推理共用，保证模板绝对一致"""
    user = f"资料：\n{context}\n\n问题：{question}" if context else question
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]

def render_prompt(tokenizer, question: str, context: str | None = None) -> str:
    """渲染为最终 prompt 字符串，add_generation_prompt 只在推理时为 True"""
    return tokenizer.apply_chat_template(
        build_messages(question, context), tokenize=False, add_generation_prompt=True,
    )
```

```python
# 单元测试：把渲染结果钉死，模板一改测试就红
def test_template_stable(tokenizer):
    got = render_prompt(tokenizer, "XJ-200 报 E041 怎么办？")
    assert got.startswith("<|im_start|>system")
    assert got.endswith("<|im_start|>assistant\n")
```

**预防**：训练脚本启动时**打印第一条样本的完整渲染结果**到日志，推理服务启动时也打印一条。两边日志人工比对一次，之后写进测试。

---

### F10 · `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**出现场景**：开了梯度检查点或用了 QLoRA 之后。

**根因**：所有参数都被冻结了，没有任何一个 `requires_grad=True`；或者梯度检查点需要输入 embedding 参与梯度图但被切断了。

**解决**：

```python
from peft import prepare_model_for_kbit_training

# QLoRA 必须先 prepare，它会处理 norm 层精度、开启输入梯度等一系列细节
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, lora_config)

# 兜底：显式开启输入 embedding 的梯度
model.enable_input_require_grads()

# 验证：必须有非空输出
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"可训练参数 {len(trainable)} 个，例如: {trainable[:3]}")
assert trainable, "没有任何可训练参数，检查 LoRA 是否正确挂载"
```

**预防**：`model.print_trainable_parameters()` 的输出写进训练日志的第一行。这一行同时能挡住 F02（学习不动）和 F10（梯度断了）两个问题。

---

### F11 · `ValueError: Attempting to unscale FP16 gradients`

**出现场景**：用 `fp16=True` 训练 LoRA。

**根因**：LoRA 的适配器参数被加载成了 FP16，而 GradScaler 要求可训练参数是 FP32。

**解决**：

```python
# 方案 A（推荐）：改用 bf16，从根上避开 GradScaler
args = TrainingArguments(bf16=True, fp16=False, ...)

# 方案 B：老卡只能 fp16 时，把 LoRA 参数转回 fp32
for name, param in model.named_parameters():
    if "lora" in name:
        param.data = param.data.float()
```

**预防**：Ampere（算力 8.0）及以上的卡一律用 BF16。只有 V100/T4 这类老卡才需要 FP16，那时候要额外小心数值稳定性。

---

### F12 · `ValueError: Target modules ['q_proj', 'v_proj'] not found in the base model`

**出现场景**：把针对 Llama 写的 LoRA 配置直接用到别的架构上。

**根因**：不同模型架构的线性层命名不一样。

**解决**：先把模型的线性层名字打出来。

```python
import torch.nn as nn

def list_linear_modules(model) -> set[str]:
    """列出模型里所有 nn.Linear 的短名，用于确定 LoRA 的 target_modules"""
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(name.split(".")[-1])
    return names

print(sorted(list_linear_modules(model)))
# Qwen2/Llama 系: {'down_proj','gate_proj','k_proj','lm_head','o_proj','q_proj','up_proj','v_proj'}
```

或者干脆让 peft 自动挂全部线性层：

```python
lora_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules="all-linear",       # peft 0.13 支持，自动挂所有线性层（会跳过 lm_head）
)
```

**预防**：换模型时**永远先打印结构**，不要复制粘贴别人的配置。

---

### F13 · checkpoint 只有几 MB，以为训练失败了

**出现场景**：LoRA 训练完看 `checkpoint-1000` 目录只有 40MB。

**根因**：**这是正常的**。LoRA 只保存适配器权重（几十 MB），不保存 base 模型（十几 GB）。真正的问题是很多人因此误以为失败，去改配置反而搞坏了。

**解决**：确认适配器内容正常即可。

```python
from safetensors.torch import load_file

sd = load_file("./output/qlora-qwen25-7b/checkpoint-1000/adapter_model.safetensors")
print(f"参数张量数: {len(sd)}")
total = sum(v.numel() for v in sd.values())
print(f"参数总量: {total/1e6:.2f}M")
# r=32 挂 7 个模块的 Qwen2.5-7B，量级在 1000 万 ~ 1 亿之间属正常

# 检查权重不是全 0（全 0 说明压根没训）
import torch
nonzero = sum(1 for v in sd.values() if torch.any(v != 0))
print(f"非全零张量: {nonzero}/{len(sd)}")
```

**预防**：在部署文档里写清楚"adapter 需要配 base 一起用"，并在 adapter 的 `meta.json` 里记下 base 的路径和哈希。**单独把 adapter 拷给别人是没用的。**

---

### F14 · 训练日志显示样本数远少于数据集条数 / 长样本被静默丢弃

**出现场景**：数据集 8000 条，训练日志显示 `num_examples = 3200`。

**根因**：预处理时用 `filter` 丢掉了超长样本；或者用了 packing（把多条短样本拼成一条定长序列）导致条数变化；或者 `max_seq_length` 设太小，`truncation=True` 把长样本的答案部分整个截没了——**这种最恶劣，样本还在但标签没了**。

**解决**：

```python
def audit_lengths(dataset, tokenizer, max_len: int) -> None:
    """审计样本长度分布与截断影响，超长比例过高就报警"""
    lens = [len(tokenizer(ex["instruction"] + ex["output"])["input_ids"]) for ex in dataset]
    lens.sort()
    n = len(lens)
    over = sum(1 for l in lens if l > max_len)
    print(f"样本数={n} p50={lens[n//2]} p90={lens[int(n*0.9)]} p99={lens[int(n*0.99)]} max={lens[-1]}")
    print(f"超过 max_len={max_len} 的样本: {over} 条 ({over/n:.1%})")
    if over / n > 0.05:
        print("[警告] 超长样本超过 5%，建议提高 max_seq_length 或改用父子切分")
```

**预防**：`max_seq_length` 按数据的 **p95 长度**来定，而不是拍脑袋定 512 或 2048。同时在训练日志里打印"实际参与训练的样本数"和"被过滤/截断的样本数"，让损耗可见。

---

## 7. API 调用

本书主用 DeepSeek（`deepseek-chat` / `deepseek-reasoner`）与通义千问等 OpenAI 兼容端点。
这一组的坑大多不在代码逻辑，而在**网络、配额和契约**。

```python
# 本组的通用诊断：把完整的请求和响应打出来
import httpx, os, json

r = httpx.post(
    "https://api.deepseek.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
             "Content-Type": "application/json"},
    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "你好"}]},
    timeout=30,
)
print(r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:1000])
```

### A01 · `RateLimitError: Error code: 429 - Rate limit reached`

**出现场景**：批量跑评测、批量生成训练数据、或者线上并发突增。

**根因**：超过了服务方的 RPM（每分钟请求数）或 TPM（每分钟 token 数）限额。

**解决**：客户端限流 + 指数退避重试。

```python
import asyncio, random
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError

client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
SEM = asyncio.Semaphore(8)        # 并发上限：按你的配额调

async def chat_with_retry(messages: list[dict], max_retry: int = 5, **kw) -> str:
    """带指数退避与抖动的聊天补全，专门处理限流与瞬时网络错误"""
    for attempt in range(max_retry):
        try:
            async with SEM:
                resp = await client.chat.completions.create(
                    model="deepseek-chat", messages=messages, timeout=60, **kw,
                )
            return resp.choices[0].message.content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == max_retry - 1:
                raise
            # 指数退避 + 抖动，避免所有失败请求同时重试形成二次冲击
            delay = min(2 ** attempt, 32) + random.uniform(0, 1)
            print(f"[retry {attempt+1}/{max_retry}] {type(e).__name__}，{delay:.1f}s 后重试")
            await asyncio.sleep(delay)
```

**预防**：批量任务要有**全局速率控制**（令牌桶），并把并发度作为配置项而不是硬编码。评测这种"一次跑几千条"的场景，宁可跑慢点也不要触发限流——重试的总耗时通常比降并发更长。见 [08-评测体系/04-RAGAS与自动化评测流水线](../08-评测体系/04-RAGAS与自动化评测流水线.md)。

---

### A02 · `AuthenticationError: Error code: 401 - Incorrect API key provided`

**出现场景**：刚配好环境第一次调用。

**根因**：key 没读到（`.env` 没加载）、key 复制时带了空格或换行、key 用错了服务商、或者 key 已失效。

**解决**：

```python
import os
from dotenv import load_dotenv

load_dotenv()      # 确保 .env 被加载

key = os.environ.get("DEEPSEEK_API_KEY", "")
print(f"key 长度={len(key)} 前缀={key[:6]}... 有无空白字符={key != key.strip()}")
assert key, "DEEPSEEK_API_KEY 未设置"
assert key == key.strip(), "key 含首尾空白，检查 .env 里有没有多余空格或引号"
```

```bash
# 最小验证：绕过所有框架直接打接口
curl -s https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" | head -20
```

**预防**：`.env` 文件必须在 `.gitignore` 里；配置读取统一走 `core/config.py` 的 pydantic-settings（本书已有），它在字段缺失时会在**启动阶段**就报错，而不是等到第一次调用。

---

### A03 · `BadRequestError: This model's maximum context length is 65536 tokens, however you requested 71024 tokens`

**出现场景**：RAG 场景下检索到的上下文太长；或者多轮对话历史越攒越多。

**根因**：`prompt tokens + max_tokens > 模型上下文上限`。注意**输出预算也算在里面**。

**解决**：做上下文预算管理。

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """粗略估算 token 数。非 OpenAI 模型用兼容编码器估算，误差通常在 10% 内"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def fit_context(chunks: list[str], budget: int) -> list[str]:
    """按 token 预算从前往后装填上下文，装不下的丢弃（chunks 应已按相关性排序）"""
    used, kept = 0, []
    for c in chunks:
        t = count_tokens(c)
        if used + t > budget:
            break
        kept.append(c)
        used += t
    return kept

# 预算分配示例：总上限 65536
MAX_CTX = 65536
RESERVED_OUTPUT = 2048          # 留给模型输出
SYSTEM_TOKENS = count_tokens(SYSTEM_PROMPT)
HISTORY_BUDGET = 4096
DOC_BUDGET = MAX_CTX - RESERVED_OUTPUT - SYSTEM_TOKENS - HISTORY_BUDGET
```

中文 token 估算经验：**1 个中文字约 0.6~1 个 token**，英文 1 个 token 约 4 个字符。粗算时按「中文字数 × 0.8」即可。

**预防**：上下文预算要在**代码里显式分配**并在超限时主动裁剪，而不是等 API 报错。同时监控「实际 prompt token 数」的 P99，它是成本和延迟的直接驱动因素。

---

### A04 · `APITimeoutError` / `httpx.ReadTimeout`

**出现场景**：长文本生成、推理模型（`deepseek-reasoner` 会先输出思维链）、或者网络抖动。

**根因**：默认超时太短。生成 2000 token 的回答可能要 60 秒以上，推理模型更久。

**解决**：分段设置超时。

```python
import httpx
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    timeout=httpx.Timeout(
        connect=10.0,      # 建连超时：短，快速发现网络不通
        read=180.0,        # 读超时：长，容忍慢生成
        write=10.0,
        pool=5.0,
    ),
    max_retries=0,         # 自己管重试，不要让 SDK 偷偷重试搞乱统计
)
```

**更好的做法是用流式**——流式下每个 token 都会刷新读超时，长生成不会整体超时，而且用户能更早看到内容：

```python
async def stream_answer(messages: list[dict]):
    """流式生成，规避长响应超时，同时改善首字延迟体验"""
    stream = await client.chat.completions.create(
        model="deepseek-chat", messages=messages, stream=True, timeout=180,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

**预防**：给用户可见的接口一律走流式。后台批量任务可以用非流式，但超时要按 `max_tokens / 预期生成速度` 算，留 3 倍余量。

---

### A05 · `SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate`

**出现场景**：公司网络有 SSL 中间人代理（安全审计设备）。

**根因**：公司代理用自签 CA 重新签发证书，Python 的 certifi 证书库里没有这个 CA。

**解决**：把公司 CA 加进信任链（**不要关校验**）。

```bash
# 1. 拿到公司 CA 证书（找 IT 要，或从浏览器导出）
# 2. 指给 Python 用
export SSL_CERT_FILE=/etc/ssl/certs/company-ca.pem
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/company-ca.pem

# 3. 或者合并进 certifi 的证书库
python -c "import certifi; print(certifi.where())"
cat company-ca.pem | sudo tee -a $(python -c "import certifi; print(certifi.where())")
```

```python
# 仅限内网测试环境的临时办法，生产严禁
# import httpx
# client = AsyncOpenAI(http_client=httpx.AsyncClient(verify=False))
```

**预防**：把公司 CA 打进基础 Docker 镜像，所有服务从这个镜像派生，一次配置全员受益。**永远不要在代码里写 `verify=False`**——它一旦进了仓库就会被复制到生产。

---

### A06 · `httpx.RemoteProtocolError: peer closed connection without sending complete message body`（流式中断）

**出现场景**：流式输出到一半突然断掉。

**根因**：网络抖动、服务端主动断连、中间有代理（Nginx/网关）设了响应超时或缓冲区限制。

**解决**：流式要设计成**可恢复**的。

```python
async def robust_stream(messages: list[dict], max_retry: int = 2):
    """流式生成，中断后带着已生成内容续写，对用户表现为一次连续输出"""
    accumulated = ""
    for attempt in range(max_retry + 1):
        try:
            msgs = messages if not accumulated else messages + [
                {"role": "assistant", "content": accumulated},
            ]
            stream = await client.chat.completions.create(
                model="deepseek-chat", messages=msgs, stream=True, timeout=180,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    accumulated += delta
                    yield delta
            return
        except (httpx.RemoteProtocolError, httpx.ReadError) as e:
            if attempt == max_retry:
                yield f"\n\n[生成中断: {type(e).__name__}]"
                return
            await asyncio.sleep(1)
```

Nginx 侧关掉缓冲：

```nginx
location /stream {
    proxy_pass http://app:8080;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    chunked_transfer_encoding on;
    add_header X-Accel-Buffering no;
}
```

**预防**：前端也要处理断流——SSE 的 `onerror` 里要有重连逻辑，并给用户明确提示，而不是界面就那么卡着。

---

### A07 · function calling 返回的 `arguments` 不是合法 JSON

**出现场景**：Agent 调工具时。

**根因**：模型生成的参数字符串本身就是坏的（尤其小模型、或者参数里有未转义的引号和换行）；或者流式模式下 `arguments` 是分片到达的，你没拼完就解析了。

**解决**：

```python
import json

def parse_tool_call(tool_call) -> dict:
    """解析工具调用参数，失败时降级为空参数并记录，而不是让整个 Agent 崩掉"""
    raw = tool_call.function.arguments
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 尝试修复常见问题：单引号、尾逗号、中文引号
    fixed = (raw.replace("'", '"')
                .replace("，", ",")
                .replace("：", ":")
                .replace("“", '"').replace("”", '"'))
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)      # 去掉尾逗号
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.error("工具参数解析失败 tool=%s raw=%r err=%s", tool_call.function.name, raw, e)
        return {}
```

流式下必须**先拼完再解析**：

```python
buffers: dict[int, dict] = {}
async for chunk in stream:
    for tc in (chunk.choices[0].delta.tool_calls or []):
        buf = buffers.setdefault(tc.index, {"name": "", "arguments": ""})
        if tc.function.name:
            buf["name"] += tc.function.name
        if tc.function.arguments:
            buf["arguments"] += tc.function.arguments     # 分片追加，不要边拼边解析
# 流结束后统一解析 buffers
```

**预防**：工具参数 schema 尽量简单（扁平对象、少用嵌套数组），并且**工具执行前用 pydantic 再校验一次**，参数非法就把错误信息回灌给模型让它重新调用——这比直接崩掉体验好得多。

---

### A08 · 模型明明被要求输出 JSON，却给了带解释的 markdown

**出现场景**：与 L08 同源，但发生在直接调 API 而非用 LangChain parser 时。

**根因**：prompt 约束不够硬；temperature 太高；或者模型不支持 JSON mode 你却以为它支持。

**解决**：优先用服务端的 JSON mode。

```python
resp = await client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是信息抽取助手。只输出 JSON 对象，不要任何解释、不要 markdown 代码块。"},
        {"role": "user", "content": f"从工单中抽取型号和故障码：\n{ticket_text}\n\n"
                                     f'输出格式：{{"model_no": "...", "error_code": "..."}}'},
    ],
    response_format={"type": "json_object"},     # 服务端保证输出是合法 JSON
    temperature=0.0,
)
data = json.loads(resp.choices[0].message.content)
```

注意：**`json_object` 模式只保证语法合法，不保证 schema 符合你的预期**，字段缺失、类型不对仍然可能发生。所以解析后必须用 pydantic 校验：

```python
from pydantic import BaseModel, ValidationError

class Extracted(BaseModel):
    model_no: str
    error_code: str | None = None

try:
    result = Extracted.model_validate(data)
except ValidationError as e:
    logger.warning("抽取结果不符合 schema: %s，原始输出: %r", e, resp.choices[0].message.content)
```

**预防**：prompt 里给**一个具体的输出示例**比写十句"必须是 JSON"有效得多。同时把「解析失败率」作为一个线上监控指标——它突然上升通常意味着模型版本变了。

---

### A09 · `APIConnectionError: Connection error` / 404 Not Found

**出现场景**：配 base_url 时。

**根因**：`base_url` 少了或多了 `/v1`。OpenAI SDK 会在 `base_url` 后面拼 `/chat/completions`，所以 base_url 必须正好到 `/v1` 为止。

**解决**：

```python
# 正确
client = AsyncOpenAI(base_url="https://api.deepseek.com/v1", api_key=key)
# 本地 vLLM（本书端口约定 8001）
client = AsyncOpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
# 本地 Ollama
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 错误写法
# base_url="https://api.deepseek.com"                    -> 少 /v1
# base_url="https://api.deepseek.com/v1/chat/completions" -> 多了路径
```

```bash
# 验证端点：能列出模型说明 base_url 对了
curl -s http://localhost:8001/v1/models | python -m json.tool
```

**预防**：把各家的 base_url 写成常量表放在 `core/llm.py`，业务代码只传 provider 名字。本书的统一客户端就是这么设计的。

---

### A10 · `BadRequestError: messages with role 'tool' must be a response to a preceding message with 'tool_calls'`

**出现场景**：手写 Agent 的 tool-calling 循环。

**根因**：消息序列不合法。正确的序列必须是：`assistant`（带 `tool_calls`）→ 每个 tool_call 对应一条 `tool` 消息（`tool_call_id` 要对上）→ 下一条 `assistant`。漏掉某个 tool_call 的响应、顺序错了、或者 `tool_call_id` 对不上，都会被拒。

**解决**：

```python
async def agent_loop(user_input: str, tools: list[dict], max_steps: int = 8) -> str:
    """手写 tool-calling 循环，严格维护消息序列的合法性"""
    messages = [{"role": "user", "content": user_input}]
    for step in range(max_steps):
        resp = await client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=tools, tool_choice="auto",
        )
        msg = resp.choices[0].message
        # 关键 1：把 assistant 消息原样追加（包括 tool_calls 字段）
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content

        # 关键 2：每一个 tool_call 都必须有且只有一条对应的 tool 消息
        for tc in msg.tool_calls:
            try:
                result = await execute_tool(tc.function.name, parse_tool_call(tc))
            except Exception as e:
                result = f"工具执行失败: {e}"          # 失败也要返回，不能跳过
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,                 # 关键 3：id 必须原样对应
                "content": str(result)[:8000],         # 截断，防止工具返回撑爆上下文
            })
    return "已达到最大步数上限，请拆分问题后重试。"
```

**预防**：把消息序列的合法性做成一个校验函数，在每次请求前跑一遍（开发环境开启，生产环境可关）：

```python
def validate_messages(messages: list[dict]) -> None:
    """校验 tool-calling 消息序列合法性"""
    pending: set[str] = set()
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            pending = {tc["id"] for tc in m["tool_calls"]}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in pending, f"孤立的 tool 消息: {m['tool_call_id']}"
            pending.discard(m["tool_call_id"])
        elif m["role"] == "assistant":
            assert not pending, f"有 tool_call 未被响应: {pending}"
```

---

### A11 · 推理模型（`deepseek-reasoner` 类）报 400：不支持某些参数

**出现场景**：把普通模型的调用参数原样用在推理模型上。

**根因**：推理类模型通常对采样参数有限制（如不支持 `temperature`、`top_p`、`presence_penalty` 等），也不支持某些功能组合。

**解决**：按模型能力分流参数。

```python
REASONING_MODELS = {"deepseek-reasoner"}

def build_params(model: str, **kw) -> dict:
    """按模型类别过滤不支持的参数，避免 400"""
    params = {"model": model, **kw}
    if model in REASONING_MODELS:
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs"):
            params.pop(k, None)
    return params
```

推理模型的输出里会带思维链字段，业务上通常**只展示最终答案，思维链只记日志**：

```python
resp = await client.chat.completions.create(**build_params("deepseek-reasoner", messages=msgs))
msg = resp.choices[0].message
reasoning = getattr(msg, "reasoning_content", None)      # 字段名以官方文档为准
if reasoning:
    logger.debug("思维链长度=%d", len(reasoning))
answer = msg.content
```

**预防**：在模型网关层维护一张**模型能力表**（是否支持 tools / json_mode / 采样参数 / 最大上下文），调用前统一裁剪参数。参数兼容性是多模型路由必须处理的问题，见 [01-大模型基础与技术选型/02-模型全景图与选型方法论](../01-大模型基础与技术选型/02-模型全景图与选型方法论.md)。

---

### A12 · `httpx.PoolTimeout: ` / 高并发下大量请求排队

**出现场景**：压测时 QPS 上不去，日志里全是连接池超时。

**根因**：httpx 默认连接池很小（`max_connections=100`，`max_keepalive_connections=20`），高并发下连接不够用。或者你每次调用都新建一个 client（连接完全复用不起来）。

**解决**：全局单例 client + 调大连接池。

```python
import httpx
from openai import AsyncOpenAI
from functools import lru_cache

@lru_cache(maxsize=8)
def get_client(provider: str = "deepseek") -> AsyncOpenAI:
    """全局复用的异步客户端，按 provider 缓存，避免重复建连接池"""
    limits = httpx.Limits(
        max_connections=200,
        max_keepalive_connections=50,
        keepalive_expiry=30.0,
    )
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        http_client=httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(connect=10, read=180, write=10, pool=10)),
        max_retries=0,
    )
```

FastAPI 里用 lifespan 管理生命周期：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = get_client()
    yield
    await app.state.client.close()      # 优雅关闭，释放连接

app = FastAPI(lifespan=lifespan)
```

**预防**：**任何 HTTP 客户端都不要在函数内部创建**。这是 Python 服务性能问题的第一大来源，不止 LLM 场景。压测时盯住「连接池等待时间」这个指标。

---
## 8. 评测

评测环节的故障有个共性：**它们会让你得到一个错误的数字，然后你基于这个数字做出错误的决策**。
比一个跑不起来的评测更危险的，是一个跑得很顺但算错了的评测。

### V01 · Judge 模型的输出解析不了，评测集大面积失败

**出现场景**：用 LLM-as-Judge 打分，跑完发现 30% 的样本没有分数。

**根因**：judge 的输出格式不稳定——有时候是 `4`，有时候是 `分数：4`，有时候是 `我认为这个回答值 4 分，因为……`。

**解决**：约束输出 + 稳健解析 + 失败重试，三件套。

```python
JUDGE_PROMPT = """你是严格的评测员。请对候选答案打分。

评分标准（1-5 分整数）：
5 = 完全正确且完整，有明确依据
4 = 正确但有小的遗漏
3 = 部分正确，有明显遗漏
2 = 大部分错误
1 = 完全错误或答非所问

问题：{question}
参考答案：{reference}
候选答案：{candidate}

只输出如下 JSON，不要任何其他文字：
{{"score": <1-5 的整数>, "reason": "<不超过 50 字的理由>"}}"""

import re, json

def parse_judge(raw: str) -> dict | None:
    """解析 judge 输出，多级降级：JSON -> 正则抓分数 -> 失败返回 None"""
    try:
        d = parse_json_robust(raw)          # 见 L08
        s = int(d["score"])
        if 1 <= s <= 5:
            return {"score": s, "reason": d.get("reason", "")}
    except Exception:
        pass
    m = re.search(r"\b([1-5])\b", raw)      # 降级：抓第一个 1-5 的数字
    if m:
        return {"score": int(m.group(1)), "reason": "[降级解析]"}
    return None
```

**关键纪律：解析失败的样本必须单独统计并报告，不能静默丢弃。**

```python
results = [parse_judge(r) for r in raw_outputs]
failed = sum(1 for r in results if r is None)
valid = [r for r in results if r is not None]
print(f"有效 {len(valid)}/{len(results)}，解析失败 {failed} ({failed/len(results):.1%})")
assert failed / len(results) < 0.05, "解析失败率超过 5%，评测结果不可信，请先修 prompt"
```

**预防**：judge 用 `temperature=0`；能用结构化输出就用；评测 harness 里把"解析失败率"作为一个**一等公民指标**打印在报告最上方。详见 [08-评测体系/03-DeepSeek-Harness自研评测框架](../08-评测体系/03-DeepSeek-Harness自研评测框架.md)。

---

### V02 · 指标算出来是 `NaN` 或 `0.0`

**出现场景**：RAGAS 或自建 harness 跑完，某几个指标是 NaN。

**根因**：① 分母为 0（没有检索到任何文档就算 context precision）；② 某些样本的必需字段缺失（`ground_truth` 为空）；③ judge 调用失败被当成 0 分混进平均值；④ 对包含 NaN 的列直接求 `mean()`。

**解决**：

```python
import numpy as np

def safe_mean(values: list[float | None], name: str) -> dict:
    """带缺失统计的安全均值，明确区分「低分」和「没算出来」"""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    missing = len(values) - len(arr)
    if len(arr) == 0:
        return {"metric": name, "value": None, "n": 0, "missing": missing,
                "note": "全部样本缺失，该指标不可用"}
    return {"metric": name, "value": float(arr.mean()), "n": len(arr),
            "missing": missing, "std": float(arr.std()),
            "ci95": float(1.96 * arr.std() / np.sqrt(len(arr)))}
```

分母为 0 的场景要显式定义语义：

```python
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    """Precision@k。检索结果为空时返回 None（不可计算）而不是 0.0（表现很差）"""
    top = retrieved[:k]
    if not top:
        return None
    return len(set(top) & relevant) / len(top)
```

**预防**：**区分"指标值很低"和"指标算不出来"**，这是评测工程里最重要的一条原则。报告里必须同时展示 `value`、`n`（有效样本数）和 `missing`（缺失数）。一个基于 30% 样本算出来的 0.92 毫无意义。

---

### V03 · 金标集里的 doc_id 与检索结果对不上，召回率恒为 0

**出现场景**：算 Recall@k 时发现分子永远是 0。

**根因**：id 体系不一致。金标集里标的是"文档编号"（如 `XJ-200-manual`），检索返回的是"chunk 编号"（如 `XJ-200-manual#chunk-017`）；或者一边带路径前缀一边不带；或者重建索引后 id 全变了。

**解决**：定义一个**归一化的 id 映射**，两边都过一遍。

```python
def normalize_doc_id(raw: str) -> str:
    """把各种形态的 id 归一到「文档级」标识，用于与金标集比对"""
    s = raw.strip()
    s = s.split("#")[0]                 # 去掉 chunk 后缀
    s = s.rsplit("/", 1)[-1]            # 去掉路径前缀
    if s.lower().endswith((".pdf", ".docx", ".md", ".csv")):
        s = s.rsplit(".", 1)[0]         # 去掉扩展名
    return s.lower()

# 评测前先做一次对齐体检
gold_ids = {normalize_doc_id(i) for q in gold_set for i in q["relevant_doc_ids"]}
index_ids = {normalize_doc_id(d["doc_id"]) for d in all_indexed_docs}
missing = gold_ids - index_ids
print(f"金标集引用了 {len(gold_ids)} 个文档，索引里有 {len(index_ids)} 个")
print(f"金标集中有 {len(missing)} 个文档不在索引里: {sorted(missing)[:10]}")
assert len(missing) / max(len(gold_ids), 1) < 0.02, "金标集与索引严重不对齐，评测无意义"
```

**预防**：id 生成规则要**稳定且可复现**——本书用 `sha1(源文件相对路径 + chunk 序号)` 的前 16 位，重建索引后同一份内容的 id 不变。金标集标注时记录的是**文档级 id + 原文片段**，评测时用片段做二次匹配兜底。见 [08-评测体系/02-LLM-Wiki金标集构建工程](../08-评测体系/02-LLM-Wiki金标集构建工程.md)。

---

### V04 · 评测账单失控（跑一次评测花掉一周的预算）

**出现场景**：300 条金标集 × 4 个指标 × 3 个系统对比 = 3600 次 judge 调用，每次带上下文几千 token。

**根因**：评测的 token 消耗常被严重低估。judge 的 prompt 里要放问题 + 参考答案 + 候选答案 + 检索到的上下文，单次几千 token 很正常。

**解决**：四条控本手段，按优先级。

```python
# 1) 缓存：同样的 (judge_model, prompt_hash) 不重复调用
import hashlib, json, sqlite3

class JudgeCache:
    """基于 sqlite 的 judge 结果缓存，跨次运行有效"""
    def __init__(self, path: str = "evals/.judge_cache.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")

    @staticmethod
    def _key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str):
        row = self.conn.execute("SELECT v FROM cache WHERE k=?", (self._key(model, prompt),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, prompt: str, value: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                          (self._key(model, prompt), json.dumps(value, ensure_ascii=False)))
        self.conn.commit()
```

```python
# 2) 分层评测：便宜的先跑，贵的只跑必要的
#    L1 规则指标（0 成本）-> L2 小模型 judge（低成本）-> L3 强模型 judge（只跑抽样或争议样本）

# 3) 抽样：日常回归跑 50 条冒烟集，发版才跑 300 条全量

# 4) 预算硬上限：超了直接停
class BudgetGuard:
    """评测预算守卫，累计 token 超限立即中止"""
    def __init__(self, max_tokens: int):
        self.max_tokens, self.used = max_tokens, 0

    def charge(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.used += prompt_tokens + completion_tokens
        if self.used > self.max_tokens:
            raise RuntimeError(f"评测 token 预算耗尽: {self.used}/{self.max_tokens}")
```

**预防**：评测 harness 必须**打印每次运行的 token 消耗和预估成本**，让成本可见。CI 里的评测门禁用冒烟集（几十条），只有发版流程才跑全量。

---

### V05 · RAGAS 在中文语料上指标异常（faithfulness 恒为 0 或恒为 1）

**出现场景**：直接用 RAGAS 默认配置评中文 RAG。

**根因**：RAGAS 内部会用 LLM 做"陈述拆解"和"NLI 判断"，默认 prompt 是英文的，对中文的拆解质量差；另外默认 LLM/embedding 指向 OpenAI，国内环境要显式换掉。

**解决**：显式配置国产模型 + 中文化。

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

judge_llm = LangchainLLMWrapper(ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
))
emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True},
))

for m in (faithfulness, answer_relevancy, context_precision, context_recall):
    m.llm = judge_llm
    if hasattr(m, "embeddings"):
        m.embeddings = emb

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

**用之前必须做一次「已知答案校准」**：

```text
构造 3 类样本各 10 条，看指标是否符合预期：
  A. 答案完全来自上下文     -> faithfulness 应接近 1
  B. 答案完全是编造的       -> faithfulness 应接近 0
  C. 答案一半来自上下文     -> faithfulness 应在中间

如果 A 和 B 的分数拉不开差距，说明该指标在你的语料上不可用，
应该换成自建的规则 + judge 组合指标，而不是硬用。
```

**预防**：任何第三方评测库在用于决策之前，都要先用**已知答案的样本做校准**。指标本身也是需要被评测的。本书 [08-评测体系/04-RAGAS与自动化评测流水线](../08-评测体系/04-RAGAS与自动化评测流水线.md) 完整演示了这套校准流程。

---

### V06 · `ImportError: cannot import name 'xxx' from 'ragas.metrics'`

**出现场景**：RAGAS 从 0.1 升到 0.2 之后。

**根因**：RAGAS 在 0.2 做了较大重构，指标类名、包路径、`evaluate` 的入参形态都有变化。

**解决**：

```bash
uv pip install "ragas==0.2.*"
python -c "import ragas; print(ragas.__version__)"
python -c "import ragas.metrics as m; print([x for x in dir(m) if not x.startswith('_')])"
```

用最后一行把**当前版本实际可用的指标名**打出来，照着写，不要照抄博客。

**预防**：评测依赖必须钉死小版本并写进 lockfile。评测结果的可比性建立在"评测工具本身没变"的前提上——**升级评测库之后，历史数据全部作废，必须重跑基线**。这一点要写进评测规范。

---

### V07 · 同一套配置跑两次，评测结果差异很大

**出现场景**：优化前后对比，发现"提升"的幅度小于两次重复跑的波动。

**根因**：随机性来源没控住。① 被测模型 `temperature > 0`；② judge 模型有随机性；③ 检索里有随机采样；④ 样本量太小，统计涨落大。

**解决**：

```python
# 1. 固定所有随机源
GEN_PARAMS = {"temperature": 0.0, "top_p": 1.0, "seed": 42}     # seed 视服务端是否支持

# 2. 重复跑 N 次取均值 + 置信区间，而不是单次结果
import numpy as np
from scipy import stats

def compare_with_significance(baseline: list[float], treatment: list[float], alpha: float = 0.05) -> dict:
    """配对 t 检验，判断改进是否统计显著"""
    b, t = np.array(baseline), np.array(treatment)
    diff = t - b
    stat, p = stats.ttest_rel(t, b)
    return {
        "baseline_mean": float(b.mean()),
        "treatment_mean": float(t.mean()),
        "delta": float(diff.mean()),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "note": "p >= 0.05 时不能宣称有提升" if p >= alpha else "差异统计显著",
    }
```

**样本量的经验参考**：要检出 5 个百分点的差异，通常需要几百条样本量级；要检出 1 个百分点，需要的量会大一个数量级。**小样本上的"提升 3 个点"极可能是噪声。**

**预防**：评测报告里**必须带置信区间**。本书的规矩是：没有置信区间的对比数字不进决策会。见 [08-评测体系/01-大模型与RAG评测方法论](../08-评测体系/01-大模型与RAG评测方法论.md) 的统计显著性一节。

---

### V08 · 评测脚本在 Jupyter 里跑到一半卡死不动

**出现场景**：RAGAS 或自建异步 harness 在 notebook 里运行。

**根因**：事件循环冲突（同 L05）+ 并发控制缺失导致的死锁。RAGAS 内部用 asyncio，notebook 环境需要 `nest_asyncio`。

**解决**：

```python
import nest_asyncio
nest_asyncio.apply()

# RAGAS 里限制并发，默认并发可能打满配额触发限流然后卡住
from ragas import evaluate
from ragas.run_config import RunConfig

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    run_config=RunConfig(max_workers=4, timeout=180, max_retries=3),
)
```

**预防**：**评测一律用脚本跑，不要用 notebook。** notebook 适合探索，不适合跑几十分钟的批量任务——内核一断全白干，而且没有日志留痕。本书的 harness 设计成 CLI：`python -m evals.cli run --suite smoke --out reports/`，可重跑、可归档、可进 CI。

---

## 9. 生产

生产环境的问题和开发环境完全是两个物种：**开发环境的问题是"跑不起来"，生产环境的问题是"跑得起来但偶尔出事"**。
本组的排查依赖三样东西：**指标（metrics）看趋势、日志（logs）看细节、链路（trace）看因果**。缺一样都会排查困难。

### P01 · 上线后 P99 延迟突然翻倍，P50 却没变

**出现场景**：监控告警，用户投诉"有时候很慢"。

**根因**：P50 没变说明**大部分请求正常**，问题出在长尾。常见根源：① 某类长查询触发了更多检索或更长生成；② 缓存命中率下降（新版本改了 cache key）；③ 下游某个依赖偶发慢（向量库某个 segment 没 load、API 供应商抖动）；④ GC 或连接池排队；⑤ 有慢请求把连接占住形成排队。

**解决**：按"分位数分解"排查，不要看平均值。

```python
# 在 core/instrument.py 的埋点基础上，按阶段记录耗时
STAGES = ["rewrite", "retrieve_dense", "retrieve_bm25", "rerank", "llm_ttft", "llm_total"]

# 查询 trace 数据，看是哪个阶段的 P99 涨了
"""
SELECT stage,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_ms) AS p50,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99
FROM spans
WHERE ts > now() - interval '1 hour'
GROUP BY stage
ORDER BY p99 DESC;
"""
```

排查清单：

| 检查项 | 命令/方法 | 正常范围（示例性，需按自己环境建基线） |
|---|---|---|
| 缓存命中率 | 监控指标 | 突然下降就是嫌疑 |
| 检索 P99 | 分阶段 trace | 看 `ef` 是否被改大 |
| LLM TTFT | 分阶段 trace | 上游抖动时会整体抬升 |
| 输入 token P99 | 埋点统计 | 突增说明有超长输入 |
| 并发排队时间 | 连接池指标 | 见 P03 |

**预防**：**上线前建立基线**——记录优化前的各阶段 P50/P95/P99，上线后对比。没有基线就没有"突增"这个判断。本书 [03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战](../03-RAG进阶与性能优化/04-性能突围-延迟优化10倍实战.md) 的第一节就是讲怎么建基线，那不是形式主义。

---

### P02 · 内存持续增长，几天后被 OOM Kill

**出现场景**：服务跑三五天就重启一次。

**根因**：候选：① 全局缓存无上限（`dict` 当缓存用，只增不减）；② 对话历史/会话状态没清理；③ 每次请求新建客户端或模型对象；④ 日志对象持有大字符串引用；⑤ asyncio task 泄漏（创建了没 await 也没取消）。

**解决**：先定位再修。

```python
# 1. 用 tracemalloc 做内存快照对比
import tracemalloc

tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
# ... 跑 1000 次请求 ...
snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, "lineno")[:15]:
    print(stat)
```

```python
# 2. 所有缓存必须有上限和过期
from cachetools import TTLCache
import threading

_cache = TTLCache(maxsize=10_000, ttl=3600)      # 而不是裸 dict
_lock = threading.Lock()

def cached_embed(text: str) -> list[float]:
    """带上限与过期的 embedding 缓存"""
    key = hashlib.sha256(text.encode()).hexdigest()
    with _lock:
        if key in _cache:
            return _cache[key]
    vec = embedder.encode(text, normalize_embeddings=True).tolist()
    with _lock:
        _cache[key] = vec
    return vec
```

```python
# 3. 会话状态要有清理策略
#    LangGraph 的 checkpointer 表会无限增长，必须定期清理
"""
DELETE FROM checkpoints WHERE created_at < now() - interval '30 days';
"""
```

```bash
# 4. 监控 asyncio task 数量
python -c "
import asyncio
print(len(asyncio.all_tasks()))   # 在服务里定期打点，只增不减就是泄漏
"
```

**预防**：**任何缓存都要有 maxsize**，这是硬规矩。服务加一个 `/debug/memory` 端点（内网可见）返回 RSS、缓存条数、活跃 task 数，让内存状况随时可查。

---

### P03 · 连接池耗尽，请求大量排队超时

**出现场景**：并发一上来，日志里全是 `PoolTimeout` 或 `QueuePool limit ... overflow`。

**根因**：连接池太小；或者连接用完没归还（忘了 `close`，或者异常路径下泄漏）；或者慢请求长时间占着连接。

**解决**：

```python
# 1. HTTP 客户端（见 A12）：全局单例 + 调大 limits

# 2. 数据库连接池
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pwd@127.0.0.1:5433/app",
    pool_size=20,             # 常驻连接
    max_overflow=10,          # 峰值可临时多开
    pool_timeout=30,          # 等待连接的超时
    pool_recycle=1800,        # 30 分钟回收一次，防止被数据库/中间件掐断
    pool_pre_ping=True,       # 取连接前 ping 一下，避免拿到死连接
)
```

```python
# 3. 用上下文管理器，保证异常路径也归还
async with engine.begin() as conn:
    await conn.execute(...)
# 而不是手动 acquire/release
```

```python
# 4. 给整个请求设置超时上限，别让慢请求无限占资源
import asyncio

async def handle_with_deadline(coro, seconds: float = 30.0):
    """给请求加硬超时，超时即释放所有资源"""
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning("请求超时被中止，deadline=%ss", seconds)
        raise
```

**预防**：连接池大小要**按并发量算**而不是拍脑袋：`pool_size ≈ 峰值 QPS × 平均持有时间(秒)`，再乘 1.5 的余量。并把「连接池使用率」做成监控指标，超过 80% 就告警。

---

### P04 · 向量索引持续膨胀，磁盘和内存都在涨

**出现场景**：知识库每天增量更新，跑了三个月磁盘涨了 5 倍。

**根因**：向量库的"删除"通常是**标记删除**，物理空间要等 compaction 才释放。频繁更新（先删后插）会积累大量墓碑数据。另外重复入库（同一份文档被灌了多次）也很常见。

**解决**：

```python
from pymilvus import Collection, utility

col = Collection("huacheng_kb")
col.flush()
print("实体数:", col.num_entities)

# 手动触发 compaction 回收空间
job_id = col.compact()
print("compaction 状态:", col.get_compaction_state())
```

```python
# 去重检查：同一份内容不应该有多条
from collections import Counter
rows = col.query(expr="pk > 0", output_fields=["doc_id", "chunk_hash"], limit=16384)
dup = [k for k, v in Counter(r["chunk_hash"] for r in rows).items() if v > 1]
print(f"重复 chunk: {len(dup)} 个")
```

**预防**：采用**版本化索引 + 别名切换**，而不是原地增删改：

```python
# 每次全量重建写入新 collection，建好后切别名，再删老的
new_name = f"huacheng_kb_v{int(time.time())}"
build_collection(new_name)                       # 全量构建
utility.alter_alias(collection_name=new_name, alias="huacheng_kb")   # 原子切换
utility.drop_collection(old_name)                # 观察一段时间后再删
```

这样既避免了膨胀，又天然支持回滚。增量更新只用于时效性要求高的小批量数据。

---

### P05 · 缓存击穿：热点 key 过期瞬间大量请求直接打到后端

**出现场景**：每天早上 9 点（上班高峰）服务抖一下。

**根因**：热点问题的缓存同时过期，N 个并发请求同时发现缓存没了，同时去调用 LLM，后端瞬间被打爆。

**解决**：单飞（singleflight）+ 随机过期时间。

```python
import asyncio, random
from typing import Callable, Awaitable

class SingleFlight:
    """同一 key 的并发请求只真正执行一次，其余等待复用结果"""
    def __init__(self):
        self._inflight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, fn: Callable[[], Awaitable]):
        async with self._lock:
            if key in self._inflight:
                fut = self._inflight[key]
                wait = True
            else:
                fut = asyncio.get_event_loop().create_future()
                self._inflight[key] = fut
                wait = False
        if wait:
            return await fut
        try:
            result = await fn()
            fut.set_result(result)
            return result
        except Exception as e:
            fut.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)

sf = SingleFlight()

async def answer_cached(question: str) -> str:
    """带单飞保护与抖动过期的问答缓存"""
    key = hashlib.sha256(question.encode()).hexdigest()
    if (hit := await redis.get(key)) is not None:
        return hit
    result = await sf.do(key, lambda: rag_chain.ainvoke({"question": question}))
    # 过期时间加 ±10% 抖动，避免同批缓存同时失效
    ttl = int(3600 * random.uniform(0.9, 1.1))
    await redis.set(key, result, ex=ttl)
    return result
```

**预防**：对**可预测的热点**（如每天固定的高频问题）做**预热**：凌晨低峰期主动跑一遍 Top 100 问题填充缓存。同时给缓存加"逻辑过期"——过期后先返回旧值，后台异步刷新，用户永远不等。

---

### P06 · 并发下不同用户的结果串味（A 用户看到 B 用户的上下文）

**出现场景**：压测或高峰期，偶发地有用户收到不属于自己的内容。

**根因**：**共享可变状态**。典型元凶：模块级的全局变量存请求上下文、单例对象上挂了 `self.current_query`、用类属性当缓存、或者 `contextvars` 用错（在错误的上下文里 set）。这是生产环境最严重的一类事故——**它是数据泄漏，不只是 bug**。

**解决**：

```python
# 错误示范：模块级可变状态
# CURRENT_USER = None                       # 绝对禁止
# class RagService:
#     def __init__(self): self.last_docs = None    # 实例被共享时同样危险

# 正确做法 1：请求上下文用 contextvars，天然隔离到协程
from contextvars import ContextVar
from dataclasses import dataclass

@dataclass
class RequestContext:
    trace_id: str
    user_id: str
    tenant_id: str

_ctx: ContextVar[RequestContext | None] = ContextVar("request_ctx", default=None)

def get_ctx() -> RequestContext:
    """获取当前请求上下文，未设置时直接报错而不是返回默认值"""
    c = _ctx.get()
    if c is None:
        raise RuntimeError("请求上下文未初始化")
    return c
```

```python
# 正确做法 2：中间件里为每个请求设置上下文
from fastapi import Request
import uuid

@app.middleware("http")
async def context_middleware(request: Request, call_next):
    token = _ctx.set(RequestContext(
        trace_id=request.headers.get("x-trace-id", str(uuid.uuid4())),
        user_id=request.headers.get("x-user-id", "anonymous"),
        tenant_id=request.headers.get("x-tenant-id", "default"),
    ))
    try:
        return await call_next(request)
    finally:
        _ctx.reset(token)          # 必须 reset，否则上下文会泄漏到下一个请求
```

```python
# 正确做法 3：所有服务对象设计成无状态的，状态只在参数里流动
class RagService:
    """无状态服务：所有依赖在构造时注入且不可变，请求数据只走参数"""
    def __init__(self, retriever, llm):
        self._retriever = retriever      # 只读依赖
        self._llm = llm

    async def answer(self, question: str, ctx: RequestContext) -> dict:
        docs = await self._retriever.aget(question, tenant=ctx.tenant_id)
        ...
```

**预防**：代码审查时**专门看有没有模块级可变变量和实例上的请求态字段**。写一条压测用例：100 个并发，每个请求带唯一标识，断言返回内容里包含且只包含自己的标识。这条用例进 CI。缓存的 key 必须包含租户/用户维度，见 [10-工程化与生产落地/03-安全合规与幻觉治理](../10-工程化与生产落地/03-安全合规与幻觉治理.md)。

---

### P07 · Langfuse / 可观测上报阻塞主流程，接口延迟被拖慢

**出现场景**：接入 trace 上报后，接口 P99 明显上升；或者 Langfuse 挂了导致整个服务不可用。

**根因**：同步上报，或者 flush 在请求路径上。**监控系统绝不能成为业务的强依赖。**

**解决**：异步 + 降级 + 超时。

```python
import asyncio
from contextlib import suppress

class SafeTracer:
    """可观测上报的安全包装：异步、有界队列、失败静默、绝不阻塞业务"""
    def __init__(self, handler, queue_size: int = 10_000):
        self._handler = handler
        self._q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None
        self._dropped = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._worker())

    def emit(self, event: dict) -> None:
        """非阻塞投递。队列满了就丢弃并计数，绝不让业务等待"""
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    async def _worker(self) -> None:
        while True:
            batch = [await self._q.get()]
            # 攒批，减少上报次数
            with suppress(asyncio.QueueEmpty):
                for _ in range(49):
                    batch.append(self._q.get_nowait())
            try:
                await asyncio.wait_for(self._handler.send_batch(batch), timeout=5.0)
            except Exception as e:
                logger.warning("trace 上报失败（已忽略，不影响业务）: %s", e)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
```

**预防**：三条铁律。① 上报永远异步；② 上报失败只记日志不抛异常；③ 队列有界，满了就丢并统计丢弃量（丢弃量本身是个监控指标）。定期做一次"**把 Langfuse 停掉，看业务是否完全正常**"的演练。

---

### P08 · 发版时正在处理的请求被中断，用户收到 502

**出现场景**：滚动发布或重启服务时。

**根因**：进程收到 SIGTERM 立刻退出，正在生成的长响应直接断掉。LLM 应用的单请求耗时比普通 Web 服务长一个数量级（几十秒），不做优雅停机必然中断。

**解决**：

```python
from contextlib import asynccontextmanager
import signal, asyncio

_shutting_down = False
_inflight = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = get_client()
    yield
    # 收到停机信号后：先拒绝新请求，再等存量请求做完
    global _shutting_down
    _shutting_down = True
    deadline = 60
    while _inflight > 0 and deadline > 0:
        logger.info("优雅停机中，剩余在途请求 %d，最多再等 %ds", _inflight, deadline)
        await asyncio.sleep(1)
        deadline -= 1
    await app.state.client.close()

@app.middleware("http")
async def inflight_counter(request, call_next):
    global _inflight
    if _shutting_down and request.url.path != "/health":
        return JSONResponse({"detail": "服务正在重启，请稍后重试"}, status_code=503)
    _inflight += 1
    try:
        return await call_next(request)
    finally:
        _inflight -= 1
```

```yaml
# K8s 侧配合：给足 terminationGracePeriod，并用 preStop 先摘流量
spec:
  terminationGracePeriodSeconds: 90
  containers:
    - name: app
      lifecycle:
        preStop:
          exec:
            command: ["sh", "-c", "sleep 10"]     # 等负载均衡把本实例摘掉
      readinessProbe:
        httpGet: { path: /health/ready, port: 8080 }
        periodSeconds: 5
```

**预防**：健康检查要区分 **liveness**（进程活着吗）和 **readiness**（能接新请求吗）。停机时先让 readiness 变红把流量摘掉，再等在途请求结束。发版流程里加一条"观察期"：新版本先接 5% 流量跑 10 分钟。

---

### P09 · 日志把磁盘打爆 / 日志里出现用户隐私数据

**出现场景**：运行一个月后磁盘告警；或者安全审计发现日志里有手机号、身份证号。

**根因**：① 没配日志轮转；② debug 级别把完整 prompt 和检索内容全打了（RAG 的 prompt 动辄几千字，QPS 稍高一天就是几十 GB）；③ 没做脱敏。

**解决**：

```python
# 1. 脱敏：core/logger.py 已提供，这里是核心逻辑
import re

PATTERNS = [
    (re.compile(r"1[3-9]\d{9}"), lambda m: m.group()[:3] + "****" + m.group()[-4:]),
    (re.compile(r"\d{17}[\dXx]"), lambda m: m.group()[:6] + "********" + m.group()[-4:]),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), lambda m: m.group()[:2] + "***@" + m.group().split("@")[1]),
    (re.compile(r"(sk-|hf_)[A-Za-z0-9]{16,}"), lambda m: m.group()[:6] + "***REDACTED***"),
]

def desensitize(text: str) -> str:
    """对日志文本做脱敏，覆盖手机号、身份证、邮箱、API key"""
    for pat, repl in PATTERNS:
        text = pat.sub(repl, text)
    return text
```

```python
# 2. 分级：完整内容只在采样时记录，且进专门的通道
import random

def log_llm_call(prompt: str, completion: str, sample_rate: float = 0.01) -> None:
    """LLM 调用日志：默认只记摘要，按比例抽样记全文"""
    logger.info("llm_call prompt_len=%d completion_len=%d", len(prompt), len(completion))
    if random.random() < sample_rate:
        logger.debug("llm_call_full prompt=%r completion=%r",
                     desensitize(prompt[:2000]), desensitize(completion[:2000]))
```

```yaml
# 3. 轮转（Docker 侧，见 D14 的 daemon.json，或 compose 里单独配）
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

**预防**：**全文内容应该进 trace 系统（Langfuse）而不是进日志文件**——trace 有结构、有保留策略、有权限控制，日志文件没有。日志只留结构化的指标字段（耗时、token 数、命中与否、错误码）。脱敏函数要有单元测试，覆盖各种边界情况。

---

## 10. 排错通用方法论

前面 114 条是"已知问题的答案"。但你迟早会遇到表里没有的问题。这一节讲**怎么自己找到答案**。

### 10.1 先分类：你遇到的是哪种问题

```mermaid
flowchart TD
    Start([出问题了]) --> Q1{程序抛异常了吗}
    Q1 -->|是| Q2{以前跑通过吗}
    Q1 -->|否, 只是结果不对| Q5[质量问题<br/>走 10.5 的链路二分]

    Q2 -->|从来没跑通过| A1[环境/配置问题<br/>查版本、查路径、查权限<br/>见 §1 §2]
    Q2 -->|以前能跑| Q3{改了什么}
    Q3 -->|改了代码| A2[git diff / git bisect<br/>见 10.2]
    Q3 -->|改了依赖| A3[比对 lockfile<br/>见 10.2]
    Q3 -->|什么都没改| A4[外部变化:<br/>数据量涨了/磁盘满了/<br/>上游 API 变了/证书过期<br/>见 10.6]

    Q5 --> Q6{是必现还是偶现}
    Q6 -->|必现| A5[最小复现 → 二分定位<br/>见 10.3 10.5]
    Q6 -->|偶现| A6[并发问题/资源问题/<br/>数据分布问题<br/>先加埋点扩大观测面<br/>见 10.6]
```

**第一个问题永远是：这是确定性问题还是随机性问题。**
确定性问题（必现）用二分法几乎总能定位；随机性问题（偶现）必须先增加可观测性，否则是在黑暗中挥拳。

---

### 10.2 怎么二分定位

二分的本质是：**每做一次实验，都要能排除掉一半的可能性。** 随机试探是效率最低的排错方式。

**三种维度的二分：**

**① 时间维度（改动二分）**

```bash
# 代码改动用 git bisect，它是二分法的自动化实现
git bisect start
git bisect bad                    # 当前版本有问题
git bisect good v1.2.0            # 这个版本是好的
# git 自动 checkout 中间的提交，你测一次，告诉它 good 还是 bad
git bisect good        # 或 git bisect bad
# 重复 log2(N) 次，就能精确定位到引入问题的那个提交
git bisect reset

# 依赖变更用 lockfile 比对
diff <(git show HEAD~10:requirements.lock) requirements.lock
```

**② 空间维度（链路二分）**

RAG 链路有 6 个环节：解析 → 切分 → 向量化 → 检索 → 重排 → 生成。
不要从第一个开始查，**从中间开始**：

```python
# 在 rerank 之后、生成之前打断点，看送进 LLM 的上下文对不对
ctx = format_docs(reranked_docs)
print("=" * 60)
print(f"上下文长度: {len(ctx)} 字符")
print(ctx[:2000])
print("=" * 60)
# 上下文里有正确答案 -> 问题在生成侧（后 1/2）
# 上下文里没有       -> 问题在检索侧（前 1/2）
# 一次实验排除一半，再在剩下的一半里继续二分
```

**③ 配置维度（变量二分）**

同时改了 5 个参数，效果变差了。不要一个个往回改，**先把 5 个全部还原确认能恢复，然后一次加一半**：

```text
基线 A（全部还原）      -> 好
实验 B（加参数 1,2,3）  -> 坏   => 问题在 1,2,3 里
实验 C（加参数 1,2）    -> 好   => 问题是参数 3
共 3 次实验定位，而不是 5 次
```

**二分法的前提条件**：必须有一个**快速、稳定的判定方法**。如果验证一次要 10 分钟，先想办法把它压到 10 秒（缩小数据集、固定随机种子、缓存中间结果）。**投资在"快速验证"上的时间，永远是值得的。**

---

### 10.3 怎么读 traceback

Python 的 traceback 是**从外到内**的调用栈，最后一行是异常，倒数第二段是真正出错的代码位置。

```text
Traceback (most recent call last):
  File "app/main.py", line 42, in answer              <- 你的代码，入口
    result = await chain.ainvoke({"question": q})
  File ".../langchain_core/runnables/base.py", line 2875, in ainvoke
    input = await step.ainvoke(input, config)
  File ".../langchain_core/runnables/base.py", line 5352, in ainvoke
    return await self._acall_with_config(...)
  File "app/rag/retriever.py", line 88, in _aget_relevant_documents   <- 你的代码，最后一次出现
    return self._rrf_fuse(dense, sparse)[: self.top_k]
  File "app/rag/retriever.py", line 120, in _rrf_fuse
    key = d.metadata["doc_id"]                        <- 真正出错的这一行
KeyError: 'doc_id'                                     <- 异常类型和值
```

**阅读顺序（这个顺序是有讲究的）：**

1. **先看最后一行**：异常类型 + 值。`KeyError: 'doc_id'` 告诉你缺了什么。
2. **再看倒数第二段**：出错的代码行。这是"案发现场"。
3. **然后从下往上找到最后一个属于你自己代码的帧**：这是你能改的地方。上面那一堆框架内部的帧通常不是重点。
4. **最后看第一段**：入口在哪，这次调用是什么场景触发的。

**几个实用技巧：**

```python
# 1. 异常链：看 "The above exception was the direct cause of" 和 "During handling of"
#    前者是显式 raise ... from e，后者是在 except 块里又出错了——后者往往掩盖了真实原因

# 2. 打印更完整的上下文变量
import traceback, sys

try:
    risky_call()
except Exception:
    exc_type, exc_value, tb = sys.exc_info()
    while tb.tb_next:
        tb = tb.tb_next
    print("出错帧的局部变量:")
    for k, v in tb.tb_frame.f_locals.items():
        print(f"  {k} = {str(v)[:200]}")
    traceback.print_exc()
```

```bash
# 3. 异步代码的 traceback 经常被截断，开 debug 模式能看到完整栈
PYTHONASYNCIODEBUG=1 python app/main.py
```

```python
# 4. 第三方库内部报错时，去看它的源码——路径就在 traceback 里
#    不要害怕读库的源码，报错行号已经精确指给你了
```

**特别提醒**：LangChain / LangGraph 的 traceback 会很长（框架层层包装），**不要被吓退**。用 `grep` 过滤出自己的代码：

```bash
python app/main.py 2>&1 | grep -E "File \"(app|core|rag|agents)/"
```

---

### 10.4 怎么做最小复现

最小复现（Minimal Reproducible Example）是排错效率的分水岭。一个 500 行的复现脚本和一个 15 行的复现脚本，定位速度差一个数量级。

**构造步骤：**

```text
1. 固定随机性       -> temperature=0, seed 固定, 关掉缓存
2. 砍掉外部依赖     -> 用固定的假数据替代真实检索/数据库
3. 砍掉无关代码     -> 二分删除：删一半，还能复现吗？能就继续删
4. 缩小数据规模     -> 10000 条 -> 100 条 -> 10 条 -> 1 条
5. 降低抽象层次     -> 去掉框架，用最底层的 API 直接调
```

**第 5 步特别重要**。很多"LangChain 的 bug"用原生 SDK 复现一遍就发现是自己参数传错了：

```python
# 复现脚本模板：repro.py
"""最小复现：<一句话描述问题>
环境：python 3.11 / langchain 0.3.7 / openai 1.54.3
预期：返回包含 E041 处理步骤的答案
实际：返回空字符串
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")

FIXED_CONTEXT = "E041 表示主轴过载。处理步骤：1. 断电 2. 检查负载 3. 复位"   # 写死，不走检索

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": f"资料：{FIXED_CONTEXT}\n问题：E041 怎么处理？"}],
    temperature=0.0,
)
print(repr(resp.choices[0].message.content))
```

**最小复现的额外收益：**
- 它就是你提 issue 时最有力的材料（维护者最烦"我的代码不工作"）
- 它可以直接变成一条回归测试，防止问题复发
- 构造它的过程中，有相当比例的问题会自己浮出水面——**"砍到只剩 20 行还是有问题"和"砍掉某一行就好了"，这两种结果都直接指向答案**

---

### 10.5 什么时候看日志，什么时候看 trace，什么时候看指标

这三样东西不是重复建设，它们回答的是**不同的问题**。

| 工具 | 回答什么问题 | 数据形态 | 什么时候用 | 本书章节 |
|---|---|---|---|---|
| **指标 Metrics** | "有没有问题？多严重？" | 时间序列聚合值 | 发现问题、判断影响面、看趋势 | [10.2 可观测性](../10-工程化与生产落地/02-可观测性与链路追踪.md) |
| **链路 Trace** | "这一次请求发生了什么？慢在哪？" | 单次请求的调用树 | 定位某个具体请求的问题 | [10.2 可观测性](../10-工程化与生产落地/02-可观测性与链路追踪.md) |
| **日志 Logs** | "那一刻代码内部的状态是什么？" | 离散的文本事件 | 已经定位到代码，要看细节 | [00.2 Python 工程基础](../00-前置准备/02-Python工程基础速补.md) |

**标准排查流程（从粗到细）：**

```mermaid
flowchart LR
    A[告警/投诉] --> B[看指标 Metrics]
    B --> C{影响面多大?<br/>什么时候开始的?}
    C --> D[按时间窗口筛 Trace]
    D --> E{哪个阶段慢/错?}
    E --> F[看该阶段的日志]
    F --> G{代码内部状态?}
    G --> H[本地最小复现]
    H --> I[修复 + 加回归测试]
```

**具体的判断规则：**

- **"系统是不是出问题了"** → 看指标。不要去翻日志，日志里永远有 warning，看不出趋势。
- **"这个用户投诉的这次对话为什么答错了"** → 看 trace。有 `trace_id` 就直接查，这是最快的路径。
- **"为什么这个函数返回了空列表"** → 看日志。trace 只记录跨度和耗时，函数内部的分支逻辑要靠日志。
- **"为什么昨天比前天慢"** → 看指标对比 + 抽样 trace 对比。
- **"这个异常一天出现几次"** → 看日志聚合（按错误指纹分组计数）。

**LLM 应用特有的一条**：**prompt 和 completion 的完整内容应该进 trace，不进日志。**
理由见 P09——trace 有结构化字段、有保留期策略、有权限控制、能按 session 串起来看；日志文件三样都没有，还会打爆磁盘。

**埋点的最小集合**（本书 `core/instrument.py` 已实现，这里列的是"至少要有哪些"）：

```python
# 每个 span 至少记录
{
    "trace_id": "...",          # 串起一次完整请求
    "span_name": "retrieve_dense",
    "start_ts": ...,
    "duration_ms": 87,
    "status": "ok",             # ok / error
    # 业务字段
    "query_len": 23,
    "top_k": 10,
    "hit_count": 10,
    "cache_hit": False,
    # LLM 特有
    "model": "deepseek-chat",
    "prompt_tokens": 3241,
    "completion_tokens": 412,
    "ttft_ms": 680,
}
```

**没有 `trace_id` 贯穿全链路的系统，生产排障基本靠猜。** 如果你的系统现在还没有，这件事的优先级高于任何性能优化。

---

### 10.6 偶现问题的专门打法

偶现问题（复现率 1% 甚至更低）无法用二分法，因为**你无法判断一次实验的结果**。打法完全不同：

**第一步：不要急着修，先提高可观测性。**

```python
# 在怀疑的路径上加详细埋点，把「现场」保存下来
def retrieve(query: str, ctx: RequestContext) -> list[Document]:
    """检索，异常时保存完整现场供事后分析"""
    try:
        docs = _do_retrieve(query)
        if not docs:
            # 空结果是个值得记录的异常状态，即使它不抛异常
            logger.warning("retrieve_empty trace_id=%s query=%r expr=%r",
                           ctx.trace_id, query[:200], build_expr(ctx))
        return docs
    except Exception:
        # 保存现场：输入、配置、环境
        snapshot = {"trace_id": ctx.trace_id, "query": query,
                    "collection": settings.milvus_collection,
                    "params": SEARCH_PARAMS, "ts": time.time()}
        Path(f"/data/incidents/{ctx.trace_id}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False))
        raise
```

**第二步：找相关性。** 把出问题的样本捞出来，和正常样本比对：

```text
问自己这些问题：
  - 出问题的请求有什么共同点？（同一个租户？同一类问题？同一个时段？）
  - 集中在某几个实例上吗？（那可能是某台机器的环境问题）
  - 和流量高峰相关吗？（那是并发或资源问题）
  - 和数据更新相关吗？（那是索引或缓存一致性问题）
  - 输入长度分布有差异吗？（那可能是截断或超限）
```

**第三步：按偶现问题的常见成因逐一排除。**

| 成因 | 特征 | 验证方法 |
|---|---|---|
| 并发竞态 | 高峰期集中出现 | 压测复现；审查共享状态（见 P06） |
| 资源耗尽 | 运行一段时间后出现 | 看内存/连接池/文件句柄曲线 |
| 上游抖动 | 随机分布，重试就好 | 看上游的错误率指标 |
| 数据分布 | 集中在某类输入 | 按输入特征分组统计错误率 |
| 缓存不一致 | 更新后一段时间内出现 | 清缓存看是否立刻恢复 |
| 时序依赖 | 特定操作序列后出现 | 从 trace 里还原操作序列 |

**第四步：如果实在定位不了，加"防御性兜底"降低影响面，同时继续观察。**
这不是投降——**在生产环境，把故障率从 1% 降到 0.01% 的价值，常常大于花两周找到根因。** 但兜底必须带告警，否则问题会被永久掩盖。

---

### 10.7 一份通用的排错检查清单

遇到没见过的问题时，按这个顺序过一遍。经验表明，**大部分问题在前 6 项就能解决**。

```text
□  1. 完整读一遍报错信息（不是只看最后一行，中间的提示常有答案）
□  2. 确认版本：python / 关键库 / CUDA / 服务端。和文档要求的一致吗？
□  3. 确认这段代码/配置以前跑通过吗？没跑通过就是环境问题
□  4. 确认改了什么：git diff / lockfile diff / 配置 diff / 数据变更
□  5. 确认路径和权限：文件存在吗？能读吗？绝对路径对吗？
□  6. 确认环境变量：env | grep -i 相关关键字，确认真的被读到了
□  7. 用最原始的方式验证依赖：curl 打接口 / psql 连库 / python -c 导包
□  8. 把报错信息（去掉自己的路径和变量名）搜一遍：官方 issue > Stack Overflow > 博客
□  9. 去读源码：traceback 已经把文件和行号给你了
□ 10. 构造最小复现（见 10.4）
□ 11. 二分定位（见 10.2）
□ 12. 还是不行：提 issue，附上最小复现 + 环境信息 + 完整 traceback
```

**关于搜索报错信息的技巧：**

```text
好的搜索词：
  "Can receive only one value per step" langgraph
  milvus "fail to search on all shard leaders"
  ragas faithfulness nan chinese

差的搜索词：
  langgraph 报错怎么办
  我的 RAG 检索不准
  python 报错 KeyError
```

**去掉专属信息**（你的路径、你的变量名、你的具体数字），**保留框架的固定措辞**，加上**库名作为限定词**。这是搜索报错的核心技巧。

---

### 10.8 最后：建立你自己的报错手册

本附录是作者的经验汇总，但**最有价值的报错手册永远是你自己那一份**——它记录的是你的环境、你的技术栈、你的团队踩过的坑。

建议在项目里维护 `docs/troubleshooting.md`，每解决一个非平凡的问题就补一条：

```markdown
## [2026-09-17] Milvus 检索偶发返回空

**现象**：约 0.3% 的请求检索结果为空，重试即正常
**排查**：从 trace 发现全部集中在凌晨 2:10-2:15，与增量索引任务时间重合
**根因**：索引任务执行 release + load，期间检索请求落到未 load 的 collection
**修复**：改用别名切换方案（见 P04），新索引建好再切，不再 release 线上 collection
**验证**：连续观察 7 天，未再出现
**相关**：commit a3f2b1c，附录 A 的 P04
```

这份文档的价值随时间指数增长。三个月后新人遇到同样的问题，能从 30 分钟缩短到 3 分钟。

记录时注意四要素：**现象（可搜索的原文）、排查路径（怎么想的）、根因（为什么）、修复与验证（怎么确认好了）**。
只写"改了 xx 配置就好了"的记录，三个月后你自己都看不懂。

---

## 附：本表与各章的对应关系

| 本表分组 | 主要对应章节 |
|---|---|
| §1 环境与安装 | [00-前置准备/01-开发环境搭建](../00-前置准备/01-开发环境搭建（Python-CUDA-Docker）.md) |
| §2 Docker 与服务 | [00-前置准备/01-开发环境搭建](../00-前置准备/01-开发环境搭建（Python-CUDA-Docker）.md)、[10-工程化与生产落地/01-生产架构设计与部署拓扑](../10-工程化与生产落地/01-生产架构设计与部署拓扑.md) |
| §3 模型加载与推理 | [01-大模型基础与技术选型/03-本地部署实战](../01-大模型基础与技术选型/03-本地部署实战（Ollama-vLLM-SGLang）.md)、[01-大模型基础与技术选型/04-算力配置与成本测算](../01-大模型基础与技术选型/04-算力配置与成本测算.md) |
| §4 RAG 链路 | [02-RAG基础篇](../02-RAG基础篇/)、[03-RAG进阶与性能优化](../03-RAG进阶与性能优化/) 全部章节 |
| §5 LangChain / LangGraph | [04-LangChain与工程框架](../04-LangChain与工程框架/) 全部章节 |
| §6 微调 | [05-微调LoRA与PEFT](../05-微调LoRA与PEFT/) 全部章节 |
| §7 API 调用 | [01-大模型基础与技术选型/05-提示工程与结构化输出](../01-大模型基础与技术选型/05-提示工程与结构化输出.md)、[06-Agent智能体/02-Function-Calling与工具设计](../06-Agent智能体/02-Function-Calling与工具设计.md) |
| §8 评测 | [08-评测体系](../08-评测体系/) 全部章节 |
| §9 生产 | [10-工程化与生产落地](../10-工程化与生产落地/) 全部章节 |
| §10 通用方法论 | 贯穿全书 |

---

**上一章** [11-能力模型与职业发展/01-大模型技术人才能力模型与市场分析](../11-能力模型与职业发展/01-大模型技术人才能力模型与市场分析.md) | **下一章** [附录 B-术语中英对照表](B-术语中英对照表.md)
