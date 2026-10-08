# 环境报告（P0 产出）

> 采集时间：2026-10-08
> 采集方式：全部为本机实际执行的命令输出，无估算、无引用文档数字。
> 对应提示词：`docs/prompts-for-windows-agent.md` 的 **P0 · 环境搭建**。

---

## 1. 硬件与系统

| 项 | 实测值 | 备注 |
|---|---|---|
| 操作系统 | Microsoft Windows 11 家庭版 中文版 | `10.0.26200` Build 26200，64 位 |
| 机器 | LENOVO / 82YA | |
| CPU | 13th Gen Intel Core i7-13700H | 14 核 20 线程 |
| 物理内存 | **15.8 GB**（2×8GB Samsung 5600） | |
| 显卡 | NVIDIA GeForce RTX 4060 **Laptop** GPU | 单卡，无第二张 GPU |
| **显存** | **8188 MiB（8 GB）** | ← 与原文档假设的 16G 不符，见第 3 节 |
| 驱动 | 566.24（CUDA Version 12.7） | |
| 磁盘 C: | 200.0 GB，装完后剩余 **57.6 GB** | 装前 67.7 GB |
| 磁盘 D: | 751.6 GB，剩余 **328.0 GB** | 项目与索引放这里 |

---

## 2. 软件栈

| 组件 | 版本 | 路径 / 说明 |
|---|---|---|
| Miniconda | conda **26.7.1** | 装在 `C:\Users\28620\miniconda3`（标准位置，非工作区内） |
| Python | **3.11.17** | `C:\Users\28620\miniconda3\envs\ec\python.exe` |
| PyTorch | **2.6.0+cu126** | 见下方「为什么不是文档写的 2.4.1」 |
| torchvision | 0.21.0+cu126 | |
| CUDA 运行时 | 12.6 | `torch.version.cuda` |
| transformers | 5.19.0 | |
| bitsandbytes | 0.50.2 | Windows 原生 **有** wheel |
| 向量库 | milvus-lite **3.2.1** / faiss-cpu 1.15.1 | 两者都可用，见第 3 节 |
| 依赖总数 | 132 个 | 完整清单见第 5 节 |

### 为什么不是文档写的 torch 2.4.1

`docs/00-setup-windows.md` 第 2 节钉的是 `torch==2.4.1+cu121`，但这条与 `requirements.txt`
的下限约束**互相矛盾**，实测会直接崩：

```
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1+cu121
NameError: name 'nn' is not defined
```

原因链（逐条查过 PyPI 元数据）：

- `requirements.txt` 只写 `transformers>=4.44`，pip 会装到 **transformers 5.19.0**
- transformers 5.x 运行时**硬性要求 torch ≥ 2.5**
- 而 cu121 通道最高只到 torch 2.5.1，`cu124` 到 2.6.0，**`cu126` 才能拿到 2.6+**

各 CUDA 通道的 torch 实际上限（`download.pytorch.org` 实测，cp311/win_amd64）：

| 通道 | 可用版本区间 |
|---|---|
| cu121 | 2.1.0 → 2.5.1 |
| cu124 | 2.4.0 → 2.6.0 |
| **cu126** | **2.6.0 → 2.14.1** ← 本机采用 |
| cu128 | 2.7.0 → 2.11.0（需驱动 ≥ 12.8，本机驱动只到 12.7，**用不了**） |

**最终选择 `torch 2.6.0+cu126`**：满足 transformers ≥ 2.5 的要求，同时是 bitsandbytes 0.50.2
（`torch<3,>=2.4`）等一整套微调依赖都被充分验证过的版本，比直接上 2.14.1 稳。

---

## 3. 三条与原文档不符的实测发现

### 3.1 显存是 8GB，不是 16GB ★

三种独立方法一致：

| 方法 | 结果 |
|---|---|
| `nvidia-smi` | `1230MiB / 8188MiB` |
| `nvidia-smi --query-gpu=memory.total` | `8188 MiB` |
| `torch.cuda.get_device_properties(0).total_memory` | **8.0 GB** |

> 「16G」实际是**系统内存**（15.8 GB）。Windows 显示适配器属性里的「总可用图形内存」
> ≈ 专用显存 8GB + 共享系统内存 8GB ≈ 16GB，但共享部分 **CUDA 用不了**。
> 另外 RTX 4060 **Laptop** 这个型号本身只有 8GB 版本（4050=6G / 4060=8G / 4070=8G /
> 4080=12G / 4090=16G）。

**影响**：`AGENTS.md` 第 1.5 节的显存预算表和 `docs/00-setup-windows.md` 第 5 节整张表都是按 16G 写的，
需要按下表修正。**P0–P12 的 RAG 与 Agent 链路不受影响，只影响 P14/P15 微调。**

| 任务 | 原表（按 16G 写） | 8G 卡上的修正 |
|---|---|---|
| Qwen3-8B 推理 4bit | ~6.5G ✅ 舒服 | 加上系统占用后**不可行**，降到 4B |
| Qwen3-14B 推理 4bit | ~9.5G ✅ 可行 | **不可行**（本来也只标云端可选） |
| Qwen3-4B QLoRA 训练 | ~9–11G ✅ 主线 | **超预算**，必须按下方降级 |
| Qwen3-14B QLoRA 训练 | 16G+ ❌ | ❌ 依旧不可行 |

**微调降级预案**（P14 执行前需再确认）：`seq 1024→512`、`batch_size 1`、
`gradient_accumulation 16`、开 gradient checkpointing、LoRA `r=8/alpha=16`；
若仍 OOM，退到 **Qwen3-1.7B** 跑通方法论。
> 本次已验证 0.5B 模型在 4bit + LoRA 下完整训练步峰值仅 **1.29 GB**，
> 说明小模型链路完全跑得通；4B 的实际峰值待 P14 实测。

### 3.2 `milvus-lite` 在 Windows 原生**可用**（推翻了文档前提）

`AGENTS.md` 铁律第 4 条与 `docs/00-setup-windows.md` 第 0 节都断言
「milvus-lite 在 Windows 原生装不上」。实测**不成立**：

- `pip install -r requirements.txt` 整份成功（wheel 标签是 `py3-none-any`，pip 不会拒绝）
- `python scripts/check_env.py`（默认后端）输出：
  `✅ 向量库 · Milvus Lite  建库/写入/检索全部正常，命中 id=1`
- 主线要用的 **HNSW 索引也建得起来**：`M=16, efConstruction=200`，检索 `ef=64` 正常
- 正确性验证：500 条归一化向量、20 次自身检索，**20/20 命中且余弦相似度 = 1.000000**

> 也就是说 `configs/base.yaml` 里的 `dense.backend: milvus_lite` **不需要**改成 faiss。
> 但 P3 仍建议按 `plan.md` 的规划先用 `faiss IndexFlatIP` 做精确检索当真值上界，
> 再对比 HNSW 近似检索的差距——这是实验设计需要，不是因为 milvus 装不上。

### 3.3 HuggingFace 直连不通，必须走镜像

```
https://huggingface.co      ->  curl: (28) Connection timed out after 20002 ms
https://hf-mirror.com       ->  200   time=0.75s   ✅
https://www.modelscope.cn   ->  302   time=0.71s   ✅
```

**所有 HF 模型下载都要挂镜像**，否则会卡死在下载上：

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
```

HF 缓存默认落在 `C:\Users\28620\.cache\huggingface`（当前已占 953 MB）。
**注意**：C: 只剩 57.6 GB，而后续要下 BGE-M3（2.2G）+ bge-reranker-v2-m3（2.2G）
+ Qwen3-4B（约 8G），建议改 `HF_HOME` 到 D 盘，否则 C 盘会紧张。

---

## 4. 微调链路验证（`scripts/check_finetune_stack.py`）

用 Qwen2.5-0.5B-Instruct 把 QLoRA 全链路跑了一遍，**6 步全绿**：

| 步骤 | 结果 |
|---|---|
| 1. 版本与 CUDA 可用性 | torch 2.6.0+cu126 \| transformers 5.19.0 \| RTX 4060 Laptop \| 8.0 GB ✅ |
| 2. bitsandbytes 兼容性 | 0.50.2，functional 可导入 ✅ |
| 3. 4bit 量化配置 | BitsAndBytesConfig(NF4 + double quant + bf16) ✅ |
| 4. 模型 4bit 加载 + 前向 | 成功，峰值显存 **0.43 GB** ✅ |
| 5. LoRA 挂载 | 可训练参数 **1,081,344 / 316,200,832 = 0.342%** ✅ |
| 6. 完整训练步（前向+反向） | 成功，**loss=5.8672**，峰值显存 **1.29 GB** ✅ |

`pip check`：`No broken requirements found.`（退出码 0）

---

## 5. 依赖完整清单（132 个，`pip freeze` 原样）

```
absl-py==2.5.0                   accelerate==1.15.0               aiohappyeyeballs==2.7.1
aiohttp==3.14.4                  aiosignal==1.4.0                 annotated-doc==0.0.5
annotated-types==0.8.0           anyio==4.15.1                    attrs==26.1.0
bitsandbytes==0.50.2             cachetools==7.2.1                certifi==2026.7.22
charset-normalizer==3.5.2        click==8.5.0                     cloudpickle==3.1.2
colorama==0.4.6                  datasets==5.1.0                  dill==0.4.1
distro==1.9.0                    faiss-cpu==1.15.1                fastapi==0.142.4
filelock==3.32.3                 FlagEmbedding==1.4.2             frozenlist==1.8.0
fsspec==2026.7.0                 grpcio==1.84.0                   h11==0.16.0
hf-xet==1.7.0                    httpcore==1.0.9                  httpcore2==2.13.1
httpx==0.28.1                    httpx2==2.13.1                   httpx-sse==0.4.3
huggingface_hub==1.33.0          idna==3.20                       iniconfig==2.3.1
ir_datasets==0.6.3               jieba==0.42.1                    Jinja2==3.1.6
jiter==0.17.0                    joblib==1.6.0                    jsonpatch==1.33
jsonpointer==3.2.0               langchain==1.4.3                 langchain-classic==1.0.8
langchain-community==0.4.2       langchain-core==1.6.7            langchain-openai==1.6.7
langchain-protocol==0.0.19       langchain-text-splitters==1.1.3  langgraph==1.2.14
langgraph-checkpoint==4.2.0      langgraph-prebuilt==1.1.0        langgraph-sdk==0.4.6
langsmith==0.14.4                lxml==6.1.3                      lz4==4.4.5
Markdown==3.11                   markdown-it-py==4.2.0            MarkupSafe==3.0.3
mdurl==0.1.2                     milvus-lite==3.2.1               mpmath==1.3.0
multidict==6.9.1                 multiprocess==0.70.19            narwhals==2.26.0
networkx==3.6.1                  numpy==2.4.6                     openai==3.26.0
opentelemetry-api==1.45.1        orjson==3.13.0                   ormsgpack==1.12.2
packaging==26.3                  pandas==3.0.6                    peft==0.21.2
pillow==12.3.0                   pluggy==1.6.0                    propcache==0.5.4
protobuf==7.36.2                 psutil==7.2.2                    pyarrow==25.0.1
pydantic==2.13.5                 pydantic_core==2.46.5            pydantic-settings==2.15.0
Pygments==2.21.0                 pymilvus==3.0.2                  pytest==9.1.1
python-dateutil==2.9.0.post0     python-dotenv==1.2.4             PyYAML==6.0.3
rank-bm25==0.2.2                 redis==8.1.0                     regex==2026.9.29
requests==2.34.2                 requests-toolbelt==1.0.0         rich==15.0.0
safetensors==0.8.0               scikit-learn==1.9.1              scipy==1.17.1
sentencepiece==0.2.2             sentence-transformers==6.1.0     shellingham==1.5.4
six==1.17.0                      sniffio==1.3.1                   SQLAlchemy==2.1.4
sqlglot==30.21.0                 starlette==1.7.0                 sympy==1.13.1
tenacity==9.2.1                  tensorboard==2.21.0              tensorboard-data-server==0.7.2
threadpoolctl==3.7.0             tiktoken==0.14.0                 tokenizers==0.23.2
torch==2.6.0+cu126               torchvision==0.21.0+cu126        tqdm==4.70.1
transformers==5.19.0             trl==1.14.2                      truststore==0.10.4
typer==0.27.3                    typing_extensions==4.16.0        typing-inspection==0.4.4
tzdata==2026.5                   urllib3==2.8.0                   uuid_utils==0.17.1
uvicorn==0.54.0                  websockets==16.1.1               Werkzeug==3.1.9
xxhash==4.0.1                    yarl==1.25.1                     zstandard==0.25.0
```

> ⚠️ 注意这些包普遍是 2026 年的**大版本**（transformers 5.x、pandas 3.x、langchain 1.x、
> datasets 5.x、trl 1.x）。`requirements.txt` 只写了下限，pip 就抓了最新。
> P3 之后的代码若用到大版本间有破坏性变更的 API，需以**本机实际装的版本**为准写，不要照抄旧文档。

---

## 6. 实际执行的命令与耗时

| 步骤 | 命令 | 耗时 | 结果 |
|---|---|---|---|
| 装 Miniconda | 清华镜像下载 + `/S` 静默安装 | 25.9s + 50.6s | ✅ 校验：Authenticode 签名 `Anaconda, Inc.` 有效，SHA256 与官方列表页一致 |
| 建环境 | `conda create -n ec python=3.11 -y --override-channels -c <清华源>` | 31.8s | ✅ Python 3.11.17 |
| 装 torch（首次） | `pip install torch==2.4.1 torchvision==0.19.1 --index-url .../cu121` | 11.0 min | ⚠️ 装上了但**与 transformers 5.19 不兼容** |
| 装主依赖 | `pip install -r requirements.txt -i <清华 PyPI>` | 4.2 min | ✅ 132 包中的主体 |
| 换 torch | `pip uninstall -y torch torchvision` + `pip cache purge` + `pip install torch==2.6.0 torchvision --index-url .../cu126` | 6.4 min | ✅ 缓存清掉 2777.4 MB |
| 装微调依赖 | `pip install -r requirements-finetune.txt` | 数秒 | ✅ 仅补装 tensorboard 及依赖 |
| 依赖自检 | `pip check` | 秒 | ✅ No broken requirements found |
| 微调链路验证 | `python scripts/check_finetune_stack.py` | 1.8 min | ✅ 6/6 通过 |
| 环境自检 | `python scripts/check_env.py` | 秒 | ✅ 关键项全绿（LLM 项除外，见下） |

### 项目自带自检块（额外验证，非 P0 要求）

```
eval/metrics.py     -> HitRate@1 = 0.5 ✅  HitRate@5 = 1.0 ✅  MRR@10 = 0.75 ✅  自检通过
src/retrieve/bm25.py -> 4 条中文 query 全部命中正确 chunk（jieba 分词正常）
```

---

## 7. 未测 / 不确定的地方（如实列出）

1. **未测**：Qwen3-4B 在 8G 卡上的 QLoRA 实际显存峰值。预算表里 9–11G 是 16G 卡的数，
   本文第 3.1 节的降级方案属于**推断**，需 P14 实测确认。
2. **未启动 LLM 服务**：`check_env.py` 的「LLM 服务」一项报 ⚠️
   （`127.0.0.1:11434` 连接被拒），因为 **ollama 尚未安装**。
   该项在脚本里是 `warn_only`，不影响「关键项全部通过」的结论，但 P6 之后用到 LLM 时必须补上。
3. **未装 Redis**：Windows 原生需 Memurai 或 WSL。`configs/base.yaml` 的
   `memory.short_term.backend` 默认是 `memory`（进程内字典），暂不需要；P7 做长期记忆时再处理。
4. **WSL 未安装任何发行版**：最终全部在 Windows 原生完成，未使用 WSL。
5. **`torchvision` 本项目和当前用不到**，是按 `docs/00-setup-windows.md` 的命令一并装的；
   若后续确认无用可以卸载。
6. **pip 的 `pip check` 查不出 transformers 对 torch 的版本要求**——因为它在元数据里写在
   `extra` 标签下，只有运行到才会暴露。这就是本次冲突第一次没被 `pip check` 拦住的**原因**，
   也说明为什么 `check_finetune_stack.py` 这种「真跑一遍」的脚本是必要的。

---

## 8. 复现本环境的完整命令

```powershell
# 1) Miniconda（清华镜像，装到标准位置）
curl.exe -L -o Miniconda3.exe https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
Start-Process .\Miniconda3.exe -ArgumentList '/InstallationType=JustMe','/RegisterPython=1','/S','/D=C:\Users\28620\miniconda3' -Wait

# 2) Python 3.11 环境
C:\Users\28620\miniconda3\Scripts\conda.exe create -n ec python=3.11 -y `
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

# 3) ★ 必须先装 CUDA 版 torch。漏掉 --index-url 会装成 CPU 版，后面全白干。
$py = 'C:\Users\28620\miniconda3\envs\ec\python.exe'
& $py -m pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 4) 主依赖 + 微调依赖
& $py -m pip install -r requirements.txt          -i https://pypi.tuna.tsinghua.edu.cn/simple
& $py -m pip install -r requirements-finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5) 验证
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& $py scripts/check_env.py
$env:HF_ENDPOINT = 'https://hf-mirror.com'        # ★ HF 直连不通，必须挂镜像
& $py scripts/check_finetune_stack.py
```

> 激活环境：`conda activate ec`（若 conda 未初始化到当前 shell，用
> `& C:\Users\28620\miniconda3\Shell\condabin\conda-hook.ps1`）。
