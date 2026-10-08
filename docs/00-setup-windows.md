# 环境搭建（RTX 4060 Laptop **8GB** / Windows / Python 从未装过）

> 目标：一条命令验证「GPU 可用 + 模型能跑 + 向量库能查」，然后才动业务代码。
> 预计耗时：1.5–2.5 小时（大部分时间在下模型）。
>
> ⚠️ **本文档已按 2026-10 的实机结果订正过三处错误假设**，见下方「实机订正记录」。

---

## 实机订正记录（实测，优先于本文档其余推测）

原始规划基于三个假设，全部被实机推翻：

| # | 原假设 | 实测结果 | 影响 |
|---|---|---|---|
| 1 | 显存 16GB | **8188 MiB（8 GB）** —— RTX 4060 **Laptop 只有 8GB 版本**；任务管理器显示的 "16GB" 是 8G 专用 + 8G 共享系统内存，后者 CUDA 用不了 | 第 5 节显存预算表整张作废，已重写 |
| 2 | milvus-lite 在 Windows 原生不可用 | **可用**。建集合/写入/检索/HNSW 索引全部正常；归一化向量 20/20 命中、余弦相似度 1.000000 | 不必降级到 FAISS，也不必为此装 WSL2 |
| 3 | HuggingFace 可直连 | **直连 20s 超时**，`hf-mirror.com` 0.75s 正常 | 所有 HF 操作必须挂 `HF_ENDPOINT` |

已确认可用的环境（Windows 那台）：

```
Python 3.11.17 | torch 2.6.0+cu126 | transformers 5.19.0 | bitsandbytes 0.50.2
NVIDIA GeForce RTX 4060 Laptop GPU | 8188 MiB | 驱动 566.24 | CUDA 12.6
conda: C:\Users\28620\miniconda3\envs\ec    项目/索引: D 盘（剩余 328 GB）
```

---

## 0. 先做选择：WSL2 还是 Windows 原生

**推荐 WSL2 + Ubuntu 22.04。** 理由（都有具体坑，不是偏好问题）：

| 组件 | Windows 原生 | WSL2 |
|---|---|---|
| PyTorch + CUDA | 可以（已实测通过） | 可以（Windows 驱动直通，**不要在 WSL 里装显卡驱动**） |
| Milvus Lite | **可以**（已实测：建库/写入/检索/HNSW 全通过） | 可用 |
| Redis | 要装 Memurai/WSL | `apt install redis-server` 一条命令 |
| LLaMA-Factory 微调 | 能跑但偶发路径/编码问题 | 稳 |
| 显存访问 | 完整 | 完整 |
| 训练中断 | 关窗即断 | 同样（建议 `tmux`） |

**但既然 Windows 原生已实测跑通整条链路，不必为了 Milvus 去装 WSL2。**
WSL2 现在只剩「Redis 方便」和「LLaMA-Factory 更稳」两个好处，
按需决定即可——不要为了环境整洁搭进去一天。

本项目的代码对两者**都兼容**：`dense.backend` 配置项支持 `milvus_lite | milvus | faiss`。

---

## 0.5 Windows 原生必做的环境变量（否则会卡死）

```powershell
# 永久写入用户环境变量，别每次手敲
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")

# HF 缓存默认落 C:，而 C: 只剩 57.6 GB。
# 后续 BGE-M3 + reranker + Qwen3-4B 约 13 GB，建议挪到 D 盘（**改前先确认**）
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf-cache", "User")
```

⚠️ **HuggingFace 直连实测 20 秒超时**。不设 `HF_ENDPOINT` 的话，
任何 `from_pretrained` / `hf_hub_download` 都会卡死在下载上，
而且报错信息往往看不出是网络问题。

---

## 1. 装 Miniconda（别用系统 Python）

```bash
# WSL2/Ubuntu
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
~/miniconda/bin/conda init bash && exec bash

conda create -n ec python=3.11 -y
conda activate ec
```

为什么是 3.11 而不是 3.12/3.13：`milvus-lite`、`bitsandbytes`、部分 `vllm`
对 3.12 的 wheel 支持仍不齐，3.11 是目前最不容易踩坑的版本。

---

## 2. 装 CUDA 版 PyTorch（最容易装错的一步）

```bash
# 先确认 CUDA 版本上限，Windows 侧执行 nvidia-smi 看右上角 "CUDA Version"
nvidia-smi
```

### 2.1 先选 CUDA 通道（这决定你能装到哪个 torch）

官方 index 的**真实版本上限**（实测，别凭记忆）：

| 通道 | 可用 torch 版本 | 驱动要求 |
|---|---|---|
| `cu121` | 2.1.0 → **2.5.1 就停** | ≥ 525 |
| `cu124` | 2.4.0 → **2.6.0 就停** | ≥ 550 |
| `cu126` | **2.6.0 → 2.14.1** | ≥ 560 |
| `cu130` | 2.9.0 → 2.14.1 | ≥ 580 |

⚠️ **关键**：`transformers 5.19` 要求 `torch>=2.5`。
如果你想要新版 transformers，就只能走 `cu126` 及以上——
因为 **`cu121` 通道最高只到 torch 2.5.1，`cu124` 最高只到 2.6.0**。

### 2.2 推荐钉版（QLoRA 全链路已验证互相兼容）

```bash
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 其余按 requirements.txt，但注意下面两条硬约束：
#   bitsandbytes>=0.50.2  ← 旧版（0.43 那批）在 torch 2.6+ 上会因 pytree 改动而崩
#   transformers>=5.19    ← 与 torch>=2.5 配对
```

这套组合的约束链，逐条核对过 PyPI 元数据：

```
torch 2.6.0+cu126
  ├─ transformers 5.19.0  需要 torch>=2.5        ✅
  ├─ bitsandbytes 0.50.2  需要 torch<3,>=2.4     ✅
  ├─ peft 0.21.2          需要 torch>=1.13       ✅
  ├─ trl 1.14.2           需要 transformers>=4.56.2 ✅
  └─ accelerate 1.15.0    需要 torch>=2.0        ✅
```

**为什么不用最新的 torch 2.14.1**：能用，但太新，
bitsandbytes / peft 尚未在它上面被广泛验证。2.6.0 是"够新且已被生态测过"的甜点。

**降级备选**：如果驱动太老只能走 `cu121`，那就
`torch==2.4.1+cu121` + `transformers==4.46.*`（4.x 分支，无 torch>=2.5 要求）。
代价是 Qwen3 支持可能不够新，且要自己确认 Qwen3 架构被该版本认识。

### 2.3 验证（不过就停下来）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出类似：`2.6.0+cu126 True NVIDIA GeForce RTX 4060`

**排错**：
- `False` → 装成了 CPU 版。⚠️ **PyPI 上默认的 torch wheel 是 CPU-only 的**，
  必须带 `--index-url`，否则拿到的就是没 CUDA 的版本
- WSL 里 `False` → Windows 侧驱动太旧，去 NVIDIA 官网更新 **Windows 侧**驱动
- 显存数字对不上 → 以 `torch.cuda.get_device_properties(0).total_memory` 为准。
  ⚠️ 任务管理器显示 "16GB" 是 **8G 专用 + 8G 共享系统内存**，共享部分 CUDA 用不了。
  本项目实机为 **8188 MiB (8GB)**
- `bitsandbytes` 导入报 pytree 相关错 → bitsandbytes 版本太老，升到 0.50.2+

---

## 3. 项目依赖

```bash
pip install \
  "pymilvus>=2.4" milvus-lite \
  sentence-transformers FlagEmbedding \
  jieba PyYAML tqdm pandas \
  sqlglot openai rank-bm25 \
  langchain langchain-community langgraph \
  peft transformers datasets accelerate bitsandbytes \
  pytest rich

# Redis（短会话缓存；装不上不影响主线，会自动降级内存字典）
sudo apt install redis-server -y && sudo service redis-server start
```

`rank_bm25` 我们其实不用（自研了 `src/retrieve/bm25.py` 以便调 k1/b），
但留着方便做交叉验证：如果自己的实现和它差太多，说明分词或公式有问题。

---

## 4. 模型从哪下

**结论：国内一律走 ModelScope（魔搭），HuggingFace 直连会卡到怀疑人生。**

```bash
pip install modelscope
export HF_ENDPOINT=https://hf-mirror.com   # 万一必须用 HF 时的镜像

# BGE-M3 向量模型（约 2.2G，走 CPU 也行）
modelscope download --model BAAI/bge-m3 --local_dir /mnt/d/models/bge-m3
# 精排模型（约 2.2G，也可换成 base 版省显存）
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir /mnt/d/models/bge-reranker-v2-m3
```

大模型选型（**受 8GB 显存约束**）：

```bash
# 方案 A：Ollama 起服务（最省事，自带量化，Windows 原生可用）
ollama pull qwen3:4b        # ✅ 约 3G，8G 卡上最舒服，主力选择
ollama pull qwen3:8b        # ⚠️ 约 5.5G，余量小但能跑，可做模型大小对照
# ❌ 不要 pull qwen3:14b —— 4bit 约 9G > 8G，必 OOM

# 方案 B：vLLM（吞吐高、原生支持 LoRA 热加载，但 Windows 原生不支持，必须 WSL/Linux）
# ⚠️ vLLM 默认会预占大量显存，8G 卡上必须显式压 utilization，且多半仍不够
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --max-model-len 4096 --gpu-memory-utilization 0.75
```

`configs/base.yaml` 里 `llm.provider` 和 `llm.base_url` 按实际改，代码走 OpenAI 兼容协议，两种都能切。

---

## 5. 显存预算表（**8GB 卡上的真实账**，已按实测重写）

| 任务 | 配置 | 显存占用 | 8G 可行 |
|---|---|---|---|
| Qwen3-4B 推理 | 4bit (NF4) | ~3–3.5 G | ✅ **推荐主力** |
| Qwen3-8B 推理 | 4bit (NF4) | ~5.5–6 G | ⚠️ 余量小 |
| Qwen3-14B 推理 | 4bit + double quant | ~9.5 G | ❌ **必 OOM** |
| BGE-M3 embedding | fp16, batch 16 | ~1.8 G | ⚠️ 建议走 **CPU** |
| bge-reranker-v2-m3 | fp16, batch 8 | ~2.5 G | ⚠️ 换 `bge-reranker-base`，或与大模型串行 |
| **Qwen3-4B QLoRA 训练** | seq≤512, bs 1, ga 16, grad ckpt | ~6–8 G（**估**） | ⚠️ **边界，必须先实测** |
| Qwen3-1.7B QLoRA 训练 | 同上 | ~3–4 G（**估**） | ✅ 稳妥降级方案 |
| Qwen3-8B QLoRA 训练 | 同上 | 8 G+ | ❌ 基本不可行 |

**唯一实测锚点**：Qwen2.5-0.5B（4bit，同配置）单步训练峰值 **1.29 GB**。
表中的 4B/1.7B 数字**都是我按缩放推断的，不是实测** —— 必须由 P14 打表验证。

### 8G 显存下的三条硬规则

1. **大模型、embedding 模型、reranker 三者不要同时驻留。**
   8G 装不下两个以上。RAG 链路用 CPU 跑 embedding，
   或者处理完 embedding 就 `del model; torch.cuda.empty_cache()`。
2. **QLoRA 一切参数按最保守开**：`seq 512` → 不行降 384；`bs 1`；
   `ga 16` 补等效 batch；`gradient_checkpointing=True`；`double_quant=True`。
3. **启动训练前先把 Ollama/浏览器/其他占显存的进程关掉。**
   `nvidia-smi` 确认空闲显存再开训，否则 OOM 无法归因。

### 微调目标的调整

原计划是「Qwen3-4B QLoRA 跑通方法论」。8G 卡上这仍可尝试，但要有降级预案：

```
先试 Qwen3-4B QLoRA（seq 512）
  ├─ 成功 → 用它做主线，记录真实显存峰值
  └─ OOM → 退到 Qwen3-1.7B QLoRA（稳），
           方法论完全相同，面试讲"1.7B 验证方案，4B 是放大"
```

**不要为了 4B 硬调参数到 seq 128** —— 那样训练出的模型没有实际意义，
反而让"我验证了方法论"这个说法站不住。

---

## 6. 验收脚本

```bash
python scripts/check_env.py                        # Windows 原生直接用 milvus_lite
python scripts/check_env.py --skip-llm             # 还没起 Ollama 时
```

期望全绿：Python 版本 / torch.cuda / 依赖 / 向量库读写 / 磁盘。
「LLM 服务」项在没起 Ollama 时会是 ⚠️（该项是 `warn_only`，不影响结论）。

⚠️ 该脚本每次运行都会在 `data/milvus/` 下留下 `_env_check.db/` 测试目录
（它只 drop collection，不删库目录）。已改为用完即清理，见脚本内说明。
