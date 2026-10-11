# 学习日志：把这个项目里每个"为什么"补回来

> 为什么有这份文档：项目前期我（AI）在自问自答地干活——改配置、修脚本、装依赖、
> 跑验证——然后把「已完成」的结论丢给用户。结果用户看不到任何决策过程，学不到东西。
> 这份文档把那些**隐式决策**全部摊开：当时观察到什么、我怎么判断、为什么这么定、
> 参数含义、以及**怎么自己验证**。
>
> 读法建议：每条都按「观察 → 判断 → 原理 → 参数 → 验证」五段组织。
> 带 ★ 的是我认为最值得记的点。

---

## 日志 1 ★ 为什么是 torch 2.6.0（不是 2.4.1，也不是最新的 2.14.1）

### 观察

在 Windows 机上装依赖时报冲突：`transformers 5.19` 与 `torch 2.4.1` 不兼容。

### 判断过程

**第一步：不要凭记忆猜版本约束，去查元数据。** 版本地狱问题的正解不是背版本号，
而是查两个权威来源：

```bash
# 来源 1：PyPI 元数据里的 requires_dist —— 看真实约束
curl -s https://pypi.org/pypi/transformers/json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for r in (d['info'].get('requires_dist') or []):
    if 'torch' in r.lower(): print(r)
"
# 输出: torch>=2.5; extra == \"torch\"
```

→ 确认 `transformers 5.19` 硬性要求 `torch>=2.5`。所以 2.4.1 真的不满足。

**第二步：查"官方轮子索引"看某个版本是否真的存在。** 光知道"需要 >=2.5"没用，
还得知道**哪个 CUDA 通道能装到 2.5+**：

```bash
for cu in cu121 cu124 cu126 cu128 cu130; do
  echo -n "$cu : "
  curl -s "https://download.pytorch.org/whl/$cu/torch/" \
    | grep -oE "torch-[0-9]+\.[0-9]+\.[0-9]+%2B$cu-cp311" \
    | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | sort -uV | tr '\n' ' '
  echo
done
```

实测结果（这是关键数据，值得记住）：

| 通道 | 最高 torch 版本 |
|---|---|
| `cu121` | **2.5.1 就停** |
| `cu124` | **2.6.0 就停** |
| `cu126` | 2.6.0 → 2.14.1 |
| `cu130` | 2.9.0 → 2.14.1 |

→ **这就是"为什么升 torch 必须同时换 CUDA 通道"**：cu121 上根本不存在 2.5 以上的版本。

**第三步：查完整约束链，避免顾此失彼。** 微调涉及四个互相约束的库：

```bash
for pkg in torch transformers bitsandbytes peft trl accelerate; do
  curl -s "https://pypi.org/pypi/$pkg/json" | python3 -c "
import json,sys; i=json.load(sys.stdin)['info']
print(i['name'], i['version'], [r for r in (i.get('requires_dist') or [])
      if any(k in r.lower() for k in ('torch','transformers','accelerate'))][:4])
"
done
```

得到的约束网：

```
torch 2.6.0+cu126
  ├─ transformers 5.19.0  需要 torch>=2.5          ✅
  ├─ bitsandbytes 0.50.2  需要 torch<3,>=2.4       ✅
  ├─ peft 0.21.2          需要 torch>=1.13         ✅
  ├─ trl 1.14.2           需要 transformers>=4.56.2 ✅
  └─ accelerate 1.15.0    需要 torch>=2.0          ✅
```

### 原理 ★

**为什么会有"通道"这个东西？**
PyTorch 的 CUDA 支持是**编译期绑定**的：wheel 里打包了针对特定 CUDA 版本的
kernel 二进制。`cu121` 的 wheel 是拿 CUDA 12.1 编译的，装不上 CUDA 12.6 的驱动生态
（反过来说驱动向后兼容，所以能用更高驱动跑更低 cuda 版本的 wheel）。

**为什么老通道会停更？**
新 torch 版本发布时，PyTorch 团队只为**当前主流 CUDA 版本**构建 wheel。
`cu121` 到 torch 2.5.1 就停止更新，是因为 2.6 起他们不再为 CUDA 12.1 出包了。

**为什么 `bitsandbytes` 有个 `>=2.4` 下限？**
因为 torch 2.6 移除了私有 API `torch.utils._pytree._register_pytree_node`，
老版 bitsandbytes 在这里会崩。`0.50.2` 起已适配，所以下限被抬到 2.4。

### 参数含义

| 参数 | 含义 | 怎么定 |
|---|---|---|
| `--index-url .../whl/cu126` | 指定从哪个通道下载 | 由你机器的驱动版本决定（`nvidia-smi` 右上角） |
| `torch==2.6.0` | 定死版本 | 满足所有约束的**最低**版本 —— 选它而非 2.14.1 是**故意保守** |

**为什么故意不选最新？** 2.14.1 能用，但太新，`bitsandbytes`/`peft`
尚未在它上面被广泛验证。**"够新且被生态测过"比"最新"重要**——这是工程判断，不是保守。

### 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.6.0+cu126 True
```

⚠️ **注意：PyPI 上默认的 `torch` wheel 是 CPU-only 的。**
不加 `--index-url` 就会装成没 CUDA 的版本——这是最常见的隐性错误，
而且它不会报错，只会让 `cuda.is_available()` 悄悄返回 False。

---

## 日志 2 ★★ 为什么 Mac 上做不了 QLoRA（bitsandbytes 是 CUDA-only）

### 观察

换到 Apple M1 后，`import bitsandbytes` 直接失败。

### 判断过程

先确认不是"没装"而是"装不了"：

```bash
pip install bitsandbytes
# → 装上了，但 import 失败
```

`bitsandbytes` 的 4bit/8bit 量化是**自定义 CUDA kernel** 实现的
（`NF4`/`FP4` 的反量化、`LLM.int8()` 的混合精度矩阵乘）。
这些 kernel 用 CUDA C++ 写、编译成 `.so`/`.pyd`，**没有 MPS 版本**。

### 原理 ★★

必须先分清两个概念，很多人在这里混淆：

| | QLoRA | LoRA |
|---|---|---|
| 基座权重 | **4bit 量化**存放 | fp16/bf16 原精度 |
| 可训练部分 | LoRA 适配器（fp16） | LoRA 适配器（fp16） |
| 内存收益 | 权重从 2 字节/参数 → 0.5 字节/参数 | 无 |
| 依赖 | **必须** bitsandbytes | 只需 peft |

所以：

```
没有 bitsandbytes  ⇒  没有 4bit 量化  ⇒  没有 QLoRA
                                       ⇒  只剩 LoRA（非量化）
```

**为什么量化能省内存？** 一个 14B 模型：

```
fp16:  14e9 × 2 字节 = 28 GB   ← 装不进任何消费级显卡
4bit:  14e9 × 0.5 字节 = 7 GB  ← 8GB 卡勉强能放
```

这就是 QLoRA 论文的核心贡献——**让 65B 模型能在单张 48GB 卡上微调**。

### 参数含义

Mac 上"量化相关"的参数全部失效，不用去调它们：

| 参数 | 在 CUDA 上 | 在 Mac 上 |
|---|---|---|
| `load_in_4bit=True` | 省 4 倍权重内存 | ❌ 不可用 |
| `bnb_4bit_quant_type="nf4"` | 正态分布 4bit，比 fp4 精度高 | ❌ 不可用 |
| `bnb_4bit_use_double_quant=True` | 对量化常数再量化，再省约 0.4 bit/参数 | ❌ 不可用 |
| `bnb_4bit_compute_dtype=bfloat16` | 反量化后用 bf16 算，稳 | ❌ 不可用 |

**Mac 上仍然有效的参数**（见日志 4）：`r`、`lora_alpha`、`lora_dropout`、
`target_modules`、`gradient_checkpointing`。

### 验证

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.device import detect
i = detect()
print('backend:', i.backend, '| 支持 4bit:', i.supports_4bit)
"
# Mac 期望: backend: mps | 支持 4bit: False
```

---

## 日志 3 ★ 内存怎么量才对（MPS 没有"真峰值"统计）

### 观察

第一次在 M1 上跑验证，脚本报 `峰值占用 0.95 GB`。
但我在 Windows 上（CUDA）报的是 `峰值 1.29 GB`。**这两个数字能直接比吗？**

### 判断过程

我去查了 API：

| 后端 | 可用 API | 语义 |
|---|---|---|
| CUDA | `torch.cuda.max_memory_allocated()` | **历史峰值**（从上次 reset 起） |
| CUDA | `torch.cuda.reset_peak_memory_stats()` | 重置峰值统计 |
| MPS | `torch.mps.current_allocated_memory()` | **当前瞬时值** |
| MPS | ~~`max_memory_allocated`~~ | **不存在** |

→ **不能直接比。** 我把瞬时值标成"峰值"，是在制造一个假的可比性。

而且更根本的问题：**MPS 的 `current_allocated_memory` 只统计 MPS 分配器手上的张量**，
不含 Python 堆、共享库、以及 MPS 已释放但未归还的缓存。
在 16GB **统一内存**的机器上，我真正该盯的是**进程 RSS**。

### 原理 ★

**统一内存（Apple Silicon）vs 专用显存（独显）的区别：**

```
独显（RTX 4060 8GB）:
  [专用 VRAM 8GB]  ← 只有 GPU 能用，CPU 碰不到
  [系统 RAM 16GB]  ← 只有 CPU 用

M1（统一内存 16GB）:
  [统一内存 16GB]  ← CPU 和 GPU 共用同一块物理内存
                     代价：模型占了，系统就少了
```

所以在 M1 上**没有"显存够不够"这回事**，只有"整机内存够不够"。
16GB 统一内存的真实可用余量 ≈ 16 − 系统(3~5) − 其他进程(1~3) = **6~9GB**。

### 参数含义

| 指标 | 用途 | 注意 |
|---|---|---|
| `max_memory_allocated` | CUDA 上判断"模型+激活峰值" | 只在 CUDA 有 |
| `current_allocated_memory` | MPS 上粗略看当前占用 | **不是峰值**，会低估 |
| 进程 RSS | **统一内存机器上最诚实的指标** | 含 Python 堆与未归还缓存 |
| `vm_stat` | 系统级内存压力 | macOS 用，看 `Pages free` 与 swap |

### 验证

```bash
# 看系统整体内存压力（macOS）
vm_stat | head -5
# Pages free × 16384 字节 = 实际空闲

# 看单个进程的峰值 RSS
/usr/bin/time -l .venv/bin/python scripts/check_finetune_stack.py --load-only
# 输出里的 "maximum resident set size" 就是峰值 RSS
```

**我据此改了脚本**：`peak_mem_gb()` 现在返回 `(值, 口径)` 元组，
强制调用方显式处理"这是峰值还是瞬时值"，而不是让一个含糊的"峰值"标签误导对比。

---

## 日志 4 ★★★ LoRA 可训练参数量怎么算（实测与推算完全对上）

### 观察

实测：`可训练参数 1,081,344 / 495,114,112 = 0.218%`（0.5B）
实测：`可训练参数 2,179,072 / 1,545,893,376 = 0.141%`（1.5B）

而且**0.5B 的这个数字在 Windows（4bit）和 Mac（非量化）上完全一样**。为什么？

### 判断过程

**关键推理**：LoRA 的参数量只取决于三件事：

```
可训练参数 = rank × (输入维度 + 输出维度) × 目标模块数
```

**与基座是否量化无关**——量化改的是权重怎么存，不改 LoRA 适配器的结构。
所以两个平台数字一致是**应该的**，这反过来验证了 LoRA 实现没问题。

然后我照真实模型结构推算（**不能假设所有投影都是方阵**）：

```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
# hidden_size = 896, num_hidden_layers = 24
# num_attention_heads = 14, num_key_value_heads = 2   ← 注意这里！
# head_dim = 896 / 14 = 64
# KV 维度 = num_key_value_heads × head_dim = 2 × 64 = 128   ← 不是 896！
```

### 原理 ★★★ GQA（分组查询注意力）导致 k/v 投影不是方阵

Qwen2.5 用的是 **GQA**：14 个 query 头，但只有 **2** 个 key/value 头
（每 7 个 query 头共享一组 KV）。所以：

```
q 投影: 896 → 896   LoRA 参数 = r×(896+896) = 14,336
k 投影: 896 → 128   LoRA 参数 = r×(896+128) =  8,192   ← 输出维度小得多
v 投影: 128 → 896   LoRA 参数 = r×(128+896) =  8,192
o 投影: 896 → 896   LoRA 参数 = r×(896+896) = 14,336
─────────────────────────────────────────────────
每层合计              45,056
× 24 层           = 1,081,344   ✅ 与实测完全一致
```

**如果忽略 GQA**（假设四个投影都是 896×896 方阵）：

```
4 × r×(896+896) = 57,344/层 × 24 = 1,376,256   ❌ 差 27%
```

→ **教训：公式要照真实结构算，不能套简化假设。**

**顺带解释 GQA 为什么存在**：推理时 KV cache 是显存瓶颈。
标准 MHA 下 14 个头全存，GQA 只存 2 个头的 KV，**KV cache 直接省 7 倍**，
而质量损失很小。这是 2023 年后主流模型的标配。

### 参数含义（LoRA 的核心四个）

| 参数 | 含义 | 取值 | 为什么 |
|---|---|---|---|
| `r`（rank） | 低秩矩阵的秩，决定**可训练参数量** | 8 | 参数量 ∝ r。r=8 是常见起点；任务复杂可到 16/32 |
| `lora_alpha` | 缩放系数，实际缩放 = `alpha/r` | 16 | `16/8 = 2`，即 LoRA 更新被放大 2 倍。经验：`alpha = 2r` |
| `lora_dropout` | 适配器上的 dropout | 0.05 | 小数据集防过拟合 |
| `target_modules` | 给哪些层挂适配器 | `[q,k,v,o]` | 只挂注意力层最省；挂 MLP（`gate/up/down_proj`）效果更好但参数多 |

**关于 `lora_alpha` 的常见困惑**：为什么不直接用学习率控制强度？
因为 LoRA 的更新是 `ΔW = (alpha/r) · B·A`，这个 `alpha/r` 缩放**与学习率解耦**——
换 `r` 时改 `alpha` 可以保持有效更新幅度不变，便于做消融对比。

### 验证

```bash
# 推算 vs 实测对照
.venv/bin/python -c "
from transformers import AutoConfig
import sys; sys.path.insert(0,'.')
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
d, L, r = c.hidden_size, c.num_hidden_layers, 8
kv = c.num_key_value_heads * (d // c.num_attention_heads)
per = r*(d+d) + r*(d+kv) + r*(kv+d) + r*(d+d)
print('推算 LoRA 参数:', f'{per*L:,}')
"
# 期望: 1,081,344  （与脚本实测值一致）
```

---

## 日志 5 为什么用项目内 `.venv` 而不是 conda 环境

### 观察

`conda create -n ec python=3.11` 报错：

```
NoWritableEnvsDirError: No writeable envs directories configured.
  - /Users/Zhuanz/.conda/envs
  - /opt/anaconda3/envs
```

### 判断过程

两个主流选择：

| 方案 | 优点 | 缺点 |
|---|---|---|
| conda env | 能装非 Python 依赖（CUDA 工具链等） | 写 `envs` 目录需要权限；环境散落在几个 GB 的 anaconda 目录里 |
| 项目内 `.venv` | 完全自包含，删目录即卸载；不需要额外权限 | 只能装纯 Python 包 |

对本项目：**依赖全是纯 Python 包**（torch / transformers / peft 都是 wheel），
不需要 conda 的二进制依赖管理能力。→ 选 `.venv`。

**顺便发现**：本机已有的 `lora` conda 环境（torch 2.6.0 + transformers 4.57.3）
当时不可写，所以也无法复用。

### 原理

**为什么 Python 版本也有讲究？** 当时用 3.13.2 而非 homebrew 默认的 3.14.6：

```
3.14 太新 → 部分包的 wheel 还没出，会退到源码编译（慢且易失败）
3.11 最稳 → 但本机没装
3.12       → 有 anaconda base，但当时不可写
3.13.2     → 折中：够新且有 arm64 wheel
```

**可复现性的关键**：`pip freeze > requirements.lock.txt`。
不锁版本的话，半年后别人装出来的环境和你不一样，"我跑通了"就无法复核。

### 验证

```bash
.venv/bin/python -V                      # 3.13.2
.venv/bin/python -m pip list | wc -l     # 装了多少个包
.venv/bin/python -m pip freeze > requirements.lock.txt   # 锁版本
```

---

## 日志 6 诚实记录：我的三次判断失误

写这一条是因为**错误比正确更有教育意义**，而且我不想让文档显得我一路都对。

### 失误 1：凭记忆说"torch 最新是 2.x 左右"

**错在哪**：我的训练知识有截止时间，而 torch 已经到 2.14.1。
**怎么发现**：查 PyPI 元数据，`latest: 2.14.1`。而那个 Windows agent 报的版本号全是真的。
**教训**：**版本号这种易变事实必须查，不能凭记忆。**

### 失误 2：以为"torch 这么大，沙箱肯定限速装不了"

**错在哪**：依据只有"pip 卡了 4 分钟"。而真实原因只是**下载慢**（约 200MB / 10 分钟）。
**后果**：我把这个错误推断写进了计划，还让你去终端手动装。
**后来更打脸**：权限策略又放宽到全盘可写，那条限制整个作废。
**教训** ★：**样本不足时的推断要标注"待验证"，绝不能写进计划当约束。**
这跟另一个 agent 把文件清成 0 字节是**同一类错误**——对未验证的中间状态下过早结论。

### 失误 3：把 MPS 的瞬时内存占用标成"峰值"

**错在哪**：为了输出好看，用了和 CUDA 一样的措辞，制造了虚假的可比性。
**怎么发现**：自查 API 时发现 MPS 根本没有 `max_memory_allocated`。
**教训**：**指标的口径必须和数值一起报。** 现在脚本返回 `(值, 口径)` 元组，
强制调用方处理这个问题。

---

## 日志 7 怎么自己复核 AI 说的话（元技能）

这一条可能比上面所有技术细节都有用。

**本项目里验证一个说法有三种手段，按可靠性排序：**

| 手段 | 例子 | 可靠性 |
|---|---|---|
| **查权威元数据** | PyPI `requires_dist`、官方轮子索引目录 | 最高——机器生成，无法粉饰 |
| **写代码从第一性原理推算** | LoRA 参数量 = f(rank, 结构) | 高——对上了就是真对上了 |
| **实际跑一遍看输出** | `check_finetune_stack.py` | 高，但受测试设计影响 |
| 我的口头说明 | "应该可以" | ❌ 不算证据 |

**给你一个可复用的检查清单**：任何技术结论到手时问三个问题：

1. **这个数字/说法来自哪里？** （记忆？实测？文档？）
2. **我能用什么独立手段复现它？** （命令 / 推算 / 查元数据）
3. **如果它是错的，会有什么症状？** （比如装成 CPU 版 → `cuda.is_available()` 返回 False）

第 3 条最重要：**能预判"错了会怎样"，才知道该验什么。**

---

## 待办：这份日志还欠的条目

以下是已完成但还没展开写的决策，后续补上：

- [ ] 为什么 `chunk_size=512 / overlap=80`（等 D3 实测后再写，要有对照数据）
- [ ] RRF 的 `k=60` 到底在平滑什么
- [ ] HNSW 的 `M` / `efConstruction` / `ef` 三个参数分别控制什么
- [ ] BM25 的 `k1` / `b` 怎么影响排序（要有实际文档的例子）
- [ ] Cross-Encoder vs Bi-Encoder 的结构差异
- [ ] 为什么测评要先有 gold 标注才能算 MRR
