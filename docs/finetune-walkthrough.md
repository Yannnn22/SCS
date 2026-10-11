# 微调实战：电商客服场景（一步步做，每步都讲为什么）

> 面向：想亲手做一遍 LoRA 微调的学习者。
> 硬件：Apple M1 / 16GB 统一内存 / 无 CUDA。
> 前置阅读：[`learning-log.md`](learning-log.md) 的日志 2（为什么没有 QLoRA）和日志 4（LoRA 参数量）。

---

## 0. 先明确：我们到底在训什么

**业务目标**：让模型能按电商客服的规范回答用户问题。

**技术目标**（这两个别混）：

| | 内容 |
|---|---|
| **能改变的** | 输出格式、话术风格、角色口吻、结构化字段（如意图标签、SQL 格式） |
| **不能改变的** | 模型的知识边界、幻觉。**LoRA 微调不会让模型"学会"新的业务规则** |

**为什么"不能改变知识"很重要**：很多人以为微调 = 灌输知识，
于是把几百条 FAQ 塞进训练集，期待模型背下来。**这必然失败**——
知识应该放 RAG（检索到的文档进 Prompt），微调只管"怎么说"。

所以本次微调的目标定成三件可量化的事：

1. **格式合规率**：输出是否符合要求的 JSON/话术结构
2. **意图识别准确率**：能否正确判断 `scene` / `need_tool`
3. **表述规范性**：是否用了客服口吻（"您"、不承诺做不到的事、不泄露内部规则）

---

## 1. 决策：选哪个基座

**这是整件事最关键的一步**，选错了要么训练不了，要么白训。

| 基座 | fp16 权重 | 训练时估计 | M1 16GB 可行性 |
|---|---|---|---|
| Qwen2.5-**0.5B**-Instruct | ~1.0 GB | ~3–4 GB | ✅ 稳妥 |
| Qwen2.5-**1.5B**-Instruct | ~3.1 GB | ~6–8 GB | ⚠️ 边界，**本次实测确认** |
| Qwen2.5-3B-Instruct | ~6.2 GB | ~11 GB+ | ❌ 不可行 |

### 为什么必须用 `-Instruct` 版本而不是 base 版本？

```
Qwen2.5-0.5B          ← base：只会续写文本，不懂"对话"
Qwen2.5-0.5B-Instruct ← 已经过指令微调，懂对话格式与角色
```

在 **Instruct** 版本上做 LoRA 叫**领域适配**（domain adaptation）；
在 base 版本上做叫**指令微调**（instruction tuning）——后者需要几十万条数据才有效果。
我们只有几百条，**必须站在 Instruct 的肩膀上**。

### 为什么不用 Qwen3 而用 Qwen2.5？

| | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|
| 推理模式 | 无 | 有 `思考模式`（thinking），输出带 `思考过程` |
| 微调友好度 | 好 | 稍麻烦——思考模式的 token 会干扰 loss 计算 |
| 生态成熟度 | 高，教程多 | 较新 |

**Qwen3 的思考模式对微调是个真实的坑**：它的输出形如
`思考过程...<｜end▁of▁thinking｜>真正的回答`，如果你不想训思考能力，
得在数据处理时把思考段落 mask 掉，否则模型会学出奇怪的格式。

→ **建议用 Qwen2.5-1.5B-Instruct 起步**（若 OOM 则退 0.5B）。

---

## 2. 数据：这一步决定成败（占整个工作的 70%）

**微调的效果上限由数据决定，不由参数决定。**
几百条高质量数据 > 几千条垃圾数据。

### 2.1 数据从哪来（三条路，按可用性排序）

| 来源 | 说明 | 量 |
|---|---|---|
| **A. 自己写 + 模板扩展** | 最可靠。手写 50–80 条种子，再用模板/LLM 扩写 | 300–800 条 |
| B. 公开数据集 | 如 `shibing624/alpaca-zh` 等中文指令集，需筛选电商相关 | 不定 |
| C. LLM 合成 | 用强模型（DeepSeek/GPT）按你的规范生成 | 快但需人工抽查 |

**推荐 A + C 组合**：手写种子定风格（保证质量），LLM 扩写补量（保证规模），
然后**人工抽查 10%**。

⚠️ **必须警惕"用同一个模型生成数据又用它训练"**——这会放大它的缺点。
如果合成数据，用**更强的模型**（比如用 DeepSeek API 生成，训 Qwen2.5-1.5B）。

### 2.2 数据格式：Alpaca 格式（最通用）

```json
{
  "instruction": "你是电商客服助手。请回答用户的售后问题。",
  "input": "我买的衣服七天无理由能退吗？",
  "output": "您好，服装类商品支持七天无理由退货，需保持吊牌完整、未洗涤。\n退货入口：我的订单 → 申请售后 → 选择退货退款。\n退款将在商家确认收货后 1–3 个工作日退回原支付渠道。"
}
```

**三个字段的含义**：

| 字段 | 作用 | 是否参与 loss 计算 |
|---|---|---|
| `instruction` | 系统提示，定义角色与任务 | ❌ 不算 |
| `input` | 用户输入 | ❌ 不算 |
| `output` | 期望输出 | ✅ **只算这个** |

### 2.3 ★ 关键概念：loss masking（只学回答，不学问题）

**这是微调最容易做错的地方。**

朴素做法是把整段文本拼起来算 loss：

```
"你是电商客服助手。请回答用户的售后问题。我买的衣服七天无理由能退吗？您好，服装类商品支持..."
 └────────────────── 这段也在算 loss ──────────────────┘└──── 这段也在算 ────┘
                        ❌ 错
```

**为什么错**：模型被迫去"预测用户会问什么"。
它会把容量浪费在模仿提问上，而且会**降低回答部分的 loss 贡献占比**。

正确做法是**只在 output 部分计算 loss**：

```
<prompt 部分>  labels = -100（忽略）     ← -100 是 PyTorch 的 ignore_index
<answer 部分>  labels = 真实 token id   ← 只有这里算 loss
```

**代码上怎么实现**：找到 response 的起始位置，把它之前的 label 全部置为 `-100`。

```python
# 简化示意（完整版见 src/finetune/build_sft.py）
prompt_ids = tokenizer(prompt_text).input_ids
answer_ids = tokenizer(answer_text).input_ids
input_ids = prompt_ids + answer_ids
labels = [-100] * len(prompt_ids) + answer_ids   # ← 关键这一行
```

⚠️ 注意还要处理：`answer_ids` 结尾若是 EOS，要保留（让模型学会"说完就停"）。

### 2.4 ★ 数据划分：为什么必须留测试集

```
train : val : test = 8 : 1 : 1
```

- **train**：训练用
- **val**：每轮结束评估，判断何时该停（早停）
- **test**：**只在最后碰一次**，用来报告最终效果

**为什么 test 不能参与调参**：如果你根据 test 的表现去改参数，
你实际上是在"对测试集过拟合"，报出来的分数就虚高了。
这是学术界的硬规矩，工业界同样适用。

**必须做的重叠检查**：

```bash
# 训练集与测试集不能有重复问题
python -c "
import json
tr = {json.loads(l)['input'] for l in open('data/sft/train.jsonl', encoding='utf-8')}
te = {json.loads(l)['input'] for l in open('data/sft/test.jsonl', encoding='utf-8')}
print('重叠条数:', len(tr & te))   # 必须是 0
"
```

---

## 3. 训练：参数怎么定，每个参数在干什么

### 3.1 完整命令

```bash
cd "/Users/Zhuanz/untitled folder/ec-agent-rebuild"
export HF_HOME=/tmp/hf-cache HF_ENDPOINT=https://hf-mirror.com

.venv/bin/python src/finetune/train_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-file data/sft/train.jsonl \
  --val-file   data/sft/val.jsonl \
  --output-dir outputs/qwen2.5-1.5b-ecomcs \
  --max-seq-len 512 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
  --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 16 \
  --warmup-ratio 0.05 --eval-steps 20 --save-steps 100 \
  --device mps
```

### 3.2 参数逐个讲

#### （1）LoRA 结构参数

| 参数 | 值 | 含义 | 怎么调 |
|---|---|---|---|
| `--lora-r` | 8 | 低秩矩阵的秩，**参数量 ∝ r** | 任务简单 4–8；复杂 16–32。调大→参数多、更易过拟合 |
| `--lora-alpha` | 16 | 缩放系数，有效缩放 = `alpha/r` = 2 | 经验 `alpha = 2r`。改 r 时同步改 alpha 可保持更新幅度 |
| `--lora-dropout` | 0.05 | 适配器上的 dropout | 小数据集(几百条)用 0.05–0.1 防过拟合 |
| `target_modules` | q,k,v,o | 给哪些层挂适配器 | 只挂注意力层最省；挂上 MLP(`gate/up/down_proj`)效果更好但参数约翻 2 倍 |

**为什么只挂注意力层？** 注意力层负责"如何关注上下文"，是风格与格式的主要来源；
MLP 层更接近"知识存储"。我们只想改风格，所以先只动注意力层。

#### （2）优化器与学习率

| 参数 | 值 | 含义 |
|---|---|---|
| `--lr` | 2e-4 | 学习率。**LoRA 常用 1e-4 ~ 3e-4，比全参微调大 10–100 倍** |
| 优化器 | AdamW | 标准选择 |
| `--warmup-ratio` | 0.05 | 前 5% 步数线性升温 |

**为什么 LoRA 的学习率要大得多？**

```
全量微调：直接更新 15 亿个原始权重 → 步长必须很小(1e-5量级)，否则崩
LoRA 微调：只更新 100 万个新增的小矩阵，初始为 0
           → 从零学起，需要更大步长才能快速收敛 → 常用 2e-4
```

#### （3）显存与批大小（8GB 卡 / 16GB 内存上的核心权衡）

| 参数 | 值 | 作用 |
|---|---|---|
| `--batch-size` | 1 | 单次前向的样本数。**显存第一杀手** |
| `--grad-accum` | 16 | 累积 16 步梯度再更新一次 |
| `--max-seq-len` | 512 | 单条样本最大 token 数。**显存第二杀手**（注意力是 O(n²)） |
| `gradient_checkpointing` | True | 用计算换内存：不存中间激活，反向时重算 |

**关键理解：等效批大小 = batch_size × grad_accum = 1 × 16 = 16。**

为什么用「小 batch + 梯度累积」而不是直接 batch_size=16？

```
batch_size=16  →  16 条样本的激活值同时存在内存里  →  OOM
bs=1, ga=16    →  每次只放 1 条，算完梯度累加，16 次后才更新
               →  数学上等价于 batch_size=16，但内存只需 1 条的
```

代价：**慢 16 倍**（16 次前向才更新一次参数）。这是内存换时间的经典交易。

**`max_seq_len=512` 够不够？** 电商客服的问答通常 100–300 token，
512 足够。但要注意：**超长样本会被截断**，如果截断掉了 answer 部分，
那条样本的 loss 会变成 0（全是 -100），等于白训。**必须在数据处理时过滤或统计超长样本。**

#### （4）训练轮数与早停

| 参数 | 值 | 说明 |
|---|---|---|
| `--epochs` | 5 | 最多跑 5 轮 |
| `--eval-steps` | 20 | 每 20 步在 val 上评一次 |
| early stopping | patience=2 | val loss 连续 2 次不降就停 |

**为什么几百条数据 5 轮就可能过拟合？**

```
300 条 × 5 轮 = 1500 次参数更新
1,081,344 个可训练参数  ÷  1500 次更新
→ 平均每个参数只被"见过"不到 1.4 次的有效信息
```

小数据集上模型很容易**记住训练样本**而不是学到规律。症状：

```
train loss ↓↓↓  持续下降
val   loss ↓↑   先降后升   ← 这就是过拟合的信号，应该在这里停
```

---

## 4. 训练过程要看什么

### 4.1 loss 曲线的四种形态与对策

| 形态 | 诊断 | 对策 |
|---|---|---|
| train ↓, val ↓ | 健康 | 继续跑 |
| train ↓↓↓, val ↓ 然后 ↑ | **过拟合** | 早停；或减 r、加 dropout、加数据 |
| train 几乎不动 | **学习率太小 或 数据有问题** | 调大 lr 到 5e-4；检查 loss masking 是否把 answer 也 mask 了 |
| train 剧烈震荡 / NaN | **学习率太大**（或 fp16 溢出） | 降到 1e-4；MPS 上用 fp32 试 |

### 4.2 必须记录的数字

```python
torch.mps.current_allocated_memory()     # MPS 当前占用
resource.getrusage(...).ru_maxrss        # 进程峰值 RSS（统一内存机器上更诚实）
每 epoch 耗时、总耗时
实际步数（可能早停）
```

⚠️ **MPS 没有 `max_memory_allocated()`**，别照抄 CUDA 的脚本（见 learning-log 日志 3）。

---

## 5. 评估：训完怎么判断"有没有用"

### 5.1 三个维度分开报，绝不合成一个"准确率"

| 维度 | 怎么测 | 判定方式 |
|---|---|---|
| **格式合规率** | 检查输出是否符合要求结构 | 程序化（正则/JSON 解析） |
| **意图识别准确率** | 让模型输出 `scene` 字段，与标注比对 | 程序化 |
| **表述规范性** | 是否用"您"、有无越权承诺 | 人工抽样 20 条打分 |

### 5.2 对照必须严格

```
同一份 test.jsonl
同一个 prompt 模板
同一套解码参数（temperature / max_tokens / top_p）
只换模型：基座 vs 微调后
```

**任何一处不一致，对照就失效。**

### 5.3 ★ 幻觉不会因为微调而消失

这是最需要诚实的地方。**LoRA 微调改的是"怎么说"，不改"知道什么"。**

验证方法：从测试集抽 20 条，**问训练数据里没有的业务规则**（比如
"你们家的会员等级有几档"），看模型会不会编。**大概率会编**——因为它不知道正确答案，
而微调没有给它新知识。

**正确做法**：业务知识走 RAG（检索到的规则文档进 Prompt），
并在 Prompt 里约束"只依据参考资料回答，不知道就说不知道"。

---

## 6. 完整操作步骤（照着做）

### 准备

```bash
cd "/Users/Zhuanz/untitled folder/ec-agent-rebuild"
export HF_HOME=/tmp/hf-cache                    # 模型缓存位置
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像（直连会超时）
```

### Step 1：验证链路（5 分钟，别跳过）

```bash
.venv/bin/python scripts/check_finetune_stack.py --model Qwen/Qwen2.5-0.5B-Instruct
```

**为什么先跑 0.5B**：它只要 1GB 左右，能快速确认
「torch+MPS+peft+transformers 这四件套版本兼容」。
用 1.5B 去试错，每次失败要重下 3GB，浪费时间。

期望：6 步全绿。

### Step 2：造数据

```bash
# 我会提供 src/finetune/build_sft.py 和数据模板
.venv/bin/python src/finetune/build_sft.py --out data/sft --n 400
```

产出 `train.jsonl` / `val.jsonl` / `test.jsonl`，并打印：
各文件条数、超长样本数、train/test 重叠数（必须为 0）。

### Step 3：先测基座（建立基线，训之前）

```bash
.venv/bin/python src/finetune/eval_sft.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --test-file data/sft/test.jsonl \
  --tag baseline
```

**为什么训之前要先测**：没有基线就没法证明微调有用。
很多人的"提升"其实是评测方式变了，而不是模型变好了。

### Step 4：训练

```bash
.venv/bin/python src/finetune/train_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-file data/sft/train.jsonl --val-file data/sft/val.jsonl \
  --output-dir outputs/ecomcs-lora \
  --max-seq-len 512 --lora-r 8 --lora-alpha 16 \
  --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 16 \
  --device mps
```

**盯三个东西**：loss 曲线、内存峰值、是否早停。

### Step 5：评估微调后

```bash
.venv/bin/python src/finetune/eval_sft.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/ecomcs-lora \
  --test-file data/sft/test.jsonl \
  --tag finetuned
```

### Step 6：对照与结论

```bash
.venv/bin/python src/finetune/compare.py --baseline baseline --finetuned finetuned
```

输出三维度对照表 + 20 条幻觉抽样对照。

---

## 7. 常见失败与排查

| 现象 | 根因 | 处理 |
|---|---|---|
| `RuntimeError: MPS backend out of memory` | 内存不足 | 降 `max-seq-len` 到 256；或退到 0.5B |
| loss 一直不降 | loss masking 写错（把 answer 也 mask 了） | 打印一条样本的 `labels`，确认非 -100 的比例 > 0 |
| loss 变 NaN | lr 太大 / fp16 溢出 | lr 降到 1e-4；或用 fp32 |
| 输出重复啰嗦 | 训练轮数过多，过拟合 | 减 epochs；加 dropout；加数据 |
| 输出格式对但内容胡说 | **正常现象** | 知识靠 RAG，不靠微调。见 5.3 |
| 下载卡住 | HF 直连超时 | 确认 `HF_ENDPOINT` 已设 |
| `PermissionError when downloading` | 缓存目录不可写 | `export HF_HOME=可写路径` |

---

## 8. 做完这一遍，你真正掌握了什么

| 能力 | 具体表现 |
|---|---|
| 判断可行性 | 知道 bitsandbytes 不可用 ⇒ 没有 QLoRA ⇒ 只能 LoRA + 小模型 |
| 数据工程 | 会做 loss masking、会划 train/val/test、会查重叠 |
| 参数理解 | 说得出 `r` / `alpha` / `lr` / `grad_accum` / `seq_len` 各自在控什么 |
| 调试 | 看 loss 曲线能诊断过拟合/欠拟合/学习率问题 |
| 评估严谨性 | 知道要先建基线、要同条件对照、不能用 test 调参 |
| **边界认知** | **知道微调改不了知识，幻觉要靠 RAG 解决** |

最后一条最重要——**知道什么做不到，比知道怎么做更值钱**。
