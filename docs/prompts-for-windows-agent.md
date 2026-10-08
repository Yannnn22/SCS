# ⛔ 已作废：这是给 Windows 那台机器的提示词集

> **2026-10 更新：本项目已改为单机 Mac M1 上完成，Windows 机器不再参与。**
> 本文件**不再使用**，保留仅供追溯当时的协作设计。
>
> 现状与调整后的计划见 [`plan-mac-m1.md`](plan-mac-m1.md)。
>
> 作废的原因有两个：
> 1. **硬件假设不存在**：这套提示词围绕「RTX 4060 Laptop 8GB 显存」写的，
>    而实际参与的是 Apple M1 / 16GB 统一内存 / 无 CUDA。
> 2. **协作模式改变**：原设计是「我在 Mac 规划 → Windows 上的 agent 执行」；
>    现在改为单机，且我（沙箱）只能写工作区，**重的安装与训练由用户本人执行**。
>
> ⚠️ 另外原提示词里「QT 用 4bit/QLoRA」的部分在 Mac 上**根本不可行**
> （bitsandbytes 是 CUDA-only），不要照它做。

---

# 提示词集：让 Windows 上的 Agent 帮你完成复现

## 怎么用

1. 把整个 `ec-agent-rebuild/` 目录拷到 Windows 那台电脑（`AGENTS.md` 和 `configs/base.yaml` 必须一起来）。
2. Agent **每次开工先让它读 `AGENTS.md` 和 `docs/plan.md`**，再贴当天的提示词。
3. **一天一个提示词**，别一次贴三个。让它交付完再进下一个。
4. 验收看它贴的**实际命令输出**。它说"应该可以"就是没跑过，让它真跑。

> 每个提示词末尾的「验收标准」不要删——这是你唯一的进度判据。

---

# P0 · 环境搭建（对应 D1）

```
读项目根目录的 AGENTS.md 和 docs/plan.md，然后执行环境搭建。

【先决策】你在这台 Windows 电脑上工作，硬件 RTX 4060 **Laptop，显存只有 8GB**
（注意：Windows 任务管理器会显示 "16GB"，那是 8G 专用 + 8G 共享系统内存，
共享部分 CUDA 用不了；以 torch.cuda.get_device_properties(0).total_memory 为准）。
我还没装过 Python。请你先判断用 WSL2 还是 Windows 原生：
- 原计划倾向 WSL2（理由是 milvus-lite 在 Windows 原生支持残缺）
- 但请**先实测再下结论**：直接在 Windows 原生装一下 milvus-lite 试试。
  如果它能正常工作，就不必为了它装 WSL2。
- 如果你判断我的机器装 WSL2 有具体障碍，直接告诉我，并给出原生方案

【网络前提】huggingface.co 直连会超时，必须用镜像：
所有 HF 相关操作都要设 HF_ENDPOINT=https://hf-mirror.com（建议写成用户环境变量）。
另外 HF 缓存默认落 C:，而 C: 余量紧张，建议改 HF_HOME 到 D 盘 —— 改之前先问我。

【任务】
1. 装 Miniconda + 建 Python 3.11 环境（命名 ec）
2. 装 CUDA 版 PyTorch —— ⚠️ 别直接照抄 cu121，先看下面
   - 官方各 CUDA 通道的 torch 版本上限不同（实测）：
       cu121 最高 2.5.1 / cu124 最高 2.6.0 / cu126 最高 2.14.1
   - 而 transformers 5.19 要求 torch>=2.5，所以升 torch 必须同时换通道
   - 先跑 nvidia-smi 看驱动版本，再决定通道，把驱动版本贴给我
   - 推荐：pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126
   - bitsandbytes 必须 >=0.50.2（旧版在 torch 2.6+ 上会崩）
   ⚠️ 绝对不要不加 index-url 直接 pip install torch，会装成 CPU 版
3. 按 requirements.txt 装依赖
4. 逐个验证并原样贴出输出：
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python scripts/check_env.py --backend faiss
5. 把环境信息写入 docs/env-report.md：
   OS / Python 版本 / torch 版本 / CUDA 版本 / 显卡型号 / 总显存 / 各依赖版本

【已知的大文件下载，我提前授权了】
- Miniconda ~500MB
- torch CUDA 版 ~2.5GB
- BGE-M3 / bge-reranker-v2-m3 各 ~2.2GB
- qwen3:8b ~5GB
这些不用再问我，直接下。用 ModelScope 或国内镜像，别用 HuggingFace 直连。

【验收标准】
- torch.cuda.is_available() 返回 True
- check_env.py 的关键项（Python/PyTorch/依赖）全绿
- docs/env-report.md 存在且信息真实

【做不到就停下来问我】不要自己编一个"应该可以"的结论。
```

---

# P1 · 语料 + 三种切分策略（对应 D2）

```
读 AGENTS.md。这是一个电商风控/客服知识库项目，现在做语料与切分。

【任务 1】造语料 data/raw/
写 60–120 篇电商 FAQ 与规则文档，用 Markdown，每篇带标题。必须覆盖这些主题：
- 物流：发货时效、延迟、丢件、改地址、拒收
- 退换：7天无理由、生鲜例外、退货运费、换货、退款时效
- 活动：优惠券、满减、限时秒杀、赠品、活动价保护
- 账单：支付方式、发票、分期、退款到账
- 风控：异常交易判定、账号异地登录、大额交易复核、刷单识别、账号解封
每篇 300–800 字。内容要像真的平台规则（有具体数字、时效、例外条款），
不要写"根据相关规定"这种空话——因为后续评测要靠这些具体数字区分 chunk。

【任务 2】实现 src/ingest/parse.py 和 src/ingest/chunk.py
支持三种策略，由 configs/base.yaml 的 chunking 段控制：
1. fixed：按 chunk_size / chunk_overlap 定长切，尽量在句号处断
2. semantic：相邻句 embedding 相似度低于 semantic_threshold 处断开
3. parent_child：父块 parent_size 给模型，子块 child_size 做检索，保留父子映射

每个 chunk 必须带 metadata：chunk_id / source / title / strategy / char_len / parent_id
chunk_id 规则：{source}_{strategy}_{序号}，必须稳定可复现（重新切分 id 不变）

【任务 3】对照实验
同一批语料跑三种策略，输出 data/chunks/ 下的三个文件，并打印一张表：
策略 | chunk 总数 | 平均长度 | 最大长度 | <80字碎块数 | 切分耗时

【任务 4】模块自检
每个文件都要有 if __name__ == "__main__" 的内置中文样例自检块，不依赖外部文件也能跑。

【验收标准】
- 三种策略都能跑，输出对照表（贴实际输出）
- chunk_id 稳定：连跑两次生成的 id 完全一致（贴出验证）
- 碎块（<80字）数量要说明，如果偏多要告诉我怎么改参数

【关键】告诉我 chunk_size=512/overlap=80 这个组合适不适合中文规则文档，
以及你在实测中观察到的具体问题。不要只报数字。
```

---

# P2 · 评测集 + 评测脚本（对应 D3，最不能省的一天）

```
读 AGENTS.md 和 docs/plan.md。今天只做评测集和评测脚本，这是整个项目的地基。

【背景】没有 gold 标注就算不出 MRR / nDCG，整个评测体系是空中楼阁。
原实习文档声称"召回率提升15.2%、Top1命中率提升10.1%"，我必须用自己的测试集验证。

【任务 1】data/eval/testset.jsonl —— 150–200 条
每条格式：{"id":"q001", "query":"...", "gold_chunk_ids":["..."], "category":"...", "note":"..."}

四类必须平衡，每类 40–50 条：
1. normal   常规咨询："退款多久到账"
2. colloquial 口语化模糊："我那个钱咋还没退回来啊"，"买东西买错了咋整"
3. multi_intent 多意图混合："我要退货，顺便问下我账号为啥登不上"
4. risk     风控核查："这个订单是不是刷单"，"我的号是不是被盗了"

⚠️ 每条必须标 2–4 个 gold_chunk_ids，且必须是 data/chunks/ 里真实存在的 id。
只有 1 个 gold 的话 Recall 会退化成 HitRate，指标就没区分度了。
标完必须程序化校验：所有 gold_chunk_ids 都能在 chunk 文件里找到，贴出校验结果。

【任务 2】eval/run_eval.py
- 读 configs/base.yaml，支持用 --level 指定跑哪个消融层（L0_bm25_only 等）
- 加载对应检索链路，跑完整测试集
- 调用 eval/metrics.py 输出指标：HitRate@1/5/10/20/50、Recall@10/20/50、MRR@10、nDCG@10、p50/p95 延迟
- 结果写 eval/results/{level}.json 和 {level}.md
【任务 3】再加一个能力：按 category 分组输出指标
因为我要验证一个假设：mixed检索的增益主要来自 colloquial 类，normal 类几乎无差。

【验收标准】
- testset.jsonl 条数 ≥150，四类均衡（贴出统计）
- gold_chunk_ids 校验 100% 通过（贴出）
- run_eval.py 能跑，并贴出 L0_bm25_only 的真实指标表
- 按 category 分组的表也贴出来

【注意】eval/metrics.py 我已经写好了，直接 import 用，不要重写。
它的接口是 evaluate(records: list[QueryResult], cfg) -> dict。
```

---

# P3 · 向量检索 L1（对应 D4）

```
读 AGENTS.md。现在加稠密向量检索，和已有的 BM25 形成对照。

【8GB 显存约束 — 先看这条再动手】
embedding 模型和 LLM **不要同时驻留在显存里**，8G 装不下。
本阶段（P3–P5）根本不调 LLM，所以：
- BGE-M3 的 device 请在 configs 里设成 **cpu**。
  300–500 个 chunk 的 embedding 用 CPU 是分钟级的事，不值得为它挤显存。
- 如果你认为 GPU 明显更快，可以试，但**必须报告 GPU 和 CPU 两种模式的耗时与显存占用**，
  然后由数据决定默认值，不要凭感觉选。

【任务 1】src/retrieve/dense.py
- 用 BGE-M3（BAAI/bge-m3），输出 1024 维
- 注意：BGE-M3 同时能出 dense / sparse / colbert 三种表示。
  本任务是 dense，但请顺便把 sparse 也存下来，后面有对照实验要用。
- 向量必须归一化（configs 里 dense.normalize=true），这样内积等价余弦
- 后端按 configs 的 dense.backend 走：faiss | milvus_lite
  （milvus_lite 在 Windows 原生已实测可用，含 HNSW 索引，放心用）

【任务 2】为什么要先做 FAISS 层？
先用 FAISS IndexFlatIP 做**精确检索**（暴力全量算），这是"真值"。
因为 Milvus 的 HNSW 是近似检索，后面要对比它和精确检索的差距。
这一步的指标会成为后续所有近似索引对比的上界。

【任务 3】跑指标
跑 L1_dense_only，输出和 L0_bm25_only 的对照表。

【任务 4】写一个离线批处理
embedding 很贵，必须缓存：把 chunk 向量存成 data/chunks/embeddings.npy，
同时存一份 chunk_id 顺序索引。第二次运行直接读缓存。

【验收标准】
- 贴出 L0 vs L1 的完整指标对照表
- 贴出按 category 分组的对照（重点看 colloquial 类，向量检索应该在这里明显赢 BM25）
- 贴出 embedding 缓存的耗时对比（首次 vs 命中缓存）
- 报告 BGE-M3 在你的显卡上跑 embedding 的显存占用和速度

【关键问题，必须回答】
BM25 在 colloquial 类上的具体表现如何？是不是真的像预期那样崩了？
贴出 3 个 BM25 检索失败的具体 query 例子。
```

---

# P4 · RRF 融合 + 加权融合对照（对应 D5）

```
读 AGENTS.md。现在做多路检索融合，核心是验证"RRF 是否真的赢过手调加权"。

【任务 1】src/retrieve/hybrid.py
实现两种融合：
1. RRF：score(d) = Σ 1/(rank(d) + k)，k 从 configs 的 fusion.rrf_k 读
   ⚠️ 必须自己实现公式，禁止调用现成的 RRF 库——我要能从代码里看清公式
2. weighted：alpha * minmax_scaled(bm25_score) + (1-alpha) * minmax_scaled(vec_score)
   alpha 从 configs 的 fusion.weighted_alpha 读
   ⚠️ 必须做归一化，因为 BM25 分数和余弦相似度的值域完全不同，
      不做归一化的加权融合是错的（这正是 RRF 要解决的问题）

【任务 2】融合后必须截断
configs 里有 fusion.candidate_k=30。融合结果先截到 30 条，再交下游。
原因：Rerank 是 O(n) 次前向，如果 50+50 融合后还把 50 条全送精排，延迟会爆炸。
这是原实习文档里的一个错误，不要复现。

【任务 3】跑 L2_hybrid_rrf 和 L2b_hybrid_weight
输出 L0 / L1 / L2 / L2b 四层对照表 + 按 category 分组表。

【任务 4】RRF 的 k 值敏感性
k 取 [10, 30, 60, 100] 各跑一次，看 HitRate@10 / MRR@10 怎么变。
把结果画成一张表或图（表就够了）。

【验收标准】
- 四层对照表（贴实际输出）
- RRF k 值敏感性表
- 贴出延迟：L2 vs L2b 的 p50/p95，说明融合带来的额外开销

【关键问题，必须诚实回答】
RRF 相对加权融合，在你的测试集上到底是赢还是输？赢多少？输多少？
如果 RRF 没赢，请直说——这本身就是一个有价值的发现，
不要为了迎合"RRF 更好"的预期去调参数凑结果。
```

---

# P5 · Rerank 精排（对应 D6）

```
读 AGENTS.md。现在做 Cross-Encoder 精排。

【术语纠正】重排模型是 Cross-**Encoder**，不要写成 Cross-BERT。
Cross-Encoder 把 [query, doc] 拼成一个序列送进模型，输出一个相关性分数，
精度远高于双塔（bi-encoder）但无法预先建索引，只能对少量候选做精算。
这就是它只能放在粗排之后的原因——这也是为什么候选池不能太大。

【任务 1】src/retrieve/rerank.py
- 模型 BAAI/bge-reranker-v2-m3（显存不够就换 `bge-reranker-base`，
  但**必须在报告里说明换了哪个、为什么换**，不能悄悄换）
  ⚠️ 8GB 显存：reranker 若要上 GPU，必须先确认此刻没有别的模型驻留显存。
  用完记得 `del model; torch.cuda.empty_cache()`。
- 对融合后的 candidate_k（30）条算分
- 按 configs 的 rerank.score_threshold 过滤，再取 top_n 条

【任务 2】两组网格搜索，都要出真实数字
网格1（Rerank 阈值）：[0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
  → 看 HitRate@1 / HitRate@5 / MRR@10 怎么变
网格2（最终入 Prompt 的 top_n）：[3, 5, 6, 8, 10]
  → 关键在"收益递减点在哪"，这直接决定 Prompt 的 Token 成本

【任务 3】跑 L3_hybrid_rerank，输出 L0→L3 五层完整对照表

【任务 4】延迟账必须算清楚
分阶段计时：BM25 / 向量 / 融合 / Rerank 各自耗时，p50 和 p95。
Rerank 30 条需要多少 ms？这是精排的真实代价。

【验收标准】
- 五层对照表（贴实际输出）
- 两组网格搜索表
- 分阶段延迟分解表
- 回答：configs 里的 0.75 阈值是不是最优？如果不是，最优是多少？

【关键问题，必须诚实回答】
精排主要拉高的是 HitRate 还是 MRR？
提示：精排不增加召回（候选池没变），它改变的是**排序**，
所以理论上主要提升 MRR 和 HitRate@1，HitRate@50 应该几乎不变。
请用你的数据验证这个判断对不对。
```

---

# P6 · Query 改写与多意图拆解（对应 D7 + L4）

```
读 AGENTS.md。现在做 L4：查询理解层。

【动机】前面 P3/P4 应该会发现：混合检索的增益主要来自 colloquial 类 query。
那 normal 类是不是根本不需要这么复杂的链路？L4 就是针对这两类做专项优化。

【任务 1】src/retrieve/rewrite.py，三个策略
1. 口语化归一：把"钱咋还没退回来"→"退款到账时间"。用规则词典 + LLM 兜底
2. 同义词扩展：BGE-M3 只能缓解不能消除，补一层显式同义词表
   （退货/退款/退钱、登录/登陆、券/优惠券、秒杀/限时抢购…）
3. 多意图拆解：一条 query 拆成多条子 query 分别检索再合并去重
   判定阈值用 configs 的 nl2sql.multi_intent_threshold

【任务 2】按 category 分组评估 L3 vs L4
这是今天最重要的产出：我要看到 L4 到底帮了哪一类、有没有副作用。
如果 L4 在 normal 类上没增益却在 colloquial 类上大涨，说明这个模块值得留；
如果在 normal 类上还掉分了，说明改写引入噪声，需要加保护条件。

【任务 3】消融掉三个子策略
单独关掉"同义词扩展"/"多意图拆解"各跑一次，看各自贡献多少。
不要三个一起上就说有效——那等于什么都不知道。

【验收标准】
- L3 vs L4 分组对照表
- 三个子策略的单独贡献表
- 贴出 5 个改写前后的具体例子（query 原文 → 改写后 → gold chunk 命中变化）
- 延迟变化（改写要调 LLM，代价不小）

【诚实要求】如果某个子策略是负收益，直接报告，不要隐藏。
```

---

# P7 · 分层记忆（对应 D13）

```
读 AGENTS.md。现在做 L5：短期 + 长期分层记忆。

【任务 1】src/memory/short.py
- 会话级缓存，TTL 30 分钟，最多 12 轮，参数从 configs.memory.short_term 读
- 后端优先 Redis，连不上自动降级进程内 dict（必须实现降级，不能崩）
- 必须能回答："这个用户在本次会话里说过什么"

【任务 2】src/memory/long.py
- 关键信息抽取：判断一轮对话是否值得长期留存
  判据用 configs 的 key_info_threshold=0.7
- 抽取后向量化存 Milvus/FAISS 的独立 collection（user_memory）
- 跨会话召回 Top3（recall_top_k）
- 关键点：长期记忆只存**业务特征**（历史风控记录、账号风险、
  咨询偏好、历史交易），闲聊和语气词全部过滤掉

【任务 3】构造多轮测试用例
至少 10 组，每组 5–8 轮对话，必须体现跨会话场景。例如：
  第1轮：我账号昨天异地登录了
  （对话结束，新会话）
  第2轮：帮我看看我的账号有风险吗
  → 有长期记忆时，Agent 应该记得之前的异地登录；没有时会答得很泛

【任务 4】对照实验（人工判 + 程序化）
L4（无记忆）vs L5（有记忆），在 10 组多轮用例上评：
- 上下文连贯性（人工 1-5 分）
- 决策前后一致性（有没有自相矛盾）
- 是否引用了历史信息（有/无）
程序化能测的是第三项，人工项我会自己填，你给我打分表模板。

【验收标准】
- 短/长期记忆各自的自检输出
- Redis 降级测试：故意关掉 Redis，确认脚本不崩（贴输出）
- 10 组多轮用例 + 对照打分表
- 长期记忆库里实际存了什么（贴 5 条真实记录，确认没存闲聊）

【关键问题】长期记忆会不会存进噪声？阈值 0.7 是偏松还是偏紧？用你的数据说明。
```

---

# P8 · 滑动窗口压缩（对应 D14 + L5 收尾）

```
读 AGENTS.md。现在做上下文压缩，目标是"不丢关键信息的前提下砍 Prompt"。

【任务 1】src/memory/compress.py，按 configs.memory.compression 实现
- 窗口 12 轮，单轮上限 800 token，全局 Prompt ≤4000 token
- 占用达 80%（trigger_ratio）自动触发压缩，提前规避超限
- 冗余过滤：相似度 >0.6 的内容剔除（用 embedding 算）
- 关键词强制置顶保留：configs 里的 keep_keywords（风控、订单、退款…）
  ⚠️ 关键词命中的内容**不参与淘汰**，这是硬规则

【任务 2】对照实验
L5（有记忆无压缩）vs L5b（有记忆+压缩），同一批多轮用例：
- 指标：多轮一致性得分（沿用 P7 的打分表）
- 成本：平均 Prompt token 数、平均推理耗时、实际显存峰值
- 风险：压缩后有没有丢掉关键信息（列出被压缩掉的句子，人工判是否该丢）

【任务 3】Token 计数要准
不要用 len(text)//2 这种估算糊弄，用真 tokenizer 计数。
如果本地没加载 tokenizer，就明确说明是估算并标注误差范围。

【验收标准】
- 压缩前后 token 数、耗时、显存对照表
- 被压缩掉的句子清单（至少 10 条）+ 是否该丢的判断
- 关键信息有没有被误删（关键词保护机制是否生效的验证）

【诚实要求】如果压缩导致指标下降，如实报告并分析原因。
"省了 token"和"答得更差"哪个更值，这个权衡结论要写清楚。
```

---

# P9 · SQLite 业务库 + NL2SQL（对应 D11–D12）

```
读 AGENTS.md。现在做 NL2SQL，这部分是原文档里最难自证的一环，要认真做。

【任务 1】src/db/schema.py + seed.py
建 5 张表并灌入真实感数据：
- orders（订单：订单号、用户id、金额、状态、商品类目、下单/支付时间、设备id、ip）
- users（用户：id、注册时间、等级、实名状态、手机号、地址）
- devices（设备：设备id、用户id、机型、首次/最近出现时间）
- risk_logs（风控日志：日志id、订单id、规则命中、风险分、处置结果、时间）
- tickets（工单：工单id、用户id、类型、状态、创建时间、解决时间）

数据量：orders ≥2000，users ≥500，risk_logs ≥800，tickets ≥600，devices ≥800
⚠️ 数据要"有故事"：故意埋一些可用于风控查询的模式，
   比如某个用户 1 小时内下 15 单、同一设备关联 20 个账号、
   某 IP 段集中在凌晨下单。这样后面的风控查询才有真结果。
手机号/地址等敏感字段要能脱敏。

【任务 2】src/db/text2sql.py —— 五层架构
1. schema 检索：先按 query 检索相关表（不是把 5 张表全塞进 Prompt）
2. 字段映射词典：口语化别名 → 真实字段（"花了多少钱"→pay_amount）
3. few-shot 注入：**动态检索**configs.nl2sql.few_shot_n 条最相似示例，
   ⚠️ 不要按原文档那样"硬塞 200 条进 Prompt"，那是 token 灾难
4. 语法校验 + 只读拦截：用 sqlglot 解析；
   禁止 drop/delete/update/insert/alter（readonly_only=true）；
   限制最多 3 张关联表、20 个字段（SQL 太复杂要拒绝并降级）
5. 执行兜底：超时 2s 熔断，报错返回可读提示而不是抛异常

【任务 3】SQL 缓存/词典命中要走捷径
高频 query 直接映射到已验证的 SQL，不要每次都问 LLM。这是工业实践的关键。

【任务 4】评测
造 40–60 条 NL2SQL 测试用例（query + 期望 SQL + 期望结果），评估：
- 工具调用成功率（对标原文档的 72%→93%）
- SQL 可执行率（生成的 SQL 能不能真跑）
- 结果正确率（跑出来的数对不对）
三个指标分开报，**不要混成一个"准确率"**

【验收标准】
- 5 张表的 schema 和实际行数（贴输出）
- 贴出 10 条(query → 生成SQL → 执行结果)的实际记录
- 三个指标的真实数字
- 恶意 SQL 拦截测试：故意让它生成 "delete from orders"，
  确认被拦住（贴输出）

【诚实要求】"工具调用成功率"这个指标很容易通过放宽标准做好看。
定义必须在报告里写清楚：什么算成功？（生成成功？执行不报错？结果正确？）
三者差别很大。
```

---

# P10 · LangGraph Agent 闭环（对应 D8–D10）

```
读 AGENTS.md。现在把检索、SQL、规则引擎组装成 Agent 执行闭环。

【任务 1】src/agent/state.py + nodes.py + graph.py
用 LangGraph StateGraph，5 个节点：
  intent（场景识别）→ reason（工具决策）→ act（工具调用）→ verify（结果校验）→ respond（输出）
- 条件边：verify 通过→respond；不通过→回到 reason 重试；
  超过 max_iter（5）→ 走 fallback 兜底
- 所有 LLM 调用走 src/llm.py 的统一接口（带超时、重试、全链路日志）
- 每个节点都要记录：耗时、输入、输出、状态流转

【任务 2】src/agent/tools.py —— 3 个工具
- search_kb(query)：调 RAG 链路
- query_sql(question)：调 NL2SQL
- rule_check(order_id)：查风控规则命中情况
统一约束：参数合法性校验 + 3s 超时 + 2 次重试 + 结果空值过滤

【任务 3】强制结构化推理
reason 节点的 Prompt 必须强制模型输出结构化思考：
{"need_tool": bool, "tool": str, "args": {...}, "reasoning": str}
这样你才能程序化检查它有没有"无依据决策"。

【任务 4】src/llm.py —— 统一模型接口
支持 OpenAI 兼容协议（Ollama 和 vLLM 都能接）。
必须实现：超时、重试、异常拦截、Prompt/输出全量日志落盘（logs/serve.jsonl）
日志格式要能支撑 P13 的线上问题排查：请求id/场景/检索结果/工具入参出参/耗时/token数

【验收标准】
- 端到端跑通至少 6 个 case（2 个风控、2 个账号、2 个售后），贴出完整链路日志
- 超时熔断测试：把某工具人为 sleep 5s，确认 3s 熔断并触发重试（贴输出）
- 死循环防护测试：构造一个必然失败的 case，确认 5 轮后走 fallback（贴输出）
- 贴出 logs/serve.jsonl 的 3 条真实记录

【关键问题】max_iter=5 够不够？有没有 case 是 5 轮还没收敛的？贴出来。
```

---

# P11 · 三场景 Agent 拆分 + 意图路由（对应 D8）

```
读 AGENTS.md。现在做三场景拆分，但要用数据回答"到底该不该拆"。

【任务 1】先用**单 Agent** 做基线
一个通用 Agent + 3 个工具，在测试集上跑，记录：
- 工具选错率（该查 SQL 却去查知识库）
- 意图识别准确率
这个基线很重要——不拆的代价必须量化出来，否则"拆成三个"只是架构偏好。

【任务 2】再拆成三个场景 Agent
按 configs.agent.scenes 配置差异化参数和工具白名单：
- risk：temperature=0.1, top_p=0.7, 工具 [rule_check, query_sql]
- account：temperature=0.1, top_p=0.7, 工具 [query_sql, rule_check]
- service：temperature=0.3, top_p=0.9, 工具 [search_kb, query_sql]
⚠️ 工具白名单必须**代码级强制**，不能只靠 Prompt 里写"你只能用这两个工具"
   —— Prompt 是软的，模型会越界。要在工具调度器里硬拦。

【任务 3】查场景时要用小模型
加一个轻量意图分类（可以用 BGE-small 微调，或用规则+LLM 兜底）。
⚠️ 不要为了分类每次都调 14B 大模型，那延迟不可接受。

【任务 4】对照实验
单 Agent vs 三 Agent，在同一批测试用例上比：
工具选错率 / 意图准确率 / 端到端延迟 / 平均 token 消耗
拆分会增加一次路由 LLM 调用，延迟上升是必然的——要量化这个代价值不值。

【验收标准】
- 单 vs 三的对照表（真实数字）
- 场景识别准确率（贴混淆矩阵）
- 工具白名单硬拦截测试：构造一个让 risk Agent 去查知识库的请求，
  确认被代码拦住（贴输出）

【诚实要求】如果三 Agent 拆分在你的规模下收益很小，直说。
原文档把拆分说得很必要，但它的收益建立在大规模、多团队、独立迭代的前提下。
你的小规模实验如果只提升 1-2 个点却增加 2 倍延迟，这个结论必须写进 findings。
```

---

# P12 · 全链路消融报告（对应 D18）

```
读 AGENTS.md。现在做总收官：跑完全部消融层，出最终报告。

【任务 1】一键跑全部层级
L0_bm25_only → L1_dense_only → L2_hybrid_rrf → L2b_hybrid_weight
→ L3_hybrid_rerank → L4_query_rewrite → L5_memory → L6_lora
输出一张大表：行=层级，列=HitRate@1/5/10/20/50、Recall@10/20/50、MRR@10、nDCG@10、p50/p95延迟

【任务 2】按 category 分组的明细表
每个层级 × 四个类别（normal / colloquial / multi_intent / risk）的 HitRate@10 和 MRR@10。
这张表是回答"哪个优化帮了哪类 query"的关键。

【任务 3】写 docs/findings.md，必须包含
1. 每一层的增益和代价（延迟、token、复杂度）
2. **至少 3 个反直觉发现**。预期方向（要自己验证，不许照抄）：
   - RRF 不一定赢手调加权
   - Rerank 阈值 0.75 不一定最优，top_n 收益递减点在 6 附近
   - 混合检索的增益主要来自 colloquial 类，normal 类几乎无差
   - 三 Agent 拆分在小规模下可能不划算
3. **推翻或确认原文档的哪些说法**。原文档声称：
   "召回率提升15.2%、Top1命中率提升10.1%、MRR提升8.7%"，
   你的数字是多少？差多少？为什么？
4. 每个"有效优化"的适用边界（什么场景下会失效）

【任务 4】图表
至少两张：①各层指标柱状图 ②延迟 vs 效果的散点/折线（展示边际收益递减）

【验收标准】
- 大表 + 分组明细表，全部真实数字
- findings.md 存在，含反直觉发现和与原文档的差异对比
- 图表文件生成在 eval/results/

【最重要】不许编数字。跑不出的层级就标"未测"并说明原因。
这份报告的价值在于真实——一个诚实的负面结论比一个编造的正面数字值钱得多。
```

---

# P13 · 微调数据构造（对应 D15）

```
读 AGENTS.md。现在准备 QLoRA 微调的数据。

【先明确预期，避免白干】
SFT 微调能改变的是：输出格式、字段识别、话术风格、SQL 语法规范
SFT 微调**不能**改变的是：模型的知识边界、幻觉
所以微调的目标是"让输出更规范、更少格式错误"，不是"让模型更懂电商"。
这个预期必须在报告里写清楚，不要声称微调消除了幻觉。

【任务 1】src/finetune/build_sft.py，造 1500–3000 条
三类数据混合：
1. 售后问答（约 40%）：用户问 → 规范话术答。要有多轮样本
2. 意图分类（约 20%）：对话 → {"scene":"risk|account|service", "need_tool":bool, "tool":...}
3. NL2SQL（约 40%）：自然语言 → 标准 SQL

⚠️ SQL 数据必须**真执行验证**：只留能在 SQLite 上跑通的，报错的直接丢。
   这是数据质量的关键，也是"SQL 可执行率"这个指标能提上去的原因。

【任务 2】清洗流程（每步都要报丢弃数量）
去重（精确 + 相似度 >0.9 的近似去重）→ 脱敏 → 格式标准化 → SQL 执行校验

【任务 3】划分 train / val
按 9:1 划分。⚠️ 必须保证**测试集和训练集不重叠**：
如果我在 P2 造的评测集里的 query 出现在训练集里，微调后的评测就是作弊。
请程序化检查并贴出重叠检查结果。

【任务 4】固化基线（微调前）
在评测集上先跑一遍**未微调**的模型，记录：
- 格式合规率（输出是否符合要求的 JSON/Markdown 格式）
- 意图分类准确率
- SQL 可执行率
这三个数就是微调的对照基线，必须现在测、现在存。

【验收标准】
- train/val 条数、三类数据分布（贴统计）
- 清洗每步的丢弃数量
- **训练集与评测集重叠检查结果**（必须为 0）
- 微调前的三个基线指标（真实数字）

【必须停下来问我】如果数据量凑不到 1500 条，或者你发现生成的 SQL
可执行率低于 80%，先告诉我，不要用低质量数据凑数。
```

---

# P14 · Qwen3-4B QLoRA 训练（对应 D16）

```
读 AGENTS.md。现在跑 QLoRA 微调。

【硬件约束】RTX 4060 Laptop，**显存只有 8GB**（不是 16GB）。
配置必须在预算内，且**必须先实测**：
- 基座 Qwen3-4B（⚠️ 绝对不要试 14B：4bit 推理就要 ~9.5G，必 OOM）
- 4bit 量化（NF4）+ double quant + gradient checkpointing
- LoRA rank=8, alpha=16, lr=2e-4, epoch≤10 + early stopping（patience=2）
- batch_size=1, gradient_accumulation=16
- max_seq_len 从 **512** 起（8G 卡不要一上来就 1024），不够再降到 384

【必须先做显存实测，再决定基座大小】
参考锚点：Qwen2.5-0.5B（同配置）单步训练峰值 1.29 GB，**但那不能直接外推**。
请按这个顺序做，每步都记录 torch.cuda.max_memory_allocated：

  1. 先跑一个 Qwen3-1.7B 的 20 步最小训练，量出峰值
  2. 再试 Qwen3-4B（seq 512），量出峰值
  3. 如果 4B OOM，**不要靠把 seq 压到 128 来硬塞** —— 那样训出来的模型
     没有实际意义，"我验证了方法论"这个说法就站不住了。
     退到 1.7B 做主线，并明确记录"4B 在 8G 上不可行及其原因"。

  这个降级结论本身就是有价值的发现，如实记录，不要为了跑通 4B 而牺牲序列长度。

【任务 1】用 LLaMA-Factory（配置化，便于讲参数）
写 src/finetune/train_lora.yaml，字段要能对应到上述参数。
如果你判断手写 peft 训练循环更适合我理解，告诉我理由并二选一。

【任务 2】训练过程必须记录
- loss 曲线（训练集 + 验证集），存成图
- **实际显存峰值**（torch.cuda.max_memory_allocated）
- 每 epoch 耗时、总耗时
- 是否触发 early stopping，在第几轮
- 开训前的 nvidia-smi（确认没有别的进程占着显存，否则 OOM 无法归因）

【任务 3】防止过拟合
epoch=10 对 2000 条小数据集很可能过拟合。请同时监控验证集 loss，
如果验证 loss 在第 3-4 轮就开始上升，如实报告并建议更优的 epoch 数。

【任务 4】保存适配器 + 推理验证
适配器存到 models/qwen3-4b-lora/。
用 3 条未见过的样本做推理，贴出原始输出，确认适配器能正常加载。

【验收标准】
- train_lora.yaml 完整内容
- loss 曲线图 + 显存峰值 + 总耗时（真实数字）
- early stopping 在第几轮（或说明没触发）
- 3 条推理样本的原始输出
- 适配器目录大小

【必须停下来问我】
- 如果 OOM：告诉我具体是哪一步 OOM，以及你建议的降级方案，不要自己反复试
- 如果验证 loss 早早上升：如实说，不要为了"训练完 10 轮"硬跑完
```

---

# P15 · 微调前后对照（对应 D17）

```
读 AGENTS.md。现在做微调前后的严格对照。

【核心原则】**同一套测试集、同一套 Prompt、同一套解码参数**，只换模型。
任何一处不一致，对照就无效。

【任务 1】三个维度分开评，不要合成一个"准确率"
1. 格式合规率：输出是否符合要求的 JSON/Markdown 结构（程序化判定）
2. 意图分类准确率：scene / need_tool / tool 三个字段的准确率
3. SQL 可执行率：生成的 SQL 能否在 SQLite 真跑通
每个指标都要 [基座] vs [微调后] 并排，贴真实数字。

【任务 2】幻觉抽样
从测试集随机抽 20 条，让基座和微调模型各答一次，输出对照表：
query | 基座回答 | 微调回答 | 是否有事实错误 | 错误类型
⚠️ 是否有事实错误由我人工判，你只负责生成对照表。
   但你可以程序化标出"回答里出现了知识库中不存在的数字/条款"这类可疑项，帮我聚焦。

【任务 3】诚实结论
微调到底在哪个维度有效、在哪个维度无效？
预期结论很可能是：格式和 SQL 明显改善，幻觉**没有**改善。
如果确实如此，如实写，这本身就是标准答案。

【验收标准】
- 三个维度的并排对照表（真实数字）
- 20 条幻觉抽样对照表
- 一段结论：SFT 改变了什么、没改变什么、为什么
- 微调后的适配器推理耗时 vs 基座（LoRA 应该几乎无额外开销，验证一下）

【禁止】不要写"微调后准确率提升 X%"这种混合口径的说法。
必须说清是哪个维度、提升多少、用什么标准判定的。
```

---

# P16 · 最终报告 + 面试材料（对应 D19–D20）

```
读 AGENTS.md。现在做最终交付。

【任务 1】README.md 完善
- 一条命令跑通全链路
- 指标总表（各层级 × 各指标 × 延迟）
- 目录结构说明
- 每个模块的核心设计决定和理由

【任务 2】docs/findings.md（如果 P12 写过就合并增强）
- 核心发现清单，按"有效/无效/负收益"分类
- 与原实习文档所有数字的差异对照表
- 每个优化的适用边界

【任务 3】docs/interview-notes.md —— 20 题的"我自己的答案"
针对原文档的 20 个问题，每题写三部分：
1. **我实际做了什么**（引用我项目里的真实文件和数字）
2. **我踩过的坑**（这个最有说服力：8G 显存撞墙、Milvus 预设被实测推翻、
   torch/CUDA 通道版本地狱、RRF 没赢、HF 直连超时……）
3. **如果被追问怎么答**（列出 2-3 个可能的追问点和回答方向）
⚠️ 必须用我自己的实验数字替换原文档的数字，不要保留原文档那些无法验证的说法。

【任务 4】把以下技术错误，在 interview-notes.md 里明确标注为"已修正"：
1. Cross-BERT → Cross-Encoder
2. HitRate 混同召回率 → 两者分开
3. IVF_FLAT 却声称低延迟高并发 → HNSW
4. 融合后 50 条全送 Rerank → 先截断到 30
5. 声称 LoRA 消除幻觉 → SFT 改格式不改知识
6. **14B QLoRA 不可行 → 且原因比原计划更严苛**：本机是 8GB 显存（不是 16GB），
   连 14B 的 4bit **推理**都跑不动（约 9.5G），更不用说 QLoRA 训练
7. **"milvus-lite 在 Windows 原生装不上" → 实测可用**（含 HNSW 索引）。
   这条要写成"我先入为主地假设了它不可用，实测发现假设是错的" ——
   面试里讲"我推翻了自己的预设"比讲"我照文档说的做"有价值得多
每题注明：原说法是什么、为什么错、正确说法是什么、**我是怎么发现的（实测命令+输出）**。

【验收标准】
- README.md 的一条命令能在干净环境跑通（贴出你的验证过程）
- findings.md / interview-notes.md 存在且内容完整
- 所有数字都能追溯到 eval/results/ 里的原始文件（不要有孤立数字）

【最后】给我一份"项目一句话总结"，控制在 3 句话内，
说清：做了什么、关键数字是多少、最难的点是什么。
```

---

# 附 · 遇到问题时的通用提示词

```
我遇到一个问题，先不要改代码，按以下格式帮我定位：

【现象】<原样贴报错或异常输出>
【复现命令】<我执行的命令>
【我的环境】<OS / Python / torch / CUDA / 显存>

请按顺序做：
1. 判断这是环境问题还是代码问题（给出判断依据，不要猜）
2. 如果是环境问题：给出最小验证命令，先确认再修
3. 如果是代码问题：指出具体文件行号，说明根因
4. 给出修复方案，并说明这个修复会不会影响已完成的指标（这点很重要，
   如果修一个 bug 会让之前跑出的数字全部失效，我必须知道）

【禁止】不要在我没同意的情况下重构代码结构或更换依赖。
【禁止】不要为了"让它跑起来"而注释掉报错的部分。
```

---

# 附 · 每日验收卡（你自己核对，不用给 agent）

| 检查项 | 不合格的表现 |
|---|---|
| 它贴了**实际命令输出**吗？ | 只有"应该可以正常运行" |
| 有**按 category 分组**的数字吗？ | 只有一个总平均分 |
| 有**延迟/成本**数字吗？ | 只报效果不报代价 |
| 说了**负面结论**吗？ | 全是正面提升，没有一处失败 |
| 参数来自 yaml 吗？ | 代码里出现硬编码数字 |
| 自检块能独立跑吗？ | 必须依赖外部文件才能验证 |
| 数字能追溯到原始文件吗？ | 只有汇总表，找不到 raw results |

**连续两天出现"全是正面结论"就要警惕**——真实的调优过程一定会有负收益的实验。
