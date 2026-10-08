# 智能电商风控与客服 Agent 平台 —— 复现计划

## 这份文件是什么

对 `智能电商风控与客服Agent平台实习项目总结+20道面试题及详解.docx` 的**独立复现计划**。
目标不是抄一遍，而是**跑出你自己的真数字**，从而：
1. 把 20 道题从“背话术”变成“我做过、我知道哪里会崩”；
2. 拿到可验证的指标对照表，面试时任何追问都接得住；
3. 顺手修正原文档里的技术硬伤（见下）。

技术选型：RTX 4060 Laptop **8GB 显存** / Windows 原生（已实测跑通）/
教学式最小可跑全链路 / 周期 2–3 周。

> ⚠️ 2026-10 实机订正：显存是 **8GB 不是 16GB**；`milvus-lite` 在 Windows 原生**可用**
> （含 HNSW），不必为此装 WSL2；HF 需走 `hf-mirror.com` 镜像。
> 详见 [`00-setup-windows.md`](00-setup-windows.md) 顶部的「实机订正记录」。

---

## 一、原文档的 6 处技术硬伤（复现时必须改）

| # | 原文 | 问题 | 修正 |
|---|---|---|---|
| 1 | Rerank 用 “轻量级 **Cross-BERT** 架构” | 术语错误，是 Cross-**Encoder**。BERT 是双塔 embedding 侧 | bge-reranker-v2-m3（Cross-Encoder），输入 `[query, doc]` |
| 2 | “HitRate@50 统计召回覆盖度” | HitRate 是二值命中，**不等于 Recall**；要算 Recall 必须每条 query 标多个 gold doc | 两个指标都实现，测试集每条标 2–4 个 gold chunk |
| 3 | Milvus 选 **IVF_FLAT** + 万级 QPS + 10ms 延迟 | 自相矛盾：IVF_FLAT 是暴力精算，低延迟高并发要用 HNSW。万级 QPS 在单机 4060 上不可能 | 主线 HNSW（M=16/efC=200/ef=64），IVF_FLAT 作为对照实验跑一次 |
| 4 | Top50+Top50 融合后送 Rerank | Rerank 是 O(n) 次前向，50 条会拖垮延迟 | 融合后取 **Top20–30** 送精排，最终 Top6 入 Prompt；延迟实测记录 |
| 5 | 3000 条数据 LoRA，准确率 +12% | SFT 改的是**输出风格/格式/字段识别**，改不了知识、救不了幻觉 | 微调前后同集对照，**分维度报**：格式合规率 / 意图准确率 / SQL 可执行率 |
| 6 | Qwen3-14B LoRA | 14B QLoRA 在 8GB 上不可能；**连 4bit 推理都要 ~9.5G** | 本机 4B QLoRA 跑通方法论（边界，需实测）；OOM 则退 1.7B；14B 标为云端可选 |

另有两个隐藏事实：
- **BGE-M3 同时输出 dense + sparse + colbert**，原文档只用 dense，浪费了一半能力 →
  复现时加一组「BM25 vs BGE-M3 sparse」对照，是加分项。
- **中文 BM25 必须分词**（jieba），不做这一步 BM25 效果差到会让你误判混合检索无用。

---

## 二、复现策略：可开关的消融实验

同一个评测集，逐层加模块，用数字说话。`configs/base.yaml` 的 `ablation` 段已定义好分层。

| 层 | 配置 | 想验证什么 |
|---|---|---|
| L0 | BM25 only | 关键词基线能到多少 |
| L1 | 向量 only | 语义检索对口语化 query 的增益 |
| L2 | BM25 + 向量 + RRF | 混合是否真的 1+1>2 |
| L2b | 同上但**加权融合** 0.5:0.5 | RRF 是否真的赢过手调加权（很可能在小编测试集上赢不了） |
| L3 | L2 + Rerank 精排 | 精排把 MRR 拉起来多少，代价是多少 ms |
| L4 | + Query 改写/多意图拆解 | 口语化和多意图两类 query 的专项增益 |
| L5 | + 分层记忆 | 多轮对话一致性（人工判 + 抽样） |
| L6 | + LoRA 微调 | 只影响生成/SQL 分支，看格式与可执行率 |

报告每层：`HitRate@1/5/10/20/50`、`Recall@10/20/50`、`MRR@10`、`nDCG@10`、`p50/p95 延迟`。

---

## 三、目录结构

```
ec-agent-rebuild/
├─ configs/base.yaml            # 所有参数集中（已建）
├─ data/
│  ├─ raw/                      # 原始规则/FAQ 文本 + stopwords.txt
│  ├─ chunks/                   # 切分产物（含 gold 标注映射）
│  ├─ eval/testset.jsonl        # ★ 150–250 条，每条 2–4 个 gold chunk_id
│  └─ sft/{train,val}.jsonl     # 微调数据
├─ src/
│  ├─ ingest/    parse.py chunk.py
│  ├─ retrieve/  bm25.py ✅ dense.py hybrid.py rerank.py
│  ├─ db/        schema.py seed.py text2sql.py
│  ├─ agent/     state.py nodes.py graph.py tools.py intent.py
│  ├─ memory/    short.py long.py compress.py
│  └─ finetune/  build_sft.py train_lora.yaml infer_compare.py
├─ eval/  metrics.py ✅ run_eval.py report.ipynb
├─ scripts/ check_env.py
└─ docs/  00-setup-windows.md ✅  plan.md ✅  findings.md  interview-notes.md
```

**已完成**：`configs/base.yaml`、`eval/metrics.py`（含自检）、`src/retrieve/bm25.py`（含 demo）、
本文件、`docs/00-setup-windows.md`。

---

## 四、2–3 周日程

### 第 1 周 —— RAG 做出可信数字（必须保底完成）
- **D1** 环境：Python + CUDA torch + 模型下载 + `check_env.py` 全绿
- **D2** 语料整理（300–500 chunk）+ 三种切分策略对照（400/512/800 + overlap）
- **D3** ★ **写评测集**（150–250 条 + gold 标注）+ `run_eval.py` 串起来
- **D4** BM25 单路 / 向量单路 → 跑出 L0、L1
- **D5** RRF 自己实现 + 加权融合对照 → L2、L2b
- **D6** Rerank + TopK×阈值网格 → L3，记录延迟
- **D7** 复盘写 `findings.md`：文档的“召回率+15.2% / Top1+10.1%”在我的测试集上成立吗

### 第 2 周 —— Agent 闭环 + 记忆 + NL2SQL
- **D8** 意图分类 + 工具白名单路由（先单 Agent 再拆三 Agent，对比“拆与不拆”）
- **D9** LangGraph 状态机：意图→推理→工具→校验→输出 + 条件边 + 重试 + 兜底
- **D10** 工具层：`search_kb` / `query_sql` / `rule_check` + 参数校验 + 3s 超时 + 2 次重试
- **D11** SQLite 造 5 张业务表（订单/用户/设备/风控日志/工单）+ 灌数据
- **D12** NL2SQL 五层：schema 检索→字段词典→few-shot→sqlglot 校验+只读拦截→执行兜底（对标 72%→93%）
- **D13** 分层记忆：短期 30min/12 轮 + 长期 Milvus 向量记忆（阈值 0.7，跨会话 Top3）
- **D14** 滑动窗口压缩（80% 触发、关键词置顶）+ 压缩前后多轮一致性与耗时对照

### 第 3 周 —— 微调 + 报告
- **D15** 造 SFT 数据 1500–3000 条（SQL 必须真执行验证过）
- **D16** Qwen3-4B QLoRA 训练（rank=8/alpha=16/lr=2e-4/epoch≤10 + early stop），记录 loss 与显存峰值
- **D17** 微调前后对照：格式合规率 / 意图准确率 / SQL 可执行率 / 幻觉抽样 20 例人工判
- **D18** 全链路回归 L0→L6，出总表 + 图
- **D19** 写 `findings.md`（含反直觉发现）+ 修正版 20 题答案
- **D20** 缓冲：补实验、代码答疑、准备面试叙述

---

## 五、数据集构造（最容易卡住的一步）

原文档只说“3000+ 条专属数据”，没给来源。复现用三层构造，全部可公开说明：

1. **知识库语料（RAG 用）**：平台公开规则页（7 天无理由、发货与物流、违规行为判定）+
   自写 60–120 篇 FAQ（物流、退换、活动、账单、风控标准）→ 切到 300–500 chunk。
2. **评测集（地基，值得花一整天）**：150–250 条人工 query，四类平衡：
   常规咨询 / 口语化模糊 / 多意图混合 / 风控核查；每条标 2–4 个 gold chunk_id。
   **没有 gold 标注就没有 MRR/nDCG，整个评测体系是空中楼阁。**
3. **微调数据**：模板 + LLM 合成 1500–3000 条（售后问答 / 意图分类 / NL2SQL），
   SQL 用 SQLite 真执行过滤，只留能跑的。

---

## 六、验收标准（交付物）

1. `README.md`：一条命令跑通全链路
2. **指标对照表**：每层配置 × 各指标 × 延迟，全是自测真数字
3. `findings.md`：至少 3 个反直觉发现。预期会撞到的：
   - RRF 在小编测试集上**未必**赢过手调加权；
   - Rerank 阈值 0.75 不是普适最优，TopK 从 6 涨到 10 收益迅速递减；
   - 混合检索的增益**主要来自口语化那 1/4 query**，常规 query 几乎无差。
4. 微调报告：loss 曲线 + 前后对照 + 显存/耗时
5. 修正版 20 题答案（换成自己的数字）

---

## 七、风险与止损

| 风险 | 止损方案 |
|---|---|
| Milvus 装不上 / Windows 原生 | `dense.backend=faiss`，链路不变，只换后端 |
| BGE-M3 下载慢或显存紧 | embedding 走 CPU；reranker 换 `bge-reranker-base` |
| 微调 OOM | seq→512、bs→1、开 gradient checkpointing；再不行换 Qwen3-1.7B 跑通流程 |
| 评测集写不动 | 先 60 条出趋势，第 2 周补到 200 |
| 时间不足 | **死保 D1–D7**（RAG 闭环 + 真指标），微调降级为演示级 |
