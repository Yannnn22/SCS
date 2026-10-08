# ec-agent-rebuild

对《智能电商风控与客服 Agent 平台实习项目总结》的**独立复现**——
目标不是重抄一遍话术，而是用同一套评测集跑出**自己的真数字**，验证原文档里那些提升幅度到底成不成立。

> 完整规划见 [`docs/plan.md`](docs/plan.md)，环境搭建见 [`docs/00-setup-windows.md`](docs/00-setup-windows.md)。

---

## 快速开始

```bash
conda create -n ec python=3.11 -y && conda activate ec

# 1) 先装 CUDA 版 torch（装错成 CPU 版后面全白干）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) 其余依赖
pip install -r requirements.txt

# 3) 环境自检：5 项全绿再往下走
python scripts/check_env.py                     # WSL/Linux
python scripts/check_env.py --backend faiss     # Windows 原生（Milvus Lite 不支持）

# 4) 已验证可跑的两个模块
python eval/metrics.py            # 评测指标自检（纯标准库）
python src/retrieve/bm25.py       # BM25 基线 demo
```

---

## 协作方式

本机（Mac）只负责**规划、写提示词、验证依赖是否可用**；实际动手在 Windows 那台
（RTX 4060）的 DSH 里执行。流程：

1. 把整个 `ec-agent-rebuild/` 拷到 Windows
2. 让那边的 DSH **先读 `AGENTS.md` 和 `docs/plan.md`**
3. 从 [`docs/prompts-for-windows-agent.md`](docs/prompts-for-windows-agent.md) 里
   **一次粘一段**（P0 → P16），它交付完再粘下一段
4. 把它的交付（实际命令输出、指标表、报错）贴回来一起看

> 提示词文件里的「验收标准」不要删，那是唯一的进度判据。
> 验收看它贴的**实际命令输出**，不是"应该可以跑"。

---

## 进度

| 阶段 | 模块 | 状态 |
|---|---|---|
| 0 | 规划 / 硬伤修正 | ✅ `docs/plan.md` |
| 0 | 环境搭建文档 | ✅ `docs/00-setup-windows.md` |
| 0 | **16 段可粘贴提示词（P0–P16）** | ✅ `docs/prompts-for-windows-agent.md` |
| 0 | Agent 项目契约 | ✅ `AGENTS.md` |
| 0 | 全局参数配置 | ✅ `configs/base.yaml`（含 L0–L3 消融分层） |
| 0 | 评测指标 | ✅ `eval/metrics.py`（HitRate/Recall/Precision/MRR/nDCG + 延迟分位） |
| 0 | BM25 基线 | ✅ `src/retrieve/bm25.py`（自研，k1/b 可调，jieba 分词） |
| 0 | 环境自检 | ✅ `scripts/check_env.py` |
| 1 | 语料整理 + 三种切分策略 | ⬜ P1 |
| 1 | ★ 评测集（150–250 条 + gold 标注） | ⬜ P2 |
| 1 | 向量检索 / RRF / Rerank | ⬜ P3–P5 |
| 2 | Query 改写 + LangGraph Agent + NL2SQL + 分层记忆 | ⬜ P6–P11 |
| 3 | QLoRA 微调 + 前后对照 + 总报告 | ⬜ P12–P16 |

---

## 关键设计决定（都有理由，不是随手选）

1. **消融开关化**：所有参数集中在 `configs/base.yaml`，`ablation` 段定义 L0→L3 分层。
   同一份代码 + 不同 yaml = 可对照实验。报告里每一层都有独立数字。
2. **BM25 自研而非用 `rank_bm25`**：官方实现不易从外部覆写 k1/b，而消融实验必须能调。
   自己写 60 行也才能真正讲清 IDF 平滑项为什么是 `(N-n+0.5)/(n+0.5)`。
3. **中文必须 jieba 分词**：不切词的话 BM25 在中文上几近随机，会让人误判“混合检索没用”。
   `jieba` 缺失时自动降级为「单字 + 相邻二字」，保证裸环境也能跑。
4. **向量后端可切换**（`milvus_lite | milvus | faiss`）：本机跑通不被环境卡死。
5. **指标全自己实现**：`eval/metrics.py` 有断言自检，面试被问公式时答得出实现细节。
6. **HitRate ≠ Recall**：原文档把两者混为一谈。测试集每条 query 必须标 2–4 个 gold chunk，
   否则 Recall 恒等于 HitRate，整个评测失去区分度。

---

## 已修正的原文档技术硬伤

| 原文 | 修正 |
|---|---|
| Rerank 用 “Cross-**BERT** 架构” | Cross-**Encoder**（bge-reranker-v2-m3） |
| “HitRate@50 统计召回覆盖度” | HitRate 是二值命中；Recall 需多个 gold doc，两者分开实现 |
| Milvus 用 **IVF_FLAT** + 万级 QPS + 10ms | 主线 **HNSW**（M=16/efConstruction=200/ef=64），IVF_FLAT 仅作对照 |
| Top50+Top50 融合后送 Rerank | 融合后取 **Top20–30** 送精排（Rerank 是 O(n) 次前向，50 条拖垮延迟） |
| 3000 条数据 LoRA → 准确率 +12% | SFT 改的是格式/字段识别，改不了知识；须**分维度**报（格式合规/意图准确/SQL 可执行率） |
| Qwen3-14B LoRA | 本机只有 **8GB 显存**，14B 连 4bit 推理（~9.5G）都跑不动，标为云端可选 |

---

## 目录

```
configs/base.yaml          # 全部可调参数 + L0~L3 消融分层
docs/plan.md               # 复现计划、日程、风险止损
docs/00-setup-windows.md   # 4060/Windows 环境搭建（含显存预算表）
scripts/check_env.py       # 环境自检，5 项全绿再动手
eval/metrics.py            # 检索指标（含自检）
src/retrieve/bm25.py       # BM25 基线（k1/b 可调 + 中文分词）
data/eval/testset.jsonl    # ★ 评测集（待建，D3 的核心产出）
```
