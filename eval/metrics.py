"""检索评测指标。

设计原则：只依赖标准库，纯函数，方便单测和离线批量跑。

关键概念澄清（这是原实习文档里被混用的一组概念）：
  * HitRate@K : TopK 中是否【至少存在一个】相关文档 -> 二值，衡量"覆盖/不遗漏"
  * Recall@K  : 召回的【相关文档数量】/ 该 query 的【全部相关文档数量】
                必须先给每条 query 标注多个 gold doc，否则 Recall 恒等于 HitRate
  * Precision@K: TopK 中相关文档占比 -> 衡量"结果干不干净"
  * MRR@K     : 第一个相关文档排名的倒数均值 -> 只看"首个命中排多靠前"
  * nDCG@K    : 带位置折损的排序质量 -> 对"多个相关文档的排序"最敏感

面试常追问：HitRate 高但 MRR 低说明什么？
答案：召回到了（覆盖够），但排得很靠后（排序差），用户/模型在 Top 里头根本看不到，
      典型原因是粗排能用、精排没做，或 Rerank 阈值/权重配错了。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# 单条 query 的指标
# --------------------------------------------------------------------------
def hit_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """TopK 内是否存在至少一个相关文档。"""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids 不能为空：没有标注的 query 无法评 HitRate")
    return 1.0 if gold & set(ranked_ids[:k]) else 0.0


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """TopK 覆盖了多少比例的相关文档。"""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids 不能为空")
    return len(gold & set(ranked_ids[:k])) / len(gold)


def precision_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return len(gold & set(topk)) / len(topk)


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """首个相关文档的排名倒数；TopK 内没命中记 0。"""
    gold = set(gold_ids)
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    # 标准 nDCG：位置 i（从 1 开始）折损为 log2(i+1)
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """二值相关度下的 nDCG@K。"""
    gold = set(gold_ids)
    if not gold:
        raise ValueError("gold_ids 不能为空")
    gains = [1.0 if d in gold else 0.0 for d in ranked_ids[:k]]
    ideal = [1.0] * min(len(gold), k)
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# 整个测试集的聚合
# --------------------------------------------------------------------------
@dataclass
class EvalConfig:
    hit_k: Sequence[int] = (1, 5, 10, 20, 50)
    recall_k: Sequence[int] = (10, 20, 50)
    mrr_k: int = 10
    ndcg_k: int = 10


@dataclass
class QueryResult:
    """一条 query 的检索结果，ranked_ids 按相关性从高到低排好序。"""

    query_id: str
    ranked_ids: list[str]
    gold_ids: list[str]
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)


def evaluate(records: Sequence[QueryResult], cfg: EvalConfig | None = None) -> dict:
    """聚合成一张可直接写进报告的表。"""
    cfg = cfg or EvalConfig()
    if not records:
        raise ValueError("records 为空")

    n = len(records)
    agg: dict[str, float] = {}

    for k in cfg.hit_k:
        agg[f"HitRate@{k}"] = sum(hit_at_k(r.ranked_ids, r.gold_ids, k) for r in records) / n
    for k in cfg.recall_k:
        agg[f"Recall@{k}"] = sum(recall_at_k(r.ranked_ids, r.gold_ids, k) for r in records) / n
    for k in cfg.hit_k:
        agg[f"Precision@{k}"] = sum(precision_at_k(r.ranked_ids, r.gold_ids, k) for r in records) / n
    agg[f"MRR@{cfg.mrr_k}"] = sum(reciprocal_rank(r.ranked_ids, r.gold_ids, cfg.mrr_k) for r in records) / n
    agg[f"nDCG@{cfg.ndcg_k}"] = sum(ndcg_at_k(r.ranked_ids, r.gold_ids, cfg.ndcg_k) for r in records) / n

    latencies = sorted(r.latency_ms for r in records)
    agg["latency_p50_ms"] = _percentile(latencies, 50)
    agg["latency_p95_ms"] = _percentile(latencies, 95)
    agg["n_queries"] = float(n)
    return agg


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[int(rank)])
    return float(sorted_values[lo] * (hi - rank) + sorted_values[hi] * (rank - lo))


# --------------------------------------------------------------------------
# 报告辅助：对比两个配置，算出"提升"和"代价"
# --------------------------------------------------------------------------
def compare(baseline: dict, improved: dict, metrics: Sequence[str] | None = None) -> list[dict]:
    """生成 baseline -> improved 的对照行，用于 README 的指标总表。"""
    keys = metrics or [k for k in improved if k.startswith(("HitRate", "Recall", "MRR", "nDCG", "latency"))]
    rows = []
    for k in keys:
        b, i = baseline.get(k, 0.0), improved.get(k, 0.0)
        rows.append(
            {
                "metric": k,
                "baseline": round(b, 4),
                "improved": round(i, 4),
                "delta": round(i - b, 4),
                # 相对提升。文档里的"召回率提升15.2%"必须写清是绝对值还是相对值，
                # 这两个口径差很多，面试官一定会追问。
                "delta_pct_abs": f"{(i - b) * 100:+.2f}pp",
                "delta_pct_rel": f"{((i - b) / b * 100 if b else 0):+.2f}%",
            }
        )
    return rows


if __name__ == "__main__":
    # 自检：用手工算得出的例子验证公式，不依赖任何第三方库
    ranked = ["d3", "d1", "d7", "d2", "d9"]
    gold = ["d1", "d2"]

    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5          # 只捞回 d1
    assert recall_at_k(ranked, gold, 4) == 1.0          # d1+d2 都捞回
    assert abs(reciprocal_rank(ranked, gold, 5) - 0.5) < 1e-9   # d1 排第 2
    assert precision_at_k(ranked, gold, 4) == 0.5

    recs = [QueryResult("q1", ranked, gold, latency_ms=12.0),
            QueryResult("q2", ["d2", "d5"], ["d2"], latency_ms=20.0)]
    table = evaluate(recs)
    print("HitRate@1 =", table["HitRate@1"], "(期望 0.5)")
    print("HitRate@5 =", table["HitRate@5"], "(期望 1.0)")
    print("MRR@10    =", round(table["MRR@10"], 4), "(期望 0.75: q1=1/2=0.5, q2=1/1=1.0)")
    print("P95 延迟  =", table["latency_p95_ms"], "ms")
    print("metrics.py 自检通过 ✅")
