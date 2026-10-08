"""中文 BM25 检索（基线 L0）。

为什么不用 rank_bm25？三个原因，都是踩过的坑：
  1. rank_bm25.BM25Okapi 的公式里默认 k1=1.5 / b=0.75 且不好从外部覆写，
     而消融实验要求 k1、b 可配。
  2. BM25Plus / BM25L 那些变体混在一起容易让人搞不清用的是哪个公式。
  3. 自己写 60 行能彻底看懂 idf 为什么取 log((N-n+0.5)/(n+0.5))，
     面试问「BM25 为什么好」时才答得出来，而不是背结论。

公式：score(q, d) = Σ_t IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
     IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )      <- +1 的平滑保证 IDF 非负
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Sequence


class BM25:
    """极简 BM25，支持中英文分词器注入，参数可配。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or default_tokenizer
        self.doc_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}

    # ---------------- 建索引 ----------------
    def fit(self, docs: Sequence[tuple[str, str]]):
        """docs: [(chunk_id, text), ...]"""
        self.doc_ids = [d[0] for d in docs]
        self.doc_tokens = [self.tokenizer(d[1]) for d in docs]
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        self.df = Counter()
        for toks in self.doc_tokens:
            self.df.update(set(toks))

        n_docs = len(self.doc_ids)
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self.df.items()
        }
        return self

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score), ...]，按 score 降序。"""
        q_tokens = self.tokenizer(query)
        scores: list[tuple[str, float]] = []
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += idf * (freq * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((self.doc_ids[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --------------------------------------------------------------------------
# 中文分词：不做这一步，BM25 在中文上几乎等于随机
# --------------------------------------------------------------------------
_JIEBA = None
_STOPWORDS: set[str] = set()
_STOPWORD_PATHS = ("data/raw/stopwords.txt",)


def load_stopwords(paths: Sequence[str] = _STOPWORD_PATHS) -> set[str]:
    words: set[str] = set()
    for p in paths:
        f = Path(p)
        if f.exists():
            words |= {line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()}
    return words


def default_tokenizer(text: str) -> list[str]:
    """jieba 精确模式 + 小写化 + 停用词/单字过滤。

    jieba 不可用时退化为「按字切分 + 相邻二字组合」，保证脚本在裸环境也能跑通。
    """
    global _JIEBA, _STOPWORDS
    if not _STOPWORDS:
        _STOPWORDS = load_stopwords()

    try:
        if _JIEBA is None:
            import jieba

            _JIEBA = jieba
        tokens = [t.strip().lower() for t in _JIEBA.lcut(text)]
    except ImportError:
        text = text.lower()
        chars = list(text)
        bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
        tokens = chars + bigrams

    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


class Bm25Retriever:
    """对齐全项目统一的 retrieve(query, top_k) -> [(chunk_id, score)] 接口。"""

    name = "bm25"

    def __init__(self, chunks: Sequence[tuple[str, str]], cfg: dict | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.index = BM25(k1=cfg.get("k1", 1.5), b=cfg.get("b", 0.75)).fit(chunks)
        self.top_k = cfg.get("top_k", 50)

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        return self.index.search(query, top_k or self.top_k)


if __name__ == "__main__":
    corpus = [
        ("c1", "订单支付成功后，如未收到货可以申请退款，退款将在3个工作日内到账。"),
        ("c2", "账号异地登录会触发安全校验，需要短信验证码二次确认。"),
        ("c3", "7天无理由退货适用于大部分商品，生鲜类目除外。"),
        ("c4", "大额交易超过5000元将进入人工复核流程。"),
    ]
    r = Bm25Retriever(corpus, {"k1": 1.5, "b": 0.75, "top_k": 5})
    for q in ["退款多久到账", "异地登录怎么办", "生鲜能退吗", "多少钱算大额"]:
        print(q, "->", [(cid, round(s, 3)) for cid, s in r.retrieve(q, 2)])
