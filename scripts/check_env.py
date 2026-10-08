"""环境自检：在写任何业务代码之前，先确认 5 件事全绿。

用法：
    python scripts/check_env.py
    python scripts/check_env.py --backend faiss      # 跳过 Milvus（Windows 原生时用）
    python scripts/check_env.py --skip-llm           # 还没起模型服务时先跳过

设计取舍：所有检查都【不抛异常】，只报告状态，因为环境问题要一次看全，
而不是修一个报一个。
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import tempfile

OK, WARN, FAIL = "\033[32m✅\033[0m", "\033[33m⚠️ \033[0m", "\033[31m❌\033[0m"

results: list[tuple[str, str, str]] = []  # (状态, 检查项, 说明)


def record(ok: bool, item: str, detail: str, warn_only: bool = False):
    mark = OK if ok else (WARN if warn_only else FAIL)
    results.append((mark, item, detail))


# ---------------------------------------------------------------- 1. Python
def check_python():
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 11)
    detail = f"{platform.python_version()} @ {sys.executable}"
    if not ok and (v.major, v.minor) == (3, 12):
        detail += "  ← 3.12 的 milvus-lite / bitsandbytes wheel 仍不齐，建议 conda create -n ec python=3.11"
    record(ok, "Python 版本", detail, warn_only=True)


# ---------------------------------------------------------------- 2. PyTorch + CUDA
def check_torch():
    try:
        import torch
    except ImportError:
        return record(False, "PyTorch + CUDA", "未安装。pip install torch --index-url https://download.pytorch.org/whl/cu121")

    if not torch.cuda.is_available():
        return record(
            False,
            "PyTorch + CUDA",
            f"torch {torch.__version__} 但 cuda.is_available()=False。"
            "多为装成了 CPU 版，或（WSL）Windows 侧显卡驱动过旧",
        )

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    detail = f"torch {torch.__version__} | {name} | {total_gb:.1f} GB | CUDA {torch.version.cuda}"
    # 8GB 是本项目实机（RTX 4060 Laptop）的规格。任务管理器显示 "16GB" 是
    # 8G 专用 + 8G 共享系统内存，共享那部分 CUDA 用不了，别被它误导。
    if total_gb < 7.5:
        detail += "  ← ＜8G，Qwen3-4B QLoRA 大概率要退到 1.7B"
    elif total_gb < 11:
        detail += "  ← 8G 档：14B 推理与 8B 训练不可行；4B QLoRA 属边界，需实测"
    record(True, "PyTorch + CUDA", detail)


# ---------------------------------------------------------------- 3. 关键依赖
def check_deps():
    groups = {
        "检索/向量": ["jieba", "yaml", "numpy"],
        "模型": ["sentence_transformers"],
        "Agent": ["langgraph", "langchain"],
        "NL2SQL": ["sqlglot"],
        "微调": ["peft", "transformers", "datasets", "bitsandbytes"],
    }
    for group, mods in groups.items():
        missing = []
        for m in mods:
            try:
                __import__(m)
            except ImportError:
                missing.append(m)
        record(not missing, f"依赖 · {group}", "全部就绪" if not missing else f"缺失 {missing}")


# ---------------------------------------------------------------- 4. 向量库
def check_vectorstore(backend: str):
    if backend == "faiss":
        try:
            import faiss  # noqa: F401
            import numpy as np

            idx = faiss.IndexFlatIP(4)
            idx.add(np.random.rand(3, 4).astype("float32"))
            record(True, "向量库 · FAISS", "读写正常（Windows 原生推荐用这个）")
        except ImportError:
            record(False, "向量库 · FAISS", "未安装：pip install faiss-cpu", warn_only=True)
        return

    try:
        from pymilvus import MilvusClient
    except ImportError:
        return record(False, "向量库 · Milvus", "pymilvus 未安装：pip install 'pymilvus>=2.4' milvus-lite")

    # 用独立临时目录，而不是项目内的 data/milvus/。
    # 原因：Milvus Lite 会在库文件旁建一个同名的 .db 目录，drop_collection 不删它，
    # 于是自检每跑一次就往项目里吐一次垃圾。放临时目录后用完自动消失，零副作用。
    tmpdir = tempfile.mkdtemp(prefix="dsh_env_check_")
    path = os.path.join(tmpdir, "check.db")
    try:
        client = MilvusClient(path)
        coll = "_env_check"
        if client.has_collection(coll):
            client.drop_collection(coll)
        client.create_collection(coll, dimension=8, metric_type="IP")
        client.insert(coll, [{"id": 1, "vector": [0.1] * 8}])
        got = client.search(coll, data=[[0.1] * 8], limit=1)
        client.drop_collection(coll)
        record(True, "向量库 · Milvus Lite", f"建库/写入/检索全部正常，命中 id={got[0][0]['id']}")
    except Exception as e:  # noqa: BLE001 - 环境检查就是要兜住一切
        record(
            False,
            "向量库 · Milvus Lite",
            f"{type(e).__name__}: {e}  ← 若确认装不上，改用 --backend faiss",
            warn_only=True,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------- 5. LLM 服务
def check_llm(base_url: str, model: str):
    import json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "只回复两个字：就绪"}], "max_tokens": 16}
    ).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
        text = body["choices"][0]["message"]["content"].strip()
        record(True, "LLM 服务", f"{model} @ {base_url} 返回: {text}")
    except urllib.error.URLError as e:
        record(False, "LLM 服务", f"连不上 {url} ({e})。先启动 ollama serve 或 vLLM，或用 --skip-llm", warn_only=True)
    except Exception as e:  # noqa: BLE001
        record(False, "LLM 服务", f"{type(e).__name__}: {e}", warn_only=True)


# ---------------------------------------------------------------- 6. 磁盘
def check_disk():
    target = os.environ.get("MODEL_DIR", "/mnt/d/models")
    base = target if os.path.isdir(target) else "."
    free_gb = shutil.disk_usage(base).free / 1024**3
    detail = f"{base} 剩余 {free_gb:.0f} GB"
    if free_gb < 40:
        detail += "  ← 模型+索引+适配器建议预留 40G 以上"
    record(free_gb >= 40, "磁盘空间", detail, warn_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="milvus_lite", choices=["milvus_lite", "faiss"])
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="qwen3:8b")
    args = ap.parse_args()

    check_python()
    check_torch()
    check_deps()
    check_vectorstore(args.backend)
    if not args.skip_llm:
        check_llm(args.base_url, args.model)
    check_disk()

    print("\n" + "=" * 72)
    print("环境自检报告")
    print("=" * 72)
    for mark, item, detail in results:
        print(f"{mark}  {item:<18} {detail}")

    hard_fail = [r for r in results if r[0] == FAIL]
    print("=" * 72)
    if hard_fail:
        print(f"有 {len(hard_fail)} 项硬失败，必须先解决：{[r[1] for r in hard_fail]}")
        return 1
    print("关键项全部通过，可以开始 D2（语料整理与切分）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
