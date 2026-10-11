"""微调效果评估：基座 vs 微调后，同一测试集、同一套解码参数。

为什么必须"先测基座再测微调后"：
    没有基线就没有对照。很多人的"提升"其实是评测方式变了、
    或者 prompt 写法变了，而不是模型变好了。
    **唯一可信的证据是：同条件、只换模型。**

三维度分开报，绝不合成一个"准确率"：
    1. 格式合规率  —— 程序化判定（JSON 可解析 / 结构符合要求）
    2. 回答覆盖率  —— 生成答案与参考答案的关键词重合度（程序化）
    3. 风格指标    —— 长度分布、敬语使用率（程序化）
    （"表述是否得体"需要人工判断，脚本只产出对照表供人工看）

用法：
    # 基线
    python src/finetune/eval_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --test-file data/sft/test.jsonl --tag baseline
    # 微调后
    python src/finetune/eval_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/ecomcs-lora --test-file data/sft/test.jsonl --tag finetuned
    # 对照
    python src/finetune/eval_sft.py --compare baseline finetuned
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_DIR = Path("eval/results")


# --------------------------------------------------------------------------
# 程序化判定函数
# --------------------------------------------------------------------------
def is_valid_json(text: str) -> tuple[bool, str]:
    """判定输出是否为合法 JSON，且字段符合要求。

    为什么这是微调最可靠的收益点：
    "格式合规"是**可程序化判定**的客观指标，不依赖人工打分。
    基座模型常常在 JSON 前后加解释文字（"好的，这是结果：{...}"），
    微调后通常能学会"只输出 JSON"。
    """
    t = text.strip()
    # 常见失败：包了 markdown 代码块
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        obj = json.loads(t)
    except Exception:
        return False, "not-json"
    if not isinstance(obj, dict):
        return False, "not-object"
    required = {"scene", "need_tool", "tool", "urgency"}
    missing = required - set(obj.keys())
    if missing:
        return False, f"missing:{','.join(sorted(missing))}"
    return True, "ok"


def keyword_coverage(pred: str, ref: str) -> float:
    """参考答案的关键内容有多少出现在预测里。

    ★ 这个指标我第一版写错了，值得记下来：

    第一版把参考答案里「4 字以上连续中文」当作关键词，要求逐字匹配：
        ref:  "拒收后包裹会退回商家...运费可能由您承担"
        pred: "可以的，您可以直接拒收商品...联系快递公司确认..."
        score: 0.3    ← 明明答对了要点，分数却极低

    原因是"逐字精确匹配"对中文太苛刻：换同义词、调整语序、
    加修饰词都会导致匹配失败。实测基线覆盖率仅 0.069，明显失真。

    修正后改用两层：
      1. **硬信息**（数字、时效、专有名词）——这些必须说对，逐字匹配合理
      2. **内容重合度**——用字符级重合率衡量"大致说了什么"，对改写宽容
    """
    # 第 1 层：硬信息。数字与时效说错了是实质错误，必须严格。
    hard = set(re.findall(r"\d+[个天小时分秒日年月%]?", ref))
    hard |= set(re.findall(r"[A-Za-z_]{3,}", ref))
    hard_hit = sum(1 for k in hard if k in pred) / len(hard) if hard else 1.0

    # 第 2 层：内容重合度。用字符 F1，对同义改写与语序调整宽容。
    #   为什么用字符级而不是分词：中文分词本身有歧义，
    #   而且客服回答里的同义替换（"退钱"/"退款"）会导致词级零匹配。
    r = set(ref) - set("，。！？、：；（）\n ")
    p = set(pred) - set("，。！？、：；（）\n ")
    if not r:
        return 1.0
    overlap = len(r & p) / len(r)
    return round(0.6 * hard_hit + 0.4 * overlap, 4)


def style_metrics(text: str) -> dict:
    return {
        "len": len(text),
        "has_honorific": int(bool(re.search(r"[您请]", text))),
        "has_steps": int(bool(re.search(r"(\d[\.、]|→|·)", text))),
        "has_newline": int("\n" in text),
    }


# --------------------------------------------------------------------------
def load_model(model_name: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)

    if adapter:
        # ★ 加载 LoRA 适配器：基座 + 适配器 = 微调后的模型
        #   注意 adapter 目录里同时存了 tokenizer（见 train_lora.py 的 save），
        #   所以 tokenizer 也一并从那里读，保证与训练时一致。
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, adapter)
        tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
        )

    model = model.to(device)
    model.eval()
    return model, tok


def generate(model, tok, instruction: str, user_input: str, device: str,
             max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_input}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            # ★ 解码参数必须两套评测完全一致，否则对照失效
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def run_eval(args) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.device import detect

    device = args.device or detect().device
    rows = [json.loads(l) for l in Path(args.test_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    print(f"加载模型（{'基座+适配器' if args.adapter else '仅基座'}）device={device} ...")
    model, tok = load_model(args.model, args.adapter, device)

    print(f"评估 {len(rows)} 条（max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}）")
    details, t0 = [], time.time()
    for i, r in enumerate(rows):
        pred = generate(model, tok, r["instruction"], r["input"], device,
                        args.max_new_tokens, args.temperature, args.top_p)
        ok, why = is_valid_json(pred)
        details.append({
            "i": i,
            "input": r["input"],
            "ref": r["output"],
            "pred": pred,
            "json_valid": ok,
            "json_reason": why,
            "coverage": round(keyword_coverage(pred, r["output"]), 4),
            **style_metrics(pred),
            "ref_len": len(r["output"]),
        })
        if (i + 1) % 5 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)} ...")

    elapsed = time.time() - t0

    # 只对"本应输出 JSON"的样本统计格式合规率
    json_rows = [d for d in details if d["ref"].strip().startswith("{")]
    qa_rows = [d for d in details if not d["ref"].strip().startswith("{")]
    n = len(details)
    summary = {
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "n_samples": n,
        "json_samples": len(json_rows),
        "json_valid_rate": round(sum(d["json_valid"] for d in json_rows) / len(json_rows), 4) if json_rows else None,
        "qa_samples": len(qa_rows),
        "avg_coverage": round(sum(d["coverage"] for d in details) / n, 4),
        "avg_len": round(sum(d["len"] for d in details) / n, 1),
        "avg_ref_len": round(sum(d["ref_len"] for d in details) / n, 1),
        "honorific_rate": round(sum(d["has_honorific"] for d in details) / n, 4),
        "steps_rate": round(sum(d["has_steps"] for d in details) / n, 4),
        "elapsed_s": round(elapsed, 1),
        "decode": {"max_new_tokens": args.max_new_tokens,
                   "temperature": args.temperature, "top_p": args.top_p},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.tag}.json").write_text(
        json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    print(f"\n明细已写入 {RESULTS_DIR / (args.tag + '.json')}")
    return summary


def compare(tags: list[str]) -> None:
    """生成对照表。这是最终要放进报告的东西。"""
    data = {}
    for t in tags:
        p = RESULTS_DIR / f"{t}.json"
        if not p.exists():
            print(f"❌ 缺少 {p}，请先跑 --tag {t}")
            return
        data[t] = json.loads(p.read_text(encoding="utf-8"))

    metrics = ["json_valid_rate", "avg_coverage", "avg_len", "honorific_rate", "steps_rate"]
    names = {
        "json_valid_rate": "JSON 格式合规率", "avg_coverage": "关键词覆盖率",
        "avg_len": "平均回答长度", "honorific_rate": "敬语使用率", "steps_rate": "分步骤率",
    }
    print("=" * 76)
    print("对照表（同一测试集、同一套解码参数）")
    print("=" * 76)
    header = f"{'指标':<18}" + "".join(f"{t:>16}" for t in tags) + f"{'变化':>16}"
    print(header)
    print("-" * 76)
    for m in metrics:
        vals = [data[t]["summary"].get(m) for t in tags]
        delta = ""
        if len(vals) == 2 and isinstance(vals[0], (int, float)) and isinstance(vals[1], (int, float)):
            d = vals[1] - vals[0]
            pct = (d / vals[0] * 100) if vals[0] else 0
            delta = f"{d:+.4f} ({pct:+.1f}%)"
        row = f"{names.get(m, m):<18}" + "".join(
            f"{(v if v is not None else 'n/a'):>16}" for v in vals
        )
        print(row + f"{delta:>16}")
    print("=" * 76)

    # 抽样对照：这个必须人工看
    print("\n抽样对照（前 5 条，供人工判断表述是否得体）:")
    t0, t1 = tags[0], tags[1] if len(tags) > 1 else tags[0]
    for i in range(min(5, len(data[t0]["details"]))):
        a = data[t0]["details"][i]
        b = data[t1]["details"][i]
        print(f"\n--- 样本 {i} ---")
        print(f"用户: {a['input']}")
        print(f"参考: {a['ref'][:110]}...")
        print(f"[{t0}] {a['pred'][:110]}...")
        print(f"[{t1}] {b['pred'][:110]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA 适配器目录；不给则评基座")
    ap.add_argument("--test-file", default="data/sft/test.jsonl")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 条（快速试跑）")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    # 默认 temperature=0（贪心解码）—— 评测要可复现，随机采样会让两次结果不同
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compare", nargs="+", default=None, help="只做对照：--compare baseline finetuned")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    run_eval(args)


if __name__ == "__main__":
    main()
