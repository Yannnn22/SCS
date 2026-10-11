"""LoRA 微调训练脚本（MPS / CUDA / CPU 后端无关）。

本脚本的两个核心难点，都在代码里详细注释了：
  1. ★ Loss Masking —— 只对 answer 计算 loss，不对 prompt 计算
  2. ★ 梯度累积 —— 用小 batch 模拟大 batch，省内存

用法：
    # M1 Mac（本次主场景）
    export HF_HOME=/tmp/hf-cache HF_ENDPOINT=https://hf-mirror.com
    python src/finetune/train_lora.py \
      --base-model Qwen/Qwen2.5-1.5B-Instruct \
      --train-file data/sft/train.jsonl --val-file data/sft/val.jsonl \
      --output-dir outputs/ecomcs-lora \
      --max-seq-len 512 --lora-r 8 --lora-alpha 16 \
      --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 16

    # 干跑（不训练，只验证数据管线与 loss masking 是否正确）
    python src/finetune/train_lora.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss 的 ignore_index 默认值


# ==========================================================================
# ★★★ 核心 1：Loss Masking
# ==========================================================================
def build_example(tokenizer, instruction: str, user_input: str, output: str, max_seq_len: int) -> dict:
    """把一条 Alpaca 样本转成 input_ids + labels，并做 loss masking。

    ---------------------------------------------------------------------
    为什么必须做 loss masking？

    朴素做法：把 prompt 和 answer 拼成一整段，全部算 loss。

        "你是客服助手。用户问：能退货吗？答：可以，条件..."
         └──────── 这些 token 也在算 loss ────────┘└── 这些也在算 ──┘
                          ❌ 错

    后果有两个，第二个更隐蔽：
      1. 模型被迫去预测"用户会问什么"——这毫无用处，浪费模型容量
      2. ★ answer 的 loss 被大量 prompt token 稀释。
         一批样本里 answer 可能只占 20% 的 token，
         那么 loss 数值里只有 20% 在指导"怎么答"。
         模型会更快学会"复述问题"而不是"给出好答案"。

    正确做法：prompt 部分的 label 置为 -100（PyTorch 会忽略这些位置），
             只有 answer 部分的 label 是真实 token id。
    ---------------------------------------------------------------------

    实现方式对比（这里选了最稳的一种）：

    方案 A：分别 tokenize prompt 和 answer，然后拼接
        问题：`tokenizer(prompt+answer) != tokenizer(prompt) + tokenizer(answer)`，
              因为 BPE 在拼接边界处的合并行为不同，会错位几个 token。

    方案 B（本实现）：整段 tokenize，然后【定位 answer 起点】
        做法：在 token 序列里搜索 assistant 回复头的 token 序列，
              找到它的结束位置，那之后就是 answer。
        优点：与真实推理时的 token 序列完全一致，不会错位。
    """
    # 用模型自带的聊天模板构造完整对话。
    # 为什么不手工拼字符串：不同模型的模板不同（<|im_start|> / [INST] / <start_of_turn>），
    # 用 apply_chat_template 才能保证与推理时一致。训练与推理的模板不一致是隐蔽的 bug 源。
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": output},
    ]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 只取到 assistant 回复头为止，用来定位 mask 边界
    prompt_messages = messages[:2]  # system + user
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    labels = list(full_ids)
    # 把 prompt 长度范围内的 label 全部置为 -100
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = IGNORE_INDEX

    # 截断（注意：必须同时截断 input_ids 与 labels）
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    return {
        "input_ids": full_ids,
        "labels": labels,
        "n_prompt": n_prompt,
        "n_answer": sum(1 for x in labels if x != IGNORE_INDEX),
    }


class SftDataset:
    """按需构造样本（不预先 tokenize 全部，省内存）。"""

    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return build_example(self.tok, r["instruction"], r["input"], r["output"], self.max_seq_len)


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """把一批变长样本 pad 成矩形。

    注意 labels 的 pad 也要用 -100，否则模型会去学"预测 padding"。
    """
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


# ==========================================================================
# 模型装配
# ==========================================================================
def load_model_and_tokenizer(base_model: str, lora_r: int, lora_alpha: int,
                             lora_dropout: float, target_modules: list[str], device: str):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        # 很多因果 LM 没有 pad token。用 eos 顶替是常规做法，
        # 但必须保证 collate 里 labels 的 pad 用的是 -100（见上）。
        tok.pad_token = tok.eos_token

    # 非量化加载（Mac 上没有 bitsandbytes，见 learning-log 日志 2）。
    # MPS 上 float16 比 bfloat16 稳；CUDA 上用 bfloat16 更稳。
    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model = model.to(device)

    # ★ 梯度检查点：用计算换内存。
    #   原理：默认情况下反向传播需要前向的中间激活值，全部要存着 → 内存 O(层数)。
    #        开启后只存每层的输入，反向时重新算一遍前向 → 内存降到 O(1)，
    #        代价是训练变慢约 20-30%。
    #   8GB/16GB 级别的机器上这个开关是必须的。
    model.gradient_checkpointing_enable()
    # ★ 配合 checkpointing 必须调这个：
    #   否则被 checkpoint 的层输入不需要梯度，反向时梯度传不回 embedding，
    #   表现是 loss 完全不降（很隐蔽的坑）。
    model.enable_input_require_grads()

    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, cfg)
    return model, tok


def count_trainable(model) -> tuple[int, int]:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot


def proc_mem_gb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    except Exception:  # noqa: BLE001
        return 0.0


# ==========================================================================
# 训练循环
# ==========================================================================
def train(args) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.device import detect

    info = detect()
    device = args.device or info.device
    print("=" * 72)
    print("LoRA 微调")
    print("=" * 72)
    print(f"后端 {info.backend} | 设备 {device} | 4bit 可用 {info.supports_4bit}")
    if not info.supports_4bit:
        print("  注：无 bitsandbytes ⇒ 走非量化 LoRA（Mac 上的正常情况）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer(
        args.base_model, args.lora_r, args.lora_alpha, args.lora_dropout,
        args.target_modules, device,
    )
    tr, tot = count_trainable(model)
    print(f"\nLoRA r={args.lora_r} alpha={args.lora_alpha} target={args.target_modules}")
    print(f"可训练参数 {tr:,} / {tot:,} = {tr / tot * 100:.3f}%")

    train_ds = SftDataset(args.train_file, tok, args.max_seq_len)
    val_ds = SftDataset(args.val_file, tok, args.max_seq_len) if args.val_file else None
    print(f"\n训练集 {len(train_ds)} 条 | 验证集 {len(val_ds) if val_ds else 0} 条")

    # ---- 数据自检：这一步能提前发现 loss masking 写错 ----
    print("\n--- 数据管线自检（前 3 条）---")
    zero_answer = 0
    for i in range(min(3, len(train_ds))):
        ex = train_ds[i]
        print(f"  样本{i}: 总长 {len(ex['input_ids'])} | prompt {ex['n_prompt']} "
              f"| answer {ex['n_answer']} (非 -100 的 label 数)")
        assert ex["n_answer"] > 0, f"样本{i} 的 answer 全被 mask 掉了！loss 会恒为 0"
    for i in range(len(train_ds)):
        if train_ds[i]["n_answer"] == 0:
            zero_answer += 1
    if zero_answer:
        print(f"  ⚠️ 有 {zero_answer} 条样本 answer 被完全截断（loss 恒 0，等于白训）")
    else:
        print("  ✅ 所有样本的 answer 都未被完全截断")

    if args.dry_run:
        print("\n--dry-run：数据管线验证完毕，未进入训练。")
        # 干跑时也做一次前向+反向，确认 loss masking 真的生效
        batch = collate([train_ds[i] for i in range(min(2, len(train_ds)))], tok.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.train()
        out = model(**batch)
        out.loss.backward()
        print(f"  干跑前向+反向成功，loss={out.loss.item():.4f}，进程 RSS {proc_mem_gb():.2f} GB")
        print("  ★ 若 loss 是 0 或异常小，检查 loss masking（answer 可能被误 mask）")
        return

    # ---- 优化器与调度 ----
    # LoRA 常用 lr=1e-4~3e-4，比全参微调大 10-100 倍。
    # 原因：全参微调直接更新 15 亿个原始权重，步长必须极小；
    #      LoRA 只更新 200 万个新增小矩阵且初始为 0，从零学起需要更大步长。
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    n_opt_steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = n_opt_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_at(step: int) -> float:
        """线性 warmup + 余弦退火。

        warmup 的作用：训练初期参数是随机的，梯度方向噪声大，
        直接用大学习率会震荡甚至发散。前若干步线性升温，让模型先"站稳"。
        """
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * prog))

    print(f"\n总优化步数 {total_steps}（warmup {warmup_steps} 步）")
    print(f"等效批大小 = batch_size({args.batch_size}) × grad_accum({args.grad_accum}) "
          f"= {args.batch_size * args.grad_accum}")

    log_path = out_dir / "train_log.jsonl"
    history: list[dict] = []
    global_step = 0
    best_val = float("inf")
    bad_evals = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        idx = list(range(len(train_ds)))
        random.shuffle(idx)
        model.train()
        optimizer.zero_grad()
        running, n_micro = 0.0, 0
        epoch_t0 = time.time()

        for step_in_epoch, i in enumerate(idx):
            batch = collate([train_ds[i]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            # ★ 梯度累积：损失要除以 accum 步数，否则等效于把学习率放大了 accum 倍。
            #   这是最常见的实现错误之一——loss 不除，lr 就悄悄变成了 32 倍。
            loss = out.loss / args.grad_accum
            loss.backward()

            running += out.loss.item()
            n_micro += 1

            if (step_in_epoch + 1) % args.grad_accum == 0 or (step_in_epoch + 1) == len(idx):
                # 梯度裁剪：防梯度爆炸（小数据集上偶发）
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                lr = lr_at(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_steps == 0 or step_in_epoch + 1 == len(idx):
                    avg = running / max(1, n_micro)
                    rec = {
                        "step": global_step, "epoch": round(epoch + step_in_epoch / len(idx), 2),
                        "train_loss": round(avg, 4), "lr": round(lr, 8),
                        "rss_gb": round(proc_mem_gb(), 2), "elapsed_s": round(time.time() - t0, 1),
                    }
                    # 验证
                    if val_ds is not None:
                        vloss = evaluate(model, val_ds, tok, device, args.eval_batches)
                        rec["val_loss"] = round(vloss, 4)
                        # 早停判断
                        if vloss < best_val - 1e-4:
                            best_val = vloss
                            bad_evals = 0
                            rec["best"] = True
                        else:
                            bad_evals += 1
                            rec["bad_evals"] = bad_evals
                    history.append(rec)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"  step {global_step:5d} | ep {rec['epoch']:.2f} | "
                          f"train {avg:.4f}" + (f" | val {rec.get('val_loss', float('nan')):.4f}" if val_ds else "")
                          + f" | lr {lr:.2e} | rss {rec['rss_gb']:.2f}G | {rec['elapsed_s']}s")

                    if val_ds is not None and bad_evals >= args.patience:
                        print(f"\n⏹ 早停：val_loss 连续 {bad_evals} 次未改善（best={best_val:.4f}）")
                        save(model, tok, out_dir, args, best_val, global_step, history)
                        print(f"总耗时 {time.time() - t0:.0f}s")
                        return
                    running, n_micro = 0.0, 0

        print(f"--- epoch {epoch} 完成，用时 {time.time() - epoch_t0:.0f}s ---")
        save(model, tok, out_dir, args, best_val, global_step, history)

    print(f"\n训练完成，总耗时 {time.time() - t0:.0f}s")
    final = evaluate(model, val_ds, tok, device, args.eval_batches) if val_ds else None
    print(f"最终 val_loss: {final if final is None else round(final, 4)}")
    save(model, tok, out_dir, args, best_val, global_step, history)


def evaluate(model, ds, tok, device, max_batches: int | None = None) -> float:
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            if max_batches and n >= max_batches:
                break
            batch = collate([ds[i]], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total += loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def save(model, tok, out_dir: Path, args, best_val, global_step, history) -> None:
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta = {
        "base_model": args.base_model, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "max_seq_len": args.max_seq_len,
        "global_step": global_step, "best_val_loss": best_val,
        "peak_rss_gb": round(proc_mem_gb(), 2),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-file", default="data/sft/train.jsonl")
    ap.add_argument("--val-file", default="data/sft/val.jsonl")
    ap.add_argument("--output-dir", default="outputs/ecomcs-lora")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--eval-batches", type=int, default=20, help="验证时最多评多少条，控制耗时")
    ap.add_argument("--patience", type=int, default=2, help="val_loss 连续多少次不降就早停")
    ap.add_argument("--device", default=None, help="默认自动探测")
    ap.add_argument("--dry-run", action="store_true", help="只验证数据管线与 loss masking")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
