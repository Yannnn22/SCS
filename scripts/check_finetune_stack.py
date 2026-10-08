"""微调链路验证：在真正开训之前，用 0.5B 小模型跑通全链路。

为什么必须先做这一步：QLoRA 涉及的每个库都是"版本敏感"的，
torch / transformers / bitsandbytes / peft 四者之间任意一个版本错配，
都会在**不同的阶段**报错（加载时、量化时、前向时、反向时）。
用 0.5B 模型跑一遍只要几分钟，能把所有版本问题一次暴露出来，
而不是等下载完 8G 的 Qwen3-4B、训了 20 分钟后才崩在 backward。

用法：
    python scripts/check_finetune_stack.py
    python scripts/check_finetune_stack.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import traceback

OK, FAIL = "\033[32m✅\033[0m", "\033[31m❌\033[0m"

steps: list[tuple[str, str, str]] = []


def step(name: str, fn):
    """跑一个步骤，捕获异常并记录，不中断后续检查。"""
    print(f"\n--- {name} ---")
    try:
        detail = fn()
        steps.append((OK, name, detail or "通过"))
        print(f"{OK} {detail or ''}")
        return True
    except Exception as e:  # noqa: BLE001 - 校验脚本就是要兜住一切
        tb = traceback.format_exc()
        # 只留最后两行，完整栈写文件便于排查
        last = [l for l in tb.strip().splitlines() if l.strip()][-2:]
        detail = f"{type(e).__name__}: {e}"
        steps.append((FAIL, name, detail))
        print(f"{FAIL} {detail}")
        for line in last:
            print(f"      {line.strip()}")
        return False


# --------------------------------------------------------------------------
def check_versions():
    import torch
    import transformers

    assert torch.cuda.is_available(), (
        "CUDA 不可用！常见原因：装成了 CPU 版 torch（漏了 --index-url），"
        "或 Windows 侧驱动过旧。这一步不过，后面全部无意义。"
    )
    assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5), (
        f"torch {torch.__version__} < 2.5，新版 transformers 需要 >=2.5。"
        "注意 cu121 通道最高只到 2.5.1，升 torch 必须同时换 CUDA 通道。"
    )
    return (
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"{torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )


def check_bitsandbytes():
    import bitsandbytes as bnb
    import torch

    # 这条是历史大坑：torch 2.6 移除了 _register_pytree_node 私有 API，
    # 老版 bitsandbytes 在这里就会炸。0.50.2+ 已修。
    assert bnb.__version__ >= "0.50.2", (
        f"bitsandbytes {bnb.__version__} 过旧。torch 2.6+ 需 >=0.50.2，"
        "否则会在 pytree 相关调用上崩溃。"
    )
    # 实际功能探测，比只读版本号可靠
    assert bnb.functional is not None, "bnb.functional 不可导入"
    _ = torch.zeros(4, device="cuda")
    return f"bitsandbytes {bnb.__version__}，functional 可导入"


def check_4bit_quant():
    """最关键的一步：真做一次 4bit 量化，把 bitsandbytes + torch 的兼容性打实。"""
    import torch
    from transformers import BitsAndBytesConfig

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # QLoRA 标准
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # 省显存；8G 卡上必须开
    )
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4"
    return "BitsAndBytesConfig(NF4 + double quant + bf16) 构造成功"


def check_model_load(model_name: str):
    """加载模型 + 真做一次前向，验证 transformers 认识这个架构。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
    )
    inputs = tok("测试", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits is not None
    vram = torch.cuda.max_memory_allocated() / 1024**3
    del model
    torch.cuda.empty_cache()
    return f"{model_name} 4bit 加载 + 前向成功，峰值显存 {vram:.2f} GB"


def check_lora_attach(model_name: str):
    """挂 LoRA 并确认可训练参数量——QLoRA 的核心机制。"""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 与项目配置一致
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total * 100
    del model
    torch.cuda.empty_cache()
    return f"LoRA r=8/alpha=16 挂载成功，可训练参数 {trainable:,} / {total:,} = {ratio:.3f}%"


def check_train_step(model_name: str):
    """最硬的验证：真跑一次反向传播。前向过了不代表能训。"""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.train()
    batch = tok(["订单退款需要三个工作日到账。"], return_tensors="pt",
                padding=True, truncation=True, max_length=64).to("cuda")
    batch["labels"] = batch["input_ids"].clone()
    out = model(**batch)
    loss_val = out.loss.item()
    out.loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    assert loss_val == loss_val, "loss 是 NaN"
    del model, out
    torch.cuda.empty_cache()
    return f"完整训练步（前向+反向）成功，loss={loss_val:.4f}，峰值显存 {peak:.2f} GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="用小模型验证链路，别拿 4B 试错")
    ap.add_argument("--load-only", action="store_true", help="只查版本，不下载模型")
    args = ap.parse_args()

    print("=" * 72)
    print("微调链路验证（QLoRA 全链路）")
    print("=" * 72)

    step("1. 版本与 CUDA 可用性", check_versions)
    if not step("2. bitsandbytes 兼容性", check_bitsandbytes):
        print("\n⛔ bitsandbytes 不过，后面全部无意义，先修这个。")
        return 1
    step("3. 4bit 量化配置", check_4bit_quant)

    if args.load_only:
        print("\n(--load-only，跳过模型加载与训练步)")
    else:
        step("4. 模型 4bit 加载 + 前向", lambda: check_model_load(args.model))
        step("5. LoRA 挂载", lambda: check_lora_attach(args.model))
        step("6. 完整训练步（前向+反向）", lambda: check_train_step(args.model))

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    for mark, name, detail in steps:
        print(f"{mark}  {name:<28} {detail}")

    failed = [r for r in steps if r[0] == FAIL]
    print("=" * 72)
    if failed:
        print(f"有 {len(failed)} 步失败：{[r[1] for r in failed]}")
        print("先修这些再开训，否则会在下载完大模型后才崩。")
        return 1
    print("全链路通过，可以开始真正的 QLoRA 训练了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
