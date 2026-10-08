"""微调链路验证：在真正开训之前，用小模型把全链路跑通。

为什么必须先做这一步：微调涉及的每个库都是"版本敏感"的，
torch / transformers / bitsandbytes / peft 四者之间任意一个版本错配，
都会在**不同的阶段**报错（加载时、量化时、前向时、反向时）。
用 0.5B 模型跑一遍只要几分钟，能把所有版本问题一次暴露出来，
而不是等下载完 8G 的 Qwen3-4B、训了 20 分钟后才崩在 backward。

★ 后端无关（本脚本同时服务两台机器）：
    CUDA（RTX 4060 8GB）  → QLoRA：4bit NF4 + double quant + grad ckpt
    MPS （Apple M1 16GB） → LoRA ：非量化 fp16（**bitsandbytes 是 CUDA-only，Mac 上装不了**）
    CPU                   → 仅验证流程，速度无参考价值

用法：
    python scripts/check_finetune_stack.py                       # 自动选后端
    python scripts/check_finetune_stack.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/check_finetune_stack.py --load-only           # 只查版本，不下载模型
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

OK, FAIL = "\033[32m✅\033[0m", "\033[31m❌\033[0m"

steps: list[tuple[str, str, str]] = []
DEVICE = None  # 由 main() 填入 src.utils.device.DeviceInfo


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
        last = [line for line in tb.strip().splitlines() if line.strip()][-2:]
        detail = f"{type(e).__name__}: {e}"
        steps.append((FAIL, name, detail))
        print(f"{FAIL} {detail}")
        for line in last:
            print(f"      {line.strip()}")
        return False


# --------------------------------------------------------------------------
# 后端相关的两个开关：是否量化、用什么 dtype
# --------------------------------------------------------------------------
def is_quantized() -> bool:
    """是否走 4bit 量化（QLoRA）。MPS 上必然为 False。"""
    return bool(DEVICE and DEVICE.supports_4bit)


def load_kwargs() -> dict:
    """按后端拼出 from_pretrained 的加载参数。"""
    import torch

    kw: dict = {"device_map": {"": DEVICE.device} if DEVICE.backend == "cuda" else None}
    if DEVICE.backend == "cuda":
        kw["device_map"] = {"": 0}
    if is_quantized():
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        # 非量化：MPS 上 float16 比 bfloat16 稳，CPU 只能用 float32
        if DEVICE.backend == "mps":
            kw["torch_dtype"] = torch.float16
        elif DEVICE.backend == "cpu":
            kw["torch_dtype"] = torch.float32
        else:
            kw["torch_dtype"] = torch.bfloat16
    return kw


def peak_mem_gb() -> float:
    """读取峰值内存占用。CUDA 有专用 API，MPS/CPU 没有等价物。"""
    import torch

    if DEVICE.backend == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    # MPS 无 max_memory_allocated；用当前分配量近似（会低估峰值）
    if DEVICE.backend == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024**3
    return 0.0


def reset_peak() -> None:
    import torch

    if DEVICE.backend == "cuda":
        torch.cuda.reset_peak_memory_stats()


def free_mem() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if DEVICE.backend == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.backend == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
def check_hf_cache():
    """确认 HF 缓存目录可写。

    为什么单列一步：缓存不可写时报的是
    `OSError: PermissionError ... when downloading ...`，
    那句话会让人以为是网络问题或模型仓库问题，实际上只是目录权限。
    早一步拦下来，能省掉一轮排查。

    两种常见的不可写场景：
      1. 沙箱/受限环境只允许写工作区，而默认缓存落在 ~/.cache/huggingface
      2. HF_HOME 被指到了一个没有权限的路径
    """
    import os
    from pathlib import Path

    default = Path.home() / ".cache" / "huggingface"
    hf_home = os.environ.get("HF_HOME")
    target = Path(hf_home) if hf_home else default / "hub"

    target.mkdir(parents=True, exist_ok=True)
    probe = target / ".__write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:  # noqa: BLE001
        raise PermissionError(
            f"HF 缓存目录不可写: {target} ({type(e).__name__})\n"
            f"      修法：把缓存指到可写目录后再跑，例如\n"
            f"        export HF_HOME=/tmp/hf-cache        # 或工作区内的目录\n"
            f"      注意：仅看到 'PermissionError when downloading' 不代表网络问题，"
            f"先查这里。"
        ) from e

    mirror = os.environ.get("HF_ENDPOINT", "")
    note = f"HF 缓存可写: {target}"
    if "hf-mirror" in mirror:
        note += f" | 镜像: {mirror}"
    else:
        note += " | ⚠️ 未设 HF_ENDPOINT 镜像（国内直连 huggingface.co 常超时）"
    return note


def check_versions():
    import torch
    import transformers

    assert DEVICE.backend != "cpu", (
        "既没有 CUDA 也没有 MPS，退到了 CPU。流程能跑但慢到没有参考价值；"
        "如果这是 Mac，检查 torch 是否是官方 wheel（MPS 需要 macOS 12.3+ 与 arm64 wheel）"
    )
    if is_quantized():
        assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5), (
            f"torch {torch.__version__} < 2.5，新版 transformers 需要 >=2.5。"
            "注意 cu121 通道最高只到 2.5.1，升 torch 必须同时换 CUDA 通道。"
        )
    mode = "QLoRA(4bit)" if is_quantized() else "LoRA(非量化)"
    return (
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"{DEVICE.name} | ≈{DEVICE.memory_gb:.1f} GB | 模式={mode}"
    )


def check_bitsandbytes():
    """只在 4bit 可用时才是硬检查；Mac 上跳过并说明原因。"""
    import bitsandbytes as bnb  # noqa: F401

    assert bnb.__version__ >= "0.50.2", (
        f"bitsandbytes {bnb.__version__} 过旧。torch 2.6+ 需 >=0.50.2，"
        "否则会在 pytree 相关调用上崩溃。"
    )
    return f"bitsandbytes {bnb.__version__}，functional 可导入"


def check_4bit_quant():
    import torch
    from transformers import BitsAndBytesConfig

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4"
    return "BitsAndBytesConfig(NF4 + double quant + bf16) 构造成功"


def check_model_load(model_name: str):
    """加载模型 + 真做一次前向，验证 transformers 认识这个架构。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = load_kwargs()
    if kw.get("device_map") is None:
        kw.pop("device_map", None)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kw)
    if DEVICE.backend != "cuda":
        model = model.to(DEVICE.device)

    inputs = tok("测试", return_tensors="pt").to(DEVICE.device)
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits is not None

    # 关键：显式跑一次 backward 用的图，确认该后端支持训练而非仅推理
    peak = peak_mem_gb()
    del out
    free_mem()
    return f"{model_name} 加载 + 前向成功（{'量化' if is_quantized() else '非量化'}），当前占用 {peak:.2f} GB"


def check_lora_attach(model_name: str):
    """挂 LoRA 并确认可训练参数量——这是微调的核心机制。"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    kw = load_kwargs()
    if kw.get("device_map") is None:
        kw.pop("device_map", None)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kw)
    if DEVICE.backend != "cuda":
        model = model.to(DEVICE.device)

    # prepare_model_for_kbit_training 只对量化模型有意义。
    # 非量化时也必须开 gradient checkpointing，否则 16G 内存装不下激活值。
    if is_quantized():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    del model
    free_mem()
    return f"LoRA r=8/alpha=16 挂载成功，可训练参数 {trainable:,} / {total:,} = {trainable / total * 100:.3f}%"


def check_train_step(model_name: str):
    """最硬的验证：真跑一次反向传播。前向过了不代表能训。"""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = load_kwargs()
    if kw.get("device_map") is None:
        kw.pop("device_map", None)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kw)
    if DEVICE.backend != "cuda":
        model = model.to(DEVICE.device)

    if is_quantized():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.train()
    reset_peak()

    batch = tok(["订单退款需要三个工作日到账。"], return_tensors="pt",
                padding=True, truncation=True, max_length=64).to(DEVICE.device)
    batch["labels"] = batch["input_ids"].clone()

    out = model(**batch)
    loss_val = out.loss.item()
    out.loss.backward()
    peak = peak_mem_gb()
    assert loss_val == loss_val, "loss 是 NaN"
    del model, out
    free_mem()
    return f"完整训练步（前向+反向）成功，loss={loss_val:.4f}，峰值占用 {peak:.2f} GB"


def main():
    global DEVICE

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="用小模型验证链路，别拿 4B 试错")
    ap.add_argument("--load-only", action="store_true", help="只查版本，不下载模型")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils.device import detect

    DEVICE = detect()

    print("=" * 72)
    print("微调链路验证（后端无关：CUDA→QLoRA / MPS→LoRA / CPU→仅流程）")
    print("=" * 72)
    print(f"后端: {DEVICE.backend} | 设备: {DEVICE.name} | 内存 ≈{DEVICE.memory_gb:.1f} GB "
          f"| 4bit 可用: {DEVICE.supports_4bit}")
    for n in DEVICE.notes:
        print(f"  · {n}")

    step("1. 版本与加速后端", check_versions)
    # 缓存可写是"下载类"步骤的前置条件。它失败时必须【立即终止】，
    # 否则后面每一步都会重复报同一个下载错误，把真正的原因淹没在噪声里。
    if not args.load_only:
        if not step("1b. HF 缓存可写性", check_hf_cache):
            print("\n⛔ HF 缓存不可写，后续所有下载都会失败。按上面的提示修好再跑。")
            return 1

    if is_quantized():
        if not step("2. bitsandbytes 兼容性", check_bitsandbytes):
            print("\n⛔ bitsandbytes 不过，后面全部无意义，先修这个。")
            return 1
        step("3. 4bit 量化配置", check_4bit_quant)
    else:
        print(f"\n--- 2/3. 跳过 bitsandbytes 与 4bit 检查 ---")
        print(f"{OK} 当前后端（{DEVICE.backend}）不支持 4bit，走非量化 LoRA，属预期行为")
        steps.append((OK, "2/3. 量化检查", f"{DEVICE.backend} 无 bitsandbytes，改用非量化 LoRA"))

    if args.load_only:
        print("\n(--load-only，跳过模型加载与训练步)")
    else:
        step("4. 模型加载 + 前向", lambda: check_model_load(args.model))
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
    print(f"全链路通过（模式：{'QLoRA 4bit' if is_quantized() else 'LoRA 非量化'}），可以开始真正的微调了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
