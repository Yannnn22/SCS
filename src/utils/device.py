"""设备与环境探测：让同一套脚本在 CUDA（Windows/Linux）和 MPS（Apple Silicon）上都能跑。

为什么要这个模块：
    本项目最初在 RTX 4060 上开发，代码里到处假设 CUDA + bitsandbytes。
    换到 M1 Mac 后有两个硬约束：
      1. 没有 CUDA，只有 MPS；
      2. **bitsandbytes 是 CUDA-only 的，在 Mac 上根本装不了** ——
         这意味着 QLoRA（4bit 量化微调）这条路在 Mac 上不存在。
    把"探测 + 能力判断"集中在这里，其他脚本只需问能力，不用自己判断平台。
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    backend: str          # "cuda" | "mps" | "cpu"
    device: str           # 传给 torch 的字符串，如 "cuda:0" / "mps" / "cpu"
    name: str
    memory_gb: float      # 显存（CUDA）或统一内存（MPS）或系统内存（CPU）
    supports_4bit: bool   # 能否做 4bit 量化（= 能否 QLoRA）
    notes: list[str]


def detect(verbose: bool = False) -> DeviceInfo:
    """探测最佳可用后端。任何 torch 缺失/异常都降级到 CPU，绝不抛异常。"""
    notes: list[str] = []

    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", "cpu", "未安装 torch", 0.0, False, ["torch 未安装"])

    # ---- 1. CUDA（Windows/Linux + NVIDIA）----
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        mem = props.total_memory / 1024**3
        bnb_ok = _has_bitsandbytes()
        if not bnb_ok:
            notes.append("torch 认到 CUDA，但 bitsandbytes 不可导入 → 不能用 4bit/QLoRA")
        return DeviceInfo(
            backend="cuda",
            device=f"cuda:{idx}",
            name=torch.cuda.get_device_name(idx),
            memory_gb=mem,
            supports_4bit=bnb_ok,
            notes=notes,
        )

    # ---- 2. MPS（Apple Silicon）----
    mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if mps_available:
        # MPS 没有"显存"概念，用的是统一内存。用系统内存近似，
        # 但必须说明：这只是上限，实际可用的要扣掉系统和其它进程占用。
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
        mem = mem_bytes / 1024**3
        notes.append(
            f"MPS 使用统一内存，{mem:.0f} GB 是整机内存上限，"
            "非专用显存；实际可用需扣掉系统与其它进程占用"
        )
        # 关键结论：Mac 上没有 bitsandbytes ⇒ 没有 4bit ⇒ 没有 QLoRA
        notes.append("bitsandbytes 为 CUDA-only，Mac 上不可用 → QLoRA 不可行，只能跑非量化的 LoRA")
        return DeviceInfo(
            backend="mps",
            device="mps",
            name=f"Apple {platform.processor() or 'Silicon'} (MPS)",
            memory_gb=mem,
            supports_4bit=False,
            notes=notes,
        )

    # ---- 3. CPU 兜底 ----
    notes.append("无 CUDA 也无 MPS，退到 CPU：能跑通流程但很慢")
    return DeviceInfo("cpu", "cpu", platform.processor() or "CPU", 0.0, False, notes)


def _has_bitsandbytes() -> bool:
    """bitsandbytes 能否导入。注意：它在 Mac 上是 CUDA-only，导入会失败。"""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - 导入失败的原因很多，全当不可用
        return False


def resolve_dtype():
    """给模型加载挑一个合理的 dtype。MPS 上 bfloat16 支持不完整，用 float16 更稳。"""
    import torch

    info = detect()
    if info.backend == "cuda":
        return torch.bfloat16
    if info.backend == "mps":
        return torch.float16
    return torch.float32


def describe() -> str:
    """一行式描述，给日志用。"""
    i = detect()
    return f"backend={i.backend} device={i.device} name={i.name} mem≈{i.memory_gb:.1f}GB 4bit={i.supports_4bit}"


if __name__ == "__main__":
    info = detect()
    print("=" * 68)
    print("设备探测")
    print("=" * 68)
    print(f"平台        : {platform.platform()}")
    print(f"Python      : {sys.version.split()[0]} @ {sys.executable}")
    print(f"后端        : {info.backend}")
    print(f"设备字符串  : {info.device}")
    print(f"设备名      : {info.name}")
    print(f"可用内存    : ≈{info.memory_gb:.1f} GB")
    print(f"支持 4bit   : {info.supports_4bit}   （决定能否 QLoRA）")
    if info.notes:
        print("\n说明：")
        for n in info.notes:
            print(f"  - {n}")

    # 真做一次张量运算，确认后端不是"名义可用"
    if info.backend != "cpu":
        try:
            import torch

            x = torch.randn(256, 256, device=info.device)
            y = (x @ x.T).sum().item()
            print(f"\n后端实测    : ✅ 张量运算成功（sum={y:.2f}）")
        except Exception as e:  # noqa: BLE001
            print(f"\n后端实测    : ❌ 运算失败 {type(e).__name__}: {e}")
            print("              → 建议改用 cpu 后端")
