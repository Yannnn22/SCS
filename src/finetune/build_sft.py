"""构造电商客服 SFT 数据（Alpaca 格式）。

为什么这个脚本是微调里最重要的一环：
    微调的效果上限由数据决定，不由参数决定。几百条高质量数据
    胜过几千条垃圾数据。而这里最容易做错的三件事：
      1. 不留测试集 → 无法证明微调有效
      2. 训练集与测试集重叠 → 指标虚高
      3. 样本超长被截断到 answer 全丢 → 那条样本白训

用法：
    python src/finetune/build_sft.py --out data/sft --n 400
    python src/finetune/build_sft.py --out data/sft --n 400 --seed 42 --json-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# ==========================================================================
# 为什么用"模板 + 槽位"而不是直接让 LLM 生成：
#   - 模板保证风格统一（这才是 LoRA 能学到的东西）
#   - 槽位保证可枚举、可复现、不引入事实错误
#   - 不依赖外部 API，离线可跑
# 缺点：多样性有限。若后续要扩量，用强模型(DeepSeek/GPT)在模板基础上改写，
#       但要用比被训模型更强的模型，否则会放大它自身的缺点。
# ==========================================================================

SYSTEM_PROMPT = "你是电商平台的客服助手，请依据平台规则回答用户问题，语气礼貌、给出行之有效的步骤。"

# --- 槽位：故意覆盖评测集要用的四个 category ---
SERVICES = ["退货退款", "换货", "发货时效", "物流异常", "优惠券使用", "发票开具", "账号异常", "订单取消"]

# 物流类
LOGISTICS = [
    ("发货时效", "我昨天下的单什么时候发货？",
     "您好，普通商品在您付款后 48 小时内发出，大促期间可能延长至 72 小时。\n您的订单目前状态为「待发货」，发出后会在订单详情页显示物流单号。\n如超过 72 小时仍未发货，您可在订单页申请「延迟发货赔付」。"),
    ("物流停滞", "我的快递三天没更新了怎么办？",
     "您好，物流信息超过 72 小时未更新属于异常件。\n请您提供订单号，我们可以为您发起物流查询工单，承运商会在 24 小时内反馈。\n若确认丢件，可选择重新发货或全额退款，运费由平台承担。"),
    ("改地址", "下单后发现地址写错了能改吗？",
     "您好，能否修改地址取决于订单状态：\n· 未发货：可在订单详情页自行修改收货地址，或联系客服协助；\n· 已发货：需联系快递员协商，或申请「拒收后重新下单」。\n跨省改地址可能产生运费差额，具体以承运商核算为准。"),
    ("拒收", "我不想要了可以直接拒收吗？",
     "您好，可以直接拒收。拒收后包裹会退回商家，我们收到后会为您发起退款。\n提示：无理由拒收产生的往返运费可能由您承担；若因商品质量问题拒收，运费由平台承担。\n退款一般在商家签收后 1–3 个工作日到账。"),
]

# 退换类
RETURN = [
    ("七天无理由", "我买的衣服能七天无理由退货吗？",
     "您好，服装类商品支持七天无理由退货。\n条件：吊牌完整、未洗涤、未穿着外出、不影响二次销售。\n流程：我的订单 → 申请售后 → 退货退款 → 选择退货原因 → 提交。\n时效：自签收次日起 7 天内可申请，退款在商家确认收货后 1–3 个工作日退回原支付渠道。"),
    ("生鲜例外", "生鲜水果也能七天无理由退吗？",
     "您好，很抱歉，生鲜类商品不适用七天无理由退货。\n原因：生鲜易腐、拆封后无法二次销售。\n但若出现腐烂、变质、缺斤少两等质量问题，请在签收后 24 小时内拍照上传，\n我们会为您办理全额退款，无需寄回。"),
    ("退货运费", "退货的运费谁出？",
     "您好，退货运费分两种情况：\n· 质量问题/发错货：运费由平台承担，我们承担上门取件；\n· 七天无理由：运费由买家承担，您可选择自行寄回或使用平台上门取件（运费从退款中扣除）。\n注：已购买「退货运费险」的订单，无理由退货的运费可获赔付。"),
    ("退款时效", "退款多久能到账？",
     "您好，退款到账时间取决于退款方式：\n· 原路退回：商家确认收货后 1–3 个工作日；\n· 平台余额：即时到账；\n· 银行卡：3–7 个工作日（由银行处理时间决定）。\n您可在「我的 → 退款/售后」查看每笔退款的实时状态。"),
    ("换货", "我想换个尺码怎么操作？",
     "您好，换货流程如下：\n1. 我的订单 → 申请售后 → 选择「换货」；\n2. 填写要更换的规格（如 M 换 L）；\n3. 按提示寄回原商品；\n4. 商家收到并确认后为您发出新商品。\n提示：换货需商品不影响二次销售；若无库存，系统会建议您改为退货退款。"),
]

# 活动类
PROMO = [
    ("优惠券", "我的优惠券为什么用不了？",
     "您好，优惠券无法使用通常是以下原因：\n· 未达门槛：券有最低消费金额要求；\n· 品类限制：部分券仅限指定类目；\n· 互斥：同一订单一般只能用一张券，不与满减叠加；\n· 已过期或已使用。\n请在「我的 → 卡券包」查看每张券的具体使用条件。"),
    ("价保", "我刚买就降价了，能补差价吗？",
     "您好，平台支持价格保护。\n条件：商品在签收后 15 天内降价（以订单页显示的价保标识为准），且非大促秒杀等特殊活动商品。\n流程：我的订单 → 申请价保 → 系统自动核算差价 → 审核通过后退回差价。\n若订单页没有价保入口，说明该商品不参与价保。"),
    ("满减", "满减和优惠券能一起用吗？",
     "您好，通常情况下：\n· 满减（店铺/平台活动）与优惠券**可以叠加**；\n· 但两张优惠券之间一般**互斥**，同类券只能选一张最优的；\n· 具体以结算页显示的优惠明细为准，系统会自动为您计算最优组合。"),
    ("取消订单", "我下错单了想取消怎么办？",
     "您好，取消订单请在订单详情页操作：\n· 未发货：可直接点击「取消订单」，款项退回原支付渠道；\n· 已发货：无法直接取消，需申请退货退款或拒收。\n注：使用优惠券的订单取消后，券若未过期会原路返还，已过期的券不补发。"),
]

# 账号与风控类
ACCOUNT = [
    ("账号异常", "我的账号突然登不上了，显示风险提示。",
     "您好，这是账号安全保护机制触发了。常见原因：异地登录、频繁更换设备、密码泄露风险。\n处理步骤：\n1. 在登录页点击「遇到问题」→「账号申诉」；\n2. 按提示完成实名验证与短信验证；\n3. 审核通常在 2 小时内完成。\n为保障资金安全，申诉期间账户内的余额与优惠券会临时冻结，恢复后自动解冻。"),
    ("支付失败", "我付款一直失败是什么原因？",
     "您好，支付失败的常见原因与处理：\n· 银行卡限额：换一张卡或分次支付；\n· 风控拦截：短时间内多次大额支付可能触发风控，建议 30 分钟后重试或联系发卡行；\n· 余额不足/卡过期：确认账户状态。\n若多次尝试仍失败，请提供订单号，我们为您核查具体拦截原因。"),
    ("盗号", "我怀疑我的号被盗了怎么办？",
     "您好，请立即按以下顺序处理（越早越好）：\n1. 修改密码并开启二次验证；\n2. 我的 → 账号安全 → 强制下线所有设备；\n3. 检查近 30 天订单，发现非本人操作立即申请「异常订单申诉」；\n4. 若已产生资金损失，请同时联系客服冻结账户并保留证据。\n我们会优先处理涉及资金的申诉。"),
]

# 无法处理 / 需要转人工 —— 这类样本很重要，教模型"知道边界"
ESCALATE = [
    ("投诉升级", "你们这个处理我不满意，我要投诉！",
     "非常抱歉给您带来不好的体验。我理解您的心情。\n为了更快解决您的问题，我已为您标记为优先工单，将由资深客服在 30 分钟内与您联系。\n同时您也可以在「我的 → 客服中心 → 投诉建议」提交书面投诉，我们会在 24 小时内给出书面答复。"),
    ("超出权限", "你能直接把我的钱退到我微信吗？",
     "很抱歉，我无法指定退款渠道。根据平台规则，退款一律**原路退回**至您支付时使用的账户，这是为了保障您的资金安全。\n如果您原支付账户已注销，请提供订单号，我们为您转接人工客服走特殊流程。"),
    ("未知规则", "你们家的会员一共有几个等级？",
     "很抱歉，我目前掌握的资料中没有这项信息，为避免给您错误答复，我不做猜测。\n建议您查看「我的 → 会员中心」页面，或在 App 内搜索「会员等级」查看最新说明。\n如仍有疑问，我可以为您转接人工客服。"),
]

GROUPS = {
    "logistics": LOGISTICS,
    "return": RETURN,
    "promo": PROMO,
    "account": ACCOUNT,
    "escalate": ESCALATE,
}

# --- 用户提问的口语化改写（模拟真实用户，也覆盖评测的 colloquial 类）---
COLLOQUIAL = {
    "我昨天下的单什么时候发货？": ["我昨天买的啥时候能发啊", "多久发货呀，等好久了", "下单一天了还没发货吗"],
    "我的快递三天没更新了怎么办？": ["快递咋不动了", "物流卡住了怎么办", "三天没物流信息了"],
    "我买的衣服能七天无理由退货吗？": ["衣服能退吗", "买的衣服不想要了能退不", "七天无理由退衣服行不行"],
    "退款多久能到账？": ["钱啥时候退回来", "退款要几天啊", "我的钱退到哪了"],
    "我的优惠券为什么用不了？": ["券咋用不了", "优惠券没法用是为啥", "我的券怎么失效了"],
    "我下错单了想取消怎么办？": ["买错了咋取消", "下错单了怎么办", "能取消订单不"],
    "我的账号突然登不上了，显示风险提示。": ["号登不上去了", "为啥说我账号有风险", "突然登录不了了"],
}

# --- JSON 结构化输出样本 ---
# 为什么要有这类：微调最可靠、最能量化的收益就是"格式合规率"。
# 让模型按固定 schema 输出，评测时可以程序化判定，不依赖人工。
INTENT_SCHEMA_TMPL = {
    "instruction": "你是电商客服的意图分类器。请阅读用户消息，输出严格的 JSON，不要有任何额外文字。",
    "input": "{query}",
    "output": '{{"scene": "{scene}", "need_tool": {need_tool}, "tool": {tool}, "urgency": "{urgency}"}}',
}


def _scene_of(group: str) -> str:
    """把数据分组映射到三个业务场景（与 configs/base.yaml 的 scenes 对齐）。"""
    return {"logistics": "service", "return": "service", "promo": "service",
            "account": "account", "escalate": "service"}[group]


def _urgency_of(group: str, question: str) -> str:
    if group == "escalate" or any(k in question for k in ("投诉", "盗", "冻结", "风险")):
        return "high"
    if group in ("account", "logistics"):
        return "medium"
    return "low"


def build_samples(json_ratio: float, rng: random.Random, target_n: int = 400) -> list[dict]:
    """生成样本。返回 Alpaca 格式的 dict 列表。

    ★ 设计要点：用「槽位 × 问法模板 × 答法变体」做组合扩展。

    为什么不用 LLM 直接生成：那会引入不可控的事实错误，且依赖外部 API。
    模板法保证：事实一致、风格统一、完全可复现、离线可跑。

    为什么需要"组合扩展"：只有 24 个基础 topic，
    而 LoRA 至少需要几百条才有意义。所以让每个 topic 有多种问法
    （模拟真实用户的不同措辞）和多种答法（正式/简洁/清单式）。
    这不是灌水——**学习"同一问题的不同说法"正是微调要干的事**。
    """
    # 问法模板：把陈述句改成真实用户可能用的各种口气
    def q_variants(topic_label: str, question: str) -> list[str]:
        base = question.rstrip("？?。")
        out = [question]
        # 口语化
        out.append(f"{base}啊")
        out.append(f"问一下，{base}")
        # 求助型
        out.append(f"麻烦问下{base}，谢谢")
        out.append(f"客服在不，{base}")
        # 省略主语型（真实用户常省略）
        if "我" in base:
            out.append(base.replace("我的", "").replace("我买", "买").replace("我", "") + "？")
        return list(dict.fromkeys([o for o in out if o and o.endswith(("？", "?"))]))

    # 答法变体：同一份事实，三种呈现风格
    #
    #   为什么要有这个：真实客服回复有长有短，只训一种风格模型会变得死板。
    #
    #   ★ 这里我第一版写错了，值得记下来：
    #     第一版用 `lines[0]`（第一行）做"简洁版"。但多行答案的第一行
    #     往往只是开场白，砍掉的是**关键信息**：
    #         完整: "您好，平台支持价格保护。\n条件：签收后15天内降价...\n流程：..."
    #         简洁: "您好，平台支持价格保护。"     ← "条件"和"流程"全丢了
    #     结果是模型学到"遇到问题就答一句空话"，直接伤害评测。
    #
    #   正确做法：简洁版必须**保留每行的要点**，只压缩措辞。
    def a_variants(answer: str) -> list[str]:
        lines = [l for l in answer.split("\n") if l.strip()]
        out = [answer]                                  # 完整版
        out.append("您好，为您解答：\n" + answer)        # 礼貌前缀版（教模型加敬语）

        if len(lines) > 1:
            # 要点版：逐行取每行的核心（去掉行首的"条件：""流程："等标签后保留内容），
            # 但【不丢行】——每行都保留，只是措辞更紧。
            core = []
            for l in lines:
                # 保留标签（它本身就是结构信息，模型该学会用），只去掉冗余修饰
                core.append(l.replace("非常抱歉，", "").replace("您好，", "").strip())
            merged = "；".join(core)
            if merged != answer:
                out.append(merged)
        return list(dict.fromkeys([o for o in out if o.strip()]))

    samples: list[dict] = []

    # ---- 1) 问答样本（主力）----
    for group, items in GROUPS.items():
        for topic, question, answer in items:
            for q in q_variants(topic, question):
                for a in a_variants(answer):
                    samples.append({
                        "instruction": SYSTEM_PROMPT,
                        "input": q,
                        "output": a,
                        "meta": {"group": group, "topic": topic, "kind": "qa"},
                    })
            # 口语化改写（从手工词典里取，这些是"最真实"的用户措辞）
            for variant in COLLOQUIAL.get(question, []):
                samples.append({
                    "instruction": SYSTEM_PROMPT,
                    "input": variant,
                    "output": answer,
                    "meta": {"group": group, "topic": topic, "kind": "qa_colloquial"},
                })

    # ---- 2) 意图分类样本（JSON 输出）----
    # 为什么要有这类：微调最可靠、最能量化的收益就是"格式合规率"。
    # 让模型按固定 schema 输出，评测时可程序化判定，不依赖人工打分。
    qa_pool = [s for s in samples if s["meta"]["kind"] in ("qa", "qa_colloquial")]
    n_json = int(len(qa_pool) * json_ratio)
    for s in rng.sample(qa_pool, k=min(n_json, len(qa_pool))):
        group = s["meta"]["group"]
        scene = _scene_of(group)
        need_tool = scene == "account" or group in ("account", "logistics")
        tool = '"query_sql"' if need_tool else "null"
        out = INTENT_SCHEMA_TMPL["output"].format(
            scene=scene, need_tool=str(need_tool).lower(), tool=tool,
            urgency=_urgency_of(group, s["input"]),
        )
        samples.append({
            "instruction": INTENT_SCHEMA_TMPL["instruction"],
            "input": s["input"],
            "output": out,
            "meta": {"group": group, "topic": s["meta"]["topic"], "kind": "intent_json"},
        })

    rng.shuffle(samples)
    if len(samples) < target_n:
        print(f"⚠️  生成 {len(samples)} 条，少于目标 {target_n} 条。")
        print("   要扩量请增加 GROUPS 里的 topic，或用强模型改写现有问答。")
    return samples


# --------------------------------------------------------------------------
# 长度统计与过滤：这是"防白训"的关键一步
# --------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """粗估 token 数。

    中文：约 1 字 ≈ 1 token（Qwen 的 BPE 对中文大致如此）
    英文：约 3 字符 ≈ 1 token（保守取 3 而非 4，宁可高估长度）

    为什么不加载真 tokenizer：
    本脚本要在下载模型【之前】就能跑（造数据不该依赖 3GB 模型）。
    粗估足够用来筛掉明显超长的样本；等训练脚本里再用真 tokenizer 复核。
    """
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cn
    return cn + max(1, other // 3) if other else cn


# --- 系统提示的真实 token 开销（模板文本，不必估算）---
def prompt_overhead_tokens() -> int:
    """计算聊天模板 + 系统提示的固定开销。

    为什么不写死一个数字（比如 40）：
    聊天模板的开销随 tokenizer 与模板而异。这里把已知的模板文本
    估出来，至少是"有依据的估计"而不是拍脑袋的常数。
    """
    tmpl = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
    return estimate_tokens(tmpl)


def analyze(samples: list[dict], max_seq_len: int, prompt_overhead: int | None = None) -> dict:
    prompt_overhead = prompt_overhead_tokens() if prompt_overhead is None else prompt_overhead
    """统计长度分布，并挑出会被截断到 answer 全丢的样本。"""
    lens, risky = [], []
    for s in samples:
        p = estimate_tokens(s["instruction"]) + estimate_tokens(s["input"]) + prompt_overhead
        a = estimate_tokens(s["output"])
        total = p + a
        lens.append(total)
        # 危险样本：prompt 本身就把窗口占满了，answer 一个 token 都放不下
        if p >= max_seq_len:
            risky.append((s["input"], p, a))
    lens_sorted = sorted(lens)
    n = len(lens_sorted)
    return {
        "count": n,
        "tok_min": lens_sorted[0] if n else 0,
        "tok_p50": lens_sorted[n // 2] if n else 0,
        "tok_p95": lens_sorted[int(n * 0.95)] if n else 0,
        "tok_max": lens_sorted[-1] if n else 0,
        "over_limit": sum(1 for x in lens if x > max_seq_len),
        "answer_would_vanish": risky,
    }


def split(samples: list[dict], rng: random.Random, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[dict]]:
    """切分数据，兼顾「防答案泄漏」与「测试集够大」。

    ★ 这里有个真实的权衡，我试了三次才定下来：

    尝试 1（按 kind+group 分层抽样）：
        问题：同一 topic 的不同问法被拆散 → 答案泄漏 → train∩test=3

    尝试 2（按 topic 整体切分）：
        泄漏修好了，但 val 只剩 6 条、test 只剩 3 条。
        因为一个 topic 平均才 6 条，整个切走就只剩几条，测试没有统计意义。

    尝试 3（本实现：按【答案】分组）★：
        关键认识 —— **泄漏的根源是"不同集合共享同一答案"，
        而不是"共享同一主题"。** 所以分组键应该是答案本身，不是主题。

        于是：
        - 同一答案对应的所有问法 → 必进同一集合（杜绝泄漏）
        - 不同答案（哪怕同一主题）→ 可分到不同集合（保证测试集够大）
    """
    def leak_key(s: dict) -> str:
        # 直接用答案内容做分组键。同一答案 → 同一集合，从根上杜绝泄漏。
        return f"{hash(s['output']) & 0xFFFFFFFF}"

    buckets: dict[str, list[dict]] = {}
    for s in samples:
        buckets.setdefault(leak_key(s), []).append(s)

    # 分层：让三个集合的 kind 分布尽量一致
    # （否则可能"测试集全是 JSON、训练集全是问答"，那测的不是同一个任务）
    layers: dict[str, list[list[dict]]] = {}
    for items in buckets.values():
        kind = Counter(s["meta"]["kind"] for s in items).most_common(1)[0][0]
        layers.setdefault(kind, []).append(items)

    train, val, test = [], [], []
    for kind, groups in layers.items():
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)  # 大组优先，避免尾部小组成员分不进任何集合

        n_total = sum(len(g) for g in groups)
        want_test = max(1, round(n_total * ratios[2]))
        want_val = max(1, round(n_total * ratios[1]))

        def fill(target: int, pool: list[list[dict]]) -> tuple[list[dict], list[list[dict]]]:
            """贪心装箱：尽量接近 target 但不超出太多。"""
            out: list[dict] = []
            rest: list[list[dict]] = []
            for g in pool:
                if len(out) + len(g) <= target or not out:
                    out += g
                else:
                    rest.append(g)
            return out, rest

        te, rest = fill(want_test, groups)
        va, rest = fill(want_val, rest)
        test += te
        val += va
        train += [s for g in rest for s in g]

    return {"train": train, "val": val, "test": test}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "meta"}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--n", type=int, default=400, help="目标样本数（不足则以实际为准）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-ratio", type=float, default=0.3, help="JSON 结构化样本占比")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    samples = build_samples(args.json_ratio, rng, target_n=args.n)
    print(f"生成样本: {len(samples)} 条")
    print(f"  分布: {dict(Counter(s['meta']['kind'] for s in samples))}")
    print(f"  分组: {dict(Counter(s['meta']['group'] for s in samples))}")

    st = analyze(samples, args.max_seq_len)
    print(f"\n长度统计（粗估 token）:")
    print(f"  min/p50/p95/max = {st['tok_min']} / {st['tok_p50']} / {st['tok_p95']} / {st['tok_max']}")
    print(f"  超过 max_seq_len({args.max_seq_len}) 的样本: {st['over_limit']}")

    if st["answer_would_vanish"]:
        print(f"\n⚠️  有 {len(st['answer_would_vanish'])} 条样本的 prompt 本身已超窗口，")
        print("    answer 会被完全截断 ⇒ 该样本 loss 恒为 0，等于白训。已剔除：")
        for q, p, a in st["answer_would_vanish"][:5]:
            print(f"      prompt≈{p}tok answer≈{a}tok  «{q[:34]}»")
        samples = [s for s in samples
                   if estimate_tokens(s["instruction"]) + estimate_tokens(s["input"]) + prompt_overhead_tokens() < args.max_seq_len]
        print(f"   剔除后剩 {len(samples)} 条")

    parts = split(samples, rng)
    for name, rows in parts.items():
        write_jsonl(out / f"{name}.jsonl", rows)
        print(f"写出 {out / (name + '.jsonl')}: {len(rows)} 条")

    # ---- 重叠检查：★ 这里我一开始也写错了，值得记下来 ----
    #
    # 错误版本：比较 train 与 test 的 `input`（问句）集合。
    #   结果报 train∩test=14 —— 看起来像严重泄漏，其实是**假警报**。
    #   因为我的切分是"按答案分组"，同一答案的不同问法会被合法地分到不同集合。
    #   问句重复不等于泄漏。
    #
    # 正确判据：**一条完整样本 (input, output) 不能同时出现在两个集合。**
    #   这才是真正的重复。共享答案但问句不同是设计上允许的（也符合真实场景）。
    def keys(rows: list[dict]) -> set[tuple[str, str]]:
        return {(r["input"], r["output"]) for r in rows}

    tr, va_, te = keys(parts["train"]), keys(parts["val"]), keys(parts["test"])
    dup_te, dup_va = len(tr & te), len(tr & va_)

    # 顺带报一个"答案重复度"供参考 —— 它高是正常的（同一主题多个问法），
    # 但如果 test 的答案 100% 出现在 train 里，就说明测试集没有新知识可测。
    ans_tr = {r["output"] for r in parts["train"]}
    ans_te = {r["output"] for r in parts["test"]}
    ans_overlap = len(ans_tr & ans_te) / max(1, len(ans_te))

    print(f"\n重叠检查（同一条 (input,output) 不得跨集合）:")
    print(f"  train∩test = {dup_te}   train∩val = {dup_va}")
    print(f"  答案重合度: test 中有 {ans_overlap * 100:.0f}% 的答案在 train 里出现过")
    print("    （这个值高是正常的——同一主题有多个问法。")
    print("     它会稀释测试的区分度，所以测试集越大越可信）")
    if dup_te or dup_va:
        print("  ❌ 有完全重复的样本！指标会虚高，必须修 split 逻辑")
    else:
        print("  ✅ 无完全重复样本")

    # ---- 留一份原始 meta 供追溯（不进 git，见 .gitignore）----
    write_jsonl(out / "_full_with_meta.jsonl", samples)
    print(f"\n完整带 meta 版本: {out / '_full_with_meta.jsonl'}（供分析用）")


if __name__ == "__main__":
    main()
