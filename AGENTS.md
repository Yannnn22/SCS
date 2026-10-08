# AGENTS.md —— 给执行 Agent 的项目契约

> 把本文件放在项目根目录。Agent 每次开工先读这里，再读 `docs/plan.md`。
> 本文件是**约束**，不是建议。与本文冲突的实现一律不接受。

---

## 0. 你的角色

你在一台 Windows 电脑上工作，硬件是 **RTX 4060 Laptop，显存仅 8GB**，用户是正在复现一个
「智能电商风控与客服 Agent 平台」实习项目的学习者。用户的目的是**通过动手真正搞懂**，
不是为了尽快产出代码。所以：

- **宁可慢，不可糊**。任何"跑通了但我不知道为什么"的代码都是失败。
- 每个模块交付时必须附带：**为什么这么写** + **参数怎么调** + **实测数字**。
- **禁止编造数字**。跑不出结果就如实报"未测"，绝不允许写一个看起来合理的数字。

---

## 1. 环境铁律

1. Python 必须是 **3.11**。3.12 的 `milvus-lite` / `bitsandbytes` wheel 不齐。
2. `torch` 必须是 **CUDA 版**。每写一段涉及 GPU 的代码，先验证：
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   输出 `False` 就停下来报告，不要继续往下做。装成 CPU 版是最常见的隐性错误。
   **实测钉版（Windows 那台已装好）**：`torch==2.6.0+cu126` + 驱动 566.24。
   ⚠️ 官方各 CUDA 通道的 torch 版本上限不同（cu121 最高 2.5.1、cu124 最高 2.6.0、
   cu126 最高 2.14.1），而 `transformers 5.19` 要求 `torch>=2.5` ——
   所以要升 torch 必须同时换通道。另外 `bitsandbytes` 必须 `>=0.50.2`。
3. **不要在 WSL 里装 NVIDIA 驱动**，只用 Windows 侧驱动。WSL 里 CUDA 不可用
   99% 是 Windows 驱动过旧。
4. `milvus-lite` **在 Windows 原生可用**（2026-10 实测：建集合/写入/检索全部正常，
   HNSW 索引亦正常，归一化向量 20/20 命中、余弦相似度 1.000000）。
   若某台机器上装不上，只改一个配置项 `dense.backend: faiss`，禁止为此重构代码结构。
5. **显存只有 8 GB（不是 16 GB）** ★ 这是最容易搞错的一条：
   **RTX 4060 Laptop 只有 8GB 版本**。Windows 任务管理器里显示 "16GB" 是
   8G 专用 + 8G 共享系统内存，**共享部分 CUDA 用不了**。
   `torch.cuda.get_device_properties(0).total_memory` 实测 = **8188 MiB**。

   8 GB 下的能力边界：
   | 任务 | 结论 |
   |---|---|
   | Qwen3-4B **推理**（4bit） | ✅ 约 3–3.5 G，舒服 |
   | Qwen3-8B 推理（4bit） | ⚠️ 约 5.5–6 G，可跑但余量小 |
   | Qwen3-14B 推理（4bit） | ❌ 约 9.5 G，**必 OOM，不要试** |
   | Qwen3-4B **QLoRA 训练** | ⚠️ 边界。seq≤512 + bs1 + ga16 + grad ckpt，**必须先实测** |
   | Qwen3-1.7B QLoRA 训练 | ✅ 稳妥的降级方案 |
   | BGE-M3 embedding | 建议走 **CPU**（300–500 chunk 场景 CPU 足够） |
   | reranker | 用 `bge-reranker-base`，或与大模型**串行**加载，不要同时驻留 |

   参考实测：Qwen2.5-0.5B 4bit 单步训练峰值 **1.29 GB**。
   **任何"能不能装下"的结论都必须先实测，不许按公式推断。**
6. **HuggingFace 直连不通**（实测 20s 超时），镜像 `hf-mirror.com` 正常（0.75s）。
   所有 HF 相关操作都必须设 `HF_ENDPOINT=https://hf-mirror.com`，
   否则脚本会卡死在模型下载上。建议把它写进系统环境变量，别每次手敲。
7. **HF 缓存默认落在 C:**，而 C 盘余量紧张（实测仅剩 57.6 GB）。
   建议把 `HF_HOME` 指向 D 盘。动这个之前先跟用户确认。

---

## 2. 参数必须集中在 `configs/base.yaml`

**禁止把任何魔法数字写在代码里。** k1、b、top_k、chunk_size、阈值、超时、轮次，
全部从 yaml 读。原因：本项目的一切价值都来自「同代码 + 不同配置 = 可对照的消融实验」。
代码里出现硬编码参数 = 消融实验做不了 = 项目价值归零。

---

## 3. 禁止复现以下技术错误

这些是原实习文档里的硬伤，属于面试官会当场抓的点：

| 禁止 | 正确做法 |
|---|---|
| 把 Rerank 说成 "Cross-BERT" | 是 Cross-**Encoder**（bge-reranker-v2-m3） |
| 把 HitRate 说成召回率 | HitRate@K 是二值命中；Recall@K 需要每条 query 多个 gold doc，**两者分开实现** |
| Milvus 用 IVF_FLAT 却声称低延迟高并发 | 主线 **HNSW**（M=16, efConstruction=200, ef=64）；IVF_FLAT 只作对照实验跑一次 |
| Top50+Top50 融合后 50 条全送 Rerank | Rerank 是 O(n) 次前向。融合后取 **Top20–30** 送精排，最终 Top5–6 入 Prompt |
| 声称 LoRA 微调消除了幻觉 | SFT 改的是**输出格式/字段识别**，改不了知识。幻觉靠 RAG + 约束 |
| 微调前后用不同测试集 | **同一套固定测试集**，分维度报：格式合规率 / 意图准确率 / SQL 可执行率 |

---

## 4. 代码规范

- 目录结构严格遵循 `docs/plan.md` 第三节，不要自创新目录。
- 每个可执行模块都要有 `if __name__ == "__main__":` 的自检/demo 块，
  用**内置的少量中文样例**跑通，不依赖外部文件就能验证。
- 检索器统一接口：`retrieve(query: str, top_k: int) -> list[tuple[chunk_id, score]]`。
  所有检索器（bm25 / dense / hybrid / rerank）都实现这个签名，才能自由拼接做消融。
- **优雅降级**：重依赖缺失时不能让脚本崩。参考 `src/retrieve/bm25.py` 里
  jieba 缺失时降级为「单字 + 相邻二字」的写法。
- 中文注释，解释**为什么**，不解释**是什么**（`i += 1  # 自增` 这种注释不要）。
- 不引入未在 `requirements.txt` 里的新依赖；确实需要就先问用户。

---

## 5. 每轮交付的固定格式

```
## 交付
- 新增/修改文件：（逐个列出）
## 验证
- 我实际运行的命令：（原样贴出）
- 实际输出：（原样贴出，不要改写）
## 参数说明
- 关键参数及取值理由：
## 不确定的地方
- 如实列出没把握的点（禁止隐瞒）
```

**验收靠"实际输出"，不靠"我认为应该能跑"。** 没跑过就不要声称跑通了。

---

## 6. 必须停下来问用户的情况

1. 需要下载超过 1GB 的模型或数据
2. 需要安装新依赖
3. CUDA 不可用 / 显存 OOM，需要改配置降级
4. 需要改动 `docs/plan.md` 定下的目录结构或接口签名
5. 某个验收标准你无法达成
6. **git 操作超出第 7 节允许的范围**（尤其是要 force push 或改写已推送的历史）

---

## 7. Git 协议（两个 agent 必须严格遵守）

远程仓库：`https://github.com/Yannnn22/SCS.git`

### 核心约束

**这个项目有两台机器在跑同一个仓库**（Mac 负责规划/提示词，Windows 负责执行）。
两台机器的本地历史**没有共同祖先**（各自 `git init`），这是刻意的设计——不要试图"修复"它。

| 机器 | 推送分支 | 说明 |
|---|---|---|
| Mac | `main` | 已有首个提交，含规划文档与提示词 |
| Windows | `win/main` | **独立起点**，不要尝试与 `main` 建立祖先关系 |

`win/main` 是独立历史，**push 时不需要 `--force`**，也不要试图 rebase 到 `main` 上。
需要汇总时由用户在本机决定怎么合，**不要自作主张合并**。

### 每个 P 阶段结束时

```bash
git add -A
git diff --cached --stat          # ★ 先看这个，确认没有大文件、没有不该提交的东西
git commit -m "P<n>: <一句话说明这阶段做了什么>"
git push origin <你的分支>        # Mac: main   /   Windows: win/main
```

### 提交前必须自检（这条比提交本身重要）

```bash
# 1. 有没有大文件混进来？任何 >5MB 的文件都要停下来问用户
git diff --cached --name-only | xargs -I{} du -h {} 2>/dev/null | sort -rh | head -10

# 2. 模型/索引有没有被误加？下面这些路径一个都不该出现
git diff --cached --name-only | grep -E "\.safetensors|\.bin$|\.pt$|\.gguf|hf-cache|data/milvus|\.db$|models/"
# 有输出 = 立刻 git reset，检查 .gitignore
```

`.gitignore` 已经覆盖了 HF 缓存、模型权重、向量库、embedding 缓存、LoRA 适配器。
**如果你发现某个该忽略的路径没被忽略，先改 `.gitignore` 再提交，不要用 `git add -f` 绕过。**

### 允许的操作

- `git add` / `git commit` / `git push origin <自己的分支>`
- `git log` / `git status` / `git diff` / `git show`
- 创建自己的新分支

### 禁止的操作（需要先问用户）

- `git push --force` / `--force-with-lease`（任何情况下都不要）
- 改写已推送的历史：`rebase`、`commit --amend`（已推送的）、`reset --hard`（已推送的）
- 删除任何远程分支
- 合并 `main` 与 `win/main`
- 提交 `docs/env-report.md` 之外的秘密信息（API key、token、账号密码）
  ⚠️ **凭据绝对不入库**。如果发现某处硬编码了 key，立即报告并改用环境变量。

### 提交信息规范

```
P<n>: <做了什么>
```
例如：`P1: 语料整理与三种切分策略对照`、`P14: 1.7B QLoRA 基线训练与显存实测`

**不要把多个 P 阶段压成一个提交**——每阶段一个提交，
这样出问题时能单独回退到某个阶段。
