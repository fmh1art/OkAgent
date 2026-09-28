# job01 深化方法：尝试与结果总结

**整理日期：** 2026-09-28  
**代码分支：** `codex/job01-multi-proxy-router`  
**适用数据集：** job01

## 1. 结论摘要

本轮工作围绕“简历切分、LLM 批量标注、多标签 proxy、多个 proxy 的 router”完成了一套可运行、可恢复、可审计的实验框架，并通过本地单元测试和合成 DuckDB 端到端测试验证。当前实现已经能够：

- 把一份简历按候选人维度聚合为 global、experience、credentials 三类信息；
- 用一次 LLM 请求批量标注多位候选人，并同时返回 7 个连续维度标签；
- 分别训练 3 个 section proxy；
- 使用 OOF stacking router 或 embedding router 组合多个 proxy；
- 采用 disagreement-driven acquisition 选择下一批最值得标注的数据；
- 严格排除训练样本后计算 AP、P@R80、P@R90，并进行多随机种子 recall 校准；
- 实验结束后自动写出 JSON 和 Markdown 报告，无需持续监视进程。

实现验证结果为 `47 passed, 2 skipped`，完整合成端到端流程通过。但由于服务器 `mengsq@10.77.110.188` 的 SSH 非交互认证始终失败，本轮没有真正启动 job01 服务器实验，因此目前不能报告新方法在 job01 上的正式 AP、P@R80 或 P@R90，也不能声称其优于 USA 基线。

## 2. 出发点与已有基线

已有项目进度记录显示，job01 包含 34,761 名候选人，其中历史正例 382 个，正例率约 1.10%。已有结果可作为后续实验的参照，但它们不是本轮新方法产生的结果。

| 方法 | Unsampled AP | P@R80 | P@R90 | 说明 |
|---|---:|---:|---:|---|
| USA | 0.4798 | 5.80% | 2.98% | 现有 proxy 基线 |
| Jevons 直接 LLM | 0.6251 | 28.01% | 9.87% | 已有直接 LLM 结果 |
| Doubao 直接 LLM | 0.5642 | 45.13% | 44.40% | 已有直接 LLM 结果 |
| Qwen-Doubao 严格 unsampled | 0.4414 | 10.06% | 4.94% | 已有蒸馏结果 |

已有 Qwen-Doubao 方案暴露出两个值得优先处理的问题：简历分块的 max aggregation 可能被单个高分片段支配；单一或重复标签也不足以支持更细粒度的 proxy 学习。本轮设计主要针对这两个问题。

## 3. 尝试一：候选人级简历切分与三个 section proxy

### 3.1 方法

将原始简历 section 固定映射为三组：

| Proxy | 输入 section | 目标 |
|---|---|---|
| global | `unstructured` | 捕获整份简历的综合匹配信息 |
| experience | `projects`、`work` | 捕获项目与工作经历证据 |
| credentials | `education`、`awards`、`languages`、`applications` | 捕获教育、荣誉、语言等资质证据 |

每个候选人的同组 section embedding 先做均值池化，再进入训练。这样一个候选人在一个 expert 中只对应一行和一次损失，避免 section 多的候选人被隐式重复加权。

每个 section 训练一个独立的逻辑回归 proxy，得到 global、experience、credentials 三个基础评分。

### 3.2 验证结果

- 候选人级池化、缺失 section、单类标签降级路径均已有测试覆盖。
- 三个 expert 能在完整端到端合成数据上训练、保存并生成候选人分数。
- 测试发现并修复了一个实际 router 问题：候选人缺少 experience section 时，原权重总和可能小于 1；现在只在可用 expert 上重新归一化，最终权重和恒为 1。
- 每位候选人的 expert 原始分数、可用性、router 权重和融合分数都会被记录，便于后续分析错误样本。

### 3.3 尚未得到的结果

因为 job01 正式运行未启动，尚不能判断三个 section proxy 中哪一个单独效果最好，也不能量化候选人级 mean pooling 相对原 max aggregation 的提升。

## 4. 尝试二：LLM batch prompting 与多维标签

### 4.1 方法

Batch prompting 不只是并发发送单样本请求，而是把多位候选人放入同一次请求，共享职位描述和系统提示词，要求模型按候选人返回结构化结果。同时，多个 batch 请求可以并发执行，而结果写入 SQLite 时保持串行事务，避免并发写损坏。

每位候选人生成以下 7 个 0 到 1 的连续标签：

1. `overall`
2. `education`
3. `experience`
4. `technical_skills`
5. `projects`
6. `research`
7. `evidence_quality`

另外记录 `missing_requirements`，用于保留硬性条件缺失的解释信息。候选人的内部稳定 ID 在 prompt 中会替换为 `candidate_000` 等批内化名，降低不必要的标识符泄露。

连续 teacher 标签通过 class-balanced soft BCE 训练。实现上把一个软标签等价展开为带权正、负两行，从而复用成熟的逻辑回归训练流程，并保留连续标签中的置信度信息。

### 4.2 可靠性与成本记录

- 所有请求、响应、状态和累计尝试次数都写入可恢复 SQLite ledger。
- 中断后可继续运行，不需要重标已经成功的候选人。
- malformed JSON、上下文过长等批次失败会自动重试，并可二分批次定位问题样本。
- 成本统计包含成功请求、格式错误但已经计费的请求、失败父批次以及跨恢复运行的历史尝试，避免只统计最终成功调用。
- 支持多轮累计标注，并检测同一候选人不同轮次间的冲突。

### 4.3 当前结果

批量标注解析、重试、恢复、失败拆分、跨轮累计和成本核算均通过本地测试。尚未在 job01 上实际调用 teacher，因此没有可报告的真实 token 节省率、请求数量、标注吞吐或标签分布。

## 5. 尝试三：多个 proxy 的 router

### 5.1 OOF stacking router

Stacking router 的输入包括三个 expert 概率和各 section 是否存在的标志。训练 router 时，每个训练候选人的 expert 分数都来自 out-of-fold 预测；router 训练完成后才用全量训练集拟合最终 expert。这样避免 router 读取 expert 对自身训练样本的过拟合分数。

### 5.2 Embedding router

Embedding router 使用 global 简历 embedding 预测“哪个 expert 对当前候选人更可靠”，再只对该候选人实际存在的 expert 归一化权重并融合评分。这条路线允许不同类型的简历采用不同 proxy，而不是所有候选人共享固定权重。

### 5.3 三分数直接组合

实现同时保留单个 expert 分数与简单平均结果，作为最低复杂度对照组。正式实验将比较：

- global proxy；
- experience proxy；
- credentials proxy；
- 三 proxy 简单平均；
- OOF stacking router；
- embedding router。

### 5.4 当前结果

两个 router 均已通过合成端到端测试，模型持久化和逐候选人路由诊断也已实现。由于没有 job01 正式结果，当前不能判断动态路由是否优于简单平均，也不能确认它在 P@R80/P@R90 上是否有稳定收益。

## 6. 尝试四：disagreement-driven 批量选样

为提高新增 LLM 标签的价值，实现了一个固定预算的混合 acquisition 策略，候选来源包括：

- 当前模型高分样本；
- 决策边界附近样本；
- 三个 expert 之间分歧最大的样本；
- stacking router 与 embedding router 分歧最大的样本；
- 随机覆盖样本。

选样过程保证最终候选人唯一且精确满足预算，并且不读取训练集之外的历史评估标签。这样可以分轮运行：先用已有 2,000 个标签训练，再挑选一批信息量高的候选人进行 batch prompting，随后累积重训。

当前只完成算法和测试验证，尚无 job01 上“每增加多少 teacher 标签可提升多少指标”的学习曲线。

## 7. 评估协议与防泄漏措施

本轮特别加强了实验协议，防止结果看起来提升但实际混入评估标签：

- 训练 ID 固定来自 JSON 或已有 teacher ledger，并记录来源与哈希。
- 所有训练 ID 从 strict-unsampled 指标中排除。
- 评分冻结前，只允许查询冻结训练 ID 的历史标签；batch teacher 模式下评分前甚至不需要打开完整历史 truth。
- 完整未采样 truth 只在所有候选人分数冻结后用于最终评估。
- router 使用 OOF expert 特征，禁止使用 in-sample expert 预测。
- 报告明确标注协议属于 `architecture_only` 还是可复现的 `usa_seed11_exact`。
- 提供 seed-11 ID 恢复清单工具；只有精确恢复 USA seed-11 训练 ID 后，才能声称与 USA 严格可比。

最终指标包括 unsampled AP、P@R80、P@R90。Recall 阈值会在 20 个随机种子上做经验校准，并额外提供基于 Clopper-Pearson 下界的保守 recall 约束版本，减少小校准集偶然达到 80%/90% recall 的风险。

## 8. 实现与测试结果

### 8.1 自动化测试

本地 UTF-8 模式运行结果：

```text
pytest -q
47 passed, 2 skipped
```

两个 skipped 为项目原有的可选集成路径。Windows 非 UTF-8 locale 曾导致一个原有测试用 GBK 解码 UTF-8 prompt；使用 `PYTHONUTF8=1` 后得到上述结果。

### 8.2 合成端到端测试

合成 DuckDB 端到端测试已覆盖：

- section 数据读取；
- 候选人级 pooling；
- 三个 expert 训练；
- 两种 OOF router；
- strict-unsampled 排除；
- 多随机种子 calibration；
- 模型保存；
- 压缩分数输出；
- `report.json` 和 `report.md` 生成。

合成测试只证明流程能够正确运行，不代表 job01 的实际性能，因此合成指标没有写入正式结果表。

## 9. job01 正式实验结果

截至本总结完成时，新的 job01 实验状态如下：

| 训练标签 | 方法 | Unsampled AP | P@R80 | P@R90 | 状态 |
|---|---|---:|---:|---:|---|
| 已有冻结 2,000 IDs | 三 proxy + routers | — | — | — | 未启动 |
| Batch multi-label 2,000 IDs | 三 proxy + routers | — | — | — | 未启动 |
| 多轮 disagreement acquisition | 累积重训 | — | — | — | 未启动 |

服务器登录被多次尝试，但 `mengsq@10.77.110.188` 均返回 `Permission denied (publickey,password)`。由于无法建立非交互 SSH 会话，后台 launcher 没有被执行，服务器上也没有本轮进程或结果文件可供等待。因此这里保留空值，而不是用本地合成数据代替。

这意味着目前还不能回答以下关键实验问题：

- 三 proxy 或 router 是否超过 USA 的 AP 0.4798；
- P@R80、P@R90 是否显著改善；
- batch prompting 相对逐条请求节约了多少 token 和请求；
- 多维软标签是否优于已有单标签 teacher；
- disagreement acquisition 是否提高单位标注成本的收益。

## 10. 服务器恢复后的执行方式

代码已经推送到远端分支。服务器认证恢复后，可由服务器端直接执行一次后台启动命令：

```bash
cd /home/mengsq/projects/OkAgent &&
git fetch origin codex/job01-multi-proxy-router &&
git show origin/codex/job01-multi-proxy-router:benchmarks/launch_job01_router_server.sh >/tmp/launch_job01_router.sh &&
bash /tmp/launch_job01_router.sh
```

launcher 是幂等的：已有进程运行时不会重复启动，已有完整结果时不会覆盖。实验结束后重点读取：

```text
results/comparison/multi_proxy_router_qwen_ids_job01_r1/report.json
results/comparison/multi_proxy_router_qwen_ids_job01_r1/report.md
```

如需先确认是否能恢复 USA seed-11 的精确训练 ID，可运行：

```bash
python -m benchmarks.recover_job01_protocol \
  --root results/comparison \
  --output reports/protocol/job01_id_source_inventory.json
```

建议实验顺序为：

1. 先用已有 Qwen-Doubao 2,000 ID ledger 运行 architecture-only 对照，验证完整 job01 管线。
2. 恢复并冻结 USA seed-11 ID，运行严格可比实验。
3. 在相同预算下生成 batch multi-label 标签，比较单标签与多标签 teacher。
4. 追加 disagreement acquisition 轮次，记录累计标签数、调用成本和指标变化。
5. 把协议哈希、运行时间、峰值内存、请求/token 用量以及各方法指标回填到正式实验日志。

## 11. 主要代码与记录位置

- `benchmarks/multi_proxy_router_job01.py`：三 proxy、两种 router、校准和评估主流程。
- `benchmarks/batch_label_job01.py`：批量多维 teacher 标注与可恢复 ledger。
- `benchmarks/recover_job01_protocol.py`：seed-11 协议恢复清单。
- `benchmarks/launch_job01_router_server.sh`：服务器幂等后台启动器。
- `benchmarks/multi_proxy_router_job01.md`：运行参数与方法说明。
- `reports/job01_deeper_methods_experiment_log.md`：正式实验状态和待回填指标表。
- `tests/test_multi_proxy_router_job01.py`：核心 pipeline 测试。
- `tests/test_batch_label_job01.py`：batch prompting、恢复和成本核算测试。

## 12. 总体判断

本轮产出已经把最初的研究想法推进为可执行实验系统，尤其补齐了候选人级简历切分、真正的多候选人 batch prompting、多维连续标签、三个 expert、动态 router、主动选样和严格评估协议。现阶段最重要的结果是“方法已实现且管线验证通过”，而不是模型性能结论。

下一步的唯一关键依赖是恢复服务器执行权限并完成 job01 正式运行。只有获得真实 report 后，才能对 proxy 数量、路由方式、batch 标签和主动选样的收益作出数据层面的判断。
