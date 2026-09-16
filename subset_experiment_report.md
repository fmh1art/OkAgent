# Baseline / LO-PH / Hydra：job01 1k 子集实验报告

实验日期：2026-09-16  
实验目的：在启动全量实验前，以小数据集验证 Qwen proxy、真实标注预算、三种方法的端到端流程和结果产物。

## 1. 结论摘要

本轮在 `job01` 的 1,000 人子集上完成了 baseline、LO-PH 和 Hydra 三组真实 API 实验，每组预算上限均为 200 次标注请求。

| 方法 | API 调用 | 选中人数 | TP | FP | FN | TN | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 200 | 317 | 215 | 102 | 167 | 516 | 56.28% | 67.82% | 61.52% |
| LO-PH | 185 | 589 | 321 | 268 | 61 | 350 | **84.03%** | 54.50% | **66.12%** |
| Hydra | 196 | 270 | 181 | 89 | 201 | 529 | 47.38% | 67.04% | 55.52% |

LO-PH 在本子集上的 recall 和 F1 最高，但该结果不能直接证明“LO-PH + Qwen”最好：LO-PH 的最终 physical proxy 未能稳定复用预计算 Qwen 特征，最后回退到了数据库内原有的 2,048 维分段均值向量。baseline 和 Hydra 使用的是 Qwen3-Embedding-0.6B 的 256 维特征。因此，这轮首先是端到端验收和问题定位，尚不是严格同特征后端的最终横向结论。

## 2. 实验设置

- 数据：`hiring_job01_1k_segvec.db`，共 1,000 名候选人。
- 金标准：1,000 人均有离线评测标签；正例 382 人、负例 618 人，正例率 38.2%。金标准只在实验完成后用于评测，不参与训练、阈值选择或在线标注。
- 岗位：`job01`，图片生成算法实习生。
- 标注服务：火山方舟 OpenAI-compatible API，模型 endpoint `ep-20250612104210-ss27q`。API key 仅保存在服务器忽略提交的 `_config/llm.json` 中，没有写入代码或报告。
- 计算环境：服务器 CPU 部署，无 GPU。
- 统一预算：每个方法最多 200 次真实标注请求，分别使用独立的 SQLite 调用账本。
- Qwen 配置：`Qwen/Qwen3-Embedding-0.6B`、256 维、最大 128 tokens、每段最多 1,000 字符、batch size 32。
- 预计算：1,000 人共 32 个 batch，约 3 分钟；缓存位于服务器 `/home/mengsq/.cache/okagent/qwen/`。
- 随机种子：42。

子集类别分布与 full 数据可能明显不同，因此这些绝对指标不能外推为全量效果。

## 3. 实验执行方式

### 3.1 预计算共享 Qwen 特征

```bash
export OKAGENT_QWEN_CACHE=/home/mengsq/.cache/okagent/qwen
export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export TOKENIZERS_PARALLELISM=false

.venv/bin/python benchmarks/run_full.py precompute \
  --job job01 --dataset 1k --max-calls 200 \
  --run-dir results/subset-qwen-v1-precompute-job01
```

预计算不调用标注 API。baseline、Hydra 和按约定实现的 LO-PH physical proxy 应复用同一个特征缓存。

### 3.2 启动三方法矩阵

```bash
.venv/bin/python benchmarks/run_matrix.py \
  --tag qwen-1k-v1 --jobs job01 --dataset 1k \
  --max-calls 200 --max-parallel 3
```

每个方法的 run-dir、输出、日志和标注账本相互隔离。矩阵首次执行时，baseline 正常完成；Hydra 在一次模型返回错误 candidate ID 后退出；LO-PH 随 SSH 前台会话断开而中止。两项中断任务随后在原 run-dir 中续跑，成功标签由 SQLite 缓存复用，没有重复支付前面的成功调用。

### 3.3 各方法实际流程

#### baseline

baseline code agent 生成并执行了 Qwen 特征 + balanced logistic regression 流水线。共标注 200 人，其中固定验证集 40 人；在验证集上选择满足 recall ≥ 0.9 的最大阈值，最终阈值为 `0.45686549093028944`。验证集观测 precision 0.80、recall 0.9091，但全库金标准 recall 仅 0.5628，说明 40 人验证集方差较大。

#### Hydra

Hydra 走固定实现而不是让 agent 临时生成训练代码：

- 用 Qwen query/candidate 相似度初始化主动学习顺序；
- 128 个训练标签拟合带 L2 正则的 logistic regression；
- 72 次重要性校准抽样估计 target-recall 阈值；
- 校准为有放回抽样，重复候选人直接命中缓存，所以最终只产生 196 次唯一 API 调用；
- 最终阈值为 `0.3379347180129763`。

Hydra 内部校准估计 recall 为 0.9468，而完整金标准 recall 为 0.4738，表明当前 1k 分布上的重要性校准/概率排序存在明显失配，需要在全量前继续诊断。

#### LO-PH

LO-PH 由 logical agent 安排 partition、sample、label、proxy 和 deploy，再由独立 physical agent 实现各算子。续跑后的最终 proxy 使用 40 个训练样本（7 正）和 40 个验证样本（9 正）；整个工作区累计有 185 个成功缓存标签，部署时这些标签覆盖模型预测。

最终 physical proxy 报告 Qwen 特征加载失败，回退到数据库原有 2,048 维分段平均特征，使用 balanced L2 logistic regression（`C=10`）。验证阈值为约 `0.0011`，验证集 recall 1.0、precision 0.3214；全库缓存覆盖后选中 589 人。高 recall 的代价是较多 FP，precision 为 54.50%。

## 4. 代码改动位置

### Qwen CPU proxy

- `src/okagent/qwen_proxy.py`
  - 新增 Qwen3-Embedding-0.6B CPU 特征提取；
  - 对岗位描述和候选人分段文本做截断、批处理和归一化；
  - 生成 256 维特征；
  - 使用原子写入的共享 `.npz` 缓存，缓存 key 包含数据库和配置身份。
- `src/okagent/prompts.py`
  - baseline prompt 明确优先使用小型 Qwen CPU embedding，而不是 TF-IDF/自回归式传统 proxy；
  - 要求缓存复用、验证集隔离、阈值选择和真实标签覆盖。
- `src/okagent/lo_ph_agent.py`
  - LO-PH 的 proxy contract 和 physical prompt 增加相同的 Qwen、缓存、泄漏防护与部署一致性要求。

### Hydra live

- `other_methods/hydra/hiring.py`
  - 新增 `feature_backend="qwen"` 路径，加载同一份 Qwen 特征和 query vector；
  - 将特征元数据写入 `hydra_summary.json`。
- `other_methods/hydra/method.py`
  - 新增 `Config.hiring_cpu(...)`；
  - 使用 L2 正则、并行标签请求、固定随机种子和 target recall 0.9；
  - 小预算下按约 64%/36% 自动拆分训练和校准预算，本轮为 128/72；2,000 次预算仍保留最多 1,280 个训练标签。

### 实验入口与可复现性

- `src/okagent/data.py`
  - `prepare(...)` 新增 `dataset` 和 `max_calls` 参数；
  - 支持显式选择 `full`、`1k` 等数据库变体；
  - 把数据变体和预算写入 `task.json`。
- `benchmarks/run_full.py`
  - 统一 precompute、baseline、LO-PH、Hydra 四种运行入口；
  - 支持 `--dataset`、`--max-calls`，Hydra live 使用真实 `SemanticOperator`。
- `benchmarks/run_matrix.py`
  - 并发运行 job × method 矩阵；
  - 为每项建立独立目录和日志；
  - 记录源码 commit、运行时间、return code 和评测结果。
- `src/okagent/semantic.py`
  - 对格式错误、candidate ID 错误或非布尔结果最多重试两次；
  - 每次尝试仍真实计入预算，HTTP 错误不做隐藏重试；
  - 成功标签继续跨进程持久化缓存。
- `tests/test_qwen_proxy.py`、`tests/test_hiring.py`、`tests/test_agent.py`、`other_methods/hydra/tests/test_method.py`
  - 增加缓存、子集、预算拆分、Qwen backend 和异常响应重试测试。

对应主要提交：

- `e878b4e`：Qwen proxy 与 Hydra CPU profile；
- `984e0c5`：CPU 编码参数调优；
- `b9d38fc`、`8b2dd7d`：实验矩阵和指标 manifest；
- `462c07d`：1k 子集入口与小预算拆分；
- `5a03a6d`、`b233d75`：异常标签响应重试及测试。

## 5. 异常、限制与解读边界

1. **LO-PH 最终没有使用 Qwen 特征。** Physical agent 两次自行指定局部 cache 目录，重复执行了 CPU 编码；最终又因接口使用错误回退到已有 2,048 维向量。因此本轮 LO-PH 与另两项不是严格同后端比较。
2. **LO-PH 是续跑结果。** 首次运行已缓存 120 个标签，续跑又产生新标签，最终 185 次；最终 proxy 只使用续跑计划中的 40/40 训练/验证划分，其余缓存主要在 deploy 阶段覆盖预测。
3. **Hydra 曾遇到一次无效模型响应。** endpoint 返回了错误 candidate ID。该请求仍计费，修复后从已有缓存续跑并完成；这促成了结构错误的预算内重试机制。
4. **验证集很小。** baseline 和 LO-PH 的阈值判断只基于约 40 个验证标签，观测 recall 与完整金标准 recall 差距较大。
5. **仅一个岗位、一个子集、一个随机种子。** 当前排名不应解读为方法的稳定总体排名。

## 6. 服务器结果位置

```text
/home/mengsq/projects/OkAgent/results/qwen-1k-v1-baseline-job01/
/home/mengsq/projects/OkAgent/results/qwen-1k-v1-lo_ph-job01/
/home/mengsq/projects/OkAgent/results/qwen-1k-v1-hydra-job01/
/home/mengsq/projects/OkAgent/results/matrix-qwen-1k-v1-logs/
```

每个完成目录中的 `evaluation.json` 是上表指标的直接来源；`workspace/output/semantic.sqlite` 是实际 API 调用和成功标签的审计账本。

## 7. 全量前建议

1. 把 Qwen 特征加载从 LO-PH physical agent 的自由生成代码中移到受控 operator，实现层强制注入共享特征路径，禁止静默 fallback。
2. 用全新的 run-dir 重跑 1k LO-PH，确认三项的 `features.backend` 都是 `qwen`，再做严格横向比较。
3. 为阈值选择增加更大的独立校准集或多次随机划分，重点检查 Hydra 的估计 recall 与真实 recall 偏差。
4. 小子集验收通过且上述问题修正后，再启动 full 数据；full 结果应使用新的独立 2,000 次账本，不能复用本轮标签。
