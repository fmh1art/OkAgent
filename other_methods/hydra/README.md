# Hydra 招聘筛选方法

按 `hydra-master(1).zip` 中 **`benchmark/hiring/config.ini` 的实际默认配置**复现：
`ml → active_learning → LR → target_recall_threshold`。直接调用 Python 函数，无 CLI 或服务。

## 使用

依赖：`uv pip install --python .venv/bin/python -r other_methods/hydra/requirements.txt`。
在 OkAgent 项目根目录运行：

```python
from okagent.data import prepare
from okagent.evaluation import evaluate
from other_methods.hydra import run_hiring, Config

prepare("_config/hiring.json", "results/hydra-job01", job="job01")
result = run_hiring("results/hydra-job01")
report = evaluate("results/hydra-job01")
```

默认与压缩包的 `mock_llm=true` 一致：只在采样请求时向 oracle 查询历史 LLM 标签，
没有真实 API 开销。`usage.json` 的 `llm_calls` 在 replay 模式是模拟请求数。
训练默认最多 1024 人，校准最多 512 次抽样；每请求默认 1 人，总请求上限读取工作区预算。
可用 `Config(batch_size=8, target_recall=0.9)` 调整；batch 算一次请求，同时记录实际标注人数。

若要实际调用 LLM，传入 `label(ids) -> [0/1, ...]` 回调，每次回调最多一次请求，禁止回调内部偷偷重试。
回调自行读取 `workspace/job.json`、`label_prompt.txt` 和请求 ID 的简历文本。每次调用前计数，
失败立即抛错，不自动重试，也不输出本次成功结果。方法不内置 API 密钥或内部服务地址。
也可复用主项目的 `SemanticOperator.label`，保持 `batch_size=1`；环境变量及接入示例见
[根目录 README](../../README.md#5-复现-other_methodshydra)。

## 算法与替换

| 模块 | 实现 |
| --- | --- |
| `method.py` | L2 归一化、主动采样、逻辑回归、Cohen κ 停止条件、预算内调用 |
| `calibration.py` | 经验 CDF、带放回重要性采样、缓存修正、recall 阈值及可选置信修正 |
| `hiring.py` | DuckDB 数据与 oracle 适配，写入现有评估格式 |

首轮和单类标签阶段：原算法选择 query 相似度最高/最低各一半。压缩包没有填写 embedding 模型名；
我们的简历库记录为 `doubao-embedding-vision-251215`、2048 维。当前按用户要求改为随机采样。
传入同模型的 `query_vector` 即恢复相似度方式，运行记录明确标明 `initialization`。

有两类标签后：10% 随机探索；少数类占比低于 0.35 时，按“不确定性 + 0.75 × 少数类概率”
补充少数类候选，最多占当前批次的 80%；剩余样本优先选分类边界附近的人。
每批重新训练。最近 3 个有效 Cohen κ 的均值大于 0.995 时提前停止；常量预测导致的未定义 κ 不计入。

校准：对全库模型分数做保留并列的 mid-rank CDF；训练标签作为精确质量纳入，
在其余候选人上按 `0.9 × sqrt(CDF) / sum(sqrt(CDF)) + 0.1 / N_remaining` 有放回抽样。
重复抽到同一人只标注一次，但估计时保留抽样次数与逆概率权重。
按原代码加入已知正例覆盖模型负判的质量修正，选择满足目标 recall 的最大阈值。
校准结束不重新训练；最终已标注正负例均覆盖 proxy 预测。

`mlpack` 的 LR 用 NumPy 手写损失/梯度、SciPy L-BFGS 替代，保持
`sum(logloss) + lambda/2 * ||w||²` 且不惩罚截距，目标定义见
[mlpack 4.7.0 源码](https://github.com/mlpack/mlpack/blob/4.7.0/src/mlpack/methods/logistic_regression/logistic_regression_function_impl.hpp)。
随机数生成器及优化器与 C++ 不同，因此不承诺逐行相同的抽样和浮点结果。
内部 `bytedance.hydra`/`bbthread` 运行依赖以直接函数、DuckDB 和串行回调替代。

保持原配置的行为：只有单类训练标签时返回该常量类别；校准失败回到默认 LR 阈值 0.5。
目标 recall 是估计目标，不保证真实全库一定达到；默认 `failure_probability=-1` 不作置信修正。
README 提及的 `two_threshold_cascade`/SCJC、RF 和 SQL 引擎均不是实际默认路径，本实现不冒充这些分支。
这条单阈值路径在部署阶段不额外调用 LLM。

## 来源与产物

压缩包版本：`22d3873c4d7e47f7fdb08508b9ce6661b787f213`。
对应源码：`src/semantic_operator/transforms/semantic_filter_transform_blocking.cpp`
（SelectNextActiveLearningBatch、finalizeMLTrainingActiveLearning、calibrateMLTargetRecall）和
`src/semantic_operator/model/logistic_regression_model.cpp`。原压缩包保持不变。

输出仍是 `workspace/output/candidate_ids.json`、`usage.json`，另保存
`hydra_summary.json`（参数、训练/校准索引、阈值、轮次）和 `hydra_model.npz`（LR 参数或单类常量）。
LR 参数前 D 项是权重，末项是截距；推理前同样对每人的向量做 L2 归一化。
金标准只用于 oracle 与事后评估，不作为模型输入。需要每人有 unstructured 向量，缺失时明确报错。

测试：`.venv/bin/python -m pytest other_methods/hydra/tests -q`。

已验证 job01 全库，seed=42、随机首轮、默认配置：34,761 人，928 个训练标签，
503 个新增校准标签，共 1,431 次模拟请求；筛出 755 人，recall=65.71%、precision=33.25%。
结果位于 `results/hydra-job01/evaluation.json`，属于离线 replay，不是真实 LLM 调用结果。
