# 招聘机器学习代理模型真实数据样例

## 简单 baseline：一次训练/测试划分

`baseline.py` 参考 `smoke_test.py`，直接读取岗位 01 全量 DuckDB，使用全部 34,761 名候选人。
按候选人分层划分 80% 训练集、20% 测试集（seed=42），使用 `unstructured` 的 2048 维向量，
L2 归一化后训练 `LogisticRegression(class_weight="balanced", max_iter=2000)`。
只在测试集计算指标，不使用交叉验证，不调用模型 API。

```bash
python baseline.py --data-dir /home/mengsq/datasets/20260722
```

默认保存到脚本目录内新的 `runs/baseline_时间戳/`。也可用 `--output-dir` 指定一个尚不存在的目录。
输出 `metrics.json`（指标）、`split.csv`（全部候选人的划分）、`test_predictions.csv`（测试集预测）、
`model.joblib`（包含归一化的完整模型）。指标对照的是历史 LLM 标签。

## 自主 Agent：岗位 01 全量实验

使用 `operator_agent.py --full-data-dir /home/mengsq/datasets/20260722` 启动新版 Agent。
Planner 每次选一个算子，Coder 执行一次后返回结果；支持自定义/重复算子和自主模型选择，
没有固定流程和最大轮数。使用验证集迭代，结束后单独评估测试集。
详见 [FULL_EVALUATION.md](FULL_EVALUATION.md)，每轮轨迹保存为 `trajectory.jsonl` 和 `rounds/`。

这是从 `job01` 招聘数据集中提取的一个小型、自包含子集，用于流水线冒烟测试。

## 内容

- `full/full_00001.jsonl`：全部 3000 名抽样候选人。
- `train/train_00001.jsonl`：用于模型拟合的 2400 名候选人。
- `test/test_00001.jsonl`：用于冒烟评估的 600 名候选人。
- `embeddings/embeddings_00001.parquet`：所有被引用的 2048 维分段嵌入向量。
- `job_description.json`：对应的职位描述。
- `llm_prompt.txt`：渲染后的匹配提示词。
- `smoke_test.py`：验证连接、哈希与数据集划分，并使用 `unstructured` 嵌入训练一个最小逻辑回归分类器。
- `metadata.json`：机器可读的抽样与数据结构信息。
- `MANIFEST.sha256`：除清单文件自身以外，包内所有文件的校验和。

## 样本设计

- 抽样种子：`20260903`。
- 标签：正样本 33 条，负样本 2967 条。
- 源数据标签：正样本 382 条，负样本 34379 条（正样本率为 1.098933%）。
- 抽样后的正样本率：1.100000%；为保持源数据的正样本比例，使用了最接近的整数样本数。
- 数据集划分：训练集 2400 条，测试集 600 条，并按标签进行分层抽样。
- 按候选人 ID 的 SHA-256 排序进行确定性选择。
- 候选人 ID、标签、分段文本和嵌入值均从源数据快照原样复制，未作修改。

该样本在整数取整精度下保持了源数据的类别比例。由于正样本数量较少，训练集和测试集各自的类别比例可能略有差异。此数据集仍然只是一个小型冒烟测试样本，因此不得将其指标作为模型质量指标报告。

## 快速检查

在当前目录中，确保已安装 `duckdb`、`numpy` 和 `scikit-learn`，然后运行：

```bash
python smoke_test.py
```

## 逐算子 ReAct Agent

`operator_agent.py` 实现了一个 Planner-Coder-Executor 循环：

1. Planner 使用 `dataset_profile` 和 `inspect_schema` 只读工具了解数据；
2. Planner 每轮只输出一个 `<operator>{...}</operator>`；
3. 独立 Coder Agent 为该算子输出 `<code>...</code>` Python 脚本；
4. Executor 校验并运行脚本，将 JSON observation 写回 Planner 的 `context.history`；
5. Planner 根据执行结果生成下一个算子，直到完成 `Deploy`。

原子集模式由 Planner 自主安排算子，生成代码通过 AST 检查后在独立子进程中运行，并设有超时；这不是操作系统级安全沙箱。允许标准库以及 DuckDB、NumPy、scikit-learn、joblib。`Label` 复用已有 LLM 标签。

全量模式可不调用模型 API，使用固定算子完成真实训练和评估：

```bash
python operator_agent.py --full-data-dir /home/mengsq/datasets/20260722 --offline --verbose
```

如需让真实模型分别担任 Planner 和 Coder，请安装 SDK 并设置 API Key：

```bash
pip install -r requirements.txt
$env:OPENAI_API_KEY="your-key"
python operator_agent.py --model gpt-5-mini --verbose
```

兼容 OpenAI Responses API 的其他服务可以通过 `--base-url` 与对应模型 endpoint 运行：

```bash
$env:ARK_API_KEY="your-key"
python operator_agent.py --base-url https://ark.cn-beijing.volces.com/api/v3 --model your-endpoint --verbose
```

运行单元测试：

```bash
python -m unittest test_operator_agent.py
```

## 数据处理

此数据包包含少量真实简历或候选人档案，以及内部招聘背景信息。数据包不包含 API 凭证和绝对源路径，但仍可能包含个人信息。请仅向符合内部数据处理政策、获得相应授权的接收者共享；不要将其上传至公共代码仓库或外部服务。
