# OkAgent

在固定 LLM 查询预算下，从简历库中筛选符合岗位要求的候选人，重点提高 recall，同时兼顾 precision。

项目包含两部分：

- **主项目**：准备工作区和任务 prompt，由最基础的 mini-swe-agent 编写、执行采样、标注、proxy 训练及预测代码，最后统一评估。
- **复现方法**：`other_methods/hydra` 实现 Hydra 的主动采样、逻辑回归与 recall 阈值校准，可直接运行。
- **CPU Qwen proxy**：三个方法可复用 Qwen3-Embedding-0.6B 的缓存语义特征；Qwen 只提取特征，不充当标签 oracle。

所有入口都是 Python 函数。code agent 只有“模型 → bash → 观察结果”的循环，逐人标注由 `SemanticOperator` 提供。

## 1. 项目结构

```text
OkAgent/
├── _config/hiring.json           # 数据路径、岗位、日期、预算
├── _config/llm.example.json       # 模型配置模板；实际 llm.json 不入 Git
├── src/okagent/
│   ├── data.py                   # prepare：准备工作区
│   ├── prompts.py                # 简短的系统 prompt 和训练任务 prompt
│   ├── agent.py                  # mini-swe-agent 运行入口
│   ├── lo_ph_agent.py            # logical 函数规划 → physical 子 agent 执行
│   ├── semantic.py               # 逐人 LLM 标注、预算与标签缓存
│   └── evaluation.py             # evaluate：统一评估
├── other_methods/
│   ├── hydra-master(1).zip       # 本地参考源码包，不入 Git
│   └── hydra/
│       ├── method.py             # 主动采样、LR、停止条件与预算计数
│       ├── calibration.py        # 重要性采样与阈值校准
│       ├── hiring.py             # 招聘数据、标注回调和输出适配
│       ├── requirements.txt      # 复现方法的额外依赖
│       ├── tests/
│       └── README.md             # 算法细节与原实现的差异
├── tests/                       # 主项目测试
├── pyproject.toml               # 主项目依赖与安装配置
└── results/                     # 生成的工作区和结果，不入 Git
```

## 2. 环境配置

需要 **Python 3.10 或以上**。以下命令适用于 Linux，在项目根目录执行。
主项目和 Hydra 复现可以共用一个 `.venv`，Hydra 使用 CPU，不要求 GPU，也不需要编译原始 C++ 引擎或安装内部包。

### 使用 uv

```bash
cd /home/fanmeihao/projects/OkAgent
uv venv --python 3.10 .venv
source .venv/bin/activate

# 主项目、code agent、训练工具及测试依赖
uv pip install -e '.[agent,test]'

# 启用 CPU Qwen proxy（transformers>=4.51）
uv pip install -e '.[agent,test,proxy]'

# other_methods/hydra 的复现环境
uv pip install -r other_methods/hydra/requirements.txt
```

不运行 Hydra 时可省略最后一条命令；不需要测试时使用 `'.[agent]'`。
只准备/评估数据或运行 Hydra replay 时，主包选择 `'.[test]'` 即可，Hydra 仍需安装其 `requirements.txt`；无需 agent 依赖和模型接口。

### 没有 uv 时

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,test]' -r other_methods/hydra/requirements.txt
```

后续使用激活环境中的 `python`，或显式使用 `.venv/bin/python`，避免误用系统 Python。
**Python 示例也应在项目根目录运行**：`other_methods` 按本地源码导入，不会由主项目的 editable install 安装到其他目录。

### 已验证的版本

| 依赖 | 用途 | 已验证版本 |
| --- | --- | --- |
| Python | 运行环境 | 3.10.20 |
| DuckDB | 主项目及 Hydra 数据读取 | 1.5.5 |
| NumPy | Hydra 向量计算与采样 | 2.2.6 |
| SciPy | Hydra LR 优化 | 1.15.3 |
| pytest | 测试 | 8.4.2 |
| mini-swe-agent | code agent 基础循环 | 2.4.5 |
| LiteLLM | code agent 模型适配 | 1.101.0 |
| OpenAI SDK | 逐人标注的兼容接口客户端 | 2.54.0 |
| Jinja2 | 填充原始匹配模板 | 3.1.6 |
| scikit-learn | agent 可使用的 TF-IDF、分类器和评估工具 | 1.7.2 |

依赖文件使用兼容版本范围。需要对齐本次复现环境时，可在安装后固定上述库版本：

```bash
uv pip install 'duckdb==1.5.5' 'numpy==2.2.6' 'scipy==1.15.3' 'pytest==8.4.2'
# 运行 code agent 时还可固定以下版本
uv pip install 'litellm==1.101.0' 'openai==2.54.0' 'jinja2==3.1.6' 'scikit-learn==1.7.2'
```

若使用上面的 pip 安装方式，将这条命令的 `uv pip install` 换成 `python -m pip install`。

## 3. 数据与配置

原始数据从 `hydra-hiring-agent` 只读引用，不需要解压 Hydra 源码包来运行复现。
配置文件为 [_config/hiring.json](_config/hiring.json)：

```json
{
  "raw_root": "../../hydra-hiring-agent/benchmarks/hiring/raw",
  "job": "job01",
  "as_of": "2026-07-22",
  "max_calls": 2000
}
```

`raw_root` 相对于**配置文件所在目录**解析，也支持绝对路径。换机器时主要修改这个字段。
`as_of` 是解释相对时间条件的基准日期；`max_calls` 是数据标注请求预算，不包含 code agent 自身的推理调用。

数据目录应包含：

```text
raw/
├── job-description-20260722/*.json
├── job-candidate-embedding-20260722/
│   ├── hiring_job01_full_segvec.db   # candidate_segments 表
│   ├── hiring_job01_llm_pass.db      # llm_pass 表
│   └── ...
└── llm_prompt.txt
```

当前可运行的 full 岗位为 **job01、job04、job05、job07**；job06、job08 缺少简历库。
每条简历包含多个分段，分段表为 `candidate_segments(candidate_id, segment, text, vec)`。
金标准是历史 LLM 的 0/1 匹配判断，并非真实录用结果。

正式评估使用 full 数据。原始 job01 的 1k 库正例占 38.2%，而 full 为 382/34,761（约 1.1%），两者不能作为相同分布比较。

## 4. 主项目：运行 baseline code agent

### 配置两个模型

code agent 的模型负责写代码；标注模型负责判断单个候选人是否匹配。两者可以使用不同模型。
默认共用 `_config/llm.json` 中的接口与模型。本机已配置；新环境复制模板后填写 `key`：

```bash
cp _config/llm.example.json _config/llm.json
```

```json
{
  "openai_base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "llm_name": "ep-20250612104210-ss27q",
  "key": "填写你的API-key"
}
```

`llm.json` 已加入 `.gitignore`，不会随代码提交。可用 `OKAGENT_LLM_CONFIG` 指定其他配置文件路径。
以下环境变量会覆盖文件配置，只有需要覆盖时才设置。
下面以共用一个 OpenAI 兼容 Chat Completions 接口为例，替换占位值：

```bash
export OKAGENT_CODE_MODEL='openai/你的code模型名'
export OKAGENT_LABEL_MODEL='你的标注模型名'
export OKAGENT_API_BASE='https://你的模型服务/v1'
export OPENAI_API_KEY='你的API-key'
```

`OKAGENT_CODE_MODEL` 使用 LiteLLM 的 `provider/model` 格式，该模型须支持 `bash` function calling。
`OKAGENT_LABEL_MODEL` 使用接口实际模型名（也接受 `openai/` 前缀），须返回 JSON 文本。
`OKAGENT_API_BASE` 是可选的共同接口地址；配置文件和环境变量均未设置地址时使用 OpenAI 默认服务。
需要给 code agent 单独指定服务或参数时，可给 `run_agent(..., model=...)` 传入已配置的 mini-swe-agent Model 实例。

### 准备工作区

```python
from okagent.data import prepare

run_dir = "results/agent-job01"
prompt_path = prepare("_config/hiring.json", run_dir, job="job01")
print(prompt_path.read_text(encoding="utf-8"))
```

`prepare` 创建新目录，不覆盖已有目录。重做实验请换一个 `run_dir`；继续已有任务时跳过准备步骤。
旧版准备的工作区缺少 `settings.json` 和新的任务 prompt，运行 code agent 时请重新准备一个新目录。

```text
results/agent-job01/
├── task.json                  # 评估需要的数据路径和预算，留在工作区外
├── agent.trajectory.json      # run_agent 运行后生成，含命令、观察与退出状态
├── evaluation.json            # 成功提交后自动评估
└── workspace/
    ├── data.duckdb            # 简历库的符号链接，不复制整库
    ├── job.json               # 完整岗位要求
    ├── settings.json          # 标注日期与总预算
    ├── label_prompt.txt       # 原始标注模板
    ├── prompt.md              # 交给 code agent 的任务
    └── output/                # agent 写入结果
```

### 执行训练流程

```python
from okagent.agent import run_agent

result = run_agent("results/agent-job01", step_limit=80, command_timeout=1800)
print(result["evaluation"])
```

`run_agent` 将 `prompt.md` 作为初始任务传给 mini-swe-agent 的 `DefaultAgent`，用本地 shell 执行生成的代码。
参考的是 `OptiHarnessForCost/agent/mini-swe-agent` 的基础结构，运行时通过已安装的包调用，不依赖参考项目路径。
组件组合方式见 [mini-swe-agent Python 用法](https://mini-swe-agent.com/latest/advanced/cookbook/)。

prompt 要求 agent 实际完成：随机留出验证集 → 采样标注 → CPU Qwen3-Embedding-0.6B 缓存特征 + 加权逻辑回归/ensemble → 增量采样 → recall 优先的阈值选择 → 全库预测。
同时约束类别极不均衡、单类训练、验证集无正例、采样偏差与标签覆盖预测等情况。
Partition、Sample、Label、Proxy、Deploy 只表示代码阶段。具体训练脚本由 code agent 编写。

`step_limit` 限制 code agent 的模型轮数，`command_timeout` 是单个 shell 命令的秒数，均独立于 `max_calls`。
每条命令启动新 shell，默认使用启动 `run_agent` 的 Python 环境；请在 `.venv` 中运行。
完成后检查训练产物并自动调用 `evaluate`；到达步数上限或输出不完整时明确报错，保留 trajectory。
重新调用 `run_agent` 会开始新的对话并覆盖 trajectory，但同一工作区的标注预算和缓存继续累积；比较实验请使用新目录。

简历数据库须以 `duckdb.connect(..., read_only=True)` 打开。历史评测标签不交给 agent；目录分离是实验约定，不是操作系统沙箱。

### 逐人 semantic operator

在准备好的 `workspace/` 中使用：

```python
from okagent.semantic import SemanticOperator

op = SemanticOperator(".")
label = op.label("实际候选人ID")  # 返回 int 0/1
labels = op.label_many(ids, workers=8)  # 并发的独立单人请求，按输入顺序返回
cache = op.cached_labels()  # 只读取成功标签，不发请求；get_label(id) 返回标签或 None
print(op.usage())                # llm_calls、max_calls、remaining
```

每次 `label` 读取该候选人的全部非空分段，填充岗位、基准日期和原始 `label_prompt.txt` 的匹配规则。
原模板依赖的结构化候选人对象改为现有分段原文，保留匹配规则和 JSON 输出要求，不截断简历文本。
因此与历史标签生成时的输入表示可能存在差异，评估衡量的是与历史 LLM 标签的一致程度。

- **一人一次请求**；批量处理使用 Python 循环，无法用一条请求标注多人。
- 成功标签缓存在 `output/semantic.sqlite`，重复查询直接返回；多个脚本共用计数与缓存。
- 每个 ID 单独去重，预算预留时短暂加全局锁；独立候选人的网络请求可并发。失败占预算且不自动重试。
- `_config/llm.json` 可用 `label_kwargs` 设置标注请求参数；示例关闭豆包深度思考并设 temperature=0，不影响 code agent 的模型参数。
- 示例启用严格 JSON schema，仅返回 ID 和布尔匹配结果；保留原始判定规则，省去长说明。解析失败的响应也保存用量。
- 多个实验共用接口时，可设置 `label_interval`（秒，缺省 0）；同一用户、同一 endpoint/model 的请求共用本机节奏。示例为 0.65 秒，以减少共享 TPM 限流；具体数值按接口额度调整。
- 请求前持久化计数，失败、超时和非法 JSON 都占一次预算，不自动重试；再次显式调用失败 ID 会再计一次。
- 预算耗尽抛出 `BudgetExceeded`，成功缓存仍可读取。不要修改缓存或在同一工作区更换岗位、模型和规则。
- `output/usage.json` 自动更新；评估直接从 SQLite 请求记录读取次数，`usage_source="semantic_operator"`。

标注端设置 `max_retries=0`，关闭 SDK 默认重试，参见 [OpenAI Docs：Retries](https://developers.openai.com/api/reference/python#retries)。
预算记录是本地实验计数，不等于服务端账单；进程在预记账后中断也保守计入。

### 输出约定

agent 在工作区保存可重跑的 `pipeline.py`，并在 `workspace/output/` 保存：

| 文件 | 格式 |
| --- | --- |
| `candidate_ids.json` | 匹配候选人的字符串 ID 数组，例如 `["id_a", "id_b"]`；无匹配时为 `[]` |
| `usage.json` | 标注器自动生成，包含 `llm_calls`，次数必须是非负整数 |
| `proxy.pkl` | 特征变换、分类器和阈值；无法训练时保存明确的兜底规则 |
| `report.md` | 采样策略、预算分配、标签分布、验证结果与局限 |
| `semantic.sqlite` | 标注器自动记录每次尝试及成功的原始 JSON 判断 |

code agent 的编写和执行轨迹保存在工作区外的 `agent.trajectory.json`，其中模型调用数独立统计。
过长的命令输出只将首尾各 10,000 字符传给模型，完整输出留在轨迹中，避免单行 JSON 撑满上下文。

### 评估结果

```python
from okagent.evaluation import evaluate

report = evaluate("results/agent-job01")
print(report)
```

结果写入运行目录的 `evaluation.json`，包含 recall、precision、F1、TP/FP/FN/TN、标签覆盖数量、调用次数与 `within_budget`。
评估以全库为范围：未返回的候选人视为不匹配；重复或未知 ID 会报错；缺少金标准的候选人不当作负例；指标分母为零时返回 `null`。
超预算时仍计算指标，但 `within_budget` 为 `false`。

### Logical / Physical 两层 agent

使用相同依赖和模型配置，以一个新工作区运行：

```python
from okagent.data import prepare
from okagent.lo_ph_agent import run_lo_ph_agent

run_dir = "results/lo-ph-job01"
prepare("_config/hiring.json", run_dir)
result = run_lo_ph_agent(run_dir)
print(result["evaluation"])
```

规划 agent 只能调用 `partition / sample / label / proxy / deploy / finish` 函数工具。
前五个函数每次启动一个新的 mini-swe-agent，使用关注实现和自检的初始 prompt；规划 prompt 关注数据拆分、采样、预算和步骤依赖。
两层 agent 共享同一个 workspace、数据和标注缓存，每次实现 agent 的对话独立。

也可以直接调用 logical operator 的 Python 函数：

```python
from okagent.lo_ph_agent import LogicalOperators

ops = LogicalOperators("results/lo-ph-job01/workspace")
sample = ops.sample({"table": "data.duckdb"}, "seed=42，均匀随机抽取 100 人")
labels = ops.label({"table": sample["artifacts"]["table"]}, "逐人标注这 100 人")
```

每个函数接收 `inputs`（输入名称到工作区文件路径的映射）和 `instruction`（方法及参数），返回产物路径、摘要和剩余预算。
实现代码、产物和独立 trajectory 保存到 `workspace/operators/<算子>-<调用ID>/`；这里只是产物目录，执行 cwd 仍是共享 workspace。
规划轨迹保存为 `logical.trajectory.json`。实现失败会返回规划 agent 修正；最终 `finish` 提交已完成 deploy 的结果，再统一评估。
Label 提交时检查完整 ID 覆盖和成功缓存中的标签；模型、特征与阈值保存在对应 `operators/` 产物中。
`step_limit` 默认 40 个规划轮次，`physical_step_limit` 默认每次实现 40 轮，均不计入逐人标注预算。
可传入 `planner_model` 和 `physical_model_factory` 分别指定两层模型；默认共用 `_config/llm.json`。

## 5. 复现 other_methods/hydra

先完成第 2 节的 Hydra 依赖安装。该实现复现压缩包实际启用的路径：
**主动采样 → 逻辑回归 → 目标 recall 单阈值校准 → 全库预测**。
NumPy/SciPy 替代原模型实现，DuckDB 和 Python 回调替代内部运行依赖。

优化后的 CPU 招聘配置可用 `Config.hiring_cpu(max_calls=2000)`；配合
`run_hiring(..., feature_backend="qwen")` 使用 256 维、128-token、按简历分段均衡保留文本的 Qwen 语义特征和岗位 query 初始化，
并启用正则化、更大的训练/校准预算。默认 `Config()` 和 `feature_backend="stored"` 仍保留，便于复现旧结果。

统一的全量实验入口为：

```bash
python benchmarks/run_full.py precompute --job job01 --run-dir results/qwen-job01
python benchmarks/run_full.py baseline --job job01 --run-dir results/baseline-job01
python benchmarks/run_full.py lo_ph --job job01 --run-dir results/lo-ph-job01
python benchmarks/run_full.py hydra --job job01 --run-dir results/hydra-job01
```

设置 `OKAGENT_QWEN_CACHE` 可让不同工作区共享一次编码结果。正式比较仍应为每个方法使用新的
run-dir 和独立 2,000 次标注账本；`precompute` 不调用标注 API。
完成岗位特征预计算后，可用下列命令并发运行完整比较矩阵；每组输出、日志和账本相互隔离，
`manifest.json` 记录退出码与耗时：

```bash
python benchmarks/run_matrix.py --tag qwen-v1 --jobs job01 job04 --max-parallel 3
```

### 离线 replay：无需模型接口或 API key

```python
from okagent.data import prepare
from okagent.evaluation import evaluate
from other_methods.hydra import run_hiring

run_dir = "results/hydra-job01-repeat"
prepare("_config/hiring.json", run_dir, job="job01")
result = run_hiring(run_dir)
report = evaluate(run_dir)

print(result["usage"])
print(report)
```

默认通过标注回调查询历史 LLM 标签，只有请求到的标签参与训练和校准，不会调用真实 LLM。
此时 `usage.json` 的 `llm_calls` 是**模拟请求数**，不是实际 API 账单。

| 默认参数 | 值 |
| --- | --- |
| 训练样本上限 `sample_size` | 1024 |
| 校准抽样次数 `calibration_sample_size` | 512 |
| 目标 recall `target_recall` | 0.7 |
| 每轮采样数 `step` | 32 |
| 每请求人数 `batch_size` | 1 |
| 随机种子 `seed` | 42 |
| 请求上限 `max_calls` | 默认读取工作区的预算 |

通过 `Config` 调整参数，例如将调用改为：

```python
from other_methods.hydra import Config

config = Config(target_recall=0.9, batch_size=8, max_calls=2000, seed=42)
result = run_hiring(run_dir, config=config)
```

显式配置的 `max_calls` 不能超过工作区预算。每个 `run_hiring` 调用是一次新实验，计数从零开始；重复运行同一目录会覆盖方法输出，比较不同配置时应使用不同目录。

### Query embedding 与真实标注

当前按约定采用**随机首轮采样**。简历库记录的向量模型为 `doubao-embedding-vision-251215`，维度为 2048。
如果取得同模型生成的岗位 query 向量，传入 `run_hiring(run_dir, query_vector=query_vector)` 即切回原来的相似度首轮采样。
不能用其他模型的同维度向量代替。Hydra 适配器要求每人有一条有效的 `unstructured` 分段向量，缺失时会报错。

需要真实标注时，传入 `run_hiring(run_dir, label=my_label)`。
`my_label(ids)` 必须按输入顺序返回 0/1 整数或布尔值列表，每次回调最多一次请求；回调负责简历读取、提示词填充及模型认证。
实现会在调用前计数，失败立即报错，不自动重试。也可以安装 `.[agent]` 并配置第 4 节的标注模型，复用逐人标注器：

```python
from okagent.semantic import SemanticOperator
from other_methods.hydra import Config, run_hiring

# run_dir 必须是已经 prepare 的独立 Hydra 实验目录
op = SemanticOperator(f"{run_dir}/workspace")
result = run_hiring(run_dir, config=Config(batch_size=1, max_calls=2000),
                    label=lambda ids: [op.label(cid) for cid in ids])
```

此适配必须保持 `batch_size=1`，确保每次 Hydra 回调最多一次真实请求。
SQLite 预算跨运行累计，评估以该记录为准；Hydra summary 中的计数仍为本次运行的回调次数。

除通用的 `candidate_ids.json` 和 `usage.json`，Hydra 还输出 `hydra_summary.json`（参数、采样索引、阈值及轮次）和 `hydra_model.npz`（模型参数）。
算法细节、原配置范围和替换差异见 [Hydra 复现说明](other_methods/hydra/README.md)。

### 已有复现结果

job01 full，默认配置、随机首轮、seed=42、离线 replay：

| 候选人数 | 模拟请求数 | 筛出人数 | Recall | Precision |
| --- | --- | --- | --- | --- |
| 34,761 | 1,431 / 2,000 | 755 | 65.71% | 33.25% |

对应本地文件为 `results/hydra-job01/evaluation.json`。目标 recall 是估计目标，不保证真实全库达到；这里实际 recall 低于配置的 70%。
版本、随机数实现与数据变化可能影响结果。`results/` 不入 Git；已有结果可从文末的 ModelScope 归档恢复，也可自行运行生成。

## 6. 验证环境与测试

在项目根目录、激活 `.venv` 后：

```bash
# 检查主项目和复现方法能否导入
python -c 'from okagent.agent import run_agent; from okagent.semantic import SemanticOperator; from other_methods.hydra import run_hiring; print("imports OK")'

# 主项目测试
python -m pytest tests -q

# Hydra 方法测试
python -m pytest other_methods/hydra/tests -q

# 全部测试
python -m pytest tests other_methods/hydra/tests -q
```

安装 `.[agent,test]` 后，测试覆盖逐人 HTTP 请求、失败计数、跨进程缓存、并发预算、baseline 与 logical/physical 两层执行、模型配置，以及 Hydra 采样和阈值校准。
agent 测试使用本地模拟模型接口与确定性 code agent 消息，实际执行 Python/训练代码，不产生外部 API 开销；它们不代表真实模型的实验效果。
未安装 agent 依赖时，agent 测试会跳过。
直接运行 `python -m pytest -q` 默认只发现主项目的 `tests/`；检查复现方法时需要显式指定上面的路径。

## Agent 与 Hydra 实验比较

实验结果和逐轮修改见 [comparison_report.md](comparison_report.md)，不含简历内容的统计保存在 `benchmarks/results.json`。
安装上述 agent 环境和 `other_methods/hydra/requirements.txt` 后，从项目根目录调用函数：

```python
from benchmarks.compare import snapshot, run_case

snapshot('my_comparison')  # 名称不可重复；保存源代码、参数和依赖版本
run_case('my_comparison', 'baseline', 'job01')
run_case('my_comparison', 'lo_ph', 'job01')
run_case('my_comparison', 'hydra', 'job01')  # 真实逐人 LLM 标注
# 可选诊断：回放历史标签，无真实标注请求，不与真实标注结果混为一谈
run_case('my_comparison', 'hydra_replay', 'job01')
```

长任务建议设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1`。每个方法有独立 workspace、2,000 次预算、标签缓存和轨迹；运行前须配置本地 `_config/llm.json`。结果包含简历信息，保留在 Git 忽略的 `results/comparison/` 中。

每一轮使用新的 Python 进程，避免上一轮导入的代码快照留在模块缓存中。失败后可在原工作区继续：

```python
run_case('my_comparison', 'baseline', 'job01', resume=True)
# 同样支持 lo_ph 和 hydra；复用预算和文件，旧轨迹归档到 previous_attempts/

from benchmarks.summarize import export
export()  # 将全库评估、调用量和 token 等聚合数据写入 benchmarks/results.json
```

需要换 prompt 迭代时，先保存新轮 snapshot，再传入 `previous_run`、原 `validation_ids`（整个禁止训练的验证池）以及可选的 `validation_sample_ids`（已冻结的验证样本）。它会复制原账本，总预算不会重置。报告中的 R2→R3→R4 即这种续跑；它们不是独立重复实验。

## 下载完整实验结果

完整 `results/` 归档位于公开数据集 [ModelScope：fmh1art/OkAgent_results](https://modelscope.cn/datasets/fmh1art/OkAgent_results)。

实际文件约 **1.84 GiB**，`results.tar.gz` 约 **964 MiB**，因此继续由 `.gitignore` 排除，不放入 GitHub。归档包含全部 14,214 个实际文件和 29 个软链接，包括 R0–R6 的工作区、生成代码、模型、标签缓存、完整轨迹及修复记录；另附报告、聚合结果、逐文件 `manifest.json` 和 `SHA256SUMS`。

可通过 ModelScope 网页下载，或在单独的下载环境安装 `modelscope-hub==0.4.3` 后使用：

```python
from modelscope_hub import HubApi

HubApi().download_repo(
    "fmh1art/OkAgent_results", repo_type="dataset", local_dir="OkAgent_results"
)
```

进入下载目录，先校验，再解压到尚未包含 `results/` 的 OkAgent 项目，避免覆盖已有实验：

```bash
sha256sum -c SHA256SUMS
tar -xzf results.tar.gz -C /path/to/OkAgent
```

归档保留软链接，不复制其指向的外部原始数据库。换机器运行前，需要准备原始数据、重建 `workspace/data.duckdb` 链接，并调整 `task.json` 和历史脚本中的本机路径。具体恢复说明见数据集 README。密钥及原始参考 zip 不在归档中。
