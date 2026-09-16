# OkAgent 招聘 proxy 实验比较报告

实验日期：2026-09-16（Asia/Shanghai）。本报告保留失败轮次、同预算续跑和人工修复记录。

## 1. 最终结果

两个 agent 均生成了两个岗位的训练和部署产物，全部运行在各自 2,000 次标注尝试上限内。job01 的两层方法还需要 R6 的部署修复和人工阈值索引修正；不能把这些结果称为无人干预的一次成功。以下为 R4 agent、job01 两层方法的 R6 部署修复和 R2 Hydra live；它们是开发过程中最终保留的运行，不是严格配对的独立测试。

| 岗位 | 方法（轮次） | 尝试 / 成功标签 | 入选人数 | Recall | Precision | F1 | TP / FP / FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| job01 | baseline (r4) | 2000 / 1929 | 6879 | 79.06% | 4.39% | 8.32% | 302 / 6577 / 80 |
| job01 | lo_ph (r6) | 1917 / 1877 | 766 | 54.19% | 27.02% | 36.06% | 207 / 559 / 175 |
| job01 | Hydra live (r2) | 1540 / 1530 | 20157 | 80.37% | 1.52% | 2.99% | 307 / 19850 / 75 |
| job04 | baseline (r4) | 1965 / 1924 | 1500 | 46.34% | 10.13% | 16.63% | 152 / 1348 / 176 |
| job04 | lo_ph (r4) | 1913 / 1889 | 3439 | 71.04% | 6.78% | 12.37% | 233 / 3206 / 95 |
| job04 | Hydra live (r2) | 1547 / 1530 | 3320 | 61.28% | 6.05% | 11.02% | 201 / 3119 / 127 |


本次比较的结论：

- **job01：** baseline 的 recall 接近 Hydra（79.06% / 80.37%），precision 更高（4.39% / 1.52%）；两层方法修复后 precision=27.02%，但 recall 下降到54.19%，不符合“召回优先”的优势预期。
- **job04：** 两层方法的 recall/precision 均高于本轮 Hydra（71.04%/6.78% 对61.28%/6.05%），同时使用更多标注尝试（1913 对1547）。
- **没有稳定的总体赢家，也没有方法达到90%的历史标签召回。** 两层方法增加了代码调用和跨步骤错误；现有证据不足以证明更复杂的架构更适合本场景。


## 2. 实验范围与计数规则

| 数据 | job01 | job04 |
| --- | ---: | ---: |
| 岗位 | 图片生成算法实习生-Seed | 多模态数据工程实习生 |
| 全库候选人 | 34,761 | 18,183 |
| 历史 LLM 正例 | 382（1.10%） | 328（1.80%） |
| 简历分段 | 239,818 | 128,752 |

数据来自 `hydra-hiring-agent/benchmarks/hiring/raw`，使用 full 库，日期固定 `2026-07-22`。库的 `_meta` 记录 embedding 为 `doubao-embedding-vision-251215`，2048 维；未取得同模型岗位 query 向量，因此 Hydra 按此前约定随机首轮采样。seed=42；未使用聚类或关键词硬筛选。

每个方法、每个岗位最多 **2,000 次逐人标注尝试**，成功、失败、超时及中断后尚未完成的预留请求均计数；缓存命中免费且不重复请求。R2→R3→R4 是同一账本续跑，**不会重新获得 2,000 次预算**。agent 自身编程/规划调用另计，不计入这 2,000 次，但计入整体开销。R0、R1 为独立开发轮，不能混入最终方法预算。

统一评估在工作区外读取历史 `llm_pass`：全库为分母，未入选候选人算预测负例，计算 TP/FP/FN/TN、recall、precision、F1。agent 输入不包含历史标签或评估结果。**这里衡量与历史 LLM 判断的一致程度，不是人工招聘真值上的准确率。** 实时标注读取全部非空简历分段，而历史生成流程的输入表示可能不同。

实际返回模型为 `doubao-seed-2-0-lite-260215`。最终标注配置：关闭 thinking，temperature=0，严格 schema，仅返回 candidate_id 和布尔 is_match.result，max_tokens=256；保留原模板中的岗位匹配规则和完整简历分段。code agent 使用同一 endpoint，最大输出 8,192 tokens。标注 SDK 不自动重试，各工作区共用 0.65 秒请求启动间隔，可有多个请求在途。

job01 用于开发，job04 最初用于跨岗位复核。R3 后已检查两岗位的输出，并在 R4 修正共性代码问题，因此 **job04 也不是完全未触碰的最终测试集**。仅运行一个 seed；即使 temperature=0，真实模型仍可能返回不同标签或程序。

### 比较方法

- **baseline**：单个 mini-swe-agent 自主写、运行及修正采样、训练和全库部署代码。
- **lo_ph**：规划 agent 调用 `partition/sample/label/proxy/deploy` Python function；每次 function 启动新的 physical code agent，所有步骤复用一个 workspace 和标注账本。
- **Hydra live**：`other_methods/hydra` 的 NumPy/SciPy 复现，调用同一 SemanticOperator。目标 recall=0.9，训练上限 1,024、校准抽样 512、step=32、batch_size=1、seed=42、label_workers=8；其余为复现默认参数。主动训练与重要性加权校准按复现实现执行，不由代理 agent 改写。
- **Hydra replay**：相同配置，但只向历史标签回放器查询；无真实标注 API，单列为诊断对照。

这是一次真实接口上的开发迭代比较，不是严格配对消融：验证样本、实际使用预算、生成的特征方法不同，R1/R2 中途有工程修复，续跑继承早期标签。不能把结果变化全部归因于 prompt 或两层架构。

## 3. 迭代与运行故障

### R0：暴露运行瓶颈，未完成部署

起点 `1d12748`。约 437 秒主动停止三个 job01 运行；baseline / lo_ph / Hydra 分别尝试 11 / 13 / 15 次，成功 10 / 12 / 14 次。**未完成不记为零召回。**

整段网络调用持有全局锁，导致并发请求串行；改为每 ID 去重锁和短预算锁，增加 `label_many`、只读 `cached_labels/get_label`，Hydra 增加可选 label_workers，默认仍为 1。baseline 自行从失败的 nano 改用 heredoc，prompt 明确禁止交互编辑器。默认深度思考导致每条约 25–30 秒；关闭 thinking 后，独立 8 人真实探针在 10.23 秒完成。修复后 38 项测试通过，包括并发预算、去重和 Hydra 串行/并发算法等价。

### R1：原策略端到端尝试

起点 `73d80da`，保留原始策略，在运行中修复 JSON 解析及格式。

- baseline：800 次尝试、778 个成功标签、10 个正例，最终 `RepeatedFormatError`；长工具参数被截断，修复脚本还曾把 shell `sleep` 写成 Python。无可评分部署结果。
- lo_ph：250 次尝试、249 个标签、5 个正例；summary 对象/字符串、JSONL/JSON 数组反复不符，约 709 秒主动停止。planner 一度换样本回避格式错误。
- Hydra：保留账本多次恢复后完成；job01 recall=62.57%，precision=9.78%，F1=16.91%，1,552 次尝试。不是一次无故障运行。

baseline 生成的 PR 阈值取第一个合格项，趋于过低阈值；后续 prompt 明确取满足召回约束的**最大**阈值。解析器容忍围栏和字符串换行，但仍检查候选人 ID 和布尔标签。

### R2：加入 proxy skill，并处理共享接口故障

起点 `effd064`，对两个岗位运行三个方法。

嵌入简短 skill：随机探索、稀少正例的高分/边界/随机混合采样、L2 归一化 embedding + balanced L2 LR、固定独立验证、最大合格阈值、批量读库、成功缓存覆盖。强调聚类弱不等于监督向量分类无效。

同时运行触发 `RateLimitExceeded.EndpointTPMExceeded`。失败仍计预算；曾暂停实验、中断子脚本再从缓存继续。加入按 endpoint 共享的本机请求节奏（`6002757`），没有新增服务或队列。部分 agent 因限流把 worker 降至 1；只恢复其 worker 数，补丁保存于本地 repairs。

长解释 JSON 又出现 malformed JSON、4,096 tokens 截断。仅缩小 schema 时，原详细输出要求仍与之冲突，产生空白输出直至截断；最终同步把输出要求缩成 ID + bool。4 个先前失败候选人的探针全部成功，32 输出 tokens/人，共 7.11 秒。code agent 输出上限提高到 8,192，并要求分段写文件。

lo_ph 曾将 400 人输入中的 187 人部分标签表交作成功结果；增加必要的 Label 检查：输入 ID 完整且唯一，0/1 标签必须与成功缓存一致。上述修复没有改动 function → 新 code agent → 共享 workspace 的结构。

R2 四个 agent 保存检查点，转入 R3；它们未完成部署，不评分。两项 Hydra live 在原账本中完成，作为主要 Hydra 对照。所有运行修复及源文件副本在 `results/comparison/r1/repairs/`、`r2/repairs/`；R1/R2 标签包含不同输出格式阶段，不伪称全程同一协议。

### R3：固定工程配置，同预算完成四个 agent

起点 `58a749d`。在新工作区复制各自 SQLite 账本和规范化标签，不复制旧生成程序。初始调用数为 baseline job01/job04 的 507/279、lo_ph job01/job04 的 387/440。验证划分保持不变；命令超时设为 7,200 秒，以容纳共享限速。四个 agent 均完成，详细指标见结果表。

审查生成代码和实际模型，发现：

1. baseline job04 只使用固定 400 人中的已标注 240 人（3 个正例），没有补齐；训练达到 51 个正例便停止，剩余 523 次。
2. baseline job01 报告中的 precision=2.05% 是旧变量，实际最终 proxy 在验证集为 7.14%（6 TP、78 FP）；全库评估从未采用 agent 自报指标。
3. lo_ph job01 的 Deploy 在缓存覆盖后算验证指标，得到人为完美结果；正确的覆盖前 precision=16%、recall=100%，仅 4 个正例。部署阈值从文字摘要取 0.4397，原值 0.4397163596771684；本例未改变验证召回，但不应靠四舍五入文字传参。
4. lo_ph job01 原保留 6,953 人验证池，后续采样排除了最终 400 人，却进入池内其余 122 人。最终训练 947 人中，严格排除整个池后仅 825 人；违反原续跑划分约定。不能称其验证池始终隔离，但不等同于最终 400 人直接参与训练。
5. lo_ph job04 中间模型曾用任意首个分段向量，后续又恢复分段均值；最终训练和部署一致，但阶段间特征漂移影响采样决策。

完整聚合审查记录见 `benchmarks/audit.json`。R3 结果已提交并推送 `7cc0f15`，原始结果保留。

### R4：依据代码审查修正，预算不重置

起点 `e8c8a2b`，只增加短 prompt 约束和续跑时显式固定验证样本的参数：补齐原样本；排除整个验证池；累计训练标签；复用同一特征函数；保存完整阈值；覆盖缓存前重新计算模型指标；少量验证正例时，不以训练正例数或固定轮数提前停止。

四项运行分别继承 R3 的 2,000 / 1,477 / 1,387 / 1,362 次尝试。baseline job01 已无标注余额，本轮只能重建模型和修正产物；其余只能消耗各自剩余额度。R3 和 R4 不是独立重复实验，且重新生成程序，所以不是只有一个变量变化的 prompt 消融。

R4 的真实运行还暴露三类问题，均保留失败证据后修复：

- baseline job04 在并发保存标注响应时出现 `sqlite3.OperationalError: database is locked`，随后父进程消失。最后保存动作为查看数据库句柄，轨迹不能确认进程退出的具体原因。把响应更新也纳入已有短写锁，显式关闭实验账本复制连接，并保留旧轨迹后恢复原工作区；禁止生成脚本杀死其他数据库使用者。没有重置预算。
- baseline job01 用 head/tail 查看单行候选人 JSON，实际把超过 30 万字符加入上下文，接口拒绝超长请求。通过 mini-swe 原有 observation_template 只向模型传首尾各 10,000 字符，完整输出仍留轨迹；新增回归测试，再由 agent 检查已有模型和输出后提交。
- 两项 lo_ph 的首批 Sample 从 pickle 字典只取 classifier，漏掉训练时 L2 归一化。已发生的采样不能撤销；明确追加变换要求到后续物理算子读取的 continuation，上游成功标签保留。项目 physical prompt 要求优先保存 sklearn Pipeline，并在读字典时应用完整变换。**这批 R4 采样仍属于有偏差的生成实现，不能描述成全程严格遵守 skill。**

修复提交依次为 `8636ebe`（缓存写入和中断证据）、`bcd3b6f`（模型可见输出限长）。运行轨迹显示 agent 能自行修复若干类型、字典读取、Python 字符串和特征接口问题，但仍需要上述人工运行审查；报告不将其表述为无人干预的一次成功。

baseline job04 的初次 R4 输出为 recall=46.34%、precision=10.13%，累计1965次查询；但自修复只在临时脚本生成了模型，pipeline.py 仍有变量顺序和尾部追加代码问题。按用户“继续迭代”的澄清，在相同工作区重启 code agent，仅用现有标签修复可从头执行的 pipeline 并重新部署，不以消耗剩余35次为目标。旧指标和轨迹归档保留，最终表采用修复后输出。

R4 第二批主动采样又暴露排除集问题：job01 把 JSON 行字典整体转成字符串作为 ID，100 个样本中 70 个已经标注；job04 只使用启动时训练文件，120 个样本中 69 个命中旧缓存。缓存去重正确，没有重复收费，但实际新增样本少于计划，浪费了 agent 计算步骤。已在 physical prompt 和后续算子上下文明确从当前 cached_labels 全量键构造排除集，并要求 Proxy 使用全部验证池外缓存标签，避免漏合并中间批次。先前采样仍保留，不能追溯当作正确的独立新样本。

### R5：针对排除集缺陷的真实 physical agent 验证

使用最终 prompt（`43cea16`）在两个独立工作区复制 R4 账本，分别让真实 physical agent 编写 Sample：seed=42，均匀随机抽 32 人，排除当前全部缓存 ID、原验证池，并正确读取旧训练 JSON 行数组。外部再次核验，两项均为 32 个唯一合法 ID，与完整缓存及验证池的交集均为 0；新增候选人标注为 0。耗时 job01 74.79 秒、job04 78.64 秒。

这是函数级回归验证，验证了本次具体的缓存排除/行格式修正；不等于重新完成一轮主动评分采样和全库质量实验。代码 agent 调用计入项目总开销，结果不并入 R4 的质量表。

### R6：只修复 job01 的 Deploy，模型和标签保持不变

R4 job01 的 lo_ph 输出了 32,886 人，历史 recall=86.13%、precision=1.00%。最终审查发现它在验证集使用 `LIMIT 1` 首个分段，在全库预测却使用所有分段均值；验证得到的阈值约0.0537失真。**这是实现错误造成的几乎全选，不能当作有效的质量改进。** R4 job04 没有这一错误。

用 `5503836` 的 physical prompt，在新工作区复制 job01 的已训练 Pipeline（记录SHA-256）、同一查询账本和固定验证样本，仅调用 `LogicalOperators.deploy` 修正。要求验证和全库预测使用同一个分段均值特征函数，核验同一ID的向量/分数一致，重新选择阈值再覆盖缓存；禁止重新训练或新增标注。

R6 是有明确人工错误定位的部署修复，并未重新运行完整规划 agent。最终质量表采用该修复后的 job01 输出，并明确标记；R4 错误结果仍在表和原始目录中保留。其他三项 agent 结果仍为 R4，不能把不同范围的修复宣称为独立端到端重复实验。

R6 的首个 physical 实现修正了特征函数，却在按分数降序累计召回后仍取“最后一个达标下标”，实际选择了最低阈值0.051001127，而非最大合格阈值。保留原代码、指标、候选集合及评估到 `r6/repairs/` 后，人工将该下标改为第一个，并增加与 `precision_recall_curve` 结果完全一致的断言，重新执行部署。此处为明确记录的人工一行逻辑修复，不能声称 agent 已自行可靠完成所有阈值实现。最终 physical prompt 也补充了升序PR曲线与降序累计召回两种索引方向的区别。


## 4. 修正前后与验证审查

| 岗位 | 方法 | R3 Recall → R4 | R3 Precision → R4 | R4 新增 / 累计尝试 |
| --- | --- | --- | --- | --- |
| job01 | baseline | 79.06% → 79.06% | 4.39% → 4.39% | 0 / 2000 |
| job01 | lo_ph | 67.02% → 86.13% | 14.45% → 1.00% | 530 / 1917 |
| job04 | baseline | 36.59% → 46.34% | 11.48% → 10.13% | 488 / 1965 |
| job04 | lo_ph | 78.35% → 71.04% | 5.08% → 6.78% | 551 / 1913 |

表中 job01 lo_ph 的 R4 指标含已确认的部署特征错误，86.13% recall 对应几乎全选，属于无效的质量改进；R6 修复后的指标见最终表。job01 lo_ph 从 R3 到修复后的 R6，recall 为67.02%→54.19%，precision 为14.45%→27.02%；增加训练标签没有带来召回提升。以上保留相邻轮次原始变化；R0、R1 agent 和 R2 检查点没有合格部署结果，不填造质量指标。R1 Hydra 的完整结果另保留于聚合 JSON。

最终模型的独立重算结果（缓存覆盖前；“可用训练标签”排除整个原验证池）：

| 岗位 | 方法 | 可用训练标签 / 正例 | 验证人数 / 正例 | 阈值 | 验证 Recall / Precision（覆盖前） |
| --- | --- | --- | --- | --- | --- |
| job01 | baseline | 1529 / 207 | 400 / 6 | 0.19549370 | 100.00% / 7.14% |
| job04 | baseline | 1524 / 71 | 400 / 4 | 0.28175746 | 100.00% / 9.09% |
| job04 | lo_ph | 1489 / 75 | 400 / 5 | 0.25967972 | 100.00% / 6.10% |
| job01 | lo_ph | 1355 / 149 | 400 / 4 | 0.44911671 | 100.00% / 22.22% |


重新加载 pickle，按生成代码的特征处理从原始数据库重建验证向量，计算模型概率、PR 阈值和混淆矩阵。R4 的固定验证样本均为原来的 400 人，未按历史评估结果重抽；可用训练标签始终不包括整个验证池。最终部署的特征和阈值也已结合对应生成代码审查；job01 两层方法的错误正是在这一步被发现并修正。

baseline job01 无剩余标注预算，本轮重新生成训练代码后的全库入选集合与 R3 相同；job04 补齐验证集、增加已有预算内训练数据后，recall 提升但 precision 下降，不能说两个指标同时提升。job04 最终保留 35 次余额；完成脚本修复时没有为用完预算而追加标签。

本轮仍包含已经记录的采样特征错误、重启和提示修正；这些干预及随机程序变化限制因果解释。验证集反复用于选择阈值，不是独立的最终测试集，四到六个正例也不足以给出可靠的全库高召回保证。

两层 agent 的 R4 最终仍分别剩83/87次预算，未用完全部上限。job01 的 R6 只修复部署；其验证特征与训练/预测统一为分段均值，另有人工阈值索引修复及PR曲线断言。函数级R5已验证新的缓存排除要求，但不证明所有未来生成程序都会遵循这些提示。


## 5. 实时标签与历史 oracle 的分歧

已查询样本中的两种判断存在明显分歧。由于部署强制以实时缓存覆盖 proxy，凡“实时判负、历史判正”的候选人都成为历史评估中的 FN。即使对其余所有人都预测正确，历史 recall 也至多为 `1 − 这类人数 / 全库历史正例数`。

| 岗位 | 方法 | 已查询：实时+/历史− | 已查询：实时−/历史+ | 缓存覆盖后的历史召回上限 | 实际历史召回 |
| --- | --- | --- | --- | --- | --- |
| job01 | baseline | 60 | 67 | 82.46% | 79.06% |
| job01 | lo_ph | 57 | 53 | 86.13% | 54.19% |
| job01 | Hydra live | 67 | 66 | 82.72% | 80.37% |
| job04 | baseline | 34 | 75 | 77.13% | 46.34% |
| job04 | lo_ph | 42 | 64 | 80.49% | 71.04% |
| job04 | Hydra live | 36 | 62 | 81.10% | 61.28% |

最终六项 live 结果的这个上限都低于90%，因此当前缓存覆盖规则下都无法达到该历史目标；它不是模型召回的置信区间。已查询集合包含主动富集样本，不能将其一致率当成随机全库的一致率，也无法在没有人工审查的情况下判定哪一个 oracle 更正确。

### 历史标签回放诊断

| 岗位 | 虚拟查询数 | 入选 | Recall | Precision | F1 | 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| job01 | 1431 | 9162 | 95.81% | 3.99% | 7.67% | 25.75 |
| job04 | 1525 | 7740 | 99.70% | 4.22% | 8.11% | 15.98 |

回放只在请求到的 ID 上提供历史标签给训练/校准，未调用真实 LLM。较高 recall 伴随约 4% precision；这些虚拟请求数和十几秒耗时不能当作真实接口成本或速度。它与 live 的差异同时包含 oracle、标注输入及标签路径变化，不能归为单一因素。

## 6. 成本、可靠性与时间

| 岗位 | 方法 | 标注 input / output tokens | agent query 轮次（累计） | agent input / output tokens（累计） | 活动运行分钟（累计） |
| --- | --- | --- | --- | --- | --- |
| job01 | baseline | 14,801,999 / 174,890 | 53 | 423,750 / 36,788 | 120.4 |
| job01 | lo_ph | 14,807,740 / 128,653 | 236 | 1,899,986 / 135,031 | 169.3 |
| job01 | Hydra live | 12,043,706 / 185,963 | 0 | 0 / 0 | 80.7 |
| job04 | baseline | 16,093,463 / 93,133 | 73 | 637,153 / 34,753 | 123.6 |
| job04 | lo_ph | 15,980,967 / 83,311 | 258 | 1,879,697 / 131,787 | 156.5 |
| job04 | Hydra live | 13,446,524 / 164,291 | 0 | 0 / 0 | 79.1 |

同预算续跑的 label tokens 取最终账本，**不重复加上已复制的旧标签**；agent tokens 和 query 轮次沿 R2→R3→R4→R6 的实际继承链累加（R6 仅 job01 lo_ph）。上表 Hydra 没有编程 agent 开销。这些用量不包含本助手开发/审查编排的开销。agent query 轮次来自 mini-swe 统计，不包含其 SDK 内部每次重试，故不是完整 HTTP 请求数。已保存的格式错误响应也纳入 token 汇总；接口拒绝、进程中断或未返回 usage 的请求仍可能缺少 token，属于可观测用量下界。

全项目开发与验证轮次（R0–R6，去除继承账本重复）共记账 **13,523 次标注尝试**，记录标注 108,449,921 tokens、agent 5,701,848 tokens。另有 12 次候选人标注探针（8 人并发检查 + 4 人紧凑输出检查）和两个通用格式探针，均不属于方法预算。历史 replay 是虚拟查询，未计入真实标注总数。

本项目未核验该 endpoint 的实际计费档位，**不把 LiteLLM 的 cost=0 当成免费，也不报告未经验证的金额或加速倍数**。TPM 拒绝没有可用 token 统计；缓存计费折扣也不能仅凭总 tokens 推算。

时间为各运行阶段的墙钟之和：包含模型请求、代码生成、特征读取、同轮暂停和共享接口竞争；不含轮与轮之间等待继续的间隔。各方法并行，不能把这些分钟相加当作项目总耗时，也不能作为严格速度排名。两层 agent 的大量独立物理步骤、反复读库和修复调用是其可见开销来源。

## 7. 方法启发与当前结论的边界

[Google Cloud 博客](https://cloud.google.com/blog/products/data-analytics/more-than-100x-faster-and-cheaper-llm-powered-sql-queries-with-proxy-models) 及其 [SIGMOD 2026 论文](https://arxiv.org/abs/2603.15970) 讨论用轻量 proxy 近似昂贵的逐行 LLM 查询，并指出类别不均衡、特征质量及评估方式会影响效果。这里吸收的是监督 embedding 分类、平衡正负类、验证阈值与明确质量限制；**没有复现或引用其“100×”为本项目结果**。

Hydra 的启发是将有限预算分配给主动采样与校准，并兼顾正例补充和边界样本。agent 版本采用更简单的混合采样和 balanced L2 LR，具体实现由 agent 生成；不是把 Hydra 的全部算法复制进 prompt。

本次结果支持以下有限结论：

- 两种 agent 已能实际生成、训练并部署 proxy，工程上仍会出现格式、类型、缓存、上下文和规划错误；最终可运行不等于从头一次成功。
- 两层划分让阶段产物可检查，但会增加规划与物理 agent 调用，并引入特征/阈值/数据集合在阶段间不一致的风险。本实验不支持仅凭架构推断其效果或成本必然优于 baseline。
- 极低正例比例下，400 人随机验证集只得到几个实时正例。反复在同一个小集合调阈值，即使表面 recall=100%，也不能保证全库 recall≥90%；没有独立的阈值测试集或统计置信保证。
- 缓存覆盖要求实时判断替代 proxy，但与历史 oracle 不一致时会固定产生一部分历史 FN。继续增加查询不一定单调改善“历史标签召回率”。
- 下一步更值得先统一实时标注输入与历史标签生成协议，并建立独立审查集，再做多 seed、同一验证划分、固定特征和匹配预算的消融。当前材料不足以给出统计显著性或真实招聘质量排名。

没有新增 CLI、服务或多层规划框架。业务代码仍为数据准备、SemanticOperator、两种 agent 入口、评估六个模块；额外比较和聚合函数位于 `benchmarks/`。

## 8. 环境、证据与复现

实际环境：Python 3.10.20，mini-swe-agent 2.4.5，OpenAI SDK 2.54.0，LiteLLM 1.101.0，DuckDB 1.5.5，scikit-learn 1.7.2，NumPy 2.2.6，SciPy 1.15.3，CPU 运行。安装主项目与 Hydra：

```bash
cd /home/fanmeihao/projects/OkAgent
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e '.[agent,test]' -r other_methods/hydra/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
```

按 `_config/llm.example.json` 创建本地 `_config/llm.json`，填写授权接口与 key；按 `_config/hiring.json` 设置本地 raw_root。密钥、简历、候选人 ID、轨迹和参考 zip 均未提交。详细环境说明见 [README](README.md) 和 [Hydra README](other_methods/hydra/README.md)。

新实验从项目根目录以 Python 函数运行（新名称，避免覆盖已有证据）：

```python
from benchmarks.compare import snapshot, run_case
snapshot('reproduce')
for job in ('job01', 'job04'):
    for method in ('baseline', 'lo_ph', 'hydra', 'hydra_replay'):
        run_case('reproduce', method, job)
```

同一轮可顺序运行以减少接口竞争；每个新轮使用新的 Python 进程，避免旧快照的模块缓存。顺序运行的耗时自然不同于本次并发、暂停和恢复的墙钟。

失败后原工作区恢复（保留预算；归档先前轨迹）：

```python
run_case('reproduce', 'baseline', 'job01', resume=True)
```

新 prompt 轮继承旧账本的例子：

```python
import json
from pathlib import Path
from benchmarks.compare import snapshot, run_case
snapshot('reproduce_next')
old = Path('results/comparison/r3/baseline-job04')
ids = json.loads((old / 'workspace/continuation_validation_pool.json').read_text())
run_case('reproduce_next', 'baseline', 'job04', previous_run=old,
         validation_ids=ids, validation_sample_ids=ids)
```

`previous_run` 必须指向自己的实际旧运行。对于 lo_ph job01，validation_ids 是整个保留池，validation_sample_ids 仅为已经冻结的 400 人样本；不能把池内其余 ID 重新划到训练。新机器没有本地 results 时，应从新实验开始；不能仅凭 Git 中的聚合表还原旧 API 响应。

证据位置：

| 内容 | 路径 |
| --- | --- |
| 全部轮次的聚合指标、tokens、状态 | [`benchmarks/results.json`](benchmarks/results.json) |
| 模型/验证/划分审查聚合数据 | [`benchmarks/audit.json`](benchmarks/audit.json) |
| 原始工作区、生成代码、SQLite、完整轨迹 | `results/comparison/r0` 至 `r6`（本地，Git 忽略） |
| R2–R6 开始时的依赖及代码哈希 | 各轮 `manifest.json` |
| 运行中的改动与原文件 | 对应轮次 `repairs/` |
| 原工作区恢复前的轨迹和状态 | `r4/*/previous_attempts/` |
| 聚合函数 | `from benchmarks.summarize import export; export()` |

R0/R1 的早期目录没有完整 manifest，仍保留运行元数据和源码副本。R4 有中途工程补丁，R6 有人工部署代码修正；初始 manifest 与最终运行文件不能混为一谈，修复点及原文件已在报告和 repairs 记录。

已验证测试涵盖真实本地 HTTP 客户端、子进程执行、并发去重/预算/缓存、失败记账、两层调用、Label 完整性、长输出截断及 Hydra 串并行等价。模拟接口测试证明这些机制的行为，质量表来自真实接口运行。

复现本次函数级检查及部署修复的本地脚本为 `results/comparison/check_sampling.py` 和 `fix_deploy.py`，部署修复后的实际代码位于 `r6/lo_ph-job01/workspace/operators/deploy-d8c5e15bbe11/implementation.py`。这些依赖私有工作区的实验产物保留本地；重新运行脚本仍需审查生成代码，不能保证随机agent再次生成完全相同程序。

最终测试：`python -m pytest tests other_methods/hydra/tests -q --tb=short`，**42 passed**，仅一个 Pydantic TypedDict ReadOnly 提示。
