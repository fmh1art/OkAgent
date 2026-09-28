# OkAgent job01 深化实验周报

**时间：2026 年 9 月 23 日—2026 年 9 月 28 日**

## 一、本周目标

本周围绕“如何把简历匹配实验做得更深入”开展工作，重点验证三条思路：一是按简历不同部分训练多个 proxy；二是通过 router 自动融合多个 proxy 的判断；三是用 batch prompting 批量调用 LLM，生成更丰富的多维标签，为后续 proxy 和 Qwen 模型训练提供监督信号。

## 二、本周完成的方法

### 1. 简历切分与多 Proxy 建模

本实验没有再调用 LLM 对简历重新切段，而是直接使用 job01 数据库 `candidate_segments` 中已有的 section 及其预计算 embedding。首先按 `candidate_id` 聚合数据，再为每位候选人构造三个 2048 维视角：

| Proxy | 读取的 section | 候选人级输入如何构造 | 负责判断的内容 |
|---|---|---|---|
| Global | `unstructured` | 读取整份简历对应的 embedding | 候选人与岗位的总体匹配度 |
| Experience | `projects`、`work` | 对候选人所有项目和工作 section 的 embedding 求均值 | 工作年限、岗位职责、项目经历是否匹配 |
| Credentials | `education`、`awards`、`languages`、`applications` | 对学历、奖项、语言和申请材料 embedding 求均值 | 学历、资质、语言等条件是否匹配 |

聚合在训练之前完成：无论一名候选人有 1 段还是 10 段工作经历，在每个 proxy 中都只形成一个向量、贡献一次 loss，避免长简历因为 section 更多而被重复加权。同时为每个 proxy 保存 `presence mask`；若某位候选人缺少对应 section，该 proxy 不参与其动态加权，缺失分数则用训练集中该任务的先验均值填充。

每个 proxy 使用 `StandardScaler(with_mean=False) + LogisticRegression`，参数为 `C=0.25`、`class_weight=balanced`、`solver=liblinear`、`max_iter=2000`。本周已完成的二分类实验中，三个 proxy 都以固定 2,000 名候选人的历史 `llm_pass` 为监督信号。正在运行的多标签实验会改用软标签：Global 学习 `overall`；Experience 学习 `(experience + projects) / 2`；Credentials 学习 `(education + technical_skills + research) / 3`。连续标签通过带样本权重的 soft-BCE 等价展开训练，而不是先粗暴取整成 0/1。

### 2. 多 Proxy 融合与 Router

为避免 router 直接读取 proxy 对自身训练样本的过拟合分数，先在 2,000 条训练数据上执行 5 折分层交叉验证（`seed=11`）。每一折只用其余 4 折训练三个 proxy，再对留出折预测；拼接五个留出折后，每名训练候选人获得三个真正的 out-of-fold（OOF）概率。只有 router 特征和目标固定之后，三个最终 proxy 才在全部 2,000 条数据上重新拟合。

在相同的三个 proxy 输出之上比较以下方法：

1. **Expert Mean**：只对该候选人实际存在的 proxy 分数求平均。例如缺少教育等资质 section 时，只平均 Global 和 Experience，而不是把 Credentials 当成 0。
2. **OOF Stacking Router**：二级逻辑回归的输入是 6 个特征，即三个 OOF 概率和三个 section presence 标志；监督目标仍为候选人的总体 `llm_pass`。模型直接学习不同 proxy 分数的组合系数和截距，输出最终通过概率。这里的“router”不是硬选一个专家，而是对多个专家证据进行有监督融合。
3. **Embedding Router**：先计算每名训练候选人在三个 OOF proxy 上的二分类 log-loss，把损失最小且 section 存在的 proxy 作为“最佳专家”标签；再用 Global 的完整简历 embedding 训练多分类逻辑回归（`C=0.1`、`class_weight=balanced`）预测三类专家权重。推理时将不存在的专家权重置零，对剩余权重重新归一化，最后计算三个 proxy 分数的加权和。

因此两个 router 的区别是：Stacking 直接根据三个专家“对当前样本打了多少分”学习最终判断；Embedding Router 则试图仅从简历整体表征判断“当前样本应更相信哪个专家”。实验还保存了每名候选人的三个原始分数、section 是否存在、路由权重和最终分数，便于分析 router 的具体选择。

### 3. Batch Prompting 与多标签标注

标注输入使用结构化简历文本而不是 embedding。对每名候选人去掉重复的 `unstructured` 全文，优先保留 `projects`、`work`、`education`、`applications`、`awards` 和 `languages`；每个 section 最多截取 2,000 字符，每名候选人的总输入不超过 8,000 字符。这样既保留了 section 边界，又控制了长简历对上下文窗口和费用的影响。

一次 HTTP 请求放入 6 名候选人，只发送一份共享的岗位描述、硬性要求、加分要求和评估日期。发送前将真实 `candidate_id` 替换为请求内的 `candidate_000` 等临时编号，返回后再映射回真实 ID。模型设置 `temperature=0`，并强制返回 JSON。每名候选人必须返回以下字段：

- 7 个 0—100 整数分数：`overall`、`education`、`experience`、`technical_skills`、`projects`、`research`、`evidence_quality`；
- `missing_requirements`：简历中缺少证据的硬性岗位要求列表。

程序会逐项校验候选人数量、顺序、临时 ID、分数范围和缺失条件的数据类型；只有整个响应通过校验才写入结果。写入时把 0—100 分归一化为 0—1。当前两个标注任务各使用 2 个并发 worker，总并发为 4，但每个请求本身仍是真正的 6 人 batch prompting，而不是把 6 个单人请求并发发送。

为了支持长时间运行，使用 SQLite WAL ledger 分别记录每次请求的协议哈希、状态、尝试次数、模型名以及输入/输出 token，并以“候选人 ID 列表 + 协议哈希”生成稳定 request ID。程序启动时先查询已经写入的候选人，只处理缺失部分。请求超时或返回格式错误时采用指数退避重试；一个 6 人批次多次失败后会二分为两个 3 人批次，继续失败则拆到单人，从而隔离异常简历。所有 HTTP 请求可以并发，但 SQLite 写入集中在主线程串行执行，避免数据库锁冲突。统计成本时累计所有已计费尝试，而不只计算最终成功请求。

### 4. 分歧采样

第二轮不是从未标注池中随机抽 1,000 人，而是读取首轮二分类模型已经冻结的全库分数，并先排除首轮 2,000 个训练 ID。采样过程不读取剩余候选人的历史真值，按固定 `seed=11` 分配五类预算：

| 采样策略 | 数量 | 排序依据 | 目的 |
|---|---:|---|---|
| Top score | 300 | OOF Stacking 最终分数从高到低 | 补充模型认为最可能通过的候选人 |
| Boundary | 250 | `abs(stacking_score - 0.5)` 从小到大 | 补充最接近决策边界、最不确定的样本 |
| Expert disagreement | 250 | 三个 proxy 分数的标准差从大到小 | 找到整体、经历和资质判断互相冲突的简历 |
| Router disagreement | 100 | Stacking 与 Embedding Router 分数差的绝对值从大到小 | 找到两种融合机制意见不一致的样本 |
| Coverage | 100 | 固定随机种子随机排序 | 保留一定的分布覆盖，避免全部集中在难例上 |

五类样本按顺序选择并全局去重；如果同一候选人同时满足多种策略导致某一配额不足，就继续按 boundary 不确定性补足，最终严格得到 1,000 个唯一 ID。程序同时保存每个入选样本的来源策略、三个 proxy 分数、两个 router 分数和专家标准差，便于之后分析哪类采样真正带来提升。

需要特别说明：这 1,000 条数据是由首轮**历史二分类标签训练的 router**选出的，并不是由尚未完成的多标签模型选择；当前也还没有单独运行“随机新增 1,000 条”的同成本控制组。因此，后续即使累计 3,000 条模型优于 2,000 条模型，也只能说明“增加这些分歧样本后有提升”，暂时不能把全部增益归因于主动采样策略本身。

## 三、已完成实验结果

已完成的正式实验使用固定的 2,000 条历史二分类标注训练轻量 proxy/router，在 job01 全部 34,761 名候选人上评估；未采样测试集包含 32,761 人，其中正例 294 人。

| 方法 | Unsampled AP | P@R80 | P@R90 |
|---|---:|---:|---:|
| Global proxy | 0.3595 | **11.07%** | 4.74% |
| Experience proxy | 0.3341 | 4.45% | 2.95% |
| Credentials proxy | 0.0767 | 2.13% | 1.59% |
| Expert Mean | 0.3472 | 7.57% | 5.20% |
| **OOF Stacking Router** | **0.3928** | 8.67% | **5.76%** |
| Embedding Router | 0.2255 | 6.17% | 3.44% |

主要结论如下：

- OOF stacking router 的 AP 为 **0.3928**，相比简单平均的 0.3472 提升 0.0456，约为 **13.1% 相对提升**，说明多个专业 proxy 的输出具有互补性，学习式融合优于直接平均。
- Global proxy 获得最高的单点 P@R80（11.07%），但在 20 个校准种子下达到 80% 召回的比例只有 35%，说明该高精度点的稳定性仍不足。
- Experience proxy 有一定独立贡献，stacking 中其系数最高；Credentials proxy 单独预测能力较弱，但 router 仍可能将其作为辅助信号使用。
- Embedding router 的 AP 只有 0.2255，明显弱于 stacking。当前数据量下，直接从高维 embedding 学习“应该相信哪个专家”较困难。

20 个种子的 job01-only R80 校准结果进一步显示：Global、Expert Mean 和 Stacking Router 的校准 F1 分别为 19.25%、14.41% 和 15.77%；对应的 Oracle-Overwrite AP 分别为 0.5790、0.5770 和 0.6063。这里的校准 F1 与历史表格中的 shared-mixture/max-F1 口径不同，因此不能直接横向比较。

## 四、与历史结果的对比

| 方法 | Unsampled AP | P@R80 | 说明 |
|---|---:|---:|---|
| LR-U | 0.3508 | 2.59% | 历史单模型基线 |
| US | 0.4155 | — | 历史结果未保存可重算 P@R80 的完整分数 |
| US3 | 0.4511 | — | 同上 |
| USA | **0.4798** | 5.80% | 当前历史最佳 AP |
| Qwen-Jev | 0.2121 | 1.32% | 历史 Qwen 结果 |
| Qwen-Doubao | 0.4414 | 10.06% | 历史 Qwen 结果 |
| Global proxy | 0.3595 | **11.07%** | 本周实验 |
| OOF Stacking Router | 0.3928 | 8.67% | 本周实验 |

Stacking Router 的 AP 比 LR-U 高 0.0420，但仍低于 US、US3、USA 和 Qwen-Doubao；Global proxy 的单点 P@R80 比 Qwen-Doubao 高 1.01 个百分点，但其跨种子稳定性不足。因此，本周结果支持“简历分块 + 多 proxy + router”这一方向，但还不能宣称全面超过已有最佳方法。另因历史实验与本周实验的采样 ID、随机种子并非完全一致，当前对比属于架构级参考，后续需要在同一固定 ID 上重跑才能形成严格结论。

### 实验结果分析

第一，**简历分块本身并不会自动提升效果，关键在于能否正确融合不同视角**。Global proxy 的 AP 为 0.3595，而三个 proxy 直接平均后降至 0.3472，说明把弱的 Credentials 分数等权加入会稀释 Global 的有效信号。OOF Stacking Router 则把 AP 提升到 0.3928：相比 Global 提高 0.0333（相对提升 9.3%），相比 Expert Mean 提高 0.0456（相对提升 13.1%）。这说明 Experience 和 Credentials 虽然单独排序能力较弱，但其中仍包含 Global 没有充分表达的补充证据，前提是由有监督模型学习其权重，而不是简单平均。

第二，**不同指标对应不同使用场景**。Global proxy 的 P@R80 最高，适合强调“召回 80% 时尽量少筛人”的场景；Stacking Router 的 AP 和 P@R90 在本组方法中最高，更适合关心整体排序质量或更高召回目标的场景。但 Global 的 R80 阈值在 20 个校准种子中只有 35% 真正达到目标，因此 11.07% 是一个较好的单点结果，不能理解为稳定性能。实际部署时仍应优先选择跨种子更可靠的阈值，或用召回置信下界约束阈值。

第三，**Embedding Router 目前没有证明“看简历后硬选专家”有效**。它的 AP 仅为 0.2255。主要原因可能包括：2,000 条数据相对高维 Global embedding 太少；“哪个 proxy 的单样本 log-loss 最小”这一三分类标签本身噪声较大；硬选最佳专家也会丢失多个 section 之间的互补信息。现阶段应保留 Stacking 的软融合思路，不宜继续把 Embedding Router 作为主模型扩大训练。

第四，**当前 LR Router 不能替代 Qwen，但显示了可与 Qwen 组合的价值**。Qwen-Doubao 的 AP 为 0.4414，比 Stacking Router 高 0.0486，说明 Qwen 对原始文本中的细粒度语义和岗位要求仍有明显优势；另一方面，Global proxy 的单点 P@R80 为 11.07%，比 Qwen-Doubao 的 10.06% 高 1.01 个百分点，说明 embedding 模型在部分高召回区域可能提供互补排序信号。合理方向不是在 LR 和 Qwen 中二选一，而是把 Qwen 作为更强的文本专家加入 section-aware router。

## 五、正在运行的实验

截至本周报记录时：

- 首轮 2,000 条多标签数据已完成 **1,143 条**；
- 第二轮 1,000 条分歧采样数据已完成 **979 条**；
- 两个最终模型（2,000 条多标签模型、累计 3,000 条模型）仍在等待标注完成，因而目前**没有可报告的最终多标签模型指标**。

Batch prompting 的阶段性效率较好：首轮已完成 192 次请求，第二轮已完成 165 次请求；相对于逐条请求，调用次数分别减少约 83.20% 和 83.15%。当前累计统计的输入/输出 token 分别为：首轮 2,154,070/655,483，第二轮 1,543,960/579,615。实验中曾出现代理连接超时，但 ledger 保留了失败记录，任务可从已有进度恢复，不需要重新标注已完成样本。

## 六、Qwen 模型说明

本周已完成的是基于现有 embedding 的轻量 LR proxy/router 架构实验，**尚未完成新的 Qwen 多标签训练**。原有 Qwen-Doubao 训练代码和结果均保持不变；本轮正在生成的多维标签，是下一阶段训练多任务 Qwen 或多个 Qwen proxy 的数据基础。因此目前不能把 0.3928 等指标表述为新的 Qwen 结果。

下一阶段可按以下两步把本周方法与 Qwen 连接起来。

### 1. 低成本方案：把 Qwen-Doubao 作为第四个专家

保留现有三个 LR section proxy，增加 Qwen-Doubao 对“JD + 完整简历文本”输出的通过概率，形成四个基础分数：

```text
Global LR ────────┐
Experience LR ────┤
Credentials LR ───┼─> OOF Stacking Router ─> 最终匹配概率
Qwen-Doubao ──────┘
```

Stacking 输入由原来的“三个 proxy 概率 + 三个 presence 标志”扩展为“四个专家概率 + section presence 标志”。该实验可以直接回答：Qwen 已经学到的文本语义，与 section embedding 模型提供的结构化证据是否互补。

需要注意，Qwen 在 router 训练集上的分数也必须是 OOF 分数。如果 Qwen-Doubao 已经用这 2,000 人训练过，就不能直接把其训练内预测交给 stacker，否则会产生泄漏。严格做法是对 Qwen 进行相同的 5 折训练并生成 OOF 分数；若计算成本过高，也可以从训练数据中单独留出一组只训练 router 的 calibration set，但不能同时训练 Qwen。

### 2. 完整方案：训练 Section-aware 多任务 Qwen

把一份简历按三个视角组织成带明确边界的文本输入：

| Qwen 分支 | 模型输入 | 训练目标 |
|---|---|---|
| Qwen-Global | JD + 完整简历 | `overall` |
| Qwen-Experience | JD + `work` + `projects` | `(experience + projects) / 2` |
| Qwen-Credentials | JD + `education` + `awards` + `languages` + `applications` | `(education + technical_skills + research) / 3` |

实现上优先采用“一个共享 Qwen 主干 + 三个回归/分类头”，而不是直接训练三套完整 Qwen：共享主干学习岗位和简历语义，三个任务头分别输出 Global、Experience、Credentials 分数。`evidence_quality` 可以作为样本权重，降低证据不足样本对训练的影响；`missing_requirements` 后续可以增加为辅助多标签任务，使模型显式识别缺失的硬性条件。三个 Qwen 分支的 OOF 输出再交给 Stacking Router 学习最终通过概率。

建议按下表组织对照实验，所有方法必须使用相同的 2,000 个训练 ID、相同的严格未采样测试集和相同的校准种子：

| 实验 | 模型 | 要回答的问题 |
|---|---|---|
| E0 | 原始 Qwen-Doubao | 固定协议下的 Qwen 基线是多少 |
| E1 | 三个 LR proxy + Stacking | 轻量 section 架构的基线是多少 |
| E2 | Qwen-Doubao + 三个 LR proxy + Stacking | Qwen 与 embedding section 信号是否互补 |
| E3 | 三头多任务 Qwen + Stacking | 多标签、分 section 监督是否优于单一 overall 标签 |
| E4 | E3 使用累计 3,000 条分歧采样标签 | 增加主动采样数据能否继续提升 |
| E4-Random | E3 使用 2,000 条原数据 + 随机 1,000 条 | E4 的提升来自采样策略还是仅来自数据量 |

最终统一报告 Unsampled AP、P@R80、P@R90、20-seed 校准达标率、训练/推理成本。只有 E4 明显超过 E4-Random，才能说明分歧采样本身有效；只有 E2 超过 E0 和 E1，才能证明 LR section proxy 与 Qwen 确实存在可利用的互补性。

## 七、问题与下一步计划

本周的主要发现是：学习式 stacking 能有效利用多个简历视角，但简单的 embedding router 效果较弱；Credentials 分支的标签或特征质量也需要加强。下一步计划为：

1. 完成剩余多标签标注，得到 2,000 条基线模型和累计 3,000 条分歧采样模型的正式结果；
2. 在固定训练/测试 ID 上比较二分类标签、多标签 2,000 条和多标签 3,000 条，统一报告 AP、P@R80、P@R90、校准可靠性和 LLM 成本；
3. 增加“随机新增 1,000 条”对照组，验证提升来自分歧采样策略，而不只是标注数量增加；
4. 将多标签监督接入 Qwen，优先尝试共享编码器加三个任务头（整体、经历/项目、资质），再在三个 Qwen 输出之上训练 router；
5. 找回并固定 USA seed11 的精确样本 ID，使新方法与历史最佳结果能够严格同协议比较。

## 八、本周总结

本周完成了从“单一整份简历评分”到“简历分块、多个 proxy、学习式 router”的完整实现和正式评估。OOF Stacking Router 在固定 2,000 条二分类标注下将 AP 从简单平均的 0.3472 提升到 0.3928，证明多视角专家存在可利用的互补信息；同时，batch prompting 已把 LLM 请求量降低约 83%，并接近完成 3,000 条多维标签。当前最重要的后续工作是完成多标签模型和 Qwen 训练，并通过固定样本 ID 与随机增量对照形成更严格、更有说服力的实验结论。
