# OkAgent 实验比较报告（进行中）

## 实验设计

- 比较：单层 baseline code agent、logical/physical 两层 agent、`other_methods/hydra`。
- 开发任务：job01 full，34,761 名候选人。每个方法、每次新运行最多 2,000 次逐人标注请求。
- 所有方法使用 `_config/llm.json` 指定的相同真实接口；运行返回的实际模型版本为 `doubao-seed-2-0-lite-260215`。
- 对照 Hydra 使用 `target_recall=0.9`，与 agent 的参考目标一致；其余保持复现默认参数，首轮随机、seed=42。
- 评估：对全库的历史 LLM 标签计算 precision、recall、F1 与混淆矩阵。历史标签不是人工真值；实时标注模型与历史标签可能不一致。
- 每轮保存独立工作区、代码快照、轨迹、标注缓存与耗时，位于 `results/comparison/`。agent 不接触历史评估标签。
- 记录成功率、标注请求数、成功标签数、标注 token、agent 自身调用/token、墙钟耗时。模型端未提供本项目价格表，不将未知价格或 `cost_tracking=ignore_errors` 的 0 当作免费。
- job01 用于迭代开发；对它反复优化后的结果不是独立测试表现。后续将用未参与 prompt 调优的岗位复核。

## 迭代记录

### R0：原始 prompt 与运行流程

代码起点：`1d12748`。本轮仅增加实验快照/运行记录，以及成功标注响应的 token 和耗时记录；不改变标签决策或采样算法。

本轮在约 437 秒时主动终止，未产生可比较的部署结果。baseline / lo_ph / Hydra 分别记账 11 / 13 / 15 次，成功 10 / 12 / 14 次；中止时各有 1 个未完成请求。

发现并修复：
- 整段网络调用持有全局锁，独立候选人被串行处理；改为每 ID 去重锁 + 短预算锁，增加 `label_many`，Hydra 增加可选 `label_workers`（默认仍为 1）。
- baseline 先用 `nano` 失败，随后自行改用 heredoc；生成的 pipeline 还引用了不存在的 `get_label`，会在标注后失败。新增只读缓存接口，并在 prompt 明确非交互写文件与缓存覆盖方法。
- 原标注约 25–30 秒/人，含深度思考。R1 起所有方法统一使用 `thinking.type=disabled, temperature=0`，原匹配规则不变；独立 8 人真实请求探针在 10.23 秒内完成。这 8 次属于工程检查，不计入任何方法实验。

修复后 38 项测试通过，包含并发同 ID 去重、独立请求重叠、并发不超预算，以及 Hydra 串行/并发输出一致。R0 是工程诊断轮，不把未完成实验记为零召回。

### R1：并发真实调用与端到端运行

起点 `73d80da`，最大并发请求数的 prompt 建议为 8。方法可选择更低并发，因此墙钟耗时同时包含 agent 的实现选择。

运行中修复（同一轮保留缓存与预算）：
- Hydra 在第 128 次尝试后因 JSON 说明文字中的未转义换行中止；修复解析器后从缓存重新执行确定性的采样流程，原失败仍记账。
- baseline 的无必要 matplotlib 导入失败，由 agent 自行删去；后续出现 JSON 分隔符错误。标注接口统一启用 JSON 输出模式，另一次通用 API 探针验证成功，消耗 64 tokens。
- 解析器容忍代码围栏和字符串换行，仍核验 candidate_id 与布尔 is_match.result；保存失败响应的可用 token 统计。
- 修复前后的源文件和配置保存在 `results/comparison/r1/repairs/`。因此 R1 是含工程干预的开发轮，后续新工作区检验稳定配置。

两层 agent 在约 709 秒时因多次重复的产物格式错误被停止（250 次尝试、249 个成功标签、5 个正例），未形成部署结果。baseline 和 Hydra 继续运行。baseline 自行生成的阈值代码取 PR 曲线第一个合格项，实际趋向最低阈值；后续完整结果将保留这一缺陷，不由评测脚本替它调参。

### R2：将运行经验和 proxy 方法写入简短 prompt

- 两个 agent 共用 `PROXY_SKILL`：监督 embedding + 平衡 LR、随机与高分/边界混合采样、独立验证、选满足召回目标的最大阈值。
- 解释聚类弱不等于监督向量分类无效；批量读库并缓存特征，避免数万次逐人 SQL。
- 物理 agent 提交前核验 JSON 数组和字符串 summary；接口给出字段级错误并拒绝 JSONL，规划 agent 必须修复原产物格式，不能换样本回避。
- 保持 function 调用 → 同 workspace 新 code agent 的结构；没有增加服务、CLI 或新规划框架。

R2 从全新缓存运行，标签内容配置从开始固定为 JSON 输出、关闭思考、temperature=0。R1 剩余方法仍在后台完成，R2 改动依据已发现的具体故障和代码问题，不使用尚未产生的评估结果。

### R2 运行修复：共享接口限流

同时运行两个岗位后，接口返回 `RateLimitExceeded.EndpointTPMExceeded`。失败请求继续计入各自 2,000 次预算；它们没有可用 token 统计，不能据此推算账单。曾短暂暂停实验进程，并中断正在执行的脚本让 agent 从缓存恢复；这些干预均留档，墙钟包含等待，因此本轮不作严格速度排名。

新增可选 `label_interval`，缺省 0，本实验为 0.65 秒。按用户和 endpoint/model 共享一个本机锁文件，只串行约束请求起始时间，网络请求仍可重叠；不改变各 workspace 的预算、去重和标签语义。没有新增服务或队列框架。之后所有实验使用相同节奏，Hydra 的确定性流程从原缓存恢复。

## 方法启发

[Google Cloud 博客](https://cloud.google.com/blog/products/data-analytics/more-than-100x-faster-and-cheaper-llm-powered-sql-queries-with-proxy-models) 和其 [SIGMOD 2026 论文](https://arxiv.org/abs/2603.15970) 提醒：proxy 的效果需要通过独立样本验证，不能只凭训练指标决定替代 LLM；极端类别不均衡与 embedding 质量会限制简单 proxy。后续改动会作为本项目的具体经验写入 prompt，不把论文中其他数据集的加速比例当作本项目结果。

最终结果、各轮修改、运行故障和复现命令将在实验完成后补齐。
