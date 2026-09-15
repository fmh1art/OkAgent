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

## 方法启发

[Google Cloud 博客](https://cloud.google.com/blog/products/data-analytics/more-than-100x-faster-and-cheaper-llm-powered-sql-queries-with-proxy-models) 和其 [SIGMOD 2026 论文](https://arxiv.org/abs/2603.15970) 提醒：proxy 的效果需要通过独立样本验证，不能只凭训练指标决定替代 LLM；极端类别不均衡与 embedding 质量会限制简单 proxy。后续改动会作为本项目的具体经验写入 prompt，不把论文中其他数据集的加速比例当作本项目结果。

最终结果、各轮修改、运行故障和复现命令将在实验完成后补齐。
