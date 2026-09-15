# 岗位 01 自主 Agent（v2）

处理全部 34,761 名候选人，以现有的 2048 维整体画像向量和历史 LLM 标签开展模型实验。
独立的 `baseline.py` 仍保留原来的单次 80/20 逻辑回归基线。

## 逐轮决策

每轮只执行：Planner 生成一个逻辑算子 → Coder 生成本轮 Python 代码 → 子进程执行 →
返回实际 observation → Planner 决定下一轮。

- 算子名不限于 Partition/Sample/Label/Proxy/Deploy，可以自定义、重复、跳过。
- 没有固定算子顺序，没有最大 12 轮或其他总轮数限制。
- Planner 可自主选择已安装库中的模型、参数、特征变换、采样和阈值。
- Coder 编写实际实现代码，不再被限制为固定函数调用模板。
- 模型效果不好时，把验证指标交回 Planner，由它决定重复训练、更换模型或做其他分析。
- Planner 输出 `<done>` 时结束；也可由用户中断。API 鉴权/网络故障在 SDK 重试耗尽后暂停并留存轨迹，
  不会在无法访问模型时无限扣费。单次代码执行保留超时（默认 7200 秒），它不是总轮数上限。

输出示例仅是协议示范，不是算子目录或强制流程：

```text
<operator>{"operator":"TrainProxy","params":{"model":"自选模型"},"reason":"本轮目的"}</operator>
```

```text
<done>{"summary":"选择验证表现最好的模型","model_round":4}</done>
```

done 中不指定 model_round 时，默认选择验证 F1 最高的成功模型。

## 数据与评估

为支持反复调参，准备阶段一次性固定训练、验证、测试划分；它不是强制的逻辑算子。
默认先划分 80% 开发数据和 20% 测试数据，再从开发数据中划出 20% 作验证集。

- 训练：22,246 人；验证：5,562 人；测试：6,953 人。种子为 42。
- Planner 可以反复查看验证结果，但不把测试集结果用于挑选模型。
- Coder 返回一个已保存的完整 sklearn 模型/流水线时，独立子进程重新计算验证指标，
  包括 precision、recall、F1、AP、ROC-AUC、混淆矩阵。
- 每轮模型保存独立快照，不锁死后续轮次的模型设置。允许同一个模型更换阈值再评估。
- 模型支持 predict_proba、decision_function 或 predict。decision_function 经 sigmoid 变为分数，
  该分数不应视为已校准的概率。
- done 后将所选模型克隆并在训练+验证的 27,808 人上重训，最终只对保留测试集评估。
- 所选模型需支持 sklearn.base.clone；预处理应一起放进 Pipeline，确保重训与推理一致。
- 标签代表历史 LLM 匹配结果，不等同于人工确认或实际录用结果。

输入库以只读方式打开。Planner/Coder 获得统计、路径和汇总观察，不发送候选人文本或向量。
生成代码使用独立进程且不继承 API Key；保留基础 AST 检查，但这不是操作系统级安全沙箱。

## 运行与轨迹

服务器目录：`/home/mengsq/apps/job01-full-agent`。
方舟配置继续使用 `~/.config/job01-full-agent/ark.env`，不放入项目或部署包。

```bash
cd /home/mengsq/apps/job01-full-agent
nohup bash run_server_job01.sh "$PWD/runs/job01_adaptive_v2" \
  > runs/job01_adaptive_v2.log 2>&1 < /dev/null &
tail -f runs/job01_adaptive_v2.log
```

相同目录可恢复中断的运行，不会覆盖旧轮次；新实验使用新的目录。
不要用旧的五折运行目录启动 v2。代码会拒绝配置或数据不一致的恢复。

每轮都会留下轨迹，失败也记录：

```text
runs/job01_adaptive_v2/
  trajectory.jsonl              # 每轮完整记录：算子、结果、状态、耗时
  events.jsonl                  # 规划/生成/执行等阶段事件，实时追加
  checkpoint.json               # 已完成轮次
  rounds/000001/
    round.json                  # 本轮状态，生成过程中也更新
    planner_input.json          # Planner 当轮看到的上下文
    planner_response.txt        # Planner 原始输出
    operator.json               # 本轮逻辑算子与参数
    coder_response.txt          # Coder 原始输出
    code.py                     # 实际执行代码
    payload.json                # 代码输入
    stdout.log / stderr.log     # 包括报错/超时前的输出
    validation.json             # 本轮模型的独立验证结果（有模型时）
    validated_model.joblib      # 本轮模型快照（有模型时）
  result.json                   # 完成后的最终结果
  final_metrics.json            # 独立测试结果
  final_model.joblib            # 选定设置重训后的最终模型
  test_predictions.csv          # 最终测试集逐人预测
```

查看算子轨迹：

```bash
tail -f runs/job01_adaptive_v2/trajectory.jsonl
```

`--offline` 是不调用 API 的真实训练冒烟模式；它使用本地默认决策，不能当成在线自主规划的证据。

## 代码与测试

- `react_runtime.py`：无轮数上限的逐轮循环、代码执行、失败反馈、轨迹与恢复。
- `job01_full_agent.py`：Planner/Coder 提示词、全量数据入口、模型验证与结束处理。
- `adaptive_eval.py`：准备数据、独立验证、选定模型的最终重训和测试。
- `operator_agent.py`：保留原 JSONL 子集入口，也使用新循环；传 --full-data-dir 进入新版全量模式。
- `job01_full_eval.py`：保留旧五折实现供历史结果复核；新版只复用它的只读数据检查函数。

```bash
python -m unittest test_react_runtime test_job01_full -v
```

本地还可加入 `test_operator_agent` 检查原 3,000 人 JSONL 子集入口（该样例数据未复制到服务器）。
测试包含 16 次重复自定义算子、超过 12 轮仍继续、每轮反馈、执行失败后的重试、
中断恢复、完整训练/验证/测试流程和数据 ID 检查。旧版本主要文件备份在本地
`.agent_backups/before-autonomous/`，历史结果不改写。
