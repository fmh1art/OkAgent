"""Autonomous Planner -> Coder -> execution -> observation, without a round cap."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from adaptive_eval import prepare_search
from job01_full_eval import source_identity
from react_runtime import run_react, save, parse_action

PLANNER_PROMPT = """你是人岗匹配实验的 Planner。根据数据画像和每轮实际 observation，逐步改善验证集上的模型效果。
每次只输出一个 <operator>{"operator":"自选逻辑算子名","params":{...},"reason":"本轮目的"}</operator>。
算子名、流程和模型由你决定；可以跳过不需要的步骤、重复算子、更换模型、调整参数、特征变换或阈值。
可用已安装的机器学习库，模型不限于逻辑回归和随机森林。不一次输出整个计划或多轮算子。
执行完本轮代码后你会收到真实结果，再决定下一轮。若不再值得继续改进，输出
<done>{"summary":"结论","model_round":最佳模型所在轮号}</done>。没有固定轮数上限。
训练数据和验证数据已准备好。验证集用于模型选择；最终测试由程序在done之后单独执行，不用于调参。
每轮输出model_file的模型会被独立计算验证指标。模型类别、参数、采样和阈值均由你选择。
关注类别不均衡，综合考虑precision、recall、F1和AP；不要只追求accuracy。不要重做历史LLM标注。
"""

CODER_PROMPT = """你是独立的 Coder，只实现这一次逻辑算子，返回 <code>完整Python脚本</code>，不加Markdown围栏。
脚本从sys.argv[1]读取JSON payload。包含 operator、round_dir、run_dir、context。
例如 p=json.load(open(sys.argv[1])); params=p['operator']['params']。operator已经是字典，无需json.loads。
context.dataset_contract给出训练与验证npz路径（每个文件含x、y、ids），历史observation给出先前产物。
你可以自由编写实际数据分析、抽样、特征变换、模型训练或阈值选择代码；不需要固定模板或固定算子目录。
使用已安装的numpy/scipy/sklearn/joblib等库。需要并行时使用n_jobs=2。只访问本次运行目录，产物写入本轮round_dir。
训练只用train；validation用于比较与调参。不要读取test数据或全量原始库，不访问网络或凭证。
将预处理和估计器一起保存为可由sklearn.base.clone重建的完整模型/流水线（joblib），便于最终重训。
训练模型或复用旧模型调整阈值后，stdout只输出单个JSON：
{"status":"ok","model_file":"实际保存的模型路径","threshold":0.5,"message":"简短说明"}。
模型应支持predict_proba、decision_function或predict；decision_function的分数会经sigmoid转换。
非训练算子输出真实的汇总JSON observation即可。不要输出候选人ID、原始文本或向量。
不要伪造指标；程序会独立验证模型。不要调用其他agent入口，不要预先执行未来算子。
"""


class ModelAgent:
    def __init__(self, client, model, api_style="chat"):
        self.client, self.model, self.api_style = client, model, api_style

    def complete(self, prompt, context):
        content = json.dumps(context, ensure_ascii=False)
        if self.api_style == "responses":
            return self.client.responses.create(model=self.model, instructions=prompt, input=content, max_output_tokens=8192).output_text
        response = self.client.chat.completions.create(model=self.model, messages=[
            {"role": "system", "content": prompt}, {"role": "user", "content": content}], max_tokens=8192)
        return response.choices[0].message.content or ""

    def next_operator(self, context):
        return self.complete(PLANNER_PROMPT, context)

    def write_code(self, operator_xml, context):
        _, operator = parse_action(operator_xml)
        request = {"operator": operator, "context": context}
        if context.get("history"):
            last = context["history"][-1]
            if last.get("status") == "error":
                code_file = Path(last["round_dir"]) / "code.py"
                if code_file.exists():
                    request["previous_failed_code"] = code_file.read_text(encoding="utf-8")[-12000:]
                    request["previous_error"] = last.get("observation")
        return self.complete(CODER_PROMPT, request)


class OfflineAgent:
    """A real local training smoke run, not a simulation of an online Planner."""
    def next_operator(self, context):
        successful = [r for r in context["history"] if r.get("observation", {}).get("validation")]
        if successful:
            return '<done>' + json.dumps({"summary": "Offline smoke complete", "model_round": successful[-1]["round"]}) + '</done>'
        return '<operator>{"operator":"TrainBaseline","params":{}}</operator>'

    def write_code(self, operator_xml, context):
        return '''<code>import json,sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
data=np.load(p["context"]["dataset_contract"]["train_file"])
model=make_pipeline(Normalizer(),LogisticRegression(class_weight="balanced",max_iter=2000,random_state=42))
model.fit(data["x"],data["y"])
path=Path(p["round_dir"])/"model.joblib"
joblib.dump(model,path)
print(json.dumps({"status":"ok","model_file":str(path)}))</code>'''


def trusted_evaluate(action, request, run_dir, directory, timeout):
    path = directory / (action + "_request.json")
    save(path, request)
    env = {"PATH": os.defpath, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("adaptive_eval.py")), action, str(path)],
                            env=env, cwd=run_dir, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    (directory / (action + "_stderr.log")).write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return json.loads(result.stdout)


def run_full_pipeline(config, run_dir, agent=None, timeout=7200, verbose=False):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt
            lock.write(b"0"); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {"test_size": 0.2, "validation_size": 0.2, **config,
                  "mode": "adaptive_v2", "source_identity": source_identity(Path(config["data_dir"]))}
        config_path = run_dir / "run_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Configuration/source changed or old fixed-flow run; use a new directory")
        save(config_path, config)
        data = prepare_search(config, run_dir)
        profile = {k: v for k, v in data.items() if k not in {"source_identity"}}
        context = {"goal": "Improve job01 matching on validation data, then select a final model",
                   "data_profile": profile,
                   "dataset_contract": {"train_file": str(run_dir / "data/train.npz"),
                                        "validation_file": str(run_dir / "data/validation.npz"),
                                        "arrays": ["x", "y", "ids"], "dimension": config["dimension"]},
                   "libraries": ["numpy", "scipy", "sklearn", "joblib", "duckdb"]}

        def observe(observation, directory):
            if "model_file" not in observation:
                return observation
            return trusted_evaluate("validate", {"run_dir": str(run_dir), "round_dir": str(directory),
                                                  "observation": observation}, run_dir, directory, timeout)

        def finalize(done, history):
            candidates = [r for r in history if r.get("status") == "ok" and r.get("observation", {}).get("validation")]
            if not candidates:
                raise ValueError("No validated model yet; train and save a model before done")
            if "model_round" in done:
                chosen = next((r for r in candidates if r["round"] == done["model_round"]), None)
                if chosen is None:
                    raise ValueError("model_round must identify a successfully validated model")
            else:
                chosen = max(candidates, key=lambda r: r["observation"]["validation"]["f1"])
            result = trusted_evaluate("finalize", {"run_dir": str(run_dir), "chosen": chosen}, run_dir, run_dir, timeout)
            return {**result, "agent_mode": "online_planner_coder" if agent else "offline_smoke",
                    "full_rows": data["row_count"], "splits": data["splits"]}

        actor = agent or OfflineAgent()
        return run_react(actor, actor, context, run_dir, observe=observe, finalize=finalize,
                         timeout=timeout, verbose=verbose)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2, help="Validation fraction within the non-test data")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "ep-20250612104210-ss27q"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"))
    parser.add_argument("--api-style", choices=["chat", "responses"], default="chat")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-code execution timeout; not a round limit")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        parser.error("Split fractions must be between 0 and 1")
    config = {"data_dir": str(args.full_data_dir.resolve()), "seed": args.seed, "expected_count": 34761,
              "dimension": 2048, "test_size": args.test_size, "validation_size": args.validation_size,
              "model": args.model, "base_url": args.base_url, "api_style": args.api_style, "offline": args.offline}
    actor = None
    if not args.offline:
        from openai import OpenAI
        key = os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            parser.error("Set ARK_API_KEY or OPENAI_API_KEY")
        actor = ModelAgent(OpenAI(api_key=key, base_url=args.base_url, timeout=180, max_retries=2), args.model, args.api_style)
    run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / ("job01_adaptive_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    result = run_full_pipeline(config, run_dir, actor, args.timeout, args.verbose)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
