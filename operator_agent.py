#!/usr/bin/env python3
"""逐算子 ReAct：Planner 选算子，Coder 写代码，Executor 运行并反馈。"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
SPLITS = {"full", "train", "test"}
OPERATOR_RE = re.compile(r"\A\s*<operator>(.*?)</operator>\s*\Z", re.DOTALL)
CODE_RE = re.compile(r"\A\s*<code>(.*?)</code>\s*\Z", re.DOTALL)
DONE_RE = re.compile(r"\A\s*<done>(.*?)</done>\s*\Z", re.DOTALL)


def _load_rows(split: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / split).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def dataset_profile(split: str) -> dict[str, Any]:
    """只返回聚合信息，不把候选人原文交给 Planner。"""
    rows = _load_rows(split)
    labels = Counter(str(row.get("label")) for row in rows)
    attributes = Counter(name for row in rows for name in row.get("attributes", {}))
    return {
        "split": split,
        "row_count": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "attribute_presence": dict(sorted(attributes.items())),
    }


def inspect_schema(split: str) -> dict[str, Any]:
    rows = _load_rows(split)
    row_keys: set[str] = set()
    attribute_keys: dict[str, set[str]] = {}
    for row in rows:
        row_keys.update(row)
        for name, value in row.get("attributes", {}).items():
            attribute_keys.setdefault(name, set()).update(value)
    return {
        "split": split,
        "row_keys": sorted(row_keys),
        "attribute_keys": {
            name: sorted(keys) for name, keys in sorted(attribute_keys.items())
        },
    }


DATA_TOOLS = {"dataset_profile": dataset_profile, "inspect_schema": inspect_schema}

TOOLS = [
    {
        "type": "function",
        "name": "dataset_profile",
        "description": "读取指定数据集的行数、标签分布和字段覆盖率。",
        "parameters": {
            "type": "object",
            "properties": {"split": {"type": "string", "enum": sorted(SPLITS)}},
            "required": ["split"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "inspect_schema",
        "description": "读取顶层和嵌套字段名，不返回个人信息值。",
        "parameters": {
            "type": "object",
            "properties": {"split": {"type": "string", "enum": sorted(SPLITS)}},
            "required": ["split"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


PLANNER_PROMPT = """你是逐轮决策的 Planner。每次根据数据画像和上一轮实际 observation，
只输出一个 <operator>{"operator":"自选算子名","params":{...},"reason":"本轮目的"}</operator>。
算子名、顺序、模型和参数由你决定，可以重复或跳过算子，自定义新算子。不要一次生成整个流程。
可用只读工具了解数据。Coder执行后会把结果交回给你，再决定下一步。
效果不足时可以改变模型、参数、采样或特征；从训练集内部划分验证集用于调整，不要用测试集调参。
觉得目标达成时输出 <done>完成摘要</done>。没有固定轮数上限。
"""

CODER_PROMPT = """你是独立 Coder，只为当前这一个逻辑算子生成真实可执行的 Python 代码。
返回 <code>完整脚本</code>，脚本从sys.argv[1]读取JSON payload，含root、run_dir、round_dir、operator和context。
context.dataset_contract描述数据格式。标准库和已安装的机器学习库可用，自主实现模型和处理方法，
不用固定调用模板。原始数据只读，产物写在本轮round_dir。stdout只输出一个汇总JSON observation。
从train内部划分validation比较模型；标签使用已有label，不访问网络，不读取凭证。
不得输出候选人文本、ID或向量到observation。训练或评估错误如实报告，禁止伪造结果。
"""


class Planner(Protocol):
    def next_operator(self, context: dict[str, Any]) -> str: ...


class Coder(Protocol):
    def write_code(self, operator_xml: str, context: dict[str, Any]) -> str: ...


def parse_operator(text: str, row_count: int) -> dict[str, Any]:
    from react_runtime import parse_action
    kind, spec = parse_action(text)
    if kind != "operator":
        raise ValueError("Expected one logical operator")
    return spec


def parse_done(text: str) -> str:
    match = DONE_RE.fullmatch(text)
    if not match:
        raise ValueError("Planner 输出既非 <operator> 也非 <done>")
    summary = match.group(1).strip()
    if not summary:
        raise ValueError("<done> 内容不能为空")
    return summary


def parse_code(text: str) -> str:
    match = CODE_RE.fullmatch(text)
    code = (match.group(1) if match else text).strip()
    fenced = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", code, re.DOTALL | re.IGNORECASE)
    code = fenced.group(1).strip() if fenced else code
    if not code:
        raise ValueError("Coder returned empty code")
    ast.parse(code)
    return code + "\n"


ALLOWED_IMPORTS = {
    "__future__", "importlib",
    "json", "sys", "pathlib", "collections", "hashlib", "random", "math",
    "glob", "os", "duckdb", "numpy", "sklearn", "joblib",
    "scipy", "pandas", "warnings", "csv", "time", "datetime", "statistics",
    "itertools", "functools", "typing", "threadpoolctl", "xgboost", "lightgbm", "catboost",
}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
BANNED_ATTRIBUTES = {
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
    "spawnve", "spawnvp", "spawnvpe", "remove", "removedirs", "rmdir", "unlink",
    "rename", "renames", "chmod", "chown", "getenv", "environ",
}


def validate_generated_code(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names):
                raise ValueError("生成代码导入了未允许的模块")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError("生成代码导入了未允许的模块")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BANNED_CALLS:
                raise ValueError(f"生成代码调用了禁用函数 {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            raise ValueError(f"生成代码访问了禁用属性 {node.attr}")


def execute_code(
    code_xml: str,
    payload: dict[str, Any],
    run_dir: Path,
    step_number: int,
    timeout: int = 120,
) -> dict[str, Any]:
    code = parse_code(code_xml)
    validate_generated_code(code)
    name = payload["operator"]["operator"].lower()
    script_path = run_dir / f"step_{step_number:02d}_{name}.py"
    payload_path = run_dir / f"step_{step_number:02d}_input.json"
    script_path.write_text(code, encoding="utf-8")
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
        "PATH": os.defpath,
    }
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-I", str(script_path), str(payload_path)],
        cwd=run_dir,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"算子代码退出码 {completed.returncode}: {completed.stderr.strip()[:1000]}"
        )
    output = completed.stdout.strip()
    try:
        observation = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"算子 stdout 不是单个 JSON：{output[:500]}") from exc
    if not isinstance(observation, dict):
        raise ValueError("算子 observation 必须是 JSON 对象")
    observation["code_file"] = str(script_path)
    return observation


class OpenAIPlanner:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def next_operator(self, context: dict[str, Any]) -> str:
        inputs: list[Any] = [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        for _ in range(6):
            response = self.client.responses.create(
                model=self.model,
                instructions=PLANNER_PROMPT,
                input=inputs,
                tools=TOOLS,
                parallel_tool_calls=False,
                max_output_tokens=1200,
            )
            inputs.extend(response.output)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                return response.output_text
            for call in calls:
                args = json.loads(call.arguments)
                try:
                    result = DATA_TOOLS[call.name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
        raise RuntimeError("Planner 工具调用超过限制")


class OpenAICoder:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def write_code(self, operator_xml: str, context: dict[str, Any]) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=CODER_PROMPT,
            input=json.dumps(
                {"operator_xml": operator_xml, "context": context}, ensure_ascii=False
            ),
            max_output_tokens=3500,
        )
        return response.output_text


def _operator_xml(spec: dict[str, Any]) -> str:
    return f"<operator>{json.dumps(spec, ensure_ascii=False)}</operator>"


def run_pipeline(planner: Planner, coder: Coder, run_dir: Path,
                 verbose: bool = False) -> dict[str, Any]:
    from react_runtime import run_react
    context = {
        "goal": "自主改进岗位匹配，逐轮执行并观察结果",
        "root": str(ROOT),
        "data_profile": dataset_profile("full"),
        "schema": inspect_schema("full"),
        "dataset_contract": {
            "train_glob": "train/*.jsonl", "test_glob": "test/*.jsonl",
            "embedding_file": "embeddings/embeddings_00001.parquet",
            "embedding_join_key": "attributes.unstructured.embedding_id -> embedding_id",
            "embedding_column": "embedding", "embedding_dimension": 2048,
            "existing_label": "label",
        },
    }
    return run_react(planner, coder, context, run_dir, verbose=verbose)


def main() -> None:
    if "--full-data-dir" in sys.argv:
        from job01_full_agent import main as full_main
        full_main()
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="逐算子 Planner-Coder ReAct Agent")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or ROOT / ".agent_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("请先运行 pip install -r requirements.txt") from exc
    api_key = os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请设置 ARK_API_KEY 或 OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    planner = OpenAIPlanner(client, args.model)
    coder = OpenAICoder(client, args.model)
    result = run_pipeline(planner, coder, run_dir.resolve(), args.verbose)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
