"""Bind the original mini-swe-agent loop to the hiring workspace."""
import os
import sys
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from .data import llm_config, write_json
from .evaluation import evaluate
from .prompts import SYSTEM_PROMPT
from .semantic import SemanticOperator


def make_model():
    config = llm_config()
    name = os.environ.get("OKAGENT_CODE_MODEL") or (f"openai/{config['model']}" if config["model"] else None)
    if not name:
        raise ValueError("Configure _config/llm.json or OKAGENT_CODE_MODEL")
    if config["api_key"]:
        os.environ.setdefault("OPENAI_API_KEY", config["api_key"])
    return LitellmModel(model_name=name, cost_tracking="ignore_errors",
                        model_kwargs={"api_base": config["base_url"], "timeout": 120})


def make_environment(workspace, command_timeout=1800):
    return LocalEnvironment(cwd=str(workspace), timeout=command_timeout, env={
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })


def run_agent(run_dir, *, model=None, step_limit=80, command_timeout=1800):
    """Run code generation/execution, then evaluate a successfully submitted result."""
    if type(step_limit) is not int or step_limit <= 0 or command_timeout <= 0:
        raise ValueError("step_limit and command_timeout must be positive")
    run_dir = Path(run_dir).resolve()
    workspace = run_dir / "workspace"
    prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
    if model is None:
        if not os.environ.get("OKAGENT_LABEL_MODEL") and not llm_config()["model"]:
            raise ValueError("Configure _config/llm.json or OKAGENT_LABEL_MODEL")
        model = make_model()
    write_json(workspace / "output/usage.json", SemanticOperator(workspace).usage())
    env = make_environment(workspace, command_timeout)
    agent = DefaultAgent(model, env, system_template=SYSTEM_PROMPT, instance_template="{{ task }}",
                         step_limit=step_limit, cost_limit=0, output_path=run_dir / "agent.trajectory.json")
    result = agent.run(prompt)
    if result.get("exit_status") != "Submitted":
        raise RuntimeError(f"Code agent stopped: {result.get('exit_status')}; see agent.trajectory.json")
    for name in ("pipeline.py", "output/proxy.pkl", "output/report.md"):
        if not (workspace / name).is_file():
            raise FileNotFoundError(f"Code agent submitted without {name}; see agent.trajectory.json")
    return {"agent": result, "evaluation": evaluate(run_dir)}
