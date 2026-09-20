"""Shared LoRA trainer/scorer for the online-Qwen proxy experiment.

The code agent supplies paths and a small Qwen model name; this module owns text
construction, fine-tuning, validation, threshold selection, and scoring so every
stage uses exactly the same implementation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path


BACKEND = "qwen_online_lora"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
TEXT_POLICY = "full_unstructured_chunks_v1"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_labels(paths: list[Path]) -> list[dict]:
    labels: dict[str, int] = {}
    for path in paths:
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
                raise ValueError(f"{path}: invalid label row")
            label = row.get("label")
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"{path}: label must be integer 0/1")
            cid = row["candidate_id"]
            if cid in labels and labels[cid] != label:
                raise ValueError(f"conflicting labels for {cid}")
            labels[cid] = label
    return [{"candidate_id": cid, "label": labels[cid]} for cid in sorted(labels)]


def read_ids(path: Path) -> list[str]:
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array")
    ids = [row["candidate_id"] if isinstance(row, dict) else row for row in rows]
    if any(not isinstance(cid, str) for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: candidate IDs must be unique strings")
    return ids


def load_resume_texts(workspace: Path, ids: list[str]) -> dict[str, str]:
    import duckdb

    wanted = set(ids)
    parts: dict[str, list[str]] = {}
    with duckdb.connect(str(workspace / "data.duckdb"), read_only=True) as con:
        rows = con.execute(
            "SELECT candidate_id, text FROM candidate_segments "
            "WHERE segment='unstructured' ORDER BY candidate_id, text"
        ).fetchall()
    for cid, value in rows:
        if cid not in wanted:
            continue
        parts.setdefault(cid, []).append(value or "")
    missing = wanted - parts.keys()
    if missing:
        raise ValueError(f"missing unstructured text for {len(missing)} candidates")
    return {cid: "\n\n".join(values) for cid, values in parts.items()}


def job_context(workspace: Path) -> str:
    job = _read_json(workspace / "job.json")
    requirements = job.get("must_have_qualifications", [])
    if not isinstance(requirements, list):
        raise ValueError("job.must_have_qualifications must be a list")
    return "岗位：{}\n必须条件：\n{}".format(
        job.get("job_title", "未知岗位"),
        "\n".join(f"- {item}" for item in requirements),
    )


def _prompt_parts(context: str) -> tuple[str, str]:
    return ("你是招聘匹配分类器。根据岗位必须条件判断候选人是否匹配。\n"
            f"{context}\n\n候选人简历：\n", "\n\n输出匹配类别。")


def build_text(context: str, resume: str) -> str:
    prefix, suffix = _prompt_parts(context)
    return prefix + resume + suffix


def resume_chunks(tokenizer, context: str, resume: str, max_length: int) -> list[list[int]]:
    """Encode every resume token exactly once, repeating job context per chunk."""
    add_special = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    wrap = add_special if callable(add_special) else lambda ids: ids
    prefix, suffix = _prompt_parts(context)
    prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False, truncation=False)["input_ids"]
    resume_ids = tokenizer(resume, add_special_tokens=False, truncation=False)["input_ids"]
    special_count = len(wrap([]))
    capacity = max_length - len(prefix_ids) - len(suffix_ids) - special_count
    if capacity <= 0:
        raise ValueError(f"max_length={max_length} cannot fit the job prompt and classification suffix")
    parts = [resume_ids[start:start + capacity] for start in range(0, len(resume_ids), capacity)]
    if not parts:
        parts = [[]]
    chunks = [wrap(prefix_ids + part + suffix_ids) for part in parts]
    if any(len(chunk) > max_length for chunk in chunks):
        raise AssertionError("resume chunk exceeded max_length")
    return chunks


def _seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _imports():
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return torch, LoraConfig, PeftModel, TaskType, get_peft_model, AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_name: str, *, adapter: Path | None, trainable: bool, device: str,
               rank: int = 8, alpha: int = 16):
    (torch, LoraConfig, PeftModel, TaskType, get_peft_model,
     AutoModelForSequenceClassification, AutoTokenizer) = _imports()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA PyTorch is unavailable")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, dtype=dtype, local_files_only=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    if adapter is None:
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score"],
        )
        model = get_peft_model(base, config)
    else:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    model.to(device)
    return torch, tokenizer, model


def _batches(rows: list[dict], batch_size: int, *, shuffle: bool, seed: int):
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start:start + batch_size]]


def train_adapter(*, workspace: Path, rows: list[dict], model_name: str, output: Path,
                  previous_adapter: Path | None, device: str, max_length: int,
                  batch_size: int, epochs: int, learning_rate: float, seed: int):
    torch, tokenizer, model = load_model(
        model_name, adapter=previous_adapter, trainable=True, device=device,
    )
    counts = [sum(row["label"] == value for row in rows) for value in (0, 1)]
    if min(counts) == 0:
        raise ValueError("online Qwen training requires both classes")
    weights = torch.tensor(
        [len(rows) / (2 * count) for count in counts], dtype=torch.float32, device=device,
    )
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("LoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    context = job_context(workspace)
    chunks = []
    chunks_per_candidate = {}
    for row in rows:
        encoded_chunks = resume_chunks(tokenizer, context, row["text"], max_length)
        chunks_per_candidate[row["candidate_id"]] = len(encoded_chunks)
        chunks.extend({"candidate_id": row["candidate_id"], "label": row["label"],
                       "input_ids": ids, "candidate_weight": 1 / len(encoded_chunks)}
                      for ids in encoded_chunks)
    model.train()
    losses = []
    mean_chunks = len(chunks) / len(rows)
    for epoch in range(epochs):
        for batch in _batches(chunks, batch_size, shuffle=True, seed=seed + epoch):
            encoded = tokenizer.pad({"input_ids": [row["input_ids"] for row in batch]},
                                    padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([row["label"] for row in batch], dtype=torch.long, device=device)
            candidate_weights = torch.tensor([row["candidate_weight"] for row in batch],
                                             dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device == "cuda" and torch.cuda.is_bf16_supported()):
                logits = model(**encoded).logits
                per_chunk = torch.nn.functional.cross_entropy(
                    logits.float(), labels, weight=weights, reduction="none")
                loss = (per_chunk * candidate_weights).sum() * mean_chunks / len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_counts": counts,
        "class_weights": [float(value) for value in weights.detach().cpu()],
        "mean_loss": sum(losses) / len(losses),
        "steps": len(losses),
        "trainable_parameters": sum(param.numel() for param in params),
        "candidate_count": len(rows),
        "chunk_count": len(chunks),
        "max_chunks_per_candidate": max(chunks_per_candidate.values()),
        "text_policy": TEXT_POLICY,
    }


def score_rows(*, workspace: Path, rows: list[dict], model_name: str, adapter: Path,
               device: str, max_length: int, batch_size: int) -> list[dict]:
    torch, tokenizer, model = load_model(
        model_name, adapter=adapter, trainable=False, device=device,
    )
    context = job_context(workspace)
    model.eval()
    scores = {}
    pending = []

    def flush():
        if not pending:
            return
        encoded = tokenizer.pad({"input_ids": [part["input_ids"] for part in pending]},
                                padding=True, return_tensors="pt").to(device)
        probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1].cpu().tolist()
        for part, probability in zip(pending, probabilities):
            cid = part["candidate_id"]
            scores[cid] = max(scores.get(cid, 0.0), float(probability))
        pending.clear()

    with torch.inference_mode():
        for row in rows:
            for ids in resume_chunks(tokenizer, context, row["text"], max_length):
                pending.append({"candidate_id": row["candidate_id"], "input_ids": ids})
                if len(pending) >= batch_size:
                    flush()
        flush()
    return [{"candidate_id": row["candidate_id"], "score": scores[row["candidate_id"]]}
            for row in rows]


def select_threshold(labels: list[int], scores: list[float], target_recall: float,
                     min_validation_positive: int = 3):
    from sklearn.metrics import precision_recall_curve

    if len(set(labels)) < 2 or sum(labels) < min_validation_positive:
        return None, {"proxy_valid": False,
                      "reason": f"validation needs both classes and at least {min_validation_positive} positives",
                      "validation_positive": sum(labels)}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = [index for index in range(len(thresholds)) if recall[index] >= target_recall]
    if not eligible:
        return None, {"proxy_valid": False, "reason": "no threshold reaches target recall"}
    best = max(eligible, key=lambda index: (precision[index], thresholds[index]))
    threshold = float(thresholds[best])
    predictions = [score >= threshold for score in scores]
    tp = sum(pred and label == 1 for pred, label in zip(predictions, labels))
    fp = sum(pred and label == 0 for pred, label in zip(predictions, labels))
    fn = sum(not pred and label == 1 for pred, label in zip(predictions, labels))
    return threshold, {
        "proxy_valid": True,
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn,
    }


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def fit(args):
    workspace = Path(args.workspace).resolve()
    output = _resolve(workspace, args.output_dir).resolve()
    train_paths = [_resolve(workspace, value) for value in args.train]
    validation_path = _resolve(workspace, args.validation)
    train_labels = merge_labels(train_paths)
    validation_labels = merge_labels([validation_path])
    train_ids = {row["candidate_id"] for row in train_labels}
    validation_ids = {row["candidate_id"] for row in validation_labels}
    if train_ids & validation_ids:
        raise ValueError("training and validation IDs overlap")
    texts = load_resume_texts(workspace, sorted(train_ids | validation_ids))
    train_rows = [{**row, "text": texts[row["candidate_id"]]} for row in train_labels]
    validation_rows = [{**row, "text": texts[row["candidate_id"]]} for row in validation_labels]
    previous_adapter = None
    if args.previous_artifact:
        previous = pickle.loads(_resolve(workspace, args.previous_artifact).read_bytes())
        if (previous.get("backend") != BACKEND or previous.get("model_name") != args.model_name
                or previous.get("text_policy") != TEXT_POLICY):
            raise ValueError("previous artifact backend/model/text policy does not match")
        previous_adapter = _resolve(workspace, previous["adapter_path"])
    _seed_everything(args.seed)
    adapter = output / "adapter"
    training = train_adapter(
        workspace=workspace, rows=train_rows, model_name=args.model_name, output=adapter,
        previous_adapter=previous_adapter, device=args.device, max_length=args.max_length,
        batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
        seed=args.seed,
    )
    validation_scores = score_rows(
        workspace=workspace, rows=validation_rows, model_name=args.model_name, adapter=adapter,
        device=args.device, max_length=args.max_length, batch_size=args.score_batch_size,
    )
    threshold, metrics = select_threshold(
        [row["label"] for row in validation_rows],
        [row["score"] for row in validation_scores], args.target_recall,
        args.min_validation_positive,
    )
    _write_json(output / "cumulative_train.json", train_labels)
    _write_json(output / "validation_scores.json", validation_scores)
    artifact = {
        "backend": BACKEND,
        "model_name": args.model_name,
        "text_policy": TEXT_POLICY,
        "chunk_aggregation": "max_probability",
        "adapter_path": str(adapter.relative_to(workspace)),
        "max_length": args.max_length,
        "threshold": threshold,
        "proxy_valid": bool(metrics["proxy_valid"]),
        "target_recall": args.target_recall,
        "min_validation_positive": args.min_validation_positive,
        "train_count": len(train_rows),
        "train_positive": sum(row["label"] for row in train_rows),
        "validation_count": len(validation_rows),
        "validation_positive": sum(row["label"] for row in validation_rows),
        "training": training,
        "validation_metrics": metrics,
        "seed": args.seed,
    }
    (output / "proxy.pkl").write_bytes(pickle.dumps(artifact))
    _write_json(output / "metadata.json", artifact)
    print(json.dumps(artifact, ensure_ascii=False))


def score(args):
    workspace = Path(args.workspace).resolve()
    artifact_path = _resolve(workspace, args.artifact)
    artifact = pickle.loads(artifact_path.read_bytes())
    if artifact.get("backend") != BACKEND or artifact.get("text_policy") != TEXT_POLICY:
        raise ValueError(f"artifact must use backend {BACKEND} and text policy {TEXT_POLICY}; retrain old adapters")
    ids = read_ids(_resolve(workspace, args.ids))
    texts = load_resume_texts(workspace, ids)
    rows = [{"candidate_id": cid, "text": texts[cid]} for cid in ids]
    scores = score_rows(
        workspace=workspace, rows=rows, model_name=artifact["model_name"],
        adapter=_resolve(workspace, artifact["adapter_path"]), device=args.device,
        max_length=artifact["max_length"], batch_size=args.batch_size,
    )
    _write_json(_resolve(workspace, args.output), scores)
    print(json.dumps({"count": len(scores), "output": args.output}, ensure_ascii=False))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--workspace", default=".")
    fit_parser.add_argument("--model-name", default=DEFAULT_MODEL)
    fit_parser.add_argument("--train", action="append", required=True)
    fit_parser.add_argument("--validation", required=True)
    fit_parser.add_argument("--output-dir", required=True)
    fit_parser.add_argument("--previous-artifact")
    fit_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    fit_parser.add_argument("--max-length", type=int, default=512)
    fit_parser.add_argument("--batch-size", type=int, default=8)
    fit_parser.add_argument("--score-batch-size", type=int, default=16)
    fit_parser.add_argument("--epochs", type=int, default=2)
    fit_parser.add_argument("--learning-rate", type=float, default=2e-4)
    fit_parser.add_argument("--target-recall", type=float, default=0.9)
    fit_parser.add_argument("--min-validation-positive", type=int, default=3)
    fit_parser.add_argument("--seed", type=int, default=42)
    fit_parser.set_defaults(func=fit)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--workspace", default=".")
    score_parser.add_argument("--artifact", required=True)
    score_parser.add_argument("--ids", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    score_parser.add_argument("--batch-size", type=int, default=16)
    score_parser.set_defaults(func=score)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("batch size must be positive")
    if hasattr(args, "epochs") and args.epochs <= 0:
        raise ValueError("epochs must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
