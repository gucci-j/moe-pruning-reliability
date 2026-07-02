from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


ReferenceMode = Literal["input", "target"]
HalluGranularity = Literal["document", "sentence"]


@dataclass
class EvalRecord:
    source: str
    subset: str
    sample_id: str
    prediction: str
    reference: str
    target: str
    metadata: Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _pick_first_key(item: Dict[str, Any], candidates: Sequence[str]) -> str:
    for key in candidates:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for element in value:
                if isinstance(element, str) and element.strip():
                    parts.append(element.strip())
            if parts:
                return "\n".join(parts)
    available = ", ".join(sorted(item.keys()))
    raise KeyError(
        f"None of the expected keys {list(candidates)} found with non-empty string values. "
        f"Available keys: [{available}]"
    )


def load_rct_records(path: Path) -> List[EvalRecord]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("RCT JSON must be a list of objects.")

    records: List[EvalRecord] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"RCT item at index {idx} is not an object.")

        prediction = _pick_first_key(item, ["prediction", "generated", "summary", "output", "candidate"])
        reference = _pick_first_key(item, ["reference", "document", "source", "context", "input"])
        raw_target = item.get("target")
        if isinstance(raw_target, list):
            target = "\n".join([str(v).strip() for v in raw_target if str(v).strip()])
        else:
            target = str(raw_target or "").strip()
        review_id = str(item.get("review_id") or idx)

        records.append(
            EvalRecord(
                source="rct",
                subset="rct",
                sample_id=f"rct:{review_id}",
                prediction=prediction,
                reference=reference,
                target=target,
                metadata={
                    "review_id": review_id,
                    "num_articles": item.get("num_articles"),
                },
            )
        )

    return records


def _load_medinst_inputs_for_task(task_name: str, config_suffix: str) -> List[str]:
    try:
        import datasets  # type: ignore
    except Exception as exc:
        raise ImportError(
            "datasets package is required for --medinst-reference-mode input. "
            "Install with pip install datasets"
        ) from exc

    config_name = f"{task_name}{config_suffix}"
    dataset = datasets.load_dataset("LiinXemmon/MedINST32", config_name)["test"]

    inputs: List[str] = []
    for row in dataset:
        inputs.append(str(row.get("input", "")).strip())
    return inputs


def load_medinst_records(
    path: Path,
    reference_mode: ReferenceMode,
    config_suffix: str,
    task_filter: str,
    skip_empty: bool = True,
) -> List[EvalRecord]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("MedINST output must be a dict keyed by categories/tasks.")

    records: List[EvalRecord] = []
    for category, tasks in payload.items():
        if not isinstance(tasks, dict):
            continue

        for task_name, task_blob in tasks.items():
            if not isinstance(task_blob, dict):
                continue
            if task_filter.lower() != "all" and task_name.lower() != task_filter.lower():
                continue

            generated = task_blob.get("generated", [])
            if not isinstance(generated, list) or not generated:
                continue

            task_inputs: Optional[List[str]] = None
            if reference_mode == "input":
                task_inputs = _load_medinst_inputs_for_task(task_name=task_name, config_suffix=config_suffix)

            for idx, sample in enumerate(generated):
                if not isinstance(sample, dict):
                    continue

                prediction = str(sample.get("prediction") or "").strip()
                raw_target = sample.get("target")
                if isinstance(raw_target, list):
                    target = "\n".join([str(v).strip() for v in raw_target if str(v).strip()])
                else:
                    target = str(raw_target or "").strip()
                if reference_mode == "input":
                    reference = task_inputs[idx].strip() if task_inputs and idx < len(task_inputs) else ""
                else:
                    reference = target

                if skip_empty and (not prediction or not reference):
                    continue

                records.append(
                    EvalRecord(
                        source="medinst32",
                        subset=f"{category}/{task_name}",
                        sample_id=f"medinst32:{category}:{task_name}:{idx}",
                        prediction=prediction,
                        reference=reference,
                        target=target,
                        metadata={
                            "category": category,
                            "task": task_name,
                            "index": idx,
                            "reference_mode": reference_mode,
                            "config_suffix": config_suffix,
                        },
                    )
                )

    return records


def _add_local_metric_repos_to_path(repo_root: Path) -> None:
    prunehall_root = repo_root / "external" / "prunehall"
    summac_root = repo_root / "external" / "summac"

    if not prunehall_root.exists():
        raise FileNotFoundError(f"Could not find prunehall folder at {prunehall_root}.")
    if not summac_root.exists():
        raise FileNotFoundError(f"Could not find summac folder at {summac_root}.")

    if str(summac_root) not in sys.path:
        sys.path.insert(0, str(summac_root))
    if str(prunehall_root) not in sys.path:
        sys.path.insert(0, str(prunehall_root))


def _aggregate(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if len(values) == 1:
        scalar = float(values[0])
        return {"mean": scalar, "std": 0.0, "min": scalar, "max": scalar}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _compute_group_aggregate(scored_rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored_rows:
        group_name = str(row.get(key, "unknown"))
        grouped.setdefault(group_name, []).append(row)

    payload: Dict[str, Dict[str, Any]] = {}
    for group_name, rows in grouped.items():
        payload[group_name] = {
            "num_samples": len(rows),
            "harim_plus": _aggregate([float(r["harim_plus"]) for r in rows]),
            "summac_zs": _aggregate([float(r["summac_zs"]) for r in rows]),
            "summac_conv": _aggregate([float(r["summac_conv"]) for r in rows]),
            "rougeL": _aggregate([float(r["rougeL"]) for r in rows]),
            "bertscore_precision": _aggregate([float(r["bertscore_precision"]) for r in rows]),
            "bertscore_recall": _aggregate([float(r["bertscore_recall"]) for r in rows]),
            "bertscore_f1": _aggregate([float(r["bertscore_f1"]) for r in rows]),
        }
    return payload


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _split_into_sentences(text: str) -> List[str]:
    """Best-effort sentence splitter.

    SummaC uses NLTK internally, but some generations are newline/bullet-separated
    without strong punctuation; this helper treats newlines as hard boundaries and
    then applies NLTK sent_tokenize when available.
    """

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    raw_lines = [line.strip() for line in normalized.split("\n")]
    lines: List[str] = []
    for line in raw_lines:
        if not line:
            continue
        # Strip common bullet/list prefixes.
        line = line.lstrip("-*•\t ")
        if line:
            lines.append(line)

    # Optional NLTK refinement within each line.
    try:
        import nltk  # type: ignore

        sent_tokenize = nltk.tokenize.sent_tokenize  # type: ignore[attr-defined]
    except Exception:
        sent_tokenize = None

    sentences: List[str] = []
    for line in lines:
        if sent_tokenize is None:
            parts = [p.strip() for p in line.replace("!", ".").replace("?", ".").split(".")]
            sentences.extend([p for p in parts if p])
            continue

        # If the line is short and unpunctuated, sent_tokenize often returns it as-is.
        try:
            parts = [s.strip() for s in sent_tokenize(line) if str(s).strip()]
        except Exception:
            parts = [line]

        sentences.extend(parts)

    return [s for s in sentences if s]


def _sentence_level_pairs(
    records: List[EvalRecord],
    max_prediction_sentences: int,
    max_prediction_words_for_summac: int,
    max_reference_words_for_summac: int,
) -> Tuple[List[str], List[str], List[Tuple[int, int]], List[int]]:
    """Expand records into sentence-level (prediction_sentence, reference_document) pairs.

    Returns:
        predictions: flattened list of prediction sentences
        references: flattened list of corresponding references
        mapping: (record_idx, sentence_idx_within_record) for each flattened pair
        sentence_counts: number of sentences used per record (aligned with records)
    """

    expanded_predictions: List[str] = []
    expanded_references: List[str] = []
    mapping: List[Tuple[int, int]] = []
    sentence_counts: List[int] = []

    for record_idx, record in enumerate(records):
        sentences = _split_into_sentences(record.prediction)
        if max_prediction_sentences > 0:
            sentences = sentences[:max_prediction_sentences]

        safe_reference = _truncate_words(record.reference, max_reference_words_for_summac)

        used = 0
        for sent_idx, sentence in enumerate(sentences):
            safe_sentence = _truncate_words(sentence, max_prediction_words_for_summac)
            if not safe_sentence.strip() or not safe_reference.strip():
                continue
            expanded_predictions.append(safe_sentence)
            expanded_references.append(safe_reference)
            mapping.append((record_idx, sent_idx))
            used += 1

        sentence_counts.append(used)

    return expanded_predictions, expanded_references, mapping, sentence_counts


def save_flat_records(records: List[EvalRecord], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "source": record.source,
            "subset": record.subset,
            "sample_id": record.sample_id,
            "prediction": record.prediction,
            "reference": record.reference,
            "target": record.target,
            "metadata": record.metadata,
        }
        for record in records
    ]
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def evaluate_records(
    records: List[EvalRecord],
    repo_root: Path,
    device: str,
    max_prediction_words_for_summac: int,
    max_reference_words_for_summac: int,
    hallu_granularity: HalluGranularity,
    max_prediction_sentences: int,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("No evaluation records were provided.")

    _add_local_metric_repos_to_path(repo_root)

    from pruning_study.eval_funcs import ExperimentEvaluator  # type: ignore[import-not-found]

    predictions = [record.prediction for record in records]
    references = [record.reference for record in records]

    # Always compute doc-level safe strings for output/debug payload fields.
    safe_predictions = [_truncate_words(text, max_prediction_words_for_summac) for text in predictions]
    safe_references = [_truncate_words(text, max_reference_words_for_summac) for text in references]

    evaluator = ExperimentEvaluator(device=device)

    per_record_sentence_results: List[List[Dict[str, Any]]]

    if hallu_granularity == "document":
        results = evaluator.evaluate_hallucunations(
            prediction=safe_predictions,
            reference=safe_references,
        )
        per_record_harim = [float(v) for v in results.harim_plus]
        per_record_summac_zs = [float(v) for v in results.summac_zs]
        per_record_summac_conv = [float(v) for v in results.summac_conv]
        sentence_counts = [1] * len(records)
        per_record_sentence_metrics: List[Dict[str, Any]] = [
            {"num_sentences": 1} for _ in records
        ]
        per_record_sentence_results = [
            [
                {
                    "sentence_index": 0,
                    "sentence": safe_predictions[idx],
                    "harim_plus": float(per_record_harim[idx]),
                    "summac_zs": float(per_record_summac_zs[idx]),
                    "summac_conv": float(per_record_summac_conv[idx]),
                }
            ]
            for idx in range(len(records))
        ]
    elif hallu_granularity == "sentence":
        (
            expanded_predictions,
            expanded_references,
            mapping,
            sentence_counts,
        ) = _sentence_level_pairs(
            records=records,
            max_prediction_sentences=max_prediction_sentences,
            max_prediction_words_for_summac=max_prediction_words_for_summac,
            max_reference_words_for_summac=max_reference_words_for_summac,
        )

        if not expanded_predictions:
            raise ValueError(
                "No sentence-level prediction/reference pairs were produced. "
                "Check that predictions are non-empty and sentence splitting works for your data."
            )

        sentence_results = evaluator.evaluate_hallucunations(
            prediction=expanded_predictions,
            reference=expanded_references,
        )

        # Aggregate sentence scores back to per-record scores.
        harim_by_record: List[List[float]] = [[] for _ in records]
        summac_zs_by_record: List[List[float]] = [[] for _ in records]
        summac_conv_by_record: List[List[float]] = [[] for _ in records]

        per_record_sentence_results = [[] for _ in records]

        for flat_idx, (record_idx, _sent_idx) in enumerate(mapping):
            harim_value = float(sentence_results.harim_plus[flat_idx])
            summac_zs_value = float(sentence_results.summac_zs[flat_idx])
            summac_conv_value = float(sentence_results.summac_conv[flat_idx])

            harim_by_record[record_idx].append(harim_value)
            summac_zs_by_record[record_idx].append(summac_zs_value)
            summac_conv_by_record[record_idx].append(summac_conv_value)

            per_record_sentence_results[record_idx].append(
                {
                    "sentence_index": int(_sent_idx),
                    "sentence": expanded_predictions[flat_idx],
                    "harim_plus": harim_value,
                    "summac_zs": summac_zs_value,
                    "summac_conv": summac_conv_value,
                }
            )

        per_record_harim = [float(statistics.mean(v)) if v else 0.0 for v in harim_by_record]
        per_record_summac_zs = [float(statistics.mean(v)) if v else 0.0 for v in summac_zs_by_record]
        per_record_summac_conv = [float(statistics.mean(v)) if v else 0.0 for v in summac_conv_by_record]
        per_record_sentence_metrics = [
            {
                "num_sentences": sentence_counts[idx],
                "harim_plus": _aggregate(harim_by_record[idx]),
                "summac_zs": _aggregate(summac_zs_by_record[idx]),
                "summac_conv": _aggregate(summac_conv_by_record[idx]),
            }
            for idx in range(len(records))
        ]
    else:
        raise ValueError(f"Unsupported hallu_granularity: {hallu_granularity}")

    quality_references = [record.target if record.target.strip() else record.reference for record in records]
    summary_results = evaluator.evaluate_summary(
        prediction=predictions,
        reference=quality_references,
    )

    scored_rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        scored_rows.append(
            {
                "id": idx,
                "sample_id": record.sample_id,
                "source": record.source,
                "subset": record.subset,
                "prediction": record.prediction,
                "reference": record.reference,
                "prediction_for_metrics": safe_predictions[idx],
                "reference_for_metrics": safe_references[idx],
                "target": record.target,
                "harim_plus": float(per_record_harim[idx]),
                "summac_zs": float(per_record_summac_zs[idx]),
                "summac_conv": float(per_record_summac_conv[idx]),
                "rougeL": float(summary_results.rouge["rougeL"][idx]),
                "bertscore_precision": float(summary_results.bertscore["precision"][idx]),
                "bertscore_recall": float(summary_results.bertscore["recall"][idx]),
                "bertscore_f1": float(summary_results.bertscore["f1"][idx]),
                "metadata": record.metadata,
                "hallu_granularity": hallu_granularity,
                "prediction_sentence_metrics": per_record_sentence_metrics[idx],
                "prediction_sentence_results": per_record_sentence_results[idx],
            }
        )

    return {
        "num_samples": len(scored_rows),
        "aggregate": {
            "harim_plus": _aggregate([row["harim_plus"] for row in scored_rows]),
            "summac_zs": _aggregate([row["summac_zs"] for row in scored_rows]),
            "summac_conv": _aggregate([row["summac_conv"] for row in scored_rows]),
            "rougeL": _aggregate([row["rougeL"] for row in scored_rows]),
            "bertscore_precision": _aggregate([row["bertscore_precision"] for row in scored_rows]),
            "bertscore_recall": _aggregate([row["bertscore_recall"] for row in scored_rows]),
            "bertscore_f1": _aggregate([row["bertscore_f1"] for row in scored_rows]),
        },
        "by_source": _compute_group_aggregate(scored_rows, key="source"),
        "by_subset": _compute_group_aggregate(scored_rows, key="subset"),
        "results": scored_rows,
    }


def save_payload(payload: Dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hallucination metrics (HaRiM+, SummaC-ZS, SummaC-Conv) from "
            "eval_medinst32.py and/or generate_rct_summaries.py outputs."
        )
    )

    parser.add_argument(
        "--rct-json",
        type=str,
        default=None,
        help="Path to JSON output from generate_rct_summaries.py",
    )
    parser.add_argument(
        "--medinst-json",
        type=str,
        default=None,
        help="Path to JSON output from eval_medinst32.py",
    )

    parser.add_argument(
        "--medinst-reference-mode",
        type=str,
        default="input",
        choices=["input", "target"],
        help=(
            "Reference used for MedINST hallucination scoring. "
            "input: task input text from HF dataset (recommended). "
            "target: gold output label/text."
        ),
    )
    parser.add_argument(
        "--medinst-config-suffix",
        type=str,
        default="",
        help=(
            "Config suffix appended to MedINST task names when loading dataset input references. "
            "Use '-zs' if your eval_medinst32 run used --zero."
        ),
    )
    parser.add_argument(
        "--medinst-task-filter",
        type=str,
        default="Multi-XScience",
        help=(
            "MedINST task name filter when --medinst-json is provided. "
            "Default: Multi-XScience. Use 'all' to include all MedINST tasks."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device passed to prunehall ExperimentEvaluator.",
    )
    parser.add_argument(
        "--max-prediction-words-for-summac",
        type=int,
        default=384,
        help=(
            "Max words kept from each prediction before SummaC scoring. "
            "Prevents tokenizer truncation failures on very long generations. "
            "Set <=0 to disable clipping."
        ),
    )
    parser.add_argument(
        "--max-reference-words-for-summac",
        type=int,
        default=768,
        help=(
            "Max words kept from each reference before SummaC scoring. "
            "Set <=0 to disable clipping."
        ),
    )

    parser.add_argument(
        "--hallu-granularity",
        type=str,
        default="sentence",
        choices=["document", "sentence"],
        help=(
            "Granularity for HaRiM+/SummaC scoring. "
            "sentence: split predictions into sentences, score each, and average per sample (recommended). "
            "document: score each sample as a single sequence (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--max-prediction-sentences",
        type=int,
        default=0,
        help=(
            "Only used when --hallu-granularity=sentence. "
            "If >0, limit the number of prediction sentences scored per sample (0 = no limit)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation/logs_hallu/hallu_metrics.json",
        help="Path to final metrics JSON payload.",
    )
    parser.add_argument(
        "--output-records-json",
        type=str,
        default="evaluation/logs_hallu/hallu_input_records.json",
        help="Path to save flattened merged input records before scoring.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()

    all_records: List[EvalRecord] = []

    if args.rct_json:
        rct_records = load_rct_records(Path(args.rct_json))
        all_records.extend(rct_records)
        print(f"Loaded RCT records: {len(rct_records)}")

    if args.medinst_json:
        medinst_records = load_medinst_records(
            path=Path(args.medinst_json),
            reference_mode=args.medinst_reference_mode,
            config_suffix=args.medinst_config_suffix,
            task_filter=args.medinst_task_filter,
            skip_empty=True,
        )
        all_records.extend(medinst_records)
        print(f"Loaded MedINST32 records: {len(medinst_records)}")

    if not all_records:
        raise ValueError("Provide at least one of --rct-json or --medinst-json")

    records_path = Path(args.output_records_json)
    save_flat_records(all_records, records_path)

    payload = evaluate_records(
        records=all_records,
        repo_root=repo_root,
        device=args.device,
        max_prediction_words_for_summac=args.max_prediction_words_for_summac,
        max_reference_words_for_summac=args.max_reference_words_for_summac,
        hallu_granularity=args.hallu_granularity,
        max_prediction_sentences=args.max_prediction_sentences,
    )

    output_path = Path(args.output_json)
    save_payload(payload, output_path)

    print(f"Saved merged records: {records_path}")
    print(f"Saved hallu metrics: {output_path}")
    print(
        "Aggregate means -> "
        f"harim_plus={payload['aggregate']['harim_plus']['mean']:.6f}, "
        f"summac_zs={payload['aggregate']['summac_zs']['mean']:.6f}, "
        f"summac_conv={payload['aggregate']['summac_conv']['mean']:.6f}, "
        f"rougeL={payload['aggregate']['rougeL']['mean']:.6f}, "
        f"bertscore_f1={payload['aggregate']['bertscore_f1']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
