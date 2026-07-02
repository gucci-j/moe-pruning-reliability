from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def tqdm(iterable, **kwargs):  # type: ignore
	return iterable


def split_response(response: str) -> str:
	"""Split responses produced by GPT-OSS-style chat models that include thinking sections."""
	analysis_match = re.search(r"^(.*?)assistantfinal", response, flags=re.DOTALL)
	if analysis_match:
		final_content = re.sub(r"^.*?assistantfinal\s*", "", response, flags=re.DOTALL).strip()
		return final_content
	if response.lower().startswith("analysis"):
		parts = re.split(r"analysis\s*[:\-]*", response, flags=re.IGNORECASE, maxsplit=1)
		if len(parts) == 2:
			return parts[1].strip()
	return response.strip()


def _normalize_text(value: Any) -> str:
	return str(value if value is not None else "").strip().lower()


def _normalize_for_exact(value: Any) -> str:
	text = _normalize_text(value)
	text = text.translate(str.maketrans("", "", string.punctuation))
	return " ".join(text.split())


def parse_model_output(raw_text: str) -> str:
	text = (raw_text or "").strip()
	text = text.split("Stop Here")[0].strip()
	fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
	if fenced:
		return fenced.group(1).strip()
	return text


def _available_mmlu_tasks() -> List[str]:
	return ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']


TASKS: Tuple[str, ...] = tuple(_available_mmlu_tasks())


@dataclass
class EvalRow:
	task: str
	id: str
	prompt: str
	raw_output: str
	parsed_output: str
	target: Any
	is_correct: bool


@dataclass
class TaskMetrics:
	task: str
	total: int
	correct: int
	wrong: int

	@property
	def accuracy(self) -> float:
		if self.total == 0:
			return 0.0
		return (self.correct / self.total) * 100.0


class VLLMPredictor:
	def __init__(
		self,
		model_name_or_path: str,
		max_new_tokens: int,
		temperature: float,
		top_p: float,
		max_model_len: Optional[int],
		enforce_eager: bool,
	):
		try:
			torch_mod = importlib.import_module("torch")
			vllm_mod = importlib.import_module("vllm")
			llm_cls = getattr(vllm_mod, "LLM")
			sampling_params_cls = getattr(vllm_mod, "SamplingParams")
		except Exception as exc:
			raise ImportError("vllm/torch package not available; install both for vllm backend.") from exc

		num_gpus = torch_mod.cuda.device_count() if torch_mod.cuda.is_available() else 1
		self.llm = llm_cls(
			model=model_name_or_path,
			trust_remote_code=True,
			tensor_parallel_size=max(1, num_gpus),
			enforce_eager=enforce_eager,
			gpu_memory_utilization=0.9,
			max_model_len=max_model_len,
		)
		self.tokenizer = self.llm.get_tokenizer()
		self.sampling_params = sampling_params_cls(
			max_tokens=max_new_tokens,
			temperature=temperature,
			top_p=top_p,
		)
		self.model_name_or_path = model_name_or_path

	def _format_prompt(self, prompt: str) -> str:
		messages = [{"role": "user", "content": prompt}]
		if hasattr(self.tokenizer, "apply_chat_template"):
			try:
				if "gpt-oss-20b" in self.model_name_or_path:
					enable_thinking = True
					chat_template_args = {"reasoning_effort": "low"}
				else:
					enable_thinking = False
					chat_template_args = None
				return str(
					self.tokenizer.apply_chat_template(
						messages,
						tokenize=False,
						add_generation_prompt=True,
						enable_thinking=enable_thinking,
						**(chat_template_args or {}),
					)
				)
			except Exception:
				return prompt
		return prompt

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
		outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
		return [str(o.outputs[0].text) for o in outputs]


class HFPredictor:
	def __init__(self, model_name_or_path: str, max_new_tokens: int, temperature: float, top_p: float, device: str, dtype: str):
		try:
			torch_mod = __import__("torch")
			transformers_mod = __import__("transformers")
			auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")
			auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")
		except Exception as exc:
			raise ImportError("transformers/torch not available; install to use hf backend.") from exc

		dtype_map = {
			"float16": torch_mod.float16,
			"bfloat16": torch_mod.bfloat16,
			"float32": torch_mod.float32,
		}
		torch_dtype = dtype_map.get(dtype, torch_mod.bfloat16)

		self.tokenizer = auto_tokenizer_cls.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		device_map = "auto" if device == "auto" else None
		self.model = auto_model_cls.from_pretrained(
			model_name_or_path,
			torch_dtype=torch_dtype,
			device_map=device_map,
			trust_remote_code=True,
		)
		if device != "auto" and hasattr(self.model, "to"):
			self.model.to(device)

		self.max_new_tokens = max_new_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.do_sample = temperature > 0.0
		self.torch = torch_mod

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		outputs: List[str] = []
		batch_size = 1
		for i in tqdm(range(0, len(prompts), batch_size), desc="HF generation", leave=False):
			batch_prompts = list(prompts[i : i + batch_size])
			model_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
			model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
			input_len = model_inputs["input_ids"].shape[1]

			with self.torch.no_grad():
				generated = self.model.generate(
					**model_inputs,
					max_new_tokens=self.max_new_tokens,
					do_sample=self.do_sample,
					temperature=self.temperature if self.do_sample else 1.0,
					top_p=self.top_p if self.do_sample else 1.0,
					eos_token_id=self.tokenizer.eos_token_id,
					pad_token_id=self.tokenizer.pad_token_id,
				)

			continuations = generated[:, input_len:]
			decoded = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
			outputs.extend(decoded)

		return outputs


def _load_rows_from_datasets(task: str, split: Optional[str] = None, num_samples: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
	try:
		datasets_mod = importlib.import_module("datasets")
	except Exception:
		raise ImportError("datasets package not available; install 'datasets' to load HuggingFace datasets")

	use_split = split or "test"
	dataset_name = "cais/mmlu"

	# Allow inputs like 'mmlu_anatomy' or just 'anatomy'
	subset = task
	if task.startswith("mmlu_"):
		subset = task.replace("mmlu_", "")

	rows: List[Dict[str, Any]] = []

	# If user requested 'all', load all available cached configs
	if subset.lower() == "all":
		try:
			configs = datasets_mod.get_dataset_config_names(dataset_name)
		except Exception as exc:
			raise ValueError("Unable to enumerate cached MMLU configs; ensure 'datasets' can access the cache") from exc

		for cfg in configs:
			if cfg == "all":
				continue
			ds = datasets_mod.load_dataset(dataset_name, name=cfg, split=use_split)
			rows.extend([dict(r) for r in ds])

		# optional sampling across the concatenated rows
		if num_samples and 0 < num_samples < len(rows):
			rng = random.Random(seed)
			indices = sorted(rng.sample(range(len(rows)), num_samples))
			rows = [rows[i] for i in indices]

		return rows

	# Try loading requested subset; if missing, raise with available cached configs
	try:
		ds = datasets_mod.load_dataset(dataset_name, name=subset, split=use_split)
		rows = [dict(r) for r in ds]
	except ValueError as exc:
		try:
			configs = datasets_mod.get_dataset_config_names(dataset_name)
		except Exception:
			configs = []
		raise ValueError(
			f"Couldn't find cache for {dataset_name} for config '{subset}'\nAvailable configs in the cache: {configs}\nUse 'all' or one of the listed config names."
		) from exc

	if num_samples and 0 < num_samples < len(rows):
		rng = random.Random(seed)
		indices = sorted(rng.sample(range(len(rows)), num_samples))
		rows = [rows[i] for i in indices]

	return rows


def _get_prompt_for_row(task: str, row: Dict[str, Any]) -> str:
	q = str(row.get("question", row.get("prompt", row.get("text", "")))).strip()

	choices = row.get("choices") or row.get("options") or row.get("answer_choices") or row.get("answers") or []
	choice_list: List[str] = []
	if isinstance(choices, dict):
		for key in ("A", "B", "C", "D"):
			if key in choices:
				choice_list.append(str(choices[key]))
	elif isinstance(choices, (list, tuple)):
		choice_list = [str(x) for x in choices]
	else:
		choice_list = [
			str(row.get("choiceA", row.get("A", ""))),
			str(row.get("choiceB", row.get("B", ""))),
			str(row.get("choiceC", row.get("C", ""))),
			str(row.get("choiceD", row.get("D", ""))),
		]

	while len(choice_list) < 4:
		choice_list.append("")

	prompt = q + "\n"
	prompt += f"A. {choice_list[0]}\n"
	prompt += f"B. {choice_list[1]}\n"
	prompt += f"C. {choice_list[2]}\n"
	prompt += f"D. {choice_list[3]}\n"
	prompt += "Answer: Respond with ONLY a single letter: A, B, C, or D"
	return prompt


def _get_target_for_row(task: str, row: Dict[str, Any]) -> Any:
	# Handle explicit 'answer' key first
	if "answer" in row:
		val = row["answer"]
		if isinstance(val, (int, float)):
			try:
				return chr(ord("A") + int(val))
			except Exception:
				return val
		if isinstance(val, str) and val.isdigit():
			try:
				return chr(ord("A") + int(val))
			except Exception:
				return val
		return val

	for key in ("target", "labels", "gold", "answers", "cop", "final_decision", "label"):
		if key in row:
			val = row[key]
			# numeric index -> letter mapping
			if isinstance(val, (int, float)):
				try:
					return chr(ord("A") + int(val))
				except Exception:
					return val
			if isinstance(val, str) and val.isdigit():
				try:
					return chr(ord("A") + int(val))
				except Exception:
					return val
			return val
	return None


def _is_correct(task: str, row: Dict[str, Any], output: str) -> bool:
	target = _get_target_for_row(task, row)
	pred = _normalize_text(output)
	if target is None:
		return False

	if isinstance(target, list):
		for t in target:
			if _normalize_for_exact(t) == _normalize_for_exact(pred):
				return True
		for t in target:
			if _normalize_for_exact(t) in _normalize_for_exact(pred):
				return True
		return False

	tnorm = _normalize_for_exact(target)
	pnorm = _normalize_for_exact(pred)
	if pnorm == tnorm:
		return True
	if pnorm.startswith(tnorm):
		return True
	return False


def evaluate_task(task: str, rows: List[Dict[str, Any]], raw_outputs: List[str]) -> Tuple[TaskMetrics, List[EvalRow]]:
	eval_rows: List[EvalRow] = []
	correct = 0
	for row, raw in zip(rows, raw_outputs):
		parsed = parse_model_output(split_response(raw))
		is_corr = _is_correct(task=task, row=row, output=parsed)
		if is_corr:
			correct += 1
		eval_rows.append(
			EvalRow(
				task=task,
				id=str(row.get("id", row.get("idx", ""))),
				prompt=_get_prompt_for_row(task, row),
				raw_output=raw,
				parsed_output=parsed,
				target=_get_target_for_row(task, row),
				is_correct=is_corr,
			)
		)

	total = len(eval_rows)
	metrics = TaskMetrics(task=task, total=total, correct=correct, wrong=(total - correct))
	return metrics, eval_rows


def _write_task_predictions_csv(task: str, eval_rows: List[EvalRow], output_dir: Path) -> Path:
	import csv

	csv_path = output_dir / f"{task}.csv"
	with csv_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		for item in eval_rows:
			writer.writerow([item.id, item.parsed_output])
	return csv_path


def _write_task_examples_json(task: str, eval_rows: List[EvalRow], output_dir: Path) -> Path:
	out_path = output_dir / f"{task}_examples.json"
	payload = [
		{
			"id": row.id,
			"task": row.task,
			"prompt": row.prompt,
			"raw_output": row.raw_output,
			"parsed_output": row.parsed_output,
			"target": row.target,
			"is_correct": row.is_correct,
		}
		for row in eval_rows
	]
	out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	return out_path


def _build_predictor(args: argparse.Namespace):
	if args.backend == "vllm":
		model_name = args.hf_model or args.model
		return VLLMPredictor(
			model_name_or_path=model_name,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			max_model_len=args.max_model_len if args.max_model_len != -1 else None,
			enforce_eager=args.enforce_eager,
		)

	if args.backend == "hf":
		model_name = args.hf_model or args.model
		return HFPredictor(
			model_name_or_path=model_name,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			device=args.device,
			dtype=args.dtype,
		)

	raise ValueError(f"Unsupported backend: {args.backend}")


def _parse_task_list(task_arg: str) -> List[str]:
	available_tasks = list(TASKS) or _available_mmlu_tasks()
	if task_arg.strip().lower() == "all":
		if not available_tasks:
			raise ValueError("Unable to resolve MMLU task list. Install 'datasets' and retry.")
		return available_tasks

	tasks = [t.strip() for t in task_arg.split(",") if t.strip()]
	if not available_tasks:
		return tasks

	invalid = [t for t in tasks if t not in available_tasks]
	if invalid:
		raise ValueError(f"Unsupported task(s): {invalid}. Valid tasks: {available_tasks}")
	return tasks


def main() -> None:
	parser = argparse.ArgumentParser(description="MMLU generation-style evaluation script.")
	parser.add_argument("--name", type=str, required=True, help="Run name used in output filenames.")
	parser.add_argument("--output-dir", type=str, required=True, help="Directory where evaluation artifacts are saved.")
	parser.add_argument("--tasks", type=str, default="all", help="Comma-separated MMLU subjects or 'all'.")
	parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on.")
	parser.add_argument("--num-samples-per-task", type=int, default=0, help="<=0 uses full task dataset.")
	parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
	parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")

	parser.add_argument("--backend", type=str, choices=["vllm", "hf"], default="hf", help="Inference backend.")
	parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name for API or default for HF backend.")
	parser.add_argument("--max-new-tokens", type=int, default=64)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--max-model-len", type=int, default=-1, help="Only for vLLM backend; ignored otherwise.")
	parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs for vLLM.")
	parser.add_argument("--hf_model", type=str, default=None, help="HF model id/path (overrides --model for HF backend).")
	parser.add_argument("--device", type=str, default="auto", help="HF device: auto, cuda:0, cpu, etc.")
	parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")

	args = parser.parse_args()

	output_dir = Path(args.output_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	tasks = _parse_task_list(args.tasks)
	predictor = _build_predictor(args)

	overall_total = 0
	overall_correct = 0
	per_task_results: List[Dict[str, Any]] = []

	for task in tasks:
		rows = _load_rows_from_datasets(task=task, split=args.split, num_samples=args.num_samples_per_task, seed=args.seed)
		prompts = [_get_prompt_for_row(task, r) for r in rows]
		raw_outputs: List[str] = []

		if args.batch_size == -1:
			raw_outputs = predictor.generate_batch(prompts)
		else:
			for start in tqdm(range(0, len(prompts), max(1, args.batch_size)), desc=f"{task} inference"):
				batch_prompts = prompts[start : start + max(1, args.batch_size)]
				outputs = predictor.generate_batch(batch_prompts)
				raw_outputs.extend(outputs)

		task_metrics, eval_rows = evaluate_task(task=task, rows=rows, raw_outputs=raw_outputs)
		overall_total += task_metrics.total
		overall_correct += task_metrics.correct
		per_task_results.append({
			"task": task,
			"total": task_metrics.total,
			"correct": task_metrics.correct,
			"wrong": task_metrics.wrong,
			"accuracy": round(task_metrics.accuracy, 4),
		})

		print(f"[{task}] total={task_metrics.total} correct={task_metrics.correct} wrong={task_metrics.wrong} acc={task_metrics.accuracy:.3f}%")

	# Macro-average across tasks (average of per-task accuracies)
	macro_accuracy = round(sum(r["accuracy"] for r in per_task_results) / len(per_task_results), 4) if per_task_results else 0.0
	micro_accuracy = round((overall_correct / overall_total * 100.0) if overall_total else 0.0, 4)
	summary = {
		"name": args.name,
		"tasks": tasks,
		"per_task": per_task_results,
		"accuracy": macro_accuracy,
		"micro_total": overall_total,
		"micro_correct": overall_correct,
		"micro_accuracy": micro_accuracy,
	}
	summary_path = output_dir / f"{args.name}.json"
	summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

	print("MMLU generation evaluation complete.")
	print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
	main()
