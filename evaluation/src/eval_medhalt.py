from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import random
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

def tqdm(iterable, **kwargs):  # type: ignore
	return iterable

# Optional dependencies
try:
	from openai import OpenAI  # type: ignore
except Exception:
	OpenAI = None
 

def split_response(response: str) -> str:
    """Split the response into analysis and final content. This is for gpt-oss models."""
    # Find the analysis part (everything before "assistantfinal")
    analysis_match = re.search(r'^(.*?)assistantfinal', response, flags=re.DOTALL)

    if analysis_match:
        analysis_content = analysis_match.group(1).strip()
        # Remove "analysis" prefix if present
        analysis_content = re.sub(r'^analysis\s*', '', analysis_content)
        
        # Find the final content (everything after "assistantfinal")
        final_content = re.sub(r'^.*?assistantfinal\s*', '', response, flags=re.DOTALL).strip()
    else:
        # If no "assistantfinal" found, check for just "analysis" prefix
        if response.startswith('analysis'):
            analysis_content = response.strip()
            final_content = ""
        else:
            # No clear separation, treat entire response as final content
            analysis_content = ""
            final_content = response.strip()
    
    return final_content


TASKS: Tuple[str, ...] = ("abs2pub", "pmid2title", "url2title", "title2pub", "fake", "FCT", "Nota")
TASK_TO_DATASET: Dict[str, str] = {
	"abs2pub": "IR_abstract2pubmedlink",
	"pmid2title": "IR_pmid2title",
	"url2title": "IR_pubmedlink2title",
	"title2pub": "IR_title2pubmedlink",
	"fake": "reasoning_fake",
	"FCT": "reasoning_FCT",
	"Nota": "reasoning_nota",
}

REFUSAL_TERMS = [
	"i do not know",
	"conceding defeat",
	"admit",
	"none of the above",
	"acknowled",
	"irrelevant",
	"fiction",
	"all of the above",
	"nonsensical",
	"no correct",
	"absurd",
	"defy",
	"i don\"t know.",
	"defies",
]


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _default_medhalt_root() -> Path:
	return _repo_root() / "external" / "medhalt"


def _add_medhalt_to_path(medhalt_root: Path) -> None:
	medhalt_pkg_root = medhalt_root / "medhalt"
	if not medhalt_pkg_root.exists():
		raise FileNotFoundError(f"Could not find MedHALT package at {medhalt_pkg_root}")
	if str(medhalt_root) not in sys.path:
		sys.path.insert(0, str(medhalt_root))


def _normalize_text(value: Any) -> str:
	return str(value if value is not None else "").strip().lower()


def _normalize_for_exact(value: Any) -> str:
	text = _normalize_text(value)
	text = text.translate(str.maketrans("", "", string.punctuation))
	return " ".join(text.split())


def _sample_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
	if n <= 0 or n >= len(rows):
		return rows
	rng = random.Random(seed)
	indices = sorted(rng.sample(range(len(rows)), n))
	return [rows[idx] for idx in indices]


def _parse_key_values(out_str: str) -> List[Tuple[str, str]]:
	regex = re.compile(r"""['\"](.*?)['\"]\s*:\s*['\"]*(.*?)['\"]*\s*[,}]""")
	return regex.findall(out_str)


def _fallback_parse_dict(out_str: str) -> Dict[str, str]:
	kvs = _parse_key_values(out_str)
	return {key.replace("\\", ""): value for key, value in kvs}


def parse_model_output(raw_text: str) -> Dict[str, str]:
	clean = (raw_text or "").strip()
	clean = clean.split("Stop Here")[0].strip()

	if not clean:
		return {}

	candidates: List[str] = [clean]

	fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, flags=re.IGNORECASE)
	if fenced_match:
		fenced_content = fenced_match.group(1).strip()
		if fenced_content:
			candidates.append(fenced_content)

	brace_match = re.search(r"\{[\s\S]*\}", clean)
	if brace_match:
		brace_content = brace_match.group(0).strip()
		if brace_content:
			candidates.append(brace_content)

	for candidate in candidates:
		if not candidate:
			continue
		try:
			parsed = ast.literal_eval(candidate)
			if isinstance(parsed, dict):
				return {str(key): str(value) for key, value in parsed.items()}
		except Exception:
			pass

		try:
			parsed_json = json.loads(candidate)
			if isinstance(parsed_json, dict):
				return {str(key): str(value) for key, value in parsed_json.items()}
		except Exception:
			pass

	return _fallback_parse_dict(candidates[-1] if candidates else clean)


@dataclass
class EvalRow:
	task: str
	id: str
	prompt: str
	raw_output: str
	parsed_output: Dict[str, str]
	target: Dict[str, str]
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


class APIPredictor:
	def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int, temperature: float, top_p: float):
		if OpenAI is None:
			raise ImportError("openai package not available; install openai to use api backend.")
		self.client = OpenAI(api_key=api_key, base_url=base_url)
		self.model = model
		self.max_tokens = max_tokens
		self.temperature = temperature
		self.top_p = top_p

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		outputs: List[str] = []
		for prompt in tqdm(prompts, desc="API generation", leave=False):
			completion = self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": prompt}],
				max_tokens=self.max_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
			)
			outputs.append(str(completion.choices[0].message.content or ""))
		return outputs


class VLLMPredictor:
	def __init__(
		self,
		model_name_or_path: str,
		max_new_tokens: int,
		temperature: float,
		top_p: float,
		max_model_len: int,
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
				return str(self.tokenizer.apply_chat_template(
        			messages, tokenize=False, add_generation_prompt=True,
					enable_thinking=enable_thinking, **(chat_template_args or {})
           		))
			except Exception:
				return prompt
		return prompt

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
		outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=True)
		return [str(o.outputs[0].text) for o in outputs]


class HFPredictor:
	def __init__(
		self,
		model_name_or_path: str,
		max_new_tokens: int,
		temperature: float,
		top_p: float,
		device: str,
		dtype: str,
	):
		try:
			torch_mod = importlib.import_module("torch")
			transformers_mod = importlib.import_module("transformers")
			auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")
			auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")
		except Exception as exc:
			raise ImportError("transformers/torch not available; install to use hf backend.") from exc

		self.torch = torch_mod

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


def _load_medhalt_rows(medhalt_root: Path, task: str, shots: int, prompt_version: str) -> List[Dict[str, Any]]:
	_add_medhalt_to_path(medhalt_root)
	prompt_utils = importlib.import_module("medhalt.prompts.utils")
	get_samples = getattr(prompt_utils, "get_samples")
	rows = get_samples(dataset_name=task, shots=shots, prompt_version=prompt_version)
	if not isinstance(rows, list):
		raise ValueError(f"Unexpected medhalt samples type for task={task}: {type(rows)}")
	return [dict(row) for row in rows]


def _get_target(task: str, row: Dict[str, Any]) -> Dict[str, str]:
	if task in {"pmid2title", "url2title"}:
		return {"paper_title": str(row.get("Title", ""))}
	if task in {"title2pub", "abs2pub"}:
		return {"url": str(row.get("url", ""))}
	if task == "Nota":
		return {
			"cop": str(row.get("correct_answer", "")),
			"cop_index": str(row.get("correct_index", "")),
		}
	if task == "FCT":
		return {
			"correct_answer": str(row.get("correct_answer", "")),
			"cop_index": str(row.get("correct_index", "")),
		}
	if task == "fake":
		return {"cop": "I do not know"}
	raise ValueError(f"Unsupported task: {task}")


def _is_correct(task: str, row: Dict[str, Any], output: Dict[str, str]) -> bool:
	if task in {"pmid2title", "url2title"}:
		return _normalize_for_exact(output.get("paper_title", "")) == _normalize_for_exact(row.get("Title", ""))

	if task in {"title2pub", "abs2pub"}:
		return _normalize_for_exact(output.get("url", "")) == _normalize_for_exact(row.get("url", ""))

	if task == "Nota":
		return _normalize_for_exact(output.get("cop", "")) == _normalize_for_exact(row.get("correct_answer", ""))

	if task == "FCT":
		possible_keys = [
			"correct_answer",
			"answer",
			"correct answer",
			"corrent_answer",
			"Correct Answer",
			"Answer",
			"Correct_answer",
			"Correct answer",
			"cop",
		]
		predicted_answer = ""
		for key in possible_keys:
			if key in output and _normalize_text(output[key]):
				predicted_answer = output[key]
				break
		return _normalize_for_exact(predicted_answer) == _normalize_for_exact(row.get("correct_answer", ""))

	if task == "fake":
		prediction = _normalize_text(output.get("cop", ""))
		return any(term in prediction for term in REFUSAL_TERMS)

	raise ValueError(f"Unsupported task: {task}")


def evaluate_task(task: str, rows: List[Dict[str, Any]], raw_outputs: List[str]) -> Tuple[TaskMetrics, List[EvalRow]]:
	parsed_outputs = [parse_model_output(text) for text in raw_outputs]
	eval_rows: List[EvalRow] = []

	correct = 0
	for row, raw_text, parsed in zip(rows, raw_outputs, parsed_outputs):
		is_correct = _is_correct(task=task, row=row, output=parsed)
		if is_correct:
			correct += 1
		eval_rows.append(
			EvalRow(
				task=task,
				id=str(row.get("id", "")),
				prompt=str(row.get("prompt", "")),
				raw_output=raw_text,
				parsed_output=parsed,
				target=_get_target(task, row),
				is_correct=is_correct,
			)
		)

	total = len(eval_rows)
	metrics = TaskMetrics(task=task, total=total, correct=correct, wrong=(total - correct))
	return metrics, eval_rows


def _write_task_predictions_csv(task: str, eval_rows: List[EvalRow], output_dir: Path) -> Path:
	csv_path = output_dir / f"{task}.csv"
	with csv_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		for item in eval_rows:
			writer.writerow([item.id, str(item.parsed_output)])
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
	if args.backend in {"openai", "api"}:
		return APIPredictor(
			api_key=args.key,
			base_url=args.base_url,
			model=args.model,
			max_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
		)

	if args.backend == "vllm":
		model_name = args.model
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
	if task_arg.strip().lower() == "all":
		return list(TASKS)
	tasks = [t.strip() for t in task_arg.split(",") if t.strip()]
	invalid = [t for t in tasks if t not in TASKS]
	if invalid:
		raise ValueError(f"Unsupported task(s): {invalid}. Valid tasks: {list(TASKS)}")
	return tasks


def main() -> None:
	parser = argparse.ArgumentParser(description="Official MedHALT research evaluation script.")
	parser.add_argument("--name", type=str, required=True, help="Run name used in output filenames.")
	parser.add_argument("--output-dir", type=str, required=True, help="Directory where evaluation artifacts are saved.")
	parser.add_argument(
		"--medhalt-root",
		type=str,
		default=str(_default_medhalt_root()),
		help="Path to external/medhalt repository root.",
	)
	parser.add_argument(
		"--tasks",
		type=str,
		default="all",
		help="Comma-separated task keys or 'all'. Keys: abs2pub,pmid2title,url2title,title2pub,fake,FCT,Nota",
	)
	parser.add_argument("--num-samples-per-task", type=int, default=0, help="<=0 uses full task dataset.")
	parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
	parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
	parser.add_argument("--prompt-shots", type=int, default=2, help="Number of MedHALT few-shot examples.")
	parser.add_argument("--prompt-version", type=str, default="v0", help="MedHALT prompt version id.")

	parser.add_argument(
		"--backend",
		type=str,
		choices=["vllm", "hf", "openai", "api"],
		default="hf",
		help="Inference backend.",
	)
	parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name for API or default for HF/vLLM.")
	parser.add_argument("--max-new-tokens", type=int, default=128)
	parser.add_argument("--max-model-len", type=int, default=-1, help="Only for vLLM backend; ignored otherwise.")
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top-p", type=float, default=0.95)

	parser.add_argument("--key", type=str, default="", help="API key for OpenAI-compatible backend.")
	parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="OpenAI-compatible API URL.")

	parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs for vLLM.")
	parser.add_argument("--hf_model", type=str, default=None, help="HF model id/path (overrides --model for HF backend).")
	parser.add_argument("--device", type=str, default="auto", help="HF device: auto, cuda:0, cpu, etc.")
	parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")

	args = parser.parse_args()

	medhalt_root = Path(args.medhalt_root).resolve()
	output_dir = Path(args.output_dir).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	tasks = _parse_task_list(args.tasks)
	predictor = _build_predictor(args)

	run_payload: Dict[str, Any] = {
		"name": args.name,
		"backend": args.backend,
		"model": args.hf_model if args.backend == "hf" and args.hf_model else args.model,
		"medhalt_root": str(medhalt_root),
		"tasks": tasks,
		"num_samples_per_task": args.num_samples_per_task,
		"seed": args.seed,
		"batch_size": args.batch_size,
		"prompt_shots": args.prompt_shots,
		"prompt_version": args.prompt_version,
		"max_new_tokens": args.max_new_tokens,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"task_metrics": [],
	}

	overall_total = 0
	overall_correct = 0

	for task_idx, task in enumerate(tasks):
		rows = _load_medhalt_rows(
			medhalt_root=medhalt_root,
			task=task,
			shots=args.prompt_shots,
			prompt_version=args.prompt_version,
		)
		rows = _sample_rows(
			rows=rows,
			n=args.num_samples_per_task,
			seed=args.seed + task_idx,
		)

		prompts = [str(row.get("prompt", "")) for row in rows]
		raw_outputs: List[str] = []
		if args.batch_size == -1:
			raw_outputs = predictor.generate_batch(prompts)
			if "gpt-oss-20b" in args.model:
				outputs = [output.split("<|message|>")[-1].strip() for output in raw_outputs]
				outputs = [split_response(output) for output in outputs]
				raw_outputs = outputs
		else:
			for start in tqdm(range(0, len(prompts), max(1, args.batch_size)), desc=f"{task} inference"):
				batch_prompts = prompts[start : start + max(1, args.batch_size)]
				outputs = predictor.generate_batch(batch_prompts)
				if "gpt-oss-20b" in args.model:
					outputs = [output.split("<|message|>")[-1].strip() for output in outputs]
					outputs = [split_response(output) for output in outputs]
				raw_outputs.extend(outputs)

		task_metrics, eval_rows = evaluate_task(task=task, rows=rows, raw_outputs=raw_outputs)

		_write_task_predictions_csv(task=task, eval_rows=eval_rows, output_dir=output_dir)
		_write_task_examples_json(task=task, eval_rows=eval_rows, output_dir=output_dir)

		run_payload["task_metrics"].append(
			{
				"task": task,
				"dataset_name": TASK_TO_DATASET[task],
				"total": task_metrics.total,
				"correct": task_metrics.correct,
				"wrong": task_metrics.wrong,
				"accuracy": round(task_metrics.accuracy, 4),
			}
		)

		overall_total += task_metrics.total
		overall_correct += task_metrics.correct

		print(
			f"[{task}] total={task_metrics.total} correct={task_metrics.correct} "
			f"wrong={task_metrics.wrong} acc={task_metrics.accuracy:.3f}%"
		)

	overall_accuracy = (overall_correct / overall_total * 100.0) if overall_total else 0.0
	run_payload["overall"] = {
		"total": overall_total,
		"correct": overall_correct,
		"wrong": overall_total - overall_correct,
		"accuracy": round(overall_accuracy, 4),
	}

	summary_path = output_dir / f"{args.name}.json"
	summary_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	print("MedHALT evaluation complete.")
	print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
	main()
