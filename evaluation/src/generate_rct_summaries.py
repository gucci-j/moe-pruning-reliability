from __future__ import annotations

import re
import argparse
import csv
import importlib
import json
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def tqdm(iterable, **kwargs):  # type: ignore
	return iterable


try:
	from openai import OpenAI  # type: ignore
except Exception:
	OpenAI = None


@dataclass
class RCTExample:
	review_id: str
	source: str
	reference: str
	num_articles: int


def split_response(response: str) -> str:
	"""Split the response into analysis and final content. This is for gpt-oss models."""
	analysis_match = re.search(r"^(.*?)assistantfinal", response, flags=re.DOTALL)

	if analysis_match:
		final_content = re.sub(r"^.*?assistantfinal\s*", "", response, flags=re.DOTALL).strip()
		if not final_content:
			final_content = response.strip()
	else:
		final_content = response.strip()

	return final_content


def _is_empty(text: object) -> bool:
	return not str(text or "").strip()


def _postprocess_gpt_oss_prediction(prediction: str) -> str:
	"""Best-effort extraction of the final text for GPT-OSS style outputs."""
	text = str(prediction or "")
	if "<|message|>" in text:
		parts = [part.strip() for part in text.split("<|message|>") if part.strip()]
		if parts:
			text = parts[-1]
	final = split_response(text)
	if final.strip():
		return final.strip()
	return text.strip()


class APIPredictor:
	def __init__(self, api_key: str, base_url: str, model: str, max_new_tokens: int, temperature: float, top_p: float):
		if OpenAI is None:
			raise ImportError("openai package not available; install openai to use api backend.")
		self.client = OpenAI(api_key=api_key, base_url=base_url)
		self.model = model
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature
		self.top_p = top_p

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		outputs: List[str] = []
		for prompt in tqdm(prompts, desc="API generation", leave=False):
			completion = self.client.chat.completions.create(
				model=self.model,
				messages=[{"role": "user", "content": prompt}],
				max_tokens=self.max_new_tokens,
				temperature=self.temperature,
				top_p=self.top_p,
			)
			outputs.append(str(completion.choices[0].message.content or ""))
		return outputs


class VLLMPredictor:
	def __init__(
		self,
		model_name_or_path: str,
		min_new_tokens: int,
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
			max_model_len=max_model_len
		)
		self.tokenizer = self.llm.get_tokenizer()
		# SamplingParams kwargs can differ across vLLM versions; only pass supported args.
		params_kwargs: Dict[str, Any] = {
			"max_tokens": max_new_tokens,
			"temperature": temperature,
			"top_p": top_p,
		}
		try:
			sig = inspect.signature(sampling_params_cls)
			if "min_tokens" in sig.parameters and min_new_tokens > 0:
				params_kwargs["min_tokens"] = min_new_tokens
		except Exception:
			pass
		self.sampling_params = sampling_params_cls(**params_kwargs)
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
               			messages, tokenize=False, add_generation_prompt=True,
						enable_thinking=enable_thinking, **(chat_template_args or {})
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
	def __init__(
		self,
		model_name_or_path: str,
		min_new_tokens: int,
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
		self.max_new_tokens = max_new_tokens
		self.min_new_tokens = max(0, int(min_new_tokens))
		self.temperature = temperature
		self.top_p = top_p
		self.do_sample = temperature > 0.0

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
		self.model_name_or_path = model_name_or_path

	def _format_prompt(self, prompt: str) -> str:
		if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
			messages = [{"role": "user", "content": prompt}]
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
                    )
           		)
			except Exception:
				return prompt
		return prompt

	def generate_batch(self, prompts: Sequence[str]) -> List[str]:
		outputs: List[str] = []
		for prompt in tqdm(prompts, desc="HF generation", leave=False):
			formatted_prompt = self._format_prompt(prompt)
			model_inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
			model_inputs = {key: value.to(self.model.device) for key, value in model_inputs.items()}
			input_len = model_inputs["input_ids"].shape[1]

			with self.torch.no_grad():
				try:
					generated = self.model.generate(
						**model_inputs,
						max_new_tokens=self.max_new_tokens,
						min_new_tokens=self.min_new_tokens,
						temperature=self.temperature,
						top_p=self.top_p,
						do_sample=self.do_sample,
						pad_token_id=self.tokenizer.pad_token_id,
						eos_token_id=self.tokenizer.eos_token_id,
					)
				except TypeError:
					# Older transformers versions may not support min_new_tokens.
					generated = self.model.generate(
						**model_inputs,
						max_new_tokens=self.max_new_tokens,
						temperature=self.temperature,
						top_p=self.top_p,
						do_sample=self.do_sample,
						pad_token_id=self.tokenizer.pad_token_id,
						eos_token_id=self.tokenizer.eos_token_id,
					)

			new_tokens = generated[:, input_len:]
			text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
			outputs.append(str(text).strip())

		return outputs


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _default_rct_dir() -> Path:
	return _repo_root() / "external" / "rct"


def _clean(value: Optional[str]) -> str:
	return str(value or "").strip()


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8", newline="") as file:
		reader = csv.DictReader(file)
		rows: List[Dict[str, str]] = []
		for row in reader:
			rows.append({key: _clean(value) for key, value in row.items()})
		return rows


def _make_source_block(review_id: str, input_rows: List[Dict[str, str]], max_source_chars: int) -> str:
	blocks: List[str] = []
	for idx, row in enumerate(input_rows, start=1):
		pmid = _clean(row.get("PMID"))
		title = _clean(row.get("Title"))
		abstract = _clean(row.get("Abstract"))
		article_text = (
			f"Article {idx}\n"
			f"PMID: {pmid}\n"
			f"Title: {title}\n"
			f"Abstract: {abstract}"
		)
		blocks.append(article_text)

	source = f"ReviewID: {review_id}\n\n" + "\n\n".join(blocks)
	if max_source_chars > 0 and len(source) > max_source_chars:
		return source[:max_source_chars]
	return source


def load_rct_examples(rct_dir: Path, split: str, max_examples: int, max_source_chars: int) -> List[RCTExample]:
	inputs_path = rct_dir / f"{split}-inputs.csv"
	targets_path = rct_dir / f"{split}-targets.csv"

	if not inputs_path.exists():
		raise FileNotFoundError(f"Missing inputs file: {inputs_path}")
	if not targets_path.exists():
		raise FileNotFoundError(f"Missing targets file: {targets_path}")

	input_rows = _load_csv_rows(inputs_path)
	target_rows = _load_csv_rows(targets_path)

	grouped_inputs: Dict[str, List[Dict[str, str]]] = {}
	for row in input_rows:
		review_id = _clean(row.get("ReviewID"))
		if not review_id:
			continue
		grouped_inputs.setdefault(review_id, []).append(row)

	grouped_targets: Dict[str, List[str]] = {}
	ordered_review_ids: List[str] = []
	seen_review_ids = set()
	for row in target_rows:
		review_id = _clean(row.get("ReviewID"))
		target = _clean(row.get("Target"))
		if not review_id or not target:
			continue
		grouped_targets.setdefault(review_id, []).append(target)
		if review_id not in seen_review_ids:
			ordered_review_ids.append(review_id)
			seen_review_ids.add(review_id)

	examples: List[RCTExample] = []
	for review_id in ordered_review_ids:
		articles = grouped_inputs.get(review_id, [])
		targets = grouped_targets.get(review_id, [])

		# Keep only strict one-to-one mappings: exactly one abstract and one target summary.
		if len(articles) != 1 or len(targets) != 1:
			continue

		target = targets[0]

		source = _make_source_block(
			review_id=review_id,
			input_rows=articles,
			max_source_chars=max_source_chars,
		)
		examples.append(
			RCTExample(
				review_id=review_id,
				source=source,
				reference=target,
				num_articles=len(articles),
			)
		)
		if max_examples > 0 and len(examples) >= max_examples:
			break

	return examples


def build_prompt(source: str) -> str:
	return (
		"You are a medical evidence summarization assistant. "
		"Given a randomized controlled trial abstract, write a concise factual summary in 2-4 sentences. "
		"Do not include information that is not supported by the abstract.\n\n"
		"Source documents:\n"
		f"{source}\n\n"
		"Summary:"
	)


def build_predictor(args: argparse.Namespace):
	if args.backend == "api":
		if not args.api_key:
			raise ValueError("--api-key is required when backend=api")
		return APIPredictor(
			api_key=args.api_key,
			base_url=args.base_url,
			model=args.model,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
		)

	if args.backend == "vllm":
		return VLLMPredictor(
			model_name_or_path=args.model,
			min_new_tokens=args.min_new_tokens,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			enforce_eager=args.enforce_eager,
			max_model_len=args.max_model_len if args.max_model_len != -1 else None,
		)

	if args.backend == "hf":
		return HFPredictor(
			model_name_or_path=args.model,
			min_new_tokens=args.min_new_tokens,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
			top_p=args.top_p,
			device=args.device,
			dtype=args.dtype,
		)

	raise ValueError(f"Unsupported backend: {args.backend}")


def _save_lines(path: Path, values: Sequence[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file:
		for value in values:
			file.write(value.replace("\n", " ").strip() + "\n")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate RCT summaries and save outputs for hallucination metric evaluation."
	)
	parser.add_argument("--rct-dir", type=str, default=str(_default_rct_dir()), help="Path to RCT CSV folder.")
	parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="RCT split to use.")
	parser.add_argument("--max-examples", type=int, default=0, help="Limit number of ReviewIDs. 0 means all.")
	parser.add_argument(
		"--max-source-chars",
		type=int,
		default=0,
		help="Truncate concatenated source text at this char length. 0 means no truncation.",
	)
	parser.add_argument("--backend", type=str, default="vllm", choices=["hf", "vllm", "api"], help="Generation backend.")
	parser.add_argument("--model", type=str, required=True, help="Model id or local model path.")
	parser.add_argument(
		"--min-new-tokens",
		type=int,
		default=16,
		help="Minimum generated tokens (best-effort). Helps avoid empty outputs for HF/vLLM.",
	)
	parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generated tokens.")
	parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
	parser.add_argument("--top-p", type=float, default=0.8, help="Top-p for sampling.")
	parser.add_argument("--max-model-len", type=int, default=-1, help="Max model context length (for vLLM backend). -1 means use model default.")
	parser.add_argument("--device", type=str, default="auto", help="HF device. e.g. auto, cuda, cpu.")
	parser.add_argument(
		"--dtype",
		type=str,
		default="bfloat16",
		choices=["float16", "bfloat16", "float32"],
		help="HF torch dtype.",
	)
	parser.add_argument("--api-key", type=str, default="", help="API key for backend=api.")
	parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
	parser.add_argument("--enforce-eager", action="store_true", help="Enable vLLM eager mode.")
	parser.add_argument(
		"--output-json",
		type=str,
		default="debug/rct_summaries_predictions.json",
		help="Path to output JSON list for hallu_metrics --input-json.",
	)
	parser.add_argument(
		"--predictions-file",
		type=str,
		default="debug/rct_summaries_predictions.txt",
		help="Path to write line-aligned predictions.",
	)
	parser.add_argument(
		"--references-file",
		type=str,
		default="debug/rct_summaries_references.txt",
		help="Path to write line-aligned source documents.",
	)
	parser.add_argument(
		"--sleep-seconds",
		type=float,
		default=0.0,
		help="Optional delay between prompts for API rate limiting.",
	)
	parser.add_argument(
		"--retry-empty",
		type=int,
		default=10000,
		help="Retry generation for prompts that produced an empty summary (after post-processing).",
	)
	parser.add_argument(
		"--fail-on-empty",
		action="store_true",
		help="If any empty summaries remain after retries, raise an error.",
	)
	return parser.parse_args()


def _generate_with_optional_sleep(
	predictor: Any,
	prompts: Sequence[str],
	backend: str,
	sleep_seconds: float,
) -> List[str]:
	if backend == "api" and sleep_seconds > 0:
		outputs: List[str] = []
		for prompt in tqdm(prompts, desc="API generation", leave=False):
			outputs.extend(predictor.generate_batch([prompt]))
			time.sleep(sleep_seconds)
		return outputs
	return predictor.generate_batch(prompts)


def main() -> None:
	args = parse_args()

	rct_dir = Path(args.rct_dir)
	examples = load_rct_examples(
		rct_dir=rct_dir,
		split=args.split,
		max_examples=args.max_examples,
		max_source_chars=args.max_source_chars,
	)
	if not examples:
		raise ValueError("No usable RCT examples found.")

	prompts = [build_prompt(example.source) for example in examples]
	predictor = build_predictor(args)

	predictions = _generate_with_optional_sleep(
		predictor=predictor,
		prompts=prompts,
		backend=args.backend,
		sleep_seconds=args.sleep_seconds,
	)
	
	if "gpt-oss-20b" in args.model:
		predictions = [_postprocess_gpt_oss_prediction(pred) for pred in predictions]

	# Retry empty outputs (often caused by immediate stop tokens / short generations).
	if args.retry_empty > 0:
		empty_indices = [i for i, pred in enumerate(predictions) if _is_empty(pred)]
		for attempt in range(1, args.retry_empty + 1):
			if not empty_indices:
				break
			print(f"Warning: {len(empty_indices)} empty summaries; retrying (attempt {attempt}/{args.retry_empty})")
			retry_prompts = [prompts[i] for i in empty_indices]
			retry_outputs = _generate_with_optional_sleep(
				predictor=predictor,
				prompts=retry_prompts,
				backend=args.backend,
				sleep_seconds=args.sleep_seconds,
			)
			if "gpt-oss-20b" in args.model:
				retry_outputs = [_postprocess_gpt_oss_prediction(pred) for pred in retry_outputs]
			for idx, retry_pred in zip(empty_indices, retry_outputs):
				predictions[idx] = str(retry_pred).strip()
			empty_indices = [i for i in empty_indices if _is_empty(predictions[i])]

	if args.fail_on_empty:
		remaining_empty = [i for i, pred in enumerate(predictions) if _is_empty(pred)]
		if remaining_empty:
			sample_ids = [examples[i].review_id for i in remaining_empty[:10]]
			raise RuntimeError(
				"Empty summaries remain after retries. "
				f"count={len(remaining_empty)} sample_review_ids={sample_ids}"
			)

	if len(predictions) != len(examples):
		raise RuntimeError(
			"Generation count mismatch: "
			f"got {len(predictions)} predictions for {len(examples)} examples"
		)

	rows: List[Dict[str, Any]] = []
	for idx, (example, prediction, prompt) in enumerate(zip(examples, predictions, prompts)):
		rows.append(
			{
				"id": idx,
				"review_id": example.review_id,
				"num_articles": example.num_articles,
				"prediction": str(prediction).strip(),
				"reference": example.source,
				"target": example.reference,
				"prompt": prompt,
			}
		)

	output_json = Path(args.output_json)
	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as file:
		json.dump(rows, file, ensure_ascii=False, indent=2)
	print(f"Loaded {len(examples)} examples from split={args.split} at {rct_dir}")
	print(f"Saved JSON: {output_json}")
	print(f"Saved predictions file: {args.predictions_file}")
	print(f"Saved references file: {args.references_file}")
	print("Use with hallu_metrics.py: --input-json <json> OR --predictions-file <pred> --references-file <ref>")


if __name__ == "__main__":
	main()
