from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

try:
	from tqdm import tqdm  # type: ignore[reportMissingModuleSource]
except Exception:
	tqdm = None

try:
	from deepeval.metrics import HallucinationMetric  # type: ignore[reportMissingImports]
	from deepeval.models import DeepEvalBaseLLM  # type: ignore[reportMissingImports]
	from deepeval.test_case import LLMTestCase  # type: ignore[reportMissingImports]
except Exception:
	HallucinationMetric = None
	DeepEvalBaseLLM = object
	LLMTestCase = None

try:
	from openai import AzureOpenAI, OpenAI  # type: ignore[reportMissingImports]
except Exception:
	AzureOpenAI = None
	OpenAI = None

try:
	import boto3  # type: ignore[reportMissingImports]
	from botocore.config import Config  # type: ignore[reportMissingImports]
except Exception:
	boto3 = None
	Config = None


GatewayEndpoint = Literal["azure-openai", "vertex-ai-openai", "bedrock"]


@dataclass
class PredictionRecord:
	sample_id: str
	task: str
	prediction: str
	reference: str
	target: str


def _repo_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _read_gateway_key(token_path: Path) -> Optional[str]:
	if token_path.exists():
		return token_path.read_text(encoding="utf-8").strip() or None
	return None


def _load_dotenv_fallback(env_path: Path) -> Dict[str, str]:
	if not env_path.exists():
		return {}

	values: Dict[str, str] = {}
	for line in env_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		values[key.strip()] = value.strip().strip('"').strip("'")
	return values


def load_gateway_config(token_path: Path, env_path: Path, endpoint_kind: GatewayEndpoint) -> Dict[str, Any]:
	gateway_url = os.environ.get("AI_GATEWAY_URL", "https://ai-gateway.astrazeneca.net")
	api_key = os.environ.get("AI_GATEWAY_KEY")

	if not api_key:
		api_key = _read_gateway_key(token_path)

	if not api_key:
		dotenv_values = _load_dotenv_fallback(env_path)
		gateway_url = gateway_url or dotenv_values.get("AI_GATEWAY_URL", "")
		api_key = api_key or dotenv_values.get("AI_GATEWAY_KEY")

	if not gateway_url or not api_key:
		raise ValueError(
			"Missing AI gateway config. Set AI_GATEWAY_URL and AI_GATEWAY_KEY, "
			f"or provide token file at {token_path}."
		)

	return {
		"gateway_url": gateway_url,
		"endpoint": f"{gateway_url.rstrip('/')}/{endpoint_kind}",
		"endpoint_kind": endpoint_kind,
		"api_key": api_key,
	}


def load_predictions(
	path: Path,
	category: str,
	task_name: str,
) -> List[PredictionRecord]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError(f"Expected dict at {path}, got {type(payload).__name__}")
	target_subset = f"{category}/{task_name}"

	# Preferred format: evaluation/logs_hallu/source/*.json
	if isinstance(payload.get("results"), list):
		rows: List[PredictionRecord] = []
		for idx, item in enumerate(payload["results"]):
			if not isinstance(item, dict):
				continue

			source = str(item.get("source") or "")
			subset = str(item.get("subset") or "")
			if source != "medinst32" or subset != target_subset:
				continue

			prediction = str(item.get("prediction_for_metrics") or item.get("prediction") or "").strip()
			reference = str(item.get("reference_for_metrics") or item.get("reference") or "").strip()
			target = str(item.get("target") or "").strip()
			if not prediction or not reference:
				continue

			rows.append(
				PredictionRecord(
					sample_id=str(item.get("sample_id") or item.get("id") or idx),
					task=task_name,
					prediction=prediction,
					reference=reference,
					target=target,
				)
			)

		return rows

	# Backward-compatible format: evaluation/logs_medinst32/source/*.json
	task_blob = payload.get(category, {}).get(task_name, {})
	if not isinstance(task_blob, dict):
		raise ValueError(f"Could not find task '{category}/{task_name}' in {path}")

	generated = task_blob.get("generated", [])
	if not isinstance(generated, list):
		raise ValueError(f"Expected 'generated' list under '{category}/{task_name}'")

	rows: List[PredictionRecord] = []
	for idx, item in enumerate(generated):
		if not isinstance(item, dict):
			continue

		prediction = str(item.get("prediction") or "").strip()
		target = str(item.get("target") or "").strip()
		reference = str(item.get("reference") or target).strip()

		if not prediction or not reference:
			continue

		rows.append(
			PredictionRecord(
				sample_id=str(idx),
				task=task_name,
				prediction=prediction,
				reference=reference,
				target=target,
			)
		)

	return rows


class AIGatewayModel(DeepEvalBaseLLM):
	def __init__(
		self,
		model_name: str,
		endpoint: str,
		endpoint_kind: GatewayEndpoint,
		api_key: str,
		api_versions: List[str],
		aws_region_name: str = "us-east-1",
		max_completion_tokens: int = 512,
		temperature: float = 0.0,
	):
		self.model_name = model_name
		self.endpoint = endpoint
		self.endpoint_kind = endpoint_kind
		self.api_key = api_key
		self.api_versions = api_versions
		self.aws_region_name = aws_region_name
		self.max_completion_tokens = int(max_completion_tokens)
		self.temperature = float(temperature)

		endpoint_lower = (endpoint or "").rstrip("/").lower()
		model_lower = (model_name or "").lower()
		is_gemini = model_lower.startswith("google/") or "gemini" in model_lower
		if is_gemini and endpoint_lower.endswith("/azure-openai"):
			raise ValueError(
				"Gemini/Google models must use the AI Gateway 'vertex-ai-openai' endpoint. "
				"Pass --gateway-endpoint vertex-ai-openai (or set AI_GATEWAY_ENDPOINT=vertex-ai-openai)."
			)

		is_bedrock_model = model_lower.startswith("us.") or "anthropic" in model_lower
		if is_bedrock_model and not endpoint_lower.endswith("/bedrock"):
			raise ValueError(
				"Bedrock modelIds (e.g. 'us.anthropic.*') must use the AI Gateway 'bedrock' endpoint. "
				"Pass --gateway-endpoint bedrock (or set AI_GATEWAY_ENDPOINT=bedrock)."
			)

		self.client_kind = self._infer_client_kind(model_name=model_name, endpoint=endpoint)
		self._active_api_version = api_versions[0] if api_versions else ""
		self.client = self._get_client(api_version=self._active_api_version)

	@staticmethod
	def _infer_client_kind(model_name: str, endpoint: str) -> Literal["azure", "openai", "bedrock"]:
		ep = (endpoint or "").rstrip("/")
		if ep.endswith("/vertex-ai-openai"):
			return "openai"
		if ep.endswith("/azure-openai"):
			return "azure"
		if ep.endswith("/bedrock"):
			return "bedrock"

		m = (model_name or "").lower()
		if m.startswith("us.") or "anthropic" in m:
			return "bedrock"
		if m.startswith("google/") or "gemini" in m:
			return "openai"
		return "azure"

	def _get_client(self, api_version: str) -> Any:
		if self.client_kind == "bedrock":
			if boto3 is None or Config is None:
				raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

			os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.api_key
			return boto3.client(
				region_name=self.aws_region_name,
				service_name="bedrock-runtime",
				endpoint_url=self.endpoint,
				aws_access_key_id="",
				aws_secret_access_key="",
				config=Config(retries={"max_attempts": 0}),
			)

		if self.client_kind == "openai":
			if OpenAI is None:
				raise ImportError("openai is required. Install with: pip install openai")
			return OpenAI(api_key=self.api_key, base_url=self.endpoint)

		if AzureOpenAI is None:
			raise ImportError("openai is required. Install with: pip install openai")
		return AzureOpenAI(api_key=self.api_key, api_version=api_version, azure_endpoint=self.endpoint)

	def _chat(self, prompt: str) -> str:
		if self.client_kind == "bedrock":
			resp = self.client.converse(
				modelId=self.model_name,
				messages=[{"role": "user", "content": [{"text": str(prompt)}]}],
				inferenceConfig={
					"maxTokens": int(self.max_completion_tokens),
					"temperature": float(self.temperature),
				},
			)
			return resp["output"]["message"]["content"][0]["text"]

		if self.client_kind == "openai":
			resp = self.client.chat.completions.create(
				model=self.model_name,
				messages=[{"role": "user", "content": prompt}],
			)
			return resp.choices[0].message.content or ""

		last_error = None
		for api_version in self.api_versions:
			try:
				if api_version != self._active_api_version:
					self.client = self._get_client(api_version)
					self._active_api_version = api_version

				resp = self.client.chat.completions.create(
					model=self.model_name,
					messages=[{"role": "user", "content": prompt}],
				)
				return resp.choices[0].message.content or ""
			except Exception as exc:  # pragma: no cover - network dependent
				last_error = exc
				continue

		versions = ", ".join(self.api_versions)
		raise RuntimeError(
			f"All API versions failed for model '{self.model_name}'. Tried: {versions}."
		) from last_error

	def load_model(self):
		return self.client

	def get_model_name(self):
		return self.model_name

	def generate(self, prompt: str) -> str:
		return self._chat(prompt)

	async def a_generate(self, prompt: str) -> str:
		import asyncio
		return await asyncio.to_thread(self._chat, prompt)


def _metric_fields(metric_obj: Any) -> Dict[str, Any]:
	return {
		"score": float(getattr(metric_obj, "score", 0.0) or 0.0),
		"threshold": float(getattr(metric_obj, "threshold", 0.0) or 0.0),
		"success": bool(getattr(metric_obj, "success", False)),
		"reason": str(getattr(metric_obj, "reason", "") or ""),
	}


def _metric_error_fields(threshold: float, error: Exception) -> Dict[str, Any]:
	return {
		"score": 0.0,
		"threshold": float(threshold),
		"success": False,
		"reason": "",
		"error": f"{type(error).__name__}: {error}",
	}


def _measure_metric_with_retry(
	metric_name: str,
	model: Any,
	threshold: float,
	test_case: Any,
	max_retries: int = 2,
) -> Dict[str, Any]:
	last_error: Optional[Exception] = None

	for _ in range(max_retries + 1):
		try:
			if metric_name == "hallucination":
				metric = HallucinationMetric(
					threshold=threshold,
					model=model,
					verbose_mode=False,
					include_reason=True,
				)
			else:
				raise ValueError(f"Unsupported metric: {metric_name}")

			metric.measure(test_case)
			return _metric_fields(metric)
		except Exception as exc:
			last_error = exc
			continue

	if last_error is None:
		last_error = RuntimeError("Unknown metric error")
	return _metric_error_fields(threshold=threshold, error=last_error)


def _safe_mean(values: List[float]) -> float:
	return float(statistics.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
	if len(values) <= 1:
		return 0.0
	return float(statistics.pstdev(values))


def _evaluate_record(
	row: PredictionRecord,
	model: Any,
	threshold: float,
) -> Dict[str, Any]:
	hallucination_case = LLMTestCase(
		input="Writing the related-work section of a paper based on its abstract and the articles it references.", # The input you supplied to your LLM application.
		actual_output=row.prediction,
		context=[row.reference],
		expected_output=row.target,
	)

	hallucination_result = _measure_metric_with_retry(
		metric_name="hallucination",
		model=model,
		threshold=threshold,
		test_case=hallucination_case,
	)

	has_error = bool(hallucination_result.get("error"))

	return {
		"id": row.sample_id,
		"task": row.task,
		"prediction": row.prediction,
		"reference": row.reference,
		"target": row.target,
		"hallucination": hallucination_result,
		"has_error": has_error,
	}


def _worker_eval(
	records_with_index: List[tuple[int, PredictionRecord]],
	model_name: str,
	endpoint: str,
	endpoint_kind: GatewayEndpoint,
	api_key: str,
	api_versions: List[str],
	threshold: float,
	aws_region_name: str,
	progress_callback: Optional[Callable[[], None]] = None,
) -> List[Dict[str, Any]]:
	model = AIGatewayModel(
		model_name=model_name,
		endpoint=endpoint,
		endpoint_kind=endpoint_kind,
		api_key=api_key,
		api_versions=api_versions,
		aws_region_name=aws_region_name,
	)

	results: List[Dict[str, Any]] = []
	for idx, row in records_with_index:
		item = _evaluate_record(row=row, model=model, threshold=threshold)
		item["_index"] = idx
		results.append(item)
		if progress_callback is not None:
			progress_callback()
	return results


def evaluate_dataset(
	records: List[PredictionRecord],
	model_name: str,
	endpoint: str,
	endpoint_kind: GatewayEndpoint,
	api_key: str,
	api_versions: List[str],
	threshold: float,
	log_every: int,
	num_workers: int,
	aws_region_name: str,
) -> Dict[str, Any]:
	if HallucinationMetric is None or LLMTestCase is None:
		raise ImportError("deepeval is required. Install with: pip install deepeval")
	if endpoint_kind != "bedrock" and AzureOpenAI is None and OpenAI is None:
		raise ImportError("openai is required. Install with: pip install openai")
	if endpoint_kind == "bedrock" and (boto3 is None or Config is None):
		raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

	total = len(records)
	start_time = time.perf_counter()
	log_every = max(1, log_every)
	num_workers = max(1, num_workers)

	print(f"Starting Multi-XScience evaluation for {total} samples with {num_workers} worker(s)...")

	processed_count = 0
	processed_lock = threading.Lock()
	all_results: List[Dict[str, Any]] = []

	def _on_progress() -> None:
		nonlocal processed_count
		with processed_lock:
			processed_count += 1
			if processed_count % log_every == 0 or processed_count == total:
				elapsed = time.perf_counter() - start_time
				rate = processed_count / elapsed if elapsed > 0 else 0.0
				remaining = (total - processed_count) / rate if rate > 0 else 0.0
				print(
					f"Progress: {processed_count}/{total} | elapsed: {elapsed:.1f}s | "
					f"rate: {rate:.2f} samples/s | eta: {remaining:.1f}s"
				)

	records_with_index = list(enumerate(records))

	if num_workers == 1:
		if tqdm is not None:
			iterator = tqdm(records_with_index, desc="DeepEval Multi-XScience", unit="sample")
		else:
			iterator = records_with_index

		worker_results = _worker_eval(
			records_with_index=list(iterator),
			model_name=model_name,
			endpoint=endpoint,
			endpoint_kind=endpoint_kind,
			api_key=api_key,
			api_versions=api_versions,
			threshold=threshold,
			aws_region_name=aws_region_name,
			progress_callback=_on_progress,
		)
		all_results.extend(worker_results)
	else:
		chunks: List[List[tuple[int, PredictionRecord]]] = [
			records_with_index[i::num_workers] for i in range(num_workers)
		]

		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
			futures = [
				executor.submit(
					_worker_eval,
					chunk,
					model_name,
					endpoint,
					endpoint_kind,
					api_key,
					api_versions,
					threshold,
					aws_region_name,
					_on_progress,
				)
				for chunk in chunks
				if chunk
			]

			for future in concurrent.futures.as_completed(futures):
				all_results.extend(future.result())

	all_results.sort(key=lambda item: int(item["_index"]))
	individual_results: List[Dict[str, Any]] = []
	for item in all_results:
		item.pop("_index", None)
		individual_results.append(item)

	successful_results = [item for item in individual_results if not item.get("has_error", False)]
	hallucination_scores = [float(item["hallucination"]["score"]) for item in successful_results]
	num_errors = len(individual_results) - len(successful_results)

	aggregate = {
		"num_samples": len(individual_results),
		"num_successful": len(successful_results),
		"num_failed": num_errors,
		"hallucination": {
			"mean": _safe_mean(hallucination_scores),
			"std": _safe_std(hallucination_scores),
			"min": float(min(hallucination_scores)) if hallucination_scores else 0.0,
			"max": float(max(hallucination_scores)) if hallucination_scores else 0.0,
			"pass_rate": (
				float(sum(score <= threshold for score in hallucination_scores) / len(hallucination_scores))
				if hallucination_scores
				else 0.0
			),
		},
		"threshold": threshold,
	}

	return {
		"aggregate": aggregate,
		"individual": individual_results,
	}


def parse_args() -> argparse.Namespace:
	repo_root = _repo_root()
	parser = argparse.ArgumentParser(
		description="Evaluate Multi-XScience summaries with DeepEval Hallucination metrics."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=repo_root / "evaluation" / "logs_hallu" / "source" / "gpt-oss-20b.json",
		help="Path to hallu-source JSON or MedINST JSON.",
	)
	parser.add_argument(
		"--category",
		type=str,
		default="SUM",
		help="Top-level category in the MedINST output JSON.",
	)
	parser.add_argument(
		"--task-name",
		type=str,
		default="Multi-XScience",
		help="Task key under category.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=repo_root / "evaluation" / "logs_hallu" / "source" / "deepeval",
		help="Output directory for aggregate and individual results.",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=os.environ.get("DEEPEVAL_MODEL", "gpt-5.4"),
		help="Azure model deployment name.",
	)
	parser.add_argument(
		"--api-versions",
		nargs="+",
		default=["2025-02-01-preview", "2025-01-01-preview"],
		help="Azure OpenAI API versions to try in order.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Threshold for Hallucination metrics.",
	)
	parser.add_argument(
		"--log-every",
		type=int,
		default=10,
		help="Print progress every N samples.",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=1,
		help="Number of threaded workers. Each worker has its own model/client instance.",
	)
	parser.add_argument(
		"--gateway-endpoint",
		type=str,
		choices=["azure-openai", "vertex-ai-openai", "bedrock"],
		default=os.environ.get("AI_GATEWAY_ENDPOINT", "azure-openai"),
		help=(
			"AI Gateway endpoint path. Use 'vertex-ai-openai' for Gemini-style models (OpenAI client), "
			"'azure-openai' for GPT-style models (AzureOpenAI client), and 'bedrock' for Bedrock modelIds "
			"(e.g. us.anthropic.* via boto3 converse)."
		),
	)
	parser.add_argument(
		"--aws-region",
		type=str,
		default=os.environ.get("AWS_REGION_NAME", "us-east-1"),
		help="AWS region for Bedrock (only used when --gateway-endpoint bedrock).",
	)
	parser.add_argument(
		"--token-path",
		type=Path,
		default=Path.home() / "openai" / "token",
		help="Path to gateway token file.",
	)
	parser.add_argument(
		"--env-path",
		type=Path,
		default=repo_root / ".env",
		help="Path to .env fallback file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	output_root = args.output_dir / args.model / args.task_name
	output_root.mkdir(parents=True, exist_ok=True)
	aggregate_path = output_root / "aggregate_results.json"
	individual_path = output_root / "individual_results.json"
	expected_samples = 200
	existing_individual: Optional[List[Dict[str, Any]]] = None
	is_rerun_for_missing_only = False

	records = load_predictions(
		path=args.input,
		category=args.category,
		task_name=args.task_name,
	)
	if not records:
		raise RuntimeError(f"No valid records found for {args.category}/{args.task_name} in {args.input}")
	else:
		print(f"Loaded {len(records)} prediction records for evaluation.")

	if individual_path.exists():
		try:
			existing_any = json.loads(individual_path.read_text(encoding="utf-8"))
			if not isinstance(existing_any, list):
				raise ValueError("Existing individual_results.json must be a JSON list.")

			existing_individual = list(existing_any)
			existing_count = len(existing_individual)
			if existing_count >= expected_samples:
				print(
					f"Existing results already contain {existing_count} samples. Skipping evaluation."
				)
				return
		except Exception as exc:
			print(f"Error loading existing results: {exc}. Proceeding with full evaluation.")
			existing_individual = None

	if existing_individual is not None:
		existing_ids = {str(item.get("id") or "").strip() for item in existing_individual}
		existing_ids = {sid for sid in existing_ids if sid}
		missing_records = [r for r in records if r.sample_id not in existing_ids]
		if not missing_records:
			print("Existing results already cover all samples in the input. Skipping evaluation.")
			return

		print(
			f"Found {len(missing_records)} missing samples. Running only missing predictions."
		)
		records = missing_records
		is_rerun_for_missing_only = True

	gateway_cfg = load_gateway_config(
		token_path=args.token_path,
		env_path=args.env_path,
		endpoint_kind=args.gateway_endpoint,  # type: ignore[arg-type]
	)

	results = evaluate_dataset(
		records=records,
		model_name=args.model,
		endpoint=gateway_cfg["endpoint"],
		endpoint_kind=gateway_cfg["endpoint_kind"],
		api_key=gateway_cfg["api_key"],
		api_versions=args.api_versions,
		threshold=args.threshold,
		log_every=args.log_every,
		num_workers=args.num_workers,
		aws_region_name=str(args.aws_region),
	)

	if is_rerun_for_missing_only:
		print("Merging missing-sample results with existing outputs.")
		if existing_individual is None:
			raise RuntimeError("Internal error: expected existing individual results for merge.")

		merged_individual = existing_individual + results["individual"]
		hallucination_scores = [
			float(item["hallucination"]["score"])
			for item in merged_individual
			if not item.get("has_error", False)
		]
		num_errors = sum(1 for item in merged_individual if item.get("has_error", False))

		merged_aggregate = {
			"num_samples": len(merged_individual),
			"num_successful": len(merged_individual) - num_errors,
			"num_failed": num_errors,
			"hallucination": {
				"mean": _safe_mean(hallucination_scores),
				"std": _safe_std(hallucination_scores),
				"min": float(min(hallucination_scores)) if hallucination_scores else 0.0,
				"max": float(max(hallucination_scores)) if hallucination_scores else 0.0,
				"pass_rate": (
					float(sum(score <= args.threshold for score in hallucination_scores) / len(hallucination_scores))
					if hallucination_scores
					else 0.0
				),
			},
			"threshold": float(args.threshold),
		}

		aggregate_path.write_text(
			json.dumps(merged_aggregate, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		individual_path.write_text(
			json.dumps(merged_individual, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		results = {"aggregate": merged_aggregate, "individual": merged_individual}
	else:
		aggregate_path.write_text(
			json.dumps(results["aggregate"], ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		individual_path.write_text(
			json.dumps(results["individual"], ensure_ascii=False, indent=2),
			encoding="utf-8",
		)

	print(f"Saved aggregate results to: {aggregate_path}")
	print(f"Saved individual results to: {individual_path}")
	print(f"Evaluated samples: {results['aggregate']['num_samples']}")


if __name__ == "__main__":
	main()
