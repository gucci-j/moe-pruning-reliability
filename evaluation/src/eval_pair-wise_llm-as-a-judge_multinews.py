from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
	from tqdm import tqdm  # type: ignore[reportMissingModuleSource]
except Exception:
	tqdm = None

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
Choice = Literal["A", "B", "tie"]

fixed_selected_ids = [
	2466,
	2782,
	4329,
	3699,
	5494,
	1347,
	1345,
	1665,
	274,
	3584,
	1069,
	4324,
	859,
	4772,
	3442,
	2608,
	1589,
	3529,
	5602,
	4082,
	4084,
	1992,
	829,
	4957,
	2366,
	5438,
	3806,
	3681,
	4256,
	1680,
	593,
	1797,
	195,
	834,
	4856,
	4946,
	2124,
	5233,
	1076,
	3449,
	4074,
	4273,
	1392,
	1184,
	440,
	4034,
	394,
	1198,
	3185,
	2453
]

@dataclass(frozen=True)
class MultiNewsSource:
	example_idx: int
	source: str
	reference: str


@dataclass(frozen=True)
class PairRecord:
	sample_id: str
	example_idx: int
	source_document: str
	vanilla_summary: str
	pruned_summary: str
	metadata: Dict[str, Any]


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


def _normalize_source(raw_source: object) -> str:
	if isinstance(raw_source, list):
		parts = [str(part).strip() for part in raw_source if str(part).strip()]
		return "\n\n".join(parts)
	return str(raw_source or "").strip()


def load_multinews_sources(cleansed_dir: Path, split: str) -> Dict[int, MultiNewsSource]:
	files = sorted(cleansed_dir.glob("cleansed_data_*.json"))
	if not files:
		raise FileNotFoundError(f"No cleansed_data_*.json files found in {cleansed_dir}")

	rows: Dict[int, MultiNewsSource] = {}
	for path in files:
		with path.open("r", encoding="utf-8") as f:
			try:
				data = json.load(f)
			except Exception:
				continue
		for item in data:
			item_split = str(item.get("split", "")).lower()
			if item_split != split.lower():
				continue

			source = _normalize_source(item.get("cleansed_document", ""))
			reference = str(item.get("summary", "") or "").strip()
			if not source:
				continue

			example_idx = int(item.get("idx", len(rows)))
			rows[example_idx] = MultiNewsSource(
				example_idx=example_idx,
				source=source,
				reference=reference,
			)

	return rows


def _load_predictions(path: Path) -> Dict[int, Dict[str, Any]]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, list):
		raise ValueError(f"Expected list at {path}, got {type(payload).__name__}")

	rows: Dict[int, Dict[str, Any]] = {}
	for idx, item in enumerate(payload):
		if not isinstance(item, dict):
			continue

		prediction = str(item.get("prediction") or "").strip()
		if not prediction:
			continue

		example_idx_raw = item.get("example_idx")
		if example_idx_raw is None:
			example_idx_raw = item.get("id", idx)
		try:
			example_idx = int(example_idx_raw)
		except Exception:
			continue
		
		if item.get("id", idx) in fixed_selected_ids:
			rows[example_idx] = {
				"prediction": prediction,
				"id": item.get("id", idx),
			}

	return rows


def _eligible_example_ids(
	path: Path,
	sources_by_idx: Dict[int, MultiNewsSource],
) -> List[int]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, list):
		raise ValueError(f"Expected list at {path}, got {type(payload).__name__}")

	ordered_ids: List[int] = []
	for idx, item in enumerate(payload):
		if not isinstance(item, dict):
			continue

		prediction = str(item.get("prediction") or "").strip()
		if not prediction:
			continue

		example_idx_raw = item.get("example_idx")
		if example_idx_raw is None:
			example_idx_raw = item.get("id", idx)
		try:
			example_idx = int(example_idx_raw)
		except Exception:
			continue

		source_row = sources_by_idx.get(example_idx)
		if not source_row or not source_row.source:
			continue
		if not str(source_row.reference or "").strip():
			continue

		ordered_ids.append(example_idx)

	return ordered_ids


def build_pairs(
	vanilla_path: Path,
	pruned_path: Path,
	sources_by_idx: Dict[int, MultiNewsSource],
) -> List[PairRecord]:
	vanilla_map = _load_predictions(vanilla_path)
	pruned_map = _load_predictions(pruned_path)

	common = sorted(set(vanilla_map.keys()) & set(pruned_map.keys()) & set(sources_by_idx.keys()))
	if not common:
		raise ValueError("No overlapping example_idx values across vanilla, pruned, and sources.")
	if set(vanilla_map.keys()) != set(pruned_map.keys()):
		raise ValueError("Inconsistent example_idx values between vanilla and pruned predictions.")
	print(f"Found {len(common)} common example_idx values across vanilla, pruned, and sources.")

	pairs: List[PairRecord] = []
	for example_idx in common:
		source = sources_by_idx[example_idx].source
		if not source:
			continue

		vanilla_summary = str(vanilla_map[example_idx]["prediction"]).strip()
		pruned_summary = str(pruned_map[example_idx]["prediction"]).strip()
		if not vanilla_summary or not pruned_summary:
			continue

		pairs.append(
			PairRecord(
				sample_id=str(example_idx),
				example_idx=example_idx,
				source_document=source,
				vanilla_summary=vanilla_summary,
				pruned_summary=pruned_summary,
				metadata={
					"vanilla_path": str(vanilla_path),
					"pruned_path": str(pruned_path),
					"vanilla_id": vanilla_map[example_idx].get("id"),
					"pruned_id": pruned_map[example_idx].get("id"),
				},
			)
		)

	return pairs


def _stable_ab_assignment(seed: int, sample_id: str) -> Tuple[Literal["vanilla", "pruned"], Literal["vanilla", "pruned"]]:
	digest = hashlib.md5(f"{seed}::{sample_id}".encode("utf-8")).digest()
	bit = digest[0] & 1
	if bit == 0:
		return ("vanilla", "pruned")
	return ("pruned", "vanilla")


def _strip_code_fences(text: str) -> str:
	t = (text or "").strip()
	if t.startswith("```"):
		lines = t.splitlines()
		lines = lines[1:]
		if lines and lines[-1].strip().startswith("```"):
			lines = lines[:-1]
		t = "\n".join(lines).strip()
	return t


def _extract_first_json_obj(text: str) -> Dict[str, Any]:
	t = _strip_code_fences(text)
	start = t.find("{")
	end = t.rfind("}")
	if start == -1 or end == -1 or end <= start:
		raise ValueError("No JSON object found in response text.")
	blob = t[start : end + 1]
	parsed = json.loads(blob)
	if not isinstance(parsed, dict):
		raise ValueError(f"Expected JSON object, got {type(parsed).__name__}.")
	return parsed


def _normalize_choice(value: Any, allow_tie: bool) -> Choice:
	v = str(value or "").strip().upper()
	if v in ("A", "B"):
		return v  # type: ignore[return-value]
	if allow_tie and v in ("TIE", "EQUAL", "SAME"):
		return "tie"
	expected = "'A'/'B'/'tie'" if allow_tie else "'A'/'B'"
	raise ValueError(f"Invalid choice: {value!r} (expected {expected}).")


def _build_prompt(document: str, summary_a: str, summary_b: str, allow_tie: bool) -> str:
	tie_clause = "You may answer 'tie' only if truly indistinguishable." if allow_tie else "You MUST choose 'A' or 'B' (no ties)."

	return f"""You are an impartial evaluator. Compare Summary A vs Summary B against the Source Document.
Use ONLY the Source Document to judge support. Do NOT use outside knowledge.
Be strict about factual support.

Answer these questions:

Q1. Hallucinations: Which summary contains MORE hallucinations (unsupported content)?
Q2. Omission: Which summary is missing MORE crucial information from the document?
Q3. Repetition: Which summary contains MORE repetitive information?
Q4. Alignment: Which summary is MORE semantically aligned with the source document?

Return ONLY valid JSON with exactly these keys:
- q1_hallucinations_more: "A" or "B"{' or "tie"' if allow_tie else ""}
- q2_omission_more: "A" or "B"{' or "tie"' if allow_tie else ""}
- q3_repetition_more: "A" or "B"{' or "tie"' if allow_tie else ""}
- q4_alignment_more: "A" or "B"{' or "tie"' if allow_tie else ""}

Rules:
- {tie_clause}
- Keep outputs to JSON only (no markdown).

Source Document:
<<<DOC
{document}
DOC>>>

Summary A:
<<<A
{summary_a}
A>>>

Summary B:
<<<B
{summary_b}
B>>>
"""


def _should_retry_error(exc: Exception) -> bool:
	message = str(exc).lower()
	if "content_filter" in message or "responsibleaipolicyviolation" in message:
		return True
	return any(
		key in message
		for key in (
			"invalid json",
			"internalserverexception",
			"server error",
			"service unavailable",
			"timeout",
			"temporarily",
			"rate limit",
			"too many requests",
			"throttl",
			"connection reset",
			"connection aborted",
		)
	)


class AIGatewayJudge:
	def __init__(
		self,
		model_name: str,
		endpoint: str,
		api_key: str,
		api_versions: List[str],
		aws_region_name: str = "us-east-1",
	):
		self.model_name = model_name
		self.endpoint = endpoint
		self.api_key = api_key
		self.api_versions = api_versions
		self.aws_region_name = aws_region_name
		self.client_kind = self._infer_client_kind(model_name=model_name, endpoint=endpoint)

		model_lower = (model_name or "").lower()
		is_gemini = model_lower.startswith("google/") or "gemini" in model_lower
		endpoint_lower = (endpoint or "").rstrip("/").lower()
		if is_gemini and endpoint_lower.endswith("/azure-openai"):
			raise ValueError(
				"Gemini/Google models must use the AI Gateway 'vertex-ai-openai' endpoint (OpenAI client). "
				"Pass --gateway-endpoint vertex-ai-openai (or set AI_GATEWAY_ENDPOINT=vertex-ai-openai)."
			)

		is_bedrock_model = model_lower.startswith("us.") or "anthropic" in model_lower
		if is_bedrock_model and not endpoint_lower.endswith("/bedrock"):
			raise ValueError(
				"Bedrock modelIds (e.g. 'us.anthropic.*') must use the AI Gateway 'bedrock' endpoint. "
				"Pass --gateway-endpoint bedrock (or set AI_GATEWAY_ENDPOINT=bedrock)."
			)

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
				raise ImportError("boto3 is required for Bedrock judging. Install with: pip install boto3")

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

	def chat(self, messages: List[Dict[str, str]], temperature: float, max_completion_tokens: int) -> str:
		if self.client_kind == "bedrock":
			system_blocks: List[Dict[str, str]] = []
			bedrock_messages: List[Dict[str, Any]] = []
			for m in messages:
				role = str(m.get("role") or "user")
				content = str(m.get("content") or "")
				if role == "system":
					if content.strip():
						system_blocks.append({"text": content})
					continue

				if role not in ("user", "assistant"):
					role = "user"

				if bedrock_messages and bedrock_messages[-1].get("role") == role:
					prev = bedrock_messages[-1]["content"][0].get("text", "")
					bedrock_messages[-1]["content"][0]["text"] = (prev + "\n\n" + content).strip()
				else:
					bedrock_messages.append({"role": role, "content": [{"text": content}]})

			kwargs: Dict[str, Any] = {
				"modelId": self.model_name,
				"messages": bedrock_messages,
				"inferenceConfig": {"maxTokens": int(max_completion_tokens), "temperature": float(temperature)},
			}
			if system_blocks:
				kwargs["system"] = system_blocks

			resp = self.client.converse(**kwargs)
			return resp["output"]["message"]["content"][0]["text"]

		if self.client_kind == "openai":
			resp = self.client.chat.completions.create(
				model=self.model_name,
				messages=messages,
				temperature=temperature,
				max_completion_tokens=max_completion_tokens,
			)
			return resp.choices[0].message.content or ""

		last_error: Optional[Exception] = None
		for api_version in self.api_versions:
			try:
				if api_version != self._active_api_version:
					self.client = self._get_client(api_version)
					self._active_api_version = api_version

				resp = self.client.chat.completions.create(
					model=self.model_name,
					messages=messages,
					temperature=temperature,
					max_completion_tokens=max_completion_tokens,
				)
				return resp.choices[0].message.content or ""
			except Exception as exc:  # pragma: no cover - network dependent
				last_error = exc
				continue
		raise RuntimeError(
			f"All API versions failed for model '{self.model_name}'. Tried: {', '.join(self.api_versions)}."
		) from last_error


def _judge_one(
	record: PairRecord,
	judge: AIGatewayJudge,
	seed: int,
	allow_tie: bool,
	temperature: float,
	max_completion_tokens: int,
	max_retries: int,
	retry_backoff: float,
) -> Dict[str, Any]:
	a_model, b_model = _stable_ab_assignment(seed=seed, sample_id=record.sample_id)
	summary_a = record.vanilla_summary if a_model == "vanilla" else record.pruned_summary
	summary_b = record.pruned_summary if a_model == "vanilla" else record.vanilla_summary

	prompt = _build_prompt(
		document=record.source_document,
		summary_a=summary_a,
		summary_b=summary_b,
		allow_tie=allow_tie,
	)

	base_messages = [
		{"role": "system", "content": "You are a careful, unbiased judge."},
		{"role": "user", "content": prompt},
	]

	last_error: Optional[str] = None
	raw_text: str = ""
	for attempt in range(max_retries + 1):
		messages = list(base_messages)
		if attempt > 0:
			messages.append(
				{
					"role": "user",
					"content": "Your previous response was invalid. Return ONLY valid JSON with the required keys.",
				}
			)

		try:
			raw_text = judge.chat(
				messages=messages,
				temperature=temperature,
				max_completion_tokens=max_completion_tokens,
			)
			parsed = _extract_first_json_obj(raw_text)

			q1 = _normalize_choice(parsed.get("q1_hallucinations_more"), allow_tie=allow_tie)
			q2 = _normalize_choice(parsed.get("q2_omission_more"), allow_tie=allow_tie)
			q3 = _normalize_choice(parsed.get("q3_repetition_more"), allow_tie=allow_tie)
			q4 = _normalize_choice(parsed.get("q4_alignment_more"), allow_tie=allow_tie)

			return {
				"sample_id": record.sample_id,
				"example_idx": record.example_idx,
				"mapping": {"A": a_model, "B": b_model},
				"answers": {
					"q1_hallucinations_more": q1,
					"q2_omission_more": q2,
					"q3_repetition_more": q3,
					"q4_alignment_more": q4,
				},
				"raw_response": raw_text,
				"error": None,
				"record": {
					"document": record.source_document,
					"vanilla_summary": record.vanilla_summary,
					"pruned_summary": record.pruned_summary,
					"metadata": record.metadata,
				},
			}
		except Exception as exc:
			last_error = f"{type(exc).__name__}: {exc}"
			if attempt < max_retries and _should_retry_error(exc):
				wait_seconds = max(0.0, retry_backoff) * (attempt + 1)
				print(
					f"Retrying sample {record.sample_id} after error "
					f"(attempt {attempt + 1}/{max_retries}, wait {wait_seconds:.1f}s): {exc}"
				)
				if wait_seconds:
					time.sleep(wait_seconds)
				continue
			break

	return {
		"sample_id": record.sample_id,
		"example_idx": record.example_idx,
		"mapping": {"A": a_model, "B": b_model},
		"answers": None,
		"raw_response": raw_text,
		"error": last_error or "Unknown error",
		"record": {
			"document": record.source_document,
			"vanilla_summary": record.vanilla_summary,
			"pruned_summary": record.pruned_summary,
			"metadata": record.metadata,
		},
	}


def _aggregate_choices(values: List[str]) -> Dict[str, Any]:
	counts: Dict[str, int] = {}
	for v in values:
		counts[v] = counts.get(v, 0) + 1
	n = sum(counts.values())
	dist = {k: (counts[k] / n if n else 0.0) for k in sorted(counts.keys())}
	return {"n": n, "counts": counts, "dist": dist}


def _compute_aggregate(individual: List[Dict[str, Any]], seed: int, allow_tie: bool) -> Dict[str, Any]:
	successful = [r for r in individual if not r.get("error")]
	failed = [r for r in individual if r.get("error")]

	def _model_for_answer(r: Dict[str, Any], answer: Choice) -> str:
		m = r.get("mapping") or {}
		if answer == "tie":
			return "tie"
		return str(m.get(answer, "unknown"))

	q1_more: List[str] = []
	q2_more: List[str] = []
	q3_more: List[str] = []
	q4_more: List[str] = []

	for r in successful:
		answers = r.get("answers")
		if not isinstance(answers, dict):
			continue
		q1_more.append(_model_for_answer(r, answers["q1_hallucinations_more"]))
		q2_more.append(_model_for_answer(r, answers["q2_omission_more"]))
		q3_more.append(_model_for_answer(r, answers["q3_repetition_more"]))
		q4_more.append(_model_for_answer(r, answers["q4_alignment_more"]))

	return {
		"num_samples": len(individual),
		"num_successful": len(successful),
		"num_failed": len(failed),
		"seed": int(seed),
		"allow_tie": bool(allow_tie),
		"questions": {
			"q1_hallucinations_more": _aggregate_choices(q1_more),
			"q2_omission_more": _aggregate_choices(q2_more),
			"q3_repetition_more": _aggregate_choices(q3_more),
			"q4_alignment_more": _aggregate_choices(q4_more),
		},
	}


def _merge_individual_results(
	existing_individual: List[Dict[str, Any]],
	rerun_individual: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	rerun_by_id: Dict[str, Dict[str, Any]] = {}
	for item in rerun_individual:
		sid = str(item.get("sample_id") or "").strip()
		if sid:
			rerun_by_id[sid] = item

	merged: List[Dict[str, Any]] = []
	seen: set[str] = set()

	for item in existing_individual:
		sid = str(item.get("sample_id") or "").strip()
		if sid and sid in rerun_by_id:
			merged.append(rerun_by_id[sid])
			seen.add(sid)
		else:
			merged.append(item)
			if sid:
				seen.add(sid)

	for sid, item in sorted(rerun_by_id.items(), key=lambda kv: kv[0]):
		if sid not in seen:
			merged.append(item)

	return merged


def evaluate_pairs(
	pairs: List[PairRecord],
	model_name: str,
	endpoint: str,
	endpoint_kind: GatewayEndpoint,
	api_key: str,
	api_versions: List[str],
	seed: int,
	allow_tie: bool,
	temperature: float,
	max_completion_tokens: int,
	max_retries: int,
	retry_backoff: float,
	log_every: int,
	num_workers: int,
	aws_region_name: str,
	rerun_failed: int,
) -> Dict[str, Any]:
	if endpoint_kind != "bedrock" and AzureOpenAI is None and OpenAI is None:
		raise ImportError("openai is required. Install with: pip install openai")
	if endpoint_kind == "bedrock" and (boto3 is None or Config is None):
		raise ImportError("boto3 is required for Bedrock judging. Install with: pip install boto3")

	total = len(pairs)
	if total == 0:
		raise ValueError("No pairs provided.")

	log_every = max(1, log_every)
	num_workers = max(1, num_workers)
	start_time = time.perf_counter()

	processed = 0
	processed_lock = threading.Lock()

	def _on_progress() -> None:
		nonlocal processed
		with processed_lock:
			processed += 1
			if processed % log_every == 0 or processed == total:
				elapsed = time.perf_counter() - start_time
				rate = processed / elapsed if elapsed > 0 else 0.0
				eta = (total - processed) / rate if rate > 0 else 0.0
				print(
					f"Progress: {processed}/{total} | elapsed: {elapsed:.1f}s | rate: {rate:.2f}/s | eta: {eta:.1f}s"
				)

	indexed = list(enumerate(pairs))
	index_to_pair = {idx: rec for idx, rec in indexed}

	def _worker(chunk: List[Tuple[int, PairRecord]]) -> List[Dict[str, Any]]:
		judge = AIGatewayJudge(
			model_name=model_name,
			endpoint=endpoint,
			api_key=api_key,
			api_versions=api_versions,
			aws_region_name=aws_region_name,
		)
		out: List[Dict[str, Any]] = []
		for idx, rec in chunk:
			item = _judge_one(
				record=rec,
				judge=judge,
				seed=seed,
				allow_tie=allow_tie,
				temperature=temperature,
				max_completion_tokens=max_completion_tokens,
				max_retries=max_retries,
				retry_backoff=retry_backoff,
			)
			item["_index"] = idx
			out.append(item)
			_on_progress()
		return out

	def _worker_no_progress(chunk: List[Tuple[int, PairRecord]]) -> List[Dict[str, Any]]:
		judge = AIGatewayJudge(
			model_name=model_name,
			endpoint=endpoint,
			api_key=api_key,
			api_versions=api_versions,
			aws_region_name=aws_region_name,
		)
		out: List[Dict[str, Any]] = []
		for idx, rec in chunk:
			item = _judge_one(
				record=rec,
				judge=judge,
				seed=seed,
				allow_tie=allow_tie,
				temperature=temperature,
				max_completion_tokens=max_completion_tokens,
				max_retries=max_retries,
				retry_backoff=retry_backoff,
			)
			item["_index"] = idx
			out.append(item)
		return out

	all_results: List[Dict[str, Any]] = []
	if num_workers == 1:
		iterator = tqdm(indexed, desc="LLM-as-a-Judge", unit="sample") if tqdm is not None else indexed
		all_results.extend(_worker(list(iterator)))
	else:
		chunks: List[List[Tuple[int, PairRecord]]] = [indexed[i::num_workers] for i in range(num_workers)]
		with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
			futures = [ex.submit(_worker, c) for c in chunks if c]
			for fut in concurrent.futures.as_completed(futures):
				all_results.extend(fut.result())

	rerun_failed = max(0, int(rerun_failed))
	if rerun_failed:
		result_by_index = {item["_index"]: item for item in all_results}
		for rerun_round in range(1, rerun_failed + 1):
			failed_indices = [idx for idx, item in result_by_index.items() if item.get("error")]
			if not failed_indices:
				break
			print(f"Rerunning {len(failed_indices)} failed samples (round {rerun_round}/{rerun_failed})")

			failed_records = [(idx, index_to_pair[idx]) for idx in failed_indices if idx in index_to_pair]
			if not failed_records:
				break

			if num_workers == 1:
				new_results = _worker_no_progress(failed_records)
			else:
				chunks = [failed_records[i::num_workers] for i in range(num_workers)]
				new_results = []
				with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
					futures = [ex.submit(_worker_no_progress, c) for c in chunks if c]
					for fut in concurrent.futures.as_completed(futures):
						new_results.extend(fut.result())

			for item in new_results:
				result_by_index[item["_index"]] = item

		all_results = list(result_by_index.values())

	all_results.sort(key=lambda x: int(x["_index"]))
	for item in all_results:
		item.pop("_index", None)

	aggregate = _compute_aggregate(all_results, seed=seed, allow_tie=allow_tie)
	return {"aggregate": aggregate, "individual": all_results}


def parse_args() -> argparse.Namespace:
	repo_root = _repo_root()
	p = argparse.ArgumentParser(description="LLM-as-a-Judge pairwise evaluation for MultiNews (vanilla vs pruned).")

	p.add_argument(
		"--vanilla",
		type=Path,
		required=True,
		help="Path to vanilla predictions JSON list.",
	)
	p.add_argument(
		"--pruned",
		type=Path,
		required=True,
		help="Path to pruned predictions JSON list.",
	)
	p.add_argument(
		"--cleansed-dir",
		type=Path,
		default=repo_root / "external" / "multi_news_plus" / "cleansing" / "cleansed_data",
		help="Path to cleansed MultiNews data directory.",
	)
	p.add_argument(
		"--split",
		type=str,
		default="test",
		choices=["train", "dev", "test"],
		help="Dataset split for sources/references.",
	)
	p.add_argument(
		"--output-dir",
		type=Path,
		default=repo_root / "evaluation" / "logs_pair-wise" / "multinews",
		help="Output folder.",
	)

	p.add_argument(
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
	p.add_argument(
		"--aws-region",
		type=str,
		default=os.environ.get("AWS_REGION_NAME", "us-east-1"),
		help="AWS region for Bedrock (only used when --gateway-endpoint bedrock).",
	)
	p.add_argument(
		"--model",
		type=str,
		default=os.environ.get("JUDGE_MODEL", "gpt-5.1"),
		help="Azure model deployment name.",
	)
	p.add_argument(
		"--api-versions",
		nargs="+",
		default=["2025-02-01-preview", "2025-01-01-preview"],
		help="Azure OpenAI API versions.",
	)
	
	p.add_argument("--seed", type=int, default=123, help="Seed controlling A/B assignment (bias mitigation).")
	p.add_argument("--allow-tie", action="store_true", help="Allow 'tie' outputs (default: forced A/B).")
	p.add_argument(
		"--sample-size",
		type=int,
		default=50,
		help="If >0, randomly sample this many joined pairs after filtering (default: 50).",
	)

	p.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature.")
	p.add_argument("--max-completion-tokens", type=int, default=400, help="Max completion tokens for the judge response.")
	p.add_argument("--max-retries", type=int, default=2, help="Retries if response is invalid JSON or transient errors.")
	p.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff seconds between retries.")

	p.add_argument("--log-every", type=int, default=10, help="Print progress every N samples.")
	p.add_argument("--num-workers", type=int, default=1, help="Threaded workers; each has its own client.")
	p.add_argument(
		"--rerun-failed",
		type=int,
		default=3,
		help="Rerun failed samples this many extra rounds after the initial pass (default: 3).",
	)

	p.add_argument("--token-path", type=Path, default=Path.home() / "openai" / "token", help="Path to gateway token file.")
	p.add_argument("--env-path", type=Path, default=repo_root / ".env", help="Path to .env fallback file.")
	return p.parse_args()


def main() -> None:
	args = parse_args()

	sources_by_idx = load_multinews_sources(args.cleansed_dir, args.split)
	if not sources_by_idx:
		raise RuntimeError("No usable MultiNews sources found (non-empty documents).")

	pairs = build_pairs(args.vanilla, args.pruned, sources_by_idx)
	if not pairs:
		raise RuntimeError("No usable pairs found after joining vanilla/pruned predictions.")

	gateway_cfg = load_gateway_config(
		token_path=args.token_path,
		env_path=args.env_path,
		endpoint_kind=args.gateway_endpoint,  # type: ignore[arg-type]
	)

	input_tag = f"{args.vanilla.parent.name}__VS__{args.pruned.parent.name}"
	out_root = args.output_dir / args.model / input_tag
	out_root.mkdir(parents=True, exist_ok=True)
	aggregate_path = out_root / "aggregate_results.json"
	individual_path = out_root / "individual_results.json"

	# If output already exists:
	# (i) rerun only failed samples when previous run had failures,
	# (ii) otherwise skip when all samples were already successful.
	is_rerun_for_failed_only = False
	existing_individual: Optional[List[Dict[str, Any]]] = None
	if aggregate_path.exists() and individual_path.exists():
		try:
			aggregate_existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
			individual_any = json.loads(individual_path.read_text(encoding="utf-8"))

			if not isinstance(aggregate_existing, dict):
				raise ValueError("Existing aggregate_results.json must be a JSON object.")
			if not isinstance(individual_any, list):
				raise ValueError("Existing individual_results.json must be a JSON list.")

			existing_individual = list(individual_any)
			num_samples = int(aggregate_existing.get("num_samples", 0) or 0)
			num_failed = int(aggregate_existing.get("num_failed", 0) or 0)

			if num_samples and num_samples != len(pairs):
				print(
					"Warning: Existing aggregate count disagrees with current input pairs "
					f"(aggregate num_samples={num_samples}, input pairs={len(pairs)})."
				)

				present_ids = {str(r.get("sample_id") or "").strip() for r in existing_individual}
				present_ids = {pid for pid in present_ids if pid}
				missing_ids = [p.sample_id for p in pairs if p.sample_id not in present_ids]

				if missing_ids:
					print(f"Found {len(missing_ids)} missing samples in existing results. Rerunning only missing samples.")
					missing_id_set = set(missing_ids)
					pairs = [p for p in pairs if p.sample_id in missing_id_set]
					is_rerun_for_failed_only = True
				else:
					print("Existing output covers current input pairs. Skipping evaluation.")
					return
			elif num_failed > 0:
				print(f"Found existing results with {num_failed} failed samples. Rerunning only failed samples.")
				failed_sample_ids = {
					str(r.get("sample_id") or "").strip() for r in existing_individual if r.get("error")
				}
				failed_sample_ids = {sid for sid in failed_sample_ids if sid}

				if not failed_sample_ids:
					print("Existing aggregate indicates failures, but no failed sample_ids found. Rerunning full evaluation.")
					existing_individual = None
				else:
					pairs = [p for p in pairs if p.sample_id in failed_sample_ids]
					if not pairs:
						print("No matching pairs found for failed samples. Exiting.")
						return
					is_rerun_for_failed_only = True
			else:
				print(f"All {num_samples or len(existing_individual)} samples already successfully evaluated. Skipping evaluation.")
				return
		except Exception as exc:
			print(f"Error loading existing results: {exc}. Proceeding with full evaluation.")
			if not is_rerun_for_failed_only:
				existing_individual = None

	results = evaluate_pairs(
		pairs=pairs,
		model_name=args.model,
		endpoint=gateway_cfg["endpoint"],
		endpoint_kind=gateway_cfg["endpoint_kind"],
		api_key=gateway_cfg["api_key"],
		api_versions=args.api_versions,
		seed=int(args.seed),
		allow_tie=bool(args.allow_tie),
		temperature=float(args.temperature),
		max_completion_tokens=int(args.max_completion_tokens),
		max_retries=int(args.max_retries),
		retry_backoff=float(args.retry_backoff),
		log_every=int(args.log_every),
		num_workers=int(args.num_workers),
		aws_region_name=str(args.aws_region),
		rerun_failed=int(args.rerun_failed),
	)

	if is_rerun_for_failed_only:
		print("Rerun for failed/missing samples only. Merging new results with existing results.")
		if existing_individual is None:
			raise RuntimeError("Internal error: expected existing individual results for merge.")

		merged_individual = _merge_individual_results(existing_individual, results["individual"])
		merged_aggregate = _compute_aggregate(
			merged_individual,
			seed=int(args.seed),
			allow_tie=bool(args.allow_tie),
		)

		aggregate_path.write_text(json.dumps(merged_aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
		individual_path.write_text(json.dumps(merged_individual, ensure_ascii=False, indent=2), encoding="utf-8")
		results = {"aggregate": merged_aggregate, "individual": merged_individual}
	else:
		aggregate_path.write_text(json.dumps(results["aggregate"], ensure_ascii=False, indent=2), encoding="utf-8")
		individual_path.write_text(json.dumps(results["individual"], ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"Saved aggregate results to: {aggregate_path}")
	print(f"Saved individual results to: {individual_path}")
	print(
		"Evaluated samples: "
		f"{results['aggregate']['num_samples']} (ok={results['aggregate']['num_successful']}, "
		f"failed={results['aggregate']['num_failed']})"
	)


if __name__ == "__main__":
	main()
