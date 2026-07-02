from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    source: str
    subset: str
    document: str
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_hallu_log_rows(path: Path) -> List[Dict[str, Any]]:
    """Load rows from a hallu metrics payload.

    Expected format:
    - {"results": [...]} (evaluation/logs_hallu/**/<model>.json)

    Note: This intentionally does NOT accept `*_records.json` list payloads.
    """

    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
    else:
        raise ValueError(
            "Unsupported input JSON format. Expected a dict with a 'results' list (hallu metrics log). "
            f"Got: {type(payload).__name__} from {path}"
        )

    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        # Keep the object as-is; downstream code picks keys.
        item.setdefault("_row_index", idx)
        cleaned.append(item)
    return cleaned


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
        f"None of the expected keys {list(candidates)} found with non-empty values. "
        f"Available keys: [{available}]"
    )


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
        # drop first fence line
        lines = lines[1:]
        # drop last fence if present
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

            # AI Gateway uses bearer-token auth for Bedrock.
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
            # Bedrock Converse supports a separate `system` field.
            system_blocks: List[Dict[str, str]] = []
            bedrock_messages: List[Dict[str, Any]] = []
            for m in messages:
                role = str(m.get("role") or "user")
                content = str(m.get("content") or "")
                if role == "system":
                    if content.strip():
                        system_blocks.append({"text": content})
                    continue

                # Bedrock validation is stricter than OpenAI; keep roles to user/assistant.
                if role not in ("user", "assistant"):
                    role = "user"

                # Coalesce adjacent same-role messages (our retry loop may append consecutive user prompts).
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


def load_pairs_from_logs(vanilla_log: Path, pruned_log: Path) -> List[PairRecord]:
    """Load and join vanilla vs pruned samples from hallu metrics logs."""

    v_rows = _load_hallu_log_rows(vanilla_log)
    p_rows = _load_hallu_log_rows(pruned_log)

    def _idx(rows: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for item in rows:
            sid = str(item.get("sample_id") or item.get("id") or item.get("_row_index") or "").strip()
            if not sid:
                continue
            if sid in index:
                raise ValueError(f"Duplicate sample_id={sid!r} in {label}.")
            index[sid] = item
        return index

    v_index = _idx(v_rows, f"vanilla_log={vanilla_log}")
    p_index = _idx(p_rows, f"pruned_log={pruned_log}")

    common_ids = sorted(set(v_index.keys()) & set(p_index.keys()))
    if not common_ids:
        raise ValueError("No overlapping sample_id values between vanilla and pruned logs.")

    pairs: List[PairRecord] = []
    for sid in common_ids:
        v = v_index[sid]
        p = p_index[sid]

        source = str(v.get("source") or p.get("source") or "unknown")
        subset = str(v.get("subset") or p.get("subset") or "unknown")

        v_doc = str(v.get("reference_for_metrics") or v.get("reference") or "").strip()
        p_doc = str(p.get("reference_for_metrics") or p.get("reference") or "").strip()
        document = v_doc or p_doc
        if not document:
            raise ValueError(f"Missing reference document for sample_id={sid!r}")

        v_sum = str(v.get("prediction_for_metrics") or v.get("prediction") or "").strip()
        p_sum = str(p.get("prediction_for_metrics") or p.get("prediction") or "").strip()
        if not v_sum or not p_sum:
            raise ValueError(f"Missing summaries for sample_id={sid!r}")

        pairs.append(
            PairRecord(
                sample_id=sid,
                source=source,
                subset=subset,
                document=document,
                vanilla_summary=v_sum,
                pruned_summary=p_sum,
                metadata={
                    "vanilla_log_path": str(vanilla_log),
                    "pruned_log_path": str(pruned_log),
                    "vanilla_row_index": v.get("_row_index"),
                    "pruned_row_index": p.get("_row_index"),
                    "vanilla_metadata": v.get("metadata", {}),
                    "pruned_metadata": p.get("metadata", {}),
                },
            )
        )

    return pairs


def load_pairs_from_pairs_json(path: Path) -> List[PairRecord]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("--pairs-json must be a JSON list.")

    pairs: List[PairRecord] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id") or item.get("id") or f"idx_{idx}").strip()
        source = str(item.get("source") or item.get("dataset") or "unknown")
        subset = str(item.get("subset") or "unknown")
        document = _pick_first_key(item, ["document", "source_document", "article", "input", "reference", "context"])
        vanilla = _pick_first_key(item, ["vanilla_summary", "summary_vanilla", "vanilla", "original", "summary_a"])
        pruned = _pick_first_key(item, ["pruned_summary", "summary_pruned", "pruned", "compressed", "summary_b"])

        pairs.append(
            PairRecord(
                sample_id=sample_id,
                source=source,
                subset=subset,
                document=document,
                vanilla_summary=vanilla,
                pruned_summary=pruned,
                metadata={k: v for k, v in item.items() if k not in ("document", "vanilla_summary", "pruned_summary")},
            )
        )
    return pairs


def _judge_one(
    record: PairRecord,
    judge: AIGatewayJudge,
    seed: int,
    allow_tie: bool,
    temperature: float,
    max_completion_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
    a_model, b_model = _stable_ab_assignment(seed=seed, sample_id=record.sample_id)
    summary_a = record.vanilla_summary if a_model == "vanilla" else record.pruned_summary
    summary_b = record.pruned_summary if a_model == "vanilla" else record.vanilla_summary

    prompt = _build_prompt(
        document=record.document,
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
                "source": record.source,
                "subset": record.subset,
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
                    "document": record.document,
                    "vanilla_summary": record.vanilla_summary,
                    "pruned_summary": record.pruned_summary,
                    "metadata": record.metadata,
                },
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

    return {
        "sample_id": record.sample_id,
        "source": record.source,
        "subset": record.subset,
        "mapping": {"A": a_model, "B": b_model},
        "answers": None,
        "raw_response": raw_text,
        "error": last_error or "Unknown error",
        "record": {
            "document": record.document,
            "vanilla_summary": record.vanilla_summary,
            "pruned_summary": record.pruned_summary,
            "metadata": record.metadata,
        },
    }


def _safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _aggregate_choices(values: List[str]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    n = sum(counts.values())
    dist = {k: (counts[k] / n if n else 0.0) for k in sorted(counts.keys())}
    return {"n": n, "counts": counts, "dist": dist}


def _compute_aggregate_from_individual(individual: List[Dict[str, Any]], seed: int, allow_tie: bool) -> Dict[str, Any]:
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

    # Append any rerun items that weren't in the existing file.
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
    log_every: int,
    num_workers: int,
    aws_region_name: str,
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
                print(f"Progress: {processed}/{total} | elapsed: {elapsed:.1f}s | rate: {rate:.2f}/s | eta: {eta:.1f}s")

    indexed = list(enumerate(pairs))

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
            )
            item["_index"] = idx
            out.append(item)
            _on_progress()
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

    all_results.sort(key=lambda x: int(x["_index"]))
    for item in all_results:
        item.pop("_index", None)

    successful = [r for r in all_results if not r.get("error")]
    failed = [r for r in all_results if r.get("error")]

    def _model_for_answer(r: Dict[str, Any], answer: Choice) -> str:
        m = r.get("mapping") or {}
        if answer == "tie":
            return "tie"
        return str(m.get(answer, "unknown"))

    # Collect per-question distributions over which model was selected by the question semantics.
    # Q1/Q2/Q3 ask "which is MORE (bad thing)" -> model counted is the one judged worse on that dimension.
    # Q4 asks "which is MORE aligned" -> model counted is the one judged better on alignment.
    q1_more: List[str] = []
    q2_more: List[str] = []
    q3_more: List[str] = []
    q4_more: List[str] = []

    for r in successful:
        a = r["answers"]
        q1_more.append(_model_for_answer(r, a["q1_hallucinations_more"]))
        q2_more.append(_model_for_answer(r, a["q2_omission_more"]))
        q3_more.append(_model_for_answer(r, a["q3_repetition_more"]))
        q4_more.append(_model_for_answer(r, a["q4_alignment_more"]))

    aggregate = {
        "num_samples": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(failed),
        "seed": seed,
        "allow_tie": allow_tie,
        "questions": {
            "q1_hallucinations_more": _aggregate_choices(q1_more),
            "q2_omission_more": _aggregate_choices(q2_more),
            "q3_repetition_more": _aggregate_choices(q3_more),
            "q4_alignment_more": _aggregate_choices(q4_more),
        },
    }

    return {"aggregate": aggregate, "individual": all_results}


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    p = argparse.ArgumentParser(description="LLM-as-a-Judge pairwise evaluation (vanilla vs pruned) for Q1–Q4.")

    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--pairs-json", type=Path, default=None, help="JSON list with document + vanilla/pruned summaries.")
    inp.add_argument(
        "--vanilla-log",
        type=Path,
        default=None,
        help="Path to vanilla hallu metrics log JSON (dict with a 'results' list).",
    )

    p.add_argument(
        "--pruned-log",
        type=Path,
        default=None,
        help="Path to pruned hallu metrics log JSON (dict with a 'results' list). Required with --vanilla-log.",
    )

    p.add_argument("--output-dir", type=Path, default=repo_root / "evaluation" / "logs_pair-wise", help="Output folder.")

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

    p.add_argument("--model", type=str, default=os.environ.get("JUDGE_MODEL", "gpt-5.1"), help="Azure model deployment name.")
    p.add_argument("--api-versions", nargs="+", default=["2025-02-01-preview", "2025-01-01-preview"], help="Azure OpenAI API versions.")

    p.add_argument("--seed", type=int, default=123, help="Seed controlling A/B assignment (bias mitigation).")
    p.add_argument("--allow-tie", action="store_true", help="Allow 'tie' outputs (default: forced A/B).")

    p.add_argument(
        "--source-filter",
        nargs="+",
        default=None,
        help="If provided, only keep samples whose 'source' is in this list (e.g., rct medinst32).",
    )
    p.add_argument(
        "--subset-filter",
        nargs="+",
        default=None,
        help="If provided, only keep samples whose 'subset' is in this list (e.g., rct SUM/Multi-XScience).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="If >0, randomly sample this many joined pairs after filtering (without replacement).",
    )

    p.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature.")
    p.add_argument("--max-completion-tokens", type=int, default=400, help="Max completion tokens for the judge response.")
    p.add_argument("--max-retries", type=int, default=2, help="Retries if response is invalid JSON.")

    p.add_argument("--log-every", type=int, default=10, help="Print progress every N samples.")
    p.add_argument("--num-workers", type=int, default=1, help="Threaded workers; each has its own client.")

    p.add_argument("--token-path", type=Path, default=Path.home() / "openai" / "token", help="Path to gateway token file.")
    p.add_argument("--env-path", type=Path, default=repo_root / ".env", help="Path to .env fallback file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def _load_pairs_and_tag() -> Tuple[List[PairRecord], str]:
        if args.pairs_json:
            return load_pairs_from_pairs_json(args.pairs_json), args.pairs_json.stem

        if not args.vanilla_log or not args.pruned_log:
            raise ValueError("Provide both --vanilla-log and --pruned-log.")
        pairs_local = load_pairs_from_logs(vanilla_log=args.vanilla_log, pruned_log=args.pruned_log)
        return pairs_local, f"{args.vanilla_log.stem}__VS__{args.pruned_log.stem}"

    pairs, input_tag = _load_pairs_and_tag()
    
    # If output_dir/model/pairing_tag already exists, check that:
    # (i) If there is any missing evaluation output (i.e., num_failed > 0), rerun only the failed samples instead of re-evaluating all samples.
    # (ii) If all samples are already successfully evaluated, skip evaluation and exit.
    existing_path = args.output_dir / args.model / input_tag
    is_rerun_for_failed_only = False
    existing_individual: Optional[List[Dict[str, Any]]] = None
    if existing_path.exists():
        aggregate_path = existing_path / "aggregate_results.json"
        individual_path = existing_path / "individual_results.json"
        if aggregate_path.exists() and individual_path.exists():
            try:
                aggregate = _load_json(aggregate_path)
                individual_any = _load_json(individual_path)

                if not isinstance(aggregate, dict):
                    raise ValueError("Existing aggregate_results.json must be a JSON object.")
                if not isinstance(individual_any, list):
                    raise ValueError("Existing individual_results.json must be a JSON list.")

                # Keep the loaded individual results for potential merging.
                existing_individual = list(individual_any)

                num_samples = int(aggregate.get("num_samples", 0) or 0)
                num_failed = int(aggregate.get("num_failed", 0) or 0)
                # Compare reported aggregate count against the number of input pairs.
                # If counts disagree, prefer running only the missing sample IDs from the current `pairs`.
                if num_samples and num_samples != len(pairs):
                    print(
                        "Warning: Existing aggregate count disagrees with current input pairs "
                        f"(aggregate num_samples={num_samples}, input pairs={len(pairs)})."
                    )
                    # Determine which sample_ids are already present in the existing individual results.
                    present_ids = {str(r.get("sample_id") or "").strip() for r in existing_individual}
                    present_ids = {pid for pid in present_ids if pid}

                    # Find which requested pairs are missing from existing results.
                    missing_ids = [p.sample_id for p in pairs if p.sample_id not in present_ids]
                    if missing_ids:
                        print(f"Found {len(missing_ids)} missing samples in existing results. Rerunning only missing samples.")
                        pairs = [p for p in pairs if p.sample_id in set(missing_ids)]
                        is_rerun_for_failed_only = True
                    else:
                        # If nothing is missing, nothing to do.
                        print("Existing output covers current input pairs. Skipping evaluation.")
                        return
                else:
                    if num_failed > 0:
                        print(f"Found existing results with {num_failed} failed samples. Rerunning only failed samples.")
                        failed_sample_ids = {
                            str(r.get("sample_id") or "").strip() for r in existing_individual if r.get("error")
                        }
                        failed_sample_ids = {sid for sid in failed_sample_ids if sid}

                        if not failed_sample_ids:
                            print("Existing aggregate indicates failures, but no failed sample_ids found. Rerunning full evaluation.")
                            existing_individual = None
                        else:
                            # Reload full pair list from the same inputs, then filter to failed ids.
                            all_pairs, _ = _load_pairs_and_tag()
                            pairs = [p for p in all_pairs if p.sample_id in failed_sample_ids]
                            if not pairs:
                                print("No matching pairs found for failed samples. Exiting.")
                                return
                            is_rerun_for_failed_only = True
                    else:
                        print(f"All {num_samples or len(existing_individual)} samples already successfully evaluated. Skipping evaluation.")
                        return
            except Exception as exc:
                print(f"Error loading existing results: {exc}. Proceeding with full evaluation.")
                # If we decided to rerun full evaluation, drop the cached individual so we don't merge.
                if not is_rerun_for_failed_only:
                    existing_individual = None

    # Optional filters.
    if args.source_filter:
        keep = set(str(s) for s in args.source_filter)
        pairs = [p for p in pairs if p.source in keep]
    if args.subset_filter:
        keep = set(str(s) for s in args.subset_filter)
        pairs = [p for p in pairs if p.subset in keep]

    # Optional sampling.
    if args.sample_size and args.sample_size > 0:
        if args.sample_size > len(pairs):
            raise ValueError(f"--sample-size={args.sample_size} exceeds available pairs={len(pairs)}")
        rng = random.Random(int(args.seed))
        pairs = rng.sample(pairs, k=int(args.sample_size))

    if not pairs:
        raise RuntimeError("No pairs loaded.")

    # When rerunning only failed samples, the pairs list will be a subset — skip the full-count check.
    if not is_rerun_for_failed_only and len(pairs) != 252:
        print(f"Warning: Loaded {len(pairs)} pairs, but expected 252. Check your filters and inputs.")
        return

    if is_rerun_for_failed_only:
        if len(pairs) + len(existing_individual or []) != 252:
            print(
                f"Warning: Rerun for failed samples only, but combined count of rerun pairs ({len(pairs)}) "
                f"and existing individual results ({len(existing_individual or [])}) does not equal 252."
            )
            return

    gateway_cfg = load_gateway_config(
        token_path=args.token_path,
        env_path=args.env_path,
        endpoint_kind=args.gateway_endpoint,  # type: ignore[arg-type]
    )

    out_root = args.output_dir / args.model / input_tag
    out_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_root / "aggregate_results.json"
    individual_path = out_root / "individual_results.json"

    results = evaluate_pairs(
        pairs=pairs,
        model_name=args.model,
        endpoint=gateway_cfg["endpoint"],
        endpoint_kind=gateway_cfg["endpoint_kind"],
        api_key=gateway_cfg["api_key"],
        api_versions=args.api_versions,
        seed=args.seed,
        allow_tie=bool(args.allow_tie),
        temperature=float(args.temperature),
        max_completion_tokens=int(args.max_completion_tokens),
        max_retries=int(args.max_retries),
        log_every=int(args.log_every),
        num_workers=int(args.num_workers),
        aws_region_name=str(args.aws_region),
    )

    if is_rerun_for_failed_only:
        print("Rerun for failed samples only. Merging new results with existing successful results.")
        if existing_individual is None:
            raise RuntimeError("Internal error: expected existing individual results for merge.")

        merged_individual = _merge_individual_results(existing_individual, results["individual"])
        merged_aggregate = _compute_aggregate_from_individual(
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
    print(f"Evaluated samples: {results['aggregate']['num_samples']} (ok={results['aggregate']['num_successful']}, failed={results['aggregate']['num_failed']})")


if __name__ == "__main__":
    main()
