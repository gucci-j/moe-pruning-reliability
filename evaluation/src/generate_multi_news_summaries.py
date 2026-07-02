from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import re
import inspect
import importlib
from typing import Any, Dict, List, Sequence, Optional


def tqdm(iterable, **kwargs):  # type: ignore
    return iterable


try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


@dataclass
class MultiNewsExample:
    idx: int
    source: str
    reference: str


def split_response(response: str) -> str:
    # Find the analysis part (everything before "assistantfinal")
    analysis_match = re.search(r'^(.*?)assistantfinal', response, flags=re.DOTALL)

    if analysis_match:
        # Find the final content (everything after "assistantfinal")
        final_content = re.sub(r'^.*?assistantfinal\s*', '', response, flags=re.DOTALL).strip()
        if not final_content:
            final_content = response.strip()
    else:
        final_content = response.strip()

    return final_content


def _postprocess_gpt_oss_prediction(prediction: str) -> str:
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
            max_model_len=max_model_len,
        )
        self.tokenizer = self.llm.get_tokenizer()
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
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **(chat_template_args or {})
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
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking, **(chat_template_args or {})
                ))
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


def _is_empty(text: object) -> bool:
    return not str(text or "").strip()


def load_multi_news_examples(cleansed_dir: Path, split: str, max_examples: int, max_source_chars: int) -> List[MultiNewsExample]:
    files = sorted(cleansed_dir.glob("cleansed_data_*.json"))
    if not files:
        raise FileNotFoundError(f"No cleansed_data_*.json files found in {cleansed_dir}")

    examples: List[MultiNewsExample] = []
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
            cleansed = item.get("cleansed_document", "")
            if isinstance(cleansed, list):
                source = "\n\n".join([str(s).strip() for s in cleansed if str(s).strip()])
            else:
                source = str(cleansed or "").strip()
            if max_source_chars > 0 and len(source) > max_source_chars:
                source = source[:max_source_chars]

            reference = str(item.get("summary", "") or "").strip()
            examples.append(MultiNewsExample(idx=item.get("idx", len(examples)), source=source, reference=reference))
            if max_examples > 0 and len(examples) >= max_examples:
                return examples

    return examples


def build_prompt(source: str) -> str:
    return (
        "You are a news summarization assistant. "
        "Given documents from multiple news sources about the same event, write a concise factual summary in 2-4 sentences. "
        "Do not include information that is not supported by the source.\n\n"
        "Source documents:\n"
        f"{source}\n\n"
        "Summary:"
    )


def _generate_with_optional_sleep(predictor: Any, prompts: Sequence[str], backend: str, sleep_seconds: float) -> List[str]:
    if backend == "api" and sleep_seconds > 0:
        outputs: List[str] = []
        for prompt in prompts:
            outputs.extend(predictor.generate_batch([prompt]))
            time.sleep(sleep_seconds)
        return outputs
    return predictor.generate_batch(prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MultiNews summaries from cleansed_data using model backends")
    parser.add_argument("--cleansed-dir", type=str, default=str(Path(__file__).resolve().parents[2] / "external" / "multi_news_plus" / "cleansing" / "cleansed_data"), help="Path to cleansed_data JSON files.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Split to use.")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit number of examples. 0 means all.")
    parser.add_argument("--max-source-chars", type=int, default=0, help="Truncate source text at this char length. 0 means no truncation.")

    # Reuse generation args pattern from RCT script
    parser.add_argument("--backend", type=str, default="vllm", choices=["hf", "vllm", "api"], help="Generation backend.")
    parser.add_argument("--model", type=str, required=True, help="Model id or local model path.")
    parser.add_argument("--min-new-tokens", type=int, default=16, help="Minimum generated tokens (best-effort).")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling.")
    parser.add_argument("--max-model-len", type=int, default=-1, help="Max model context length (for vLLM). -1 means default.")
    parser.add_argument("--device", type=str, default="auto", help="HF device (auto, cuda, cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="HF torch dtype.")
    parser.add_argument("--api-key", type=str, default="", help="API key for backend=api.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--enforce-eager", action="store_true", help="Enable vLLM eager mode.")
    parser.add_argument("--output-json", type=str, default="debug/multi_news_predictions.json", help="Path to output JSON list.")
    parser.add_argument("--predictions-file", type=str, default="debug/multi_news_predictions.txt", help="Path to write line-aligned predictions.")
    parser.add_argument("--references-file", type=str, default="debug/multi_news_references.txt", help="Path to write line-aligned references.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between prompts for API rate limiting.")
    parser.add_argument("--retry-empty", type=int, default=10000, help="Retry generation for empty summaries.")
    parser.add_argument("--fail-on-empty", action="store_true", help="Raise an error if empty summaries remain after retries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cleansed_dir = Path(args.cleansed_dir)
    examples = load_multi_news_examples(cleansed_dir=cleansed_dir, split=args.split, max_examples=args.max_examples, max_source_chars=args.max_source_chars)
    if not examples:
        raise ValueError("No usable MultiNews examples found.")

    prompts = [build_prompt(ex.source) for ex in examples]

    # Build predictor using local builder
    predictor = build_predictor(args)

    existing_by_example: Dict[str, Dict[str, Any]] = {}
    output_json = Path(args.output_json)
    if output_json.exists():
        try:
            existing_rows = json.loads(output_json.read_text(encoding="utf-8"))
            if isinstance(existing_rows, list):
                for row in existing_rows:
                    if not isinstance(row, dict):
                        continue
                    key = str(row.get("example_idx") or "").strip()
                    if key:
                        existing_by_example[key] = row
        except Exception as exc:
            print(f"Warning: failed to load existing output JSON: {exc}. Regenerating all.")
            existing_by_example = {}

    predictions: List[str] = [""] * len(examples)
    to_generate: List[int] = []
    for idx, example in enumerate(examples):
        existing = existing_by_example.get(str(example.idx))
        existing_pred = str(existing.get("prediction") or "").strip() if isinstance(existing, dict) else ""
        if existing_pred and not _is_empty(existing_pred):
            predictions[idx] = existing_pred
        else:
            to_generate.append(idx)

    if to_generate:
        gen_prompts = [prompts[i] for i in to_generate]
        gen_outputs = _generate_with_optional_sleep(
            predictor=predictor,
            prompts=gen_prompts,
            backend=args.backend,
            sleep_seconds=args.sleep_seconds,
        )

        if "gpt-oss-20b" in args.model:
            gen_outputs = [_postprocess_gpt_oss_prediction(pred) for pred in gen_outputs]

        for idx, pred in zip(to_generate, gen_outputs):
            predictions[idx] = str(pred).strip()

    # Retry empty outputs
    if args.retry_empty > 0:
        empty_indices = [i for i, pred in enumerate(predictions) if _is_empty(pred)]
        for attempt in range(1, args.retry_empty + 1):
            if not empty_indices:
                break
            print(f"Warning: {len(empty_indices)} empty summaries; retrying (attempt {attempt}/{args.retry_empty})")
            retry_prompts = [prompts[i] for i in empty_indices]
            retry_outputs = _generate_with_optional_sleep(predictor=predictor, prompts=retry_prompts, backend=args.backend, sleep_seconds=args.sleep_seconds)
            if "gpt-oss-20b" in args.model:
                retry_outputs = [_postprocess_gpt_oss_prediction(pred) for pred in retry_outputs]
            for idx, retry_pred in zip(empty_indices, retry_outputs):
                predictions[idx] = str(retry_pred).strip()
            empty_indices = [i for i in empty_indices if _is_empty(predictions[i])]

    if args.fail_on_empty:
        remaining_empty = [i for i, pred in enumerate(predictions) if _is_empty(pred)]
        if remaining_empty:
            sample_ids = [examples[i].idx for i in remaining_empty[:10]]
            raise RuntimeError(
                "Empty summaries remain after retries. "
                f"count={len(remaining_empty)} sample_ids={sample_ids}"
            )

    if len(predictions) != len(examples):
        raise RuntimeError(
            "Generation count mismatch: "
            f"got {len(predictions)} predictions for {len(examples)} examples"
        )

    rows: List[Dict[str, Any]] = []
    for idx, (ex, pred, prompt) in enumerate(zip(examples, predictions, prompts)):
        rows.append({
            "id": idx,
            "example_idx": ex.idx,
            "prediction": str(pred).strip(),
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    print(f"Loaded {len(examples)} examples from split={args.split} at {cleansed_dir}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
