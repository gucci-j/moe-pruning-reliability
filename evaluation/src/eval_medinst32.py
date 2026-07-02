# Original file downloaded from https://github.com/aialt/MedINST/blob/main/evaluation.py

import argparse
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm
import datasets
import numpy as np
from typing import List, Dict
import re
import string
from sklearn.metrics import f1_score
from collections import Counter
import evaluate as hf_evaluate

# Note: OpenAI is optional; only import if using API backend
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # will validate at runtime if used

# Optional: transformers/torch only when using HF backend
try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Optional: vLLM only when using vLLM backend
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

tests = {'QA': ['BioASQ-Task-B-yesno', 'PubMedQA-labeled', 'MedQA'],
         'TE': ['SciFact', 'ManConCorpus', 'CoVERt'],
         'NER': ['NCBI-disease',
                 'BC5CDR',
                 'AnEM',
                 'BioNLP-2009',
                 'BioNLP-2011-GE',
                 'BioNLP-2011-ID',
                 'BioNLP-2011-REL',
                 'BioNLP-2013-CG',
                 'BioNLP-2013-GE',
                 'BioNLP-2013-GRO',
                 'BioNLP-2013-PC',
                 'BioRED',
                 'tmVar-v3'],
         'TXTCLASS': ['Hallmarks-of-Cancer', 'MedDialog'],
         'NED': ['MeDAL', 'tmVar-v3-NED'],
         'RE': ['AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE'],
         'COREF': ['AnEM-COREF', 'MLEE-COREF'],
         'SUM': ['Multi-XScience'],
         'EE': ['MLEE-EE'],
         'STS': ['BIOSSES'],
         'TRANSL': ['ParaMed']}

bio = ['NCBI-disease', 'BC5CDR']
cls = ['BioASQ-Task-B-yesno', 'PubMedQA-labeled', 'SciFact', 'ManConCorpus', 'CoVERt', 'MedDialog']
entity = ['AnEM-COREF', 'MLEE-COREF', 'tmVar-v3-NED', 'AnEM', 'BioNLP-2009', 'BioNLP-2011-GE', 'BioNLP-2011-ID',
          'BioNLP-2011-REL', 'BioNLP-2013-CG',
          'BioNLP-2013-GE', 'BioNLP-2013-GRO', 'BioNLP-2013-PC', 'BioRED',
          'tmVar-v3', 'AnEM-RE', 'BC5CDR-RE', 'BioInfer-RE',
          'MLEE-EE'
          ]
em = ['MedQA', 'MeDAL']
mse = ['BIOSSES']
multicls = ['Hallmarks-of-Cancer']


def _parse_csv_selection(raw: str):
    values = [item.strip() for item in str(raw or '').split(',') if item.strip()]
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _load_existing_results(path: Path):
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f'Existing results file must contain a JSON object: {path}')
    return loaded


def _is_empty_prediction(prediction: object) -> bool:
    return not str(prediction or '').strip()


def _retry_empty_batch_predictions(
    predictor,
    examples,
    predictions,
    retry_empty: int,
    fail_on_empty: bool,
    flag: bool,
    task_name: str,
):
    if retry_empty <= 0:
        if fail_on_empty:
            remaining_empty = [i for i, pred in enumerate(predictions) if _is_empty_prediction(pred)]
            if remaining_empty:
                raise RuntimeError(
                    f'Empty predictions remain for {task_name}: count={len(remaining_empty)} '
                    f'sample_indices={remaining_empty[:10]}'
                )
        return predictions

    empty_indices = [i for i, pred in enumerate(predictions) if _is_empty_prediction(pred)]
    for attempt in range(1, retry_empty + 1):
        if not empty_indices:
            break
        print(
            f"Warning: {len(empty_indices)} empty predictions for {task_name}; "
            f"retrying (attempt {attempt}/{retry_empty})"
        )
        retry_examples = [examples[i] for i in empty_indices]
        retry_results = predictor.generate_batch(retry_examples, flag=flag)
        retry_predictions = [res[0] for res in retry_results]
        for idx, retry_pred in zip(empty_indices, retry_predictions):
            predictions[idx] = str(retry_pred).strip()
        empty_indices = [i for i in empty_indices if _is_empty_prediction(predictions[i])]

    if fail_on_empty:
        remaining_empty = [i for i, pred in enumerate(predictions) if _is_empty_prediction(pred)]
        if remaining_empty:
            raise RuntimeError(
                f'Empty predictions remain for {task_name} after retries: '
                f'count={len(remaining_empty)} sample_indices={remaining_empty[:10]}'
            )

    return predictions


def load_metric(metric_name: str, **kwargs):
    default_external = Path(__file__).resolve().parents[2] / "external"
    external_path = Path(os.getenv("EXTERNAL_PATH", str(default_external))).expanduser().resolve()
    metric_dir = external_path / "evaluate" / "metrics" / metric_name

    if metric_dir.exists():
        return hf_evaluate.load(str(metric_dir), **kwargs)
    return hf_evaluate.load(metric_name, **kwargs)


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

        # If final content is empty, it might be that the model didn't generate anything after "assistantfinal"
        if not final_content:
            final_content = analysis_content
            analysis_content = ""
    else:
        # If no "assistantfinal" found, treat the entire response as final content
        final_content = response.strip()
        analysis_content = ""
    
    return final_content


def build_messages(ex: Dict) -> List[Dict[str, str]]:
    """Build a chat-style message list from dataset example.

    - instruction -> system message (when available)
    - history -> interleaved user/assistant messages
    - input -> final user message
    """
    messages: List[Dict[str, str]] = []
    instruction = ex.get('instruction', '')
    if instruction:
        messages.append({"role": "system", "content": instruction})
    if 'history' in ex and ex['history']:
        for h in ex['history']:
            # Expecting (user, assistant)
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                messages.append({"role": "user", "content": str(h[0])})
                messages.append({"role": "assistant", "content": str(h[1])})
    input_text = ex.get('input', '')
    if input_text:
        messages.append({"role": "user", "content": input_text})
    return messages


class APIPredictor:
    """OpenAI-compatible API predictor (works for vLLM server or OpenAI).

    Expects an OpenAI-compatible endpoint at base_url.
    """

    def __init__(self, api_key: str, base_url: str, model: str, max_tokens: int = 1024, temperature: float = 0.0):
        if OpenAI is None:
            raise ImportError("openai package not available; install openai to use API/vLLM backend.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, ex: Dict, flag: bool = False):
        messages = build_messages(ex)

        try_times = 10
        while try_times > 0:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                response = completion.choices[0].message.content
                num_input = getattr(completion.usage, 'prompt_tokens', None)
                num_output = getattr(completion.usage, 'completion_tokens', None)
                if flag:
                    print(messages)
                    print(response)
                return response, (num_input or 0), (num_output or 0)
            except Exception as e:
                print(f"Waiting for the server... ({e})")
                time.sleep(30)
                try_times -= 1
        return "", 0, 0


class VLLMPredictor:
    """Offline vLLM predictor using the Python library directly."""

    def __init__(
        self, 
        model: str, 
        max_tokens: int = 1024,
        temperature: float = 0.0, 
        max_model_len: int = None,
        enforce_eager: bool = False,
        batch_size: int = 32,
        restart_every: int = -1,
    ):
        if LLM is None:
            raise ImportError("vllm package not available; install vllm to use this backend.")
        
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.batch_size = max(1, batch_size)
        self.restart_every = restart_every
        self.llm = None
        self._initialize_llm()

    def restart(self):
        """Public method to force a restart of the vLLM engine."""
        print("Reinitializing vLLM engine...")
        self._initialize_llm()

    def _initialize_llm(self):
        """Internal helper to (re)initialize the vLLM engine."""
        import gc
        if self.llm is not None:
            # Shutdown and cleanup to free GPU memory
            # vLLM doesn't have a formal 'shutdown', so we delete and gc
            try:
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except ImportError:
                pass
            del self.llm
            self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5) # Give it a moment to clear
            
        num_gpus = torch.cuda.device_count() if torch and torch.cuda.is_available() else 1
        self.llm = LLM(
            model=self.model_name, 
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=0.9,
            max_model_len=self.max_model_len
        )
        self.sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name_or_path = self.model_name

    def generate(self, ex: Dict, flag: bool = False):
        # Build prompt
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            enable_thinking = False
            chat_template_args = None
            if "gpt-oss-20b" in self.model_name_or_path:
                enable_thinking = True
                chat_template_args = {"reasoning_effort": "low"}
            messages = build_messages(ex)
            # Note: enable_thinking is specific to some tokenizers, removing if it causes issues
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=enable_thinking, 
                    **(chat_template_args or {})
                )
            except TypeError:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_text = (ex.get('instruction', '') + '\n' + ex.get('input', '')).strip()

        # Generate
        outputs = self.llm.generate([prompt_text], self.sampling_params, use_tqdm=False)
        output = outputs[0]
        if "gpt-oss-20b" in self.model_name_or_path:
            generated_text = output.outputs[0].text.split("<|message|>")[-1].strip()
            if generated_text != "":
                generated_text = split_response(generated_text)
            else:
                generated_text = split_response(output.outputs[0].text)
        else:
            generated_text = output.outputs[0].text
        
        num_input = len(output.prompt_token_ids)
        num_output = len(output.outputs[0].token_ids)

        if flag:
            print({'prompt': prompt_text})
            print(generated_text)
        return generated_text, num_input, num_output

    def generate_batch(self, examples: List[Dict], flag: bool = False):
        prompts = []
        is_gpt_oss = "gpt-oss-20b" in self.model_name_or_path
        
        for ex in examples:
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                enable_thinking = False
                chat_template_args = None
                if is_gpt_oss:
                    if flag and not prompts:
                        print("[chat_template] Detected gpt-oss tokenizer; enabling thinking and generation prompts for better alignment.")
                    enable_thinking = True
                    chat_template_args = {"reasoning_effort": "low"}
                
                messages = build_messages(ex)
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True, 
                        enable_thinking=enable_thinking, 
                        **(chat_template_args or {})
                    )
                except TypeError:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt_text = (ex.get('instruction', '') + '\n' + ex.get('input', '')).strip()
            prompts.append(prompt_text)

        if flag and len(prompts) > 0:
            print({'prompt_example': prompts[0]})

        # Run in chunks to avoid large shared-memory pressure from one giant request.
        prompt_chunks = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]

        outputs = []
        for i, chunk in enumerate(tqdm(prompt_chunks, desc='vLLM chunks')):
            # Handle model-specific max length constraints
            if "Phi-3.5-MoE-instruct" in self.model_name_or_path:
                max_model_len = 4096
                prompt_token_ids = [self.tokenizer.encode(p) for p in chunk]
                truncated_token_ids = []
                for ids in prompt_token_ids:
                    if len(ids) > max_model_len:
                        # Truncate from the left to keep the most recent context
                        # Leave some space for generation
                        truncated_token_ids.append(ids[-(max_model_len - 512):])
                    else:
                        truncated_token_ids.append(ids)
                # Decode back to text for generation
                truncated_prompts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in truncated_token_ids]
                batch_outputs = self.llm.generate(truncated_prompts, sampling_params=self.sampling_params, use_tqdm=False)
            else:
                # Regular generation
                batch_outputs = self.llm.generate(chunk, sampling_params=self.sampling_params, use_tqdm=False)
            outputs.extend(batch_outputs)
        
        results = []
        for i, output in enumerate(outputs):
            if is_gpt_oss:
                # Restoration: Split gpt-oss-20b output to get actual content
                raw_text = output.outputs[0].text
                generated_text = raw_text.split("<|message|>")[-1].strip()
                if generated_text != "":
                    generated_text = split_response(generated_text)
                else:
                    generated_text = split_response(raw_text)
            else:
                generated_text = output.outputs[0].text
            
            num_input = len(output.prompt_token_ids)
            num_output = len(output.outputs[0].token_ids)
            results.append((generated_text, num_input, num_output))
            
            if flag and i == 0:
                print(generated_text)

        return results


class HFPredictor:
    """Local Hugging Face transformers predictor."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda:0',
        dtype: str = 'bfloat16',
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        use_chat_template: bool = True,
    ):
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            raise ImportError("transformers/torch not available; install to use HF backend.")

        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            # set pad token to eos if missing
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer device_map='auto' to utilize multiple GPUs if available
        # When a specific device like 'cuda:0' is provided, map to it explicitly
        device_map = 'auto' if device == 'auto' else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device != 'auto' and hasattr(self.model, 'to'):
            self.model.to(device)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = temperature is not None and temperature > 0.0
        self.use_chat_template = use_chat_template
        self.model_name_or_path = model_name_or_path

    def generate(self, ex: Dict, flag: bool = False):
        # Prefer chat template if available for instruction-tuned models
        if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = build_messages(ex)
            enable_thinking = False
            chat_template_args = None
            if "gpt-oss-20b" in self.model_name_or_path:
                enable_thinking = True
                chat_template_args = {"enable_thinking": True}
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **(chat_template_args or {})
            )
        else:
            # Fallback: simple instruction + input concatenation
            prompt_text = (ex.get('instruction', '') + '\n' + ex.get('input', '')).strip()

        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.model.device)
        input_len = inputs['input_ids'].shape[-1]
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "gpt-oss-20b" in self.model_name_or_path:
            text = text.split("<|message|>")[-1].strip()
            if text != "":
                text = split_response(text)
            else:
                text = split_response(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        if flag:
            print({'prompt': prompt_text})
            print(text)
        return text, input_len, gen_ids.shape[-1]


def mse_score(targets, preds):
    def extract_integers_from_string(s):
        integers = re.findall(r'\d+', s)
        integers = [int(num) for num in integers]
        return list(set(integers))

    ts = []
    ps = []
    for t, p in zip(targets, preds):
        t_numbers = extract_integers_from_string(t)
        p_numbers = extract_integers_from_string(p)
        if len(t_numbers) != 1 or len(p_numbers) != 1:
            t_num = 0
            p_num = 5
        elif t_numbers[0] not in [0, 1, 2, 3, 4, 5] or p_numbers[0] not in [0, 1, 2, 3, 4, 5]:
            t_num = 0
            p_num = 5
        else:
            t_num = t_numbers[0]
            p_num = p_numbers[0]
        ts.append(t_num)
        ps.append(p_num)
    n = len(ts)
    # print(ts)
    # print(ps)
    mse = sum((x - y) ** 2 for x, y in zip(ts, ps)) / n
    return mse


def post_bio(target, pred):
    def extract_tags(input_string):
        pattern = r'\[B\]|\[I\]|\[O\]'
        matches = re.findall(pattern, input_string)
        return matches

    perd_labels = extract_tags(pred)
    target_labels = extract_tags(target)
    return target_labels, perd_labels


def post_entity(target, pred):
    def extract_entities_with_stack(s):
        stack = []
        entities = []
        current_entity = []
        for char in s:
            if char == '[':
                if stack:
                    current_entity.append(char)
                stack.append(char)
            elif char == ']':
                if not stack:
                    current_entity = []
                    continue
                stack.pop()
                if stack:
                    current_entity.append(char)
                else:
                    entities.append(normalize_answer(''.join(current_entity)))
                    current_entity = []
            elif stack:
                current_entity.append(char)
        return entities

    target_entities = extract_entities_with_stack(target)
    pred_entities = extract_entities_with_stack(pred)
    return target_entities, pred_entities


def label_level_f1(targets, preds):
    macro_f1 = f1_score(targets, preds, labels=sorted(set(targets)), average='macro')
    return macro_f1


def entity_level_f1(target, pred):
    
    true_counter = Counter(target)
    pred_counter = Counter(pred)

   
    tp = sum((true_counter & pred_counter).values())

   
    fp = sum((pred_counter - true_counter).values())

    
    fn = sum((true_counter - pred_counter).values())


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return score


def rouge(prediction: str, ground_truth: str, rouge_metric):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_aggregator': False, 'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0]


def word_level_f1(prediction: str, ground_truth: str):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(targets: List[str], predictions: List[str], evaluation_types: List[str], rouge_metric) -> Dict:
    assert len(predictions) == len(targets), \
        f"The pred file does not have the same length as the gold data: {len(targets)} vs {len(predictions)}"

    metrics = {}

    for idx, (gold, pred) in datasets.tqdm(enumerate(zip(targets, predictions))):

        if 'rouge' not in metrics:
            metrics['rouge'] = 0
        metrics['rouge'] += rouge(pred, gold, rouge_metric)

        if 'f1' not in metrics:
            metrics['f1'] = 0
        metrics['f1'] += word_level_f1(pred, gold)

        if 'entity' in evaluation_types:
            ts, ps = post_entity(gold, pred)
            if 'entity_level_f1' not in metrics:
                metrics['entity_level_f1'] = 0
            metrics['entity_level_f1'] += entity_level_f1(ts, ps)

        if 'multicls' in evaluation_types:
            ts = gold.split(', ')
            ps = pred.split(', ')
            ts = [t.lower().strip() for t in ts]
            ps = [p.lower().strip() for p in ps]
            if 'entity_level_f1' not in metrics:
                metrics['entity_level_f1'] = 0
            metrics['entity_level_f1'] += entity_level_f1(ts, ps)

        if 'em' in evaluation_types:
            if 'em' not in metrics:
                metrics['em'] = 0
            metrics['em'] += exact_match_score(pred, gold)

    # e.g., selecting A, B, C, etc.
    # normalize tne metrics
    for key in metrics.keys():
        metrics[key] /= len(predictions)

    if 'bio' in evaluation_types:
        ts = []
        ps = []
        for t, p in zip(targets, predictions):
            post_t, post_p = post_bio(t, p)
            if len(post_t) > len(post_p):
                post_p = post_p + ['N' for _ in range(len(post_t) - len(post_p))]
            else:
                post_p = post_p[:len(post_t)]
            ts.extend(post_t)
            ps.extend(post_p)
        metrics['label_leval_f1'] = label_level_f1(ts, ps)

    if "cls" in evaluation_types:
        ts = [normalize_answer(t) for t in targets]
        ps = [normalize_answer(p) for p in predictions]
        metrics['label_level_f1'] = label_level_f1(ts, ps)

    if 'mse' in evaluation_types:
        metrics['mse'] = mse_score(targets, predictions)

    return metrics


def predict(ex, predictor, flag=False):
    """Unified predict function over different backends.

    predictor must implement .generate(ex: Dict, flag: bool) -> (text, prompt_tokens, completion_tokens)
    """
    return predictor.generate(ex, flag=flag)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MedINST32 benchmark with various backends.')
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument('--zero', action='store_true')
    # Backend selection
    parser.add_argument('--backend', type=str, choices=['vllm', 'hf', 'openai', 'api'], default='hf',
                        help='Backend to use: vllm/openai/api for OpenAI-compatible server; hf for local transformers')
    # API/vLLM settings
    parser.add_argument('--key', type=str, default='', help='API key for OpenAI-compatible server (vLLM/OpenAI)')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1',
                        help='Base URL for OpenAI-compatible API (use your vLLM server URL here)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name for API backend or default for HF')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_model_len', type=int, default=-1, help='Max model context length (for vLLM backend)')
    parser.add_argument('--vllm_batch_size', type=int, default=32,
                        help='Number of prompts per vLLM generate() call')
    parser.add_argument('--vllm_restart_every', type=int, default=-1,
                        help='Restart vLLM engine every N chunks to clear memory/stalls (e.g., 10)')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--enforce_eager', action='store_true', help='Disable CUDA graphs (useful for ROCm/AMD GPUs)')
    # HF settings
    parser.add_argument('--hf_model', type=str, default=None, help='HuggingFace model id or local path')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device for HF backend: e.g., 'cuda:0', 'cpu', 'auto'")
    parser.add_argument('--dtype', type=str, choices=['float16', 'bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--no_chat_template', action='store_true',
                        help='Disable chat template usage for HF backend (defaults to enabled if available)')
    parser.add_argument(
        '--categories',
        type=str,
        default='',
        help='Comma-separated MedINST categories to run. Empty means all categories.',
    )
    parser.add_argument(
        '--sum-retry-empty',
        type=int,
        default=10000,
        help='Retry empty predictions for SUM/Multi-XScience after generation.',
    )
    parser.add_argument(
        '--sum-fail-on-empty',
        action='store_true',
        help='Raise if empty predictions remain for SUM/Multi-XScience after retries.',
    )
    args = parser.parse_args()
    name = args.name
    dir = args.dir
    zero = args.zero
    backend = args.backend
    model = args.model
    selected_categories = _parse_csv_selection(args.categories) or list(tests.keys())
    selected_categories = [category for category in selected_categories if category in tests]
    if not selected_categories:
        raise ValueError('No valid categories selected.')

    # Build predictor based on backend
    if backend in ['openai', 'api']:
        predictor = APIPredictor(api_key=args.key, base_url=args.base_url, model=model,
                                 max_tokens=args.max_new_tokens, temperature=args.temperature)
    elif backend == 'vllm':
        predictor = VLLMPredictor(
            model=model, 
            max_tokens=args.max_new_tokens, 
            temperature=args.temperature, 
            enforce_eager=args.enforce_eager, 
            max_model_len=args.max_model_len if args.max_model_len != -1 else None,
            batch_size=args.vllm_batch_size,
            restart_every=args.vllm_restart_every
        )
    elif backend == 'hf':
        hf_model = args.hf_model or model
        predictor = HFPredictor(model_name_or_path=hf_model, device=args.device, dtype=args.dtype,
                                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                use_chat_template=(not args.no_chat_template))
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Prepare ROUGE metric
    try:
        rouge_metric = load_metric('rouge', experiment_id=str(random.randint(1, 888888)))
    except Exception as e:
        raise RuntimeError("ROUGE metric is required for evaluation. Please ensure it is available locally or can be downloaded from Hugging Face hub. If you have a local copy, set the EXTERNAL_PATH environment variable to its parent `external` directory of `moe-pruning`.")

    # Evaluation
    os.makedirs(dir, exist_ok=True)
    output_path = Path(dir) / f'{name}.json'
    preserve_existing = set(selected_categories) != set(tests.keys())
    results = _load_existing_results(output_path) if preserve_existing else {}
    first_test = True
    for cat, test_names in tests.items():
        if cat not in selected_categories:
            continue
        results.setdefault(cat, {})
        for test in test_names:
            # Per-task restart for vLLM to prevent stalls/memory leaks
            if backend == 'vllm' and not first_test:
                predictor.restart()
            first_test = False

            flag = True
            print(f'------------------------ {cat}: {test} ------------------------')
            results[cat][test] = {}
            config = '-zs' if zero else ''
            data = datasets.load_dataset('LiinXemmon/MedINST32', test+config)['test']
            targets = []
            predictions = []
            if hasattr(predictor, 'generate_batch'):
                targets = [d['output'] for d in data]
                batch_results = predictor.generate_batch(list(data), flag=flag)
                predictions = [res[0] for res in batch_results]
                if cat == 'SUM' and test == 'Multi-XScience':
                    predictions = _retry_empty_batch_predictions(
                        predictor=predictor,
                        examples=list(data),
                        predictions=predictions,
                        retry_empty=args.sum_retry_empty,
                        fail_on_empty=args.sum_fail_on_empty,
                        flag=False,
                        task_name=f'{cat}:{test}',
                    )
            else:
                tqdm_data = tqdm(data)
                for d in tqdm_data:
                    targets.append(d['output'])
                    if not flag:
                        flag = random.randint(1, 50) == 1
                    pre, len_prompt, len_gen = predict(d, predictor, flag=flag)
                    predictions.append(pre)
                    tqdm_data.set_description(f'Inp: {len_prompt} Gen: {len_gen}')
                    flag = False
            results[cat][test]['generated'] = [{'prediction': pre, 'target': target} for pre, target in
                                               zip(predictions, targets)]
            types = []
            if test in cls:
                types.append('cls')
            elif test in em:
                types.append('em')
            elif test in entity:
                types.append('entity')
            elif test in multicls:
                types.append('multicls')
            elif test in bio:
                types.append('bio')
            elif test in mse:
                types.append('mse')

            results[cat][test]['metrics'] = evaluate(targets, predictions, types, rouge_metric)
            print(f"{test}: ", results[cat][test]['metrics'])

    with output_path.open('w', encoding='utf-8') as out:
        json.dump(results, out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
