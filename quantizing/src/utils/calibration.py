from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .format_chat_data import format_chat_with_tokenizer


def parse_dataset_types(dataset_type: str | None) -> list[str]:
    if not dataset_type:
        return []
    return [d.strip() for d in dataset_type.split(",") if d.strip()]


def load_calibration_dataset(
    ds_type: str,
    n_samples: int,
    *,
    seed: int = 42,
    buffer_size: int = 10_000,
):
    if ds_type == "medinst":
        full_train_ds = load_dataset(
            "aialt/MedINST32",
            name="unseen_train_100k",
        )["train"]
        dataset = full_train_ds.shuffle(seed=seed)
        dataset = dataset.select(range(min(len(dataset), n_samples)))
        del full_train_ds
        return dataset

    if ds_type == "general":
        full_train_ds = load_dataset(
            "allenai/Dolci-Instruct-SFT-No-Tools",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        full_train_ds = full_train_ds.shuffle(seed=seed, buffer_size=buffer_size)
        dataset = full_train_ds.take(n_samples)
        del full_train_ds
        return dataset

    raise ValueError(f"Please provide a valid dataset type for calibration. Unknown: {ds_type}")


def format_calibration_text(
    tokenizer: AutoTokenizer,
    sample,
    *,
    ds_type: str,
    no_chat_template_domain: bool,
) -> Optional[str]:
    try:
        use_template = True
        if no_chat_template_domain and ds_type != "general":
            use_template = False
        return format_chat_with_tokenizer(tokenizer, sample, use_template=use_template)
    except NotImplementedError:
        return None


@torch.no_grad()
def run_calibration(
    model,
    tokenizer: AutoTokenizer,
    dataset: Iterable,
    *,
    ds_type: str,
    device: str,
    n_samples: int,
    no_chat_template_domain: bool,
    verbose: bool,
    max_length: int = 2048,
):
    model.eval()
    for i, sample in enumerate(tqdm(dataset, total=n_samples, desc="Calibration")):
        if i >= n_samples:
            break

        text = format_calibration_text(
            tokenizer,
            sample,
            ds_type=ds_type,
            no_chat_template_domain=no_chat_template_domain,
        )
        if text is None:
            continue

        if verbose:
            print(f"Processing sample {i}: {text[:200]}...")

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        model(**inputs)

