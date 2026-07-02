import argparse
import sys
import os
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)

# We import the local utilities from the copied utils folder
from utils.calibration import (
    load_calibration_dataset,
    parse_dataset_types,
    format_calibration_text,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.utils import load_context


def _expects_visual_weights(config) -> bool:
    """Return True when config indicates multimodal visual parameters are expected."""
    archs = getattr(config, "architectures", []) or []
    has_conditional_arch = any("ConditionalGeneration" in arch for arch in archs)
    has_vision_cfg = getattr(config, "vision_config", None) is not None
    text_cfg = getattr(config, "text_config", None)
    text_has_vision_cfg = False
    if text_cfg is not None:
        text_has_vision_cfg = getattr(text_cfg, "vision_config", None) is not None
    return has_conditional_arch or has_vision_cfg or text_has_vision_cfg


def _build_quantization_ignore_list(is_multimodal: bool) -> list[str]:
    """Return the layer patterns that should remain in higher precision."""
    ignore = ["lm_head", "re:.*gate$", "re:.*router$"]
    if is_multimodal:
        # Keep the visual tower in full precision; its MLP projections are not
        # always divisible by the GPTQ group size used here.
        ignore.append("re:^model\\.visual\\.")
    return ignore


def main(args):
    print(f"Loading config from {args.model_name_or_path}...")
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    is_multimodal = _expects_visual_weights(config)

    print(f"Loading model and tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    processor = None
    feature_extractor = None

    # Load model inside load_context to ensure proper MoE layer handling
    with load_context():
        if is_multimodal:
            print("Detected multimodal config. Loading conditional generation model to preserve visual weights...")
            # We import Qwen3_5MoeForConditionalGeneration dynamically if available/required,
            # otherwise fall back to AutoModelForCausalLM/AutoModelForVision2Seq
            try:
                from transformers import Qwen3_5MoeForConditionalGeneration
                model_cls = Qwen3_5MoeForConditionalGeneration
            except ImportError:
                print("Warning: Qwen3_5MoeForConditionalGeneration not found in transformers, falling back to AutoModelForCausalLM")
                model_cls = AutoModelForCausalLM

            model = model_cls.from_pretrained(
                args.model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            try:
                processor = AutoProcessor.from_pretrained(
                    args.model_name_or_path,
                    trust_remote_code=True,
                )
                feature_extractor = getattr(processor, "image_processor", None)
            except Exception as proc_err:
                print(f"Warning: could not load processor: {proc_err}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

    # Choose Quantization Recipe
    ignore_patterns = _build_quantization_ignore_list(is_multimodal)
    if args.quant_type == "gptq":
        print("Configuring GPTQ (W4A16) modifier...")
        recipe = GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=ignore_patterns,
        )
    elif args.quant_type == "fp8":
        print("Configuring FP8 modifier...")
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=ignore_patterns,
        )
    else:
        raise ValueError(f"Invalid quant_type: {args.quant_type}")

    # Load and preprocess dataset
    dataset_types = parse_dataset_types(args.dataset_type)
    if not dataset_types:
        raise ValueError("Please provide a valid calibration --dataset_type (e.g., medinst)")

    all_texts = []
    for ds_type in dataset_types:
        print(f"Loading calibration dataset: {ds_type}")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        
        count = 0
        for sample in dataset:
            if count >= args.n_samples:
                break
            text = format_calibration_text(
                tokenizer,
                sample,
                ds_type=ds_type,
                no_chat_template_domain=args.no_chat_template_domain,
            )
            if text is not None:
                all_texts.append(text)
                count += 1

    print(f"Collected {len(all_texts)} total samples for calibration.")
    if not all_texts:
        raise ValueError("No calibration samples were successfully loaded and formatted.")

    ds = Dataset.from_dict({"text": all_texts})

    # Tokenize the processed texts
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Configure oneshot kwargs
    oneshot_kwargs = {
        "model": model,
        "dataset": ds,
        "recipe": recipe,
        "max_seq_length": args.max_seq_length,
        "num_calibration_samples": len(ds),
    }

    print("Starting quantization oneshot (this can take some time)...")
    oneshot(**oneshot_kwargs)

    # Save the compressed artifact
    print(f"Saving quantized model to {args.save_path}...")
    model.save_pretrained(args.save_path, save_compressed=True)
    tokenizer.save_pretrained(args.save_path)
    if processor is not None:
        processor.save_pretrained(args.save_path)
    elif feature_extractor is not None:
        feature_extractor.save_pretrained(args.save_path)

    print("Quantization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to FP8 or GPTQ (4-bit) using llm-compressor.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the unquantized or pruned model")
    parser.add_argument("--quant_type", type=str, required=True, choices=["gptq", "fp8"], help="Type of quantization to apply")
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of calibration dataset (e.g., medinst), comma-separated for multiple domains")
    parser.add_argument("--n_samples", type=int, default=128, help="Number of calibration samples per dataset type")
    parser.add_argument("--data_seed", type=int, default=42, help="Random seed for shuffling/sampling calibration datasets")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for tokenization")
    parser.add_argument("--no_chat_template_domain", action="store_true", help="Do not use chat templates for domain calibration data")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map configuration (e.g. auto, cpu)")
    parser.add_argument("--sequential_targets", type=str, default=None, help="Comma-separated target layers for sequential processing (e.g. GroupAttention,MoEBlocks) to reduce memory spike")
    parser.add_argument("--fp8_block_n", type=int, default=None, help="FP8 block size (n) to use for FP8_BLOCK scheme; leave empty to use default")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the compressed model")

    args = parser.parse_args()
    main(args)
