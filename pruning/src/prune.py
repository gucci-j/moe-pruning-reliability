import argparse
import json
import os

import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
)

from utils import (
    EANCounter,
    EasyEP,
    FrequencyCounter,
    GatingCounter,
    REAPCounter,
    get_moe_info,
    get_moe_layers,
    patch_moe_forward,
)
from utils.calibration import (
    load_calibration_dataset,
    parse_dataset_types,
    run_calibration
)
import time

def _select_random_experts(*, moe_count: int, n_experts: int) -> list[list[int]]:
    k = n_experts // 2
    top_experts: list[list[int]] = []
    for _ in range(moe_count):
        indices = torch.randperm(n_experts)[:k].tolist()
        top_experts.append(indices)
    return top_experts


def _select_easyep_experts(*, model, tokenizer, moe_count: int, n_experts: int, device: str, args) -> list[list[int]]:
    print("EasyEP pruning selected.")
    easy_ep = EasyEP(moe_count, n_experts)
    print("Patching model to collect scores...")
    patch_moe_forward(model, easy_ep)

    dataset_types = parse_dataset_types(args.dataset_type)

    easy_ep.reset()
    for ds_type in dataset_types:
        print(f"Processing dataset: {ds_type}")
        print("Loading dataset...")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        print(f"Running calibration for {ds_type}...")
        run_calibration(
            model,
            tokenizer,
            dataset,
            ds_type=ds_type,
            device=device,
            n_samples=args.n_samples,
            no_chat_template_domain=False,
            verbose=args.verbose,
        )
    top_experts = easy_ep.get_top_experts(args.n_experts_to_keep)
    print(f"Selected top {args.n_experts_to_keep} experts for each layer.")
    return top_experts


def _select_frequency_experts(*, model, tokenizer, moe_count: int, n_experts: int, device: str, args) -> list[list[int]]:
    print("Frequency pruning selected.")
    freq = FrequencyCounter(moe_count, n_experts, device=device)
    print("Patching model...")
    patch_moe_forward(model, freq)

    dataset_types = parse_dataset_types(args.dataset_type)

    freq.reset()
    for ds_type in dataset_types:
        print(f"Processing dataset: {ds_type}")
        print("Loading dataset...")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        print(f"Running calibration for {ds_type}...")
        run_calibration(
            model,
            tokenizer,
            dataset,
            ds_type=ds_type,
            device=device,
            n_samples=args.n_samples,
            no_chat_template_domain=False,
            verbose=args.verbose,
        )

    top_experts = freq.get_top_experts(args.n_experts_to_keep)
    print(f"Selected top {args.n_experts_to_keep} experts for each layer.")
    return top_experts


def _select_gating_experts(*, model, tokenizer, moe_count: int, n_experts: int, device: str, args) -> list[list[int]]:
    print("Gating pruning selected.")
    gating = GatingCounter(moe_count, n_experts, device=device)
    print("Patching model...")
    patch_moe_forward(model, gating)

    dataset_types = parse_dataset_types(args.dataset_type)

    gating.reset()
    for ds_type in dataset_types:
        print(f"Processing dataset: {ds_type}")
        print("Loading dataset...")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        print(f"Running calibration for {ds_type}...")
        run_calibration(
            model,
            tokenizer,
            dataset,
            ds_type=ds_type,
            device=device,
            n_samples=args.n_samples,
            no_chat_template_domain=False,
            verbose=args.verbose,
        )

    top_experts = gating.get_top_experts(args.n_experts_to_keep)
    print(f"Selected top {args.n_experts_to_keep} experts for each layer.")
    return top_experts


def _select_ean_experts(*, model, tokenizer, moe_count: int, n_experts: int, device: str, args) -> list[list[int]]:
    print("EAN pruning selected.")
    ean = EANCounter(
        moe_count,
        n_experts,
        device=device,
        max_tokens_per_batch=256,
        token_sampling="first",
        seed=0,
    )
    print("Patching model...")
    patch_moe_forward(model, ean)

    dataset_types = parse_dataset_types(args.dataset_type)

    ean.reset()
    for ds_type in dataset_types:
        print(f"Processing dataset: {ds_type}")
        print("Loading dataset...")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        print(f"Running calibration for {ds_type}...")
        run_calibration(
            model,
            tokenizer,
            dataset,
            ds_type=ds_type,
            device=device,
            n_samples=args.n_samples,
            no_chat_template_domain=False,
            verbose=args.verbose,
        )

    top_experts = ean.get_top_experts(args.n_experts_to_keep)
    print(f"Selected top {args.n_experts_to_keep} experts for each layer.")
    return top_experts


def _select_reap_experts(*, model, tokenizer, moe_count: int, n_experts: int, device: str, args) -> list[list[int]]:
    print("REAP pruning selected.")
    reap = REAPCounter(moe_count, n_experts, device=device)
    print("Patching model...")
    patch_moe_forward(model, reap)

    dataset_types = parse_dataset_types(args.dataset_type)

    reap.reset()
    for ds_type in dataset_types:
        print(f"Processing dataset: {ds_type}")
        print("Loading dataset...")
        dataset = load_calibration_dataset(ds_type, args.n_samples, seed=args.data_seed)
        print(f"Running calibration for {ds_type}...")
        run_calibration(
            model,
            tokenizer,
            dataset,
            ds_type=ds_type,
            device=device,
            n_samples=args.n_samples,
            no_chat_template_domain=False,
            verbose=args.verbose,
        )

    top_experts = reap.get_top_experts(args.n_experts_to_keep)
    print(f"Selected top {args.n_experts_to_keep} experts for each layer.")
    return top_experts


def _is_fused_experts(experts_module):
    """Returns True if module uses GPT-OSS style fused 3D weight tensors rather than a ModuleList."""
    return (
        hasattr(experts_module, 'gate_up_proj') and
        isinstance(experts_module.gate_up_proj, torch.nn.Parameter) and
        experts_module.gate_up_proj.dim() == 3
    )


def _normalize_tied_weight_metadata_for_save(model) -> None:
    """Normalize tied-weight metadata across transformers/model version mismatches.

    Some model implementations expose `_tied_weights_keys` as a list/tuple while
    newer transformers internals expect a dict-like object with `.keys()`.
    """
    for _name, module in model.named_modules():
        for attr in ("_tied_weights_keys", "_dynamic_tied_weights_keys"):
            tied = getattr(module, attr, None)
            if isinstance(tied, (list, tuple, set)):
                setattr(module, attr, {str(k): str(k) for k in tied})


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


def _save_state_dict_without_pretrained_rewrite(
    *,
    model,
    state_dict: dict[str, torch.Tensor],
    save_path: str,
) -> str:
    """Save exact in-memory state_dict to disk without model.save_pretrained key rewriting."""
    weights_path = os.path.join(save_path, "model.safetensors")
    serializable_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # Clone to break shared storage so safetensors can persist reliably.
        serializable_state[key] = tensor.clone()

    save_safetensors_file(serializable_state, weights_path)
    model.config.save_pretrained(save_path)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(save_path)
    return weights_path


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #####
    # Load model and tokenizer
    #####
    print(f"Loading model from {args.model_name_or_path}...")
    feature_extractor = None
    processor = None
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        is_multimodal = _expects_visual_weights(config)

        if is_multimodal:
            print("Detected multimodal config. Loading conditional generation model to preserve visual weights...")
            model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                config=config,
                dtype=torch.bfloat16,
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
                print(f"Warning: could not load processor for multimodal model: {proc_err}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        print(f"🔍 CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"🔍 Current CUDA device: {torch.cuda.current_device()}")
            print(f"🔍 CUDA device name: {torch.cuda.get_device_name()}")
            print(
                f"🔍 CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )

        # Check model device placement
        print(f"🔍 Model class: {model.__class__.__name__}")
        print(f"🔍 Model config type: {model.config.model_type}")

        # Check device placement of first few layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            print(
                f"🔍 First layer device: {next(model.model.layers[0].parameters()).device}"
            )
            if len(model.model.layers) > 0 and hasattr(model.model.layers[0], "mlp"):
                mlp = model.model.layers[0].mlp
                if hasattr(mlp, "experts"):
                    print(f"🔍 Experts type: {type(mlp.experts)}")
                    print(f"🔍 Experts device: {next(mlp.experts.parameters()).device}")
                    
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
            print(
                f"🔍 First layer device: {next(model.backbone.layers[0].parameters()).device}"
            )
            # Check if model_type" is "nemotron_h"
            if getattr(model.config, "model_type", "") == "nemotron_h":
                print("🔍 Detected Nemotron-like architecture")
                if hasattr(model.backbone.layers[1], "mixer") and hasattr(model.backbone.layers[1].mixer, "experts"):
                    print(f"🔍 Experts type: {type(model.backbone.layers[1].mixer.experts)}")
                    print(f"🔍 Experts device: {next(model.backbone.layers[1].mixer.experts.parameters()).device}")
            else:
                raise ValueError("Unrecognized model architecture; cannot determine expert module structure.")
        
        print("🔍 Model successfully loaded!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    #####
    # Determine number of layers and experts
    #####
    moe_count, n_experts = get_moe_info(model)
    if moe_count == 0 or n_experts == 0:
        raise ValueError("No MoE layers or experts found in the model.")
    print(f"Detected {moe_count} MoE layers and {n_experts} experts.")
    
    # Sanity checks
    if args.pruning_method != "random":
        if not args.dataset_type:
            raise ValueError("--dataset_type must be provided for this pruning method")
    if args.n_experts_to_keep > n_experts:
        raise ValueError(
            f"n_experts_to_keep ({args.n_experts_to_keep}) cannot be greater than total experts ({n_experts})."
        )
    
    #####
    # Expert selection (one function per method)
    #####
    if args.pruning_method == "random":
        print("Random pruning selected. Pruning 50% of experts randomly.")
        args.n_experts_to_keep = n_experts // 2
        top_experts = _select_random_experts(moe_count=moe_count, n_experts=n_experts)
        print(f"Selected {args.n_experts_to_keep} random experts for each layer.")
    else:
        selectors = {
            "easyep": _select_easyep_experts,
            "frequency": _select_frequency_experts,
            "gating": _select_gating_experts,
            "ean": _select_ean_experts,
            "reap": _select_reap_experts,
        }
        if args.pruning_method not in selectors:
            raise ValueError("Invalid pruning method specified.")

        top_experts = selectors[args.pruning_method](
            model=model,
            tokenizer=tokenizer,
            moe_count=moe_count,
            n_experts=n_experts,
            device=device,
            args=args,
        )
    
    #####
    # Save mask (i.e., selected experts per layer)
    #####
    os.makedirs(args.save_path, exist_ok=True)
    mask_path = os.path.join(args.save_path, "expert_mask.json")
    with open(mask_path, 'w') as f:
        json.dump(top_experts, f)
    print(f"Expert mask saved to {mask_path}")
    
    #####
    # Prune model
    #####
    print(f"Pruning model with {args.pruning_type} method...")
    moe_layers = get_moe_layers(model)
    for layer_idx, (name, module) in enumerate(moe_layers):
        experts_to_keep = top_experts[layer_idx]
        experts_to_keep.sort()
        print(f"Pruning layer {layer_idx} ({name}): keeping {len(experts_to_keep)} experts")

        if args.pruning_type == "physical":
            # 1. Prune experts
            if hasattr(module, 'experts'):
                old_experts = module.experts
                if _is_fused_experts(old_experts):
                    # GPT-OSS style: fused 3D weight tensors — slice first dimension in-place
                    idx = torch.tensor(experts_to_keep, device=old_experts.gate_up_proj.device)
                    old_experts.gate_up_proj = torch.nn.Parameter(old_experts.gate_up_proj.data[idx])
                    old_experts.down_proj = torch.nn.Parameter(old_experts.down_proj.data[idx])
                    if hasattr(old_experts, 'gate_up_proj_bias') and old_experts.gate_up_proj_bias is not None:
                        old_experts.gate_up_proj_bias = torch.nn.Parameter(old_experts.gate_up_proj_bias.data[idx])
                    if hasattr(old_experts, 'down_proj_bias') and old_experts.down_proj_bias is not None:
                        old_experts.down_proj_bias = torch.nn.Parameter(old_experts.down_proj_bias.data[idx])
                    old_experts.num_experts = len(experts_to_keep)
                else:
                    new_experts = torch.nn.ModuleList([old_experts[i] for i in experts_to_keep])
                    module.experts = new_experts
            else:
                raise ValueError(f"Experts not found in module {name}.")

            # 2. Prune gate/router
            gate_module = None
            if hasattr(module, 'gate'):
                gate_module = module.gate
            elif hasattr(module, 'router'):
                gate_module = module.router

            if gate_module is not None and isinstance(gate_module, torch.nn.Linear):
                old_weight = gate_module.weight.data
                # Gate weights are (num_experts, hidden_dim)
                new_weight = old_weight[experts_to_keep, :]

                new_gate = torch.nn.Linear(
                    gate_module.in_features,
                    len(experts_to_keep),
                    bias=gate_module.bias is not None,
                    device=gate_module.weight.device,
                    dtype=gate_module.weight.dtype
                )
                new_gate.weight.data = new_weight
                if gate_module.bias is not None:
                    new_gate.bias.data = gate_module.bias.data[experts_to_keep]

                if hasattr(module, 'gate'):
                    module.gate = new_gate
                else:
                    module.router = new_gate
                    
            elif gate_module is not None and hasattr(gate_module, 'weight') and hasattr(gate_module, 'num_experts'):
                # GptOssTopKRouter style: direct nn.Parameter weight/bias (not an nn.Linear)
                gate_module.weight = torch.nn.Parameter(gate_module.weight.data[experts_to_keep])
                if hasattr(gate_module, 'bias') and gate_module.bias is not None:
                    gate_module.bias = torch.nn.Parameter(gate_module.bias.data[experts_to_keep])
                gate_module.num_experts = len(experts_to_keep)
            
            elif gate_module is not None and hasattr(gate_module, 'weight') and hasattr(gate_module, 'e_score_correction_bias'):
                # NemotronHTopkRouter style
                gate_module.n_routed_experts = len(experts_to_keep)
                gate_module.weight = torch.nn.Parameter(
                    gate_module.weight.data[experts_to_keep]
                )
                gate_module.e_score_correction_bias = torch.nn.Parameter(
                    gate_module.e_score_correction_bias.data[experts_to_keep]
                )
            
            else:
                raise ValueError(f"Gate/router not found or not a Linear layer in module {name}.")
    
    # Update config
    if args.pruning_type == "physical":
        if hasattr(model.config, 'num_local_experts'):
            model.config.num_local_experts = args.n_experts_to_keep
        elif hasattr(model.config, 'num_experts'):
            model.config.num_experts = args.n_experts_to_keep
        elif hasattr(model.config, 'n_routed_experts'):
            model.config.n_routed_experts = args.n_experts_to_keep
        elif hasattr(model.config.text_config, 'num_experts'):
            model.config.text_config.num_experts = args.n_experts_to_keep

    print(f"Saving pruned model to {args.save_path}...")
    if getattr(model.config, "model_type", "") == "nemotron_h":
        _normalize_tied_weight_metadata_for_save(model)
    if is_multimodal or model.config.model_type == "nemotron_h" or model.config.model_type == "gpt_oss":
        # For multimodal models, we avoid using model.save_pretrained to prevent unintended key rewrites that break visual weight loading.
        # For Nemotron-H, we also observed that save_pretrained omits backbone.embeddings, so we use the same custom save logic to ensure all weights and tied-weight metadata are preserved exactly.
        # For GPT-OSS models, we also observed that save_pretrained adds extra weird down_proj$ and gate_up_proj$ keys that should not exist in the original state_dict, so we use the same custom save logic to preserve the original key structure.
        state_dict_for_save = model.state_dict()
        saved_weights_path = _save_state_dict_without_pretrained_rewrite(
            model=model,
            state_dict=state_dict_for_save,
            save_path=args.save_path,
        )
        print(f"Model weights saved in {saved_weights_path}")
    else:
        model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    if processor is not None:
        processor.save_pretrained(args.save_path)
    elif feature_extractor is not None:
        feature_extractor.save_pretrained(args.save_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_type", type=str, default=None, help="Type of calibration dataset (e.g., medinst) for EasyEP. Comma-separated for multiple domains.")
    parser.add_argument("--n_samples", type=int, default=32, help="Number of calibration samples")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Random seed used when shuffling/sampling calibration and iterative validation datasets.",
    )
    parser.add_argument("--n_experts_to_keep", type=int, default=64, help="Number of experts to keep")
    parser.add_argument("--pruning_method", type=str, default="easyep", choices=["easyep", "frequency", "gating", "ean", "reap", "random"], help="Pruning method to use")
    parser.add_argument("--pruning_type", type=str, default="physical", choices=["physical"], help="Pruning type: physical (remove experts)")
    parser.add_argument("--save_path", type=str, default="pruned_model", help="Path to save the pruned model/mask")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample debug lines during calibration.",
    )
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
