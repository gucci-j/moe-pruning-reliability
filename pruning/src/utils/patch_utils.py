import torch
import torch.nn.functional as F
from torch import nn
from .model_utils import get_moe_layers


def _linear_any_layout(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """Apply linear projection when weight may be [in, out] or [out, in]."""
    in_dim = x.shape[-1]
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(w.shape)}")

    if w.shape[0] == in_dim:
        y = x @ w
    elif w.shape[1] == in_dim:
        y = x @ w.transpose(0, 1)
    else:
        raise ValueError(
            f"Incompatible linear shapes: x={tuple(x.shape)}, w={tuple(w.shape)}"
        )

    if b is not None:
        y = y + b
    return y


def _run_fused_expert(experts_module, expert_id: int, x: torch.Tensor) -> torch.Tensor:
    """Run one fused expert across architectures (GPT-OSS/Qwen fused layouts)."""
    gate_up_w = experts_module.gate_up_proj[expert_id]
    gate_up_b = experts_module.gate_up_proj_bias[expert_id] if hasattr(experts_module, "gate_up_proj_bias") else None
    gate_up = _linear_any_layout(x, gate_up_w, gate_up_b)

    gate, up = gate_up.chunk(2, dim=-1)

    # GPT-OSS uses custom bounded GLU; most others use SwiGLU.
    if hasattr(experts_module, "alpha") and hasattr(experts_module, "limit"):
        gate = gate.clamp(max=experts_module.limit)
        up = up.clamp(min=-experts_module.limit, max=experts_module.limit)
        glu = gate * torch.sigmoid(gate * experts_module.alpha)
        hidden = (up + 1) * glu
    else:
        act_fn = getattr(experts_module, "act_fn", F.silu)
        hidden = act_fn(gate) * up

    down_w = experts_module.down_proj[expert_id]
    down_b = experts_module.down_proj_bias[expert_id] if hasattr(experts_module, "down_proj_bias") else None
    return _linear_any_layout(hidden, down_w, down_b)


def patch_moe_forward(model, pruner, top_k: int = 2):
    """
    Patch the MoE layers to capture intermediate states and update EasyEP.
    This is specific to Hugging Face Transformers implementations of MoE.

    Args:
        model: The Hugging Face Transformers model with MoE layers.
        pruner: An instance of a pruning metric accumulator (EasyEP/Frequency/EAN/REAP/VITA/JointCal, etc.).
        top_k: Number of top experts selected by the router.
    """

    #####
    # Identify MoE layers
    #####
    moe_layers = get_moe_layers(model)
    print(f"Found {len(moe_layers)} MoE layers to patch.")
    
    # Ensure accumulator is initialized with correct number of layers
    if getattr(pruner, "n_layers", None) != len(moe_layers):
        raise ValueError("Pruner n_layers does not match number of MoE layers in the model.")

    cfg_top_k = getattr(getattr(model, "config", None), "num_experts_per_tok", None)
    if cfg_top_k is not None:
        top_k = int(cfg_top_k)
    else:
        top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"Invalid top_k={top_k}. Set model.config.num_experts_per_tok or pass top_k>0")

    #####
    # Patch forward methods to capture necessary data for pruning
    #####
    hook_handles = []
    for layer_idx, (name, module) in enumerate(moe_layers):
        print(f"\tPatching MoE layer {layer_idx}: {name}")

        # Avoid stacking multiple hooks across repeated patching (iterative pruning).
        old_handle = getattr(module, "_moe_prune_forward_hook_handle", None)
        if old_handle is not None:
            try:
                old_handle.remove()
            except Exception:
                pass
            setattr(module, "_moe_prune_forward_hook_handle", None)

        def hook_fn(module, args, kwargs, output, idx=layer_idx):
            # Try to get hidden_states from args or kwargs
            hidden_states = None
            if len(args) > 0:
                hidden_states = args[0]
            elif kwargs and 'hidden_states' in kwargs:
                hidden_states = kwargs['hidden_states'] # Input hidden states to the MoE layer. [Batch, Seq, Hidden]
            if hidden_states is None:
                print("Could not find hidden_states in args or kwargs.")
                return

            # IMPORTANT: the computations below are for pruning metrics only.
            # Detach to avoid building extra autograd graphs.
            hs = hidden_states.detach() # [B, S, H]
            batch_size, seq_len, hidden_dim = hs.shape

            # All-experts mode (e.g., EAN): evaluate every expert on (a sample of) tokens.
            if getattr(pruner, "requires_all_experts", False):
                if not hasattr(module, "experts"):
                    print("No experts found in MoE module.")
                    return
                if not hasattr(pruner, "update_all"):
                    raise ValueError("requires_all_experts=True but pruning object has no update_all(...) method")

                x = hs.reshape(batch_size * seq_len, hidden_dim)

                # Optional token subsampling (implemented by the pruning object)
                if hasattr(pruner, "sample_token_indices"):
                    token_idx = pruner.sample_token_indices(x.shape[0])
                    x = x[token_idx]

                # Accumulate sum ||E_i(x_t)|| over sampled tokens
                if isinstance(module.experts, nn.ModuleList):
                    n_experts = len(module.experts)
                else:
                    n_experts = module.experts.num_experts
                sums = torch.zeros(n_experts, device=x.device, dtype=torch.float32)
                for expert_id in range(n_experts):
                    with torch.no_grad():
                        if isinstance(module.experts, nn.ModuleList):
                            out = module.experts[expert_id](x)
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                        else:
                            out = _run_fused_expert(module.experts, int(expert_id), x)
                    sums[expert_id] = torch.norm(out, p=2, dim=-1).sum().to(dtype=torch.float32)

                pruner.update_all(idx, sums)
                return
            
            #####
            # 1. Run the gate to get routing weights
            #####
            if hasattr(module, 'gate'):
                router_output = module.gate(hs)
            elif hasattr(module, 'router'):
                # GPT-OSS router is implemented on flattened tokens [B*S, H].
                if getattr(getattr(model, "config", None), "model_type", "") == "gpt_oss":
                    router_output = module.router(hs.reshape(batch_size * seq_len, hidden_dim))
                else:
                    router_output = module.router(hs)
            else:
                print("No gate or router found in MoE module.")
                return
            
            # Handle different router return types
            if isinstance(router_output, (tuple, list)):
                if model.config.model_type == "nemotron_h":
                    # Nemotron 3 style: router returns (indices, weights) already processed
                    selected_experts, routing_weights = router_output
                    # Nemotron flattens batch and seq dimensions into the first dimension, so we need to unflatten them
                    selected_experts = selected_experts.view(batch_size, seq_len, -1)
                    routing_weights = routing_weights.view(batch_size, seq_len, -1)
                else:
                    # GPT-OSS style router outputs vary by transformers version:
                    # - (weights, indices)
                    # - (logits, weights, indices)
                    # -> [B, S, num_experts], [B, S, top_k], [B, S, top_k] respectively
                    tensors = [t for t in router_output if isinstance(t, torch.Tensor)]
                    if len(tensors) < 2:
                        raise ValueError(
                            f"Unexpected GPT-OSS router output: expected >=2 tensors, got {len(tensors)}"
                        )

                    selected_experts = None
                    for t in tensors:
                        if t.dtype in (torch.int32, torch.int64):
                            selected_experts = t
                            break
                    if selected_experts is None:
                        raise ValueError(
                            "Unexpected GPT-OSS router output: could not find integer expert indices tensor"
                        )

                    # If router returned flattened tokens [B*S, ...], restore [B, S, ...].
                    if selected_experts.ndim == 2 and selected_experts.shape[0] == batch_size * seq_len:
                        selected_experts = selected_experts.view(batch_size, seq_len, -1)
                    elif selected_experts.ndim == 2 and batch_size == 1:
                        selected_experts = selected_experts.unsqueeze(0)

                    # Prefer top-k weights tensor whose trailing shape matches selected indices.
                    routing_weights = None
                    for t in tensors:
                        if not torch.is_floating_point(t):
                            continue
                        tc = t
                        if tc.ndim == 2 and tc.shape[0] == batch_size * seq_len:
                            tc = tc.view(batch_size, seq_len, -1)
                        elif tc.ndim == 2 and batch_size == 1:
                            tc = tc.unsqueeze(0)
                        if tc.shape == selected_experts.shape:
                            # Keep routing weights in float for numerical stability in downstream metrics.
                            routing_weights = tc.float()
                            break
                    if routing_weights is not None:
                        # GPT-OSS may flatten [B, S] into [B*S, ...]. Unflatten to [B, S, K].
                        if routing_weights.ndim == 2 and batch_size == 1:
                            routing_weights = routing_weights.unsqueeze(0)
                        elif routing_weights.shape[0] == hs.shape[0] * hs.shape[1]:
                            routing_weights = routing_weights.view(batch_size, seq_len, -1)
                        routing_weights = routing_weights.float()
                    
                    # Fallback: use dense logits/probabilities and gather with indices.
                    if routing_weights is None:
                        dense_scores = None
                        for t in tensors:
                            if not torch.is_floating_point(t):
                                continue
                            if t.ndim >= 2:
                                dense_scores = t
                                break
                        if dense_scores is None:
                            raise ValueError(
                                "Unexpected GPT-OSS router output: could not find floating-point router scores"
                            )

                        if dense_scores.ndim == 2 and hs.shape[0] == 1:
                            dense_scores = dense_scores.unsqueeze(0)
                        elif dense_scores.shape[0] == hs.shape[0] * hs.shape[1]:
                            dense_scores = dense_scores.view(hs.shape[0], hs.shape[1], -1)

                        routing_weights = dense_scores.gather(-1, selected_experts).float()
                    
                    # Convert sparse [B, S, num_experts] -> dense [B, S, top_k]
                    # if the selected weights are still in full expert space.
                    if routing_weights.shape[-1] != selected_experts.shape[-1]:
                        routing_weights = routing_weights.gather(-1, selected_experts)
                        
                        # Keep a consistent interpretation across router variants: top-k weights per token sum to 1.
                        denom = routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        routing_weights = routing_weights / denom

            else:
                # Standard MoE: router returns raw logits, we do softmax + topk
                # router_output is typically [B, S, n_experts]; softmax must be over expert dimension
                routing_weights = F.softmax(router_output, dim=-1, dtype=torch.float)
                
                # Top-k selection
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                
                # Normalize routing weights
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            #####
            # 2. Run selected experts and collect their outputs (only if needed)
            #####
            need_outputs = getattr(pruner, "requires_expert_outputs", True)
            expert_outputs = None

            if need_outputs:
                if not hasattr(module, 'experts'):
                    raise ValueError("Pruner requires expert outputs but module has no experts")

                batch_size, seq_len, hidden_dim = hs.shape
                selected_k = int(selected_experts.shape[-1])
                expert_outputs = torch.zeros(batch_size, seq_len, selected_k, hidden_dim, device=hs.device, dtype=hs.dtype)

                # For each unique expert, run it on the corresponding inputs
                unique_experts = torch.unique(selected_experts).tolist()
                for expert_id in unique_experts:
                    # Find where this expert is used
                    mask = (selected_experts == expert_id)  # [B, S, K]
                    if not mask.any():
                        continue

                    b_idx, s_idx, k_idx = torch.where(mask)

                    # Gather inputs
                    inputs = hs[b_idx, s_idx]  # [N_active, H]

                    # Detect architecture and compute accordingly
                    if isinstance(module.experts, nn.ModuleList):
                        # Qwen3 v4: Subscriptable ModuleList
                        expert = module.experts[int(expert_id)]
                        out = expert(inputs)
                        
                    else:
                        out = _run_fused_expert(module.experts, int(expert_id), inputs)
                    
                    expert_outputs[b_idx, s_idx, k_idx] = out.to(expert_outputs.dtype)

            # Update pruning metric once per forward.
            pruner.update(idx, hs, routing_weights, expert_outputs, selected_experts)

            return

        # Register the hook
        handle = module.register_forward_hook(hook_fn, with_kwargs=True)
        setattr(module, "_moe_prune_forward_hook_handle", handle)
        hook_handles.append(handle)

    return hook_handles
