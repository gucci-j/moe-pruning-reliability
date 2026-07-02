
def get_moe_layers(model):
    """
    Get all MoE layers from the model.
    Assumes MoE layers have 'experts' attribute.

    Returns:
        List of (layer_name, layer_module) tuples.
    """
    moe_layers = []
    if model.config.model_type == "nemotron_h":
        print("🔍 Detected Nemotron-like hybrid architecture")
        for name, module in model.backbone.layers.named_modules():
            if name.endswith('.mixer') and hasattr(module, 'experts'):
                moe_layers.append((name, module))
                # An MoE layer has: mixer.experts, mixer.gate, and mixer.shared_experts
    elif model.config.model_type == "qwen3_5_moe":
        print("🔍 Detected Qwen3.5-MoE architecture")
        for name, module in model.model.language_model.named_modules():
            if name.endswith('.mlp') and hasattr(module, 'experts'):
                moe_layers.append((name, module))
    else:
        for name, module in model.named_modules():
            if name.endswith('.mlp') or name.endswith('.block_sparse_moe'):
                moe_layers.append((name, module))
    return moe_layers


def get_moe_info(model):
    """
    Get the number of MoE layers and the number of experts per layer.
    Assumes all MoE layers have the same number of experts.

    References:
    - https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json
    - https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
    """
    if model.config.model_type == "nemotron_h":
        print("🔍 Detected Nemotron-like hybrid architecture")
        n_layers = len([
            module for name, module in model.backbone.layers.named_modules() if name.endswith('.mixer') and hasattr(module, 'experts')
        ]) # Some layers may not have experts, so we count them directly
    elif model.config.model_type == "qwen3_5_moe":
        print("🔍 Detected Qwen3.5-MoE architecture")
        n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config.text_config, 'num_hidden_layers') else 0
        n_experts = model.config.text_config.num_experts if hasattr(model.config.text_config, 'num_experts') else 0
        return n_layers, n_experts
    else:
        n_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 0
        
    n_experts = model.config.num_local_experts if hasattr(model.config, 'num_local_experts') else 0
    if n_experts == 0:
        n_experts = model.config.num_experts if hasattr(model.config, 'num_experts') else 0
    if n_experts == 0:
        n_experts = model.config.n_routed_experts if hasattr(model.config, 'n_routed_experts') else 0
    return n_layers, n_experts
