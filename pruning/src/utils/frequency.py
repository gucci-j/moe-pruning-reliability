import torch


class FrequencyCounter:
    """Counts expert selection frequency.

    Implements: sum_{t=1..T} 1(i in top-k)

    `patch_moe_forward` will call `update(...)` with `expert_indices` shaped [B, S, K].
    We count one hit per (token, selected-expert) occurrence.
    """

    def __init__(self, n_layers: int, n_experts: int, device: str | torch.device = "cpu"):
        self.requires_expert_outputs = False
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.device = device
        self.counts = [torch.zeros(n_experts, device=device, dtype=torch.float32) for _ in range(n_layers)]

    def reset(self):
        self.counts = [torch.zeros(self.n_experts, device=self.device, dtype=torch.float32) for _ in range(self.n_layers)]

    def update(self, layer_idx, hidden_states, router_weights, expert_outputs, expert_indices):
        # We only need expert_indices. Keep signature compatible with EasyEP.
        expert_indices = expert_indices.detach()

        if self.counts[layer_idx] is None:
            self.counts[layer_idx] = torch.zeros(self.n_experts, device=expert_indices.device, dtype=torch.float32)

        # In sharded setups (device_map="auto"), different MoE layers can live on
        # different GPUs. Keep each per-layer accumulator on that layer's device.
        if self.counts[layer_idx].device != expert_indices.device:
            self.counts[layer_idx] = self.counts[layer_idx].to(expert_indices.device)

        flat = expert_indices.reshape(-1).to(dtype=torch.long, device=expert_indices.device)
        ones = torch.ones(flat.shape, device=expert_indices.device, dtype=torch.float32)
        tmp = torch.zeros(self.n_experts, device=expert_indices.device, dtype=torch.float32)
        tmp.index_add_(0, flat, ones)
        self.counts[layer_idx] += tmp

    def get_top_experts(self, k: int):
        top_experts = []
        for layer_counts in self.counts:
            if layer_counts is None:
                top_experts.append([])
                continue
            _, indices = torch.topk(layer_counts, k)
            top_experts.append(indices.cpu().tolist())
        return top_experts
