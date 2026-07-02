import torch


class GatingCounter:
    """Accumulates per-expert summed routing weights.

    Definition (per layer l, expert i):
      r_li = sum over tokens t and selected slots k of routing_weight[t, k]
             where expert_indices[t, k] == i.

    `patch_moe_forward` passes:
      - router_weights: [B, S, K] (post-softmax, top-k, and renormalized)
      - expert_indices: [B, S, K]

    This metric is a natural companion to `FrequencyCounter` (which sums 1s).
    """

    def __init__(self, n_layers: int, n_experts: int, device: str | torch.device = "cpu"):
        self.requires_expert_outputs = False
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.device = device
        self.sums = [torch.zeros(n_experts, device=device, dtype=torch.float32) for _ in range(n_layers)]

    def reset(self):
        self.sums = [torch.zeros(self.n_experts, device=self.device, dtype=torch.float32) for _ in range(self.n_layers)]

    def update(self, layer_idx, hidden_states, router_weights, expert_outputs, expert_indices):
        # Keep signature compatible with other pruners.
        expert_indices = expert_indices.detach()
        router_weights = router_weights.detach().to(device=expert_indices.device, dtype=torch.float32)

        if self.sums[layer_idx] is None:
            self.sums[layer_idx] = torch.zeros(self.n_experts, device=expert_indices.device, dtype=torch.float32)

        # In sharded setups (device_map="auto"), MoE layers can be distributed
        # across multiple GPUs. Keep each per-layer accumulator on that layer's
        # active device to avoid cross-device adds.
        if self.sums[layer_idx].device != expert_indices.device:
            self.sums[layer_idx] = self.sums[layer_idx].to(expert_indices.device)

        flat_idx = expert_indices.reshape(-1).to(device=expert_indices.device, dtype=torch.long)
        flat_w = router_weights.reshape(-1)

        tmp = torch.zeros(self.n_experts, device=expert_indices.device, dtype=torch.float32)
        tmp.index_add_(0, flat_idx, flat_w)
        self.sums[layer_idx] += tmp

    def get_top_experts(self, k: int):
        top_experts = []
        for layer_sums in self.sums:
            if layer_sums is None:
                top_experts.append([])
                continue
            _, indices = torch.topk(layer_sums, k)
            top_experts.append(indices.cpu().tolist())
        return top_experts
