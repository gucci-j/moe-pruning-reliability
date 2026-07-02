import torch


class REAPCounter:
    """REAP: Router-weighted Expert Activation Power.

    Implements:
      sum_{t=1..T} G(x_t)_i * ||E_i(x_t)||

    where G(x_t)_i is the router probability/weight for expert i at token t,
    and E_i(x_t) is the expert output (only computed for top-k routed experts).

    This is similar to the c_{i,t} term in EasyEP, but without the token-level
    similarity scaling.
    """

    def __init__(self, n_layers: int, n_experts: int, device: str | torch.device = "cpu"):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.device = device
        self.scores = [torch.zeros(n_experts, device=device, dtype=torch.float32) for _ in range(n_layers)]
        self.requires_expert_outputs = True

    def reset(self):
        self.scores = [torch.zeros(self.n_experts, device=self.device, dtype=torch.float32) for _ in range(self.n_layers)]

    def update(self, layer_idx, hidden_states, router_weights, expert_outputs, expert_indices):
        # Detach to save memory
        router_weights = router_weights.detach()
        expert_outputs = expert_outputs.detach()
        expert_indices = expert_indices.detach()

        # ||E_i(x_t)|| over hidden dim
        e_norm = torch.norm(expert_outputs, p=2, dim=-1)  # [B, S, K]
        contrib = (router_weights * e_norm).reshape(-1).to(dtype=torch.float32)  # [B*S*K]
        flat_idx = expert_indices.reshape(-1)

        tmp = torch.zeros(self.n_experts, device=expert_indices.device, dtype=torch.float32)
        tmp.index_add_(0, flat_idx, contrib)
        self.scores[layer_idx] += tmp.to(self.scores[layer_idx].device)

    def get_top_experts(self, k: int):
        top_experts = []
        for layer_scores in self.scores:
            _, idx = torch.topk(layer_scores, k)
            top_experts.append(idx.cpu().tolist())
        return top_experts
