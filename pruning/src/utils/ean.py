import torch


class EANCounter:
    """Expert Activation Norm (EAN) accumulator.

    EAN per expert i is:
      sum_{t=1..T} || E_i(x_t) ||

    This ignores router selection and instead evaluates every expert on a sample
    of token hidden-states.

    Notes:
    - To keep compute bounded, we optionally subsample tokens per forward.
    - `patch_moe_forward` detects `requires_all_experts=True` and will call `update_all`.
    """

    requires_all_experts = True

    def __init__(
        self,
        n_layers: int,
        n_experts: int,
        device: str | torch.device = "cpu",
        max_tokens_per_batch: int = 256,
        token_sampling: str = "first",
        seed: int = 0,
    ):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.device = device
        self.max_tokens_per_batch = max_tokens_per_batch
        self.token_sampling = token_sampling
        self.seed = seed
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        self.requires_expert_outputs = True

        self.sums = [torch.zeros(n_experts, device=device, dtype=torch.float32) for _ in range(n_layers)]

    def reset(self):
        self.sums = [torch.zeros(self.n_experts, device=self.device, dtype=torch.float32) for _ in range(self.n_layers)]

    def update_all(self, layer_idx: int, batch_sums: torch.Tensor):
        """batch_sums is a [n_experts] float tensor to add."""
        self.sums[layer_idx] += batch_sums.to(self.sums[layer_idx].device, dtype=self.sums[layer_idx].dtype)

    def sample_token_indices(self, n_tokens: int) -> torch.Tensor:
        if self.max_tokens_per_batch is None or self.max_tokens_per_batch <= 0 or self.max_tokens_per_batch >= n_tokens:
            return torch.arange(n_tokens)

        k = int(self.max_tokens_per_batch)
        if self.token_sampling == "first":
            return torch.arange(k)
        if self.token_sampling == "random":
            return torch.randperm(n_tokens, generator=self._gen)[:k]
        raise ValueError(f"Unknown token_sampling: {self.token_sampling}")

    def get_top_experts(self, k: int):
        top_experts = []
        for layer_sums in self.sums:
            _, idx = torch.topk(layer_sums, k)
            top_experts.append(idx.cpu().tolist())
        return top_experts
