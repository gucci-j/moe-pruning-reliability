import torch
import torch.nn.functional as F

class EasyEP:
    def __init__(self, n_layers, n_experts):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.expert_scores = [None] * n_layers
        self.requires_expert_outputs = True

    def reset(self):
        self.expert_scores = [None] * self.n_layers

    def update(
        self, 
        layer_idx, 
        hidden_states, 
        router_weights, 
        expert_outputs, 
        expert_indices
    ):
        """
        Update the expert scores for a given layer.

        Args:
            layer_idx (int): The index of the layer.
            hidden_states (torch.Tensor): Input hidden states to the MoE layer. [Batch, Seq, Hidden]
            router_weights (torch.Tensor): The gating weights for the selected experts. [Batch, Seq, K]
            expert_outputs (torch.Tensor): The outputs of the selected experts. [Batch, Seq, K, Hidden]
            expert_indices (torch.Tensor): The indices of the selected experts. [Batch, Seq, K]
        """
        # Ensure inputs are detached to save memory
        hidden_states = hidden_states.detach()
        router_weights = router_weights.detach()
        expert_outputs = expert_outputs.detach()
        expert_indices = expert_indices.detach()

        # Initialize score tensor if not exists
        if self.expert_scores[layer_idx] is None:
            self.expert_scores[layer_idx] = torch.zeros(self.n_experts, device=hidden_states.device)

        #####
        # 1. Calculate Expert-Level Token Contribution (s_t^l)
        # - "Given the representations before and after the routed expert modules hlt and  ̃hlt, we compute the cosine similarity between these representations"
        # s_t^l = 1 - Sim(h, h_tilde)
        # h_tilde = h + sum(g * e)
        
        # Weighted sum of expert outputs
        # expert_outputs: [B, S, K, H]
        # router_weights: [B, S, K]
        moe_out = torch.sum(expert_outputs * router_weights.unsqueeze(-1), dim=2) # [B, S, H]
        h_tilde = hidden_states + moe_out
        
        # Cosine similarity
        # We compute similarity along the hidden dimension
        # Add epsilon to avoid division by zero
        sim = F.cosine_similarity(hidden_states, h_tilde, dim=-1, eps=1e-8) # [B, S]
        s_t = 1.0 - sim # [B, S]
        
        #####
        # 2. Calculate Output-Aware Expert Importance (c_{i,t}^l)
        # c_{i,t}^l = g_{i,t}^l * ||e_{i,t}^l||_2
        
        e_norm = torch.norm(expert_outputs, p=2, dim=-1) # [B, S, K]
        c_it = router_weights * e_norm # [B, S, K]
        
        #####
        # 3. Final Score
        # I = c_it * s_t
        score_contribution = c_it * s_t.unsqueeze(-1)  # [B, S, K]
        # Aggregate contributions across batch and sequence dimensions in a vectorized manner.
        # Flatten indices and scores, then index_add into the expert score vector.
        flat_indices = expert_indices.reshape(-1)  # [B*S*K]
        flat_scores = score_contribution.reshape(-1)  # [B*S*K]
        # Accumulate into a temporary buffer and then add to persistent layer scores.
        tmp_scores = torch.zeros(self.n_experts, device=hidden_states.device, dtype=flat_scores.dtype)
        tmp_scores.index_add_(0, flat_indices, flat_scores)
        self.expert_scores[layer_idx] += tmp_scores # [N_experts]
                    

    def get_top_experts(self, k):
        """
        Get the indices of the top-k experts for each layer.
        """
        top_experts = []
        for scores in self.expert_scores:
            if scores is None:
                top_experts.append([])
                continue
            _, indices = torch.topk(scores, k)
            top_experts.append(indices.cpu().tolist())
        return top_experts
