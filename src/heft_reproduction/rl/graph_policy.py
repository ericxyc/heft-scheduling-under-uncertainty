"""Pure-PyTorch message-passing features for MaskablePPO."""

from __future__ import annotations

import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import (
    MaskableMultiInputActorCriticPolicy,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GraphCandidateExtractor(BaseFeaturesExtractor):
    """Encode DAG nodes, edges, candidates, and global scheduler state."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        candidate_dim: int = 16,
        message_passing_steps: int = 3,
    ) -> None:
        candidate_shape = observation_space["candidates"].shape
        node_shape = observation_space["graph_nodes"].shape
        edge_shape = observation_space["graph_edge_features"].shape
        global_dim = observation_space["global"].shape[0]
        self.max_candidates = candidate_shape[0]
        self.max_nodes = node_shape[0]
        self.candidate_dim = candidate_dim
        self.context_dim = hidden_dim + global_dim
        features_dim = (
            self.max_candidates * candidate_dim + hidden_dim + global_dim
        )
        super().__init__(observation_space, features_dim=features_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_shape[1], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        message_input = hidden_dim + edge_shape[1]
        self.forward_messages = nn.ModuleList()
        self.reverse_messages = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(message_passing_steps):
            self.forward_messages.append(
                nn.Sequential(
                    nn.Linear(message_input, hidden_dim), nn.SiLU()
                )
            )
            self.reverse_messages.append(
                nn.Sequential(
                    nn.Linear(message_input, hidden_dim), nn.SiLU()
                )
            )
            self.updates.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
            )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(hidden_dim + candidate_shape[1], hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, candidate_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _aggregate(
        messages: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        output = messages.new_zeros((output_size, messages.shape[-1]))
        output.index_add_(0, indices, messages * weights)
        degree = messages.new_zeros((output_size, 1))
        degree.index_add_(0, indices, weights)
        return output / degree.clamp_min(1.0)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = observations["graph_nodes"].float()
        node_mask = observations["graph_node_mask"].float()
        edge_index = observations["graph_edge_index"].long()
        edge_features = observations["graph_edge_features"].float()
        edge_mask = observations["graph_edge_mask"].float()
        batch_size, node_count, _ = nodes.shape
        edge_count = edge_index.shape[1]

        hidden = self.node_encoder(nodes) * node_mask.unsqueeze(-1)
        offsets = (
            torch.arange(batch_size, device=nodes.device).view(-1, 1)
            * node_count
        )
        source = (edge_index[:, :, 0] + offsets).reshape(-1)
        target = (edge_index[:, :, 1] + offsets).reshape(-1)
        flat_edges = edge_features.reshape(
            batch_size * edge_count, edge_features.shape[-1]
        )
        weights = edge_mask.reshape(-1, 1)
        output_size = batch_size * node_count

        for forward_net, reverse_net, update_net in zip(
            self.forward_messages, self.reverse_messages, self.updates
        ):
            flat_hidden = hidden.reshape(output_size, -1)
            forward = forward_net(
                torch.cat((flat_hidden[source], flat_edges), dim=-1)
            )
            reverse = reverse_net(
                torch.cat((flat_hidden[target], flat_edges), dim=-1)
            )
            from_parents = self._aggregate(
                forward, target, weights, output_size
            )
            from_children = self._aggregate(
                reverse, source, weights, output_size
            )
            updated = update_net(
                torch.cat((flat_hidden, from_parents, from_children), dim=-1)
            )
            hidden = (
                flat_hidden + updated
            ).reshape(batch_size, node_count, -1)
            hidden = hidden * node_mask.unsqueeze(-1)

        candidate_nodes = observations["candidate_nodes"].long()
        gather_index = candidate_nodes.unsqueeze(-1).expand(
            -1, -1, hidden.shape[-1]
        )
        candidate_graph = torch.gather(hidden, 1, gather_index)
        candidate_raw = observations["candidates"].float()
        candidate_features = self.candidate_encoder(
            torch.cat((candidate_graph, candidate_raw), dim=-1)
        )
        candidate_features = candidate_features * observations[
            "action_mask"
        ].float().unsqueeze(-1)
        pooled = hidden.sum(dim=1) / node_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return torch.cat(
            (
                candidate_features.flatten(start_dim=1),
                pooled,
                observations["global"].float(),
            ),
            dim=-1,
        )


class CandidateSetActionNet(nn.Module):
    """Score each candidate with shared weights and invariant pooled context."""

    def __init__(
        self,
        max_candidates: int,
        candidate_dim: int,
        context_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.max_candidates = max_candidates
        self.candidate_dim = candidate_dim
        self.context_dim = context_dim
        self.scorer = nn.Sequential(
            nn.Linear(candidate_dim * 2 + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        candidate_width = self.max_candidates * self.candidate_dim
        candidates = latent[:, :candidate_width].reshape(
            -1, self.max_candidates, self.candidate_dim
        )
        context = latent[:, candidate_width : candidate_width + self.context_dim]
        present = candidates.abs().sum(dim=-1, keepdim=True).gt(0).float()
        pooled = (candidates * present).sum(dim=1) / present.sum(
            dim=1
        ).clamp_min(1.0)
        expanded = torch.cat(
            (
                candidates,
                pooled.unsqueeze(1).expand(-1, self.max_candidates, -1),
                context.unsqueeze(1).expand(-1, self.max_candidates, -1),
            ),
            dim=-1,
        )
        return self.scorer(expanded).squeeze(-1)


class PermutationInvariantGraphPolicy(MaskableMultiInputActorCriticPolicy):
    """MaskablePPO policy with candidate-permutation-equivariant actor logits."""

    def __init__(self, *args, **kwargs) -> None:
        observation_space = args[0] if args else kwargs["observation_space"]
        lr_schedule = args[2] if len(args) > 2 else kwargs["lr_schedule"]
        extractor_kwargs = kwargs.get("features_extractor_kwargs") or {}
        self._candidate_count = observation_space["candidates"].shape[0]
        self._candidate_dim = int(extractor_kwargs.get("candidate_dim", 16))
        hidden_dim = int(extractor_kwargs.get("hidden_dim", 64))
        global_dim = observation_space["global"].shape[0]
        super().__init__(*args, **kwargs)
        self.action_net = CandidateSetActionNet(
            max_candidates=self._candidate_count,
            candidate_dim=self._candidate_dim,
            context_dim=hidden_dim + global_dim,
            hidden_dim=hidden_dim,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
