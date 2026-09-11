"""Deterministic current-board production features; no history or hidden information."""
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from catanatron.models.map import build_map


class ProductionFeatures(BaseFeaturesExtractor):
    """Expose per-node resource yields already derivable from the public board."""
    def __init__(self, observation_space):
        board = build_map("BASE")
        coordinates = sorted(board.land_tiles)
        nodes = sorted(board.land_nodes)
        self.node_start = len(coordinates) * 8 + len(nodes) * 6 + len(coordinates)
        self.node_end = self.node_start + len(nodes) * 4
        super().__init__(observation_space, observation_space.shape[0] + len(nodes) * 5 + 10)
        incidence = np.array([[node in board.land_tiles[c].nodes.values()
                               for c in coordinates] for node in nodes], dtype=np.float32)
        self.register_buffer("incidence", torch.as_tensor(incidence))

    def forward(self, observations):
        tiles = observations[:, :152].reshape(-1, 19, 8)
        yields = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8])
        buildings = observations[:, self.node_start:self.node_end].reshape(-1, 54, 4)
        own = buildings[:, :, 0] + 2 * buildings[:, :, 1]
        enemy = buildings[:, :, 2] + 2 * buildings[:, :, 3]
        own_yields = (yields * own.unsqueeze(-1)).sum(dim=1) / 30
        enemy_yields = (yields * enemy.unsqueeze(-1)).sum(dim=1) / 30
        return torch.cat((observations, yields.flatten(1) / 3, own_yields, enemy_yields), dim=1)
