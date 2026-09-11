"""Memoryless shared action scorer using only the encoded public/current state."""
import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from catanatron.models.enums import RESOURCES, ActionType as A
from .game import CatanEnv
from .features import ProductionFeatures


class ActorValue(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.latent_dim_pi, self.latent_dim_vf = size, 128
        self.value = nn.Sequential(nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.value(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class ActionScorer(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        env = CatanEnv()
        self.production = ProductionFeatures(observation_space)
        kinds = list(A)
        node_ids = {node: i for i, node in enumerate(env.nodes)}
        tile_ids = {coordinate: i for i, coordinate in enumerate(env.coordinates)}
        kind_ids, nodes, node_masks, tiles, tile_masks, give, take, costs = [], [], [], [], [], [], [], []
        for kind, value in env.encoder.actions:
            kind_ids.append(kinds.index(kind))
            ns = ([value] if kind in (A.BUILD_CITY, A.BUILD_SETTLEMENT) else
                  list(value) if kind == A.BUILD_ROAD else [])
            nodes.append([node_ids[ns[0]], node_ids[ns[-1]]] if ns else [0, 0])
            node_masks.append(bool(ns))
            tiles.append(tile_ids[value[0]] if kind == A.MOVE_ROBBER else 0)
            tile_masks.append(kind == A.MOVE_ROBBER)
            g, t, cost = np.zeros(5), np.zeros(5), 0
            if kind == A.MARITIME_TRADE:
                g[RESOURCES.index(value[0])] = 1
                t[RESOURCES.index(value[-1])] = 1
                cost = len(value) - 1
            elif kind == A.DISCARD_RESOURCE:
                g[RESOURCES.index(value)] = 1
            elif kind == A.PLAY_MONOPOLY:
                t[RESOURCES.index(value)] = 1
            elif kind == A.PLAY_YEAR_OF_PLENTY:
                for resource in value:
                    t[RESOURCES.index(resource)] += .5
            give.append(g)
            take.append(t)
            costs.append(cost / 4)
        env.close()
        for name, values, dtype in (("kinds", kind_ids, torch.long), ("nodes", nodes, torch.long),
                                    ("node_masks", node_masks, torch.float32), ("tiles", tiles, torch.long),
                                    ("tile_masks", tile_masks, torch.float32), ("give", give, torch.float32),
                                    ("take", take, torch.float32), ("costs", costs, torch.float32)):
            self.register_buffer(name, torch.as_tensor(np.asarray(values), dtype=dtype))
        self.kind_embedding = nn.Embedding(len(kinds), 8)
        self.context = nn.Sequential(nn.Linear(78, 32), nn.Tanh())
        self.kind_scores = nn.Linear(32, len(kinds))
        self.local_scores = nn.Sequential(nn.Linear(32 + 8 + 20, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observations):
        batch = observations.shape[0]
        derived = self.production(observations)
        yields = derived[:, 923:-10].reshape(batch, 54, 5)
        endpoints = yields[:, self.nodes]
        local = endpoints.mean(dim=2) * self.node_masks[None, :, None]
        best = endpoints.sum(dim=-1).max(dim=2).values * self.node_masks
        buildings = observations[:, 495:711].reshape(batch, 54, 4)
        own = buildings[:, :, 0] + 2 * buildings[:, :, 1]
        enemy = buildings[:, :, 2] + 2 * buildings[:, :, 3]
        own_rob = (own @ self.production.incidence)[:, self.tiles] * self.tile_masks / 6
        enemy_rob = (enemy @ self.production.incidence)[:, self.tiles] * self.tile_masks / 6
        hand = observations[:, 883:888]
        give_hand = hand @ self.give.T
        take_hand = hand @ self.take.T
        context = self.context(torch.cat((observations[:, 855:], derived[:, -10:]), dim=1))
        local_features = torch.cat((local, best.unsqueeze(-1), own_rob.unsqueeze(-1), enemy_rob.unsqueeze(-1),
                                    self.give.expand(batch, -1, -1), self.take.expand(batch, -1, -1),
                                    give_hand.unsqueeze(-1), take_hand.unsqueeze(-1)), dim=-1)
        # All locations share parameters; no learned action-ID-specific shortcut.
        features = torch.cat((context[:, None].expand(-1, len(self.kinds), -1),
                              self.kind_embedding(self.kinds).expand(batch, -1, -1), local_features), dim=-1)
        return self.kind_scores(context)[:, self.kinds] + self.local_scores(features).squeeze(-1)


class ActionFeaturePolicy(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorValue(self.features_dim)

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (923,) or action_space.n != 332:
            raise ValueError("ActionFeaturePolicy requires observation/action format 3")
        self.action_net = ActionScorer(observation_space)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
