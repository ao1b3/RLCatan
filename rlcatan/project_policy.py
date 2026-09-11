"""Format-3 graph/action policies with an optional learned economic-project residual."""
import numpy as np
import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.map import build_map

from .game import CatanEnv
from .action_policy import ActorValue


class BoardGraph(nn.Module):
    """Two small public-board message-passing rounds over fixed Catan nodes."""
    def __init__(self):
        super().__init__()
        env = CatanEnv()
        nodes, coordinates = env.nodes, env.coordinates
        env.close()
        board = build_map('BASE')
        incidence = np.array([[node in STATIC_GRAPH and node in board.land_tiles[tile].nodes.values()
                               for tile in coordinates] for node in nodes], dtype=np.float32)
        adjacency = np.array([[STATIC_GRAPH.has_edge(a, b) for b in nodes] for a in nodes], dtype=np.float32)
        edges = tuple(sorted(tuple(sorted(edge)) for edge in get_edges(nodes)))
        edge_incidence = np.array([[node in edge for edge in edges] for node in nodes], dtype=np.float32)
        self.register_buffer('incidence', torch.as_tensor(incidence))
        self.register_buffer('adjacency', torch.as_tensor(adjacency))
        self.register_buffer('edge_incidence', torch.as_tensor(edge_incidence))
        self.input = nn.Sequential(nn.Linear(22, 32), nn.Tanh())
        self.message1 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.message2 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())

    def node_inputs(self, obs):
        batch = len(obs)
        tiles = obs[:, :152].reshape(batch, 19, 8)
        nominal = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8])
        robber = obs[:, 476:495]
        blocked = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8] * (1 - robber).unsqueeze(-1))
        ports = obs[:, 152:476].reshape(batch, 54, 6)
        buildings = obs[:, 495:711].reshape(batch, 54, 4)
        roads = obs[:, 711:855].reshape(batch, 72, 2)
        access = (self.edge_incidence @ roads) / 3
        return torch.cat((nominal / 3, blocked / 3, ports, buildings, access), -1)

    def forward(self, obs):
        nodes = self.input(self.node_inputs(obs))
        degree = self.adjacency.sum(-1).clamp_min(1)
        for layer in (self.message1, self.message2):
            nodes = layer(torch.cat((nodes, self.adjacency @ nodes / degree[None, :, None]), -1))
        return nodes


class GraphScorer(nn.Module):
    def __init__(self, projects=False):
        super().__init__()
        self.projects = projects
        self.graph = BoardGraph()
        env = CatanEnv()
        node_ids = {node: i for i, node in enumerate(env.nodes)}
        tile_ids = {tile: i for i, tile in enumerate(env.coordinates)}
        kinds = list(A)
        kind_ids, endpoints, endpoint_masks, tiles, tile_masks, delta, give, take, count = [], [], [], [], [], [], [], [], []
        costs = {A.BUILD_ROAD: (1, 1, 0, 0, 0), A.BUILD_SETTLEMENT: (1, 1, 1, 1, 0),
                 A.BUILD_CITY: (0, 0, 0, 2, 3), A.BUY_DEVELOPMENT_CARD: (0, 0, 1, 1, 1)}
        for kind, value in env.encoder.actions:
            kind_ids.append(kinds.index(kind))
            ns = ([value] if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY) else list(value) if kind == A.BUILD_ROAD else [])
            endpoints.append([node_ids[ns[0]], node_ids[ns[-1]]] if ns else [0, 0])
            endpoint_masks.append(float(bool(ns)))
            tiles.append(tile_ids[value[0]] if kind == A.MOVE_ROBBER else 0)
            tile_masks.append(float(kind == A.MOVE_ROBBER))
            change = np.zeros(5)
            outgoing, incoming = np.zeros(5), np.zeros(5)
            if kind in costs:
                change -= costs[kind]
            elif kind == A.MARITIME_TRADE:
                change[RESOURCES.index(value[0])] -= len(value) - 1
                change[RESOURCES.index(value[-1])] += 1
                outgoing[RESOURCES.index(value[0])] = 1
                incoming[RESOURCES.index(value[-1])] = 1
            elif kind == A.DISCARD_RESOURCE:
                change[RESOURCES.index(value)] -= 1
                outgoing[RESOURCES.index(value)] = 1
            elif kind == A.PLAY_MONOPOLY:
                incoming[RESOURCES.index(value)] = 1
            elif kind == A.PLAY_YEAR_OF_PLENTY:
                for resource in value:
                    change[RESOURCES.index(resource)] += 1
                    incoming[RESOURCES.index(resource)] += .5
            delta.append(change)
            give.append(outgoing)
            take.append(incoming)
            count.append(abs(change).sum() / 4)
        env.close()
        for name, values, dtype in (("kinds", kind_ids, torch.long), ("endpoints", endpoints, torch.long),
                                    ("endpoint_masks", endpoint_masks, torch.float32), ("tiles", tiles, torch.long),
                                    ("tile_masks", tile_masks, torch.float32), ("delta", delta, torch.float32),
                                    ("give", give, torch.float32), ("take", take, torch.float32),
                                    ("count", count, torch.float32),
                                    ("targets", ((1, 1, 0, 0, 0), (1, 1, 1, 1, 0),
                                                 (0, 0, 0, 2, 3), (0, 0, 1, 1, 1)), torch.float32)):
            self.register_buffer(name, torch.as_tensor(np.asarray(values), dtype=dtype))
        self.kinds_count = len(kinds)
        self.kind = nn.Embedding(len(kinds), 8)
        self.context = nn.Sequential(nn.Linear(923, 64), nn.Tanh())
        self.kind_scores = nn.Linear(64, len(kinds))
        self.context_local = nn.Sequential(nn.Linear(64, 16), nn.Tanh())
        self.node_local = nn.Sequential(nn.Linear(32, 16), nn.Tanh())
        self.tile_local = nn.Sequential(nn.Linear(38, 16), nn.Tanh())
        self.resource_local = nn.Sequential(nn.Linear(21, 16), nn.Tanh())
        self.local = nn.Sequential(nn.Linear(16 + 32 + 16 + 16 + 8, 16), nn.Tanh(), nn.Linear(16, 1))
        if projects:
            self.latent_weights = nn.Linear(64, 4)
            self.latent_projects = nn.Embedding(4, 12)
            self.project_global = nn.Sequential(nn.Linear(16 + 12 + 20, 16), nn.Tanh())
            self.project_residual = nn.Sequential(nn.Linear(16 + 5 + 20 + 2, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.project_residual[-1].weight)
            nn.init.zeros_(self.project_residual[-1].bias)

    def resource_features(self, obs):
        hand = obs[:, 883:888]
        return torch.cat((hand[:, None].expand(-1, len(self.kinds), -1), self.give[None].expand(len(obs), -1, -1),
                          self.take[None].expand(len(obs), -1, -1), self.delta[None].expand(len(obs), -1, -1) / 4,
                          self.count[None, :, None].expand(len(obs), -1, -1)), -1)

    def project_features(self, obs):
        hand = obs[:, 883:888] * 19
        road = self.kinds == list(A).index(A.BUILD_ROAD)
        settlement = self.kinds == list(A).index(A.BUILD_SETTLEMENT)
        free = ((obs[:, 913:914] > 0) & (road | settlement)) | ((obs[:, 916:917] > 0) & road)
        remaining = hand[:, None] + self.delta[None] * (~free).unsqueeze(-1)
        deficits = (self.targets[None, None] - remaining[:, :, None]).clamp_min(0).flatten(2) / 3
        raw = self.graph.node_inputs(obs)
        nominal, blocked = raw[:, :, :5], raw[:, :, 5:10]
        buildings = obs[:, 495:711].reshape(-1, 54, 4)
        own = buildings[:, :, 0] + 2 * buildings[:, :, 1]
        enemy = buildings[:, :, 2] + 2 * buildings[:, :, 3]
        incomes = torch.cat(((own.unsqueeze(-1) * nominal).sum(1), (own.unsqueeze(-1) * blocked).sum(1),
                             (enemy.unsqueeze(-1) * nominal).sum(1), (enemy.unsqueeze(-1) * blocked).sum(1)), -1) / 10
        tiles = obs[:, :152].reshape(-1, 19, 8)
        threat = torch.stack(((own @ self.graph.incidence)[:, self.tiles],
                              (enemy @ self.graph.incidence)[:, self.tiles]), -1) * self.tile_masks[None, :, None] / 3
        return remaining / 19, deficits, incomes, threat

    def tile_features(self, obs, nodes):
        batch = len(obs)
        tiles = obs[:, :152].reshape(batch, 19, 8)
        pooled = (self.graph.incidence.T @ nodes) / self.graph.incidence.sum(0)[None, :, None].clamp_min(1)
        return torch.cat((tiles[:, self.tiles, :5], tiles[:, self.tiles, 7:8], pooled[:, self.tiles]), -1)

    def forward(self, obs):
        batch = len(obs)
        nodes = self.graph(obs)
        context = self.context(obs)
        node_embeddings = self.node_local(nodes)
        ends = node_embeddings[:, self.endpoints]
        endpoints = torch.cat((ends.mean(2), ends.amax(2)), -1) * self.endpoint_masks[None, :, None]
        tile = self.tile_local(self.tile_features(obs, nodes)) * self.tile_masks[None, :, None]
        resources = self.resource_local(self.resource_features(obs))
        local = torch.cat((self.context_local(context)[:, None].expand(-1, len(self.kinds), -1),
                           self.kind(self.kinds).expand(batch, -1, -1), endpoints, tile, resources), -1)
        scores = self.kind_scores(context)[:, self.kinds] + self.local(local).squeeze(-1)
        if self.projects:
            remaining, deficits, incomes, threat = self.project_features(obs)
            weights = self.latent_weights(context).softmax(-1) @ self.latent_projects.weight
            global_project = self.project_global(torch.cat((self.context_local(context), weights, incomes), -1))
            values = torch.cat((global_project[:, None].expand(-1, len(self.kinds), -1), remaining, deficits, threat), -1)
            scores = scores + self.project_residual(values).squeeze(-1)
        return scores


class GraphPolicy(MaskableActorCriticPolicy):
    """Graph-only ablation with the same actor/critic architecture as ProjectPolicy."""
    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorValue(923)

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (923,) or action_space.n != 332:
            raise ValueError("GraphPolicy requires observation/action format 3")
        self.action_net = GraphScorer()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class ProjectPolicy(GraphPolicy):
    """GraphPolicy plus four-way latent conditioning of exact resource-deficit features."""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.action_net = GraphScorer(projects=True)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
