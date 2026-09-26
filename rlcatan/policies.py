"""The networks that score actions.

Every network reads the current observation only. It has no memory of past
actions and it cannot see a hidden hand.

A network scores a place or a resource, not an action number. The same weights
serve every node and every tile. A model can therefore judge a node that it has
never built on.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType as A, RESOURCES

from .game import (BUILDINGS, COORDINATES, EDGES, HAND, INCIDENCE, NODES, OBSERVATION_SIZE,
                   PLAYER_SIZE, PORTS, PUBLIC, ROADS, ROAD_BUILDING, ROBBER, SETUP_PHASE,
                   TILES, action_table)

# The cost of each project, in the resource order of RESOURCES.
COSTS = {A.BUILD_ROAD: (1, 1, 0, 0, 0), A.BUILD_SETTLEMENT: (1, 1, 1, 1, 0),
         A.BUILD_CITY: (0, 0, 0, 2, 3), A.BUY_DEVELOPMENT_CARD: (0, 0, 1, 1, 1)}
KINDS = list(A)


def action_tables(multiplayer=False):
    """Return the fixed tables that describe every action.

    Each row is one action. The tables say which kind of action it is, which
    nodes it touches, which tile the robber moves to, and which resources it
    gives away or takes. A mask is 1 when the field applies to that action.
    """
    node_ids = {node: index for index, node in enumerate(NODES)}
    tile_ids = {coordinate: index for index, coordinate in enumerate(COORDINATES)}
    kinds, endpoints, endpoint_masks, tiles, tile_masks, give, take = [], [], [], [], [], [], []
    for kind, value in action_table(multiplayer):
        kinds.append(KINDS.index(kind))
        ends = ([value] if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY) else
                list(value) if kind == A.BUILD_ROAD else [])
        endpoints.append([node_ids[ends[0]], node_ids[ends[-1]]] if ends else [0, 0])
        endpoint_masks.append(float(bool(ends)))
        tiles.append(tile_ids[value[0]] if kind == A.MOVE_ROBBER else 0)
        tile_masks.append(float(kind == A.MOVE_ROBBER))
        out, into = np.zeros(5), np.zeros(5)
        if kind == A.MARITIME_TRADE:
            out[RESOURCES.index(value[0])] = 1
            into[RESOURCES.index(value[-1])] = 1
        elif kind == A.DISCARD_RESOURCE:
            out[RESOURCES.index(value)] = 1
        elif kind == A.PLAY_MONOPOLY:
            into[RESOURCES.index(value)] = 1
        elif kind == A.PLAY_YEAR_OF_PLENTY:
            for resource in value:
                into[RESOURCES.index(resource)] += .5
        give.append(out)
        take.append(into)
    return {"kinds": np.asarray(kinds, np.int64),
            "endpoints": np.asarray(endpoints, np.int64),
            "endpoint_masks": np.asarray(endpoint_masks, np.float32),
            "tiles": np.asarray(tiles, np.int64),
            "tile_masks": np.asarray(tile_masks, np.float32),
            "give": np.asarray(give, np.float32),
            "take": np.asarray(take, np.float32)}


def building_weights(obs):
    """Return how much each node pays its owner, and each enemy node its owner.

    A city pays twice as much as a settlement.
    """
    buildings = obs[:, BUILDINGS].reshape(len(obs), 54, 4)
    return (buildings[:, :, 0] + 2 * buildings[:, :, 1],
            buildings[:, :, 2] + 2 * buildings[:, :, 3])


def register_tables(module, names=None):
    """Store the action tables on a network. Torch saves them with the weights.

    Pass names to store a table under an older name. The name must match the
    name inside the saved models.
    """
    for name, values in action_tables().items():
        module.register_buffer(names.get(name, name) if names else name, torch.as_tensor(values))


class ProductionFeatures(BaseFeaturesExtractor):
    """Add the resource income of every node to the observation.

    The network could work this income out from the tiles. This saves it the
    work.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0] + len(NODES) * 5 + 10)
        self.register_buffer("incidence", torch.as_tensor(INCIDENCE))

    def forward(self, observations):
        tiles = observations[:, TILES].reshape(-1, 19, 8)
        yields = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8])
        own, enemy = building_weights(observations)
        own_yields = (yields * own.unsqueeze(-1)).sum(dim=1) / 30
        enemy_yields = (yields * enemy.unsqueeze(-1)).sum(dim=1) / 30
        return torch.cat((observations, yields.flatten(1) / 3, own_yields, enemy_yields), dim=1)


class ActorValue(nn.Module):
    """Pass the features straight to the actor. Give the critic its own layers."""

    def __init__(self, size):
        super().__init__()
        self.latent_dim_pi, self.latent_dim_vf = size, 128
        self.value = nn.Sequential(nn.Linear(size, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh())

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.value(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class ActionScorer(nn.Module):
    """Score every action from the kind of action and from its place."""

    def __init__(self, observation_space):
        super().__init__()
        self.production = ProductionFeatures(observation_space)
        register_tables(self, {"endpoints": "nodes", "endpoint_masks": "node_masks"})
        # Saved models hold this table, so it must stay. The network does not
        # read it.
        costs = [(len(value) - 1) / 4 if kind == A.MARITIME_TRADE else 0
                 for kind, value in action_table(False)]
        self.register_buffer("costs", torch.as_tensor(costs, dtype=torch.float32))
        self.kind_embedding = nn.Embedding(len(KINDS), 8)
        self.context = nn.Sequential(nn.Linear(78, 32), nn.Tanh())
        self.kind_scores = nn.Linear(32, len(KINDS))
        self.local_scores = nn.Sequential(nn.Linear(32 + 8 + 20, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, observations):
        batch = observations.shape[0]
        derived = self.production(observations)
        yields = derived[:, OBSERVATION_SIZE:-10].reshape(batch, 54, 5)
        endpoints = yields[:, self.nodes]
        local = endpoints.mean(dim=2) * self.node_masks[None, :, None]
        best = endpoints.sum(dim=-1).max(dim=2).values * self.node_masks
        own, enemy = building_weights(observations)
        own_robber = (own @ self.production.incidence)[:, self.tiles] * self.tile_masks / 6
        enemy_robber = (enemy @ self.production.incidence)[:, self.tiles] * self.tile_masks / 6
        hand = observations[:, HAND]
        context = self.context(torch.cat((observations[:, PUBLIC:], derived[:, -10:]), dim=1))
        local_features = torch.cat((local, best.unsqueeze(-1), own_robber.unsqueeze(-1),
                                    enemy_robber.unsqueeze(-1), self.give.expand(batch, -1, -1),
                                    self.take.expand(batch, -1, -1),
                                    (hand @ self.give.T).unsqueeze(-1),
                                    (hand @ self.take.T).unsqueeze(-1)), dim=-1)
        features = torch.cat((context[:, None].expand(-1, len(self.kinds), -1),
                              self.kind_embedding(self.kinds).expand(batch, -1, -1),
                              local_features), dim=-1)
        return self.kind_scores(context)[:, self.kinds] + self.local_scores(features).squeeze(-1)


class ActionFeaturePolicy(MaskableActorCriticPolicy):
    """The two player policy that scores actions by place and by resource."""

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorValue(self.features_dim)

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (OBSERVATION_SIZE,) or action_space.n != 332:
            raise ValueError("ActionFeaturePolicy needs observation format 3")
        self.action_net = ActionScorer(observation_space)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class BoardGraph(nn.Module):
    """Pass messages twice between neighbouring nodes of the board.

    Each node starts with what it yields, its port, its buildings and its
    roads. It then takes in the state of its neighbours. Two rounds are enough
    to see a road two steps away.
    """

    def __init__(self):
        super().__init__()
        adjacency = np.array([[STATIC_GRAPH.has_edge(a, b) for b in NODES] for a in NODES],
                             dtype=np.float32)
        edge_incidence = np.array([[node in edge for edge in EDGES] for node in NODES],
                                  dtype=np.float32)
        self.register_buffer("incidence", torch.as_tensor(INCIDENCE))
        self.register_buffer("adjacency", torch.as_tensor(adjacency))
        self.register_buffer("edge_incidence", torch.as_tensor(edge_incidence))
        self.input = nn.Sequential(nn.Linear(22, 32), nn.Tanh())
        self.message1 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.message2 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())

    def node_inputs(self, obs):
        """Return what each node is worth, before any message passes."""
        batch = len(obs)
        tiles = obs[:, TILES].reshape(batch, 19, 8)
        yields = tiles[:, :, :5] * tiles[:, :, 7:8]
        nominal = self.incidence @ yields
        robber = obs[:, ROBBER]
        blocked = self.incidence @ (yields * (1 - robber).unsqueeze(-1))
        ports = obs[:, PORTS].reshape(batch, 54, 6)
        buildings = obs[:, BUILDINGS].reshape(batch, 54, 4)
        roads = obs[:, ROADS].reshape(batch, 72, 2)
        access = (self.edge_incidence @ roads) / 3
        return torch.cat((nominal / 3, blocked / 3, ports, buildings, access), -1)

    def forward(self, obs):
        nodes = self.input(self.node_inputs(obs))
        degree = self.adjacency.sum(-1).clamp_min(1)
        for layer in (self.message1, self.message2):
            nodes = layer(torch.cat((nodes, self.adjacency @ nodes / degree[None, :, None]), -1))
        return nodes


class GraphScorer(nn.Module):
    """Score every action from the board graph and from the next project.

    The project part asks one question: how many resources are still missing
    for a road, a settlement, a city or a development card. It starts at zero,
    so a model that transfers into this network begins unchanged.
    """

    def __init__(self):
        super().__init__()
        self.graph = BoardGraph()
        register_tables(self)
        delta, count = [], []
        for kind, value in action_table(False):
            change = np.zeros(5)
            if kind in COSTS:
                change -= COSTS[kind]
            elif kind == A.MARITIME_TRADE:
                change[RESOURCES.index(value[0])] -= len(value) - 1
                change[RESOURCES.index(value[-1])] += 1
            elif kind == A.DISCARD_RESOURCE:
                change[RESOURCES.index(value)] -= 1
            elif kind == A.PLAY_YEAR_OF_PLENTY:
                for resource in value:
                    change[RESOURCES.index(resource)] += 1
            delta.append(change)
            count.append(abs(change).sum() / 4)
        self.register_buffer("delta", torch.as_tensor(np.asarray(delta), dtype=torch.float32))
        self.register_buffer("count", torch.as_tensor(count, dtype=torch.float32))
        self.register_buffer("targets", torch.as_tensor(
            [COSTS[kind] for kind in (A.BUILD_ROAD, A.BUILD_SETTLEMENT,
                                      A.BUILD_CITY, A.BUY_DEVELOPMENT_CARD)], dtype=torch.float32))
        self.kind = nn.Embedding(len(KINDS), 8)
        self.context = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.Tanh())
        self.kind_scores = nn.Linear(64, len(KINDS))
        self.context_local = nn.Sequential(nn.Linear(64, 16), nn.Tanh())
        self.node_local = nn.Sequential(nn.Linear(32, 16), nn.Tanh())
        self.tile_local = nn.Sequential(nn.Linear(38, 16), nn.Tanh())
        self.resource_local = nn.Sequential(nn.Linear(21, 16), nn.Tanh())
        self.local = nn.Sequential(nn.Linear(16 + 32 + 16 + 16 + 8, 16), nn.Tanh(), nn.Linear(16, 1))
        self.latent_weights = nn.Linear(64, 4)
        self.latent_projects = nn.Embedding(4, 12)
        self.project_global = nn.Sequential(nn.Linear(16 + 12 + 20, 16), nn.Tanh())
        self.project_residual = nn.Sequential(nn.Linear(16 + 5 + 20 + 2, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.project_residual[-1].weight)
        nn.init.zeros_(self.project_residual[-1].bias)

    def resource_features(self, obs):
        hand = obs[:, HAND]
        batch = len(obs)
        return torch.cat((hand[:, None].expand(-1, len(self.kinds), -1),
                          self.give[None].expand(batch, -1, -1),
                          self.take[None].expand(batch, -1, -1),
                          self.delta[None].expand(batch, -1, -1) / 4,
                          self.count[None, :, None].expand(batch, -1, -1)), -1)

    def project_features(self, obs):
        """Return what is left in hand, what is missing, the income and the threat."""
        hand = obs[:, HAND] * 19
        road = self.kinds == KINDS.index(A.BUILD_ROAD)
        settlement = self.kinds == KINDS.index(A.BUILD_SETTLEMENT)
        # Opening roads and settlements are free. Road Building roads are free.
        free = ((obs[:, SETUP_PHASE] > 0) & (road | settlement)) | ((obs[:, ROAD_BUILDING] > 0) & road)
        remaining = hand[:, None] + self.delta[None] * (~free).unsqueeze(-1)
        deficits = (self.targets[None, None] - remaining[:, :, None]).clamp_min(0).flatten(2) / 3
        raw = self.graph.node_inputs(obs)
        nominal, blocked = raw[:, :, :5], raw[:, :, 5:10]
        own, enemy = building_weights(obs)
        incomes = torch.cat(((own.unsqueeze(-1) * nominal).sum(1), (own.unsqueeze(-1) * blocked).sum(1),
                             (enemy.unsqueeze(-1) * nominal).sum(1),
                             (enemy.unsqueeze(-1) * blocked).sum(1)), -1) / 10
        threat = torch.stack(((own @ self.graph.incidence)[:, self.tiles],
                              (enemy @ self.graph.incidence)[:, self.tiles]), -1)
        threat = threat * self.tile_masks[None, :, None] / 3
        return remaining / 19, deficits, incomes, threat

    def tile_features(self, obs, nodes):
        batch = len(obs)
        tiles = obs[:, TILES].reshape(batch, 19, 8)
        pooled = (self.graph.incidence.T @ nodes) / self.graph.incidence.sum(0)[None, :, None].clamp_min(1)
        return torch.cat((tiles[:, self.tiles, :5], tiles[:, self.tiles, 7:8], pooled[:, self.tiles]), -1)

    def forward(self, obs):
        batch = len(obs)
        nodes = self.graph(obs)
        context = self.context(obs)
        ends = self.node_local(nodes)[:, self.endpoints]
        endpoints = torch.cat((ends.mean(2), ends.amax(2)), -1) * self.endpoint_masks[None, :, None]
        tile = self.tile_local(self.tile_features(obs, nodes)) * self.tile_masks[None, :, None]
        resources = self.resource_local(self.resource_features(obs))
        local = torch.cat((self.context_local(context)[:, None].expand(-1, len(self.kinds), -1),
                           self.kind(self.kinds).expand(batch, -1, -1), endpoints, tile, resources), -1)
        scores = self.kind_scores(context)[:, self.kinds] + self.local(local).squeeze(-1)
        remaining, deficits, incomes, threat = self.project_features(obs)
        weights = self.latent_weights(context).softmax(-1) @ self.latent_projects.weight
        shared = self.project_global(torch.cat((self.context_local(context), weights, incomes), -1))
        values = torch.cat((shared[:, None].expand(-1, len(self.kinds), -1),
                            remaining, deficits, threat), -1)
        return scores + self.project_residual(values).squeeze(-1)


class ProjectPolicy(MaskableActorCriticPolicy):
    """The two player policy that reads the board as a graph."""

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActorValue(OBSERVATION_SIZE)

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (OBSERVATION_SIZE,) or action_space.n != 332:
            raise ValueError("ProjectPolicy needs observation format 3")
        self.action_net = GraphScorer()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class PlayerFeatures(BaseFeaturesExtractor):
    """Read the player blocks of a format 4 observation.

    One small network reads every block, with the same weights each time. The
    opponent blocks are also joined into a mean and a maximum. The network
    therefore works for two, three or four players.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, 128 + 6 * 64)
        self.global_size = observation_space.shape[0] - 4 * PLAYER_SIZE
        self.board = nn.Sequential(nn.Linear(self.global_size, 128), nn.ReLU())
        self.player = nn.Sequential(nn.Linear(PLAYER_SIZE, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU())

    def forward(self, obs):
        slots = obs[:, self.global_size:].reshape(-1, 4, PLAYER_SIZE)
        present = slots[:, :, :1]
        encoded = self.player(slots) * present
        opponents = encoded[:, 1:]
        mean = opponents.sum(1) / present[:, 1:].sum(1).clamp_min(1)
        maximum = opponents.max(1).values
        return torch.cat((self.board(obs[:, :self.global_size]), encoded.flatten(1), mean, maximum), dim=1)
