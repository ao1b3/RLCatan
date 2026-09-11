"""Format 4: public player slots in relative turn order, padded to four."""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import MlpExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS, ActionPrompt, ActionType as A
from catanatron.models.map import build_map
from catanatron.models.board import get_edges
from catanatron.state_functions import player_key
from .game import ActionEncoder


PLAYER_SIZE = 197  # presence, turn, 54*2 buildings, 72 roads, 14 public, discard
GLOBAL_SIZE = 1320 - 4 * PLAYER_SIZE


def observe(env, color):
    state = env.game.state
    ps = state.player_state
    start = state.colors.index(color)
    colors = state.colors[start:] + state.colors[:start]
    global_features = list(env.static)
    global_features.extend(float(state.board.robber_coordinate == c) for c in env.coordinates)
    key = player_key(state, color)
    global_features.extend(ps[key + '_' + r + '_IN_HAND'] / 19 for r in RESOURCES)
    global_features.extend(ps[key + '_' + c + '_IN_HAND'] / 14 for c in DEVELOPMENT_CARDS)
    global_features.extend(float(ps[key + '_' + c + '_OWNED_AT_START']) for c in DEVELOPMENT_CARDS if c != 'VICTORY_POINT')
    global_features.extend((ps[key + '_ACTUAL_VICTORY_POINTS'] / 12,
                            float(ps[key + '_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN'])))
    global_features.extend(float(n > 0) for n in state.resource_freqdeck)
    global_features.append(len(state.development_listdeck) / 25)
    global_features.extend(float(state.current_prompt == p) for p in ActionPrompt)
    global_features.extend((float(state.is_initial_build_phase), float(state.is_discarding),
                            float(state.is_road_building), state.free_roads_available / 2,
                            env.config.target_vp / 12, min(state.num_turns / env.config.max_turns, 1),
                            min(env.actions / env.config.max_actions, 1), len(colors) / 4))
    slots = np.zeros((4, PLAYER_SIZE), dtype=np.float32)
    public = (('VICTORY_POINTS', 12), ('ROADS_AVAILABLE', 15), ('SETTLEMENTS_AVAILABLE', 5),
              ('CITIES_AVAILABLE', 4), ('HAS_ROAD', 1), ('HAS_ARMY', 1), ('HAS_ROLLED', 1),
              ('LONGEST_ROAD_LENGTH', 15))
    for index, owner in enumerate(colors):
        key = player_key(state, owner)
        values = [1., float(state.colors[state.current_turn_index] == owner)]
        for node in env.nodes:
            building = state.board.buildings.get(node)
            values.extend((building == (owner, 'SETTLEMENT'), building == (owner, 'CITY')))
        values.extend(state.board.roads.get(edge) == owner for edge in env.edges)
        values.extend(ps[key + '_' + name] / scale for name, scale in public)
        values.extend(ps[key + '_PLAYED_' + c] / 14 for c in DEVELOPMENT_CARDS if c != 'VICTORY_POINT')
        values.append(sum(ps[key + '_' + r + '_IN_HAND'] for r in RESOURCES) / 95)
        values.append(sum(ps[key + '_' + c + '_IN_HAND'] for c in DEVELOPMENT_CARDS) / 25)
        values.append(state.discard_counts[state.color_to_index[owner]] / 48)
        slots[index] = values
    return np.clip(np.concatenate((global_features, slots.ravel())).astype(np.float32), 0, 1)


class PlayerFeatures(BaseFeaturesExtractor):
    """Shared player weights, masked opponent pooling, and ordered target slots.

    Ordered embeddings retain the identities needed to select robber victims;
    pooled summaries give the critic a consistent view across player counts.
    """
    def __init__(self, observation_space):
        super().__init__(observation_space, 128 + 6 * 64)
        self.global_size = observation_space.shape[0] - 4 * PLAYER_SIZE
        self.board = nn.Sequential(nn.Linear(self.global_size, 128), nn.ReLU())
        self.player = nn.Sequential(nn.Linear(PLAYER_SIZE, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

    def forward(self, obs):
        slots = obs[:, self.global_size:].reshape(-1, 4, PLAYER_SIZE)
        present = slots[:, :, :1]
        encoded = self.player(slots) * present
        opponents = encoded[:, 1:]
        mean = opponents.sum(1) / present[:, 1:].sum(1).clamp_min(1)
        maximum = opponents.max(1).values
        return torch.cat((self.board(obs[:, :self.global_size]), encoded.flatten(1), mean, maximum), dim=1)


class PlayerProductionFeatures(PlayerFeatures):
    """The baseline embedding plus public nominal and robber-adjusted production."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        if observation_space.shape != (1320,):
            raise ValueError("PlayerProductionFeatures requires observation format 4")
        board = build_map("BASE")
        coordinates, nodes = sorted(board.land_tiles), sorted(board.land_nodes)
        incidence = np.array([[node in board.land_tiles[c].nodes.values() for c in coordinates]
                              for node in nodes], dtype=np.float32)
        self.register_buffer("incidence", torch.as_tensor(incidence))
        # 54 nodes × 5 resources × nominal/robber-adjusted, then 4 player totals.
        self.production = nn.Sequential(nn.Linear(54 * 5 * 2 + 4 * 5 * 2, 64), nn.Tanh(), nn.Linear(64, 512))
        nn.init.zeros_(self.production[-1].weight)
        nn.init.zeros_(self.production[-1].bias)

    def production_features(self, obs):
        tiles = obs[:, :152].reshape(-1, 19, 8)
        resource_yields = tiles[:, :, :5] * tiles[:, :, 7:8]
        nominal = self.incidence @ resource_yields
        robber = obs[:, 476:495]
        adjusted = self.incidence @ (resource_yields * (1 - robber).unsqueeze(-1))
        buildings = obs[:, GLOBAL_SIZE:].reshape(-1, 4, PLAYER_SIZE)[:, :, 2:110].reshape(-1, 4, 54, 2)
        weights = buildings[:, :, :, :1] + 2 * buildings[:, :, :, 1:]
        totals = torch.cat(((weights * nominal[:, None]).sum(2), (weights * adjusted[:, None]).sum(2)), -1)
        return torch.cat((nominal.flatten(1) / 3, adjusted.flatten(1) / 3, totals.flatten(1) / 30), -1)

    def forward(self, obs):
        return super().forward(obs) + self.production(self.production_features(obs))


class RawPlayerProductionFeatures(PlayerProductionFeatures):
    """Opt-in candidate-policy extractor: 512 transferred features plus raw public state."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self._features_dim += observation_space.shape[0]

    def forward(self, obs):
        return torch.cat((super().forward(obs), obs), -1)


class PlayerProductionPolicy(MaskableActorCriticPolicy):
    """Opt-in format-4 production policy with a transfer-neutral residual."""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs["features_extractor_class"] = PlayerProductionFeatures
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (1320,) or action_space.n != 370:
            raise ValueError("PlayerProductionPolicy requires observation/action format 4")
        for extractor in (self.features_extractor, self.pi_features_extractor, self.vf_features_extractor):
            nn.init.zeros_(extractor.production[-1].weight)
            nn.init.zeros_(extractor.production[-1].bias)


class RawMlpExtractor(MlpExtractor):
    """Keep baseline MLP tensors while carrying raw format-4 observations to the actor."""
    def __init__(self, net_arch, activation_fn, device, raw_size=1320):
        super().__init__(512, net_arch, activation_fn, device)
        self.latent_dim_pi += raw_size

    def forward_actor(self, features):
        return torch.cat((super().forward_actor(features[:, :512]), features[:, 512:]), -1)

    def forward_critic(self, features):
        return super().forward_critic(features[:, :512])


class MultiplayerCandidateNet(nn.Linear):
    """Dense baseline logits plus a zero-start shared location/robber residual."""
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        board = build_map("BASE")
        coordinates, nodes = sorted(board.land_tiles), sorted(board.land_nodes)
        encoder = ActionEncoder(True)
        node_ids = {node: i for i, node in enumerate(nodes)}
        tile_ids = {tile: i for i, tile in enumerate(coordinates)}
        kinds = list(A)
        kind_ids, endpoints, endpoint_masks, tiles, tile_masks, victims, victim_masks = [], [], [], [], [], [], []
        for kind, value in encoder.actions:
            kind_ids.append(kinds.index(kind))
            ns = ([value] if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY) else
                  list(value) if kind == A.BUILD_ROAD else [])
            endpoints.append([node_ids[ns[0]], node_ids[ns[-1]]] if ns else [0, 0])
            endpoint_masks.append(float(bool(ns)))
            tiles.append(tile_ids[value[0]] if kind == A.MOVE_ROBBER else 0)
            tile_masks.append(float(kind == A.MOVE_ROBBER))
            victim = value[1] if kind == A.MOVE_ROBBER else None
            victims.append((encoder.colors.index(victim) if victim is not None else 0))
            victim_masks.append(float(victim is not None))
        for name, values, dtype in (("kinds", kind_ids, torch.long), ("endpoints", endpoints, torch.long),
                                    ("endpoint_masks", endpoint_masks, torch.float32), ("tiles", tiles, torch.long),
                                    ("tile_masks", tile_masks, torch.float32), ("victims", victims, torch.long),
                                    ("victim_masks", victim_masks, torch.float32)):
            self.register_buffer(name, torch.as_tensor(values, dtype=dtype))
        incidence = np.array([[node in board.land_tiles[c].nodes.values() for c in coordinates]
                              for node in nodes], dtype=np.float32)
        self.register_buffer("incidence", torch.as_tensor(incidence))
        edges = sorted(tuple(sorted(edge)) for edge in get_edges(board.land_nodes))
        edge_incidence = np.array([[node in edge for edge in edges] for node in nodes], dtype=np.float32)
        self.register_buffer("edge_incidence", torch.as_tensor(edge_incidence))
        self.kind = nn.Embedding(len(kinds), 8)
        self.context = nn.Sequential(nn.Linear(in_features, 64), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(64 + 8 + 80, 64), nn.Tanh(), nn.Linear(64, 1))

    def local_features(self, obs):
        batch = len(obs)
        tiles = obs[:, :152].reshape(batch, 19, 8)
        yields = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8])
        robber = obs[:, 476:495]
        adjusted = self.incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8] * (1 - robber).unsqueeze(-1))
        ends = torch.cat((yields[:, self.endpoints].mean(2), adjusted[:, self.endpoints].mean(2)), -1)
        ends *= self.endpoint_masks[None, :, None]
        slots = obs[:, GLOBAL_SIZE:].reshape(batch, 4, PLAYER_SIZE)
        building_kinds = slots[:, :, 2:110].reshape(batch, 4, 54, 2)
        buildings = building_kinds.sum(-1)
        ownership = buildings[:, :, self.endpoints].permute(0, 2, 3, 1).flatten(2)
        ownership *= self.endpoint_masks[None, :, None]
        ports = obs[:, 152:476].reshape(batch, 54, 6)[:, self.endpoints].flatten(2)
        ports *= self.endpoint_masks[None, :, None]
        roads = slots[:, :, 110:182].reshape(batch, 4, 72)
        access = torch.einsum("ne,bpe->bpn", self.edge_incidence, roads)
        own_access = access[:, :1, self.endpoints].permute(0, 2, 3, 1).flatten(2)
        enemy_access = access[:, 1:, self.endpoints].amax(1).unsqueeze(-1)
        road_access = torch.cat((own_access, enemy_access.flatten(2)), -1) * self.endpoint_masks[None, :, None]
        tile_features = torch.cat((tiles[:, self.tiles, :5], tiles[:, self.tiles, 7:8]), -1) * self.tile_masks[None, :, None]
        tile_buildings = torch.einsum("nt,bpn->bpt", self.incidence,
                                      building_kinds[:, :, :, :1].squeeze(-1) + 2 * building_kinds[:, :, :, 1])
        denied = (tile_buildings[:, :, self.tiles].permute(0, 2, 1).unsqueeze(-1)
                  * (tiles[:, self.tiles, :5] * tiles[:, self.tiles, 7:8]).unsqueeze(2))
        denied *= self.tile_masks[None, :, None, None]
        target_denied = denied.gather(2, self.victims[None, :, None, None].expand(batch, -1, 1, 5)).squeeze(2)
        target_denied *= self.victim_masks[None, :, None]
        denied = denied.flatten(2)
        target = slots[:, self.victims, 182:] * self.victim_masks[None, :, None]
        return torch.cat((ends, ownership, ports, road_access, tile_features, denied, target_denied, target), -1)

    def forward(self, latent):
        base, obs = latent[:, :self.in_features], latent[:, self.in_features:]
        logits = super().forward(base)
        local = self.local_features(obs)
        batch = len(obs)
        context = self.context(base)[:, None].expand(-1, len(self.kinds), -1)
        features = torch.cat((context, self.kind(self.kinds).expand(batch, -1, -1), local), -1)
        return logits + self.candidate(features).squeeze(-1)


class MultiplayerCandidatePolicy(MaskableActorCriticPolicy):
    """Format-4 policy retaining transferable dense actor/critic/action tensors."""
    def _build_mlp_extractor(self):
        self.mlp_extractor = RawMlpExtractor(self.net_arch, self.activation_fn, self.device)

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        kwargs["features_extractor_class"] = RawPlayerProductionFeatures
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if observation_space.shape != (1320,) or action_space.n != 370:
            raise ValueError("MultiplayerCandidatePolicy requires observation/action format 4")
        self.action_net = MultiplayerCandidateNet(self.mlp_extractor.latent_dim_pi - 1320, action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        for extractor in (self.features_extractor, self.pi_features_extractor, self.vf_features_extractor):
            nn.init.zeros_(extractor.production[-1].weight)
            nn.init.zeros_(extractor.production[-1].bias)
        nn.init.zeros_(self.action_net.candidate[-1].weight)
        nn.init.zeros_(self.action_net.candidate[-1].bias)
