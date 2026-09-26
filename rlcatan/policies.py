import numpy as np
import torch
from functools import lru_cache
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType as A, RESOURCES

from .game import (BUILDINGS, COLORS, COORDINATES, COUNTED_SIZE, EDGES, EDGE_INDEX,
                   ENEMY_HAND, ENEMY_UNKNOWN, HAND, INCIDENCE, NODES, OBSERVATION_SIZE,
                   PORTS, PUBLIC, ROADS, ROAD_LENGTH, ROAD_TITLE, ROBBER, SEATS, SEAT_BLOCK,
                   SEAT_BUILDINGS, SEAT_DISCARD, SEAT_HAND, SEAT_PRESENT, SEAT_PUBLIC,
                   SEAT_ROADS, SEAT_SIZE, SEAT_UNKNOWN, TILES, TRADE_ASKING, TRADE_OFFERER,
                   TRADE_OFFERING, TRADE_SIZE, action_table)

COSTS = {A.BUILD_ROAD: (1, 1, 0, 0, 0), A.BUILD_SETTLEMENT: (1, 1, 1, 1, 0),
         A.BUILD_CITY: (0, 0, 0, 2, 3), A.BUY_DEVELOPMENT_CARD: (0, 0, 1, 1, 1)}
KINDS = list(A)
KIND_IDS = {kind: index for index, kind in enumerate(KINDS)}


@lru_cache(maxsize=4)
def _action_metadata(relative=False, trading=False):
    node_ids = {node: i for i, node in enumerate(NODES)}
    tile_ids = {coordinate: i for i, coordinate in enumerate(COORDINATES)}
    kinds, endpoints, endpoint_masks, tiles, tile_masks, give, take, victims = [], [], [], [], [], [], [], []
    delta, count, edges, edge_masks = [], [], [], []
    for kind, value in action_table(relative, trading):
        ends = ([value] if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY)
                else list(value) if kind == A.BUILD_ROAD else [])
        kinds.append(KIND_IDS[kind])
        endpoints.append([node_ids[ends[0]], node_ids[ends[-1]]] if ends else [0, 0])
        endpoint_masks.append(float(bool(ends)))
        tiles.append(tile_ids[value[0]] if kind == A.MOVE_ROBBER else 0)
        tile_masks.append(float(kind == A.MOVE_ROBBER))
        victims.append(COLORS.index(value[1]) if kind == A.MOVE_ROBBER and value[1] is not None
                       else COLORS.index(value) if kind == A.CONFIRM_TRADE else 0)
        out, into = np.zeros(5), np.zeros(5)
        if kind == A.MARITIME_TRADE:
            out[RESOURCES.index(value[0])] = 1
            into[RESOURCES.index(value[-1])] = 1
        elif kind == A.OFFER_TRADE:
            out[:], into[:] = value[:5], value[5:]
        elif kind == A.DISCARD_RESOURCE:
            out[RESOURCES.index(value)] = 1
        elif kind == A.PLAY_MONOPOLY:
            into[RESOURCES.index(value)] = 1
        elif kind == A.PLAY_YEAR_OF_PLENTY:
            for resource in value:
                into[RESOURCES.index(resource)] += .5
        change = np.zeros(5)
        if kind in COSTS:
            change -= COSTS[kind]
        elif kind == A.OFFER_TRADE:
            change += into - out
        elif kind == A.MARITIME_TRADE:
            change[RESOURCES.index(value[0])] -= sum(x is not None for x in value[:-1])
            change[RESOURCES.index(value[-1])] += 1
        elif kind == A.DISCARD_RESOURCE:
            change[RESOURCES.index(value)] -= 1
        elif kind == A.PLAY_YEAR_OF_PLENTY:
            for resource in value:
                change[RESOURCES.index(resource)] += 1
        delta.append(change)
        count.append(abs(change).sum() / 4)
        edges.append(EDGE_INDEX[tuple(sorted(value))] if kind == A.BUILD_ROAD else 0)
        edge_masks.append(float(kind == A.BUILD_ROAD))
        give.append(out)
        take.append(into)
    return {"kinds": np.asarray(kinds, np.int64), "endpoints": np.asarray(endpoints, np.int64),
            "endpoint_masks": np.asarray(endpoint_masks, np.float32), "tiles": np.asarray(tiles, np.int64),
            "tile_masks": np.asarray(tile_masks, np.float32), "give": np.asarray(give, np.float32),
            "take": np.asarray(take, np.float32), "victims": np.asarray(victims, np.int64),
            "delta": np.asarray(delta, np.float32), "count": np.asarray(count, np.float32),
            "edges": np.asarray(edges, np.int64), "edge_masks": np.asarray(edge_masks, np.float32)}


def action_tables(relative=False, trading=False):
    return {name: _action_metadata(relative, trading)[name].copy() for name in
            ("kinds", "endpoints", "endpoint_masks", "tiles", "tile_masks", "give", "take", "victims")}


def building_weights(obs):
    buildings = obs[:, BUILDINGS].reshape(len(obs), 54, 4)
    return buildings[:, :, 0] + 2 * buildings[:, :, 1], buildings[:, :, 2] + 2 * buildings[:, :, 3]


def resource_deltas(relative=False, trading=False):
    meta = _action_metadata(relative, trading)
    return meta["delta"].copy(), meta["count"].copy()


def port_values(obs, incidence):
    batch = len(obs)
    tiles = obs[:, TILES].reshape(batch, 19, 8)
    nominal = incidence @ (tiles[:, :, :5] * tiles[:, :, 7:8])
    ports = obs[:, PORTS].reshape(batch, 54, 6)
    own, _ = building_weights(obs)
    after = (own.unsqueeze(-1) * nominal).sum(1, keepdim=True) + nominal
    return torch.cat(((after * ports[:, :, :5]).sum(-1, keepdim=True) / 3,
                      after.sum(-1, keepdim=True) * ports[:, :, 5:] / 10), -1)


def edge_table(relative=False, trading=False):
    meta = _action_metadata(relative, trading)
    return meta["edges"].copy(), meta["edge_masks"].copy()


def register_tables(module, names=None, relative=False, persistent=True, trading=False):
    for name, values in action_tables(relative, trading).items():
        module.register_buffer(names.get(name, name) if names else name, torch.as_tensor(values),
                               persistent=persistent and name != "victims")


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feed = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, tokens, bias):
        normed = self.norm1(tokens)
        tokens = tokens + self.attention(normed, normed, normed, attn_mask=bias, need_weights=False)[0]
        return tokens + self.feed(self.norm2(tokens))


class BoardTransformer(nn.Module):
    NODE_END, TILE_END, TOKENS = len(NODES), len(NODES) + len(COORDINATES), len(NODES) + len(COORDINATES) + 1
    PLAYER_FEATURES = 1 + 14 + 1 + 5 + 1 + 1

    def __init__(self, width=64, heads=4, layers=3, counted=False, seats=False, trading=False):
        super().__init__()
        self.counted, self.seats, self.trading = counted, seats, trading
        adjacency = np.array([[STATIC_GRAPH.has_edge(a, b) for b in NODES] for a in NODES], dtype=np.float32)
        edge_incidence = np.array([[node in edge for edge in EDGES] for node in NODES], dtype=np.float32)
        self.register_buffer("incidence", torch.as_tensor(INCIDENCE))
        self.register_buffer("adjacency", torch.as_tensor(adjacency))
        self.register_buffer("edge_incidence", torch.as_tensor(edge_incidence))
        self.latent_dim_pi, self.latent_dim_vf, self.heads = width, 128, heads
        two_steps = ((adjacency @ adjacency) > 0) & (adjacency == 0) & ~np.eye(len(NODES), dtype=bool)
        tiles_touch = (INCIDENCE.T @ INCIDENCE) >= 2
        relation = np.zeros((self.TOKENS, self.TOKENS), np.int64)
        relation[:self.NODE_END, :self.NODE_END] = 2 * (adjacency > 0) + 3 * two_steps
        relation[:self.NODE_END, self.NODE_END:self.TILE_END] = 4 * INCIDENCE
        relation[self.NODE_END:self.TILE_END, :self.NODE_END] = 4 * INCIDENCE.T
        relation[self.NODE_END:self.TILE_END, self.NODE_END:self.TILE_END] = 5 * tiles_touch
        relation[-1, :] = relation[:, -1] = 6
        np.fill_diagonal(relation, 1)
        if seats:
            players = np.full((self.TOKENS + 4, self.TOKENS + 4), 7, np.int64)
            players[:self.TOKENS, :self.TOKENS] = relation
            players[self.TOKENS:, self.TOKENS:] = 8
            np.fill_diagonal(players, 1)
            relation = players
        self.register_buffer("relation", torch.as_tensor(relation))
        self.bias = nn.Embedding(9 if seats else 7, heads)
        nn.init.zeros_(self.bias.weight)
        self.node_in = nn.Linear(22 + 2 * counted + 9 * seats, width)
        self.tile_in = nn.Linear(9, width)
        self.global_in = nn.Linear(OBSERVATION_SIZE - PUBLIC + 6 * counted + (TRADE_SIZE - SEAT_SIZE) * trading, width)
        self.position = nn.Parameter(torch.randn(self.TILE_END, width) * .02)
        if seats:
            self.player_in = nn.Linear(self.PLAYER_FEATURES, width)
            self.player_position = nn.Parameter(torch.randn(4, width) * .02)
        self.layers = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.value = nn.Sequential(nn.Linear(4 * width + width * seats, 128), nn.GELU(),
                                   nn.Linear(128, 128), nn.GELU())

    def node_inputs(self, obs):
        batch = len(obs)
        tiles = obs[:, TILES].reshape(batch, 19, 8)
        yields = tiles[:, :, :5] * tiles[:, :, 7:8]
        nominal = self.incidence @ yields
        blocked = self.incidence @ (yields * (1 - obs[:, ROBBER]).unsqueeze(-1))
        ports = obs[:, PORTS].reshape(batch, 54, 6)
        buildings = obs[:, BUILDINGS].reshape(batch, 54, 4)
        roads = obs[:, ROADS].reshape(batch, 72, 2)
        inputs = [nominal / 3, blocked / 3, ports, buildings, (self.edge_incidence @ roads) / 3]
        if self.counted:
            inputs.append(port_values(obs, self.incidence))
        if self.seats:
            blocks = obs[:, SEATS].reshape(batch, 3, SEAT_BLOCK)
            inputs += [blocks[:, :, SEAT_BUILDINGS].reshape(batch, 3, 54, 2).permute(0, 2, 1, 3).flatten(2),
                       (self.edge_incidence @ blocks[:, :, SEAT_ROADS].permute(0, 2, 1)) / 3]
        return torch.cat(inputs, -1)

    def players(self, obs):
        batch = len(obs)
        blocks = obs[:, SEATS].reshape(batch, 3, SEAT_BLOCK)
        rolled = obs[:, PUBLIC + 6:PUBLIC + 7]
        me = torch.cat((torch.ones(batch, 1, device=obs.device), obs[:, PUBLIC:PUBLIC + 14], obs[:, 917:918],
                        obs[:, HAND], torch.zeros(batch, 1, device=obs.device), rolled), 1)
        present = torch.cat((torch.ones(batch, 1, device=obs.device), blocks[:, :, SEAT_PRESENT]), 1)
        others = torch.cat((blocks[:, :, SEAT_PRESENT:SEAT_PRESENT + 1], blocks[:, :, SEAT_PUBLIC],
                            blocks[:, :, SEAT_DISCARD:SEAT_DISCARD + 1], blocks[:, :, SEAT_HAND],
                            blocks[:, :, SEAT_UNKNOWN:SEAT_UNKNOWN + 1],
                            blocks[:, :, SEAT_PUBLIC.start + 6:SEAT_PUBLIC.start + 7]), -1)
        return torch.cat((me.unsqueeze(1), others), 1), present

    def tokens(self, obs):
        batch = len(obs)
        tiles = obs[:, TILES].reshape(batch, 19, 8)
        tokens = torch.cat((self.node_in(self.node_inputs(obs)),
                            self.tile_in(torch.cat((tiles, obs[:, ROBBER].unsqueeze(-1)), -1))), 1) + self.position
        rest = obs[:, PUBLIC:OBSERVATION_SIZE]
        if self.counted:
            rest = torch.cat((rest, obs[:, ENEMY_HAND.start:ENEMY_UNKNOWN.stop]), 1)
        if self.trading:
            rest = torch.cat((rest, obs[:, SEAT_SIZE:TRADE_SIZE]), 1)
        tokens = torch.cat((tokens, self.global_in(rest).unsqueeze(1)), 1)
        bias = self.bias(self.relation).permute(2, 0, 1).repeat(batch, 1, 1)
        if self.seats:
            players, present = self.players(obs)
            tokens = torch.cat((tokens, self.player_in(players) + self.player_position), 1)
            absent = torch.cat((torch.zeros(batch, self.TOKENS, dtype=torch.bool, device=obs.device), present < .5), 1)
            bias = bias.masked_fill(absent.repeat_interleave(self.heads, 0).unsqueeze(1), float("-inf"))
        for layer in self.layers:
            tokens = layer(tokens, bias)
        return self.norm(tokens)

    def value_latent(self, tokens, obs):
        nodes = tokens[:, :self.NODE_END]
        own, enemy = building_weights(obs)
        parts = [tokens[:, self.TOKENS - 1], nodes.mean(1), (own.unsqueeze(-1) * nodes).sum(1) / 5,
                 (enemy.unsqueeze(-1) * nodes).sum(1) / 5]
        if self.seats:
            _, present = self.players(obs)
            players = tokens[:, self.TOKENS:]
            parts.append((players[:, 1:] * present[:, 1:, None]).sum(1) /
                         present[:, 1:].sum(1, keepdim=True).clamp_min(1))
        return self.value(torch.cat(parts, -1))

    def forward_actor(self, obs):
        return self.tokens(obs), obs

    def forward_critic(self, obs):
        return self.value_latent(self.tokens(obs), obs)

    def forward(self, obs):
        tokens = self.tokens(obs)
        return (tokens, obs), self.value_latent(tokens, obs)


class AttentionScorer(nn.Module):
    def __init__(self, width, counted=False, ports=False, seats=False, trading=False):
        super().__init__()
        self.counted, self.ports, self.seats, self.trading = counted, ports, seats, trading
        register_tables(self, relative=seats, persistent=not seats, trading=trading)
        self.register_buffer("incidence", torch.as_tensor(INCIDENCE), persistent=False)
        delta, count = resource_deltas(seats, trading)
        self.register_buffer("delta", torch.as_tensor(delta), persistent=not seats)
        self.register_buffer("count", torch.as_tensor(count), persistent=not seats)
        for name, kind in (("offers", A.OFFER_TRADE), ("accepts", A.ACCEPT_TRADE), ("answers", None), ("confirms", A.CONFIRM_TRADE)):
            match = [A.ACCEPT_TRADE, A.REJECT_TRADE] if kind is None else [kind]
            self.register_buffer(name, torch.as_tensor([float(k in match) for k, _ in action_table(seats, trading)]), persistent=False)
        self.register_buffer("costs", torch.tensor([[1., 1, 1, 1, 0], [0, 0, 0, 2, 3], [0, 0, 1, 1, 1]]), persistent=False)
        edges, edge_masks = edge_table(seats, trading)
        self.register_buffer("edges", torch.as_tensor(edges), persistent=False)
        self.register_buffer("edge_masks", torch.as_tensor(edge_masks), persistent=False)
        self.kind = nn.Embedding(len(KINDS), 8)
        self.kind_scores = nn.Linear(width, len(KINDS))
        self.local = nn.Sequential(nn.Linear(8 + 4 * width + 21 + 10 * counted + ports + width * seats + 4 * trading, 64),
                                   nn.GELU(), nn.Linear(64, 1))

    def forward(self, latent):
        tokens, obs = latent
        batch = len(obs)
        nodes, tiles, context = tokens[:, :BoardTransformer.NODE_END], tokens[:, BoardTransformer.NODE_END:BoardTransformer.TILE_END], tokens[:, BoardTransformer.TOKENS - 1]
        ends = nodes[:, self.endpoints]
        endpoints = torch.cat((ends.mean(2), ends.amax(2)), -1) * self.endpoint_masks[None, :, None]
        tile = tiles[:, self.tiles] * self.tile_masks[None, :, None]
        give, take = self.give[None].expand(batch, -1, -1), self.take[None].expand(batch, -1, -1)
        delta, count = self.delta[None].expand(batch, -1, -1), self.count[None, :, None].expand(batch, -1, -1)
        if self.trading:
            offering, asking = obs[:, TRADE_OFFERING] * 4, obs[:, TRADE_ASKING] * 4
            give = give + self.accepts[None, :, None] * asking[:, None] + self.confirms[None, :, None] * offering[:, None]
            take = take + self.accepts[None, :, None] * offering[:, None] + self.confirms[None, :, None] * asking[:, None]
            moved = (self.accepts + self.confirms)[None, :, None] * (take - give)
            delta, count = delta + moved, count + moved.abs().sum(-1, keepdim=True) / 4
        resources = torch.cat((obs[:, HAND][:, None].expand(-1, len(self.kinds), -1), give, take, delta / 4, count), -1)
        local = [self.kind(self.kinds).expand(batch, -1, -1), endpoints, tile, resources,
                 context[:, None].expand(-1, len(self.kinds), -1)]
        if self.counted:
            enemy = obs[:, ENEMY_HAND]
            roads = torch.stack((obs[:, ROAD_LENGTH][:, self.edges], obs[:, ROAD_TITLE][:, self.edges]), -1)
            monopoly = (self.kinds == KIND_IDS[A.PLAY_MONOPOLY]).float()
            gain = (enemy @ self.take.T) * monopoly
            tile_resources = obs[:, TILES].reshape(batch, 19, 8)[:, self.tiles, :5]
            stock = (tile_resources * enemy[:, None]).sum(-1) * self.tile_masks
            local += [obs[:, ENEMY_HAND.start:ENEMY_UNKNOWN.stop][:, None].expand(-1, len(self.kinds), -1),
                      roads * self.edge_masks[None, :, None], gain.unsqueeze(-1), stock.unsqueeze(-1)]
        if self.ports:
            settlement = (self.kinds == KIND_IDS[A.BUILD_SETTLEMENT]).float()
            port = port_values(obs, self.incidence).sum(-1)[:, self.endpoints[:, 0]] * settlement
            local.append(port.unsqueeze(-1))
        if self.seats:
            players = tokens[:, BoardTransformer.TOKENS:]
            party = players[:, self.victims] * (self.victims > 0).float()[None, :, None]
            if self.trading:
                offerer = (obs[:, TRADE_OFFERER][:, :, None] * players).sum(1)
                party = party + self.answers[None, :, None] * offerer[:, None]
            local.append(party)
        if self.trading:
            blocks = obs[:, SEATS].reshape(batch, 3, SEAT_BLOCK)
            hands, present = blocks[:, :, SEAT_HAND] * 19, blocks[:, :, SEAT_PRESENT]
            asked, offered = self.take[None, :, None], self.give[None, :, None]
            payable = ((hands[:, None] >= asked - 1e-6).all(-1).float() * present[:, None]).sum(-1)
            likely = (((hands[:, None] >= asked + (asked > 0).float() - 1e-6).all(-1) &
                       ((hands[:, None] * (offered > 0).float()).sum(-1) < .5)).float() * present[:, None]).sum(-1)
            able = (hands[:, :, None] >= self.costs - 1e-6).all(-1)
            finishing = (((hands[:, None] + offered)[:, :, :, None] >= self.costs - 1e-6).all(-1) & ~able[:, None]).any(-1)
            finishing = (finishing.float() * present[:, None]).sum(-1)
            for scalar in payable, likely, finishing:
                local.append((scalar / 3 * self.offers[None]).unsqueeze(-1))
            mine = obs[:, HAND] * 19
            after = mine[:, None] + take - give
            short = ~(mine[:, None] >= self.costs - 1e-6).all(-1)
            pays = ((after[:, :, None] >= self.costs - 1e-6).all(-1) & short[:, None]).any(-1)
            local.append((pays.float() * (self.offers + self.accepts + self.confirms)[None]).unsqueeze(-1))
        return self.kind_scores(context)[:, self.kinds] + self.local(torch.cat(local, -1)).squeeze(-1)


class AttentionPolicy(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = BoardTransformer(**self.transformer)

    def __init__(self, observation_space, action_space, lr_schedule, width=64, heads=4, layers=3,
                 counted=False, ports=False, seats=False, trading=False, **kwargs):
        self.transformer = dict(width=width, heads=heads, layers=layers, counted=counted or seats,
                                seats=seats, trading=trading)
        self.ports, self.seats, self.trading = ports, seats, trading
        kwargs.setdefault("ortho_init", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        size, actions = ((TRADE_SIZE, 436) if trading else (SEAT_SIZE, 370) if seats else
                         (COUNTED_SIZE, 332) if counted else (OBSERVATION_SIZE, 332))
        if observation_space.shape != (size,) or action_space.n != actions:
            raise ValueError("AttentionPolicy needs the observation format it was built for")
        self.action_net = AttentionScorer(width, counted or seats, self.ports, seats, trading)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def prime_trading(self):
        like = {A.OFFER_TRADE: A.MARITIME_TRADE, A.ACCEPT_TRADE: A.MARITIME_TRADE,
                A.CONFIRM_TRADE: A.MARITIME_TRADE, A.REJECT_TRADE: A.END_TURN, A.CANCEL_TRADE: A.END_TURN}
        with torch.no_grad():
            for kind, model in like.items():
                i, j = KIND_IDS[kind], KIND_IDS[model]
                self.action_net.kind.weight[i] = self.action_net.kind.weight[j]
                self.action_net.kind_scores.weight[i] = self.action_net.kind_scores.weight[j]
                self.action_net.kind_scores.bias[i] = self.action_net.kind_scores.bias[j]
