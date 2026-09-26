"""Catan rules, card beliefs, fixed model formats, and the Gym environment."""
from dataclasses import dataclass
from functools import lru_cache

import random

import gymnasium as gym
import numpy as np
from catanatron.game import Game as CoreGame
from catanatron.apply_action import yield_resources
from catanatron.models.actions import Action, generate_playable_actions
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.board import STATIC_GRAPH, get_edges
from catanatron.models.enums import ActionType, ActionPrompt, RESOURCES, DEVELOPMENT_CARDS
from catanatron.models.map import build_map, number_probability
from catanatron.models.player import Color, Player
from catanatron.state_functions import (get_actual_victory_points, get_player_freqdeck,
                                        player_key, player_num_dev_cards,
                                        player_num_resource_cards)

# The shape of the board never changes. Only the resources and the numbers on
# the tiles change from game to game. Read the shape once.
_TEMPLATE = build_map("BASE")
COORDINATES = tuple(sorted(_TEMPLATE.land_tiles))
NODES = tuple(sorted(_TEMPLATE.land_nodes))
EDGES = tuple(sorted(tuple(sorted(edge)) for edge in get_edges(_TEMPLATE.land_nodes)))
EDGE_INDEX = {edge: index for index, edge in enumerate(EDGES)}
COLORS = (Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE)
# One row for each node and one column for each tile. The value is 1 when the
# node touches the tile.
INCIDENCE = np.array([[node in _TEMPLATE.land_tiles[c].nodes.values() for c in COORDINATES]
                      for node in NODES], dtype=np.float32)

# Where each group of numbers sits inside an observation. The tiles and the
# ports are in the same place in both formats.
TILES = slice(0, 152)
PORTS = slice(152, 476)
ROBBER = slice(476, 495)
BUILDINGS = slice(495, 711)
ROADS = slice(711, 855)
PUBLIC = 855
HAND = slice(883, 888)
SETUP_PHASE = slice(913, 914)
ROAD_BUILDING = slice(916, 917)
OBSERVATION_SIZE = 923
# A counted observation adds what a player who counts cards can work out at
# the table: the other hand, as far as it can be known, and the longest road
# that each road would make.
ENEMY_HAND = slice(923, 928)
ENEMY_UNKNOWN = slice(928, 929)
ROAD_LENGTH = slice(929, 1001)
ROAD_TITLE = slice(1001, 1073)
COUNTED_SIZE = 1073
# The seat format (format 5) serves two to four players in one layout. Its
# first 1073 values are the counted two player observation, where "the
# enemy" is the leading opponent and the enemy buildings and roads are those
# of every opponent. Then come the players in turn order after the acting
# player, one block each, and the number of players.
SEAT_BLOCK = 54 * 2 + 72 + 14 + 1 + 5 + 1 + 1
SEAT_BUILDINGS, SEAT_ROADS, SEAT_PUBLIC = slice(0, 108), slice(108, 180), slice(180, 194)
SEAT_DISCARD, SEAT_HAND, SEAT_UNKNOWN, SEAT_PRESENT = 194, slice(195, 200), 200, 201
SEATS = slice(COUNTED_SIZE, COUNTED_SIZE + 3 * SEAT_BLOCK)
SEAT_SIZE = COUNTED_SIZE + 3 * SEAT_BLOCK + 1
# Trading adds the offer on the table: what is offered and asked, who offers
# (me, or a seat), who has accepted, and how many offers I may still make.
TRADE_OFFERING = slice(SEAT_SIZE, SEAT_SIZE + 5)
TRADE_ASKING = slice(SEAT_SIZE + 5, SEAT_SIZE + 10)
TRADE_OFFERER = slice(SEAT_SIZE + 10, SEAT_SIZE + 14)
TRADE_ACCEPTEES = slice(SEAT_SIZE + 14, SEAT_SIZE + 17)
TRADE_OFFERS_LEFT = SEAT_SIZE + 17
TRADE_SIZE = SEAT_SIZE + 18
OFFERS_PER_TURN = 2
TRADE_KINDS = (ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE,
               ActionType.CONFIRM_TRADE, ActionType.CANCEL_TRADE)
# Name and scale of every public player value. A player cannot hide these.
PUBLIC_VALUES = (("VICTORY_POINTS", 12), ("ROADS_AVAILABLE", 15),
                 ("SETTLEMENTS_AVAILABLE", 5), ("CITIES_AVAILABLE", 4),
                 ("HAS_ROAD", 1), ("HAS_ARMY", 1), ("HAS_ROLLED", 1),
                 ("LONGEST_ROAD_LENGTH", 15))

@dataclass(frozen=True)
class Config:
    """The rules of the game and the limits of one episode."""
    target_vp: int = 6
    max_turns: int = 300
    max_actions: int = 2000
    players: int = 2
    player_counts: tuple = ()
    # Add the other hand and the longest road that each road would make.
    counted: bool = False
    # The seat format: two to four players, counted, for the attention policy.
    seats: bool = False
    # Player to player trades: offers of one or two cards for one or two.
    # offers is the number a player may make each turn; 0 keeps the trading
    # observation but stops every offer, to measure what trading is worth.
    trading: bool = False
    offers: int = 2

    def __post_init__(self):
        object.__setattr__(self, "player_counts", tuple(self.player_counts))
        if self.players not in (2, 3, 4) or any(n not in (2, 3, 4) for n in self.player_counts):
            raise ValueError("players and player_counts must be 2, 3 or 4")
        if (self.players != 2 or self.player_counts) and not self.seats:
            raise ValueError("More than two players needs the seat format")
        if self.trading and not self.seats:
            raise ValueError("Trading needs the seat format")
        if type(self.target_vp) is not int or self.target_vp < 3 or self.max_turns < 1 or self.max_actions < 1:
            raise ValueError("target_vp must be an integer of at least 3, and both limits must be positive")
        if type(self.offers) is not int or self.offers < 0:
            raise ValueError("offers must be a nonnegative integer")
        if self.max_actions <= 2 * (max((self.players, *self.player_counts)) - 1):
            raise ValueError("max_actions must allow the last seat's first setup decision")

    @property
    def relative_actions(self):
        """Whether the action table names every seat, relative to the actor."""
        return self.seats

    @property
    def table(self):
        """The action table: which seats it names, and whether it trades."""
        return self.relative_actions, self.trading

def winner(game):
    """Return the colour of the winner, or None."""
    state = game.state
    color = state.colors[state.current_turn_index]
    return color if get_actual_victory_points(state, color) >= game.vps_to_win else None

def public_values(state, owner):
    """Return the numbers that every player can see about one player."""
    key = player_key(state, owner)
    values = [state.player_state[key + "_" + name] / scale for name, scale in PUBLIC_VALUES]
    values.extend(state.player_state[key + "_PLAYED_" + card] / 14
                  for card in DEVELOPMENT_CARDS if card != "VICTORY_POINT")
    values.append(player_num_resource_cards(state, owner) / 95)
    values.append(player_num_dev_cards(state, owner) / 25)
    return values

def private_values(state, color):
    """Return the hand of one player, and what is left in the bank."""
    key = player_key(state, color)
    ps = state.player_state
    values = [count / 19 for count in get_player_freqdeck(state, color)]
    values.extend(ps[key + "_" + card + "_IN_HAND"] / 14 for card in DEVELOPMENT_CARDS)
    values.extend(float(ps[key + "_" + card + "_OWNED_AT_START"])
                  for card in DEVELOPMENT_CARDS if card != "VICTORY_POINT")
    values.extend((get_actual_victory_points(state, color) / 12,
                   float(ps[key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])))
    values.extend(float(count > 0) for count in state.resource_freqdeck)
    values.append(len(state.development_listdeck) / 25)
    values.extend(float(state.current_prompt == prompt) for prompt in ActionPrompt)
    return values

def node_yield(state, node, blocked=None):
    """Return how many of each resource one node pays per roll."""
    yields = np.zeros(5)
    for tile in state.board.map.adjacent_tiles[node]:
        if tile.resource is not None and tile is not blocked:
            yields[RESOURCES.index(tile.resource)] += number_probability(tile.number)
    return yields

def longest_road(board, color):
    """Return the longest road of one player, by the rules."""
    best = 0
    for start in {node for edge, owner in board.roads.items() if owner == color for node in edge}:
        agenda = [(start, ())]
        while agenda:
            node, path = agenda.pop()
            best = max(best, len(path))
            if path and board.is_enemy_node(node, color):
                continue
            for neighbour in STATIC_GRAPH.neighbors(node):
                edge = tuple(sorted((node, neighbour)))
                if edge not in path and board.is_friendly_road(edge, color):
                    agenda.append((neighbour, path + (edge,)))
    return best

def road_length_after(board, color, edge):
    """Return the longest road of one player once one more road is laid."""
    a, b = edge
    board.roads[(a, b)] = board.roads[(b, a)] = color
    try:
        return longest_road(board, color)
    finally:
        del board.roads[(a, b)], board.roads[(b, a)]

def player_income(state, color, blocked=None):
    """Return how many of each resource one player collects per roll."""
    income = np.zeros(5)
    for node, (owner, building) in state.board.buildings.items():
        if owner == color:
            income += node_yield(state, node, blocked) * (2 if building == "CITY" else 1)
    return income

class CardCount:
    """What one player knows about the other hand from watching the table."""

    def __init__(self):
        self.certain = np.zeros(5)
        self.pool = 0.0
        self.mix = np.full(5, .2)

    def copy(self):
        other = CardCount.__new__(CardCount)
        other.certain, other.pool, other.mix = self.certain.copy(), self.pool, self.mix.copy()
        return other

    def expected(self):
        return self.certain + self.pool * self.mix

    def _set_mix(self, cards):
        self.mix = cards / cards.sum() if cards.sum() > 1e-9 else np.full(5, .2)

    def seen(self, delta):
        """Count cards that were seen to arrive or leave, by resource."""
        self.certain += np.maximum(delta, 0)
        lost = np.maximum(-delta, 0)
        from_certain = np.minimum(lost, self.certain)
        self.certain -= from_certain
        revealed = lost - from_certain
        if revealed.sum():
            # The pool held at least these cards. Take them out of it.
            remaining = np.maximum(self.pool * self.mix - revealed, 0)
            self.pool = max(self.pool - revealed.sum(), 0.0)
            self._set_mix(remaining)

    def lost_unknown(self, cards, size):
        """Cards left the hand unseen: a discard, or a theft seen from the side."""
        before = self.expected()
        self.certain = np.maximum(self.certain - cards, 0)
        self.pool = max(size - self.certain.sum(), 0.0)
        self._set_mix(np.maximum(before - self.certain, 0))

    def revealed(self, cards):
        """The hand was shown to hold at least these cards: an offer shows the cards offered, an acceptance
        the cards asked."""
        extra = np.maximum(cards - self.certain, 0)
        taken = min(extra.sum(), self.pool)
        if taken > 0:
            extra *= taken / extra.sum()
            self.certain += extra
            remaining = np.maximum(self.pool * self.mix - extra, 0)
            self.pool -= taken
            self._set_mix(remaining)

    def gained_unknown(self, cards, mix):
        """Cards arrived unseen, drawn from a hand with the given mix."""
        self._set_mix(self.pool * self.mix + cards * mix)
        self.pool += cards

    def resync(self, size):
        """Match the public size of the hand. A gap is counted as uncertain."""
        gap = size - self.certain.sum() - self.pool
        if gap >= 0:
            self.pool += gap
        else:
            self.pool = max(self.pool + gap, 0.0)
            if self.certain.sum() > size:
                self.certain *= size / self.certain.sum()

class Game(CoreGame):
    """Catanatron with corrected victory, road, bank-shortage, Year of Plenty and trade rules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each player counts every other hand.
        colors = self.state.colors if getattr(self, "state", None) else ()
        self.counts = {c: {o: CardCount() for o in colors if o != c} for c in colors}

    def winning_color(self):
        return winner(self)

    def copy(self):
        """Copy the game for a simulation. The copy keeps these rules and rolls its own dice, so a
        simulation cannot move the real game."""
        game = super().copy()
        game.__class__ = type(self)
        # The dice and the card draws read the state's generator.
        game.random = game.state.random = random.Random(0)
        game.counts = {c: {o: count.copy() for o, count in others.items()} for c, others in self.counts.items()}
        return game

    def execute(self, action, validate_action=True, action_record=None):
        state = self.state
        road_owner_before = state.board.road_color
        bank = state.resource_freqdeck.copy()
        hands = {c: np.array(get_player_freqdeck(state, c)) for c in state.colors}
        result = super().execute(action, validate_action, action_record)
        if action.action_type == ActionType.ROLL and sum(result.result) != 7:
            self._pay_lone_claimant(bank, sum(result.result))
        if action.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
            self._skip_offerer()
        self._count(action, hands)
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
            self._award_longest_road(road_owner_before, action)
        if sum(self.state.resource_freqdeck) >= 2:
            # The library offers a one card Year of Plenty while the bank still
            # holds two cards. Remove those actions.
            self.playable_actions = [a for a in self.playable_actions
                                     if a.action_type != ActionType.PLAY_YEAR_OF_PLENTY
                                     or len(a.value) == 2]
        return result

    def _count(self, action, hands_before):
        """Update what every player knows about every other hand."""
        state = self.state
        deltas = {c: np.array(get_player_freqdeck(state, c)) - hands_before[c] for c in state.colors}
        sizes = {c: player_num_resource_cards(state, c) for c in state.colors}
        kind, actor = action.action_type, action.color
        victim = action.value[1] if kind == ActionType.MOVE_ROBBER else None
        stolen = victim is not None and deltas[victim].sum() < 0
        shown = None
        if kind == ActionType.OFFER_TRADE:
            shown = actor, np.array(action.value[:5], dtype=float)
        elif kind == ActionType.ACCEPT_TRADE:
            shown = actor, np.array(action.value[5:10], dtype=float)
        changed = {c for c, delta in deltas.items() if delta.any()}
        if shown:
            changed.add(actor)
        for observer, counts in self.counts.items():
            for other, count in counts.items():
                if other not in changed:
                    continue
                if shown and shown[0] == other:
                    count.revealed(shown[1])
                if kind == ActionType.DISCARD_RESOURCE and actor == other:
                    count.lost_unknown(-deltas[other].sum(), sizes[other])
                elif stolen and observer not in (actor, victim) and other == victim:
                    count.lost_unknown(1, sizes[other])
                elif stolen and observer not in (actor, victim) and other == actor:
                    count.gained_unknown(1, counts[victim].mix if counts[victim].pool else
                                         counts[victim].expected() / max(counts[victim].expected().sum(), 1e-9))
                else:
                    count.seen(deltas[other])
                count.resync(sizes[other])

    def _skip_offerer(self):
        """Move past the offerer when the answers come round to them."""
        from catanatron.apply_action import reset_trading_state
        state = self.state
        offerer = state.current_trade[10]
        if state.current_prompt != ActionPrompt.DECIDE_TRADE or state.current_player_index != offerer:
            return
        later = [i for i in range(state.current_player_index + 1, len(state.colors)) if i != offerer]
        if later:
            state.current_player_index = later[0]
        elif any(state.acceptees):
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES
        else:
            reset_trading_state(state)
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.PLAY_TURN
        self.playable_actions = generate_playable_actions(state)

    def _pay_lone_claimant(self, bank, roll):
        """Give a player the rest of the bank when no other player wants it."""
        demand, _ = yield_resources(self.state.board, [100] * 5, roll)
        paid = False
        for index, resource in enumerate(RESOURCES):
            claimants = [color for color, amounts in demand.items() if amounts[index]]
            if len(claimants) == 1 and 0 < bank[index] < demand[claimants[0]][index]:
                key = player_key(self.state, claimants[0])
                self.state.player_state[key + "_" + resource + "_IN_HAND"] += bank[index]
                self.state.resource_freqdeck[index] -= bank[index]
                paid = True
        if paid:
            self.playable_actions = generate_playable_actions(self.state)

    def _award_longest_road(self, road_owner_before, action):
        """Measure the roads that the action changed, give the longest road to the right player, and fix
        the scores."""
        board = self.state.board
        if action.action_type == ActionType.BUILD_ROAD:
            changed = {action.color}
        else:
            changed = {board.roads[edge] for edge in STATIC_GRAPH.edges(action.value) if edge in board.roads}
        for color in changed:
            board.road_lengths[color] = longest_road(board, color)
        longest = max(board.road_lengths.values(), default=0)
        leaders = [c for c, length in board.road_lengths.items() if length == longest and length >= 5]
        owner = road_owner_before if road_owner_before in leaders else leaders[0] if len(leaders) == 1 else None
        board.road_color, board.road_length = owner, longest
        for color in self.state.colors:
            key = player_key(self.state, color)
            ps = self.state.player_state
            change = 2 * (int(color == owner) - int(ps[key + "_HAS_ROAD"]))
            ps[key + "_HAS_ROAD"] = color == owner
            ps[key + "_LONGEST_ROAD_LENGTH"] = board.road_lengths[color]
            ps[key + "_VICTORY_POINTS"] += change
            ps[key + "_ACTUAL_VICTORY_POINTS"] += change

@lru_cache(maxsize=1)
def trade_offers():
    """Return every offer a player may make: one or two of a resource for one or two of another, as the ten
    numbers the engine expects."""
    offers = []
    for give, take in ((1, 1), (2, 1), (1, 2)):
        for i in range(5):
            for j in range(5):
                if i != j:
                    offering, asking = [0] * 5, [0] * 5
                    offering[i], asking[j] = give, take
                    offers.append(tuple(offering + asking))
    return tuple(offers)

OFFER_CARDS = np.asarray(trade_offers())

@lru_cache(maxsize=4)
def action_table(relative, trading=False):
    """Return every action of the BASE game, in a fixed order."""
    table = list(get_action_array(COLORS if relative else COLORS[:2], "BASE"))
    if trading:
        table.extend((ActionType.OFFER_TRADE, offer) for offer in trade_offers())
        table.extend(((ActionType.ACCEPT_TRADE, None), (ActionType.REJECT_TRADE, None),
                      (ActionType.CANCEL_TRADE, None)))
        table.extend((ActionType.CONFIRM_TRADE, seat) for seat in COLORS[1:])
    return tuple(table)

@lru_cache(maxsize=4)
def _action_indices(relative, trading=False):
    return {action: i for i, action in enumerate(action_table(relative, trading))}

def relative_color(color, other, colors, relative):
    """Return the colour of another player as the acting player sees it."""
    if relative:
        offset = colors.index(color)
        return COLORS[(colors[offset:] + colors[:offset]).index(other)]
    return Color.BLUE if other == color else Color.RED

def action_id(action, color, colors, relative, trading=False):
    """Return the number of one action, as the acting player sees it."""
    value = action.value
    if action.action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        value = (coordinate, None if victim is None else relative_color(color, victim, colors, relative))
    elif action.action_type == ActionType.BUILD_ROAD:
        value = tuple(sorted(value))
    elif action.action_type == ActionType.OFFER_TRADE:
        value = tuple(value[:10])
    elif action.action_type == ActionType.CONFIRM_TRADE:
        value = relative_color(color, value[10], colors, relative)
    elif action.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE):
        value = None
    return _action_indices(relative, trading)[(action.action_type, value)]

class CatanEnv(gym.Env):
    """A learner plays against scripted or saved opponents, with optional player trades."""
    metadata = {"render_modes": []}
    # Subclasses that display the log keep it; overriding _execute instead
    # silently drops the per-turn offer bookkeeping that offers() relies on.
    clear_records = True

    def __init__(self, config=Config(), opponent="random"):
        super().__init__()
        if not callable(opponent) and opponent not in ("random", "greedy"):
            raise ValueError("opponent must be random, greedy, or a callable(env, color)")
        self.config, self.opponent = config, opponent
        self.action_space = gym.spaces.Discrete(len(action_table(*config.table)))
        self.reset(seed=0)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=self.observe().shape, dtype=np.float32)

    def legal(self, color):
        """Return the current action-number mapping; callers must not mutate it."""
        if self.game.state.current_color() != color:
            raise ValueError("Legal actions were asked for a player who is not acting")
        cached = getattr(self, "_legal_cache", None)
        # Engine actions get a new list after execution, including copied-game simulations.
        if cached and cached[0] is self.game and cached[1] is self.game.playable_actions and cached[2] == color:
            return cached[3]
        relative, trading = self.config.table
        colors = self.game.state.colors
        actions = list(self.game.playable_actions)
        if trading:
            actions.extend(self.offers(color))
        result = {action_id(action, color, colors, relative, trading): action for action in actions}
        if len(result) != len(actions):
            raise RuntimeError("Two actions share one action number")
        self._legal_cache = self.game, self.game.playable_actions, color, result
        return result

    def offers(self, color):
        """Return the offers one player may make now."""
        from catanatron.state_functions import player_has_rolled
        state = self.game.state
        if (state.current_prompt != ActionPrompt.PLAY_TURN or not player_has_rolled(state, color)
                or state.is_initial_build_phase or state.is_resolving_trade or state.is_road_building):
            return []
        made = self._offers_made(color)
        if len(made) >= self.config.offers:
            return []
        hand = np.array(get_player_freqdeck(state, color))
        bounds = np.asarray([count.certain + count.pool for count in self.game.counts[color].values()])
        payable = (OFFER_CARDS[:, :5] <= hand).all(1)
        possible = (OFFER_CARDS[None, :, 5:] <= bounds[:, None]).all(2).any(0)
        return [Action(color, ActionType.OFFER_TRADE, trade_offers()[i])
                for i in np.flatnonzero(payable & possible) if trade_offers()[i] not in made]

    def _offers_made(self, color):
        """Return the offers one player has made this turn."""
        offers = self.__dict__.setdefault("_offers", {})
        turn = self.game.state.num_turns
        if color not in offers or offers[color][0] != turn:
            offers[color] = (turn, set())
        return offers[color][1]

    def _mask_of(self, legal):
        """Return one flag for each action. The flag is true when it is legal."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[list(legal)] = True
        return mask

    def mask_for(self, color):
        return self._mask_of(self.legal(color))

    def action_masks(self):
        return self._mask.copy()

    def _cache_map(self):
        """Read the tiles and the ports of this board. They do not change."""
        board_map = self.game.state.board.map
        features = []
        for coordinate in COORDINATES:
            tile = board_map.land_tiles[coordinate]
            features.extend(float(tile.resource == resource) for resource in [*RESOURCES, None])
            features.extend((0 if tile.number is None else tile.number / 12,
                             0 if tile.resource is None else number_probability(tile.number) * 6))
        for node in NODES:
            features.extend(float(node in board_map.port_nodes[resource]) for resource in [*RESOURCES, None])
        self.static = np.asarray(features, dtype=np.float32)
        self.production = {node: node_yield(self.game.state, node).sum() for node in NODES}

    def observe(self, color=None, counted=None, seats=None, trading=None):
        """Encode the actor first; legacy formats keep their exact feature offsets."""
        color = self.learner if color is None else color
        counted = self.config.counted if counted is None else counted
        seats = self.config.seats if seats is None else seats
        trading = self.config.trading if trading is None else trading
        state, board = self.game.state, self.game.state.board
        start = state.colors.index(color)
        order = state.colors[start:] + state.colors[:start]
        others = order[1:]
        enemy = max(others, key=lambda c: state.player_state[player_key(state, c) + "_VICTORY_POINTS"])
        size = TRADE_SIZE if trading else SEAT_SIZE if seats else COUNTED_SIZE if counted else OBSERVATION_SIZE
        obs = np.zeros(size, np.float32)
        obs[:ROBBER.start] = self.static
        obs[ROBBER.start + COORDINATES.index(board.robber_coordinate)] = 1
        buildings, roads = np.zeros((4, 54, 2), np.float32), np.zeros((4, 72), np.float32)
        for node, (owner, kind) in board.buildings.items():
            buildings[order.index(owner), NODES.index(node), int(kind == "CITY")] = 1
        for edge, owner in board.roads.items():
            if edge in EDGE_INDEX:
                roads[order.index(owner), EDGE_INDEX[edge]] = 1
        obs[BUILDINGS] = np.concatenate((buildings[0], buildings[1:].sum(0)), axis=1).ravel()
        obs[ROADS] = np.column_stack((roads[0], roads[1:].sum(0))).ravel()
        obs[PUBLIC:OBSERVATION_SIZE] = [
            *public_values(state, color), *public_values(state, enemy), *private_values(state, color),
            state.colors[state.current_turn_index] == color, state.is_initial_build_phase, state.is_discarding,
            state.current_prompt == ActionPrompt.MOVE_ROBBER, state.is_road_building, state.free_roads_available / 2,
            *(state.discard_counts[state.color_to_index[c]] / 48 for c in (color, enemy)),
            self.config.target_vp / 12, state.num_turns / self.config.max_turns, self.actions / self.config.max_actions]
        if counted or seats:
            count = self.game.counts[color][enemy]
            obs[ENEMY_HAND], obs[ENEMY_UNKNOWN] = count.expected() / 19, count.pool / 19
            if state.current_color() == color:
                for action in self.game.playable_actions:
                    if action.action_type == ActionType.BUILD_ROAD:
                        edge = EDGE_INDEX[tuple(sorted(action.value))]
                        length = road_length_after(board, color, action.value)
                        obs[ROAD_LENGTH.start + edge] = length / 15
                        # The holder retains the title on a tie.
                        obs[ROAD_TITLE.start + edge] = length >= 5 and (
                            board.road_color == color or length > board.road_length)
        if seats:
            blocks = obs[SEATS].reshape(3, SEAT_BLOCK)
            for i, other in enumerate(others):
                count = self.game.counts[color][other]
                blocks[i] = [*buildings[i + 1].ravel(), *roads[i + 1], *public_values(state, other),
                             state.discard_counts[state.color_to_index[other]] / 48,
                             *(count.expected() / 19), count.pool / 19, 1]
            obs[SEAT_SIZE - 1] = len(order) / 4
        if trading:
            if state.is_resolving_trade:
                trade = state.current_trade
                obs[TRADE_OFFERING.start:TRADE_ASKING.stop] = np.asarray(trade[:10]) / 4
                obs[TRADE_OFFERER.start + order.index(state.colors[trade[10]])] = 1
                obs[TRADE_ACCEPTEES.start:TRADE_ACCEPTEES.start + len(others)] = [
                    state.acceptees[state.colors.index(c)] for c in others]
            obs[TRADE_OFFERS_LEFT] = (self.config.offers - len(self._offers_made(color))) / max(2, self.config.offers)
        return np.clip(obs, 0, 1, out=obs)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.seed_value = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.num_players = (int(self.np_random.choice(self.config.player_counts))
                            if self.config.player_counts else self.config.players)
        self.seat = options.get("seat", int(self.np_random.integers(self.num_players)))
        if self.seat not in range(self.num_players):
            raise ValueError("seat must be inside the player count")
        self.game = Game([Player(c) for c in COLORS[:self.num_players]], seed=self.seed_value,
                         vps_to_win=self.config.target_vp)
        self.learner = self.game.state.colors[self.seat]
        self.actions = 0
        self.done = False
        self._offers, self._legal_cache = {}, None
        self._cache_map()
        self._advance()
        self._refresh()
        return self.observe(), self._info()

    def _refresh(self):
        """Store the legal actions of the learner, and the matching flags."""
        acting = self.game.state.current_color() == self.learner
        self._legal = self.legal(self.learner) if acting else {}
        self._mask = self._mask_of(self._legal)

    def _execute(self, action):
        if winner(self.game) is None:
            if action.action_type == ActionType.OFFER_TRADE:
                self._offers_made(action.color).add(tuple(action.value[:10]))
            self.game.execute(action)
            self.actions += 1
            if self.clear_records:
                # The learner sees the current board only. It has no memory of
                # past actions, so the record is not needed and would grow
                # forever. A caller that shows the log turns this off.
                self.game.state.action_records.clear()

    def _advance(self):
        """Play every opponent action until the learner must choose."""
        from .opponents import choose_action
        while not self._limits() and winner(self.game) is None and self.game.state.current_color() != self.learner:
            color = self.game.state.current_color()
            chosen = self.opponent(self, color) if callable(self.opponent) else choose_action(self, color, self.opponent)
            legal = self.legal(color)
            if chosen not in legal:
                raise ValueError(f"The opponent chose an illegal action {chosen}")
            self._execute(legal[chosen])

    def _limits(self):
        return [name for name, value in (("max_turns", self.game.state.num_turns), ("max_actions", self.actions))
                if value >= getattr(self.config, name)]

    def _info(self, outcome=None):
        state = self.game.state
        winning = winner(self.game)
        limits = self._limits() if outcome == "truncated" else []
        name = getattr(self, "opponent_name", self.opponent if isinstance(self.opponent, str) else "callable")
        return dict(opponent=name, outcome=outcome, winner=None if winning is None else winning.name,
                    truncation_limits=limits,
                    learner_vp=get_actual_victory_points(state, self.learner),
                    opponent_vp=max(state.player_state[player_key(state, c) + "_VICTORY_POINTS"]
                                    for c in state.colors if c != self.learner),
                    players=self.num_players, turns=state.num_turns, actions=self.actions,
                    seat=self.seat, seed=self.seed_value)

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() after an episode ends")
        if not isinstance(action, (int, np.integer)) or int(action) not in self._legal:
            raise ValueError(f"Illegal action {action}")
        self._execute(self._legal[int(action)])
        self._advance()
        winning = winner(self.game)
        terminated = winning is not None
        truncated = not terminated and bool(self._limits())
        self.done = terminated or truncated
        outcome = ("win" if winning == self.learner else "loss") if terminated else "truncated" if truncated else None
        self._refresh()
        reward = (1.0 if winning == self.learner else -1.0) if terminated else 0.0
        return self.observe(), reward, terminated, truncated, self._info(outcome)
