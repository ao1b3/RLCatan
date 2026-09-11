"""The Catan environment for reinforcement learning.

One step is one learner action. After that action the environment plays every
opponent action. The environment stops again when the learner must choose.

There are two observation formats. Format 3 is for two players. Format 4 holds
one block of numbers for each of two, three or four players.
"""
from dataclasses import dataclass
from functools import lru_cache

import gymnasium as gym
import numpy as np
from catanatron.game import Game as CoreGame
from catanatron.apply_action import yield_resources
from catanatron.models.actions import generate_playable_actions
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.board import get_edges
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
# Format 4 gives each player one block. Four blocks always exist. A block of
# zeros means that the player is not in the game.
PLAYER_SIZE = 197
MULTIPLAYER_OBSERVATION_SIZE = 1320
GLOBAL_SIZE = MULTIPLAYER_OBSERVATION_SIZE - 4 * PLAYER_SIZE
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
    multiplayer: bool = False
    player_counts: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "player_counts", tuple(self.player_counts))
        if self.players not in (2, 3, 4) or any(n not in (2, 3, 4) for n in self.player_counts):
            raise ValueError("players and player_counts must be 2, 3 or 4")
        if (self.players != 2 or self.player_counts) and not self.multiplayer:
            raise ValueError("More than two players needs the multiplayer format")
        if not 3 <= self.target_vp <= 10 or self.max_turns < 1 or self.max_actions < 1:
            raise ValueError("target_vp must be 3 to 10, and both limits must be positive")


def winner(game):
    """Return the colour of the winner, or None.

    A player can only win on their own turn. Check only the player whose turn
    it is.
    """
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
    """Return the hand of one player, and what is left in the bank.

    Only the player who asks may see these numbers.
    """
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
    """Return how many of each resource one node pays per roll.

    Pass the tile under the robber as blocked to ignore that tile.
    """
    yields = np.zeros(5)
    for tile in state.board.map.adjacent_tiles[node]:
        if tile.resource is not None and tile is not blocked:
            yields[RESOURCES.index(tile.resource)] += number_probability(tile.number)
    return yields


def player_income(state, color, blocked=None):
    """Return how many of each resource one player collects per roll.

    A city pays twice as much as a settlement.
    """
    income = np.zeros(5)
    for node, (owner, building) in state.board.buildings.items():
        if owner == color:
            income += node_yield(state, node, blocked) * (2 if building == "CITY" else 1)
    return income


class Game(CoreGame):
    """The catanatron game with three rule fixes.

    The library gets three rules wrong:

    1. It looks for a winner on every turn, not only on the turn of the player
       who scored.
    2. It does not move the longest road when another player cuts the road.
    3. It pays nothing when the bank is short, even when one player alone
       claims the cards.
    """

    def winning_color(self):
        return winner(self)

    def execute(self, action, validate_action=True, action_record=None):
        road_owner_before = self.state.board.road_color
        bank = self.state.resource_freqdeck.copy() if action.action_type == ActionType.ROLL else None
        result = super().execute(action, validate_action, action_record)
        if bank is not None and sum(result.result) != 7:
            self._pay_lone_claimant(bank, sum(result.result))
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
            self._award_longest_road(road_owner_before)
        if sum(self.state.resource_freqdeck) >= 2:
            # The library offers a one card Year of Plenty while the bank still
            # holds two cards. Remove those actions.
            self.playable_actions = [a for a in self.playable_actions
                                     if a.action_type != ActionType.PLAY_YEAR_OF_PLENTY
                                     or len(a.value) == 2]
        return result

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

    def _award_longest_road(self, road_owner_before):
        """Give the longest road to the right player, and fix the scores.

        The road must be five segments or longer. The player who holds the road
        keeps it when another player only draws level.
        """
        board = self.state.board
        longest = max(board.road_lengths.values(), default=0)
        leaders = [c for c, length in board.road_lengths.items() if length == longest and length >= 5]
        owner = road_owner_before if road_owner_before in leaders else leaders[0] if len(leaders) == 1 else None
        board.road_color, board.road_length = owner, longest
        for color in self.state.colors:
            key = player_key(self.state, color)
            ps = self.state.player_state
            change = 2 * (int(color == owner) - int(ps[key + "_HAS_ROAD"]))
            ps[key + "_HAS_ROAD"] = color == owner
            ps[key + "_VICTORY_POINTS"] += change
            ps[key + "_ACTUAL_VICTORY_POINTS"] += change


@lru_cache(maxsize=2)
def action_table(multiplayer):
    """Return every action of the BASE game, in a fixed order.

    The position in this tuple is the number of the action. The numbers never
    change, so a saved model always means the same action.
    """
    return tuple(get_action_array(COLORS if multiplayer else COLORS[:2], "BASE"))


@lru_cache(maxsize=2)
def _action_indices(multiplayer):
    return {action: i for i, action in enumerate(action_table(multiplayer))}


def action_id(action, color, colors, multiplayer):
    """Return the number of one action, as the acting player sees it.

    The acting player is always BLUE. In a two player game the other player is
    always RED. In a longer game the other players follow the turn order.
    """
    value = action.value
    if action.action_type == ActionType.MOVE_ROBBER:
        coordinate, victim = value
        if multiplayer:
            offset = colors.index(color)
            relative = colors[offset:] + colors[:offset]
            value = (coordinate, None if victim is None else COLORS[relative.index(victim)])
        else:
            value = (coordinate, None if victim is None else Color.BLUE if victim == color else Color.RED)
    elif action.action_type == ActionType.BUILD_ROAD:
        value = tuple(sorted(value))
    return _action_indices(multiplayer)[(action.action_type, value)]


class CatanEnv(gym.Env):
    """A Catan game that one learner plays against scripted or saved players.

    The environment offers no player to player trades. It keeps bank trades,
    port trades, the robber, development cards and the opening placements.
    """
    metadata = {"render_modes": []}

    def __init__(self, config=Config(), opponent="random"):
        super().__init__()
        if not callable(opponent) and opponent not in ("random", "greedy"):
            raise ValueError("opponent must be random, greedy, or a callable(env, color)")
        self.config, self.opponent = config, opponent
        self.action_space = gym.spaces.Discrete(len(action_table(config.multiplayer)))
        self.reset(seed=0)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=self.observe().shape, dtype=np.float32)

    def legal(self, color):
        """Return the legal actions of one player, keyed by action number."""
        if self.game.state.current_color() != color:
            raise ValueError("Legal actions were asked for a player who is not acting")
        multiplayer = self.config.multiplayer
        colors = self.game.state.colors
        result = {action_id(action, color, colors, multiplayer): action
                  for action in self.game.playable_actions}
        if len(result) != len(self.game.playable_actions):
            raise RuntimeError("Two actions share one action number")
        return result

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

    def observe(self, color=None):
        """Return the observation of one player."""
        color = self.learner if color is None else color
        if self.config.multiplayer:
            return self._observe_multiplayer(color)
        state = self.game.state
        enemy = next(c for c in state.colors if c != color)
        features = [float(state.board.robber_coordinate == c) for c in COORDINATES]
        for node in NODES:
            building = state.board.buildings.get(node)
            features.extend((building == (color, "SETTLEMENT"), building == (color, "CITY"),
                             building == (enemy, "SETTLEMENT"), building == (enemy, "CITY")))
        for edge in EDGES:
            owner = state.board.roads.get(edge)
            features.extend((owner == color, owner == enemy))
        for owner in (color, enemy):
            features.extend(public_values(state, owner))
        features.extend(private_values(state, color))
        features.extend((float(state.colors[state.current_turn_index] == color),
                         float(state.is_initial_build_phase), float(state.is_discarding),
                         float(state.current_prompt == ActionPrompt.MOVE_ROBBER),
                         float(state.is_road_building), state.free_roads_available / 2))
        features.extend(state.discard_counts[state.color_to_index[c]] / 48 for c in (color, enemy))
        features.extend((self.config.target_vp / 12,
                         min(state.num_turns / self.config.max_turns, 1),
                         min(self.actions / self.config.max_actions, 1)))
        return np.concatenate((self.static, np.clip(np.asarray(features, dtype=np.float32), 0, 1)))

    def _observe_multiplayer(self, color):
        """Return the observation of one player in format 4.

        The player blocks start with the player who asks. The other blocks
        follow the turn order. This keeps the view the same for every seat.
        """
        state = self.game.state
        start = state.colors.index(color)
        colors = state.colors[start:] + state.colors[:start]
        shared = [float(state.board.robber_coordinate == c) for c in COORDINATES]
        shared.extend(private_values(state, color))
        shared.extend((float(state.is_initial_build_phase), float(state.is_discarding),
                       float(state.is_road_building), state.free_roads_available / 2,
                       self.config.target_vp / 12,
                       min(state.num_turns / self.config.max_turns, 1),
                       min(self.actions / self.config.max_actions, 1), len(colors) / 4))
        slots = np.zeros((4, PLAYER_SIZE), dtype=np.float32)
        for index, owner in enumerate(colors):
            values = [1., float(state.colors[state.current_turn_index] == owner)]
            for node in NODES:
                building = state.board.buildings.get(node)
                values.extend((building == (owner, "SETTLEMENT"), building == (owner, "CITY")))
            values.extend(state.board.roads.get(edge) == owner for edge in EDGES)
            values.extend(public_values(state, owner))
            values.append(state.discard_counts[state.color_to_index[owner]] / 48)
            slots[index] = values
        shared = np.asarray(shared, dtype=np.float32)
        return np.clip(np.concatenate((self.static, shared, slots.ravel())), 0, 1)

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
            self.game.execute(action)
            self.actions += 1
            # The learner sees the current board only. It has no memory of past
            # actions, so the record is not needed and would grow forever.
            self.game.state.action_records.clear()

    def _advance(self):
        """Play every opponent action until the learner must choose."""
        from .opponents import choose_action
        while winner(self.game) is None and self.game.state.current_color() != self.learner:
            color = self.game.state.current_color()
            chosen = self.opponent(self, color) if callable(self.opponent) else choose_action(self, color, self.opponent)
            legal = self.legal(color)
            if chosen not in legal:
                raise ValueError(f"The opponent chose an illegal action {chosen}")
            self._execute(legal[chosen])

    def _info(self, outcome=None):
        state = self.game.state
        winning = winner(self.game)
        limits = [name for name, reached in (("max_turns", state.num_turns >= self.config.max_turns),
                                             ("max_actions", self.actions >= self.config.max_actions))
                  if reached] if outcome == "truncated" else []
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
        truncated = not terminated and (self.game.state.num_turns >= self.config.max_turns
                                        or self.actions >= self.config.max_actions)
        self.done = terminated or truncated
        outcome = ("win" if winning == self.learner else "loss") if terminated else "truncated" if truncated else None
        self._refresh()
        reward = (1.0 if winning == self.learner else -1.0) if terminated else 0.0
        return self.observe(), reward, terminated, truncated, self._info(outcome)
