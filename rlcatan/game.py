"""Catan learning adapter, with legacy two-player and shared 2–4-player formats."""
from dataclasses import dataclass

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
from catanatron.state_functions import player_key


@dataclass(frozen=True)
class Config:
    target_vp: int = 6
    max_turns: int = 300
    max_actions: int = 2000
    players: int = 2
    multiplayer: bool = False
    player_counts: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'player_counts', tuple(self.player_counts))
        if self.players not in (2, 3, 4) or any(n not in (2, 3, 4) for n in self.player_counts):
            raise ValueError("players and player_counts must be 2..4")
        if (self.players != 2 or self.player_counts) and not self.multiplayer:
            raise ValueError("Use multiplayer format for variable player counts")
        if not 3 <= self.target_vp <= 10 or self.max_turns < 1 or self.max_actions < 1:
            raise ValueError("target_vp must be 3..10 and limits positive")


class Game(CoreGame):
    """Repair upstream victory, interrupted-road and sole-claimant shortage rules."""
    def winning_color(self):
        return winner(self)

    def execute(self, action, validate_action=True, action_record=None):
        previous = self.state.board.road_color
        bank = self.state.resource_freqdeck.copy() if action.action_type == ActionType.ROLL else None
        result = super().execute(action, validate_action, action_record)
        if bank is not None and sum(result.result) != 7:
            demand, _ = yield_resources(self.state.board, [100] * 5, sum(result.result))
            repaired = False
            for index, resource in enumerate(RESOURCES):
                claimants = [color for color, amounts in demand.items() if amounts[index]]
                if len(claimants) == 1 and 0 < bank[index] < demand[claimants[0]][index]:
                    key = player_key(self.state, claimants[0])
                    self.state.player_state[key + "_" + resource + "_IN_HAND"] += bank[index]
                    self.state.resource_freqdeck[index] -= bank[index]
                    repaired = True
            if repaired:
                self.playable_actions = generate_playable_actions(self.state)
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_ROAD):
            board = self.state.board
            longest = max(board.road_lengths.values(), default=0)
            leaders = [c for c, length in board.road_lengths.items() if length == longest and length >= 5]
            owner = previous if previous in leaders else leaders[0] if len(leaders) == 1 else None
            board.road_color, board.road_length = owner, longest
            for color in self.state.colors:
                key = player_key(self.state, color)
                ps = self.state.player_state
                adjustment = 2 * (int(color == owner) - int(ps[key + "_HAS_ROAD"]))
                ps[key + "_HAS_ROAD"] = color == owner
                ps[key + "_VICTORY_POINTS"] += adjustment
                ps[key + "_ACTUAL_VICTORY_POINTS"] += adjustment
        if sum(self.state.resource_freqdeck) >= 2:
            self.playable_actions = [a for a in self.playable_actions
                                     if a.action_type != ActionType.PLAY_YEAR_OF_PLENTY or len(a.value) == 2]
        return result


def winner(game):
    """Catan victory is checked only for the owner of the current turn."""
    state = game.state
    color = state.colors[state.current_turn_index]
    return color if state.player_state[player_key(state, color) + "_ACTUAL_VICTORY_POINTS"] >= game.vps_to_win else None


class ActionEncoder:
    """Stable BASE action IDs; BLUE always means self and RED opponent."""
    def __init__(self, multiplayer=False):
        self.colors = (Color.BLUE, Color.RED, Color.ORANGE, Color.WHITE)
        self.multiplayer = multiplayer
        self.actions = tuple(get_action_array(self.colors if multiplayer else self.colors[:2], "BASE"))
        self.indices = {action: i for i, action in enumerate(self.actions)}

    def index(self, action, color, colors=None):
        value = action.value
        if action.action_type == ActionType.MOVE_ROBBER:
            coordinate, victim = value
            if self.multiplayer:
                offset = colors.index(color)
                relative = colors[offset:] + colors[:offset]
                value = (coordinate, None if victim is None else self.colors[relative.index(victim)])
            else:
                value = (coordinate, None if victim is None else Color.BLUE if victim == color else Color.RED)
        elif action.action_type == ActionType.BUILD_ROAD:
            value = tuple(sorted(value))
        return self.indices[(action.action_type, value)]

    def legal(self, game, color):
        if game.state.current_color() != color:
            raise ValueError("Legal actions requested for a player who is not acting")
        result = {self.index(action, color, game.state.colors): action for action in game.playable_actions}
        if len(result) != len(game.playable_actions):
            raise RuntimeError("Action encoding collision")
        return result


class CatanEnv(gym.Env):
    """A transition spans one learner action and all intervening opponent actions.

    No player trades are offered. Standard bank/port trades, robber, development
    cards and learned snake setup remain. Timeouts are learner decision boundaries.
    """
    metadata = {"render_modes": []}

    def __init__(self, config=Config(), opponent="random"):
        super().__init__()
        if not callable(opponent) and opponent not in ("random", "greedy"):
            raise ValueError("opponent must be random, greedy, or callable(env, color)")
        self.config, self.opponent = config, opponent
        self.encoder = ActionEncoder(config.multiplayer)
        template = build_map("BASE")
        self.nodes = tuple(sorted(template.land_nodes))
        self.edges = tuple(sorted(tuple(sorted(e)) for e in get_edges(template.land_nodes)))
        self.coordinates = tuple(sorted(template.land_tiles))
        self.action_space = gym.spaces.Discrete(len(self.encoder.actions))
        self.reset(seed=0)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=self.observe().shape, dtype=np.float32)

    def _cache_map(self):
        board_map = self.game.state.board.map
        features = []
        for coordinate in self.coordinates:
            tile = board_map.land_tiles[coordinate]
            features.extend(float(tile.resource == resource) for resource in [*RESOURCES, None])
            features.extend((0 if tile.number is None else tile.number / 12,
                             0 if tile.resource is None else number_probability(tile.number) * 6))
        for node in self.nodes:
            features.extend(float(node in board_map.port_nodes[resource]) for resource in [*RESOURCES, None])
        self.static = np.asarray(features, dtype=np.float32)
        self.production = {node: sum(number_probability(tile.number) for tile in board_map.adjacent_tiles[node]
                                     if tile.resource is not None) for node in self.nodes}

    def observe(self, color=None):
        color = self.learner if color is None else color
        if self.config.multiplayer:
            from .multiplayer import observe
            return observe(self, color)
        state = self.game.state
        enemy = next(c for c in state.colors if c != color)
        features = []
        features.extend(float(state.board.robber_coordinate == c) for c in self.coordinates)
        for node in self.nodes:
            building = state.board.buildings.get(node)
            features.extend((building == (color, "SETTLEMENT"), building == (color, "CITY"),
                             building == (enemy, "SETTLEMENT"), building == (enemy, "CITY")))
        for edge in self.edges:
            owner = state.board.roads.get(edge)
            features.extend((owner == color, owner == enemy))
        ps = state.player_state
        public = (("VICTORY_POINTS", 12), ("ROADS_AVAILABLE", 15),
                  ("SETTLEMENTS_AVAILABLE", 5), ("CITIES_AVAILABLE", 4),
                  ("HAS_ROAD", 1), ("HAS_ARMY", 1), ("HAS_ROLLED", 1),
                  ("LONGEST_ROAD_LENGTH", 15))
        for owner in (color, enemy):
            key = player_key(state, owner)
            features.extend(ps[key + "_" + name] / scale for name, scale in public)
            features.extend(ps[key + "_PLAYED_" + card] / 14 for card in DEVELOPMENT_CARDS if card != "VICTORY_POINT")
            features.append(sum(ps[key + "_" + resource + "_IN_HAND"] for resource in RESOURCES) / 95)
            features.append(sum(ps[key + "_" + card + "_IN_HAND"] for card in DEVELOPMENT_CARDS) / 25)
        key = player_key(state, color)
        features.extend(ps[key + "_" + resource + "_IN_HAND"] / 19 for resource in RESOURCES)
        features.extend(ps[key + "_" + card + "_IN_HAND"] / 14 for card in DEVELOPMENT_CARDS)
        features.extend(float(ps[key + "_" + card + "_OWNED_AT_START"]) for card in DEVELOPMENT_CARDS if card != "VICTORY_POINT")
        features.extend((ps[key + "_ACTUAL_VICTORY_POINTS"] / 12,
                         float(ps[key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"])))
        features.extend(float(count > 0) for count in state.resource_freqdeck)
        features.append(len(state.development_listdeck) / 25)
        features.extend(float(state.current_prompt == prompt) for prompt in ActionPrompt)
        features.extend((float(state.colors[state.current_turn_index] == color),
                         float(state.is_initial_build_phase), float(state.is_discarding),
                         float(state.current_prompt == ActionPrompt.MOVE_ROBBER), float(state.is_road_building),
                         state.free_roads_available / 2))
        features.extend(state.discard_counts[state.color_to_index[c]] / 48 for c in (color, enemy))
        features.extend((self.config.target_vp / 12, min(state.num_turns / self.config.max_turns, 1),
                         min(self.actions / self.config.max_actions, 1)))
        return np.concatenate((self.static, np.clip(np.asarray(features, dtype=np.float32), 0, 1)))

    def mask_for(self, color):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[list(self.encoder.legal(self.game, color))] = True
        return mask

    def action_masks(self):
        return self._mask.copy()

    def _refresh(self):
        self._legal = self.encoder.legal(self.game, self.learner) if self.game.state.current_color() == self.learner else {}
        self._mask = np.zeros(self.action_space.n, dtype=bool)
        self._mask[list(self._legal)] = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.seed_value = int(seed if seed is not None else self.np_random.integers(0, 2**31))
        self.num_players = int(self.np_random.choice(self.config.player_counts)) if self.config.player_counts else self.config.players
        self.seat = options.get("seat", int(self.np_random.integers(self.num_players)))
        if self.seat not in range(self.num_players):
            raise ValueError("seat must be within the player count")
        self.game = Game([Player(c) for c in self.encoder.colors[:self.num_players]], seed=self.seed_value,
                         vps_to_win=self.config.target_vp)
        self.learner = self.game.state.colors[self.seat]
        self.actions = 0
        self.done = False
        self._cache_map()
        self._advance()
        self._refresh()
        return self.observe(), self._info()

    def _execute(self, action):
        if winner(self.game) is None:
            self.game.execute(action)
            self.actions += 1
            # ponytail: memoryless partial observation; add public event history only if benchmarks justify it.
            # Upstream rules never read action_records; clearing bounds episode memory.
            self.game.state.action_records.clear()

    def _advance(self):
        from .agents import choose_action
        while winner(self.game) is None and self.game.state.current_color() != self.learner:
            color = self.game.state.current_color()
            action_id = self.opponent(self, color) if callable(self.opponent) else choose_action(self, self.opponent, color)
            legal = self.encoder.legal(self.game, color)
            if action_id not in legal:
                raise ValueError(f"Opponent selected illegal action {action_id}")
            self._execute(legal[action_id])

    def _info(self, outcome=None):
        state = self.game.state
        winning = winner(self.game)
        return dict(opponent=getattr(self, "opponent_name", self.opponent if isinstance(self.opponent, str) else "callable"), outcome=outcome, winner=None if winning is None else winning.name,
                    truncation_limits=([name for name, reached in (
                        ("max_turns", state.num_turns >= self.config.max_turns),
                        ("max_actions", self.actions >= self.config.max_actions)) if reached]
                        if outcome == "truncated" else []),
                    learner_vp=state.player_state[player_key(state, self.learner) + "_ACTUAL_VICTORY_POINTS"],
                    opponent_vp=max(state.player_state[player_key(state, c) + "_VICTORY_POINTS"] for c in state.colors if c != self.learner),
                    players=self.num_players,
                    turns=state.num_turns, actions=self.actions, seat=self.seat, seed=self.seed_value)

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() after an episode ends")
        if not isinstance(action, (int, np.integer)) or int(action) not in self._legal:
            raise ValueError(f"Illegal action {action}")
        self._execute(self._legal[int(action)])
        self._advance()
        winning = winner(self.game)
        terminated = winning is not None
        truncated = not terminated and (self.game.state.num_turns >= self.config.max_turns or self.actions >= self.config.max_actions)
        self.done = terminated or truncated
        outcome = ("win" if winning == self.learner else "loss") if terminated else "truncated" if truncated else None
        self._refresh()
        reward = (1.0 if winning == self.learner else -1.0) if terminated else 0.0
        return self.observe(), reward, terminated, truncated, self._info(outcome)
