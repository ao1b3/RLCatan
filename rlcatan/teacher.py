"""A separate city-focused demonstrator. The greedy benchmark is unchanged."""
import numpy as np
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.state_functions import player_key
from catanatron.models.map import number_probability
from catanatron.models.board import STATIC_GRAPH


def builder(env, color, style="builder"):
    """Current-state styles; builder retains its original benchmark behavior."""
    state = env.game.state
    key = player_key(state, color)
    hand = np.array([state.player_state[key + '_' + r + '_IN_HAND'] for r in RESOURCES])
    income = np.zeros(5)
    for node, (owner, kind) in state.board.buildings.items():
        if owner == color:
            for tile in state.board.map.adjacent_tiles[node]:
                if tile.resource:
                    income[RESOURCES.index(tile.resource)] += number_probability(tile.number) * (2 if kind == 'CITY' else 1)
    preferences = {"builder": [.7, .7, 1., 1.7, 1.7],
                   "expansion": [1.5, 1.5, 1., 1., .6],
                   "development": [.5, .5, 1.5, 1.5, 1.5]}
    weights = np.array(preferences[style]) / np.sqrt(.08 + income)
    def production(node):
        return sum(number_probability(t.number) * weights[RESOURCES.index(t.resource)]
                   for t in state.board.map.adjacent_tiles[node] if t.resource)
    city_cost = np.array([0, 0, 0, 2, 3])
    settlement_cost = np.array([1, 1, 1, 1, 0])
    development_cost = np.array([0, 0, 1, 1, 1])
    has_settlement = any(owner == color and kind == 'SETTLEMENT' for owner, kind in state.board.buildings.values())
    save_city = has_settlement and hand[4] >= 2 and hand[3] >= 1
    target = city_cost if has_settlement else settlement_cost
    if not has_settlement and state.player_state[key + '_SETTLEMENTS_AVAILABLE'] == 0:
        target = development_cost
    if style == "expansion":
        target = settlement_cost
        save_city = False
    elif style == "development":
        target = development_cost
        save_city = False
    def score(action):
        kind, value = action.action_type, action.value
        if kind == A.BUILD_SETTLEMENT:
            return 100 + production(value)
        if kind == A.BUILD_CITY:
            return (90 if style == "expansion" else 120) + production(value)
        if kind == A.BUILD_ROAD:
            if state.is_initial_build_phase or state.is_road_building:
                return 20 + max(production(n) for n in value)
            candidates = [n for n in value if n not in state.board.buildings
                          and all(other not in state.board.buildings for other in STATIC_GRAPH.neighbors(n))]
            return 20 + max(map(production, candidates)) if candidates and not save_city else 0
        if kind == A.BUY_DEVELOPMENT_CARD:
            return 0 if save_city else 130 if style == "development" else 50
        if kind == A.MARITIME_TRADE:
            give, take = RESOURCES.index(value[0]), RESOURCES.index(value[-1])
            cost = len(value) - 1
            if hand[give] - cost < target[give] or hand[take] >= target[take]:
                return 0
            return 30 + (target[take] - hand[take]) / (income[take] + .1)
        if kind == A.DISCARD_RESOURCE:
            return (hand[RESOURCES.index(value)] - target[RESOURCES.index(value)]) / (income[RESOURCES.index(value)] + .1)
        if kind == A.MOVE_ROBBER:
            tile = state.board.map.land_tiles[value[0]]
            return sum((1 if owner != color else -2) * (2 if building == 'CITY' else 1)
                       for node in tile.nodes.values() if (entry := state.board.buildings.get(node))
                       for owner, building in [entry]) * (0 if tile.number is None else number_probability(tile.number))
        if kind == A.PLAY_MONOPOLY:
            return 35 + (target[RESOURCES.index(value)] - hand[RESOURCES.index(value)]) / (income[RESOURCES.index(value)] + .1)
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 35 + sum((target[RESOURCES.index(r)] - hand[RESOURCES.index(r)]) / (income[RESOURCES.index(r)] + .1) for r in value)
        return {A.ROLL: 50, A.PLAY_KNIGHT_CARD: 35, A.PLAY_ROAD_BUILDING: 35, A.END_TURN: 10}.get(kind, 0)
    legal = env.encoder.legal(env.game, color)
    scores = {index: score(action) for index, action in legal.items()}
    best = max(scores.values())
    return int(env.np_random.choice([index for index, score in scores.items() if score == best]))


STYLES = ("random", "greedy", "builder", "expansion", "development")
OPPONENTS = (*STYLES, "noisy-builder", "planner", "planner-contender", "planner-feasible", "planner-available")


def opponent_for(name, config):
    from functools import partial
    from .agents import choose_action
    if name in ("random", "greedy"):
        return lambda env, color: choose_action(env, name, color)
    if name == "noisy-builder":
        def noisy(env, color):
            legal = env.encoder.legal(env.game, color)
            if len(legal) > 1 and env.np_random.random() < .05:
                return int(env.np_random.choice(sorted(legal)))
            return builder(env, color)
        return noisy
    if name in ('planner', 'planner-contender', 'planner-feasible', 'planner-available'):
        from .strong_teacher import planner, planner_available, planner_contender, planner_feasible
        return {'planner': planner, 'planner-contender': planner_contender,
                'planner-feasible': planner_feasible, 'planner-available': planner_available}[name]
    if name in STYLES:
        return partial(builder, style=name)
    from .training import frozen_opponent
    return frozen_opponent(name, config)
