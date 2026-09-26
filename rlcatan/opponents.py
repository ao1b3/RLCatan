"""The scripted players that the learner trains and competes against.

Every player here reads the public board and its own hand. No player copies the
game to look ahead, and no player sees a hidden hand.

Each player gives every legal action a score. The player then takes the best
action. The named styles below differ only in how they score.
"""
from functools import partial

import numpy as np
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.models.map import number_probability
from catanatron.state_functions import (get_player_freqdeck, player_key,
                                        player_num_resource_cards)

from .game import node_yield, player_income

ROAD = np.array([1, 1, 0, 0, 0])
SETTLEMENT = np.array([1, 1, 1, 1, 0])
CITY = np.array([0, 0, 0, 2, 3])
DEVELOPMENT = np.array([0, 0, 1, 1, 1])


def pick_best(env, scores):
    """Return the action with the best score. Break a tie at random.

    The caller decides the order of the scores. That order decides which action
    a tie gives, so it must stay the same to repeat a game exactly.
    """
    best = max(scores.values())
    return int(env.np_random.choice([action for action, score in scores.items() if score == best]))


def tile_buildings(state, tile):
    """Return the owner and the kind of every building on one tile."""
    return [entry for node in tile.nodes.values()
            if (entry := state.board.buildings.get(node)) is not None]


def choose_action(env, color=None, kind="random"):
    """Take a random action, or the action with the best immediate value."""
    color = env.learner if color is None else color
    legal = env.legal(color)
    if kind == "random":
        return int(env.np_random.choice(sorted(legal)))
    if kind != "greedy":
        raise ValueError(f"Unknown baseline {kind}")
    state = env.game.state
    hand = np.array(get_player_freqdeck(state, color))

    def score(action):
        kind, value = action.action_type, action.value
        if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY):
            return (100 if kind == A.BUILD_CITY else 90) + env.production[value]
        if kind == A.BUILD_ROAD:
            return 20 + max(env.production[n] for n in value if n in env.production)
        if kind == A.DISCARD_RESOURCE:
            return hand[RESOURCES.index(value)]
        if kind == A.MOVE_ROBBER:
            tile = state.board.map.land_tiles[value[0]]
            return sum((1 if owner != color else -2) * (2 if building == "CITY" else 1)
                       for owner, building in tile_buildings(state, tile))
        if kind == A.PLAY_MONOPOLY:
            return -hand[RESOURCES.index(value)] / 100 + 35
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 35 - sum(hand[RESOURCES.index(r)] for r in value) / 100
        if kind == A.MARITIME_TRADE:
            # This looks one step ahead only. Search the tree if a stronger
            # opponent proves to be worth the time.
            return 10 + (hand[RESOURCES.index(value[0])] - hand[RESOURCES.index(value[-1])]) / 100
        return {A.ROLL: 50, A.BUY_DEVELOPMENT_CARD: 40, A.PLAY_KNIGHT_CARD: 35,
                A.PLAY_ROAD_BUILDING: 35, A.END_TURN: 0}.get(kind, 0)

    return pick_best(env, {action_id: score(legal[action_id]) for action_id in sorted(legal)})


def builder(env, color, style="builder"):
    """Build towards one goal, and trade for what that goal still needs.

    The builder style saves for cities. The expansion style spreads out with
    settlements. The development style buys development cards.
    """
    state = env.game.state
    key = player_key(state, color)
    hand = np.array(get_player_freqdeck(state, color))
    income = player_income(state, color)
    preferences = {"builder": [.7, .7, 1., 1.7, 1.7],
                   "expansion": [1.5, 1.5, 1., 1., .6],
                   "development": [.5, .5, 1.5, 1.5, 1.5]}
    # A resource that the player already collects is worth less.
    weights = np.array(preferences[style]) / np.sqrt(.08 + income)

    def production(node):
        return float(node_yield(state, node) @ weights)

    has_settlement = any(owner == color and building == "SETTLEMENT"
                         for owner, building in state.board.buildings.values())
    save_city = style == "builder" and has_settlement and hand[4] >= 2 and hand[3] >= 1
    if style == "expansion":
        target = SETTLEMENT
    elif style == "development":
        target = DEVELOPMENT
    elif has_settlement:
        target = CITY
    elif state.player_state[key + "_SETTLEMENTS_AVAILABLE"] == 0:
        target = DEVELOPMENT
    else:
        target = SETTLEMENT

    def score(action):
        kind, value = action.action_type, action.value
        if kind == A.BUILD_SETTLEMENT:
            return 100 + production(value)
        if kind == A.BUILD_CITY:
            return (90 if style == "expansion" else 120) + production(value)
        if kind == A.BUILD_ROAD:
            if state.is_initial_build_phase or state.is_road_building:
                return 20 + max(production(n) for n in value)
            free = [n for n in value if n not in state.board.buildings
                    and all(other not in state.board.buildings for other in STATIC_GRAPH.neighbors(n))]
            return 20 + max(map(production, free)) if free and not save_city else 0
        if kind == A.BUY_DEVELOPMENT_CARD:
            return 0 if save_city else 130 if style == "development" else 50
        if kind == A.MARITIME_TRADE:
            give, take = RESOURCES.index(value[0]), RESOURCES.index(value[-1])
            if hand[give] - (len(value) - 1) < target[give] or hand[take] >= target[take]:
                return 0
            return 30 + (target[take] - hand[take]) / (income[take] + .1)
        if kind == A.DISCARD_RESOURCE:
            resource = RESOURCES.index(value)
            return (hand[resource] - target[resource]) / (income[resource] + .1)
        if kind == A.MOVE_ROBBER:
            tile = state.board.map.land_tiles[value[0]]
            value_to_others = sum((1 if owner != color else -2) * (2 if building == "CITY" else 1)
                                  for owner, building in tile_buildings(state, tile))
            return value_to_others * (0 if tile.number is None else number_probability(tile.number))
        if kind == A.PLAY_MONOPOLY:
            resource = RESOURCES.index(value)
            return 35 + (target[resource] - hand[resource]) / (income[resource] + .1)
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 35 + sum((target[RESOURCES.index(r)] - hand[RESOURCES.index(r)])
                            / (income[RESOURCES.index(r)] + .1) for r in value)
        return {A.ROLL: 50, A.PLAY_KNIGHT_CARD: 35, A.PLAY_ROAD_BUILDING: 35,
                A.END_TURN: 10}.get(kind, 0)

    legal = env.legal(color)
    return pick_best(env, {action_id: score(action) for action_id, action in legal.items()})


def noisy_builder(env, color):
    """The builder style, but it takes a random action once in twenty."""
    legal = env.legal(color)
    if len(legal) > 1 and env.np_random.random() < .05:
        return int(env.np_random.choice(sorted(legal)))
    return builder(env, color)


def _readiness(cards, can_settle, can_city, can_buy):
    """Return how close a hand is to its best project.

    A project that is out of reach scores lower. Each missing card costs the
    same amount, so a trade that fills a gap always raises this number.
    """
    projects = []
    if can_buy:
        projects.append(180 - 35 * np.maximum(DEVELOPMENT - cards, 0).sum())
    if can_settle:
        projects.append(240 - 40 * np.maximum(SETTLEMENT - cards, 0).sum())
    if can_city:
        projects.append(300 - 50 * np.maximum(CITY - cards, 0).sum())
    return max(projects, default=0)


def planner(env, color, contender=False, feasible=False, available=False):
    """Score each legal action from the public board and from the hand.

    contender values a spread of resources more highly. feasible prices a paid
    road and a discard by the hand that is left. available stops the planner
    from saving for a development card once the deck is empty.
    """
    state = env.game.state
    key = player_key(state, color)
    hand = np.array(get_player_freqdeck(state, color))
    blocked = state.board.map.land_tiles[state.board.robber_coordinate]
    income = player_income(state, color, blocked)
    has_settlement = any(owner == color and building == "SETTLEMENT"
                         for owner, building in state.board.buildings.values())
    can_settle = (state.player_state[key + "_SETTLEMENTS_AVAILABLE"] > 0
                  and bool(state.board.buildable_node_ids(color)))
    can_city = state.player_state[key + "_CITIES_AVAILABLE"] > 0 and has_settlement
    can_buy = bool(state.development_listdeck) if available else True

    def readiness(cards):
        return _readiness(cards, can_settle, can_city, can_buy)

    current = readiness(hand)

    def site(node):
        """Return how good a node is to build on."""
        resources = node_yield(state, node, blocked)
        spread = np.count_nonzero(resources) * (18 if contender else 12)
        # A resource that the player already collects adds less.
        return (resources.sum() * (90 if contender else 80) + spread
                - (income * resources).sum() * (3 if contender else 5))

    def robbed(tile, owner):
        """Return how much income the robber takes from one player."""
        return sum(number_probability(tile.number) * (2 if building == "CITY" else 1)
                   for entry_owner, building in tile_buildings(state, tile) if entry_owner == owner)

    def score(action):
        kind, value = action.action_type, action.value
        if kind == A.BUILD_CITY:
            return 1000 + site(value)
        if kind == A.BUILD_SETTLEMENT:
            return 900 + site(value)
        if kind == A.BUILD_ROAD:
            free = [node for node in value if node not in state.board.buildings
                    and not any(other in state.board.buildings
                                for other in STATIC_GRAPH.neighbors(node))]
            reach = max(map(site, free)) / 20 if free else 0
            if feasible and not (state.is_initial_build_phase or state.is_road_building):
                return reach + readiness(hand - ROAD) - current
            return reach
        if kind == A.MARITIME_TRADE:
            after = hand.copy()
            after[RESOURCES.index(value[0])] -= len(value) - 1
            after[RESOURCES.index(value[-1])] += 1
            return readiness(after) - current
        if kind == A.BUY_DEVELOPMENT_CARD:
            return 220 + (30 if state.player_state[key + "_HAS_ARMY"] else 0)
        if kind == A.MOVE_ROBBER:
            tile, victim = value
            cards = 0 if victim is None else player_num_resource_cards(state, victim)
            land = state.board.map.land_tiles[tile]
            return 100 * (robbed(land, victim) - 1.5 * robbed(land, color)) + min(cards, 7)
        if kind == A.DISCARD_RESOURCE:
            resource = RESOURCES.index(value)
            if feasible:
                after = hand.copy()
                after[resource] -= 1
                return readiness(after) + 1e-6 * (hand[resource] - income[resource])
            missing = max((CITY - hand)[resource], (SETTLEMENT - hand)[resource],
                          (DEVELOPMENT - hand)[resource])
            return hand[resource] - 8 * missing - income[resource]
        if kind == A.PLAY_MONOPOLY:
            resource = RESOURCES.index(value)
            held = sum(player_num_resource_cards(state, other)
                       for other in state.colors if other != color)
            return (250 + 20 * max(CITY[resource] - hand[resource],
                                   SETTLEMENT[resource] - hand[resource]) + held / 5)
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 250 + sum(max(CITY[RESOURCES.index(r)] - hand[RESOURCES.index(r)],
                                 SETTLEMENT[RESOURCES.index(r)] - hand[RESOURCES.index(r)])
                             for r in value)
        return {A.ROLL: 150, A.PLAY_KNIGHT_CARD: 140, A.PLAY_ROAD_BUILDING: 130,
                A.END_TURN: 0}.get(kind, 20)

    legal = env.legal(color)
    # The planner always breaks a tie the same way, so its games repeat exactly.
    return max(legal, key=lambda action_id: (score(legal[action_id]), -action_id))


# Every scripted player, by the name that the command line uses.
SCRIPTED = {
    "random": partial(choose_action, kind="random"),
    "greedy": partial(choose_action, kind="greedy"),
    "builder": partial(builder, style="builder"),
    "expansion": partial(builder, style="expansion"),
    "development": partial(builder, style="development"),
    "noisy-builder": noisy_builder,
    "planner": planner,
    "planner-contender": partial(planner, contender=True),
    "planner-feasible": partial(planner, feasible=True),
    "planner-available": partial(planner, feasible=True, available=True),
}


def opponent_for(name, config):
    """Return a scripted player, or a saved model that plays as an opponent."""
    if name in SCRIPTED:
        return SCRIPTED[name]
    from .training import frozen_opponent
    return frozen_opponent(name, config)
