"""Experimental public-state planner; evaluates legal actions without copying a game."""
import numpy as np
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.models.map import number_probability
from catanatron.state_functions import player_key


_CITY = np.array([0, 0, 0, 2, 3])
_SETTLEMENT = np.array([1, 1, 1, 1, 0])
_DEVELOPMENT = np.array([0, 0, 1, 1, 1])
_ROAD = np.array([1, 1, 0, 0, 0])


def _road_hand_after(hand, state):
    """Return the post-road hand; setup and Road Building roads are free."""
    if state.is_initial_build_phase or state.is_road_building:
        return hand
    return hand - _ROAD


def _readiness(cards, can_settle, can_city, development_available=True):
    projects = []
    if development_available:
        projects.append(180 - 35 * np.maximum(_DEVELOPMENT - cards, 0).sum())
    if can_settle:
        projects.append(240 - 40 * np.maximum(_SETTLEMENT - cards, 0).sum())
    if can_city:
        projects.append(300 - 50 * np.maximum(_CITY - cards, 0).sum())
    return max(projects, default=0)


def choose_action(env, color, contender=False, feasible=False, available=False,
                  resource_weights=None, diversity_bonus=None):
    """Choose a legal action from public board state and the acting player's hand."""
    state = env.game.state
    key = player_key(state, color)
    hand = np.array([state.player_state[key + '_' + r + '_IN_HAND'] for r in RESOURCES])
    blocked = state.board.map.land_tiles[state.board.robber_coordinate]

    def production(node):
        return sum(number_probability(tile.number) for tile in state.board.map.adjacent_tiles[node]
                   if tile.resource and tile is not blocked)

    income = np.zeros(5)
    for node, (owner, building) in state.board.buildings.items():
        if owner == color:
            for tile in state.board.map.adjacent_tiles[node]:
                if tile.resource and tile is not blocked:
                    income[RESOURCES.index(tile.resource)] += number_probability(tile.number) * (2 if building == 'CITY' else 1)

    has_settlement = any(owner == color and building == 'SETTLEMENT'
                         for owner, building in state.board.buildings.values())
    can_settle = state.player_state[key + '_SETTLEMENTS_AVAILABLE'] > 0 and bool(state.board.buildable_node_ids(color))
    can_city = state.player_state[key + '_CITIES_AVAILABLE'] > 0 and has_settlement

    def readiness(cards):
        return _readiness(cards, can_settle, can_city,
                          bool(state.development_listdeck) if available else True)

    def site(node):
        resources = np.zeros(5)
        for tile in state.board.map.adjacent_tiles[node]:
            if tile.resource and tile is not blocked:
                resources[RESOURCES.index(tile.resource)] += number_probability(tile.number)
        production = resources.sum() if resource_weights is None else np.dot(resources, resource_weights)
        diversity = (18 if contender else 12) if diversity_bonus is None else diversity_bonus
        return production * (90 if contender else 80) + np.count_nonzero(resources) * diversity - (income * resources).sum() * (3 if contender else 5)

    def robbed(tile, owner):
        return sum(number_probability(tile.number) * (2 if building == 'CITY' else 1)
                   for node in tile.nodes.values() if (entry := state.board.buildings.get(node))
                   for building_owner, building in [entry] if building_owner == owner)

    def score(action):
        kind, value = action.action_type, action.value
        if kind == A.BUILD_CITY:
            return 1000 + site(value)
        if kind == A.BUILD_SETTLEMENT:
            return 900 + site(value)
        if kind == A.BUILD_ROAD:
            destinations = [node for node in value if node not in state.board.buildings
                            and not any(other in state.board.buildings for other in STATIC_GRAPH.neighbors(node))]
            site_score = max(map(site, destinations)) / 20 if destinations else 0
            if feasible:
                return site_score + readiness(_road_hand_after(hand, state)) - readiness(hand)
            return site_score
        if kind == A.MARITIME_TRADE:
            give, take = RESOURCES.index(value[0]), RESOURCES.index(value[-1])
            after = hand.copy()
            after[give] -= len(value) - 1
            after[take] += 1
            return readiness(after) - readiness(hand)
        if kind == A.BUY_DEVELOPMENT_CARD:
            return 220 + (30 if state.player_state[key + '_HAS_ARMY'] else 0)
        if kind == A.MOVE_ROBBER:
            tile, victim = value
            victim_cards = 0 if victim is None else sum(state.player_state[player_key(state, victim) + '_' + r + '_IN_HAND'] for r in RESOURCES)
            return 100 * (robbed(state.board.map.land_tiles[tile], victim) - 1.5 * robbed(state.board.map.land_tiles[tile], color)) + min(victim_cards, 7)
        if kind == A.DISCARD_RESOURCE:
            resource = RESOURCES.index(value)
            if feasible:
                after = hand.copy()
                after[resource] -= 1
                return readiness(after) + 1e-6 * (hand[resource] - income[resource])
            return hand[resource] - 8 * max((_CITY - hand)[resource], (_SETTLEMENT - hand)[resource], (_DEVELOPMENT - hand)[resource]) - income[resource]
        if kind == A.PLAY_MONOPOLY:
            resource = RESOURCES.index(value)
            opponents = sum(sum(state.player_state[player_key(state, other) + '_' + r + '_IN_HAND'] for r in RESOURCES)
                            for other in state.colors if other != color)
            return 250 + 20 * max(_CITY[resource] - hand[resource], _SETTLEMENT[resource] - hand[resource]) + opponents / 5
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 250 + sum(max(_CITY[RESOURCES.index(r)] - hand[RESOURCES.index(r)],
                                 _SETTLEMENT[RESOURCES.index(r)] - hand[RESOURCES.index(r)]) for r in value)
        return {A.ROLL: 150, A.PLAY_KNIGHT_CARD: 140, A.PLAY_ROAD_BUILDING: 130,
                A.END_TURN: 0}.get(kind, 20)

    legal = env.encoder.legal(env.game, color)
    return max(legal, key=lambda action_id: (score(legal[action_id]), -action_id))


def planner(env, color):
    return choose_action(env, color)


def planner_contender(env, color):
    return choose_action(env, color, contender=True)


def planner_feasible(env, color):
    """Planner variant that prices paid roads and discards by aftermath."""
    return choose_action(env, color, feasible=True)


def planner_available(env, color):
    """Feasible planner that also excludes unavailable development purchases."""
    return choose_action(env, color, feasible=True, available=True)


def _route_score(env, color, action):
    """Best public settlement site one to three edges beyond a legal road."""
    state = env.game.state
    blocked = state.board.map.land_tiles[state.board.robber_coordinate]
    buildings = state.board.buildings
    roads = state.board.roads
    endpoints = tuple(action.value)
    own_nodes = {node for node, (owner, _) in buildings.items() if owner == color}
    own_edges = {edge for edge, owner in roads.items() if owner == color}
    starts = [node for node in endpoints if node not in own_nodes and not any(node in edge for edge in own_edges)]
    starts = starts or list(endpoints)

    income = np.zeros(5)
    for node, (owner, building) in buildings.items():
        if owner == color:
            for tile in state.board.map.adjacent_tiles[node]:
                if tile.resource and tile is not blocked:
                    income[RESOURCES.index(tile.resource)] += number_probability(tile.number) * (2 if building == 'CITY' else 1)

    def site(node):
        if node in buildings or any(neighbor in buildings for neighbor in STATIC_GRAPH.neighbors(node)):
            return 0.
        resources = np.zeros(5)
        for tile in state.board.map.adjacent_tiles[node]:
            if tile.resource and tile is not blocked:
                resources[RESOURCES.index(tile.resource)] += number_probability(tile.number)
        return resources.sum() * 90 + np.count_nonzero(resources) * 18 - (income * resources).sum() * 3

    best = 0.
    for start in starts:
        frontier, seen = [(start, 0)], {start}
        while frontier:
            node, depth = frontier.pop(0)
            if depth:
                best = max(best, site(node) / depth)
            if depth == 3 or (node in buildings and node not in own_nodes):
                continue
            for neighbor in STATIC_GRAPH.neighbors(node):
                edge = tuple(sorted((node, neighbor)))
                if neighbor in seen or (edge in roads and roads[edge] != color):
                    continue
                seen.add(neighbor)
                frontier.append((neighbor, depth + 1))
    return best


def planner_route(env, color):
    """Frozen planner plus public route selection when it has chosen a road."""
    action_id = planner(env, color)
    legal = env.encoder.legal(env.game, color)
    if legal[action_id].action_type != A.BUILD_ROAD:
        return action_id
    roads = [candidate for candidate, action in legal.items() if action.action_type == A.BUILD_ROAD]
    return max(roads, key=lambda candidate: (_route_score(env, color, legal[candidate]), -candidate))
