"""Small legal-action baselines sharing the learner's action representation."""
import numpy as np
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.state_functions import player_key


def choose_action(env, kind="random", color=None):
    color = env.learner if color is None else color
    legal = env.encoder.legal(env.game, color)
    ids = sorted(legal)
    if kind == "random":
        return int(env.np_random.choice(ids))
    if kind != "greedy":
        raise ValueError(f"Unknown baseline {kind}")
    state = env.game.state
    key = player_key(state, color)
    hand = {r: state.player_state[key + "_" + r + "_IN_HAND"] for r in RESOURCES}

    def score(action):
        kind, value = action.action_type, action.value
        if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY):
            return (100 if kind == A.BUILD_CITY else 90) + env.production[value]
        if kind == A.BUILD_ROAD:
            return 20 + max(env.production[n] for n in value if n in env.production)
        if kind == A.DISCARD_RESOURCE:
            return hand[value]
        if kind == A.MOVE_ROBBER:
            tile = state.board.map.land_tiles[value[0]]
            return sum((1 if b[0] != color else -2) * (2 if b[1] == "CITY" else 1)
                       for n in tile.nodes.values() if (b := state.board.buildings.get(n)))
        if kind == A.PLAY_MONOPOLY:
            return -hand[value] / 100 + 35
        if kind == A.PLAY_YEAR_OF_PLENTY:
            return 35 - sum(hand[r] for r in value) / 100
        if kind == A.MARITIME_TRADE:
            # ponytail: one-step shortage heuristic; search only if measured stronger opponents are needed.
            return 10 + (hand[value[0]] - hand[value[-1]]) / 100
        return {A.ROLL: 50, A.BUY_DEVELOPMENT_CARD: 40, A.PLAY_KNIGHT_CARD: 35,
                A.PLAY_ROAD_BUILDING: 35, A.END_TURN: 0}.get(kind, 0)

    scores = np.asarray([score(legal[i]) for i in ids])
    best = np.flatnonzero(scores == scores.max())
    return ids[int(env.np_random.choice(best))]
