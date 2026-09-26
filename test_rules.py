"""Exercise our engine corrections and card beliefs: python test_rules.py [seeds]."""
import sys
from collections import Counter
import numpy as np
from catanatron.apply_action import yield_resources
from catanatron.models.board import Board
from catanatron.models.enums import ActionType as A, ActionPrompt
from catanatron.models.map import build_map
from catanatron.models.player import Color
from catanatron.state_functions import get_player_freqdeck, player_key
from rlcatan.game import CatanEnv, Config, longest_road

def test_rules(seeds=6):
    seen = Counter()
    for players in (2, 3, 4):
        env = CatanEnv(Config(target_vp=10, players=players, seats=True, trading=True, max_actions=3000))
        env.clear_records = False
        for seed in range(seeds):
            env.reset(seed=seed, options={"seat": 0})
            game, offers = env.game, Counter()
            execute = game.execute
            def checked(action, *args, **kwargs):
                state, board = game.state, game.state.board
                hand = lambda c: np.array(get_player_freqdeck(state, c))
                if seed % 2 and not state.is_initial_build_phase and sum(state.resource_freqdeck) > 60:
                    state.resource_freqdeck = [3] * 5
                bank, before = np.array(state.resource_freqdeck), {c: hand(c) for c in state.colors}
                if action.action_type == A.OFFER_TRADE:
                    offers[action.color, state.num_turns] += 1
                    assert offers[action.color, state.num_turns] <= env.config.offers
                result = execute(action, *args, **kwargs)
                after = {c: hand(c) for c in state.colors}
                assert np.array_equal(bank + sum(before.values()), state.resource_freqdeck + sum(after.values()))
                assert min(state.resource_freqdeck) >= 0 and all((h >= 0).all() for h in after.values())
                if action.action_type == A.ROLL and sum(result.result) != 7:
                    demand, _ = yield_resources(board, [100] * 5, sum(result.result))
                    for resource in range(5):
                        claimants = [c for c, amounts in demand.items() if amounts[resource]]
                        total = sum(demand[c][resource] for c in claimants)
                        for c in state.colors:
                            due = demand.get(c, [0] * 5)[resource]
                            expected = due if total <= bank[resource] else min(due, bank[resource]) if len(claimants) == 1 else 0
                            assert after[c][resource] - before[c][resource] == expected
                        seen["bank shortage"] += total > bank[resource]
                if action.action_type == A.PLAY_YEAR_OF_PLENTY:
                    assert len(action.value) == 2 or bank.sum() < 2
                if state.is_resolving_trade and state.current_prompt == ActionPrompt.DECIDE_TRADE:
                    assert state.current_player_index != state.current_trade[10]
                if action.action_type == A.CONFIRM_TRADE:
                    give, take, partner = np.array(action.value[:5]), np.array(action.value[5:10]), action.value[10]
                    assert np.array_equal(after[action.color] - before[action.color], take - give)
                    assert np.array_equal(after[partner] - before[partner], give - take)
                for observer, counts in game.counts.items():
                    for other, count in counts.items():
                        assert np.all(count.certain <= after[other] + 1e-6)
                        assert abs(count.expected().sum() - after[other].sum()) < 1e-6
                for color in state.colors:
                    ps, key = state.player_state, player_key(state, color)
                    pieces = [kind for owner, kind in board.buildings.values() if owner == color]
                    points = pieces.count("SETTLEMENT") + 2 * pieces.count("CITY")
                    points += 2 * (ps[key + "_HAS_ARMY"] + ps[key + "_HAS_ROAD"])
                    assert ps[key + "_VICTORY_POINTS"] == points
                    assert ps[key + "_ACTUAL_VICTORY_POINTS"] == points + ps[key + "_VICTORY_POINT_IN_HAND"]
                    assert ps[key + "_HAS_ROAD"] == (board.road_color == color)
                    assert board.road_lengths[color] == longest_road(board, color)
                lengths = list(board.road_lengths.values())
                if board.road_color is not None:
                    assert board.road_lengths[board.road_color] == max(lengths) >= 5
                else:
                    assert max(lengths) < 5 or lengths.count(max(lengths)) > 1
                current = state.colors[state.current_turn_index]
                won = state.player_state[player_key(state, current) + "_ACTUAL_VICTORY_POINTS"] >= game.vps_to_win
                assert game.winning_color() == (current if won else None)
                seen[action.action_type.name] += 1
                return result
            game.execute = checked
            while not env.done:
                env.step(int(env.np_random.choice(sorted(env._legal))))
            assert env.actions <= env.config.max_actions and env.game.state.action_records
    for kind in ("bank shortage", "OFFER_TRADE", "CONFIRM_TRADE", "PLAY_YEAR_OF_PLENTY", "BUILD_CITY"):
        assert seen[kind], f"Check did not exercise {kind}"
    board = Board(build_map("BASE"))
    board.build_settlement(Color.BLUE, 20, initial_build_phase=True)
    board.build_settlement(Color.RED, 46, initial_build_phase=True)
    board.build_road(Color.RED, (19, 46))
    board.build_road(Color.RED, (19, 20))
    assert longest_road(board, Color.RED) == 2  # The segment reaching an enemy building counts.
    print("rules ok")

if __name__ == "__main__":
    test_rules(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
