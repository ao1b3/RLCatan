"""Measure the habits of a player: the robber, exposure to a seven, builds it passes up.

    .venv/bin/python tools/habits.py runs/long-a runs/attn-c3 planner --opponent planner --pairs 100

Every player plays every seat of the same boards. For each robber move the
value of a tile is the income per roll it takes from the opponent less what
it takes from the mover. A turn is exposed when it ends with more than seven
cards. A build is passed when a settlement or city was legal at END_TURN.
"""
import argparse
import json
from collections import Counter
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import torch
from catanatron.models.enums import ActionType as A, RESOURCES
from catanatron.models.map import number_probability
from catanatron.state_functions import get_actual_victory_points, get_player_freqdeck, player_key

from rlcatan.game import CatanEnv, Config, node_yield, winner
from rlcatan.opponents import CITY, DEVELOPMENT, LIBRARY, SCRIPTED, SETTLEMENT, opponent_for
from rlcatan.training import load_policy


SHAPES = {(1, 1): "1_for_1", (2, 1): "2_for_1", (1, 2): "1_for_2"}  # cards given, cards asked

def robbed(state, tile, owner):
    """Return the income per roll the robber takes from one player on one tile."""
    if tile.number is None:
        return 0.0
    return sum(number_probability(tile.number) * (2 if kind == "CITY" else 1)
               for node in tile.nodes.values()
               for color, kind in [state.board.buildings.get(node, (None, None))] if color == owner)


def finishes_a_build(hand, after):
    """Whether a hand that could not pay for a settlement, a city or a development
    card can pay for one after a trade."""
    return any((after >= cost).all() and not (hand >= cost).all() for cost in (SETTLEMENT, CITY, DEVELOPMENT))


def habits(player, opponent, pairs, seed, robber_rule=False, players=None, trading=False, offers=None,
           answer_rule=False):
    """Play the boards and count the habits. robber_rule replaces the player's robber
    choice with the tile of most net income, and answer_rule its answer to an offer
    with acceptance whenever the trade finishes a build, to measure what the habit costs."""
    torch.set_num_threads(1)
    if player in SCRIPTED or player in LIBRARY:
        model, config = None, Config(target_vp=10, seats=players is not None, trading=trading, players=players or 2)
    else:
        model, config = load_policy(player)
        if players:
            config = replace(config, players=players, player_counts=())
    if offers is not None and config.trading:
        config = replace(config, offers=offers)
    if model is None:
        decide = opponent_for(player, config)
    env = CatanEnv(config, opponent_for(opponent, config))
    totals = Counter()
    ratios, hands = [], []
    for board in range(seed, seed + pairs):
        for seat in range(config.players):
            obs, _ = env.reset(seed=board, options={"seat": seat})
            me = env.learner
            opponents = tuple(c for c in env.game.state.colors if c != me)
            done = False
            win_turn, confirm_turn = None, None
            while not done:
                mask = env.action_masks()
                legal = env._legal
                if model is None:
                    action = decide(env, me)
                else:
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
                chosen = legal[action]
                state = env.game.state
                kind = chosen.action_type
                if answer_rule and kind in (A.ACCEPT_TRADE, A.REJECT_TRADE):
                    accept = next((a for a, m in legal.items() if m.action_type == A.ACCEPT_TRADE), None)
                    hand = np.array(get_player_freqdeck(state, me))
                    value = legal[accept].value if accept is not None else None
                    take = accept is not None and finishes_a_build(hand, hand + np.array(value[:5]) - np.array(value[5:10]))
                    action = accept if take else next(a for a, m in legal.items() if m.action_type == A.REJECT_TRADE)
                    chosen = legal[action]
                    kind = chosen.action_type
                if kind == A.BUILD_SETTLEMENT and state.is_initial_build_phase:
                    # The opening: how good a spot, out of the spots on offer.
                    spots = {a: node_yield(state, m.value).sum() for a, m in legal.items()
                             if m.action_type == A.BUILD_SETTLEMENT}
                    yields = node_yield(state, chosen.value)
                    totals["openings"] += 1
                    totals["opening_yield"] += yields.sum()
                    totals["opening_best_yield"] += max(spots.values())
                    totals["opening_top3"] += spots[action] >= sorted(spots.values())[-3]
                    totals["opening_port"] += chosen.value in {n for r in state.board.map.port_nodes.values() for n in r}
                    totals["opening_resources"] += int((yields > 0).sum())
                if (get_actual_victory_points(state, me) >= config.target_vp - 2 and mask.sum() > 1
                        and kind not in (A.ROLL, A.DISCARD_RESOURCE, A.MOVE_ROBBER)):
                    # Is a winning move on the table, and does the player take it?
                    winning = set()
                    for a, m in legal.items():
                        if m.action_type in (A.BUILD_SETTLEMENT, A.BUILD_CITY, A.BUILD_ROAD, A.PLAY_KNIGHT_CARD):
                            trial = env.game.copy()
                            trial.execute(m)
                            if winner(trial) == me:
                                winning.add(a)
                    turn = (board, seat, state.num_turns)
                    if winning and turn != win_turn:
                        # Count the turn once; a trade before the winning build is no miss.
                        win_turn = turn
                        totals["win_available"] += 1
                        for kind_name in {legal[a].action_type.value for a in winning}:
                            totals["win_kind_" + kind_name] += 1
                    if action in winning and turn == win_turn:
                        totals["win_taken"] += 1
                        totals["win_taken_kind_" + chosen.action_type.value] += 1
                if kind == A.PLAY_KNIGHT_CARD:
                    totals["knights"] += 1
                if kind == A.PLAY_MONOPOLY:
                    enemy_cards = sum((np.array(get_player_freqdeck(state, other)) for other in opponents),
                                      np.zeros(len(RESOURCES), dtype=int))
                    totals["monopolies"] += 1
                    totals["monopoly_cards"] += enemy_cards[RESOURCES.index(chosen.value)]
                    totals["monopoly_best_cards"] += enemy_cards.max()
                if kind == A.MOVE_ROBBER and mask.sum() > 1:
                    tiles = state.board.map.land_tiles
                    # Every opposing seat can gain from a robber block, not only seat one.
                    theirs = {a: sum(robbed(state, tiles[m.value[0]], other) for other in opponents)
                              for a, m in legal.items() if m.action_type == A.MOVE_ROBBER}
                    values = {a: theirs[a] - robbed(state, tiles[m.value[0]], me)
                              for a, m in legal.items() if m.action_type == A.MOVE_ROBBER}
                    if robber_rule:
                        # Prefer a tile with a victim to steal from when the income ties.
                        action = max(values, key=lambda a: (values[a], legal[a].value[1] is not None))
                        chosen = legal[action]
                    best = max(values.values())
                    totals["robber_moves"] += 1
                    totals["robber_best_net"] += values[action] >= best - 1e-9
                    totals["robber_best_theirs"] += theirs[action] >= max(theirs.values()) - 1e-9
                    totals["robber_self_harm"] += robbed(state, tiles[chosen.value[0]], me) > 0
                    totals["robber_no_steal"] += chosen.value[1] is None and any(m.value[1] for m in legal.values()
                                                                                 if m.action_type == A.MOVE_ROBBER)
                    ratios.append(theirs[action] / max(theirs.values()) if max(theirs.values()) > 0 else 1.0)
                    # Blocking what the other player is saving for: the tile's
                    # resource is one they hold two or more of.
                    tile = tiles[chosen.value[0]]
                    owners = {state.board.buildings.get(n, (None,))[0] for n in tile.nodes.values()} & set(opponents)
                    if tile.resource is not None and owners:
                        totals["robber_on_their_tiles"] += 1
                        resource = RESOURCES.index(tile.resource)
                        totals["robber_blocks_their_stock"] += any(
                            get_player_freqdeck(state, owner)[resource] >= 2 for owner in owners)
                elif kind == A.END_TURN and mask.sum() > 1:
                    hand = sum(get_player_freqdeck(state, me))
                    hands.append(hand)
                    totals["end_turns"] += 1
                    totals["exposed_end_turns"] += hand > 7
                    totals["expected_seven_loss"] += (hand // 2) / 6 if hand > 7 else 0
                    kinds = {m.action_type for m in legal.values()}
                    totals["passed_city"] += A.BUILD_CITY in kinds
                    totals["passed_settlement"] += A.BUILD_SETTLEMENT in kinds
                    totals["passed_dev_card"] += A.BUY_DEVELOPMENT_CARD in kinds
                elif kind == A.DISCARD_RESOURCE:
                    totals["cards_discarded"] += 1
                elif kind == A.MARITIME_TRADE:
                    totals["trades"] += 1
                    totals[f"trades_{sum(g is not None for g in chosen.value[:-1])}_to_1"] += 1
                elif kind == A.BUY_DEVELOPMENT_CARD:
                    totals["dev_cards"] += 1
                if kind == A.OFFER_TRADE:
                    totals["offers"] += 1
                    totals["offer_" + SHAPES[sum(chosen.value[:5]), sum(chosen.value[5:10])]] += 1
                elif kind in (A.ACCEPT_TRADE, A.REJECT_TRADE):
                    totals["asked"] += 1
                    accept = next((m for m in legal.values() if m.action_type == A.ACCEPT_TRADE), None)
                    totals["could_accept"] += accept is not None
                    totals["accepted"] += kind == A.ACCEPT_TRADE
                    if accept is not None:
                        # A good offer finishes a build for the player that its hand alone does not.
                        hand = np.array(get_player_freqdeck(state, me))
                        good = finishes_a_build(hand, hand + np.array(accept.value[:5]) - np.array(accept.value[5:10]))
                        totals["good_offers"] += good
                        totals["good_accepted"] += good and kind == A.ACCEPT_TRADE
                elif kind == A.CONFIRM_TRADE:
                    totals["confirmed"] += 1
                    totals["done_" + SHAPES[sum(chosen.value[:5]), sum(chosen.value[5:10])]] += 1
                    partner_vp = state.player_state[player_key(state, chosen.value[10]) + "_VICTORY_POINTS"]
                    my_vp = state.player_state[player_key(state, me) + "_VICTORY_POINTS"]
                    totals["confirmed_with_leader"] += partner_vp >= my_vp + 2
                    confirm_turn = state.num_turns
                elif kind == A.CANCEL_TRADE:
                    totals["cancelled"] += 1
                if kind in (A.BUILD_SETTLEMENT, A.BUILD_CITY, A.BUY_DEVELOPMENT_CARD, A.BUILD_ROAD) and confirm_turn == state.num_turns:
                    totals["built_after_trade"] += 1
                    confirm_turn = None
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            key = player_key(env.game.state, me)
            ps = env.game.state.player_state
            totals["games"] += 1
            totals["wins"] += info["outcome"] == "win"
            totals["vp"] += info["learner_vp"]
            totals["turns"] += info["turns"]
            totals["settlements"] += 5 - ps[key + "_SETTLEMENTS_AVAILABLE"]
            totals["cities"] += 4 - ps[key + "_CITIES_AVAILABLE"]
            totals["roads"] += 15 - ps[key + "_ROADS_AVAILABLE"]
            totals["has_road"] += bool(ps[key + "_HAS_ROAD"])
            totals["has_army"] += bool(ps[key + "_HAS_ARMY"])
            totals["lost_at_9"] += info["outcome"] != "win" and info["learner_vp"] >= config.target_vp - 1
            totals["vp_cards"] += ps[key + "_VICTORY_POINT_IN_HAND"]
    games = totals["games"]
    return player, {
        "games": games, "win_rate": round(totals["wins"] / games, 3),
        "vp": round(totals["vp"] / games, 2), "turns": round(totals["turns"] / games, 1),
        "robber": {"moves_per_game": round(totals["robber_moves"] / games, 2),
                   "best_tile_net": round(totals["robber_best_net"] / max(totals["robber_moves"], 1), 3),
                   "best_tile_for_them": round(totals["robber_best_theirs"] / max(totals["robber_moves"], 1), 3),
                   "value_taken_vs_best": round(float(np.mean(ratios)), 3) if ratios else None,
                   "hits_own_tile": round(totals["robber_self_harm"] / max(totals["robber_moves"], 1), 3),
                   "skips_a_steal": round(totals["robber_no_steal"] / max(totals["robber_moves"], 1), 3),
                   "blocks_a_resource_they_stock": round(totals["robber_blocks_their_stock"] / max(totals["robber_on_their_tiles"], 1), 3)},
        "seven": {"end_turns_over_7": round(totals["exposed_end_turns"] / max(totals["end_turns"], 1), 3),
                  "mean_hand_at_end_turn": round(float(np.mean(hands)), 2) if hands else None,
                  "expected_cards_lost_per_game": round(totals["expected_seven_loss"] / games, 2),
                  "cards_discarded_per_game": round(totals["cards_discarded"] / games, 2)},
        "opening": {"yield_vs_best_spot": round(totals["opening_yield"] / max(totals["opening_best_yield"], 1e-9), 3),
                    "pips_per_settlement": round(36 * totals["opening_yield"] / max(totals["openings"], 1), 2),
                    "in_top_3_spots": round(totals["opening_top3"] / max(totals["openings"], 1), 3),
                    "on_a_port": round(totals["opening_port"] / max(totals["openings"], 1), 3),
                    "resources_per_settlement": round(totals["opening_resources"] / max(totals["openings"], 1), 2)},
        "monopoly": {"played_per_game": round(totals["monopolies"] / games, 2),
                     "cards_taken_vs_best": round(totals["monopoly_cards"] / max(totals["monopoly_best_cards"], 1e-9), 3)},
        "races": {"longest_road_held": round(totals["has_road"] / games, 3),
                  "largest_army_held": round(totals["has_army"] / games, 3),
                  "knights_played": round(totals["knights"] / games, 2),
                  "vp_cards_held": round(totals["vp_cards"] / games, 2)},
        "trading": {"offers_per_game": round(totals["offers"] / games, 2),
                    "offer_kinds": {k[6:]: v for k, v in totals.items() if k.startswith("offer_")},
                    "offers_confirmed": round(totals["confirmed"] / max(totals["offers"], 1), 3),
                    "trade_kinds": {k[5:]: v for k, v in totals.items() if k.startswith("done_")},
                    "offers_cancelled": round(totals["cancelled"] / max(totals["offers"], 1), 3),
                    "trades_per_game": round(totals["confirmed"] / games, 2),
                    "trades_with_leader": round(totals["confirmed_with_leader"] / max(totals["confirmed"], 1), 3),
                    "built_same_turn_after_trade": round(totals["built_after_trade"] / max(totals["confirmed"], 1), 3),
                    "asked_per_game": round(totals["asked"] / games, 2),
                    "answers_per_game": round(totals["could_accept"] / games, 2),
                    "accepts_when_able": round(totals["accepted"] / max(totals["could_accept"], 1), 3),
                    "good_offers_per_game": round(totals["good_offers"] / games, 2),
                    "accepts_good_offers": round(totals["good_accepted"] / max(totals["good_offers"], 1), 3)},
        "endgame": {"winning_turn_converted": round(totals["win_taken"] / max(totals["win_available"], 1), 3),
                    "winning_turns_seen": totals["win_available"],
                    "missed_by_kind": {k[9:]: totals[k] - totals["win_taken_kind_" + k[9:]]
                                       for k in totals if k.startswith("win_kind_")},
                    "lost_with_9_plus_vp": round(totals["lost_at_9"] / games, 3)},
        "builds": {"settlements": round(totals["settlements"] / games, 2), "cities": round(totals["cities"] / games, 2),
                   "roads": round(totals["roads"] / games, 2), "dev_cards": round(totals["dev_cards"] / games, 2),
                   "trades": round(totals["trades"] / games, 2),
                   "trade_ratios": {k[7:]: round(v / games, 2) for k, v in totals.items() if k.startswith("trades_")},
                   "end_turns_with_city_legal": round(totals["passed_city"] / max(totals["end_turns"], 1), 3),
                   "end_turns_with_settlement_legal": round(totals["passed_settlement"] / max(totals["end_turns"], 1), 3),
                   "end_turns_with_dev_card_legal": round(totals["passed_dev_card"] / max(totals["end_turns"], 1), 3)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("players", nargs="+", help="Saved runs or scripted names")
    parser.add_argument("--opponent", default="planner")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9_200_000)
    parser.add_argument("--robber-rule", action="store_true",
                        help="Override each player's robber choice with the tile of most net income")
    parser.add_argument("--table", type=int, choices=(2, 3, 4), help="Player count; scripted players then use the seat format")
    parser.add_argument("--trading", action="store_true", help="Scripted players may trade")
    parser.add_argument("--offers", type=int, help="Offers per turn for a trading player; 0 turns trading off")
    parser.add_argument("--answer-rule", action="store_true",
                        help="Override each answer to an offer with acceptance when the trade finishes a build")
    args = parser.parse_args()
    with Pool(len(args.players)) as pool:
        results = dict(pool.starmap(habits, [(p, args.opponent, args.pairs, args.seed, args.robber_rule,
                                              args.table, args.trading, args.offers, args.answer_rule)
                                             for p in args.players]))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
