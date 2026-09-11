"""Measure how well a model plays, and how fast the code runs.

Every seat plays the same board, so a lucky board helps nobody. The game, the
observation and the network are timed apart.
"""
import argparse
import json
import statistics
import resource
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from catanatron.state_functions import player_key

from .game import CatanEnv, Config
from .opponents import choose_action, opponent_for
from .training import League, file_sha256, load_policy, runtime

BASELINES = ("random", "greedy", "builder", "expansion", "development")
STYLES = ("greedy", "builder", "expansion", "development", "mixed")
_EVALUATOR_SHA = file_sha256(__file__)


def select(env, policy, obs):
    """Return the action of one player. The player may be a name or a model."""
    if isinstance(policy, str):
        return (choose_action(env, kind=policy) if policy in ("random", "greedy")
                else opponent_for(policy, env.config)(env, env.learner))
    action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
    return int(action)


def select_batch(policy, active):
    """Return one action for each game that is running.

    One batch is faster than one call per game. A different batch size can move
    the last digits of a score, so score a near tie one game at a time.
    """
    policy.policy.set_training_mode(False)
    with torch.no_grad():
        observations, _ = policy.policy.obs_to_tensor(np.stack([game[1] for game in active]))
        distribution = policy.policy.get_distribution(
            observations, action_masks=np.stack([game[0].action_masks() for game in active]))
        logits = distribution.distribution.logits
        actions = logits.argmax(-1).cpu().numpy()
        top = logits.topk(2).values
    for index in torch.where(top[:, 0] - top[:, 1] < 1e-5)[0].tolist():
        env, obs = active[index][0], active[index][1]
        actions[index] = select(env, policy, obs)
    return actions


def paired_interval(values, seed=0):
    """Return a 95 percent interval. Resample whole boards, not single seats.

    The seats of one board share the same tiles, so they are not independent.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate(policy="random", opponent="greedy", config=None, pairs=100, seed=100_000, envs=1):
    """Play every seat of each board seed, and report how the games ended."""
    if pairs < 1 or envs < 1:
        raise ValueError("pairs and envs must be positive")
    config = config or Config()
    # Evaluation fixes the player count and rotates every seat of a board.
    config = replace(config, player_counts=())
    players = config.players
    if opponent == "mixed":
        opponent = League(["greedy", "builder", "expansion", "development"], config)
    elif isinstance(opponent, str) and opponent not in ("random", "greedy"):
        opponent = opponent_for(opponent, config)
    rows = []
    started = time.perf_counter()
    jobs = iter((board, seat) for board in range(seed, seed + pairs) for seat in range(players))

    def start(env):
        job = next(jobs, None)
        if job is None:
            return None
        obs, _ = env.reset(seed=job[0], options={"seat": job[1]})
        return [env, obs, 0, Counter()]

    active = [start(CatanEnv(config, opponent)) for _ in range(min(envs, players * pairs))]
    while active:
        if envs == 1 or isinstance(policy, str):
            actions = [select(env, policy, obs) for env, obs, _, _ in active]
        else:
            actions = select_batch(policy, active)
        remaining = []
        for (env, obs, decisions, action_counts), action in zip(active, actions):
            action = int(action)
            action_counts[env._legal[action].action_type.value] += 1
            obs, _, terminated, truncated, info = env.step(action)
            decisions += 1
            if not (terminated or truncated):
                remaining.append([env, obs, decisions, action_counts])
                continue
            state = env.game.state
            key = player_key(state, env.learner)
            rows.append({**info, "decisions": decisions, "action_counts": dict(action_counts),
                         "cities": 4 - state.player_state[key + "_CITIES_AVAILABLE"],
                         "settlements": 5 - state.player_state[key + "_SETTLEMENTS_AVAILABLE"],
                         "roads": 15 - state.player_state[key + "_ROADS_AVAILABLE"],
                         "has_army": bool(state.player_state[key + "_HAS_ARMY"]),
                         "has_road": bool(state.player_state[key + "_HAS_ROAD"])})
            next_game = start(env)
            if next_game:
                remaining.append(next_game)
            else:
                env.close()
        active = remaining
    rows.sort(key=lambda row: (row["seed"], row["seat"]))
    wins = [int(row["outcome"] == "win") for row in rows]
    pair_wins = np.array(wins).reshape(-1, players).mean(axis=1)
    elapsed = time.perf_counter() - started
    return {"game": asdict(config), "seed_start": seed, "pairs": pairs,
            "games": len(rows), "evaluation_envs": envs,
            "evaluation_source_sha256": _EVALUATOR_SHA,
            "outcomes": {kind: sum(row["outcome"] == kind for row in rows)
                         for kind in ("win", "loss", "truncated")},
            "truncation_limits": {limit: sum(limit in row["truncation_limits"] for row in rows)
                                  for limit in ("max_turns", "max_actions")},
            "win_rate_all_games": float(np.mean(wins)),
            "win_rate_95pct_paired_bootstrap": paired_interval(pair_wins),
            "win_rate_by_seat": [float(np.mean(wins[s::players])) for s in range(players)],
            "mean_learner_vp": statistics.mean(row["learner_vp"] for row in rows),
            "stalled_games": sum(row["cities"] == 0 and row["settlements"] == 2 for row in rows),
            "two_vp_games": sum(row["learner_vp"] <= 2 for row in rows),
            "mean_turns": statistics.mean(row["turns"] for row in rows),
            "seconds": elapsed, "games_per_second": len(rows) / elapsed,
            "rows": rows}


def compare(policy, opponent="greedy", config=None, pairs=100, seed=100_000, reference=None, envs=1):
    """Play the model and the baselines on the same seeds, and report the gap."""
    policies = {"random": "random", "greedy": "greedy", "model": policy}
    if reference is not None:
        policies["initial"] = reference
    reports = {name: evaluate(p, opponent, config, pairs, seed, envs) for name, p in policies.items()}
    learned = np.array([r["outcome"] == "win" for r in reports["model"]["rows"]])
    players = (config or Config()).players
    for baseline in policies.keys() - {"model"}:
        other = np.array([r["outcome"] == "win" for r in reports[baseline]["rows"]])
        differences = (learned.astype(float) - other).reshape(-1, players).mean(axis=1)
        reports["model"]["win_rate_delta_vs_" + baseline] = {
            "mean": float(differences.mean()),
            "95pct_paired_bootstrap": paired_interval(differences)}
    return reports


def speed(steps=5000, repeats=3, policy=None):
    """Time the game, the observation, the mask and the network apart."""
    if min(steps, repeats) < 1:
        raise ValueError("steps and repeats must be positive")
    rates, encoding, masks, inference = [], [], [], []
    env = CatanEnv()
    for repeat in range(repeats):
        obs, _ = env.reset(seed=repeat)
        if policy is not None:
            select(env, policy, obs)
        primitive = 0
        started = time.perf_counter()
        for _ in range(steps):
            before = env.actions
            obs, _, terminated, truncated, _ = env.step(choose_action(env))
            primitive += env.actions - before
            if terminated or truncated:
                obs, _ = env.reset()
        elapsed = time.perf_counter() - started
        rates.append((steps / elapsed, primitive / elapsed))
        for samples, function in ((encoding, lambda: env.observe(env.learner)),
                                  (masks, env.action_masks)):
            started = time.perf_counter()
            for _ in range(steps):
                function()
            samples.append((time.perf_counter() - started) * 1e6 / steps)
        if policy is not None:
            started = time.perf_counter()
            for _ in range(min(steps, 1000)):
                select(env, policy, obs)
            inference.append((time.perf_counter() - started) * 1e6 / min(steps, 1000))
    # Measure memory apart from time. Tracing makes the code slower.
    tracemalloc.start()
    obs, _ = env.reset(seed=99)
    for _ in range(min(steps, 1000)):
        obs, _, terminated, truncated, _ = env.step(choose_action(env))
        if terminated or truncated:
            env.reset()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"steps_per_repeat": steps, "repeats": repeats,
              "learner_steps_per_second_median": statistics.median(r[0] for r in rates),
              "primitive_actions_per_second_median": statistics.median(r[1] for r in rates),
              "encoding_us_median": statistics.median(encoding),
              "cached_mask_us_median": statistics.median(masks),
              "traced_python_peak_mib": peak / 2**20,
              "process_peak_rss_mib": rss / (2**20 if sys.platform == "darwin" else 1024),
              "observation_floats": env.observation_space.shape[0],
              "actions": int(env.action_space.n), **runtime()}
    if inference:
        result["single_policy_inference_us_median"] = statistics.median(inference)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="A saved run; leave out to measure a baseline")
    parser.add_argument("--opponent", default="greedy")
    parser.add_argument("--players", type=int, choices=(2, 3, 4), help="Fixed player count")
    parser.add_argument("--suite", action="store_true",
                        help="Play 2, 3 and 4 players against each style and a mixed league")
    parser.add_argument("--policy", choices=BASELINES, default="random")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--eval-envs", type=int, default=16,
                        help="Play this many games at once; 1 scores one game at a time")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, config = load_policy(args.model) if args.model else (args.policy, Config())
    if args.players:
        config = replace(config, players=args.players, player_counts=(),
                         multiplayer=config.multiplayer or (not args.model and args.players != 2))
    if args.compare and not args.model:
        parser.error("--compare needs --model")
    if args.suite:
        if args.speed or args.compare:
            parser.error("--suite cannot be used with --speed or --compare")
        if args.model and not config.multiplayer:
            parser.error("--suite needs a multiplayer model")
        result = {}
        for players in (2, 3, 4):
            for style in STYLES:
                key = f"{players}p-{style}"
                result[key] = evaluate(model, style,
                                       replace(config, players=players, multiplayer=True,
                                               player_counts=()),
                                       args.pairs, args.seed, args.eval_envs)
                print(f'{key}: {result[key]["win_rate_all_games"]:.1%}, '
                      f'{result[key]["outcomes"]}', flush=True)
    elif args.compare and not args.speed:
        initial = Path(args.model)
        initial = (initial if initial.is_dir() else initial.parent) / "initial.zip"
        reference = load_policy(initial)[0] if initial.exists() else None
        result = compare(model, args.opponent, config, args.pairs, args.seed, reference, args.eval_envs)
    elif args.speed:
        result = speed(policy=model if args.model else None)
    else:
        result = evaluate(model, args.opponent, config, args.pairs, args.seed, args.eval_envs)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    if args.suite or (args.compare and not args.speed):
        result = {name: {k: v for k, v in report.items() if k != "rows"}
                  for name, report in result.items()}
    else:
        result = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
