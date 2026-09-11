"""Paired-seat evaluation and separated simulation/encoding/inference timing."""
import argparse
import hashlib
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import tracemalloc

import numpy as np
import torch

from .agents import choose_action
from .game import CatanEnv, Config
from .training import load_policy, runtime


_EVALUATOR_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def select(env, policy, obs):
    if isinstance(policy, str):
        if policy in ("random", "greedy"):
            return choose_action(env, policy)
        from .teacher import opponent_for
        return opponent_for(policy, env.config)(env, env.learner)
    action, _ = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
    return int(action)


def paired_interval(values, seed=0):
    """Resample BOARD/SEED pairs, not correlated individual seats."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = [rng.choice(values, len(values), replace=True).mean() for _ in range(2000)]
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate(policy="random", opponent="greedy", config=None, pairs=100, seed=100_000, envs=1):
    if pairs < 1 or envs < 1:
        raise ValueError("pairs and envs must be positive")
    config = config or Config()
    # Evaluation fixes the count and rotates every seat on each board seed.
    config = replace(config, player_counts=())
    players = config.players
    if opponent == 'mixed':
        from .training import LeagueOpponents
        opponent = LeagueOpponents(['greedy', 'builder', 'expansion', 'development'], config)
    if isinstance(opponent, str) and opponent not in ("random", "greedy"):
        from .teacher import opponent_for
        opponent = opponent_for(opponent, config)
    rows = []
    started = time.perf_counter()
    jobs = iter((game_seed, seat) for game_seed in range(seed, seed + pairs) for seat in range(players))
    active = []
    def reset(env):
        job = next(jobs, None)
        if job is None:
            return None
        obs, _ = env.reset(seed=job[0], options={"seat": job[1]})
        return [env, obs, 0, Counter()]
    for _ in range(min(envs, players * pairs)):
        active.append(reset(CatanEnv(config, opponent)))
    while active:
        if envs == 1 or isinstance(policy, str):
            actions = [select(env, policy, obs) for env, obs, _, _ in active]
        else:
            policy.policy.set_training_mode(False)
            with torch.no_grad():
                observations, _ = policy.policy.obs_to_tensor(np.stack([a[1] for a in active]))
                distribution = policy.policy.get_distribution(observations,
                    action_masks=np.stack([a[0].action_masks() for a in active]))
                logits = distribution.distribution.logits
                actions = logits.argmax(-1).cpu().numpy()
                top = logits.topk(2).values
                # Different GEMM batch sizes can perturb near ties. Use the
                # original single-state inference for these decisions.
                for index in torch.where(top[:, 0] - top[:, 1] < 1e-5)[0].tolist():
                    env, obs, _, _ = active[index]
                    actions[index] = select(env, policy, obs)
        remaining = []
        for (env, obs, decisions, action_counts), action in zip(active, actions):
            action = int(action)
            action_counts[env._legal[action].action_type.value] += 1
            obs, _, terminated, truncated, info = env.step(action)
            decisions += 1
            if terminated or truncated:
                from catanatron.state_functions import player_key
                state = env.game.state
                key = player_key(state, env.learner)
                rows.append({**info, "decisions": decisions, "action_counts": dict(action_counts),
                             "cities": 4 - state.player_state[key + "_CITIES_AVAILABLE"],
                             "settlements": 5 - state.player_state[key + "_SETTLEMENTS_AVAILABLE"],
                             "roads": 15 - state.player_state[key + "_ROADS_AVAILABLE"],
                             "has_army": bool(state.player_state[key + "_HAS_ARMY"]),
                             "has_road": bool(state.player_state[key + "_HAS_ROAD"])})
                next_game = reset(env)
                if next_game:
                    remaining.append(next_game)
                else:
                    env.close()
            else:
                remaining.append([env, obs, decisions, action_counts])
        active = remaining
    rows.sort(key=lambda row: (row['seed'], row['seat']))
    wins = [int(row["outcome"] == "win") for row in rows]
    pair_wins = np.array(wins).reshape(-1, players).mean(axis=1)
    counts = {kind: sum(row["outcome"] == kind for row in rows)
              for kind in ("win", "loss", "truncated")}
    elapsed = time.perf_counter() - started
    return {"game": asdict(config), "seed_start": seed, "pairs": pairs,
            "games": len(rows), "evaluation_envs": envs, "evaluation_source_sha256": _EVALUATOR_SHA, "outcomes": counts,
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
    """Same test seeds for random, heuristic and model, independent of training."""
    policies = {"random": "random", "greedy": "greedy", "model": policy}
    if reference is not None:
        policies["initial"] = reference
    reports = {name: evaluate(p, opponent, config, pairs, seed, envs) for name, p in policies.items()}
    learned = np.array([r["outcome"] == "win" for r in reports["model"]["rows"]])
    for baseline in policies.keys() - {"model"}:
        reference = np.array([r["outcome"] == "win" for r in reports[baseline]["rows"]])
        differences = (learned.astype(float) - reference).reshape(-1, (config or Config()).players).mean(axis=1)
        reports["model"]["win_rate_delta_vs_" + baseline] = {
            "mean": float(differences.mean()),
            "95pct_paired_bootstrap": paired_interval(differences)}
    return reports


def speed(steps=5000, repeats=3, policy=None):
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
            obs, _, term, trunc, _ = env.step(choose_action(env))
            primitive += env.actions - before
            if term or trunc:
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
    # Separate allocation measurement so tracing does not contaminate timing.
    tracemalloc.start()
    obs, _ = env.reset(seed=99)
    for _ in range(min(steps, 1000)):
        obs, _, term, trunc, _ = env.step(choose_action(env))
        if term or trunc:
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
    parser.add_argument("--model", help="Saved run; omitted for baseline evaluation")
    parser.add_argument("--opponent", default="greedy")
    parser.add_argument('--players', type=int, choices=(2, 3, 4), help='Fixed evaluation player count')
    parser.add_argument('--suite', action='store_true', help='Evaluate 2/3/4 players against each style and mixed lineups')
    parser.add_argument("--policy", choices=("random", "greedy", "builder", "expansion", "development"), default="random")
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--eval-envs", type=int, default=16, help="Batch independent games; 1 uses original inference")
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
    opponent = args.opponent
    if args.compare and not args.model:
        parser.error("--compare requires --model")
    if args.suite:
        if args.speed or args.compare:
            parser.error('--suite cannot be combined with --speed or --compare')
        if args.model and not config.multiplayer:
            parser.error('--suite requires a multiplayer checkpoint')
        result = {}
        for players in (2, 3, 4):
            for style in ('greedy', 'builder', 'expansion', 'development', 'mixed'):
                key = f'{players}p-{style}'
                result[key] = evaluate(model, style, replace(config, players=players, multiplayer=True,
                                       player_counts=()), args.pairs, args.seed, args.eval_envs)
                report = result[key]
                print(f'{key}: {report["win_rate_all_games"]:.1%}, {report["outcomes"]}', flush=True)
    elif args.compare and not args.speed:
        initial = Path(args.model)
        initial = (initial if initial.is_dir() else initial.parent) / "initial.zip"
        reference = load_policy(initial)[0] if initial.exists() else None
        result = compare(model, opponent, config, args.pairs, args.seed, reference, args.eval_envs)
    else:
        result = (speed(policy=model if args.model else None) if args.speed else
                  evaluate(model, opponent, config, args.pairs, args.seed, args.eval_envs))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    if args.suite or (args.compare and not args.speed):
        result = {k: {a: b for a, b in v.items() if a != "rows"} for k, v in result.items()}
    else:
        result = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
