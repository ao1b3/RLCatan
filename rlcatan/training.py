"""Train masked PPO and record its rules, opponents, sources and metrics."""
import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from collections import Counter, defaultdict
from importlib.metadata import requires, version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from catanatron.models.actions import Action
from catanatron.models.enums import ActionType
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points

from .game import CatanEnv, Config, action_table, player_income, winner

SOURCE_FILES = {path.name: path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py"))}

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def model_path(path):
    """Resolve a run directory to model.zip."""
    path = Path(path)
    return path / "model.zip" if path.is_dir() else path

def runtime():
    """Record the machine, source hashes and package versions."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "source_sha256": {name: hashlib.sha256(data).hexdigest()
                              for name, data in SOURCE_FILES.items()},
            "packages": {p: version(p) for p in ("catanatron", "numpy", "torch", "gymnasium",
                                                 "stable-baselines3", "sb3-contrib")}}

def config_from_metadata(metadata):
    """Validate saved observation format and discard the retired multiplayer flag."""
    if metadata["format"] not in (3, 5):
        raise ValueError("Unsupported observation format")
    # Old metadata keeps this flag even when the retired format was unused.
    rules = dict(metadata["game"])
    if rules.pop("multiplayer", False):
        raise ValueError("Format 4 models are no longer supported")
    config = Config(**rules)
    if config.seats != (metadata["format"] == 5):
        raise ValueError("The saved format does not match the saved rules")
    return config

def load_policy(path, **overrides):
    """Load rules and weights; inference needs only a tiny unused rollout buffer."""
    path = model_path(path)
    metadata = json.loads((path.parent / "run.json").read_text())
    config = config_from_metadata(metadata)
    overrides = {"n_steps": 2, "n_envs": 1, "batch_size": 2, **overrides}
    try:
        model = MaskablePPO.load(path, device="cpu", custom_objects=overrides)
    except RuntimeError as error:
        if "size mismatch for " not in str(error):
            raise
        # Zero new inputs when loading weights saved before the network grew.
        from stable_baselines3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(path, device="cpu", custom_objects=overrides)
        model = MaskablePPO(data["policy_class"], None, device="cpu", _init_setup_model=False)
        model.__dict__.update(data)
        model._setup_model()
        transfer_weights(model.policy, params["policy"])
    # The critic of a shaped run scores a position less its potential.
    model.shaping = metadata.get("shaping", 0.0)
    return model, config

def transfer_weights(policy, source):
    """Copy a smaller network, zeroing grown weights; keep current code-defined buffers."""
    buffers = {name for name, _ in policy.named_buffers()}
    state = policy.state_dict()
    tables = {"kinds", "endpoints", "endpoint_masks", "tiles", "tile_masks", "give", "take", "delta", "count"}
    unexpected = [key for key in source if key not in state and key.rsplit(".", 1)[-1] not in tables]
    missing = [key for key in state if key not in source and key not in buffers]
    if unexpected or any(not key.startswith("mlp_extractor.") for key in missing):
        raise ValueError(f"transfer does not fit this network: {missing} {unexpected}")
    for key, value in source.items():
        if key in buffers or key not in state:
            continue
        target = state[key]
        if value.shape != target.shape:
            if value.dim() != target.dim() or any(v > t for v, t in zip(value.shape, target.shape)):
                raise ValueError(f"{key}: {tuple(value.shape)} does not fit {tuple(target.shape)}")
            grown = torch.zeros_like(target)
            grown[tuple(slice(0, n) for n in value.shape)] = value
            value = grown
        state[key] = value
    policy.load_state_dict(state)

def frozen_opponent(path, config):
    """Play saved weights using their original observation and action table.

    Unnamed robber victims map to the leading opponent; unsupported trades are declined."""
    from .game import action_id
    model, rules = load_policy(path)
    if rules.target_vp != config.target_vp:
        raise ValueError("A saved opponent must use the same victory target")
    # Prediction uses the policy without retaining the PPO rollout buffer.
    policy = model.policy
    translate = rules.table != config.table
    size = len(action_table(*rules.table))

    def decide(env, color):
        legal = env.legal(color)
        if len(legal) == 1:
            return next(iter(legal))
        obs = env.observe(color, rules.counted, rules.seats, rules.trading)
        if not translate:
            action, _ = policy.predict(obs, deterministic=True, action_masks=env._mask_of(legal))
            return int(action)
        colors = env.game.state.colors
        offered = {}
        for number, action in legal.items():
            if (not rules.relative_actions and action.action_type == ActionType.MOVE_ROBBER
                    and action.value[1] is not None):
                action = Action(color, action.action_type, (action.value[0], Color.RED))
                key = action_id(action, Color.BLUE, (Color.BLUE, Color.RED), False, rules.trading)
                offered.setdefault(key, number)
                continue
            try:
                offered.setdefault(action_id(action, color, colors, *rules.table), number)
            except KeyError:
                continue
        if not offered:
            # Nothing it can name: the legal actions are answers to an offer.
            return next(iter(legal))
        mask = np.zeros(size, dtype=bool)
        mask[list(offered)] = True
        action, _ = policy.predict(obs, deterministic=True, action_masks=mask)
        return offered[int(action)]

    return decide

class League:
    """Choose one opponent per seat, weighted by learner losses during training.

    Evaluation freezes uniform assignments by board seed and learner seat."""

    def __init__(self, names, config, floor=.25, adaptive=True):
        from .opponents import opponent_for
        if not names:
            raise ValueError("A league needs at least one opponent")
        self.names = list(map(str, names))
        self.opponents = [opponent_for(name, config) for name in self.names]
        self.floor = floor
        self.adaptive = adaptive
        self.wins = np.zeros(len(self.names))
        self.games = np.zeros(len(self.names))

    def weights(self):
        # Start every member near even, then follow the learner's losses.
        odds = self.floor + (self.games - self.wins + 1) / (self.games + 2)
        return odds / odds.sum()

    def record(self, game, picks, learner):
        """Count each participating member when the game has a winner."""
        won = winner(game)
        if won is None:
            return
        for index in picks.values():
            self.games[index] += 1
            self.wins[index] += won == learner

    def __call__(self, env, color):
        chosen = getattr(env, "league_choice", None)
        if chosen is None or chosen[0] is not env.game:
            if chosen is not None and self.adaptive:
                self.record(*chosen)
            # Evaluation assigns opponents from the board and seat, independent
            # of game scheduling and of the evaluated policy's random draws.
            rng = env.np_random if self.adaptive else np.random.default_rng([env.seed_value, env.seat])
            weights = self.weights() if self.adaptive else None
            picks = {c: int(rng.choice(len(self.opponents), p=weights))
                     for c in env.game.state.colors if c != env.learner}
            chosen = (env.game, picks, env.learner)
            env.league_choice = chosen
        env.opponent_name = "+".join(self.names[i] for i in chosen[1].values())
        return self.opponents[chosen[1][color]](env, color)

def potential(env, scale):
    """Progress potential: add it back to a shaped critic to recover true value."""
    state = env.game.state
    points = get_actual_victory_points(state, env.learner)
    # Reward resource spread; ignore the robber so moving it cannot inflate income.
    income = player_income(state, env.learner)
    return scale * (points / env.config.target_vp + .2 * np.sqrt(income).sum())

class TrainingEnv(gym.Wrapper):
    """Fold forced actions into one decision, then apply potential shaping.

    Wins zero the potential; time limits retain it for bootstrap values.
    """
    def __init__(self, env, gamma=1.0, scale=0.0, skip_forced=False):
        super().__init__(env)
        self.gamma, self.scale, self.skip_forced = gamma, scale, skip_forced

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.previous = potential(self.unwrapped, self.scale) if self.scale else 0.0
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        forced = 0
        while self.skip_forced and not (terminated or truncated):
            legal = np.flatnonzero(self.env.action_masks())
            if len(legal) != 1:
                break
            obs, extra, terminated, truncated, info = self.env.step(int(legal[0]))
            reward += extra
            forced += 1
        if self.skip_forced:
            info = {**info, "forced_learner_actions": forced}
        if self.scale:
            current = 0.0 if terminated else potential(self.unwrapped, self.scale)
            reward += self.gamma * current - self.previous
            self.previous = current
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()

class Outcomes(BaseCallback):
    """Count endings by opponent and player count, including capped-game details."""

    def __init__(self):
        super().__init__()
        self.counts = Counter(win=0, loss=0, truncated=0)
        self.by_opponent = defaultdict(lambda: Counter(win=0, loss=0, truncated=0))
        self.by_players = defaultdict(lambda: Counter(win=0, loss=0, truncated=0))
        self.truncation_limits = Counter(max_turns=0, max_actions=0)
        self.forced_actions = 0
        self.capped_games = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.forced_actions += info.get("forced_learner_actions", 0)
            if not done:
                continue
            self.counts[info["outcome"]] += 1
            for store, key in ((self.by_opponent, info.get("opponent", "unknown")),
                               (self.by_players, info.get("players", 2))):
                store[key][info["outcome"]] += 1
            if info["outcome"] == "truncated":
                self.capped_games.append({k: info[k] for k in
                                          ("opponent", "seed", "seat", "turns", "actions",
                                           "learner_vp", "opponent_vp", "truncation_limits")
                                          if k in info})
            self.truncation_limits.update(info["truncation_limits"])
        return True

    def _on_rollout_end(self):
        for name, count in {**self.counts, **self.truncation_limits}.items():
            self.logger.record("games/" + name, count)
        for prefix, store in (("opponents", self.by_opponent), ("players", self.by_players)):
            for name, counts in store.items():
                self.logger.record(f"{prefix}/{name}/win_rate", counts["win"] / counts.total())

def train(output="runs/ppo", steps=100_000, seed=0, opponent="random",
          config=None, resume=None, envs=4, rollout=512, batch=256,
          gamma=None, shaping=0.0, gae_lambda=None, entropy=None,
          skip_forced=False, league=None, learning_rate=None,
          transfer=None, league_floor=.25, checkpoint_every=0):
    """Train and save weights, rules and metrics. Resume keeps optimiser state;
    transfer grows the network with silent new inputs and a fresh optimiser."""
    from .opponents import LIBRARY, SCRIPTED, opponent_for
    config = config or Config()
    if steps < 0 or min(envs, rollout, batch) < 1 or envs * rollout < 2:
        raise ValueError("steps cannot be negative, and the sizes must be positive")
    if batch < 2 or (envs * rollout) % batch:
        raise ValueError("batch must be 2 or more, and must divide envs * rollout")
    if gamma is not None and not 0 < gamma <= 1:
        raise ValueError("gamma must be more than 0 and no more than 1")
    if shaping < 0 or (entropy is not None and entropy < 0) or (gae_lambda is not None
                                                                and not 0 <= gae_lambda <= 1):
        raise ValueError("shaping and entropy cannot be negative, and gae_lambda must be 0 to 1")
    if resume and transfer:
        raise ValueError("Use resume or transfer, not both")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("RLCATAN_THREADS", 1)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    opponent_name = str(opponent)
    if league:
        opponent = League(league, config, league_floor)
    elif opponent not in ("random", "greedy"):
        opponent = opponent_for(opponent, config)
    if resume:
        # A resumed run may collect and batch differently from the saved one.
        model, _ = load_policy(resume, n_steps=rollout, batch_size=batch, n_envs=envs)
    learning_rate = (learning_rate if learning_rate is not None
                     else float(model.lr_schedule(1)) if resume else 3e-4)
    gamma = gamma if gamma is not None else model.gamma if resume else .995

    def make_env():
        env = CatanEnv(config, opponent)
        return TrainingEnv(env, gamma, shaping, skip_forced) if skip_forced or shaping else env

    # One process runs every game. Profile before you pay for more processes.
    vec = DummyVecEnv([make_env for _ in range(envs)])
    vec.seed(seed)
    callback = Outcomes()
    started = time.perf_counter()
    logger = None
    try:
        if resume:
            model.gamma = model.rollout_buffer.gamma = gamma
            if gae_lambda is not None:
                model.gae_lambda = model.rollout_buffer.gae_lambda = gae_lambda
            if entropy is not None:
                model.ent_coef = entropy
            from stable_baselines3.common.utils import FloatSchedule
            model.learning_rate = learning_rate
            model.lr_schedule = FloatSchedule(learning_rate)
            model.set_env(vec)
            model.set_random_seed(seed)
        else:
            from .policies import AttentionPolicy
            # A counted run also scores each settlement by its port.
            policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                             "counted": config.counted or config.seats,
                             "ports": config.counted or config.seats,
                             "seats": config.seats, "trading": config.trading}
            model = MaskablePPO(AttentionPolicy, vec, learning_rate=learning_rate,
                                n_steps=rollout, batch_size=batch, n_epochs=4, gamma=gamma,
                                gae_lambda=.95 if gae_lambda is None else gae_lambda,
                                ent_coef=.01 if entropy is None else entropy, target_kl=0.03,
                                seed=seed, device="cpu", policy_kwargs=policy_kwargs)
            if transfer:
                source, _ = load_policy(transfer)
                transfer_weights(model.policy, source.policy.state_dict())
                if config.trading and not getattr(source.policy, "trading", False):
                    model.policy.prime_trading()
                # Loading restores the seed of the source. This run keeps its own.
                model.set_random_seed(seed)
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
        source_dir = output / "source"
        source_dir.mkdir()
        for name, data in SOURCE_FILES.items():
            (source_dir / name).write_bytes(data)
        (source_dir / "requirements.txt").write_text("\n".join(requires("rlcatan") or []) + "\n")
        saved = [p for p in (resume, transfer, *(league or []), opponent_name)
                 if p and str(p) not in SCRIPTED and str(p) not in LIBRARY]
        metadata = {"format": 5 if config.seats else 3, "game": asdict(config), "seed": seed,
                    "opponent": "+".join(map(str, league)) if league else opponent_name,
                    "requested_steps": steps,
                    "envs": envs, "rollout": rollout, "batch": batch,
                    "gamma": gamma, "gae_lambda": model.gae_lambda, "entropy": model.ent_coef,
                    "status": "training",
                    "ppo_epochs": model.n_epochs, "target_kl": model.target_kl,
                    "policy_class": type(model.policy).__name__,
                    "policy_kwargs": str(model.policy_kwargs),
                    "shaping": shaping, "shaping_version": "resource-income-sqrt-v1",
                    "league_weighting": "inverse-win-rate-v1", "league_floor": league_floor,
                    "skip_forced": skip_forced, "learning_rate": learning_rate,
                    "league": list(map(str, league or [])),
                    "dependencies": [{"path": str(model_path(p)), "sha256": file_sha256(model_path(p))}
                                     for p in dict.fromkeys(saved)],
                    "resume": str(resume) if resume else None,
                    "transfer": str(transfer) if transfer else None, **runtime()}

        def save_metadata():
            (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

        logger = configure(str(output), ["csv"])
        model.set_logger(logger)
        save_metadata()
        model.save(output / "initial.zip")
        before = model.num_timesteps
        if steps:
            callbacks = [callback]
            if checkpoint_every:
                # Each checkpoint loads with the run.json beside it.
                from stable_baselines3.common.callbacks import CheckpointCallback
                callbacks.append(CheckpointCallback(max(1, checkpoint_every // envs), str(output), "model"))
            model.learn(total_timesteps=steps, callback=callbacks,
                        reset_num_timesteps=not bool(resume))
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise RuntimeError("Training produced parameters that are not finite")
        logger.record("time/total_timesteps", model.num_timesteps)
        logger.dump(step=model.num_timesteps)
        model.save(output / "model.zip")
        metadata["status"] = "complete"
        metadata["model_sha256"] = file_sha256(output / "model.zip")
        save_metadata()
        elapsed = time.perf_counter() - started
        metrics = {"steps": model.num_timesteps - (before if resume else 0),
                   "seconds": elapsed, "games": callback.counts,
                   "forced_learner_actions": callback.forced_actions,
                   "opponent_outcomes": callback.by_opponent,
                   "player_outcomes": callback.by_players,
                   "truncation_limits": callback.truncation_limits,
                   "capped_games": callback.capped_games,
                   "parameters": sum(p.numel() for p in model.policy.parameters()),
                   "policy_mib": sum(p.numel() * p.element_size()
                                     for p in model.policy.parameters()) / 2**20}
        metrics["run_learner_steps_per_second"] = metrics["steps"] / elapsed
        (output / "train.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return model, metrics
    finally:
        if logger is not None:
            logger.close()
        vec.close()

def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/ppo")
    for name, default in (("steps", 100_000), ("seed", 0), ("target-vp", 6)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("max-turns", 300), ("max-actions", 2000),
                          ("envs", 4), ("rollout", 512), ("batch", 256)):
        parser.add_argument("--" + name, type=positive, default=default)
    for name in ("gamma", "gae-lambda", "entropy", "learning-rate"):
        parser.add_argument("--" + name, type=float)
    for name, help_text in (("counted", "Observe counted hands and roads' outcomes"),
                            ("seats", "Seat format: two to four players, counted, attention policy"),
                            ("trading", "Player to player trades"),
                            ("skip-forced", "Train on real choices only")):
        parser.add_argument("--" + name, action="store_true", help=help_text)
    parser.add_argument("--opponent", default="random", help="random, greedy or a saved run")
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--player-counts", type=int, nargs="+", default=(),
                        help="Pick a count each game; repeat a count to make it more likely")
    parser.add_argument("--offers", type=int, default=2, help="Offers a player may make each turn")
    parser.add_argument("--resume", help="A saved run; the rules may change for a curriculum")
    parser.add_argument("--transfer", help="A saved run whose weights start a new network")
    parser.add_argument("--shaping", type=float, default=0.0)
    parser.add_argument("--league", nargs="+", help="Scripted players and saved runs")
    parser.add_argument("--league-floor", type=float, default=.25,
                        help="Least weight of a beaten league member; large means uniform")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Also save the model every this many steps")
    args = vars(parser.parse_args())
    args["config"] = Config(**{k: args.pop(k) for k in
                               ("target_vp", "max_turns", "max_actions", "players",
                                "player_counts", "counted", "seats", "trading", "offers")})
    _, result = train(**args)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
