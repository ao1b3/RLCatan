"""Train a small masked actor-critic; SB3 owns PPO, GAE and timeout bootstrap."""
import argparse
import csv
import shutil
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import time
import weakref

import gymnasium as gym

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from .compat import LEGACY_MODULES, install_legacy_module_aliases
from .game import CatanEnv, Config


_SOURCE_FILES = {name: Path(__file__).with_name(name).read_bytes()
                  for name in ("game.py", "agents.py", "training.py", "benchmark.py",
                               "features.py", "multiplayer.py", "action_policy.py", "teacher.py",
                               "project_policy.py", "strong_teacher.py", "compat.py")}
_SOURCE_HASHES = {name: hashlib.sha256(data).hexdigest() for name, data in _SOURCE_FILES.items()}


def runtime():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": _SOURCE_HASHES,
            "packages": {p: version(p) for p in
                         ("catanatron", "numpy", "torch", "gymnasium",
                          "stable-baselines3", "sb3-contrib")}}


def load_policy(path):
    """Load our checkpoint and its matching rules, never infer rules from shape."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.zip"
    metadata = json.loads((path.parent / "run.json").read_text())
    if metadata["format"] not in (3, 4):
        raise ValueError("Unsupported checkpoint observation/action format")
    config = Config(**metadata["game"])
    if config.multiplayer != (metadata['format'] == 4):
        raise ValueError("Checkpoint format does not match game configuration")
    try:
        model = MaskablePPO.load(path, device="cpu")
    except ModuleNotFoundError as error:
        if error.name not in LEGACY_MODULES:
            raise
        install_legacy_module_aliases()
        model = MaskablePPO.load(path, device="cpu")
    return model, config


def average_policy_parameters(model, paths, config):
    """Average matching learned parameters; preserve identical static buffers."""
    if len(paths) < 2:
        raise ValueError("Checkpoint averaging needs at least two sources")
    sources = [load_policy(path) for path in paths]
    parameters = dict(model.policy.named_parameters())
    buffers = dict(model.policy.named_buffers())
    source_parameters = []
    for source, rules in sources:
        if type(source.policy) is not type(model.policy) or rules.target_vp != config.target_vp:
            raise ValueError("Averaging requires matching policy architecture and victory target")
        values = dict(source.policy.named_parameters())
        static = dict(source.policy.named_buffers())
        if (values.keys() != parameters.keys() or static.keys() != buffers.keys()
                or any(values[k].shape != p.shape for k, p in parameters.items())
                or any(not torch.equal(static[k], b) for k, b in buffers.items())):
            raise ValueError("Averaging requires matching parameter shapes and static buffers")
        source_parameters.append(values)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(torch.stack([values[name] for values in source_parameters]).mean(0))


def frozen_opponent(path, config):
    model, rules = load_policy(path)
    if rules.target_vp != config.target_vp or rules.multiplayer != config.multiplayer:
        raise ValueError("Frozen opponent must use the same victory target")

    def decide(env, color):
        mask = env.mask_for(color)
        if mask.sum() == 1:
            return int(mask.argmax())
        action, _ = model.predict(env.observe(color), deterministic=True, action_masks=mask)
        return int(action)

    return decide


class OpponentPool:
    """Immutable policies; sample once per game, including across vector resets."""
    def __init__(self, paths, config):
        if not paths:
            raise ValueError("A pool needs at least one checkpoint")
        self.models = []
        for path in paths:
            model, rules = load_policy(path)
            if rules.target_vp != config.target_vp or rules.multiplayer != config.multiplayer:
                raise ValueError("Pool opponents must use the same victory target")
            self.models.append(model)
        self.episodes = weakref.WeakKeyDictionary()

    def __call__(self, env, color):
        from .agents import choose_action
        entry = self.episodes.get(env)
        if entry is None or entry[0] is not env.game:
            indices = {}
            for owner in env.game.state.colors:
                if owner == env.learner:
                    continue
                draw = env.np_random.random()
                indices[owner] = (-1 if draw < .2 else len(self.models) - 1 if draw < .6
                                  else int(env.np_random.integers(max(1, len(self.models) - 1))))
            entry = (env.game, indices)
            self.episodes[env] = entry
        if entry[1][color] == -1:
            return choose_action(env, "greedy", color)
        # Sample with the environment RNG so evaluation/training interleaving is reproducible.
        model = self.models[entry[1][color]]
        with torch.no_grad():
            obs, _ = model.policy.obs_to_tensor(env.observe(color))
            distribution = model.policy.get_distribution(obs, action_masks=env.mask_for(color))
            probabilities = distribution.distribution.probs.cpu().numpy()[0].astype(float)
        probabilities /= probabilities.sum()
        return int(env.np_random.choice(len(probabilities), p=probabilities))


class LeagueOpponents:
    """Uniform, immutable opponent roster; fixed identity for each episode."""
    def __init__(self, names, config):
        from .teacher import opponent_for
        if not names:
            raise ValueError("League needs opponents")
        self.names = list(map(str, names))
        self.opponents = [opponent_for(name, config) for name in self.names]
        self.episodes = weakref.WeakKeyDictionary()

    def __call__(self, env, color):
        entry = self.episodes.get(env)
        if entry is None or entry[0] is not env.game:
            entry = (env.game, {c: int(env.np_random.integers(len(self.opponents)))
                               for c in env.game.state.colors if c != env.learner})
            self.episodes[env] = entry
        env.opponent_name = '+'.join(self.names[i] for i in entry[1].values())
        return self.opponents[entry[1][color]](env, color)


class DecisionSteps(gym.Wrapper):
    """Count genuine choices as training steps; execute mandatory actions directly."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        forced = 0
        while not (terminated or truncated):
            legal = np.flatnonzero(self.env.action_masks())
            if len(legal) != 1:
                break
            obs, extra, terminated, truncated, info = self.env.step(int(legal[0]))
            reward += extra
            forced += 1
        return obs, reward, terminated, truncated, {**info, "forced_learner_actions": forced}

    def action_masks(self):
        return self.env.action_masks()


class PotentialReward(gym.Wrapper):
    """Training-only shaping; terminal potential zero, timeout potential retained."""
    def __init__(self, env, gamma, scale):
        super().__init__(env)
        self.gamma, self.scale = gamma, scale

    def potential(self):
        from catanatron.state_functions import player_key
        from catanatron.models.enums import RESOURCES
        from catanatron.models.map import number_probability
        env = self.unwrapped
        state = env.game.state
        points = state.player_state[player_key(state, env.learner) + "_ACTUAL_VICTORY_POINTS"]
        income = np.zeros(len(RESOURCES))
        for node, (color, kind) in state.board.buildings.items():
            if color == env.learner:
                for tile in state.board.map.adjacent_tiles[node]:
                    if tile.resource:
                        income[RESOURCES.index(tile.resource)] += number_probability(tile.number) * (2 if kind == "CITY" else 1)
        # Diminishing returns favor resource coverage; use nominal income so robber moves cannot farm this term.
        return self.scale * (points / env.config.target_vp + .2 * np.sqrt(income).sum())

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.previous = self.potential()
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        potential = 0.0 if terminated else self.potential()
        reward += self.gamma * potential - self.previous
        self.previous = potential
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()


def imitate(model, config, samples, seed, epochs=10, teacher="greedy", learner_mix=0.0, output=None, demonstration_cache=None, setup_only=False):
    """Supervised warm start on greedy self-play; omit forced decisions."""
    from .agents import choose_action
    from .teacher import opponent_for
    if demonstration_cache:
        raise ValueError('Demonstration caches were removed; collect a fresh warm start')
    else:
        decide = opponent_for(teacher, config)
        env = CatanEnv(config, decide)
        obs, _ = env.reset(seed=seed)
        observations, masks, actions = [], [], []
        while len(actions) < samples:
            mask = env.action_masks()
            action = decide(env, env.learner)
            if mask.sum() > 1:
                observations.append(obs)
                masks.append(mask)
                actions.append(action)
            behavior = action
            if learner_mix and env.np_random.random() < learner_mix and mask.sum() > 1:
                behavior, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, _, term, trunc, _ = env.step(int(behavior))
            if term or trunc or (setup_only and not env.game.state.is_initial_build_phase):
                obs, _ = env.reset()
        env.close()
    observations = torch.as_tensor(np.asarray(observations), device=model.device)
    masks = np.asarray(masks)
    actions = torch.as_tensor(actions, device=model.device)
    rng = np.random.default_rng(seed)
    model.policy.set_training_mode(True)
    history = []
    for epoch in range(epochs):
        total_loss, correct = 0.0, 0
        for start in range(0, samples, 256):
            # Each epoch visits every demonstration once.
            if start == 0:
                order = rng.permutation(samples)
            indices = order[start:start + 256]
            distribution = model.policy.get_distribution(observations[indices], action_masks=masks[indices])
            loss = -distribution.log_prob(actions[indices]).mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            model.policy.optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            correct += int((distribution.distribution.probs.argmax(-1) == actions[indices]).sum())
        history.append({"epoch": epoch + 1, "loss": total_loss / samples, "accuracy": correct / samples})
        if output:
            with (Path(output) / "imitation.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0])
                writer.writeheader()
                writer.writerows(history)
    model.policy.set_training_mode(False)
    return {"samples": samples, "epochs": epochs, "last_loss": history[-1]["loss"], "learner_mix": learner_mix, "setup_only": setup_only}


class Outcomes(BaseCallback):
    def __init__(self):
        super().__init__()
        self.counts = {"win": 0, "loss": 0, "truncated": 0}
        self.truncation_limits = {"max_turns": 0, "max_actions": 0}
        self.forced_actions = 0
        self.by_opponent = {}
        self.by_players = {}
        self.capped_games = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.forced_actions += info.get("forced_learner_actions", 0)
            if done:
                self.counts[info["outcome"]] += 1
                name = info.get("opponent", "unknown")
                counts = self.by_opponent.setdefault(name, {"win": 0, "loss": 0, "truncated": 0})
                counts[info["outcome"]] += 1
                counts = self.by_players.setdefault(info.get('players', 2), {"win": 0, "loss": 0, "truncated": 0})
                counts[info['outcome']] += 1
                if info["outcome"] == "truncated":
                    self.capped_games.append({k: info[k] for k in
                        ("opponent", "seed", "seat", "turns", "actions",
                         "learner_vp", "opponent_vp", "truncation_limits") if k in info})
                for limit in info["truncation_limits"]:
                    self.truncation_limits[limit] += 1
        return True

    def _on_rollout_end(self):
        for outcome, count in self.counts.items():
            self.logger.record("games/" + outcome, count)
        for name, counts in self.by_opponent.items():
            self.logger.record("opponents/" + name + "/win_rate", counts["win"] / sum(counts.values()))
        for limit, count in self.truncation_limits.items():
            self.logger.record("games/" + limit, count)
        for players, counts in self.by_players.items():
            self.logger.record(f"players/{players}/win_rate", counts['win'] / sum(counts.values()))


def train(output="runs/ppo", steps=100_000, seed=0, opponent="random",
          config=None, resume=None, envs=4, rollout=512, batch=256,
          gamma=None, shaping=0.0, imitation=0, imitation_epochs=10, pool=None,
          teacher="greedy", policy="mlp", width=128, gae_lambda=None, entropy=None,
          skip_forced=False, league=None, initialize=None, learning_rate=None, imitation_mix=0.0, average=None,
          demonstration_cache=None, imitation_setup=False):
    """Fixed opponents per run; resume supports an explicit 4→6→10 VP curriculum."""
    config = config or Config()
    if steps < 0 or min(envs, rollout, batch) < 1 or envs * rollout < 2:
        raise ValueError("Nonnegative steps, positive sizes and at least two rollout samples required")
    if batch < 2 or (envs * rollout) % batch:
        raise ValueError("Batch must be >=2 and divide envs * rollout")
    if gamma is not None and not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if shaping < 0 or imitation < 0 or imitation_epochs < 1:
        raise ValueError("Invalid shaping or imitation settings")
    if config.multiplayer:
        if policy not in ('mlp', 'players'):
            raise ValueError("Multiplayer training requires a players policy")
        if policy == 'mlp':
            policy = 'players'
    elif policy == 'players':
        raise ValueError("players policies require multiplayer format")
    if policy not in ("mlp", "actions", "project", "players"):
        raise ValueError("Unknown policy")
    if demonstration_cache and (not imitation or imitation_mix):
        raise ValueError('Demonstration cache requires imitation and zero learner mix')
    if imitation_setup and not imitation:
        raise ValueError('--imitation-setup requires --imitation samples')
    if width < 1 or (gae_lambda is not None and not 0 <= gae_lambda <= 1) or (entropy is not None and entropy < 0):
        raise ValueError("Invalid policy or PPO settings")
    if not 0 <= imitation_mix <= 1 or (learning_rate is not None and learning_rate <= 0):
        raise ValueError("Invalid imitation mix or learning rate")
    if sum(bool(x) for x in (pool, league)) > 1 or sum(bool(x) for x in (initialize, resume, average)) > 1:
        raise ValueError("Use pool or league, and only one of initialize, resume, or average")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    opponent_name = str(opponent)
    if league:
        opponent = LeagueOpponents(league, config)
    elif pool:
        opponent = OpponentPool(pool, config)
    elif opponent not in ("random", "greedy"):
        from .teacher import opponent_for
        opponent = opponent_for(opponent, config)
    # ponytail: synchronous vectorization; add processes only if profiling pays for IPC.
    if resume:
        model, _ = load_policy(resume)
    learning_rate = learning_rate if learning_rate is not None else float(model.lr_schedule(1)) if resume else 3e-4
    effective_gamma = gamma if gamma is not None else model.gamma if resume else .995
    def make_env():
        env = CatanEnv(config, opponent)
        if skip_forced:
            env = DecisionSteps(env)
        return PotentialReward(env, effective_gamma, shaping) if shaping else env
    vec = DummyVecEnv([make_env for _ in range(envs)])
    vec.seed(seed)
    callback = Outcomes()
    started = time.perf_counter()
    logger = None
    try:
        if resume:
            if model.n_steps != rollout or model.batch_size != batch or model.n_envs != envs:
                raise ValueError("Resume requires the saved envs, rollout and batch sizes")
            model.gamma = effective_gamma
            model.rollout_buffer.gamma = effective_gamma
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
            from .action_policy import ActionFeaturePolicy
            from .project_policy import ProjectPolicy
            policy_kwargs = {"net_arch": {"pi": [width, width], "vf": [width, width]}}
            if policy == 'players':
                from .multiplayer import PlayerFeatures
                policy_kwargs['features_extractor_class'] = PlayerFeatures
            model = MaskablePPO(
                {"actions": ActionFeaturePolicy, "project": ProjectPolicy}.get(policy, "MlpPolicy"), vec, learning_rate=learning_rate, n_steps=rollout,
                batch_size=batch, n_epochs=4, gamma=effective_gamma, gae_lambda=.95 if gae_lambda is None else gae_lambda,
                ent_coef=.01 if entropy is None else entropy, target_kl=0.03, seed=seed, device="cpu",
                policy_kwargs=policy_kwargs,
            )
        # Imitation uses the optimizer directly, before PPO updates its schedule.
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
        if initialize:
            source, _ = load_policy(initialize)
            missing, unexpected = model.policy.load_state_dict(source.policy.state_dict(), strict=False)
            allowed = ("action_net.",)
            if unexpected or any(not k.startswith(allowed) for k in missing):
                raise ValueError(f"Incompatible initialization: {missing}, {unexpected}")
            # SB3 loading restores the source seed; transfer must use this run's seed.
            model.set_random_seed(seed)
        if average:
            average_policy_parameters(model, average, config)
            model.set_random_seed(seed)
        source_dir = output / "source"
        source_dir.mkdir()
        for name, data in _SOURCE_FILES.items():
            (source_dir / name).write_bytes(data)
        # requirements.txt sits beside the package, and is absent once installed.
        requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
        if requirements.exists():
            shutil.copy2(requirements, source_dir / requirements.name)
        def checkpoint_identity(path):
            path = Path(path)
            path = path / "model.zip" if path.is_dir() else path
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        from .teacher import OPPONENTS
        dependencies = [p for p in [resume, initialize, *(pool or []), *(league or []), *(average or []), opponent_name,
                                   teacher if imitation else None]
                        if p and str(p) not in OPPONENTS]
        imitation_metrics = None
        logger = configure(str(output), ["csv"])
        model.set_logger(logger)
        metadata = {"format": 4 if config.multiplayer else 3, "game": asdict(config), "seed": seed,
                    "opponent": opponent_name, "requested_steps": steps,
                    "envs": envs, "rollout": rollout, "batch": batch,
                    "gamma": effective_gamma, "gae_lambda": model.gae_lambda, "entropy": model.ent_coef,
                    "status": "imitation" if imitation else "training",
                    "imitation_requested": {"samples": imitation, "epochs": imitation_epochs, "learner_mix": imitation_mix, "setup_only": imitation_setup},
                    "ppo_epochs": model.n_epochs, "target_kl": model.target_kl,
                    "policy_class": type(model.policy).__name__, "policy_kwargs": str(model.policy_kwargs),
                    "shaping": shaping, "shaping_version": "resource-income-sqrt-v1", "imitation": imitation_metrics, "teacher": teacher,
                    "skip_forced": skip_forced, "learning_rate": learning_rate,
                    "league": list(map(str, league or [])), "average": list(map(str, average or [])), "initialize": str(initialize) if initialize else None,
                    "dependencies": [checkpoint_identity(p) for p in dict.fromkeys(dependencies)],
                    "pool": [str(p) for p in pool] if pool else [],
                    "resume": str(resume) if resume else None, **runtime()}
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if imitation:
            metadata["imitation"] = imitate(model, config, imitation, seed, imitation_epochs, teacher,
                                            imitation_mix, output, demonstration_cache, imitation_setup)
            if demonstration_cache:
                metadata['demonstration_cache'] = checkpoint_identity(demonstration_cache)
        metadata["status"] = "training"
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        model.save(output / "initial.zip")
        before = model.num_timesteps
        if steps:
            model.learn(total_timesteps=steps, callback=callback,
                        reset_num_timesteps=not bool(resume))
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise RuntimeError("Training produced nonfinite parameters")
        logger.record("time/total_timesteps", model.num_timesteps)
        logger.dump(step=model.num_timesteps)
        model.save(output / "model.zip")
        metadata["status"] = "complete"
        metadata["model_sha256"] = hashlib.sha256((output / "model.zip").read_bytes()).hexdigest()
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/ppo")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", help="random, greedy or saved run")
    parser.add_argument("--target-vp", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument('--players', type=int, choices=(2, 3, 4), default=2)
    parser.add_argument('--multiplayer', action='store_true', help='Use format 4, including for two players')
    parser.add_argument('--player-counts', type=int, nargs='+', default=(), help='Sample counts each episode; repeat counts to weight them')
    parser.add_argument("--resume", help="Saved run directory; rules may change for curriculum")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--shaping", type=float, default=0.0)
    parser.add_argument("--imitation", type=int, default=0, help="Greedy demonstrations before PPO")
    parser.add_argument("--imitation-epochs", type=int, default=10)
    parser.add_argument("--imitation-setup", action="store_true", help="Imitate opening placements only")
    parser.add_argument("--pool", nargs="+", help="Frozen run directories, oldest to newest")
    parser.add_argument("--teacher", default="greedy", help="Scripted opponent name or frozen checkpoint")
    parser.add_argument("--policy", choices=("mlp", "actions", "project", "players"), default="mlp")
    parser.add_argument('--demonstration-cache', help='Reuse an exact teacher-only .npz dataset across students')
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy", type=float)
    parser.add_argument("--skip-forced", action="store_true", help="Train only on genuine action choices")
    parser.add_argument("--league", nargs="+", help="Uniform roster of styles and frozen run paths")
    parser.add_argument("--average", nargs="+", help="Average matching checkpoints into a fresh policy and optimizer")
    parser.add_argument("--initialize", help="Transfer policy weights into a new architecture/run")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--imitation-mix", type=float, default=0.0, help="Fraction of learner actions during demonstration collection")
    args = vars(parser.parse_args())
    args["config"] = Config(**{k: args.pop(k) for k in
                              ("target_vp", "max_turns", "max_actions", "players", "multiplayer", "player_counts")})
    _, result = train(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
