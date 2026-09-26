"""Train a policy with masked PPO.

The stable-baselines3 library does the PPO maths. This file prepares the
environment, the opponents and the warm start, and it records what each run
used.
"""
import argparse
import csv
import hashlib
import importlib
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from catanatron.state_functions import get_actual_victory_points

from .game import CatanEnv, Config, player_income

# Models trained before these modules moved into the rlcatan package name their
# classes by the old module. Point the old names at the file that now holds the
# class, so that those saved models still load.
LEGACY_MODULES = {"game": "game", "agents": "opponents", "teacher": "opponents",
                  "strong_teacher": "opponents", "features": "policies",
                  "action_policy": "policies", "project_policy": "policies",
                  "multiplayer": "policies", "training": "training", "benchmark": "benchmark"}
SOURCE_FILES = {path.name: path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py"))}


def install_legacy_module_aliases():
    for old, new in LEGACY_MODULES.items():
        sys.modules.setdefault(old, importlib.import_module("." + new, __package__))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_path(path):
    """Return the path of the saved model. A run directory holds model.zip."""
    path = Path(path)
    return path / "model.zip" if path.is_dir() else path


def runtime():
    """Return what produced a run: the machine, the sources and the packages."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": {name: hashlib.sha256(data).hexdigest()
                              for name, data in SOURCE_FILES.items()},
            "packages": {p: version(p) for p in ("catanatron", "numpy", "torch", "gymnasium",
                                                 "stable-baselines3", "sb3-contrib")}}


def load_policy(path):
    """Load a saved model and the rules that it was trained on.

    Read the rules from the run.json beside the model. Never guess the rules
    from the size of the arrays.
    """
    path = model_path(path)
    metadata = json.loads((path.parent / "run.json").read_text())
    if metadata["format"] not in (3, 4):
        raise ValueError("Unsupported observation format")
    config = Config(**metadata["game"])
    if config.multiplayer != (metadata["format"] == 4):
        raise ValueError("The saved format does not match the saved rules")
    try:
        model = MaskablePPO.load(path, device="cpu")
    except ModuleNotFoundError as error:
        if error.name not in LEGACY_MODULES:
            raise
        install_legacy_module_aliases()
        model = MaskablePPO.load(path, device="cpu")
    return model, config


def frozen_opponent(path, config):
    """Return a player that takes the best action of a saved model."""
    model, rules = load_policy(path)
    if rules.target_vp != config.target_vp or rules.multiplayer != config.multiplayer:
        raise ValueError("A saved opponent must use the same rules")
    # Keep the network only. The optimiser and the rollout buffer of the saved
    # model are large and are never used here.
    policy = model.policy

    def decide(env, color):
        mask = env.mask_for(color)
        if mask.sum() == 1:
            return int(mask.argmax())
        action, _ = policy.predict(env.observe(color), deterministic=True, action_masks=mask)
        return int(action)

    return decide


class League:
    """A fixed roster of opponents.

    Each player in a game keeps one opponent for the whole game. The roster
    itself never changes.
    """

    def __init__(self, names, config):
        from .opponents import opponent_for
        if not names:
            raise ValueError("A league needs at least one opponent")
        self.names = list(map(str, names))
        self.opponents = [opponent_for(name, config) for name in self.names]

    def __call__(self, env, color):
        chosen = getattr(env, "league_choice", None)
        if chosen is None or chosen[0] is not env.game:
            picks = {c: int(env.np_random.integers(len(self.opponents)))
                     for c in env.game.state.colors if c != env.learner}
            chosen = (env.game, picks)
            env.league_choice = chosen
        env.opponent_name = "+".join(self.names[i] for i in chosen[1].values())
        return self.opponents[chosen[1][color]](env, color)


class DecisionSteps(gym.Wrapper):
    """Count a step only when the learner has a real choice.

    Play a forced action at once, and add its reward to the step that follows.
    """

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
    """Add a small reward for progress. Use this during training only.

    The value is zero when a player wins, so the shaping cannot change who the
    winner is. The value stays when the game hits a time limit.
    """

    def __init__(self, env, gamma, scale):
        super().__init__(env)
        self.gamma, self.scale = gamma, scale

    def potential(self):
        env = self.unwrapped
        state = env.game.state
        points = get_actual_victory_points(state, env.learner)
        # The square root makes a wide spread of resources worth more than a
        # lot of one resource. The income ignores the robber, so a player
        # cannot raise this reward by moving the robber.
        income = player_income(state, env.learner)
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


def imitate(model, config, samples, seed, epochs, teacher, output, setup_only):
    """Teach the policy to copy a scripted player before PPO starts.

    Collect demonstrations, then train on them. Skip a forced action, because
    the policy learns nothing from a choice of one.
    """
    from .opponents import opponent_for
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
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated or (setup_only and not env.game.state.is_initial_build_phase):
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
        order = rng.permutation(samples)
        for start in range(0, samples, 256):
            indices = order[start:start + 256]
            distribution = model.policy.get_distribution(observations[indices],
                                                         action_masks=masks[indices])
            loss = -distribution.log_prob(actions[indices]).mean()
            model.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            model.policy.optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            correct += int((distribution.distribution.probs.argmax(-1) == actions[indices]).sum())
        history.append({"epoch": epoch + 1, "loss": total_loss / samples,
                        "accuracy": correct / samples})
        if output:
            with (Path(output) / "imitation.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0])
                writer.writeheader()
                writer.writerows(history)
    model.policy.set_training_mode(False)
    return {"samples": samples, "epochs": epochs, "last_loss": history[-1]["loss"],
            "setup_only": setup_only}


class Outcomes(BaseCallback):
    """Count how each game ended, by opponent and by player count."""

    def __init__(self):
        super().__init__()
        self.by_opponent, self.by_players = {}, {}
        self.truncation_limits = {"max_turns": 0, "max_actions": 0}
        self.forced_actions = 0
        self.capped_games = []

    @property
    def counts(self):
        totals = {"win": 0, "loss": 0, "truncated": 0}
        for counts in self.by_opponent.values():
            for outcome, count in counts.items():
                totals[outcome] += count
        return totals

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            self.forced_actions += info.get("forced_learner_actions", 0)
            if not done:
                continue
            for store, key in ((self.by_opponent, info.get("opponent", "unknown")),
                               (self.by_players, info.get("players", 2))):
                counts = store.setdefault(key, {"win": 0, "loss": 0, "truncated": 0})
                counts[info["outcome"]] += 1
            if info["outcome"] == "truncated":
                self.capped_games.append({k: info[k] for k in
                                          ("opponent", "seed", "seat", "turns", "actions",
                                           "learner_vp", "opponent_vp", "truncation_limits")
                                          if k in info})
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
            self.logger.record(f"players/{players}/win_rate", counts["win"] / sum(counts.values()))


def train(output="runs/ppo", steps=100_000, seed=0, opponent="random",
          config=None, resume=None, envs=4, rollout=512, batch=256,
          gamma=None, shaping=0.0, imitation=0, imitation_epochs=10,
          teacher="greedy", policy="mlp", gae_lambda=None, entropy=None,
          skip_forced=False, league=None, learning_rate=None, imitation_setup=False):
    """Train one model and write it, its rules and its metrics to output.

    The opponents do not change during a run. Use resume to raise the victory
    target between runs.
    """
    from .opponents import SCRIPTED, opponent_for
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
    if policy not in ("mlp", "actions", "project"):
        raise ValueError("policy must be mlp, actions or project")
    if config.multiplayer and policy != "mlp":
        raise ValueError("The multiplayer format uses the mlp policy")
    if imitation_setup and not imitation:
        raise ValueError("imitation_setup needs imitation samples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    opponent_name = str(opponent)
    if league:
        opponent = League(league, config)
    elif opponent not in ("random", "greedy"):
        opponent = opponent_for(opponent, config)
    if resume:
        model, _ = load_policy(resume)
    learning_rate = (learning_rate if learning_rate is not None
                     else float(model.lr_schedule(1)) if resume else 3e-4)
    gamma = gamma if gamma is not None else model.gamma if resume else .995

    def make_env():
        env = CatanEnv(config, opponent)
        if skip_forced:
            env = DecisionSteps(env)
        return PotentialReward(env, gamma, shaping) if shaping else env

    # One process runs every game. Profile before you pay for more processes.
    vec = DummyVecEnv([make_env for _ in range(envs)])
    vec.seed(seed)
    callback = Outcomes()
    started = time.perf_counter()
    logger = None
    try:
        if resume:
            if model.n_steps != rollout or model.batch_size != batch or model.n_envs != envs:
                raise ValueError("resume needs the saved envs, rollout and batch sizes")
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
            from .policies import ActionFeaturePolicy, PlayerFeatures, ProjectPolicy
            classes = {"actions": ActionFeaturePolicy, "project": ProjectPolicy}
            policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
            if config.multiplayer:
                # Format 4 always reads the player blocks.
                policy_kwargs["features_extractor_class"] = PlayerFeatures
            model = MaskablePPO(classes.get(policy, "MlpPolicy"), vec, learning_rate=learning_rate,
                                n_steps=rollout, batch_size=batch, n_epochs=4, gamma=gamma,
                                gae_lambda=.95 if gae_lambda is None else gae_lambda,
                                ent_coef=.01 if entropy is None else entropy, target_kl=0.03,
                                seed=seed, device="cpu", policy_kwargs=policy_kwargs)
        # Imitation uses the optimiser directly, before PPO sets its own rate.
        for group in model.policy.optimizer.param_groups:
            group["lr"] = learning_rate
        source_dir = output / "source"
        source_dir.mkdir()
        for name, data in SOURCE_FILES.items():
            (source_dir / name).write_bytes(data)
        # requirements.txt sits beside the package. It is absent once installed.
        requirements = Path(__file__).resolve().parent.parent / "requirements.txt"
        if requirements.exists():
            shutil.copy2(requirements, source_dir / requirements.name)
        saved = [p for p in (resume, *(league or []), opponent_name,
                             teacher if imitation else None)
                 if p and str(p) not in SCRIPTED]
        metadata = {"format": 4 if config.multiplayer else 3, "game": asdict(config), "seed": seed,
                    "opponent": "+".join(map(str, league)) if league else opponent_name,
                    "requested_steps": steps,
                    "envs": envs, "rollout": rollout, "batch": batch,
                    "gamma": gamma, "gae_lambda": model.gae_lambda, "entropy": model.ent_coef,
                    "status": "imitation" if imitation else "training",
                    "imitation_requested": {"samples": imitation, "epochs": imitation_epochs,
                                            "setup_only": imitation_setup},
                    "ppo_epochs": model.n_epochs, "target_kl": model.target_kl,
                    "policy_class": type(model.policy).__name__,
                    "policy_kwargs": str(model.policy_kwargs),
                    "shaping": shaping, "shaping_version": "resource-income-sqrt-v1",
                    "imitation": None, "teacher": teacher,
                    "skip_forced": skip_forced, "learning_rate": learning_rate,
                    "league": list(map(str, league or [])),
                    "dependencies": [{"path": str(model_path(p)), "sha256": file_sha256(model_path(p))}
                                     for p in dict.fromkeys(saved)],
                    "resume": str(resume) if resume else None, **runtime()}

        def save_metadata():
            (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

        logger = configure(str(output), ["csv"])
        model.set_logger(logger)
        save_metadata()
        if imitation:
            metadata["imitation"] = imitate(model, config, imitation, seed, imitation_epochs,
                                            teacher, output, imitation_setup)
        metadata["status"] = "training"
        save_metadata()
        model.save(output / "initial.zip")
        before = model.num_timesteps
        if steps:
            model.learn(total_timesteps=steps, callback=callback,
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
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", help="random, greedy or a saved run")
    parser.add_argument("--target-vp", type=int, default=6)
    parser.add_argument("--max-turns", type=positive, default=300)
    parser.add_argument("--max-actions", type=positive, default=2000)
    parser.add_argument("--players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--multiplayer", action="store_true",
                        help="Use format 4, even for two players")
    parser.add_argument("--player-counts", type=int, nargs="+", default=(),
                        help="Pick a count each game; repeat a count to make it more likely")
    parser.add_argument("--resume", help="A saved run; the rules may change for a curriculum")
    parser.add_argument("--envs", type=positive, default=4)
    parser.add_argument("--rollout", type=positive, default=512)
    parser.add_argument("--batch", type=positive, default=256)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--shaping", type=float, default=0.0)
    parser.add_argument("--imitation", type=int, default=0, help="Demonstrations before PPO")
    parser.add_argument("--imitation-epochs", type=positive, default=10)
    parser.add_argument("--imitation-setup", action="store_true",
                        help="Copy the opening placements only")
    parser.add_argument("--teacher", default="greedy", help="A scripted player or a saved run")
    parser.add_argument("--policy", choices=("mlp", "actions", "project"), default="mlp")
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--entropy", type=float)
    parser.add_argument("--skip-forced", action="store_true",
                        help="Train on real choices only")
    parser.add_argument("--league", nargs="+", help="Scripted players and saved runs")
    parser.add_argument("--learning-rate", type=float)
    args = vars(parser.parse_args())
    args["config"] = Config(**{k: args.pop(k) for k in
                               ("target_vp", "max_turns", "max_actions", "players",
                                "multiplayer", "player_counts")})
    _, result = train(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
