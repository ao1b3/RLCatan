"""Portable model, training and evaluation contracts: python test_rlcatan.py."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from catanatron.models.enums import ActionType as A
from rlcatan.game import CatanEnv, Config, action_table, winner
from rlcatan.opponents import maritime_terms
from rlcatan.policies import AttentionPolicy, resource_deltas
from rlcatan.training import League, config_from_metadata, load_policy, train, transfer_weights
from rlcatan.benchmark import evaluate, speed

def logits(model, obs, mask):
    model.policy.set_training_mode(False)
    with torch.no_grad():
        tensor = torch.as_tensor(np.asarray(obs)[None])
        return model.policy.get_distribution(tensor, action_masks=mask).distribution.logits

def test_models():
    models = []
    for flags, shape in (({}, (923, 332)), ({"counted": True}, (1073, 332)),
                         ({"seats": True, "players": 3}, (1680, 370)),
                         ({"seats": True, "players": 4, "trading": True}, (1698, 436))):
        config = Config(**flags)
        env = CatanEnv(config)
        kwargs = dict(counted=config.counted or config.seats, ports=config.counted or config.seats,
                      seats=config.seats, trading=config.trading)
        model = MaskablePPO(AttentionPolicy, env, n_steps=2, batch_size=2, policy_kwargs=kwargs, seed=3)
        assert (env.observation_space.shape[0], env.action_space.n) == shape
        assert len(set(action_table(*config.table))) == shape[1]
        deltas, _ = resource_deltas(*config.table)
        for i, (kind, value) in enumerate(action_table(*config.table)):
            if kind == A.PLAY_MONOPOLY:
                assert not deltas[i].any()  # Unknown gains cannot be encoded as a fixed one-card reward.
            if kind == A.PLAY_YEAR_OF_PLENTY:
                assert deltas[i].sum() == len(value)
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            for _ in range(6):
                mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                assert mask[int(action)]
                obs, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
        models.append((model, env))
    report = speed(steps=2, repeats=1, policy=model, config=config)
    assert (report["observation_floats"], report["actions"]) == (1698, 436)
    assert report["single_policy_inference_us_median"] > 0
    old, plain = models[0]
    new, counted = models[1]
    transfer_weights(new.policy, old.policy.state_dict())
    for seed in range(4):
        a, _ = plain.reset(seed=seed)
        b, _ = counted.reset(seed=seed)
        assert torch.allclose(logits(old, a, plain.action_masks()), logits(new, b, counted.action_masks()))
        with torch.no_grad():
            assert torch.allclose(old.policy.predict_values(torch.as_tensor(a[None])),
                                  new.policy.predict_values(torch.as_tensor(b[None])))
    for _, env in models:
        env.close()

def test_training(directory):
    config = Config(seats=True, trading=True, players=4)
    run = directory / "new"
    model, _ = train(run, steps=4, envs=2, rollout=2, batch=4, config=config,
                     shaping=.2, skip_forced=True, checkpoint_every=1)
    loaded, rules = load_policy(run)
    assert rules == config and (loaded.n_steps, loaded.n_envs, loaded.batch_size) == (2, 1, 2)
    assert list(run.glob("model_*_steps.zip"))
    env = CatanEnv(config)
    obs, _ = env.reset(seed=19)
    assert torch.equal(logits(model, obs, env.action_masks()), logits(loaded, obs, env.action_masks()))
    resumed, metrics = train(directory / "resumed", steps=2, envs=1, rollout=2, batch=2,
                             config=config, resume=run, shaping=.2, skip_forced=True)
    assert resumed.n_envs == 1 and metrics["steps"] == 2
    assert config_from_metadata({"format": 3, "game": {"multiplayer": False}}) == Config()
    env.close()

def test_evaluation_and_limits():
    config = Config(seats=True, players=4)
    before = evaluate("builder", "mixed", config, pairs=4, seed=77, envs=1)
    after = evaluate("builder", "mixed", config, pairs=4, seed=77, envs=16)
    assert before["rows"] == after["rows"]
    league = League(["greedy", "builder"], Config())
    league.games[:] = 40
    league.wins[:] = (40, 0)
    assert league.weights()[1] > 3 * league.weights()[0]
    for players in (2, 3, 4):
        limit = 2 * (players - 1) + 1
        env = CatanEnv(Config(seats=True, players=players, max_actions=limit))
        env.reset(seed=1, options={"seat": players - 1})
        _, _, terminated, truncated, _ = env.step(next(iter(env._legal)))
        assert truncated and not terminated and env.actions == limit
    env = CatanEnv(Config(target_vp=100))
    env.reset(seed=1, options={"seat": 0})
    key = f"P{env.game.state.current_turn_index}_ACTUAL_VICTORY_POINTS"
    for points in (10, 99, 100):
        env.game.state.player_state[key] = points
        assert (winner(env.game) is not None) == (points >= 100)
        assert env.observation_space.contains(env.observe())
    env.close()
    for paid in (2, 3, 4):
        value = ("WOOD",) * paid + (None,) * (4 - paid) + ("BRICK",)
        assert maritime_terms(value)[2] == paid

if __name__ == "__main__":
    torch.set_num_threads(1)
    test_models()
    with tempfile.TemporaryDirectory() as directory:
        test_training(Path(directory))
    test_evaluation_and_limits()
    print("models, training and evaluation ok")
