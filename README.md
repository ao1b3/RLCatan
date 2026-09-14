# RLCatan

Train and evaluate masked-PPO Settlers of Catan agents on top of
[Catanatron](https://github.com/bcollazo/catanatron), instead of trying to simulate internally...

This repository holds the environment, training, and evaluation code. The
browser interface for playing against a trained model is developed separately
and depends on this package.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Train

```sh
.venv/bin/python -m rlcatan.training --output runs/new --policy project --steps 100000
.venv/bin/python -m rlcatan.benchmark --model runs/new --pairs 100
```

The `rlcatan-train` and `rlcatan-benchmark` console scripts are equivalent.

`project` is the two-player policy. `actions` is the compact two-player action
policy. `mlp` is the baseline. Add `--multiplayer` to train on two to four
players; that format always reads the per-player blocks, so it uses `mlp`.

Use `--shaping .5` to reward progress toward victory and balanced resource
production during training. Evaluation and browser play use the original game
rewards. To continue project-ten:

```sh
.venv/bin/python -m rlcatan.training --output runs/project-coverage --resume runs/project-ten \
  --target-vp 10 --envs 8 --rollout 128 --batch 256 --steps 131072 --seed 8300000 \
  --league builder planner-available expansion development runs/project-ten \
  --shaping .5 --skip-forced --gamma .999 --gae-lambda .99 --learning-rate .0001 --entropy .01
```

Use a fresh output directory for each run. Compare candidates on the same
evaluation seeds. `stalled_games` counts games ending with two settlements and
no cities; `two_vp_games` counts games ending at two or fewer actual victory
points. Development-card strategies can win without expanding, so inspect these
counts alongside wins.

`--imitation 4000 --imitation-epochs 5 --imitation-setup --teacher planner-available`
trains on opening placements before PPO. Omit `--imitation-setup` to imitate
full games. Opening practice updates shared network weights, so evaluate full
games afterward.

## Checkpoints

Each run directory keeps `model.zip` beside the `run.json` that records its
rules, seeds, and source hashes; the two must stay together. Runs trained
before the modules moved into the `rlcatan` package pickled their policy
classes under the old top-level names. `LEGACY_MODULES` in
`rlcatan.training` maps each old name to the file that now holds the class, so
those checkpoints still load.

Generated runs and models are ignored by Git.

## History

The original notebook-and-`RLmodels` project that preceded this rewrite is kept
on the `legacy-rlmodels` branch and the `legacy-rlmodels-final` tag.
