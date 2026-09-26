# RLCatan

Train and evaluate Catan bots with masked PPO, a board transformer, and
[Catanatron](https://github.com/bcollazo/catanatron) as the game engine.

## Setup

Python 3.12+ (checks run on 3.13):

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m rlcatan.training --output runs/first --seats --trading \
  --players 4 --target-vp 10 --steps 100000 --league builder planner
.venv/bin/python -m rlcatan.benchmark --model runs/first --suite --pairs 100 --seed 9500000
```

## Method

- Featurized board state, own hand, public information, and approximate card counting.
- Three attention layers, width 64, four heads; shared action scorer and value head.
- Maskable PPO with potential-based reward shaping and scripted/frozen opponent leagues.
- Evaluation across starting seats and fixed boards, with board-level confidence intervals.

[Methodology](METHODOLOGY.md) explains the representations, architecture, and
training choices, with a diagram of the path from game state to decisions.

## Play

The browser demo supports two to four players and optional negotiation.

Weights are local artifacts, not included in a clone. Keep each `model.zip`
with its `run.json`; load only checkpoints you trust. Training produces both.
The separately developed sibling `RLCatan-play` provides
browser play; with both projects and the checkpoints available:

```sh
cd ../RLCatan-play
../RLCatan/.venv/bin/python -m pip install -r requirements-play.txt
../RLCatan/.venv/bin/python play.py
```

## Code and checks

`rlcatan/`: `game.py` (environment), `policies.py` (network), `opponents.py`
(opponents), `training.py` (PPO), `benchmark.py` (evaluation).
`tools/habits.py` measures openings, building choices, robber use, and trades.

```sh
.venv/bin/python test_rlcatan.py
.venv/bin/python test_rules.py
```

Public demo hosting, checkpoint distribution, and a project license remain
release tasks. Earlier notebook code is on `legacy-rlmodels`.
