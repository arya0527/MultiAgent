# Emergent Coordination Without Language

Research project on coordination between heterogeneous driving agents at an unsignalized intersection, without explicit communication.

The repository contains independently trained Stage 1 behavior policies and a Stage 2 three-agent MAPPO environment. Stage 2 uses decentralized actors with a centralized training critic; background traffic is enabled throughout MAPPO training and evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Run the Stage 2 environment regressions:

```bash
python src/stage2/test_stage2.py
```

Train Stage 2 MAPPO:

```bash
python src/stage2/training/train_mappo.py --total-updates 50 --eval-interval 10 --eval-episodes 5 --final-eval-episodes 20
```

Add `--device cuda` or `--device cpu` to choose a device. Training writes checkpoints and evaluation reports under `models/stage2_mappo/` by default.

Run the live intersection visualization:

```bash
python visualization/server.py
```

Open the local page served by the visualization server. Vehicles follow road-bounded routes; live policy actions are filtered against predicted agent and background-traffic arrival times at shared conflict points. The status cards show the chosen action, any yielding/following decision, speed, acceleration, and route progress. Safety TTC readings use predicted route conflicts where applicable.

The server appends every simulation frame as one JSON object per line to `visualization/readings.jsonl`. Each record includes a UTC timestamp, episode/step, raw 47-value sensor observations per agent, controlled and background vehicle positions and motion, proposed and applied actions, decisions, ToM beliefs/action predictions, rewards, and safety metrics. Use the **Readings** button in the visualizer or open `http://localhost:8766/readings.jsonl` to download the current log.

## Project Layout

- `src/agents/` contains Stage 1 behavior profiles and PPO implementation.
- `src/envs/` contains the Stage 1 driving environment.
- `src/stage2/` contains the intersection environment, diagnostics, tests, and MAPPO trainer.
- `models/` contains selected Stage 1 policies and Stage 2 checkpoints.