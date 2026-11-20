# Emergent Coordination Without Language in Multi-Agent Systems

This project investigates whether heterogeneous autonomous vehicles can learn to coordinate at an unsignalized intersection without verbal communication or an explicit message channel.

Three agents with different learned driving behaviors enter an intersection with background traffic. Each agent observes only a local 47-dimensional sensor state and selects one of four actions:

- Brake
- Yield
- Maintain speed
- Accelerate

The system is trained with **multi-agent proximal policy optimization (MAPPO)**. The actors are decentralized, while a centralized critic observes the combined state during training. Coordination must therefore emerge from shared training and local observations rather than direct communication or a learned communication protocol.

## Research Pipeline

### Stage 1: Heterogeneous behavior policies

The project first trains three PPO policies representing different driving styles:

| Behavior | Checkpoint |
| --- | --- |
| Aggressive | `models/aggressive_seed45.pt` |
| Neutral | `models/neutral_seed46.pt` |
| Conservative | `models/conservative_seed50.pt` |

Stage 1 includes the behavior-profile definitions, training scripts, and single-agent or baseline intersection environment used to establish these policies.

### Stage 2: Coordinated MAPPO

Stage 2 places the three policies in the same unsignalized intersection:

- `agent_0` uses aggressive behavior.
- `agent_1` uses neutral behavior.
- `agent_2` uses conservative behavior.

The Stage 2 environment adds:

- Three controlled agents and configurable background traffic
- 47-dimensional local observations per controlled agent
- A 141-dimensional centralized joint observation
- Discrete action filtering for intersection safety
- Route, maneuver, TTC, collision, progress, and completion metrics
- Optional Stage 1 checkpoint warm starts
- A curriculum that can begin without traffic and continue with full traffic

The decentralized actors do not receive direct messages from one another. Theory-of-mind and belief-prediction components are included for analysis and visualization, but they are not communication channels.

## Key Features

- Heterogeneous agent behaviors
- Decentralized actors with centralized MAPPO training
- Warm-starting from independently trained PPO policies
- Background-traffic simulation
- Road-bounded routes and maneuver handling
- Collision, near-collision, unsafe-TTC, and route-progress diagnostics
- Live policy visualization
- JSON Lines logs of observations, actions, decisions, beliefs, rewards, and safety metrics

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

The main runtime dependencies are PyTorch, NumPy, and WebSockets.

To use CUDA, install a PyTorch build compatible with the available NVIDIA driver and GPU.

## Running the Tests

Run the Stage 2 environment regression suite:

```bash
python src/stage2/test_stage2.py
```

Additional environment, route, safety, and diagnostic scripts are available under `src/`.

## Training

Train Stage 2 MAPPO from the project root:

```bash
python src/stage2/training/train_mappo.py --total-updates 50 --eval-interval 10 --eval-episodes 5 --final-eval-episodes 20
```

The default trainer configuration uses:

- Seed 42
- 2048 rollout steps
- PPO clipping and GAE
- Observation and advantage normalization
- Stage 1 warm starts
- Background traffic
- CPU or CUDA, depending on availability

Override the device explicitly:

```bash
python src/stage2/training/train_mappo.py --device cpu
python src/stage2/training/train_mappo.py --device cuda
```

Training outputs checkpoints, training history, final evaluations, and configuration data under `models/stage2_mappo/`. Use `--help` to list the available command-line options.

## Live Visualization

Start the local visualization server:

```bash
python visualization/server.py
```

Open `http://localhost:8766/` in a browser. The interface shows:

- Controlled and background vehicle positions
- Proposed and applied actions
- Yielding and following decisions
- Speed and acceleration
- Route progress
- Predicted conflict timing
- Safety and TTC readings
- Theory-of-mind predictions and belief summaries

The server writes each simulation frame to `visualization/readings.jsonl`. Each line is a JSON object containing the UTC timestamp, episode and step, 47-value observations for each controlled agent, vehicle positions and motion, actions, rewards, decisions, beliefs, predictions, and safety metrics.

The current log can also be downloaded from `http://localhost:8766/readings.jsonl`.

## Evaluation

Stage 2 evaluation reports include episode rewards, completion and collision outcomes, unsafe and near-collision events, route completion, and per-vehicle behavior statistics. Use the scripts and JSON reports under `src/evaluation/` and `models/stage2_mappo/` to compare policies and training phases.

## Project Layout

```text
src/
  agents/       PPO and Stage 1 behavior implementations
  envs/         Stage 1 intersection and traffic environments
  stage2/       Stage 2 MAPPO environment, tests, diagnostics, and trainer
  tom/          Theory-of-mind analysis and prediction components
  evaluation/   Behavioral and policy comparison tools
  training/     Stage 1 training entry points
models/
  Stage 1 behavior checkpoints
  Stage 2 MAPPO checkpoints and evaluation reports
visualization/
  Local browser UI and WebSocket/server implementation
README.md
requirements.txt
```
