"""
Visualization runner server.

Runs the Stage-2 MAPPO environment with Theory of Mind enabled,
streams real-time state to a browser via WebSockets (using the
built-in asyncio + websockets library, no Flask dependency needed).

Start with:
    python visualization/server.py

Then open visualization/index.html in a browser.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow running from any CWD
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.stage2.intersection_env import (
    Stage2IntersectionEnv,
    Stage2Config,
    SELECTED_MODELS,
    ACTION_NAMES,
)
from src.tom.theory_of_mind import MultiAgentToMManager

# ---------------------------------------------------------------------------
# Reproduce the ActorCritic architecture from src/agents/ppo.py
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor  = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.actor(x), self.critic(x).squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ---------------------------------------------------------------------------
# Load Stage-1 checkpoints
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT / "models"
DEVICE = torch.device("cpu")
READINGS_PATH = Path(__file__).resolve().parent / "readings.jsonl"

def load_stage1_policies(obs_dim: int):
    """Load Stage-1 policies. Returns (policies_dict, normers_dict)."""
    policies: Dict[str, ActorCritic] = {}
    normers:  Dict[str, RunningNorm]  = {}

    for agent_id, meta in SELECTED_MODELS.items():
        ckpt_path = MODELS_DIR / meta["checkpoint"]

        if ckpt_path.exists():
            payload = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
            ckpt_obs_dim = int(payload.get("obs_dim", obs_dim))
            model = ActorCritic(ckpt_obs_dim, action_dim=4).to(DEVICE)
            model.load_state_dict(payload["model_state_dict"])
            print(f"[ok] Loaded {meta['checkpoint']} for {agent_id} (obs_dim={ckpt_obs_dim})")

            # Restore running normalizer stats
            norm = RunningNorm(ckpt_obs_dim)
            if "obs_mean" in payload and payload["obs_mean"] is not None:
                norm.mean  = payload["obs_mean"]
                norm.var   = payload["obs_var"]
                norm.count = float(payload.get("obs_count", 1e4))
        else:
            print(f"[warn] Checkpoint not found: {ckpt_path}, using random weights")
            model = ActorCritic(obs_dim, action_dim=4).to(DEVICE)
            norm  = RunningNorm(obs_dim)

        model.eval()
        policies[agent_id] = model
        normers[agent_id]  = norm

    return policies, normers



# ---------------------------------------------------------------------------
# Observation normalizer (running stats – mirrors training)
# ---------------------------------------------------------------------------

class RunningNorm:
    def __init__(self, dim: int) -> None:
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var  = np.ones(dim,  dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        delta = x[0] - self.mean
        self.count += 1
        self.mean += delta / self.count
        self.var = np.maximum(
            self.var + delta * (x[0] - self.mean) / self.count, 1e-8
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Simulation state (shared between sim thread and WS server)
# ---------------------------------------------------------------------------

class SimState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[dict] = None
        self.running = True
        self.paused  = False
        self.speed   = 1.0   # sim speed multiplier
        self.episode = 0
        self.step    = 0
        self.total_collisions = 0
        self.total_completions = 0


SIM_STATE = SimState()


# ---------------------------------------------------------------------------
# Color mapping helpers
# ---------------------------------------------------------------------------

AGENT_COLORS = {
    "agent_0": "#FF4B4B",   # aggressive – red
    "agent_1": "#4BD4FF",   # neutral    – cyan
    "agent_2": "#7BFF4B",   # conservative – green
}

BEHAVIOR_COLORS = {
    "aggressive":   "#FF4B4B",
    "neutral":      "#4BD4FF",
    "conservative": "#7BFF4B",
}

ACTION_COLORS = {
    0: "#FF2222",   # brake      – bright red
    1: "#FFB822",   # yield      – amber
    2: "#4BD4FF",   # maintain   – blue
    3: "#22FF88",   # accelerate – green
}


# ---------------------------------------------------------------------------
# Convert environment state to serializable frame
# ---------------------------------------------------------------------------

def build_frame(
    env: Stage2IntersectionEnv,
    observations: Dict[str, np.ndarray],
    actions: Dict[str, int],
    proposed_actions: Dict[str, int],
    decisions: Dict[str, str],
    rewards: Dict[str, float],
    infos: Dict,
    tom_manager: MultiAgentToMManager,
    episode: int,
    step: int,
    total_collisions: int,
    total_completions: int,
    episode_rewards: Dict[str, float],
) -> dict:

    # ---- learning vehicles ----
    vehicles_data = []
    for agent_id, vehicle in env.vehicles.items():
        behavior = env.behaviors[agent_id]
        action   = actions.get(agent_id, 2)
        vehicles_data.append({
            "id":       agent_id,
            "behavior": behavior,
            "color":    AGENT_COLORS[agent_id],
            "x":        round(vehicle.x,   3),
            "y":        round(vehicle.y,   3),
            "heading":  round(vehicle.heading, 4),
            "speed":    round(vehicle.speed, 3),
            "acceleration": round(vehicle.acceleration, 3),
            "action":   action,
            "action_name": ACTION_NAMES[action],
            "action_color": ACTION_COLORS[action],
            "proposed_action": proposed_actions.get(agent_id, action),
            "proposed_action_name": ACTION_NAMES[
                proposed_actions.get(agent_id, action)
            ],
            "decision": decisions.get(agent_id, "Policy"),
            "completed": vehicle.completed,
            "collided":  vehicle.collided,
            "progress":  round(vehicle.progress, 2),
            "approach":  vehicle.approach,
            "maneuver":  vehicle.maneuver,
            "lane":      vehicle.lane,
            "reward":    round(rewards.get(agent_id, 0.0), 4),
            "ep_reward": round(episode_rewards.get(agent_id, 0.0), 4),
        })

    # ---- background vehicles ----
    bg_data = []
    for bv in env.background_vehicles:
        if bv.completed:
            continue
        bg_data.append({
            "id":      bv.vehicle_id,
            "x":       round(bv.x, 3),
            "y":       round(bv.y, 3),
            "heading": round(bv.heading, 4),
            "speed":   round(bv.speed, 3),
            "acceleration": round(bv.acceleration, 3),
            "progress": round(bv.progress, 2),
            "approach": bv.approach,
            "maneuver": bv.maneuver,
            "lane": bv.lane,
        })

    # ---- Theory of Mind ----
    tom_summary   = tom_manager.get_full_summary()
    adapt_hints   = tom_manager.get_adaptation_hints()

    tom_data: Dict[str, dict] = {}
    for agent_id in env.agents:
        mod = tom_manager.modules[agent_id]
        agent_tom: Dict[str, dict] = {}
        for other_id in mod.other_agent_ids:
            bs = mod.beliefs[other_id]
            agent_tom[other_id] = {
                "beliefs": {
                    "aggressive":   round(float(bs.beliefs[0]), 3),
                    "neutral":      round(float(bs.beliefs[1]), 3),
                    "conservative": round(float(bs.beliefs[2]), 3),
                },
                "dominant_behavior": bs.dominant_behavior,
                "confidence":        round(bs.dominant_confidence, 3),
                "predicted_actions": {
                    "brake":       round(float(bs.predicted_action_probs[0]), 3),
                    "yield":       round(float(bs.predicted_action_probs[1]), 3),
                    "maintain":    round(float(bs.predicted_action_probs[2]), 3),
                    "accelerate":  round(float(bs.predicted_action_probs[3]), 3),
                },
                "risk_tolerance": round(bs.inferred_risk_tolerance, 3),
                "svo":            round(bs.social_value_orientation, 3),
                "adaptation":     adapt_hints[agent_id].get(other_id, {}),
            }
        tom_data[agent_id] = agent_tom

    # ---- global info ----
    global_info = infos.get("global", {})

    return {
        "type":     "state",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "episode":  episode,
        "step":     step,
        "total_collisions":  total_collisions,
        "total_completions": total_completions,
        "vehicles":  vehicles_data,
        "bg_vehicles": bg_data,
        "sensor_readings": {
            agent_id: np.asarray(observations[agent_id], dtype=np.float32)
            .round(6)
            .tolist()
            for agent_id in env.agents
        },
        "tom":       tom_data,
        "global": {
            "agent_collisions":   global_info.get("agent_collisions", 0),
            "traffic_collisions": global_info.get("traffic_collisions", 0),
            "near_collisions":    global_info.get("near_collisions", 0),
            "min_distance":       round(global_info.get("minimum_distance", 999.0), 3),
            "min_ttc":            round(min(global_info.get("minimum_ttc", 99.0), 99.0), 3),
            "step_count":         step,
        },
        "config": {
            "intersection_half_width": env.config.intersection_half_width,
            "approach_length":         env.config.approach_length,
            "lane_width":              env.config.lane_width,
            "lanes_per_approach":      env.config.lanes_per_approach,
        },
    }


def _project_stage2_observation(observation: np.ndarray) -> np.ndarray:
    """Project the 47-D Stage-2 sensor vector into the Stage-1 policy layout."""
    if observation.shape != (47,):
        raise ValueError(
            f"Expected Stage-2 observation shape (47,), got {observation.shape}"
        )

    x, y, speed, acceleration = observation[:4]
    sin_heading, cos_heading = observation[4:6]
    maneuver = observation[8:11]
    angle = math.atan2(float(sin_heading), float(cos_heading))
    directions = (
        (0.0, -math.pi / 2.0),
        (0.33, math.pi / 2.0),
        (0.66, math.pi),
        (1.0, 0.0),
    )
    direction = min(
        directions,
        key=lambda item: abs(math.atan2(
            math.sin(angle - item[1]),
            math.cos(angle - item[1]),
        )),
    )[0]
    maneuver_value = (
        0.5 if maneuver[1] > 0.5
        else 1.0 if maneuver[2] > 0.5
        else 0.0
    )
    result = [
        float(x), float(y), float(speed), float(acceleration),
        0.0, direction, maneuver_value,
    ]

    others = []
    for start in (11, 19):
        dx, dy, rel_speed, distance = observation[start:start + 4]
        ttc = observation[start + 4]
        others.append((
            float(distance), float(dx), float(dy),
            float(rel_speed), float(ttc),
        ))
    for start in (27, 32, 37, 42):
        dx, dy, rel_speed, distance, ttc = observation[start:start + 5]
        others.append((
            float(distance), float(dx), float(dy),
            float(rel_speed), float(ttc),
        ))

    min_distance = min((item[0] for item in others), default=1.0)
    min_ttc = min((item[4] for item in others), default=1.0)
    result.extend((
        float(np.clip(min_distance, 0.0, 1.0)),
        float(np.clip(min_ttc, 0.0, 1.0)),
        float(np.clip(math.hypot(float(x), float(y)) / 120.0, 0.0, 1.0)),
    ))
    others.sort(key=lambda item: item[0])
    for distance, dx, dy, rel_speed, ttc in others[:5]:
        result.extend((
            dx, dy, rel_speed,
            float(np.clip(distance, 0.0, 1.0)),
            0.0,
            float(np.clip(ttc, 0.0, 1.0)),
        ))
    result.extend([0.0] * (40 - len(result)))
    return np.asarray(result[:40], dtype=np.float32)


def _predictive_safety_filter(
    env: Stage2IntersectionEnv,
    proposed_actions: Dict[str, int],
    tom_manager: MultiAgentToMManager,
) -> tuple[Dict[str, int], Dict[str, str]]:
    """Apply route-conflict checks using ToM-predicted action distributions."""
    predicted_action_probs = {}
    for agent_id in env.agents:
        observations = []
        for observer in tom_manager.modules.values():
            belief = observer.beliefs.get(agent_id)
            if belief is not None:
                observations.append(belief.predicted_action_probs)
        if observations:
            predicted_action_probs[agent_id] = np.mean(observations, axis=0)
    return env.filter_actions(proposed_actions, predicted_action_probs)


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------

def run_simulation(state: SimState) -> None:
    config = Stage2Config(max_steps=800)
    env    = Stage2IntersectionEnv(seed=42, config=config, background_traffic=True)

    stage2_obs_dim = env.observation_dim

    # Load Stage-1 policies + their saved normalizer stats
    policies, normers = load_stage1_policies(stage2_obs_dim)

    # Determine per-agent obs_dim from loaded policies
    policy_obs_dims: Dict[str, int] = {}
    for aid, pol in policies.items():
        # First linear layer input size
        policy_obs_dims[aid] = pol.shared[0].in_features

    tom_manager = MultiAgentToMManager(
        agent_ids=env.agents,
        alpha=0.85,
    )

    episode = 0
    total_collisions  = 0
    total_completions = 0

    while state.running:
        episode += 1
        obs_dict  = env.reset(seed=episode * 17 + 3)
        ep_rewards: Dict[str, float] = {aid: 0.0 for aid in env.agents}
        terminated = truncated = False
        step = 0

        while not (terminated or truncated) and state.running:
            if state.paused:
                time.sleep(0.05)
                continue

            # ---- select actions via Stage-1 policies ----
            proposed_actions: Dict[str, int] = {}
            for agent_id in env.agents:
                raw_obs  = np.asarray(obs_dict[agent_id], dtype=np.float32)
                projected = _project_stage2_observation(raw_obs)
                pol_dim = policy_obs_dims[agent_id]
                if projected.shape[0] != pol_dim:
                    raise ValueError(
                        f"{agent_id} policy expects {pol_dim} observations, "
                        f"got {projected.shape[0]}"
                    )

                norm = normers[agent_id]
                norm_obs = norm.normalize(projected)
                obs_t    = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action_t, _, _, _ = policies[agent_id].get_action(
                        obs_t, deterministic=True
                    )
                proposed_actions[agent_id] = int(action_t.item())

            actions, decisions = _predictive_safety_filter(
                env, proposed_actions, tom_manager
            )

            # ---- Theory of Mind update ----
            tom_manager.update_all(actions)

            # ---- environment step ----
            obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            step += 1

            for aid in env.agents:
                ep_rewards[aid] += rewards.get(aid, 0.0)

            if terminated:
                if env.total_collisions > 0:
                    total_collisions += 1
                else:
                    completed_count = sum(
                        1 for v in env.vehicles.values() if v.completed
                    )
                    if completed_count == len(env.agents):
                        total_completions += 1

            # ---- build and store frame ----
            frame = build_frame(
                env=env,
                observations=obs_dict,
                actions=actions,
                proposed_actions=proposed_actions,
                decisions=decisions,
                rewards=rewards,
                infos=infos,
                tom_manager=tom_manager,
                episode=episode,
                step=step,
                total_collisions=total_collisions,
                total_completions=total_completions,
                episode_rewards=ep_rewards,
            )

            with READINGS_PATH.open("a", encoding="utf-8") as readings_file:
                readings_file.write(
                    json.dumps(frame, separators=(",", ":")) + "\n"
                )
                readings_file.flush()

            with state.lock:
                state.frame   = frame
                state.episode = episode
                state.step    = step
                state.total_collisions  = total_collisions
                state.total_completions = total_completions

            # ---- timing ----
            sleep_time = max(0.01, 0.08 / max(state.speed, 0.1))
            time.sleep(sleep_time)

    print("[sim] Simulation thread exiting.")


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

_connected_clients = set()


async def ws_handler(websocket) -> None:
    global _connected_clients
    _connected_clients.add(websocket)
    print(f"[ws] Client connected. Total: {len(_connected_clients)}")
    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
                cmd = msg.get("cmd", "")
                if cmd == "pause":
                    SIM_STATE.paused = True
                elif cmd == "resume":
                    SIM_STATE.paused = False
                elif cmd == "speed":
                    SIM_STATE.speed = float(msg.get("value", 1.0))
            except Exception:
                pass
    except Exception:
        pass
    finally:
        _connected_clients.discard(websocket)
        print(f"[ws] Client disconnected. Total: {len(_connected_clients)}")


async def broadcast_loop() -> None:
    global _connected_clients
    while SIM_STATE.running:
        await asyncio.sleep(0.05)   # ~20 fps to browser
        if not _connected_clients:
            continue
        with SIM_STATE.lock:
            frame = SIM_STATE.frame
        if frame is None:
            continue
        msg = json.dumps(frame)
        dead = set()
        for client in list(_connected_clients):
            try:
                await client.send(msg)
            except Exception:
                dead.add(client)
        _connected_clients -= dead


async def http_handler(reader, writer) -> None:
    """Minimal HTTP/1.1 server for serving index.html and static assets."""
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        request_line = request_line.decode(errors="replace").strip()

        # Drain headers
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if line in (b"\r\n", b"\n", b""):
                break

        path = "/"
        if request_line.startswith("GET"):
            parts = request_line.split()
            if len(parts) >= 2:
                path = parts[1]

        # Serve index.html for root
        if path in ("/", "/index.html"):
            html_path = Path(__file__).parent / "index.html"
            try:
                content = html_path.read_bytes()
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/html; charset=utf-8\r\n"
                    b"Access-Control-Allow-Origin: *\r\n"
                    + f"Content-Length: {len(content)}\r\n\r\n".encode()
                    + content
                )
            except FileNotFoundError:
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\nNot Found")
        elif path == "/readings.jsonl":
            if not READINGS_PATH.exists():
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\nNo readings recorded yet")
            else:
                file_size = READINGS_PATH.stat().st_size
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/x-ndjson; charset=utf-8\r\n"
                    b"Content-Disposition: attachment; filename=readings.jsonl\r\n"
                    + f"Content-Length: {file_size}\r\n\r\n".encode()
                )
                await writer.drain()
                remaining = file_size
                with READINGS_PATH.open("rb") as readings_file:
                    while remaining:
                        chunk = readings_file.read(min(65536, remaining))
                        if not chunk:
                            break
                        writer.write(chunk)
                        await writer.drain()
                        remaining -= len(chunk)
                return
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\n\r\nNot Found")

        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()


async def main_async() -> None:
    import websockets

    sim_thread = threading.Thread(target=run_simulation, args=(SIM_STATE,), daemon=True)
    sim_thread.start()

    # HTTP server for the HTML page
    http_server = await asyncio.start_server(http_handler, "localhost", 8766)

    print("\n" + "="*60)
    print(" Multi-Agent Intersection -- Theory of Mind Visualizer")
    print("="*60)
    print(" [OK] Simulation started")
    print(" [WEB] Open in browser -> http://localhost:8766")
    print(" [WS]  WebSocket stream -> ws://localhost:8765")
    print("="*60 + "\n")

    # Auto-open browser
    import webbrowser
    import time as _time
    _time.sleep(1.5)  # give sim thread a moment
    webbrowser.open("http://localhost:8766")

    async with websockets.serve(ws_handler, "localhost", 8765):
        async with http_server:
            await broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        SIM_STATE.running = False
        print("\n[server] Stopped.")
