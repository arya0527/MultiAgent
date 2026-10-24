# src/stage2/intersection_env.py

from __future__ import annotations

import math
import random
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# ENUM-LIKE CONSTANTS
# ============================================================

APPROACHES = ("north", "south", "east", "west")
MANEUVERS = ("straight", "left", "right")

# Stage-1-compatible actions
BRAKE = 0
YIELD = 1
MAINTAIN = 2
ACCELERATE = 3

ACTION_NAMES = {
    BRAKE: "brake",
    YIELD: "yield",
    MAINTAIN: "maintain",
    ACCELERATE: "accelerate",
}

BEHAVIORS = (
    "aggressive",
    "neutral",
    "conservative",
)


# ============================================================
# SELECTED STAGE-1 AGENTS
# ============================================================

SELECTED_MODELS = {
    "agent_0": {
        "behavior": "aggressive",
        "checkpoint": "aggressive_seed45.pt",
    },
    "agent_1": {
        "behavior": "neutral",
        "checkpoint": "neutral_seed46.pt",
    },
    "agent_2": {
        "behavior": "conservative",
        "checkpoint": "conservative_seed50.pt",
    },
}


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

@dataclass
class Stage2Config:
    # --------------------------------------------------------
    # Road geometry
    # --------------------------------------------------------

    approach_length: float = 120.0
    exit_length: float = 120.0

    intersection_half_width: float = 12.0

    lane_width: float = 3.5
    lanes_per_approach: int = 2

    stop_line_distance: float = 12.0

    # --------------------------------------------------------
    # Vehicle geometry
    # --------------------------------------------------------

    vehicle_length: float = 4.5
    vehicle_width: float = 1.8

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    dt: float = 0.1
    max_steps: int = 1800

    max_speed: float = 13.0
    min_speed: float = 0.0

    max_acceleration: float = 2.0
    max_braking: float = -4.0
    yield_acceleration: float = -2.0

    # --------------------------------------------------------
    # Traffic
    # --------------------------------------------------------

    background_vehicle_count: int = 8

    background_min_speed: float = 4.0
    background_max_speed: float = 11.0
    minimum_same_lane_spawn_gap: float = 20.0

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    collision_distance: float = 2.5
    near_collision_distance: float = 6.0

    dangerous_ttc: float = 3.0

    # --------------------------------------------------------
    # Reward
    # --------------------------------------------------------

    collision_penalty: float = 300.0
    completion_reward: float = 2000.0
    team_completion_reward: float = 3000.0
    reward_scale: float = 0.01
    progress_scale: float = 2.0
    time_penalty: float = 0.01

    unsafe_penalty: float = 0.10


# ============================================================
# VEHICLE
# ============================================================

@dataclass
class Vehicle:
    vehicle_id: str

    behavior: Optional[str]

    approach: str
    maneuver: str
    lane: int

    x: float
    y: float

    speed: float
    acceleration: float
    heading: float

    progress: float = 0.0
    progress_delta: float = 0.0

    completed: bool = False
    collided: bool = False
    agent_collided: bool = False
    traffic_collided: bool = False
    collision_event: bool = False
    completion_reward_given: bool = False

    is_learning: bool = False

    route_points: List[Tuple[float, float]] = field(
        default_factory=list
    )

    route_distance: float = 0.0

    # Cached route geometry. These are immutable for a vehicle
    # during an episode and avoid rebuilding NumPy arrays on
    # every simulation step.
    route_segment_lengths: List[float] = field(default_factory=list)
    route_cumulative_lengths: List[float] = field(default_factory=list)
    route_points_np: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.route_points:
            points = np.asarray(self.route_points, dtype=np.float64)
            self.route_points_np = points
            segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.route_segment_lengths = segments.tolist()
            self.route_cumulative_lengths = np.concatenate(
                ([0.0], np.cumsum(segments))
            ).tolist()

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    @property
    def length(self) -> float:
        return 4.5

    @property
    def width(self) -> float:
        return 1.8

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return np.array(
            [self.x, self.y],
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        acceleration: float,
        dt: float,
        max_speed: float,
    ) -> None:

        self.acceleration = float(acceleration)

        old_speed = self.speed

        self.speed += (
            self.acceleration * dt
        )

        self.speed = float(
            np.clip(
                self.speed,
                0.0,
                max_speed,
            )
        )

        distance = (
            0.5
            * (
                old_speed
                + self.speed
            )
            * dt
        )

        self.progress += max(
            distance,
            0.0,
        )

        self.progress_delta = max(
            distance,
            0.0,
        )

        self._update_position_from_route()

    # --------------------------------------------------------
    # Route interpolation
    # --------------------------------------------------------

    def _update_position_from_route(self) -> None:

        if not self.route_points_np is not None:
            return

        if self.progress <= 0.0:
            point = self.route_points_np[0]
            self.x = float(point[0])
            self.y = float(point[1])
            self.heading = math.atan2(
                self.route_points_np[1, 1] - self.route_points_np[0, 1],
                self.route_points_np[1, 0] - self.route_points_np[0, 0],
            )
            return

        total = self.route_cumulative_lengths[-1]
        if self.progress >= total:
            point = self.route_points_np[-1]
            self.x = float(point[0])
            self.y = float(point[1])
            if len(self.route_points_np) >= 2:
                p1 = self.route_points_np[-2]
                p2 = self.route_points_np[-1]
                self.heading = math.atan2(
                    p2[1] - p1[1],
                    p2[0] - p1[0],
                )
            self.completed = True
            self.speed = 0.0
            return

        # Same segment-selection convention as the original loop,
        # but O(log N) instead of scanning all route samples.
        segment_index = bisect_left(
            self.route_cumulative_lengths,
            self.progress,
        ) - 1
        segment_index = max(0, min(segment_index, len(self.route_segment_lengths) - 1))

        p1 = self.route_points_np[segment_index]
        p2 = self.route_points_np[segment_index + 1]
        segment_start = self.route_cumulative_lengths[segment_index]
        segment = max(self.route_segment_lengths[segment_index], 1e-8)
        ratio = (self.progress - segment_start) / segment
        point = p1 + ratio * (p2 - p1)

        self.x = float(point[0])
        self.y = float(point[1])
        self.heading = math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0],
        )


# ============================================================
# MAIN ENVIRONMENT
# ============================================================

class Stage2IntersectionEnv:
    """
    Large four-way unsignalized intersection for Stage 2 MARL.

    Three learning vehicles:

        agent_0 -> aggressive
        agent_1 -> neutral
        agent_2 -> conservative

    Selected Stage-1 checkpoints:

        aggressive_seed45.pt
        neutral_seed46.pt
        conservative_seed50.pt

    The environment itself does not load neural-network
    checkpoints. It provides the physical environment in which
    those policies will later operate.

    Action space:

        0 = Brake
        1 = Yield
        2 = Maintain
        3 = Accelerate
    """

    def __init__(
        self,
        seed: int = 42,
        config: Optional[Stage2Config] = None,
        background_traffic: bool = True,
    ) -> None:

        self.config = (
            config
            if config is not None
            else Stage2Config()
        )

        self.seed = seed

        self.rng = random.Random(seed)

        self.background_traffic_enabled = (
            background_traffic
        )

        self.agents = list(
            SELECTED_MODELS.keys()
        )

        self.behaviors = {
            agent_id: SELECTED_MODELS[
                agent_id
            ]["behavior"]
            for agent_id in self.agents
        }

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.vehicles: Dict[
            str,
            Vehicle,
        ] = {}

        self.background_vehicles: List[
            Vehicle
        ] = []

        self.step_count = 0

        self.total_collisions = 0
        self.agent_collisions = 0
        self.traffic_collisions = 0
        self.near_collisions = 0

        self.minimum_distance = float("inf")
        self.minimum_ttc = float("inf")
        self._agent_collision_pairs = set()
        self._traffic_collision_pairs = set()
        self.team_completion_reward_given = False

        # Route geometry is fixed within an episode. Cache pairwise
        # conflict distances once instead of recomputing sampled
        # route intersections at every observation.
        self._conflict_cache = {}

        # ----------------------------------------------------
        # Observation
        # ----------------------------------------------------

        self.observation_dim = (
            self._calculate_observation_dimension()
        )

        self.action_dim = 4

    # ========================================================
    # RESET
    # ========================================================

        # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed: Optional[int] = None,
        scenario: Optional[List[dict]] = None,
    ):
        """
        Reset the Stage-2 environment.

        Random resets:
        - assign unique (approach, lane) spawn slots to learning agents
        - generate background traffic
        - reject any initially overlapping vehicle state

        Explicit scenarios are preserved as supplied.
        """

        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        # ----------------------------------------------------
        # Explicit scenario
        # ----------------------------------------------------
        if scenario is not None:
            self._reset_state()

            self._build_learning_vehicles(scenario)

            if self.background_traffic_enabled:
                self._create_background_traffic()

            if self._initial_state_has_collision():
                raise ValueError(
                    "Explicit scenario creates an initial collision."
                )

            return {
                agent_id: self._get_observation(agent_id)
                for agent_id in self.agents
            }

        # ----------------------------------------------------
        # Random scenario
        # ----------------------------------------------------
        max_reset_attempts = 100

        for _ in range(max_reset_attempts):
            self._reset_state()

            generated_scenario = (
                self._random_learning_scenario()
            )

            self._build_learning_vehicles(
                generated_scenario
            )

            if self.background_traffic_enabled:
                self._create_background_traffic()

            if not self._initial_state_has_collision():
                return {
                    agent_id: self._get_observation(agent_id)
                    for agent_id in self.agents
                }

        raise RuntimeError(
            "Unable to generate a collision-free initial "
            f"Stage-2 state after {max_reset_attempts} attempts. "
            "Check spawn geometry or background traffic generation."
        )

    # ========================================================
    # RESET HELPERS
    # ========================================================

    def _reset_state(self) -> None:
        """Clear all episode state before constructing a reset."""

        self.step_count = 0

        self.total_collisions = 0
        self.agent_collisions = 0
        self.traffic_collisions = 0
        self.near_collisions = 0

        self.minimum_distance = float("inf")
        self.minimum_ttc = float("inf")
        self._agent_collision_pairs.clear()
        self._traffic_collision_pairs.clear()
        self.team_completion_reward_given = False

        self.vehicles.clear()
        self.background_vehicles.clear()

    def _build_learning_vehicles(
        self,
        scenario: List[dict],
    ) -> None:
        """Construct the three learning vehicles from a scenario."""

        if len(scenario) != len(self.agents):
            raise ValueError(
                "Scenario must contain exactly "
                f"{len(self.agents)} learning vehicles."
            )

        scenario_agents = {
            item["agent_id"]
            for item in scenario
        }

        if scenario_agents != set(self.agents):
            raise ValueError(
                "Scenario agent IDs do not match expected agents: "
                f"{self.agents}"
            )

        # Explicitly reject duplicate physical spawn slots.
        slots = [
            (
                item["approach"],
                int(item["lane"]),
            )
            for item in scenario
        ]

        if len(slots) != len(set(slots)):
            raise ValueError(
                "Learning agents cannot share the same "
                "(approach, lane) spawn slot."
            )

        for item in scenario:
            agent_id = item["agent_id"]

            vehicle = self._create_learning_vehicle(
                agent_id=agent_id,
                approach=item["approach"],
                maneuver=item["maneuver"],
                lane=int(item["lane"]),
                speed=float(
                    item.get(
                        "speed",
                        self.rng.uniform(5.0, 8.0),
                    )
                ),
            )

            self.vehicles[agent_id] = vehicle

    def _initial_state_has_collision(self) -> bool:
        """
        Return True if any pair of vehicles is already within
        the configured collision distance at reset.
        """

        all_vehicles = (
            list(self.vehicles.values())
            + list(self.background_vehicles)
        )

        for i in range(len(all_vehicles)):
            for j in range(i + 1, len(all_vehicles)):
                distance = self._distance(
                    all_vehicles[i],
                    all_vehicles[j],
                )

                if distance <= self.config.collision_distance:
                    return True

        for learning in self.vehicles.values():
            for traffic in self.background_vehicles:
                if (
                    learning.approach == traffic.approach
                    and learning.lane == traffic.lane
                    and abs(learning.progress - traffic.progress)
                    < self.config.minimum_same_lane_spawn_gap
                ):
                    return True

        return False

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        actions: Dict[str, int],
    ):

        self.step_count += 1

        # ----------------------------------------------------
        # Validate actions
        # ----------------------------------------------------

        for agent_id in self.agents:

            if agent_id not in actions:
                raise ValueError(
                    f"Missing action for {agent_id}"
                )

            action = int(
                actions[agent_id]
            )

            if action not in ACTION_NAMES:
                raise ValueError(
                    f"Invalid action {action}"
                )

        # ----------------------------------------------------
        # Simultaneous action conversion
        # ----------------------------------------------------

        accelerations = {}

        for agent_id in self.agents:

            vehicle = self.vehicles[
                agent_id
            ]

            accelerations[
                agent_id
            ] = self._action_to_acceleration(
                action=actions[
                    agent_id
                ],
                vehicle=vehicle,
            )

        # ----------------------------------------------------
        # Move learning vehicles
        # ----------------------------------------------------

        for agent_id in self.agents:

            vehicle = self.vehicles[
                agent_id
            ]

            vehicle.progress_delta = 0.0
            if vehicle.completed:
                continue

            vehicle.update(
                accelerations[
                    agent_id
                ],
                self.config.dt,
                self.config.max_speed,
            )

        # ----------------------------------------------------
        # Move background traffic
        # ----------------------------------------------------

        self._update_background_traffic()

        # ----------------------------------------------------
        # Collision detection
        # ----------------------------------------------------

        for vehicle in self.vehicles.values():
            vehicle.collision_event = False

        self._check_collisions()

        # ----------------------------------------------------
        # Safety metrics
        # ----------------------------------------------------

        self._update_safety_metrics()

        # ----------------------------------------------------
        # Rewards
        # ----------------------------------------------------

        rewards = {}

        for agent_id in self.agents:

            rewards[
                agent_id
            ] = self._calculate_reward(
                agent_id,
                actions[agent_id],
            )

        # ----------------------------------------------------
        # Termination
        # ----------------------------------------------------

        all_completed = all(
            self.vehicles[
                agent_id
            ].completed
            for agent_id in self.agents
        )

        if all_completed and not self.team_completion_reward_given:
            for agent_id in self.agents:
                rewards[agent_id] += self.config.team_completion_reward
            self.team_completion_reward_given = True

        terminated = (
            all_completed
            or self.total_collisions > 0
        )

        truncated = (
            self.step_count
            >= self.config.max_steps
        )

        observations = {
            agent_id: self._get_observation(
                agent_id
            )
            for agent_id in self.agents
        }

        infos = {
            agent_id: self._get_agent_info(
                agent_id,
                actions[agent_id],
            )
            for agent_id in self.agents
        }

        infos["global"] = (
            self._get_global_info()
        )

        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )

    def filter_actions(
        self,
        proposed_actions: Dict[str, int],
        predicted_action_probs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """Apply a predictive safety layer to policy actions."""
        if set(proposed_actions) != set(self.agents):
            raise ValueError("Proposed actions must include every learning agent.")
        for action in proposed_actions.values():
            if int(action) not in ACTION_NAMES:
                raise ValueError(f"Invalid proposed action: {action}")

        actions = {agent_id: int(action) for agent_id, action in proposed_actions.items()}
        decisions = {agent_id: "Policy" for agent_id in self.agents}
        accel_by_action = (
            self.config.max_braking,
            self.config.yield_acceleration,
            0.0,
            self.config.max_acceleration,
        )
        behavior_bias = {
            "aggressive": -0.4,
            "neutral": 0.0,
            "conservative": 0.4,
        }
        predicted_acceleration = {}

        for agent_id in self.agents:
            probabilities = (
                None
                if predicted_action_probs is None
                else predicted_action_probs.get(agent_id)
            )
            if probabilities is None:
                predicted = 0.0
            else:
                probabilities = np.asarray(probabilities, dtype=np.float64)
                if (
                    probabilities.shape != (self.action_dim,)
                    or not np.all(np.isfinite(probabilities))
                    or np.any(probabilities < 0.0)
                    or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
                ):
                    raise ValueError(
                        f"{agent_id} predicted action probabilities must be "
                        "a finite 4-element distribution."
                    )
                predicted = float(np.dot(probabilities, accel_by_action))
            policy_acceleration = accel_by_action[actions[agent_id]]
            predicted_acceleration[agent_id] = 0.5 * (
                predicted + policy_acceleration
            )

        for agent_id in self.agents:
            ego = self.vehicles[agent_id]
            if ego.completed or ego.collided:
                continue

            ego_acceleration = accel_by_action[actions[agent_id]]
            others = (
                [
                    vehicle
                    for other_id, vehicle in self.vehicles.items()
                    if other_id != agent_id
                ]
                + self.background_vehicles
            )
            for other in others:
                if other.completed or other.collided:
                    continue

                dx = other.x - ego.x
                dy = other.y - ego.y
                longitudinal_gap = (
                    dx * math.cos(ego.heading)
                    + dy * math.sin(ego.heading)
                )
                lateral_gap = abs(
                    -dx * math.sin(ego.heading)
                    + dy * math.cos(ego.heading)
                )
                same_direction = (
                    math.cos(ego.heading - other.heading) > 0.5
                )
                closing_speed = ego.speed - other.speed
                required_gap = max(8.0, 5.0 + ego.speed * 1.2)
                if (
                    same_direction
                    and 0.0 < longitudinal_gap < required_gap
                    and lateral_gap < self.config.lane_width
                    and closing_speed > 0.0
                ):
                    stopping_gap = (
                        ego.speed**2
                        / (2.0 * abs(self.config.max_braking))
                        + 2.0
                    )
                    actions[agent_id] = (
                        BRAKE if longitudinal_gap < max(6.0, stopping_gap)
                        else YIELD
                    )
                    decisions[agent_id] = (
                        f"Following {other.vehicle_id} "
                        f"({longitudinal_gap:.1f} m)"
                    )
                    ego_acceleration = accel_by_action[actions[agent_id]]

                other_acceleration = (
                    predicted_acceleration[other.vehicle_id]
                    if other.is_learning
                    else self._background_acceleration(other)
                )
                predicted_close_time = self._predicted_close_approach(
                    ego,
                    other,
                    ego_acceleration,
                    other_acceleration,
                )
                if predicted_close_time is not None:
                    ego_rank = behavior_bias[self.behaviors[agent_id]]
                    if other.is_learning:
                        other_rank = behavior_bias[
                            self.behaviors[other.vehicle_id]
                        ]
                        if (
                            self._is_in_intersection(other)
                            and not self._is_in_intersection(ego)
                        ):
                            other_rank -= 2.0
                        elif (
                            self._is_in_intersection(ego)
                            and not self._is_in_intersection(other)
                        ):
                            ego_rank -= 2.0
                    elif (
                        same_direction
                        and longitudinal_gap > 0.0
                        and lateral_gap < self.config.lane_width
                    ):
                        other_rank = -1.0
                    elif (
                        self._is_in_intersection(other)
                        and not self._is_in_intersection(ego)
                    ):
                        other_rank = -2.0
                    else:
                        other_rank = 1.0
                    if (ego_rank, agent_id) > (other_rank, other.vehicle_id):
                        actions[agent_id] = (
                            BRAKE if predicted_close_time < 2.0 else YIELD
                        )
                        decisions[agent_id] = (
                            f"Predicted conflict with {other.vehicle_id} "
                            f"({predicted_close_time:.1f} s)"
                        )
                        ego_acceleration = accel_by_action[actions[agent_id]]

                ego_conflict, other_conflict = self._get_conflict_distances(
                    ego, other
                )
                if ego_conflict is None or other_conflict is None:
                    continue

                ego_remaining = ego_conflict - ego.progress
                other_remaining = other_conflict - other.progress
                if (
                    ego_remaining < -ego.length
                    or other_remaining < -other.length
                ):
                    continue

                ego_time = self._time_to_distance(
                    ego_remaining, ego.speed, ego_acceleration
                )
                other_time = self._time_to_distance(
                    other_remaining, other.speed, other_acceleration
                )
                if not math.isfinite(ego_time) or not math.isfinite(other_time):
                    continue

                safety_gap = min(
                    4.0, 1.5 + (ego.speed + other.speed) / 10.0
                )
                if (
                    min(ego_time, other_time) > 8.0
                    or abs(ego_time - other_time) > safety_gap
                ):
                    continue

                ego_rank = (
                    ego_time + behavior_bias[self.behaviors[agent_id]]
                )
                other_inside = self._is_in_intersection(other)
                ego_inside = self._is_in_intersection(ego)
                if ego_inside and not other_inside:
                    ego_rank -= 2.0
                if other.is_learning:
                    other_rank = (
                        other_time
                        + behavior_bias[self.behaviors[other.vehicle_id]]
                    )
                    if other_inside and not ego_inside:
                        other_rank -= 2.0
                else:
                    other_rank = other_time + (
                        -2.0 if other_inside and not ego_inside else 0.5
                    )
                if (ego_rank, agent_id) > (other_rank, other.vehicle_id):
                    actions[agent_id] = BRAKE if ego_time < 2.0 else YIELD
                    decisions[agent_id] = (
                        f"Yielding to {other.vehicle_id} "
                        f"(ETA {other_time:.1f} s)"
                    )
                    ego_acceleration = accel_by_action[actions[agent_id]]

        return actions, decisions

    def _is_in_intersection(self, vehicle: Vehicle) -> bool:
        extent = (
            self.config.intersection_half_width
            + vehicle.length / 2.0
        )
        return abs(vehicle.x) <= extent and abs(vehicle.y) <= extent

    @staticmethod
    def _time_to_distance(
        distance: float,
        speed: float,
        acceleration: float,
    ) -> float:
        """Estimate arrival time under bounded constant acceleration."""
        if distance <= 0.0:
            return 0.0
        acceleration = float(np.clip(acceleration, -4.0, 2.0))
        speed = max(float(speed), 0.0)
        if abs(acceleration) < 1e-6:
            return distance / max(speed, 0.5)
        discriminant = speed * speed + 2.0 * acceleration * distance
        if discriminant < 0.0:
            return float("inf")
        return max((-speed + math.sqrt(discriminant)) / acceleration, 0.0)

    def _predicted_close_approach(
        self,
        a: Vehicle,
        b: Vehicle,
        acceleration_a: float,
        acceleration_b: float,
    ) -> Optional[float]:
        """Return when planned routes come within the collision buffer."""
        dt = 0.2
        horizon = 4.0
        threshold = self.config.collision_distance + 1.0
        progress_a = a.progress
        progress_b = b.progress
        speed_a = a.speed
        speed_b = b.speed

        for index in range(int(horizon / dt) + 1):
            if index:
                old_speed_a = speed_a
                old_speed_b = speed_b
                speed_a = float(np.clip(
                    speed_a + acceleration_a * dt,
                    0.0,
                    self.config.max_speed,
                ))
                speed_b = float(np.clip(
                    speed_b + acceleration_b * dt,
                    0.0,
                    self.config.background_max_speed,
                ))
                progress_a += 0.5 * (old_speed_a + speed_a) * dt
                progress_b += 0.5 * (old_speed_b + speed_b) * dt

            position_a = self._position_at_progress(a, progress_a)
            position_b = self._position_at_progress(b, progress_b)
            if self._distance_points(position_a, position_b) <= threshold:
                return index * dt
        return None

    @staticmethod
    def _position_at_progress(
        vehicle: Vehicle,
        progress: float,
    ) -> np.ndarray:
        if progress <= 0.0:
            return vehicle.route_points_np[0]
        if progress >= vehicle.route_cumulative_lengths[-1]:
            return vehicle.route_points_np[-1]

        segment_index = bisect_left(
            vehicle.route_cumulative_lengths,
            progress,
        ) - 1
        segment_index = max(
            0,
            min(segment_index, len(vehicle.route_segment_lengths) - 1),
        )
        segment_length = max(
            vehicle.route_segment_lengths[segment_index],
            1e-8,
        )
        ratio = (
            progress - vehicle.route_cumulative_lengths[segment_index]
        ) / segment_length
        start = vehicle.route_points_np[segment_index]
        end = vehicle.route_points_np[segment_index + 1]
        return start + ratio * (end - start)

    @staticmethod
    def _distance_points(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # ========================================================
    # LEARNING VEHICLE CREATION
    # ========================================================

    def _create_learning_vehicle(
        self,
        agent_id: str,
        approach: str,
        maneuver: str,
        lane: int,
        speed: float,
    ) -> Vehicle:

        route = self._build_route(
            approach,
            maneuver,
            lane,
        )

        x, y = route[0]

        heading = self._route_heading(
            route,
            0,
        )

        return Vehicle(
            vehicle_id=agent_id,
            behavior=self.behaviors[
                agent_id
            ],
            approach=approach,
            maneuver=maneuver,
            lane=lane,
            x=x,
            y=y,
            speed=speed,
            acceleration=0.0,
            heading=heading,
            route_points=route,
            route_distance=self._route_length(
                route
            ),
            is_learning=True,
        )

    # ========================================================
    # RANDOM LEARNING SCENARIO
    # ========================================================

    def _random_learning_scenario(
        self,
    ) -> List[dict]:
        """
        Generate a random learning scenario.

        The physical spawn slot is (approach, lane). These slots
        are sampled uniquely first. Maneuver selection is then
        performed independently, so two learning agents can never
        spawn at the same physical position.
        """

        # Four approaches x two lanes = eight physical slots.
        spawn_slots = [
            (approach, lane)
            for approach in APPROACHES
            for lane in range(
                self.config.lanes_per_approach
            )
        ]

        if len(spawn_slots) < len(self.agents):
            raise RuntimeError(
                "Not enough unique learning spawn slots."
            )

        self.rng.shuffle(spawn_slots)

        selected_slots = spawn_slots[:len(self.agents)]

        scenario = []

        for agent_id, (
            approach,
            lane,
        ) in zip(
            self.agents,
            selected_slots,
        ):
            legal_maneuvers = self._legal_maneuvers(lane)

            maneuver = self.rng.choice(
                legal_maneuvers
            )

            speed = self.rng.uniform(
                5.0,
                8.0,
            )

            scenario.append(
                {
                    "agent_id": agent_id,
                    "approach": approach,
                    "maneuver": maneuver,
                    "lane": lane,
                    "speed": speed,
                }
            )

        return scenario

    # ========================================================
    # LEGAL MANEUVERS
    # ========================================================

    @staticmethod
    def _legal_maneuvers(
        lane: int,
    ) -> Tuple[str, ...]:

        # Right lane:
        # straight / right
        if lane == 0:
            return (
                "straight",
                "right",
            )

        # Left lane:
        # straight / left
        return (
            "straight",
            "left",
        )

    # ========================================================
    # ROUTE GENERATION
    # ========================================================

    def _build_route(
        self,
        approach: str,
        maneuver: str,
        lane: int,
        samples: int = 401,
    ) -> List[Tuple[float, float]]:

        entry = self._entry_point(
            approach,
            lane,
        )

        exit_approach = (
            self._exit_approach(
                approach,
                maneuver,
            )
        )

        exit_lane = (
            self._exit_lane(
                lane,
                maneuver,
            )
        )

        exit = self._exit_point(
            exit_approach,
            exit_lane,
        )

        if maneuver == "straight":

            return self._line_route(
                entry,
                exit,
                samples,
            )

        # ----------------------------------------------------
        # Turn route
        # ----------------------------------------------------

        return self._bezier_route(
            entry,
            exit,
            approach,
            maneuver,
            samples,
        )

    # ========================================================
    # ENTRY / EXIT GEOMETRY
    # ========================================================

    def _entry_point(
        self,
        approach: str,
        lane: int,
    ) -> Tuple[float, float]:

        offset = self._lane_offset(
            lane
        )

        distance = (
            self.config.approach_length
        )

        if approach == "north":

            return (
                -offset,
                distance,
            )

        if approach == "south":

            return (
                offset,
                -distance,
            )

        if approach == "east":

            return (
                distance,
                offset,
            )

        return (
            -distance,
            -offset,
        )

    def _exit_point(
        self,
        approach: str,
        lane: int,
    ) -> Tuple[float, float]:

        offset = self._lane_offset(
            lane
        )

        distance = (
            self.config.exit_length
        )

        if approach == "north":

            return (
                offset,
                distance,
            )

        if approach == "south":

            return (
                -offset,
                -distance,
            )

        if approach == "east":

            return (
                distance,
                -offset,
            )

        return (
            -distance,
            offset,
        )

    # ========================================================
    # LANE OFFSET
    # ========================================================

    def _lane_offset(
        self,
        lane: int,
    ) -> float:

        return (
            self.config.lanes_per_approach
            - lane
            - 0.5
        ) * self.config.lane_width

    # ========================================================
    # DESTINATION APPROACH
    # ========================================================

    @staticmethod
    def _exit_approach(
        approach: str,
        maneuver: str,
    ) -> str:

        mapping = {

            "north": {
                "straight": "south",
                "left": "east",
                "right": "west",
            },

            "south": {
                "straight": "north",
                "left": "west",
                "right": "east",
            },

            "east": {
                "straight": "west",
                "left": "south",
                "right": "north",
            },

            "west": {
                "straight": "east",
                "left": "north",
                "right": "south",
            },
        }

        return mapping[
            approach
        ][maneuver]

    # ========================================================
    # EXIT LANE
    # ========================================================

    def _exit_lane(
        self,
        entry_lane: int,
        maneuver: str,
    ) -> int:

        # Lane numbering is relative to each approach's direction of travel,
        # so straight-through vehicles retain their lane index.
        return entry_lane

    # ========================================================
    # LINE ROUTE
    # ========================================================

    @staticmethod
    def _line_route(
        start: Tuple[float, float],
        end: Tuple[float, float],
        samples: int,
    ) -> List[Tuple[float, float]]:

        points = []

        for t in np.linspace(
            0.0,
            1.0,
            samples,
        ):

            x = (
                start[0]
                + t
                * (
                    end[0]
                    - start[0]
                )
            )

            y = (
                start[1]
                + t
                * (
                    end[1]
                    - start[1]
                )
            )

            points.append(
                (
                    float(x),
                    float(y),
                )
            )

        return points

    # ========================================================
    # BEZIER TURN ROUTE
    # ========================================================

    def _bezier_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        approach: str,
        maneuver: str,
        samples: int,
    ) -> List[Tuple[float, float]]:

        p0 = np.asarray(
            start,
            dtype=np.float64,
        )

        p3 = np.asarray(
            end,
            dtype=np.float64,
        )

        half = (
            self.config.intersection_half_width
        )

        control_distance = (
            self.config.approach_length
            * 0.35
        )

        incoming = {
            "north": np.array(
                [0.0, -1.0]
            ),
            "south": np.array(
                [0.0, 1.0]
            ),
            "east": np.array(
                [-1.0, 0.0]
            ),
            "west": np.array(
                [1.0, 0.0]
            ),
        }[approach]

        outgoing_approach = (
            self._exit_approach(
                approach,
                maneuver,
            )
        )

        outgoing = {
            "north": np.array(
                [0.0, 1.0]
            ),
            "south": np.array(
                [0.0, -1.0]
            ),
            "east": np.array(
                [1.0, 0.0]
            ),
            "west": np.array(
                [-1.0, 0.0]
            ),
        }[outgoing_approach]

        turn_extent = half + 4.0
        turn_start = p0 + incoming * (
            self.config.approach_length - turn_extent
        )
        turn_end = p3 - outgoing * (
            self.config.exit_length - turn_extent
        )
        turn_control = min(half * 0.55, control_distance)
        p1 = turn_start + incoming * turn_control
        p2 = turn_end - outgoing * turn_control

        straight_samples = max(2, samples // 3)
        turn_samples = max(2, samples - 2 * straight_samples + 2)
        points = []

        for t in np.linspace(0.0, 1.0, straight_samples):
            point = p0 + t * (turn_start - p0)
            points.append((float(point[0]), float(point[1])))

        for t in np.linspace(0.0, 1.0, turn_samples)[1:]:
            one_minus = 1.0 - t
            point = (
                one_minus**3 * turn_start
                + 3.0 * one_minus**2 * t * p1
                + 3.0 * one_minus * t**2 * p2
                + t**3 * turn_end
            )
            points.append((float(point[0]), float(point[1])))

        for t in np.linspace(0.0, 1.0, straight_samples)[1:]:
            point = turn_end + t * (p3 - turn_end)
            points.append((float(point[0]), float(point[1])))

        return points

    # ========================================================
    # ROUTE UTILITIES
    # ========================================================

    @staticmethod
    def _route_length(
        route: List[Tuple[float, float]],
    ) -> float:

        total = 0.0

        for i in range(
            len(route) - 1
        ):

            p1 = np.asarray(
                route[i]
            )

            p2 = np.asarray(
                route[i + 1]
            )

            total += float(
                np.linalg.norm(
                    p2 - p1
                )
            )

        return total

    @staticmethod
    def _route_heading(
        route: List[Tuple[float, float]],
        index: int,
    ) -> float:

        if len(route) < 2:
            return 0.0

        index = min(
            index,
            len(route) - 2,
        )

        p1 = np.asarray(
            route[index]
        )

        p2 = np.asarray(
            route[index + 1]
        )

        return math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0],
        )

    # ========================================================
    # BACKGROUND TRAFFIC
    # ========================================================

    def _create_background_traffic(
        self,
    ) -> None:

        for i in range(
            self.config.background_vehicle_count
        ):

            approach = self.rng.choice(
                APPROACHES
            )

            lane = self.rng.randrange(
                self.config.lanes_per_approach
            )

            maneuver = self.rng.choice(
                self._legal_maneuvers(
                    lane
                )
            )

            route = self._build_route(
                approach,
                maneuver,
                lane,
            )

            # Put traffic at different
            # points along its route.

            route_length = (
                self._route_length(
                    route
                )
            )

            progress = self.rng.uniform(
                0.0,
                min(
                    80.0,
                    route_length * 0.65,
                ),
            )

            x, y = self._point_at_progress(
                route,
                progress,
            )

            heading = self._heading_at_progress(
                route,
                progress,
            )

            vehicle = Vehicle(
                vehicle_id=f"traffic_{i}",
                behavior=None,
                approach=approach,
                maneuver=maneuver,
                lane=lane,
                x=x,
                y=y,
                speed=self.rng.uniform(
                    self.config.background_min_speed,
                    self.config.background_max_speed,
                ),
                acceleration=0.0,
                heading=heading,
                progress=progress,
                route_points=route,
                route_distance=route_length,
                is_learning=False,
            )

            self.background_vehicles.append(
                vehicle
            )

    def _update_background_traffic(
        self,
    ) -> None:

        for vehicle in self.background_vehicles:

            if vehicle.completed:
                continue

            acceleration = (
                self._background_acceleration(
                    vehicle
                )
            )

            vehicle.update(
                acceleration,
                self.config.dt,
                self.config.background_max_speed,
            )

    def _background_acceleration(
        self,
        vehicle: Vehicle,
    ) -> float:

        # Background traffic keeps a stable cruise speed and yields at
        # imminent route conflicts with learning vehicles.
        target = (
            self.config.background_min_speed
            + self.config.background_max_speed
        ) / 2.0

        if self._is_in_intersection(vehicle):
            if vehicle.speed < target - 0.5:
                return 1.0
            return 0.0

        for agent in self.vehicles.values():
            before_intersection = min(
                agent.progress,
                vehicle.progress,
            ) < self.config.approach_length
            if (
                agent.approach == vehicle.approach
                and agent.lane == vehicle.lane
                and before_intersection
            ):
                if (
                    agent.progress < vehicle.progress
                    and vehicle.progress - agent.progress < 30.0
                    and agent.speed > vehicle.speed + 0.1
                ):
                    target = self.config.background_max_speed
                elif (
                    vehicle.progress < agent.progress
                    and agent.progress - vehicle.progress < 30.0
                    and vehicle.speed > agent.speed + 0.1
                ):
                    return self.config.yield_acceleration

            dx = agent.x - vehicle.x
            dy = agent.y - vehicle.y
            longitudinal_gap = (
                dx * math.cos(vehicle.heading)
                + dy * math.sin(vehicle.heading)
            )
            lateral_gap = abs(
                -dx * math.sin(vehicle.heading)
                + dy * math.cos(vehicle.heading)
            )
            same_direction = (
                math.cos(vehicle.heading - agent.heading) > 0.5
            )

            if (
                same_direction
                and lateral_gap < self.config.lane_width
                and 0.0 < longitudinal_gap < 30.0
                and vehicle.speed > agent.speed + 0.1
            ):
                return self.config.yield_acceleration
            if (
                same_direction
                and lateral_gap < self.config.lane_width
                and -30.0 < longitudinal_gap < 0.0
                and agent.speed > vehicle.speed + 0.1
            ):
                target = self.config.background_max_speed

            traffic_ttc, agent_ttc = self._time_to_conflict(
                vehicle,
                agent,
            )
            if (
                traffic_ttc < 16.0
                and agent_ttc < 16.0
                and abs(traffic_ttc - agent_ttc) < 5.0
            ):
                return self.config.yield_acceleration

        if vehicle.speed < target - 0.5:
            return 1.0

        if vehicle.speed > target + 0.5:
            return -0.8

        return 0.0

    # ========================================================
    # POINT / HEADING AT PROGRESS
    # ========================================================

    @staticmethod
    def _point_at_progress(
        route: List[Tuple[float, float]],
        progress: float,
    ) -> Tuple[float, float]:
        points = np.asarray(route, dtype=np.float64)
        if progress <= 0.0:
            return float(points[0, 0]), float(points[0, 1])
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        if progress >= cumulative[-1]:
            return float(points[-1, 0]), float(points[-1, 1])
        i = bisect_left(cumulative.tolist(), progress) - 1
        i = max(0, min(i, len(segments) - 1))
        ratio = (progress - cumulative[i]) / max(segments[i], 1e-8)
        point = points[i] + ratio * (points[i + 1] - points[i])
        return float(point[0]), float(point[1])

    @staticmethod
    def _heading_at_progress(
        route: List[Tuple[float, float]],
        progress: float,
    ) -> float:
        points = np.asarray(route, dtype=np.float64)
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segments)))
        if progress >= cumulative[-1]:
            i = len(segments) - 1
        else:
            i = bisect_left(cumulative.tolist(), max(progress, 0.0)) - 1
            i = max(0, min(i, len(segments) - 1))
        p1 = points[i]
        p2 = points[i + 1]
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    # ========================================================
    # ACTION -> ACCELERATION
    # ========================================================

    def _action_to_acceleration(
        self,
        action: int,
        vehicle: Vehicle,
    ) -> float:

        if action == BRAKE:
            return self.config.max_braking

        if action == YIELD:
            return self.config.yield_acceleration

        if action == MAINTAIN:
            return 0.0

        if action == ACCELERATE:
            return self.config.max_acceleration

        raise ValueError(
            f"Unknown action: {action}"
        )

    # ========================================================
    # COLLISION DETECTION
    # ========================================================

    def _check_collisions(
        self,
    ) -> None:

        all_learning = [
            vehicle
            for vehicle in self.vehicles.values()
            if not vehicle.completed
        ]

        # ----------------------------------------------------
        # Learning ↔ learning
        # ----------------------------------------------------

        for i in range(
            len(all_learning)
        ):

            for j in range(
                i + 1,
                len(all_learning),
            ):

                a = all_learning[i]
                b = all_learning[j]

                distance = self._distance(
                    a,
                    b,
                )

                if (
                    distance
                    <= self.config.collision_distance
                ):
                    pair = tuple(sorted((a.vehicle_id, b.vehicle_id)))
                    if pair not in self._agent_collision_pairs:
                        self._agent_collision_pairs.add(pair)
                        self.total_collisions += 1
                        self.agent_collisions += 1
                        a.collision_event = True
                        b.collision_event = True
                    a.collided = True
                    b.collided = True
                    a.agent_collided = True
                    b.agent_collided = True

        # ----------------------------------------------------
        # Learning ↔ traffic
        # ----------------------------------------------------

        for agent in all_learning:

            for traffic in self.background_vehicles:
                if traffic.completed:
                    continue

                distance = self._distance(
                    agent,
                    traffic,
                )

                if (
                    distance
                    <= self.config.collision_distance
                ):
                    pair = (agent.vehicle_id, traffic.vehicle_id)
                    if pair not in self._traffic_collision_pairs:
                        self._traffic_collision_pairs.add(pair)
                        self.total_collisions += 1
                        self.traffic_collisions += 1
                        agent.collision_event = True
                    agent.collided = True
                    agent.traffic_collided = True
                    traffic.collided = True

    # ========================================================
    # SAFETY METRICS
    # ========================================================

    def _update_safety_metrics(
        self,
    ) -> None:

        learning_vehicles = [
            vehicle
            for vehicle in self.vehicles.values()
            if not vehicle.completed
        ]
        background_vehicles = [
            vehicle
            for vehicle in self.background_vehicles
            if not vehicle.completed
        ]

        # Safety metrics describe the learning task.  Background-vs-background
        # encounters are not counted as near-collisions or TTC events.
        pairs = []
        for i, a in enumerate(learning_vehicles):
            for b in learning_vehicles[i + 1:]:
                pairs.append((a, b))
            for b in background_vehicles:
                pairs.append((a, b))

        for a, b in pairs:

                distance = self._distance(
                    a,
                    b,
                )

                self.minimum_distance = min(
                    self.minimum_distance,
                    distance,
                )

                if (
                    distance
                    < self.config.near_collision_distance
                ):

                    self.near_collisions += 1

                ttc = self._approximate_ttc(
                    a,
                    b,
                )

                if np.isfinite(ttc):

                    self.minimum_ttc = min(
                        self.minimum_ttc,
                        ttc,
                    )

    # ========================================================
    # DISTANCE
    # ========================================================

    @staticmethod
    def _distance(
        a: Vehicle,
        b: Vehicle,
    ) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return math.hypot(dx, dy)

    # ========================================================
    # APPROXIMATE TTC
    # ========================================================

    def _approximate_ttc(
        self,
        a: Vehicle,
        b: Vehicle,
    ) -> float:

        conflict_a, conflict_b = self._get_conflict_distances(a, b)
        if conflict_a is not None and conflict_b is not None:
            remaining_a = conflict_a - a.progress
            remaining_b = conflict_b - b.progress
            if remaining_a < -a.length or remaining_b < -b.length:
                return float("inf")

            time_a = max(remaining_a, 0.0) / max(a.speed, 0.5)
            time_b = max(remaining_b, 0.0) / max(b.speed, 0.5)
            if abs(time_a - time_b) <= self.config.dangerous_ttc:
                return min(time_a, time_b)
            return float("inf")

        dx = b.x - a.x
        dy = b.y - a.y
        distance = math.hypot(dx, dy)

        if distance <= 0.01:
            return 0.0

        inv_distance = 1.0 / distance
        dir_x = dx * inv_distance
        dir_y = dy * inv_distance

        va_x = a.speed * math.cos(a.heading)
        va_y = a.speed * math.sin(a.heading)
        vb_x = b.speed * math.cos(b.heading)
        vb_y = b.speed * math.sin(b.heading)

        closing_speed = (va_x - vb_x) * dir_x + (va_y - vb_y) * dir_y

        if closing_speed <= 0.1:
            return float("inf")

        return (
            distance
            / closing_speed
        )

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    def _route_pair_key(self, a: Vehicle, b: Vehicle):
        return (
            a.approach, a.maneuver, a.lane,
            b.approach, b.maneuver, b.lane,
        )

    def _get_conflict_distances(
        self,
        a: Vehicle,
        b: Vehicle,
    ) -> Tuple[Optional[float], Optional[float]]:
        key = self._route_pair_key(a, b)
        cached = self._conflict_cache.get(key)
        if cached is not None:
            return cached

        if a.approach == b.approach and a.lane == b.lane:
            result = (None, None)
            self._conflict_cache[key] = result
            return result

        # Preserve the original geometry rule exactly: every ego
        # route segment start is checked against every 10th point
        # on the other route. Vectorization removes the Python nested
        # loops while keeping the same first-hit and midpoint logic.
        route_a = a.route_points_np
        route_b = b.route_points_np
        sample_b = route_b[::10]
        sample_a = route_a[::10]

        d2_a = np.sum(
            (route_a[:-1, None, :] - sample_b[None, :, :]) ** 2,
            axis=2,
        )
        hit_rows_a = np.flatnonzero(np.any(d2_a < 9.0, axis=1))

        d2_b = np.sum(
            (route_b[:-1, None, :] - sample_a[None, :, :]) ** 2,
            axis=2,
        )
        hit_rows_b = np.flatnonzero(np.any(d2_b < 9.0, axis=1))

        if hit_rows_a.size == 0 or hit_rows_b.size == 0:
            result = (None, None)
        else:
            ia = int(hit_rows_a[0])
            ib = int(hit_rows_b[0])
            da = (
                a.route_cumulative_lengths[ia]
                + a.route_segment_lengths[ia] / 2.0
            )
            db = (
                b.route_cumulative_lengths[ib]
                + b.route_segment_lengths[ib] / 2.0
            )
            result = (float(da), float(db))

        self._conflict_cache[key] = result
        return result

    def _routes_conflict(
        self,
        a: Vehicle,
        b: Vehicle,
    ) -> bool:
        da, db = self._get_conflict_distances(a, b)
        return da is not None and db is not None

    def _time_to_conflict(
        self,
        ego: Vehicle,
        other: Vehicle,
    ) -> Tuple[float, float]:
        ego_distance, other_distance = self._get_conflict_distances(ego, other)

        if ego_distance is None or other_distance is None:
            return float("inf"), float("inf")

        ego_remaining = max(ego_distance - ego.progress, 0.0)
        other_remaining = max(other_distance - other.progress, 0.0)

        ego_time = ego_remaining / max(ego.speed, 0.5)
        other_time = other_remaining / max(other.speed, 0.5)

        return ego_time, other_time

    def _find_conflict_distance(
        self,
        ego: Vehicle,
        other: Vehicle,
    ) -> Optional[float]:
        ego_distance, _ = self._get_conflict_distances(ego, other)
        return ego_distance

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(
        self,
        agent_id: str,
    ) -> np.ndarray:

        ego = self.vehicles[
            agent_id
        ]

        values = []

        # ----------------------------------------------------
        # SELF
        # ----------------------------------------------------

        values.extend(
            [
                ego.x / 120.0,
                ego.y / 120.0,
                ego.speed
                / self.config.max_speed,
                ego.acceleration
                / abs(
                    self.config.max_braking
                ),
                math.sin(
                    ego.heading
                ),
                math.cos(
                    ego.heading
                ),
                ego.progress
                / max(
                    ego.route_distance,
                    1.0,
                ),
                self._distance_to_stop_line(
                    ego
                )
                / 120.0,
            ]
        )

        # ----------------------------------------------------
        # Maneuver one-hot
        # ----------------------------------------------------

        values.extend(
            [
                float(
                    ego.maneuver
                    == "straight"
                ),
                float(
                    ego.maneuver
                    == "left"
                ),
                float(
                    ego.maneuver
                    == "right"
                ),
            ]
        )

        # ----------------------------------------------------
        # Other learning agents
        # ----------------------------------------------------

        for other_id in self.agents:

            if other_id == agent_id:
                continue

            other = self.vehicles[
                other_id
            ]

            dx = (
                other.x
                - ego.x
            )

            dy = (
                other.y
                - ego.y
            )

            distance = math.sqrt(
                dx * dx
                + dy * dy
            )

            relative_speed = (
                other.speed
                - ego.speed
            )

            ego_ttc, other_ttc = (
                self._time_to_conflict(
                    ego,
                    other,
                )
            )

            if np.isfinite(
                ego_ttc
            ) and np.isfinite(
                other_ttc
            ):

                arrival_gap = abs(
                    ego_ttc
                    - other_ttc
                )

                conflict = 1.0

            else:

                arrival_gap = 0.0
                conflict = 0.0

            values.extend(
                [
                    dx / 120.0,
                    dy / 120.0,
                    relative_speed
                    / self.config.max_speed,
                    distance / 120.0,
                    self._normalize_ttc(
                        ego_ttc
                    ),
                    self._normalize_ttc(
                        other_ttc
                    ),
                    self._normalize_time_gap(
                        arrival_gap
                    ),
                    conflict,
                ]
            )

        # ----------------------------------------------------
        # Nearest background traffic
        # ----------------------------------------------------

        traffic = sorted(
            [
                vehicle
                for vehicle in self.background_vehicles
                if not vehicle.completed
            ],
            key=lambda v: self._distance(
                ego,
                v,
            ),
        )

        # Fixed-size observation.
        for i in range(4):

            if i < len(traffic):

                other = traffic[i]

                dx = (
                    other.x
                    - ego.x
                )

                dy = (
                    other.y
                    - ego.y
                )

                distance = self._distance(
                    ego,
                    other,
                )

                ttc = (
                    self._approximate_ttc(
                        ego,
                        other,
                    )
                )

                values.extend(
                    [
                        dx / 120.0,
                        dy / 120.0,
                        (
                            other.speed
                            - ego.speed
                        )
                        / self.config.max_speed,
                        distance / 120.0,
                        self._normalize_ttc(
                            ttc
                        ),
                    ]
                )

            else:

                values.extend(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ]
                )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ========================================================
    # OBSERVATION HELPERS
    # ========================================================

    def _calculate_observation_dimension(
        self,
    ) -> int:

        # Self:
        # 8 physical + 3 maneuver
        self_dim = 11

        # Two other learning agents:
        # 8 each
        learning_dim = 2 * 8

        # Four background vehicles:
        # 5 each
        traffic_dim = 4 * 5

        return (
            self_dim
            + learning_dim
            + traffic_dim
        )

    def _normalize_ttc(
        self,
        ttc: float,
    ) -> float:

        if not np.isfinite(ttc):
            return 1.0

        return float(
            np.clip(
                ttc / 10.0,
                0.0,
                1.0,
            )
        )

    def _normalize_time_gap(
        self,
        gap: float,
    ) -> float:

        if not np.isfinite(gap):
            return 1.0

        return float(
            np.clip(
                gap / 10.0,
                0.0,
                1.0,
            )
        )

    def _distance_to_stop_line(
        self,
        vehicle: Vehicle,
    ) -> float:

        return max(
            self._distance_to_intersection_entry(
                vehicle
            ),
            0.0,
        )

    def _distance_to_intersection_entry(
        self,
        vehicle: Vehicle,
    ) -> float:

        remaining = (
            vehicle.route_distance
            - vehicle.progress
        )

        return min(
            remaining,
            self.config.approach_length,
        )

    # ========================================================
    # REWARD
    # ========================================================

    def _calculate_reward(
        self,
        agent_id: str,
        action: int,
    ) -> float:

        vehicle = self.vehicles[
            agent_id
        ]

        reward = 0.0

        # Pay only for distance gained this step. Cumulative progress
        # rewards repeatedly pay for the same distance and favor stalling.
        reward += (
            vehicle.progress_delta
            * self.config.progress_scale
        )

        # ----------------------------------------------------
        # Small time cost
        # ----------------------------------------------------

        reward -= self.config.time_penalty

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if vehicle.collision_event:
            reward -= (
                self.config.collision_penalty
            )

        # ----------------------------------------------------
        # Completion
        # ----------------------------------------------------

        if vehicle.completed and not vehicle.completion_reward_given:
            reward += self.config.completion_reward
            vehicle.completion_reward_given = True

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        safety_risk = (
            self._agent_safety_risk(
                vehicle
            )
        )

        reward -= (
            self.config.unsafe_penalty
            * safety_risk
        )

        return float(
            reward
            * self.config.reward_scale
        )

    # ========================================================
    # AGENT SAFETY RISK
    # ========================================================

    def _agent_safety_risk(
        self,
        ego: Vehicle,
    ) -> float:

        risk = 0.0

        all_other = (
            [
                v
                for v in self.vehicles.values()
                if v.vehicle_id
                != ego.vehicle_id
                and not v.completed
            ]
            + [
                vehicle
                for vehicle in self.background_vehicles
                if not vehicle.completed
            ]
        )

        for other in all_other:

            distance = self._distance(
                ego,
                other,
            )

            if distance < 20.0:

                proximity = np.clip(
                    1.0
                    - distance / 20.0,
                    0.0,
                    1.0,
                )

                risk = max(
                    risk,
                    float(proximity),
                )

            ttc = (
                self._approximate_ttc(
                    ego,
                    other,
                )
            )

            if np.isfinite(ttc):

                ttc_risk = np.clip(
                    1.0
                    - (
                        ttc
                        / self.config.dangerous_ttc
                    ),
                    0.0,
                    1.0,
                )

                risk = max(
                    risk,
                    float(ttc_risk),
                )

        return risk

    # ========================================================
    # INFO
    # ========================================================

    def _get_agent_info(
        self,
        agent_id: str,
        action: int,
    ) -> dict:

        vehicle = self.vehicles[
            agent_id
        ]

        return {
            "agent_id": agent_id,
            "behavior": vehicle.behavior,
            "checkpoint": SELECTED_MODELS[
                agent_id
            ]["checkpoint"],
            "approach": vehicle.approach,
            "maneuver": vehicle.maneuver,
            "lane": vehicle.lane,
            "speed": float(
                vehicle.speed
            ),
            "acceleration": float(
                vehicle.acceleration
            ),
            "progress": float(
                vehicle.progress
            ),
            "completed": bool(
                vehicle.completed
            ),
            "collided": bool(
                vehicle.collided
            ),
            "agent_collided": bool(
                vehicle.agent_collided
            ),
            "traffic_collided": bool(
                vehicle.traffic_collided
            ),
            "action": int(action),
            "action_name": ACTION_NAMES[
                int(action)
            ],
        }

    def _get_global_info(
        self,
    ) -> dict:

        return {
            "step": self.step_count,
            "total_collisions": (
                self.total_collisions
            ),
            "agent_involved_collisions": (
                self.total_collisions
            ),
            "agent_collisions": (
                self.agent_collisions
            ),
            "agent_agent_collisions": (
                self.agent_collisions
            ),
            "traffic_collisions": (
                self.traffic_collisions
            ),
            "agent_traffic_collisions": (
                self.traffic_collisions
            ),
            "near_collisions": (
                self.near_collisions
            ),
            "minimum_distance": (
                self.minimum_distance
            ),
            "minimum_ttc": (
                self.minimum_ttc
            ),
            "all_completed": all(
                v.completed
                for v in self.vehicles.values()
            ),
        }


# ============================================================
# SIMPLE MANUAL TEST
# ============================================================

if __name__ == "__main__":

    env = Stage2IntersectionEnv(
        seed=42,
        background_traffic=True,
    )

    observations = env.reset()

    print("=" * 70)
    print("STAGE 2 INTERSECTION ENVIRONMENT")
    print("=" * 70)

    print(
        f"Observation dimension: "
        f"{env.observation_dim}"
    )

    print(
        f"Action dimension: "
        f"{env.action_dim}"
    )

    print("\nLearning agents:")

    for agent_id in env.agents:

        vehicle = env.vehicles[
            agent_id
        ]

        print(
            f"  {agent_id}: "
            f"{vehicle.behavior:12s} "
            f"{SELECTED_MODELS[agent_id]['checkpoint']}"
        )

    print(
        f"\nBackground vehicles: "
        f"{len(env.background_vehicles)}"
    )

    # One simultaneous action.
    actions = {
        "agent_0": MAINTAIN,
        "agent_1": MAINTAIN,
        "agent_2": MAINTAIN,
    }

    (
        observations,
        rewards,
        terminated,
        truncated,
        infos,
    ) = env.step(actions)

    print("\nAfter one simultaneous step:")

    for agent_id in env.agents:

        info = infos[
            agent_id
        ]

        print(
            f"  {agent_id}: "
            f"speed={info['speed']:.2f} "
            f"progress={info['progress']:.2f} "
            f"action={info['action_name']}"
        )

    print("\nEnvironment test complete.")