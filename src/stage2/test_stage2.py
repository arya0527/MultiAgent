# ============================================================
# STAGE 2 ENVIRONMENT TESTS
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from intersection_env import (
    ACCELERATE,
    BRAKE,
    MAINTAIN,
    YIELD,
    ACTION_NAMES,
    APPROACHES,
    SELECTED_MODELS,
    Stage2IntersectionEnv,
    Stage2Config,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_pass(name: str) -> None:
    print(f"{name}: PASS")


def initial_collision_count(
    env: Stage2IntersectionEnv,
) -> int:
    vehicles = (
        list(env.vehicles.values())
        + list(env.background_vehicles)
    )

    collisions = 0

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if (
                env._distance(
                    vehicles[i],
                    vehicles[j],
                )
                <= env.config.collision_distance
            ):
                collisions += 1

    return collisions


def main() -> None:
    total = 0
    passed = 0

    def run_test(name: str, function) -> None:
        nonlocal total, passed

        total += 1
        function()
        passed += 1
        print_pass(name)

    # --------------------------------------------------------
    # 1. Environment creation
    # --------------------------------------------------------
    def test_creation():
        env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=True,
        )

        check(
            len(env.agents) == 3,
            "Expected exactly 3 learning agents.",
        )

        check(
            env.observation_dim == 47,
            "Expected observation dimension 47.",
        )

        check(
            env.action_dim == 4,
            "Expected action dimension 4.",
        )

    run_test("Environment creation", test_creation)

    # --------------------------------------------------------
    # 2. Reset
    # --------------------------------------------------------
    env = Stage2IntersectionEnv(
        seed=42,
        background_traffic=True,
    )

    observations = env.reset()

    def test_reset():
        check(
            isinstance(observations, dict),
            "Reset must return observation dict.",
        )

        check(
            set(observations.keys()) == set(env.agents),
            "Reset agent IDs mismatch.",
        )

        for agent_id in env.agents:
            check(
                observations[agent_id].shape == (47,),
                f"{agent_id} observation shape incorrect.",
            )

    run_test("Reset", test_reset)

    # --------------------------------------------------------
    # 3. Agent identities
    # --------------------------------------------------------
    def test_identities():
        expected = {
            "agent_0": "aggressive",
            "agent_1": "neutral",
            "agent_2": "conservative",
        }

        for agent_id, behavior in expected.items():
            check(
                SELECTED_MODELS[agent_id]["behavior"] == behavior,
                f"{agent_id} behavior mismatch.",
            )

    run_test(
        "Agent identities/behaviors",
        test_identities,
    )

    # --------------------------------------------------------
    # 4. Selected Stage-1 checkpoint references
    # --------------------------------------------------------
    def test_checkpoints():
        expected = {
            "agent_0": "aggressive_seed45.pt",
            "agent_1": "neutral_seed46.pt",
            "agent_2": "conservative_seed50.pt",
        }

        for agent_id, checkpoint in expected.items():
            check(
                SELECTED_MODELS[agent_id]["checkpoint"]
                == checkpoint,
                f"{agent_id} checkpoint mismatch.",
            )

    run_test(
        "Selected Stage-1 checkpoints",
        test_checkpoints,
    )

    # --------------------------------------------------------
    # 5. Action space
    # --------------------------------------------------------
    def test_actions():
        check(
            ACTION_NAMES[BRAKE] == "brake",
            "Brake action incorrect.",
        )
        check(
            ACTION_NAMES[YIELD] == "yield",
            "Yield action incorrect.",
        )
        check(
            ACTION_NAMES[MAINTAIN] == "maintain",
            "Maintain action incorrect.",
        )
        check(
            ACTION_NAMES[ACCELERATE] == "accelerate",
            "Accelerate action incorrect.",
        )

    run_test("Action space", test_actions)

    # --------------------------------------------------------
    # 6. Observation structure
    # --------------------------------------------------------
    def test_observations():
        for agent_id in env.agents:
            observation = observations[agent_id]

            check(
                isinstance(observation, np.ndarray),
                "Observation must be numpy array.",
            )

            check(
                observation.shape == (47,),
                "Observation must be 47-D.",
            )

            check(
                np.all(np.isfinite(observation)),
                "Observation contains NaN/Inf.",
            )

    run_test(
        "Observation structure",
        test_observations,
    )

    # --------------------------------------------------------
    # 7. Background traffic
    # --------------------------------------------------------
    def test_background():
        check(
            len(env.background_vehicles) == 8,
            "Expected 8 background vehicles.",
        )

    run_test(
        "Background traffic",
        test_background,
    )

    # --------------------------------------------------------
    # 8. Background traffic disabled
    # --------------------------------------------------------
    env_no_traffic = Stage2IntersectionEnv(
        seed=42,
        background_traffic=False,
    )

    env_no_traffic.reset()

    def test_no_background():
        check(
            len(env_no_traffic.background_vehicles) == 0,
            "Background traffic should be disabled.",
        )

    run_test(
        "Background traffic disabled",
        test_no_background,
    )

    # --------------------------------------------------------
    # 9. Vehicle geometry
    # --------------------------------------------------------
    def test_geometry():
        for agent_id in env.agents:
            vehicle = env.vehicles[agent_id]

            check(
                vehicle.length == 4.5,
                "Vehicle length incorrect.",
            )

            check(
                vehicle.width == 1.8,
                "Vehicle width incorrect.",
            )

    run_test(
        "Vehicle geometry",
        test_geometry,
    )

    # --------------------------------------------------------
    # 10. Route generation
    # --------------------------------------------------------
    def test_routes():
        road_half_width = (
            env.config.lane_width
            * env.config.lanes_per_approach
        )
        intersection_half_width = env.config.intersection_half_width
        for approach in APPROACHES:
            for lane in range(2):
                for maneuver in env._legal_maneuvers(lane):
                    route = env._build_route(
                        approach,
                        maneuver,
                        lane,
                    )

                    check(
                        len(route) > 10,
                        "Route is too short.",
                    )

                    check(
                        np.all(
                            np.isfinite(
                                np.asarray(route)
                            )
                        ),
                        "Route contains invalid values.",
                    )

                    for x, y in route:
                        on_intersection = (
                            abs(x) <= intersection_half_width
                            and abs(y) <= intersection_half_width
                        )
                        on_vertical_road = (
                            abs(x) <= road_half_width
                            and abs(y) <= (
                                env.config.approach_length
                                + intersection_half_width
                            )
                        )
                        on_horizontal_road = (
                            abs(y) <= road_half_width
                            and abs(x) <= (
                                env.config.approach_length
                                + intersection_half_width
                            )
                        )
                        check(
                            on_intersection
                            or on_vertical_road
                            or on_horizontal_road,
                            (
                                f"{approach}/{lane}/{maneuver} leaves "
                                f"the road at ({x:.2f}, {y:.2f})."
                            ),
                        )

    run_test(
        "Routes remain on road",
        test_routes,
    )

    # --------------------------------------------------------
    # 11. All legal maneuvers
    # --------------------------------------------------------
    def test_maneuvers():
        count = 0

        for approach in APPROACHES:
            for lane in range(2):
                for maneuver in env._legal_maneuvers(lane):
                    route = env._build_route(
                        approach,
                        maneuver,
                        lane,
                    )

                    check(
                        len(route) > 0,
                        "Missing route.",
                    )

                    count += 1

        check(
            count == 16,
            f"Expected 16 route combinations, got {count}.",
        )

    run_test(
        "All legal maneuvers",
        test_maneuvers,
    )

    # --------------------------------------------------------
    # 12. Simultaneous actions
    # --------------------------------------------------------
    env_step = Stage2IntersectionEnv(
        seed=42,
        background_traffic=True,
    )

    env_step.reset()

    actions = {
        "agent_0": MAINTAIN,
        "agent_1": BRAKE,
        "agent_2": ACCELERATE,
    }

    result = env_step.step(actions)

    def test_step():
        check(
            len(result) == 5,
            "Step must return 5 values.",
        )

        obs, rewards, terminated, truncated, infos = result

        check(
            set(obs.keys()) == set(env_step.agents),
            "Step observations mismatch.",
        )

        check(
            set(rewards.keys()) == set(env_step.agents),
            "Step rewards mismatch.",
        )

        check(
            isinstance(terminated, bool),
            "terminated must be bool.",
        )

        check(
            isinstance(truncated, bool),
            "truncated must be bool.",
        )

        check(
            "global" in infos,
            "Missing global info.",
        )

    run_test(
        "Simultaneous action stepping",
        test_step,
    )

    # --------------------------------------------------------
    # 13. Action validation
    # --------------------------------------------------------
    env_validation = Stage2IntersectionEnv(
        seed=42,
        background_traffic=False,
    )

    env_validation.reset()

    def test_action_validation():
        try:
            env_validation.step(
                {
                    "agent_0": 99,
                    "agent_1": MAINTAIN,
                    "agent_2": MAINTAIN,
                }
            )
        except ValueError:
            return

        raise AssertionError(
            "Invalid action was not rejected."
        )

    run_test(
        "Action validation",
        test_action_validation,
    )

    # --------------------------------------------------------
    # Predictive route-conflict safety filter
    # --------------------------------------------------------
    def test_predictive_safety_filter():
        safety_env = Stage2IntersectionEnv(
            seed=1,
            config=Stage2Config(background_vehicle_count=0),
            background_traffic=False,
        )
        safety_env.reset(
            scenario=[
                {
                    "agent_id": "agent_0",
                    "approach": "north",
                    "maneuver": "straight",
                    "lane": 0,
                    "speed": 5.0,
                },
                {
                    "agent_id": "agent_1",
                    "approach": "east",
                    "maneuver": "straight",
                    "lane": 0,
                    "speed": 5.0,
                },
                {
                    "agent_id": "agent_2",
                    "approach": "west",
                    "maneuver": "straight",
                    "lane": 0,
                    "speed": 5.0,
                },
            ]
        )
        first = safety_env.vehicles["agent_0"]
        second = safety_env.vehicles["agent_1"]
        first_conflict, second_conflict = (
            safety_env._get_conflict_distances(first, second)
        )
        check(
            first_conflict is not None and second_conflict is not None,
            "Expected the two routes to share a conflict point.",
        )
        first.progress = first_conflict - 10.0
        second.progress = second_conflict - 10.0
        first._update_position_from_route()
        second._update_position_from_route()

        ttc = safety_env._approximate_ttc(first, second)
        check(
            abs(ttc - 2.0) < 1e-6,
            f"Expected a 2-second crossing TTC, got {ttc}.",
        )
        probabilities = {
            agent_id: np.array([0.0, 0.0, 1.0, 0.0])
            for agent_id in safety_env.agents
        }
        filtered, reasons = safety_env.filter_actions(
            {agent_id: MAINTAIN for agent_id in safety_env.agents},
            probabilities,
        )
        check(
            filtered["agent_0"] == YIELD
            and filtered["agent_1"] == MAINTAIN,
            "The approaching vehicle did not yield to traffic in the intersection.",
        )
        check(
            "agent_1" in reasons["agent_0"]
            and (
                reasons["agent_0"].startswith("Yielding to")
                or reasons["agent_0"].startswith("Predicted conflict with")
            ),
            "The safety decision did not report its predicted conflict.",
        )

    run_test(
        "Predictive conflict arbitration and TTC",
        test_predictive_safety_filter,
    )

    # --------------------------------------------------------
    # 14. Initial state collision-free
    # --------------------------------------------------------
    def test_initial_state():
        env_initial = Stage2IntersectionEnv(
            seed=42,
            background_traffic=True,
        )

        env_initial.reset()

        check(
            initial_collision_count(env_initial) == 0,
            "Initial state contains a collision.",
        )

    run_test(
        "Initial state collision-free",
        test_initial_state,
    )

    # --------------------------------------------------------
    # 15. CRITICAL regression: seed 43
    # --------------------------------------------------------
    def test_seed_43():
        env_43 = Stage2IntersectionEnv(
            seed=43,
            background_traffic=True,
        )

        env_43.reset()

        check(
            initial_collision_count(env_43) == 0,
            "Seed 43 still starts with a collision.",
        )

        slots = [
            (
                env_43.vehicles[agent_id].approach,
                env_43.vehicles[agent_id].lane,
            )
            for agent_id in env_43.agents
        ]

        check(
            len(slots) == len(set(slots)),
            "Learning agents share a spawn slot.",
        )

    run_test(
        "Seed-43 regression",
        test_seed_43,
    )

    # --------------------------------------------------------
    # 16. Unique learning spawn slots over 100 seeds
    # --------------------------------------------------------
    def test_spawn_uniqueness():
        for seed in range(100):
            test_env = Stage2IntersectionEnv(
                seed=seed,
                background_traffic=True,
            )

            test_env.reset()

            slots = [
                (
                    test_env.vehicles[agent_id].approach,
                    test_env.vehicles[agent_id].lane,
                )
                for agent_id in test_env.agents
            ]

            check(
                len(slots) == len(set(slots)),
                f"Duplicate learning spawn slot at seed {seed}.",
            )

    run_test(
        "Unique learning spawn slots (100 seeds)",
        test_spawn_uniqueness,
    )

    # --------------------------------------------------------
    # 17. Collision-free initial states over 100 seeds
    # --------------------------------------------------------
    def test_collision_free_resets():
        for seed in range(100):
            test_env = Stage2IntersectionEnv(
                seed=seed,
                background_traffic=True,
            )

            test_env.reset()

            check(
                initial_collision_count(test_env) == 0,
                f"Initial collision at seed {seed}.",
            )

    run_test(
        "Collision-free resets (100 seeds)",
        test_collision_free_resets,
    )

    # --------------------------------------------------------
    # 18. One-step safety regression over 100 seeds
    # --------------------------------------------------------
    def test_one_step_many_seeds():
        for seed in range(100):
            test_env = Stage2IntersectionEnv(
                seed=seed,
                background_traffic=True,
            )

            test_env.reset()

            actions = {
                agent_id: MAINTAIN
                for agent_id in test_env.agents
            }

            (
                _,
                _,
                _terminated,
                _truncated,
                infos,
            ) = test_env.step(actions)

            check(
                infos["global"]["total_collisions"] == 0,
                (
                    "Collision occurred on first "
                    f"step at seed {seed}."
                ),
            )

    run_test(
        "One-step safety regression (100 seeds)",
        test_one_step_many_seeds,
    )

    # --------------------------------------------------------
    # 19. Reset reproducibility
    # --------------------------------------------------------
    def test_reproducibility():
        env_a = Stage2IntersectionEnv(
            seed=123,
            background_traffic=True,
        )

        obs_a = env_a.reset()

        state_a = [
            (
                agent_id,
                env_a.vehicles[agent_id].approach,
                env_a.vehicles[agent_id].maneuver,
                env_a.vehicles[agent_id].lane,
                env_a.vehicles[agent_id].speed,
            )
            for agent_id in env_a.agents
        ]

        env_b = Stage2IntersectionEnv(
            seed=123,
            background_traffic=True,
        )

        obs_b = env_b.reset()

        state_b = [
            (
                agent_id,
                env_b.vehicles[agent_id].approach,
                env_b.vehicles[agent_id].maneuver,
                env_b.vehicles[agent_id].lane,
                env_b.vehicles[agent_id].speed,
            )
            for agent_id in env_b.agents
        ]

        check(
            state_a == state_b,
            "Same seed produced different initial states.",
        )

        for agent_id in env_a.agents:
            check(
                np.allclose(
                    obs_a[agent_id],
                    obs_b[agent_id],
                ),
                f"Observation mismatch for {agent_id}.",
            )

    run_test(
        "Reset reproducibility",
        test_reproducibility,
    )

    # --------------------------------------------------------
    # 20. Controlled scenario
    # --------------------------------------------------------
    def test_controlled_scenario():
        controlled = [
            {
                "agent_id": "agent_0",
                "approach": "south",
                "maneuver": "straight",
                "lane": 1,
                "speed": 6.0,
            },
            {
                "agent_id": "agent_1",
                "approach": "east",
                "maneuver": "right",
                "lane": 0,
                "speed": 6.0,
            },
            {
                "agent_id": "agent_2",
                "approach": "west",
                "maneuver": "straight",
                "lane": 1,
                "speed": 6.0,
            },
        ]

        controlled_env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=False,
        )

        observations = controlled_env.reset(
            scenario=controlled
        )

        check(
            len(observations) == 3,
            "Controlled scenario reset failed.",
        )

        for item in controlled:
            agent_id = item["agent_id"]
            vehicle = controlled_env.vehicles[agent_id]

            check(
                vehicle.approach == item["approach"],
                "Controlled approach changed.",
            )

            check(
                vehicle.maneuver == item["maneuver"],
                "Controlled maneuver changed.",
            )

            check(
                vehicle.lane == item["lane"],
                "Controlled lane changed.",
            )

    run_test(
        "Controlled scenario",
        test_controlled_scenario,
    )

    # --------------------------------------------------------
    # 21. Agent-to-traffic collision is terminal and counted once
    # --------------------------------------------------------
    def test_agent_traffic_collision_terminates():
        traffic_env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=True,
        )
        observations = traffic_env.reset(seed=42)
        learner = traffic_env.vehicles["agent_0"]
        traffic = traffic_env.background_vehicles[0]
        traffic.x = learner.x
        traffic.y = learner.y
        traffic.speed = learner.speed
        traffic.heading = learner.heading
        traffic_env._check_collisions()
        global_info = traffic_env._get_global_info()

        _, _, terminated, truncated, infos = traffic_env.step(
            {
                agent_id: BRAKE
                for agent_id in traffic_env.agents
            }
        )

        global_info = infos["global"]
        check(
            global_info["agent_collisions"] == 0,
            "Unexpected learning-agent collision in forced traffic-contact case.",
        )
        check(
            global_info["agent_involved_collisions"] > 0
            and global_info["agent_traffic_collisions"] > 0,
            "Agent-to-traffic contact was not counted as agent-involved.",
        )
        check(
            terminated and not truncated,
            "Agent-to-traffic collision did not terminate the episode.",
        )

        collision_count = traffic_env.total_collisions
        traffic_env._check_collisions()
        check(
            traffic_env.total_collisions == collision_count,
            "A persistent traffic contact was counted more than once.",
        )

    run_test(
        "Agent-to-traffic collision terminates",
        test_agent_traffic_collision_terminates,
    )

    # --------------------------------------------------------
    # 22. Background-only collisions are not learning collisions
    # --------------------------------------------------------
    def test_background_only_collision_is_ignored():
        traffic_env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=True,
        )
        traffic_env.reset(seed=42)
        first, second = traffic_env.background_vehicles[:2]
        first.x = second.x = 500.0
        first.y = second.y = 500.0

        traffic_env._check_collisions()
        global_info = traffic_env._get_global_info()

        check(
            global_info["agent_involved_collisions"] == 0,
            "A background-to-background overlap was counted against learning agents.",
        )
        check(
            global_info["agent_traffic_collisions"] == 0,
            "A background-to-background overlap was classified as agent-to-traffic.",
        )

    run_test(
        "Background-only collision is ignored",
        test_background_only_collision_is_ignored,
    )

    # --------------------------------------------------------
    # 23. Joint completion reward is paid once to every agent
    # --------------------------------------------------------
    def test_joint_completion_reward():
        completion_env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=False,
        )
        completion_env.reset(seed=42)
        for vehicle in completion_env.vehicles.values():
            vehicle.completed = True
            vehicle.completion_reward_given = True

        _, rewards, terminated, truncated, _ = completion_env.step(
            {
                agent_id: MAINTAIN
                for agent_id in completion_env.agents
            }
        )

        check(
            terminated and not truncated,
            "Joint completion did not terminate the episode.",
        )
        for agent_id in completion_env.agents:
            check(
                rewards[agent_id] >= (
                    completion_env.config.team_completion_reward
                    * completion_env.config.reward_scale
                    - completion_env.config.time_penalty
                    * completion_env.config.reward_scale
                    - completion_env.config.unsafe_penalty
                    * completion_env.config.reward_scale
                ),
                f"Missing joint completion reward for {agent_id}.",
            )
        check(
            completion_env.team_completion_reward_given,
            "Joint completion reward was not marked as paid.",
        )

        completion_env.reset(seed=43)
        check(
            not completion_env.team_completion_reward_given,
            "Joint completion reward flag leaked across reset.",
        )

    run_test(
        "Joint completion reward",
        test_joint_completion_reward,
    )

    # --------------------------------------------------------
    # 24. Same-lane learning/traffic starts have safe headway
    # --------------------------------------------------------
    def test_same_lane_spawn_headway():
        for seed in range(100):
            headway_env = Stage2IntersectionEnv(
                seed=seed,
                background_traffic=True,
            )
            headway_env.reset(seed=seed)
            for learner in headway_env.vehicles.values():
                for traffic in headway_env.background_vehicles:
                    if (
                        learner.approach == traffic.approach
                        and learner.lane == traffic.lane
                    ):
                        check(
                            abs(learner.progress - traffic.progress)
                            >= headway_env.config.minimum_same_lane_spawn_gap,
                            f"Unsafe same-lane headway at seed {seed}.",
                        )

    run_test(
        "Same-lane traffic spawn headway (100 seeds)",
        test_same_lane_spawn_headway,
    )

    # --------------------------------------------------------
    # 25. Background traffic handles same-lane headway
    # --------------------------------------------------------
    def test_background_same_lane_headway():
        headway_env = Stage2IntersectionEnv(
            seed=42,
            background_traffic=True,
        )
        headway_env.reset(seed=42)
        learner = headway_env.vehicles["agent_1"]
        traffic = headway_env.background_vehicles[0]
        traffic.approach = learner.approach
        traffic.lane = learner.lane

        learner.progress = 10.0
        learner.speed = 7.0
        traffic.progress = 25.0
        traffic.speed = 5.0
        check(
            headway_env._background_acceleration(traffic)
            == 1.0,
            "Traffic did not accelerate to clear a faster learner behind it.",
        )

        learner.progress = 25.0
        learner.speed = 5.0
        traffic.progress = 10.0
        traffic.speed = 7.0
        check(
            headway_env._background_acceleration(traffic)
            == headway_env.config.yield_acceleration,
            "Traffic did not yield when closing on a learner ahead.",
        )

    run_test(
        "Background same-lane headway",
        test_background_same_lane_headway,
    )

    print()
    print("=" * 70)
    print(f"ALL TESTS PASSED: {passed}/{total}")
    print("=" * 70)


if __name__ == "__main__":
    main()
