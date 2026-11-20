import unittest
from types import SimpleNamespace

from src.stage2.intersection_env import (
    ACCELERATE,
    BRAKE,
    MAINTAIN,
    SELECTED_MODELS,
    YIELD,
)
from visualization.server import AGENT_COLORS, _keep_agents_moving


class ContinuousMovementTests(unittest.TestCase):
    def test_policy_stops_are_replaced_without_mutating_policy_actions(self):
        env = SimpleNamespace(
            behaviors={
                "agent_0": "aggressive",
                "agent_1": "neutral",
                "agent_2": "conservative",
            },
            vehicles={
                "agent_0": SimpleNamespace(
                    completed=False,
                    collided=False,
                    speed=6.0,
                ),
                "agent_1": SimpleNamespace(
                    completed=False,
                    collided=False,
                    speed=0.0,
                ),
                "agent_2": SimpleNamespace(
                    completed=False,
                    collided=False,
                    speed=3.0,
                ),
            }
        )
        policy_actions = {
            "agent_0": BRAKE,
            "agent_1": YIELD,
            "agent_2": BRAKE,
        }

        actions, overridden = _keep_agents_moving(env, policy_actions)

        self.assertEqual(
            actions,
            {
                "agent_0": ACCELERATE,
                "agent_1": ACCELERATE,
                "agent_2": MAINTAIN,
            },
        )
        self.assertEqual(overridden, set(policy_actions))
        self.assertEqual(
            policy_actions,
            {
                "agent_0": BRAKE,
                "agent_1": YIELD,
                "agent_2": BRAKE,
            },
        )

    def test_preserves_accelerate_and_completed_agent_actions(self):
        env = SimpleNamespace(
            behaviors={
                "agent_0": "aggressive",
                "agent_1": "neutral",
            },
            vehicles={
                "agent_0": SimpleNamespace(
                    completed=False,
                    collided=False,
                    speed=0.0,
                ),
                "agent_1": SimpleNamespace(
                    completed=True,
                    collided=False,
                    speed=0.0,
                ),
            }
        )
        policy_actions = {
            "agent_0": ACCELERATE,
            "agent_1": BRAKE,
        }

        actions, overridden = _keep_agents_moving(env, policy_actions)

        self.assertEqual(actions, policy_actions)
        self.assertFalse(overridden)

    def test_behavior_checkpoint_and_color_mapping(self):
        self.assertEqual(SELECTED_MODELS["agent_0"]["behavior"], "aggressive")
        self.assertEqual(SELECTED_MODELS["agent_1"]["behavior"], "neutral")
        self.assertEqual(SELECTED_MODELS["agent_2"]["behavior"], "conservative")
        self.assertEqual(AGENT_COLORS["agent_0"], "#FF4B4B")
        self.assertEqual(AGENT_COLORS["agent_1"], "#3B82F6")
        self.assertEqual(AGENT_COLORS["agent_2"], "#7BFF4B")


if __name__ == "__main__":
    unittest.main()
