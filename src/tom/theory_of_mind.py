# src/tom/theory_of_mind.py
"""
Theory of Mind (ToM) Module for Multi-Agent Intersection Navigation.

Each agent maintains a belief distribution over the behavioral types
(aggressive, neutral, conservative) of ALL other learning agents.

The belief is updated in real-time using a Bayesian update rule
conditioned on observed actions and the likelihood of those actions
under each behavioral profile.

The ToM module also computes:
  - Predicted next-action distribution per other agent
  - Inferred behavioral intent (estimated risk tolerance, speed preference)
  - Social value orientation score (how much other agents yield vs push)

These signals are injected as additional features into the MAPPO
observation for each agent, augmenting the ego-centric sensor data.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# BEHAVIORAL PRIORS
# ============================================================
# Prior probability of each behavior type (uniform).
BEHAVIOR_TYPES = ["aggressive", "neutral", "conservative"]
NUM_BEHAVIORS = len(BEHAVIOR_TYPES)

# Each behavior's likelihood of taking each action:
#   0 = Brake, 1 = Yield, 2 = Maintain, 3 = Accelerate
#
# These are soft hand-crafted likelihoods that encode
# the behavioral profiles already defined in the codebase.
# They are NOT trained; they are interpretable domain priors.
#
# Format: P(action | behavior), shape (4,)
_ACTION_LIKELIHOODS: Dict[str, np.ndarray] = {
    "aggressive":    np.array([0.05, 0.08, 0.27, 0.60], dtype=np.float32),
    "neutral":       np.array([0.15, 0.20, 0.40, 0.25], dtype=np.float32),
    "conservative":  np.array([0.35, 0.38, 0.22, 0.05], dtype=np.float32),
}

# Ensure rows sum to 1.0
for _btype in BEHAVIOR_TYPES:
    _ACTION_LIKELIHOODS[_btype] /= _ACTION_LIKELIHOODS[_btype].sum()

# ============================================================
# RISK-TOLERANCE ESTIMATES PER BEHAVIOR
# ============================================================
# Used by other agents to infer the expected risk acceptance
# of an identified agent, influencing their own planning.
BEHAVIOR_RISK_TOLERANCE: Dict[str, float] = {
    "aggressive":   0.75,
    "neutral":      0.45,
    "conservative": 0.20,
}

# Expected relative speed preference (fraction of max_speed)
BEHAVIOR_SPEED_PREFERENCE: Dict[str, float] = {
    "aggressive":   0.85,
    "neutral":      0.60,
    "conservative": 0.40,
}


# ============================================================
# AGENT BELIEF STATE
# ============================================================

class AgentBeliefState:
    """
    Tracks a single other agent's inferred behavior distribution
    from the perspective of an observer agent.
    """

    def __init__(
        self,
        agent_id: str,
        alpha: float = 0.85,   # memory decay (higher = slower forgetting)
        min_belief: float = 0.02,
    ) -> None:
        self.agent_id = agent_id
        self.alpha = alpha
        self.min_belief = min_belief

        # Belief distribution over behavior types (sum = 1.0)
        self.beliefs: np.ndarray = np.ones(
            NUM_BEHAVIORS, dtype=np.float32
        ) / NUM_BEHAVIORS

        # Rolling history of observed actions (action indices)
        self.action_history: List[int] = []
        self.max_history = 64

        # Cached dominant behavior
        self._dominant: Optional[str] = None
        self._dominant_confidence: float = 1.0 / NUM_BEHAVIORS

        # Social-value orientation: accumulated cooperation signal
        # Positive = tends to yield; negative = tends to push
        self._svo_accumulator: float = 0.0
        self._svo_count: int = 0

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self, observed_action: int) -> None:
        """
        Bayesian update given an observed action.

        new_belief[b] ∝ old_belief[b] * P(action | behavior_b)

        We apply exponential decay so recent observations matter more.
        """
        assert 0 <= observed_action <= 3, (
            f"Invalid action: {observed_action}"
        )

        self.action_history.append(observed_action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)

        # Likelihood vector: P(a | each_behavior)
        likelihoods = np.array(
            [
                _ACTION_LIKELIHOODS[b][observed_action]
                for b in BEHAVIOR_TYPES
            ],
            dtype=np.float32,
        )

        # Bayesian update with exponential memory discount
        new_beliefs = self.beliefs * likelihoods
        total = new_beliefs.sum()

        if total < 1e-9:
            new_beliefs = np.ones(NUM_BEHAVIORS, dtype=np.float32)

        new_beliefs /= new_beliefs.sum()

        # Blend with prior using memory decay
        self.beliefs = (
            self.alpha * self.beliefs
            + (1.0 - self.alpha) * new_beliefs
        )

        # Floor clipping to avoid zero-probabilities
        self.beliefs = np.maximum(self.beliefs, self.min_belief)
        self.beliefs /= self.beliefs.sum()

        # Update dominant cache
        idx = int(np.argmax(self.beliefs))
        self._dominant = BEHAVIOR_TYPES[idx]
        self._dominant_confidence = float(self.beliefs[idx])

        # Update social value orientation
        # Brake/Yield = cooperative (+), Accelerate = competitive (-)
        svo_map = {0: +1.0, 1: +0.5, 2: 0.0, 3: -0.5}
        self._svo_accumulator += svo_map[observed_action]
        self._svo_count += 1

    # --------------------------------------------------------
    # PROPERTIES
    # --------------------------------------------------------

    @property
    def dominant_behavior(self) -> str:
        if self._dominant is None:
            return "neutral"
        return self._dominant

    @property
    def dominant_confidence(self) -> float:
        return self._dominant_confidence

    @property
    def predicted_action_probs(self) -> np.ndarray:
        """
        Expected action probability distribution:
        P(a) = sum_b belief[b] * P(a | behavior_b)
        """
        probs = np.zeros(4, dtype=np.float32)
        for i, b in enumerate(BEHAVIOR_TYPES):
            probs += self.beliefs[i] * _ACTION_LIKELIHOODS[b]
        return probs

    @property
    def inferred_risk_tolerance(self) -> float:
        """Weighted average risk tolerance given current beliefs."""
        rt = sum(
            self.beliefs[i] * BEHAVIOR_RISK_TOLERANCE[b]
            for i, b in enumerate(BEHAVIOR_TYPES)
        )
        return float(rt)

    @property
    def inferred_speed_preference(self) -> float:
        """Weighted average speed preference given current beliefs."""
        sp = sum(
            self.beliefs[i] * BEHAVIOR_SPEED_PREFERENCE[b]
            for i, b in enumerate(BEHAVIOR_TYPES)
        )
        return float(sp)

    @property
    def social_value_orientation(self) -> float:
        """
        SVO score in [-1, +1].
        +1 = always cooperative (yields), -1 = always competitive.
        """
        if self._svo_count == 0:
            return 0.0
        raw = self._svo_accumulator / self._svo_count
        return float(np.clip(raw, -1.0, 1.0))

    def to_feature_vector(self) -> np.ndarray:
        """
        Return a fixed-size feature vector (length = 9) encoding:
          - beliefs over 3 behavior types  (3 floats)
          - predicted action distribution  (4 floats)
          - inferred risk tolerance        (1 float)
          - social value orientation       (1 float)
        Total: 9 floats
        """
        return np.concatenate([
            self.beliefs,                            # 3
            self.predicted_action_probs,             # 4
            [self.inferred_risk_tolerance],          # 1
            [self.social_value_orientation],         # 1
        ]).astype(np.float32)


# ============================================================
# PER-AGENT THEORY OF MIND MODULE
# ============================================================

class TheoryOfMindModule:
    """
    Each learning agent has one TheoryOfMindModule that tracks
    beliefs about all OTHER learning agents.

    The module produces a fixed-size ToM feature vector that is
    appended to the agent's base observation before passing to
    the actor/critic network.

    Usage:
        tom = TheoryOfMindModule(ego_agent_id="agent_0",
                                 other_agent_ids=["agent_1", "agent_2"])

        # Every step, call update with the actions taken by others:
        tom.update({"agent_1": observed_action_1,
                    "agent_2": observed_action_2})

        # Get augmented feature vector:
        vec = tom.get_feature_vector()   # shape (18,) for 2 others
    """

    def __init__(
        self,
        ego_agent_id: str,
        other_agent_ids: List[str],
        alpha: float = 0.85,
    ) -> None:
        self.ego_agent_id = ego_agent_id
        self.other_agent_ids = sorted(other_agent_ids)

        self.beliefs: Dict[str, AgentBeliefState] = {
            aid: AgentBeliefState(aid, alpha=alpha)
            for aid in self.other_agent_ids
        }

        # Per-agent feature size = 9
        self.feature_dim_per_agent = 9
        self.feature_dim = (
            len(self.other_agent_ids) * self.feature_dim_per_agent
        )

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self, observed_actions: Dict[str, int]) -> None:
        """
        Update beliefs given the actions taken by all other agents
        in the current step.

        observed_actions: {agent_id: action_int, ...}
        """
        for aid in self.other_agent_ids:
            if aid in observed_actions:
                self.beliefs[aid].update(observed_actions[aid])

    # --------------------------------------------------------
    # FEATURE VECTOR
    # --------------------------------------------------------

    def get_feature_vector(self) -> np.ndarray:
        """
        Concatenate belief state features for all other agents.
        Shape: (9 * num_others,)
        """
        parts = [
            self.beliefs[aid].to_feature_vector()
            for aid in self.other_agent_ids
        ]
        return np.concatenate(parts).astype(np.float32)

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    def get_belief_summary(self) -> Dict[str, Dict]:
        """
        Returns a dict of human-readable belief summaries
        for all observed agents.
        """
        summary = {}
        for aid in self.other_agent_ids:
            bs = self.beliefs[aid]
            summary[aid] = {
                "dominant_behavior": bs.dominant_behavior,
                "confidence": round(bs.dominant_confidence, 3),
                "beliefs": {
                    b: round(float(bs.beliefs[i]), 3)
                    for i, b in enumerate(BEHAVIOR_TYPES)
                },
                "predicted_actions": {
                    "brake": round(float(bs.predicted_action_probs[0]), 3),
                    "yield": round(float(bs.predicted_action_probs[1]), 3),
                    "maintain": round(float(bs.predicted_action_probs[2]), 3),
                    "accelerate": round(float(bs.predicted_action_probs[3]), 3),
                },
                "inferred_risk_tolerance": round(bs.inferred_risk_tolerance, 3),
                "svo": round(bs.social_value_orientation, 3),
            }
        return summary

    def get_adaptation_hint(
        self, other_agent_id: str
    ) -> Dict[str, float]:
        """
        Computes adaptation signals the ego agent should apply
        when interacting with a specific other agent.

        Returns:
          - yield_bias:   float in [0,1], how much more the ego should yield
          - gap_buffer:   float in [0,1], how much larger a safety gap to keep
          - speed_adj:    float in [-1,1], speed adjustment direction
        """
        if other_agent_id not in self.beliefs:
            return {"yield_bias": 0.0, "gap_buffer": 0.0, "speed_adj": 0.0}

        bs = self.beliefs[other_agent_id]
        risk = bs.inferred_risk_tolerance
        svo = bs.social_value_orientation

        # If other is aggressive (high risk, negative SVO), ego yields more
        yield_bias = float(np.clip(risk - 0.45, 0.0, 1.0))
        gap_buffer = float(np.clip(risk * 0.8, 0.0, 1.0))
        speed_adj = float(np.clip(-svo * 0.5, -1.0, 1.0))

        return {
            "yield_bias": round(yield_bias, 3),
            "gap_buffer": round(gap_buffer, 3),
            "speed_adj": round(speed_adj, 3),
        }


# ============================================================
# MULTI-AGENT ToM MANAGER
# ============================================================

class MultiAgentToMManager:
    """
    Manages one TheoryOfMindModule per learning agent.

    Call update_all() every step with all agents' actions.
    Call get_tom_features(agent_id) to get the ToM feature
    vector for injection into the observation of that agent.

    This is the interface used by the visualization server
    and by the MAPPO runner.
    """

    def __init__(
        self,
        agent_ids: List[str],
        alpha: float = 0.85,
    ) -> None:
        self.agent_ids = agent_ids
        self.modules: Dict[str, TheoryOfMindModule] = {}

        for aid in agent_ids:
            others = [a for a in agent_ids if a != aid]
            self.modules[aid] = TheoryOfMindModule(
                ego_agent_id=aid,
                other_agent_ids=others,
                alpha=alpha,
            )

        # Feature dim = 9 * (num_agents - 1) per agent
        self.feature_dim_per_agent = 9 * (len(agent_ids) - 1)

    def update_all(self, actions: Dict[str, int]) -> None:
        """Update all agents' ToM modules with the latest actions."""
        for aid in self.agent_ids:
            others = {
                k: v for k, v in actions.items()
                if k != aid
            }
            self.modules[aid].update(others)

    def get_tom_features(self, agent_id: str) -> np.ndarray:
        """Get ToM feature vector for the given ego agent."""
        return self.modules[agent_id].get_feature_vector()

    def get_full_summary(self) -> Dict[str, Dict]:
        """Return beliefs of all agents about all others."""
        return {
            aid: self.modules[aid].get_belief_summary()
            for aid in self.agent_ids
        }

    def get_adaptation_hints(self) -> Dict[str, Dict[str, Dict]]:
        """
        Return adaptation hints for every (ego, other) pair.
        """
        hints = {}
        for aid in self.agent_ids:
            mod = self.modules[aid]
            hints[aid] = {
                other: mod.get_adaptation_hint(other)
                for other in mod.other_agent_ids
            }
        return hints
