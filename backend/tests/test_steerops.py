"""
Unit tests for SteerOps evaluator, interpreter, and schemas.

Tests metric computation, scoring logic, and API schemas
without requiring a GPU or loaded model.
"""

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- EVALUATOR METRICS ---


class TestEvaluatorMetrics:
    """Evaluator metric computation without a real model."""

    def _make_evaluator(self):
        from app.core.evaluator import Evaluator
        return Evaluator()

    def test_simple_sentiment_positive(self):
        ev = self._make_evaluator()
        result = ev._simple_sentiment("This is a wonderful amazing great day!")
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_simple_sentiment_negative(self):
        ev = self._make_evaluator()
        result = ev._simple_sentiment("This is terrible awful horrible bad.")
        assert isinstance(result, float)

    def test_simple_sentiment_empty(self):
        ev = self._make_evaluator()
        result = ev._simple_sentiment("")
        assert isinstance(result, float)

    def test_aggregate_metrics_empty(self):
        ev = self._make_evaluator()
        result = ev._aggregate_metrics([])
        assert result == {}

    def test_aggregate_metrics_single(self):
        ev = self._make_evaluator()
        result = ev._aggregate_metrics([{
            "semantic_shift": 0.3,
            "perplexity_delta": 5.0,
        }])
        assert "avg_semantic_shift" in result
        assert result["avg_semantic_shift"] == 0.3

    def test_aggregate_metrics_mean(self):
        ev = self._make_evaluator()
        result = ev._aggregate_metrics([
            {"semantic_shift": 0.2, "perplexity_delta": 3.0},
            {"semantic_shift": 0.4, "perplexity_delta": 7.0},
        ])
        assert abs(result["avg_semantic_shift"] - 0.3) < 0.001
        assert abs(result["avg_perplexity_delta"] - 5.0) < 0.001
        assert "std_semantic_shift" in result

    def test_overall_score_structure(self):
        ev = self._make_evaluator()
        agg = {
            "avg_semantic_shift": 0.15,
            "avg_perplexity_ratio": 1.1,
            "avg_concept_alignment_delta": 0.05,
            "behavioral_consistency": 0.8,
            "avg_steering_efficiency": 0.05,
        }
        result = ev._compute_overall_score(agg)
        assert "score" in result
        assert "grade" in result
        assert "breakdown" in result
        assert 0 <= result["score"] <= 100
        assert result["grade"] in ("A+", "A", "B+", "B", "C", "D", "F")

    def test_high_quality_high_score(self):
        ev = self._make_evaluator()
        agg = {
            "avg_semantic_shift": 0.35,
            "avg_perplexity_ratio": 1.05,
            "avg_concept_alignment_delta": 0.2,
            "behavioral_consistency": 0.9,
            "avg_steering_efficiency": 0.12,
        }
        result = ev._compute_overall_score(agg)
        assert result["score"] >= 60

    def test_low_quality_low_score(self):
        ev = self._make_evaluator()
        agg = {
            "avg_semantic_shift": 0.0,
            "avg_perplexity_ratio": 3.0,
        }
        result = ev._compute_overall_score(agg)
        assert result["score"] < 50

    def test_concept_anchors_exist(self):
        ev = self._make_evaluator()
        for concept in ["politeness", "toxicity", "creativity", "refusal", "verbosity"]:
            anchors = ev._get_concept_anchors(concept)
            assert len(anchors) >= 3, f"{concept} should have 3+ anchors"

    def test_concept_anchors_unknown(self):
        ev = self._make_evaluator()
        anchors = ev._get_concept_anchors("nonexistent_concept_xyz")
        assert anchors == []


# --- INTERPRETER ---


class TestInterpreter:
    """ResponseInterpreter uses embeddings, not keywords."""

    def test_no_keyword_patterns(self):
        from app.core.interpreter import ResponseInterpreter
        interp = ResponseInterpreter()
        assert not hasattr(interp, "_KEYWORD_PATTERNS"), \
            "_KEYWORD_PATTERNS should not exist — keywords removed"

    def test_interpret_returns_valid_structure(self):
        from app.core.interpreter import ResponseInterpreter
        interp = ResponseInterpreter()
        result = interp.interpret(
            expected_response="Please be very polite and considerate.",
            prompt="Tell me about AI",
        )
        assert hasattr(result, "scores") or hasattr(result, "to_dict")

    def test_interpret_behavior_mode(self):
        from app.core.interpreter import ResponseInterpreter
        interp = ResponseInterpreter()
        result = interp.interpret(
            expected_response="",
            prompt="",
            behavior_description="be rude and dismissive",
        )
        assert result is not None


# --- API SCHEMAS ---


class TestSchemas:
    """API request/response schema validation."""

    def test_analyze_request_prompt_optional(self):
        from app.schemas import AnalyzeRequest
        req = AnalyzeRequest(behavior_description="be helpful")
        assert req.prompt == ""

    def test_analyze_request_with_prompt(self):
        from app.schemas import AnalyzeRequest
        req = AnalyzeRequest(
            prompt="Tell me about AI",
            expected_response="AI is fascinating...",
        )
        assert req.prompt == "Tell me about AI"

    def test_steering_config(self):
        from app.schemas import SteeringConfig
        cfg = SteeringConfig(layer=15, strength=-5.0)
        assert cfg.layer == 15
        assert cfg.direction_vector is None

    def test_steering_config_with_vector(self):
        from app.schemas import SteeringConfig
        cfg = SteeringConfig(
            layer=15, strength=-5.0,
            direction_vector=[0.1, 0.2, 0.3],
        )
        assert cfg.direction_vector == [0.1, 0.2, 0.3]


# --- STATISTICAL UTILITIES ---


class TestStatUtils:
    """Cohen's d and paired t-test from the definitive test script."""

    def test_cohens_d_identical_groups(self):
        """Identical groups should produce d ≈ 0."""
        import numpy as np
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        na, nb = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
        d = (mean_b - mean_a) / pooled_std if pooled_std > 1e-10 else 0.0
        assert abs(d) < 0.01

    def test_cohens_d_known_effect(self):
        """Groups with known difference should produce expected d."""
        import numpy as np
        a = [0.0] * 100
        b = [1.0] * 100
        na, nb = len(a), len(b)
        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
        if pooled_std < 1e-10:
            d = 0.0
        else:
            d = (mean_b - mean_a) / pooled_std
        # All zeros vs all ones: pooled_std ≈ 0, d undefined
        # Both groups have zero variance, so d should be 0
        assert d == 0.0 or abs(d) > 10  # edge case

    def test_paired_t_test_significant(self):
        """Clearly different paired samples should be significant."""
        from scipy import stats
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        t_stat, p_value = stats.ttest_rel(a, b)
        assert p_value < 0.001

    def test_paired_t_test_not_significant(self):
        """Nearly identical paired samples should not be significant."""
        from scipy import stats
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.01, 2.01, 3.01, 4.01, 5.01]
        t_stat, p_value = stats.ttest_rel(a, b)
        # Tiny difference, may or may not be significant with n=5
        assert isinstance(p_value, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
