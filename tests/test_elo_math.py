from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest


def _load_module():
    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "estimate_elo.py"
    spec = importlib.util.spec_from_file_location("estimate_elo", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load estimate_elo module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EloMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_module()

    def test_logistic_transform_basics(self) -> None:
        self.assertAlmostEqual(self.mod.elo_diff_from_score(0.5), 0.0, places=6)
        self.assertGreater(self.mod.elo_diff_from_score(0.7), 0.0)
        self.assertLess(self.mod.elo_diff_from_score(0.3), 0.0)

    def test_score_ci_is_finite_and_monotonic(self) -> None:
        lo, hi = self.mod.score_confidence_interval(0.55, 0.04)
        self.assertTrue(math.isfinite(lo))
        self.assertTrue(math.isfinite(hi))
        self.assertLessEqual(lo, hi)

        elo_lo = self.mod.elo_diff_from_score(lo)
        elo_hi = self.mod.elo_diff_from_score(hi)
        self.assertLessEqual(elo_lo, elo_hi)

    def test_score_ci_clipping_bounds(self) -> None:
        lo, hi = self.mod.score_confidence_interval(0.001, 1.0)
        self.assertGreaterEqual(lo, self.mod.MIN_SCORE_FOR_ELO)
        self.assertLessEqual(hi, self.mod.MAX_SCORE_FOR_ELO)

    def test_weighted_combine_prefers_lower_variance(self) -> None:
        anchors = [
            {"elo_estimate": 1500.0, "elo_std": 10.0},
            {"elo_estimate": 1800.0, "elo_std": 80.0},
        ]
        combined = self.mod.combine_anchor_estimates(anchors)
        self.assertIsNotNone(combined["elo"])
        self.assertLess(abs(combined["elo"] - 1500.0), abs(combined["elo"] - 1800.0))


if __name__ == "__main__":
    unittest.main()
