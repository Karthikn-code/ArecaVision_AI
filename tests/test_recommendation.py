"""
test_recommendation.py
----------------------
Unit tests for recommendation engine (recommendation/engine.py).
"""

import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recommendation.engine import RecommendationEngine
from config.config import CLASS_NAMES


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RecommendationEngine()

    def test_database_loaded(self):
        """Verify the disease database JSON loads successfully."""
        self.assertTrue(len(self.engine.database) > 0)

    def test_all_class_names_present(self):
        """Verify recommendations exist for all 9 configured dataset class names."""
        for cls in CLASS_NAMES:
            rec = self.engine.get_recommendation(cls)
            self.assertIn("display_name", rec)
            self.assertIn("symptoms", rec)
            self.assertIn("treatment", rec)

    def test_fallback_behavior(self):
        """Verify fallback recommendation dictionary for unknown disease class."""
        rec = self.engine.get_recommendation("Unknown_Fake_Disease_Class")
        self.assertEqual(rec["scientific_name"], "N/A")
        self.assertIn("organic_control", rec["treatment"])


if __name__ == '__main__':
    unittest.main()
