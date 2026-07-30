"""
test_class_expansion.py
------------------------
Unit tests for the 14-class expanded taxonomy, recommendation engine,
Kannada translations, and dynamic CNN architecture shapes.
"""

import unittest
from config.config import CLASS_NAMES, DISPLAY_NAMES
from recommendation.engine import RecommendationEngine
from recommendation.translations import get_kannada_recommendation, KANNADA_TRANSLATIONS
from models.architectures import build_efficientnet_b0, build_mobilenet_v3, build_resnet50


class TestClassExpansion(unittest.TestCase):

    def test_class_names_count(self):
        self.assertEqual(len(CLASS_NAMES), 14)

    def test_display_names_mapping(self):
        for cls in CLASS_NAMES:
            self.assertIn(cls, DISPLAY_NAMES)

    def test_recommendation_database_entries(self):
        engine = RecommendationEngine()
        for cls in CLASS_NAMES:
            rec = engine.get_recommendation(cls)
            self.assertIn("scientific_name", rec)
            self.assertIn("description", rec)
            self.assertIn("treatment", rec)
            self.assertIn("preventive_measures", rec)

    def test_kannada_translations_entries(self):
        for cls in CLASS_NAMES:
            rec = get_kannada_recommendation(cls)
            self.assertIn("display_name", rec)
            self.assertIn("description", rec)
            self.assertIn("symptoms", rec)

    def test_cnn_architecture_num_classes(self):
        eff_model = build_efficientnet_b0()
        self.assertEqual(eff_model.output_shape[-1], 14)

        mob_model = build_mobilenet_v3()
        self.assertEqual(mob_model.output_shape[-1], 14)

        res_model = build_resnet50()
        self.assertEqual(res_model.output_shape[-1], 14)


if __name__ == "__main__":
    unittest.main()
