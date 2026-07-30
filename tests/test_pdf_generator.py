"""
test_pdf_generator.py
---------------------
Unit tests for PDF Diagnostic Report generator (reports/pdf_generator.py).
"""

import os
import sys
import unittest
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from reports.pdf_generator import generate_pdf_report
from config.config import RESULTS_DIR


class TestPDFGenerator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = os.path.join(RESULTS_DIR, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create dummy test images
        self.orig_img = os.path.join(self.temp_dir, "test_orig.jpg")
        self.gradcam_img = os.path.join(self.temp_dir, "test_gradcam.jpg")
        self.output_pdf = os.path.join(self.temp_dir, "test_report.pdf")

        dummy_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.orig_img, dummy_np)
        cv2.imwrite(self.gradcam_img, dummy_np)

        self.mock_rec = {
            "display_name": "Mahali / Koleroga",
            "scientific_name": "Phytophthora palmivora",
            "description": "Fungal disease causing nut rot during monsoons.",
            "cause": "Phytophthora palmivora pathogen",
            "symptoms": ["Water-soaked lesions", "Fruit rot"],
            "treatment": {
                "organic_control": "Apply Trichoderma viride.",
                "chemical_control": "Spray 1% Bordeaux mixture.",
                "recommended_fungicide": "Bordeaux Mixture 1%",
                "recommended_pesticide": "None"
            },
            "preventive_measures": ["Improve drainage", "Clean fallen fruits"]
        }

        self.mock_severity = {
            "severity_pct": 28.5,
            "severity_level": "Moderate",
            "status_color": "#E67E22"
        }

    def test_pdf_generation(self):
        """Verify PDF report file is generated successfully and non-empty."""
        generate_pdf_report(
            dest_path=self.output_pdf,
            farmer_name="Test Farmer",
            original_img_path=self.orig_img,
            gradcam_img_path=self.gradcam_img,
            predicted_disease="Mahali_Koleroga",
            confidence=0.965,
            processing_time=0.142,
            model_used="EfficientNet-B0",
            rec_details=self.mock_rec,
            severity_info=self.mock_severity
        )

        self.assertTrue(os.path.exists(self.output_pdf))
        self.assertGreater(os.path.getsize(self.output_pdf), 1000)

    def tearDown(self):
        for f in [self.orig_img, self.gradcam_img, self.output_pdf]:
            if os.path.exists(f):
                os.remove(f)


if __name__ == '__main__':
    unittest.main()
