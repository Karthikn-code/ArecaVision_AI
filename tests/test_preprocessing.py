"""
test_preprocessing.py
---------------------
Unit tests for preprocessing and image denoising pipeline.
"""

import os
import sys
import unittest
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from augmentation.augmentor import preprocess_and_denoise_image
from config.config import IMG_HEIGHT, IMG_WIDTH, RESULTS_DIR


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Create a synthetic test image file."""
        self.temp_dir = os.path.join(RESULTS_DIR, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.test_img_path = os.path.join(self.temp_dir, "test_synthetic.jpg")

        # Create 300x300 synthetic color image
        synthetic_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite(self.test_img_path, synthetic_img)

    def test_preprocess_and_denoise_shape(self):
        """Verify image is correctly resized to 224x224x3 float32."""
        processed = preprocess_and_denoise_image(self.test_img_path)
        self.assertEqual(processed.shape, (IMG_HEIGHT, IMG_WIDTH, 3))
        self.assertEqual(processed.dtype, np.float32)

    def test_invalid_path_raises_error(self):
        """Verify reading a non-existent file raises an exception."""
        with self.assertRaises(Exception):
            preprocess_and_denoise_image("non_existent_file.png")

    def tearDown(self):
        """Cleanup synthetic test image."""
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)


if __name__ == '__main__':
    unittest.main()
