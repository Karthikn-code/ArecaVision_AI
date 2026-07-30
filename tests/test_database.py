"""
test_database.py
----------------
Unit tests for the SQLite database manager (database/db_manager.py).
"""

import os
import sys
import unittest
import uuid
import pandas as pd

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from database.db_manager import (
    init_db, save_prediction, get_history_df,
    delete_prediction, clear_all_history
)


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        """Initialize database before each test."""
        init_db()

    def test_01_init_db(self):
        """Verify database initialization creates the table."""
        df = get_history_df()
        self.assertIsInstance(df, pd.DataFrame)

    def test_02_save_and_retrieve_prediction(self):
        """Verify saving a prediction record and retrieving it."""
        test_id = f"TEST_{str(uuid.uuid4())[:6]}"
        save_prediction(
            prediction_id=test_id,
            image_path="test_image.jpg",
            predicted_class="Mahali_Koleroga",
            confidence=0.954,
            processing_time=0.123,
            model_used="EfficientNet-B0"
        )

        df = get_history_df()
        self.assertFalse(df.empty)
        record = df[df['prediction_id'] == test_id]
        self.assertEqual(len(record), 1)
        self.assertEqual(record.iloc[0]['predicted_class'], "Mahali_Koleroga")
        self.assertAlmostEqual(record.iloc[0]['confidence'], 0.954, places=3)

    def test_03_delete_prediction(self):
        """Verify deleting an individual prediction record."""
        test_id = f"DEL_{str(uuid.uuid4())[:6]}"
        save_prediction(
            prediction_id=test_id,
            image_path="test_del.jpg",
            predicted_class="stem cracking",
            confidence=0.88,
            processing_time=0.05,
            model_used="MobileNetV3"
        )

        delete_prediction(test_id)
        df = get_history_df()
        record = df[df['prediction_id'] == test_id]
        self.assertEqual(len(record), 0)

    def test_04_clear_all_history(self):
        """Verify clearing all history removes all rows."""
        save_prediction(
            prediction_id="CLR_01",
            image_path="test_clr.jpg",
            predicted_class="Healthy_Leaf",
            confidence=0.99,
            processing_time=0.04,
            model_used="ResNet50"
        )

        clear_all_history()
        df = get_history_df()
        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
