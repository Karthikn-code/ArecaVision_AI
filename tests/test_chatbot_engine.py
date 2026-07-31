"""
test_chatbot_engine.py — Unit tests for ArecaBot chatbot engine using unittest.
"""

import unittest
from recommendation.chatbot_engine import get_bot_response


class TestChatbotEngine(unittest.TestCase):

    def test_greeting_response(self):
        resp = get_bot_response("Hello Namaste")
        self.assertIn("Namaskara", resp["answer"])
        self.assertEqual(resp["topic"], "Greeting")

    def test_koleroga_query(self):
        resp = get_bot_response("How to treat Koleroga disease during monsoon?")
        self.assertIn("Mahali / Koleroga", resp["answer"])
        self.assertIn("Bordeaux Mixture", resp["answer"])
        self.assertTrue(len(resp["suggestions"]) > 0)

    def test_bordeaux_recipe_query(self):
        resp = get_bot_response("How to prepare 1% Bordeaux mixture step by step?")
        self.assertIn("Copper Sulphate", resp["answer"])
        self.assertIn("Quicklime", resp["answer"])

    def test_fertilizer_query(self):
        resp = get_bot_response("What is the NPK fertilizer schedule?")
        self.assertTrue("Nitrogen" in resp["answer"] or "NPK" in resp["answer"])

    def test_context_awareness(self):
        resp = get_bot_response("What is the remedy?", current_prediction="Mahali_Koleroga")
        self.assertIn("Mahali / Koleroga", resp["answer"])


if __name__ == "__main__":
    unittest.main()
