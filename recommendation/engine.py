import json
import os
from config.config import RECOMMENDATION_JSON_PATH
from utils.logger import get_logger

logger = get_logger("RecommendationEngine")

class RecommendationEngine:
    def __init__(self, json_path=RECOMMENDATION_JSON_PATH):
        self.json_path = json_path
        self.database = {}
        self.load_database()
        
    def load_database(self):
        logger.info(f"Loading recommendation database from {self.json_path}...")
        if not os.path.exists(self.json_path):
            logger.error(f"Recommendation JSON file not found at {self.json_path}")
            self.database = {}
            return
            
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            logger.info("Recommendation database loaded successfully.")
        except Exception as e:
            logger.error(f"Error reading recommendation database: {e}")
            self.database = {}
            
    def get_recommendation(self, disease_name):
        """
        Retrieves recommendation details for a given disease name.
        Uses case-insensitive mapping if exact match fails.
        """
        if not self.database:
            self.load_database()
            
        # Try direct lookup
        if disease_name in self.database:
            return self.database[disease_name]
            
        # Try case-insensitive lookup
        for key in self.database.keys():
            if key.lower() == disease_name.lower():
                return self.database[key]
                
        # Default fallback
        logger.warning(f"No recommendations found for class: {disease_name}")
        return {
            "scientific_name": "N/A",
            "display_name": disease_name,
            "description": "No description available in database.",
            "cause": "Unknown",
            "symptoms": ["No symptoms cataloged."],
            "treatment": {
                "organic_control": "Contact your local agricultural extension service.",
                "chemical_control": "Consult an expert before applying chemical treatment.",
                "recommended_fungicide": "N/A",
                "recommended_pesticide": "N/A"
            },
            "preventive_measures": ["Monitor crop health regularly."]
        }
