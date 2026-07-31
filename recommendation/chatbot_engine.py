"""
chatbot_engine.py — Specialized AI Agronomist Knowledge & Response Engine for ArecaVision AI.

Provides expert conversational answers on:
- Areca Nut Diseases (Mahali/Koleroga, Bud Borer, Healthy Nut care)
- Areca Leaf Health (Yellow Leaf Disease, Leaf Spot, Healthy Leaf maintenance)
- Trunk & Foot Diseases (Stem Bleeding, Stem Cracking, Foot Rot / Anabe Roga)
- Treatment Recipes (1% Bordeaux mixture, Copper Oxychloride, Trichoderma viride)
- Soil & Fertilizer Management (NPK ratios, organic neem cake, liming, drainage)
- Monsoon & Seasonal Protocols (Pre-monsoon spraying, post-monsoon care)
"""

import re
import random
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger("ChatbotEngine")

# ─────────────────────────────────────────────────────────────────────────────
#  Comprehensive Areca Palm Agronomist Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
ARECA_KNOWLEDGE_BASE = {
    "koleroga": {
        "keywords": ["koleroga", "mahali", "fruit rot", "nut rot", "rotting nuts", "monsoon rot", "falling nuts"],
        "title": "Mahali / Koleroga (Fruit Rot) Management",
        "cause": "Caused by Phytophthora meadii fungus, triggered by continuous heavy monsoon rain and high humidity.",
        "organic_remedies": [
            "Apply Trichoderma viride bio-fungicide mixed with well-rotted farmyard manure around the base (2 kg/palm).",
            "Collect and burn all fallen infected nuts and bunches to destroy fungal spores.",
            "Ensure proper drainage channels in the plantation to prevent waterlogging."
        ],
        "chemical_remedies": [
            "Prophylactic spraying of 1% Bordeaux Mixture before the onset of Southwest Monsoon (May-June).",
            "Second spray with 1% Bordeaux Mixture 40 days after the first spray if rains continue.",
            "Alternatively spray Metalaxyl + Mancozeb (2 g/L) or Copper Oxychloride 0.3% (3 g/L) on nut bunches."
        ],
        "recipe_bordeaux": "To make 100L of 1% Bordeaux Mixture: Dissolve 1 kg Copper Sulphate in 50L water (non-metal bucket). Slake 1 kg Quicklime in 50L water. Slowly pour Copper Sulphate solution into lime solution while stirring. Test pH with iron nail — if rust coats the nail, add more lime solution until neutral.",
        "dosage": "Spraying volume: 2 to 3 liters of spray solution per mature palm tree targeting nut bunches directly."
    },
    "yellow_leaf": {
        "keywords": ["yellow leaf", "yld", "yellowing", "yellow leaves", "leaf yellow", "chlorosis"],
        "title": "Yellow Leaf Disease (YLD) Management",
        "cause": "Caused by Phytoplasma transmitted by plant hopper vectors, leading to chlorosis, leaf yellowing, and crown reduction.",
        "organic_remedies": [
            "Apply balanced organic nutrition: 12 kg well-decomposed compost/vermicompost + 2 kg Neem cake per palm annually.",
            "Incorporate green manure crops like Pueraria phaseoloides or Calopogonium in basin areas.",
            "Apply Micronutrient mixture (Boron, Zinc, Magnesium) @ 100g per palm in two split doses."
        ],
        "chemical_remedies": [
            "Soil application of Phorate 10G @ 10g per palm or Imidacloprid 17.8 SL @ 0.5 ml/L spray to control insect vectors.",
            "Foliar spray of Magnesium Sulphate (10 g/L) + Zinc Sulphate (5 g/L) to alleviate nutritional chlorosis."
        ],
        "dosage": "Apply fertilizers in two split doses: 1st in May-June (pre-monsoon), 2nd in September-October (post-monsoon)."
    },
    "bud_rot": {
        "keywords": ["bud rot", "spear leaf", "rotten bud", "crown rot", "spear rotting"],
        "title": "Bud Rot Disease Management",
        "cause": "Phytophthora palmivora fungus infecting the growing central spear leaf and spindle.",
        "organic_remedies": [
            "Remove affected spindle and dead tissue, treat crown cavity with Bordeaux paste.",
            "Cover the treated crown with a polythene cap to prevent rainwater ingress.",
            "Apply Pseudomonas fluorescens (20 g/L) around the crown area."
        ],
        "chemical_remedies": [
            "Apply Copper Oxychloride paste (50 g in 100 ml water) into the crown central spindle cavity.",
            "Drench surrounding healthy palms with 1% Bordeaux Mixture or Fosetyl-Al (2 g/L)."
        ],
        "dosage": "Spot treat early spindle infection immediately; remove severely dead palms to protect neighboring trees."
    },
    "stem_bleeding": {
        "keywords": ["stem bleeding", "bleeding", "trunk bleeding", "dark sap", "bark oozing", "sap oozing"],
        "title": "Stem Bleeding Disease Management",
        "cause": "Fungal pathogen Thielaviopsis paradoxa entering through stem cracks or mechanical wounds.",
        "organic_remedies": [
            "Chisel out infected bark/wood tissue up to healthy tissue and apply Coal Tar or Neem oil paste.",
            "Apply Trichoderma viride fortified farmyard manure (5 kg/palm) in root basins.",
            "Avoid root damage during intercultural operations."
        ],
        "chemical_remedies": [
            "Apply Calixin (Tridemorph) 5 ml or Hexaconazole 5 ml per liter of water on swabbed trunk wounds.",
            "Hot coal tar or Bordeaux paste paint over scraped trunk area."
        ],
        "dosage": "Scrape infected bark clean before painting. Repeat trunk painting twice a year (Pre- and Post-monsoon)."
    },
    "stem_cracking": {
        "keywords": ["stem cracking", "trunk cracking", "cracks", "vertical cracks", "sunscald"],
        "title": "Stem Cracking & Sunscald Prevention",
        "cause": "Sudden moisture fluctuations, excess nitrogen, or intense direct sun exposure (south-west sunscald).",
        "organic_remedies": [
            "White-wash the lower trunk with Lime-slurry (1 kg quicklime + 100g CuSO4 in 5L water) up to 2 meters height.",
            "Maintain steady soil moisture using organic mulching (dried areca leaves/husk) around basin.",
            "Grow shade barrier crops like banana or gliricidia on south-west plantation borders."
        ],
        "chemical_remedies": [
            "Spray 0.2% Borax (2 g/L) if micronutrient boron deficiency is suspected.",
            "Apply copper paste on deep cracks to prevent secondary fungal entry."
        ],
        "dosage": "Lime whitewashing should be completed before December (start of bright summer sunshine)."
    },
    "bordeaux_mixture": {
        "keywords": ["bordeaux", "bordeaux mixture", "bordeaux paste", "copper sulphate", "how to make bordeaux", "bordeaux recipe"],
        "title": "1% Bordeaux Mixture Preparation Guide",
        "cause": "Gold standard protective fungicide against Koleroga, Bud Rot, and Leaf Spot diseases.",
        "organic_remedies": [],
        "chemical_remedies": [
            "Step 1: Dissolve 1 kg Copper Sulphate (blue vitriol) crystals in 50 Liters of water in a plastic or earthen pot.",
            "Step 2: Slake 1 kg Quicklime (calcium oxide) with water and make up to 50 Liters in a separate container.",
            "Step 3: Pour the Copper Sulphate solution slowly into the Lime solution while continuously stirring with a wooden stick.",
            "Step 4 (Test): Dip a clean iron knife or nail into the mixture for 1 minute. If reddish copper coats the nail, add more lime solution until neutral."
        ],
        "recipe_bordeaux": "Use fresh mixture on the day of preparation. Do NOT store prepared Bordeaux mixture overnight as it loses efficacy.",
        "dosage": "Add sticker/spreader agent (e.g. Rosin soap @ 200 ml / 100L) to prevent rain wash-off during monsoon."
    },
    "fertilizer_schedule": {
        "keywords": ["fertilizer", "npk", "manure", "schedule", "nutrient", "fertilizer dose", "urea", "potash", "compost"],
        "title": "Recommended Areca Palm Fertilizer Schedule (Per Palm/Year)",
        "cause": "Optimal nutrition schedule for high yield, thick nuts, and strong disease resistance.",
        "organic_remedies": [
            "Farmyard Manure / Compost: 12 to 20 kg per palm per year (applied during Sept-Oct).",
            "Neem Cake: 2 kg per palm (controls root-knot nematodes and improves soil health)."
        ],
        "chemical_remedies": [
            "Nitrogen (N): 100 grams per palm per year (approx 220g Urea).",
            "Phosphorus (P2O5): 40 grams per palm per year (approx 250g Rock Phosphate / SSP).",
            "Potassium (K2O): 140 grams per palm per year (approx 235g Muriate of Potash - MOP)."
        ],
        "dosage": "Split into two applications: 1/3rd dosage in May-June (Pre-monsoon) and 2/3rds in Sept-Oct (Post-monsoon)."
    },
    "healthy_care": {
        "keywords": ["healthy", "maintain", "good yield", "prevention", "care", "boost yield", "irrigation"],
        "title": "Best Practices for Maintaining Healthy Areca Palms",
        "cause": "Preventive cultural practices to sustain high quality yields and disease-free gardens.",
        "organic_remedies": [
            "Irrigation: Provide 175 to 200 Liters of water per palm every 4 to 5 days during dry summer months.",
            "Mulching: Cover palm basins with dried leaves, green manures, or areca husk (thick 10cm layer) to conserve moisture.",
            "Intercropping: Grow pepper, cocoa, banana, or cardamom for microclimate moderation and additional farm income.",
            "Drainage: Construct 30cm deep drainage channels between palm rows before June."
        ],
        "chemical_remedies": [
            "Annual prophylactic 1% Bordeaux mixture spraying on nut bunches before June monsoon.",
            "Soil testing every 2 years to adjust liming (apply 1 kg lime per palm if soil pH < 5.5)."
        ],
        "dosage": "Maintain plant spacing of 2.7m x 2.7m (about 1370 palms per hectare)."
    }
}

DEFAULT_RESPONSES = [
    "ArecaVision AI Assistant is ready to help! You can ask me about **Mahali / Koleroga disease**, **Yellow Leaf Disease**, **Stem Bleeding**, **1% Bordeaux Mixture recipe**, or **NPK Fertilizer schedules**.",
    "I specialize in Areca nut, leaf, and trunk health. Try asking: *'How to prepare 1% Bordeaux mixture?'* or *'What is the treatment for Koleroga?'*",
    "For best yields and disease prevention, ensure pre-monsoon spraying in May-June. Ask me any question about remedies, dosages, or soil care!"
]


# ─────────────────────────────────────────────────────────────────────────────
#  Query Matcher & Response Generator
# ─────────────────────────────────────────────────────────────────────────────
def get_bot_response(user_query: str, current_prediction: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates structured, expert agronomist response based on farmer query.
    Optionally incorporates current image prediction context if provided.
    """
    query_clean = user_query.strip().lower()
    logger.info(f"Processing chatbot query: '{user_query}' | Context: {current_prediction}")

    # Check for greeting or generic hello
    if any(g in query_clean for g in ["hi", "hello", "hey", "namaste", "namaskara"]):
        greeting_text = "Namaskara! I am **ArecaBot**, your AI Agronomist Assistant. How can I help you with your Areca palm garden today?"
        if current_prediction and current_prediction != "Unknown":
            greeting_text += f"\n\n*(Current context: Last scanned image diagnosed as **{current_prediction}**)*"
        return {
            "answer": greeting_text,
            "topic": "Greeting",
            "suggestions": [
                "How to treat Koleroga disease?",
                "How to make 1% Bordeaux mixture?",
                "What is the NPK fertilizer dose?",
                "How to cure Yellow Leaf Disease?"
            ]
        }

    # Match topic from knowledge base
    matched_topic_key = None
    for key, data in ARECA_KNOWLEDGE_BASE.items():
        for kw in data["keywords"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', query_clean):
                matched_topic_key = key
                break
        if matched_topic_key:
            break

    # If context is available and user asks "how to treat it" or "what is the cure"
    if not matched_topic_key and current_prediction:
        pred_lower = current_prediction.lower()
        if "koleroga" in pred_lower or "mahali" in pred_lower:
            matched_topic_key = "koleroga"
        elif "yellow" in pred_lower:
            matched_topic_key = "yellow_leaf"
        elif "bleeding" in pred_lower:
            matched_topic_key = "stem_bleeding"
        elif "cracking" in pred_lower:
            matched_topic_key = "stem_cracking"
        elif "bud" in pred_lower:
            matched_topic_key = "bud_rot"
        elif "healthy" in pred_lower:
            matched_topic_key = "healthy_care"

    # Construct Answer
    if matched_topic_key:
        topic_data = ARECA_KNOWLEDGE_BASE[matched_topic_key]
        answer_md = f"### 🌴 {topic_data['title']}\n\n"
        answer_md += f"**Cause / Context:** {topic_data['cause']}\n\n"

        if topic_data.get("organic_remedies"):
            answer_md += "#### 🌿 Organic & Cultural Management:\n"
            for item in topic_data["organic_remedies"]:
                answer_md += f"- {item}\n"
            answer_md += "\n"

        if topic_data.get("chemical_remedies"):
            answer_md += "#### 🧪 Chemical / Fungicide Treatments:\n"
            for item in topic_data["chemical_remedies"]:
                answer_md += f"- {item}\n"
            answer_md += "\n"

        if topic_data.get("recipe_bordeaux"):
            answer_md += f"💡 **Recipe Note:** {topic_data['recipe_bordeaux']}\n\n"

        if topic_data.get("dosage"):
            answer_md += f"📌 **Dosage & Timing:** {topic_data['dosage']}\n"

        return {
            "answer": answer_md,
            "topic": topic_data["title"],
            "suggestions": [
                "How to prepare 1% Bordeaux mixture?",
                "What is the NPK fertilizer schedule?",
                "How to control Koleroga in heavy monsoon?",
                "Tips for healthy Areca palms"
            ]
        }

    # Fallback intelligent response for general Areca queries
    fallback_text = (
        f"Thank you for asking about **'{user_query}'**!\n\n"
        f"Here are the core agronomist recommendations for Areca Palm health:\n\n"
        f"1. **Monsoon Care**: Apply 1% Bordeaux mixture before monsoon rains (May-June) to prevent **Koleroga (Fruit Rot)**.\n"
        f"2. **Nutrition**: Provide 12-20 kg compost + 100g N, 40g P2O5, 140g K2O per palm annually in two split doses.\n"
        f"3. **Yellow Leaf Disease**: Drench with bio-fungicide Trichoderma viride and spray micronutrients (Magnesium & Boron).\n"
        f"4. **Drainage**: Ensure 30cm deep drainage trenches to avoid waterlogging and root rots.\n\n"
        f"*Try asking specifically about: 'Bordeaux recipe', 'Koleroga treatment', 'Yellow leaf disease', or 'Fertilizer dose'.*"
    )

    return {
        "answer": fallback_text,
        "topic": "General Areca Guidance",
        "suggestions": [
            "How to treat Koleroga / Mahali?",
            "How to make 1% Bordeaux mixture?",
            "Fertilizer dose per tree",
            "Yellow Leaf Disease management"
        ]
    }
