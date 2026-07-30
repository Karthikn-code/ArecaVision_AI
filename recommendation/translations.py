"""
translations.py
---------------
Multilingual translation mapping (English & Kannada / ಕನ್ನಡ)
for ArecaVision AI disease diagnostics and recommendations.
"""

KANNADA_TRANSLATIONS = {
    "bud borer": {
        "display_name": "ಮೊಗ್ಗು ಕೊರೆಕ (Bud Borer)",
        "scientific_name": "Tirathaba rufivena",
        "description": "ಸುಳಿಯ ಮೊಗ್ಗುಗಳು ಮತ್ತು ಹೂವಿನ ಗೊಂಚಲುಗಳನ್ನು ಹುಳುಗಳು ತಿಂದು ನಾಶಪಡಿಸುವ ಕೀಟ ಬಾಧೆ.",
        "cause": "ಟಿರಾತಾಬಾ ರುಫಿವೇನಾ ಪತಂಗದ ಮರಿಹುಳುಗಳು.",
        "symptoms": [
            "ಹೂವಿನ ಗೊಂಚಲಿನಲ್ಲಿ ಸಣ್ಣ ರಂಧ್ರಗಳು ಮತ್ತು ಹಿಕ್ಕೆಯ ಲಕ್ಷಣಗಳು.",
            "ಎಳೆಯ ಕಾಯಿಗಳು (ಹಿಂಗಾರ) ಉದುರುವುದು ಮತ್ತು ಹೂವು ಕಪ್ಪಾಗುವುದು.",
            "ಸುಳಿ ಒಣಗುವುದು."
        ],
        "organic_control": "ಬೇವು ಎಣ್ಣೆ (1-3%) ಅಥವಾ ಬೆಸಿಲಸ್ ತುರಿಂಜಿಯೆನ್ಸಿಸ್ (Bt) ಸಿಂಪರಣೆ ಮಾಡಿ.",
        "chemical_control": "ಕ್ಲೋರ್‌ಪೈರಿಫಾಸ್ 20 EC (2 ಮಿ.ಲೀ / ಲೀಟರ್ ನೀರಿಗೆ) ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ತೋಟವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಿ ಮತ್ತು ಸೋಂಕಿತ ಹೂವುಗಳನ್ನು ಸುಟ್ಟುಹಾಕಿ.",
            "ಬೆಳಕಿನ ಬಲೆಗಳನ್ನು ಬಳಸಿ ಪ್ರೌಢ ಪತಂಗಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ."
        ]
    },
    "healthy_foot": {
        "display_name": "ಆರೋಗ್ಯಕರ ಬುಡ (Healthy Base)",
        "scientific_name": "N/A",
        "description": "ಮರದ ಬುಡವು ಯಾವುದೇ ರೋಗ ಅಥವಾ ಕೊಳೆತವಿಲ್ಲದೆ ದೃಢವಾಗಿದೆ.",
        "cause": "ಉತ್ತಮ ನೀರು ಬಸಿಕಾಲುವೆ ಹಾಗೂ ತೋಟದ ನಿರ್ವಹಣೆ.",
        "symptoms": [
            "ದೃಢವಾದ ತೊಗಟೆ ಮತ್ತು ಆರೋಗ್ಯಕರ ಬೇರುಗಳು.",
            "ನೀರು ನಿಲ್ಲದಿರುವುದು."
        ],
        "organic_control": "ವಾರ್ಷಿಕವಾಗಿ ಸಾವಯವ ಗೊಬ್ಬರ ಮತ್ತು ಕಾಂಪೋಸ್ಟ್ ಹಾಕಿ.",
        "chemical_control": "ಯಾವುದೇ ರಾಸಾಯನಿಕದ ಅಗತ್ಯವಿಲ್ಲ.",
        "preventive_measures": [
            "ಮರದ ಬುಡದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ."
        ]
    },
    "Healthy_Leaf": {
        "display_name": "ಆರೋಗ್ಯಕರ ಗರಿ/ಎಲೆ (Healthy Leaf)",
        "scientific_name": "N/A",
        "description": "ಗರಿಗಳು ದಟ್ಟ ಹಸಿರು ಬಣ್ಣದಿಂದ ಕೂಡಿದ್ದು, ದ್ಯುತಿಸಂಶ್ಲೇಷಣೆ ಉತ್ತಮವಾಗಿದೆ.",
        "cause": "ಸರಿಯಾದ ಪೋಷಕಾಂಶಗಳ ಪೂರೈಕೆ.",
        "symptoms": [
            "ಸಮತೋಲಿತ ಹಸಿರು ಬಣ್ಣದ ಗರಿಗಳು.",
            "ಚುಕ್ಕೆ ಅಥವಾ ಹಳದಿ ಬಣ್ಣ ಇಲ್ಲದಿರುವುದು."
        ],
        "organic_control": "ಸಾರಜನಕ ಮತ್ತು ಪೊಟ್ಯಾಶ್ ಗೊಬ್ಬರ ನೀಡಿ.",
        "chemical_control": "ಅಗತ್ಯವಿಲ್ಲ.",
        "preventive_measures": [
            "ಬೇಸಿಗೆಯಲ್ಲಿ ಸೂಕ್ತ ನೀರಾವರಿ ಒದಗಿಸಿ."
        ]
    },
    "Healthy_Nut": {
        "display_name": "ಆರೋಗ್ಯಕರ ಅಡಿಕೆ ಕಾಯಿ (Healthy Nut)",
        "scientific_name": "N/A",
        "description": "ಕಾಯಿಗಳು ನಯವಾದ ಮೇಲ್ಮೈ ಮತ್ತು ಸಮಾನ ಆಕಾರ ಹೊಂದಿವೆ.",
        "cause": "ಉತ್ತಮ ನೀರು ಮತ್ತು ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶ ನಿರ್ವಹಣೆ.",
        "symptoms": [
            "ಕಾಯಿಗಳ ಉದುರುವಿಕೆ ಇಲ್ಲದಿರುವುದು.",
            "ನಯವಾದ ಮೇಲ್ಮೈ."
        ],
        "organic_control": "ಮರದ ಬುಡಕ್ಕೆ ಹೊದಿಕೆ (Mulching) ಹಾಕಿ.",
        "chemical_control": "ಅಗತ್ಯವಿಲ್ಲ.",
        "preventive_measures": [
            "ಹೂವಾಡುವಾಗ ಸೂಕ್ತ ನೀರಾವರಿ ಒದಗಿಸಿ."
        ]
    },
    "Healthy_Trunk": {
        "display_name": "ಆರೋಗ್ಯಕರ ಮರದ ಕಾಂಡ (Healthy Trunk)",
        "scientific_name": "N/A",
        "description": "ಕಾಂಡವು ನೇರವಾಗಿದ್ದು, ಯಾವುದೇ ಸೀಳಿಕೆ ಅಥವಾ ರಸ ಸೋರುವಿಕೆ ಇಲ್ಲ.",
        "cause": "ಉತ್ತಮ ಬೆಳವಣಿಗೆಯ ವಾತಾವರಣ.",
        "symptoms": [
            "ನಯವಾದ ಕಾಂಡದ ಮೇಲ್ಮೈ."
        ],
        "organic_control": "ಅಗತ್ಯವಿಲ್ಲ.",
        "chemical_control": "ಅಗತ್ಯವಿಲ್ಲ.",
        "preventive_measures": [
            "ಎಳೆಯ ಮರಗಳಿಗೆ ಬಿಸಿಲಿನಿಂದ ರಕ್ಷಣೆ ನೀಡಿ."
        ]
    },
    "Mahali_Koleroga": {
        "display_name": "ಮಹಾಳಿ / ಕೊಳೆ ರೋಗ (Koleroga - Fruit Rot)",
        "scientific_name": "Phytophthora palmivora",
        "description": "ಮಳೆಗಾಲದಲ್ಲಿ ಅಡಿಕೆ ಕಾಯಿಗಳು ಅಪಾರ ಪ್ರಮಾಣದಲ್ಲಿ ಉದುರುವ ಅಪಾಯಕಾರಿ ಶಿಲೀಂಧ್ರ ರೋಗ.",
        "cause": "ಫೈಟೋಪ್ಥೊರಾ ಪಾಲ್ಮಿವೋರಾ ಎಂಬ ಶಿಲೀಂಧ್ರ.",
        "symptoms": [
            "ಎಳೆಯ ಹಸಿರು ಕಾಯಿಗಳ ಮೇಲೆ ನೀರಿನಂತಹ ಮಚ್ಚೆಗಳು.",
            "ಅಡಿಕೆ ಕಾಯಿಗಳು ಅಧಿಕವಾಗಿ ಉದುರುವುದು.",
            "ಉದುರಿದ ಕಾಯಿಗಳ ಮೇಲೆ ಬಿಳಿ ಬಣ್ಣದ ಶಿಲೀಂಧ್ರ ಬೆಳೆ."
        ],
        "organic_control": "ಟ್ರೈಕೋಡರ್ಮಾ ವೈರೈಡ್ (Trichoderma viride) ಅನ್ನು ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರದೊಂದಿಗೆ ಬುಡಕ್ಕೆ ಹಾಕಿ.",
        "chemical_control": "ಮಳೆಗಾಲಕ್ಕೆ ಮುಂಚಿತವಾಗಿ 1% ಬೋರ್ಡೋ ಮಿಶ್ರಣ (Bordeaux Mixture) ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ಉದುರಿದ ರೋಗಗ್ರಸ್ತ ಕಾಯಿಗಳನ್ನು ಸುಟ್ಟುಹಾಕಿ.",
            "ತೋಟದಲ್ಲಿ ಗಾಳಿ-ಬೆಳಕು ಆಡುವಂತೆ ಸವರಿರಿ."
        ]
    },
    "stem cracking": {
        "display_name": "ಕಾಂಡ ಸೀಳುವಿಕೆ (Stem Cracking)",
        "scientific_name": "Physiological Disorder",
        "description": "ಮರದ ಕಾಂಡದಲ್ಲಿ ಲಂಬವಾಗಿ ಸೀಳಿಕೆ ಉಂಟಾಗುವ ಶಾರೀರಿಕ ಅಸ್ವಸ್ಥತೆ.",
        "cause": "ತೀವ್ರ ಬಿಸಿಲು ಮತ್ತು ಹಠಾತ್ ತೇವಾಂಶ ವ್ಯತ್ಯಾಸ.",
        "symptoms": [
            "ಕಾಂಡದ ಮೇಲೆ ಉದ್ದನೆಯ ಸೀಳಿಕೆಗಳು.",
            "ಮರದ ನಾರು ಹೊರಬರುವುದು."
        ],
        "organic_control": "ಮರದ ಬುಡಕ್ಕೆ ತಂಗಾಳಿ ಮತ್ತು ತೇವಾಂಶ ಕಾಯ್ದುಕೊಳ್ಳಲು ಹೊದಿಕೆ ಹಾಕಿ.",
        "chemical_control": "ಸೀಳಿಕೆಗಳಿಗೆ ಬೋರ್ಡೋ ಪೇಸ್ಟ್ (Bordeaux Paste 10%) ಹಚ್ಚಿ.",
        "preventive_measures": [
            "ಕಾಂಡಕ್ಕೆ ಸುಣ್ಣದ ಬಳಪ (Lime wash) ಬಳಿಯಿರಿ."
        ]
    },
    "Stem_bleeding": {
        "display_name": "ಕಾಂಡದಿಂದ ರಸ ಸೋರುವಿಕೆ (Stem Bleeding)",
        "scientific_name": "Thielaviopsis paradoxa",
        "description": "ಕಾಂಡದ ಒಳಭಾಗ ಕೊಳೆತು ಕಂದು ಬಣ್ಣದ ದ್ರವ ಸೋರುವ ಶಿಲೀಂಧ್ರ ರೋಗ.",
        "cause": "ಥಿಯೇಲಾವಿಯೋಪ್ಸಿಸ್ ಪ್ಯಾರಾಡಾಕ್ಸಾ ಶಿಲೀಂಧ್ರ ಸೋಂಕು.",
        "symptoms": [
            "ಕಾಂಡದ ಬಿರುಕುಗಳಿಂದ ಕಡು ಕಂದು ದ್ರವ ಹೊರಸೋರುವುದು.",
            "ಕಾಂಡದ ಒಳಭಾಗ ಕೊಳೆಯುವುದು.",
            "ಗರಿಗಳು ಹಳದಿಯಾಗುವುದು."
        ],
        "organic_control": "ಟ್ರೈಕೋಡರ್ಮಾ ಕಾಂಪೋಸ್ಟ್ ಬುಡಕ್ಕೆ ನೀಡಿ.",
        "chemical_control": "ರೋಗಗ್ರಸ್ತ ಭಾಗವನ್ನು ಕೆರೆದು ಬೋರ್ಡೋ ಪೇಸ್ಟ್ ಅಥವಾ ಕ್ಯಾಲಿಕ್ಸಿನ್ (Calixin 5%) ಹಚ್ಚಿ.",
        "preventive_measures": [
            "ಮರದ ಬುಡದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಚರಂಡಿ ಮಾಡಿ."
        ]
    },
    "yellow leaf disease": {
        "display_name": "ಹಳದಿ ಎಲೆ ರೋಗ (Yellow Leaf Disease)",
        "scientific_name": "Candidatus Phytoplasma",
        "description": "ಅಡಿಕೆ ಮರದ ಎಲೆಗಳು ಹಳದಿಯಾಗಿ ಇಡೀ ಮರ ಒಣಗಿ ಸಾಯುವ ಮಾರಕ ರೋಗ.",
        "cause": "ಫೈಟೋಪ್ಲಾಸ್ಮಾ ಸೂಕ್ಷ್ಮಜೀವಿ (ಕೀಟಗಳ ಮೂಲಕ ಹರಡುತ್ತದೆ).",
        "symptoms": [
            "ಕೆಳಗಿನ ಗರಿಗಳ ತುದಿಯಿಂದ ಹಳದಿ ಬಣ್ಣ ಪ್ರಾರಂಭವಾಗುವುದು.",
            "ಗರಿಗಳು ಚಿಕ್ಕದಾಗಿ ಮುದುಡುವುದು.",
            "ಬೇರುಗಳು ಕೊಳೆಯುವುದು."
        ],
        "organic_control": "ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ಬೇವಿನ ಕಷಾಯ ಸಿಂಪಡಿಸಿ.",
        "chemical_control": "ರೋಗ ಹರಡುವ ಕೀಟಗಳನ್ನು ನಿಯಂತ್ರಿಸಲು ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ (Imidacloprid) ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ತೀವ್ರ ರೋಗಗ್ರಸ್ತ ಮರಗಳನ್ನು ಬುಡಮೇಲು ಮಾಡಿ ಸುಟ್ಟುಹಾಕಿ.",
            "ಹೆಚ್ಚಿನ ಪೊಟ್ಯಾಶ್ ಗೊಬ್ಬರ ನೀಡಿ."
        ]
    },
    "Arecanut_YellowBrownSpot": {
        "display_name": "ಹಳದಿ ಕಂದು ಎಲೆ ಚುಕ್ಕೆ ರೋಗ (Yellow Brown Spot)",
        "scientific_name": "Helminthosporium / Colletotrichum",
        "description": "ಅಡಿಕೆ ಎಲೆಗಳ ಮೇಲೆ ಹಳದಿ ಮತ್ತು ಕಂದು ಬಣ್ಣದ ಚುಕ್ಕೆಗಳು ಉಂಟಾಗುವ ಶಿಲೀಂಧ್ರ ರೋಗ.",
        "cause": "ಕೊಲೆಟೋಟ್ರಿಕಮ್ ಅಥವಾ ಹೆಲ್ಮಿಂಥೋಸ್ಪೋರಿಯಮ್ ಶಿಲೀಂಧ್ರ.",
        "symptoms": [
            "ಎಲೆಗಳ ಮೇಲೆ ಸಣ್ಣ ಹಳದಿ ಮತ್ತು ಕಂದು ಚುಕ್ಕೆಗಳು.",
            "ಚುಕ್ಕೆಗಳು ಒಂದಕ್ಕೊಂದು ಸೇರಿ ಗರಿಗಳು ಒಣಗುವುದು."
        ],
        "organic_control": "ಬೇವಿನ ಎಣ್ಣೆ 3% ಅಥವಾ ಸೂಡೋಮೊನಾಸ್ (20 ಗ್ರಾಂ/ಲೀಟರ್) ಸಿಂಪಡಿಸಿ.",
        "chemical_control": "ಮ್ಯಾಂಕೋಜೆಬ್ (Mancozeb 2.5g/L) ಅಥವಾ ಕಾರ್ಬಂಡಸಿಮ್ ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ಎಲೆಗಳ ಮೇಲೆ ನೀರು ನಿಲ್ಲದಂತೆ ನೋಡಿಕೊಳ್ಳಿ ಮತ್ತು ಬಿದ್ದ ಗರಿಗಳನ್ನು ಸುಡಿ."
        ]
    },
    "CCI_Caterpillars": {
        "display_name": "ಕರಿ ಕೀಟ / ಕಾಯಿ ತಿನ್ನುವ ಕಂಬಳಿಹುಳು (Caterpillar Infestation)",
        "scientific_name": "Slug Caterpillar Complex",
        "description": "ಮರಿಯುಳುಗಳು ಅಡಿಕೆ ಗರಿಗಳನ್ನು ತೀವ್ರವಾಗಿ ತಿಂದು ನಾಶಪಡಿಸುವ ಕೀಟ ಬಾಧೆ.",
        "cause": "ಕಂಬಳಿಹುಳುಗಳು ಮತ್ತು ಪತಂಗದ ಮರಿಹುಳುಗಳು.",
        "symptoms": [
            "ಎಲೆಗಳ ಅಂಚುಗಳಲ್ಲಿ ರಂಧ್ರಗಳು.",
            "ಎಲೆಯ ಈರ್ಕಲು ಮಾತ್ರ ಬಾಕಿ ಉಳಿಯುವುದು."
        ],
        "organic_control": "ಬೆಸಿಲಸ್ ತುರಿಂಜಿಯೆನ್ಸಿಸ್ (Bt 2g/L) ಸಿಂಪಡಿಸಿ.",
        "chemical_control": "ಕ್ವಿನಾಲ್ಫಾಸ್ 25 EC (2 ml/L) ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ಹುಳುಗಳ ಮೊಟ್ಟೆಗಳನ್ನು ಕೈಯಿಂದ ಆರಿಸಿ ನಾಶಪಡಿಸಿ."
        ]
    },
    "WCLWD_DryingofLeaflets": {
        "display_name": "ಎಲೆ ಬಾಡುವಿಕೆ - ಒಣಗುವ ಹಂತ (Leaf Wilt - Drying)",
        "scientific_name": "WCLWD Complex",
        "description": "ಎಲೆ ಬಾಡುವ ರೋಗದ ಮುಂದುವರಿದ ಹಂತದಲ್ಲಿ ಎಲೆಗಳ ತುದಿಗಳು ಕಂದು ಬಣ್ಣಕ್ಕೆ ತಿರುಗಿ ಒಣಗುತ್ತವೆ.",
        "cause": "ಬೇರು ಕೊಳೆತ ಮತ್ತು ನಾಳಗಳ ಸೋಂಕು.",
        "symptoms": [
            "ಎಲೆಗಳ ತುದಿಗಳಿಂದ ಒಣಗುವಿಕೆ ಪ್ರಾರಂಭವಾಗುವುದು.",
            "ಗರಿಗಳು ಕಾಗದದಂತೆ ಒಣಗುವುದು."
        ],
        "organic_control": "ಟ್ರೈಕೋಡರ್ಮಾ ಕಾಂಪೋಸ್ಟ್ ಬುಡಕ್ಕೆ ಹಾಕಿ.",
        "chemical_control": "ಕಾಪರ್ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ (COC 3g/L) ಬುಡಕ್ಕೆ ನೀಡಿ.",
        "preventive_measures": [
            "ಸರಿಯಾದ ನೀರಾವರಿ ಮತ್ತು ಬಸಿಕಾಲುವೆ ಮಾಡಿ."
        ]
    },
    "WCLWD_Flaccidity": {
        "display_name": "ಎಲೆ ಬಾಡುವಿಕೆ - ತೂಗಾಡುವ ಹಂತ (Leaf Wilt - Flaccidity)",
        "scientific_name": "WCLWD Complex",
        "description": "ಎಲೆಗಳು ಶಕ್ತಿ ಕಳೆದುಕೊಂಡು ಕೆಳಕ್ಕೆ ತೂಗಾಡುವ ಮಧ್ಯಂತರ ರೋಗದ ಹಂತ.",
        "cause": "ನೀರು ಸಾಗಣೆ ನಾಳಗಳ ತಡೆಗೋಡೆ.",
        "symptoms": [
            "ಎಲೆಗಳು ಕೆಳಮುಖವಾಗಿ ಬಾಗುವುದು.",
            "ಎಲೆಯ ಪತ್ರಗಳು ಸಡಿಲವಾಗುವುದು."
        ],
        "organic_control": "ಮೈಕೋರೈಜಾ (VAM) ಜೀವಣು ಗೊಬ್ಬರ ಬಳಸಿ.",
        "chemical_control": "ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶಗಳ ಮಿಶ್ರಣ ನೀಡಿ.",
        "preventive_measures": [
            "ಬೇಸಿಗೆಯಲ್ಲಿ ನೀರಾವರಿ ಒದಗಿಸಿ."
        ]
    },
    "WCLWD_Yellowing": {
        "display_name": "ಎಲೆ ಬಾಡುವಿಕೆ - ಹಳದಿ ಹಂತ (Leaf Wilt - Yellowing)",
        "scientific_name": "WCLWD Complex",
        "description": "ಎಲೆ ಬಾಡುವ ರೋಗದ ಆರಂಭಿಕ ಹಂತದಲ್ಲಿ ಗರಿಗಳು ಹಳದಿ ಬಣ್ಣಕ್ಕೆ ತಿರುಗುತ್ತವೆ.",
        "cause": "ಬೇರು ಕಸಿ ಮತ್ತು ನಂಜು ಕೀಟಗಳ ಸೋಂಕು.",
        "symptoms": [
            "ಕೆಳಗಿನ ಗರಿಗಳಲ್ಲಿ ಬಂಗಾರದ ಹಳದಿ ಬಣ್ಣ.",
            "ಮರದ ಇಳುವರಿ ಕಡಿಮೆಯಾಗುವುದು."
        ],
        "organic_control": "ಬೇವಿನ ಹಿಂಡಿ ಗೊಬ್ಬರ ಹಾಕಿ.",
        "chemical_control": "ಮೆಗ್ನೀಷಿಯಂ ಸಲ್ಫೇಟ್ ಸಿಂಪಡಿಸಿ.",
        "preventive_measures": [
            "ಸೊಂಪಾದ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ ನೀಡಿ."
        ]
    }
}


def get_kannada_recommendation(disease_name: str) -> dict:
    """Returns Kannada translated disease recommendation details."""
    for key, val in KANNADA_TRANSLATIONS.items():
        if key.lower() == disease_name.lower():
            return val
    return {
        "display_name": disease_name,
        "scientific_name": "N/A",
        "description": "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.",
        "cause": "ಅಜ್ಞಾತ",
        "symptoms": ["ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ."],
        "organic_control": "ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        "chemical_control": "ಕೃಷಿ ತಜ್ಞರ ಸಲಹೆ ಪಡೆಯಿರಿ.",
        "preventive_measures": ["ತೋಟವನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ."]
    }
