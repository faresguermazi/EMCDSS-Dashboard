from pathlib import Path
import json
from datetime import datetime

# File path to save feedback
file_path = Path("/netscratch/fguermazi/XAI/utils_generated_files/clixm_feedback.jsonl") #CHANGE PATH IF NEEDED

# CLIX-M checklist items with associated user roles
clixm_items = [
    {"item_name": "Purpose", "input_type": "text", "required_clinician": False, "required_developer": False},

    {"item_name": "Domain Relevance", "input_type": "dropdown", "dropdown_options": [
        "Very irrelevant", "Irrelevant", "Relevant", "Very relevant"
    ], "required_clinician": True, "required_developer": False},

    {"item_name": "Reasonableness", "input_type": "dropdown", "dropdown_options": [
        "Very incoherent", "Incoherent", "Coherent", "Very coherent"
    ], "required_clinician": True, "required_developer": False},

    {"item_name": "Actionability", "input_type": "dropdown", "dropdown_options": [
        "Not actionable", "Slightly actionable", "Actionable", "Highly actionable"
    ], "required_clinician": True, "required_developer": False},

    {"item_name": "Correctness", "input_type": "scale", "required_clinician": False, "required_developer": True},
    {"item_name": "Confidence Alignment", "input_type": "scale", "required_clinician": False, "required_developer": True},
    {"item_name": "Consistency", "input_type": "scale", "required_clinician": False, "required_developer": True},
    {"item_name": "Robustness", "input_type": "scale", "required_clinician": False, "required_developer": True},
    {"item_name": "Causal Validity", "input_type": "scale", "required_clinician": False, "required_developer": True},

    {"item_name": "Narrative Reasoning", "input_type": "scale", "required_clinician": False, "required_developer": False},
    {"item_name": "Bias & Fairness", "input_type": "scale", "required_clinician": False, "required_developer": False},
    {"item_name": "Model Troubleshooting", "input_type": "scale", "required_clinician": False, "required_developer": False},

    {"item_name": "Interpretation Clarity", "input_type": "test", "required_clinician": False, "required_developer": False},

    {"item_name": "XAI Limitations", "input_type": "text", "required_clinician": False, "required_developer": False}
]


def load_clixm_items():
    return clixm_items

def save_clixm_feedback(user_role, feedback_dict):
    """
    Save feedback entry to a JSONL file with timestamp.
    Each line is a JSON object.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_role": user_role,
        "feedback": feedback_dict
    }

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
