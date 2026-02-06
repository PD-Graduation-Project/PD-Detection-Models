"""
Parkinson's Disease Prediction Test Suite

Tests all three prediction modalities:
1. Drawing analysis (spiral/wave images)
2. Questionnaire data (demographics + 28 questions)
3. Audio analysis (voice recordings)
"""

import yaml
from pathlib import Path
from utils.helper_functions import *

# =============
# DATA PATHS
# =============
drawing_data = ["examples/Healthy1.png",
                "examples/Healthy10.png",
                "examples/Parkinson1.png",
                "examples/Parkinson10.png"]

yaml_file = 'examples/user_data.yaml'

audio_data = ["examples/healthy_audio.wav",
            "examples/pd_audio.wav"]

# ============================================================================
# Test 1: Drawing Predictions
# ============================================================================

print_section_header("1. DRAWING PREDICTIONS (Spiral/Wave Analysis)")

from predict_from_drawing import predict as pred_drawing

drawing_tests = [
    ("Healthy image #1", drawing_data[0]),
    ("Healthy image #2", drawing_data[1]),
    ("PD image #1", drawing_data[2]),
    ("PD image #2", drawing_data[3]),
]

for label, path in drawing_tests:
    if Path(path).exists():
        prob = pred_drawing(path)
        print_result(label, prob)
    else:
        print(f"{label:40s} | File not found: {path}")


# ============================================================================
# Test 2: Questionnaire Predictions
# ============================================================================

print_section_header("2. QUESTIONNAIRE PREDICTIONS (Demographics + 28 Questions)")

from predict_from_questionnaire import predict as pred_questionnaire

# Load user data from YAML
user_data = load_user_data(yaml_file)

# Display loaded data
print(f"\nLoaded user data from {yaml_file}:")
print(f"  Age: {user_data['age']}")
print(f"  Height: {user_data['height']} cm")
print(f"  Weight: {user_data['weight']} kg")
print(f"  Gender: {'Female' if user_data['gender'] == 1 else 'Male'}")
print(f"  Kinship history: {'Yes' if user_data['appearance_in_kinship'] == 1 else 'No'}")
print(f"  First-grade kinship: {'Yes' if user_data['appearance_in_first_grade_kinship'] == 1 else 'No'}")
print(f"  Questions answered: {len(user_data['questions'])}/28")

# Prepare input vector
questionnaire_input = [
    user_data['age'],
    user_data['height'],
    user_data['weight'],
    user_data['gender'],
    user_data['appearance_in_kinship'],
    user_data['appearance_in_first_grade_kinship'],
    user_data['questions']
]

# Predict
prob = pred_questionnaire(questionnaire_input)
print()
print_result("User questionnaire data", prob)


# ============================================================================
# Test 3: Audio Predictions
# ============================================================================

print_section_header("3. AUDIO PREDICTIONS (Voice Analysis)")

from predict_from_audio import predict as pred_audio

# Get gender from user data if available
gender = None
if Path(yaml_file).exists():
    user_data = load_user_data(yaml_file)
    gender = 'F' if user_data['gender'] == 1 else 'M'
    print(f"\nUsing gender from user data: {gender}")

audio_tests = [
    ("Healthy voice sample", audio_data[0]),
    ("PD voice sample", audio_data[1]),
]

for label, path in audio_tests:
    if Path(path).exists():
        prob = pred_audio(path, gender=gender)
        print_result(label, prob)
    else:
        print(f"{label:40s} | File not found: {path}")
