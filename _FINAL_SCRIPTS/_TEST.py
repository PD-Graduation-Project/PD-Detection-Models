"""
Parkinson's Disease Prediction Test Suite

Tests all three prediction modalities:
1. Drawing analysis (spiral/wave images)
2. Tremor analysis (IMU wrist signals)
3. Audio analysis (voice recordings)
4. Questionnaire data (demographics + 28 questions)
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

tremor_data = [
    ("Healthy tremor", "examples/tremor/healthy/", 0, "right"),
    ("PD tremor", "examples/tremor/pd/", 10, "right"),
]

# ============================================================================
# Test 1: Drawing Predictions
# ============================================================================
def run_drawing_tests():
    print_section_header("1. DRAWING PREDICTIONS (Spiral/Wave Analysis)")
    from predict_from_drawing import predict as pred_drawing

    tests = [
        ("Healthy image #1", drawing_data[0]),
        ("Healthy image #2", drawing_data[1]),
        ("PD image #1", drawing_data[2]),
        ("PD image #2", drawing_data[3]),
    ]

    for label, path in tests:
        if Path(path).exists():
            prob = pred_drawing(path)
            print_result(label, prob)
        else:
            print(f"{label:40s} | File not found: {path}")


# ============================================================================
# Test 2: Tremor Predictions
# ============================================================================
def run_tremor_tests():
    print_section_header("2. TREMOR PREDICTIONS (IMU Wrist Signals)")
    from predict_from_tremor import predict as pred_tremor

    for label, directory, movement, handedness in tremor_data:
        if Path(directory).exists():
            prob = pred_tremor(
                txt_dir=directory,
                movement=movement,
                handedness=handedness,
            )
            print_result(label, prob)
        else:
            print(f"{label:40s} | Directory not found: {directory}")
            
# ============================================================================
# Test 3: Audio Predictions
# ============================================================================

def run_audio_tests():
    print_section_header("3. AUDIO PREDICTIONS (Voice Analysis)")
    from predict_from_audio import predict as pred_audio

    gender = None
    if Path(yaml_file).exists():
        user_data = load_user_data(yaml_file)
        gender = "F" if user_data["gender"] == 1 else "M"
        print(f"\nUsing gender from user data: {gender}")

    tests = [
        ("Healthy voice sample", audio_data[0]),
        ("PD voice sample", audio_data[1]),
    ]

    for label, path in tests:
        if Path(path).exists():
            prob = pred_audio(path, gender=gender)
            print_result(label, prob)
        else:
            print(f"{label:40s} | File not found: {path}")

# ============================================================================
# Test 4: Questionnaire Predictions
# ============================================================================

def run_questionnaire_tests():
    print_section_header("4. QUESTIONNAIRE PREDICTIONS (Demographics + 28 Questions)")
    from predict_from_questionnaire import predict as pred_questionnaire

    user_data = load_user_data(yaml_file)

    print(f"\nLoaded user data from {yaml_file}:")
    print(f"  Age: {user_data['age']}")
    print(f"  Height: {user_data['height']} cm")
    print(f"  Weight: {user_data['weight']} kg")
    print(f"  Gender: {'Female' if user_data['gender'] == 1 else 'Male'}")
    print(f"  Kinship history: {'Yes' if user_data['appearance_in_kinship'] == 1 else 'No'}")
    print(f"  First-grade kinship: {'Yes' if user_data['appearance_in_first_grade_kinship'] == 1 else 'No'}")
    print(f"  Questions answered: {len(user_data['questions'])}/28")

    questionnaire_input = [
        user_data["age"],
        user_data["height"],
        user_data["weight"],
        user_data["gender"],
        user_data["appearance_in_kinship"],
        user_data["appearance_in_first_grade_kinship"],
        user_data["questions"],
    ]

    prob = pred_questionnaire(questionnaire_input)
    print()
    print_result("User questionnaire data", prob)



# =============================================================================
# Main entry
# =============================================================================

def run_tests(mode="all"):
    """
    mode:
        - "drawing"
        - "questionnaire"
        - "audio"
        - "tremor"
        - "all"
    """
    mode = mode.lower()

    if mode in ("drawing", "all"):
        run_drawing_tests()
        
    if mode in ("tremor", "all"):
        run_tremor_tests()
        
    if mode in ("audio", "all"):
        run_audio_tests()

    if mode in ("questionnaire", "all"):
        run_questionnaire_tests()


if __name__ == "__main__":
    # Change mode here
    run_tests(mode="tremor")