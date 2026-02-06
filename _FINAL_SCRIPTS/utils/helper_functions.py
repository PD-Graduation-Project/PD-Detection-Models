import yaml
from pathlib import Path

def load_user_data(yaml_path='examples/user_data.yaml'):
    """
    Load user data from YAML file.
    
    Expected YAML format:
    ---
    age: 50
    height: 150
    weight: 100
    gender: 'male'  # or 'female'
    appearance_in_kinship: false  # or true
    appearance_in_first_grade_kinship: false  # or true
    questions: [1, 1, 1, ..., 1]  # 28 binary answers
    
    Returns:
        dict: User data with standardized keys
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Standardize gender encoding (0 = male, 1 = female)
    if isinstance(data['gender'], str):
        data['gender'] = 1 if data['gender'].lower() == 'female' else 0
    
    # Standardize kinship encoding (0 = False, 1 = True)
    if isinstance(data['appearance_in_kinship'], bool):
        data['appearance_in_kinship'] = int(data['appearance_in_kinship'])
    
    if isinstance(data['appearance_in_first_grade_kinship'], bool):
        data['appearance_in_first_grade_kinship'] = int(data['appearance_in_first_grade_kinship'])
    
    return data


def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(label, probability, threshold=0.5):
    """
    Print prediction result with interpretation.
    
    Args:
        label: Description of the test sample
        probability: Prediction probability (0-1)
        threshold: Classification threshold (default 0.5)
    """
    prediction = "PD" if probability >= threshold else "Healthy"
    print(f"{label:30s} | Prob: {probability:.4f} | Prediction: {prediction}")