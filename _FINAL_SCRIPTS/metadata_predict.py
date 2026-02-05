"""
Expected input format (STRICT):

    predict([
        age,                                    # int | float
        height,                                 # int | float (cm)
        weight,                                 # int | float (kg)
        gender,                                 # 'male' | 'female' | 0 | 1 | -1
        appearance_in_kinship,                  # 0 | 1 | -1 | 'True' | 'False'
        appearance_in_first_grade_kinship,      # 0 | 1 | -1 | 'True' | 'False'
        questions                               # list of 28 numeric values
    ])

Internal model input order (EXACT):

    [
        age_class,
        height_to_weight_class,
        gender,
        appearance_in_kinship,
        appearance_in_first_grade_kinship,
        Q01, Q02, Q03, Q04, Q05, Q06, Q07, Q08,
        Q09, Q10, Q11, Q12, Q13, Q14, Q15, Q16,
        Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24,
        Q25, Q26, Q27, Q28
    ]

Notes:
- Raw age, height, and weight are NOT passed to the model.
- Age and height/weight are converted to fixed classes.
- Feature order must match training exactly.
"""

import torch
import numpy as np
from FINAL_MODELS.densenet169 import DenseNet1691D


# -------------------------------------------------
# Fixed boundaries (must match training)
# -------------------------------------------------
AGE_BOUNDS = [58.620843499477175, 71.37997532894738]


# -------------------------------------------------
# Feature engineering helpers
# -------------------------------------------------
def age_to_class(age: float) -> float:
    if age < AGE_BOUNDS[0]:
        return 1.0
    elif age < AGE_BOUNDS[1]:
        return 2.0
    else:
        return 3.0


def height_weight_to_class(height: float, weight: float) -> float:
    if height < 172.4 and weight < 74.3:
        return 1.0
    elif 172.4 <= height < 179.1 and 74.3 <= weight < 96.1:
        return 2.0
    elif height >= 179.1 and weight >= 96.1:
        return 3.0
    else:
        return -1.0


def normalize_bool(val) -> float:
    if isinstance(val, (bool, np.bool_)):
        return 1.0 if val else 0.0
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "true":
            return 1.0
        if v == "false":
            return 0.0
    if val in [0, 1, -1]:
        return float(val)
    return -1.0


def normalize_gender(val) -> float:
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "male":
            return 1.0
        if v == "female":
            return 0.0
    if val in [0, 1, -1]:
        return float(val)
    return -1.0


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
def preprocess_tabular_input(x):
    """
    Preprocess metadata input into model-ready tensor.

    Args:
        x (list): [
            age,
            height,
            weight,
            gender,
            appearance_in_kinship,
            appearance_in_first_grade_kinship,
            questions (list of 28 values)
        ]

    Returns:
        torch.Tensor: shape (1, 33)
    """

    if not isinstance(x, list) or len(x) != 7:
        raise ValueError("Input must be a list of length 7")

    age, height, weight, gender, app_kin, app_first, questions = x

    if not isinstance(questions, list) or len(questions) != 28:
        raise ValueError("questions must be a list of length 28")

    # Feature engineering
    age_class = age_to_class(float(age))
    height_to_weight = height_weight_to_class(float(height), float(weight))
    gender = normalize_gender(gender)
    app_kin = normalize_bool(app_kin)
    app_first = normalize_bool(app_first)

    # Assemble features (EXACT order)
    features = [
        age_class,
        height_to_weight,
        gender,
        app_kin,
        app_first,
        *[float(q) for q in questions]
    ]

    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=-1.0)

    return torch.tensor(features).unsqueeze(0)


# -------------------------------------------------
# Prediction
# -------------------------------------------------
def predict(x, device=None):
    """
    Run inference on metadata input.

    Args:
        x (list): See module docstring for exact format.
        device (torch.device, optional)

    Returns:
        float: Parkinson's probability (0–1)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet1691D().to(device)
    checkpoint = torch.load(
        "FINAL_MODELS/Metadata_Model_best.pth",
        map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X = preprocess_tabular_input(x).to(device)

    with torch.inference_mode():
        logits = model(X)
        probability = torch.sigmoid(logits).item()

    return probability
