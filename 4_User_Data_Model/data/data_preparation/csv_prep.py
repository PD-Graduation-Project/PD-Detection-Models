import pandas as pd
import numpy as np
import json
import os

def create_filtered_classified_csv(
    input_csv_path,
    output_csv_path = "output.csv"
):
    """
    Creates a new CSV with selected features and fixed class rules.

    Output columns:
    age, height_to_weight, gender,
    appearance_in_kinship,
    appearance_in_first_grade_kinship,
    label
    """

    df = pd.read_csv(input_csv_path)

    # -------------------------------------------------
    # Filter only Healthy and PD
    # -------------------------------------------------
    df = df[
        (df["label"].isin([0, 1])) &
        (df["label"] != 2)
    ].copy()

    # -------------------------------------------------
    # Age classes (fixed boundaries)
    # -------------------------------------------------
    age_bounds = [58.620843499477175, 71.37997532894738]

    def age_class(age):
        if age < age_bounds[0]:
            return 1
        elif age < age_bounds[1]:
            return 2
        else:
            return 3

    df["age"] = df["age"].apply(age_class)

    # -------------------------------------------------
    # Height–Weight combined classes
    # -------------------------------------------------
    def height_weight_class(row):
        h, w = row["height"], row["weight"]

        if h < 172.4 and w < 74.3:
            return 1
        elif 172.4 <= h < 179.1 and 74.3 <= w < 96.1:
            return 2
        elif h >= 179.1 and w >= 96.1:
            return 3
        else:
            return -1  # outside defined regions

    df["height_to_weight"] = df.apply(height_weight_class, axis=1)


    # -------------------------------------------------
    # Select final columns
    # -------------------------------------------------
    final_df = df[
        [
            "id",
            "age",
            "height_to_weight",
            "gender",
            "appearance_in_kinship",
            "appearance_in_first_grade_kinship",
            "label"
        ]
    ]

    # -------------------------------------------------
    # Save new CSV
    # -------------------------------------------------
    final_df.to_csv(output_csv_path, index=False)

    return final_df


def add_questionnaire_to_csv(
    csv_path,
    json_folder_path,
    output_csv_path="output.csv",
    id_column="id"
):
    """
    Loads a processed CSV and appends 30 questionnaire answers
    from JSON files matched by subject ID.

    Parameters
    ----------
    csv_path : str
        Path to processed CSV
    json_folder_path : str
        Folder containing JSON files named by subject ID
    output_csv_path : str
        Path to save updated CSV
    id_column : str
        Column in CSV that matches JSON subject_id
    """

    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"CSV must contain '{id_column}' column for matching JSON files.")

    # -------------------------------------------------
    # Define questions (EXCLUDE 18, 19)
    # -------------------------------------------------
    question_cols = [
        f"Q{str(i).zfill(2)}"
        for i in range(1, 31)
        if i not in (18, 19)
    ]

    # -------------------------------------------------
    # Initialize columns ONCE
    # -------------------------------------------------
    for q in question_cols:
        if q not in df.columns:
            df[q] = pd.NA

    # -------------------------------------------------
    # Load JSON answers
    # -------------------------------------------------
    questionnaire_data = {}

    for fname in os.listdir(json_folder_path):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(json_folder_path, fname)
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            subject_id = str(data["subject_id"])

            # Convert boolean to int: False->0, True->1
            answers = {}
            for item in data["item"]:
                link_id = int(item["link_id"])
                
                # Skip questions 18 and 19
                if link_id in (18, 19):
                    continue
                
                # Convert boolean answer to integer
                answer_value = 1 if item["answer"] else 0
                
                answers[f"Q{str(link_id).zfill(2)}"] = answer_value

            questionnaire_data[subject_id] = answers
            print(f"Loaded data for subject: {subject_id}")
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print(f"\nTotal subjects loaded from JSON: {len(questionnaire_data)}")
    print(f"Subject IDs in JSON: {list(questionnaire_data.keys())}")
    print(f"\nTotal rows in CSV: {len(df)}")
    print(f"Subject IDs in CSV: {df[id_column].astype(str).tolist()}")

    import pandas as pd
import json
import os

def add_questionnaire_to_csv(
    csv_path,
    json_folder_path,
    output_csv_path="output.csv",
    id_column="id"
):
    """
    Loads a processed CSV and appends 30 questionnaire answers
    from JSON files matched by subject ID.

    Parameters
    ----------
    csv_path : str
        Path to processed CSV
    json_folder_path : str
        Folder containing JSON files named by subject ID
    output_csv_path : str
        Path to save updated CSV
    id_column : str
        Column in CSV that matches JSON subject_id
    """

    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"CSV must contain '{id_column}' column for matching JSON files.")

    # -------------------------------------------------
    # Define questions (EXCLUDE 18, 19)
    # -------------------------------------------------
    question_cols = [
        f"Q{str(i).zfill(2)}"
        for i in range(1, 31)
        if i not in (18, 19)
    ]

    # -------------------------------------------------
    # Initialize columns ONCE
    # -------------------------------------------------
    for q in question_cols:
        if q not in df.columns:
            df[q] = pd.NA

    # -------------------------------------------------
    # Load JSON answers
    # -------------------------------------------------
    questionnaire_data = {}

    for fname in os.listdir(json_folder_path):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(json_folder_path, fname)
        
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            subject_id = str(data["subject_id"])

            # Convert boolean to int: False->0, True->1
            answers = {}
            for item in data["item"]:
                link_id = int(item["link_id"])
                
                # Skip questions 18 and 19
                if link_id in (18, 19):
                    continue
                
                # Convert boolean answer to integer
                answer_value = 1 if item["answer"] else 0
                
                answers[f"Q{str(link_id).zfill(2)}"] = answer_value

            questionnaire_data[subject_id] = answers
            # print(f"Loaded data for subject: {subject_id}")
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    print(f"\nTotal subjects loaded from JSON: {len(questionnaire_data)}")
    # print(f"Subject IDs in JSON: {list(questionnaire_data.keys())}")
    print(f"\nTotal rows in CSV: {len(df)}")
    # print(f"Subject IDs in CSV: {df[id_column].astype(str).tolist()}")

    # -------------------------------------------------
    # Fill answers (normalize IDs by converting to int)
    # -------------------------------------------------
    matched_count = 0
    for idx, row in df.iterrows():
        # Convert CSV ID to int to match with JSON (removes leading zeros)
        csv_id = int(row[id_column])
        
        # Look for matching JSON subject by converting JSON IDs to int
        for json_id_str, answers in questionnaire_data.items():
            json_id = int(json_id_str)
            
            if csv_id == json_id:
                matched_count += 1
                for q, val in answers.items():
                    df.at[idx, q] = val
                break

    print(f"\nMatched {matched_count} subjects between CSV and JSON")

    # -------------------------------------------------
    # Reorder columns ONCE
    # -------------------------------------------------
    base_cols = [
        "id",
        "age",
        "height_to_weight",
        "gender",
        "appearance_in_kinship",
        "appearance_in_first_grade_kinship",
    ]

    final_cols = base_cols + question_cols + ["label"]
    
    # Only include columns that exist in the dataframe
    final_cols = [col for col in final_cols if col in df.columns]

    df = df[final_cols]

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved output to: {output_csv_path}")
    
    return df