import matplotlib.pyplot as plt

def plot_class_counts(df, target_col: str):
    """
    Plot the number of Parkinson's vs Healthy samples in the dataset.

    Args:
        df (pd.DataFrame): The input dataframe containing labels.
        target_col (str): Name of the column indicating PD vs Healthy.

    The function displays a bar chart with clear counts.
    """

    # Step 1: Count occurrences of each class
    # ----------------------------------------
    counts = df[target_col].value_counts()

    # Step 2: Create bar plot
    # -------------------------
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")

    # Step 3: Label and style the figure
    # -----------------------------------
    plt.title("Distribution of Parkinson's vs Healthy Samples")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Step 4: Show the plot
    # -----------------------
    plt.show()
