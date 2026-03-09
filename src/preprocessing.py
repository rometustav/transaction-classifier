"""
This script preprocesses the generated transaction dataset before model training.

Preprocessing approach:
    - Load the generated dataset from CSV.
    - Remove rows with missing target labels.
    - Normalize the transaction purpose text:
        * convert text to lowercase
        * remove non-alphanumeric characters
        * normalize whitespace
    - Remove rows where cleaned text becomes empty.
    - Reset the dataframe index after filtering.
    - Save the cleaned dataset for downstream modelling.

Usage:
    python src/preprocessing.py

Output:
    data/transactions_cleaned.csv
"""

import pandas as pd
import re

def clean_text(text: str) -> str:
    """
    Clean and normalize a transaction purpose text.

    Steps:
    1. Convert non-string values to empty string
    2. Lowercase text
    3. Remove unwanted characters
    4. Collapse multiple spaces
    5. Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Replace anything that isn't a character, number or a whitespace
    text = re.sub(r"\s+", " ", text).strip() # Replace multiple whitespaces with one whitespace, remove spaces at the beginning and end

    return text

def load_and_preprocess(input_path: str) -> pd.DataFrame:
    """
    Load transaction data and apply preprocessing.

    Expected columns:
    - purpose_text
    - transaction_type
    """
    df = pd.read_csv(input_path)

    # Remove rows where label is missing
    df = df.dropna(subset=["transaction_type"])

    # Replace missing purpose text
    df["purpose_text"] = df["purpose_text"].fillna("")

    # Clean text
    df["purpose_text"] = df["purpose_text"].apply(clean_text)

    # Remove empty texts after cleaning
    df = df[df["purpose_text"].str.strip() != ""]

    # Reset index
    df = df.reset_index(drop=True)

    return df


def save_preprocessed_data(input_path: str, output_path: str) -> None:
    df = load_and_preprocess(input_path)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed dataset saved to: {output_path}")
    print(f"Remaining rows: {len(df)}")


if __name__ == "__main__":
    save_preprocessed_data(
        input_path="data/transactions.csv",
        output_path="data/transactions_cleaned.csv"
    )